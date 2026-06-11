"""GABLS1 stable boundary-layer benchmark on the legoESM SCM.

Reproduces the Cuxart et al. (2006) intercomparison case using the
single-column model: a 400-m-deep stably stratified boundary layer
cooled by 0.25 K/hr at the surface for 9 hours.  The case is the
canonical benchmark for the MYNN-2.5 closure (Phase C) and the
prescribed-T_s forcing hook (Phase B).

Setup (mirrors jax_scm's :func:`scm.examples.gabls1.get_gabls1`):
  * f_c = 1.39e-4 (~73 deg latitude)
  * Geostrophic wind: ug=8, vg=0
  * z0m = z0h = 0.1 m
  * Initial theta: 265 K below 100 m + capping inversion 0.01*(z-100) above
  * Initial qke: 0.8*(1-z/250)^3 below 250 m
  * Surface temperature: 265 K - 0.25*t/3600 [K]
  * Domain: 9 hr, jax_scm uses 400 m / Nz=64 (geometric)

The legoESM SCM runs on a pressure-based sigma coordinate, so the
vertical levels do not coincide with jax_scm's height grid.  The
companion benchmark test (:mod:`tests.validation.test_scm_gabls1`)
performs scalar / surface-tracking comparisons against the oracle
NetCDF in ``tests/validation/scm_oracle/`` rather than profile-by-
profile matching.

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/matrix/run_scm_test_matrix.py gabls1
"""

from __future__ import annotations

import argparse
import sys

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig
from legoesm.atmosphere.physics.turbulence.config import (
    MYNN25Config,
    SurfaceLayerConfig,
)
from legoesm.atmosphere.scm import SingleColumnModel
from legoesm.atmosphere.scm_forcing import SCMForcing

from legoesm import constants

_F_C = 1.39e-4
_U_G = 8.0
_T_S_0 = 265.0
_T_S_RATE = -0.25 / 3600.0
_Z_INV = 100.0
_LAPSE_ABOVE = 0.01


def build_initial_T_profile(z_full: jnp.ndarray) -> jnp.ndarray:
    """GABLS1 initial potential temperature profile, converted to T.

    Below 100 m: theta=265 K (neutral mixed layer).  Above 100 m: theta
    increases at 0.01 K/m (capping inversion).  Conversion from theta to
    T uses an idealised exner function ``(p/p_ref)^kappa`` with
    p ~= p_s*exp(-z/H), H=8 km; for the shallow domain this stays
    within millikelvin of the analytic theta.
    """
    theta = jnp.where(
        z_full > _Z_INV,
        _T_S_0 + _LAPSE_ABOVE * (z_full - _Z_INV),
        _T_S_0,
    )
    p_s = 1.0e5
    H = 8.0e3
    p = p_s * jnp.exp(-z_full / H)
    exner = (p / constants.p_ref) ** constants.kappa
    return theta * exner


def _turbulence_config(turbulence: str) -> TurbulenceConfig:
    """GABLS1 turbulence config for the requested scheme.

    GABLS1 prescribes a *cooling surface temperature* (``prescribe="T_s"``) with
    the bulk transfer ACTIVE (``Ch_neutral=1.5e-3``), so the surface heat flux is
    computed from ``T_sfc`` by the scheme's own bulk formula — i.e. the turbulence
    scheme's NATIVE surface coupling (unlike the prescribed-kinematic-flux Wangara
    case). Prognostic CLUBB therefore receives the GABLS1 surface cooling through
    its real ``wpthlp_sfc`` lower boundary, making this a correctly-coupled stable
    boundary-layer benchmark for CLUBB."""
    surf = SurfaceLayerConfig(z0=0.1, Cd_neutral=1.5e-3, Ch_neutral=1.5e-3)
    if turbulence == "mynn25":
        return TurbulenceConfig(scheme="mynn25", mynn25=MYNN25Config(surface=surf))
    if turbulence == "clubb":
        return TurbulenceConfig(
            scheme="clubb", clubb=CLUBBConfig(prognostic=True, surface=surf))
    raise ValueError(
        f"gabls1 --turbulence={turbulence!r} unsupported; "
        "choose from 'mynn25', 'clubb'.")


def build_scm(
    nlev: int = 32, dt: float = 5.0, sigma_top: float = 0.7,
    turbulence: str = "mynn25",
) -> SingleColumnModel:
    """Configure the legoESM SCM for the GABLS1 setup."""
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top)
    p_s = 1.0e5
    H = 8.0e3
    p_full = sigma_coord.sigma_full * p_s
    z_full = -H * jnp.log(jnp.maximum(p_full / p_s, 1e-6))

    T_profile = build_initial_T_profile(z_full)

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=_turbulence_config(turbulence),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    forcing = SCMForcing(
        f_c=_F_C,
        u_geo=lambda t: jnp.full(nlev, _U_G),
        v_geo=lambda t: jnp.zeros(nlev),
        prescribe="T_s",
        T_s=lambda t: jnp.asarray(_T_S_0 + _T_S_RATE * t),
    )
    return SingleColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=dt,
        T_profile=T_profile, q_v_profile=jnp.zeros(nlev),
        u=_U_G, v=0.0,
        latitude_deg=73.0,
        sigma_top=sigma_top,
        forcing=forcing,
        time_integrator="forward_euler",
    )


def _tke_diag(phys_state, turbulence: str):
    """(near-surface TKE-like value, column max) for the active scheme. ``qke``
    for MYNN-2.5; the prognostic ``wp2`` slot of the packed CLUBB moments for
    CLUBB. Both carries are read at the genuine near-surface level: ``qke`` is
    top-down (last index) while the packed CLUBB ``wp2`` is ascending zm (first
    index = surface)."""
    if turbulence == "clubb":
        wp2 = np.asarray(phys_state.clubb_moments)[0, 4, :]
        return float(wp2[0]), float(wp2.max())     # ascending: index 0 = surface
    qke = np.asarray(phys_state.qke)[0]
    return float(qke[-1]), float(qke.max())        # top-down: last = surface


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hours", type=float, default=9.0,
                   help="Run duration [hours] (default 9.0, Cuxart 2006).")
    p.add_argument("--dt", type=float, default=5.0,
                   help="Physics time step [s].")
    p.add_argument("--nlev", type=int, default=32,
                   help="Vertical grid count.")
    p.add_argument("--sigma-top", type=float, default=0.7,
                   help="Top sigma value (smaller = taller domain; 0.7 ~ 2.5 km).")
    p.add_argument("--turbulence", choices=("mynn25", "clubb"), default="mynn25",
                   help="Turbulence scheme (clubb = full prognostic CLUBB; GABLS1's "
                        "prescribe='T_s' + active bulk flux drives CLUBB's native "
                        "surface coupling).")


def run(args: argparse.Namespace) -> int:
    scm = build_scm(nlev=args.nlev, dt=args.dt, sigma_top=args.sigma_top,
                    turbulence=args.turbulence)
    nsteps = int(args.hours * 3600.0 / args.dt)
    # Preflight: reject a zero-step request BEFORE scm.run() (which would crash
    # stacking empty history arrays) so the guard yields a controlled failure.
    if nsteps < 1:
        print(f"[FAIL] nsteps={nsteps} < 1 (hours*3600/dt truncated to zero) - "
              "nothing to integrate")
        return 1
    # Capture the INITIAL near-surface temperature so the pass gate can require a
    # genuine cooling RESPONSE (not just a plausible final value an unchanged
    # column would already satisfy).
    T_low_init = float(scm.state.T.data[0, 0, 0, -1])
    print(f"[GABLS1] nlev={args.nlev}  dt={args.dt}  nsteps={nsteps}  "
          f"hours={args.hours}  turbulence={args.turbulence}  "
          f"T_low_init={T_low_init:.3f}")

    save_every = max(1, nsteps // 60)
    final, hist = scm.run(nsteps=nsteps, save_every=save_every)

    T_low = float(final.T.data[0, 0, 0, -1])
    tke_sfc, tke_max = _tke_diag(scm.phys_state, args.turbulence)
    u_low = float(final.u.data[0, 0, 0, -1])
    v_low = float(final.v.data[0, 0, 0, -1])
    print(f"[GABLS1] final T_low = {T_low:.3f} K  (Δ={T_low - T_low_init:+.3f})  "
          f"u_low = {u_low:.3f} m/s  v_low = {v_low:.3f} m/s  "
          f"tke_sfc = {tke_sfc:.4f}  tke_max = {tke_max:.4f}")

    ok = True
    if not (255.0 < T_low < 270.0):
        print(f"[FAIL] T_low {T_low} out of GABLS1 plausible band [255, 270]")
        ok = False
    # RESPONSE check: the prescribed surface cooling must actually reach the
    # near-surface air (a no-op / uncoupled run would leave T_low unchanged).
    # The surface cools at 0.25 K/hr, so require at least ~0.05 K of cooling per
    # hour integrated (well above round-off, well below the full forced rate).
    min_cooling = 0.05 * args.hours
    if not (T_low < T_low_init - min_cooling):
        print(f"[FAIL] near-surface cooling {T_low_init - T_low:.3f} K < required "
              f"{min_cooling:.3f} K - prescribed T_s not coupled to the column")
        ok = False
    # Turbulence developed above the rest floor (shear-driven, stable BL) yet
    # stays weak and bounded (no runaway). The rest/floor wp2 is ~1e-6 and qke
    # starts near 0, so a >1e-4 peak distinguishes a real run from an unchanged
    # initial column; <5 rules out a blow-up.
    if not (1e-4 < tke_max < 5.0):
        print(f"[FAIL] tke_max {tke_max} outside developed-but-bounded stable-BL "
              "band (1e-4, 5) - turbulence did not develop, ran away, or is negative")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
