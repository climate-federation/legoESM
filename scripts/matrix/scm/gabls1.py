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
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
from legoesm.atmosphere.physics.turbulence.config import (
    MYNN25Config,
    SurfaceLayerConfig,
)
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing

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
    # ACTUAL integrated time: int() truncation means nsteps*dt can be < the
    # requested args.hours*3600 for a non-dividing dt. The prescribed surface
    # T_s(t)=265+_T_S_RATE*t and every duration-dependent gate threshold below
    # must use this ACTUAL end time, not the requested args.hours.
    t_end_hours = nsteps * args.dt / 3600.0
    # Capture the INITIAL near-surface temperature and the INITIAL column-mean
    # temperature so the pass gate can require a genuine cooling RESPONSE (not
    # just a plausible final value an unchanged column would already satisfy).
    T_low_init = float(scm.state.T.data[0, 0, 0, -1])
    T_col_init = float(np.mean(np.asarray(scm.state.T.data[0, 0, 0])))
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
    # THERMAL-COUPLING check (the non-vacuous one). radiation/convection/
    # microphysics are all "none" in build_scm, so the prescribed cooling surface
    # is the ONLY diabatic sink: the COLUMN-MEAN temperature can change solely
    # through the surface heat flux, making column-mean cooling a direct test
    # that the prescribed T_s actually reached the column (a thermally-decoupled
    # run nets ~0). NB this is column-mean, NOT the single lowest cell: that cell
    # sits ~37.6 m up (nlev=32/sigma_top=0.7) and, being bottom-trapped under the
    # growing inversion, nets ~zero / slight warming even when the column cools —
    # so a fixed lowest-cell cooling threshold is unphysical there.
    # The first ~2 h are an initial mixed-layer-adjustment transient that briefly
    # warms the near-surface, so the cooling signal is only resolvable once the
    # run is long enough. Measured column-mean ΔT (mynn25 default, sweep in
    # scripts/tmp/_probe_gabls1_thermal.py): +0.002 K @ 1 h, -0.0002 @ 2 h,
    # -0.013 @ 4 h, -0.075 @ the 9 h Cuxart default. Enforce only for runs long
    # enough to clear the transient; below that, band + TKE still apply.
    _COOLING_RESOLVABLE_HOURS = 4.0      # below this the transient masks cooling
    _MIN_COLMEAN_COOLING_K = 5.0e-3      # floor (>>roundoff, <<the -0.013 @ 4 h)
    T_col = float(np.mean(np.asarray(final.T.data[0, 0, 0])))
    if t_end_hours >= _COOLING_RESOLVABLE_HOURS:
        if not (T_col < T_col_init - _MIN_COLMEAN_COOLING_K):
            print(f"[FAIL] column-mean air did not cool (ΔT_colmean="
                  f"{T_col - T_col_init:+.4f} K over {t_end_hours:g} h) - "
                  "prescribed surface cooling not coupled to the column")
            ok = False
    else:
        print(f"[GABLS1] t_end={t_end_hours:g} h < {_COOLING_RESOLVABLE_HOURS:g}: "
              "too short to resolve the surface-cooling response (initial "
              "transient); thermal-coupling check skipped (band + TKE still hold)")
    # STABLE-STRATIFICATION sanity (mirrors the oracle-validated
    # tests/validation/test_scm_gabls1.py::test_gabls1_lowest_cell_in_stable_band):
    # the lowest air must end WARMER than the prescribed cooled surface
    # T_s(t_end)=265-0.25*hours (a stable BL, not an unphysical super-inversion).
    # T_s_end is the EXACT prescribed surface temperature at the final step,
    # using the same _T_S_0 / _T_S_RATE constants the forcing applies (no
    # duplicated 265/0.25 literals to drift).
    T_s_end = _T_S_0 + _T_S_RATE * (t_end_hours * 3600.0)
    if not (T_low > T_s_end - 1.0):
        print(f"[FAIL] lowest air T_low {T_low:.3f} K colder than prescribed "
              f"surface T_s(t_end) {T_s_end:.3f} K by >1 K - unphysical SBL / "
              "surface decoupled from the column")
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
