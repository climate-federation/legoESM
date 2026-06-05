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

from legoesm import constants
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics.turbulence.config import (
    MYNN25Config,
    SurfaceLayerConfig,
)
from legoesm.atmosphere.scm import SingleColumnModel
from legoesm.atmosphere.scm_forcing import SCMForcing


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


def build_scm(nlev: int = 32, dt: float = 5.0, sigma_top: float = 0.7) -> SingleColumnModel:
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
        turbulence=TurbulenceConfig(
            scheme="mynn25",
            mynn25=MYNN25Config(
                surface=SurfaceLayerConfig(
                    z0=0.1,
                    Cd_neutral=1.5e-3,
                    Ch_neutral=1.5e-3,
                ),
            ),
        ),
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


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hours", type=float, default=9.0,
                   help="Run duration [hours] (default 9.0, Cuxart 2006).")
    p.add_argument("--dt", type=float, default=5.0,
                   help="Physics time step [s].")
    p.add_argument("--nlev", type=int, default=32,
                   help="Vertical grid count.")
    p.add_argument("--sigma-top", type=float, default=0.7,
                   help="Top sigma value (smaller = taller domain; 0.7 ~ 2.5 km).")


def run(args: argparse.Namespace) -> int:
    scm = build_scm(nlev=args.nlev, dt=args.dt, sigma_top=args.sigma_top)
    nsteps = int(args.hours * 3600.0 / args.dt)
    print(f"[GABLS1] nlev={args.nlev}  dt={args.dt}  nsteps={nsteps}  hours={args.hours}")

    save_every = max(1, nsteps // 60)
    final, hist = scm.run(nsteps=nsteps, save_every=save_every)

    T_low = float(final.T.data[0, 0, 0, -1])
    qke_low = float(scm.phys_state.qke[0, -1])
    u_low = float(final.u.data[0, 0, 0, -1])
    v_low = float(final.v.data[0, 0, 0, -1])
    print(f"[GABLS1] final T_low = {T_low:.3f} K  "
          f"u_low = {u_low:.3f} m/s  v_low = {v_low:.3f} m/s  "
          f"qke_low = {qke_low:.4f} m2/s2")

    ok = True
    if not (255.0 < T_low < 270.0):
        print(f"[FAIL] T_low {T_low} out of GABLS1 plausible band [255, 270]")
        ok = False
    if not (0.0 < qke_low < 5.0):
        print(f"[FAIL] qke_low {qke_low} out of plausible BL band [0, 5]")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
