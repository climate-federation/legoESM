"""Neutral Ekman boundary-layer benchmark on the legoESM SCM.

Mirrors jax_scm's Andren et al. (1994) reference case:

  * f_c = 1.0e-4 (~45 deg latitude)
  * Geostrophic wind: u_g = 10 m/s, v_g = 0
  * Surface roughness z0 = 0.1 m
  * Neutral stratification: theta = 273.15 K throughout, dry, no surface
    heat or moisture flux
  * Domain: H ~ 1500 m, integrated to several inertial periods

The legoESM equivalent uses MYNN-2.5 turbulence on a pressure-sigma
column tall enough to cover the spin-up Ekman layer (``sigma_top=0.85``
~ 1.3 km).

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/matrix/run_scm_test_matrix.py ekman --hours 6
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
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing


_F_C = 1.0e-4
_U_G = 10.0
_THETA = constants.T_freeze


def build_scm(
    nlev: int = 16, dt: float = 10.0, sigma_top: float = 0.85,
) -> SingleColumnModel:
    """Configure the legoESM SCM for the neutral Ekman setup."""
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top)
    p_s = 1.0e5
    H = 8.0e3
    p_full = sigma_coord.sigma_full * p_s
    z_full = -H * jnp.log(jnp.maximum(p_full / p_s, 1e-6))
    exner = (p_full / constants.p_ref) ** constants.kappa
    T_profile = jnp.full(nlev, _THETA) * exner

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
    )
    return SingleColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=dt,
        T_profile=T_profile, q_v_profile=jnp.zeros(nlev),
        u=_U_G, v=0.0,
        latitude_deg=45.0,
        sigma_top=sigma_top,
        forcing=forcing,
        time_integrator="forward_euler",
    )


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hours", type=float, default=6.0,
                   help="Run duration [hours] (~0.35 inertial periods).")
    p.add_argument("--dt", type=float, default=10.0)
    p.add_argument("--nlev", type=int, default=16)
    p.add_argument("--sigma-top", type=float, default=0.85)


def run(args: argparse.Namespace) -> int:
    scm = build_scm(nlev=args.nlev, dt=args.dt, sigma_top=args.sigma_top)
    nsteps = int(args.hours * 3600.0 / args.dt)
    print(f"[Ekman] nlev={args.nlev}  dt={args.dt}  nsteps={nsteps}")

    scm.run(nsteps=nsteps)
    u_low = float(scm.state.u.data[0, 0, 0, -1])
    v_low = float(scm.state.v.data[0, 0, 0, -1])
    qke_low = float(scm.phys_state.qke[0, -1])
    T_low = float(scm.state.T.data[0, 0, 0, -1])
    speed_low = (u_low ** 2 + v_low ** 2) ** 0.5
    veer = jnp.rad2deg(jnp.arctan2(v_low, u_low))
    print(f"[Ekman] T_low={T_low:.3f} K  |V|_low={speed_low:.3f} m/s  "
          f"veer={float(veer):.2f}deg  qke_low={qke_low:.4f}")

    ok = True
    if not (272.0 < T_low < 274.0):
        print(f"[FAIL] T_low {T_low} out of neutral exner-projected band [272, 274]")
        ok = False
    if speed_low > _U_G:
        print(f"[FAIL] surface wind {speed_low} exceeds geostrophic {_U_G}")
        ok = False
    if not (0.0 < float(veer) < 45.0):
        print(f"[FAIL] veer angle {veer} outside Ekman band (0, 45)")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
