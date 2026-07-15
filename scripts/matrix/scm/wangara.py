"""Wangara Day 33 convective boundary-layer benchmark (legoESM SCM).

Mirrors the jax_scm Wangara case (Pierzyna 2026) with a simplified
initial profile.  Drives the SCM with the canonical Wangara surface
forcing:

  * Cosine-shaped surface kinematic heat flux peaking at 13:00 local
    time: ``w'th'(t) = 0.216 * cos(((t/3600)-13)/11 * pi)`` [K m/s].
  * Cosine-shaped surface moisture flux peaking at 13:00 local time:
    ``w'qv'(t) = 2.29e-5 * cos(((t/3600)-13)/11 * pi)`` [(kg/kg) m/s].
  * f_c = 2*Omega*sin(-34.5 deg) (southern-hemisphere; negative).
  * Height-dependent geostrophic wind (decreasing easterly).
  * Initial theta ~= 277 K (Wangara sounding average; sounding CSV
    import skipped for v1).

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/matrix/run_scm_test_matrix.py wangara --hours 4
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


_LATITUDE_DEG = -34.5
_F_C = 2.0 * constants.Omega * jnp.sin(jnp.deg2rad(_LATITUDE_DEG))
_THETA_INIT = 277.0
_T_START_S = 9.0 * 3600.0


def _w_th_s(t_s):
    """Wangara surface kinematic sensible-heat flux [K m/s]."""
    return 0.216 * jnp.cos(((t_s / 3600.0) - 13.0) / 11.0 * jnp.pi)


def _w_qv_s(t_s):
    """Wangara surface kinematic moisture flux [(kg/kg) m/s]."""
    return 2.29e-5 * jnp.cos(((t_s / 3600.0) - 13.0) / 11.0 * jnp.pi)


def _u_geo_profile(nlev: int, z_full: jnp.ndarray) -> jnp.ndarray:
    """Wangara geostrophic-wind profile (easterly, weakens with height)."""
    return jnp.where(
        z_full < 1000.0,
        -5.5 + 2.9e-3 * z_full,
        -2.6 + 1.4e-3 * (z_full - 1000.0),
    )


def build_scm(
    nlev: int = 24, dt: float = 5.0, sigma_top: float = 0.78,
) -> SingleColumnModel:
    """Configure the legoESM SCM for the Wangara setup."""
    from legoesm.grids.vertical import create_sigma_coordinate
    sigma_coord = create_sigma_coordinate(nlev, sigma_top=sigma_top)
    p_s = 1.0e5
    H = 8.0e3
    p_full = sigma_coord.sigma_full * p_s
    z_full = -H * jnp.log(jnp.maximum(p_full / p_s, 1e-6))
    exner = (p_full / constants.p_ref) ** constants.kappa
    T_profile = jnp.full(nlev, _THETA_INIT) * exner

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(
            scheme="mynn25",
            mynn25=MYNN25Config(
                surface=SurfaceLayerConfig(
                    z0=0.01,
                    Cd_neutral=1.5e-3,
                    Ch_neutral=0.0,
                ),
            ),
        ),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    u_geo = _u_geo_profile(nlev, z_full)
    forcing = SCMForcing(
        f_c=float(_F_C),
        u_geo=lambda t: u_geo,
        v_geo=lambda t: jnp.zeros(nlev),
        prescribe="fluxes",
        w_th_s=_w_th_s,
        w_qv_s=_w_qv_s,
    )
    return SingleColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=dt,
        T_profile=T_profile,
        q_v_profile=jnp.full(nlev, 5e-3),
        u=-5.5, v=0.0,
        latitude_deg=_LATITUDE_DEG,
        sigma_top=sigma_top,
        forcing=forcing,
        time_integrator="forward_euler",
        t0_seconds=_T_START_S,
    )


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hours", type=float, default=4.0,
                   help="Run duration [hours] from 09:00 local.")
    p.add_argument("--dt", type=float, default=5.0)
    p.add_argument("--nlev", type=int, default=24)
    p.add_argument("--sigma-top", type=float, default=0.78)


def run(args: argparse.Namespace) -> int:
    scm = build_scm(nlev=args.nlev, dt=args.dt, sigma_top=args.sigma_top)
    nsteps = int(args.hours * 3600.0 / args.dt)
    print(f"[Wangara] nlev={args.nlev}  dt={args.dt}  nsteps={nsteps}")

    scm.run(nsteps=nsteps)
    T_low = float(scm.state.T.data[0, 0, 0, -1])
    qke_low = float(scm.phys_state.qke[0, -1])
    qke_max = float(scm.phys_state.qke.max())
    print(f"[Wangara] T_low={T_low:.3f}  qke_low={qke_low:.4f}  "
          f"qke_max={qke_max:.4f}")

    ok = True
    if not (270.0 < T_low < 290.0):
        print(f"[FAIL] T_low {T_low} out of Wangara band [270, 290]")
        ok = False
    if not (qke_max > qke_low):
        print(f"[FAIL] qke_max {qke_max} not above surface qke {qke_low} "
              "- convective BL did not develop")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
