"""MPAS realistic-bathymetry GO spinup on ico4 with ETOPO.

P6 of ``docs/ocean_experiments/realistic_geometry_mpas_plan.md`` —
the headline validation experiment.  Loads ETOPO bathymetry onto a
Voronoi mesh, applies MEO r-factor smoothing, builds the
partial-cell coordinate, and integrates a wind-driven GO spinup
with the full P3-P5 partial-cell stack active (Adcroft PGF, hybrid
vertex thickness, donor-cell continuity, min-rule edge thickness,
bottom drag at maxLevelEdgeBot).

Mirrors the lat-lon Phase 4 Wolfe-Cessi spinup
(``run_phase4_wolfe_cessi_spinup.py``); the lat-lon equivalent ran
100 yrs and confirmed equilibrated state (max|u|=1.21 m/s).

Default config is **smoke-test scale** (subdivision=4, 30 days) so
a developer can verify the script runs end-to-end without burning
HPC budget.  Production runs use the documented flags:

    # 5-yr ico4 GO spinup (the plan-spec headline run):
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/run_mpas_etopo_spinup.py \\
        --subdivision 7 --years 5 --diag-every-days 30

    # Smoke test (default):
    JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/run_mpas_etopo_spinup.py

Acceptance gates (production run):
- Integration completes ``years`` sim-years without NaN.
- ``max|u|`` stays bounded (< 2 m/s) once spun up.
- Time-mean V_baro grid-noise σ < 3× the implicit-CN flat-bottom
  baseline (1.92e-2 m/s per ``project_mpas_barotropic_noise.md``).
- AMOC magnitude in the observed range (15-25 Sv).
- No spurious deep flow under western boundary topography (visual).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


def make_physics(tau_max: float):
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="cosine_latitude", tau_max=tau_max,
            ),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
    )


def run(args):
    print(f"[mpas-etopo] subdivision={args.subdivision}, "
          f"H_max={args.H_max} m, n_levels={args.n_levels}")
    mesh = create_voronoi_mesh(subdivision_level=args.subdivision)
    z_coord = create_ocean_z_star(n_levels=args.n_levels, H_max=args.H_max)

    if args.etopo_path:
        bathy_cfg = BathymetryConfig(
            source="file", path=args.etopo_path,
            H_max=args.H_max, H_min=args.H_min,
            smoothing_passes=args.smoothing_passes,
            enforce_straits=True,
            r_factor_max=args.r_factor_max,
            depth_is_negative=True,
        )
        print(f"[mpas-etopo] loading ETOPO from {args.etopo_path}")
    else:
        # Idealized fallback (no ETOPO file available): test the full
        # pipeline with the lat-threshold land mask.
        bathy_cfg = None
        print("[mpas-etopo] no --etopo-path given; using idealized bathymetry")

    if bathy_cfg is not None:
        state = rest_state_mpas_ocean(
            mesh, z_coord, bathymetry=bathy_cfg,
            T_surface=args.T_surf, T_deep=args.T_deep, S_uniform=args.S_uniform,
        )
    else:
        state = rest_state_mpas_ocean(
            mesh, z_coord, H_max=args.H_max,
            T_surface=args.T_surf, T_deep=args.T_deep, S_uniform=args.S_uniform,
        )

    H_bathy = state.H_bathy.data
    pc_coord = create_partial_cell_coordinate(z_coord, H_bathy)
    print(f"[mpas-etopo] nCells={mesh.nCells}, ocean cells = "
          f"{int(jnp.sum(state.land_mask.data > 0.5))}, "
          f"H range = [{float(jnp.min(jnp.where(state.land_mask.data > 0.5, H_bathy, jnp.inf))):.0f}, "
          f"{float(jnp.max(H_bathy)):.0f}] m")

    cfg = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        A_h=args.A_h, A_v=1.0e-3, K_v=1.0e-4, K_h=args.K_h,
        bottom_drag_r=1.1e-3,
        physics=make_physics(args.tau_max),
        barotropic_implicit_pcg_tol=1.0e-10,
        barotropic_implicit_pcg_maxiter=300,
        min_water_column_m=1.0,
        pgf_scheme="adcroft",
        pv_scheme="enstrophy",
        tracer_advection="upwind",
    )
    model = MPASOceanModel(mesh, pc_coord, cfg)

    n_days = args.years * 365.25
    n_steps = int(n_days * 86400.0 / args.dt)
    diag_stride = max(1, int(args.diag_every_days * 86400.0 / args.dt))
    print(f"[mpas-etopo] dt={args.dt}s, n_steps={n_steps}, "
          f"diag every {args.diag_every_days} days")

    t0 = time.time()
    s = state
    for k in range(n_steps):
        s = model.step(s, dt=args.dt)
        if (k + 1) % diag_stride == 0:
            sim_days = (k + 1) * args.dt / 86400.0
            mu = float(jnp.max(jnp.abs(s.u.data)))
            me = float(jnp.max(jnp.abs(s.eta.data)))
            mt = float(jnp.max(s.T.data))
            elapsed = time.time() - t0
            print(f"  t={sim_days:7.1f}d : max|u|={mu:.3e} m/s, "
                  f"max|eta|={me:.3e} m, max T={mt:.2f} C, "
                  f"wall={elapsed:.1f}s")
            if not (np.isfinite(mu) and np.isfinite(me)):
                raise RuntimeError(
                    f"State went non-finite at t={sim_days}d — "
                    f"investigate stability before proceeding"
                )

    print(f"[mpas-etopo] done in {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subdivision", type=int, default=4,
                   help="Voronoi subdivision level (4=2562, 7=163842 cells)")
    p.add_argument("--n-levels", dest="n_levels", type=int, default=20)
    p.add_argument("--H-max", dest="H_max", type=float, default=5500.0)
    p.add_argument("--H-min", dest="H_min", type=float, default=10.0)
    p.add_argument("--T-surf", dest="T_surf", type=float, default=20.0)
    p.add_argument("--T-deep", dest="T_deep", type=float, default=2.0)
    p.add_argument("--S-uniform", dest="S_uniform", type=float, default=35.0)
    p.add_argument("--years", type=float, default=0.083,
                   help="Simulation years (default ~30 days for smoke test)")
    p.add_argument("--dt", type=float, default=600.0,
                   help="Baroclinic timestep [s]")
    p.add_argument("--diag-every-days", dest="diag_every_days", type=float, default=5.0)
    p.add_argument("--A-h", dest="A_h", type=float, default=1.0e5)
    p.add_argument("--K-h", dest="K_h", type=float, default=1.0e3)
    p.add_argument("--tau-max", dest="tau_max", type=float, default=0.05)
    p.add_argument("--smoothing-passes", dest="smoothing_passes", type=int, default=2)
    p.add_argument("--r-factor-max", dest="r_factor_max", type=float, default=0.3)
    p.add_argument("--etopo-path", dest="etopo_path", type=str, default=None,
                   help="Path to ETOPO/GEBCO NetCDF; idealized if omitted")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
