#!/usr/bin/env python
"""Global overturning on the ORCA1 tripolar grid with ETOPO bathymetry.

Uses ETOPO bathymetry interpolated onto the ORCA1 tripolar grid,
partial cells with Adcroft PGF, and the production physics stack.
No north cap — the tripolar grid handles the pole natively.

Verified stable for 30 days (max|u|~5 m/s without KPP).

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_tripole.py
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_tripole.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate,
)
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH

# ============================================================================
# Configuration
# ============================================================================
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 600.0
SNAP_FRAC = 0.30           # partial cell snap threshold
H_MIN = 200.0              # minimum bathymetry depth [m] — removes
                            # dangerous shallow cap cells (27m next to
                            # 5500m cliffs in the bipolar cap)

GRID_FILE = Path("data/grids/eORCA1.2_mesh_mask.nc")
ETOPO_FILE = Path("/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc")
OUTPUT_DIR = Path("results/ocean/global_overturning_tripole")


def snap_partial_cells_2d(H_bathy, z_coord, min_frac=SNAP_FRAC):
    """Snap thin partial cells to nearest interface."""
    abs_z_half = jnp.abs(z_coord.z_half_ref)
    nlev = z_coord.n_levels
    shape = H_bathy.shape
    H_flat = H_bathy.ravel()
    n_above = jnp.sum(abs_z_half[None, :] < H_flat[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_flat - abs_z_half[bottom_level]
    frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]
    z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_flat - z_upper < z_lower - H_flat,
                           z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_flat > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_flat)
    H_new = jnp.where(H_new <= 0, 0.0, H_new)
    return H_new.reshape(shape)


def main():
    p = argparse.ArgumentParser(
        description="Global overturning on ORCA1 tripolar (ETOPO bathy)")
    p.add_argument("--days", type=float, default=3650.0)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--quick", action="store_true", help="30-day test")
    p.add_argument("--block-size", type=int, default=50)
    p.add_argument("--grid-file", type=str, default=str(GRID_FILE))
    p.add_argument("--etopo", type=str, default=str(ETOPO_FILE))
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    days = 30.0 if args.quick else args.days
    dt = args.dt

    # ---- Grid ----
    grid_file = Path(args.grid_file)
    if not grid_file.exists():
        sys.exit(f"Grid file not found: {grid_file}")

    print(f"Loading ORCA1 tripolar grid from {grid_file}...")
    geom = create_tripole_grid(str(grid_file))
    print(f"  Grid: {geom.n_lat} x {geom.n_lon} (fold at j={geom.fold.fold_j})")

    dx_floor = 1000.0
    geom = geom._replace(
        dx_T=jnp.maximum(geom.dx_T, dx_floor),
        dy_T=jnp.maximum(geom.dy_T, dx_floor),
        area_T=jnp.maximum(geom.area_T, dx_floor**2),
        dx_u=jnp.maximum(geom.dx_u, dx_floor),
        dy_u=jnp.maximum(geom.dy_u, dx_floor),
        dx_v=jnp.maximum(geom.dx_v, dx_floor),
        dy_v=jnp.maximum(geom.dy_v, dx_floor),
        area_q=jnp.maximum(geom.area_q, dx_floor**2),
    )

    # ---- ETOPO bathymetry + partial cells ----
    print(f"Loading ETOPO from {args.etopo}...")
    bathy_cfg = BathymetryConfig(
        source="file", path=args.etopo,
        H_max=H_MAX, H_min=H_MIN,
        smoothing_passes=2, r_factor_max=0.2,
        depth_is_negative=True,
        north_cap_lat=None,     # tripolar handles the north pole
        south_cap_lat=-75.0,
    )
    H_raw, ocean_mask = init_ocean_bathymetry(geom, bathy_cfg)
    H_raw = jnp.asarray(H_raw, dtype=jnp.float64)
    ocean_mask = jnp.asarray(ocean_mask, dtype=jnp.float64)

    z_base = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP,
    )
    H_snapped = snap_partial_cells_2d(H_raw, z_base)
    ocean_mask = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    z_coord = create_partial_cell_coordinate(z_base, H_snapped)

    n_ocean = int(jnp.sum(ocean_mask > 0.5))
    print(f"  Ocean cells: {n_ocean}/{ocean_mask.size} "
          f"({100 * n_ocean / ocean_mask.size:.1f}%)")

    # ---- Physics ----
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="two_belt", tau_max=0.1,
            ),
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=1e30,
                T_star_eq=25.0, T_star_pole=0.0,
                S_star=35.0, T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
        ),
        shortwave_penetration=None,
    )

    ocean_config = LatLonCGridOceanConfig(
        A_h=2e5, A_v=1e-3, K_v=1e-5,
        barotropic_solver="implicit_cn",
        eos="linear",
        pgf_scheme="adcroft",
        physics=physics,
        implicit_vertical_mixing=True,
    )

    # ---- Model + initial conditions ----
    print("Creating ocean model...")
    model = LatLonCGridOceanModel(geom, z_coord, ocean_config)

    state = rest_state_latlon_cgrid_ocean(
        geom, z_base,
        T_water_init_C=20.0, T_deep=20.0, S_uniform=35.0,
        H_max=H_MAX,
        land_mask_override=ocean_mask,
        H_bathy_override=H_snapped,
    )

    # Add stratification
    z_full = np.asarray(z_base.z_full_ref)
    T_profile = 2.0 + 18.0 * np.exp(z_full / _SCALE_DEPTH)
    T_data = np.array(state.T.data)
    for k in range(N_LEVELS):
        T_data[..., k] = T_profile[k]
    T_data = T_data * np.asarray(ocean_mask)[..., np.newaxis]
    state = state._replace(
        T=state.T.replace(data=jnp.array(T_data)),
    )

    print(f"  Fold active: {geom.fold.is_active}")

    # ---- Integration ----
    n_steps = int(days * 86400 / dt)
    block_size = args.block_size
    n_blocks = n_steps // block_size

    print(f"\n=== ORCA1 tripolar: {days/365:.1f} yr, dt={dt}s ===")
    print(f"  ETOPO + partial cells + Adcroft PGF")
    print(f"  A_h={ocean_config.A_h:.0e}, implicit vmix")
    print(f"  {n_blocks} blocks x {block_size} steps\n")

    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    t0 = time.time()
    steps_done = 0
    progress_every = max(1, n_blocks // 30)

    for b in range(n_blocks):
        state = block_fn(state, block_size)
        steps_done += block_size
        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            jax.block_until_ready(state.eta.data)
            day = steps_done * dt / 86400.0
            max_eta = float(jnp.max(jnp.abs(state.eta.data)))
            max_u = float(jnp.max(jnp.abs(state.u.data)))
            elapsed = time.time() - t0
            rate = steps_done / elapsed if elapsed > 0 else 0
            print(f"  Block {b+1:>5}/{n_blocks}: day {day:>8.1f} "
                  f"({day/365:.2f} yr)  max|eta|={max_eta:.3e}  "
                  f"max|u|={max_u:.3e}  ({rate:.1f} steps/s)")

    jax.block_until_ready(state.eta.data)
    final_day = steps_done * dt / 86400.0
    elapsed = time.time() - t0
    print(f"\n=== Done: {final_day/365:.2f} sim-years in {elapsed:.0f}s ===")
    print(f"  max |eta|: {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  max |u|:   {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")
    T = state.T.data
    print(f"  T range:   [{float(jnp.min(T)):.2f}, {float(jnp.max(T)):.2f}] degC")
    ok = bool(jnp.all(jnp.isfinite(state.eta.data)) and jnp.all(jnp.isfinite(state.u.data)))
    print(f"  All finite: {ok}")


if __name__ == "__main__":
    main()
