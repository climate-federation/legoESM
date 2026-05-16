#!/usr/bin/env python
"""Global overturning on the ORCA1 tripolar grid.

Uses ORCA's native bathymetry with the production physics stack
(Wright EOS, Adcroft PGF, implicit vertical mixing, Smagorinsky,
wind + SST restoring).  No GM/Redi or KPP yet (pending tripolar
validation of isopycnal slopes and KPP C-grid support).

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
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean, replace_land_mask
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_eos_config,
)
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig,
)
from legoesm.ocean.vertical import create_ocean_z_star

# ============================================================================
# Configuration
# ============================================================================
H_MAX = 5500.0
N_LEVELS = 20
DT = 600.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
NORTH_CAP_LAT = 80.0   # mask distorted cap cells above 80°N
SOUTH_CAP_LAT = -75.0  # mask converging-meridian cells below 75°S
H_SHELF_MIN = 500.0    # mask shallow shelf cells (z-star thin-cell guard)

GRID_FILE = Path("data/grids/eORCA1.2_mesh_mask.nc")
OUTPUT_DIR = Path("results/ocean/global_overturning_tripole")


def _reconstruct_bathymetry(grid_file):
    """Reconstruct H_bathy and land mask from ORCA mesh_mask."""
    import netCDF4
    ds = netCDF4.Dataset(str(grid_file), "r")
    mbathy = np.asarray(ds.variables["mbathy"][0])
    gdept_1d = np.asarray(ds.variables["gdept_1d"][0])
    tmask_surf = np.asarray(ds.variables["tmask"][0, 0])
    ds.close()
    H_bathy = np.zeros_like(mbathy, dtype=np.float64)
    n_lev = len(gdept_1d)
    for j in range(mbathy.shape[0]):
        for i in range(mbathy.shape[1]):
            k = int(mbathy[j, i])
            if k > 0:
                H_bathy[j, i] = gdept_1d[min(k, n_lev - 1)]
    return H_bathy, tmask_surf.astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=float, default=3650.0)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--quick", action="store_true", help="30-day test")
    p.add_argument("--block-size", type=int, default=100)
    p.add_argument("--grid-file", type=str, default=str(GRID_FILE))
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    days = 30.0 if args.quick else args.days
    dt = args.dt

    # ---- Grid ----
    grid_file = Path(args.grid_file)
    if not grid_file.exists():
        print(f"Grid file not found: {grid_file}")
        sys.exit(1)

    print(f"Loading ORCA1 tripolar grid from {grid_file}...")
    geom = create_tripole_grid(str(grid_file))
    print(f"  Grid: {geom.n_lat} x {geom.n_lon} (fold at j={geom.fold.fold_j})")

    # Metric floor for degenerate cells at bipolar fold seam
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

    # ---- Bathymetry from ORCA mesh_mask ----
    print("Reconstructing bathymetry from mesh_mask...")
    H_bathy_raw, land_mask = _reconstruct_bathymetry(grid_file)
    Hc = np.clip(H_bathy_raw, 0, H_MAX)
    lat_deg = np.asarray(geom.lat_T) * 180 / np.pi

    # Polar caps
    if NORTH_CAP_LAT is not None:
        n_north = int(np.sum((lat_deg > NORTH_CAP_LAT) & (land_mask > 0)))
        if n_north > 0:
            print(f"  North cap: masking {n_north} cells above {NORTH_CAP_LAT}°N")
        Hc[lat_deg > NORTH_CAP_LAT] = 0.0
        land_mask[lat_deg > NORTH_CAP_LAT] = 0.0
    if SOUTH_CAP_LAT is not None:
        n_south = int(np.sum((lat_deg < SOUTH_CAP_LAT) & (land_mask > 0)))
        if n_south > 0:
            print(f"  South cap: masking {n_south} cells below {SOUTH_CAP_LAT}°S")
        Hc[lat_deg < SOUTH_CAP_LAT] = 0.0
        land_mask[lat_deg < SOUTH_CAP_LAT] = 0.0

    # Shelf mask
    too_shallow = (Hc > 0) & (Hc < H_SHELF_MIN)
    n_shelf = int(np.sum(too_shallow))
    if n_shelf > 0:
        print(f"  Shelf: masking {n_shelf} cells with H < {H_SHELF_MIN:.0f} m")
    Hc[too_shallow] = 0.0
    land_mask[too_shallow] = 0.0

    n_ocean = int(np.sum(land_mask > 0.5))
    print(f"  Ocean cells: {n_ocean}")

    # ---- Vertical coordinate ----
    config = GlobalOverturningConfig(H_max=H_MAX, n_levels=N_LEVELS)
    z_coord = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP,
    )
    eos_config = create_eos_config(config)

    # ---- Physics ----
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind", tau_max=0.1,
                tropical_wind_scale=0.5, tropical_wind_lat_deg=15.0,
            ),
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=2592000.0,
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
        A_h=1e4, C_smag_lap=0.33, A_v=1e-4, K_v=1e-5,
        bottom_drag_r=1e-3, bottom_drag_bbl_thickness=100.0,
        bottom_drag_bg_velocity=0.1,
        barotropic_solver="implicit_cn",
        implicit_vertical_mixing=True,
        eos="linear", eos_linear=eos_config,
        physics=physics,
    )

    # ---- Model + initial conditions ----
    print("Creating ocean model...")
    model = LatLonCGridOceanModel(geom, z_coord, ocean_config)

    state = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        T_surface=config.T_surface, T_deep=config.T_surface,
        S_uniform=config.S_uniform,
    )
    state = state._replace(H_bathy=Field(jnp.array(Hc)))
    state = replace_land_mask(state, jnp.array(land_mask), grid=geom)

    # Add stratification
    from legoesm.ocean.eos import scale_depth as _SD
    z_full = np.asarray(z_coord.z_full_ref)
    T_profile = config.T_deep + (config.T_surface - config.T_deep) * np.exp(z_full / _SD)
    T_data = np.array(state.T.data)
    for k in range(N_LEVELS):
        T_data[..., k] = T_profile[k]
    state = state._replace(
        T=Field(jnp.array(T_data), name="T", dims=state.T.dims, units=state.T.units),
    )

    print(f"  Fold active: {geom.fold.is_active}")

    # ---- Integration ----
    n_steps = int(days * 86400 / dt)
    block_size = args.block_size
    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size

    print(f"\n=== ORCA1 tripolar: {days/365:.1f} yr, dt={dt}s ===")
    print(f"  A_h={ocean_config.A_h:.0e}, C_smag={ocean_config.C_smag_lap}")
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

    if n_remainder > 0:
        state = block_fn(state, n_remainder)
        steps_done += n_remainder

    jax.block_until_ready(state.eta.data)
    final_day = steps_done * dt / 86400.0
    elapsed = time.time() - t0
    print(f"\n=== Done: {final_day/365:.2f} sim-years in {elapsed:.0f}s ===")
    print(f"  max |eta|: {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  max |u|:   {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")
    T = state.T.data
    print(f"  T range:   [{float(jnp.min(T)):.2f}, {float(jnp.max(T)):.2f}] degC")
    print(f"  All finite: {bool(jnp.all(jnp.isfinite(state.eta.data)) and jnp.all(jnp.isfinite(state.u.data)))}")


if __name__ == "__main__":
    main()
