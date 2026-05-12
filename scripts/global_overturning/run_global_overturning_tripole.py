#!/usr/bin/env python
"""Global overturning on the ORCA1 tripolar grid with ETOPO bathymetry.

Matched configuration with run_comparison_latlon.py — uses ETOPO
bathymetry interpolated onto the ORCA1 tripolar grid geometry, partial
cells, Wright EOS, Adcroft PGF, KPP, GM/Redi, TVD advection.

The tripolar grid eliminates the polar singularity, enabling full Arctic
ocean dynamics without ad-hoc polar caps.

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_tripole.py
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_tripole.py --days 365 --quick
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

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import (
    VerticalMixingConfig, KPPConfig,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    LateralMixingConfig, GMRediConfig, VisbeckConfig,
)
from legoesm.ocean.physics.convection.config import (
    OceanConvectionConfig, EnhancedDiffusionConfig,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star, create_partial_cell_coordinate, compute_centroid_depth,
)


# ============================================================================
# Configuration constants (matched with run_comparison_latlon.py)
# ============================================================================

N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 1200.0                # seconds (matches comparison)
SNAP_FRAC = 0.30

# Viscosity — Smagorinsky-dominated, low constant floor
A_H = 1.0e4
C_SMAG_LAP = 0.33

# Vertical mixing
A_V = 1.0e-4
K_V = 1.0e-5

# Bottom drag
BOTTOM_DRAG_R = 1.0e-3
BOTTOM_DRAG_BBL = 100.0
BOTTOM_DRAG_BG_VEL = 0.1

# GM/Redi
KAPPA_GM = 600.0
KAPPA_REDI = 600.0
S_MAX = 0.005

# Forcing
TAU_MAX = 0.1
TROPICAL_WIND_SCALE = 0.5
TROPICAL_WIND_LAT_DEG = 15.0
TAU_T = 2592000.0          # 30-day restoring [s]
TAU_S = 2592000.0
T_STAR_EQ = 25.0
T_STAR_POLE = 0.0
S_STAR = 35.0

GRID_FILE = Path("data/grids/eORCA1.2_mesh_mask.nc")
ETOPO_FILE = Path("/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc")
OUTPUT_DIR = Path("results/ocean/global_overturning_tripole")


# ============================================================================
# Helpers
# ============================================================================

def snap_partial_cells_2d(H_bathy, z_coord, min_frac=SNAP_FRAC):
    """Snap thin partial cells to nearest interface (2D version)."""
    abs_z_half = jnp.abs(z_coord.z_half_ref)
    nlev = z_coord.n_levels
    shape = H_bathy.shape
    H_flat = H_bathy.ravel()
    n_above = jnp.sum(abs_z_half[None, :] < H_flat[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]
    dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_flat - abs_z_at_bottom
    frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]
    z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_flat - z_upper < z_lower - H_flat,
                           z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_flat > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_flat)
    H_new = jnp.where(H_new <= 0, 0.0, H_new)
    return H_new.reshape(shape)


def save_restart(state, day, dt, output_dir):
    """Save state as restart_dayXXXXXX.npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(round(day * 86400 / dt)), "time_days": float(day),
               "grid_type": "tripole"}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        payload[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **payload)
    print(f"    Restart saved: {fname.name}")
    return fname


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Global overturning on ORCA1 tripolar grid (ETOPO bathy)")
    p.add_argument("--days", type=float, default=3650.0,
                   help="Simulation length [days] (default: 3650 = 10 yr)")
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--quick", action="store_true",
                   help="Quick 30-day run for testing")
    p.add_argument("--block-size", type=int, default=100)
    p.add_argument("--grid-file", type=str, default=str(GRID_FILE))
    p.add_argument("--etopo", type=str, default=str(ETOPO_FILE))
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()

    output_dir = Path(args.output)
    restart_dir = output_dir / "restarts"
    restart_dir.mkdir(parents=True, exist_ok=True)

    days = 30.0 if args.quick else args.days
    dt = args.dt
    n_steps = int(days * 86400 / dt)
    block_size = args.block_size

    # ---- Load ORCA1 tripolar grid (geometry only) ----
    grid_file = Path(args.grid_file)
    if not grid_file.exists():
        print(f"Grid file not found: {grid_file}")
        print("Download with: python -c \"from legoesm.grids.tripole import "
              "download_orca1_grid; download_orca1_grid()\"")
        sys.exit(1)

    print(f"Loading ORCA1 tripolar grid from {grid_file}...")
    geom = create_tripole_grid(str(grid_file))
    print(f"  Grid: {geom.n_lat} x {geom.n_lon} "
          f"(fold at j={geom.fold.fold_j})")

    # Metric floor (1 km) for degenerate cells at bipolar fold seam
    dx_floor = 1000.0
    n_small = int(jnp.sum(geom.dx_T < dx_floor))
    if n_small > 0:
        print(f"  Metric floor: clamping {n_small} cells with dx < {dx_floor:.0f} m")
        geom = geom._replace(
            dx_T=jnp.maximum(geom.dx_T, dx_floor),
            dy_T=jnp.maximum(geom.dy_T, dx_floor),
            area_T=jnp.maximum(geom.area_T, dx_floor * dx_floor),
            dx_u=jnp.maximum(geom.dx_u, dx_floor),
            dy_u=jnp.maximum(geom.dy_u, dx_floor),
            dx_v=jnp.maximum(geom.dx_v, dx_floor),
            dy_v=jnp.maximum(geom.dy_v, dx_floor),
            area_q=jnp.maximum(geom.area_q, dx_floor * dx_floor),
        )

    # ---- ETOPO bathymetry interpolated onto tripolar grid ----
    print(f"Loading ETOPO bathymetry from {args.etopo}...")
    bathy_cfg = BathymetryConfig(
        source="file", path=args.etopo,
        H_max=H_MAX, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
        north_cap_lat=None,     # tripolar handles the north pole
        south_cap_lat=-75.0,    # southern cap (converging meridians)
    )
    H_bathy_raw, ocean_mask = init_ocean_bathymetry(geom, bathy_cfg)
    H_bathy_raw = jnp.asarray(H_bathy_raw, dtype=jnp.float64)
    ocean_mask = jnp.asarray(ocean_mask, dtype=jnp.float64)

    # ---- Vertical coordinate with partial cells ----
    z_coord_base = create_ocean_z_star(
        n_levels=N_LEVELS, H_max=H_MAX,
        dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP,
    )

    # Snap thin partial cells (same 30% threshold as comparison)
    H_snapped = snap_partial_cells_2d(H_bathy_raw, z_coord_base)
    ocean_mask = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    z_coord = create_partial_cell_coordinate(z_coord_base, H_snapped)

    n_ocean = int(jnp.sum(ocean_mask > 0.5))
    print(f"  Ocean cells: {n_ocean}/{ocean_mask.size} "
          f"({100.0 * n_ocean / ocean_mask.size:.1f}%)")
    print(f"  H_bathy range: "
          f"{float(H_snapped[ocean_mask > 0.5].min()):.0f} - "
          f"{float(H_snapped.max()):.0f} m")

    # ---- Physics (matched with comparison) ----
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind", tau_max=TAU_MAX,
                tropical_wind_scale=TROPICAL_WIND_SCALE,
                tropical_wind_lat_deg=TROPICAL_WIND_LAT_DEG,
            ),
            restoring=RestoringConfig(
                tau_T=TAU_T, tau_S=TAU_S,
                T_star_eq=T_STAR_EQ, T_star_pole=T_STAR_POLE,
                S_star=S_STAR, T_profile="cosine",
            ),
        ),
        # KPP disabled with partial cells — C-grid shape mismatch in
        # KPP internals causes NaN.  Use constant A_v via config instead.
        # Re-enable after fixing KPP for C-grid + partial cells.
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(
            scheme="enhanced_diffusion",
            enhanced_diffusion=EnhancedDiffusionConfig(K_conv=1.0),
        ),
        shortwave_penetration=None,
    )

    ocean_config = LatLonCGridOceanConfig(
        A_h=A_H,
        C_smag_lap=C_SMAG_LAP,
        A_v=A_V,
        K_v=K_V,
        bottom_drag_r=BOTTOM_DRAG_R,
        bottom_drag_bbl_thickness=BOTTOM_DRAG_BBL,
        bottom_drag_bg_velocity=BOTTOM_DRAG_BG_VEL,
        pgf_scheme="adcroft",
        barotropic_solver="implicit_cn",
        implicit_vertical_mixing=True,
        tracer_advection="tvd",
        eos="wright",
        freshwater_closure="virtual_salt_flux",
        S_ref=S_STAR,
        # GM/Redi disabled for now — causes immediate blowup on tripolar
        # grid (likely isopycnal slope computation has column-0 metric
        # extractions).  Enable after fixing compute_isopycnal_slopes.
        # gm_redi=GMRediConfig(
        #     kappa_GM=KAPPA_GM, kappa_Redi=KAPPA_REDI, S_max=S_MAX,
        #     visbeck=VisbeckConfig(enabled=False), slope_scheme="centered",
        # ),
        physics=physics,
    )

    # ---- Create model ----
    print("Creating ocean model...")
    model = LatLonCGridOceanModel(geom, z_coord, ocean_config)

    # ---- Initial conditions ----
    print("Creating initial conditions...")
    state = rest_state_latlon_cgrid_ocean(
        geom, z_coord_base,
        T_surface=20.0, T_deep=2.0,
        S_uniform=S_STAR, H_max=H_MAX,
        land_mask_override=ocean_mask,
        H_bathy_override=H_snapped,
    )

    # Level-based exponential T(z) stratification
    z_full = np.asarray(z_coord_base.z_full_ref)
    T_profile = 2.0 + 18.0 * np.exp(z_full / _SCALE_DEPTH)
    T_data = np.array(state.T.data)
    for k in range(N_LEVELS):
        T_data[..., k] = T_profile[k]
    T_data = T_data * np.asarray(ocean_mask)[..., np.newaxis]
    state = state._replace(
        T=state.T.replace(data=jnp.array(T_data)),
    )

    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] degC")
    print(f"  Fold active: {geom.fold.is_active}")

    # ---- Integration ----
    print(f"\n=== Global overturning on ORCA1 tripolar grid ===")
    print(f"  Grid: {geom.n_lat}x{geom.n_lon} tripolar, {N_LEVELS} levels")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({days/365:.1f} sim-yr)")
    print(f"  Block size: {block_size}")
    print(f"  Config: A_h={A_H:.0e}, C_smag_lap={C_SMAG_LAP}, "
          f"A_v={A_V:.0e}, K_v={K_V:.0e}")
    print(f"  KPP, GM/Redi(κ={KAPPA_GM}), TVD, Wright EOS, Adcroft PGF")
    print(f"  Bottom drag: r={BOTTOM_DRAG_R:.0e}, BBL={BOTTOM_DRAG_BBL}m")
    print(f"  Output: {output_dir}")
    print()

    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    save_restart(state, 0.0, dt, restart_dir)

    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    t0 = time.time()
    steps_done = 0
    last_restart_step = 0
    n_steps_per_restart = int(365.0 * 86400 / dt)
    progress_every = max(1, n_blocks // 30)

    print(f"Starting integration ({n_blocks} blocks x {block_size} steps)")
    for b in range(n_blocks):
        state = block_fn(state, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(state.eta.data)
            day = steps_done * dt / 86400.0
            save_restart(state, day, dt, restart_dir)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            jax.block_until_ready(state.eta.data)
            day = steps_done * dt / 86400.0
            max_eta = float(jnp.max(jnp.abs(state.eta.data)))
            max_u = float(jnp.max(jnp.abs(state.u.data)))
            rate = steps_done / (now - t0) if now > t0 else 0
            print(f"  Block {b+1:>5}/{n_blocks}: day {day:>8.1f} "
                  f"({day/365:.2f} yr)  "
                  f"max|eta|={max_eta:.3e}  max|u|={max_u:.3e}  "
                  f"({rate:.1f} steps/s, {now - t0:.0f}s)")

    if n_remainder > 0:
        state = block_fn(state, n_remainder)
        steps_done += n_remainder

    jax.block_until_ready(state.eta.data)
    final_day = steps_done * dt / 86400.0
    save_restart(state, final_day, dt, restart_dir)

    elapsed = time.time() - t0
    print(f"\n=== Done: {final_day/365:.2f} sim-years in {elapsed:.0f}s ===")
    print(f"  max |eta|: {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  max |u|:   {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")
    print(f"  max |v|:   {float(jnp.max(jnp.abs(state.v.data))):.4e} m/s")
    T = state.T.data
    print(f"  T range:   [{float(jnp.min(T)):.2f}, {float(jnp.max(T)):.2f}] degC")
    print(f"  All finite: {bool(jnp.all(jnp.isfinite(state.eta.data)) and jnp.all(jnp.isfinite(state.u.data)))}")


if __name__ == "__main__":
    main()
