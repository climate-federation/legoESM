#!/usr/bin/env python
"""Global overturning on the ORCA1 tripolar grid.

Runs the global overturning circulation experiment (wind + SST restoring)
on the eORCA1 tripolar grid (362x332, ~1 deg) with the implicit CN
barotropic solver.  Uses ORCA's native bathymetry and land mask instead
of the idealized flat-bottom + polar-cap domain.

This is the Phase 6 validation run for the tripolar grid implementation.
The tripolar grid eliminates the polar singularity, enabling full Arctic
ocean dynamics without ad-hoc polar caps.

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_tripole.py
    JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_tripole.py --days 30 --quick
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.tripole import create_tripole_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.init_latlon_cgrid import (
    rest_state_latlon_cgrid_ocean,
    replace_land_mask,
)
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig,
    create_forcings,
    create_eos_config,
    create_gm_redi_config,
)


OUTPUT_DIR = Path("results/ocean/global_overturning_tripole")
GRID_FILE = Path("data/grids/eORCA1.2_mesh_mask.nc")


def _reconstruct_bathymetry(grid_file: str | Path):
    """Reconstruct bathymetry [m] and land mask from ORCA mesh_mask.

    Returns
    -------
    H_bathy : (n_lat, n_lon) array of ocean depth [m]. Zero on land.
    land_mask : (n_lat, n_lon) array, 1 = ocean, 0 = land.
    """
    import netCDF4

    ds = netCDF4.Dataset(str(grid_file), "r")
    mbathy = np.asarray(ds.variables["mbathy"][0])  # (332, 362) int
    gdept_1d = np.asarray(ds.variables["gdept_1d"][0])  # (75,)

    # tmask at the surface gives the land/ocean mask
    tmask_surf = np.asarray(ds.variables["tmask"][0, 0])  # (332, 362)
    ds.close()

    # Reconstruct depth from level index
    H_bathy = np.zeros_like(mbathy, dtype=np.float64)
    n_lev = len(gdept_1d)
    for j in range(mbathy.shape[0]):
        for i in range(mbathy.shape[1]):
            k = int(mbathy[j, i])
            if k > 0:
                H_bathy[j, i] = gdept_1d[min(k, n_lev - 1)]

    land_mask = tmask_surf.astype(np.float64)

    return H_bathy, land_mask


def _add_stratification(state, z_coord, config: GlobalOverturningConfig):
    """Add exponential temperature stratification."""
    z_full = np.asarray(z_coord.z_full_ref)
    decay = np.exp(z_full / config.T_scale_depth)
    T_profile = config.T_deep + (config.T_surface - config.T_deep) * decay

    T_data = np.array(state.T.data)
    for k in range(z_coord.n_levels):
        T_data[..., k] = T_profile[k]

    return state._replace(
        T=Field(jnp.array(T_data), name="T",
                dims=state.T.dims, units=state.T.units),
    )


def _save_restart(state, day, output_dir):
    npz = {
        "step": int(round(day * 86400 / 600)),
        "time_days": float(day),
        "grid_type": "tripole",
    }
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        npz[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **npz)
    print(f"    Restart saved: {fname.name}")


def _make_step_block(model, dt):
    """JIT-compiled n-step scan of model.step."""
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Global overturning on ORCA1 tripolar grid",
    )
    p.add_argument("--days", type=float, default=3650.0,
                   help="Simulation length [days] (default: 3650 = 10 yr)")
    p.add_argument("--dt", type=float, default=600.0,
                   help="Timestep [s] (default: 600)")
    p.add_argument("--quick", action="store_true",
                   help="Quick 30-day run for testing")
    p.add_argument("--nlev", type=int, default=20,
                   help="Number of vertical levels (default: 20)")
    p.add_argument("--block-size", type=int, default=100,
                   help="Steps per JIT block (default: 100)")
    p.add_argument("--gm-redi", action="store_true",
                   help="Enable GM/Redi mesoscale parameterization")
    p.add_argument("--grid-file", type=str, default=str(GRID_FILE),
                   help="Path to ORCA mesh_mask NetCDF file")
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    days = 30.0 if args.quick else args.days
    dt = args.dt
    n_steps = int(days * 86400 / dt)
    block_size = args.block_size

    # ---- Load ORCA1 tripolar grid ----
    grid_file = Path(args.grid_file)
    if not grid_file.exists():
        print(f"Grid file not found: {grid_file}")
        print("Download with: python -c \"from legoesm.grids.tripole import "
              "download_orca1_grid; download_orca1_grid()\"")
        sys.exit(1)

    print(f"Loading ORCA1 tripolar grid from {grid_file}...")
    geom = create_tripole_grid(str(grid_file))
    print(f"  Grid: {geom.n_lat} x {geom.n_lon} "
          f"(fold at j={geom.fold.fold_j}, cap at j={geom.fold.cap_j})")
    print(f"  Total area: {float(geom.total_area):.4e} m^2")

    # ---- Reconstruct bathymetry from mesh_mask ----
    print("Reconstructing bathymetry from mesh_mask...")
    H_bathy_raw, land_mask_raw = _reconstruct_bathymetry(grid_file)
    print(f"  Ocean cells: {int(np.sum(land_mask_raw > 0))}")
    print(f"  H_bathy range: {H_bathy_raw[land_mask_raw > 0].min():.0f} "
          f"- {H_bathy_raw.max():.0f} m")

    # ---- Vertical coordinate ----
    config = GlobalOverturningConfig(
        use_gm_redi=args.gm_redi,
        H_max=float(H_bathy_raw.max()),
        n_levels=args.nlev,
    )
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels,
        H_max=config.H_max,
        dz_surface=config.dz_surface,
        dz_deep=config.dz_deep,
    )

    # ---- Physics / forcing ----
    # The forcing functions (wind, T-restoring) are latitude-dependent
    # analytical profiles that work on any grid.
    physics = create_forcings("latlon", geom, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=config.A_h, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        eos="linear", eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        barotropic_solver="implicit_cn",
    )

    # ---- Create model ----
    print("Creating ocean model...")
    model = LatLonCGridOceanModel(geom, z_coord, ocean_config)

    # ---- Initial conditions ----
    # Start from rest with stratified T/S, then apply ORCA land mask
    print("Creating initial conditions...")
    state = rest_state_latlon_cgrid_ocean(
        geom, z_coord,
        T_surface=config.T_surface, T_deep=config.T_surface,
        S_uniform=config.S_uniform,
    )

    # Apply ORCA bathymetry and land mask
    # Clamp H_bathy to our z-coordinate range
    H_bathy_clamped = np.clip(H_bathy_raw, 0, config.H_max)
    state = state._replace(
        H_bathy=Field(jnp.array(H_bathy_clamped)),
    )
    state = replace_land_mask(state, jnp.array(land_mask_raw))

    # Add stratification
    state = _add_stratification(state, z_coord, config)

    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] degC")
    print(f"  S uniform: {float(jnp.mean(state.S.data)):.2f} PSU")
    print(f"  Fold active: {geom.fold.is_active}")
    print()

    # ---- Integration ----
    print(f"=== Global overturning on ORCA1 tripolar grid ===")
    print(f"  Grid: {geom.n_lat}x{geom.n_lon} tripolar, {args.nlev} levels")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({days/365:.1f} sim-yr)")
    print(f"  Block size: {block_size} steps ({n_steps // block_size} blocks)")
    print(f"  Barotropic solver: {ocean_config.barotropic_solver}")
    print(f"  GM/Redi: {args.gm_redi}")
    print(f"  Output: {output_dir}")
    print()

    block_fn = _make_step_block(model, dt)

    # Save day-0 restart
    _save_restart(state, 0.0, output_dir)

    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size

    print(f"Starting integration ({n_blocks} blocks x {block_size} steps "
          f"+ {n_remainder} remainder)")
    t0 = time.time()
    last_print = t0
    steps_done = 0
    restart_every_days = 365.0
    n_steps_per_restart = int(restart_every_days * 86400 / dt)
    last_restart_step = 0
    progress_every = max(1, n_blocks // 30)

    for b in range(n_blocks):
        state = block_fn(state, block_size)
        steps_done += block_size

        # Periodic restarts
        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(state.eta.data)
            day = steps_done * dt / 86400.0
            _save_restart(state, day, output_dir)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(state.eta.data)
                day = steps_done * dt / 86400.0
                eta = state.eta.data
                u = state.u.data
                max_eta = float(jnp.max(jnp.abs(eta)))
                max_u = float(jnp.max(jnp.abs(u)))
                elapsed = now - t0
                rate = steps_done / elapsed if elapsed > 0 else 0
                print(f"  Block {b+1:>5}/{n_blocks}: day {day:>8.1f} "
                      f"({day/365:.2f} yr)  "
                      f"max|eta|={max_eta:.3e}  max|u|={max_u:.3e}  "
                      f"({rate:.1f} steps/s, {elapsed:.0f}s elapsed)")
                last_print = now

    # Handle remainder
    if n_remainder > 0:
        state = block_fn(state, n_remainder)
        steps_done += n_remainder

    # Final save
    jax.block_until_ready(state.eta.data)
    final_day = steps_done * dt / 86400.0
    _save_restart(state, final_day, output_dir)

    elapsed = time.time() - t0
    print()
    print(f"=== Done: {final_day/365:.2f} sim-years in {elapsed:.0f}s ===")

    # Final diagnostics
    eta = state.eta.data
    u = state.u.data
    v = state.v.data
    T = state.T.data
    print(f"  max |eta|: {float(jnp.max(jnp.abs(eta))):.4e} m")
    print(f"  max |u|:   {float(jnp.max(jnp.abs(u))):.4e} m/s")
    print(f"  max |v|:   {float(jnp.max(jnp.abs(v))):.4e} m/s")
    print(f"  T range:   [{float(jnp.min(T)):.2f}, {float(jnp.max(T)):.2f}] degC")
    print(f"  All finite: {bool(jnp.all(jnp.isfinite(eta)) and jnp.all(jnp.isfinite(u)))}")


if __name__ == "__main__":
    main()
