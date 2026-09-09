#!/usr/bin/env python
"""MPAS global overturning with ETOPO bathymetry at ico5 (~1°).

Uses the production recipe established on 2026-05-10:
  - Real ETOPO bathymetry with 30% partial-cell snap
  - 20 stretched vertical levels (dz_surface=20m, dz_deep=500m)
  - Implicit vertical mixing (backward Euler) — THE key fix
  - Implicit Crank-Nicolson barotropic solver
  - KPP boundary-layer mixing (K_conv=0, shear-driven)
  - Wind + T/S restoring forcing
  - A_h=5e4, A_v=5e-3, K_v=1e-4, bottom_drag_r=1e-3

This is the first multi-year MPAS ocean run with realistic bathymetry.
Target: 10+ years stable spinup demonstrating global overturning
circulation on an unstructured Voronoi mesh.

Usage:
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_mpas_etopo.py --years 10
    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_mpas_etopo.py --years 5 --restart results/ocean/global_overturning_mpas_etopo/restarts/restart_day001825.npz
"""
from __future__ import annotations

import argparse
import csv
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.bathymetry import BathymetryConfig, load_bathymetry_mpas
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
from legoesm.ocean.init_mpas import rest_state_mpas_ocean, reconstruct_cell_velocity
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig, RestoringConfig, SurfaceForcingConfig,
)
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig, KPPConfig
from legoesm.ocean.physics.lateral_mixing.config import (
    LateralMixingConfig, GMRediConfig, VisbeckConfig,
)
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("results/ocean/global_overturning_mpas_etopo_tropred")
SUBDIVISION = 5       # ico5 ~ 120 km ~ 1°
N_LEVELS = 20
H_MAX = 5500.0
DZ_SURFACE = 20.0
DZ_DEEP = 500.0
DT = 300.0            # seconds
SNAP_FRAC = 0.30


# ============================================================================
# Helpers
# ============================================================================

def snap_partial_cells(H_bathy, z_coord, min_frac=SNAP_FRAC):
    """Round H_bathy to nearest interface when bottom partial cell < min_frac."""
    abs_z_half = jnp.abs(z_coord.z_half_ref)
    nlev = z_coord.n_levels
    n_above = jnp.sum(abs_z_half[None, :] < H_bathy[:, None], axis=1)
    bottom_level = jnp.clip(n_above - 1, 0, nlev - 1)
    abs_z_at_bottom = abs_z_half[bottom_level]
    dz_at_bottom = z_coord.dz_ref[bottom_level]
    partial_thick = H_bathy - abs_z_at_bottom
    frac = partial_thick / jnp.maximum(dz_at_bottom, 1e-10)
    z_upper = abs_z_half[bottom_level]
    z_lower = abs_z_half[jnp.minimum(bottom_level + 1, nlev)]
    H_snapped = jnp.where(H_bathy - z_upper < z_lower - H_bathy, z_upper, z_lower)
    needs_snap = (frac < min_frac) & (frac > 0) & (H_bathy > 0)
    H_new = jnp.where(needs_snap, H_snapped, H_bathy)
    return jnp.where(H_new <= 0, 0.0, H_new)


def save_restart(state, day, output_dir):
    """Save state as restart_dayXXXXXX.npz."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(round(day * 86400 / DT)), "time_days": float(day),
               "grid_type": "mpas", "subdivision": SUBDIVISION}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        payload[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
    np.savez_compressed(fname, **payload)
    return fname


def load_restart(restart_path, template_state):
    """Load restart npz into template state."""
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = Field(data=arr, name=obj.name, dims=obj.dims, units=obj.units)
    return template_state._replace(**replacements), restart_day


def save_snapshot(state, mesh, ocean_mask, day, output_dir):
    """Save 6-panel diagnostic PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_np = np.asarray(ocean_mask) > 0.5
    lon = np.degrees(np.asarray(mesh.lonCell))
    lat = np.degrees(np.asarray(mesh.latCell))

    u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
    speed = np.asarray(jnp.sqrt(u_east**2 + v_north**2))
    eta = np.asarray(state.eta.data)
    sst = np.asarray(state.T.data[:, 0])

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Surface speed
    spd = np.where(mask_np, speed[:, 0], np.nan)
    vmax = max(0.01, np.nanpercentile(spd, 99))
    sc = axes[0, 0].scatter(lon, lat, c=spd, s=3, cmap="magma", vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=axes[0, 0], label="m/s"); axes[0, 0].set_title("Surface speed")

    # SSH
    eta_p = np.where(mask_np, eta, np.nan)
    vm = max(0.01, np.nanmax(np.abs(eta_p)))
    sc = axes[0, 1].scatter(lon, lat, c=eta_p, s=3, cmap="RdBu_r", vmin=-vm, vmax=vm)
    plt.colorbar(sc, ax=axes[0, 1], label="m"); axes[0, 1].set_title("SSH")

    # SST
    sst_p = np.where(mask_np, sst, np.nan)
    sc = axes[0, 2].scatter(lon, lat, c=sst_p, s=3, cmap="RdYlBu_r")
    plt.colorbar(sc, ax=axes[0, 2], label="°C"); axes[0, 2].set_title("SST")

    # Max-depth speed
    spd_max = np.where(mask_np, np.max(speed, axis=1), np.nan)
    sc = axes[1, 0].scatter(lon, lat, c=spd_max, s=3, cmap="magma",
                            vmin=0, vmax=max(0.01, np.nanpercentile(spd_max, 99)))
    plt.colorbar(sc, ax=axes[1, 0], label="m/s"); axes[1, 0].set_title("Max-depth speed")

    # SSS
    sss = np.asarray(state.S.data[:, 0])
    sss_p = np.where(mask_np, sss, np.nan)
    sc = axes[1, 1].scatter(lon, lat, c=sss_p, s=3, cmap="YlGnBu")
    plt.colorbar(sc, ax=axes[1, 1], label="PSU"); axes[1, 1].set_title("SSS")

    # Deep T (level 15 ~ 3300m)
    deep_lev = min(15, state.T.data.shape[1] - 1)
    T_deep = np.asarray(state.T.data[:, deep_lev])
    T_deep_p = np.where(mask_np, T_deep, np.nan)
    sc = axes[1, 2].scatter(lon, lat, c=T_deep_p, s=3, cmap="RdYlBu_r")
    plt.colorbar(sc, ax=axes[1, 2], label="°C")
    axes[1, 2].set_title(f"T at level {deep_lev}")

    for ax in axes.flat:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")

    fig.suptitle(f"MPAS ETOPO global overturning — year {day/365.25:.1f}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"snapshot_day{int(round(day)):06d}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="MPAS global overturning with ETOPO at ico5.")
    p.add_argument("--years", type=float, default=10.0)
    p.add_argument("--restart", default=None)
    p.add_argument("--checkpoint-years", type=float, default=1.0)
    p.add_argument("--etopo", default=os.environ.get("LEGOESM_ETOPO_PATH", "data/bathymetry/etopo_1deg.nc"))
    args = p.parse_args()

    outdir = OUTPUT_DIR
    restart_dir = outdir / "restarts"
    snapshot_dir = outdir / "snapshots"
    restart_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # --- Grid ---
    print(f"=== MPAS global overturning: ETOPO ico{SUBDIVISION}, {args.years} years ===")
    mesh = create_voronoi_mesh(subdivision_level=SUBDIVISION)
    z_coord = create_ocean_z_star(n_levels=N_LEVELS, H_max=H_MAX,
                                  dz_surface=DZ_SURFACE, dz_deep=DZ_DEEP)

    # --- Bathymetry ---
    bathy_cfg = BathymetryConfig(
        source="file", path=args.etopo,
        H_max=H_MAX, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
    )
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    n_ocean = int(jnp.sum(ocean_mask_new > 0.5))
    print(f"  Mesh: nCells={mesh.nCells}, nEdges={mesh.nEdges}")
    print(f"  Ocean cells: {n_ocean}/{mesh.nCells}")
    print(f"  Vertical: {N_LEVELS} levels, dz_sfc={DZ_SURFACE}m, dz_deep={DZ_DEEP}m")

    # --- Physics ---
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(
                wind_profile="global_wind", tau_max=0.1,
                tropical_wind_scale=0.5, tropical_wind_lat_deg=15.0,
            ),
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=2592000.0,  # 30-day restoring
                T_star_eq=25.0, T_star_pole=0.0,
                S_star=35.0, T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(
            scheme="kpp",
            kpp=KPPConfig(K_conv=0.0),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )

    # --- Model config: production-tuned recipe (2026-05-11) ---
    # H1: quadratic bottom drag (bg_velocity=0.1 → Cd=0.01)
    # H3: TVD tracer advection (warmer SST, less spurious flow)
    # M1: GM/Redi centered κ=600 (isopycnal mixing for ACC + MOC structure)
    config = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=0.0,                        # Smagorinsky provides damping
        A_v=1e-4,                       # production scalar (MOM6/NEMO/E3SM)
        C_smag_lap=0.33,                # Laplacian Smagorinsky
        K_v=1e-5,
        bottom_drag_r=1e-3,
        bottom_drag_bbl_thickness=100.0,
        bottom_drag_bg_velocity=0.1,    # H1: quadratic-with-floor
        K_zeta_bih=None,                # Voronoi checkerboard damping: DERIVED
                                        # from this mesh's spacing (dx^3,
                                        # anchored on the ico6 mesh where 1e14
                                        # was tuned)
        equatorial_visc_boost=0.0,
        pgf_scheme="centered",
        implicit_vertical_mixing=True,
        tracer_advection="tvd",         # H3: less spurious diffusion
        gm_redi=GMRediConfig(           # M1: isopycnal eddy mixing
            kappa_GM=600.0,
            kappa_Redi=600.0,
            S_max=0.005,
            visbeck=VisbeckConfig(enabled=False),
            slope_scheme="centered",
        ),
        physics=physics,
    )

    model = MPASOceanModel(mesh, pc_coord, config)

    # --- Initial condition ---
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, N_LEVELS))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active, jnp.full_like(T_data, 35.0), 0.0)

    state = rest_state_mpas_ocean(
        mesh, z_coord, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=H_MAX, land_lat_threshold=90.0,
    )
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask_new.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S", dims=("nCells", "nlev"), units="PSU"),
    )

    # --- Restart? ---
    start_day = 0.0
    if args.restart:
        state, start_day = load_restart(args.restart, state)
        print(f"  Resumed from {args.restart} at day {start_day:.0f}")

    # --- Print config ---
    print(f"\n  Config:")
    print(f"    implicit_vertical_mixing = True")
    print(f"    A_h={config.A_h:.0e}, A_v={config.A_v:.0e}, K_v={config.K_v:.0e}")
    print(f"    bottom_drag_r={config.bottom_drag_r:.0e}, KPP(K_conv=0)")
    print(f"    barotropic_solver=implicit_cn")
    print(f"    dt={DT}s, snap_frac={SNAP_FRAC}")

    # --- Time loop ---
    total_days = args.years * 365.25
    dt = DT
    n_steps = int(total_days * 86400 / dt)
    diag_every = int(86400 / dt)  # daily
    checkpoint_steps = int(args.checkpoint_years * 365.25 * 86400 / dt)

    csv_path = outdir / "timeseries.csv"
    csv_exists = csv_path.exists() and args.restart
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["day", "year", "max_u", "max_eta", "mean_SST", "wall_s"])

    print(f"\n  Running {total_days:.0f} days ({args.years} years)")
    print(f"  Checkpoints every {args.checkpoint_years} years\n")

    t0 = time.time()
    for k in range(n_steps):
        state = model.step(state, dt=dt)
        day = start_day + (k + 1) * dt / 86400.0

        if (k + 1) % diag_every == 0:
            mu = float(jnp.max(jnp.abs(state.u.data)))
            me = float(jnp.max(jnp.abs(state.eta.data)))
            mSST = float(jnp.where(ocean_mask_new > 0.5, state.T.data[:, 0], 0.0).sum()
                         / jnp.maximum(jnp.sum(ocean_mask_new > 0.5), 1))
            wall = time.time() - t0
            year = day / 365.25

            csv_writer.writerow([f"{day:.2f}", f"{year:.4f}",
                                 f"{mu:.6e}", f"{me:.6e}", f"{mSST:.4f}",
                                 f"{wall:.1f}"])
            csv_file.flush()

            if int(day) % 90 == 0 or k == n_steps - 1:
                print(f"  year {year:5.2f}  max|u|={mu:.4f}  "
                      f"max|eta|={me:.4f}  <SST>={mSST:.2f}  "
                      f"wall={wall/60:.1f}min")

            if not np.isfinite(mu) or mu > 20:
                print(f"  *** BLOWUP at year {year:.2f} ***")
                save_restart(state, day, restart_dir)
                save_snapshot(state, mesh, ocean_mask_new, day, snapshot_dir)
                break

        if (k + 1) % checkpoint_steps == 0 or k == n_steps - 1:
            fname = save_restart(state, day, restart_dir)
            print(f"  checkpoint: {fname}")
            save_snapshot(state, mesh, ocean_mask_new, day, snapshot_dir)

    csv_file.close()
    wall_total = time.time() - t0
    print(f"\nDone: {day/365.25:.1f} years in {wall_total/60:.1f} min "
          f"({wall_total/3600:.2f} h)")


if __name__ == "__main__":
    main()
