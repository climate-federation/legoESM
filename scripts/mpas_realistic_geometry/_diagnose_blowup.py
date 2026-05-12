"""Diagnose WHERE the ETOPO bathymetry blowup initiates.

Restarts from day 120 checkpoint (before exponential growth begins)
and saves spatial diagnostics every 5 days to identify the failure mode.

Output: outputs/mpas_etopo_spinup_bdrag1e-3_topo/diagnostics/
  - diag_dayXXX.npz: per-cell speed, SSH, T, and per-edge |u|
  - diag_summary.csv: top-20 fastest edges with lat/lon/depth
  - growth_maps/growth_dayXXX.png: spatial maps of velocity growth

Usage:
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 python scripts/mpas_realistic_geometry/_diagnose_blowup.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

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
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate


def snap_partial_cells(H_bathy, z_coord, min_frac=0.30):
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


def load_restart(restart_path, template_state):
    """Load restart npz into template state."""
    data = np.load(restart_path)
    restart_day = float(data["time_days"])
    restart_step = int(data["step"])
    replacements = {}
    for f in template_state._fields:
        if f not in data:
            continue
        obj = getattr(template_state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        arr = jnp.asarray(data[f], dtype=obj.data.dtype)
        replacements[f] = Field(data=arr, name=obj.name, dims=obj.dims, units=obj.units)
    state = template_state._replace(**replacements)
    return state, restart_day, restart_step


def compute_edge_horiz_T_gradient(state, mesh, pc_coord):
    """Compute |dT/dx| at each edge (surface level) as proxy for PGF source."""
    T = state.T.data[:, 0]  # surface T
    c0 = mesh.cellsOnEdge[0, :]  # (2, nEdges) layout
    c1 = mesh.cellsOnEdge[1, :]
    dT = jnp.abs(T[c0] - T[c1])
    # Approximate dx from edge length (dcEdge)
    dx = mesh.dcEdge
    return dT / jnp.maximum(dx, 1.0)  # K/m


def diagnose_snapshot(state, mesh, pc_coord, ocean_mask, prev_u_abs=None):
    """Compute spatial diagnostic fields."""
    u_abs = jnp.abs(state.u.data)  # (nEdges, nlev)
    u_max_edge = jnp.max(u_abs, axis=1)  # max over depth per edge

    # Growth rate (if we have previous snapshot)
    growth = None
    if prev_u_abs is not None:
        growth = u_abs - prev_u_abs  # positive = growing

    # Per-cell speed (reconstructed)
    u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
    speed = jnp.sqrt(u_east**2 + v_north**2)
    speed_sfc = speed[:, 0]
    speed_max = jnp.max(speed, axis=1)

    # Horizontal T gradient at edges
    dTdx = compute_edge_horiz_T_gradient(state, mesh, pc_coord)

    # Step-edge indicator: edges where adjacent cells have different bottom level
    H = state.H_bathy.data
    c0 = mesh.cellsOnEdge[0, :]
    c1 = mesh.cellsOnEdge[1, :]
    dH = jnp.abs(H[c0] - H[c1])

    return {
        "u_abs": u_abs,
        "u_max_edge": u_max_edge,
        "growth": growth,
        "speed_sfc": speed_sfc,
        "speed_max": speed_max,
        "dTdx_sfc": dTdx,
        "dH_edge": dH,
        "eta": state.eta.data,
        "sst": state.T.data[:, 0],
    }


def plot_growth_map(diag, mesh, ocean_mask, day, output_dir):
    """Plot spatial maps showing where instability is growing."""
    lon_cell = np.degrees(np.asarray(mesh.lonCell))
    lat_cell = np.degrees(np.asarray(mesh.latCell))
    lon_edge = np.degrees(np.asarray(mesh.lonEdge))
    lat_edge = np.degrees(np.asarray(mesh.latEdge))
    mask = np.asarray(ocean_mask) > 0.5

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Panel 1: Surface speed
    spd = np.where(mask, np.asarray(diag["speed_sfc"]), np.nan)
    vmax = max(0.1, np.nanpercentile(spd, 99))
    sc = axes[0, 0].scatter(lon_cell, lat_cell, c=spd, s=3, cmap="magma", vmin=0, vmax=vmax)
    plt.colorbar(sc, ax=axes[0, 0], label="m/s")
    axes[0, 0].set_title(f"Surface speed (max={np.nanmax(spd):.2f})")

    # Panel 2: Edge velocity max-over-depth
    u_edge = np.asarray(diag["u_max_edge"])
    vmax_e = max(0.1, np.percentile(u_edge[u_edge > 0], 99))
    sc = axes[0, 1].scatter(lon_edge, lat_edge, c=u_edge, s=1, cmap="hot", vmin=0, vmax=vmax_e)
    plt.colorbar(sc, ax=axes[0, 1], label="m/s")
    axes[0, 1].set_title(f"Edge |u| max-depth (max={u_edge.max():.2f})")

    # Panel 3: SSH
    eta = np.where(mask, np.asarray(diag["eta"]), np.nan)
    vm = max(0.1, np.nanmax(np.abs(eta)))
    sc = axes[0, 2].scatter(lon_cell, lat_cell, c=eta, s=3, cmap="RdBu_r", vmin=-vm, vmax=vm)
    plt.colorbar(sc, ax=axes[0, 2], label="m")
    axes[0, 2].set_title(f"SSH (max|eta|={vm:.2f})")

    # Panel 4: Surface dT/dx at edges
    dTdx = np.asarray(diag["dTdx_sfc"])
    vmax_t = max(1e-7, np.percentile(dTdx[dTdx > 0], 99))
    sc = axes[1, 0].scatter(lon_edge, lat_edge, c=dTdx, s=1, cmap="YlOrRd", vmin=0, vmax=vmax_t)
    plt.colorbar(sc, ax=axes[1, 0], label="K/m")
    axes[1, 0].set_title(f"Surface |dT/dx| at edges")

    # Panel 5: SST
    sst = np.where(mask, np.asarray(diag["sst"]), np.nan)
    sc = axes[1, 1].scatter(lon_cell, lat_cell, c=sst, s=3, cmap="RdYlBu_r")
    plt.colorbar(sc, ax=axes[1, 1], label="degC")
    axes[1, 1].set_title(f"SST (mean={np.nanmean(sst):.2f})")

    # Panel 6: Bathymetry step height at edges (context)
    dH = np.asarray(diag["dH_edge"])
    sc = axes[1, 2].scatter(lon_edge, lat_edge, c=dH, s=1, cmap="copper", vmin=0,
                            vmax=max(100, np.percentile(dH, 95)))
    plt.colorbar(sc, ax=axes[1, 2], label="m")
    axes[1, 2].set_title("Bathymetry step |dH| at edges")

    for ax in axes.flat:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_xlim(0, 360); ax.set_ylim(-90, 90)

    fig.suptitle(f"Blowup diagnostics — day {day:.0f}", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"growth_day{int(day):04d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_top_edges(diag, mesh, day, output_dir, n_top=50):
    """Plot location of top-N fastest edges, colored by latitude band."""
    u_max = np.asarray(diag["u_max_edge"])
    top_idx = np.argsort(u_max)[-n_top:][::-1]

    lon_e = np.degrees(np.asarray(mesh.lonEdge))
    lat_e = np.degrees(np.asarray(mesh.latEdge))

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    # Background: all edges faintly
    ax.scatter(lon_e, lat_e, c="lightgray", s=0.5, alpha=0.3)
    # Top edges colored by speed
    sc = ax.scatter(lon_e[top_idx], lat_e[top_idx], c=u_max[top_idx],
                    s=40, cmap="hot", edgecolors="black", linewidths=0.5,
                    vmin=0, vmax=u_max[top_idx[0]])
    plt.colorbar(sc, ax=ax, label="|u| m/s")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"Top-{n_top} fastest edges — day {day:.0f}")
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)

    # Annotate lat histogram
    lats_top = lat_e[top_idx]
    eq_frac = np.sum(np.abs(lats_top) < 10) / n_top
    ax.text(0.02, 0.02, f"|lat|<10°: {eq_frac*100:.0f}%\n"
            f"|lat|<30°: {np.sum(np.abs(lats_top)<30)/n_top*100:.0f}%",
            transform=ax.transAxes, fontsize=11, va="bottom",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / f"top_edges_day{int(day):04d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    # --- Setup (identical to spinup script) ---
    mesh = create_voronoi_mesh(subdivision_level=5)
    z_coord = create_ocean_z_star(n_levels=10, H_max=5500.0)

    bathy_cfg = BathymetryConfig(
        source="file", path="/home/dbalwada/legoESM/data/bathymetry/etopo_1deg.nc",
        H_max=5500.0, H_min=10.0, smoothing_passes=2,
        r_factor_max=0.2, depth_is_negative=True,
    )
    H_bathy_raw, ocean_mask = load_bathymetry_mpas(mesh, bathy_cfg)
    H_snapped = snap_partial_cells(H_bathy_raw, z_coord, min_frac=0.30)
    ocean_mask_new = jnp.where(H_snapped > 0, ocean_mask, 0.0)
    pc_coord = create_partial_cell_coordinate(z_coord, H_snapped)

    # Physics (same as bdrag test)
    physics = OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="combined",
            prescribed=PrescribedForcingConfig(wind_profile="global_wind", tau_max=0.1),
            restoring=RestoringConfig(
                tau_T=2592000.0, tau_S=2592000.0,
                T_star_eq=25.0, T_star_pole=0.0,
                S_star=35.0, T_profile="cosine",
            ),
        ),
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
    )

    config = MPASOceanConfig(
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1e-10,
        barotropic_implicit_pcg_maxiter=300,
        A_h=5e4,
        A_v=1e-2,
        K_v=1e-4,
        bottom_drag_r=1e-3,
        bottom_drag_bbl_thickness=100.0,
        equatorial_visc_boost=0.0,
        pgf_scheme="centered",
        physics=physics,
    )
    model = MPASOceanModel(mesh, pc_coord, config)

    # --- Template state + restart ---
    T_ref = 2.0 + 18.0 * jnp.exp(z_coord.z_full_ref / _SCALE_DEPTH)
    T_data = jnp.broadcast_to(T_ref[None, :], (mesh.nCells, 10))
    T_data = jnp.where(pc_coord.is_active, T_data, 0.0)
    S_data = jnp.where(pc_coord.is_active, jnp.full_like(T_data, 35.0), 0.0)

    state = rest_state_mpas_ocean(
        mesh, z_coord, T_surface=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=5500.0, land_lat_threshold=90.0,
    )
    dtype = state.eta.data.dtype
    state = state._replace(
        H_bathy=Field(data=H_snapped.astype(dtype), name="H_bathy", dims=("nCells",), units="m"),
        land_mask=Field(data=ocean_mask_new.astype(dtype)),
        T=Field(data=T_data.astype(dtype), name="T", dims=("nCells", "nlev"), units="degC"),
        S=Field(data=S_data.astype(dtype), name="S", dims=("nCells", "nlev"), units="PSU"),
    )

    restart_path = "outputs/mpas_etopo_spinup_bdrag1e-3_topo/restarts/restart_day000120.npz"
    state, start_day, start_step = load_restart(restart_path, state)
    print(f"Loaded restart: day {start_day:.0f}, step {start_step}")

    # --- Output dirs ---
    base_dir = Path("outputs/mpas_etopo_spinup_bdrag1e-3_topo/diagnostics")
    growth_dir = base_dir / "growth_maps"
    growth_dir.mkdir(parents=True, exist_ok=True)

    # --- Run 120 days (day 120 → 240) with diagnostics every 5 days ---
    dt = 300.0
    diag_interval_days = 5.0
    diag_steps = int(diag_interval_days * 86400 / dt)
    total_steps = int(120 * 86400 / dt)  # 120 more days

    # CSV for top-edge summary
    csv_path = base_dir / "top_edges_summary.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["day", "max_u", "top1_lon", "top1_lat", "top1_speed",
                         "frac_eq10", "frac_eq30", "top1_depth_km",
                         "top1_dH_m", "top1_dTdx"])

    prev_u_abs = jnp.abs(state.u.data)
    print(f"\nRunning diagnostics: day {start_day:.0f} → {start_day + 120:.0f}")
    print(f"  Saving spatial maps every {diag_interval_days} days\n")

    t0 = time.time()
    for k in range(total_steps):
        state = model.step(state, dt=dt)
        step = start_step + k + 1
        day = start_day + (k + 1) * dt / 86400.0

        if (k + 1) % diag_steps == 0:
            mu = float(jnp.max(jnp.abs(state.u.data)))
            print(f"  day {day:6.1f}  max|u|={mu:.4f} m/s  wall={time.time()-t0:.0f}s")

            if np.isnan(mu):
                print("  *** NaN — stopping ***")
                break

            diag = diagnose_snapshot(state, mesh, pc_coord, ocean_mask_new, prev_u_abs)

            # Plot spatial maps
            plot_growth_map(diag, mesh, ocean_mask_new, day, growth_dir)
            plot_top_edges(diag, mesh, day, growth_dir, n_top=50)

            # Top-edge CSV
            u_max = np.asarray(diag["u_max_edge"])
            top_idx = np.argsort(u_max)[-50:][::-1]
            lon_e = np.degrees(np.asarray(mesh.lonEdge))
            lat_e = np.degrees(np.asarray(mesh.latEdge))
            dH = np.asarray(diag["dH_edge"])
            dTdx = np.asarray(diag["dTdx_sfc"])

            lats_top = lat_e[top_idx]
            eq10 = np.sum(np.abs(lats_top) < 10) / 50
            eq30 = np.sum(np.abs(lats_top) < 30) / 50

            # Top-1 edge info
            t1 = top_idx[0]
            c0 = int(mesh.cellsOnEdge[0, t1])
            c1 = int(mesh.cellsOnEdge[1, t1])
            H_avg = float((state.H_bathy.data[c0] + state.H_bathy.data[c1]) / 2)

            csv_writer.writerow([
                f"{day:.1f}", f"{mu:.4f}",
                f"{lon_e[t1]:.1f}", f"{lat_e[t1]:.1f}", f"{u_max[t1]:.4f}",
                f"{eq10:.2f}", f"{eq30:.2f}",
                f"{H_avg/1000:.2f}", f"{dH[t1]:.0f}", f"{dTdx[t1]:.2e}",
            ])
            csv_file.flush()

            prev_u_abs = jnp.abs(state.u.data)

    csv_file.close()
    print(f"\nDiagnostics saved to: {base_dir}")
    print(f"  growth_maps/*.png — spatial snapshots")
    print(f"  top_edges_summary.csv — top-50 edge tracking")


if __name__ == "__main__":
    main()
