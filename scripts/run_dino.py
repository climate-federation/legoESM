#!/usr/bin/env python
"""DINO (Diabatic Neverworld Ocean) standalone production script.

Replicates the Kamm, Deshayes & Madec (2025, GMD) DINO 1° R1
configuration on the lat-lon Mercator grid OR on an MPAS regional
Voronoi mesh, with full surface forcing (wind + T/S restoring +
Q_sr split + Jerlov SW penetration) and physics (KPP + GM/Redi +
enhanced-diffusion convection).

Quick start::

    JAX_ENABLE_X64=1 python scripts/run_dino.py --days 10
    JAX_ENABLE_X64=1 python scripts/run_dino.py --grid mpas --days 10
    JAX_ENABLE_X64=1 python scripts/plot_dino.py results/dino   # visualize

For the full list of options::

    python scripts/run_dino.py --help

This script is **portable** by design: no project-internal CI hooks,
no test-matrix integration. The output directory is self-contained
(NPZ snapshots + a JSON config dump) so it can be moved to a GPU
machine for production runs.

See ``docs/ocean_experiments/README.md`` for a 1-minute orientation
and ``docs/ocean_experiments/dino_replication_plan.md`` for the full
scientific configuration, decisions log, and stability investigation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    apply_dino_lat_lon_surface_forcing,
    apply_dino_mpas_surface_forcing,
    create_dino_z_star,
    dino_lat_lon_grid,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
    dino_lat_lon_surface_forcing_arrays,
    dino_mpas_model_config,
    dino_mpas_state,
    dino_mpas_surface_forcing_arrays,
)


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--grid", choices=("latlon", "mpas"), default="latlon",
        help="Horizontal grid: 'latlon' (Mercator) or 'mpas' (regional "
             "Voronoi with periodic_x=True + seam wall).",
    )
    p.add_argument(
        "--n-lon", type=int, default=50,
        help="Zonal cell count for the Mercator grid (lat-lon only; "
             "50 = 1° R1, default).",
    )
    p.add_argument(
        "--mpas-resolution-km", type=float, default=97.0,
        help="Cell-spacing target for the MPAS Voronoi mesh "
             "(MPAS only). Default 97 km gives 9686 cells, matching "
             "the 50-col Mercator basin's 9900 cells within 2%% -- cross-"
             "grid comparison is at equivalent mean cell area. "
             "(Theoretical area-equivalent is ~82 km but the regional "
             "Voronoi generator has quantization gaps below ~85 km; "
             "97 km is the closest working value to 9900-cell match.)",
    )
    p.add_argument(
        "--days", type=float, default=10.0,
        help="Total simulated duration in days (default 10; capped at "
             "365 by local-machine policy — see plan).",
    )
    p.add_argument(
        "--snapshot-every-days", type=float, default=1.0,
        help="Snapshot interval in days (default 1).",
    )
    p.add_argument(
        "--dt", type=float, default=None,
        help="Baroclinic timestep [s]. Default uses DINOConfig.dt = 2700.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/dino"),
        help="Directory to write snapshots + log. Default 'results/dino' "
             "(relative to CWD). Note: `results/` should be gitignored — "
             "snapshots can be many MB.",
    )
    p.add_argument(
        "--no-forcing", action="store_true",
        help="Run dycore-only from rest (no wind, no restoring) — for "
             "shake-down tests of the model + bathymetry combination.",
    )
    p.add_argument(
        "--physics-off", action="store_true",
        help="Disable KPP / GM-Redi / convection — dycore only.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def _diagnose(state, grid, grid_kind: str):
    """Compact per-snapshot diagnostics. ``grid_kind`` ∈ {"latlon","mpas"}."""
    cell_mask = np.asarray(state.land_mask.data) > 0.5
    u = np.asarray(state.u.data)
    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    eta = np.asarray(state.eta.data)

    u_max = float(np.max(np.abs(u)))
    eta_max = float(np.max(np.abs(eta)))
    T_max = float(np.max(T[cell_mask, :])) if cell_mask.any() else float("nan")
    T_min = float(np.min(T[cell_mask, :])) if cell_mask.any() else float("nan")
    S_max = float(np.max(S[cell_mask, :])) if cell_mask.any() else float("nan")
    S_min = float(np.min(S[cell_mask, :])) if cell_mask.any() else float("nan")

    if grid_kind == "latlon":
        v = np.asarray(state.v.data)
        v_max = float(np.max(np.abs(v)))
        # u: (n_lat, n_lon+1, nlev), v: (n_lat+1, n_lon, nlev) → cell centers
        u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
        v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        ke = 0.5 * (u_cc ** 2 + v_cc ** 2)
        area = np.asarray(grid.area)            # (n_lat, n_lon)
        ke_total = float(np.sum(ke * area[..., None] * cell_mask[..., None]))
    else:  # mpas
        # u is edge-normal (nEdges, nlev); approximate per-cell KE as
        # the mean of |u|² over the cell's edges. Cheap and good enough
        # for a monitor.
        v_max = float("nan")
        ke_edge = 0.5 * u ** 2                  # (nEdges, nlev)
        area = np.asarray(grid.areaCell)        # (nCells,)
        ke_per_cell = float(np.mean(ke_edge) * np.sum(area * cell_mask))
        ke_total = ke_per_cell

    return {
        "u_max": u_max, "v_max": v_max, "eta_max": eta_max,
        "T_max": T_max, "T_min": T_min,
        "S_max": S_max, "S_min": S_min,
        "ke_total": ke_total,
    }


def _save_snapshot(state, t_seconds, snapshot_dir: Path, idx: int, grid_kind: str):
    """Save a snapshot as NPZ. Handles both lat-lon (has v) and MPAS (no v)."""
    fields = {
        "time_seconds": t_seconds,
        "time_days": t_seconds / 86400.0,
        "eta": np.asarray(state.eta.data),
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data),
        "land_mask": np.asarray(state.land_mask.data),
        "H_bathy": np.asarray(state.H_bathy.data),
    }
    if grid_kind == "latlon":
        fields["v"] = np.asarray(state.v.data)
    np.savez_compressed(snapshot_dir / f"snapshot_{idx:05d}.npz", **fields)


def _save_run_metadata(args, cfg: DINOConfig, grid, z, output_dir: Path,
                       grid_kind: str):
    """Write a JSON file with the run config + grid info."""
    if grid_kind == "latlon":
        grid_info = {
            "kind": "latlon-mercator",
            "n_lat": int(grid.n_lat),
            "n_lon": int(grid.n_lon),
            "radius": float(grid.radius),
            "lat_min_deg": float(np.degrees(np.min(grid.lat))),
            "lat_max_deg": float(np.degrees(np.max(grid.lat))),
            "dx_eq_km": float(np.max(grid.dx) / 1000.0),
            "dx_pole_km": float(np.min(grid.dx) / 1000.0),
        }
    else:
        grid_info = {
            "kind": "mpas-regional-voronoi",
            "nCells": int(grid.nCells),
            "nEdges": int(grid.nEdges),
            "radius": float(grid.radius),
            "median_dcEdge_km": float(np.median(grid.dcEdge) / 1000.0),
        }

    metadata = {
        "args": vars(args),
        "config": dataclasses.asdict(cfg),
        "grid": grid_info,
        "vertical": {
            "n_levels": int(z.n_levels),
            "H_max": float(z.H_max),
            "dz_top": float(z.dz_ref[0]),
            "dz_bot": float(z.dz_ref[-1]),
        },
    }
    metadata["config"]["wind_tau_lats_deg"] = list(cfg.wind_tau_lats_deg)
    metadata["config"]["wind_tau_values"] = list(cfg.wind_tau_values)
    metadata["args"]["output_dir"] = str(metadata["args"]["output_dir"])
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = _parse_args()

    # JAX x64 sanity check: DINO uses Wright EOS + barotropic split;
    # both need 64-bit precision to avoid silent eta drift and EOS noise.
    import jax
    if not jax.config.x64_enabled:
        import warnings
        warnings.warn(
            "JAX_ENABLE_X64 is OFF. DINO needs 64-bit for the Wright EOS "
            "and barotropic-baroclinic split — silent precision artifacts "
            "will appear at multi-day integration. Re-run with "
            "`JAX_ENABLE_X64=1 python scripts/run_dino.py ...`.",
            stacklevel=2,
        )

    # Original local-machine policy capped at 1 yr. Removed 2026-05-18:
    # V100S GPU completes 1 yr in ~4.4 min, so multi-year runs are feasible.

    cfg = DINOConfig()
    if args.dt is not None:
        cfg = dataclasses.replace(cfg, dt=args.dt)
    dt = cfg.dt
    grid_kind = args.grid

    # Build grid, state, model — branch on grid type
    z = create_dino_z_star(cfg)
    if grid_kind == "latlon":
        grid = dino_lat_lon_grid(cfg, n_lon=args.n_lon)
        state = dino_lat_lon_state(grid, z, cfg)
        model_cfg, _ = dino_lat_lon_model_config(
            grid, cfg, physics=not args.physics_off,
        )
        model = LatLonCGridOceanModel(grid, z, model_cfg)
        forcing = (None if args.no_forcing
                   else dino_lat_lon_surface_forcing_arrays(grid, cfg))
        apply_forcing = apply_dino_lat_lon_surface_forcing
        grid_desc = f"{grid.n_lat}x{grid.n_lon} lat-lon Mercator"
    else:  # mpas
        grid = create_regional_voronoi_mesh(
            lon_range=(cfg.lon_west_deg, cfg.lon_east_deg),
            lat_range=(-cfg.lat_max_deg, cfg.lat_max_deg),
            resolution_km=args.mpas_resolution_km,
            periodic_x=True,
        )
        state = dino_mpas_state(grid, z, cfg)
        model_cfg, _ = dino_mpas_model_config(
            grid, cfg, physics=not args.physics_off,
        )
        model = MPASOceanModel(grid, z, model_cfg)
        forcing = (None if args.no_forcing
                   else dino_mpas_surface_forcing_arrays(grid, cfg))
        apply_forcing = apply_dino_mpas_surface_forcing
        grid_desc = f"{grid.nCells} cells MPAS regional Voronoi"

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.output_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    _save_run_metadata(args, cfg, grid, z, args.output_dir, grid_kind)

    # Time loop
    n_steps_total = int(round(args.days * 86400.0 / dt))
    snapshot_every_steps = max(1, int(round(args.snapshot_every_days * 86400.0 / dt)))

    print(f"DINO {grid_desc}: {z.n_levels} levels, dt={dt}s")
    print(f"Run: {n_steps_total} steps = {args.days:.2f} days, "
          f"snapshot every {snapshot_every_steps} steps "
          f"= {snapshot_every_steps * dt / 86400.0:.2f} days")
    print(f"Forcing: {'OFF (dycore only)' if args.no_forcing else 'wind + T/S restoring (Q_sr split) + Jerlov-I SW penetration'}")
    print(f"Physics: {'OFF' if args.physics_off else 'KPP + GM/Redi + enhanced-diffusion convection'}")
    print(f"Output:  {args.output_dir}")
    print()
    print(f"{'step':>6} {'day':>7} {'|u|':>10} {'|v|':>10} {'|eta|':>10} "
          f"{'T_max':>7} {'T_min':>7} {'KE':>10}")

    t_wall_start = time.time()
    snapshot_idx = 0
    _save_snapshot(state, 0.0, snapshot_dir, snapshot_idx, grid_kind)

    for k in range(n_steps_total):
        if forcing is not None:
            state = apply_forcing(state, forcing, z, cfg, dt)

        state = model.step(state, dt=dt)

        is_last = (k == n_steps_total - 1)
        if (k + 1) % snapshot_every_steps == 0 or is_last:
            snapshot_idx += 1
            t_seconds = (k + 1) * dt
            _save_snapshot(state, t_seconds, snapshot_dir, snapshot_idx, grid_kind)
            d = _diagnose(state, grid, grid_kind)
            print(f"{k+1:6d} {t_seconds/86400.0:7.2f} "
                  f"{d['u_max']:10.4e} {d['v_max']:10.4e} {d['eta_max']:10.4e} "
                  f"{d['T_max']:7.2f} {d['T_min']:7.2f} {d['ke_total']:10.4e}")

    wall = time.time() - t_wall_start
    print()
    print(f"Done. Wall time: {wall:.1f}s ({wall/n_steps_total*1000:.1f} ms/step). "
          f"Snapshots: {snapshot_idx + 1}")


if __name__ == "__main__":
    main()
