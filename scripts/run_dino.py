#!/usr/bin/env python
"""DINO (Diabatic Neverworld Ocean) standalone production script.

Replicates Kamm et al. (2025) DINO 1° R1 configuration on the lat-lon
Mercator grid (Phase 4 v0). MPAS path will be added in v1.

Run with::

    JAX_ENABLE_X64=1 python scripts/run_dino.py --days 10

For the full list of options::

    python scripts/run_dino.py --help

This script is **portable** by design: no project-internal CI hooks,
no test-matrix integration. The output directory is self-contained
(NPZ snapshots + a JSON config dump) so it can be moved to a GPU
machine for production runs.

See ``docs/ocean_experiments/dino_replication_plan.md`` for the
scientific configuration and decisions log.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_regional_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    create_dino_z_star,
    dino_lat_lon_grid,
    dino_lat_lon_model_config,
    dino_lat_lon_state,
    dino_mpas_model_config,
    dino_mpas_state,
    dino_Q_sr_annual_mean,
    dino_S_star,
    dino_T_star_annual_mean,
    dino_top_layer_S_tendency,
    dino_top_layer_T_tendency,
    dino_top_layer_u_tendency,
    dino_wind_stress,
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
             "the 50-col Mercator basin's 9900 cells within 2% — cross-"
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
        "--output-dir", type=Path, default=Path("dino_output"),
        help="Directory to write snapshots + log.",
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
# DINO surface forcing on the lat-lon Mercator grid
#
# Applied as explicit external tendencies after each model step. This
# sidesteps legoESM's surface_forcing physics convention (which uses
# timescales, not heat-flux coefficients, and doesn't subtract Q_sr
# per paper eq 8). Q_sr split: surface T tendency uses the FULL eq 8
# formula `(A_θ(T*-T) - Q_sr) / (ρ₀ c_p dz_0)`. Jerlov shortwave
# penetration through the column is NOT applied in v0 — the column
# simply doesn't see solar heating below the surface. This means the
# subsurface stratification will be biased (no penetrating SW). v1
# will add proper Jerlov penetration.
# ---------------------------------------------------------------------

def _build_lat_lon_forcing_arrays(grid, cfg: DINOConfig):
    """Pre-compute forcing fields that don't depend on the state.

    Returns dict with:
      tau_u_face: (n_lat, n_lon+1) — wind stress at u-faces
      T_star_2d:  (n_lat, n_lon)
      S_star_2d:  (n_lat, n_lon)
      Q_sr_2d:    (n_lat, n_lon) — annual-mean solar
    """
    lat_1d = jnp.degrees(grid.lat)              # (n_lat,)

    T_star_1d = dino_T_star_annual_mean(lat_1d, cfg)
    S_star_1d = dino_S_star(lat_1d, cfg)
    Q_sr_1d = dino_Q_sr_annual_mean(lat_1d, cfg)
    tau_u_1d = dino_wind_stress(lat_1d, cfg)    # (n_lat,)

    # Broadcast to 2D
    T_star_2d = jnp.broadcast_to(T_star_1d[:, None], (grid.n_lat, grid.n_lon))
    S_star_2d = jnp.broadcast_to(S_star_1d[:, None], (grid.n_lat, grid.n_lon))
    Q_sr_2d = jnp.broadcast_to(Q_sr_1d[:, None], (grid.n_lat, grid.n_lon))

    # Wind at u-faces: u shape (n_lat, n_lon+1), zonally uniform
    tau_u_face = jnp.broadcast_to(tau_u_1d[:, None], (grid.n_lat, grid.n_lon + 1))

    return {
        "tau_u_face": tau_u_face,
        "T_star_2d": T_star_2d,
        "S_star_2d": S_star_2d,
        "Q_sr_2d": Q_sr_2d,
    }


def apply_dino_lat_lon_surface_forcing(state, forcing, z_coord, cfg, dt):
    """Apply DINO surface forcing on the lat-lon Mercator grid.

    Components (paper eqs 7-10):
      - Wind:  τ_u → top-layer u tendency (eq 7)
      - T:     non-solar restoring (A_θ(T*-T) - Q_sr) at top layer (eq 8)
      - S:     A_S(S*-S) at top layer (eq 9)
      - SW:    Jerlov type I column-distributed Q_sr through all
               levels (eq 10) — uses
               ``legoesm.ocean.physics.shortwave_penetration``.

    Returns
    -------
    LatLonCGridOceanState
    """
    dz_0 = float(z_coord.dz_ref[0])
    cell_mask = state.land_mask.data

    # T tendency at top layer (eq 8; subtracts Q_sr from non-solar component)
    T_top = state.T.data[..., 0]
    dT_dt_top = dino_top_layer_T_tendency(
        T_top, forcing["T_star_2d"], forcing["Q_sr_2d"], dz_0, cfg,
    )

    # Jerlov SW penetration through the column (eq 10): full 3D tendency
    # added to dT/dt at every level. The reference Jacobian ≈ 1
    # (η/H_max ~ 1e-4) — using 1.0 keeps the code simple and matches
    # standard practice in MOM6 / NEMO / POP.
    jacobian = jnp.ones_like(state.eta.data)
    sw_cfg = ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type)
    dT_dt_sw = shortwave_penetration_tendency(
        sw_down=forcing["Q_sr_2d"],
        z_coord_dz_ref=z_coord.dz_ref,
        z_coord_z_half_ref=z_coord.z_half_ref,
        jacobian=jacobian,
        config=sw_cfg,
        rho_0=cfg.rho_0,
        c_sw=cfg.c_p,
    )  # (n_lat, n_lon, nlev)

    # Combine: top layer gets eq-8 surface flux + eq-10 surface absorption;
    # subsurface levels get only eq-10 SW absorption.
    new_T = state.T.data + dt * dT_dt_sw * cell_mask[..., None]
    new_T_top = new_T[..., 0] + dt * dT_dt_top * cell_mask
    new_T = new_T.at[..., 0].set(new_T_top)

    # S tendency at top layer (eq 9)
    S_top = state.S.data[..., 0]
    dS_dt_top = dino_top_layer_S_tendency(
        S_top, forcing["S_star_2d"], dz_0, cfg,
    )
    new_S_top = S_top + dt * dS_dt_top * cell_mask
    new_S = state.S.data.at[..., 0].set(new_S_top)

    # u tendency at u-faces (eq 7)
    u_top = state.u.data[..., 0]
    du_dt_top = dino_top_layer_u_tendency(forcing["tau_u_face"], dz_0, cfg)
    u_face_mask = state.u_mask.data
    new_u_top = u_top + dt * du_dt_top * u_face_mask
    new_u = state.u.data.at[..., 0].set(new_u_top)

    return state._replace(
        T=Field(data=new_T, name=state.T.name, dims=state.T.dims, units=state.T.units),
        S=Field(data=new_S, name=state.S.name, dims=state.S.dims, units=state.S.units),
        u=Field(data=new_u, name=state.u.name, dims=state.u.dims, units=state.u.units),
    )


# ---------------------------------------------------------------------
# DINO surface forcing on the MPAS regional mesh
# ---------------------------------------------------------------------

def _build_mpas_forcing_arrays(mesh, cfg: DINOConfig):
    """Pre-compute MPAS forcing fields.

    Wind τ_u(lat) is computed at edge latitudes and projected onto the
    edge-normal direction via ``cos(angleEdge)`` (tau_v=0 for DINO).
    T*, S*, Q_sr are computed at cell latitudes.

    Returns dict with keys:
      tau_normal: (nEdges,) — wind stress along edge normal
      T_star_1d: (nCells,)
      S_star_1d: (nCells,)
      Q_sr_1d:   (nCells,)
    """
    # Cell-center latitudes (radians → degrees)
    lat_c_deg = jnp.degrees(mesh.latCell)
    T_star_1d = dino_T_star_annual_mean(lat_c_deg, cfg)
    S_star_1d = dino_S_star(lat_c_deg, cfg)
    Q_sr_1d = dino_Q_sr_annual_mean(lat_c_deg, cfg)

    # Edge-projected wind stress (tau_v = 0 for DINO so tau_n = tau_u·cos(angle))
    lat_e_deg = jnp.degrees(mesh.latEdge)
    tau_u_e = dino_wind_stress(lat_e_deg, cfg)         # (nEdges,)
    tau_normal = tau_u_e * jnp.cos(mesh.angleEdge)     # (nEdges,)

    return {
        "tau_normal": tau_normal,
        "T_star_1d": T_star_1d,
        "S_star_1d": S_star_1d,
        "Q_sr_1d": Q_sr_1d,
    }


def apply_dino_mpas_surface_forcing(state, forcing, z_coord, cfg, dt):
    """Apply DINO surface forcing on the MPAS regional mesh.

    Same physics as the lat-lon applicator (paper eqs 7-10) with edge-
    projected wind and 1D cell-indexed T*, S*, Q_sr.

    Returns
    -------
    MPASOceanState
    """
    dz_0 = float(z_coord.dz_ref[0])
    cell_mask = state.land_mask.data                  # (nCells,)

    # T tendency at top layer (eq 8)
    T_top = state.T.data[:, 0]                         # (nCells,)
    dT_dt_top = dino_top_layer_T_tendency(
        T_top, forcing["T_star_1d"], forcing["Q_sr_1d"], dz_0, cfg,
    )

    # Jerlov SW penetration through the column (eq 10)
    jacobian = jnp.ones_like(state.eta.data)           # (nCells,)
    sw_cfg = ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type)
    dT_dt_sw = shortwave_penetration_tendency(
        sw_down=forcing["Q_sr_1d"],
        z_coord_dz_ref=z_coord.dz_ref,
        z_coord_z_half_ref=z_coord.z_half_ref,
        jacobian=jacobian,
        config=sw_cfg,
        rho_0=cfg.rho_0,
        c_sw=cfg.c_p,
    )  # (nCells, nlev)

    new_T = state.T.data + dt * dT_dt_sw * cell_mask[:, None]
    new_T_top = new_T[:, 0] + dt * dT_dt_top * cell_mask
    new_T = new_T.at[:, 0].set(new_T_top)

    # S tendency at top layer (eq 9)
    S_top = state.S.data[:, 0]
    dS_dt_top = dino_top_layer_S_tendency(
        S_top, forcing["S_star_1d"], dz_0, cfg,
    )
    new_S_top = S_top + dt * dS_dt_top * cell_mask
    new_S = state.S.data.at[:, 0].set(new_S_top)

    # u tendency at edges (eq 7): du/dt = tau_normal / (rho_0 · dz_0).
    # The MPAS dynamics handles dry-edge masking internally (edges to
    # land cells have their fluxes zeroed by the operators), so we
    # don't apply a per-edge mask here.
    u_top = state.u.data[:, 0]                          # (nEdges,)
    du_dt_top = dino_top_layer_u_tendency(forcing["tau_normal"], dz_0, cfg)
    new_u_top = u_top + dt * du_dt_top
    new_u = state.u.data.at[:, 0].set(new_u_top)

    return state._replace(
        T=Field(data=new_T, name=state.T.name, dims=state.T.dims, units=state.T.units),
        S=Field(data=new_S, name=state.S.name, dims=state.S.dims, units=state.S.units),
        u=Field(data=new_u, name=state.u.name, dims=state.u.dims, units=state.u.units),
    )


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

    # Local-machine policy: cap at 1 yr (decision logged in plan)
    if args.days > 365.0:
        raise SystemExit(
            f"--days={args.days} exceeds the 1-year local-machine cap. "
            "Long spin-ups should run on a GPU machine — see plan."
        )

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
                   else _build_lat_lon_forcing_arrays(grid, cfg))
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
                   else _build_mpas_forcing_arrays(grid, cfg))
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
