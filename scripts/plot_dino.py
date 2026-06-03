#!/usr/bin/env python
"""Visualize a DINO run produced by ``scripts/run_dino.py``.

Reads the NPZ snapshots and ``run_metadata.json`` from a run directory
and writes diagnostic PNGs alongside them. Handles both lat-lon
Mercator and MPAS regional Voronoi outputs (auto-detected from the
metadata).

Quick start::

    JAX_ENABLE_X64=1 python scripts/plot_dino.py results/dino

For a specific snapshot index::

    JAX_ENABLE_X64=1 python scripts/plot_dino.py results/dino --snapshot 5

Plots written:
  - ``snapshots_evolution.png`` — surface T, eta, |u| at 4 evenly-spaced
    snapshots through the run
  - ``timeseries.png`` — |u|, |v| (lat-lon only), |eta|, T extrema, KE
    proxy vs time on log axes (so blowups stand out)
  - ``snapshot_<idx>.png`` — full 4-panel detail of a single snapshot
    (use --snapshot)
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path,
                   help="Run output directory (must contain "
                        "run_metadata.json and snapshots/).")
    p.add_argument("--snapshot", type=int, default=None,
                   help="Render full detail panel for a single "
                        "snapshot index (in addition to the overview "
                        "plots).")
    return p.parse_args()


# ----------------- Lat-lon helpers -----------------

def _mercator_lat_lon_deg(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct cell-center lat/lon (degrees) from run metadata."""
    n_lon = int(meta["grid"]["n_lon"])
    n_lat = int(meta["grid"]["n_lat"])
    cfg = meta["config"]
    dlon_deg = (cfg["lon_east_deg"] - cfg["lon_west_deg"]) / n_lon
    j = np.arange(-(n_lat // 2) + 0.5, n_lat // 2 + 0.5)
    lat_deg = (180.0 / math.pi) * np.arcsin(
        np.tanh(dlon_deg * math.pi / 180.0 * j)
    )
    lon_deg = cfg["lon_west_deg"] + dlon_deg * (np.arange(n_lon) + 0.5)
    return lon_deg, lat_deg


def _surface_speed_latlon(snap: dict) -> np.ndarray:
    u_sfc = 0.5 * (snap["u"][:, :-1, 0] + snap["u"][:, 1:, 0])
    v_sfc = 0.5 * (snap["v"][:-1, :, 0] + snap["v"][1:, :, 0])
    return np.sqrt(u_sfc ** 2 + v_sfc ** 2)


def _plot_evolution_latlon(meta: dict, snaps: list[dict],
                            out_path: Path) -> None:
    lon, lat = _mercator_lat_lon_deg(meta)
    # Drop post-blowup snapshots so pcolormesh doesn't get NaN-poisoned panels
    finite_snaps = [
        s for s in snaps
        if np.isfinite(np.asarray(s["T"][s["land_mask"] > 0.5])).any()
    ]
    if not finite_snaps:
        raise SystemExit("All snapshots are non-finite. Nothing to plot.")
    if len(finite_snaps) < len(snaps):
        print(f"  (Skipping {len(snaps) - len(finite_snaps)} post-blowup snapshot(s) for evolution plot.)")
    snaps = finite_snaps
    n_show = min(4, len(snaps))
    idxs = np.linspace(0, len(snaps) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12),
                              constrained_layout=True)
    fig.suptitle(f"DINO lat-lon Mercator — {len(snaps)} snapshots, "
                 f"showing {n_show} evenly spaced",
                 fontsize=13)
    for col, idx in enumerate(idxs):
        s = snaps[int(idx)]
        day = float(s["time_days"])
        mask = s["land_mask"] > 0.5

        T_sfc = s["T"][:, :, 0].copy()
        T_sfc[~mask] = np.nan
        ax = axes[0, col]
        im = ax.pcolormesh(lon, lat, T_sfc, cmap="RdYlBu_r",
                            vmin=-2, vmax=28, shading="auto")
        ax.set_title(f"Day {day:.1f}: surface T (°C)")
        plt.colorbar(im, ax=ax)

        eta = s["eta"].copy()
        eta[~mask] = np.nan
        ax = axes[1, col]
        em = float(np.nanmax(np.abs(eta))) or 1.0
        im = ax.pcolormesh(lon, lat, eta, cmap="RdBu_r",
                            vmin=-em, vmax=em, shading="auto")
        ax.set_title(f"Day {day:.1f}: η (m, ±{em:.2f})")
        plt.colorbar(im, ax=ax)

        speed = _surface_speed_latlon(s)
        speed[~mask] = np.nan
        ax = axes[2, col]
        sm = float(np.nanmax(speed)) or 1.0
        im = ax.pcolormesh(lon, lat, speed, cmap="magma",
                            vmin=0, vmax=sm, shading="auto")
        ax.set_title(f"Day {day:.1f}: |u| surface (m/s, max {sm:.2f})")
        plt.colorbar(im, ax=ax)

    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# ----------------- MPAS helpers -----------------

def _build_mpas_triangulation(meta: dict):
    """Rebuild the MPAS mesh + Delaunay (drops triangles spanning the
    periodic seam) so we can draw smooth contour fills."""
    from legoesm.grids.voronoi import create_regional_voronoi_mesh
    from scipy.spatial import Delaunay
    cfg = meta["config"]
    res_km = float(meta["args"]["mpas_resolution_km"])
    mesh = create_regional_voronoi_mesh(
        lon_range=(cfg["lon_west_deg"], cfg["lon_east_deg"]),
        lat_range=(-cfg["lat_max_deg"], cfg["lat_max_deg"]),
        resolution_km=res_km,
        periodic_x=True,
    )
    lon = (np.degrees(np.asarray(mesh.lonCell)) + 180.0) % 360.0 - 180.0
    lat = np.degrees(np.asarray(mesh.latCell))
    tri = Delaunay(np.column_stack([lon, lat]))

    def _max_edge(simplex):
        pts = np.column_stack([lon[simplex], lat[simplex]])
        return max(np.linalg.norm(pts[(i + 1) % 3] - pts[i]) for i in range(3))

    edge_lens = np.array([_max_edge(s) for s in tri.simplices])
    keep = tri.simplices[edge_lens < 5.0]
    return mesh, lon, lat, keep


def _plot_evolution_mpas(meta: dict, snaps: list[dict],
                          out_path: Path) -> None:
    from matplotlib.tri import Triangulation
    mesh, lon, lat, keep = _build_mpas_triangulation(meta)
    # Drop snapshots that are post-blowup (T is all NaN)
    finite_snaps = [
        s for s in snaps
        if np.isfinite(np.asarray(s["T"][s["land_mask"] > 0.5])).any()
    ]
    if not finite_snaps:
        raise SystemExit("All snapshots are non-finite (run blew up before "
                         "any clean output). Nothing to plot.")
    if len(finite_snaps) < len(snaps):
        print(f"  (Skipping {len(snaps) - len(finite_snaps)} post-blowup snapshot(s) for evolution plot.)")
    snaps = finite_snaps
    n_show = min(4, len(snaps))
    idxs = np.linspace(0, len(snaps) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(3, n_show, figsize=(5 * n_show, 12),
                              constrained_layout=True)
    fig.suptitle(f"DINO MPAS regional Voronoi ({mesh.nCells} cells) — "
                 f"{len(snaps)} snapshots, showing {n_show}",
                 fontsize=13)

    for col, idx in enumerate(idxs):
        s = snaps[int(idx)]
        day = float(s["time_days"])
        mask = s["land_mask"] > 0.5
        ocean_tris = keep[mask[keep].min(axis=1) > 0.5]
        mtri_oc = Triangulation(lon, lat, triangles=ocean_tris)

        ax = axes[0, col]
        cf = ax.tricontourf(mtri_oc, s["T"][:, 0],
                             levels=np.linspace(0, 28, 15),
                             cmap="RdYlBu_r", extend="both")
        ax.scatter(lon[~mask], lat[~mask], c="dimgray", s=3, marker="s")
        ax.set_title(f"Day {day:.1f}: surface T (°C)")
        plt.colorbar(cf, ax=ax)

        em = max(float(np.nanmax(np.abs(s["eta"]))), 1e-6)
        ax = axes[1, col]
        cf = ax.tricontourf(mtri_oc, s["eta"],
                             levels=np.linspace(-em, em, 21),
                             cmap="RdBu_r", extend="both")
        ax.scatter(lon[~mask], lat[~mask], c="dimgray", s=3, marker="s")
        ax.set_title(f"Day {day:.1f}: η (m, ±{em:.2f})")
        plt.colorbar(cf, ax=ax)

        # |u_edge| → cell-mean
        u_top = np.asarray(s["u"][:, 0])
        edges_on_cell = np.asarray(mesh.edgesOnCell)
        n_edges_on_cell = np.asarray(mesh.nEdgesOnCell)
        u_per_cell = np.zeros(mesh.nCells)
        for ic in range(mesh.nCells):
            ne = int(n_edges_on_cell[ic])
            if ne > 0:
                u_per_cell[ic] = float(np.mean(np.abs(u_top[edges_on_cell[:ne, ic]])))
        sm = max(float(u_per_cell[mask].max()) if mask.any() else 0.0, 1e-6)
        ax = axes[2, col]
        cf = ax.tricontourf(mtri_oc, u_per_cell,
                             levels=np.linspace(0, sm, 16),
                             cmap="magma", extend="max")
        ax.scatter(lon[~mask], lat[~mask], c="dimgray", s=3, marker="s")
        ax.set_title(f"Day {day:.1f}: |u_edge| surface (m/s, max {sm:.2f})")
        plt.colorbar(cf, ax=ax)

    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# ----------------- Time series (works for both grid types) -----------

def _plot_timeseries(snaps: list[dict], grid_kind: str, out_path: Path):
    times, u_max, v_max, eta_max, T_max, T_min, ke_proxy = (
        [], [], [], [], [], [], [],
    )
    for s in snaps:
        mask = s["land_mask"] > 0.5
        times.append(float(s["time_days"]))
        u_max.append(float(np.nanmax(np.abs(s["u"]))))
        v_max.append(
            float(np.nanmax(np.abs(s["v"]))) if "v" in s else float("nan")
        )
        eta_max.append(float(np.nanmax(np.abs(s["eta"]))))
        T_oc = s["T"][mask, :] if mask.any() else np.array([np.nan])
        T_max.append(float(np.nanmax(T_oc)))
        T_min.append(float(np.nanmin(T_oc)))
        # crude KE proxy: mean(u²) over all faces
        ke_proxy.append(0.5 * float(np.nanmean(s["u"] ** 2)))

    t = np.array(times)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"DINO {grid_kind} run — diagnostics evolution",
                 fontsize=13)

    ax = axes[0, 0]
    ax.semilogy(t, np.abs(u_max), "o-", color="C0", label="|u| max")
    if grid_kind == "latlon":
        ax.semilogy(t, np.abs(v_max), "s-", color="C1", label="|v| max")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.4)
    ax.set_ylabel("velocity max (m/s)")
    ax.set_xlabel("time (days)")
    ax.set_title("Max velocity")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(t, np.abs(eta_max), "o-", color="C2")
    ax.set_ylabel("|η| max (m)")
    ax.set_xlabel("time (days)")
    ax.set_title("Max sea-surface height")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, T_max, "o-", color="C3", label="T max")
    ax.plot(t, T_min, "s-", color="C0", label="T min")
    ax.set_ylabel("T (°C)")
    ax.set_xlabel("time (days)")
    ax.set_title("Ocean T extrema")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(t, ke_proxy, "o-", color="C4")
    ax.set_ylabel("KE proxy = ½ ⟨u²⟩")
    ax.set_xlabel("time (days)")
    ax.set_title("Bulk KE proxy")
    ax.grid(alpha=0.3)

    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------- Main -----------------

def main():
    args = _parse_args()

    meta_path = args.run_dir / "run_metadata.json"
    snap_dir = args.run_dir / "snapshots"
    if not meta_path.exists():
        raise SystemExit(f"No run_metadata.json in {args.run_dir}; is "
                         f"this a DINO run output directory?")
    if not snap_dir.is_dir():
        raise SystemExit(f"No snapshots/ in {args.run_dir}.")

    meta = json.loads(meta_path.read_text())
    grid_kind = meta["grid"]["kind"]
    is_latlon = grid_kind.startswith("latlon")

    snaps = []
    for snap_path in sorted(snap_dir.glob("snapshot_*.npz")):
        with np.load(snap_path) as f:
            snaps.append({k: f[k] for k in f.files})
    if not snaps:
        raise SystemExit(f"No snapshots found in {snap_dir}.")

    print(f"Loaded {len(snaps)} snapshot(s) from {snap_dir}")
    print(f"Grid: {grid_kind}")

    evol_path = args.run_dir / "snapshots_evolution.png"
    ts_path = args.run_dir / "timeseries.png"

    if is_latlon:
        _plot_evolution_latlon(meta, snaps, evol_path)
    else:
        _plot_evolution_mpas(meta, snaps, evol_path)
    print(f"Wrote: {evol_path}")

    _plot_timeseries(snaps, "lat-lon" if is_latlon else "MPAS", ts_path)
    print(f"Wrote: {ts_path}")

    if args.snapshot is not None:
        if not (0 <= args.snapshot < len(snaps)):
            raise SystemExit(f"--snapshot {args.snapshot} out of range "
                             f"[0, {len(snaps)-1}]")
        # For now, the evolution plot includes evenly-spaced snapshots.
        # A dedicated detail plot can be added later if needed.
        warnings.warn("--snapshot detail plot not implemented; the "
                      "evolution figure shows 4 evenly-spaced snapshots.",
                      stacklevel=2)


if __name__ == "__main__":
    main()
