#!/usr/bin/env python3
"""Run MPAS (icosahedral) Williamson shallow-water cases with diagnostics.

Outputs are written under ``results/atmosphere/shallow_water`` by default.

This script complements the pytest suite by producing runnable case artifacts:
- per-case `results.txt`
- time-varying diagnostics CSV + plots (mean + conservation views)
- field snapshots (`h`) at initial/mid/final times on native and lat-lon grids
- root `summary.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root is importable when executed as a file script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Avoid matplotlib cache permission issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm import constants
from legoesm.atmosphere.dynamics.shallow_water_mpas import (
    MPASShallowWaterConfig,
    MPASShallowWaterModel,
)
from legoesm.core.operators_voronoi import (
    kinetic_energy_cell,
    potential_vorticity_vertex,
    vertex_thickness,
)
from legoesm.grids.voronoi import VoronoiMesh, create_voronoi_mesh
from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
    compute_error_norms_mpas,
    williamson_test2_mpas,
    williamson_test5_mpas,
    williamson_test6_mpas,
)


@dataclass(frozen=True)
class CaseSpec:
    tag: str
    label: str
    default_days: float
    init_fn: callable


CASE_SPECS: dict[str, CaseSpec] = {
    "tc2": CaseSpec(
        tag="tc2",
        label="Williamson Test 2",
        default_days=5.0,
        init_fn=williamson_test2_mpas,
    ),
    "tc5": CaseSpec(
        tag="tc5",
        label="Williamson Test 5",
        default_days=5.0,
        init_fn=williamson_test5_mpas,
    ),
    "tc6": CaseSpec(
        tag="tc6",
        label="Williamson Test 6",
        default_days=5.0,
        init_fn=williamson_test6_mpas,
    ),
}


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _weighted_mean(x: jax.Array, w: jax.Array) -> float:
    return float(jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0e-30))


def _estimate_dt(mesh: VoronoiMesh, h: jax.Array, cfl: float, g: float) -> float:
    dx_min = float(jnp.min(mesh.dcEdge))
    c = float(jnp.sqrt(g * jnp.max(h)))
    return max(cfl * dx_min / max(c, 1.0e-12), 1.0)


def _diagnostics(state, mesh: VoronoiMesh, g: float) -> dict[str, float]:
    h = state.h.data
    u = state.u.data
    h_s = state.h_s.data

    area_cell = mesh.areaCell
    area_tri = mesh.areaTriangle

    mass = jnp.sum(h * area_cell)

    ke_cell = kinetic_energy_cell(u, mesh)
    kinetic = jnp.sum(ke_cell * h * area_cell)
    potential = jnp.sum(0.5 * g * (h + h_s) ** 2 * area_cell)
    energy = kinetic + potential

    pv_v = potential_vorticity_vertex(u, h, mesh.fVertex, mesh)
    h_v = vertex_thickness(h, mesh)
    enstrophy = 0.5 * jnp.sum((pv_v ** 2) * h_v * area_tri)

    return {
        "mass": float(mass),
        "energy": float(energy),
        "enstrophy": float(enstrophy),
        "h_mean": _weighted_mean(h, area_cell),
        "h_min": float(jnp.min(h)),
        "h_max": float(jnp.max(h)),
        "u_rms": float(jnp.sqrt(jnp.mean(u ** 2))),
        "u_max_abs": float(jnp.max(jnp.abs(u))),
    }


def _write_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_timeseries(rows: list[dict[str, float]], out_path: Path, title: str) -> None:
    if not rows:
        return

    t_days = np.array([r["time_days"] for r in rows], dtype=np.float64)
    mass = np.array([r["mass"] for r in rows], dtype=np.float64)
    energy = np.array([r["energy"] for r in rows], dtype=np.float64)
    enstrophy = np.array([r["enstrophy"] for r in rows], dtype=np.float64)
    h_mean = np.array([r["h_mean"] for r in rows], dtype=np.float64)
    u_rms = np.array([r["u_rms"] for r in rows], dtype=np.float64)

    def rel(x):
        return (x - x[0]) / max(abs(x[0]), 1.0e-30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t_days, rel(mass), label="mass")
    axes[0, 0].plot(t_days, rel(energy), label="energy")
    axes[0, 0].plot(t_days, rel(enstrophy), label="enstrophy")
    axes[0, 0].set_ylabel("Relative drift")
    axes[0, 0].set_xlabel("Time (days)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t_days, h_mean, color="tab:green")
    axes[0, 1].set_ylabel("Mean depth h (m)")
    axes[0, 1].set_xlabel("Time (days)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_days, u_rms, color="tab:orange")
    axes[1, 0].set_ylabel("u normal RMS (m/s)")
    axes[1, 0].set_xlabel("Time (days)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_days, np.array([r["h_min"] for r in rows]), label="h_min")
    axes[1, 1].plot(t_days, np.array([r["h_max"] for r in rows]), label="h_max")
    axes[1, 1].set_ylabel("Depth bounds (m)")
    axes[1, 1].set_xlabel("Time (days)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_h_snapshots(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    pixel_mapper,
    snapshots: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    labels = [k for k in ("initial", "mid", "final") if k in snapshots]
    if not labels:
        return

    vmin = min(float(np.nanmin(snapshots[k])) for k in labels)
    vmax = max(float(np.nanmax(snapshots[k])) for k in labels)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)

    for i, lbl in enumerate(labels):
        ax = axes[0, i]
        h_ll = pixel_mapper(snapshots[lbl])
        sc = ax.imshow(
            h_ll,
            origin="lower",
            extent=(
                float(lon2d[0, 0]),
                float(lon2d[0, -1]),
                float(lat2d[0, 0]),
                float(lat2d[-1, 0]),
            ),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(f"{lbl} h (m, lat-lon pixels)")
        ax.set_xlabel("Longitude (deg)")
        if i == 0:
            ax.set_ylabel("Latitude (deg)")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, shrink=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_h_snapshots_native(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    snapshots: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    labels = [k for k in ("initial", "mid", "final") if k in snapshots]
    if not labels:
        return

    vmin = min(float(np.nanmin(snapshots[k])) for k in labels)
    vmax = max(float(np.nanmax(snapshots[k])) for k in labels)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for i, lbl in enumerate(labels):
        ax = axes[0, i]
        sc = ax.scatter(
            lon_deg,
            lat_deg,
            c=snapshots[lbl],
            s=5.0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,
            rasterized=True,
        )
        ax.set_title(f"{lbl} h (m, native mesh)")
        ax.set_xlabel("Longitude (deg)")
        if i == 0:
            ax.set_ylabel("Latitude (deg)")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, shrink=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_placeholder_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_latlon_pixel_mapper(
    mesh: VoronoiMesh, nlon: int, nlat: int
) -> tuple[np.ndarray, np.ndarray, callable]:
    """Create a nearest-neighbor mapper from Voronoi cells to lat-lon pixels."""
    nlon = max(16, int(nlon))
    nlat = max(8, int(nlat))

    lon_centers = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=np.float64)
    lon_centers = lon_centers + 0.5 * (360.0 / nlon)
    lat_centers = np.linspace(-90.0, 90.0, nlat, dtype=np.float64)
    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)

    lon_cell = np.asarray(mesh.lonCell, dtype=np.float64)
    lat_cell = np.asarray(mesh.latCell, dtype=np.float64)
    x_cell = np.cos(lat_cell) * np.cos(lon_cell)
    y_cell = np.cos(lat_cell) * np.sin(lon_cell)
    z_cell = np.sin(lat_cell)
    xyz_cell = np.stack([x_cell, y_cell, z_cell], axis=1)  # (nCells, 3)

    lon_rad = np.deg2rad(lon2d.ravel())
    lat_rad = np.deg2rad(lat2d.ravel())
    x_pix = np.cos(lat_rad) * np.cos(lon_rad)
    y_pix = np.cos(lat_rad) * np.sin(lon_rad)
    z_pix = np.sin(lat_rad)
    xyz_pix = np.stack([x_pix, y_pix, z_pix], axis=1)  # (nPix, 3)

    nearest_idx = np.empty(xyz_pix.shape[0], dtype=np.int32)
    chunk = 2048
    for i0 in range(0, xyz_pix.shape[0], chunk):
        i1 = min(i0 + chunk, xyz_pix.shape[0])
        dots = xyz_pix[i0:i1] @ xyz_cell.T
        nearest_idx[i0:i1] = np.argmax(dots, axis=1).astype(np.int32)

    def _map(field_cell: np.ndarray) -> np.ndarray:
        field = np.asarray(field_cell, dtype=np.float64).reshape(-1)
        return field[nearest_idx].reshape(nlat, nlon)

    return lon2d, lat2d, _map


def _run_case(
    out_root: Path,
    mesh_level: int,
    case_spec: CaseSpec,
    days: float,
    config: MPASShallowWaterConfig,
    cfl: float,
    lloyd_iterations: int,
    save_every: int,
    latlon_nlon: int,
    latlon_nlat: int,
) -> dict:
    mesh = create_voronoi_mesh(mesh_level, lloyd_iterations=lloyd_iterations)
    state0 = case_spec.init_fn(mesh)
    state = state0

    dt = _estimate_dt(mesh, state.h.data, cfl=cfl, g=config.g)
    n_steps = max(1, int(days * 86400.0 / dt))
    save_every = max(1, save_every)
    model = MPASShallowWaterModel(mesh, config)

    case_dir = out_root / f"{case_spec.tag}_L{mesh_level}_{config.time_integrator}"
    case_dir.mkdir(parents=True, exist_ok=True)

    lon_deg = np.asarray(mesh.lonCell, dtype=np.float64) * 180.0 / np.pi
    lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180.0 / np.pi
    lon_deg = (lon_deg + 360.0) % 360.0
    lon2d, lat2d, pixel_mapper = _build_latlon_pixel_mapper(
        mesh, nlon=latlon_nlon, nlat=latlon_nlat
    )

    rows: list[dict[str, float]] = []
    snapshots: dict[str, np.ndarray] = {"initial": np.asarray(state.h.data, dtype=np.float64)}
    stable = True
    blowup_step = -1

    start = time.time()
    for step in range(1, n_steps + 1):
        state = model.step(state, dt)

        if step == n_steps // 2:
            snapshots["mid"] = np.asarray(state.h.data, dtype=np.float64)
        if step == n_steps:
            snapshots["final"] = np.asarray(state.h.data, dtype=np.float64)

        if (step % save_every == 0) or (step == n_steps):
            jax.block_until_ready(state.h.data)
            diag = _diagnostics(state, mesh, config.g)
            diag["step"] = float(step)
            diag["time_seconds"] = float(step * dt)
            diag["time_days"] = float(step * dt / 86400.0)
            rows.append(diag)

            if not np.isfinite(np.asarray(state.h.data)).all():
                stable = False
                blowup_step = step
                break

    wall = time.time() - start

    if "final" not in snapshots:
        snapshots["final"] = np.asarray(state.h.data, dtype=np.float64)

    _write_csv(rows, case_dir / "diagnostics_timeseries.csv")
    _write_csv(rows, case_dir / "mean_timeseries.csv")
    _write_csv(rows, case_dir / "integrated_timeseries.csv")
    _plot_timeseries(
        rows,
        case_dir / "diagnostics_timeseries.png",
        title=f"{case_spec.label} | icosahedral level {mesh_level}",
    )
    shutil.copy2(case_dir / "diagnostics_timeseries.png", case_dir / "mean_timeseries.png")
    shutil.copy2(case_dir / "diagnostics_timeseries.png", case_dir / "integrated_timeseries.png")
    _plot_h_snapshots(
        lon2d,
        lat2d,
        pixel_mapper,
        snapshots,
        case_dir / "field_snapshots_latlon_pixels.png",
        title=f"{case_spec.label} | icosahedral level {mesh_level}",
    )
    _plot_h_snapshots_native(
        lon_deg,
        lat_deg,
        snapshots,
        case_dir / "field_snapshots_native.png",
        title=f"{case_spec.label} | icosahedral level {mesh_level}",
    )
    shutil.copy2(case_dir / "field_snapshots_latlon_pixels.png", case_dir / "field_snapshots.png")
    latlon_snapshots = {f"{k}_latlon": pixel_mapper(v) for k, v in snapshots.items()}
    np.savez(
        case_dir / "h_snapshots.npz",
        lon_cell_deg=lon_deg,
        lat_cell_deg=lat_deg,
        lon2d=lon2d,
        lat2d=lat2d,
        **snapshots,
        **latlon_snapshots,
    )

    # Conservation diagnostics from global invariants.
    if rows:
        t_days = np.array([float(r["time_days"]) for r in rows], dtype=np.float64)
        mass = np.array([float(r["mass"]) for r in rows], dtype=np.float64)
        energy = np.array([float(r["energy"]) for r in rows], dtype=np.float64)
        mass0 = max(abs(float(mass[0])), 1.0e-30)
        energy0 = max(abs(float(energy[0])), 1.0e-30)
        mrel = (mass - mass[0]) / mass0
        erel = (energy - energy[0]) / energy0

        with (case_dir / "conservation_timeseries.csv").open("w") as f:
            f.write("step,time_days,mass_proxy,energy_proxy,mass_rel,energy_rel\n")
            for i, r in enumerate(rows):
                f.write(
                    f"{int(r['step'])},{r['time_days']:.8f},{mass[i]:.12e},{energy[i]:.12e},"
                    f"{mrel[i]:.12e},{erel[i]:.12e}\n",
                )

        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True)
        axes[0].plot(t_days, mrel, lw=1.8, color="tab:blue")
        axes[0].axhline(0.0, color="0.3", lw=0.8, ls="--")
        axes[0].set_ylabel("Mass drift (rel.)")
        axes[0].set_title("Mass conservation")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t_days, erel, lw=1.8, color="tab:red")
        axes[1].axhline(0.0, color="0.3", lw=0.8, ls="--")
        axes[1].set_ylabel("Energy drift (rel.)")
        axes[1].set_xlabel("Time (days)")
        axes[1].set_title("Energy conservation")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(case_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Shallow-water has no vertical dimension; emit explicit placeholders for
    # standardized atmosphere post-processing interfaces.
    _write_placeholder_plot(
        case_dir / "vertical_profiles.png",
        "Vertical Profile Evolution",
        "N/A for shallow-water (single-layer) dynamics.",
    )
    _write_placeholder_plot(
        case_dir / "zonal_cross_sections.png",
        "Latitude-Vertical Snapshots",
        "N/A for shallow-water (single-layer) dynamics.",
    )
    _write_placeholder_plot(
        case_dir / "meridional_cross_sections.png",
        "Longitude-Vertical Snapshots",
        "N/A for shallow-water (single-layer) dynamics.",
    )
    shutil.copy2(case_dir / "vertical_profiles.png", case_dir / "mean_profiles.png")
    shutil.copy2(case_dir / "zonal_cross_sections.png", case_dir / "lat_vertical_snapshots.png")
    shutil.copy2(case_dir / "meridional_cross_sections.png", case_dir / "lon_vertical_snapshots.png")
    with (case_dir / "vertical_profiles.csv").open("w") as f:
        f.write("time_days,sigma,profile_value\n")
        f.write("0.0,1.0,nan\n")
    shutil.copy2(case_dir / "vertical_profiles.csv", case_dir / "mean_profiles.csv")

    snap_days = [0.0]
    if "mid" in snapshots:
        snap_days.append(0.5 * days)
    snap_days.append(float(days))
    with (case_dir / "snapshot_times.txt").open("w") as f:
        for d in sorted(set(snap_days)):
            f.write(f"day {d:.2f}\n")

    d0 = _diagnostics(state0, mesh, config.g)
    df = _diagnostics(state, mesh, config.g)

    case_result = {
        "case": case_spec.tag,
        "label": case_spec.label,
        "mesh_level": mesh_level,
        "n_cells": int(mesh.nCells),
        "n_edges": int(mesh.nEdges),
        "days": days,
        "dt": dt,
        "n_steps": n_steps,
        "time_integrator": config.time_integrator,
        "pv_scheme": config.pv_scheme,
        "nu_del2": config.nu_del2,
        "nu_del4": config.nu_del4,
        "stable": stable,
        "blowup_step": blowup_step,
        "wall_time_s": wall,
        "mass_drift_rel": (df["mass"] - d0["mass"]) / max(abs(d0["mass"]), 1.0e-30),
        "energy_drift_rel": (df["energy"] - d0["energy"]) / max(abs(d0["energy"]), 1.0e-30),
        "enstrophy_drift_rel": (
            (df["enstrophy"] - d0["enstrophy"]) / max(abs(d0["enstrophy"]), 1.0e-30)
        ),
        "final_h_mean": df["h_mean"],
        "final_u_rms": df["u_rms"],
        "final_u_max_abs": df["u_max_abs"],
    }

    if case_spec.tag == "tc2":
        norms = compute_error_norms_mpas(state.h.data, state0.h.data, mesh)
        case_result["tc2_error_norms"] = norms

    with (case_dir / "results.txt").open("w") as f:
        for k, v in case_result.items():
            if isinstance(v, dict):
                f.write(f"{k}: {json.dumps(v, indent=2)}\n")
            else:
                f.write(f"{k}: {v}\n")

    return case_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run icosahedral (MPAS) Williamson shallow-water cases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "results/atmosphere/shallow_water/icosahedral/williamson_cases_<timestamp>"
        ),
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="tc2,tc5,tc6",
        help="Comma-separated case list from {tc2,tc5,tc6}.",
    )
    parser.add_argument(
        "--mesh-levels",
        type=str,
        default="5",
        help="Comma-separated icosahedral refinement levels (default: 5, ~2-degree).",
    )
    parser.add_argument(
        "--latlon-nlon",
        type=int,
        default=180,
        help="Longitude pixel count for lat-lon snapshot projection.",
    )
    parser.add_argument(
        "--latlon-nlat",
        type=int,
        default=91,
        help="Latitude pixel count for lat-lon snapshot projection.",
    )
    parser.add_argument("--days-tc2", type=float, default=CASE_SPECS["tc2"].default_days)
    parser.add_argument("--days-tc5", type=float, default=CASE_SPECS["tc5"].default_days)
    parser.add_argument("--days-tc6", type=float, default=CASE_SPECS["tc6"].default_days)
    parser.add_argument("--cfl", type=float, default=0.2, help="CFL factor for dt estimate.")
    parser.add_argument("--save-every", type=int, default=20, help="Save diagnostics every N steps.")
    parser.add_argument("--lloyd-iterations", type=int, default=30)
    parser.add_argument("--integrator", choices=("rk4", "ssp_rk3"), default="rk4")
    parser.add_argument("--pv-scheme", choices=("energy", "enstrophy"), default="energy")
    parser.add_argument("--nu-del2", type=float, default=0.0)
    parser.add_argument("--nu-del4", type=float, default=0.0)
    args = parser.parse_args()

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(
            f"results/atmosphere/shallow_water/icosahedral/williamson_cases_{stamp}"
        )
    else:
        out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    case_tags = _parse_csv_list(args.cases)
    for c in case_tags:
        if c not in CASE_SPECS:
            raise ValueError(f"Unsupported case '{c}'. Valid: {sorted(CASE_SPECS)}")

    mesh_levels = _parse_csv_ints(args.mesh_levels)
    if not mesh_levels:
        raise ValueError("No mesh levels provided.")

    config = MPASShallowWaterConfig(
        g=constants.g,
        nu_del2=args.nu_del2,
        nu_del4=args.nu_del4,
        pv_scheme=args.pv_scheme,
        fix_mass=True,
        fix_energy=False,
        time_integrator=args.integrator,
    )

    days_map = {
        "tc2": float(args.days_tc2),
        "tc5": float(args.days_tc5),
        "tc6": float(args.days_tc6),
    }

    print("Running MPAS Williamson cases:")
    print(f"  output={out_root}")
    print(f"  cases={case_tags}")
    print(f"  mesh_levels={mesh_levels}")
    print(f"  latlon_pixels={args.latlon_nlat}x{args.latlon_nlon}")
    print(f"  integrator={args.integrator}, pv_scheme={args.pv_scheme}")
    print(f"  nu_del2={args.nu_del2}, nu_del4={args.nu_del4}, cfl={args.cfl}")

    all_results = []
    total_start = time.time()
    for level in mesh_levels:
        for tag in case_tags:
            spec = CASE_SPECS[tag]
            days = days_map[tag]
            print(f"\n[{tag.upper()}] level={level}, days={days:.2f} ...")
            result = _run_case(
                out_root=out_root,
                mesh_level=level,
                case_spec=spec,
                days=days,
                config=config,
                cfl=float(args.cfl),
                lloyd_iterations=int(args.lloyd_iterations),
                save_every=int(args.save_every),
                latlon_nlon=int(args.latlon_nlon),
                latlon_nlat=int(args.latlon_nlat),
            )
            all_results.append(result)
            print(
                "  stable={stable}, steps={steps}, dt={dt:.2f}s, "
                "mass_drift={md:+.3e}, energy_drift={ed:+.3e}".format(
                    stable=result["stable"],
                    steps=result["n_steps"],
                    dt=result["dt"],
                    md=result["mass_drift_rel"],
                    ed=result["energy_drift_rel"],
                )
            )

    summary = {
        "n_runs": len(all_results),
        "total_wall_time_s": time.time() - total_start,
        "runs": all_results,
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"  Total wall time: {summary['total_wall_time_s']:.2f}s")
    print(f"  Summary: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
