#!/usr/bin/env python
"""Cubed-sphere atmospheric dycore suite at ~2.5 degree with SSP(5,4).

Runs all cubed-sphere atmospheric dycore cases and writes per-case:
- snapshot maps of key fields at selected times
- time series of area-weighted mean fields
- scalar result metrics

Default configuration targets ~2.5 degree horizontal resolution (C36).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None


def _snapshot_steps(n_steps: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    return sorted({0, max(1, n_steps // 2), n_steps})


def _capture_snapshot(
    snapshots: dict[int, dict[str, np.ndarray]],
    target_steps: list[int],
    step_num: int,
    extractor,
    state,
):
    if step_num in target_steps and step_num not in snapshots:
        snapshots[step_num] = extractor(state)


def _cubed_faces_to_mosaic(field_2d_faces: np.ndarray) -> np.ndarray:
    n = field_2d_faces.shape[1]
    mosaic = np.full((2 * n, 3 * n), np.nan, dtype=np.float64)
    face_positions = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    for face, (row, col) in enumerate(face_positions):
        mosaic[row * n:(row + 1) * n, col * n:(col + 1) * n] = field_2d_faces[face]
    return mosaic


def _field_to_panel(field_data) -> np.ndarray:
    arr = np.asarray(field_data)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 6:
        return _cubed_faces_to_mosaic(arr)
    raise ValueError(f"Expected 2D lat-lon or (6,n,n) cubed-sphere field, got {arr.shape}")


def _regrid_faces_to_latlon(
    field_2d_faces: np.ndarray,
    cube_lon_deg: np.ndarray,
    cube_lat_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate cubed-sphere face samples to a regular lat-lon grid.

    Uses inverse-distance weighting of nearest neighbors in 3D Cartesian
    coordinates on the unit sphere. This avoids the Delaunay-triangulation
    seam artifacts that appear when interpolating directly in lon-lat space.
    """
    n = int(field_2d_faces.shape[1])
    n_lon = max(360, 8 * n)
    n_lat = n_lon // 2

    lon = np.asarray(cube_lon_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(cube_lat_deg, dtype=np.float64).reshape(-1)
    val = np.asarray(field_2d_faces, dtype=np.float64).reshape(-1)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = ((lon[valid] + 180.0) % 360.0) - 180.0
    lat = np.clip(lat[valid], -90.0, 90.0)
    val = val[valid]

    lon_cent = np.linspace(-180.0, 180.0, n_lon, endpoint=False) + 180.0 / n_lon
    lat_cent = np.linspace(-90.0, 90.0, n_lat)
    lon2d, lat2d = np.meshgrid(lon_cent, lat_cent)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        cos_lat = np.cos(lat_rad)
        src_xyz = np.column_stack(
            [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)],
        )

        lon_t = np.deg2rad(lon2d.reshape(-1))
        lat_t = np.deg2rad(lat2d.reshape(-1))
        cos_lat_t = np.cos(lat_t)
        tgt_xyz = np.column_stack(
            [cos_lat_t * np.cos(lon_t), cos_lat_t * np.sin(lon_t), np.sin(lat_t)],
        )

        k = min(8, src_xyz.shape[0])
        tree = cKDTree(src_xyz)
        dist, idx = tree.query(tgt_xyz, k=k)
        if k == 1:
            field_ll = val[idx].reshape(lon2d.shape)
        else:
            dist = np.maximum(dist, 1.0e-12)
            w = 1.0 / dist
            w /= np.sum(w, axis=1, keepdims=True)
            field_ll = np.sum(val[idx] * w, axis=1).reshape(lon2d.shape)
    except Exception:
        lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
        lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
        sum_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges), weights=val)
        cnt_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges))
        with np.errstate(invalid="ignore", divide="ignore"):
            field_ll = np.where(cnt_grid > 0.0, sum_grid / cnt_grid, np.nan)

    return lon2d, lat2d, field_ll


def _color_limits(panels: list[np.ndarray]) -> tuple[float, float]:
    finite_chunks = [p[np.isfinite(p)] for p in panels if p is not None]
    finite_chunks = [c for c in finite_chunks if c.size > 0]
    if not finite_chunks:
        return -1.0, 1.0

    values = np.concatenate(finite_chunks)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if vmin < 0.0 < vmax:
        vmax_abs = max(abs(vmin), abs(vmax), 1.0e-12)
        vmin, vmax = -vmax_abs, vmax_abs

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-6
        vmin -= pad
        vmax += pad

    return vmin, vmax


def _format_sim_time(step: int, dt: float) -> str:
    t_sec = step * dt
    if t_sec < 3600.0:
        return f"{t_sec / 60.0:.1f} min"
    if t_sec < 86400.0:
        return f"{t_sec / 3600.0:.2f} h"
    return f"{t_sec / 86400.0:.2f} d"


def _save_case_snapshots(
    test_dir: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    field_specs: list[tuple[str, str, str]],
    cube_lon_deg: np.ndarray,
    cube_lat_deg: np.ndarray,
    central_longitude: float = 0.0,
    cube_plot_mode: str = "latlon",
):
    if not snapshots:
        return

    snap_steps = sorted(snapshots.keys())
    n_rows = len(field_specs)
    n_cols = len(snap_steps)

    cube_mode = (cube_plot_mode or "latlon").lower()
    use_cube_latlon = cube_lon_deg is not None and cube_lat_deg is not None and cube_mode == "latlon"
    use_projected_cube = (
        HAS_CARTOPY and cube_lon_deg is not None and cube_lat_deg is not None and cube_mode == "scatter"
    )

    fig = plt.figure(figsize=((5.0 if use_projected_cube else 4.4) * n_cols + 0.9, 3.4 * n_rows))
    width_ratios = [1.0] * n_cols + [0.06]
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=width_ratios,
        hspace=0.28,
        wspace=0.18,
    )

    if use_projected_cube:
        proj = ccrs.Robinson(central_longitude=central_longitude)
        axes = [
            [fig.add_subplot(gs[row, col], projection=proj) for col in range(n_cols)]
            for row in range(n_rows)
        ]
    else:
        axes = [[fig.add_subplot(gs[row, col]) for col in range(n_cols)] for row in range(n_rows)]
    caxes = [fig.add_subplot(gs[row, n_cols]) for row in range(n_rows)]

    for row, (key, row_label, cmap) in enumerate(field_specs):
        row_panels = []
        lon_plot = None
        lat_plot = None
        for step in snap_steps:
            field_dict = snapshots.get(step, {})
            if key not in field_dict:
                row_panels.append(None)
                continue
            arr = np.asarray(field_dict[key])
            if arr.ndim == 3 and arr.shape[0] == 6 and (use_projected_cube or use_cube_latlon):
                if use_cube_latlon:
                    lon_ll, lat_ll, field_ll = _regrid_faces_to_latlon(
                        arr,
                        cube_lon_deg,
                        cube_lat_deg,
                    )
                    lon_ll = (lon_ll + 360.0) % 360.0
                    order = np.argsort(lon_ll[0, :])
                    lon_ll = lon_ll[:, order]
                    field_ll = field_ll[:, order]
                    row_panels.append(field_ll)
                    if lon_plot is None:
                        lon_plot = lon_ll
                        lat_plot = lat_ll
                else:
                    row_panels.append(arr)
            else:
                row_panels.append(_field_to_panel(arr))

        valid_panels = [p for p in row_panels if p is not None]
        vmin, vmax = _color_limits(valid_panels)
        im = None

        for col, step in enumerate(snap_steps):
            ax = axes[row][col]
            panel = row_panels[col]
            time_label = _format_sim_time(step, dt)

            if panel is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if use_projected_cube and panel.ndim == 3 and panel.shape[0] == 6:
                    lon_pts = np.asarray(cube_lon_deg, dtype=np.float64).reshape(-1)
                    lat_pts = np.asarray(cube_lat_deg, dtype=np.float64).reshape(-1)
                    val_pts = np.asarray(panel, dtype=np.float64).reshape(-1)
                    valid = (
                        np.isfinite(lon_pts)
                        & np.isfinite(lat_pts)
                        & np.isfinite(val_pts)
                    )
                    n_face = int(panel.shape[1])
                    marker_size = max(0.8, 2200.0 / float(n_face * n_face))
                    im = ax.scatter(
                        lon_pts[valid],
                        lat_pts[valid],
                        c=val_pts[valid],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=marker_size,
                        linewidths=0.0,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )
                    ax.set_global()
                    ax.coastlines(linewidth=0.35, color="0.35")
                    ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
                elif use_cube_latlon and panel.ndim == 2 and lon_plot is not None and lat_plot is not None:
                    im = ax.pcolormesh(
                        lon_plot,
                        lat_plot,
                        panel,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        shading="auto",
                    )
                    ax.set_xlim(0.0, 360.0)
                    ax.set_ylim(-90.0, 90.0)
                    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
                    ax.set_yticks([-60, -30, 0, 30, 60])
                    ax.grid(True, alpha=0.15)
                else:
                    im = ax.imshow(
                        panel,
                        origin="lower",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

            ax.set_title(f"{row_label}\nstep {step}, t={time_label}", fontsize=9)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10)

        if im is not None:
            fig.colorbar(im, cax=caxes[row], orientation="vertical")
        else:
            caxes[row].axis("off")

    fig.suptitle(f"{case_name} - Field Snapshots", fontsize=13)
    fig.subplots_adjust(top=0.92)
    fig.savefig(test_dir / "field_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(test_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for step in snap_steps:
            t_sec = step * dt
            f.write(f"{step},{t_sec:.1f},{t_sec/86400.0:.6f}\n")


def _weighted_mean(field: np.ndarray, area: np.ndarray) -> float:
    num = np.sum(field * area)
    den = np.sum(area)
    return float(num / den)


def _record_means(
    series: dict[str, list[float]],
    step: int,
    fields: dict[str, np.ndarray],
    area: np.ndarray,
):
    series["step"].append(float(step))
    for key, arr in fields.items():
        arr_np = np.asarray(arr)
        if arr_np.ndim == 3 and arr_np.shape[0] == 6:
            mean_val = _weighted_mean(arr_np, area)
        else:
            mean_val = float(np.mean(arr_np))
        series[key].append(mean_val)


def _save_mean_timeseries(
    test_dir: Path,
    case_name: str,
    series: dict[str, list[float]],
    dt: float,
    units: dict[str, str] | None = None,
):
    units = units or {}
    steps = np.asarray(series.get("step", []), dtype=float)
    if steps.size == 0:
        return

    t_sec = steps * dt
    t_days = t_sec / 86400.0

    keys = [k for k in series.keys() if k != "step"]

    csv_path = test_dir / "mean_fields_timeseries.csv"
    with open(csv_path, "w") as f:
        header = ["step", "time_seconds", "time_days"] + keys
        f.write(",".join(header) + "\n")
        for i in range(len(steps)):
            row = [f"{steps[i]:.0f}", f"{t_sec[i]:.6f}", f"{t_days[i]:.8f}"]
            row.extend(f"{series[k][i]:.10e}" for k in keys)
            f.write(",".join(row) + "\n")

    fig, axes = plt.subplots(len(keys), 1, figsize=(8.5, 2.6 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        y = np.asarray(series[key], dtype=float)
        ax.plot(t_days, y, lw=1.8)
        unit = units.get(key, "")
        ylabel = key if not unit else f"{key} ({unit})"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} - Mean Field Time Series", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(test_dir / "mean_fields_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_conservation_timeseries(
    test_dir: Path,
    case_name: str,
    series: dict[str, list[float]],
    dt: float,
):
    steps = np.asarray(series.get("step", []), dtype=float)
    if steps.size == 0:
        return

    t_sec = steps * dt
    t_days = t_sec / 86400.0
    mass = np.asarray(series.get("mass", []), dtype=float)
    energy = np.asarray(series.get("energy", []), dtype=float)

    mass_rel = (mass - mass[0]) / max(abs(mass[0]), 1.0e-30)
    energy_rel = (energy - energy[0]) / max(abs(energy[0]), 1.0e-30)

    csv_path = test_dir / "conservation_timeseries.csv"
    with open(csv_path, "w") as f:
        f.write("step,time_seconds,time_days,mass,energy,mass_rel,energy_rel\n")
        for i in range(len(steps)):
            f.write(
                f"{steps[i]:.0f},{t_sec[i]:.6f},{t_days[i]:.8f},"
                f"{mass[i]:.12e},{energy[i]:.12e},{mass_rel[i]:.12e},{energy_rel[i]:.12e}\n",
            )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    axes[0].plot(t_days, mass_rel, color="tab:blue", lw=1.8)
    axes[0].axhline(0.0, color="0.2", lw=0.8, ls="--")
    axes[0].set_ylabel("Relative drift")
    axes[0].set_title("Mass balance")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_days, energy_rel, color="tab:red", lw=1.8)
    axes[1].axhline(0.0, color="0.2", lw=0.8, ls="--")
    axes[1].set_ylabel("Relative drift")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_title("Energy balance")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"{case_name} - Conservation Time Series", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(test_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(test_dir / "mass_energy_balance.txt", "w") as f:
        f.write(f"initial_mass: {mass[0]:.12e}\n")
        f.write(f"final_mass: {mass[-1]:.12e}\n")
        f.write(f"mass_drift_rel: {mass_rel[-1]:.12e}\n")
        f.write(f"initial_energy: {energy[0]:.12e}\n")
        f.write(f"final_energy: {energy[-1]:.12e}\n")
        f.write(f"energy_drift_rel: {energy_rel[-1]:.12e}\n")


def _run_sw_williamson2(out_dir: Path, n: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel as ShallowWaterModel, CDGridShallowWaterConfig as ShallowWaterConfig
    from legoesm.core.conservation import compute_conservation_diagnostics
    from tests.test_cases.williamson import williamson_test2

    case_dir = out_dir / "01_sw_fv_williamson2"
    case_dir.mkdir(parents=True, exist_ok=True)

    dt = 450.0
    n_steps = int(5 * 86400 / dt)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi

    state0 = williamson_test2(grid)
    edge_blend = 0.25 if n >= 24 else 0.0
    config = ShallowWaterConfig(
        hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
        time_integrator=solver,
        edge_blend_strength=edge_blend,
    )
    model = ShallowWaterModel(grid, config)

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_speed": [], "mean_height": []}
    cons_series = {"step": [], "mass": [], "energy": []}

    def _extract(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        wind = np.sqrt(u * u + v * v)
        h = np.asarray(s.h.data)
        return {"wind_speed": wind, "height": h}

    state = state0
    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(series, 0, {"mean_wind_speed": f0["wind_speed"], "mean_height": f0["height"]}, area)
    d0 = compute_conservation_diagnostics(state, grid)
    cons_series["step"].append(0.0)
    cons_series["mass"].append(float(d0["total_mass"]))
    cons_series["energy"].append(float(d0["total_energy"]))

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(series, step, {"mean_wind_speed": fs["wind_speed"], "mean_height": fs["height"]}, area)
            d = compute_conservation_diagnostics(state, grid)
            cons_series["step"].append(float(step))
            cons_series["mass"].append(float(d["total_mass"]))
            cons_series["energy"].append(float(d["total_energy"]))

        if step % progress_every == 0:
            print(f"      SW W2 progress: {step}/{n_steps}")

        if not bool(jnp.all(jnp.isfinite(state.h.data))):
            stable = False
            print(f"      SW W2 non-finite at step {step}")
            break

    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    _save_case_snapshots(
        case_dir,
        f"SW FV Williamson 2 C{n} {solver}",
        snapshots,
        dt,
        [
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("height", "Fluid depth h (m)", "viridis"),
        ],
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"SW FV Williamson 2 C{n} {solver}",
        series,
        dt,
        {"mean_wind_speed": "m/s", "mean_height": "m"},
    )
    _save_conservation_timeseries(
        case_dir,
        f"SW FV Williamson 2 C{n} {solver}",
        cons_series,
        dt,
    )

    h_err = float(jnp.sqrt(jnp.mean((state.h.data - state0.h.data) ** 2)))
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write("land_mask_applied: false\n")
        f.write(f"edge_blend_strength: {edge_blend:.2f}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"l2_error_h: {h_err:.8e}\n")
        f.write(f"mass_drift_rel: {float((cons_series['mass'][-1] - cons_series['mass'][0]) / max(abs(cons_series['mass'][0]), 1.0e-30)):.8e}\n")
        f.write(f"energy_drift_rel: {float((cons_series['energy'][-1] - cons_series['energy'][0]) / max(abs(cons_series['energy'][0]), 1.0e-30)):.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "SW Williamson 2",
        "status": status,
        "metric": "L2(h)",
        "value": h_err,
        "wall_time_s": wall,
    }


def _run_sw_williamson5(out_dir: Path, n: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel as ShallowWaterModel, CDGridShallowWaterConfig as ShallowWaterConfig
    from legoesm.core.conservation import compute_conservation_diagnostics
    from tests.test_cases.williamson import williamson_test5

    case_dir = out_dir / "02_sw_fv_williamson5"
    case_dir.mkdir(parents=True, exist_ok=True)

    dt = 450.0
    n_steps = int(15 * 86400 / dt)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi

    state0 = williamson_test5(grid)
    edge_blend = 0.25 if n >= 24 else 0.0
    config = ShallowWaterConfig(
        hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
        time_integrator=solver,
        edge_blend_strength=edge_blend,
    )
    model = ShallowWaterModel(grid, config)

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_speed": [], "mean_height": []}
    cons_series = {"step": [], "mass": [], "energy": []}

    def _extract(s):
        u = np.asarray(s.u.data)
        v = np.asarray(s.v.data)
        wind = np.sqrt(u * u + v * v)
        h = np.asarray(s.h.data)
        return {"wind_speed": wind, "height": h}

    state = state0
    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(series, 0, {"mean_wind_speed": f0["wind_speed"], "mean_height": f0["height"]}, area)
    d0 = compute_conservation_diagnostics(state, grid)
    cons_series["step"].append(0.0)
    cons_series["mass"].append(float(d0["total_mass"]))
    cons_series["energy"].append(float(d0["total_energy"]))

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(series, step, {"mean_wind_speed": fs["wind_speed"], "mean_height": fs["height"]}, area)
            d = compute_conservation_diagnostics(state, grid)
            cons_series["step"].append(float(step))
            cons_series["mass"].append(float(d["total_mass"]))
            cons_series["energy"].append(float(d["total_energy"]))

        if step % progress_every == 0:
            print(f"      SW W5 progress: {step}/{n_steps}")

        if not bool(jnp.all(jnp.isfinite(state.h.data))):
            stable = False
            print(f"      SW W5 non-finite at step {step}")
            break

    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    _save_case_snapshots(
        case_dir,
        f"SW FV Williamson 5 C{n} {solver}",
        snapshots,
        dt,
        [
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("height", "Fluid depth h (m)", "viridis"),
        ],
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"SW FV Williamson 5 C{n} {solver}",
        series,
        dt,
        {"mean_wind_speed": "m/s", "mean_height": "m"},
    )
    _save_conservation_timeseries(
        case_dir,
        f"SW FV Williamson 5 C{n} {solver}",
        cons_series,
        dt,
    )

    h0 = np.asarray(state0.h.data)
    h1 = np.asarray(state.h.data)
    mass_drift = float(abs(np.mean(h1) - np.mean(h0)) / max(abs(np.mean(h0)), 1.0e-12))
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write("land_mask_applied: false\n")
        f.write(f"edge_blend_strength: {edge_blend:.2f}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"mass_drift: {mass_drift:.8e}\n")
        f.write(f"mass_drift_rel: {float((cons_series['mass'][-1] - cons_series['mass'][0]) / max(abs(cons_series['mass'][0]), 1.0e-30)):.8e}\n")
        f.write(f"energy_drift_rel: {float((cons_series['energy'][-1] - cons_series['energy'][0]) / max(abs(cons_series['energy'][0]), 1.0e-30)):.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "SW Williamson 5",
        "status": status,
        "metric": "mass_drift",
        "value": mass_drift,
        "wall_time_s": wall,
    }


def _run_hydro_held_suarez(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import CDGridPrimitiveEquationModel as PrimitiveEquationModel, CDGridPrimitiveEquationConfig as PrimitiveEquationConfig
    from legoesm.atmosphere.physics.held_suarez import held_suarez_init, held_suarez_forcing
    from legoesm.core.operators import global_integral

    case_dir = out_dir / "03_hydro_fv_held_suarez"
    case_dir.mkdir(parents=True, exist_ok=True)

    dt = 450.0
    n_steps = int(30 * 86400 / dt)

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    edge_uv = 0.15 if n >= 24 else 0.0
    edge_T = 0.10 if n >= 24 else 0.0
    edge_ps = 0.20 if n >= 24 else 0.0
    edge_wd = 2 if n >= 24 else 1

    config = PrimitiveEquationConfig(
        hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
        hyperdiff_ps_coeff=5.0e16 * (48.0 / n) ** 4,
        use_conservation_fixer=True,
        fix_mass=True,
        time_integrator=solver,
        edge_blend_uv=edge_uv,
        edge_blend_T=edge_T,
        edge_blend_p_s=edge_ps,
        edge_blend_width=edge_wd,
    )
    model = PrimitiveEquationModel(grid, sigma, config)
    state = held_suarez_init(grid, sigma)

    mass_init = float(global_integral(state.p_s, grid))

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_sfc": [], "mean_p_s": [], "mean_T_sfc": []}

    def _extract(s):
        u_sfc = np.asarray(s.u.data)[..., -1]
        v_sfc = np.asarray(s.v.data)[..., -1]
        return {
            "wind_sfc": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
            "p_s": np.asarray(s.p_s.data),
            "T_sfc": np.asarray(s.T.data)[..., -1],
        }

    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(
        series,
        0,
        {
            "mean_wind_sfc": f0["wind_sfc"],
            "mean_p_s": f0["p_s"],
            "mean_T_sfc": f0["T_sfc"],
        },
        area,
    )

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)

    for i in range(n_steps):
        state = model.step_with_physics(state, dt, held_suarez_forcing)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(
                series,
                step,
                {
                    "mean_wind_sfc": fs["wind_sfc"],
                    "mean_p_s": fs["p_s"],
                    "mean_T_sfc": fs["T_sfc"],
                },
                area,
            )

        if step % progress_every == 0:
            print(f"      Hydro HS progress: {step}/{n_steps}")

        u_max = float(jnp.max(jnp.abs(state.u.data)))
        if not bool(jnp.all(jnp.isfinite(state.u.data))) or u_max > 700.0:
            stable = False
            print(f"      Hydro HS instability at step {step}, max|u|={u_max:.2f}")
            break

    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    _save_case_snapshots(
        case_dir,
        f"Hydro FV Held-Suarez C{n}/L{nlev} {solver}",
        snapshots,
        dt,
        [
            ("wind_sfc", "Surface wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
            ("T_sfc", "Surface temperature (K)", "coolwarm"),
        ],
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"Hydro FV Held-Suarez C{n}/L{nlev} {solver}",
        series,
        dt,
        {"mean_wind_sfc": "m/s", "mean_p_s": "Pa", "mean_T_sfc": "K"},
    )

    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = abs(mass_final - mass_init) / max(abs(mass_init), 1.0e-12)
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write(f"levels: {nlev}\n")
        f.write(f"edge_blend_uv: {edge_uv:.2f}\n")
        f.write(f"edge_blend_T: {edge_T:.2f}\n")
        f.write(f"edge_blend_p_s: {edge_ps:.2f}\n")
        f.write(f"edge_blend_width: {edge_wd}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"mass_drift: {mass_drift:.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "Hydro Held-Suarez 30d",
        "status": status,
        "metric": "mass_drift",
        "value": mass_drift,
        "wall_time_s": wall,
    }


def _run_hydro_baroclinic(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import CDGridPrimitiveEquationModel as PrimitiveEquationModel, CDGridPrimitiveEquationConfig as PrimitiveEquationConfig
    from legoesm.atmosphere.physics.baroclinic_wave import baroclinic_wave_init
    from legoesm.core.operators import global_integral

    case_dir = out_dir / "04_hydro_fv_baroclinic"
    case_dir.mkdir(parents=True, exist_ok=True)

    dt = 300.0
    n_steps = int(10 * 86400 / dt)

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    edge_uv = 0.15 if n >= 24 else 0.0
    edge_T = 0.10 if n >= 24 else 0.0
    edge_ps = 0.20 if n >= 24 else 0.0
    edge_wd = 2 if n >= 24 else 1

    config = PrimitiveEquationConfig(
        hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
        hyperdiff_ps_coeff=5.0e16 * (48.0 / n) ** 4,
        use_conservation_fixer=True,
        fix_mass=True,
        time_integrator=solver,
        edge_blend_uv=edge_uv,
        edge_blend_T=edge_T,
        edge_blend_p_s=edge_ps,
        edge_blend_width=edge_wd,
    )
    model = PrimitiveEquationModel(grid, sigma, config)
    state = baroclinic_wave_init(grid, sigma, perturbed=True)
    mass_init = float(global_integral(state.p_s, grid))

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_sfc": [], "mean_p_s": [], "mean_T_sfc": []}

    def _extract(s):
        u_sfc = np.asarray(s.u.data)[..., -1]
        v_sfc = np.asarray(s.v.data)[..., -1]
        return {
            "wind_sfc": np.sqrt(u_sfc * u_sfc + v_sfc * v_sfc),
            "p_s": np.asarray(s.p_s.data),
            "T_sfc": np.asarray(s.T.data)[..., -1],
        }

    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(
        series,
        0,
        {
            "mean_wind_sfc": f0["wind_sfc"],
            "mean_p_s": f0["p_s"],
            "mean_T_sfc": f0["T_sfc"],
        },
        area,
    )

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)

    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(
                series,
                step,
                {
                    "mean_wind_sfc": fs["wind_sfc"],
                    "mean_p_s": fs["p_s"],
                    "mean_T_sfc": fs["T_sfc"],
                },
                area,
            )

        if step % progress_every == 0:
            print(f"      Hydro BW progress: {step}/{n_steps}")

        u_max = float(jnp.max(jnp.abs(state.u.data)))
        if not bool(jnp.all(jnp.isfinite(state.u.data))) or u_max > 700.0:
            stable = False
            print(f"      Hydro BW instability at step {step}, max|u|={u_max:.2f}")
            break

    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    _save_case_snapshots(
        case_dir,
        f"Hydro FV Baroclinic C{n}/L{nlev} {solver}",
        snapshots,
        dt,
        [
            ("wind_sfc", "Surface wind speed (m/s)", "magma"),
            ("p_s", "Surface pressure (Pa)", "viridis"),
            ("T_sfc", "Surface temperature (K)", "coolwarm"),
        ],
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"Hydro FV Baroclinic C{n}/L{nlev} {solver}",
        series,
        dt,
        {"mean_wind_sfc": "m/s", "mean_p_s": "Pa", "mean_T_sfc": "K"},
    )

    mass_final = float(global_integral(state.p_s, grid))
    mass_drift = abs(mass_final - mass_init) / max(abs(mass_init), 1.0e-12)
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write(f"levels: {nlev}\n")
        f.write(f"edge_blend_uv: {edge_uv:.2f}\n")
        f.write(f"edge_blend_T: {edge_T:.2f}\n")
        f.write(f"edge_blend_p_s: {edge_ps:.2f}\n")
        f.write(f"edge_blend_width: {edge_wd}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"mass_drift: {mass_drift:.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "Hydro Baroclinic 10d",
        "status": status,
        "metric": "mass_drift",
        "value": mass_drift,
        "wall_time_s": wall,
    }


def _run_nh_tc1(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int, dt: float):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerModel, CompressibleEulerConfig
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init

    case_dir = out_dir / "05_nh_fv_dcmip25_tc1"
    case_dir.mkdir(parents=True, exist_ok=True)

    hours = 1.0
    n_steps = int(hours * 3600.0 / dt)

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi

    state, height_coord, terrain_metric = dcmip25_tc1_init(grid, n_levels=nlev)
    edge_uv = 0.22 if n >= 24 else 0.0
    edge_w = 0.14 if n >= 24 else 0.0
    edge_theta = 0.12 if n >= 24 else 0.0
    edge_rho = 0.18 if n >= 24 else 0.0
    edge_wd = 3 if n >= 24 else 1
    config = CompressibleEulerConfig(
        n_acoustic_substeps=8,
        sponge_width=10000.0,
        sponge_coeff=0.06,
        outer_integrator=solver,
        edge_blend_uv=edge_uv,
        edge_blend_w=edge_w,
        edge_blend_theta=edge_theta,
        edge_blend_rho=edge_rho,
        edge_blend_width=edge_wd,
    )
    model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_low": [], "mean_rho_prime": [], "mean_w_mid": []}

    def _extract(s):
        u_low = np.asarray(s.u.data)[..., -1]
        v_low = np.asarray(s.v.data)[..., -1]
        w = np.asarray(s.w.data)
        return {
            "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
            "rho_prime": np.asarray(s.rho_prime.data)[..., -1],
            "w_mid": w[..., w.shape[-1] // 2],
        }

    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(
        series,
        0,
        {
            "mean_wind_low": f0["wind_low"],
            "mean_rho_prime": f0["rho_prime"],
            "mean_w_mid": f0["w_mid"],
        },
        area,
    )

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(
                series,
                step,
                {
                    "mean_wind_low": fs["wind_low"],
                    "mean_rho_prime": fs["rho_prime"],
                    "mean_w_mid": fs["w_mid"],
                },
                area,
            )

        if step % progress_every == 0:
            print(f"      NH TC1 progress: {step}/{n_steps}")

        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            stable = False
            print(f"      NH TC1 non-finite at step {step}")
            break

    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    _save_case_snapshots(
        case_dir,
        f"NH FV DCMIP TC1 C{n}/L{nlev} {solver}",
        snapshots,
        dt,
        [
            ("wind_low", "Low-level wind speed (m/s)", "magma"),
            ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
            ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
        ],
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"NH FV DCMIP TC1 C{n}/L{nlev} {solver}",
        series,
        dt,
        {"mean_wind_low": "m/s", "mean_rho_prime": "kg/m3", "mean_w_mid": "m/s"},
    )

    w_max = float(jnp.max(jnp.abs(state.w.data)))
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write(f"levels: {nlev}\n")
        f.write(f"edge_blend_uv: {edge_uv:.2f}\n")
        f.write(f"edge_blend_w: {edge_w:.2f}\n")
        f.write(f"edge_blend_theta: {edge_theta:.2f}\n")
        f.write(f"edge_blend_rho: {edge_rho:.2f}\n")
        f.write(f"edge_blend_width: {edge_wd}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"max_abs_w: {w_max:.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "NH DCMIP TC1 1h",
        "status": status,
        "metric": "max_abs_w",
        "value": w_max,
        "wall_time_s": wall,
    }


def _run_nh_tc2a(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int, dt: float):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerModel, CompressibleEulerConfig
    from tests.test_cases.dcmip2025 import dcmip25_tc2_init

    case_dir = out_dir / "06_nh_fv_dcmip25_tc2a"
    case_dir.mkdir(parents=True, exist_ok=True)

    hours = 0.05  # 3 minutes
    n_steps = int(hours * 3600.0 / dt)

    grid = create_cubed_sphere(n)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(grid, n_levels=nlev, subcase="a")

    area = np.asarray(small_grid.area)
    lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi
    edge_uv = 0.22 if n >= 24 else 0.0
    edge_w = 0.22 if n >= 24 else 0.0
    edge_theta = 0.12 if n >= 24 else 0.0
    edge_rho = 0.26 if n >= 24 else 0.0
    edge_wd = 4 if n >= 24 else 1

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=15000.0,
        sponge_coeff=1.0 / (0.1 * 86400.0),
        small_earth_factor=20.0,
        outer_integrator=solver,
        edge_blend_uv=edge_uv,
        edge_blend_w=edge_w,
        edge_blend_theta=edge_theta,
        edge_blend_rho=edge_rho,
        edge_blend_width=edge_wd,
    )
    model = CompressibleEulerModel(small_grid, height_coord, terrain_metric, config)

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_low": [], "mean_rho_prime": [], "mean_w_mid": []}

    def _extract(s):
        u_low = np.asarray(s.u.data)[..., -1]
        v_low = np.asarray(s.v.data)[..., -1]
        w = np.asarray(s.w.data)
        return {
            "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
            "rho_prime": np.asarray(s.rho_prime.data)[..., -1],
            "w_mid": w[..., w.shape[-1] // 2],
        }

    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(
        series,
        0,
        {
            "mean_wind_low": f0["wind_low"],
            "mean_rho_prime": f0["rho_prime"],
            "mean_w_mid": f0["w_mid"],
        },
        area,
    )

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(
                series,
                step,
                {
                    "mean_wind_low": fs["wind_low"],
                    "mean_rho_prime": fs["rho_prime"],
                    "mean_w_mid": fs["w_mid"],
                },
                area,
            )

        if step % progress_every == 0:
            print(f"      NH TC2a progress: {step}/{n_steps}")

        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            stable = False
            print(f"      NH TC2a non-finite at step {step}")
            break

    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    _save_case_snapshots(
        case_dir,
        f"NH FV DCMIP TC2a C{n}/L{nlev} {solver}",
        snapshots,
        dt,
        [
            ("wind_low", "Low-level wind speed (m/s)", "magma"),
            ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
            ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
        ],
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"NH FV DCMIP TC2a C{n}/L{nlev} {solver}",
        series,
        dt,
        {"mean_wind_low": "m/s", "mean_rho_prime": "kg/m3", "mean_w_mid": "m/s"},
    )

    w_max = float(jnp.max(jnp.abs(state.w.data)))
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write(f"levels: {nlev}\n")
        f.write(f"edge_blend_uv: {edge_uv:.2f}\n")
        f.write(f"edge_blend_w: {edge_w:.2f}\n")
        f.write(f"edge_blend_theta: {edge_theta:.2f}\n")
        f.write(f"edge_blend_rho: {edge_rho:.2f}\n")
        f.write(f"edge_blend_width: {edge_wd}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"max_abs_w: {w_max:.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "NH DCMIP TC2a 3min",
        "status": status,
        "metric": "max_abs_w",
        "value": w_max,
        "wall_time_s": wall,
    }


def _run_nh_tc3(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int, dt: float):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import CompressibleEulerModel, CompressibleEulerConfig
    from tests.test_cases.dcmip2025 import dcmip25_tc3_init

    case_dir = out_dir / "07_nh_fv_dcmip25_tc3"
    case_dir.mkdir(parents=True, exist_ok=True)

    hours = 0.05  # 3 minutes
    n_steps = int(hours * 3600.0 / dt)

    grid = create_cubed_sphere(n)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc3_init(grid, n_levels=nlev)

    area = np.asarray(small_grid.area)
    lon_deg = np.asarray(small_grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(small_grid.lat) * 180.0 / np.pi

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=5000.0,
        sponge_coeff=0.05,
        small_earth_factor=60.0,
        use_coriolis=False,
        outer_integrator=solver,
    )
    model = CompressibleEulerModel(small_grid, height_coord, terrain_metric, config)

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_wind_low": [], "mean_rho_prime": [], "mean_w_mid": []}

    def _extract(s):
        u_low = np.asarray(s.u.data)[..., -1]
        v_low = np.asarray(s.v.data)[..., -1]
        w = np.asarray(s.w.data)
        out = {
            "wind_low": np.sqrt(u_low * u_low + v_low * v_low),
            "rho_prime": np.asarray(s.rho_prime.data)[..., -1],
            "w_mid": w[..., w.shape[-1] // 2],
        }
        tracers = np.asarray(s.tracers.data)
        if tracers.ndim == 5 and tracers.shape[-1] >= 3:
            out["q_rain"] = np.clip(tracers[..., -1, 2], 0.0, None)
        return out

    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(
        series,
        0,
        {
            "mean_wind_low": f0["wind_low"],
            "mean_rho_prime": f0["rho_prime"],
            "mean_w_mid": f0["w_mid"],
        },
        area,
    )

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(
                series,
                step,
                {
                    "mean_wind_low": fs["wind_low"],
                    "mean_rho_prime": fs["rho_prime"],
                    "mean_w_mid": fs["w_mid"],
                },
                area,
            )

        if step % progress_every == 0:
            print(f"      NH TC3 progress: {step}/{n_steps}")

        if not bool(jnp.all(jnp.isfinite(state.u.data))):
            stable = False
            print(f"      NH TC3 non-finite at step {step}")
            break

    jax.block_until_ready(state.u.data)
    wall = time.time() - t0

    field_specs = [
        ("wind_low", "Low-level wind speed (m/s)", "magma"),
        ("rho_prime", "Density perturbation (kg/m3)", "RdBu_r"),
        ("w_mid", "Mid-level vertical w (m/s)", "RdBu_r"),
    ]
    if any("q_rain" in s for s in snapshots.values()):
        field_specs.append(("q_rain", "Rain tracer q_rain", "Blues"))

    _save_case_snapshots(
        case_dir,
        f"NH FV DCMIP TC3 C{n}/L{nlev} {solver}",
        snapshots,
        dt,
        field_specs,
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"NH FV DCMIP TC3 C{n}/L{nlev} {solver}",
        series,
        dt,
        {"mean_wind_low": "m/s", "mean_rho_prime": "kg/m3", "mean_w_mid": "m/s"},
    )

    w_max = float(jnp.max(jnp.abs(state.w.data)))
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write(f"levels: {nlev}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"max_abs_w: {w_max:.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "NH DCMIP TC3 3min",
        "status": status,
        "metric": "max_abs_w",
        "value": w_max,
        "wall_time_s": wall,
    }


def _run_transport(out_dir: Path, n: int, nlev: int, solver: str, mean_every: int):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.tracer_transport import TracerTransportModel, TracerTransportConfig
    from tests.test_cases.dcmip_transport import (
        dcmip11_wind,
        dcmip11_init,
        compute_tracer_error_norms,
        create_dcmip_sigma,
    )

    case_dir = out_dir / "08_transport_dcmip12_11"
    case_dir.mkdir(parents=True, exist_ok=True)

    dt = 900.0
    n_steps = int(12 * 86400 / dt)

    grid = create_cubed_sphere(n)
    sigma = create_dcmip_sigma(nlev)
    area = np.asarray(grid.area)
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi

    state0 = dcmip11_init(grid, sigma)
    config = TracerTransportConfig(hyperdiff_coeff=0.0, time_integrator=solver)
    model = TracerTransportModel(grid, sigma, dcmip11_wind, config)

    snapshots = {}
    snap_targets = _snapshot_steps(n_steps)
    series = {"step": [], "mean_q1": [], "mean_q2": []}

    def _extract(s):
        tracers = np.asarray(s.tracers.data)
        out = {"q1": np.clip(tracers[..., -1, 0], 0.0, None)}
        if tracers.shape[-1] >= 2:
            out["q2"] = np.clip(tracers[..., -1, 1], 0.0, None)
        return out

    state = state0
    _capture_snapshot(snapshots, snap_targets, 0, _extract, state)
    f0 = _extract(state)
    _record_means(series, 0, {"mean_q1": f0["q1"], "mean_q2": f0.get("q2", f0["q1"])}, area)

    t0 = time.time()
    stable = True
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        step = i + 1

        _capture_snapshot(snapshots, snap_targets, step, _extract, state)
        if step % mean_every == 0 or step == n_steps:
            fs = _extract(state)
            _record_means(series, step, {"mean_q1": fs["q1"], "mean_q2": fs.get("q2", fs["q1"])}, area)

        if step % progress_every == 0:
            print(f"      Transport progress: {step}/{n_steps}")

        if not bool(jnp.all(jnp.isfinite(state.tracers.data))):
            stable = False
            print(f"      Transport non-finite at step {step}")
            break

    jax.block_until_ready(state.tracers.data)
    wall = time.time() - t0

    field_specs = [
        ("q1", "Tracer q1 (surface)", "viridis"),
        ("q2", "Tracer q2 (surface)", "plasma"),
    ]
    _save_case_snapshots(
        case_dir,
        f"Transport DCMIP 1-1 C{n}/L{nlev} {solver}",
        snapshots,
        dt,
        field_specs,
        lon_deg,
        lat_deg,
    )
    _save_mean_timeseries(
        case_dir,
        f"Transport DCMIP 1-1 C{n}/L{nlev} {solver}",
        series,
        dt,
        {"mean_q1": "kg/kg", "mean_q2": "kg/kg"},
    )

    norms = compute_tracer_error_norms(state, state0, grid)
    l2_q1 = float(norms["l2"][0])
    status = "PASS" if stable else "FAIL"

    with open(case_dir / "results.txt", "w") as f:
        f.write(f"solver: {solver}\n")
        f.write(f"resolution: C{n}\n")
        f.write(f"levels: {nlev}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"l2_q1: {l2_q1:.8e}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")

    return {
        "case": "Transport DCMIP 1-1 12d",
        "status": status,
        "metric": "l2_q1",
        "value": l2_q1,
        "wall_time_s": wall,
    }


def _write_summary(out_dir: Path, results: list[dict], args):
    total_wall = float(sum(r.get("wall_time_s", 0.0) for r in results))
    n_pass = sum(1 for r in results if r.get("status") == "PASS")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")
    n_error = sum(1 for r in results if r.get("status") == "ERROR")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "resolution": args.resolution,
            "hydro_levels": args.hydro_levels,
            "nh_levels": args.nh_levels,
            "solver": args.solver,
            "mean_every": args.mean_every,
            "dt_nh_tc1": args.dt_nh_tc1,
            "dt_nh_small": args.dt_nh_small,
        },
        "summary": {
            "total_cases": len(results),
            "pass": n_pass,
            "fail": n_fail,
            "error": n_error,
            "total_wall_time_s": total_wall,
        },
        "cases": results,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append("# Cubed-Sphere Atmospheric Dycore Suite")
    lines.append("")
    lines.append(f"Generated: {payload['generated']}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Resolution: C{args.resolution} (~{90.0/args.resolution:.2f} degree)")
    lines.append(f"- Hydro levels: {args.hydro_levels}")
    lines.append(f"- NH levels: {args.nh_levels}")
    lines.append(f"- Solver: {args.solver}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Case | Status | Metric | Value | Wall Time (s) |")
    lines.append("|---|---|---|---:|---:|")
    for r in results:
        val = r.get("value")
        if isinstance(val, float):
            val_str = f"{val:.6e}"
        else:
            val_str = str(val)
        lines.append(
            f"| {r.get('case')} | {r.get('status')} | {r.get('metric')} | {val_str} | {r.get('wall_time_s', 0.0):.2f} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: {len(results)}")
    lines.append(f"- PASS: {n_pass}")
    lines.append(f"- FAIL: {n_fail}")
    lines.append(f"- ERROR: {n_error}")
    lines.append(f"- Total wall time: {total_wall:.1f} s ({total_wall/60.0:.2f} min)")

    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cubed-sphere atmospheric dycore suite with SSP(5,4)")
    parser.add_argument("--resolution", type=int, default=36, help="Cubed-sphere C resolution (default: 36 ~= 2.5 degree)")
    parser.add_argument("--hydro-levels", type=int, default=10, help="Vertical levels for hydro/transport")
    parser.add_argument("--nh-levels", type=int, default=20, help="Vertical levels for non-hydrostatic tests")
    parser.add_argument("--solver", type=str, default="ssp45", help="Time integrator label (e.g., ssp45)")
    parser.add_argument("--mean-every", type=int, default=20, help="Record mean fields every N steps")
    parser.add_argument("--dt-nh-tc1", type=float, default=1.0, help="DT for NH TC1 [s]")
    parser.add_argument("--dt-nh-small", type=float, default=0.2, help="DT for NH TC2a/TC3 [s]")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/atmosphere/dycore_cube_C{resolution}_{solver})",
    )
    parser.add_argument("--skip-nh", action="store_true", help="Skip NH DCMIP cases")
    parser.add_argument("--skip-transport", action="store_true", help="Skip transport case")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=("sw2", "sw5", "hydro_hs", "hydro_bw", "nh_tc1", "nh_tc2a", "nh_tc3", "transport"),
        help="Run a single case only",
    )
    args = parser.parse_args()

    out_dir = args.output or Path(f"results/atmosphere/dycore_cube_C{args.resolution}_{args.solver}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Cubed-Sphere Atmospheric Dycore Suite")
    print("=" * 72)
    print(f"Backend: {jax.default_backend()}")
    print(f"X64: {jax.config.jax_enable_x64}")
    print(f"Devices: {jax.devices()}")
    print(f"Resolution: C{args.resolution} (~{90.0/args.resolution:.2f} degree)")
    print(f"Solver: {args.solver}")
    print(f"Output: {out_dir}")

    results = []

    runners = [
        ("SW Williamson 2", lambda: _run_sw_williamson2(out_dir, args.resolution, args.solver, args.mean_every)),
        ("SW Williamson 5", lambda: _run_sw_williamson5(out_dir, args.resolution, args.solver, args.mean_every)),
        (
            "Hydro Held-Suarez",
            lambda: _run_hydro_held_suarez(
                out_dir,
                args.resolution,
                args.hydro_levels,
                args.solver,
                args.mean_every,
            ),
        ),
        (
            "Hydro Baroclinic",
            lambda: _run_hydro_baroclinic(
                out_dir,
                args.resolution,
                args.hydro_levels,
                args.solver,
                args.mean_every,
            ),
        ),
    ]

    if not args.skip_nh:
        runners.extend(
            [
                (
                    "NH TC1",
                    lambda: _run_nh_tc1(
                        out_dir,
                        args.resolution,
                        args.nh_levels,
                        args.solver,
                        args.mean_every,
                        args.dt_nh_tc1,
                    ),
                ),
                (
                    "NH TC2a",
                    lambda: _run_nh_tc2a(
                        out_dir,
                        args.resolution,
                        args.nh_levels,
                        args.solver,
                        args.mean_every,
                        args.dt_nh_small,
                    ),
                ),
                (
                    "NH TC3",
                    lambda: _run_nh_tc3(
                        out_dir,
                        args.resolution,
                        args.nh_levels,
                        args.solver,
                        args.mean_every,
                        args.dt_nh_small,
                    ),
                ),
            ]
        )

    if not args.skip_transport:
        runners.append(
            (
                "Transport",
                lambda: _run_transport(
                    out_dir,
                    args.resolution,
                    args.hydro_levels,
                    args.solver,
                    args.mean_every,
                ),
            )
        )

    if args.only is not None:
        only_map = {
            "sw2": "SW Williamson 2",
            "sw5": "SW Williamson 5",
            "hydro_hs": "Hydro Held-Suarez",
            "hydro_bw": "Hydro Baroclinic",
            "nh_tc1": "NH TC1",
            "nh_tc2a": "NH TC2a",
            "nh_tc3": "NH TC3",
            "transport": "Transport",
        }
        target = only_map[args.only]
        runners = [r for r in runners if r[0] == target]

    t_all = time.time()
    for name, fn in runners:
        print("\n" + "-" * 72)
        print(f"Running case: {name}")
        print("-" * 72)
        try:
            res = fn()
            results.append(res)
            print(f"  {res['status']} | {res['metric']}={res['value']:.6e} | {res['wall_time_s']:.1f}s")
        except Exception as exc:
            traceback.print_exc()
            results.append(
                {
                    "case": name,
                    "status": "ERROR",
                    "metric": "error",
                    "value": str(exc),
                    "wall_time_s": 0.0,
                }
            )

    total = time.time() - t_all
    _write_summary(out_dir, results, args)

    print("\n" + "=" * 72)
    print(f"Suite complete in {total:.1f}s ({total/60.0:.2f} min)")
    print("=" * 72)
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
