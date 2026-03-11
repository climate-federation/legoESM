#!/usr/bin/env python
"""Run shallow-water Williamson tests with harmonized diagnostics.

This script runs (for Williamson 2 and/or 5):
1) Spectral shallow water on Gaussian (lat-lon) grid
2) Finite-volume shallow water on cubed-sphere, remapped to lat-lon outputs
3) Finite-volume shallow water on cubed-sphere native map outputs

Each case writes identical output products:
- field_snapshots.png
- snapshot_times.txt
- mean_fields_timeseries.csv/.png
- conservation_timeseries.csv/.png
- timeseries.npz
- slab_timeseries.csv/.npz/.png
- results.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local `src/` and `tests/` imports work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    ccrs = None
    HAS_CARTOPY = False


def _snapshot_steps(n_steps: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    return sorted({0, max(1, n_steps // 2), n_steps})


def _format_sim_time(step: int, dt: float) -> str:
    t_sec = step * dt
    if t_sec < 3600.0:
        return f"{t_sec / 60.0:.1f} min"
    if t_sec < 86400.0:
        return f"{t_sec / 3600.0:.2f} h"
    return f"{t_sec / 86400.0:.2f} d"


def _weighted_mean(field: np.ndarray, area: np.ndarray) -> float:
    return float(np.sum(field * area) / np.sum(area))


def _gaussian_area_weights(grid) -> np.ndarray:
    # Area weights normalized up to a constant scale are sufficient for means.
    w_lat = np.asarray(grid.weights, dtype=np.float64)
    n_lon = int(grid.n_lon)
    return np.broadcast_to(w_lat[:, None] / float(n_lon), (w_lat.shape[0], n_lon))


def _series_init(keys: list[str]) -> dict[str, list[float]]:
    out = {"step": []}
    for k in keys:
        out[k] = []
    return out


def _series_push(series: dict[str, list[float]], step: int, vals: dict[str, float]):
    series["step"].append(float(step))
    for k, v in vals.items():
        series[k].append(float(v))


def _color_limits(arrays: list[np.ndarray], symmetric: bool = False) -> tuple[float, float]:
    finite = [a[np.isfinite(a)] for a in arrays if a is not None]
    finite = [a for a in finite if a.size > 0]
    if not finite:
        return -1.0, 1.0
    vals = np.concatenate(finite)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if symmetric or (vmin < 0.0 < vmax):
        m = max(abs(vmin), abs(vmax), 1.0e-12)
        vmin, vmax = -m, m
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-6
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _is_symmetric_snapshot_field(key: str) -> bool:
    return key in ("u", "v") or ("vorticity" in key)


def _save_snapshot_times(out_dir: Path, steps: list[int], dt: float) -> None:
    with open(out_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for st in steps:
            t_sec = st * dt
            f.write(f"{st},{t_sec:.6f},{t_sec/86400.0:.8f}\n")


def _save_mean_timeseries(
    out_dir: Path,
    case_name: str,
    series: dict[str, list[float]],
    dt: float,
    units: dict[str, str],
) -> None:
    steps = np.asarray(series.get("step", []), dtype=float)
    if steps.size == 0:
        return

    keys = [k for k in series if k != "step"]
    t_sec = steps * dt
    t_days = t_sec / 86400.0

    with open(out_dir / "mean_fields_timeseries.csv", "w") as f:
        f.write(",".join(["step", "time_seconds", "time_days"] + keys) + "\n")
        for i in range(steps.size):
            row = [f"{steps[i]:.0f}", f"{t_sec[i]:.6f}", f"{t_days[i]:.8f}"]
            row.extend(f"{series[k][i]:.10e}" for k in keys)
            f.write(",".join(row) + "\n")

    fig, axes = plt.subplots(len(keys), 1, figsize=(9.0, 2.8 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.plot(t_days, np.asarray(series[key], dtype=float), lw=1.8)
        unit = units.get(key, "")
        ax.set_ylabel(key if not unit else f"{key} ({unit})")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} - Mean Field Time Series", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "mean_fields_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _weighted_stats_2d(field_2d: np.ndarray, lat_weights: np.ndarray) -> tuple[float, float]:
    """Return area-weighted mean and std for (lat, lon) field."""
    lon_mean = np.mean(field_2d, axis=1)
    lon_mean_sq = np.mean(field_2d * field_2d, axis=1)
    wsum = float(np.sum(lat_weights))
    mean = float(np.sum(lat_weights * lon_mean) / wsum)
    mean_sq = float(np.sum(lat_weights * lon_mean_sq) / wsum)
    std = float(np.sqrt(max(mean_sq - mean * mean, 0.0)))
    return mean, std


def _lat_weights_from_lat2d(lat2d_deg: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(np.asarray(lat2d_deg, dtype=np.float64)[:, 0])
    return np.clip(np.cos(lat_rad), 1.0e-12, None)


def _compute_slab_values(
    step_idx: int,
    dt: float,
    fields: dict[str, np.ndarray],
    lat_weights: np.ndarray,
) -> dict[str, float]:
    """Compute slab/global-mean diagnostics from lat-lon fields."""
    h = np.asarray(fields["height"], dtype=np.float64)
    speed = np.asarray(fields["wind_speed"], dtype=np.float64)
    vor = np.asarray(fields["vorticity"], dtype=np.float64)

    h_mean, h_std = _weighted_stats_2d(h, lat_weights)
    speed_mean, speed_std = _weighted_stats_2d(speed, lat_weights)
    vor_mean, vor_std = _weighted_stats_2d(vor, lat_weights)
    vor_rms = float(np.sqrt(vor_mean * vor_mean + vor_std * vor_std))

    return {
        "step": float(step_idx),
        "time_days": float(step_idx * dt / 86400.0),
        "h_mean": h_mean,
        "h_std": h_std,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "vor_mean": vor_mean,
        "vor_rms": vor_rms,
    }


def _save_slab_timeseries(out_dir: Path, case_name: str, slab_values: list[dict[str, float]]) -> None:
    if not slab_values:
        return
    step = np.asarray([v["step"] for v in slab_values], dtype=np.float64)
    time_days = np.asarray([v["time_days"] for v in slab_values], dtype=np.float64)
    h_mean = np.asarray([v["h_mean"] for v in slab_values], dtype=np.float64)
    h_std = np.asarray([v["h_std"] for v in slab_values], dtype=np.float64)
    speed_mean = np.asarray([v["speed_mean"] for v in slab_values], dtype=np.float64)
    speed_std = np.asarray([v["speed_std"] for v in slab_values], dtype=np.float64)
    vor_mean = np.asarray([v["vor_mean"] for v in slab_values], dtype=np.float64)
    vor_rms = np.asarray([v["vor_rms"] for v in slab_values], dtype=np.float64)

    with open(out_dir / "slab_timeseries.csv", "w") as f:
        f.write("step,time_days,h_mean_m,h_std_m,wind_mean_ms,wind_std_ms,vort_mean_s-1,vort_rms_s-1\n")
        for i in range(step.size):
            f.write(
                f"{step[i]:.0f},{time_days[i]:.8f},"
                f"{h_mean[i]:.10e},{h_std[i]:.10e},"
                f"{speed_mean[i]:.10e},{speed_std[i]:.10e},"
                f"{vor_mean[i]:.10e},{vor_rms[i]:.10e}\n",
            )

    np.savez(
        out_dir / "slab_timeseries.npz",
        step=step,
        days=time_days,
        time_days=time_days,
        h_mean=h_mean,
        h_std=h_std,
        speed_mean=speed_mean,
        speed_std=speed_std,
        vor_mean=vor_mean,
        vor_rms=vor_rms,
    )

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 10.0), sharex=True)
    axes[0].plot(time_days, h_mean, "k-", lw=1.8, label="mean h")
    axes[0].plot(time_days, h_std, "k--", lw=1.3, label="std(h)")
    axes[0].set_ylabel("Fluid depth [m]")
    axes[0].set_title("Slab/Global Mean Depth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time_days, speed_mean, color="tab:orange", lw=1.8, label="mean |u|")
    axes[1].plot(time_days, speed_std, color="tab:orange", ls="--", lw=1.3, label="std(|u|)")
    axes[1].set_ylabel("Wind speed [m/s]")
    axes[1].set_title("Slab/Global Mean Wind")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time_days, vor_mean, color="tab:blue", lw=1.8, label="mean vort")
    axes[2].plot(time_days, vor_rms, color="tab:blue", ls="--", lw=1.3, label="rms vort")
    axes[2].set_ylabel("Vorticity [1/s]")
    axes[2].set_title("Slab/Global Mean Vorticity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Time [days]")

    fig.suptitle(f"{case_name} - Slab/Global Mean Diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "slab_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_conservation_timeseries(
    out_dir: Path,
    case_name: str,
    series: dict[str, list[float]],
    dt: float,
) -> None:
    steps = np.asarray(series.get("step", []), dtype=float)
    if steps.size == 0:
        return

    t_sec = steps * dt
    t_days = t_sec / 86400.0
    mass = np.asarray(series["mass"], dtype=float)
    energy = np.asarray(series["energy"], dtype=float)
    enstrophy = np.asarray(series["enstrophy"], dtype=float)
    mass_rel = (mass - mass[0]) / max(abs(mass[0]), 1.0e-30)
    energy_rel = (energy - energy[0]) / max(abs(energy[0]), 1.0e-30)
    enstrophy_rel = (enstrophy - enstrophy[0]) / max(abs(enstrophy[0]), 1.0e-30)

    np.savez(
        out_dir / "timeseries.npz",
        step=steps,
        days=t_days,
        time_days=t_days,
        mass=mass,
        energy=energy,
        enstrophy=enstrophy,
        mass_rel=mass_rel,
        energy_rel=energy_rel,
        enstrophy_rel=enstrophy_rel,
    )

    with open(out_dir / "conservation_timeseries.csv", "w") as f:
        f.write(
            "step,time_seconds,time_days,mass,energy,enstrophy,mass_rel,energy_rel,enstrophy_rel\n",
        )
        for i in range(steps.size):
            f.write(
                f"{steps[i]:.0f},{t_sec[i]:.6f},{t_days[i]:.8f},"
                f"{mass[i]:.12e},{energy[i]:.12e},{enstrophy[i]:.12e},"
                f"{mass_rel[i]:.12e},{energy_rel[i]:.12e},{enstrophy_rel[i]:.12e}\n",
            )

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.0), sharex=True)
    curves = [
        ("Mass balance", mass_rel, "tab:blue"),
        ("Energy balance", energy_rel, "tab:red"),
        ("Enstrophy balance", enstrophy_rel, "tab:green"),
    ]
    for ax, (title, y, color) in zip(axes, curves):
        ax.plot(t_days, y, color=color, lw=1.8)
        ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
        ax.set_ylabel("Relative drift")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} - Conservation Time Series", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "conservation_timeseries.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "conservation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _regrid_faces_to_latlon(
    field_2d_faces: np.ndarray,
    cube_lon_deg: np.ndarray,
    cube_lat_deg: np.ndarray,
    n_lon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate cubed-sphere face values to a regular lat-lon grid."""
    n = int(field_2d_faces.shape[1])
    n_lon_out = int(n_lon or max(360, 8 * n))
    n_lat_out = n_lon_out // 2

    lon = np.asarray(cube_lon_deg, dtype=np.float64).reshape(-1)
    lat = np.asarray(cube_lat_deg, dtype=np.float64).reshape(-1)
    val = np.asarray(field_2d_faces, dtype=np.float64).reshape(-1)

    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon = ((lon[valid] + 180.0) % 360.0) - 180.0
    lat = np.clip(lat[valid], -90.0, 90.0)
    val = val[valid]

    lon_cent = np.linspace(-180.0, 180.0, n_lon_out, endpoint=False) + 180.0 / n_lon_out
    lat_cent = np.linspace(-90.0, 90.0, n_lat_out)
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
        lon_edges = np.linspace(-180.0, 180.0, n_lon_out + 1)
        lat_edges = np.linspace(-90.0, 90.0, n_lat_out + 1)
        sum_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges), weights=val)
        cnt_grid, _, _ = np.histogram2d(lat, lon, bins=(lat_edges, lon_edges))
        with np.errstate(invalid="ignore", divide="ignore"):
            field_ll = np.where(cnt_grid > 0.0, sum_grid / cnt_grid, np.nan)

    return lon2d, lat2d, field_ll


def _save_snapshots_latlon(
    out_dir: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    field_specs: list[tuple[str, str, str]],
    lon2d_deg: np.ndarray,
    lat2d_deg: np.ndarray,
    *,
    projection: str = "platecarree",
    draw_coastlines: bool = False,
) -> None:
    if not snapshots:
        return
    steps = sorted(snapshots.keys())
    n_rows = len(field_specs)
    n_cols = len(steps)

    fig = plt.figure(figsize=(5.0 * n_cols + 0.9, 3.4 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        hspace=0.28,
        wspace=0.18,
    )
    proj = None
    if HAS_CARTOPY:
        if projection == "robinson":
            proj = ccrs.Robinson(central_longitude=0.0)
        else:
            proj = ccrs.PlateCarree()
    axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            if proj is None:
                row_axes.append(fig.add_subplot(gs[r, c]))
            else:
                row_axes.append(fig.add_subplot(gs[r, c], projection=proj))
        axes.append(row_axes)
    caxes = [fig.add_subplot(gs[r, n_cols]) for r in range(n_rows)]

    lon_base = np.asarray(lon2d_deg, dtype=np.float64)
    lat_base = np.asarray(lat2d_deg, dtype=np.float64)
    if HAS_CARTOPY and projection == "platecarree":
        lon_base = ((lon_base + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_base[0, :])
    lon_base = lon_base[:, order]
    lat_base = lat_base[:, order]

    lon_plot = np.concatenate([lon_base, lon_base[:, :1] + 360.0], axis=1)
    lat_plot = np.concatenate([lat_base, lat_base[:, :1]], axis=1)

    for r, (key, label, cmap) in enumerate(field_specs):
        panels = [np.asarray(snapshots[s][key], dtype=np.float64) for s in steps]
        vmin, vmax = _color_limits(panels, symmetric=_is_symmetric_snapshot_field(key))
        im = None
        for c, st in enumerate(steps):
            ax = axes[r][c]
            fld = panels[c][:, order]
            if proj is None:
                im = ax.pcolormesh(
                    lon_plot,
                    lat_plot,
                    np.concatenate([fld, fld[:, :1]], axis=1),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                    rasterized=True,
                )
                ax.set_xlim(0.0, 360.0)
                ax.set_ylim(-90.0, 90.0)
                ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
                ax.set_yticks([-60, -30, 0, 30, 60])
                ax.grid(True, alpha=0.15)
            else:
                fld_plot = np.concatenate([fld, fld[:, :1]], axis=1)
                im = ax.pcolormesh(
                    lon_plot,
                    lat_plot,
                    fld_plot,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    rasterized=True,
                )
                if projection == "platecarree":
                    ax.set_extent([-180.0, 180.0, -90.0, 90.0], ccrs.PlateCarree())
                else:
                    ax.set_global()
                if draw_coastlines:
                    ax.coastlines(linewidth=0.35, color="0.35")
                ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
            ax.set_title(f"step {st}\nt={_format_sim_time(st, dt)}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
        fig.colorbar(im, cax=caxes[r], orientation="vertical")

    fig.suptitle(f"{case_name} - Field Snapshots", fontsize=13)
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_dir / "field_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    _save_snapshot_times(out_dir, steps, dt)


def _save_snapshots_cube(
    out_dir: Path,
    case_name: str,
    snapshots: dict[int, dict[str, np.ndarray]],
    dt: float,
    field_specs: list[tuple[str, str, str]],
    cube_lon_deg: np.ndarray,
    cube_lat_deg: np.ndarray,
    filename: str = "field_snapshots_cube_native.png",
    *,
    projection: str = "platecarree",
    draw_coastlines: bool = False,
) -> None:
    if not snapshots:
        return
    steps = sorted(snapshots.keys())
    n_rows = len(field_specs)
    n_cols = len(steps)

    fig = plt.figure(figsize=(5.0 * n_cols + 0.9, 3.4 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        hspace=0.28,
        wspace=0.18,
    )
    proj = None
    if HAS_CARTOPY:
        if projection == "robinson":
            proj = ccrs.Robinson(central_longitude=0.0)
        else:
            proj = ccrs.PlateCarree()
    axes = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            if proj is None:
                row_axes.append(fig.add_subplot(gs[r, c]))
            else:
                row_axes.append(fig.add_subplot(gs[r, c], projection=proj))
        axes.append(row_axes)
    caxes = [fig.add_subplot(gs[r, n_cols]) for r in range(n_rows)]

    lon_faces = np.asarray(cube_lon_deg, dtype=np.float64)
    lat_faces = np.asarray(cube_lat_deg, dtype=np.float64)
    use_face_tiles = lon_faces.ndim == 3 and lat_faces.ndim == 3 and lon_faces.shape == lat_faces.shape
    if use_face_tiles and projection == "platecarree":
        lon_faces = ((lon_faces + 180.0) % 360.0) - 180.0
    lon_pts = lon_faces.reshape(-1)
    lat_pts = lat_faces.reshape(-1)

    for r, (key, label, cmap) in enumerate(field_specs):
        panels = [np.asarray(snapshots[s][key], dtype=np.float64) for s in steps]
        vmin, vmax = _color_limits(
            [p.reshape(-1) for p in panels],
            symmetric=_is_symmetric_snapshot_field(key),
        )
        im = None
        for c, st in enumerate(steps):
            ax = axes[r][c]
            fld_panel = panels[c]
            if proj is None:
                if use_face_tiles and fld_panel.ndim == 3 and fld_panel.shape == lon_faces.shape:
                    for fidx in range(fld_panel.shape[0]):
                        lon_f = (lon_faces[fidx] + 360.0) % 360.0
                        lat_f = lat_faces[fidx]
                        fld_f = fld_panel[fidx]
                        mask_f = ~(np.isfinite(lon_f) & np.isfinite(lat_f) & np.isfinite(fld_f))
                        fld_m = np.ma.array(fld_f, mask=mask_f)
                        im = ax.pcolormesh(
                            lon_f,
                            lat_f,
                            fld_m,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            shading="nearest",
                            rasterized=True,
                        )
                else:
                    fld = fld_panel.reshape(-1)
                    valid = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(fld)
                    im = ax.scatter(
                        ((lon_pts[valid] + 360.0) % 360.0),
                        lat_pts[valid],
                        c=fld[valid],
                        s=max(0.8, 2200.0 / max(1, fld.size)),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        linewidths=0.0,
                        rasterized=True,
                    )
                ax.set_xlim(0.0, 360.0)
                ax.set_ylim(-90.0, 90.0)
                ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
                ax.set_yticks([-60, -30, 0, 30, 60])
                ax.grid(True, alpha=0.15)
            else:
                if use_face_tiles and fld_panel.ndim == 3 and fld_panel.shape == lon_faces.shape:
                    for fidx in range(fld_panel.shape[0]):
                        lon_f = lon_faces[fidx]
                        lat_f = lat_faces[fidx]
                        fld_f = fld_panel[fidx]
                        mask_f = ~(np.isfinite(lon_f) & np.isfinite(lat_f) & np.isfinite(fld_f))
                        fld_m = np.ma.array(fld_f, mask=mask_f)
                        im = ax.pcolormesh(
                            lon_f,
                            lat_f,
                            fld_m,
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            transform=ccrs.PlateCarree(),
                            shading="nearest",
                            rasterized=True,
                        )
                else:
                    fld = fld_panel.reshape(-1)
                    valid = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(fld)
                    lon_plot = lon_pts[valid]
                    if projection == "platecarree":
                        lon_plot = ((lon_plot + 180.0) % 360.0) - 180.0
                    im = ax.scatter(
                        lon_plot,
                        lat_pts[valid],
                        c=fld[valid],
                        s=max(0.8, 2200.0 / max(1, fld.size)),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        linewidths=0.0,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )
                if projection == "platecarree":
                    ax.set_extent([-180.0, 180.0, -90.0, 90.0], ccrs.PlateCarree())
                else:
                    ax.set_global()
                if draw_coastlines:
                    ax.coastlines(linewidth=0.35, color="0.35")
                ax.gridlines(draw_labels=False, linewidth=0.2, color="0.6", alpha=0.35)
            ax.set_title(f"step {st}\nt={_format_sim_time(st, dt)}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
        fig.colorbar(im, cax=caxes[r], orientation="vertical")

    fig.suptitle(f"{case_name} - Field Snapshots", fontsize=13)
    fig.subplots_adjust(top=0.92)
    fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_spectral_latlon(
    out_dir: Path,
    trunc: int,
    days: float,
    dt: float,
    mean_every: int,
    case: str,
    projection: str,
    draw_coastlines: bool,
) -> dict:
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.spectral_sw import (
        SpectralSWConfig,
        spectral_sw_tendencies,
        spectral_to_grid,
        compute_spectral_diagnostics,
        williamson_test2_spectral,
        williamson_test5_spectral,
    )
    from legoesm.timestepping.ssp_rk54 import ssp_rk54_step

    out_dir.mkdir(parents=True, exist_ok=True)
    if case == "williamson2":
        init_fn = williamson_test2_spectral
        case_tag = "Williamson2"
    elif case == "williamson5":
        init_fn = williamson_test5_spectral
        case_tag = "Williamson5"
    else:
        raise ValueError(f"Unsupported case '{case}'")
    case_name = f"SW Spectral LatLon {case_tag} T{trunc} SSP45"

    grid = create_gaussian_grid(trunc)
    area = _gaussian_area_weights(grid)
    lon1d = np.degrees(np.asarray(grid.lon))
    lat1d = np.degrees(np.asarray(grid.lat))
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)

    state = init_fn(grid)
    n_steps = int(days * 86400.0 / dt)
    snap_targets = _snapshot_steps(n_steps)
    snapshots: dict[int, dict[str, np.ndarray]] = {}

    a = grid.radius
    eig_max = trunc * (trunc + 1) / (a * a)
    cfg = SpectralSWConfig(hyperdiff_coeff=1.0 / (1.0 * 3600.0 * eig_max ** 2), hyperdiff_order=2)

    def _tendency(s):
        return spectral_sw_tendencies(s, grid, cfg)

    stepper = jax.jit(lambda s: ssp_rk54_step(s, _tendency, dt))

    mean_series = _series_init(["mean_wind_speed", "mean_height", "mean_vorticity"])
    cons_series = _series_init(["mass", "energy", "enstrophy"])
    slab_values: list[dict[str, float]] = []
    lat_weights = np.asarray(grid.weights, dtype=np.float64)

    def _extract(s):
        f = spectral_to_grid(s, grid)
        u = np.asarray(f["u"], dtype=np.float64)
        v = np.asarray(f["v"], dtype=np.float64)
        h = np.asarray(f["h"], dtype=np.float64)
        vor = np.asarray(f["vor"], dtype=np.float64)
        return {
            "u": u,
            "v": v,
            "wind_speed": np.sqrt(u * u + v * v),
            "height": h,
            "vorticity": vor,
        }

    def _record(step_idx: int, s):
        fx = _extract(s)
        _series_push(
            mean_series,
            step_idx,
            {
                "mean_wind_speed": _weighted_mean(fx["wind_speed"], area),
                "mean_height": _weighted_mean(fx["height"], area),
                "mean_vorticity": _weighted_mean(fx["vorticity"], area),
            },
        )
        d = compute_spectral_diagnostics(s, grid)
        _series_push(
            cons_series,
            step_idx,
            {
                "mass": float(d["mass"]),
                "energy": float(d["energy"]),
                "enstrophy": float(d["enstrophy"]),
            },
        )
        slab_values.append(_compute_slab_values(step_idx, dt, fx, lat_weights))

    snapshots[0] = _extract(state)
    _record(0, state)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = stepper(state)
        st = i + 1
        if st in snap_targets:
            snapshots[st] = _extract(state)
        if st % mean_every == 0 or st == n_steps:
            _record(st, state)
        if st % progress_every == 0:
            print(f"      Spectral progress: {st}/{n_steps}")
        if not bool(jnp.all(jnp.isfinite(state.vor_hat.data))):
            stable = False
            print(f"      Spectral non-finite at step {st}")
            break
    jax.block_until_ready(state.vor_hat.data)
    wall = time.time() - t0

    _save_snapshots_latlon(
        out_dir,
        case_name,
        snapshots,
        dt,
        [
            ("u", "Zonal wind u (m/s)", "RdBu_r"),
            ("v", "Meridional wind v (m/s)", "RdBu_r"),
            ("wind_speed", "Wind speed (m/s)", "magma"),
            ("height", "Fluid depth h (m)", "viridis"),
            ("vorticity", "Relative vorticity (1/s)", "RdBu_r"),
        ],
        lon2d,
        lat2d,
        projection=projection,
        draw_coastlines=draw_coastlines,
    )
    _save_mean_timeseries(
        out_dir,
        case_name,
        mean_series,
        dt,
        {"mean_wind_speed": "m/s", "mean_height": "m", "mean_vorticity": "1/s"},
    )
    _save_conservation_timeseries(out_dir, case_name, cons_series, dt)
    _save_slab_timeseries(out_dir, case_name, slab_values)

    with open(out_dir / "results.txt", "w") as f:
        f.write(f"case: spectral_latlon_{case}\n")
        f.write(f"truncation: T{trunc}\n")
        f.write(f"duration_days: {days}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"stable: {stable}\n")
        f.write(f"wall_time_s: {wall:.2f}\n")
        f.write(
            "mass_drift_rel: "
            f"{(cons_series['mass'][-1] - cons_series['mass'][0]) / max(abs(cons_series['mass'][0]), 1.0e-30):.8e}\n",
        )
        f.write(
            "energy_drift_rel: "
            f"{(cons_series['energy'][-1] - cons_series['energy'][0]) / max(abs(cons_series['energy'][0]), 1.0e-30):.8e}\n",
        )
        f.write(
            "enstrophy_drift_rel: "
            f"{(cons_series['enstrophy'][-1] - cons_series['enstrophy'][0]) / max(abs(cons_series['enstrophy'][0]), 1.0e-30):.8e}\n",
        )
        f.write(f"final_h_mean: {slab_values[-1]['h_mean']:.8e}\n")
        f.write(f"final_speed_mean: {slab_values[-1]['speed_mean']:.8e}\n")
        f.write(f"final_vort_rms: {slab_values[-1]['vor_rms']:.8e}\n")

    return {
        "case": f"spectral_latlon_{case}",
        "stable": stable,
        "wall_time_s": wall,
        "n_steps": n_steps,
        "dt": dt,
    }


def _run_fv_cubesphere(
    out_dir_latlon: Path,
    out_dir_cube: Path,
    n: int,
    days: float,
    dt: float,
    mean_every: int,
    case: str,
    projection: str,
    draw_coastlines: bool,
    fv_variant: str,
) -> dict:
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.core.conservation import compute_conservation_diagnostics
    from legoesm.core.operators import curl_z
    from legoesm.core.operators_fv_cubed import default_div_damp_coeffs
    from tests.test_cases.williamson import williamson_test2, williamson_test5

    out_dir_latlon.mkdir(parents=True, exist_ok=True)
    out_dir_cube.mkdir(parents=True, exist_ok=True)
    if case == "williamson2":
        init_fn = williamson_test2
        case_tag = "Williamson2"
    elif case == "williamson5":
        init_fn = williamson_test5
        case_tag = "Williamson5"
    else:
        raise ValueError(f"Unsupported case '{case}'")
    fv_variant = fv_variant.lower().strip()
    if fv_variant in ("agrid", "a-grid", "a_grid"):
        from legoesm.atmosphere.dynamics.shallow_water_fv import (
            FVShallowWaterConfig as _SWConfig,
            FVShallowWaterModel as _SWModel,
        )

        variant_tag = "fv_agrid"
        variant_label = "Finite-Volume A-grid"
    elif fv_variant in ("cdgrid", "cgrid", "c-d-grid", "c_d_grid"):
        from legoesm.atmosphere.dynamics.shallow_water_cgrid import (
            CGShallowWaterCubedConfig as _SWConfig,
            CGShallowWaterCubedModel as _SWModel,
        )

        variant_tag = "fv_cdgrid"
        variant_label = "Finite-Volume C-D-grid"
    else:
        raise ValueError(
            f"Unsupported fv_variant={fv_variant!r}; choose from 'agrid', 'cdgrid'."
        )

    case_name_latlon = f"SW {variant_label} LatLon(remapped) {case_tag} C{n} SSP45"
    case_name_cube = f"SW {variant_label} CubeSphere {case_tag} C{n} SSP45"

    grid = create_cubed_sphere(n)
    area = np.asarray(grid.area, dtype=np.float64)
    lon_faces = np.asarray(grid.lon, dtype=np.float64) * 180.0 / np.pi
    lat_faces = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi

    state = init_fn(grid)
    n_steps = int(days * 86400.0 / dt)
    snap_targets = _snapshot_steps(n_steps)
    snapshots_latlon: dict[int, dict[str, np.ndarray]] = {}
    snapshots_cube: dict[int, dict[str, np.ndarray]] = {}

    nu2, nu4 = default_div_damp_coeffs(grid, dt=dt)
    if variant_tag == "fv_agrid":
        cfg = _SWConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            div_damp_2=nu2,
            div_damp_4=nu4,
            time_integrator="ssp45",
            use_limiter=True,
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
        )
    else:
        cfg = _SWConfig(
            hyperdiff_coeff=5.0e16 * (48.0 / n) ** 4,
            div_damp_2=nu2,
            div_damp_4=nu4,
            time_integrator="ssp45",
            use_conservation_fixer=True,
            fix_mass=True,
            fix_energy=False,
        )
    model = _SWModel(grid, cfg)

    mean_series = _series_init(["mean_wind_speed", "mean_height", "mean_vorticity"])
    cons_series = _series_init(["mass", "energy", "enstrophy"])
    slab_values: list[dict[str, float]] = []

    def _extract_faces(s):
        u = np.asarray(s.u.data, dtype=np.float64)
        v = np.asarray(s.v.data, dtype=np.float64)
        h = np.asarray(s.h.data, dtype=np.float64)
        wind = np.sqrt(u * u + v * v)
        vor = np.asarray(curl_z(s.u, s.v, grid).data, dtype=np.float64)
        return {"u": u, "v": v, "wind_speed": wind, "height": h, "vorticity": vor}

    def _extract_latlon(face_fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for key, fld in face_fields.items():
            lon2d, lat2d, fld_ll = _regrid_faces_to_latlon(fld, lon_faces, lat_faces)
            lon2d = (lon2d + 360.0) % 360.0
            order = np.argsort(lon2d[0, :])
            out[key] = fld_ll[:, order]
            out["_lon2d"] = lon2d[:, order]
            out["_lat2d"] = lat2d[:, order]
        return out

    def _record(
        step_idx: int,
        s,
        face_fields: dict[str, np.ndarray],
        latlon_fields: dict[str, np.ndarray],
        lat_weights: np.ndarray,
    ):
        _series_push(
            mean_series,
            step_idx,
            {
                "mean_wind_speed": _weighted_mean(face_fields["wind_speed"], area),
                "mean_height": _weighted_mean(face_fields["height"], area),
                "mean_vorticity": _weighted_mean(face_fields["vorticity"], area),
            },
        )
        d = compute_conservation_diagnostics(s, grid)
        h_arr = np.asarray(s.h.data, dtype=np.float64)
        zeta_arr = np.asarray(face_fields["vorticity"], dtype=np.float64)
        abs_vor = zeta_arr + np.asarray(grid.f, dtype=np.float64)
        q = abs_vor / np.maximum(h_arr, 1.0e-12)
        enstrophy = 0.5 * np.sum(q * q * h_arr * area)
        _series_push(
            cons_series,
            step_idx,
            {
                "mass": float(d["total_mass"]),
                "energy": float(d["total_energy"]),
                "enstrophy": float(enstrophy),
            },
        )
        slab_values.append(_compute_slab_values(step_idx, dt, latlon_fields, lat_weights))

    f0 = _extract_faces(state)
    ll0 = _extract_latlon(f0)
    snapshots_cube[0] = {k: v for k, v in f0.items()}
    snapshots_latlon[0] = {k: ll0[k] for k in ("u", "v", "wind_speed", "height", "vorticity")}
    lon2d_ref = ll0["_lon2d"]
    lat2d_ref = ll0["_lat2d"]
    lat_weights = _lat_weights_from_lat2d(lat2d_ref)
    _record(0, state, f0, snapshots_latlon[0], lat_weights)

    stable = True
    t0 = time.time()
    progress_every = max(1, n_steps // 10)
    for i in range(n_steps):
        state = model.step(state, dt)
        st = i + 1
        need_snap = st in snap_targets
        need_record = st % mean_every == 0 or st == n_steps
        if need_snap or need_record:
            f = _extract_faces(state)
            ll = _extract_latlon(f)
            ll_fields = {k: ll[k] for k in ("u", "v", "wind_speed", "height", "vorticity")}
            if need_snap:
                snapshots_cube[st] = {k: v for k, v in f.items()}
                snapshots_latlon[st] = ll_fields
            if need_record:
                _record(st, state, f, ll_fields, lat_weights)
        if st % progress_every == 0:
            print(f"      FV progress: {st}/{n_steps}")
        if not bool(jnp.all(jnp.isfinite(state.h.data))):
            stable = False
            print(f"      FV non-finite at step {st}")
            break
    jax.block_until_ready(state.h.data)
    wall = time.time() - t0

    field_specs = [
        ("u", "Zonal wind u (m/s)", "RdBu_r"),
        ("v", "Meridional wind v (m/s)", "RdBu_r"),
        ("wind_speed", "Wind speed (m/s)", "magma"),
        ("height", "Fluid depth h (m)", "viridis"),
        ("vorticity", "Relative vorticity (1/s)", "RdBu_r"),
    ]

    _save_snapshots_latlon(
        out_dir_latlon,
        case_name_latlon,
        snapshots_latlon,
        dt,
        field_specs,
        lon2d_ref,
        lat2d_ref,
        projection=projection,
        draw_coastlines=draw_coastlines,
    )
    _save_snapshots_latlon(
        out_dir_cube,
        case_name_cube,
        snapshots_latlon,
        dt,
        field_specs,
        lon2d_ref,
        lat2d_ref,
        projection=projection,
        draw_coastlines=draw_coastlines,
    )
    _save_snapshots_cube(
        out_dir_cube,
        f"{case_name_cube} (native cube scatter)",
        snapshots_cube,
        dt,
        field_specs,
        lon_faces,
        lat_faces,
        filename="field_snapshots_cube_native.png",
        projection=projection,
        draw_coastlines=draw_coastlines,
    )
    _save_snapshot_times(out_dir_cube, sorted(snapshots_latlon.keys()), dt)
    for tgt_dir, tgt_name in ((out_dir_latlon, case_name_latlon), (out_dir_cube, case_name_cube)):
        _save_mean_timeseries(
            tgt_dir,
            tgt_name,
            mean_series,
            dt,
            {"mean_wind_speed": "m/s", "mean_height": "m", "mean_vorticity": "1/s"},
        )
        _save_conservation_timeseries(tgt_dir, tgt_name, cons_series, dt)
        _save_slab_timeseries(tgt_dir, tgt_name, slab_values)
        with open(tgt_dir / "results.txt", "w") as f:
            f.write(f"case: {variant_tag}_{case}\n")
            f.write(f"fv_variant: {variant_tag}\n")
            f.write(f"resolution: C{n}\n")
            f.write(f"duration_days: {days}\n")
            f.write(f"dt: {dt}\n")
            f.write(f"n_steps: {n_steps}\n")
            f.write(f"stable: {stable}\n")
            f.write(f"wall_time_s: {wall:.2f}\n")
            f.write(
                "mass_drift_rel: "
                f"{(cons_series['mass'][-1] - cons_series['mass'][0]) / max(abs(cons_series['mass'][0]), 1.0e-30):.8e}\n",
            )
            f.write(
                "energy_drift_rel: "
                f"{(cons_series['energy'][-1] - cons_series['energy'][0]) / max(abs(cons_series['energy'][0]), 1.0e-30):.8e}\n",
            )
            f.write(
                "enstrophy_drift_rel: "
                f"{(cons_series['enstrophy'][-1] - cons_series['enstrophy'][0]) / max(abs(cons_series['enstrophy'][0]), 1.0e-30):.8e}\n",
            )
            f.write(f"final_h_mean: {slab_values[-1]['h_mean']:.8e}\n")
            f.write(f"final_speed_mean: {slab_values[-1]['speed_mean']:.8e}\n")
            f.write(f"final_vort_rms: {slab_values[-1]['vor_rms']:.8e}\n")

    return {
        "case": f"{variant_tag}_{case}",
        "stable": stable,
        "wall_time_s": wall,
        "n_steps": n_steps,
        "dt": dt,
    }


def main():
    parser = argparse.ArgumentParser(description="Run SW spectral/FV lat-lon and cubed-sphere diagnostics.")
    parser.add_argument("--output", type=Path, default=Path("results/atmosphere/shallow_water/sw_latlon_cube_compare"))
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=("all", "williamson2", "williamson5"),
        help="Williamson test case to run.",
    )
    parser.add_argument("--days", type=float, default=5.0)
    parser.add_argument("--cube-resolution", type=int, default=36)
    parser.add_argument("--spectral-truncation", type=int, default=42)
    parser.add_argument("--dt-fv", type=float, default=450.0)
    parser.add_argument("--dt-spec", type=float, default=120.0)
    parser.add_argument("--mean-every-fv", type=int, default=24, help="Record FV means/diagnostics every N steps.")
    parser.add_argument("--mean-every-spec", type=int, default=60, help="Record spectral means/diagnostics every N steps.")
    parser.add_argument(
        "--fv-variant",
        type=str,
        default="both",
        choices=("agrid", "cdgrid", "both"),
        help="Finite-volume cubed-sphere branch to run.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="platecarree",
        choices=("platecarree", "robinson"),
        help="Map projection for snapshot figures.",
    )
    parser.add_argument(
        "--coastlines",
        action="store_true",
        help="Draw coastlines on projected map snapshots.",
    )
    args = parser.parse_args()

    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    print("Running shallow-water comparison suite:")
    print(f"  output={out_root}")
    print(f"  case={args.case}")
    print(f"  days={args.days}")
    print(f"  cube_resolution=C{args.cube_resolution}, dt_fv={args.dt_fv}s")
    print(f"  spectral_truncation=T{args.spectral_truncation}, dt_spec={args.dt_spec}s")

    t0 = time.time()
    summary = {}

    case_list = ["williamson2", "williamson5"] if args.case == "all" else [args.case]
    dir_map = {
        "williamson2": (
            "01_latlon_spectral_williamson2",
            "02_latlon_fv_williamson2",
            "03_cubesphere_fv_williamson2",
        ),
        "williamson5": (
            "04_latlon_spectral_williamson5",
            "05_latlon_fv_williamson5",
            "06_cubesphere_fv_williamson5",
        ),
    }
    dir_map_cd = {
        "williamson2": (
            "07_latlon_fv_cdgrid_williamson2",
            "08_cubesphere_fv_cdgrid_williamson2",
        ),
        "williamson5": (
            "09_latlon_fv_cdgrid_williamson5",
            "10_cubesphere_fv_cdgrid_williamson5",
        ),
    }

    for idx, case in enumerate(case_list, start=1):
        print(f"\n[{idx}/{len(case_list)}] Running {case} ...")
        spec_dir, fv_ll_dir, fv_cube_dir = dir_map[case]
        summary[f"spectral_latlon_{case}"] = _run_spectral_latlon(
            out_root / spec_dir,
            trunc=args.spectral_truncation,
            days=args.days,
            dt=args.dt_spec,
            mean_every=args.mean_every_spec,
            case=case,
            projection=args.projection,
            draw_coastlines=args.coastlines,
        )

        fv_variants = ["agrid", "cdgrid"] if args.fv_variant == "both" else [args.fv_variant]
        for fv_variant in fv_variants:
            if fv_variant == "agrid":
                ll_dir = fv_ll_dir
                cube_dir = fv_cube_dir
            else:
                ll_dir, cube_dir = dir_map_cd[case]
            summary[f"finite_volume_{fv_variant}_{case}"] = _run_fv_cubesphere(
                out_root / ll_dir,
                out_root / cube_dir,
                n=args.cube_resolution,
                days=args.days,
                dt=args.dt_fv,
                mean_every=args.mean_every_fv,
                case=case,
                projection=args.projection,
                draw_coastlines=args.coastlines,
                fv_variant=fv_variant,
            )

    wall = time.time() - t0
    summary["wall_time_s_total"] = wall
    summary["args"] = {
        "days": args.days,
        "case": args.case,
        "cube_resolution": args.cube_resolution,
        "spectral_truncation": args.spectral_truncation,
        "dt_fv": args.dt_fv,
        "dt_spec": args.dt_spec,
        "mean_every_fv": args.mean_every_fv,
        "mean_every_spec": args.mean_every_spec,
        "fv_variant": args.fv_variant,
        "projection": args.projection,
        "coastlines": bool(args.coastlines),
    }

    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"  Total wall time: {wall:.2f}s")
    print(f"  Results written to: {out_root}")


if __name__ == "__main__":
    main()
