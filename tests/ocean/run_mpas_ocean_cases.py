#!/usr/bin/env python3
"""Run MPAS (icosahedral/Voronoi) ocean test cases with diagnostics.

Outputs are written under ``results/ocean`` by default and include:
- per-case scalar diagnostics and conservation drifts
- horizontal slab averages (depth profiles) at multiple times
- vertical slab averages (upper/middle/deep) over time
- surface snapshots remapped to lat-lon pixels
- per-run and global summary JSON files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ensure repo root import when invoked as a file script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Avoid matplotlib cache permission issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legoesm.grids.voronoi import VoronoiMesh, create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.init_mpas import reconstruct_cell_velocity, rest_state_mpas_ocean
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness, create_ocean_z_star


@dataclass(frozen=True)
class CaseSpec:
    tag: str
    label: str
    init_kind: str
    forcing_kind: str
    use_fixers: bool
    fix_volume: bool
    fix_heat: bool
    fix_salt: bool
    default_hours: float


CASE_SPECS: dict[str, CaseSpec] = {
    "rest_state": CaseSpec(
        tag="rest_state",
        label="Rest State",
        init_kind="rest_state",
        forcing_kind="none",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "gravity_wave": CaseSpec(
        tag="gravity_wave",
        label="Barotropic Gravity Wave",
        init_kind="gravity_wave",
        forcing_kind="none",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "wind_gyre": CaseSpec(
        tag="wind_gyre",
        label="Wind-Driven Gyre",
        init_kind="wind_gyre",
        forcing_kind="wind_gyre",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "adiabatic_topography": CaseSpec(
        tag="adiabatic_topography",
        label="Adiabatic Topographic Adjustment",
        init_kind="adiabatic_topography",
        forcing_kind="none",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "holland_lin_gyre": CaseSpec(
        tag="holland_lin_gyre",
        label="Holland-Lin Double Gyre",
        init_kind="holland_lin_gyre",
        forcing_kind="holland_lin_gyre",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "thermohaline": CaseSpec(
        tag="thermohaline",
        label="Diabatic Thermohaline Circulation",
        init_kind="thermohaline",
        forcing_kind="thermohaline",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "phillips_two_layer": CaseSpec(
        tag="phillips_two_layer",
        label="Phillips Two-Layer Test",
        init_kind="phillips_two_layer",
        forcing_kind="phillips_two_layer",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
    "taylor_column": CaseSpec(
        tag="taylor_column",
        label="Taylor Column",
        init_kind="taylor_column",
        forcing_kind="taylor_column",
        use_fixers=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        default_hours=24.0,
    ),
}


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if any(v < 1 for v in vals):
        raise ValueError(f"Expected positive integers, got {text!r}")
    return vals


def _weighted_mean(x: jax.Array, w: jax.Array) -> float:
    return float(jnp.sum(x * w) / jnp.maximum(jnp.sum(w), 1.0e-30))


def _wrap_lon_deg(lon_deg: jax.Array) -> jax.Array:
    return (lon_deg + 180.0) % 360.0 - 180.0


def _great_circle_distance_rad(
    lon: jax.Array,
    lat: jax.Array,
    lon0: float,
    lat0: float,
) -> jax.Array:
    dlon = ((lon - lon0 + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi
    cos_d = (
        jnp.sin(lat) * jnp.sin(lat0)
        + jnp.cos(lat) * jnp.cos(lat0) * jnp.cos(dlon)
    )
    return jnp.arccos(jnp.clip(cos_d, -1.0, 1.0))


def _edge_ocean_mask(mesh: VoronoiMesh, land_mask: jax.Array) -> jax.Array:
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return land_mask[c1] * land_mask[c2]


def _cell_to_edge_mean(mesh: VoronoiMesh, field_cell: jax.Array) -> jax.Array:
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return 0.5 * (field_cell[c1] + field_cell[c2])


def _ocean_diagnostics(
    state,
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
) -> dict[str, float]:
    mask = state.land_mask.data
    area = mesh.areaCell
    h_k = compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    volume = jnp.sum(state.eta.data * mask * area)
    heat = jnp.sum(state.T.data * h_k * mask[:, None] * area[:, None])
    salt = jnp.sum(state.S.data * h_k * mask[:, None] * area[:, None])

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    h_e_k = 0.5 * (h_k[c1] + h_k[c2])
    area_edge = mesh.dcEdge * mesh.dvEdge
    ke = 0.5 * jnp.sum(
        state.u.data ** 2 * h_e_k * area_edge[:, None] * edge_mask[:, None]
    )

    u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
    speed = jnp.sqrt(u_east ** 2 + v_north ** 2)
    w_area = mask * area

    return {
        "volume": float(volume),
        "heat": float(heat),
        "salt": float(salt),
        "KE": float(ke),
        "eta_mean_m": _weighted_mean(state.eta.data, w_area),
        "eta_min_m": float(jnp.min(state.eta.data)),
        "eta_max_m": float(jnp.max(state.eta.data)),
        "T_surface_mean_degC": _weighted_mean(state.T.data[:, 0], w_area),
        "S_surface_mean_psu": _weighted_mean(state.S.data[:, 0], w_area),
        "speed_surface_mean_mps": _weighted_mean(speed[:, 0], w_area),
        "speed_surface_max_mps": float(jnp.max(speed[:, 0])),
        "all_finite": bool(
            jnp.all(jnp.isfinite(state.u.data))
            and jnp.all(jnp.isfinite(state.T.data))
            and jnp.all(jnp.isfinite(state.S.data))
            and jnp.all(jnp.isfinite(state.eta.data))
        ),
    }


def _slab_bounds(n_levels: int) -> list[tuple[str, int, int]]:
    if n_levels <= 1:
        return [("full", 0, n_levels)]

    i1 = max(1, int(np.ceil(n_levels / 3.0)))
    i2 = max(i1 + 1, int(np.ceil(2.0 * n_levels / 3.0)))
    i2 = min(i2, n_levels)
    bounds = [
        ("upper", 0, i1),
        ("middle", i1, i2),
        ("deep", i2, n_levels),
    ]
    out = []
    for name, k0, k1 in bounds:
        if k1 > k0:
            out.append((name, k0, k1))
    return out


def _horizontal_profiles(
    state,
    mesh: VoronoiMesh,
) -> dict[str, np.ndarray]:
    mask = state.land_mask.data
    area = mesh.areaCell
    w = mask * area
    w_3d = w[:, None]

    u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
    speed = jnp.sqrt(u_east ** 2 + v_north ** 2)

    denom = jnp.maximum(jnp.sum(w), 1.0e-30)
    t_prof = jnp.sum(state.T.data * w_3d, axis=0) / denom
    s_prof = jnp.sum(state.S.data * w_3d, axis=0) / denom
    speed_prof = jnp.sum(speed * w_3d, axis=0) / denom

    return {
        "T_mean_degC": np.asarray(t_prof, dtype=np.float64),
        "S_mean_psu": np.asarray(s_prof, dtype=np.float64),
        "speed_mean_mps": np.asarray(speed_prof, dtype=np.float64),
    }


def _vertical_slab_means(
    state,
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
) -> dict[str, float]:
    mask = state.land_mask.data
    area = mesh.areaCell
    h_k = compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
    speed = jnp.sqrt(u_east ** 2 + v_north ** 2)

    w_vol = h_k * (mask * area)[:, None]
    out: dict[str, float] = {}
    for slab_name, k0, k1 in _slab_bounds(state.T.data.shape[1]):
        w = w_vol[:, k0:k1]
        w_sum = jnp.maximum(jnp.sum(w), 1.0e-30)
        out[f"T_{slab_name}_mean_degC"] = float(
            jnp.sum(state.T.data[:, k0:k1] * w) / w_sum
        )
        out[f"S_{slab_name}_mean_psu"] = float(
            jnp.sum(state.S.data[:, k0:k1] * w) / w_sum
        )
        out[f"speed_{slab_name}_mean_mps"] = float(
            jnp.sum(speed[:, k0:k1] * w) / w_sum
        )
    return out


def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    e = np.empty(c.size + 1, dtype=np.float64)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - (e[1] - c[0])
    e[-1] = c[-1] + (c[-1] - e[-2])
    return e


def _default_lat_bins_mpas(
    mesh: VoronoiMesh,
    mask_1d: np.ndarray,
    *,
    min_bins: int = 24,
    max_bins: int = 96,
) -> np.ndarray:
    lat_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180.0 / np.pi
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    mask = np.asarray(mask_1d, dtype=np.float64)
    wet = (mask > 0.5) & np.isfinite(area) & (area > 0.0) & np.isfinite(lat_deg)
    wet_lat = lat_deg[wet]
    if wet_lat.size < 2:
        return np.linspace(-90.0, 90.0, min_bins + 1)
    lat_lo = max(-90.0, float(np.nanmin(wet_lat)) - 1.0)
    lat_hi = min(90.0, float(np.nanmax(wet_lat)) + 1.0)
    n_wet = int(np.sum(wet))
    n_bins = int(np.clip(round(np.sqrt(max(n_wet, 4))), min_bins, max_bins))
    return np.linspace(lat_lo, lat_hi, n_bins + 1)


def _default_lon_bins_mpas(
    mesh: VoronoiMesh,
    mask_1d: np.ndarray,
    *,
    min_bins: int = 36,
    max_bins: int = 180,
) -> np.ndarray:
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    mask = np.asarray(mask_1d, dtype=np.float64)
    wet = (mask > 0.5) & np.isfinite(area) & (area > 0.0)
    n_wet = int(np.sum(wet))
    if n_wet < 4:
        return np.linspace(0.0, 360.0, min_bins + 1)
    n_bins = int(np.clip(round(1.7 * np.sqrt(n_wet)), min_bins, max_bins))
    return np.linspace(0.0, 360.0, n_bins + 1)


def _fill_missing_bands(section: np.ndarray, *, periodic: bool) -> np.ndarray:
    out = np.asarray(section, dtype=np.float64).copy()
    if out.ndim != 2:
        return out
    n_bins = out.shape[0]
    idx = np.arange(n_bins, dtype=np.float64)
    for k in range(out.shape[1]):
        col = out[:, k]
        finite = np.isfinite(col)
        if np.sum(finite) == 0:
            continue
        if np.sum(finite) == 1:
            col[~finite] = col[finite][0]
        else:
            if periodic:
                x = idx[finite]
                y = col[finite]
                x_ext = np.concatenate([x - float(n_bins), x, x + float(n_bins)])
                y_ext = np.concatenate([y, y, y])
                col[~finite] = np.interp(idx[~finite], x_ext, y_ext)
            else:
                col[~finite] = np.interp(idx[~finite], idx[finite], col[finite])
        out[:, k] = col

    finite_level = np.any(np.isfinite(out), axis=0)
    if np.any(finite_level):
        level_idx = np.arange(out.shape[1], dtype=np.int64)
        for k in level_idx[~finite_level]:
            nearest = level_idx[finite_level][
                np.argmin(np.abs(level_idx[finite_level] - k))
            ]
            out[:, k] = out[:, nearest]

    finite_all = np.isfinite(out)
    if np.any(finite_all):
        fill = float(np.nanmean(out[finite_all]))
        out[~finite_all] = fill
    else:
        out[...] = 0.0
    return out


def _compute_binned_depth_section_mpas(
    field_2d: np.ndarray,
    coord_deg_1d: np.ndarray,
    area_1d: np.ndarray,
    mask_1d: np.ndarray,
    bins_deg: np.ndarray,
    *,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field_2d, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"Expected field_2d (nCells, nLevels), got {field.shape}")

    coord = np.asarray(coord_deg_1d, dtype=np.float64).ravel()
    area = np.asarray(area_1d, dtype=np.float64).ravel()
    mask = np.asarray(mask_1d, dtype=np.float64).ravel()
    n_lev = field.shape[1]
    n_bins = int(len(bins_deg) - 1)
    section = np.full((n_bins, n_lev), np.nan, dtype=np.float64)

    if periodic:
        coord = np.mod(coord, 360.0)
    idx = np.digitize(coord, bins_deg, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    w = area * mask
    valid_geom = np.isfinite(coord) & np.isfinite(w) & (w > 0.0)

    for k in range(n_lev):
        v = field[:, k]
        valid = valid_geom & np.isfinite(v)
        if not np.any(valid):
            continue
        bins_k = idx[valid]
        w_k = w[valid]
        v_k = v[valid]
        sum_w = np.bincount(bins_k, weights=w_k, minlength=n_bins)
        sum_v = np.bincount(bins_k, weights=w_k * v_k, minlength=n_bins)
        wet = sum_w > 0.0
        section[wet, k] = sum_v[wet] / sum_w[wet]

    section = _fill_missing_bands(section, periodic=periodic)
    centers = 0.5 * (np.asarray(bins_deg[:-1]) + np.asarray(bins_deg[1:]))
    return centers, section


def _plot_depth_section_snapshots(
    output_path: Path,
    z_coord: OceanZStarCoordinate,
    coord_centers: np.ndarray,
    coord_label: str,
    section_snaps: dict[str, np.ndarray],
    field_label: str,
    cmap: str,
    *,
    symmetric: bool = False,
) -> None:
    labels = [k for k in ("initial", "q1", "mid", "q3", "final") if k in section_snaps]
    if not labels:
        return

    depth = -np.asarray(z_coord.z_full_ref, dtype=np.float64)
    depth_edges = _edges_from_centers(depth)
    coord_edges = _edges_from_centers(coord_centers)

    finite_values = []
    for lbl in labels:
        arr = np.asarray(section_snaps[lbl], dtype=np.float64)
        fin = arr[np.isfinite(arr)]
        if fin.size:
            finite_values.append(fin)
    if finite_values:
        data_all = np.concatenate(finite_values)
        if symmetric:
            vmax = max(float(np.nanmax(np.abs(data_all))), 1.0e-12)
            vmin = -vmax
        else:
            vmin = float(np.nanmin(data_all))
            vmax = float(np.nanmax(data_all))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 5.5), squeeze=False)
    axes = axes.ravel()
    for i, lbl in enumerate(labels):
        ax = axes[i]
        sec = np.asarray(section_snaps[lbl], dtype=np.float64).T
        mesh = ax.pcolormesh(
            coord_edges,
            depth_edges,
            np.ma.masked_invalid(sec),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_title(f"{field_label}\n{lbl}")
        ax.set_xlabel(coord_label)
        if i == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_ylim(float(np.nanmax(depth_edges)), float(np.nanmin(depth_edges)))
        ax.grid(True, alpha=0.25)
        fig.colorbar(mesh, ax=ax, shrink=0.9)

    fig.suptitle(f"{field_label} | {coord_label}-Depth Snapshots")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_latlon_pixel_mapper(
    mesh: VoronoiMesh, nlon: int, nlat: int
) -> tuple[np.ndarray, np.ndarray, callable]:
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
    xyz_cell = np.stack([x_cell, y_cell, z_cell], axis=1)

    lon_rad = np.deg2rad(lon2d.ravel())
    lat_rad = np.deg2rad(lat2d.ravel())
    x_pix = np.cos(lat_rad) * np.cos(lon_rad)
    y_pix = np.cos(lat_rad) * np.sin(lon_rad)
    z_pix = np.sin(lat_rad)
    xyz_pix = np.stack([x_pix, y_pix, z_pix], axis=1)

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


def _write_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_scalar_timeseries(rows: list[dict[str, float]], out_path: Path, title: str) -> None:
    if not rows:
        return
    t_h = np.array([r["time_hours"] for r in rows], dtype=np.float64)
    volume = np.array([r["volume"] for r in rows], dtype=np.float64)
    heat = np.array([r["heat"] for r in rows], dtype=np.float64)
    salt = np.array([r["salt"] for r in rows], dtype=np.float64)
    ke = np.array([r["KE"] for r in rows], dtype=np.float64)
    sspd = np.array([r["speed_surface_mean_mps"] for r in rows], dtype=np.float64)
    eta_mean = np.array([r["eta_mean_m"] for r in rows], dtype=np.float64)

    def rel(x):
        return (x - x[0]) / max(abs(x[0]), 1.0e-30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(t_h, rel(volume), label="volume")
    axes[0, 0].plot(t_h, rel(heat), label="heat")
    axes[0, 0].plot(t_h, rel(salt), label="salt")
    axes[0, 0].set_ylabel("Relative drift")
    axes[0, 0].set_xlabel("Time (hours)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t_h, ke, label="KE", color="tab:purple")
    axes[0, 1].set_ylabel("Kinetic energy")
    axes[0, 1].set_xlabel("Time (hours)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_h, eta_mean, label="eta mean", color="tab:blue")
    axes[1, 0].set_ylabel("Area-mean eta (m)")
    axes[1, 0].set_xlabel("Time (hours)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t_h, sspd, label="surface speed mean", color="tab:orange")
    axes[1, 1].set_ylabel("Surface speed mean (m/s)")
    axes[1, 1].set_xlabel("Time (hours)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_vertical_slab_timeseries(
    rows: list[dict[str, float]],
    out_path: Path,
    title: str,
) -> None:
    if not rows:
        return
    t_h = np.array([r["time_hours"] for r in rows], dtype=np.float64)
    first = rows[0]
    t_keys = sorted([k for k in first if k.startswith("T_")])
    s_keys = sorted([k for k in first if k.startswith("S_")])
    u_keys = sorted([k for k in first if k.startswith("speed_")])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for k in t_keys:
        axes[0].plot(t_h, [r[k] for r in rows], label=k.replace("_mean_degC", ""))
    for k in s_keys:
        axes[1].plot(t_h, [r[k] for r in rows], label=k.replace("_mean_psu", ""))
    for k in u_keys:
        axes[2].plot(t_h, [r[k] for r in rows], label=k.replace("_mean_mps", ""))

    axes[0].set_ylabel("T slab mean (degC)")
    axes[1].set_ylabel("S slab mean (PSU)")
    axes[2].set_ylabel("Speed slab mean (m/s)")
    axes[2].set_xlabel("Time (hours)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_horizontal_profiles(
    profile_snaps: dict[str, dict[str, np.ndarray]],
    depth_m: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    labels = [k for k in ("initial", "q1", "mid", "q3", "final") if k in profile_snaps]
    if not labels:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for label in labels:
        p = profile_snaps[label]
        axes[0].plot(p["T_mean_degC"], depth_m, label=label)
        axes[1].plot(p["S_mean_psu"], depth_m, label=label)
        axes[2].plot(p["speed_mean_mps"], depth_m, label=label)

    axes[0].set_xlabel("T horizontal mean (degC)")
    axes[1].set_xlabel("S horizontal mean (PSU)")
    axes[2].set_xlabel("Speed horizontal mean (m/s)")
    axes[0].set_ylabel("Depth (m)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_surface_snapshots(
    surface_snaps: dict[str, dict[str, np.ndarray]],
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    labels = [k for k in ("initial", "q1", "mid", "q3", "final") if k in surface_snaps]
    if not labels:
        return

    fields = [
        ("eta", "Sea-surface height eta (m)", "RdBu_r"),
        ("T_surface", "Surface temperature (degC)", "turbo"),
        ("S_surface", "Surface salinity (PSU)", "viridis"),
        ("speed_surface", "Surface speed (m/s)", "magma"),
    ]

    nrows = len(fields)
    ncols = len(labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for i, (key, label, cmap) in enumerate(fields):
        vals = [surface_snaps[t][key] for t in labels]
        vmin = float(np.nanmin([np.nanmin(v) for v in vals]))
        vmax = float(np.nanmax([np.nanmax(v) for v in vals]))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        for j, tlabel in enumerate(labels):
            ax = axes[i, j]
            arr = surface_snaps[tlabel][key]
            im = ax.imshow(
                arr,
                origin="lower",
                extent=(
                    float(lon2d[0, 0]),
                    float(lon2d[0, -1]),
                    float(lat2d[0, 0]),
                    float(lat2d[-1, 0]),
                ),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"{label}\n{tlabel}")
            ax.set_xlabel("Longitude (deg)")
            if j == 0:
                ax.set_ylabel("Latitude (deg)")
            ax.grid(True, alpha=0.15)
            fig.colorbar(im, ax=ax, shrink=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_surface_snapshots_native(
    surface_snaps: dict[str, dict[str, np.ndarray]],
    lon_cell_deg: np.ndarray,
    lat_cell_deg: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    labels = [k for k in ("initial", "q1", "mid", "q3", "final") if k in surface_snaps]
    if not labels:
        return

    fields = [
        ("eta", "Sea-surface height eta (m)", "RdBu_r"),
        ("T_surface", "Surface temperature (degC)", "turbo"),
        ("S_surface", "Surface salinity (PSU)", "viridis"),
        ("speed_surface", "Surface speed (m/s)", "magma"),
    ]

    nrows = len(fields)
    ncols = len(labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    marker_size = 8.0

    for i, (key, label, cmap) in enumerate(fields):
        vals = [surface_snaps[t][key] for t in labels]
        vmin = float(np.nanmin([np.nanmin(v) for v in vals]))
        vmax = float(np.nanmax([np.nanmax(v) for v in vals]))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        for j, tlabel in enumerate(labels):
            ax = axes[i, j]
            arr = surface_snaps[tlabel][key]
            sc = ax.scatter(
                lon_cell_deg,
                lat_cell_deg,
                c=arr,
                s=marker_size,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.0,
            )
            ax.set_title(f"{label}\n{tlabel}")
            ax.set_xlim(-180.0, 180.0)
            ax.set_ylim(-90.0, 90.0)
            ax.set_xlabel("Longitude (deg)")
            if j == 0:
                ax.set_ylabel("Latitude (deg)")
            ax.grid(True, alpha=0.15)
            fig.colorbar(sc, ax=ax, shrink=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _prepare_initial_state(
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    case: CaseSpec,
):
    if case.init_kind == "adiabatic_topography":
        state = rest_state_mpas_ocean(
            mesh,
            z_coord,
            T_surface=6.0,
            T_deep=6.0,
            S_uniform=35.0,
            H_max=4200.0,
            land_lat_threshold=90.0,
        )
        dist = _great_circle_distance_rad(mesh.lonCell, mesh.latCell, float(jnp.pi), 0.0)
        bump = 1500.0 * jnp.exp(-0.5 * (dist / (12.0 * jnp.pi / 180.0)) ** 2)
        H_bathy = jnp.clip(4200.0 - bump, 800.0, 4200.0) * state.land_mask.data
        eta0 = 0.30 * jnp.exp(-0.5 * (dist / (16.0 * jnp.pi / 180.0)) ** 2) * state.land_mask.data
        area_w = state.land_mask.data * mesh.areaCell
        eta0 = eta0 - jnp.sum(eta0 * area_w) / jnp.maximum(jnp.sum(area_w), 1.0)
        return state._replace(
            H_bathy=state.H_bathy.replace(data=H_bathy),
            eta=state.eta.replace(data=eta0.astype(state.eta.data.dtype)),
        )

    if case.init_kind == "thermohaline":
        return rest_state_mpas_ocean(
            mesh,
            z_coord,
            T_surface=24.0,
            T_deep=1.5,
            S_uniform=34.8,
            H_max=4500.0,
            land_lat_threshold=90.0,
        )

    if case.init_kind == "phillips_two_layer":
        state = rest_state_mpas_ocean(
            mesh,
            z_coord,
            T_surface=17.0,
            T_deep=7.0,
            S_uniform=35.0,
            H_max=3500.0,
            land_lat_threshold=90.0,
        )
        lat_deg = mesh.latCell * (180.0 / jnp.pi)
        u_jet_cell = 0.30 * jnp.exp(-((lat_deg - 45.0) / 14.0) ** 2) * state.land_mask.data
        u_jet_edge = _cell_to_edge_mean(mesh, u_jet_cell) * jnp.cos(mesh.angleEdge)
        u_data = jnp.zeros_like(state.u.data)
        u_data = u_data.at[:, 0].set(u_jet_edge)
        if z_coord.n_levels > 1:
            u_data = u_data.at[:, 1].set(-0.20 * u_jet_edge)
        eta_seed = 0.05 * jnp.sin(3.0 * mesh.lonCell) * jnp.cos(2.0 * mesh.latCell) * state.land_mask.data
        area_w = state.land_mask.data * mesh.areaCell
        eta_seed = eta_seed - jnp.sum(eta_seed * area_w) / jnp.maximum(jnp.sum(area_w), 1.0)
        return state._replace(
            u=state.u.replace(data=u_data.astype(state.u.data.dtype)),
            eta=state.eta.replace(data=eta_seed.astype(state.eta.data.dtype)),
        )

    if case.init_kind == "taylor_column":
        state = rest_state_mpas_ocean(
            mesh,
            z_coord,
            T_surface=8.0,
            T_deep=4.0,
            S_uniform=35.0,
            H_max=4500.0,
            land_lat_threshold=90.0,
        )
        dist = _great_circle_distance_rad(
            mesh.lonCell,
            mesh.latCell,
            float(jnp.pi),
            float(30.0 * jnp.pi / 180.0),
        )
        bump = 1200.0 * jnp.exp(-0.5 * (dist / (10.0 * jnp.pi / 180.0)) ** 2)
        H_bathy = jnp.clip(4500.0 - bump, 1200.0, 4500.0) * state.land_mask.data
        u_target_cell = 0.02 * jnp.cos(mesh.latCell) * state.land_mask.data
        u_target_edge = _cell_to_edge_mean(mesh, u_target_cell) * jnp.cos(mesh.angleEdge)
        u_data = jnp.broadcast_to(
            u_target_edge[:, None],
            state.u.data.shape,
        ).astype(state.u.data.dtype)
        return state._replace(
            H_bathy=state.H_bathy.replace(data=H_bathy),
            u=state.u.replace(data=u_data),
        )

    state = rest_state_mpas_ocean(
        mesh,
        z_coord,
        T_surface=20.0,
        T_deep=2.0,
        S_uniform=35.0,
        H_max=4000.0,
        land_lat_threshold=90.0,
    )

    if case.init_kind == "gravity_wave":
        dist = _great_circle_distance_rad(mesh.lonCell, mesh.latCell, float(jnp.pi), 0.0)
        eta0 = 1.0 * jnp.exp(-0.5 * (dist / 0.18) ** 2) * state.land_mask.data
        area_w = state.land_mask.data * mesh.areaCell
        eta0 = eta0 - jnp.sum(eta0 * area_w) / jnp.maximum(jnp.sum(area_w), 1.0)
        state = state._replace(eta=state.eta.replace(data=eta0.astype(state.eta.data.dtype)))

    if case.init_kind in ("wind_gyre", "holland_lin_gyre"):
        lat_deg = mesh.latCell * (180.0 / jnp.pi)
        lon_deg = _wrap_lon_deg(mesh.lonCell * (180.0 / jnp.pi))
        if case.init_kind == "wind_gyre":
            basin = (
                (lat_deg >= 10.0) & (lat_deg <= 70.0)
                & (lon_deg >= -80.0) & (lon_deg <= 20.0)
            )
        else:
            basin = (
                (lat_deg >= 10.0) & (lat_deg <= 70.0)
                & (lon_deg >= -70.0) & (lon_deg <= 20.0)
            )
        basin_mask = basin.astype(state.land_mask.data.dtype)
        H_bathy = 4000.0 * basin_mask
        edge_mask = _edge_ocean_mask(mesh, basin_mask)
        state = state._replace(
            land_mask=state.land_mask.replace(data=basin_mask),
            H_bathy=state.H_bathy.replace(data=H_bathy.astype(state.H_bathy.data.dtype)),
            eta=state.eta.replace(data=(state.eta.data * basin_mask).astype(state.eta.data.dtype)),
            T=state.T.replace(data=(state.T.data * basin_mask[:, None]).astype(state.T.data.dtype)),
            S=state.S.replace(data=(state.S.data * basin_mask[:, None]).astype(state.S.data.dtype)),
            u=state.u.replace(data=(state.u.data * edge_mask[:, None]).astype(state.u.data.dtype)),
        )

    return state


def _config_for_case(case: CaseSpec) -> MPASOceanConfig:
    cfg = MPASOceanConfig(
        A_h=1.0e4,
        K_h=1.0e3,
        A_v=1.0e-3,
        K_v=1.0e-4,
        n_barotropic_substeps=30,
        use_conservation_fixer=case.use_fixers,
        fix_volume=case.fix_volume,
        fix_heat=case.fix_heat,
        fix_salt=case.fix_salt,
        freshwater_closure="none",
    )
    if case.tag in ("wind_gyre", "holland_lin_gyre"):
        return cfg._replace(
            A_h=1.0e7,
            K_h=1.0e6,
            A_v=1.0e-2,
            K_v=1.0e-3,
            n_barotropic_substeps=50,
            barotropic_damping=1.0e-5,
        )
    if case.tag == "adiabatic_topography":
        return cfg._replace(
            A_h=5.0e4,
            K_h=5.0e3,
            n_barotropic_substeps=40,
            barotropic_damping=5.0e-6,
        )
    if case.tag == "thermohaline":
        return cfg._replace(
            A_h=5.0e4,
            K_h=2.0e4,
            A_v=5.0e-3,
            K_v=5.0e-4,
            n_barotropic_substeps=35,
        )
    if case.tag == "phillips_two_layer":
        return cfg._replace(
            A_h=2.0e5,
            K_h=2.0e4,
            A_v=1.0e-2,
            K_v=1.0e-3,
            n_barotropic_substeps=45,
            barotropic_damping=5.0e-6,
        )
    if case.tag == "taylor_column":
        return cfg._replace(
            A_h=5.0e5,
            K_h=5.0e4,
            A_v=1.0e-2,
            K_v=1.0e-3,
            n_barotropic_substeps=60,
            barotropic_damping=1.0e-5,
        )
    return cfg


def _forcing_step_for_case(
    case: CaseSpec,
    mesh: VoronoiMesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig,
):
    if case.forcing_kind == "none":
        return None

    if case.forcing_kind in ("wind_gyre", "holland_lin_gyre"):
        lat_edge_deg = mesh.latEdge * (180.0 / jnp.pi)
        y_norm = (lat_edge_deg - 10.0) / 60.0
        active = ((y_norm >= 0.0) & (y_norm <= 1.0)).astype(jnp.float64)
        tau_max = 0.05
        if case.forcing_kind == "wind_gyre":
            tau_x = -tau_max * jnp.sin(jnp.pi * y_norm) * active
        else:
            tau_x = -tau_max * jnp.cos(2.0 * jnp.pi * y_norm) * active
        edge_cos = jnp.cos(mesh.angleEdge)
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        drag_tau = 20.0 * 86400.0

        def _step(state, _step_num: int, dt_now: float):
            del _step_num
            mask_c = state.land_mask.data
            edge_mask = _edge_ocean_mask(mesh, mask_c)
            h_k = compute_layer_thickness(
                state.eta.data,
                state.H_bathy.data,
                z_coord,
                min_water_column_m=config.min_water_column_m,
            )
            h0 = jnp.maximum(h_k[:, 0], 1.0)
            h0_edge = jnp.maximum(0.5 * (h0[c1] + h0[c2]), 1.0)
            du0 = (tau_x * edge_cos) / (config.rho_0 * h0_edge)
            du0 = du0 * edge_mask
            drag_factor = jnp.exp(-dt_now / drag_tau)
            u_new = state.u.data * drag_factor
            u_new = u_new.at[:, 0].add(dt_now * du0)
            u_new = u_new * edge_mask[:, None]
            return state._replace(u=state.u.replace(data=u_new.astype(state.u.data.dtype)))

        return _step

    if case.forcing_kind == "thermohaline":
        lat = mesh.latCell
        T_star = 24.0 - 20.0 * jnp.sin(lat) ** 2
        S_star = 34.7 + 0.6 * jnp.cos(2.0 * lat)
        tau_T = 20.0 * 86400.0
        tau_S = 45.0 * 86400.0

        def _step(state, _step_num: int, dt_now: float):
            del _step_num
            mask = state.land_mask.data
            T = state.T.data
            S = state.S.data
            dT0 = (-(T[:, 0] - T_star) / tau_T) * mask
            dS0 = (-(S[:, 0] - S_star) / tau_S) * mask
            T_new = T.at[:, 0].set(T[:, 0] + dt_now * dT0)
            S_new = S.at[:, 0].set(S[:, 0] + dt_now * dS0)
            return state._replace(
                T=state.T.replace(data=T_new.astype(state.T.data.dtype)),
                S=state.S.replace(data=S_new.astype(state.S.data.dtype)),
            )

        return _step

    if case.forcing_kind == "phillips_two_layer":
        lat_deg = mesh.latCell * (180.0 / jnp.pi)
        n_bins = 72
        lat_edges = jnp.linspace(-90.0, 90.0, n_bins + 1)
        bin_idx = jnp.clip(
            jnp.searchsorted(lat_edges, lat_deg, side="right") - 1,
            0,
            n_bins - 1,
        ).astype(jnp.int32)
        base_weights = mesh.areaCell
        T_star_upper = 16.0 - 10.0 * jnp.sin(mesh.latCell) ** 2
        T_star_lower = 8.0 - 4.0 * jnp.sin(mesh.latCell) ** 2
        tau_relax = 15.0 * 86400.0
        tau_drag = 25.0 * 86400.0

        def _lat_bin_mean(field_1d: jax.Array, mask_1d: jax.Array) -> jax.Array:
            weights = base_weights * mask_1d
            wsum = jnp.bincount(bin_idx, weights=weights, length=n_bins)
            ssum = jnp.bincount(bin_idx, weights=field_1d * weights, length=n_bins)
            mean_bin = ssum / jnp.maximum(wsum, 1.0e-12)
            return mean_bin[bin_idx]

        def _step(state, _step_num: int, dt_now: float):
            del _step_num
            mask = state.land_mask.data
            T = state.T.data
            T_new = T
            if T.shape[1] >= 1:
                T0_zm = _lat_bin_mean(T[:, 0], mask)
                dT0 = (-(T0_zm - T_star_upper) / tau_relax) * mask
                T_new = T_new.at[:, 0].set(T[:, 0] + dt_now * dT0)
            if T.shape[1] >= 2:
                T1_zm = _lat_bin_mean(T[:, 1], mask)
                dT1 = (-(T1_zm - T_star_lower) / tau_relax) * mask
                T_new = T_new.at[:, 1].set(T[:, 1] + dt_now * dT1)
            edge_mask = _edge_ocean_mask(mesh, mask)
            drag_factor = jnp.exp(-dt_now / tau_drag)
            u_new = state.u.data * drag_factor * edge_mask[:, None]
            return state._replace(
                T=state.T.replace(data=T_new.astype(state.T.data.dtype)),
                u=state.u.replace(data=u_new.astype(state.u.data.dtype)),
            )

        return _step

    if case.forcing_kind == "taylor_column":
        u_target_edge = 0.02 * jnp.cos(mesh.latEdge) * jnp.cos(mesh.angleEdge)
        tau_restore = 30.0 * 86400.0

        def _step(state, _step_num: int, dt_now: float):
            del _step_num
            edge_mask = _edge_ocean_mask(mesh, state.land_mask.data)
            u_target = jnp.broadcast_to(u_target_edge[:, None], state.u.data.shape)
            u_new = state.u.data + dt_now * (u_target - state.u.data) / tau_restore
            u_new = u_new * edge_mask[:, None]
            return state._replace(u=state.u.replace(data=u_new.astype(state.u.data.dtype)))

        return _step

    raise ValueError(f"Unsupported forcing kind: {case.forcing_kind!r}")


def _write_profile_csv(
    profile: dict[str, np.ndarray],
    depth_m: np.ndarray,
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_m", "T_mean_degC", "S_mean_psu", "speed_mean_mps"])
        for k in range(depth_m.size):
            writer.writerow(
                [
                    float(depth_m[k]),
                    float(profile["T_mean_degC"][k]),
                    float(profile["S_mean_psu"][k]),
                    float(profile["speed_mean_mps"][k]),
                ]
            )


def _run_case(
    out_root: Path,
    mesh_level: int,
    n_levels: int,
    lloyd_iterations: int,
    case: CaseSpec,
    dt: float,
    run_hours: float | None,
    save_every: int,
    latlon_nlon: int,
    latlon_nlat: int,
) -> dict:
    mesh = create_voronoi_mesh(subdivision_level=mesh_level, lloyd_iterations=lloyd_iterations)
    if case.tag == "phillips_two_layer":
        z_coord = create_ocean_z_star(
            n_levels=2,
            H_max=3500.0,
            dz_surface=120.0,
            dz_deep=2200.0,
        )
    else:
        z_coord = create_ocean_z_star(
            n_levels=n_levels,
            H_max=5000.0,
            dz_surface=20.0,
            dz_deep=300.0,
        )

    config = _config_for_case(case)
    model = MPASOceanModel(mesh, z_coord, config)

    state0 = _prepare_initial_state(mesh, z_coord, case)
    forcing_step_fn = _forcing_step_for_case(case, mesh, z_coord, config)

    hours = float(run_hours) if run_hours is not None else float(case.default_hours)
    n_steps = max(1, int(hours * 3600.0 / dt))
    save_every = max(1, int(save_every))
    n_levels_case = int(z_coord.n_levels)
    case_dir = out_root / f"L{mesh_level}_K{n_levels_case}" / case.tag
    case_dir.mkdir(parents=True, exist_ok=True)

    lon2d, lat2d, pixel_mapper = _build_latlon_pixel_mapper(mesh, latlon_nlon, latlon_nlat)
    lon_cell_deg = _wrap_lon_deg(np.asarray(mesh.lonCell, dtype=np.float64) * 180.0 / np.pi)
    lat_cell_deg = np.asarray(mesh.latCell, dtype=np.float64) * 180.0 / np.pi
    depth_m = -np.asarray(z_coord.z_full_ref, dtype=np.float64)

    snapshot_steps = {
        0: "initial",
        max(1, n_steps // 4): "q1",
        max(1, n_steps // 2): "mid",
        max(1, (3 * n_steps) // 4): "q3",
        n_steps: "final",
    }

    scalar_rows: list[dict[str, float]] = []
    slab_rows: list[dict[str, float]] = []
    profile_snaps: dict[str, dict[str, np.ndarray]] = {}
    surface_snaps: dict[str, dict[str, np.ndarray]] = {}
    surface_native_snaps: dict[str, dict[str, np.ndarray]] = {}
    lat_depth_t_snaps: dict[str, np.ndarray] = {}
    lon_depth_t_snaps: dict[str, np.ndarray] = {}
    lat_depth_speed_snaps: dict[str, np.ndarray] = {}
    lon_depth_speed_snaps: dict[str, np.ndarray] = {}
    stable = True
    blowup_step = -1

    lat_bins = _default_lat_bins_mpas(mesh, np.asarray(state0.land_mask.data))
    lon_bins = _default_lon_bins_mpas(mesh, np.asarray(state0.land_mask.data))
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

    def record_step(step: int, state) -> None:
        diag = _ocean_diagnostics(state, mesh, z_coord, config)
        row = {
            "step": float(step),
            "time_seconds": float(step * dt),
            "time_hours": float(step * dt / 3600.0),
            **diag,
        }
        scalar_rows.append(row)

        slabs = _vertical_slab_means(state, mesh, z_coord, config)
        slab_rows.append(
            {
                "step": float(step),
                "time_seconds": float(step * dt),
                "time_hours": float(step * dt / 3600.0),
                **slabs,
            }
        )

        label = snapshot_steps.get(step, None)
        if label is not None:
            profiles = _horizontal_profiles(state, mesh)
            profile_snaps[label] = profiles
            u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
            speed_3d = np.asarray(jnp.sqrt(u_east ** 2 + v_north ** 2), dtype=np.float64)
            speed = jnp.sqrt(u_east[:, 0] ** 2 + v_north[:, 0] ** 2)
            surface_snaps[label] = {
                "eta": pixel_mapper(np.asarray(state.eta.data, dtype=np.float64)),
                "T_surface": pixel_mapper(np.asarray(state.T.data[:, 0], dtype=np.float64)),
                "S_surface": pixel_mapper(np.asarray(state.S.data[:, 0], dtype=np.float64)),
                "speed_surface": pixel_mapper(np.asarray(speed, dtype=np.float64)),
            }
            surface_native_snaps[label] = {
                "eta": np.asarray(state.eta.data, dtype=np.float64),
                "T_surface": np.asarray(state.T.data[:, 0], dtype=np.float64),
                "S_surface": np.asarray(state.S.data[:, 0], dtype=np.float64),
                "speed_surface": np.asarray(speed, dtype=np.float64),
            }
            _, lat_t = _compute_binned_depth_section_mpas(
                np.asarray(state.T.data, dtype=np.float64),
                np.asarray(mesh.latCell, dtype=np.float64) * 180.0 / np.pi,
                np.asarray(mesh.areaCell, dtype=np.float64),
                np.asarray(state.land_mask.data, dtype=np.float64),
                lat_bins,
                periodic=False,
            )
            _, lon_t = _compute_binned_depth_section_mpas(
                np.asarray(state.T.data, dtype=np.float64),
                (np.asarray(mesh.lonCell, dtype=np.float64) * 180.0 / np.pi + 360.0) % 360.0,
                np.asarray(mesh.areaCell, dtype=np.float64),
                np.asarray(state.land_mask.data, dtype=np.float64),
                lon_bins,
                periodic=True,
            )
            _, lat_speed = _compute_binned_depth_section_mpas(
                speed_3d,
                np.asarray(mesh.latCell, dtype=np.float64) * 180.0 / np.pi,
                np.asarray(mesh.areaCell, dtype=np.float64),
                np.asarray(state.land_mask.data, dtype=np.float64),
                lat_bins,
                periodic=False,
            )
            _, lon_speed = _compute_binned_depth_section_mpas(
                speed_3d,
                (np.asarray(mesh.lonCell, dtype=np.float64) * 180.0 / np.pi + 360.0) % 360.0,
                np.asarray(mesh.areaCell, dtype=np.float64),
                np.asarray(state.land_mask.data, dtype=np.float64),
                lon_bins,
                periodic=True,
            )
            lat_depth_t_snaps[label] = lat_t
            lon_depth_t_snaps[label] = lon_t
            lat_depth_speed_snaps[label] = lat_speed
            lon_depth_speed_snaps[label] = lon_speed
            _write_profile_csv(profiles, depth_m, case_dir / f"horizontal_profile_{label}.csv")

    state = state0
    start = time.time()
    record_step(0, state)

    for step in range(1, n_steps + 1):
        if forcing_step_fn is not None:
            state = forcing_step_fn(state, step, dt)
        state = model.step(state, dt)
        if (step % save_every == 0) or (step == n_steps) or (step in snapshot_steps):
            jax.block_until_ready(state.eta.data)
            record_step(step, state)
            if not scalar_rows[-1]["all_finite"]:
                stable = False
                blowup_step = step
                break

    wall_s = time.time() - start

    if "final" not in profile_snaps:
        profile_snaps["final"] = _horizontal_profiles(state, mesh)
        u_east, v_north = reconstruct_cell_velocity(state.u.data, mesh)
        speed = jnp.sqrt(u_east[:, 0] ** 2 + v_north[:, 0] ** 2)
        surface_snaps["final"] = {
            "eta": pixel_mapper(np.asarray(state.eta.data, dtype=np.float64)),
            "T_surface": pixel_mapper(np.asarray(state.T.data[:, 0], dtype=np.float64)),
            "S_surface": pixel_mapper(np.asarray(state.S.data[:, 0], dtype=np.float64)),
            "speed_surface": pixel_mapper(np.asarray(speed, dtype=np.float64)),
        }
        surface_native_snaps["final"] = {
            "eta": np.asarray(state.eta.data, dtype=np.float64),
            "T_surface": np.asarray(state.T.data[:, 0], dtype=np.float64),
            "S_surface": np.asarray(state.S.data[:, 0], dtype=np.float64),
            "speed_surface": np.asarray(speed, dtype=np.float64),
        }
        _write_profile_csv(
            profile_snaps["final"], depth_m, case_dir / "horizontal_profile_final.csv"
        )

    _write_csv(scalar_rows, case_dir / "scalar_diagnostics.csv")
    _write_csv(slab_rows, case_dir / "vertical_slab_averages.csv")
    _plot_scalar_timeseries(
        scalar_rows,
        case_dir / "scalar_diagnostics.png",
        title=f"{case.label} | MPAS ocean L{mesh_level} K{n_levels_case}",
    )
    _plot_vertical_slab_timeseries(
        slab_rows,
        case_dir / "vertical_slab_averages.png",
        title=f"{case.label} | vertical slab means",
    )
    _plot_horizontal_profiles(
        profile_snaps,
        depth_m,
        case_dir / "horizontal_slab_profiles.png",
        title=f"{case.label} | horizontal slab means by depth",
    )
    _plot_surface_snapshots(
        surface_snaps,
        lon2d,
        lat2d,
        case_dir / "surface_snapshots_latlon.png",
        title=f"{case.label} | surface maps (lat-lon pixels)",
    )
    _plot_surface_snapshots_native(
        surface_native_snaps,
        lon_cell_deg,
        lat_cell_deg,
        case_dir / "surface_snapshots_native.png",
        title=f"{case.label} | native MPAS surface maps",
    )
    _plot_depth_section_snapshots(
        case_dir / "lat_depth_sections_temperature.png",
        z_coord,
        lat_centers,
        "Latitude (deg)",
        lat_depth_t_snaps,
        "Temperature (degC)",
        "RdYlBu_r",
    )
    _plot_depth_section_snapshots(
        case_dir / "lon_depth_sections_temperature.png",
        z_coord,
        lon_centers,
        "Longitude (deg)",
        lon_depth_t_snaps,
        "Temperature (degC)",
        "RdYlBu_r",
    )
    _plot_depth_section_snapshots(
        case_dir / "lat_depth_sections_speed.png",
        z_coord,
        lat_centers,
        "Latitude (deg)",
        lat_depth_speed_snaps,
        "Speed (m/s)",
        "magma",
    )
    _plot_depth_section_snapshots(
        case_dir / "lon_depth_sections_speed.png",
        z_coord,
        lon_centers,
        "Longitude (deg)",
        lon_depth_speed_snaps,
        "Speed (m/s)",
        "magma",
    )

    np.savez(
        case_dir / "surface_snapshots_latlon.npz",
        lon2d=lon2d,
        lat2d=lat2d,
        lon_cell_deg=lon_cell_deg,
        lat_cell_deg=lat_cell_deg,
        **{f"{k}_eta": v["eta"] for k, v in surface_snaps.items()},
        **{f"{k}_T_surface": v["T_surface"] for k, v in surface_snaps.items()},
        **{f"{k}_S_surface": v["S_surface"] for k, v in surface_snaps.items()},
        **{f"{k}_speed_surface": v["speed_surface"] for k, v in surface_snaps.items()},
        **{f"{k}_native_eta": v["eta"] for k, v in surface_native_snaps.items()},
        **{f"{k}_native_T_surface": v["T_surface"] for k, v in surface_native_snaps.items()},
        **{f"{k}_native_S_surface": v["S_surface"] for k, v in surface_native_snaps.items()},
        **{f"{k}_native_speed_surface": v["speed_surface"] for k, v in surface_native_snaps.items()},
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        **{f"{k}_lat_depth_temperature": v for k, v in lat_depth_t_snaps.items()},
        **{f"{k}_lon_depth_temperature": v for k, v in lon_depth_t_snaps.items()},
        **{f"{k}_lat_depth_speed": v for k, v in lat_depth_speed_snaps.items()},
        **{f"{k}_lon_depth_speed": v for k, v in lon_depth_speed_snaps.items()},
    )

    d0 = scalar_rows[0]
    df = scalar_rows[-1]
    result = {
        "case": case.tag,
        "label": case.label,
        "mesh_level": mesh_level,
        "n_levels": n_levels_case,
        "n_cells": int(mesh.nCells),
        "n_edges": int(mesh.nEdges),
        "n_steps": int(df["step"]),
        "dt": float(dt),
        "duration_hours": float(df["time_hours"]),
        "stable": bool(stable),
        "blowup_step": int(blowup_step),
        "wall_time_s": float(wall_s),
        "volume_drift_rel": (df["volume"] - d0["volume"]) / max(abs(d0["volume"]), 1.0e-30),
        "heat_drift_rel": (df["heat"] - d0["heat"]) / max(abs(d0["heat"]), 1.0e-30),
        "salt_drift_rel": (df["salt"] - d0["salt"]) / max(abs(d0["salt"]), 1.0e-30),
        "final_eta_mean_m": df["eta_mean_m"],
        "final_surface_speed_mean_mps": df["speed_surface_mean_mps"],
        "all_finite": bool(df["all_finite"]),
        "case_dir": str(case_dir),
    }

    with (case_dir / "summary.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (case_dir / "results.txt").open("w") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MPAS icosahedral ocean test cases with slab diagnostics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/ocean/icosahedral_ocean_tests_<timestamp>).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help=f"Comma-separated cases or 'all'. Valid: {sorted(CASE_SPECS)}",
    )
    parser.add_argument(
        "--mesh-levels",
        type=str,
        default="2",
        help="Comma-separated Voronoi mesh subdivision levels (e.g., 2,3).",
    )
    parser.add_argument(
        "--n-levels",
        type=str,
        default="5",
        help="Comma-separated vertical levels (e.g., 5,10).",
    )
    parser.add_argument("--dt", type=float, default=60.0, help="Timestep in seconds.")
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="If set, override per-case default integration length with this value [hours].",
    )
    parser.add_argument("--save-every", type=int, default=10, help="Save diagnostics every N steps.")
    parser.add_argument("--lloyd-iterations", type=int, default=30)
    parser.add_argument("--latlon-nlon", type=int, default=360)
    parser.add_argument("--latlon-nlat", type=int, default=181)
    parser.add_argument("--x64", action="store_true", help="Enable JAX float64 mode.")
    args = parser.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(f"results/ocean/icosahedral_ocean_tests_{stamp}")
    else:
        out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cases.strip().lower() == "all":
        case_tags = list(CASE_SPECS.keys())
    else:
        case_tags = _parse_csv_list(args.cases)
        bad = [c for c in case_tags if c not in CASE_SPECS]
        if bad:
            raise ValueError(f"Unsupported cases {bad!r}. Valid: {sorted(CASE_SPECS)}")

    mesh_levels = _parse_csv_ints(args.mesh_levels)
    n_levels_list = _parse_csv_ints(args.n_levels)
    if not mesh_levels:
        raise ValueError("No mesh levels specified")
    if not n_levels_list:
        raise ValueError("No vertical levels specified")

    print("Running MPAS ocean test suite")
    print(f"  output={out_root}")
    print(f"  cases={case_tags}")
    print(f"  mesh_levels={mesh_levels}, n_levels={n_levels_list}")
    print(f"  dt={args.dt}, hours={args.hours}, save_every={args.save_every}, x64={args.x64}")

    all_results = []
    t0 = time.time()
    for mesh_level in mesh_levels:
        for n_levels in n_levels_list:
            for case_tag in case_tags:
                case = CASE_SPECS[case_tag]
                print(f"\n[{case_tag}] L{mesh_level} K{n_levels}")
                run = _run_case(
                    out_root=out_root,
                    mesh_level=mesh_level,
                    n_levels=n_levels,
                    lloyd_iterations=int(args.lloyd_iterations),
                    case=case,
                    dt=float(args.dt),
                    run_hours=(None if args.hours is None else float(args.hours)),
                    save_every=int(args.save_every),
                    latlon_nlon=int(args.latlon_nlon),
                    latlon_nlat=int(args.latlon_nlat),
                )
                all_results.append(run)
                print(
                    "  stable={stable}, drift(volume,heat,salt)=({dv:+.3e}, {dh:+.3e}, {ds:+.3e}), "
                    "wall={wall:.2f}s".format(
                        stable=run["stable"],
                        dv=run["volume_drift_rel"],
                        dh=run["heat_drift_rel"],
                        ds=run["salt_drift_rel"],
                        wall=run["wall_time_s"],
                    )
                )

    summary = {
        "n_runs": len(all_results),
        "total_wall_time_s": time.time() - t0,
        "runs": all_results,
    }
    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone")
    print(f"  total_wall_time_s={summary['total_wall_time_s']:.2f}")
    print(f"  summary={out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
