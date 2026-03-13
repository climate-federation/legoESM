#!/usr/bin/env python
"""Run standardized ocean model tests with visualization.

Three test cases:
  1. Rest-state adjustment — verify tendency magnitudes and conservation
  2. Barotropic gravity wave — Gaussian SSH perturbation, wave propagation
  3. Wind-driven gyre — idealized zonal wind stress, Sverdrup balance

Each test produces PNG visualizations and a summary of diagnostics.

Usage:
    cd /Users/pierregentine/legoESM
    source .venv/bin/activate

    # Quick test (C8, 10 levels, 5 days)
    python scripts/run_ocean_tests.py

    # Higher resolution (C16, 20 levels, 30 days)
    python scripts/run_ocean_tests.py --resolution 16 --levels 20 --days 30
"""

import argparse
import json
import os
import tempfile
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np


def _ensure_mpl_config_dir() -> None:
    """Ensure Matplotlib cache/config directory is writable."""
    current = os.environ.get("MPLCONFIGDIR", "")
    if current and os.path.isdir(current) and os.access(current, os.W_OK):
        return
    candidate = os.path.join(tempfile.gettempdir(), "legoesm_mplconfig")
    os.makedirs(candidate, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = candidate


_ensure_mpl_config_dir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


IDEALIZED_LAND_LAT_THRESHOLD = 90.0  # Fully oceanic mask for idealized cases

# Cases with explicit forcing — conservation drift includes forced tendency
# and should NOT be interpreted as numerical conservation error.
_FORCED_CASES = {
    "wind_gyre": "Wind stress + Rayleigh drag",
    "holland_lin_gyre": "Wind stress + Rayleigh drag",
    "thermohaline": "SST/SSS restoring (non-conservative by design)",
    "phillips_two_layer": "Zonal-mean T relaxation + Rayleigh drag",
    "taylor_column": "Background flow restoring",
}

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean import (
    OceanModel,
    OCEAN_DISCRETIZATIONS,
    OceanConfig,
    create_ocean_z_star,
    rest_state_ocean,
)
from legoesm.ocean.eos import rho_0
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.conservation import _ocean_area_sum, _ocean_volume_sum


# =====================================================================
# Helpers
# =====================================================================

def scatter_field(ax, lon_deg, lat_deg, data, cmap, vmin, vmax,
                  point_size=2.0, alpha=0.9):
    """Scatter plot a 2D field on all 6 cubed-sphere faces."""
    for face in range(6):
        sc = ax.scatter(
            lon_deg[face].ravel(), lat_deg[face].ravel(),
            c=np.asarray(data[face]).ravel(),
            s=point_size, cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), edgecolors="none", alpha=alpha,
        )
    return sc


def _idealized_projection(central_longitude: float = 0.0):
    """Map projection for idealized all-ocean experiments."""
    return ccrs.Mollweide(central_longitude=central_longitude)


def _style_idealized_axes(ax):
    """Style map axes for idealized all-ocean plots (no real coastlines)."""
    ax.gridlines(linewidth=0.3, alpha=0.4)
    ax.set_global()


def _default_lat_bins(grid, mask_2d, *, min_bins=24, max_bins=72):
    """Choose adaptive latitude bins from wet-cell coverage.

    Fixed high bin counts at coarse resolution can create empty latitude bins,
    which then appear as spurious horizontal/latitudinal bands in sections.
    """
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    mask = np.asarray(mask_2d, dtype=np.float64)
    wet_lat = lat_deg[mask > 0.5]
    if wet_lat.size < 2:
        return np.linspace(-90.0, 90.0, min_bins + 1)

    lat_lo = float(np.nanmin(wet_lat))
    lat_hi = float(np.nanmax(wet_lat))
    span = max(lat_hi - lat_lo, 1.0)
    pad = max(0.5, 0.02 * span)
    lat_lo = max(-90.0, lat_lo - pad)
    lat_hi = min(90.0, lat_hi + pad)

    n_lat, _ = _default_pixel_dims(
        grid,
        mask_2d,
        min_lat=min_bins,
        max_lat=max_bins,
        oversample=1.0,
    )
    return np.linspace(lat_lo, lat_hi, n_lat + 1)


def _default_lon_bins(grid, mask_2d, *, min_bins=36, max_bins=180):
    """Choose adaptive longitude bins from wet-cell density."""
    mask = np.asarray(mask_2d, dtype=np.float64)
    wet_count = int(np.sum(mask > 0.5))
    if wet_count < 4:
        return np.linspace(0.0, 360.0, min_bins + 1)

    min_lat = max(12, min_bins // 2)
    max_lat = max(24, max_bins // 2)
    _, n_lon = _default_pixel_dims(
        grid,
        mask_2d,
        min_lat=min_lat,
        max_lat=max_lat,
        oversample=1.0,
    )
    n_lon = int(np.clip(n_lon, min_bins, max_bins))
    return np.linspace(0.0, 360.0, n_lon + 1)


def _default_pixel_dims(grid, mask_2d, *, min_lat=24, max_lat=180, oversample=1.35):
    """Choose a regular lat-lon pixel grid compatible with wet-cell density."""
    mask = np.asarray(mask_2d, dtype=np.float64)
    n_wet = int(np.sum(mask > 0.5))
    if n_wet < 4:
        n_lat = int(min_lat)
    else:
        # n_lon ~ 2*n_lat on a lat-lon grid; pick target cells ~ O(n_wet).
        n_lat = int(np.clip(round(oversample * np.sqrt(n_wet / 2.0)), min_lat, max_lat))
    n_lon = int(np.clip(2 * n_lat, 2 * min_lat, 360))
    return n_lat, n_lon


def _fill_missing_bands(section, *, periodic=False):
    """Fill NaN bins by 1D interpolation (per depth level)."""
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

    # Fill depth levels with no finite bins from nearest valid level.
    finite_level = np.any(np.isfinite(out), axis=0)
    if np.any(finite_level):
        level_idx = np.arange(out.shape[1], dtype=np.int64)
        for k in level_idx[~finite_level]:
            nearest = level_idx[finite_level][
                np.argmin(np.abs(level_idx[finite_level] - k))
            ]
            out[:, k] = out[:, nearest]

    # Last resort: prevent NaNs in output diagnostics.
    finite_all = np.isfinite(out)
    if np.any(finite_all):
        fill = float(np.nanmean(out[finite_all]))
        out[~finite_all] = fill
    else:
        out[...] = 0.0

    return out


def _fill_missing_lat_bands(section):
    """Fill NaN latitude bins by interpolation (non-periodic)."""
    return _fill_missing_bands(section, periodic=False)


def _fill_missing_lon_bands(section):
    """Fill NaN longitude bins by interpolation (periodic)."""
    return _fill_missing_bands(section, periodic=True)


def _edges_from_centers(centers):
    """Build monotonic cell edges from cell centers."""
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    e = np.empty(c.size + 1, dtype=np.float64)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - (e[1] - c[0])
    e[-1] = c[-1] + (c[-1] - e[-2])
    return e


def _compute_lat_depth_section(field_3d, grid, mask_2d, lat_bins_deg):
    """Latitude-depth section from area-weighted zonal means in latitude bins."""
    field = np.asarray(field_3d, dtype=np.float64)
    if field.ndim != 4:
        raise ValueError(f"Expected field_3d with shape (face, x, y, level), got {field.shape}")

    n_levels = field.shape[-1]
    n_bins = int(len(lat_bins_deg) - 1)
    section = np.full((n_bins, n_levels), np.nan, dtype=np.float64)
    lat_deg = np.asarray(grid.lat, dtype=np.float64) * 180.0 / np.pi
    area = np.asarray(grid.area, dtype=np.float64)
    if mask_2d is None:
        mask = np.ones_like(area, dtype=np.float64)
    else:
        mask = np.asarray(mask_2d, dtype=np.float64)

    lat_flat = lat_deg.ravel()
    weight_flat = (area * mask).ravel()
    lat_idx = np.digitize(lat_flat, lat_bins_deg, right=False) - 1
    lat_idx = np.clip(lat_idx, 0, n_bins - 1)
    valid_geom = np.isfinite(lat_flat) & np.isfinite(weight_flat) & (weight_flat > 0.0)

    for k in range(n_levels):
        data_flat = np.asarray(field[..., k], dtype=np.float64).ravel()
        valid = valid_geom & np.isfinite(data_flat)
        if not np.any(valid):
            continue

        idx = lat_idx[valid]
        w = weight_flat[valid]
        v = data_flat[valid]
        sum_w = np.bincount(idx, weights=w, minlength=n_bins)
        sum_v = np.bincount(idx, weights=w * v, minlength=n_bins)

        wet = sum_w > 0.0
        section[wet, k] = sum_v[wet] / sum_w[wet]

    section = _fill_missing_lat_bands(section)
    lat_centers = 0.5 * (np.asarray(lat_bins_deg[:-1]) + np.asarray(lat_bins_deg[1:]))
    return lat_centers, section


def _compute_lon_depth_section(field_3d, grid, mask_2d, lon_bins_deg):
    """Longitude-depth section from area-weighted means in longitude bins."""
    field = np.asarray(field_3d, dtype=np.float64)
    if field.ndim != 4:
        raise ValueError(f"Expected field_3d with shape (face, x, y, level), got {field.shape}")

    n_levels = field.shape[-1]
    n_bins = int(len(lon_bins_deg) - 1)
    section = np.full((n_bins, n_levels), np.nan, dtype=np.float64)
    lon_deg = (np.asarray(grid.lon, dtype=np.float64) * 180.0 / np.pi + 360.0) % 360.0
    area = np.asarray(grid.area, dtype=np.float64)
    if mask_2d is None:
        mask = np.ones_like(area, dtype=np.float64)
    else:
        mask = np.asarray(mask_2d, dtype=np.float64)

    lon_flat = lon_deg.ravel()
    weight_flat = (area * mask).ravel()
    lon_idx = np.digitize(lon_flat, lon_bins_deg, right=False) - 1
    lon_idx = np.clip(lon_idx, 0, n_bins - 1)
    valid_geom = np.isfinite(lon_flat) & np.isfinite(weight_flat) & (weight_flat > 0.0)

    for k in range(n_levels):
        data_flat = np.asarray(field[..., k], dtype=np.float64).ravel()
        valid = valid_geom & np.isfinite(data_flat)
        if not np.any(valid):
            continue

        idx = lon_idx[valid]
        w = weight_flat[valid]
        v = data_flat[valid]
        sum_w = np.bincount(idx, weights=w, minlength=n_bins)
        sum_v = np.bincount(idx, weights=w * v, minlength=n_bins)

        wet = sum_w > 0.0
        section[wet, k] = sum_v[wet] / sum_w[wet]

    section = _fill_missing_lon_bands(section)
    lon_centers = 0.5 * (np.asarray(lon_bins_deg[:-1]) + np.asarray(lon_bins_deg[1:]))
    return lon_centers, section


def _plot_lat_depth_sections(
    output_path,
    z_coord,
    lat_centers,
    panels,
    suptitle,
):
    """Plot one or more latitude-depth section panels."""
    depth = np.asarray(z_coord.z_full_ref)
    if float(np.nanmean(depth)) < 0.0:
        depth = -depth

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 6.0), squeeze=False)
    axes = axes.ravel()

    for ax, (title, section, cmap, symmetric, cbar_label) in zip(axes, panels):
        data = np.asarray(section).T  # (n_levels, n_lat)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "No ocean data", ha="center", va="center")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Latitude [deg]")
            ax.set_ylabel("Depth [m]")
            continue

        if symmetric:
            vmax = float(np.max(np.abs(finite)))
            vmax = max(vmax, 1.0e-12)
            vmin = -vmax
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12

        lat_edges = _edges_from_centers(lat_centers)
        depth_edges = _edges_from_centers(depth)
        data_m = np.ma.masked_invalid(data)
        cs = ax.pcolormesh(
            lat_edges,
            depth_edges,
            data_m,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("Latitude [deg]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(float(np.nanmax(depth)), float(np.nanmin(depth)))
        ax.grid(True, alpha=0.25)
        fig.colorbar(cs, ax=ax, orientation="vertical", pad=0.02, label=cbar_label)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_lon_depth_sections(
    output_path,
    z_coord,
    lon_centers,
    panels,
    suptitle,
):
    """Plot one or more longitude-depth section panels."""
    depth = np.asarray(z_coord.z_full_ref)
    if float(np.nanmean(depth)) < 0.0:
        depth = -depth

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0 * n_panels, 6.0), squeeze=False)
    axes = axes.ravel()

    for ax, (title, section, cmap, symmetric, cbar_label) in zip(axes, panels):
        data = np.asarray(section).T  # (n_levels, n_lon)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "No ocean data", ha="center", va="center")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Depth [m]")
            continue

        if symmetric:
            vmax = float(np.max(np.abs(finite)))
            vmax = max(vmax, 1.0e-12)
            vmin = -vmax
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12

        lon_edges = _edges_from_centers(lon_centers)
        depth_edges = _edges_from_centers(depth)
        data_m = np.ma.masked_invalid(data)
        cs = ax.pcolormesh(
            lon_edges,
            depth_edges,
            data_m,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(float(np.nanmin(lon_edges)), float(np.nanmax(lon_edges)))
        ax.set_ylim(float(np.nanmax(depth)), float(np.nanmin(depth)))
        ax.grid(True, alpha=0.25)
        fig.colorbar(cs, ax=ax, orientation="vertical", pad=0.02, label=cbar_label)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _remap_to_latlon_pixels(
    data_2d,
    grid,
    mask_2d=None,
    n_lat=None,
    n_lon=None,
    *,
    fill_empty=True,
):
    """Conservative-ish remap from cubed-sphere points to regular lat-lon pixels.

    Uses area-weighted binning into pixel cells for clear visual continuity.
    """
    lon_deg = (np.asarray(grid.lon) * 180.0 / np.pi + 360.0) % 360.0
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    area = np.asarray(grid.area, dtype=np.float64)
    data = np.asarray(data_2d, dtype=np.float64)
    if mask_2d is None:
        mask = np.ones_like(area, dtype=np.float64)
    else:
        mask = np.asarray(mask_2d, dtype=np.float64)

    if n_lat is None:
        n_lat, n_lon_auto = _default_pixel_dims(grid, mask)
        if n_lon is None:
            n_lon = n_lon_auto
    if n_lon is None:
        n_lon = int(np.clip(2 * int(n_lat), 48, 360))
    n_lat = int(n_lat)
    n_lon = int(n_lon)

    lon_i = np.floor(lon_deg / 360.0 * n_lon).astype(np.int64)
    lat_i = np.floor((lat_deg + 90.0) / 180.0 * n_lat).astype(np.int64)
    lon_i = np.clip(lon_i, 0, n_lon - 1)
    lat_i = np.clip(lat_i, 0, n_lat - 1)

    weights = area * mask
    valid = np.isfinite(data) & np.isfinite(weights) & (weights > 0.0)
    idx_flat = (lat_i * n_lon + lon_i).ravel()
    idx_valid = idx_flat[valid.ravel()]
    w_valid = weights.ravel()[valid.ravel()]
    v_valid = data.ravel()[valid.ravel()]

    n_cells = n_lat * n_lon
    sum_w = np.bincount(idx_valid, weights=w_valid, minlength=n_cells)
    sum_v = np.bincount(idx_valid, weights=w_valid * v_valid, minlength=n_cells)

    out = np.full(n_cells, np.nan, dtype=np.float64)
    wet = sum_w > 0.0
    out[wet] = sum_v[wet] / sum_w[wet]
    out = out.reshape(n_lat, n_lon)
    if fill_empty:
        out = _fill_nan_pixel_gaps(out)

    lon_edges = np.linspace(0.0, 360.0, n_lon + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    return lon_edges, lat_edges, out


def _fill_nan_pixel_gaps(pix, max_iter=8):
    """Fill empty remap bins using local-neighborhood inpainting.

    Coarse cubed-sphere grids projected to fine regular lat-lon pixels leave
    many empty bins. This pass fills those bins from adjacent finite neighbors
    so the visualization is spatially continuous.
    """
    arr = np.asarray(pix, dtype=np.float64).copy()
    if arr.ndim != 2:
        return arr
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr

    for _ in range(max_iter):
        if np.all(finite):
            break

        vals = np.where(finite, arr, 0.0)
        sum_n = np.zeros_like(arr)
        cnt_n = np.zeros_like(arr, dtype=np.int32)

        # Longitude neighbors (periodic).
        for shift in (-1, 1):
            v = np.roll(vals, shift=shift, axis=1)
            m = np.roll(finite, shift=shift, axis=1)
            sum_n += v * m
            cnt_n += m.astype(np.int32)

        # Latitude neighbors (non-periodic).
        for shift in (-1, 1):
            v = np.empty_like(vals)
            m = np.empty_like(finite)
            if shift == -1:
                v[:-1, :] = vals[1:, :]
                v[-1, :] = 0.0
                m[:-1, :] = finite[1:, :]
                m[-1, :] = False
            else:
                v[1:, :] = vals[:-1, :]
                v[0, :] = 0.0
                m[1:, :] = finite[:-1, :]
                m[0, :] = False
            sum_n += v * m
            cnt_n += m.astype(np.int32)

        to_fill = (~finite) & (cnt_n > 0)
        if not np.any(to_fill):
            break
        arr[to_fill] = sum_n[to_fill] / cnt_n[to_fill]
        finite[to_fill] = True

    return arr


def _plot_latlon_pixel_snapshots(
    output_path,
    snapshots,
    dt,
    grid,
    field_getter,
    title,
    cmap,
    cbar_label,
    symmetric=False,
    n_lat=None,
    n_lon=None,
):
    """Plot up to 4 snapshots as lat-lon pixel maps."""
    items = sorted(snapshots.items())[:4]
    remapped = []
    all_finite = []
    if items and (n_lat is None or n_lon is None):
        n_lat_auto, n_lon_auto = _default_pixel_dims(
            grid,
            np.asarray(items[0][1].land_mask.data),
        )
        if n_lat is None:
            n_lat = n_lat_auto
        if n_lon is None:
            n_lon = n_lon_auto
    for step_num, snap in items:
        fld = np.asarray(field_getter(snap), dtype=np.float64)
        lon_e, lat_e, pix = _remap_to_latlon_pixels(
            fld,
            grid,
            mask_2d=np.asarray(snap.land_mask.data),
            n_lat=n_lat,
            n_lon=n_lon,
        )
        remapped.append((step_num, lon_e, lat_e, pix))
        finite = pix[np.isfinite(pix)]
        if finite.size:
            all_finite.append(finite)
    if not remapped:
        return

    if all_finite:
        concat = np.concatenate(all_finite)
        if symmetric:
            vmax = max(float(np.nanmax(np.abs(concat))), 1.0e-12)
            vmin = -vmax
        else:
            vmin = float(np.nanmin(concat))
            vmax = float(np.nanmax(concat))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.ravel()
    pcm = None
    for idx, ax in enumerate(axes):
        if idx >= len(remapped):
            ax.axis("off")
            continue
        step_num, lon_e, lat_e, pix = remapped[idx]
        day = step_num * dt / 86400.0
        pcm = ax.pcolormesh(
            lon_e,
            lat_e,
            np.ma.masked_invalid(pix),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{cbar_label}\nstep {step_num}, t={day:.2f} d",
            fontsize=11,
            fontweight="bold",
        )
    if pcm is not None:
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.90, wspace=0.13, hspace=0.32)
        cax = fig.add_axes([0.92, 0.13, 0.02, 0.72])
        fig.colorbar(pcm, cax=cax, orientation="vertical", label=cbar_label)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    if pcm is None:
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.90, wspace=0.13, hspace=0.32)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_latlon_pixel_two_panel(
    output_path,
    state_a,
    state_b,
    label_a,
    label_b,
    grid,
    field_getter,
    title,
    cmap,
    cbar_label,
    field_getter_b=None,
    symmetric=False,
    n_lat=None,
    n_lon=None,
):
    """Plot two states as side-by-side lat-lon pixel maps."""
    if field_getter_b is None:
        field_getter_b = field_getter
    if n_lat is None or n_lon is None:
        n_lat_auto, n_lon_auto = _default_pixel_dims(grid, np.asarray(state_a.land_mask.data))
        if n_lat is None:
            n_lat = n_lat_auto
        if n_lon is None:
            n_lon = n_lon_auto
    lon_e, lat_e, pix_a = _remap_to_latlon_pixels(
        np.asarray(field_getter(state_a), dtype=np.float64),
        grid,
        mask_2d=np.asarray(state_a.land_mask.data),
        n_lat=n_lat,
        n_lon=n_lon,
    )
    _, _, pix_b = _remap_to_latlon_pixels(
        np.asarray(field_getter_b(state_b), dtype=np.float64),
        grid,
        mask_2d=np.asarray(state_b.land_mask.data),
        n_lat=n_lat,
        n_lon=n_lon,
    )
    finite = np.concatenate(
        [pix_a[np.isfinite(pix_a)], pix_b[np.isfinite(pix_b)]],
        axis=0,
    )
    if finite.size == 0:
        return
    if symmetric:
        vmax = max(float(np.nanmax(np.abs(finite))), 1.0e-12)
        vmin = -vmax
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if abs(vmax - vmin) < 1.0e-12:
            vmax = vmin + 1.0e-12

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), squeeze=False)
    axes = axes.ravel()
    pcm = None
    for ax, (label, pix) in zip(axes, [(label_a, pix_a), (label_b, pix_b)]):
        pcm = ax.pcolormesh(
            lon_e,
            lat_e,
            np.ma.masked_invalid(pix),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(-90.0, 90.0)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.grid(True, alpha=0.25)
        ax.set_title(label, fontsize=12, fontweight="bold")
    if pcm is not None:
        fig.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.88, wspace=0.16)
        cax = fig.add_axes([0.92, 0.16, 0.02, 0.66])
        fig.colorbar(pcm, cax=cax, orientation="vertical", label=cbar_label)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    if pcm is None:
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.88, wspace=0.16)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _rest_state_all_ocean(grid, z_coord, **kwargs):
    """Rest state with no land mask for idealized ocean cases."""
    return rest_state_ocean(
        grid,
        z_coord,
        land_lat_threshold=IDEALIZED_LAND_LAT_THRESHOLD,
        **kwargs,
    )


def _assert_all_ocean(state, label: str):
    """Guard that idealized cases are effectively all-ocean.

    Some cubed-sphere resolutions can leave a tiny number of masked edge cells
    due to geometric thresholding in grid generation. Treat these as acceptable
    for idealized tests while still rejecting meaningful land fractions.
    """
    ocean_fraction = float(jnp.mean(state.land_mask.data))
    if ocean_fraction < 0.999:
        raise ValueError(
            f"{label}: expected fully oceanic mask, got ocean_fraction={ocean_fraction:.6f}",
        )


def compute_ocean_diagnostics(state, grid, z_coord):
    """Compute scalar diagnostics for the ocean state."""
    mask = state.land_mask.data
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)

    # Volume: integral of eta over ocean
    vol = float(_ocean_area_sum(state.eta.data, mask, grid))

    # Heat: integral of T * h_k over ocean
    heat = float(_ocean_volume_sum(state.T.data, h_k, mask, grid))

    # Salt: integral of S * h_k over ocean
    salt = float(_ocean_volume_sum(state.S.data, h_k, mask, grid))

    # Kinetic energy: 0.5 * integral of (u^2 + v^2) * h_k over ocean
    ke = float(_ocean_volume_sum(
        0.5 * (state.u.data**2 + state.v.data**2), h_k, mask, grid,
    ))

    return {
        "volume": vol,
        "heat": heat,
        "salt": salt,
        "kinetic_energy": ke,
        "SST_mean": float(jnp.sum(state.T.data[..., 0] * mask) /
                          jnp.maximum(jnp.sum(mask), 1.0)),
        "SST_max": float(jnp.max(state.T.data[..., 0] * mask)),
        "SSH_min": float(jnp.min(jnp.where(mask > 0.5, state.eta.data, 0.0))),
        "SSH_max": float(jnp.max(state.eta.data * mask)),
        "u_max": float(jnp.max(jnp.abs(state.u.data))),
        "v_max": float(jnp.max(jnp.abs(state.v.data))),
    }


def _save_case_timeseries(case_dir, diagnostics, dt, n_steps, state, grid, case_title):
    """Save standardized conservation + mean-state time series for one case."""
    os.makedirs(case_dir, exist_ok=True)

    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    vol = np.array([d["volume"] for d in diagnostics], dtype=np.float64)
    heat = np.array([d["heat"] for d in diagnostics], dtype=np.float64)
    salt = np.array([d["salt"] for d in diagnostics], dtype=np.float64)
    ke = np.array([d["kinetic_energy"] for d in diagnostics], dtype=np.float64)
    sst_mean = np.array([d["SST_mean"] for d in diagnostics], dtype=np.float64)
    ssh_min = np.array([d["SSH_min"] for d in diagnostics], dtype=np.float64)
    ssh_max = np.array([d["SSH_max"] for d in diagnostics], dtype=np.float64)
    u_max = np.array([d["u_max"] for d in diagnostics], dtype=np.float64)
    v_max = np.array([d["v_max"] for d in diagnostics], dtype=np.float64)
    speed_max = np.sqrt(u_max * u_max + v_max * v_max)

    mask = np.asarray(state.land_mask.data, dtype=np.float64)
    ocean_area = float(np.sum(mask * np.asarray(grid.area, dtype=np.float64)))
    eta_mean = vol / max(ocean_area, 1.0)

    vol_rel = (vol - vol[0]) / max(abs(vol[0]), 1.0e-30)
    heat_rel = (heat - heat[0]) / max(abs(heat[0]), 1.0e-30)
    salt_rel = (salt - salt[0]) / max(abs(salt[0]), 1.0e-30)

    with open(os.path.join(case_dir, "timeseries.csv"), "w") as f:
        f.write(
            "time_days,eta_mean_m,SSH_min_m,SSH_max_m,SST_mean_degC,"
            "u_max_ms,v_max_ms,speed_max_ms,kinetic_energy,volume,heat,salt,"
            "volume_rel,heat_rel,salt_rel\n",
        )
        for i in range(n_diag):
            f.write(
                f"{times_days[i]:.8f},{eta_mean[i]:.12e},{ssh_min[i]:.12e},{ssh_max[i]:.12e},"
                f"{sst_mean[i]:.12e},{u_max[i]:.12e},{v_max[i]:.12e},{speed_max[i]:.12e},"
                f"{ke[i]:.12e},{vol[i]:.12e},{heat[i]:.12e},{salt[i]:.12e},"
                f"{vol_rel[i]:.12e},{heat_rel[i]:.12e},{salt_rel[i]:.12e}\n",
            )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(times_days, eta_mean, "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("Mean SSH eta [m]")
    axes[0, 0].set_title("Mean Sea Surface Height eta")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(times_days, sst_mean, "g-", linewidth=1.5)
    axes[0, 1].set_ylabel("Mean SST [degC]")
    axes[0, 1].set_title("Mean Sea Surface Temperature")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(times_days, speed_max, "r-", linewidth=1.5)
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_ylabel("Max speed [m/s]")
    axes[1, 0].set_title("Maximum Horizontal Speed")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_ylabel("Kinetic energy")
    axes[1, 1].set_title("Kinetic Energy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    fig.suptitle(f"{case_title} — Mean-State Time Series", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "mean_state_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(times_days, vol_rel, "b-", linewidth=1.5)
    axes[0].set_ylabel("Relative drift")
    axes[0].set_title("Volume Conservation")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[1].plot(times_days, heat_rel, "r-", linewidth=1.5)
    axes[1].set_ylabel("Relative drift")
    axes[1].set_title("Heat Conservation")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[2].plot(times_days, salt_rel, "g-", linewidth=1.5)
    axes[2].set_ylabel("Relative drift")
    axes[2].set_xlabel("Time [days]")
    axes[2].set_title("Salt Conservation")
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    fig.suptitle(f"{case_title} — Conservation Time Series", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "conservation_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_point_profile_evolution(output_path, z_coord, snapshots, dt, extractor, x_label, title):
    """Plot vertical profile evolution at one representative ocean point."""
    z_ref = np.asarray(z_coord.z_full_ref, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    for step_num, snap in sorted(snapshots.items()):
        day = step_num * dt / 86400.0
        profile = np.asarray(extractor(snap), dtype=np.float64)
        ax.plot(profile, z_ref, linewidth=1.5, label=f"Day {day:.2f}")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Depth [m]", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def summarize_case_metrics(state, diagnostics, grid):
    """Build machine-readable summary metrics for a test case."""
    first = diagnostics[0]
    last = diagnostics[-1]
    mask = state.land_mask.data

    all_finite = bool(
        jnp.all(jnp.isfinite(state.u.data))
        & jnp.all(jnp.isfinite(state.v.data))
        & jnp.all(jnp.isfinite(state.T.data))
        & jnp.all(jnp.isfinite(state.S.data))
        & jnp.all(jnp.isfinite(state.eta.data))
    )

    land = mask < 0.5
    land_3d = jnp.broadcast_to(land[..., jnp.newaxis], state.u.data.shape)
    if bool(jnp.any(land)):
        max_land_u = float(jnp.max(jnp.abs(jnp.where(land_3d, state.u.data, 0.0))))
        max_land_v = float(jnp.max(jnp.abs(jnp.where(land_3d, state.v.data, 0.0))))
        max_land_eta = float(jnp.max(jnp.abs(jnp.where(land, state.eta.data, 0.0))))
        land_zero = max(max_land_u, max_land_v, max_land_eta) <= 1.0e-10
    else:
        max_land_u = max_land_v = max_land_eta = 0.0
        land_zero = True

    ocean_area = float(jnp.sum(mask * grid.area))
    heat_drift_rel = (
        (last["heat"] - first["heat"]) / abs(first["heat"])
        if abs(first["heat"]) > 1.0e-12 else 0.0
    )
    salt_drift_rel = (
        (last["salt"] - first["salt"]) / abs(first["salt"])
        if abs(first["salt"]) > 1.0e-12 else 0.0
    )
    volume_mean_eta_drift = (
        (last["volume"] - first["volume"]) / max(ocean_area, 1.0)
    )
    speed_max = float(jnp.max(jnp.sqrt(state.u.data**2 + state.v.data**2)))

    return {
        "SSH_min": float(last["SSH_min"]),
        "SSH_max": float(last["SSH_max"]),
        "u_max": float(last["u_max"]),
        "v_max": float(last["v_max"]),
        "speed_max": float(speed_max),
        "SST_mean": float(last["SST_mean"]),
        "SST_max": float(last["SST_max"]),
        "kinetic_energy": float(last["kinetic_energy"]),
        "heat_drift_rel": float(heat_drift_rel),
        "salt_drift_rel": float(salt_drift_rel),
        "volume_mean_eta_drift": float(volume_mean_eta_drift),
        "all_finite": bool(all_finite),
        "land_zero": bool(land_zero),
        "max_land_u": float(max_land_u),
        "max_land_v": float(max_land_v),
        "max_land_eta": float(max_land_eta),
    }


def _config_to_dict(config: OceanConfig) -> dict[str, float | int | bool]:
    """Serialize OceanConfig to JSON-safe scalar dict."""
    return {
        "A_h": float(config.A_h),
        "K_h": float(config.K_h),
        "A_v": float(config.A_v),
        "K_v": float(config.K_v),
        "hyperdiff_coeff": float(config.hyperdiff_coeff),
        "n_barotropic_substeps": int(config.n_barotropic_substeps),
        "barotropic_diffusion_alpha": float(config.barotropic_diffusion_alpha),
        "barotropic_diffusion_dt_ref": float(config.barotropic_diffusion_dt_ref),
        "use_conservation_fixer": bool(config.use_conservation_fixer),
        "fix_volume": bool(config.fix_volume),
        "fix_heat": bool(config.fix_heat),
        "fix_salt": bool(config.fix_salt),
        "enable_runtime_checks": bool(config.enable_runtime_checks),
    }


def add_wind_stress_tendency(state, grid, z_coord, config, tau_max=0.1):
    """Compute wind stress tendency for wind-driven gyre.

    Idealized zonal wind stress: tau_x = -tau_max * cos(2*pi*lat/60)
    applied in the band 15N-75N, zero elsewhere.
    Applied as du/dt = tau_x / (rho_0 * h_surface) in the top layer.
    """
    lat_deg = grid.lat * (180.0 / jnp.pi)

    # Wind stress: sinusoidal profile over Northern Hemisphere mid-latitudes
    tau_x = -tau_max * jnp.cos(2.0 * jnp.pi * (lat_deg - 45.0) / 60.0)
    # Restrict to 15N-75N ocean band
    wind_mask = jnp.where(
        (lat_deg > 15.0) & (lat_deg < 75.0) & (state.land_mask.data > 0.5),
        1.0, 0.0,
    )
    tau_x = tau_x * wind_mask

    # Apply to surface layer only
    h_k = compute_layer_thickness(state.eta.data, state.H_bathy.data, z_coord)
    h_surface = jnp.maximum(h_k[..., 0], 1.0)
    du_surface = tau_x / (rho_0 * h_surface)

    # Build full 3D tendency (only surface layer)
    du_3d = jnp.zeros_like(state.u.data)
    du_3d = du_3d.at[..., 0].set(du_surface)

    return du_3d


# =====================================================================
# Test Case 1: Rest-state adjustment
# =====================================================================

def run_rest_state_test(
    grid,
    z_coord,
    config,
    dt,
    n_steps,
    output_dir,
    point_size,
    discretization="centered",
):
    """Rest-state adjustment: start from rest, verify small tendencies."""
    print("\n" + "=" * 70)
    print("TEST 1: Rest-State Adjustment")
    print("=" * 70)

    model = OceanModel(grid, z_coord, config, discretization=discretization)
    step_fn = model.step_checked if config.enable_runtime_checks else model.step
    state = _rest_state_all_ocean(grid, z_coord)
    _assert_all_ocean(state, "rest_state_test")
    state_init = state

    print(f"  SST range: [{float(jnp.min(state.T.data[..., 0])):.1f}, "
          f"{float(jnp.max(state.T.data[..., 0])):.1f}] degC")
    ocean_S = state.S.data * state.land_mask.data[..., jnp.newaxis]
    ocean_count = jnp.sum(state.land_mask.data) * z_coord.n_levels
    print(f"  Salinity: {float(jnp.sum(ocean_S) / jnp.maximum(ocean_count, 1.0)):.1f} PSU")

    # JIT warmup
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = step_fn(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Diagnostics storage
    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
    snapshot_steps = {int(f * n_steps): f for f in snapshot_fracs}
    snapshots = {0: state}
    diag_interval = max(1, n_steps // 50)

    # Integration
    print(f"\n{'Step':>6s}  {'Day':>6s}  {'SSH_max':>10s}  {'|u|_max':>10s}  "
          f"{'SST_mean':>10s}  {'KE':>12s}")
    print("-" * 70)

    t_start = time.time()
    for step in range(1, n_steps + 1):
        state = step_fn(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:6.1f}  {diag['SSH_max']:10.2e}  "
                  f"{diag['u_max']:10.2e}  {diag['SST_mean']:10.4f}  "
                  f"{diag['kinetic_energy']:12.2e}")
        if step in snapshot_steps:
            snapshots[step] = state

    total_time = time.time() - t_start
    steps_per_s = n_steps / max(total_time, 1.0e-12)
    print(f"\nCompleted in {total_time:.1f}s ({steps_per_s:.0f} steps/s)")

    # --- Visualization ---
    os.makedirs(f"{output_dir}/rest_state", exist_ok=True)
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    if HAS_CARTOPY:
        # SST snapshots at multiple times
        fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                                  subplot_kw={"projection": _idealized_projection()})
        axes = axes.ravel()
        for idx, (step_num, snap) in enumerate(sorted(snapshots.items())):
            if idx >= 4:
                break
            day = step_num * dt / 86400.0
            ax = axes[idx]
            sst = np.asarray(snap.T.data[..., 0])
            sc = scatter_field(ax, lon_deg, lat_deg, sst, "RdYlBu_r",
                               vmin=0, vmax=22, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SST [degC] — Day {day:.2f}", fontsize=13, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                     label="Temperature [degC]", pad=0.02)
        fig.suptitle("Rest-State Test — Sea Surface Temperature", fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/rest_state/sst.png", dpi=150, bbox_inches="tight")
        plt.close()

        # SSH final
        fig, ax = plt.subplots(1, 1, figsize=(14, 7),
                                subplot_kw={"projection": _idealized_projection()})
        ssh = np.asarray(state.eta.data)
        vabs = max(abs(float(np.nanmin(ssh))), abs(float(np.nanmax(ssh))), 1e-10)
        sc = scatter_field(ax, lon_deg, lat_deg, ssh, "RdBu_r",
                           -vabs, vabs, point_size=point_size)
        _style_idealized_axes(ax)
        ax.set_title("Sea Surface Height [m]", fontsize=14, fontweight="bold")
        fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical",
                     label="SSH [m]", pad=0.02)
        plt.savefig(f"{output_dir}/rest_state/ssh.png", dpi=150, bbox_inches="tight")
        plt.close()

    _plot_latlon_pixel_snapshots(
        f"{output_dir}/rest_state/sst_latlon_pixels.png",
        snapshots,
        dt,
        grid,
        field_getter=lambda s: s.T.data[..., 0],
        title="Rest-State Test — SST Snapshots (lat-lon pixels)",
        cmap="RdYlBu_r",
        cbar_label="Sea-surface temperature [degC]",
        symmetric=False,
    )
    _plot_latlon_pixel_snapshots(
        f"{output_dir}/rest_state/ssh_snapshots_latlon_pixels.png",
        snapshots,
        dt,
        grid,
        field_getter=lambda s: s.eta.data,
        title="Rest-State Test — SSH Snapshots (lat-lon pixels)",
        cmap="RdBu_r",
        cbar_label="Sea surface height eta [m]",
        symmetric=True,
    )

    # Conservation time series
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Volume (eta integral)
    vol = np.array([d["volume"] for d in diagnostics])
    axes[0, 0].plot(times_days, vol, "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("Volume integral [m^3]")
    axes[0, 0].set_title("Volume Conservation (eta integral)")
    axes[0, 0].grid(True, alpha=0.3)

    # Heat
    heat = np.array([d["heat"] for d in diagnostics])
    if abs(heat[0]) > 0:
        heat_rel = (heat - heat[0]) / abs(heat[0])
        axes[0, 1].plot(times_days, heat_rel, "r-", linewidth=1.5)
        axes[0, 1].set_ylabel("Relative change")
    else:
        axes[0, 1].plot(times_days, heat, "r-", linewidth=1.5)
        axes[0, 1].set_ylabel("Heat integral")
    axes[0, 1].set_title("Heat Conservation")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    # Salt
    salt = np.array([d["salt"] for d in diagnostics])
    if abs(salt[0]) > 0:
        salt_rel = (salt - salt[0]) / abs(salt[0])
        axes[1, 0].plot(times_days, salt_rel, "g-", linewidth=1.5)
        axes[1, 0].set_ylabel("Relative change")
    else:
        axes[1, 0].plot(times_days, salt, "g-", linewidth=1.5)
        axes[1, 0].set_ylabel("Salt integral")
    axes[1, 0].set_title("Salt Conservation")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    # Kinetic energy
    ke = np.array([d["kinetic_energy"] for d in diagnostics])
    axes[1, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[1, 1].set_ylabel("Kinetic energy [J/m]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Kinetic Energy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    fig.suptitle("Rest-State Test — Conservation Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rest_state/conservation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Vertical temperature profile
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    z_ref = np.asarray(z_coord.z_full_ref)
    # Pick a mid-latitude ocean point (face 0, center)
    mid = grid.n // 2
    T_init = np.asarray(state_init.T.data[0, mid, mid, :])
    T_final = np.asarray(state.T.data[0, mid, mid, :])
    ax.plot(T_init, z_ref, "b-o", markersize=3, label="Initial", linewidth=1.5)
    ax.plot(T_final, z_ref, "r--s", markersize=3, label="Final", linewidth=1.5)
    ax.set_xlabel("Temperature [degC]", fontsize=12)
    ax.set_ylabel("Depth [m]", fontsize=12)
    ax.set_title("Vertical Temperature Profile (mid-latitude point)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rest_state/T_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Vertical profile evolution across snapshots
    mid = grid.n // 2
    _plot_point_profile_evolution(
        f"{output_dir}/rest_state/T_profile_over_time.png",
        z_coord,
        snapshots,
        dt,
        lambda snap: np.asarray(snap.T.data[0, mid, mid, :]),
        "Temperature [degC]",
        "Vertical Temperature Profile Evolution (mid-latitude point)",
    )

    # Latitude-depth sections (initial/final)
    lat_bins = _default_lat_bins(grid, state.land_mask.data)
    lat_centers, T_init_sec = _compute_lat_depth_section(
        state_init.T.data,
        grid,
        state_init.land_mask.data,
        lat_bins,
    )
    _, T_final_sec = _compute_lat_depth_section(
        state.T.data,
        grid,
        state.land_mask.data,
        lat_bins,
    )
    _plot_lat_depth_sections(
        f"{output_dir}/rest_state/lat_depth_sections.png",
        z_coord,
        lat_centers,
        [
            ("Initial T", T_init_sec, "RdYlBu_r", False, "Temperature [degC]"),
            ("Final T", T_final_sec, "RdYlBu_r", False, "Temperature [degC]"),
        ],
        "Rest-State Test — Latitude-Depth Temperature Sections",
    )
    lon_bins = _default_lon_bins(grid, state.land_mask.data)
    lon_centers, T_init_lon_sec = _compute_lon_depth_section(
        state_init.T.data,
        grid,
        state_init.land_mask.data,
        lon_bins,
    )
    _, T_final_lon_sec = _compute_lon_depth_section(
        state.T.data,
        grid,
        state.land_mask.data,
        lon_bins,
    )
    _plot_lon_depth_sections(
        f"{output_dir}/rest_state/lon_depth_sections.png",
        z_coord,
        lon_centers,
        [
            ("Initial T", T_init_lon_sec, "RdYlBu_r", False, "Temperature [degC]"),
            ("Final T", T_final_lon_sec, "RdYlBu_r", False, "Temperature [degC]"),
        ],
        "Rest-State Test — Longitude-Depth Temperature Sections",
    )

    _save_case_timeseries(
        f"{output_dir}/rest_state",
        diagnostics,
        dt,
        n_steps,
        state,
        grid,
        "Rest-State Test",
    )

    return state, diagnostics


# =====================================================================
# Test Case 2: Barotropic gravity wave
# =====================================================================

def run_gravity_wave_test(
    grid,
    z_coord,
    config,
    dt,
    n_steps,
    output_dir,
    point_size,
    discretization="centered",
):
    """Barotropic gravity wave: Gaussian SSH perturbation."""
    print("\n" + "=" * 70)
    print("TEST 2: Barotropic Gravity Wave")
    print("=" * 70)

    # Start from rest state, add Gaussian SSH perturbation
    state = _rest_state_all_ocean(grid, z_coord)
    _assert_all_ocean(state, "gravity_wave_test")

    # Gaussian SSH perturbation centered at (lon=180, lat=0)
    lon_rad = grid.lon
    lat_rad = grid.lat
    lon0 = jnp.pi  # 180 degrees
    lat0 = 0.0     # equator
    sigma = 10.0 * jnp.pi / 180.0  # 10 degrees

    # Great-circle distance
    dlon = lon_rad - lon0
    dist_angle = jnp.arccos(
        jnp.clip(jnp.sin(lat_rad) * jnp.sin(lat0) +
                 jnp.cos(lat_rad) * jnp.cos(lat0) * jnp.cos(dlon), -1.0, 1.0)
    )
    eta_pert = 1.0 * jnp.exp(-0.5 * (dist_angle / sigma) ** 2)  # 1m amplitude
    eta_pert = eta_pert * state.land_mask.data

    state = state._replace(
        eta=state.eta.replace(data=eta_pert.astype(jnp.float32)),
    )
    state_init = state

    print(f"  SSH perturbation: {float(jnp.max(eta_pert)):.3f} m (Gaussian, sigma=10deg)")
    c_wave = float(jnp.sqrt(9.81 * 5500.0))
    print(f"  Expected wave speed: {c_wave:.0f} m/s")

    model = OceanModel(grid, z_coord, config, discretization=discretization)
    step_fn = model.step_checked if config.enable_runtime_checks else model.step

    # JIT warmup
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = step_fn(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Diagnostics + snapshots
    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
    snapshot_steps = {int(f * n_steps): f for f in snapshot_fracs}
    snapshots = {0: state}
    diag_interval = max(1, n_steps // 50)

    print(f"\n{'Step':>6s}  {'Day':>6s}  {'SSH_min':>10s}  {'SSH_max':>10s}  "
          f"{'|u|_max':>10s}  {'KE':>12s}")
    print("-" * 70)

    t_start = time.time()
    for step in range(1, n_steps + 1):
        state = step_fn(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:6.2f}  {diag['SSH_min']:10.4f}  "
                  f"{diag['SSH_max']:10.4f}  {diag['u_max']:10.4f}  "
                  f"{diag['kinetic_energy']:12.4e}")

        if step in snapshot_steps:
            snapshots[step] = state

    total_time = time.time() - t_start
    steps_per_s = n_steps / max(total_time, 1.0e-12)
    print(f"\nCompleted in {total_time:.1f}s ({steps_per_s:.0f} steps/s)")

    # --- Visualization ---
    os.makedirs(f"{output_dir}/gravity_wave", exist_ok=True)
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    if HAS_CARTOPY:
        # SSH snapshots (4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                                  subplot_kw={"projection": _idealized_projection(central_longitude=180)})
        axes = axes.ravel()

        # Find global SSH range across all snapshots
        all_ssh = [np.asarray(s.eta.data) for s in snapshots.values()]
        ssh_max = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
        ssh_max = max(ssh_max, 0.01)

        for idx, (step_num, snap) in enumerate(sorted(snapshots.items())):
            if idx >= 4:
                break
            day = step_num * dt / 86400.0
            ax = axes[idx]
            ssh_data = np.asarray(snap.eta.data)
            sc = scatter_field(ax, lon_deg, lat_deg, ssh_data, "RdBu_r",
                               -ssh_max, ssh_max, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SSH [m] — Day {day:.2f}", fontsize=13, fontweight="bold")

        fig.suptitle("Barotropic Gravity Wave — Sea Surface Height [m]",
                     fontsize=15, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, pad=0.02, orientation="vertical",
                     label="SSH [m]")
        plt.savefig(f"{output_dir}/gravity_wave/ssh_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Velocity field at final step
        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  subplot_kw={"projection": _idealized_projection(central_longitude=180)})
        for ax, (data, title, cmap) in zip(axes, [
            (np.asarray(state.u.data[..., 0]), "Surface u [m/s]", "RdBu_r"),
            (np.asarray(state.v.data[..., 0]), "Surface v [m/s]", "RdBu_r"),
        ]):
            vabs = max(float(np.nanmax(np.abs(data))), 1e-10)
            sc = scatter_field(ax, lon_deg, lat_deg, data, cmap, -vabs, vabs,
                               point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(title, fontsize=13, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical", pad=0.02, label=title)
        fig.suptitle("Barotropic Gravity Wave — Surface Velocity (Final)",
                     fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/gravity_wave/velocity_final.png", dpi=150, bbox_inches="tight")
        plt.close()

    _plot_latlon_pixel_snapshots(
        f"{output_dir}/gravity_wave/ssh_snapshots_latlon_pixels.png",
        snapshots,
        dt,
        grid,
        field_getter=lambda s: s.eta.data,
        title="Barotropic Gravity Wave — SSH Snapshots (lat-lon pixels)",
        cmap="RdBu_r",
        cbar_label="Sea surface height eta [m]",
        symmetric=True,
    )
    _plot_latlon_pixel_two_panel(
        f"{output_dir}/gravity_wave/velocity_final_latlon_pixels.png",
        state,
        state,
        "Surface u [m/s]",
        "Surface v [m/s]",
        grid,
        field_getter=lambda s: s.u.data[..., 0],
        field_getter_b=lambda s: s.v.data[..., 0],
        title="Barotropic Gravity Wave — Surface Velocity (lat-lon pixels)",
        cmap="RdBu_r",
        cbar_label="Velocity [m/s]",
        symmetric=True,
    )

    # SSH time series
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    axes[0].plot(times_days, [d["SSH_max"] for d in diagnostics], "b-", label="SSH max")
    axes[0].plot(times_days, [d["SSH_min"] for d in diagnostics], "r-", label="SSH min")
    axes[0].set_ylabel("SSH [m]")
    axes[0].set_title("SSH Extremes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times_days, [d["kinetic_energy"] for d in diagnostics], "m-", linewidth=1.5)
    axes[1].set_ylabel("KE [J/m]")
    axes[1].set_xlabel("Time [days]")
    axes[1].set_title("Kinetic Energy")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    fig.suptitle("Barotropic Gravity Wave — Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gravity_wave/diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Latitude-depth sections at final time
    lat_bins = _default_lat_bins(grid, state.land_mask.data)
    lat_centers, speed_sec = _compute_lat_depth_section(
        jnp.sqrt(state.u.data**2 + state.v.data**2),
        grid,
        state.land_mask.data,
        lat_bins,
    )
    _, T_anom_sec = _compute_lat_depth_section(
        state.T.data - state_init.T.data,
        grid,
        state.land_mask.data,
        lat_bins,
    )
    _plot_lat_depth_sections(
        f"{output_dir}/gravity_wave/lat_depth_sections.png",
        z_coord,
        lat_centers,
        [
            ("Final speed", speed_sec, "magma", False, "Speed [m/s]"),
            ("Final T anomaly", T_anom_sec, "RdBu_r", True, "Delta T [degC]"),
        ],
        "Barotropic Gravity Wave — Latitude-Depth Sections",
    )
    lon_bins = _default_lon_bins(grid, state.land_mask.data)
    lon_centers, speed_lon_sec = _compute_lon_depth_section(
        jnp.sqrt(state.u.data**2 + state.v.data**2),
        grid,
        state.land_mask.data,
        lon_bins,
    )
    _, T_anom_lon_sec = _compute_lon_depth_section(
        state.T.data - state_init.T.data,
        grid,
        state.land_mask.data,
        lon_bins,
    )
    _plot_lon_depth_sections(
        f"{output_dir}/gravity_wave/lon_depth_sections.png",
        z_coord,
        lon_centers,
        [
            ("Final speed", speed_lon_sec, "magma", False, "Speed [m/s]"),
            ("Final T anomaly", T_anom_lon_sec, "RdBu_r", True, "Delta T [degC]"),
        ],
        "Barotropic Gravity Wave — Longitude-Depth Sections",
    )

    mid = grid.n // 2
    _plot_point_profile_evolution(
        f"{output_dir}/gravity_wave/speed_profile_over_time.png",
        z_coord,
        snapshots,
        dt,
        lambda snap: np.sqrt(
            np.asarray(snap.u.data[0, mid, mid, :], dtype=np.float64) ** 2
            + np.asarray(snap.v.data[0, mid, mid, :], dtype=np.float64) ** 2
        ),
        "Speed [m/s]",
        "Vertical Speed Profile Evolution (mid-ocean point)",
    )

    _save_case_timeseries(
        f"{output_dir}/gravity_wave",
        diagnostics,
        dt,
        n_steps,
        state,
        grid,
        "Barotropic Gravity Wave",
    )

    return state, diagnostics


# =====================================================================
# Test Case 3: Wind-driven gyre
# =====================================================================

def run_wind_driven_gyre_test(
    grid,
    z_coord,
    config,
    dt,
    n_steps,
    output_dir,
    point_size,
    discretization="centered",
):
    """Wind-driven double gyre with idealized zonal wind stress."""
    print("\n" + "=" * 70)
    print("TEST 3: Wind-Driven Gyre")
    print("=" * 70)

    # Use depth-uniform tracers for a predominantly barotropic gyre harness.
    state = _rest_state_all_ocean(grid, z_coord, T_surface=15.0, T_deep=15.0)
    _assert_all_ocean(state, "wind_driven_gyre_test")
    state_init = state

    # Keep forcing moderate at coarse resolution so the 5-day standardized
    # harness remains in a stable, interpretable regime with dt=1h.
    tau_max = 0.01  # N/m^2
    drag_timescale_days = 1.5  # linear Rayleigh damping (Stommel-style)
    drag_factor = float(jnp.exp(-dt / (drag_timescale_days * 86400.0)))
    gyre_config = config._replace(
        A_h=max(config.A_h, 1.0e7),
        K_h=max(config.K_h, 1.0e6),
        A_v=max(config.A_v, 1.0e-2),
        K_v=max(config.K_v, 1.0e-3),
        n_barotropic_substeps=max(config.n_barotropic_substeps, 30),
        barotropic_diffusion_alpha=max(config.barotropic_diffusion_alpha, 0.05),
    )
    print(f"  Wind stress: tau_max = {tau_max} N/m^2")
    print(f"  Wind pattern: -tau_max * cos(2*pi*(lat-45)/60), 15N-75N")
    print(f"  Linear drag timescale: {drag_timescale_days:.1f} days")
    print(
        "  Gyre mixing: "
        f"A_h={gyre_config.A_h:.1e}, K_h={gyre_config.K_h:.1e}, "
        f"A_v={gyre_config.A_v:.1e}, K_v={gyre_config.K_v:.1e}",
    )

    model = OceanModel(grid, z_coord, gyre_config, discretization=discretization)
    step_fn = model.step_checked if gyre_config.enable_runtime_checks else model.step

    # JIT warmup
    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    _ = step_fn(state, dt)
    jax.block_until_ready(_.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    # Diagnostics + snapshots
    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
    snapshot_steps = {int(f * n_steps): f for f in snapshot_fracs}
    snapshots = {0: state}
    diag_interval = max(1, n_steps // 50)

    print(f"\n{'Step':>6s}  {'Day':>6s}  {'SSH_max':>10s}  {'|u|_max':>10s}  "
          f"{'SST_mean':>10s}  {'KE':>12s}")
    print("-" * 70)

    t_start = time.time()
    for step in range(1, n_steps + 1):
        # Apply wind forcing to surface layer velocity
        du_wind = add_wind_stress_tendency(state, grid, z_coord, gyre_config, tau_max)
        state = state._replace(
            u=state.u.replace(data=(state.u.data + dt * du_wind) * drag_factor),
            v=state.v.replace(data=state.v.data * drag_factor),
        )

        # Model step (dynamics + barotropic + conservation)
        state = step_fn(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:6.1f}  {diag['SSH_max']:10.4f}  "
                  f"{diag['u_max']:10.4f}  {diag['SST_mean']:10.4f}  "
                  f"{diag['kinetic_energy']:12.4e}")

        if step in snapshot_steps:
            snapshots[step] = state

    total_time = time.time() - t_start
    steps_per_s = n_steps / max(total_time, 1.0e-12)
    print(f"\nCompleted in {total_time:.1f}s ({steps_per_s:.0f} steps/s)")

    # --- Visualization ---
    os.makedirs(f"{output_dir}/wind_gyre", exist_ok=True)
    lon_deg = np.asarray(grid.lon) * 180 / np.pi
    lat_deg = np.asarray(grid.lat) * 180 / np.pi

    if HAS_CARTOPY:
        # SSH snapshots (4-panel)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                                  subplot_kw={"projection": _idealized_projection()})
        axes = axes.ravel()

        all_ssh = [np.asarray(s.eta.data) for s in snapshots.values()]
        ssh_abs = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
        ssh_abs = max(ssh_abs, 0.001)

        for idx, (step_num, snap) in enumerate(sorted(snapshots.items())):
            if idx >= 4:
                break
            day = step_num * dt / 86400.0
            ax = axes[idx]
            ssh_data = np.asarray(snap.eta.data)
            sc = scatter_field(ax, lon_deg, lat_deg, ssh_data, "RdBu_r",
                               -ssh_abs, ssh_abs, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SSH [m] — Day {day:.0f}", fontsize=13, fontweight="bold")

        fig.suptitle("Wind-Driven Gyre — Sea Surface Height [m]",
                     fontsize=15, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, pad=0.02, orientation="vertical",
                     label="SSH [m]")
        plt.savefig(f"{output_dir}/wind_gyre/ssh_snapshots.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Surface velocity + speed at final time
        fig, axes = plt.subplots(1, 3, figsize=(22, 5),
                                  subplot_kw={"projection": _idealized_projection()})

        u_surf = np.asarray(state.u.data[..., 0])
        v_surf = np.asarray(state.v.data[..., 0])
        speed = np.sqrt(u_surf**2 + v_surf**2)

        for ax, (data, title, cmap, sym) in zip(axes, [
            (u_surf, "Surface u [m/s]", "RdBu_r", True),
            (v_surf, "Surface v [m/s]", "RdBu_r", True),
            (speed, "Surface speed [m/s]", "magma", False),
        ]):
            if sym:
                vabs = max(float(np.nanmax(np.abs(data))), 1e-10)
                vmin, vmax = -vabs, vabs
            else:
                vmin, vmax = 0, max(float(np.nanmax(data)), 1e-10)
            sc = scatter_field(ax, lon_deg, lat_deg, data, cmap, vmin, vmax,
                               point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(title, fontsize=12, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical", pad=0.02, label=title)

        fig.suptitle("Wind-Driven Gyre — Surface Velocity (Final)",
                     fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/wind_gyre/velocity_final.png", dpi=150, bbox_inches="tight")
        plt.close()

        # SST evolution
        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                  subplot_kw={"projection": _idealized_projection()})
        for ax, (label, snap) in zip(axes, [("Initial", state_init), ("Final", state)]):
            sst = np.asarray(snap.T.data[..., 0])
            sc = scatter_field(ax, lon_deg, lat_deg, sst, "RdYlBu_r",
                               vmin=0, vmax=22, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SST — {label}", fontsize=13, fontweight="bold")
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                     label="Temperature [degC]", pad=0.02)
        fig.suptitle("Wind-Driven Gyre — Sea Surface Temperature",
                     fontsize=15, fontweight="bold")
        plt.savefig(f"{output_dir}/wind_gyre/sst.png", dpi=150, bbox_inches="tight")
        plt.close()

    _plot_latlon_pixel_snapshots(
        f"{output_dir}/wind_gyre/ssh_snapshots_latlon_pixels.png",
        snapshots,
        dt,
        grid,
        field_getter=lambda s: s.eta.data,
        title="Wind-Driven Gyre — SSH Snapshots (lat-lon pixels)",
        cmap="RdBu_r",
        cbar_label="Sea surface height eta [m]",
        symmetric=True,
    )
    _plot_latlon_pixel_two_panel(
        f"{output_dir}/wind_gyre/velocity_final_latlon_pixels.png",
        state,
        state,
        "Surface u [m/s]",
        "Surface v [m/s]",
        grid,
        field_getter=lambda s: s.u.data[..., 0],
        field_getter_b=lambda s: s.v.data[..., 0],
        title="Wind-Driven Gyre — Surface Velocity (lat-lon pixels)",
        cmap="RdBu_r",
        cbar_label="Velocity [m/s]",
        symmetric=True,
    )
    _plot_latlon_pixel_two_panel(
        f"{output_dir}/wind_gyre/sst_initial_final_latlon_pixels.png",
        state_init,
        state,
        "Initial SST [degC]",
        "Final SST [degC]",
        grid,
        field_getter=lambda s: s.T.data[..., 0],
        title="Wind-Driven Gyre — SST (lat-lon pixels)",
        cmap="RdYlBu_r",
        cbar_label="Sea-surface temperature [degC]",
        symmetric=False,
    )

    # Diagnostics time series
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(times_days, [d["SSH_max"] for d in diagnostics], "b-", linewidth=1.5)
    axes[0, 0].set_ylabel("SSH max [m]")
    axes[0, 0].set_title("Maximum SSH")
    axes[0, 0].grid(True, alpha=0.3)

    ke = np.array([d["kinetic_energy"] for d in diagnostics])
    axes[0, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[0, 1].set_ylabel("KE [J/m]")
    axes[0, 1].set_title("Kinetic Energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    axes[1, 0].plot(times_days, [d["u_max"] for d in diagnostics], "r-", linewidth=1.5)
    axes[1, 0].set_ylabel("|u| max [m/s]")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_title("Maximum Velocity")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times_days, [d["SST_mean"] for d in diagnostics], "g-", linewidth=1.5)
    axes[1, 1].set_ylabel("Mean SST [degC]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Mean Sea Surface Temperature")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Wind-Driven Gyre — Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wind_gyre/diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Vertical temperature profile at final time
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    z_ref = np.asarray(z_coord.z_full_ref)
    mid = grid.n // 2
    T_init = np.asarray(state_init.T.data[0, mid, mid, :])
    T_final = np.asarray(state.T.data[0, mid, mid, :])
    ax.plot(T_init, z_ref, "b-o", markersize=3, label="Initial", linewidth=1.5)
    ax.plot(T_final, z_ref, "r--s", markersize=3, label="Final", linewidth=1.5)
    ax.set_xlabel("Temperature [degC]", fontsize=12)
    ax.set_ylabel("Depth [m]", fontsize=12)
    ax.set_title("Vertical T Profile (mid-latitude point)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wind_gyre/T_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Latitude-depth sections at final time
    lat_bins = _default_lat_bins(grid, state.land_mask.data)
    lat_centers, u_sec = _compute_lat_depth_section(
        state.u.data,
        grid,
        state.land_mask.data,
        lat_bins,
    )
    _, speed_sec = _compute_lat_depth_section(
        jnp.sqrt(state.u.data**2 + state.v.data**2),
        grid,
        state.land_mask.data,
        lat_bins,
    )
    _plot_lat_depth_sections(
        f"{output_dir}/wind_gyre/lat_depth_sections.png",
        z_coord,
        lat_centers,
        [
            ("Final zonal velocity u", u_sec, "RdBu_r", True, "u [m/s]"),
            ("Final speed", speed_sec, "magma", False, "Speed [m/s]"),
        ],
        "Wind-Driven Gyre — Latitude-Depth Sections",
    )
    lon_bins = _default_lon_bins(grid, state.land_mask.data)
    lon_centers, u_lon_sec = _compute_lon_depth_section(
        state.u.data,
        grid,
        state.land_mask.data,
        lon_bins,
    )
    _, speed_lon_sec = _compute_lon_depth_section(
        jnp.sqrt(state.u.data**2 + state.v.data**2),
        grid,
        state.land_mask.data,
        lon_bins,
    )
    _plot_lon_depth_sections(
        f"{output_dir}/wind_gyre/lon_depth_sections.png",
        z_coord,
        lon_centers,
        [
            ("Final zonal velocity u", u_lon_sec, "RdBu_r", True, "u [m/s]"),
            ("Final speed", speed_lon_sec, "magma", False, "Speed [m/s]"),
        ],
        "Wind-Driven Gyre — Longitude-Depth Sections",
    )

    mid = grid.n // 2
    _plot_point_profile_evolution(
        f"{output_dir}/wind_gyre/T_profile_over_time.png",
        z_coord,
        snapshots,
        dt,
        lambda snap: np.asarray(snap.T.data[0, mid, mid, :]),
        "Temperature [degC]",
        "Vertical Temperature Profile Evolution (mid-latitude point)",
    )
    _plot_point_profile_evolution(
        f"{output_dir}/wind_gyre/speed_profile_over_time.png",
        z_coord,
        snapshots,
        dt,
        lambda snap: np.sqrt(
            np.asarray(snap.u.data[0, mid, mid, :], dtype=np.float64) ** 2
            + np.asarray(snap.v.data[0, mid, mid, :], dtype=np.float64) ** 2
        ),
        "Speed [m/s]",
        "Vertical Speed Profile Evolution (mid-latitude point)",
    )

    _save_case_timeseries(
        f"{output_dir}/wind_gyre",
        diagnostics,
        dt,
        n_steps,
        state,
        grid,
        "Wind-Driven Gyre",
    )

    return state, diagnostics, gyre_config


# =====================================================================
# Additional idealized tests requested by oceanography guidance
# =====================================================================

def _great_circle_distance_rad(lon_rad, lat_rad, lon0_rad, lat0_rad):
    """Great-circle angular distance [rad]."""
    dlon = lon_rad - lon0_rad
    return jnp.arccos(
        jnp.clip(
            jnp.sin(lat_rad) * jnp.sin(lat0_rad)
            + jnp.cos(lat_rad) * jnp.cos(lat0_rad) * jnp.cos(dlon),
            -1.0,
            1.0,
        ),
    )


def _replace_static_ocean_fields(state, H_bathy=None, land_mask=None):
    """Replace static ocean fields and keep masked prognostic fields consistent."""
    H_new = state.H_bathy.data if H_bathy is None else H_bathy
    mask_new = state.land_mask.data if land_mask is None else land_mask
    H_new = H_new.astype(state.H_bathy.data.dtype)
    mask_new = mask_new.astype(state.land_mask.data.dtype)
    mask_3d = mask_new[..., jnp.newaxis]
    return state._replace(
        H_bathy=state.H_bathy.replace(data=H_new),
        land_mask=state.land_mask.replace(data=mask_new),
        eta=state.eta.replace(data=state.eta.data * mask_new),
        u=state.u.replace(data=state.u.data * mask_3d),
        v=state.v.replace(data=state.v.data * mask_3d),
    )


def _pick_representative_ocean_point(mask_2d):
    """Pick a representative ocean column near domain center for profile diagnostics."""
    mask = np.asarray(mask_2d)
    wet = np.argwhere(mask > 0.5)
    if wet.size == 0:
        return (0, 0, 0)
    center = np.array([2.5, mask.shape[1] / 2.0, mask.shape[2] / 2.0], dtype=np.float64)
    d2 = np.sum((wet.astype(np.float64) - center[None, :]) ** 2, axis=1)
    sel = wet[int(np.argmin(d2))]
    return int(sel[0]), int(sel[1]), int(sel[2])


def _plot_scalar_map(
    output_path,
    title,
    data,
    grid,
    point_size,
    cmap="viridis",
    vmin=None,
    vmax=None,
    symmetric=False,
    cbar_label="",
    central_longitude=0.0,
):
    """Plot a single scalar map on cubed-sphere points."""
    if not HAS_CARTOPY:
        return
    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return
    if symmetric:
        vabs = float(np.max(np.abs(finite)))
        vabs = max(vabs, 1.0e-12)
        vmin_auto, vmax_auto = -vabs, vabs
    else:
        vmin_auto = float(np.min(finite))
        vmax_auto = float(np.max(finite))
        if abs(vmax_auto - vmin_auto) < 1.0e-12:
            vmax_auto = vmin_auto + 1.0e-12

    if vmin is None:
        vmin = vmin_auto
    if vmax is None:
        vmax = vmax_auto

    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(14, 7),
        subplot_kw={"projection": _idealized_projection(central_longitude=central_longitude)},
    )
    sc = scatter_field(ax, lon_deg, lat_deg, arr, cmap, vmin, vmax, point_size=point_size)
    _style_idealized_axes(ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical", label=cbar_label, pad=0.02)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_standard_case_outputs(
    case_dir,
    case_title,
    state_init,
    state_final,
    snapshots,
    diagnostics,
    grid,
    z_coord,
    dt,
    n_steps,
    point_size,
    central_longitude=0.0,
):
    """Standardized snapshots + diagnostics for idealized ocean cases."""
    lon_deg = np.asarray(grid.lon) * 180.0 / np.pi
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi

    if HAS_CARTOPY:
        # SSH snapshots
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(16, 10),
            subplot_kw={"projection": _idealized_projection(central_longitude=central_longitude)},
        )
        axes = axes.ravel()
        all_ssh = [np.asarray(s.eta.data) for s in snapshots.values()]
        ssh_abs = max(float(np.nanmax(np.abs(a))) for a in all_ssh)
        ssh_abs = max(ssh_abs, 1.0e-3)
        sc = None
        for idx, (step_num, snap) in enumerate(sorted(snapshots.items())):
            if idx >= 4:
                break
            day = step_num * dt / 86400.0
            ax = axes[idx]
            sc = scatter_field(
                ax,
                lon_deg,
                lat_deg,
                np.asarray(snap.eta.data),
                "RdBu_r",
                -ssh_abs,
                ssh_abs,
                point_size=point_size,
            )
            _style_idealized_axes(ax)
            ax.set_title(
                f"Sea surface height eta [m]\nstep {step_num}, t={day:.2f} d",
                fontsize=12,
                fontweight="bold",
            )
        if sc is not None:
            fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                         label="Sea surface height eta [m]", pad=0.02)
        fig.suptitle(f"{case_title} — SSH Snapshots", fontsize=15, fontweight="bold")
        plt.savefig(os.path.join(case_dir, "ssh_snapshots.png"), dpi=150, bbox_inches="tight")
        plt.close()
        _plot_latlon_pixel_snapshots(
            os.path.join(case_dir, "ssh_snapshots_latlon_pixels.png"),
            snapshots,
            dt,
            grid,
            field_getter=lambda s: s.eta.data,
            title=f"{case_title} — SSH Snapshots (lat-lon pixels)",
            cmap="RdBu_r",
            cbar_label="Sea surface height eta [m]",
            symmetric=True,
        )

        # Final velocity maps
        u_surf = np.asarray(state_final.u.data[..., 0])
        v_surf = np.asarray(state_final.v.data[..., 0])
        speed = np.sqrt(u_surf**2 + v_surf**2)
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(22, 5),
            subplot_kw={"projection": _idealized_projection(central_longitude=central_longitude)},
        )
        for ax, (data, title, cmap, symmetric) in zip(axes, [
            (u_surf, "Surface zonal velocity u [m/s]", "RdBu_r", True),
            (v_surf, "Surface meridional velocity v [m/s]", "RdBu_r", True),
            (speed, "Surface speed [m/s]", "magma", False),
        ]):
            if symmetric:
                vmax = max(float(np.nanmax(np.abs(data))), 1.0e-10)
                vmin = -vmax
            else:
                vmin = 0.0
                vmax = max(float(np.nanmax(data)), 1.0e-10)
            sc = scatter_field(ax, lon_deg, lat_deg, data, cmap, vmin, vmax, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(title, fontsize=12, fontweight="bold")
            fig.colorbar(sc, ax=ax, shrink=0.78, orientation="vertical", label=title, pad=0.02)
        fig.suptitle(f"{case_title} — Surface Velocity (Final)", fontsize=15, fontweight="bold")
        plt.savefig(os.path.join(case_dir, "velocity_final.png"), dpi=150, bbox_inches="tight")
        plt.close()
        _plot_latlon_pixel_two_panel(
            os.path.join(case_dir, "velocity_final_latlon_pixels.png"),
            state_final,
            state_final,
            "Surface u [m/s]",
            "Surface v [m/s]",
            grid,
            field_getter=lambda s: s.u.data[..., 0],
            field_getter_b=lambda s: s.v.data[..., 0],
            title=f"{case_title} — Surface Velocity (lat-lon pixels)",
            cmap="RdBu_r",
            cbar_label="Velocity [m/s]",
            symmetric=True,
        )

        # SST initial/final
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(16, 5),
            subplot_kw={"projection": _idealized_projection(central_longitude=central_longitude)},
        )
        sc = None
        for ax, (label, snap) in zip(axes, [("Initial", state_init), ("Final", state_final)]):
            sst = np.asarray(snap.T.data[..., 0])
            vmin = float(np.nanmin(sst))
            vmax = float(np.nanmax(sst))
            if abs(vmax - vmin) < 1.0e-12:
                vmax = vmin + 1.0e-12
            sc = scatter_field(ax, lon_deg, lat_deg, sst, "RdYlBu_r", vmin, vmax, point_size=point_size)
            _style_idealized_axes(ax)
            ax.set_title(f"SST [degC] — {label}", fontsize=13, fontweight="bold")
        if sc is not None:
            fig.colorbar(sc, ax=axes.tolist(), shrink=0.78, orientation="vertical",
                         label="Sea-surface temperature [degC]", pad=0.02)
        fig.suptitle(f"{case_title} — Sea Surface Temperature", fontsize=15, fontweight="bold")
        plt.savefig(os.path.join(case_dir, "sst_initial_final.png"), dpi=150, bbox_inches="tight")
        plt.close()
        _plot_latlon_pixel_two_panel(
            os.path.join(case_dir, "sst_initial_final_latlon_pixels.png"),
            state_init,
            state_final,
            "Initial SST [degC]",
            "Final SST [degC]",
            grid,
            field_getter=lambda s: s.T.data[..., 0],
            title=f"{case_title} — SST (lat-lon pixels)",
            cmap="RdYlBu_r",
            cbar_label="SST [degC]",
            symmetric=False,
        )

    # Diagnostics time series
    n_diag = len(diagnostics)
    times_days = np.linspace(0, n_steps * dt / 86400.0, n_diag)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(times_days, [d["SSH_max"] for d in diagnostics], "b-", linewidth=1.5, label="SSH max")
    axes[0, 0].plot(times_days, [d["SSH_min"] for d in diagnostics], "c-", linewidth=1.5, label="SSH min")
    axes[0, 0].set_ylabel("SSH [m]")
    axes[0, 0].set_title("Sea Surface Height Extremes")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    ke = np.array([d["kinetic_energy"] for d in diagnostics], dtype=np.float64)
    axes[0, 1].plot(times_days, ke, "m-", linewidth=1.5)
    axes[0, 1].set_ylabel("Kinetic energy")
    axes[0, 1].set_title("Kinetic Energy")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))

    axes[1, 0].plot(times_days, [d["u_max"] for d in diagnostics], "r-", linewidth=1.5, label="|u| max")
    axes[1, 0].plot(times_days, [d["v_max"] for d in diagnostics], "orange", linewidth=1.5, label="|v| max")
    axes[1, 0].set_ylabel("Velocity [m/s]")
    axes[1, 0].set_xlabel("Time [days]")
    axes[1, 0].set_title("Maximum Velocity Components")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times_days, [d["SST_mean"] for d in diagnostics], "g-", linewidth=1.5)
    axes[1, 1].set_ylabel("Mean SST [degC]")
    axes[1, 1].set_xlabel("Time [days]")
    axes[1, 1].set_title("Mean Sea-Surface Temperature")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{case_title} — Diagnostics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "diagnostics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Latitude-depth sections
    lat_bins = _default_lat_bins(grid, state_final.land_mask.data)
    lat_centers, T_sec = _compute_lat_depth_section(state_final.T.data, grid, state_final.land_mask.data, lat_bins)
    _, speed_sec = _compute_lat_depth_section(
        jnp.sqrt(state_final.u.data**2 + state_final.v.data**2),
        grid,
        state_final.land_mask.data,
        lat_bins,
    )
    _plot_lat_depth_sections(
        os.path.join(case_dir, "lat_depth_sections.png"),
        z_coord,
        lat_centers,
        [
            ("Final temperature T", T_sec, "RdYlBu_r", False, "T [degC]"),
            ("Final speed", speed_sec, "magma", False, "Speed [m/s]"),
        ],
        f"{case_title} — Latitude-Depth Sections",
    )
    lon_bins = _default_lon_bins(grid, state_final.land_mask.data)
    lon_centers, T_lon_sec = _compute_lon_depth_section(
        state_final.T.data,
        grid,
        state_final.land_mask.data,
        lon_bins,
    )
    _, speed_lon_sec = _compute_lon_depth_section(
        jnp.sqrt(state_final.u.data**2 + state_final.v.data**2),
        grid,
        state_final.land_mask.data,
        lon_bins,
    )
    _plot_lon_depth_sections(
        os.path.join(case_dir, "lon_depth_sections.png"),
        z_coord,
        lon_centers,
        [
            ("Final temperature T", T_lon_sec, "RdYlBu_r", False, "T [degC]"),
            ("Final speed", speed_lon_sec, "magma", False, "Speed [m/s]"),
        ],
        f"{case_title} — Longitude-Depth Sections",
    )

    # Profile evolution at representative wet column
    face, i, j = _pick_representative_ocean_point(state_final.land_mask.data)
    _plot_point_profile_evolution(
        os.path.join(case_dir, "T_profile_over_time.png"),
        z_coord,
        snapshots,
        dt,
        lambda snap: np.asarray(snap.T.data[face, i, j, :], dtype=np.float64),
        "Temperature [degC]",
        f"Temperature Profile Evolution (face={face}, i={i}, j={j})",
    )
    _plot_point_profile_evolution(
        os.path.join(case_dir, "speed_profile_over_time.png"),
        z_coord,
        snapshots,
        dt,
        lambda snap: np.sqrt(
            np.asarray(snap.u.data[face, i, j, :], dtype=np.float64) ** 2
            + np.asarray(snap.v.data[face, i, j, :], dtype=np.float64) ** 2
        ),
        "Speed [m/s]",
        f"Speed Profile Evolution (face={face}, i={i}, j={j})",
    )


def _run_forced_ocean_case(
    test_label,
    case_subdir,
    case_title,
    state_init,
    grid,
    z_coord,
    case_config,
    dt,
    n_steps,
    output_dir,
    point_size,
    forcing_step_fn=None,
    central_longitude=0.0,
    extra_plot_fn=None,
    discretization="centered",
):
    """Run a generic forced idealized ocean case with standard diagnostics."""
    print("\n" + "=" * 70)
    print(test_label)
    print("=" * 70)
    print(
        "  Config: "
        f"A_h={case_config.A_h:.1e}, K_h={case_config.K_h:.1e}, "
        f"A_v={case_config.A_v:.1e}, K_v={case_config.K_v:.1e}, "
        f"n_baro={case_config.n_barotropic_substeps}",
    )

    model = OceanModel(grid, z_coord, case_config, discretization=discretization)
    step_fn = model.step_checked if case_config.enable_runtime_checks else model.step
    state = state_init

    print("  Warming up JIT...", end=" ", flush=True)
    t0 = time.time()
    warm = step_fn(state, dt)
    jax.block_until_ready(warm.eta.data)
    print(f"done ({time.time()-t0:.1f}s)")

    diagnostics = [compute_ocean_diagnostics(state, grid, z_coord)]
    snapshot_fracs = [0.0, 0.25, 0.5, 1.0]
    snapshot_steps = {int(f * n_steps): f for f in snapshot_fracs}
    snapshots = {0: state}
    diag_interval = max(1, n_steps // 50)

    print(f"\n{'Step':>6s}  {'Day':>7s}  {'SSH_max':>10s}  {'|u|_max':>10s}  "
          f"{'SST_mean':>10s}  {'KE':>12s}")
    print("-" * 74)
    t_start = time.time()
    for step in range(1, n_steps + 1):
        if forcing_step_fn is not None:
            state = forcing_step_fn(state, step, dt)
        state = step_fn(state, dt)

        if step % diag_interval == 0 or step == n_steps:
            jax.block_until_ready(state.eta.data)
            diag = compute_ocean_diagnostics(state, grid, z_coord)
            diagnostics.append(diag)
            day = step * dt / 86400.0
            print(f"{step:6d}  {day:7.2f}  {diag['SSH_max']:10.4e}  "
                  f"{diag['u_max']:10.4e}  {diag['SST_mean']:10.4f}  "
                  f"{diag['kinetic_energy']:12.4e}")

        if step in snapshot_steps:
            snapshots[step] = state

    total_time = time.time() - t_start
    steps_per_s = n_steps / max(total_time, 1.0e-12)
    print(f"\nCompleted in {total_time:.1f}s ({steps_per_s:.0f} steps/s)")

    case_dir = f"{output_dir}/{case_subdir}"
    os.makedirs(case_dir, exist_ok=True)
    _plot_standard_case_outputs(
        case_dir,
        case_title,
        state_init,
        state,
        snapshots,
        diagnostics,
        grid,
        z_coord,
        dt,
        n_steps,
        point_size,
        central_longitude=central_longitude,
    )
    _save_case_timeseries(case_dir, diagnostics, dt, n_steps, state, grid, case_title)

    if extra_plot_fn is not None:
        extra_plot_fn(case_dir, state_init, state, snapshots, diagnostics)

    return state, diagnostics, case_config


def run_adiabatic_topography_adjustment_test(
    grid, z_coord, config, dt, n_steps, output_dir, point_size, discretization="centered",
):
    """Adiabatic adjustment over topography."""
    state = _rest_state_all_ocean(
        grid,
        z_coord,
        T_surface=6.0,
        T_deep=6.0,
        S_uniform=35.0,
        H_max=4200.0,
    )
    _assert_all_ocean(state, "adiabatic_topography_test")

    bump = 1500.0 * jnp.exp(
        -0.5
        * (
            _great_circle_distance_rad(grid.lon, grid.lat, jnp.pi, 0.0)
            / (12.0 * jnp.pi / 180.0)
        ) ** 2,
    )
    H_bathy = jnp.clip(4200.0 - bump, 800.0, 4200.0)
    state = _replace_static_ocean_fields(state, H_bathy=H_bathy)

    eta0 = 0.30 * jnp.exp(
        -0.5
        * (
            _great_circle_distance_rad(grid.lon, grid.lat, jnp.pi, 0.0)
            / (16.0 * jnp.pi / 180.0)
        ) ** 2,
    )
    eta0 = eta0 * state.land_mask.data
    area_w = state.land_mask.data * grid.area
    eta0 = eta0 - jnp.sum(eta0 * area_w) / jnp.maximum(jnp.sum(area_w), 1.0)
    state = state._replace(eta=state.eta.replace(data=eta0.astype(state.eta.data.dtype)))

    case_config = config._replace(
        A_h=max(config.A_h, 5.0e4),
        K_h=max(config.K_h, 5.0e3),
        n_barotropic_substeps=max(config.n_barotropic_substeps, 40),
        barotropic_diffusion_alpha=max(config.barotropic_diffusion_alpha, 0.05),
    )

    def extra(case_dir, state_init, *_):
        _plot_scalar_map(
            os.path.join(case_dir, "bathymetry.png"),
            "Bathymetry depth H_bathy [m]",
            state_init.H_bathy.data,
            grid,
            point_size,
            cmap="terrain",
            cbar_label="Depth [m]",
        )
        _plot_scalar_map(
            os.path.join(case_dir, "eta_initial.png"),
            "Initial sea-surface height eta [m]",
            state_init.eta.data,
            grid,
            point_size,
            cmap="RdBu_r",
            symmetric=True,
            cbar_label="eta [m]",
        )

    return _run_forced_ocean_case(
        "TEST 4: Adiabatic Topographic Adjustment",
        "adiabatic_topography",
        "Adiabatic Topographic Adjustment",
        state,
        grid,
        z_coord,
        case_config,
        dt,
        n_steps,
        output_dir,
        point_size,
        forcing_step_fn=None,
        extra_plot_fn=extra,
        discretization=discretization,
    )


def run_holland_lin_double_gyre_test(
    grid, z_coord, config, dt, n_steps, output_dir, point_size, discretization="centered",
):
    """Wind-forced double gyre in a closed rectangular basin (Holland-Lin style)."""
    state = _rest_state_all_ocean(
        grid,
        z_coord,
        T_surface=15.0,
        T_deep=15.0,
        S_uniform=35.0,
        H_max=4000.0,
    )
    lat_deg = grid.lat * (180.0 / jnp.pi)
    lon_deg = (grid.lon * (180.0 / jnp.pi) + 180.0) % 360.0 - 180.0
    basin_mask = jnp.where(
        (lat_deg >= 10.0) & (lat_deg <= 70.0) & (lon_deg >= -70.0) & (lon_deg <= 20.0),
        1.0,
        0.0,
    ).astype(jnp.float32)
    if float(jnp.mean(basin_mask)) <= 0.01:
        raise ValueError("holland_lin_double_gyre_test: basin mask is empty")

    H_bathy = jnp.full_like(basin_mask, 4000.0, dtype=jnp.float32)
    state = _replace_static_ocean_fields(state, H_bathy=H_bathy, land_mask=basin_mask)

    y_norm = (lat_deg - 10.0) / 60.0
    tau_max = 0.05  # N/m^2
    tau_x = -tau_max * jnp.cos(2.0 * jnp.pi * y_norm)
    tau_x = jnp.where((y_norm >= 0.0) & (y_norm <= 1.0) & (basin_mask > 0.5), tau_x, 0.0)
    drag_factor = float(jnp.exp(-dt / (20.0 * 86400.0)))

    case_config = config._replace(
        A_h=max(config.A_h, 1.0e7),
        K_h=max(config.K_h, 1.0e6),
        A_v=max(config.A_v, 1.0e-2),
        K_v=max(config.K_v, 1.0e-3),
        n_barotropic_substeps=max(config.n_barotropic_substeps, 50),
        barotropic_diffusion_alpha=max(config.barotropic_diffusion_alpha, 0.08),
    )

    def forcing_step_fn(state_now, step, dt_now):
        del step
        h_k = compute_layer_thickness(state_now.eta.data, state_now.H_bathy.data, z_coord)
        h_surface = jnp.maximum(h_k[..., 0], 1.0)
        du_surface = tau_x / (rho_0 * h_surface)
        du_3d = jnp.zeros_like(state_now.u.data).at[..., 0].set(du_surface)
        mask_3d = state_now.land_mask.data[..., jnp.newaxis]
        u_new = (state_now.u.data + dt_now * du_3d) * drag_factor * mask_3d
        v_new = state_now.v.data * drag_factor * mask_3d
        return state_now._replace(
            u=state_now.u.replace(data=u_new),
            v=state_now.v.replace(data=v_new),
        )

    def extra(case_dir, state_init, *_):
        _plot_scalar_map(
            os.path.join(case_dir, "basin_mask.png"),
            "Closed-basin ocean mask [1=ocean]",
            state_init.land_mask.data,
            grid,
            point_size,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Mask",
        )
        _plot_scalar_map(
            os.path.join(case_dir, "wind_stress_tau_x.png"),
            "Imposed zonal wind stress tau_x [N/m^2]",
            tau_x,
            grid,
            point_size,
            cmap="RdBu_r",
            symmetric=True,
            cbar_label="tau_x [N/m^2]",
        )

    return _run_forced_ocean_case(
        "TEST 5: Holland-Lin Wind-Forced Double Gyre",
        "holland_lin_gyre",
        "Holland-Lin Double Gyre",
        state,
        grid,
        z_coord,
        case_config,
        dt,
        n_steps,
        output_dir,
        point_size,
        forcing_step_fn=forcing_step_fn,
        extra_plot_fn=extra,
        discretization=discretization,
    )


def run_diabatic_thermohaline_test(
    grid, z_coord, config, dt, n_steps, output_dir, point_size, discretization="centered",
):
    """Diabatic idealized thermohaline circulation with SST/SSS restoring."""
    state = _rest_state_all_ocean(
        grid,
        z_coord,
        T_surface=24.0,
        T_deep=1.5,
        S_uniform=34.8,
        H_max=4500.0,
    )
    _assert_all_ocean(state, "diabatic_thermohaline_test")

    lat = grid.lat
    mask = state.land_mask.data
    T_star = 24.0 - 20.0 * jnp.sin(lat) ** 2
    S_star = 34.7 + 0.6 * jnp.cos(2.0 * lat)
    tau_T = 20.0 * 86400.0
    tau_S = 45.0 * 86400.0

    case_config = config._replace(
        A_h=max(config.A_h, 5.0e4),
        K_h=max(config.K_h, 2.0e4),
        A_v=max(config.A_v, 5.0e-3),
        K_v=max(config.K_v, 5.0e-4),
        n_barotropic_substeps=max(config.n_barotropic_substeps, 35),
    )

    def forcing_step_fn(state_now, step, dt_now):
        del step
        T_data = state_now.T.data
        S_data = state_now.S.data
        dT_sfc = (-(T_data[..., 0] - T_star) / tau_T) * mask
        dS_sfc = (-(S_data[..., 0] - S_star) / tau_S) * mask
        T_new = T_data.at[..., 0].set(T_data[..., 0] + dt_now * dT_sfc)
        S_new = S_data.at[..., 0].set(S_data[..., 0] + dt_now * dS_sfc)
        return state_now._replace(
            T=state_now.T.replace(data=T_new),
            S=state_now.S.replace(data=S_new),
        )

    def extra(case_dir, *_):
        _plot_scalar_map(
            os.path.join(case_dir, "target_sst.png"),
            "Target SST restoring profile T* [degC]",
            T_star,
            grid,
            point_size,
            cmap="RdYlBu_r",
            cbar_label="T* [degC]",
        )
        _plot_scalar_map(
            os.path.join(case_dir, "target_sss.png"),
            "Target SSS restoring profile S* [PSU]",
            S_star,
            grid,
            point_size,
            cmap="viridis",
            cbar_label="S* [PSU]",
        )

    return _run_forced_ocean_case(
        "TEST 6: Diabatic Thermohaline Circulation",
        "thermohaline",
        "Diabatic Thermohaline Circulation",
        state,
        grid,
        z_coord,
        case_config,
        dt,
        n_steps,
        output_dir,
        point_size,
        forcing_step_fn=forcing_step_fn,
        extra_plot_fn=extra,
        discretization=discretization,
    )


def run_two_layer_phillips_test(
    grid, base_z_coord, config, dt, n_steps, output_dir, point_size, discretization="centered",
):
    """Two-layer Phillips-style baroclinic test with zonal-mean relaxation."""
    del base_z_coord
    z_coord_2 = create_ocean_z_star(n_levels=2)
    sim_seconds = n_steps * dt
    dt_case = min(dt, 900.0)
    n_steps_case = max(1, int(sim_seconds / dt_case))

    state = _rest_state_all_ocean(
        grid,
        z_coord_2,
        T_surface=17.0,
        T_deep=7.0,
        S_uniform=35.0,
        H_max=3500.0,
    )
    _assert_all_ocean(state, "two_layer_phillips_test")

    lat_deg = grid.lat * (180.0 / jnp.pi)
    u_jet = 0.30 * jnp.exp(-((lat_deg - 45.0) / 14.0) ** 2) * state.land_mask.data
    u_data = state.u.data.at[..., 0].set(u_jet).at[..., 1].set(-0.20 * u_jet)
    eta_seed = 0.05 * jnp.sin(3.0 * grid.lon) * jnp.cos(2.0 * grid.lat) * state.land_mask.data
    area_w = state.land_mask.data * grid.area
    eta_seed = eta_seed - jnp.sum(eta_seed * area_w) / jnp.maximum(jnp.sum(area_w), 1.0)
    state = state._replace(
        u=state.u.replace(data=u_data.astype(state.u.data.dtype)),
        eta=state.eta.replace(data=eta_seed.astype(state.eta.data.dtype)),
    )

    case_config = config._replace(
        A_h=max(config.A_h, 2.0e5),
        K_h=max(config.K_h, 2.0e4),
        A_v=max(config.A_v, 1.0e-2),
        K_v=max(config.K_v, 1.0e-3),
        n_barotropic_substeps=max(config.n_barotropic_substeps, 45),
        barotropic_diffusion_alpha=max(config.barotropic_diffusion_alpha, 0.06),
    )

    n_bins = max(24, 2 * grid.n)
    lat_edges = jnp.linspace(-90.0, 90.0, n_bins + 1)
    lat_deg_flat = (grid.lat * (180.0 / jnp.pi)).reshape(-1)
    bin_idx = jnp.clip(
        jnp.searchsorted(lat_edges, lat_deg_flat, side="right") - 1,
        0,
        n_bins - 1,
    ).astype(jnp.int32)
    weights = (grid.area * state.land_mask.data).reshape(-1)

    def lat_bin_mean(field_2d):
        flat = field_2d.reshape(-1)
        wsum = jnp.bincount(bin_idx, weights=weights, length=n_bins)
        ssum = jnp.bincount(bin_idx, weights=flat * weights, length=n_bins)
        mean_bin = ssum / jnp.maximum(wsum, 1.0e-12)
        return mean_bin[bin_idx].reshape(field_2d.shape)

    T_star_upper = 16.0 - 10.0 * jnp.sin(grid.lat) ** 2
    T_star_lower = 8.0 - 4.0 * jnp.sin(grid.lat) ** 2
    tau_relax = 15.0 * 86400.0
    drag_factor = float(jnp.exp(-dt_case / (25.0 * 86400.0)))

    def forcing_step_fn(state_now, step, dt_now):
        del step
        T_data = state_now.T.data
        mask = state_now.land_mask.data
        T0_zm = lat_bin_mean(T_data[..., 0])
        T1_zm = lat_bin_mean(T_data[..., 1])
        dT0 = (-(T0_zm - T_star_upper) / tau_relax) * mask
        dT1 = (-(T1_zm - T_star_lower) / tau_relax) * mask
        T_new = T_data.at[..., 0].set(T_data[..., 0] + dt_now * dT0)
        T_new = T_new.at[..., 1].set(T_data[..., 1] + dt_now * dT1)
        mask_3d = mask[..., jnp.newaxis]
        u_new = state_now.u.data * drag_factor * mask_3d
        v_new = state_now.v.data * drag_factor * mask_3d
        return state_now._replace(
            T=state_now.T.replace(data=T_new),
            u=state_now.u.replace(data=u_new),
            v=state_now.v.replace(data=v_new),
        )

    def extra(case_dir, *_):
        _plot_scalar_map(
            os.path.join(case_dir, "target_T_upper.png"),
            "Phillips target upper-layer T* [degC]",
            T_star_upper,
            grid,
            point_size,
            cmap="RdYlBu_r",
            cbar_label="T* upper [degC]",
        )
        _plot_scalar_map(
            os.path.join(case_dir, "target_T_lower.png"),
            "Phillips target lower-layer T* [degC]",
            T_star_lower,
            grid,
            point_size,
            cmap="RdYlBu_r",
            cbar_label="T* lower [degC]",
        )

    return _run_forced_ocean_case(
        "TEST 7: Two-Layer Phillips Baroclinic Problem",
        "phillips_two_layer",
        "Two-Layer Phillips Test",
        state,
        grid,
        z_coord_2,
        case_config,
        dt_case,
        n_steps_case,
        output_dir,
        point_size,
        forcing_step_fn=forcing_step_fn,
        extra_plot_fn=extra,
        discretization=discretization,
    )


def run_taylor_column_test(
    grid, z_coord, config, dt, n_steps, output_dir, point_size, discretization="centered",
):
    """Taylor-column style flow over a seamount in rotating stratified ocean."""
    sim_seconds = n_steps * dt
    dt_case = min(dt, 1800.0)
    n_steps_case = max(1, int(sim_seconds / dt_case))

    state = _rest_state_all_ocean(
        grid,
        z_coord,
        T_surface=8.0,
        T_deep=4.0,
        S_uniform=35.0,
        H_max=4500.0,
    )
    _assert_all_ocean(state, "taylor_column_test")

    dist = _great_circle_distance_rad(
        grid.lon,
        grid.lat,
        jnp.pi,
        30.0 * jnp.pi / 180.0,
    )
    bump = 1200.0 * jnp.exp(-0.5 * (dist / (10.0 * jnp.pi / 180.0)) ** 2)
    H_bathy = jnp.clip(4500.0 - bump, 1200.0, 4500.0)
    state = _replace_static_ocean_fields(state, H_bathy=H_bathy)

    U0 = 0.02
    u_target_2d = U0 * jnp.cos(grid.lat) * state.land_mask.data
    u_target_3d = jnp.broadcast_to(u_target_2d[..., jnp.newaxis], state.u.data.shape)
    state = state._replace(u=state.u.replace(data=u_target_3d.astype(state.u.data.dtype)))

    case_config = config._replace(
        A_h=max(config.A_h, 5.0e5),
        K_h=max(config.K_h, 5.0e4),
        A_v=max(config.A_v, 1.0e-2),
        K_v=max(config.K_v, 1.0e-3),
        n_barotropic_substeps=max(config.n_barotropic_substeps, 60),
        barotropic_diffusion_alpha=max(config.barotropic_diffusion_alpha, 0.10),
    )
    tau_restore = 30.0 * 86400.0

    def forcing_step_fn(state_now, step, dt_now):
        del step
        mask = state_now.land_mask.data
        mask_3d = mask[..., jnp.newaxis]
        u_target = jnp.broadcast_to(u_target_2d[..., jnp.newaxis], state_now.u.data.shape)
        u_new = (state_now.u.data + dt_now * (u_target - state_now.u.data) / tau_restore) * mask_3d
        v_new = state_now.v.data * jnp.exp(-dt_now / tau_restore) * mask_3d
        return state_now._replace(
            u=state_now.u.replace(data=u_new),
            v=state_now.v.replace(data=v_new),
        )

    def extra(case_dir, *_):
        _plot_scalar_map(
            os.path.join(case_dir, "bathymetry.png"),
            "Taylor-column bathymetry H_bathy [m]",
            H_bathy,
            grid,
            point_size,
            cmap="terrain",
            cbar_label="Depth [m]",
        )
        _plot_scalar_map(
            os.path.join(case_dir, "u_target.png"),
            "Background zonal flow target u_target [m/s]",
            u_target_2d,
            grid,
            point_size,
            cmap="RdBu_r",
            symmetric=True,
            cbar_label="u_target [m/s]",
        )

    return _run_forced_ocean_case(
        "TEST 8: Taylor Column over Seamount",
        "taylor_column",
        "Taylor Column Test",
        state,
        grid,
        z_coord,
        case_config,
        dt_case,
        n_steps_case,
        output_dir,
        point_size,
        forcing_step_fn=forcing_step_fn,
        extra_plot_fn=extra,
        discretization=discretization,
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Ocean model standardized tests")
    parser.add_argument("--x64", action="store_true",
                        help="Enable float64 mode (default: float32)")
    parser.add_argument("--resolution", "-n", type=int, default=8,
                        help="Cubed-sphere resolution (default: 8)")
    parser.add_argument("--levels", "-l", type=int, default=10,
                        help="Number of vertical levels (default: 10)")
    parser.add_argument("--dt", type=float, default=3600.0,
                        help="Time step in seconds (default: 3600)")
    parser.add_argument("--days", "-d", type=float, default=5.0,
                        help="Integration time in days (default: 5)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: results/ocean/ocean_tests_C{n}_L{l})")
    parser.add_argument("--test", "-t", type=str, default="all",
                        choices=[
                            "all",
                            "rest",
                            "wave",
                            "gyre",
                            "adiabatic_topography",
                            "holland_lin",
                            "thermohaline",
                            "phillips",
                            "taylor",
                        ],
                        help="Which test to run (default: all)")
    parser.add_argument("--runtime-checks", action="store_true",
                        help="Enable host-side runtime invariant checks")
    parser.add_argument(
        "--discretization",
        type=str,
        default="centered",
        choices=OCEAN_DISCRETIZATIONS,
        help="Ocean horizontal discretization (default: centered).",
    )
    args = parser.parse_args()

    if args.resolution < 1:
        raise ValueError(f"--resolution must be >= 1, got {args.resolution!r}")
    if args.levels < 1:
        raise ValueError(f"--levels must be >= 1, got {args.levels!r}")
    if args.dt <= 0.0:
        raise ValueError(f"--dt must be > 0, got {args.dt!r}")
    if args.days <= 0.0:
        raise ValueError(f"--days must be > 0, got {args.days!r}")

    jax.config.update("jax_enable_x64", bool(args.x64))

    output_dir = args.output or f"results/ocean/ocean_tests_C{args.resolution}_L{args.levels}"
    os.makedirs(output_dir, exist_ok=True)

    n_steps = int(args.days * 86400 / args.dt)
    if n_steps < 1:
        raise ValueError(
            "Integration has zero steps. Increase --days or reduce --dt "
            f"(got days={args.days}, dt={args.dt}).",
        )
    point_size = max(1.0, 120 / args.resolution)

    # Banner
    print("=" * 70)
    print("legoESM — Ocean Model Standardized Tests")
    print("=" * 70)
    print(f"  Backend:      {jax.default_backend()}")
    print(f"  Devices:      {jax.device_count()}")
    print(f"  Float dtype:  {jnp.zeros(1).dtype}")
    print(f"  Discretization: {args.discretization}")

    # Grid setup
    print(f"\nCreating C{args.resolution} cubed-sphere grid...")
    t0 = time.time()
    grid = create_cubed_sphere(args.resolution)
    print(f"  Grid created in {time.time()-t0:.1f}s")
    print(f"  Resolution: ~{grid.resolution_km:.0f} km")
    print(f"  Total cells: {grid.n_cells:,}")

    # Vertical coordinate
    print(f"\nCreating ocean z-star coordinate ({args.levels} levels)...")
    z_coord = create_ocean_z_star(n_levels=args.levels)
    print(f"  Surface dz: {float(z_coord.dz_ref[0]):.1f} m")
    print(f"  Bottom dz:  {float(z_coord.dz_ref[-1]):.1f} m")
    print(f"  Total depth: {float(z_coord.H_max):.0f} m")

    # Hyperdiffusion: scale with grid spacing
    # Ocean uses weaker hyperdiffusion than atmosphere (slower velocities)
    mean_dx = float(jnp.mean(grid.dx))
    hyperdiff_coeff = 1e-6 * mean_dx**4 / args.dt
    print(f"  Mean dx: {mean_dx/1000:.0f} km")
    print(f"  Hyperdiffusion coeff: {hyperdiff_coeff:.2e}")

    # Barotropic substeps: CFL for barotropic gravity waves
    # c = sqrt(gH) ~ 230 m/s, need dt_sub < dx / c
    c_baro = float(jnp.sqrt(9.81 * 5500.0))
    n_baro = max(10, int(2.0 * c_baro * args.dt / mean_dx) + 1)
    print(f"  Barotropic wave speed: {c_baro:.0f} m/s")
    print(f"  Barotropic substeps: {n_baro}")

    config = OceanConfig(
        hyperdiff_coeff=hyperdiff_coeff,
        n_barotropic_substeps=n_baro,
        use_conservation_fixer=True,
        fix_volume=True,
        fix_heat=True,
        fix_salt=True,
        enable_runtime_checks=args.runtime_checks,
    )

    print(f"\nIntegration: {args.days} days ({n_steps} steps, dt={args.dt:.0f}s)")

    # =====================================================================
    # Run tests
    # =====================================================================
    results = {}
    case_configs = {}
    case_errors = {}

    def _run_case(case_name: str, fn) -> None:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 3:
                state, diag, case_cfg = out
            else:
                state, diag = out
                case_cfg = config
            results[case_name] = {
                "metrics": summarize_case_metrics(state, diag, grid),
            }
            case_configs[case_name] = _config_to_dict(case_cfg)
        except Exception as exc:  # pragma: no cover - exercised in runtime matrix failures
            msg = f"{type(exc).__name__}: {exc}"
            case_errors[case_name] = msg
            print(f"\n  !!! CASE FAILED: {case_name} -> {msg}")
            traceback.print_exc()

    if args.test in ("all", "rest"):
        _run_case(
            "rest_state",
            lambda: run_rest_state_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "wave"):
        _run_case(
            "gravity_wave",
            lambda: run_gravity_wave_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "gyre"):
        _run_case(
            "wind_gyre",
            lambda: run_wind_driven_gyre_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "adiabatic_topography"):
        _run_case(
            "adiabatic_topography",
            lambda: run_adiabatic_topography_adjustment_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "holland_lin"):
        _run_case(
            "holland_lin_gyre",
            lambda: run_holland_lin_double_gyre_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "thermohaline"):
        _run_case(
            "thermohaline",
            lambda: run_diabatic_thermohaline_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "phillips"):
        _run_case(
            "phillips_two_layer",
            lambda: run_two_layer_phillips_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    if args.test in ("all", "taylor"):
        _run_case(
            "taylor_column",
            lambda: run_taylor_column_test(
                grid, z_coord, config, args.dt, n_steps, output_dir, point_size,
                args.discretization,
            ),
        )

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Resolution:  C{args.resolution} x {args.levels}L ({grid.n_cells:,} columns)")
    print(f"  Duration:    {args.days} days ({n_steps} steps, dt={args.dt:.0f}s)")
    print(f"  Runtime checks: {args.runtime_checks}")

    summary_cases = {}
    for name, res in results.items():
        m = res["metrics"]
        is_forced = name in _FORCED_CASES
        if is_forced:
            m["forced"] = True
            m["conservation_note"] = _FORCED_CASES[name]
        else:
            m["forced"] = False
        print(f"\n  --- {name} ---")
        if is_forced:
            print(f"    [FORCED: {_FORCED_CASES[name]}]")
        print(f"    Final SSH range:  [{m['SSH_min']:.4e}, {m['SSH_max']:.4e}] m")
        print(f"    Final |u| max:    {m['u_max']:.4e} m/s")
        print(f"    Final |v| max:    {m['v_max']:.4e} m/s")
        print(f"    Final speed max:  {m['speed_max']:.4e} m/s")
        print(f"    Final SST mean:   {m['SST_mean']:.4f} degC")
        print(f"    Final KE:         {m['kinetic_energy']:.4e}")
        if is_forced:
            print(f"    Heat drift rel:   {m['heat_drift_rel']:.2e}  (includes forced tendency)")
            print(f"    Salt drift rel:   {m['salt_drift_rel']:.2e}  (includes forced tendency)")
        else:
            print(f"    Heat drift rel:   {m['heat_drift_rel']:.2e}")
            print(f"    Salt drift rel:   {m['salt_drift_rel']:.2e}")
        print(f"    Mean eta drift:   {m['volume_mean_eta_drift']:.2e} m")
        print(f"    All fields finite: {m['all_finite']}")
        print(f"    Land cells zero:   {m['land_zero']}")
        summary_cases[name] = m

    if case_errors:
        print("\n  --- failed_cases ---")
        for name, msg in case_errors.items():
            print(f"    {name}: {msg}")

    summary_payload = {
        "suite": "ocean_tests",
        "meta": {
            "backend": jax.default_backend(),
            "device_count": int(jax.device_count()),
            "float_dtype": str(jnp.zeros(1).dtype),
            "resolution": int(args.resolution),
            "levels": int(args.levels),
            "dt": float(args.dt),
            "days": float(args.days),
            "n_steps": int(n_steps),
            "runtime_checks": bool(args.runtime_checks),
            "discretization": str(args.discretization),
        },
        "config": _config_to_dict(config),
        "case_configs": case_configs,
        "cases": summary_cases,
        "case_errors": case_errors,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary_payload, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("legoESM Ocean Model Standardized Tests\n")
        f.write("=" * 48 + "\n")
        f.write(f"Resolution: C{args.resolution}, Levels: {args.levels}\n")
        f.write(f"dt={args.dt:.0f}s, days={args.days:.2f}, steps={n_steps}\n")
        f.write(f"discretization={args.discretization}\n")
        f.write(f"Runtime checks: {args.runtime_checks}\n")
        f.write(
            "Config: "
            f"hyperdiff={config.hyperdiff_coeff:.3e}, "
            f"n_baro={config.n_barotropic_substeps}, "
            f"baro_alpha={config.barotropic_diffusion_alpha:.3e}\n\n"
        )
        if case_configs:
            f.write("Case configs:\n")
            for name, cfg in case_configs.items():
                f.write(
                    f"  {name}: A_h={cfg['A_h']:.3e}, K_h={cfg['K_h']:.3e}, "
                    f"A_v={cfg['A_v']:.3e}, K_v={cfg['K_v']:.3e}, "
                    f"n_baro={cfg['n_barotropic_substeps']}, "
                    f"baro_alpha={cfg['barotropic_diffusion_alpha']:.3e}\n",
                )
            f.write("\n")
        for name, m in summary_cases.items():
            f.write(f"{name}\n")
            if m.get("forced"):
                f.write(f"  [FORCED: {m.get('conservation_note', '')}]\n")
            f.write(
                f"  SSH=[{m['SSH_min']:.4e}, {m['SSH_max']:.4e}] "
                f"u_max={m['u_max']:.4e} v_max={m['v_max']:.4e} "
                f"speed_max={m['speed_max']:.4e}\n",
            )
            if m.get("forced"):
                f.write(
                    f"  heat_drift_rel={m['heat_drift_rel']:.3e} (forced) "
                    f"salt_drift_rel={m['salt_drift_rel']:.3e} (forced) "
                    f"eta_mean_drift={m['volume_mean_eta_drift']:.3e}\n",
                )
            else:
                f.write(
                    f"  heat_drift_rel={m['heat_drift_rel']:.3e} "
                    f"salt_drift_rel={m['salt_drift_rel']:.3e} "
                    f"eta_mean_drift={m['volume_mean_eta_drift']:.3e}\n",
                )
            f.write(
                f"  all_finite={m['all_finite']} land_zero={m['land_zero']}\n\n",
            )
        if case_errors:
            f.write("failed_cases\n")
            for name, msg in case_errors.items():
                f.write(f"  {name}: {msg}\n")
            f.write("\n")

    print(f"\n  Output directory: {output_dir}/")
    for name in results:
        subdir = {
            "rest_state": "rest_state",
            "gravity_wave": "gravity_wave",
            "wind_gyre": "wind_gyre",
            "adiabatic_topography": "adiabatic_topography",
            "holland_lin_gyre": "holland_lin_gyre",
            "thermohaline": "thermohaline",
            "phillips_two_layer": "phillips_two_layer",
            "taylor_column": "taylor_column",
        }[name]
        print(f"    {subdir}/")
    print("=" * 70)
    return 1 if case_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
