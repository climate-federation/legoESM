"""Diagnostic saving functions for the ocean test matrix.

Functions for writing time series, conservation diagnostics, snapshot plots,
cross-sections, profiles, and snapshot data files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ocean_test_matrix import config
from ocean_test_matrix.regridding import (
    _build_latlon_weights, _apply_weights, _bin_to_latlon,
    _regrid_land_mask, _regrid_2d, _regrid_3d_level,
    _build_voronoi_polygons, _plot_voronoi_field,
    _fill_nan_profile, _fill_nan_section,
)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_results_txt(output_dir: Path, rows: dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        for k, v in rows.items():
            f.write(f"{k}: {v}\n")


def _save_timeseries_csv(output_dir: Path, diag: dict, dt: float):
    keys = [k for k in diag if k not in ("steps", "times")]
    if not keys or not diag["steps"]:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    # Always write CSV (human-readable, quick inspection)
    with open(output_dir / "mean_timeseries.csv", "w") as f:
        f.write("step,time_days," + ",".join(keys) + "\n")
        for i in range(len(diag["steps"])):
            vals = ",".join(f"{diag[k][i]:.12e}" for k in keys)
            f.write(f"{diag['steps'][i]},{diag['times'][i]:.8f},{vals}\n")
    # Also save as xarray Dataset if format is netcdf or zarr
    if config.OUTPUT_FORMAT in ("netcdf", "zarr"):
        from ocean_test_matrix.xarray_output import timeseries_to_dataset, save_dataset
        ds = timeseries_to_dataset(diag, dt)
        if ds.data_vars:
            save_dataset(ds, output_dir / "timeseries", fmt=config.OUTPUT_FORMAT)


def _save_timeseries_plot(output_dir: Path, case_name: str, diag: dict,
                          scalar_units: dict[str, str]):
    keys = [k for k in diag if k not in ("steps", "times")]
    times = diag.get("times", [])
    if not keys or not times:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        ax.plot(times, diag[key], lw=1.5)
        unit = scalar_units.get(key, "")
        ax.set_ylabel(f"{key}" + (f" ({unit})" if unit else ""))
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} — domain-averaged time series", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_conservation(output_dir: Path, case_name: str, diag: dict,
                       vol_key: str, heat_key: str, salt_key: str):
    vol_vals = diag.get(vol_key, [])
    heat_vals = diag.get(heat_key, [])
    salt_vals = diag.get(salt_key, [])
    times = diag.get("times", [])
    if not vol_vals or not heat_vals or not times:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    vol = np.array(vol_vals, dtype=np.float64)
    heat = np.array(heat_vals, dtype=np.float64)
    t = np.array(times, dtype=np.float64)
    vol_rel = (vol - vol[0]) / max(abs(vol[0]), 1e-30)
    heat_rel = (heat - heat[0]) / max(abs(heat[0]), 1e-30)

    n_panels = 2
    has_salt = len(salt_vals) == len(times)
    if has_salt:
        salt = np.array(salt_vals, dtype=np.float64)
        salt_rel = (salt - salt[0]) / max(abs(salt[0]), 1e-30)
        n_panels = 3

    with open(output_dir / "conservation_timeseries.csv", "w") as f:
        header = "time_days,volume,heat,vol_rel,heat_rel"
        if has_salt:
            header += ",salt,salt_rel"
        f.write(header + "\n")
        for i in range(t.size):
            line = (f"{t[i]:.8f},{vol[i]:.12e},{heat[i]:.12e},"
                    f"{vol_rel[i]:.12e},{heat_rel[i]:.12e}")
            if has_salt:
                line += f",{salt[i]:.12e},{salt_rel[i]:.12e}"
            f.write(line + "\n")

    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), sharex=True)
    axes[0].plot(t, vol_rel, lw=1.5)
    axes[0].axhline(0, color="0.3", ls="--", lw=0.8)
    axes[0].set_ylabel(f"Relative {vol_key} drift")
    axes[0].set_title("Volume conservation")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(t, heat_rel, lw=1.5, color="tab:red")
    axes[1].axhline(0, color="0.3", ls="--", lw=0.8)
    axes[1].set_ylabel(f"Relative {heat_key} drift")
    axes[1].set_title("Heat conservation")
    axes[1].grid(True, alpha=0.25)
    if has_salt:
        axes[2].plot(t, salt_rel, lw=1.5, color="tab:green")
        axes[2].axhline(0, color="0.3", ls="--", lw=0.8)
        axes[2].set_ylabel(f"Relative {salt_key} drift")
        axes[2].set_title("Salt conservation")
        axes[2].grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"{case_name} — conservation diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        output_dir / "conservation_timeseries.png", dpi=150,
        bbox_inches="tight")
    plt.close(fig)


def _extract_test_case_name(case_name: str) -> str:
    """Extract test case name from full case name (e.g. 'Rest State spectral T21' -> 'rest_state')."""
    case_lower = case_name.lower()
    for test_case in config.FIELD_RANGES.keys():
        test_case_words = test_case.replace('_', ' ')
        if test_case_words in case_lower:
            return test_case
    # Fallback to first word if no match
    return case_lower.split()[0].replace(' ', '_')

def _save_snapshot_plots(output_dir: Path, case_name: str, snapshots: dict,
                         dt: float, field_specs: list[tuple[str, str, str]],
                         coord_kind: str, lon_deg: np.ndarray,
                         lat_deg: np.ndarray,
                         domain_extent: tuple[float,float,float,float] | None = None,
                         mesh=None):
    """Save snapshot evolution plots for each 2D field.

    Parameters
    ----------
    mesh : VoronoiMesh, optional
        When provided and coord_kind is ``"mpas"``, snapshot plots use
        native Voronoi polygon rendering (PolyCollection) instead of
        regrid-then-imshow, eliminating land-bleed interpolation artifacts.
    """
    if not snapshots:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_steps = sorted(snapshots.keys())
    first_saved = None

    # Extract test case for consistent field ranges
    test_case = _extract_test_case_name(case_name)
    field_ranges = config.FIELD_RANGES.get(test_case, {})

    # Use native Voronoi polygon rendering for MPAS grids when mesh is
    # available.  This eliminates all interpolation artifacts at land
    # boundaries.  (Falls back to regrid+imshow when mesh is not provided.)
    use_native_voronoi = (mesh is not None
                          and coord_kind in ("mpas", "mpas_regional", "mpas_channel"))

    for field_key, field_label, cmap in field_specs:
        steps = [s for s in valid_steps if field_key in snapshots[s]]
        if not steps:
            continue
        if len(steps) > 8:
            idx = np.linspace(0, len(steps) - 1, 8).astype(int)
            steps = [steps[i] for i in idx]

        n_cols = min(4, len(steps))
        n_rows = (len(steps) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
        axes = np.atleast_2d(axes)
        im = None

        if use_native_voronoi:
            # --- Native Voronoi polygon path (MPAS) -----------------------
            for idx, step in enumerate(steps):
                r, c = divmod(idx, n_cols)
                ax = axes[r, c]
                raw = np.asarray(snapshots[step][field_key], dtype=np.float64)
                lm = (np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
                      if "land_mask" in snapshots[step] else None)
                # Per-panel color range: use explicit range if configured,
                # otherwise fit to this panel's ocean data.
                if field_key in field_ranges and field_ranges[field_key] != (None, None):
                    vmin, vmax = field_ranges[field_key]
                else:
                    ocean_vals = raw.ravel()
                    if lm is not None:
                        ocean_vals = ocean_vals[lm.ravel() > 0.5]
                    finite = ocean_vals[np.isfinite(ocean_vals)]
                    vmin = float(finite.min()) if finite.size else None
                    vmax = float(finite.max()) if finite.size else None
                # Force symmetric colorscale centered at 0 for velocity
                # and w fields (diverging quantities)
                _sym = ("w_" in field_key or field_key in ("u_sfc", "v_sfc"))
                if vmin is not None and vmax is not None and _sym:
                    vlim = max(abs(vmin), abs(vmax))
                    vmin, vmax = -vlim, vlim
                im = _plot_voronoi_field(
                    ax, mesh, raw, land_mask=lm, cmap=cmap,
                    vmin=vmin, vmax=vmax)
                if domain_extent is not None:
                    ax.set_xlim(domain_extent[0], domain_extent[1])
                    ax.set_ylim(domain_extent[2], domain_extent[3])
                if im is not None:
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                day = step * dt / 86400.0
                ax.set_title(f"t={day:.2f} d", fontsize=9)
                if c == 0:
                    ax.set_ylabel("Latitude")
                if r == n_rows - 1:
                    ax.set_xlabel("Longitude")
        else:
            # --- Regrid + imshow path (latlon, cubed-sphere, fallback) ----
            # Pre-compute regridded fields for all steps
            all_regridded = []
            for step in steps:
                raw = np.asarray(snapshots[step][field_key], dtype=np.float64)
                # Use mask-aware IDW: pass ocean_mask so land cells are
                # excluded from the KDTree and cannot contaminate ocean values.
                land_mask_raw = None
                if "land_mask" in snapshots[step]:
                    land_mask_raw = np.asarray(snapshots[step]["land_mask"],
                                                dtype=np.float64)
                regridded = _regrid_2d(raw, lon_deg, lat_deg, coord_kind,
                                       ocean_mask=land_mask_raw)
                # Still apply land mask on the regridded grid for display
                if land_mask_raw is not None:
                    land_mask_reg = _regrid_land_mask(
                        land_mask_raw, lon_deg, lat_deg, coord_kind)
                    regridded = np.where(land_mask_reg > 0.5, regridded, np.nan)
                all_regridded.append(regridded)

            for idx, step in enumerate(steps):
                r, c = divmod(idx, n_cols)
                ax = axes[r, c]
                regridded = all_regridded[idx]

                # Per-panel color range
                if field_key in field_ranges and field_ranges[field_key] != (None, None):
                    vmin, vmax = field_ranges[field_key]
                else:
                    panel_finite = regridded.ravel()
                    panel_finite = panel_finite[np.isfinite(panel_finite)]
                    if len(panel_finite) > 0:
                        vmin, vmax = float(panel_finite.min()), float(panel_finite.max())
                    else:
                        vmin, vmax = None, None
                # Force symmetric colorscale centered at 0 for velocity
                # and w fields (diverging quantities)
                _sym = ("w_" in field_key or field_key in ("u_sfc", "v_sfc"))
                if vmin is not None and vmax is not None and _sym:
                    vlim = max(abs(vmin), abs(vmax))
                    vmin, vmax = -vlim, vlim

                # Compute extent from actual coordinates.
                # When domain_extent is given (regional experiments), use it
                # directly.  Otherwise infer from coordinate arrays or the
                # non-NaN bounding box of the regridded field.
                if domain_extent is not None:
                    # domain_extent = (lon_west, lon_east, lat_south, lat_north)
                    lon_ext = [domain_extent[0], domain_extent[1]]
                    lat_ext = [domain_extent[2], domain_extent[3]]
                    if coord_kind not in ("latlon", "gaussian"):
                        lat_1d = np.linspace(-90, 90, regridded.shape[0])
                        lon_1d = np.linspace(0, 360, regridded.shape[1])
                        r0 = max(int(np.searchsorted(lat_1d, lat_ext[0])) - 1, 0)
                        r1 = min(int(np.searchsorted(lat_1d, lat_ext[1])) + 2,
                                 len(lat_1d))
                        c0 = max(int(np.searchsorted(lon_1d, lon_ext[0])) - 1, 0)
                        c1 = min(int(np.searchsorted(lon_1d, lon_ext[1])) + 2,
                                 len(lon_1d))
                        plot_data = regridded[r0:r1, c0:c1]
                        lon_ext = [float(lon_1d[c0]),
                                   float(lon_1d[min(c1, len(lon_1d)-1)])]
                        lat_ext = [float(lat_1d[r0]),
                                   float(lat_1d[min(r1, len(lat_1d)-1)])]
                    else:
                        plot_data = regridded
                elif coord_kind in ("latlon", "gaussian"):
                    lon_flat = np.asarray(lon_deg, dtype=np.float64).ravel()
                    lat_flat = np.asarray(lat_deg, dtype=np.float64).ravel()
                    lon_ext = [float(lon_flat.min()), float(lon_flat.max())]
                    lat_ext = [float(lat_flat.min()), float(lat_flat.max())]
                    plot_data = regridded
                else:
                    lat_1d = np.linspace(-90, 90, regridded.shape[0])
                    lon_1d = np.linspace(0, 360, regridded.shape[1])
                    valid = np.isfinite(regridded)
                    if valid.any():
                        rows = np.where(valid.any(axis=1))[0]
                        cols = np.where(valid.any(axis=0))[0]
                        r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, len(lat_1d))
                        c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, len(lon_1d))
                        plot_data = regridded[r0:r1, c0:c1]
                        lon_ext = [float(lon_1d[c0]), float(lon_1d[min(c1, len(lon_1d)-1)])]
                        lat_ext = [float(lat_1d[r0]), float(lat_1d[min(r1, len(lat_1d)-1)])]
                    else:
                        plot_data = regridded
                        lon_ext = [0, 360]
                        lat_ext = [-90, 90]
                im = ax.imshow(
                    plot_data, origin="lower", aspect="auto", cmap=cmap,
                    extent=[lon_ext[0], lon_ext[1], lat_ext[0], lat_ext[1]],
                    vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Add velocity vectors for circulation visualization
                if field_key in ("SST", "speed_sfc") and "u_sfc" in snapshots[step] and "v_sfc" in snapshots[step]:
                    u_raw = np.asarray(snapshots[step]["u_sfc"], dtype=np.float64)
                    v_raw = np.asarray(snapshots[step]["v_sfc"], dtype=np.float64)
                    _lm = (np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
                            if "land_mask" in snapshots[step] else None)
                    u_reg = _regrid_2d(u_raw, lon_deg, lat_deg, coord_kind,
                                       ocean_mask=_lm)
                    v_reg = _regrid_2d(v_raw, lon_deg, lat_deg, coord_kind,
                                       ocean_mask=_lm)

                    if domain_extent is not None and coord_kind not in ("latlon", "gaussian"):
                        lat_1d = np.linspace(-90, 90, u_reg.shape[0])
                        lon_1d = np.linspace(0, 360, u_reg.shape[1])
                        r0 = max(int(np.searchsorted(lat_1d, lat_ext[0])) - 1, 0)
                        r1 = min(int(np.searchsorted(lat_1d, lat_ext[1])) + 2, len(lat_1d))
                        c0 = max(int(np.searchsorted(lon_1d, lon_ext[0])) - 1, 0)
                        c1 = min(int(np.searchsorted(lon_1d, lon_ext[1])) + 2, len(lon_1d))
                        u_plot = u_reg[r0:r1, c0:c1]
                        v_plot = v_reg[r0:r1, c0:c1]
                        lat_plot = lat_1d[r0:r1]
                        lon_plot = lon_1d[c0:c1]
                    else:
                        u_plot = u_reg
                        v_plot = v_reg
                        lat_plot = np.linspace(lat_ext[0], lat_ext[1], u_reg.shape[0])
                        lon_plot = np.linspace(lon_ext[0], lon_ext[1], u_reg.shape[1])

                    skip = 4
                    X, Y = np.meshgrid(lon_plot[::skip], lat_plot[::skip])
                    U = u_plot[::skip, ::skip]
                    V = v_plot[::skip, ::skip]

                    mask = np.isfinite(U) & np.isfinite(V) & ((np.abs(U) + np.abs(V)) > 1e-6)
                    if np.any(mask):
                        speed = np.sqrt(U[mask]**2 + V[mask]**2)
                        max_speed = np.nanmax(speed) if len(speed) > 0 else 0.01
                        scale = max_speed * 100

                        ax.quiver(X[mask], Y[mask], U[mask], V[mask],
                                 color='white', alpha=0.8, scale=scale, scale_units='xy',
                                 width=0.003, headwidth=4, headlength=6,
                                 edgecolors='black', linewidth=0.5)

                day = step * dt / 86400.0
                ax.set_title(f"t={day:.2f} d", fontsize=9)
                if c == 0:
                    ax.set_ylabel("Latitude")
                if r == n_rows - 1:
                    ax.set_xlabel("Longitude")

        for idx in range(len(steps), n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r, c].set_visible(False)

        fig.suptitle(f"{case_name} — {field_key} ({field_label})", fontsize=11)
        fig.tight_layout()
        fname = f"snapshots_{field_key}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if first_saved is None:
            first_saved = fname

    # Alias first field as field_snapshots.png
    if first_saved and (output_dir / first_saved).exists():
        shutil.copy2(output_dir / first_saved,
                     output_dir / "field_snapshots.png")


def _bin_cross_section(
    f3d: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    coord_kind: str,
    mean_axis: int,
    n_lat: int = 181,
    n_lon: int = 360,
    ocean_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a cross-section by direct binning (no interpolation).

    For regular grids (latlon, gaussian): regrid then nanmean as before.
    For unstructured grids (cube, mpas): bin source points directly into
    latitude or longitude bins with adaptive bin count ensuring >= ~10
    points per bin on average, avoiding IDW interpolation artifacts.

    Parameters
    ----------
    mean_axis : int
        0 = average over latitude → longitude-vertical section
        1 = average over longitude → latitude-vertical section
    ocean_mask : array, optional
        Boolean-like mask (>0.5 = ocean).  Excludes land cells from KDTree.

    Returns
    -------
    section : np.ndarray, shape (n_bins, nlev)
    bin_centers : np.ndarray, shape (n_bins,) — for plotting extents
    """
    # Regrid to regular lat-lon, then average over the requested axis.
    # Use k=20 neighbors for unstructured grids to avoid aliasing from
    # cubed-sphere face boundaries or icosahedral grid structure.
    if coord_kind not in ("latlon", "gaussian"):
        arr = np.asarray(f3d, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        flat = arr.reshape(-1, arr.shape[-1])
        nlev = flat.shape[1]
        idxs, w = _build_latlon_weights(lon_deg, lat_deg, n_lat, n_lon, k=20,
                                         ocean_mask=ocean_mask)
        ll = np.full((n_lat, n_lon, nlev), np.nan, dtype=np.float64)
        for lev in range(nlev):
            ll[..., lev] = _apply_weights(flat[:, lev], idxs, w, n_lat, n_lon)
    else:
        ll = _regrid_3d_level(f3d, lon_deg, lat_deg, coord_kind)

    section = np.nanmean(ll, axis=mean_axis)
    if mean_axis == 0:
        return section, np.linspace(0, 360, section.shape[0])
    return section, np.linspace(-90, 90, section.shape[0])


def _save_cross_sections(output_dir: Path, case_name: str, snapshots: dict,
                         dt: float, field_3d_key: str, coord_kind: str,
                         lon_deg: np.ndarray, lat_deg: np.ndarray,
                         levels: np.ndarray, level_label: str):
    """Save latitude-vertical and longitude-vertical cross-sections."""
    valid_steps = sorted(
        s for s in snapshots if field_3d_key in snapshots[s])
    if not valid_steps:
        return
    if len(valid_steps) > 6:
        idx = np.linspace(0, len(valid_steps) - 1, 6).astype(int)
        valid_steps = [valid_steps[i] for i in idx]

    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, mean_axis, xlabel in [
        ("latitude_vertical_cross_sections.png", 1, "Latitude"),
        ("longitude_vertical_cross_sections.png", 0, "Longitude"),
    ]:
        nc = len(valid_steps)
        fig, axes_arr = plt.subplots(
            1, nc, figsize=(4.5 * nc, 5), sharey=True)
        if nc == 1:
            axes_arr = [axes_arr]
        im = None

        # Pre-compute all cross-sections and find shared color range
        all_sections = []
        all_bin_centers = []
        for step in valid_steps:
            f3d = np.asarray(snapshots[step][field_3d_key], dtype=np.float64)
            _lm = None
            if "land_mask" in snapshots[step]:
                _lm = np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
                land_mask_3d = _lm[..., np.newaxis]
                f3d = np.where(land_mask_3d > 0.5, f3d, np.nan)
            section, bin_centers = _bin_cross_section(
                f3d, lon_deg, lat_deg, coord_kind, mean_axis,
                ocean_mask=_lm)
            section = _fill_nan_section(section)
            all_sections.append(section)
            all_bin_centers.append(bin_centers)

        all_vals = np.concatenate([s.ravel() for s in all_sections])
        all_finite = all_vals[np.isfinite(all_vals)]
        if len(all_finite) > 0:
            cs_vmin, cs_vmax = float(np.nanmin(all_finite)), float(np.nanmax(all_finite))
        else:
            cs_vmin, cs_vmax = None, None

        # Compute level interfaces for pcolormesh (accurate vertical grid representation)
        def compute_level_interfaces(level_centers):
            """Compute interface depths from level center depths."""
            if len(level_centers) == 1:
                return np.array([0.0, 2 * level_centers[0]])

            # Compute interfaces as midpoints between level centers
            interfaces = np.zeros(len(level_centers) + 1)
            interfaces[0] = 0.0  # Surface
            interfaces[1:-1] = 0.5 * (level_centers[:-1] + level_centers[1:])
            interfaces[-1] = level_centers[-1] + (level_centers[-1] - interfaces[-2])
            return interfaces

        level_interfaces = compute_level_interfaces(levels)

        for ax, step, section, bin_centers in zip(
                axes_arr, valid_steps, all_sections, all_bin_centers):
            # Use pcolormesh to show true model grid structure instead of imshow
            # This accurately represents the variable vertical grid spacing
            bin_interfaces = np.linspace(bin_centers[0] - 0.5 * (bin_centers[1] - bin_centers[0]),
                                       bin_centers[-1] + 0.5 * (bin_centers[-1] - bin_centers[-2]),
                                       len(bin_centers) + 1)
            X, Y = np.meshgrid(bin_interfaces, level_interfaces)
            im = ax.pcolormesh(
                X, Y, section.T, cmap="RdBu_r", shading='flat',
                vmin=cs_vmin, vmax=cs_vmax)

            day = step * dt / 86400.0
            ax.set_title(f"t={day:.2f} d", fontsize=9)
            ax.set_xlabel(xlabel)

        # Invert y-axis ONCE after loop (calling inside loop with sharey=True
        # toggles inversion, leaving even-panel counts non-inverted).
        axes_arr[0].invert_yaxis()  # Surface (depth=0) at top
        axes_arr[0].set_ylabel(level_label)
        if im is not None:
            fig.colorbar(
                im, ax=axes_arr, orientation="vertical", fraction=0.046,
                pad=0.04, label=field_3d_key)
        fig.suptitle(
            f"{case_name} — {field_3d_key} cross-sections", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.88, 0.95])  # More space for colorbar
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_profiles(output_dir: Path, case_name: str, snapshots: dict,
                   dt: float, field_3d_key: str, levels: np.ndarray,
                   level_label: str):
    """Save vertical profile evolution plot."""
    valid_steps = sorted(
        s for s in snapshots if field_3d_key in snapshots[s])
    if not valid_steps:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_steps)))
    for step, color in zip(valid_steps, colors):
        f3d = np.asarray(snapshots[step][field_3d_key], dtype=np.float64)

        # Apply land masking to 3D field before computing profile if available
        if "land_mask" in snapshots[step]:
            land_mask_raw = np.asarray(snapshots[step]["land_mask"], dtype=np.float64)
            land_mask_3d = land_mask_raw[..., np.newaxis]  # Expand to 3D
            f3d = np.where(land_mask_3d > 0.5, f3d, np.nan)

        profile = np.nanmean(f3d, axis=tuple(range(f3d.ndim - 1)))
        day = step * dt / 86400.0
        ax.plot(profile, levels, color=color, lw=1.5, label=f"day {day:.1f}")
    ax.set_xlabel(field_3d_key)
    ax.set_ylabel(level_label)
    ax.invert_yaxis()
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title(f"{case_name} — vertical profile evolution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / "vertical_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_velocity_profiles(output_dir: Path, case_name: str,
                            snapshots: dict, dt: float,
                            levels: np.ndarray, level_label: str = "Depth (m)"):
    """Save vertical velocity profile diagnostics (mean, max, structure).

    Requires snapshots to contain 'u_3d', 'v_3d', 'speed_3d' keys
    (set include_velocity_3d=True in _make_extract_fn).
    """
    valid_steps = sorted(
        s for s in snapshots
        if "u_3d" in snapshots[s] and "v_3d" in snapshots[s])
    if not valid_steps:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    n_snaps = len(valid_steps)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, n_snaps))

    def _mask_3d(snap):
        mask = snap["land_mask"]
        if mask.ndim == 1:  # MPAS: (nCells,)
            return mask[:, np.newaxis]
        return mask[..., np.newaxis]  # latlon: (n_lat, n_lon)

    def _horiz_axes(m3d):
        return tuple(range(m3d.ndim - 1))

    # -- 1. Domain-mean |u|, |v|, speed vs depth --
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    for idx, step in enumerate(valid_steps):
        snap = snapshots[step]
        m3d = _mask_3d(snap)
        hax = _horiz_axes(m3d)
        ocean_count = np.maximum(np.sum(m3d > 0.5, axis=hax), 1)
        day = step * dt / 86400.0
        lbl = f"day {day:.1f}"

        mean_abs_u = np.sum(np.abs(snap["u_3d"]) * m3d, axis=hax) / ocean_count
        mean_abs_v = np.sum(np.abs(snap["v_3d"]) * m3d, axis=hax) / ocean_count
        mean_spd = np.sum(snap["speed_3d"] * m3d, axis=hax) / ocean_count

        axes[0].plot(mean_abs_u, levels, color=cmap[idx], label=lbl)
        axes[1].plot(mean_abs_v, levels, color=cmap[idx], label=lbl)
        axes[2].plot(mean_spd, levels, color=cmap[idx], label=lbl)

    axes[0].set_xlabel(r"Mean $|u|$ (m/s)"); axes[0].set_title("Zonal velocity")
    axes[1].set_xlabel(r"Mean $|v|$ (m/s)"); axes[1].set_title("Meridional velocity")
    axes[2].set_xlabel("Mean speed (m/s)"); axes[2].set_title("Speed")
    axes[0].set_ylabel(level_label)
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} — domain-mean velocity profiles", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_profiles_mean.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # -- 2. Max |u|, |v|, speed vs depth --
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    for idx, step in enumerate(valid_steps):
        snap = snapshots[step]
        m3d = _mask_3d(snap)
        hax = _horiz_axes(m3d)
        day = step * dt / 86400.0
        lbl = f"day {day:.1f}"

        max_u = np.max(np.where(m3d > 0.5, np.abs(snap["u_3d"]), 0.0), axis=hax)
        max_v = np.max(np.where(m3d > 0.5, np.abs(snap["v_3d"]), 0.0), axis=hax)
        max_spd = np.max(np.where(m3d > 0.5, snap["speed_3d"], 0.0), axis=hax)

        axes[0].plot(max_u, levels, color=cmap[idx], label=lbl)
        axes[1].plot(max_v, levels, color=cmap[idx], label=lbl)
        axes[2].plot(max_spd, levels, color=cmap[idx], label=lbl)

    axes[0].set_xlabel(r"Max $|u|$ (m/s)"); axes[0].set_title("Zonal velocity")
    axes[1].set_xlabel(r"Max $|v|$ (m/s)"); axes[1].set_title("Meridional velocity")
    axes[2].set_xlabel("Max speed (m/s)"); axes[2].set_title("Speed")
    axes[0].set_ylabel(level_label)
    axes[2].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} — max velocity profiles", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_profiles_max.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # -- 3. Final-snapshot velocity structure: mean + RMS per level --
    last_step = valid_steps[-1]
    snap = snapshots[last_step]
    m3d = _mask_3d(snap)
    hax = _horiz_axes(m3d)
    day = last_step * dt / 86400.0
    ocean_count = np.maximum(np.sum(m3d > 0.5, axis=hax), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    mean_u = np.sum(snap["u_3d"] * m3d, axis=hax) / ocean_count
    mean_v = np.sum(snap["v_3d"] * m3d, axis=hax) / ocean_count
    rms_u = np.sqrt(np.sum(snap["u_3d"]**2 * m3d, axis=hax) / ocean_count)
    rms_v = np.sqrt(np.sum(snap["v_3d"]**2 * m3d, axis=hax) / ocean_count)

    axes[0].plot(mean_u, levels, 'b-o', ms=4, label="mean u")
    axes[0].plot(rms_u, levels, 'b--s', ms=4, label="RMS u")
    axes[0].set_xlabel("Velocity (m/s)"); axes[0].set_ylabel(level_label)
    axes[0].set_title(f"Zonal velocity (day {day:.1f})")
    axes[0].legend(); axes[0].axvline(0, color='k', lw=0.5, ls=':')

    axes[1].plot(mean_v, levels, 'r-o', ms=4, label="mean v")
    axes[1].plot(rms_v, levels, 'r--s', ms=4, label="RMS v")
    axes[1].set_xlabel("Velocity (m/s)")
    axes[1].set_title(f"Meridional velocity (day {day:.1f})")
    axes[1].legend(); axes[1].axvline(0, color='k', lw=0.5, ls=':')

    for ax in axes:
        ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{case_name} — velocity structure (final)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_structure_final.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def _save_snapshot_times(output_dir: Path, snapshots: dict, dt: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "snapshot_times.txt", "w") as f:
        f.write("step,time_seconds,time_days\n")
        for step in sorted(snapshots.keys()):
            t_s = step * dt
            f.write(f"{step},{t_s:.2f},{t_s / 86400:.6f}\n")


def _save_snapshot_data(
    output_dir: Path,
    snapshots: dict,
    dt: float,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    domain_extent: tuple[float, float, float, float] | None = None,
    depth_values: np.ndarray | None = None,
):
    """Save snapshot field arrays as NPZ files with proper time series format.

    NEW FORMAT: Each field is saved as a time series array with shape (n_times, ...).
    This replaces the old format where each timestep was a separate variable.
    """
    if not snapshots:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_steps = sorted(snapshots.keys())
    times_days = np.array([s * dt / 86400.0 for s in sorted_steps],
                          dtype=np.float64)

    # Common metadata for both files
    common_metadata = {
        "steps": np.array(sorted_steps, dtype=np.int64),
        "times_days": times_days,
    }

    # Collect all unique field keys across all timesteps
    all_field_keys = set()
    for step_data in snapshots.values():
        all_field_keys.update(step_data.keys())

    # Build time-series arrays for native grid
    native_arrays = dict(common_metadata)
    latlon_arrays = dict(common_metadata)
    # Store source coordinate range so downstream plotters know the
    # actual domain extent (important for regional unstructured meshes
    # where nearest-neighbour regridding bleeds beyond the domain).
    # When domain_extent is provided (regional experiments), use it
    # instead of the mesh cell coordinate range (which can extend
    # beyond the nominal domain for Voronoi meshes).
    src_lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    src_lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    if coord_kind in ("latlon", "gaussian"):
        # Native grids have lon in [0, 360); store actual coordinates
        latlon_arrays["lat"] = src_lat
        latlon_arrays["lon"] = src_lon
    else:
        # Regridded grids (cube, mpas) use [0, 360] target
        latlon_arrays["lat"] = np.linspace(-90.0, 90.0, 181)
        latlon_arrays["lon"] = np.linspace(0.0, 360.0, 360)
    if domain_extent is not None:
        latlon_arrays["source_lon_range"] = np.array(
            [domain_extent[0], domain_extent[1]])
        latlon_arrays["source_lat_range"] = np.array(
            [domain_extent[2], domain_extent[3]])
    else:
        src_lon_min, src_lon_max = float(src_lon.min()), float(src_lon.max())
        if src_lon_min < 0:
            src_lon_min = src_lon_min % 360
            src_lon_max = src_lon_max % 360
            if src_lon_max <= src_lon_min:
                src_lon_min, src_lon_max = 0.0, 360.0
        latlon_arrays["source_lon_range"] = np.array(
            [src_lon_min, src_lon_max])
        latlon_arrays["source_lat_range"] = np.array(
            [float(src_lat.min()), float(src_lat.max())])

    for field_key in all_field_keys:
        # Collect this field across all timesteps
        field_timesteps = []
        latlon_timesteps = []

        for step in sorted_steps:
            if field_key in snapshots[step]:
                arr = np.asarray(snapshots[step][field_key], dtype=np.float64)
                field_timesteps.append(arr)

                # Get ocean mask for mask-aware IDW (skip for land_mask itself)
                _lm = None
                if field_key != "land_mask" and "land_mask" in snapshots[step]:
                    _lm = np.asarray(snapshots[step]["land_mask"],
                                     dtype=np.float64)

                # Regrid to lat-lon
                target_lat = latlon_arrays["lat"]
                target_lon = latlon_arrays["lon"]
                if field_key.endswith("_3d"):
                    regridded = _regrid_3d_level(arr, lon_deg, lat_deg, coord_kind,
                                               target_lat=target_lat,
                                               target_lon=target_lon,
                                               ocean_mask=_lm)
                else:
                    regridded = _regrid_2d(arr, lon_deg, lat_deg, coord_kind,
                                         target_lat=target_lat,
                                         target_lon=target_lon,
                                         ocean_mask=_lm)
                latlon_timesteps.append(regridded)
            else:
                # Field not available at this timestep - skip or use NaN
                # For now, we'll skip incomplete time series
                break

        # Only save fields that are available at all timesteps
        if len(field_timesteps) == len(sorted_steps):
            # Stack into time series: shape (n_times, ...)
            native_arrays[field_key] = np.stack(field_timesteps, axis=0)
            latlon_arrays[field_key] = np.stack(latlon_timesteps, axis=0)

    # TODO: once postprocessing fully supports NetCDF, skip NPZ when format != "npz"
    np.savez_compressed(output_dir / "snapshots_native.npz", **native_arrays)
    np.savez_compressed(output_dir / "snapshots_latlon.npz", **latlon_arrays)

    if config.OUTPUT_FORMAT in ("netcdf", "zarr"):
        from ocean_test_matrix.xarray_output import (
            stacked_arrays_to_dataset, _arrays_to_latlon_dataset, save_dataset,
        )
        ds_native = stacked_arrays_to_dataset(
            native_arrays, coord_kind, lon_deg, lat_deg,
            depth=depth_values,
            attrs={"description": "Native grid snapshots"})
        save_dataset(ds_native, output_dir / "snapshots_native",
                     fmt=config.OUTPUT_FORMAT)
        ds_latlon = _arrays_to_latlon_dataset(latlon_arrays, depth=depth_values)
        save_dataset(ds_latlon, output_dir / "snapshots_latlon",
                     fmt=config.OUTPUT_FORMAT)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _save_case_diagnostics(
    output_dir: Path,
    case_name: str,
    dt: float,
    diag: dict,
    snapshots: dict,
    coord_kind: str,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    field_specs_2d: list[tuple[str, str, str]],
    *,
    field_3d_key: str | None = None,
    level_values: np.ndarray | None = None,
    level_label: str = "Depth (m)",
    vol_key: str | None = None,
    heat_key: str | None = None,
    salt_key: str | None = None,
    scalar_units: dict[str, str] | None = None,
    domain_extent: tuple[float, float, float, float] | None = None,
    mesh=None,
):
    """Save all standard diagnostic outputs for a test case.

    Parameters
    ----------
    domain_extent : (lon_west, lon_east, lat_south, lat_north) or None
        When set, snapshot plots are cropped to this geographic extent.
        Useful for regional experiments on unstructured grids where the
        auto-crop heuristic fails due to nearest-neighbour extrapolation.
    mesh : VoronoiMesh, optional
        When provided for MPAS grids, snapshot plots use native Voronoi
        polygon rendering instead of regrid+imshow.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    su = scalar_units or {}

    _save_timeseries_csv(output_dir, diag, dt)
    _save_timeseries_plot(output_dir, case_name, diag, su)

    if vol_key and heat_key:
        _save_conservation(output_dir, case_name, diag,
                           vol_key, heat_key, salt_key or "")

    _save_snapshot_plots(
        output_dir, case_name, snapshots, dt, field_specs_2d,
        coord_kind, lon_deg, lat_deg, domain_extent=domain_extent,
        mesh=mesh)
    _save_snapshot_times(output_dir, snapshots, dt)
    _save_snapshot_data(
        output_dir, snapshots, dt, coord_kind, lon_deg, lat_deg,
        domain_extent=domain_extent, depth_values=level_values)

    if field_3d_key and level_values is not None:
        _save_cross_sections(
            output_dir, case_name, snapshots, dt, field_3d_key,
            coord_kind, lon_deg, lat_deg, level_values, level_label)
        _save_profiles(
            output_dir, case_name, snapshots, dt, field_3d_key,
            level_values, level_label)


def _placeholder_plot(path: Path, title: str, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.6, title, ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(0.5, 0.3, text, ha="center", va="center", fontsize=8)
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _ensure_required_artifacts(output_dir: Path):
    """Guarantee standardized files exist in each case folder.

    Creates placeholder CSVs, snapshot_times.txt, and placeholder PNGs
    so that every test case directory has the full set of expected outputs,
    even when the test crashed (ERROR) or was skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_name in ["mean_timeseries.csv", "conservation_timeseries.csv"]:
        p = output_dir / csv_name
        if not p.exists():
            with open(p, "w") as f:
                f.write("# No data produced\n")
    if not (output_dir / "snapshot_times.txt").exists():
        with open(output_dir / "snapshot_times.txt", "w") as f:
            f.write("step,time_seconds,time_days\n")
    # Generate placeholder PNGs for any missing visualization files.
    case_label = "/".join(output_dir.parts[-4:])
    for png_name in [
        "mean_timeseries.png",
        "conservation_timeseries.png",
        "field_snapshots.png",
    ]:
        p = output_dir / png_name
        if not p.exists():
            _placeholder_plot(p, png_name.replace(".png", ""),
                              f"No data — {case_label}")
