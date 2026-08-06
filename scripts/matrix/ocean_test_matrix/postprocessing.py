"""Cross-grid comparison, replot, and rest-state cross-variant functions."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from legoesm import constants
from ocean_test_matrix import config
from ocean_test_matrix.testcase import TestCase
from ocean_test_matrix.diagnostic_io import (
    _extract_test_case_name,
    _save_snapshot_plots,
    _save_cross_sections,
    _save_profiles,
)


# ===========================================================================
# Cross-grid comparison helpers
# ===========================================================================


def _collect_grid_results(test_case_dir: Path) -> dict:
    """Collect results from all grids that completed for this test case.

    Returns:
        dict mapping grid_type -> {timeseries, snapshots, metadata}
    """
    grid_results = {}

    for grid_dir in test_case_dir.iterdir():
        if not grid_dir.is_dir():
            continue

        # Find resolution subdirectory (e.g., C24, 36x72, ico3, T21)
        resolution_dirs = [d for d in grid_dir.iterdir() if d.is_dir()]
        if not resolution_dirs:
            continue
        # iter-112 codex LOW-3: port the iter-108/110 collector
        # logic to this modular postprocessing path:
        # 1) filter hidden / internal dirs
        # 2) prefer grid-typed format over bare-numeric
        # 3) graceful fallback for legacy trees
        _BAD_DIRNAMES = {"__pycache__", ".ipynb_checkpoints"}
        resolution_dirs = [
            d for d in resolution_dirs
            if not d.name.startswith(".")
            and d.name not in _BAD_DIRNAMES
        ]
        if not resolution_dirs:
            continue
        grid_typed_pattern = {
            "cubed_sphere": lambda n: n.startswith("C") and n[1:].isdigit(),
            "latlon": lambda n: "x" in n and all(p.isdigit() for p in n.split("x") if p),
            "mpas": lambda n: n.startswith("ico") and n[3:].isdigit(),
            "spectral": lambda n: n.startswith("T") and n[1:].isdigit(),
            "mpas_regional": lambda n: n.endswith("km") and n[:-2].isdigit(),
            "latlon_regional": lambda n: "x" in n and all(p.isdigit() for p in n.split("x") if p),
            "cs_regional": lambda n: n.startswith("C") and n[1:].isdigit(),
        }
        matcher = grid_typed_pattern.get(grid_dir.name)
        if matcher is not None:
            typed = [d for d in resolution_dirs if matcher(d.name)]
            if typed:
                resolution_dir = sorted(typed)[0]
            else:
                # iter-115 codex iter-114-followup MEDIUM-3:
                # prefer bare-numeric isdigit() dirs over
                # arbitrary names (e.g., ``_archive`` legacy
                # dirs).  Mirrors the iter-112 fix for the
                # monolithic ocean collector.
                bare_numeric = [
                    d for d in resolution_dirs if d.name.isdigit()
                ]
                if bare_numeric:
                    resolution_dir = sorted(bare_numeric)[0]
                else:
                    resolution_dir = resolution_dirs[0]
        else:
            resolution_dir = resolution_dirs[0]

        # Check for required files
        csv_file = resolution_dir / "mean_timeseries.csv"
        npz_file = resolution_dir / "snapshots_latlon.npz"
        results_file = resolution_dir / "results.txt"

        if all(f.exists() for f in [csv_file, npz_file, results_file]):
            try:
                # Load timeseries data
                timeseries_df = pd.read_csv(csv_file)

                # Load snapshot data as NPZ (postprocessing uses .files attribute)
                snapshots_data = np.load(npz_file)

                # Parse results metadata
                metadata = {}
                with open(results_file, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metadata[key.strip()] = value.strip()

                grid_results[grid_dir.name] = {
                    'timeseries': timeseries_df,
                    'snapshots': snapshots_data,
                    'metadata': metadata,
                    'resolution': resolution_dir.name
                }
            except Exception as e:
                print(f"Warning: Failed to load data for {grid_dir.name}: {e}")
                continue

    return grid_results


def _create_comparison_timeseries(test_case_dir: Path, grid_results: dict) -> None:
    """Create 4-panel time series comparison plot across all grids."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Time Series Comparison - {test_case_dir.name}', fontsize=14, fontweight='bold')

    # Colors for different grids
    colors = {
        'cubed_sphere': 'blue', 'latlon': 'red', 'mpas': 'green',
        'spectral': 'orange', 'mpas_regional': 'tab:green',
        'latlon_regional': 'tab:red', 'cs_regional': 'tab:blue',
    }
    linestyles = {
        'cubed_sphere': '-', 'latlon': '-', 'mpas': '-',
        'spectral': '-', 'mpas_regional': '--',
        'latlon_regional': '--', 'cs_regional': '--',
    }

    for grid_name, data in grid_results.items():
        df = data['timeseries']
        color = colors.get(grid_name, 'black')
        ls = linestyles.get(grid_name, '-')

        # Panel 1: Mean eta evolution
        if 'mean_eta' in df.columns:
            axes[0,0].plot(df['time_days'], df['mean_eta'], label=grid_name, color=color, ls=ls)

        # Panel 2: Max |eta|
        if 'max_abs_eta' in df.columns:
            axes[0,1].plot(df['time_days'], df['max_abs_eta'], label=grid_name, color=color, ls=ls)

        # Panel 3: Max speed
        if 'max_speed' in df.columns:
            axes[1,0].plot(df['time_days'], df['max_speed'], label=grid_name, color=color, ls=ls)

        # Panel 4: Volume-weighted KE
        if 'mean_ke' in df.columns:
            axes[1,1].plot(df['time_days'], df['mean_ke'], label=grid_name, color=color, ls=ls)

    axes[0,0].set_ylabel('Mean η (m)')
    axes[0,0].set_title('Mean Sea Surface Height')
    axes[0,1].set_ylabel('Max |η| (m)')
    axes[0,1].set_title('Maximum SSH Amplitude')
    axes[1,0].set_ylabel('Max Speed (m/s)')
    axes[1,0].set_title('Maximum Speed')
    axes[1,1].set_ylabel('Mean KE (m²/s²)')
    axes[1,1].set_title('Volume-Weighted Kinetic Energy')

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Set common x-label
    for ax in axes[1,:]:
        ax.set_xlabel('Time (days)')

    from datetime import datetime
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()

    # Save plot
    output_file = test_case_dir / "comparison_timeseries.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def _create_comparison_snapshots(test_case_dir: Path, grid_results: dict, field: str = 'eta') -> None:
    """Create 4-panel final snapshot comparison for a given field."""
    import matplotlib.pyplot as plt

    # Determine the test case for field ranges
    test_case = _extract_test_case_name(test_case_dir.name)
    field_ranges = config.FIELD_RANGES.get(test_case, {})
    vmin, vmax = field_ranges.get(field, (None, None))

    # If no explicit range, compute shared range across all grids
    if vmin is None or vmax is None:
        all_vals = []
        for data in grid_results.values():
            snapshots = data['snapshots']
            if field in snapshots.files:
                fd = snapshots[field]
                if fd.ndim == 3:
                    fd = fd[-1]
                if 'land_mask' in snapshots.files:
                    lm = snapshots['land_mask']
                    if lm.ndim == 3:
                        lm = lm[-1]
                    fd = np.where(lm > 0.5, fd, np.nan)
                all_vals.append(fd.ravel())
            else:
                ffiles = [f for f in snapshots.files if f.startswith(f'{field}_step')]
                if ffiles:
                    steps = [int(f.split('_step')[1]) for f in ffiles]
                    fd = snapshots[f'{field}_step{max(steps)}']
                    all_vals.append(fd.ravel())
        if all_vals:
            combined = np.concatenate(all_vals)
            finite = combined[np.isfinite(combined)]
            if len(finite) > 0:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))

    # Set up grid layout (2x2 for up to 4 grids + space for colorbar)
    n_grids = len(grid_results)
    if n_grids <= 2:
        nrows, ncols = 1, 3  # Extra column for colorbar
    else:
        nrows, ncols = 2, 3  # Extra column for colorbar

    fig = plt.figure(figsize=(14, 10))  # Wider to accommodate colorbar

    # Create subplots with specific width ratios: plots get most space, colorbar gets less
    gs = fig.add_gridspec(nrows, ncols, width_ratios=[1, 1, 0.05] if ncols == 3 else [1, 1, 1, 0.05])

    axes = []
    for i in range(nrows):
        for j in range(ncols - 1):  # Don't include colorbar column
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    # Determine simulation time of the final snapshot from any grid's data
    sim_time_str = ""
    for data in grid_results.values():
        snapshots_any = data['snapshots']
        if 'times_days' in snapshots_any.files:
            t_final = float(snapshots_any['times_days'][-1])
            sim_time_str = f" (t = {t_final:.2f} days)"
            break

    fig.suptitle(f'Final {field.upper()} Snapshots - {test_case_dir.name}{sim_time_str}',
                 fontsize=14, fontweight='bold')

    # Choose colormap based on field
    if field == 'eta':
        cmap = 'RdBu_r'
    elif field in ['SST', 'T']:
        cmap = 'plasma'
    elif field in ['SSS', 'S']:
        cmap = 'viridis'
    elif field in ('w_133m', 'w_sfc'):
        cmap = 'RdBu_r'  # Diverging colormap for vertical velocity (upwelling/downwelling)
    else:
        cmap = 'viridis'

    im = None
    for i, (grid_name, data) in enumerate(grid_results.items()):
        if i >= len(axes):
            break

        ax = axes[i]
        snapshots = data['snapshots']

        # Determine plot extent: prefer source coordinate range (accurate
        # for regional unstructured meshes) over the regridded grid range.
        if 'source_lon_range' in snapshots.files:
            src_lon = snapshots['source_lon_range']
            src_lat = snapshots['source_lat_range']
            plot_extent = [float(src_lon[0]), float(src_lon[1]),
                          float(src_lat[0]), float(src_lat[1])]
        elif 'lon' in snapshots.files:
            lon_arr = snapshots['lon']
            lat_arr = snapshots['lat']
            plot_extent = [float(lon_arr.min()), float(lon_arr.max()),
                          float(lat_arr.min()), float(lat_arr.max())]
        else:
            plot_extent = [0, 360, -90, 90]

        # Handle both new format (field as time series) and old format (field_stepN)
        if field in snapshots.files:
            # NEW FORMAT: field is a time series array, take final timestep
            field_data = snapshots[field]
            if field_data.ndim >= 2:  # At least 2D (could be 3D with time)
                if field_data.ndim == 3:  # Time series: (n_times, nlat, nlon)
                    final_field = field_data[-1]  # Last time
                else:  # Single timestep: (nlat, nlon) - shouldn't happen with new format
                    final_field = field_data

                # Apply land masking if available
                if 'land_mask' in snapshots.files:
                    land_mask_data = snapshots['land_mask']
                    if land_mask_data.ndim == 3:  # Time series
                        land_mask_final = land_mask_data[-1]  # Last time
                    else:  # Single timestep
                        land_mask_final = land_mask_data
                    # Mask land areas (where land_mask ≤ 0.5) with NaN
                    final_field = np.where(land_mask_final > 0.5, final_field, np.nan)

                # Crop regridded array to source domain extent
                if 'lon' in snapshots.files:
                    lat_1d = np.asarray(snapshots['lat'])
                    lon_1d = np.asarray(snapshots['lon'])
                    r0 = max(int(np.searchsorted(lat_1d, plot_extent[2])) - 1, 0)
                    r1 = min(int(np.searchsorted(lat_1d, plot_extent[3])) + 2,
                             len(lat_1d))
                    c0 = max(int(np.searchsorted(lon_1d, plot_extent[0])) - 1, 0)
                    c1 = min(int(np.searchsorted(lon_1d, plot_extent[1])) + 2,
                             len(lon_1d))
                    if (r1 - r0) < final_field.shape[0] or (c1 - c0) < final_field.shape[1]:
                        final_field = final_field[r0:r1, c0:c1]
                        plot_extent = [float(lon_1d[c0]),
                                       float(lon_1d[min(c1, len(lon_1d)-1)]),
                                       float(lat_1d[r0]),
                                       float(lat_1d[min(r1, len(lat_1d)-1)])]

                # Create the plot with consistent color scale
                im = ax.imshow(final_field, origin='lower', aspect='auto', cmap=cmap,
                              extent=plot_extent, vmin=vmin, vmax=vmax)
                ax.set_title(f'{grid_name} ({data["resolution"]})')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            else:
                ax.text(0.5, 0.5, f'{field} wrong shape', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{grid_name} ({data["resolution"]})')
        else:
            # OLD FORMAT: Find final timestep field (fields stored as field_stepN)
            field_files = [f for f in snapshots.files if f.startswith(f'{field}_step')]
            if field_files:
                # Get the highest step number
                step_numbers = [int(f.split('_step')[1]) for f in field_files]
                final_step = max(step_numbers)
                final_field_name = f'{field}_step{final_step}'
                final_field = snapshots[final_field_name]

                # Apply land masking if available (old format)
                land_mask_name = f'land_mask_step{final_step}'
                if land_mask_name in snapshots.files:
                    land_mask_final = snapshots[land_mask_name]
                    # Mask land areas (where land_mask ≤ 0.5) with NaN
                    final_field = np.where(land_mask_final > 0.5, final_field, np.nan)

                # Crop to bounding box of non-NaN data for regional grids
                valid = np.isfinite(final_field)
                if valid.any():
                    rows = np.where(valid.any(axis=1))[0]
                    cols = np.where(valid.any(axis=0))[0]
                    r0, r1 = max(rows[0] - 1, 0), min(rows[-1] + 2, final_field.shape[0])
                    c0, c1 = max(cols[0] - 1, 0), min(cols[-1] + 2, final_field.shape[1])
                    if (r1 - r0) * (c1 - c0) < 0.8 * final_field.size:
                        lat_1d = np.linspace(plot_extent[2], plot_extent[3], final_field.shape[0])
                        lon_1d = np.linspace(plot_extent[0], plot_extent[1], final_field.shape[1])
                        final_field = final_field[r0:r1, c0:c1]
                        plot_extent = [float(lon_1d[c0]), float(lon_1d[min(c1, len(lon_1d)-1)]),
                                       float(lat_1d[r0]), float(lat_1d[min(r1, len(lat_1d)-1)])]

                # Create the plot with consistent color scale
                im = ax.imshow(final_field, origin='lower', aspect='auto', cmap=cmap,
                              extent=plot_extent, vmin=vmin, vmax=vmax)
                ax.set_title(f'{grid_name} ({data["resolution"]})')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
            else:
                ax.text(0.5, 0.5, f'{field} not available', transform=ax.transAxes,
                       ha='center', va='center')
                ax.set_title(f'{grid_name} ({data["resolution"]})')

    # Hide unused subplots
    for i in range(n_grids, len(axes)):
        axes[i].set_visible(False)

    # Add colorbar in dedicated space
    if im is not None:
        # Create colorbar axes in the rightmost column, spanning all rows
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(im, cax=cbar_ax)
        if field == 'eta':
            cbar.set_label('Sea Surface Height (m)')
        elif field in ['SST', 'T']:
            cbar.set_label('Temperature (°C)')
        elif field in ['SSS', 'S']:
            cbar.set_label('Salinity (PSU)')

    from datetime import datetime
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='right', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()

    # Save plot
    output_file = test_case_dir / f"comparison_snapshots_{field}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def _create_comparison_summary(test_case_dir: Path, grid_results: dict) -> None:
    """Create cross-grid metrics comparison table."""

    summary_file = test_case_dir / "comparison_summary.txt"

    with open(summary_file, 'w') as f:
        f.write(f"{test_case_dir.name} - Cross-Grid Comparison\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Grid':<12} {'Status':<8} {'Resolution':<10} {'Wall Time':<10} {'Notes':<30}\n")
        f.write("-" * 80 + "\n")

        # Find metrics for comparison
        eta_drifts = []
        T_drifts = []
        wall_times = []

        for grid_name, data in grid_results.items():
            metadata = data['metadata']
            status = metadata.get('status', 'N/A')
            resolution = data['resolution']
            wall_time = metadata.get('wall_time', 'N/A')
            notes = metadata.get('notes', '')

            f.write(f"{grid_name:<12} {status:<8} {resolution:<10} {wall_time:<10} {notes:<30}\n")

            # Extract drift metrics from notes if available
            if 'eta drift=' in notes:
                try:
                    eta_drift_str = notes.split('eta drift=')[1].split(',')[0].split()[0]
                    eta_drifts.append((grid_name, float(eta_drift_str)))
                except (IndexError, ValueError):
                    pass

            if 'T drift=' in notes:
                try:
                    T_drift_str = notes.split('T drift=')[1].split(',')[0].split()[0]
                    T_drifts.append((grid_name, float(T_drift_str)))
                except (IndexError, ValueError):
                    pass

            if wall_time != 'N/A':
                try:
                    wall_time_val = float(wall_time.replace('s', ''))
                    wall_times.append((grid_name, wall_time_val))
                except (IndexError, ValueError):
                    pass

        f.write("-" * 80 + "\n")

        # Summary statistics
        if eta_drifts:
            best_eta = min(eta_drifts, key=lambda x: abs(x[1]))
            worst_eta = max(eta_drifts, key=lambda x: abs(x[1]))
            f.write(f"Best η drift:    {best_eta[0]} ({best_eta[1]:.2e})\n")
            f.write(f"Worst η drift:   {worst_eta[0]} ({worst_eta[1]:.2e})\n")

        if T_drifts:
            best_T = min(T_drifts, key=lambda x: abs(x[1]))
            worst_T = max(T_drifts, key=lambda x: abs(x[1]))
            f.write(f"Best T drift:    {best_T[0]} ({best_T[1]:.2e})\n")
            f.write(f"Worst T drift:   {worst_T[0]} ({worst_T[1]:.2e})\n")

        if wall_times:
            fastest = min(wall_times, key=lambda x: x[1])
            slowest = max(wall_times, key=lambda x: x[1])
            f.write(f"Fastest:         {fastest[0]} ({fastest[1]:.1f}s)\n")
            f.write(f"Slowest:         {slowest[0]} ({slowest[1]:.1f}s)\n")

    print(f"    Saved: {summary_file.name}")


def _create_comparison_evolution(test_case_dir: Path, grid_results: dict,
                                  field: str = 'eta', max_times: int = 6) -> None:
    """Create a grid-vs-time evolution comparison: rows=grids, cols=time steps.

    Each row is a different grid, each column a snapshot in time.
    All panels share a common colorbar so fields are directly comparable.
    Skipped for rest-state cases where fields don't evolve.
    """
    import matplotlib.pyplot as plt

    # Skip for rest states — nothing evolves
    if test_case_dir.name.startswith("rest_state"):
        return

    test_case = _extract_test_case_name(test_case_dir.name)
    field_ranges = config.FIELD_RANGES.get(test_case, {})

    # Choose colormap
    if field == 'eta':
        cmap = 'RdBu_r'
    elif field in ('SST', 'T'):
        cmap = 'plasma'
    elif field in ('speed_sfc',):
        cmap = 'magma'
    elif field in ('w_133m', 'w_sfc'):
        cmap = 'RdBu_r'  # Diverging colormap for vertical velocity (upwelling/downwelling)
    else:
        cmap = 'viridis'

    grid_names = list(grid_results.keys())
    n_grids = len(grid_names)

    # Determine number of time steps available (use the grid with most)
    n_times_per_grid = {}
    for gname, data in grid_results.items():
        snaps = data['snapshots']
        if field in snaps.files and snaps[field].ndim == 3:
            n_times_per_grid[gname] = snaps[field].shape[0]
        elif 'times_days' in snaps.files:
            # Grid doesn't have this field, but use times from other data
            n_times_per_grid[gname] = len(snaps['times_days'])
        else:
            # Old format: count step files or use a default
            ffiles = [f for f in snaps.files if f.startswith(f'{field}_step')]
            n_times_per_grid[gname] = len(ffiles) if ffiles else max_times

    if not n_times_per_grid:
        return  # No grids at all

    # Ensure we have at least 2 time steps from grids that actually have the field
    has_field_times = [nt for gname, nt in n_times_per_grid.items()
                       if field in grid_results[gname]['snapshots'].files]
    if not has_field_times or max(has_field_times) < 2:
        return  # Need at least 2 time steps for evolution

    n_times_max = max(n_times_per_grid.values())
    n_cols = min(max_times, n_times_max)

    # Collect all fields for shared color range
    all_fields = []  # list of 2D arrays
    grid_field_data = {}  # gname -> list of (time_label, 2D_array or None)
    for gname, data in grid_results.items():
        snaps = data['snapshots']
        times_days = snaps['times_days'] if 'times_days' in snaps.files else None

        if field in snaps.files and snaps[field].ndim == 3:
            field_3d = snaps[field]  # (n_times, nlat, nlon)
            nt = field_3d.shape[0]
            # Select evenly spaced time indices
            if nt > n_cols:
                indices = np.linspace(0, nt - 1, n_cols).astype(int)
            else:
                indices = np.arange(nt)

            entries = []
            for idx in indices:
                f2d = field_3d[idx].copy()
                if 'land_mask' in snaps.files:
                    lm = snaps['land_mask']
                    lm2d = lm[idx] if lm.ndim == 3 else lm
                    f2d = np.where(lm2d > 0.5, f2d, np.nan)
                t_label = f"{times_days[idx]:.1f}d" if times_days is not None else f"t{idx}"
                entries.append((t_label, f2d))
                all_fields.append(f2d)
            grid_field_data[gname] = entries
        else:
            # Field not available - create placeholder entries
            entries = []
            for i in range(n_cols):
                if times_days is not None and len(times_days) > i:
                    t_label = f"{times_days[i]:.1f}d"
                else:
                    t_label = f"t{i}"
                entries.append((t_label, None))  # None indicates missing data
            grid_field_data[gname] = entries

    if not grid_field_data or not all_fields:
        return

    # Actual number of columns (may differ per grid; use max)
    actual_cols = max(len(v) for v in grid_field_data.values())
    actual_grids = grid_names  # Include all grids, even those with missing fields
    n_rows = len(actual_grids)

    fig, axes = plt.subplots(n_rows, actual_cols,
                              figsize=(3.5 * actual_cols, 3.0 * n_rows),
                              squeeze=False)

    for i_row, gname in enumerate(actual_grids):
        entries = grid_field_data[gname]

        # Color range is computed per-panel below so that spatial patterns
        # within each snapshot are visible (temporal trends otherwise
        # dominate the shared range and wash out spatial structure).

        # Get plot extent — prefer source coordinate range for accurate
        # regional domain cropping on unstructured meshes.
        snaps = grid_results[gname]['snapshots']
        if 'source_lon_range' in snaps.files:
            src_lon = snaps['source_lon_range']
            src_lat = snaps['source_lat_range']
            extent = [float(src_lon[0]), float(src_lon[1]),
                      float(src_lat[0]), float(src_lat[1])]
        elif 'lon' in snaps.files:
            lon_arr, lat_arr = snaps['lon'], snaps['lat']
            extent = [float(lon_arr.min()), float(lon_arr.max()),
                      float(lat_arr.min()), float(lat_arr.max())]
        else:
            extent = [0, 360, -90, 90]

        # Crop regridded array to source domain extent
        crop_slices = None
        if 'lon' in snaps.files:
            lat_1d = np.asarray(snaps['lat'])
            lon_1d = np.asarray(snaps['lon'])
            r0 = max(int(np.searchsorted(lat_1d, extent[2])) - 1, 0)
            r1 = min(int(np.searchsorted(lat_1d, extent[3])) + 2,
                     len(lat_1d))
            c0 = max(int(np.searchsorted(lon_1d, extent[0])) - 1, 0)
            c1 = min(int(np.searchsorted(lon_1d, extent[1])) + 2,
                     len(lon_1d))
            # Find a non-None entry to get the shape
            sample_shape = None
            if entries:
                for _, f2d in entries:
                    if f2d is not None:
                        sample_shape = f2d.shape
                        break
            if sample_shape and ((r1 - r0) < sample_shape[0]
                                 or (c1 - c0) < sample_shape[1]):
                crop_slices = (r0, r1, c0, c1)
                extent = [float(lon_1d[c0]),
                          float(lon_1d[min(c1, len(lon_1d)-1)]),
                          float(lat_1d[r0]),
                          float(lat_1d[min(r1, len(lat_1d)-1)])]

        for i_col in range(actual_cols):
            ax = axes[i_row, i_col]
            if i_col < len(entries):
                t_label, f2d = entries[i_col]
                if f2d is not None:
                    # Field data available
                    if crop_slices is not None:
                        r0, r1, c0, c1 = crop_slices
                        f2d = f2d[r0:r1, c0:c1]
                    # Per-panel color range
                    if field in field_ranges and field_ranges[field] != (None, None):
                        p_vmin, p_vmax = field_ranges[field]
                    else:
                        pf = f2d.ravel()
                        pf = pf[np.isfinite(pf)]
                        if pf.size:
                            p_vmin, p_vmax = float(pf.min()), float(pf.max())
                        else:
                            p_vmin, p_vmax = None, None
                    panel_im = ax.imshow(f2d, origin='lower', aspect='auto',
                                         cmap=cmap, extent=extent,
                                         vmin=p_vmin, vmax=p_vmax)
                    fig.colorbar(panel_im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, f'{field} not available',
                            transform=ax.transAxes,
                            ha='center', va='center', fontsize=9)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                if i_row == 0:
                    ax.set_title(t_label, fontsize=9)
                if i_col == 0:
                    res = grid_results[gname]['resolution']
                    ax.set_ylabel(f"{gname}\n({res})", fontsize=9)
                else:
                    ax.set_ylabel("")
                ax.set_xlabel("")
                ax.tick_params(labelsize=7)
            else:
                ax.set_visible(False)

    fig.suptitle(f"{field.upper()} Evolution — {test_case_dir.name}",
                 fontsize=13, fontweight='bold')

    fig.tight_layout()
    out = test_case_dir / f"comparison_evolution_{field}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.name}")


def _create_comparison_vertical_section(test_case_dir: Path, grid_results: dict) -> None:
    """Create meridional vertical cross-section comparison for baroclinic gyre."""
    import numpy as np

    # Check that we have T_3d data for all grids
    grids_with_T3d = {}
    for grid_name, data in grid_results.items():
        snapshots = data['snapshots']
        if 'T_3d' in snapshots.files:
            grids_with_T3d[grid_name] = data

    if len(grids_with_T3d) < 2:
        return

    # Domain parameters for baroclinic gyre (60° longitude × 60° latitude)
    lon_middle = 60.0  # Middle of 0-120° domain
    lat_range = (-90, 90)  # Full latitude range for averaging

    fig, axes = plt.subplots(1, len(grids_with_T3d), figsize=(5 * len(grids_with_T3d), 6),
                             sharey=True)
    if len(grids_with_T3d) == 1:
        axes = [axes]

    for idx, (grid_name, data) in enumerate(grids_with_T3d.items()):
        ax = axes[idx]
        snapshots = data['snapshots']

        # Load final temperature field (use last time step)
        T_3d_all = np.asarray(snapshots['T_3d'], dtype=np.float64)
        lon_deg = np.asarray(snapshots['lon'], dtype=np.float64)
        lat_deg = np.asarray(snapshots['lat'], dtype=np.float64)

        # Handle different data structures - T_3d is (time, lat, lon, lev)
        if T_3d_all.ndim == 4:  # (time, lat, lon, lev)
            T_3d = T_3d_all[-1]  # Take final time step → (lat, lon, lev)
        elif T_3d_all.ndim == 3:  # (lat, lon, lev) - already final state
            T_3d = T_3d_all
        else:
            print(f"Warning: unexpected T_3d shape {T_3d_all.shape} for {grid_name}")
            continue

        # Get depth levels from the first grid (they should be the same)
        if idx == 0:
            # Try to get depth data from results.txt or default levels
            results_file = test_case_dir / grid_name / 'results.txt'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    content = f.read()
                    # Look for depth line
                    for line in content.split('\n'):
                        if line.startswith('depth:'):
                            import ast
                            depth_str = line.split(':', 1)[1].strip()
                            depth_data = np.array(ast.literal_eval(depth_str))
                            break
                    else:
                        # Fallback to default depth levels
                        depth_data = np.array([26.19, 133.86, 352.12, 680.95, 1120.37,
                                             1670.37, 2330.95, 3102.12, 3983.86, 4976.19])
            else:
                # Fallback to default depth levels
                depth_data = np.array([26.19, 133.86, 352.12, 680.95, 1120.37,
                                     1670.37, 2330.95, 3102.12, 3983.86, 4976.19])

        # Extract meridional section at domain middle (longitude = 60°)
        # Find closest longitude index to domain middle
        lon_idx = np.argmin(np.abs(lon_deg - lon_middle))

        # Average over a few longitude points for smoother section
        lon_indices = slice(max(0, lon_idx-2), min(len(lon_deg), lon_idx+3))
        T_section = np.nanmean(T_3d[:, lon_indices, :], axis=1)  # T_3d is (lat, lon, lev) → (lat, lev)

        # Apply land mask if available (simplified for now)
        if 'land_mask' in snapshots.files:
            land_mask = np.asarray(snapshots['land_mask'], dtype=np.float64)
            # For now, skip land masking to get basic functionality working
            # TODO: Fix land mask broadcasting for vertical sections
            pass

        # Compute depth interfaces for proper plotting
        def compute_level_interfaces(level_centers):
            if len(level_centers) == 1:
                return np.array([0.0, 2 * level_centers[0]])
            interfaces = np.zeros(len(level_centers) + 1)
            interfaces[0] = 0.0
            interfaces[1:-1] = 0.5 * (level_centers[:-1] + level_centers[1:])
            interfaces[-1] = level_centers[-1] + (level_centers[-1] - interfaces[-2])
            return interfaces

        level_interfaces = compute_level_interfaces(depth_data)

        # Create coordinate meshes for pcolormesh
        # Ensure lat_interfaces matches the T_section shape
        if len(lat_deg) > 1:
            lat_step = (lat_deg[-1] - lat_deg[0]) / (len(lat_deg) - 1)
            lat_interfaces = np.linspace(lat_deg[0] - 0.5 * lat_step,
                                       lat_deg[-1] + 0.5 * lat_step,
                                       len(lat_deg) + 1)
        else:
            lat_interfaces = np.array([lat_deg[0] - 1.0, lat_deg[0] + 1.0])

        X, Y = np.meshgrid(lat_interfaces, level_interfaces)

        # Verify dimensions match for pcolormesh
        expected_lat_size = len(lat_interfaces) - 1  # pcolormesh expects one less than interfaces
        expected_lev_size = len(level_interfaces) - 1
        if T_section.shape != (expected_lat_size, expected_lev_size):
            print(f"Warning: T_section shape {T_section.shape} doesn't match expected "
                  f"({expected_lat_size}, {expected_lev_size}) for {grid_name}")
            continue

        # Plot with adaptive colormap range
        # For baroclinic_gyre, use data-adaptive range to show circulation patterns
        T_finite = T_section.T[np.isfinite(T_section.T)]
        if len(T_finite) > 0:
            vmin, vmax = float(np.nanmin(T_finite)), float(np.nanmax(T_finite))
        else:
            vmin, vmax = 2, 20  # fallback for edge cases

        im = ax.pcolormesh(X, Y, T_section.T, cmap='RdYlBu_r',
                          vmin=vmin, vmax=vmax, shading='flat')

        ax.set_title(f'{grid_name}', fontsize=12)
        ax.set_xlabel('Latitude (°)')
        if idx == 0:
            ax.set_ylabel('Depth (m)')
        ax.set_ylim(0, level_interfaces[-1])  # Set y-limits to actual depth range
        ax.invert_yaxis()  # Surface at top, deep at bottom
        ax.grid(True, alpha=0.3)

        # Set x-limits to regional domain extent for consistent comparison
        # Use the actual experiment domain, not the interpolation grid extent
        regional_lat_min, regional_lat_max = 15.0, 75.0  # Baroclinic gyre domain
        ax.set_xlim(regional_lat_min, regional_lat_max)

        # Add domain boundaries for regional grids
        if 'regional' in grid_name:
            ax.axvline(15, color='white', linewidth=2, linestyle='--', alpha=0.8)
            ax.axvline(75, color='white', linewidth=2, linestyle='--', alpha=0.8)

    # Add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Temperature (°C)', fontsize=12)

    fig.suptitle(f'{test_case_dir.name} — Meridional Temperature Section (60°E)',
                 fontsize=14, y=0.95)

    # Save the plot
    out_file = test_case_dir / 'comparison_vertical_section.png'
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_file.name}")


def _create_comparison_vertical_evolution(
    test_case_dir: Path, grid_results: dict, max_times: int = 6,
) -> None:
    """Meridional T cross-section evolution: rows = grids, cols = time steps.

    Each panel shows the latitude–depth temperature section at 60°E,
    with a per-panel colorbar so that spatial patterns within each
    snapshot are visible even when the full-depth stratification
    dominates the overall range.
    """
    import matplotlib.pyplot as plt

    # Skip rest-state cases
    if test_case_dir.name.startswith("rest_state"):
        return

    # Gather grids that have T_3d
    grids_with_T3d = {}
    for gname, data in grid_results.items():
        snaps = data["snapshots"]
        if "T_3d" in snaps.files:
            grids_with_T3d[gname] = data
    if len(grids_with_T3d) < 1:
        return

    # Read depth levels from results.txt of first grid
    first_grid = next(iter(grids_with_T3d))
    results_file = (test_case_dir / first_grid / "results.txt")
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                if line.startswith("depth:"):
                    import ast as _ast
                    depth_data = np.array(
                        _ast.literal_eval(line.split(":", 1)[1].strip()))
                    break
            else:
                depth_data = np.array([26.19, 133.86, 352.12, 680.95,
                                       1120.37, 1670.37, 2330.95, 3102.12,
                                       3983.86, 4976.19])
    else:
        depth_data = np.array([26.19, 133.86, 352.12, 680.95, 1120.37,
                               1670.37, 2330.95, 3102.12, 3983.86, 4976.19])

    # Level interfaces for pcolormesh
    def _level_interfaces(centers):
        if len(centers) == 1:
            return np.array([0.0, 2 * centers[0]])
        ifc = np.zeros(len(centers) + 1)
        ifc[0] = 0.0
        ifc[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        ifc[-1] = centers[-1] + (centers[-1] - ifc[-2])
        return ifc

    level_ifc = _level_interfaces(depth_data)
    lon_middle = 60.0

    grid_names = list(grids_with_T3d.keys())
    n_grids = len(grid_names)

    # Determine number of time columns
    n_times_max = max(
        grids_with_T3d[g]["snapshots"]["T_3d"].shape[0]
        for g in grid_names
        if grids_with_T3d[g]["snapshots"]["T_3d"].ndim == 4)
    n_cols = min(max_times, n_times_max)

    fig, axes = plt.subplots(n_grids, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_grids),
                              squeeze=False)

    for i_row, gname in enumerate(grid_names):
        snaps = grids_with_T3d[gname]["snapshots"]
        T_3d_all = np.asarray(snaps["T_3d"], dtype=np.float64)
        lon_deg = np.asarray(snaps["lon"], dtype=np.float64)
        lat_deg = np.asarray(snaps["lat"], dtype=np.float64)
        times_days = (np.asarray(snaps["times_days"], dtype=np.float64)
                      if "times_days" in snaps.files else None)

        if T_3d_all.ndim != 4:
            continue  # need (time, lat, lon, lev)
        nt = T_3d_all.shape[0]
        indices = (np.linspace(0, nt - 1, n_cols).astype(int)
                   if nt > n_cols else np.arange(nt))

        # Longitude slice
        lon_idx = int(np.argmin(np.abs(lon_deg - lon_middle)))
        lon_sl = slice(max(0, lon_idx - 2), min(len(lon_deg), lon_idx + 3))

        # Latitude interfaces
        if len(lat_deg) > 1:
            dlat = (lat_deg[-1] - lat_deg[0]) / (len(lat_deg) - 1)
            lat_ifc = np.linspace(lat_deg[0] - 0.5 * dlat,
                                  lat_deg[-1] + 0.5 * dlat,
                                  len(lat_deg) + 1)
        else:
            lat_ifc = np.array([lat_deg[0] - 1.0, lat_deg[0] + 1.0])

        X, Y = np.meshgrid(lat_ifc, level_ifc)

        # Use the initial condition (t=0) range for all panels so that
        # any departure from the initial stratification is visible.
        T_ic = T_3d_all[0]  # (lat, lon, lev)
        T_sec_ic = np.nanmean(T_ic[:, lon_sl, :], axis=1)
        ic_finite = T_sec_ic[np.isfinite(T_sec_ic)]
        if ic_finite.size:
            row_vmin, row_vmax = float(ic_finite.min()), float(ic_finite.max())
        else:
            row_vmin, row_vmax = 2, 20

        for i_col, ti in enumerate(indices):
            ax = axes[i_row, i_col]
            T_3d = T_3d_all[ti]  # (lat, lon, lev)
            T_sec = np.nanmean(T_3d[:, lon_sl, :], axis=1)  # (lat, lev)

            if T_sec.shape != (len(lat_ifc) - 1, len(level_ifc) - 1):
                ax.set_visible(False)
                continue

            im = ax.pcolormesh(X, Y, T_sec.T, cmap="RdYlBu_r",
                               vmin=row_vmin, vmax=row_vmax, shading="flat")
            ax.invert_yaxis()
            ax.set_xlim(15, 75)
            ax.set_ylim(level_ifc[-1], 0)
            ax.tick_params(labelsize=7)

            if i_row == 0:
                t_label = (f"{times_days[ti]:.1f}d"
                           if times_days is not None else f"t{ti}")
                ax.set_title(t_label, fontsize=9)
            if i_col == 0:
                res = grid_results[gname]["resolution"]
                ax.set_ylabel(f"{gname}\n({res})\nDepth (m)", fontsize=9)
            else:
                ax.set_ylabel("")
            if i_row == n_grids - 1:
                ax.set_xlabel("Latitude (°)", fontsize=8)
            else:
                ax.set_xlabel("")

        # Hide unused columns
        for i_col in range(len(indices), n_cols):
            axes[i_row, i_col].set_visible(False)

        # One shared colorbar per row (all panels use the IC range)
        if im is not None:
            row_axes = [axes[i_row, c] for c in range(n_cols)
                        if axes[i_row, c].get_visible()]
            fig.colorbar(im, ax=row_axes, fraction=0.02, pad=0.02,
                         label="T (°C)")

    fig.suptitle(
        f"T Section Evolution at 60°E — {test_case_dir.name}",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = test_case_dir / "comparison_evolution_vertical_section.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.name}")


def _create_cross_grid_comparisons(test_case_dir: Path, grid_results: dict) -> None:
    """Create all cross-grid comparison plots and summary for a test case."""
    if len(grid_results) < 2:
        return  # Need at least 2 grids for comparison

    print(f"  Creating cross-grid comparisons for {test_case_dir.name}...")

    # Time series comparison
    _create_comparison_timeseries(test_case_dir, grid_results)

    # Final snapshot comparisons
    for field in ['eta', 'SST', 'w_133m']:
        field_available = any(
            field in data['snapshots'].files or
            any(f.startswith(f'{field}_step') for f in data['snapshots'].files)
            for data in grid_results.values()
        )
        if field_available:
            _create_comparison_snapshots(test_case_dir, grid_results, field)
            _create_comparison_evolution(test_case_dir, grid_results, field)

    # Forcing profile plot for wind-driven cases
    if 'barotropic_wind' in test_case_dir.name:
        _save_forcing_profile(test_case_dir)

    # Vertical cross-section comparison for baroclinic gyre
    if 'baroclinic_gyre' in test_case_dir.name:
        _create_comparison_vertical_section(test_case_dir, grid_results)
        _create_comparison_vertical_evolution(test_case_dir, grid_results)

    # Summary table
    _create_comparison_summary(test_case_dir, grid_results)


def _save_forcing_profile(test_case_dir: Path) -> None:
    """Save a wind forcing profile plot (tau_x vs latitude)."""
    import matplotlib.pyplot as plt

    lat_deg = np.linspace(-90, 90, 361)
    lat_rad = lat_deg * np.pi / 180.0
    s2 = np.sin(lat_rad) ** 2
    tau_x = (-0.08 - 0.0397 * s2 + 1.9487 * s2**2 - 2.0397 * s2**3) * np.cos(lat_rad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Prescribed Wind Forcing (Nikurashin & Vallis style)',
                 fontsize=14, fontweight='bold')

    # Panel 1: Wind stress profile
    axes[0].plot(lat_deg, tau_x, 'b-', linewidth=2)
    axes[0].axhline(0, color='k', linewidth=0.5, linestyle=':')
    axes[0].axvspan(-90, -55, alpha=0.1, color='gray', label='Drake Passage open')
    axes[0].set_xlabel('Latitude (deg)')
    axes[0].set_ylabel(r'$\tau_x$ [Pa]')
    axes[0].set_title('Zonal Wind Stress')
    axes[0].set_xlim(-80, 80)
    axes[0].set_ylim(-0.12, 0.12)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('Trades', xy=(0, -0.065), ha='center', fontsize=9, color='blue')
    axes[0].annotate('Westerlies', xy=(50, 0.085), ha='center', fontsize=9, color='red')
    axes[0].annotate('Westerlies', xy=(-50, 0.085), ha='center', fontsize=9, color='red')

    # Panel 2: Wind stress curl (proportional to Sverdrup transport)
    R = constants.R_earth
    dtau_dlat = np.gradient(tau_x, lat_rad)
    curl_z = dtau_dlat / R
    axes[1].plot(lat_deg, curl_z * 1e7, 'b-', linewidth=2)
    axes[1].axhline(0, color='k', linewidth=0.5, linestyle=':')
    axes[1].axvspan(-90, -55, alpha=0.1, color='gray')
    axes[1].set_xlabel('Latitude (deg)')
    axes[1].set_ylabel(r'curl$_z(\tau)$ ($\times 10^{-7}$ N/m$^3$)')
    axes[1].set_title('Wind Stress Curl')
    axes[1].set_xlim(-80, 80)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = test_case_dir / "forcing_profile.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file.name}")


def _check_and_generate_comparisons(output_base: Path, test_case_name: str, all_results: list) -> None:
    """Check if all grids completed for a test case and generate cross-grid comparisons."""
    # Find all results for this test case
    test_results = [r for r in all_results if r['test'] == test_case_name]

    if len(test_results) < 2:
        return  # Need at least 2 grids for comparison

    # Check if we have results for multiple grids
    grid_types = set(r['grid'] for r in test_results)
    if len(grid_types) < 2:
        return  # Need different grids, not just multiple resolutions

    # Grouped variants are nested under their parent folder
    group = TestCase._CASE_TO_GROUP.get(test_case_name)
    if group is not None:
        test_case_dir = output_base / group / test_case_name
    else:
        test_case_dir = output_base / test_case_name
    if not test_case_dir.exists():
        return

    # Try to collect grid results
    grid_results = _collect_grid_results(test_case_dir)
    if len(grid_results) > 1:
        print("\n" + "-" * 60)
        print(f"  CROSS-GRID COMPARISON: {test_case_name}")
        print("-" * 60)
        _create_cross_grid_comparisons(test_case_dir, grid_results)
        print("-" * 60)
    else:
        print(f"  Note: Insufficient grid data for {test_case_name} cross-grid comparison")


# ===========================================================================
# Replot functions
# ===========================================================================


def _replot_case_snapshots(case_dir: Path) -> None:
    """Regenerate per-case snapshot plots from saved NPZ data.

    Reads ``snapshots_latlon.npz`` (regridded) and ``results.txt``
    to reconstruct the snapshot evolution plots, cross-sections, and
    vertical profiles without rerunning the simulation.
    """
    npz_path = case_dir / "snapshots_latlon.npz"
    results_path = case_dir / "results.txt"
    if not npz_path.exists():
        return

    data = np.load(npz_path)
    if "times_days" not in data.files:
        return

    times = data["times_days"]
    lat = data["lat"]
    lon = data["lon"]
    n_times = len(times)

    # Parse metadata from results.txt
    meta = {}
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    meta[k.strip()] = v.strip()
    dt_val = float(meta.get("dt", config.DEFAULT_DT))
    grid_type = meta.get("grid", "")
    resolution = meta.get("resolution", "")

    # Determine coord_kind for the regridded data — it's always on a
    # regular lat-lon grid after regridding, so we use "latlon".
    coord_kind = "latlon"

    # Build field specs from what's available
    field_specs_2d = []
    if "eta" in data.files:
        field_specs_2d.append(("eta", "SSH (m)", "RdBu_r"))
    if "SST" in data.files:
        field_specs_2d.append(("SST", "SST (degC)", "RdYlBu_r"))
    if "speed_sfc" in data.files:
        field_specs_2d.append(("speed_sfc", "Surface speed (m/s)", "magma"))
    if "u_sfc" in data.files:
        field_specs_2d.append(("u_sfc", "Zonal velocity (m/s)", "RdBu_r"))
    if "v_sfc" in data.files:
        field_specs_2d.append(("v_sfc", "Meridional velocity (m/s)", "RdBu_r"))
    if "w_133m" in data.files:
        field_specs_2d.append(("w_133m", "w at 134m (m/s)", "RdBu_r"))

    if not field_specs_2d:
        return

    # Reconstruct the snapshots dict expected by _save_snapshot_plots.
    # The NPZ stores (n_times, lat, lon) arrays.  The plotting code
    # expects  snapshots = {step: {field: 2D_array, ...}, ...}
    # with step numbers as keys.  We fake step numbers from dt.
    snapshots = {}
    steps_arr = data["steps"] if "steps" in data.files else np.arange(n_times)
    for i, step in enumerate(steps_arr):
        step = int(step)
        snap = {}
        for fk, _, _ in field_specs_2d:
            if fk in data.files:
                arr = data[fk]
                if arr.ndim == 3:  # (time, lat, lon)
                    snap[fk] = arr[i]
                elif arr.ndim == 2:
                    snap[fk] = arr
        if "land_mask" in data.files:
            lm = data["land_mask"]
            snap["land_mask"] = lm[i] if lm.ndim == 3 else lm
        # Velocity vectors for quiver overlay
        for vk in ("u_sfc", "v_sfc"):
            if vk in data.files:
                va = data[vk]
                snap[vk] = va[i] if va.ndim == 3 else va
        snapshots[step] = snap

    # Domain extent
    domain_extent = None
    if "source_lon_range" in data.files and "source_lat_range" in data.files:
        slon = data["source_lon_range"]
        slat = data["source_lat_range"]
        domain_extent = (float(slon[0]), float(slon[1]),
                         float(slat[0]), float(slat[1]))

    case_label = f"{case_dir.parent.name} {grid_type} {resolution}".strip()

    # Regenerate snapshot plots (regridded data is already on lat-lon)
    _save_snapshot_plots(
        case_dir, case_label, snapshots, dt_val, field_specs_2d,
        coord_kind, lon, lat, domain_extent=domain_extent)

    # Regenerate cross-sections and profiles if 3D data exists
    if "T_3d" in data.files:
        import ast as _ast
        T_3d_all = data["T_3d"]  # (time, lat, lon, lev)
        depth_data = None
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    if line.startswith("depth:"):
                        depth_data = np.array(
                            _ast.literal_eval(line.split(":", 1)[1].strip()))
                        break
        if depth_data is not None:
            # Add T_3d to snapshots
            for i, step in enumerate(steps_arr):
                step = int(step)
                if step in snapshots and T_3d_all.ndim == 4:
                    snapshots[step]["T_3d"] = T_3d_all[i]

            _save_cross_sections(
                case_dir, case_label, snapshots, dt_val, "T_3d",
                coord_kind, lon, lat, depth_data, "Depth (m)")
            _save_profiles(
                case_dir, case_label, snapshots, dt_val, "T_3d",
                depth_data, "Depth (m)")


def _run_replot(args) -> None:
    """Replot mode: regenerate all plots from existing NPZ data."""
    output_base = Path(args.output)
    if not output_base.exists():
        print(f"Output directory {output_base} does not exist.")
        return

    print("=" * 78)
    print("  legoESM Ocean Test Matrix — REPLOT MODE")
    print("=" * 78)
    print(f"  Output:     {output_base}")
    print(f"  Filter:     --only {args.only}  --grid {args.grid}")
    print("=" * 78)

    t_start = time.time()

    # Discover all test case directories with results
    test_case_dirs = set()
    replotted = 0

    for case_dir in sorted(output_base.rglob("snapshots_latlon.npz")):
        res_dir = case_dir.parent        # e.g., results/ocean/baroclinic_gyre/latlon_regional/24x48
        grid_dir = res_dir.parent         # e.g., results/ocean/baroclinic_gyre/latlon_regional
        test_dir = grid_dir.parent        # e.g., results/ocean/baroclinic_gyre

        grid_name = grid_dir.name
        test_name = test_dir.name

        # Apply filters
        if args.only != "all" and args.only not in test_name:
            continue
        if args.grid != "all" and args.grid != grid_name:
            continue

        print(f"  Replotting {test_name}/{grid_name}/{res_dir.name} ...")
        try:
            _replot_case_snapshots(res_dir)
            replotted += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()

        test_case_dirs.add(test_dir)

    # Regenerate cross-grid comparisons
    if test_case_dirs:
        print()
        print("=" * 78)
        print("  REGENERATING CROSS-GRID COMPARISONS")
        print("=" * 78)
        for test_dir in sorted(test_case_dirs):
            grid_results = _collect_grid_results(test_dir)
            if len(grid_results) >= 2:
                _create_cross_grid_comparisons(test_dir, grid_results)
            elif len(grid_results) == 1:
                print(f"  Skipping {test_dir.name}: only 1 grid available")

    # Rest-state cross-variant comparison
    _create_rest_state_cross_variant_comparison(output_base)

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Replotted {replotted} case(s) in {elapsed:.1f}s")
    print("=" * 78)


# ===========================================================================
# Rest-state cross-variant comparison
# ===========================================================================

REST_STATE_VARIANTS_LIST = [
    "rest_state_stratified_with_land",
    "rest_state_uniform_with_land",
    "rest_state_stratified_no_land",
    "rest_state_uniform_no_land",
]

REST_STATE_VARIANT_LABELS = {
    "rest_state_stratified_with_land": "Stratified + Land",
    "rest_state_uniform_with_land":    "Uniform T/S + Land",
    "rest_state_stratified_no_land":   "Stratified, No Land",
    "rest_state_uniform_no_land":      "Uniform T/S, No Land",
}

REST_STATE_VARIANT_COLORS = {
    "rest_state_stratified_with_land": "red",
    "rest_state_uniform_with_land":    "orange",
    "rest_state_stratified_no_land":   "blue",
    "rest_state_uniform_no_land":      "green",
}


def _create_rest_state_cross_variant_comparison(output_base: Path):
    """Create a combined comparison plot across all rest-state variants and grids.

    Produces a (3 rows × 3 cols) figure:
    - Rows: grids (cubed_sphere, latlon, mpas)
    - Cols: eta drift, T drift, S drift
    Each panel overlays the 4 rest-state variants as different colored lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = ["cubed_sphere", "latlon", "mpas"]

    # Collect conservation timeseries for all variants × grids
    data = {}  # (variant, grid) -> DataFrame
    rest_base = output_base / "rest_state"
    for variant in REST_STATE_VARIANTS_LIST:
        for grid in grids:
            # Find the resolution dir (rest states grouped under rest_state/)
            variant_dir = rest_base / variant / grid
            if not variant_dir.exists():
                continue
            res_dirs = [d for d in variant_dir.iterdir() if d.is_dir()]
            if not res_dirs:
                continue
            csv_file = res_dirs[0] / "conservation_timeseries.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    data[(variant, grid)] = df
                except Exception:
                    continue

    if not data:
        print("  No rest-state conservation data found for cross-variant comparison.")
        return

    print("\n" + "=" * 78)
    print("  REST-STATE CROSS-VARIANT COMPARISON")
    print("=" * 78)

    fig, axes = plt.subplots(len(grids), 3, figsize=(15, 3.5 * len(grids)),
                              sharex=True)
    if len(grids) == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Volume (eta) relative drift", "Heat (T) relative drift",
                  "Salt (S) relative drift"]
    col_keys = ["vol_rel", "heat_rel", "salt_rel"]

    for i_grid, grid in enumerate(grids):
        for i_col, (col_key, col_label) in enumerate(zip(col_keys, col_labels)):
            ax = axes[i_grid, i_col]
            any_plotted = False
            for variant in REST_STATE_VARIANTS_LIST:
                key = (variant, grid)
                if key not in data:
                    continue
                df = data[key]
                if col_key not in df.columns:
                    continue
                label = REST_STATE_VARIANT_LABELS.get(variant, variant)
                color = REST_STATE_VARIANT_COLORS.get(variant, "gray")
                ax.plot(df["time_days"], df[col_key], label=label, color=color,
                        linewidth=1.5)
                any_plotted = True

            ax.set_ylabel(col_label if i_grid == 0 else "")
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))
            ax.grid(True, alpha=0.3)
            if i_grid == 0:
                ax.set_title(col_label, fontsize=10)
            if i_grid == len(grids) - 1:
                ax.set_xlabel("Time (days)")
            if i_col == 0:
                ax.text(-0.25, 0.5, grid, transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='center',
                        rotation=90)
            if any_plotted and i_grid == 0 and i_col == 2:
                ax.legend(fontsize=7, loc='upper left')

    fig.suptitle("Rest-State Cross-Variant Comparison\n"
                 "(4 variants × 3 grids, conservation drift)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])

    rest_base.mkdir(parents=True, exist_ok=True)
    out_path = rest_base / "cross_variant_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also create a summary table
    summary_path = rest_base / "cross_variant_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Rest-State Cross-Variant Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Variant':<35} {'Grid':<15} {'eta drift':>12} {'T drift':>12} {'S drift':>12}\n")
        f.write("-" * 86 + "\n")
        for variant in REST_STATE_VARIANTS_LIST:
            for grid in grids:
                key = (variant, grid)
                if key not in data:
                    continue
                df = data[key]
                last = df.iloc[-1]
                eta_d = last.get("vol_rel", 0)
                t_d = last.get("heat_rel", 0)
                s_d = last.get("salt_rel", 0)
                label = REST_STATE_VARIANT_LABELS.get(variant, variant)
                f.write(f"{label:<35} {grid:<15} {eta_d:>12.2e} {t_d:>12.2e} {s_d:>12.2e}\n")
            f.write("\n")
    print(f"  Saved: {summary_path}")
