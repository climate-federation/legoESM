#!/usr/bin/env python
"""Yearly AMIP map plotter.

Reads a ``monthly_means.npz`` written by
:class:`legoesm.diagnostics.monthly_means.SpatialMonthlyAccumulator`
(the file ``ModelDriver`` writes when ``OutputConfig.monthly_means=True``)
and produces one PNG per simulated year covering the canonical
AMIP-validation fields: SST/T_sfc, near-surface T (lowest model level),
total precipitation, OLR/TOA longwave, and total cloud cover.

Why this script
~~~~~~~~~~~~~~~
``scripts/plot/plot_amip.py`` already exists but it only plots
scalar timeseries + zonal-mean cross-sections.  Spatial yearly maps
were missing — Stage 4 of the lat-lon MPI AMIP work.  This plotter:

* averages 12 consecutive months → one yearly map per field per year;
* skips years where any month is missing (so partial first / last
  years don't pollute the gallery);
* uses a CMIP-style diverging palette for anomalies relative to the
  first complete year, plus a sequential palette for absolute fields;
* writes one PNG per year with a 2x3 panel grid (SST, T_low, precip,
  OLR, cloud, p_s).

Usage
~~~~~
    python scripts/plot/plot_amip_yearly_maps.py \\
        --input results/amip_2025/monthly_means.npz \\
        --output results/amip_2025/yearly_maps \\
        --start-year 1979

Outputs::

    results/amip_2025/yearly_maps/amip_y1979.png
    results/amip_2025/yearly_maps/amip_y1980.png
    ...

The plotter is agnostic to the underlying grid (lat-lon, cubed-sphere
remapped to lat-lon, etc.) — it just reads the (n_months, nlat, nlon)
arrays the accumulator wrote.  For lat-lon FV it shows the native
grid; for other grids the accumulator already does the remap.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless — no DISPLAY on Ginsburg compute nodes
import matplotlib.pyplot as plt
import numpy as np


# Canonical AMIP-validation panels.  Each entry maps the npz-key
# (``field_2d_<name>``) → (display_title, cmap, units, vmin, vmax).
# vmin/vmax = None means use the data range (with optional symmetric
# clip around the mean — see ``_safe_minmax``).
_PANELS: list[tuple[str, str, str, str, float | None, float | None]] = [
    ("field_2d_T_sfc",     "Surface T (K)",      "viridis",   "K",       240.0, 310.0),
    ("field_2d_T_lowest",  "Low-level T (K)",    "viridis",   "K",       240.0, 310.0),
    ("field_2d_precip",    "Precipitation",      "Blues",     "mm/day",    0.0,  20.0),
    ("field_2d_OLR",       "OLR (TOA LW)",       "magma_r",   "W/m²",    180.0, 320.0),
    ("field_2d_cld_frac",  "Total cloud cover",  "Greys_r",   "[0, 1]",    0.0,   1.0),
    ("field_2d_p_s",       "Surface pressure",   "RdBu_r",    "hPa",     960.0,1040.0),
]


def _safe_minmax(arr, lo=None, hi=None):
    """Pick a finite (vmin, vmax) for matplotlib pcolormesh.

    If ``lo``/``hi`` are passed, use them.  Otherwise fall back to the
    data's [2%, 98%] percentile so a stray NaN-corner or polar
    singularity doesn't blow out the colour bar.
    """
    if lo is not None and hi is not None:
        return float(lo), float(hi)
    a = np.asarray(arr)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    return float(np.percentile(a, 2)), float(np.percentile(a, 98))


def _read_lat_lon(npz):
    """Recover the lat / lon coordinate axes from the npz.

    The SpatialMonthlyAccumulator stores ``lat`` (1-D, n_lat) and
    ``lon`` (1-D, n_lon).  Older runs may have only ``lat`` (the
    zonal-mean accumulator) — those raise here with a clear message
    so the caller can switch to plot_amip.py's timeseries plotter.
    """
    if "lat" not in npz.files:
        raise ValueError(
            "monthly_means.npz has no 'lat' axis — this file was "
            "written by the zonal-mean MonthlyAccumulator, not the "
            "SpatialMonthlyAccumulator.  Yearly map plotting requires "
            "the spatial variant; enable it in the driver with "
            "OutputConfig(monthly_means=True, spatial_monthly_means=True)."
        )
    lat = np.asarray(npz["lat"])
    if "lon" not in npz.files:
        raise ValueError(
            "monthly_means.npz has 'lat' but no 'lon' — likely the "
            "zonal-mean accumulator.  See the 'lat' error message."
        )
    lon = np.asarray(npz["lon"])
    return lat, lon


def _yearly_mean(monthly_field, years, month_nums, target_year):
    """Average all 12 months of ``target_year``.  Returns ``None`` when
    any of those months is missing."""
    mask = years == target_year
    if int(mask.sum()) != 12:
        return None
    arr = np.asarray(monthly_field[mask])  # (12, nlat, nlon)
    # nan-aware mean so a missing-data month doesn't poison the year
    return np.nanmean(arr, axis=0)


def _plot_year(year_arr, panels, lat, lon, year, out_path):
    """Render one PNG for a single year using up to ``_PANELS`` fields.

    ``year_arr`` maps panel-key → (nlat, nlon) yearly mean (or
    ``None`` if the field is absent from the npz)."""
    n_panels = sum(1 for _, _, _, _, _, _ in panels)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )
    fig.suptitle(f"AMIP yearly mean, year {year}", fontsize=14)

    for ax_idx, (key, title, cmap, units, vmin, vmax) in enumerate(panels):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        field = year_arr.get(key)
        if field is None:
            ax.set_title(f"{title}\n(not in output)")
            ax.set_axis_off()
            continue
        f = field
        if key == "field_2d_p_s":
            f = field / 100.0  # Pa → hPa for display only
        lo, hi = _safe_minmax(f, vmin, vmax)
        im = ax.pcolormesh(
            np.degrees(lon), np.degrees(lat),
            f, cmap=cmap, vmin=lo, vmax=hi, shading="auto",
        )
        ax.set_title(f"{title}\nmin={np.nanmin(f):.2f} max={np.nanmax(f):.2f}")
        ax.set_xlabel("lon [°]")
        ax.set_ylabel("lat [°]")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label(units)

    # Hide any empty panels in the last row
    for k in range(n_panels, nrows * ncols):
        axes[k // ncols, k % ncols].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to monthly_means.npz")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="Output directory for the per-year PNGs")
    parser.add_argument("--start-year", type=int, default=None,
                        help="If the npz lacks a 'years' array, assume "
                             "the first 12 months belong to this year")
    parser.add_argument("--year", type=int, default=None,
                        help="If given, plot only this one year")
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input {args.input} not found", file=sys.stderr)
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    npz = np.load(args.input, allow_pickle=True)

    lat, lon = _read_lat_lon(npz)

    # Resolve the year/month coordinates.
    if "years" in npz.files and "month_nums" in npz.files:
        years = np.asarray(npz["years"])
        month_nums = np.asarray(npz["month_nums"])
    elif args.start_year is not None:
        n_months = None
        for key, *_ in _PANELS:
            if key in npz.files:
                n_months = npz[key].shape[0]
                break
        if n_months is None:
            print("ERROR: no spatial fields found in the npz", file=sys.stderr)
            sys.exit(2)
        years = np.repeat(
            np.arange(args.start_year, args.start_year + (n_months + 11) // 12),
            12,
        )[:n_months]
        month_nums = np.tile(np.arange(1, 13), (n_months + 11) // 12)[:n_months]
    else:
        print("ERROR: npz has no 'years' array — pass --start-year",
              file=sys.stderr)
        sys.exit(2)

    # Build a panel dict of (key → monthly_array) only including
    # fields that are actually present.
    panels_present = [
        (key, title, cmap, units, vmin, vmax)
        for (key, title, cmap, units, vmin, vmax) in _PANELS
        if key in npz.files
    ]
    if not panels_present:
        print("ERROR: none of the canonical 2-D fields are in the npz",
              file=sys.stderr)
        sys.exit(2)

    unique_years = sorted(set(int(y) for y in years))
    if args.year is not None:
        unique_years = [args.year]

    n_written = 0
    for year in unique_years:
        year_arr = {}
        any_field = False
        for key, *_ in panels_present:
            ym = _yearly_mean(npz[key], years, month_nums, year)
            year_arr[key] = ym
            any_field = any_field or (ym is not None)
        if not any_field:
            print(f"[year {year}] no complete months — skipping")
            continue
        out_path = args.output / f"amip_y{year}.png"
        _plot_year(year_arr, panels_present, lat, lon, year, out_path)
        print(f"[year {year}] -> {out_path}")
        n_written += 1

    print(f"Wrote {n_written} yearly map PNGs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
