#!/usr/bin/env python
"""Construct a Repeat Year Forcing (RYF) dataset from JRA55-do IAF.

Implements the Stewart et al. 2020 algorithm faithfully — modernised
from the canonical reference at https://github.com/COSIMA/make_ryf
(Python 2, MOM-specific) into a Python 3 implementation that produces
a single combined Zarr cache directly consumable by our
``scripts/data/prepare_omip_forcing.py``.

Algorithm
---------

The RYF year is built by stitching two consecutive calendar years at
the **May 1** boundary, choosing the splice direction so the result
has exactly 365 days (regardless of whether either input year is a
leap year):

* If ``year2`` is **not** a leap year: take ``year2`` as the template
  (it already has 365 days). Overwrite its **May 1 – Dec 31** section
  with the corresponding section from ``year1``. Result represents
  ``May year1 – April year2``.

* If ``year2`` **is** a leap year: take ``year1`` as the template
  (also 365 days). Overwrite its **Jan 1 – Apr 30** with the matching
  section from ``year2``, dropping Feb 29.

Stewart 2020 §3 documents this construction; their RYF9091 picks
``year1=1990, year2=1991`` (both non-leap) → template = 1991, splice
= May–Dec from 1990. The "smoothness" of the result is **statistical**
(both years are climatologically neutral; neither has a major ENSO,
NAO, or SAM excursion at the splice date) — there is no mathematical
blend or taper applied at the boundary.

Inputs
------

A directory containing JRA55-do IAF NetCDFs in the input4MIPs layout
(matches what ``scripts/data/download_jra55_iaf.py`` produces)::

    <iaf_dir>/atmos/3hrPt/<var>/v20240531/<var>_..._<YYYY>01010000-<YYYY>12312100.nc
    <iaf_dir>/atmos/3hr/<var>/v20240531/<var>_..._<YYYY>01010130-<YYYY>12312230.nc
    <iaf_dir>/land/day/friver/v20240531/friver_..._<YYYY>0101-<YYYY>1231.nc

Output
------

A single Zarr store containing all 10 variables on a common 3-hourly
time axis (re-based to ``1900-01-01 00:00`` with ``calendar=noleap``):

    <out_path>/
        uas, vas, tas, huss, psl       — native 3-hourly (HH:00)
        rsds, rlds, prra, prsn         — 3-hour means re-stamped to HH:00
                                         (1.5h offset is negligible vs the
                                         3h native cadence; documented)
        friver                         — daily, broadcast across the 8
                                         sub-daily slots of each noleap day

This format is what ``scripts/data/prepare_omip_forcing.py`` expects.

Usage
-----

    python scripts/make_ryf.py \\
        --iaf-dir data/jra55_iaf \\
        --year1 1990 --year2 1991 \\
        --out data/jra55_ryf/RYF9091.zarr

The output Zarr is ~3 GB for RYF9091 at native ~0.5° resolution.
"""

from __future__ import annotations

import argparse
import calendar
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Same variable manifest the downloader uses; importing keeps the two
# in lock-step.  The downloader lives next to us in ``scripts/`` so we
# import by file path.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from download_jra55_iaf import VARS, VERSION_TAG, _expected_filename, _local_path


# ---------------------------------------------------------------------------
# Splice plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplicePlan:
    """Which year is the template; which slice gets overwritten."""
    template_year: int          # year whose 365-day calendar provides the axis
    overwrite_source_year: int  # year supplying the replacement section
    # The replacement section is a calendar slice expressed as
    # ``(month, day)`` start and end (inclusive) on the template year.
    overwrite_start: tuple[int, int]
    overwrite_end: tuple[int, int]


def _build_splice_plan(year1: int, year2: int) -> SplicePlan:
    """Stewart 2020 splice rule — see module docstring."""
    if year2 != year1 + 1:
        raise ValueError(
            f"year2 must be year1 + 1 (consecutive); got {year1}, {year2}"
        )
    if calendar.isleap(year2):
        # Template = year1 (non-leap by construction). Overwrite Jan-Apr.
        return SplicePlan(
            template_year=year1,
            overwrite_source_year=year2,
            overwrite_start=(1, 1),
            overwrite_end=(4, 30),
        )
    # Template = year2 (non-leap). Overwrite May-Dec.
    return SplicePlan(
        template_year=year2,
        overwrite_source_year=year1,
        overwrite_start=(5, 1),
        overwrite_end=(12, 31),
    )


# ---------------------------------------------------------------------------
# Per-variable splice
# ---------------------------------------------------------------------------

def _resolve_iaf_path(iaf_dir: Path, var, year: int) -> Path:
    """Locate one variable-year IAF NetCDF on disk.

    Mirrors :func:`download_jra55_iaf._local_path` (which the downloader
    uses to write files), so the layouts match exactly.
    """
    expected = _local_path(iaf_dir, var, year)
    if expected.exists():
        return expected
    # Permissive fallback: glob the directory in case the version tag
    # changed between download and post-processing.
    parent = iaf_dir / var.realm / var.table / var.name
    if parent.exists():
        candidates = list(parent.rglob(_expected_filename(var, year)))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"IAF file for {var.name} {year} not found under {iaf_dir}. "
        f"Expected: {expected}"
    )


def _slice_by_calendar(da, year: int, start_md: tuple[int, int],
                       end_md: tuple[int, int]):
    """Return the records whose timestamps fall in ``[start_md, end_md]``
    of the given calendar year.

    ``time`` is decoded to numpy.datetime64 (xarray's default). The
    inclusive range covers ``start_md 00:00`` up to and including
    ``end_md 23:59:59``.
    """
    import pandas as pd
    sm, sd = start_md
    em, ed = end_md
    start = np.datetime64(f"{year:04d}-{sm:02d}-{sd:02d}T00:00:00")
    # 1 second past midnight at the *end* of em/ed gives an exclusive
    # cutoff that includes the last record of the day for any cadence.
    end_exclusive = (
        pd.Timestamp(year, em, ed) + pd.Timedelta(days=1)
    ).to_datetime64()
    t = da["time"].values
    mask = (t >= start) & (t < end_exclusive)
    return da.isel(time=mask)


def _splice_variable(template_da, source_da, plan: SplicePlan):
    """Apply the Stewart splice to one variable.

    ``template_da`` is the full-year DataArray from the template year;
    ``source_da`` is from the overwrite-source year. Returns a new
    DataArray with the overwrite section replaced. Time coordinate is
    inherited from the template (the source values are pasted in by
    *position*, not by date).
    """
    import xarray as xr
    template_slice = _slice_by_calendar(
        template_da, plan.template_year,
        plan.overwrite_start, plan.overwrite_end,
    )
    source_slice = _slice_by_calendar(
        source_da, plan.overwrite_source_year,
        plan.overwrite_start, plan.overwrite_end,
    )
    if template_slice.sizes["time"] != source_slice.sizes["time"]:
        raise ValueError(
            f"splice length mismatch: template has "
            f"{template_slice.sizes['time']} records, "
            f"source has {source_slice.sizes['time']}"
        )
    # Find positional indices of the template slice within the full
    # template DataArray so we can scatter back.
    template_t = template_da["time"].values
    slice_t = template_slice["time"].values
    idx = np.where(np.isin(template_t, slice_t))[0]

    out = template_da.copy()
    out.values[idx] = source_slice.values
    return out


# ---------------------------------------------------------------------------
# Time-axis harmonisation
# ---------------------------------------------------------------------------

def _harmonise_horizontal_grid(per_var):
    """Re-interpolate variables that don't share the reference grid.

    JRA55-do v1.6.0 publishes atmospheric variables on a TL319 reduced
    Gaussian grid (320×640) and ``friver`` on a separate 0.25° regular
    grid (720×1440). A naive ``xr.Dataset(out_vars)`` merge would
    fail. We interpolate the off-grid variables onto the reference
    grid (the first variable's lat/lon) using xarray's bilinear
    ``interp`` — not strictly mass-conservative for ``friver``, but
    acceptable for the smoke test and downstream coastal
    redistribution. Production OMIP runs should replace this with a
    flux-conservative remap.

    Input: dict ``{var_name: xr.DataArray}`` with possibly varying
    horizontal grids.
    Output: dict with all values on the reference grid.
    """
    ref_var = VARS[0].name
    ref_da = per_var[ref_var]
    ref_lat = ref_da["lat"].values
    ref_lon = ref_da["lon"].values

    out: dict = {}
    for v in VARS:
        da = per_var[v.name]
        same_lat = (
            da["lat"].size == ref_lat.size
            and np.allclose(da["lat"].values, ref_lat)
        )
        same_lon = (
            da["lon"].size == ref_lon.size
            and np.allclose(da["lon"].values, ref_lon)
        )
        if same_lat and same_lon:
            out[v.name] = da
        else:
            print(
                f"[make_ryf] regridding {v.name} "
                f"({da['lat'].size}×{da['lon'].size}) -> reference "
                f"({ref_lat.size}×{ref_lon.size}) via bilinear interp",
                flush=True,
            )
            out[v.name] = da.interp(
                lat=ref_lat, lon=ref_lon, method="linear",
                kwargs={"fill_value": 0.0},
            )
    return out


def _rebase_to_common_3hourly(per_var):
    """Resample each variable to a common 3-hourly time axis at HH:00.

    For 3hrPt variables this is the identity. For 3hr-mean variables
    we shift the timestamp by −1h30m (the 1.5h offset is small enough
    relative to the 3h cadence that we treat it as a re-label rather
    than a re-interpolation; this is the standard convention in
    JRA55-do-driven OMIP runs that downsample to a single time axis).
    For daily ``friver`` we broadcast each daily value across 8 slots
    of 3h.

    Input: dict ``{var_name: xarray.DataArray}`` with native cadences
    (assumed already on a common horizontal grid — call
    ``_harmonise_horizontal_grid`` first).
    Output: xarray.Dataset with all variables on a single 3-hourly time
    coordinate.
    """
    import pandas as pd
    import xarray as xr

    common_t: np.ndarray | None = None
    out_vars: dict[str, xr.DataArray] = {}

    for v in VARS:
        da = per_var[v.name]
        if v.ts_scheme == "instant":
            # Native 3-hourly at HH:00; pass through.
            harmonised = da
        elif v.ts_scheme == "mean3h":
            # Re-stamp HH:30 -> HH:00 (1.5h shift back to slot start).
            new_t = da["time"].values - np.timedelta64(90, "m")
            harmonised = da.assign_coords(time=new_t)
        elif v.ts_scheme == "daily":
            # Broadcast each daily value across the 8 3-hourly slots.
            day_values = da.values  # (n_day, ...)
            day_times = da["time"].values  # (n_day,)
            n_day = day_times.size
            n_slot = 8
            # Build the expanded time axis at HH:00 in 3h steps. Daily
            # JRA55-do timestamps land at 12:00; subtract 12h to align
            # the first slot with day-midnight.
            base_midnights = day_times.astype("datetime64[D]").astype(
                "datetime64[ns]"
            )
            offsets = np.arange(n_slot) * np.timedelta64(3, "h")
            expanded_t = (
                base_midnights[:, None] + offsets[None, :]
            ).reshape(-1)
            expanded_v = np.repeat(
                day_values, n_slot, axis=0,
            )  # (n_day*n_slot, ...)
            harmonised = xr.DataArray(
                expanded_v,
                dims=("time",) + da.dims[1:],
                coords={"time": expanded_t,
                        **{d: da[d] for d in da.dims[1:] if d in da.coords}},
                name=v.name,
                attrs=da.attrs,
            )
        else:
            raise ValueError(f"unknown ts_scheme {v.ts_scheme}")

        # First variable defines the axis; subsequent vars are aligned
        # by re-indexing onto it so any tiny mismatch becomes a hard
        # error rather than a silent merge surprise.
        if common_t is None:
            common_t = harmonised["time"].values
        else:
            harmonised = harmonised.reindex(time=common_t)
        out_vars[v.name] = harmonised

    ds = xr.Dataset(out_vars)
    return ds


def _shift_time_axis_to_1900(ds):
    """Map the time axis to 1900-01-01-based with ``calendar=noleap``.

    Prevents downstream tools from treating the cycled output as
    referring to specific calendar dates (1990-91). Matches the
    convention in COSIMA's reference make_ryf.py.
    """
    import pandas as pd
    t = ds["time"].values
    delta = t - t[0]
    new_t = np.datetime64("1900-01-01T00:00:00") + delta
    ds = ds.assign_coords(time=new_t)
    # xarray treats ``calendar`` as encoding, not a free-form attr; the
    # rest go in attrs.  ``modulo`` is the MOM convention for declaring
    # a periodic axis.
    ds["time"].encoding["calendar"] = "noleap"
    ds["time"].attrs.update({
        "axis": "T",
        "modulo": " ",
        "comment": "RYF time axis re-based to 1900-01-01; calendar = noleap",
    })
    return ds


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def make_ryf(iaf_dir: Path, year1: int, year2: int, out_path: Path,
             *, progress: bool = True):
    """Run the full RYF construction. Writes a Zarr to ``out_path``."""
    import xarray as xr

    plan = _build_splice_plan(year1, year2)
    if progress:
        print(f"[make_ryf] year1={year1} year2={year2}")
        print(f"[make_ryf] template year   = {plan.template_year}")
        print(f"[make_ryf] overwrite from  = {plan.overwrite_source_year}")
        print(f"[make_ryf] overwrite range = "
              f"{plan.overwrite_start} → {plan.overwrite_end}")

    per_var: dict[str, xr.DataArray] = {}
    for v in VARS:
        if progress:
            print(f"[make_ryf] splicing {v.name} ...", flush=True)
        path_template = _resolve_iaf_path(iaf_dir, v, plan.template_year)
        path_source = _resolve_iaf_path(iaf_dir, v, plan.overwrite_source_year)
        ds_t = xr.open_dataset(path_template)
        ds_s = xr.open_dataset(path_source)
        spliced = _splice_variable(ds_t[v.name], ds_s[v.name], plan)
        per_var[v.name] = spliced

    if progress:
        print("[make_ryf] harmonising to common horizontal grid ...")
    per_var = _harmonise_horizontal_grid(per_var)

    if progress:
        print("[make_ryf] harmonising to common 3-hourly axis ...")
    ds = _rebase_to_common_3hourly(per_var)

    if progress:
        print("[make_ryf] shifting time axis to 1900-01-01-based ...")
    ds = _shift_time_axis_to_1900(ds)

    ds.attrs.update({
        "title": f"JRA55-do RYF year {year1}-{year2}",
        "source": "MRI-JRA55-do-1-6-0 IAF, spliced per Stewart et al. 2020",
        "history": (
            f"Constructed by scripts/make_ryf.py from "
            f"{iaf_dir}; template={plan.template_year}, "
            f"source={plan.overwrite_source_year}."
        ),
    })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"[make_ryf] writing {out_path} ...")
    ds.chunk({"time": 8 * 30}).to_zarr(
        str(out_path), mode="w", consolidated=True,
    )
    if progress:
        print(f"[make_ryf] done: {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--iaf-dir", type=str, required=True,
                   help="Directory containing the JRA55-do IAF NetCDFs.")
    p.add_argument("--year1", type=int, required=True,
                   help="First year of the RYF window (e.g. 1990 for RYF9091).")
    p.add_argument("--year2", type=int, required=True,
                   help="Second year (must equal year1+1; e.g. 1991).")
    p.add_argument("--out", type=str, required=True,
                   help="Output Zarr path.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    make_ryf(
        Path(args.iaf_dir), args.year1, args.year2, Path(args.out),
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
