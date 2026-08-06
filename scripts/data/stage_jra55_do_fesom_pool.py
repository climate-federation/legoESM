#!/usr/bin/env python
"""Merge DKRZ's per-variable JRA55-do files into one source store for the
legoESM OMIP-2 cache builder.

Why
---
FESOM2 on Levante reads JRA55-do v1.4.0 as one NetCDF **per variable per
year** out of ``/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0/`` (the paths
are literal in ``namelist.forcing``).  ``legoesm.forcing.jra55_do``'s
``build_jra55_cache`` instead wants a SINGLE store holding every variable on
ONE shared source grid and ONE shared time axis.  This script bridges the two
so a legoESM run can be forced by the SAME SOURCE FILES FESOM2 is forced by --
the prerequisite for any controlled legoESM-vs-FESOM2 comparison.

Not "byte-identical", and it should never be described that way: the time
axis is rebased, ``friver`` is regridded and held, everything is then
conservatively regridded to the model grid by the cache builder, and the
staged store is float32.  See ``docs/ocean/fidelity/fesom2_gap_analysis.md``.

RUNOFF NOTE: ``friver`` needs a COASTAL ROUTING step, which this script does
NOT do and does not need to -- routing depends on the MODEL's land mask, so it
belongs downstream, and ``run_omip.py --runoff-routing`` now performs it
(``legoesm.ocean.forcing.runoff_mapper``).  What matters here is that the
remap below conserves the global total; where that total can be delivered is
the driver's problem.  See ``docs/ocean/fidelity/fesom2_gap_analysis.md``
section 3.5 for the measured loss with and without routing, and for a
retracted first reading of ``areacello`` (it does NOT make the full-cell total
an overestimate; at 0.25 deg a river-mouth cell is essentially all ocean).

It performs three alignments and nothing else; all regridding in time or
space that the cache needs is left to ``build_jra55_cache``:

1. **Flux time stamps.**  In this distribution the *state* variables
   (``uas``, ``vas``, ``tas``, ``huss``, ``psl``) are stamped at the interval
   START (00:00, 03:00, ...) while the *flux* variables (``rsds``, ``rlds``,
   ``prra``, ``prsn``) are stamped at the interval MIDPOINT (01:30, 04:30,
   ...).  ``_build_noleap_record_index`` maps a record to a 3-hourly slot by
   ``round(hour / 3)``, which is only correct for start stamps -- midpoint
   stamps alias two records onto one slot.  We therefore rebase the flux axis
   onto the state axis by subtracting 1.5 h, i.e. we read an interval-mean
   flux as valid at the START of its interval.  That is the same reading
   FESOM2 uses: ``namelist.forcing`` sets ``nm_nc_tmid = 0`` ("time stamp
   position: 1=mid-point, 0=start of interval") for JRA55-do.

2. **Runoff cadence.**  ``friver`` is daily, stamped at the interval MIDPOINT
   (12:00).  It is held constant through ITS OWN CALENDAR DAY on the 3-hourly
   axis -- binned against the interval STARTS, not the stamps, or every day
   would run noon-to-noon and each transition would be 12 h late.  (Leaving
   it sparse is not an option: every variable in the merged store must share
   one ``time`` dimension, and NaN holes would poison the conservative
   regrid.)

3. **Runoff grid.**  ``friver`` is on the 0.25 deg (1440x720) river grid, not
   the 0.5625 deg (640x320) atmospheric grid.  ``_regrid_and_align_variable``
   builds ONE weight set for all variables and raises on a shape mismatch, so
   ``friver`` is conservatively regridded onto the atmospheric grid here,
   using ``legoesm.grids.conservative_regrid`` (the shared utility -- no
   re-derived area weights).  It is then conservatively regridded a second
   time, to the model grid, by the cache builder; conservative regridding is
   linear and area-conserving, so the composition still conserves.

Usage
-----
    python scripts/data/stage_jra55_do_fesom_pool.py \\
        --pool-dir /pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0 \\
        --years 1958 1958 \\
        --out /scratch/b/b381103/jra55_do_fesom_1958.zarr

then feed ``--out`` to ``scripts/data/prepare_omip_forcing.py --source``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# The conservative overlap weights are area ratios summed to 1; float32 would
# put their roundoff (~1e-7) at the coverage guard's own 1e-6 tolerance.  Set
# before the deferred ``import jax.numpy`` inside the regrid helper, which is
# this module's only JAX use.
os.environ.setdefault("JAX_ENABLE_X64", "1")

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

#: Variables stamped at the START of their 3-hourly interval.
STATE_VARS: tuple[str, ...] = ("uas", "vas", "tas", "huss", "psl")

#: Variables stamped at the MIDPOINT of their 3-hourly interval.
FLUX_VARS: tuple[str, ...] = ("rsds", "rlds", "prra", "prsn")

#: Daily variable on the separate river grid.
RUNOFF_VAR: str = "friver"

#: Half of the 3-hourly sampling interval, in days -- the midpoint offset
#: removed from :data:`FLUX_VARS` so every variable shares one axis.
_FLUX_MIDPOINT_OFFSET_DAYS: float = 1.5 / 24.0

#: Tolerance (days) for asserting the rebased flux axis matches the state
#: axis.  1e-6 d = 0.09 s; the stamps are exact multiples of 1/8 d in fp64.
_TIME_MATCH_TOL_DAYS: float = 1e-6

#: Tolerance (days) on "the runoff records are one day apart".  1e-6 d is
#: 0.09 s -- far below any real cadence error, far above fp64 epoch roundoff.
_DAY_TOL_DAYS: float = 1e-6


def _open_year(pool_dir: Path, var: str, year: int):
    """Open one ``<var>.<year>.nc`` with time decoding OFF.

    The cache builder parses the CF ``units`` string itself, so the raw
    numeric axis is what must survive to the merged store.
    """
    import xarray as xr

    path = pool_dir / f"{var}.{year}.nc"
    if not path.exists():
        raise FileNotFoundError(f"missing JRA55-do file: {path}")
    return xr.open_dataset(path, decode_times=False)


def _edges_from_cf_bounds(bnds, name: str) -> np.ndarray:
    """Convert a CF ``(n, 2)`` bounds array to an ``(n + 1,)`` edge array.

    Raises unless the cells are contiguous (``bnds[i, 1] == bnds[i+1, 0]``),
    because a gapped or overlapping bounds array would make the overlap
    weights silently wrong rather than fail.
    """
    b = np.asarray(bnds, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"{name}: expected (n, 2) CF bounds, got {b.shape}")
    if b[0, 0] > b[-1, 1]:          # descending axis -> flip to ascending
        b = b[::-1, ::-1]
    gap = np.max(np.abs(b[1:, 0] - b[:-1, 1]))
    if gap > 1e-9:
        raise ValueError(
            f"{name}: CF bounds are not contiguous (max gap {gap:.3e} deg)"
        )
    return np.concatenate([b[:, 0], b[-1:, 1]])


def _axis_edges(ds, axis: str, centres: np.ndarray, label: str) -> np.ndarray:
    """Cell edges [deg] for ``axis``: the file's CF bounds if it has them,
    otherwise inferred from the centres assuming uniform spacing.

    JRA55-do ships ``lat_bnds``/``lon_bnds``, and its atmospheric latitudes
    are GAUSSIAN (spacing varies 0.5569-0.5616 deg over the 320 rows), so the
    uniform-spacing inference is an approximation there and the bounds are
    not.  Prefer the bounds; keep the inference as the fallback for a file
    that lacks them.
    """
    bnds_name = ds[axis].attrs.get("bounds", f"{axis}_bnds")
    if bnds_name in ds:
        return _edges_from_cf_bounds(ds[bnds_name].values,
                                     f"{label} {bnds_name}")
    from legoesm.grids.conservative_regrid import cell_edges_1d
    asc = np.sort(np.asarray(centres, dtype=np.float64))
    return np.rad2deg(cell_edges_1d(np.deg2rad(asc)))


def _epoch_is_midnight(units) -> bool:
    """True when a CF ``days since ...`` epoch is midnight in the ZERO offset.

    Only then does an integral time value mean "a calendar midnight", which
    is what :func:`_daily_hold_index` needs to place day boundaries in
    absolute terms rather than merely consistently.

    CF allows the reference time to carry a time zone, attached (``00:00:00Z``,
    ``00:00:00+03:00``) or space-separated.  A NON-ZERO offset is NOT an
    absolute midnight -- ``00:00:00 +03:00`` is 21:00 UTC -- so it is rejected
    even though the clock field reads zero.  ``Z`` and ``+00:00`` are.
    """
    if not units:
        return False
    tail = str(units).split("since", 1)[-1].strip()
    if not tail:
        return False
    parts = tail.split()
    if len(parts) == 1:                      # date only => midnight
        return True
    clock = parts[1]
    tz = parts[2] if len(parts) > 2 else ""
    # Split an ATTACHED zone off the clock field.
    for sep in ("Z", "z", "+", "-"):
        i = clock.find(sep)
        if i > 0:
            clock, tz = clock[:i], clock[i:] + tz
            break
    if not (clock.startswith("00:00")
            and set(clock.replace(":", "")) <= {"0"}):
        return False
    if tz in ("", "Z", "z", "UTC", "utc"):
        return True
    return set(tz.replace(":", "").lstrip("+-")) <= {"0"}


def _daily_hold_index(run_time, ref_time, *, label: str,
                      units, ref_units) -> np.ndarray:
    """Index each 3-hourly slot to the daily record covering ITS calendar day.

    A daily MEAN applies through its whole day, 00:00 to 24:00.  The stamps
    are interval MIDPOINTS (12:00), so binning against them directly runs the
    hold noon-to-noon: every transition 12 h late, the first day covering 12
    three-hourly slots and the last only 4.  Binning against the interval
    STARTS -- the stamps shifted back half an interval -- gives every day
    exactly 8.

    Both guards below replace an earlier ``np.clip``, which absorbed a missing
    or irregular day by holding the previous one across the gap: a plausible
    field, silently wrong.

    Parameters
    ----------
    run_time : daily stamps [days], same epoch as ``ref_time``.
    ref_time : the 3-hourly target axis [days].
    label : file identifier for the error messages.
    """
    run_time = np.asarray(run_time, dtype=np.float64)
    ref_time = np.asarray(ref_time, dtype=np.float64)
    if run_time.size < 2:
        raise ValueError(
            f"{label}: need >= 2 records to infer the daily interval; got "
            f"{run_time.size}."
        )
    d_run = np.diff(run_time)
    if not np.all(d_run > 0) or not np.allclose(d_run, d_run[0],
                                                rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"{label}: daily time axis is not strictly increasing with "
            f"uniform spacing (min {d_run.min():.6g} d, max "
            f"{d_run.max():.6g} d). A gap would be filled by holding the "
            "previous day across it."
        )
    # Uniform is not enough -- it must be uniform at ONE DAY.  A 1.125 d
    # spacing is perfectly uniform and would hand out 9/9/6-slot "days".
    if abs(float(d_run[0]) - 1.0) > _DAY_TOL_DAYS:
        raise ValueError(
            f"{label}: records are spaced {float(d_run[0]):.6f} d apart, not "
            "1 d. This helper holds a DAILY mean over its own calendar day."
        )

    d_ref = np.diff(ref_time)
    if not np.all(d_ref > 0) or not np.allclose(d_ref, d_ref[0],
                                                rtol=1e-9, atol=1e-9):
        raise ValueError(
            f"{label}: the target axis is not uniformly spaced "
            f"(min {d_ref.min():.6g} d, max {d_ref.max():.6g} d)."
        )
    dt_ref = float(d_ref[0])

    day_start = run_time - 0.5 * float(d_run[0])
    covered_lo = float(day_start[0])
    covered_hi = float(day_start[-1] + d_run[0])
    idx = np.searchsorted(day_start, ref_time, side="right") - 1
    # Coverage is half-open [covered_lo, covered_hi): a sample exactly AT
    # covered_hi belongs to the next day, which this file does not have.
    # searchsorted saturates at the last record, so without this an axis
    # running past the daily coverage came back in-range and silently held
    # the last day over the tail.  The tolerance absorbs float64 epoch
    # roundoff (~4e-12 d on a `days since 1900` axis).
    _tol = 1e-9
    if (idx.min() < 0 or idx.max() >= run_time.size
            or float(ref_time[0]) < covered_lo - _tol
            or float(ref_time[-1]) >= covered_hi - _tol):
        raise ValueError(
            f"{label}: the 3-hourly axis spans [{ref_time[0]:.6f}, "
            f"{ref_time[-1]:.6f}] d but the {run_time.size} daily records "
            f"cover [{covered_lo:.6f}, {covered_hi:.6f}) d "
            f"(index range [{idx.min()}, {idx.max()}]). The two files do not "
            "describe the same period."
        )

    # The day boundaries must land ON target samples, not between them.  A
    # stamp phase offset (10:30 instead of 12:00) passes every check above
    # and still tiles evenly, but it shifts the hold by that offset; the pool
    # contract is noon stamps, so tolerating it would only hide a malformed
    # or mismatched source.  Tolerance in DAYS, matching every other check.
    offset_days = covered_lo - float(ref_time[0])
    if abs(offset_days / dt_ref - round(offset_days / dt_ref)) * dt_ref \
            > _DAY_TOL_DAYS:
        raise ValueError(
            f"{label}: the daily boundaries fall BETWEEN target samples "
            f"(offset {offset_days / dt_ref:.6f} steps of {dt_ref:.6f} d). "
            "The daily stamps are phase-shifted relative to the target axis, "
            "so the hold would be displaced by that offset."
        )

    # The two axes must be in the SAME frame before their numbers can be
    # compared at all.  Without this, a runoff axis on `... +03:00` against a
    # target on `...Z` passes every arithmetic check while applying days that
    # begin at a local midnight to a UTC axis.  Required, not optional: a
    # missing `units` attribute would otherwise fail open.
    if not units or not ref_units:
        raise ValueError(
            f"{label}: both time axes need CF `units` to be comparable; got "
            f"runoff={units!r}, target={ref_units!r}."
        )
    if str(units).strip() != str(ref_units).strip():
        raise ValueError(
            f"{label}: the runoff and target axes use DIFFERENT time units "
            f"({units!r} vs {ref_units!r}). Their numbers are not in the same "
            "frame, so no comparison between them is meaningful."
        )

    # ...and the day boundaries must be ABSOLUTE calendar midnights, not
    # merely consistent with the target axis.  The step check above is
    # RELATIVE: shift BOTH axes by the same 3 h (15:00 runoff stamps against
    # an 03:00 target phase) and it still passes, while every "day" actually
    # runs 03:00-03:00.  With a midnight epoch an integral value IS a
    # midnight, so require both.
    if not _epoch_is_midnight(units):
        raise ValueError(
            f"{label}: time units {units!r} do not have a midnight epoch in "
            "the zero offset, so a calendar day boundary cannot be "
            "identified. Re-express against `days since <date> 00:00:00`."
        )
    if abs(covered_lo - round(covered_lo)) > _DAY_TOL_DAYS:
        raise ValueError(
            f"{label}: the first day starts at {covered_lo:.6f} d after the "
            f"epoch, which is not a calendar midnight "
            f"({covered_lo - np.floor(covered_lo):.6f} d past one). The "
            "daily stamps are offset from the calendar, so the hold would "
            "not cover whole days."
        )

    # Every day must receive the SAME number of slots.  This is NOT implied by
    # the checks above: a target axis that starts partway through the first
    # day (e.g. at 03:00) gives 7/8/8, which every other guard accepts.  It is
    # the direct statement of what a zero-order hold over whole days needs.
    counts = np.bincount(idx, minlength=run_time.size)
    if counts.min() != counts.max():
        raise ValueError(
            f"{label}: the daily records do not tile the target axis evenly "
            f"({counts.min()}-{counts.max()} slots per day). The target axis "
            "probably starts or ends partway through a day."
        )
    return idx


def _regrid_runoff_to_atmos_grid(da_runoff, src_lat_edges_deg,
                                 src_lon_edges_deg, dst_lat_edges_deg,
                                 dst_lon_edges_deg, lon_atmos):
    """Conservatively regrid ``friver`` onto the atmospheric lat-lon grid.

    ``normalization='dstarea'`` (strictly conservative) with no polar fill,
    because runoff is an EXTENSIVE flux density whose global integral is the
    quantity that must survive, and the 0.25 deg river grid's bounds reach the
    poles exactly -- so every atmospheric destination cell is fully covered
    and there is no gap to fill.  ``require_full_coverage=True`` turns any
    violation of that into an exception rather than a quietly reduced field.

    Parameters
    ----------
    da_runoff : xr.DataArray, dims ``(time, lat, lon)`` on the river grid.
    src_lat_edges_deg : river-grid latitude edges [deg], ascending.
    dst_lat_edges_deg, dst_lon_edges_deg : atmospheric-grid edges [deg],
        ascending (latitude) / increasing eastward (longitude).
    lon_atmos : atmospheric longitude CENTRES [deg] -- used only to size the
        wrap padding.

    Returns
    -------
    np.ndarray, shape ``(n_time, n_dst_lat, n_dst_lon)`` in the destination's
    ASCENDING-latitude order.
    """
    import jax
    import jax.numpy as jnp

    from legoesm.grids.conservative_regrid import (
        apply_conservative_regrid,
        check_axis_span,
        compute_overlap_weights,
    )

    # Fail LOUDLY rather than silently regridding in float32: the overlap
    # weights are area ratios summed to 1, and float32 roundoff (~1e-7) puts
    # the conservation error at 1e-8 relative -- plausible-looking, wrong, and
    # invisible.  The module sets JAX_ENABLE_X64 at import, but that is too
    # late if something imported jax first (a pytest conftest, say).
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "JAX x64 is off; the conservative regrid would run in float32. "
            "Set JAX_ENABLE_X64=1 before importing jax, or call "
            "jax.config.update('jax_enable_x64', True)."
        )

    src_lat = np.asarray(da_runoff["lat"].values, dtype=np.float64)
    src_lon = np.asarray(da_runoff["lon"].values, dtype=np.float64)
    src = np.asarray(da_runoff.values, dtype=np.float64)

    # The edge arrays are ASCENDING; flip the data with the axis so a
    # descending-latitude file cannot silently mirror the field.
    if src_lat[0] > src_lat[-1]:
        src = src[:, ::-1, :]
    if src_lon[0] > src_lon[-1]:
        src = src[:, :, ::-1]
        src_lon = src_lon[::-1]

    # Longitude seam: the two grids have DIFFERENT origins -- river cells
    # start at edge 0.0 deg, atmospheric cells at edge -0.28125 deg.  The
    # destination's first column therefore sticks out past the source's
    # western edge and would come back partly weighted.  Wrap-pad the source
    # with ghost columns at each end (the exact periodic continuation) so it
    # brackets the destination.  No destination cell reaches a ghost AND its
    # 360-deg twin -- the destination spans exactly one period -- so nothing
    # is double counted.
    src_lon_edges = np.asarray(src_lon_edges_deg, dtype=np.float64)
    # PRECONDITION, checked BEFORE the padding below: the RAW source must
    # already tile the full 360 deg.  The wrap ghosts are the exact periodic
    # continuation only if that holds -- on a source missing a longitude
    # sector they would instead copy edge data across the hole, and
    # ``require_full_coverage=True`` (applied AFTER padding) would then report
    # complete coverage of a fabricated field.
    check_axis_span(np.deg2rad(src_lon_edges), 2.0 * np.pi,
                    name="runoff source longitude")
    dlon_src = float(src_lon[1] - src_lon[0])
    dlon_dst = float(np.abs(lon_atmos[1] - lon_atmos[0]))
    n_ghost = int(np.ceil((0.5 * dlon_dst + dlon_src) / dlon_src))
    src = np.concatenate(
        [src[:, :, -n_ghost:], src, src[:, :, :n_ghost]], axis=2
    )
    src_lon_edges = np.concatenate([
        src_lon_edges[0] - dlon_src * np.arange(n_ghost, 0, -1),
        src_lon_edges,
        src_lon_edges[-1] + dlon_src * np.arange(1, n_ghost + 1),
    ])

    weights = compute_overlap_weights(
        src_lat_edges=np.deg2rad(src_lat_edges_deg),
        src_lon_edges=np.deg2rad(src_lon_edges),
        dst_lat_edges=np.deg2rad(dst_lat_edges_deg),
        dst_lon_edges=np.deg2rad(dst_lon_edges_deg),
        normalization="dstarea",
        polar_fill=False,
        require_full_coverage=True,
    )
    return np.asarray(
        apply_conservative_regrid(jnp.asarray(src, dtype=jnp.float64), weights)
    )


def build_source_year(pool_dir: Path, year: int):
    """Merged, time-aligned JRA55-do source Dataset for ONE year.

    Every variable in ``legoesm.forcing.jra55_do.JRA55_VARIABLES`` is present,
    on the atmospheric 640x320 grid and the shared 3-hourly ``time`` axis
    carrying the source's own CF ``units``.

    One year at a time on purpose: the zero-order-held runoff alone
    materialises ~2.4 GB per year, so concatenating a multi-decade window in
    memory before writing would need ~146 GB for OMIP-2's 1958-2018.  The
    caller streams years to Zarr with ``append_dim`` instead.
    """
    import xarray as xr

    for _once in (0,):
        merged = {}
        ref_time = None
        ref_units = None
        for var in STATE_VARS:
            ds = _open_year(pool_dir, var, year)
            if ref_time is None:
                ref_time = np.asarray(ds["time"].values, dtype=np.float64)
                ref_units = ds["time"].attrs.get("units")
            merged[var] = ds[var]

        for var in FLUX_VARS:
            ds = _open_year(pool_dir, var, year)
            t = np.asarray(ds["time"].values, dtype=np.float64)
            rebased = t - _FLUX_MIDPOINT_OFFSET_DAYS
            if rebased.size != ref_time.size:
                raise ValueError(
                    f"{var}.{year}: {rebased.size} records but the state "
                    f"variables have {ref_time.size}; cannot share one axis."
                )
            drift = float(np.max(np.abs(rebased - ref_time)))
            if drift > _TIME_MATCH_TOL_DAYS:
                raise ValueError(
                    f"{var}.{year}: rebased time axis differs from the state "
                    f"axis by up to {drift * 24:.4f} h. The midpoint "
                    "assumption documented in this module does not hold for "
                    "this distribution -- inspect the file before staging."
                )
            merged[var] = ds[var].assign_coords(time=ref_time)

        ds_run = _open_year(pool_dir, RUNOFF_VAR, year)
        ds_atm = _open_year(pool_dir, STATE_VARS[0], year)
        sample = merged[STATE_VARS[0]]
        lat_atm = np.asarray(sample["lat"].values, dtype=np.float64)
        lon_atm = np.asarray(sample["lon"].values, dtype=np.float64)
        runoff_atmos = _regrid_runoff_to_atmos_grid(
            ds_run[RUNOFF_VAR],
            _axis_edges(ds_run, "lat", np.asarray(ds_run["lat"].values),
                        "river"),
            _axis_edges(ds_run, "lon", np.asarray(ds_run["lon"].values),
                        "river"),
            _axis_edges(ds_atm, "lat", lat_atm, "atmos"),
            _axis_edges(ds_atm, "lon", lon_atm, "atmos"),
            lon_atm,
        )
        # The regrid returns ASCENDING latitude; restore the file's order so
        # the merged Dataset's ``lat`` coordinate and data stay consistent.
        if lat_atm[0] > lat_atm[-1]:
            runoff_atmos = runoff_atmos[:, ::-1, :]
        # Daily -> 3-hourly zero-order hold; see _daily_hold_index.
        hold_idx = _daily_hold_index(
            np.asarray(ds_run["time"].values, dtype=np.float64),
            ref_time, label=f"{RUNOFF_VAR}.{year}",
            units=ds_run["time"].attrs.get("units"),
            ref_units=ref_units,
        )
        # float32 before the zero-order-hold expansion: the held array is
        # ``n_records`` copies of 365 daily fields and would otherwise cost
        # 4.8 GB in float64 for one year, for no precision the float32 Zarr
        # encoding keeps.
        merged[RUNOFF_VAR] = xr.DataArray(
            runoff_atmos.astype(np.float32)[hold_idx],
            dims=("time", "lat", "lon"),
            coords={"time": ref_time, "lat": sample["lat"],
                    "lon": sample["lon"]},
            attrs=dict(ds_run[RUNOFF_VAR].attrs),
        )

    ds_year = xr.Dataset(merged)
    ds_year["time"] = ref_time
    if ref_units is not None:
        ds_year["time"].attrs["units"] = ref_units
    out = ds_year
    out.attrs["legoesm_staging"] = (
        "merged from per-variable DKRZ JRA55-do v1.4.0 files; flux stamps "
        f"rebased by -{_FLUX_MIDPOINT_OFFSET_DAYS * 24:.1f} h to the state "
        "axis (FESOM2 nm_nc_tmid=0); friver conservatively regridded to the "
        "atmospheric grid and held through each calendar day"
    )
    return out


def build_source_dataset(pool_dir: Path, years: tuple[int, int]):
    """Concatenated multi-year source Dataset.

    Convenience for tests and single-year use.  :func:`main` streams year by
    year instead -- see :func:`build_source_year` for why.
    """
    import xarray as xr

    parts = [build_source_year(pool_dir, y)
             for y in range(years[0], years[1] + 1)]
    return parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pool-dir", type=str,
                   default="/pool/data/AWICM/FESOM2/FORCING/JRA55-do-v1.4.0",
                   help="Directory of per-variable JRA55-do NetCDFs.")
    p.add_argument("--years", type=int, nargs=2, metavar=("START", "END"),
                   required=True, help="Inclusive year window.")
    p.add_argument("--out", type=str, required=True,
                   help="Output Zarr path for the merged source store.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"[stage] exists, skipping: {out}")
        return 0
    if args.years[1] < args.years[0]:
        raise SystemExit(
            f"--years START must be <= END; got {args.years[0]} "
            f"{args.years[1]}."
        )

    # Stream year by year: a decade held in memory before writing would be
    # tens of GB, almost all of it the zero-order-held runoff.
    for i, year in enumerate(range(args.years[0], args.years[1] + 1)):
        ds = build_source_year(Path(args.pool_dir), year)
        if i == 0:
            print(f"[stage] merged variables : {sorted(ds.data_vars)}")
            print(f"[stage] dims per year    : {dict(ds.sizes)}")
            print(f"[stage] writing          : {out}")
            encoding = {v: {"dtype": "float32"} for v in ds.data_vars}
            ds.to_zarr(out, mode="w", encoding=encoding, consolidated=True)
        else:
            ds.to_zarr(out, append_dim="time", consolidated=True)
        print(f"[stage]   {year} written")
        ds.close()
    print(f"[stage] done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
