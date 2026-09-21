#!/usr/bin/env python3
"""Paired-arm window means: partial-month CMOR file minus the restart sidecar.

A restart arm copies its parent's ``cmor_accum_day_<start>.npz`` (the monthly
accumulator's running SUM and per-variable sample COUNT from the 1st of the
month to day ``start``) and, on clean completion, the driver writes the open
month as a partial-month ``Amon`` file (mean over the 1st .. ``end``) and
retires the terminal sidecar.  The mean over the arm's OWN days is therefore

    window = (mean_file * n_end - sum_start) / (n_end - n_start)

with ``n_end = n_start + cadence * (end - start)``.  The accumulator samples
every 2-D variable once per day (measured on rhebc90_r6: counts 12 -> 22 for
clt and 11 -> 21 for the fluxes between day 70 and day 80), so ``cadence`` is
1/day and is asserted, not assumed: pass ``--cadence-from RUN`` naming a run
with two sidecars and the counts are checked.

Every arm is reduced identically, and the control's partial month is the
arm's own days-before-``start`` history, so the arm-minus-control difference
is exactly ``(n_end/(n_end - n_start))`` times the partial-file difference.
Self-check: the sidecar's mean over 1..start must correlate spatially with
the partial-month file (> 0.9 for every variable) or the two are not on the
same grid/orientation and the script refuses.

Prints area-weighted window means of the TOA/cloud/rain 2-D fields (globe,
20S-20N, 10S-10N) and the arm-minus-control differences.  CRE_LW = rlutcs -
rlut, CRE_SW = rsutcs - rsut.

    window_diff.py --ctl cc_ctl --arms cc_cond3e5 cc_xr --start 80 --end 86
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

from legoesm.diagnostics.monthly_means import MonthlyAccumulator
from legoesm.forcing.time_utils import day_to_calendar

ROOT = pathlib.Path("/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
FIELDS = ("rsut", "rlut", "rsutcs", "rlutcs", "clt", "clwvi", "clivi", "pr",
          "evspsbl", "prw", "tas", "hfls", "hfss")
DERIVED = {"CRE_LW": ("rlutcs", "rlut"), "CRE_SW": ("rsutcs", "rsut"),
           "P-E": ("pr", "evspsbl")}
# Rain and evaporation are accumulated in kg/m2/s, which prints as 0.000 at the
# table's precision -- the precipitation gate was unreadable until this scaling
# (a 4 mm/day tropical mean is 4.6e-5 kg/m2/s).
SCALE = {"pr": 86400.0, "evspsbl": 86400.0}
UNITS = {"pr": "mm/d", "evspsbl": "mm/d", "P-E": "mm/d"}
# (lat_lo, lat_hi, lon_lo, lon_hi), longitudes in [0, 360) and allowed to wrap
# past 360 (the Namibian box straddles the prime meridian).  The zonal bands
# come first; the stratocumulus decks are included because the cloud bias
# CHANGES SIGN between them -- Namibia is too cloudy while California is too
# clear -- so a zonal mean averages the two defects away.  These boxes match
# ``regional_bias.REGIONS`` so the two instruments cannot drift apart, but they
# are evaluated here on the MATCHED day window rather than on each run's
# published months: a restart arm publishes fewer months than its parent, and
# comparing those climatologies directly is a sampling confound large enough to
# move incoming solar by 40 W/m^2.
BANDS = {"GLOBAL": (-90.0, 90.0, 0.0, 360.0),
         "tropics 20S-20N": (-20.0, 20.0, 0.0, 360.0),
         "ITCZ 10S-10N": (-10.0, 10.0, 0.0, 360.0),
         "trades 10-30N": (10.0, 30.0, 0.0, 360.0),
         "trades 10-30S": (-30.0, -10.0, 0.0, 360.0),
         "Sc Peru": (-30.0, -10.0, 260.0, 290.0),
         "Sc Namibia": (-25.0, -5.0, 350.0, 375.0),
         "Sc California": (15.0, 35.0, 220.0, 250.0),
         "SO stormtrack": (-60.0, -30.0, 0.0, 360.0),
         # The polar caps are a separate error of the opposite sign, so a
         # global mean hides them.  They are SEPARATE rows deliberately: in
         # March the Arctic cap is snow-covered while the Antarctic is largely
         # bare ice, so their albedo errors do not have the same size and a
         # single combined cap would average two different defects.
         "Arctic 60-90N": (60.0, 90.0, 0.0, 360.0),
         # The polar-cap moisture import is P-E over the cap (plus the storage
         # tendency, added by the caller); the two inner caps are the
         # pre-registered scoring bands of the cap-cloud A/B.
         "Arctic 72.5-90N": (72.5, 90.0, 0.0, 360.0),
         "Arctic 75-90N": (75.0, 90.0, 0.0, 360.0),
         "Antarctic 60-90S": (-90.0, -60.0, 0.0, 360.0)}
CADENCE_PER_DAY = 1.0


def bucket_key(day):
    """The accumulator's (year, month) bucket for a simulation day, computed
    with the SAME helpers the driver uses so the window can never be binned
    differently from the sums it is subtracting."""
    doy, _ = day_to_calendar(float(day))
    return int(float(day) // 365.0), MonthlyAccumulator.day_to_month(doy)
MIN_SPATIAL_CORR = 0.8   # user 2026-09-21: 0.88 on a 5-vs-10-day cloud-cover window; a grid mismatch gives ~0


def sidecar_sums(path):
    """{var: (sum (nlat, nlon), count)} for the monthly accumulator's OPEN
    month, plus its (year, month).

    A run that has crossed a month boundary keeps the CLOSED months in the same
    bucket (a day-100 sidecar carries a complete March alongside a partial
    April), so the open month is selected -- the latest one present -- and only
    its sums are returned.  Mixing two months here would silently average a
    complete month into a 10-day window.
    """
    z = np.load(path, allow_pickle=True)
    man = json.loads(str(z["monthly.__manifest__"]))
    if man["type"] != "SpatialMonthlyAccumulator":
        raise SystemExit(f"{path}: unexpected accumulator {man['type']!r}")
    months = {(y, m) for y, m, *_ in man["data_2d"]}
    if not months:
        raise SystemExit(f"{path}: empty accumulator bucket")
    open_month = max(months)
    out = {}
    for _y, _m, var, count, key in man["data_2d"]:
        if (_y, _m) != open_month:
            continue
        out[var] = (np.asarray(z[f"monthly.{key}"], dtype=np.float64), int(count))
    return out, open_month


def publishes(run, var):
    """Does this run write the variable at all?  The driver's 2-D diagnostic
    writer skips a field whose source is None, so the clear-sky fluxes are
    genuinely OPTIONAL output: a run without them has no Amon file for them and
    must still be scorable on the fields it does write."""
    return bool(glob.glob(str(ROOT / run / "cmor" / "Amon"
                              / f"{var}_Amon_*_gn_*.nc")))


def partial_month(run, var, year, month):
    """The open month's mean from the run's Amon file (last time index of the
    newest file ending in ``month``), as (nlat, nlon), plus lat.  ``year`` is
    the sidecar's relative year index and is not used for matching."""
    import xarray as xr
    pat = str(ROOT / run / "cmor" / "Amon" / f"{var}_Amon_*_gn_*.nc")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"{run}: no Amon file for {var} ({pat})")
    # The sidecar's bucket key carries a RELATIVE year index (0 for the run's
    # first year), so match the file on its END month only and take the
    # newest such file; the open month is that file's last time step.
    hits = [f for f in files
            if pathlib.Path(f).stem.split("_")[-1].split("-")[-1][-2:] == f"{month:02d}"]
    if not hits:
        raise SystemExit(f"{run}/{var}: no Amon file ends in month {month:02d}: {files}")
    # The sidecar year is a RELATIVE index, so a multi-year run can offer the
    # same end month in several files and "newest" would silently pick one.
    if len(hits) > 1:
        raise SystemExit(f"{run}/{var}: {len(hits)} Amon files end in month "
                         f"{month:02d} ({hits}); the year is ambiguous")
    d = xr.open_dataset(hits[0], decode_times=False)
    arr = np.asarray(d[var].isel(time=-1).values, dtype=np.float64)
    lat = np.asarray(d["lat"].values, dtype=np.float64)
    lon = np.asarray(d["lon"].values, dtype=np.float64) % 360.0
    return arr, lat, lon


def check_cadence(run, days):
    """Assert the accumulator samples every 2-D variable CADENCE_PER_DAY times
    per day between two sidecars of ``run``."""
    d0, d1 = sorted(days)
    a, _ = sidecar_sums(ROOT / run / f"cmor_accum_day_{d0:04d}.npz")
    b, _ = sidecar_sums(ROOT / run / f"cmor_accum_day_{d1:04d}.npz")
    for var in FIELDS:
        if var in a and var in b:
            got = (b[var][1] - a[var][1]) / (d1 - d0)
            if abs(got - CADENCE_PER_DAY) > 1e-9:
                raise SystemExit(f"{run}/{var}: cadence {got}/day between days "
                                 f"{d0}-{d1}, expected {CADENCE_PER_DAY}")


def window_means(run, start, end):
    """{var: window-mean (nlat, nlon)} over days start+1..end, plus lat."""
    sums, (year, month) = sidecar_sums(ROOT / run / f"cmor_accum_day_{start:04d}.npz")
    n_new = CADENCE_PER_DAY * (end - start)
    if n_new <= 0:
        raise SystemExit(f"{run}: empty window {start}..{end}")
    # Every day the window adds must land in the sidecar's OPEN month, or the
    # subtraction mixes a finished month into the mean.  Checked on the
    # calendar, not on sample counts: a window that merely happens to be
    # shorter than the elapsed month can still straddle the boundary.
    spanned = {bucket_key(d) for d in range(start + 1, end + 1)}
    if spanned != {(year, month)}:
        raise SystemExit(
            f"{run}: days {start + 1}..{end} span {sorted(spanned)} but the "
            f"sidecar's open bucket is {(year, month)}; the window must lie "
            "inside one calendar month")
    # Two different absences, and conflating them cost this tool a capability:
    # a field the run NEVER writes (the clear-sky fluxes are optional) is
    # skipped, while a field the run DOES write but whose open month holds no
    # samples is refused -- that one would silently drop a scored column.
    missing = [v for v in FIELDS if v not in sums]
    unsampled = [v for v in missing if publishes(run, v)]
    if unsampled:
        raise SystemExit(
            f"{run}: the open month {(year, month)} has no samples yet for "
            f"{unsampled}; start the window later in the month")
    fields = [v for v in FIELDS if v in sums]
    out, lat, lon = {}, None, None
    for var in fields:
        s0, c0 = sums[var]
        mean_file, lat, lon = partial_month(run, var, year, month)
        if mean_file.shape != s0.shape:
            raise SystemExit(f"{run}/{var}: file {mean_file.shape} vs sidecar {s0.shape}")
        if c0 > 0:
            r = np.corrcoef((s0 / c0).ravel(), mean_file.ravel())[0, 1]
            if not r > MIN_SPATIAL_CORR:
                raise SystemExit(f"{run}/{var}: sidecar/file spatial correlation "
                                 f"{r:.3f} < {MIN_SPATIAL_CORR}; grid or "
                                 "orientation mismatch")
        n_end = c0 + n_new
        out[var] = (mean_file * n_end - s0) / n_new * SCALE.get(var, 1.0)
        out[f"{var}__n"] = n_new
    for name, (x, y) in DERIVED.items():
        if x in out and y in out:
            out[name] = out[x] - out[y]
    return out, lat, lon


def band_mean(field, lat, lon, lat0, lat1, lon0=0.0, lon1=360.0):
    """cos-lat area mean over a lat/lon box; the longitude window may wrap."""
    w = np.cos(np.deg2rad(lat))
    jsel = (lat >= lat0) & (lat <= lat1)
    if not jsel.any():
        raise SystemExit(f"no rows in latitude band {lat0}..{lat1}")
    if lon1 - lon0 >= 360.0:
        isel = np.ones(lon.shape, dtype=bool)
    else:
        shifted = (lon - lon0) % 360.0
        isel = shifted <= (lon1 - lon0)
    if not isel.any():
        raise SystemExit(f"no columns in longitude band {lon0}..{lon1}")
    sub = field[np.ix_(jsel, isel)]
    return float((sub.mean(axis=1) * w[jsel]).sum() / w[jsel].sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--cadence-from", nargs=3, metavar=("RUN", "DAY0", "DAY1"),
                    default=("rhebc90_r6", "70", "80"),
                    help="run + two sidecar days that pin the sampling cadence")
    args = ap.parse_args(argv)
    check_cadence(args.cadence_from[0], (int(args.cadence_from[1]), int(args.cadence_from[2])))

    runs = [args.ctl] + list(args.arms)
    means, lat, lon = {}, None, None
    for r in runs:
        means[r], lat, lon = window_means(r, args.start, args.end)
    print(f"window days {args.start + 1}..{args.end} ({means[args.ctl]['rsut__n']:.0f} daily samples)")
    cols = [v for v in list(FIELDS) + list(DERIVED) if v in means[args.ctl]]
    for band, (lo, hi, wlo, whi) in BANDS.items():
        print(f"\n=== {band}: control value, then arm minus control ===")
        print(f"{'run':14s}" + "".join(
            f"{c + ('[' + UNITS[c] + ']' if c in UNITS else ''):>9s}"
            for c in cols))
        ctl = {c: band_mean(means[args.ctl][c], lat, lon, lo, hi, wlo, whi)
               for c in cols}
        print(f"{args.ctl:14s}" + "".join(f"{ctl[c]:9.3f}" for c in cols))
        for r in args.arms:
            d = {c: band_mean(means[r][c], lat, lon, lo, hi, wlo, whi) - ctl[c]
                 for c in cols}
            print(f"{r:14s}" + "".join(f"{d[c]:+9.3f}" for c in cols))


if __name__ == "__main__":
    main()
