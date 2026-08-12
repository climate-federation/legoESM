#!/usr/bin/env python3
"""Regional bias decomposition vs CERES-EBAF + ERA5: right answer, right reason.

Splits the model-minus-CERES rsut / rlut (and rsutcs / CRE when the run
publishes clear-sky) into the regions where the known AMIP failure modes
live, so a global-mean fix that trades one region against another is VISIBLE
instead of silently compensating:

  ITCZ 10S-10N | trades 10-30 (each hemi) | Sc decks (Peru/Namibia/California
  boxes) | SH storm track 30-60S (Southern Ocean bias, Trenberth & Fasullo
  2010) | NH midlat 30-60N | poles 60-90.

Method: model CMOR Amon months (calendar labels) vs the CERES-EBAF monthly
CLIMATOLOGY of the same calendar months (2000-2025 mean), CERES regridded to
the model's 5-deg lat-lon by area-weighted block averaging (coarsen 5x5 --
exact for 1-deg -> 5-deg). Epoch mismatch (run year 1923 vs CERES 2000s) is
the SAME approximation the global scorecard already makes; stated, not
hidden.

Usage: regional_bias.py <run> [<run> ...]              model - obs
       regional_bias.py --ctl <ctl> <arm> [<arm> ...]   arm - ctl (regional)
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.environ.get(
    "LEGOESM_AMIP_RUNS", "/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
CERES = "/work/bd1179/b309141/climateeval_input/observation_CERES-EBAF/mon"
ERA5 = "/work/bd1179/b309141/climateeval_input/reanalysis_ERA5/mon"
# Reference climatology period.  ERA5 monthly starts 1979-01; CERES-EBAF
# 2000-03.  Each reference is averaged over ITS OWN full record restricted to
# >=1979, and the model is compared to the SAME CALENDAR MONTHS it simulated.
REF_MIN_YEAR = 1979

# TOA fluxes: CERES only (ERA5 has no rsut/rlut here).  Cloud state: ERA5.
_SOURCE = {"rsut": CERES, "rlut": CERES, "rsutcs": CERES, "rlutcs": CERES,
           "clt": ERA5, "lwp": ERA5, "clivi": ERA5, "prw": ERA5, "pr": ERA5,
           "tas": ERA5}

REGIONS = {           # (lat_lo, lat_hi, lon_lo, lon_hi) lon in [0,360)
    "ITCZ 10S-10N":     (-10, 10, 0, 360),
    "trades 10-30N":    (10, 30, 0, 360),
    "trades 10-30S":    (-30, -10, 0, 360),
    "Sc Peru":          (-30, -10, 260, 290),
    "Sc Namibia":       (-25, -5, 350, 375),   # 10W-15E wraps: use 350-360+0-15
    "Sc California":    (15, 35, 220, 250),
    "SO stormtrack":    (-60, -30, 0, 360),
    "NH midlat":        (30, 60, 0, 360),
    "poles 60-90":      (None, None, 0, 360),  # handled specially (both caps)
    "GLOBAL":           (-90, 90, 0, 360),
}


def _load_model(run, var):
    fs = sorted(glob.glob(
        f"{ROOT}/{run}/cmor/Amon/{var}_Amon_legoESM-1-0_amip_r1i1p1f1_gn*.nc"))
    if not fs:
        return None
    try:
        return xr.open_dataset(fs[-1], decode_times=False)
    except OSError:
        return None


def _month_labels(d):
    """Calendar month numbers from the file's OWN time units.

    The run's start year is read from the ``days since YYYY-01-01`` attribute,
    never assumed: wave-1/floor arms run 1923 while the 1979-epoch reference
    run does not, and a hardcoded epoch would silently mis-label every month.
    """
    days = np.asarray(d["time"])
    out = []
    for t in days:
        doy = t % 365.0
        m = int(np.searchsorted(np.cumsum(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), doy, side="right"))
        out.append(m + 1)
    return out


def bin_to_model(arr, rlat, rlon, mlat, mlon, label="reference"):
    """Area-weighted BIN average of a (lat, lon) field onto the model grid.

    ``np.bincount`` over cos-lat weights, not ``coarsen``: CERES is 180x360 and
    ERA5 native6 is 721x1440, and only the former is an integer multiple of the
    model 36x72.  Binning is exact for both and stays conservative on the
    pole-inclusive ERA5 latitude axis.  Public so the 3-D profile probe uses
    THIS reduction rather than a second copy that could drift from it.
    """
    dlat = float(mlat[1] - mlat[0])
    dlon = float(mlon[1] - mlon[0])
    iy = np.clip(((rlat - (mlat[0] - dlat / 2)) // dlat).astype(int),
                 0, mlat.size - 1)
    ix = np.clip(((rlon - (mlon[0] - dlon / 2)) % 360.0 // dlon).astype(int),
                 0, mlon.size - 1)
    w1 = np.cos(np.deg2rad(rlat))
    w2 = np.broadcast_to(w1[:, None], arr.shape)
    flat = iy[:, None] * mlon.size + ix[None, :]
    n = mlat.size * mlon.size
    good = np.isfinite(arr)
    num = np.bincount(flat[good], weights=(arr * w2)[good], minlength=n)
    den = np.bincount(flat[good], weights=w2[good], minlength=n)
    if (den <= 0).any():
        raise SystemExit(f"FATAL: {label} left {int((den <= 0).sum())} model "
                         "cells with no reference data -- never nanmean past this")
    return (num / den).reshape(mlat.size, mlon.size)


def _ref_clim(var, months, mlat, mlon):
    """Reference monthly climatology BINNED onto the model grid, >=REF_MIN_YEAR."""
    src = _SOURCE.get(var)
    if src is None:
        return None
    fs = sorted(glob.glob(f"{src}/{var}/*.nc"))
    if not fs:
        return None
    d = xr.open_mfdataset(fs, combine="by_coords") if len(fs) > 1 \
        else xr.open_dataset(fs[0])
    if var not in d:
        return None
    v = d[var].sel(time=slice(f"{REF_MIN_YEAR}-01-01", None))
    if v.time.size == 0:
        return None
    clim = v.groupby("time.month").mean("time").sel(month=months).mean("month")
    clim = clim.load()

    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rlon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    arr = np.asarray(clim, dtype=np.float64)
    if arr.shape != (rlat.size, rlon.size):          # (lon, lat) ordering
        arr = arr.T
    assert arr.shape == (rlat.size, rlon.size), f"{var}: unexpected ref shape"

    return bin_to_model(arr, rlat, rlon, mlat, mlon, label=var)


def region_mean(field, lat, lon, box, valid=None):
    """cos-lat area mean of ``field`` over ``box``.

    ``valid`` is an optional boolean mask of cells to KEEP.  Cells outside it
    are dropped from the weights, which is not the same as ``np.nanmean``: the
    weight normalisation is recomputed, and an EMPTY region raises instead of
    returning a silent NaN.  Used by the profile probe to exclude pressure
    levels that lie below the terrain, where CMOR publishes a clamped copy of
    the lowest model level rather than a measurement.
    """
    la0, la1, lo0, lo1 = box
    latg, long_ = np.meshgrid(lat, lon, indexing="ij")
    w = np.cos(np.deg2rad(latg))
    if la0 is None:                                        # both polar caps
        m = np.abs(latg) >= 60.0
    else:
        m = (latg >= la0) & (latg <= la1)
        if lo1 > 360:                                      # wrapping box
            m &= (long_ >= lo0) | (long_ <= lo1 - 360)
        elif (lo0, lo1) != (0, 360):
            m &= (long_ >= lo0) & (long_ <= lo1)
    if valid is not None:
        m = m & valid
    wm = np.where(m, w, 0.0)
    total = wm.sum()
    if total <= 0:
        raise SystemExit("FATAL: region has no valid cells -- refusing to "
                         "return a mean of nothing")
    return float((np.where(m, field, 0.0) * wm).sum() / total)


def _region_mean(field, lat, lon, box):
    return region_mean(field, lat, lon, box)


def main(runs, ctl=None):
    ctl_fields = {}
    if ctl is not None:
        for var in ("rsut", "rlut", "rsutcs", "clt", "prw", "hfls"):
            md = _load_model(ctl, var)
            if md is not None:
                ctl_fields[var] = (np.asarray(md[var]).mean(axis=0),
                                   np.asarray(md.lat), np.asarray(md.lon))
    for run in runs:
        if ctl is not None:
            print(f"\n=== {run} - {ctl} (regional deltas) ===")
            rows = {}
            for var, (cv, lat, lon) in ctl_fields.items():
                md = _load_model(run, var)
                if md is None:
                    continue
                mv = np.asarray(md[var]).mean(axis=0)
                rows[var] = {name: _region_mean(mv - cv, lat, lon, box)
                             for name, box in REGIONS.items()}
            hdr = list(rows.keys())
            print(f"{'region':<16}" + "".join(f"{h:>10}" for h in hdr))
            for r in REGIONS:
                print(f"{r:<16}" + "".join(f"{rows[h][r]:10.2f}" for h in hdr))
            continue
        print(f"\n=== {run} (model - CERES-EBAF climatology, matched months) ===")
        rows = {}
        for var in ("rsut", "rlut", "rsutcs", "rlutcs",
                    "clt", "lwp", "clivi", "prw"):
            md = _load_model(run, var)
            if md is None:
                continue
            months = _month_labels(md)
            lat, lon = np.asarray(md.lat), np.asarray(md.lon)
            ob = _ref_clim(var, months, lat, lon)
            if ob is None:
                continue
            mv = np.asarray(md[var]).mean(axis=0)
            ov = ob
            rows[var] = {name: _region_mean(mv - ov, lat, lon, box)
                         for name, box in REGIONS.items()}
        if not rows:
            print("  no scorable variables")
            continue
        hdr = list(rows.keys())
        if "rsut" in rows and "rsutcs" in rows:
            rows["CRE_sw_bias"] = {r: rows["rsut"][r] - rows["rsutcs"][r]
                                   for r in REGIONS}
            hdr.append("CRE_sw_bias")
        print(f"{'region':<16}" + "".join(f"{h:>13}" for h in hdr))
        for r in REGIONS:
            print(f"{r:<16}" + "".join(f"{rows[h][r]:13.2f}" for h in hdr))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--ctl":
        main(sys.argv[3:], ctl=sys.argv[2])
    else:
        main(sys.argv[1:])
