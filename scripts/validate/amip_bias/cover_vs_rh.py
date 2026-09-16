#!/usr/bin/env python3
"""Is the tropical cloud-cover excess in the CLOSURE, or in the humidity?

The model carries roughly twenty percentage points too much total cloud cover in
the deep tropics.  Two explanations have opposite fixes.  Either the atmosphere
is too moist and any reasonable cover closure would then produce too much cloud,
or the atmosphere's humidity is about right and the closure turns it into far
too much cover.  Comparing cover maps cannot separate them; comparing cover AT
MATCHED HUMIDITY can:

  * plot observed cover against reanalysis humidity, and modelled cover against
    the model's OWN humidity, in the same humidity bins;
  * if the two curves lie on top of each other, the cover excess is entirely a
    humidity-distribution problem and the closure is exonerated;
  * if the model curve sits above the observed one at the same humidity, the
    closure itself makes too much cloud and no amount of drying fixes it.

Observed cover is ESACCI-CLOUD, reanalysis humidity is ERA5, both binned onto
the model's own grid with the shared conservative binning used by the rest of
these validators, over the model's own calendar months.  Relative humidity is
computed from temperature and specific humidity with the model's own saturation
curve on BOTH sides, so no part of the comparison re-derives thermodynamics.

    cover_vs_rh.py --run g30_ctl --plev 70000
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("regional_bias", _HERE / "regional_bias.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

ESACCI = "/work/bd1179/b309141/climateeval_input/observation_ESACCI-CLOUD/mon"
ERA5 = "/work/bd1179/b309141/climateeval_input/reanalysis_ERA5/mon"
# Wide enough at the ends to hold the tails, fine where the Sundqvist closure
# is steep.  The last edge is above 1 because a grid-box mean can be
# supersaturated in the model even though the observations cannot be.
RH_EDGES = np.array([0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0])
MIN_CELLS = 20


def relative_humidity(q, T, p):
    """RH from specific humidity, temperature and pressure [Pa], using the
    model's own saturation curve (never a re-derived Magnus/Tetens fit)."""
    from legoesm.thermo import saturation_mixing_ratio
    r = q / np.maximum(1.0 - q, 1e-12)          # specific humidity -> mixing ratio
    r_sat = np.asarray(saturation_mixing_ratio(T, np.broadcast_to(p, T.shape)))
    return r / np.maximum(r_sat, 1e-12)


def binned_cover(rh, cover, weights):
    """(cover per RH bin, cell count per bin), area-weighted.

    Cells where either field is missing are dropped from BOTH sides, so the two
    curves are computed over the same cells.
    """
    ok = np.isfinite(rh) & np.isfinite(cover)
    idx = np.clip(np.digitize(rh[ok], RH_EDGES) - 1, 0, len(RH_EDGES) - 2)
    w, c = weights[ok], cover[ok]
    n = len(RH_EDGES) - 1
    num = np.bincount(idx, weights=w * c, minlength=n)
    den = np.bincount(idx, weights=w, minlength=n)
    cnt = np.bincount(idx, minlength=n)
    curve = np.where(cnt >= MIN_CELLS, num / np.maximum(den, 1e-30), np.nan)
    return curve, cnt


def _level_ref(var, months, mlat, mlon, plev):
    """One pressure level of a 3-D ERA5 field, on the model grid."""
    import glob
    import xarray as xr
    fs = sorted(glob.glob(f"{ERA5}/{var}/*.nc"))
    if not fs:
        raise SystemExit(f"no ERA5 {var} under {ERA5}")
    d = xr.open_mfdataset(fs, combine="by_coords") if len(fs) > 1 else xr.open_dataset(fs[0])
    if var not in d:
        raise SystemExit(f"ERA5 {var}: variable absent from {fs}")
    v = d[var].sel(time=slice(f"{rb.REF_MIN_YEAR}-01-01", None))
    if v.time.size == 0:
        raise SystemExit(f"ERA5 {var}: no months at or after {rb.REF_MIN_YEAR}")
    lev_name = next((c for c in ("plev", "level", "lev") if c in v.dims), None)
    if lev_name is None:
        raise SystemExit(f"ERA5 {var}: no pressure dimension in {v.dims}")
    lev = np.asarray(v[lev_name], dtype=np.float64)
    # ERA5 stores hPa in some archives and Pa in others; decide from the data,
    # never from the attribute, and say which was used.
    scale = 100.0 if lev.max() < 2000.0 else 1.0
    k = int(np.argmin(np.abs(lev * scale - plev)))
    got = float(lev[k] * scale)
    if abs(got - plev) > 5000.0:
        raise SystemExit(f"ERA5 {var}: nearest level to {plev} Pa is {got} Pa")
    clim = (v.isel({lev_name: k}).groupby("time.month").mean("time")
            .sel(month=months).mean("month").load())
    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rlon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    arr = np.asarray(clim, dtype=np.float64)
    if arr.shape != (rlat.size, rlon.size):
        arr = arr.T
    return rb.bin_to_model(arr, rlat, rlon, mlat, mlon, label=f"ERA5 {var}"), got


def _level_model(run, var, plev):
    d = rb._load_model(run, var)
    if d is None:
        raise SystemExit(f"{run}: no {var} output")
    lev_name = next((c for c in ("plev", "level", "lev") if c in d[var].dims), None)
    if lev_name is None:
        raise SystemExit(f"{run}/{var}: no pressure dimension in {d[var].dims}")
    lev = np.asarray(d[lev_name], dtype=np.float64)
    scale = 100.0 if lev.max() < 2000.0 else 1.0
    k = int(np.argmin(np.abs(lev * scale - plev)))
    return (np.asarray(d[var].isel({lev_name: k}), dtype=np.float64).mean(axis=0),
            float(lev[k] * scale))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--plev", type=float, default=70000.0, help="pressure level [Pa]")
    ap.add_argument("--band", nargs=2, type=float, default=(-30.0, 30.0),
                    metavar=("LAT0", "LAT1"))
    args = ap.parse_args(argv)

    clt_d = rb._load_model(args.run, "clt")
    if clt_d is None:
        raise SystemExit(f"{args.run}: no clt output")
    months = rb._month_labels(clt_d)
    mlat = np.asarray(clt_d.lat, dtype=np.float64)
    mlon = np.asarray(clt_d.lon, dtype=np.float64) % 360.0
    clt_m = np.asarray(clt_d["clt"], dtype=np.float64).mean(axis=0)

    hus_m, lev_m = _level_model(args.run, "hus", args.plev)
    ta_m, _ = _level_model(args.run, "ta", args.plev)
    rh_m = relative_humidity(hus_m, ta_m, lev_m)

    clt_o = rb._ref_clim("clt", months, mlat, mlon, src=ESACCI)
    if clt_o is None:
        raise SystemExit("ESACCI clt unavailable")
    if np.nanmax(clt_o) <= 1.5:
        clt_o = clt_o * 100.0
    hus_o, lev_o = _level_ref("hus", months, mlat, mlon, args.plev)
    ta_o, _ = _level_ref("ta", months, mlat, mlon, args.plev)
    rh_o = relative_humidity(hus_o, ta_o, lev_o)

    lat0, lat1 = args.band
    sel = (mlat >= lat0) & (mlat <= lat1)
    w = np.broadcast_to(np.cos(np.deg2rad(mlat))[:, None], clt_m.shape)[sel]
    cm, nm = binned_cover(rh_m[sel], clt_m[sel], w)
    co, no = binned_cover(rh_o[sel], clt_o[sel], w)

    print(f"{args.run}: total cloud cover at matched relative humidity, "
          f"{lat0:g} to {lat1:g} degrees, months {months}")
    print(f"  model level {lev_m:.0f} Pa, ERA5 level {lev_o:.0f} Pa, "
          f"observed cover ESACCI-CLOUD")
    print(f"{'RH bin':>12s}{'model %':>10s}{'obs %':>10s}{'model-obs':>11s}"
          f"{'n model':>9s}{'n obs':>7s}")
    for i in range(len(RH_EDGES) - 1):
        d = cm[i] - co[i]
        print(f"{RH_EDGES[i]:5.2f}-{RH_EDGES[i+1]:<6.2f}"
              f"{cm[i]:10.1f}{co[i]:10.1f}{d:11.1f}{nm[i]:9d}{no[i]:7d}")
    okm, oko = nm >= MIN_CELLS, no >= MIN_CELLS
    print(f"\n  area-mean cover: model {np.average(clt_m[sel], weights=w):.1f} %, "
          f"observed {np.average(clt_o[sel], weights=w):.1f} %")
    print(f"  area-mean RH:    model {np.average(rh_m[sel], weights=w):.3f}, "
          f"reanalysis {np.average(rh_o[sel], weights=w):.3f}")
    both = okm & oko
    if both.any():
        print(f"  mean cover difference at MATCHED humidity: "
              f"{np.nanmean((cm - co)[both]):+.1f} percentage points "
              f"over {both.sum()} shared bins")


if __name__ == "__main__":
    main()
