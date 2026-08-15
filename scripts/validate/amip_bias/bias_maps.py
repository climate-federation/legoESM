#!/usr/bin/env python3
"""Model / observation / bias maps against the ESMValTool reference datasets.

One row per variable, three panels: the model climatology, the observational
climatology on the same calendar months, and their difference. Plus a
pressure-latitude section for air temperature, which is where a map cannot show
the error at all (the model's temperature bias reverses sign with height).

References are the ones the ESMValTool / ClimateEval suites declare for each
variable, not a single convenient one:

    albedo, rsut, rlut   CERES-EBAF 4.2.1   (2000-2025)
    clt, clivi, lwp      ESACCI-CLOUD AVHRR (1982-2016)
    tas, ta, prw, pr     ERA5               (1979-)

EPOCH, stated because it bounds every number here. Each reference is averaged
over ITS OWN full record from 1979 on, restricted to the calendar months the
run simulated. The campaign's long runs start in 1923, so a five-year map is
NOT epoch-matched to CERES or ESACCI: some of the difference is a different
climate, not model error. `ref1979` exists to give the like-for-like
comparison and is the run to quote absolute numbers from; the long run is for
the spatial PATTERN, which is what a map is good for.

Regridding: the reference is area-weighted binned onto the model's own grid
with `regional_bias.bin_to_model` -- the same reduction the regional scorecard
uses, so a map and a table cannot disagree.

Usage:
    bias_maps.py --run cldF_fsd --out /path/maps      # 5-year pattern
    bias_maps.py --run ref1979 --out /path/maps       # 12-month, epoch-matched
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = "/work/bd1179/b309141/climateeval_input"
CERES = f"{_ROOT}/observation_CERES-EBAF/mon"
ESACCI = f"{_ROOT}/observation_ESACCI-CLOUD/mon"
ERA5 = f"{_ROOT}/reanalysis_ERA5/mon"

_spec = importlib.util.spec_from_file_location("rb", _DIR / "regional_bias.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

# (reference dataset dir, label, colourmap, bias limit, units)
SPEC = {
    "albedo": (CERES, "CERES-EBAF", "viridis", 0.15, "-"),
    "rsut": (CERES, "CERES-EBAF", "viridis", 40.0, "W m-2"),
    "rlut": (CERES, "CERES-EBAF", "inferno", 40.0, "W m-2"),
    "clt": (ESACCI, "ESACCI-CLOUD", "Blues", 40.0, "%"),
    "tas": (ERA5, "ERA5", "RdYlBu_r", 10.0, "K"),
    "prw": (ERA5, "ERA5", "YlGnBu", 10.0, "kg m-2"),
    "evspsbl": (ERA5, "ERA5", "YlGnBu", 3.0, "mm day-1"),
}

# display scaling applied to BOTH model and reference before plotting, for
# variables whose CMOR unit is not the one a reader expects to see on a map.
_DISPLAY_SCALE = {"evspsbl": 86400.0}


def _model_clim(run, var):
    """Model climatology over ALL published months, plus (lat, lon, months).

    ``albedo`` is derived as rsut/rsdt on BOTH sides rather than compared as a
    flux, so the comparison is insensitive to any insolation difference between
    the model calendar and the reference epoch.
    """
    if var == "albedo":
        up = rb._load_model(run, "rsut")
        dn = rb._load_model(run, "rsdt")
        if up is None or dn is None:
            return None
        months = rb._month_labels(up)
        # Ratio of the TIME MEANS, not the mean of the ratios: the monthly
        # ratio is undefined in polar night (rsdt -> 0) and a mean over it
        # would be dominated by that noise.
        field = np.asarray(up["rsut"]).mean(axis=0) / np.maximum(
            np.asarray(dn["rsdt"]).mean(axis=0), 1.0)
        return field, np.asarray(up.lat), np.asarray(up.lon) % 360.0, months
    d = rb._load_model(run, var)
    if d is None:
        return None
    return (np.asarray(d[var]).mean(axis=0), np.asarray(d.lat),
            np.asarray(d.lon) % 360.0, rb._month_labels(d))


def _ref_clim(var, months, mlat, mlon, src):
    """Reference climatology on the model grid, same calendar months."""
    if var == "albedo":
        up = rb._ref_clim("rsut", months, mlat, mlon, src=src)
        dn = rb._ref_clim("rsdt", months, mlat, mlon, src=src)
        if up is None or dn is None:
            return None
        return up / np.maximum(dn, 1.0)
    out = rb._ref_clim(var, months, mlat, mlon, src=src)
    if out is None:
        return None
    if var == "clt" and np.nanmax(out) <= 1.5:
        out = out * 100.0          # a fraction where the model publishes %
    return out


def grid_scale_residual(f):
    """|f - mean of its 4 neighbours|, normalised by the field's own spatial sd.

    A high-pass residual, NOT a 2-dx measure -- a merely sharper smooth feature
    raises it too. It is quoted only as a RATIO of model to observation on the
    SAME grid after the SAME regridding, which is what makes it interpretable:
    a field whose ratio is ~1 is as smooth as the reference, and one at 3 is
    carrying structure the reference does not have.
    """
    n = (np.roll(f, 1, 0) + np.roll(f, -1, 0)
         + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 4.0
    return float(np.mean(np.abs(f - n)[1:-1]) / np.std(f))


def _gm(field, lat, lon):
    return rb.region_mean(field, lat, lon, rb.REGIONS["GLOBAL"])


def _panel(ax, lon, lat, field, title, cmap, vmin, vmax, units):
    import cartopy.crs as ccrs
    lon_p = np.append(lon, lon[0] + 360.0)
    fld = np.concatenate([field, field[:, :1]], axis=1)   # close the seam
    m = ax.pcolormesh(lon_p, lat, fld, cmap=cmap, vmin=vmin, vmax=vmax,
                      shading="nearest", transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.4, color="0.25")
    ax.set_global()
    ax.set_title(title, fontsize=8)
    return m


def maps(run, variables, out_dir):
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for var in variables:
        got = _model_clim(run, var)
        if got is None:
            print(f"  {var}: no model output in {run} -- skipped")
            continue
        field, mlat, mlon, months = got
        src, label, cmap, blim, units = SPEC[var]
        ref = _ref_clim(var, months, mlat, mlon, src)
        if ref is None:
            print(f"  {var}: no {label} reference -- skipped")
            continue
        scale = _DISPLAY_SCALE.get(var, 1.0)
        rows.append((var, field * scale, ref * scale, mlat, mlon, label, cmap,
                     blim, units))
    if not rows:
        raise SystemExit(f"{run}: nothing to plot")

    proj = ccrs.Robinson(central_longitude=180)
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(13.5, 3.0 * len(rows)),
        subplot_kw={"projection": proj}, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, (var, field, ref, lat, lon, label, cmap, blim, units) in enumerate(rows):
        lo = float(min(np.nanpercentile(field, 2), np.nanpercentile(ref, 2)))
        hi = float(max(np.nanpercentile(field, 98), np.nanpercentile(ref, 98)))
        m1 = _panel(axes[i, 0], lon, lat, field,
                    f"legoESM {var}   global {_gm(field, lat, lon):.3g} {units}",
                    cmap, lo, hi, units)
        _panel(axes[i, 1], lon, lat, ref,
               f"{label} {var}   global {_gm(ref, lat, lon):.3g} {units}",
               cmap, lo, hi, units)
        d = field - ref
        m3 = _panel(axes[i, 2], lon, lat, d,
                    f"bias   global {_gm(d, lat, lon):+.3g} {units}   "
                    f"RMS {np.sqrt(np.mean(d ** 2)):.3g}",
                    "RdBu_r", -blim, blim, units)
        fig.colorbar(m1, ax=axes[i, :2], shrink=0.85, pad=0.01)
        fig.colorbar(m3, ax=axes[i, 2], shrink=0.85, pad=0.01)

    print(f"  grid-scale residual (model / {'obs':>3}), same grid + same regrid:")
    for var, field, ref, *_ in rows:
        a, b = grid_scale_residual(field), grid_scale_residual(ref)
        flag = "  <-- carries structure the reference does not" if a / b > 1.3 else ""
        print(f"    {var:<8}{a:7.4f} /{b:7.4f}  = {a / b:5.2f}{flag}")

    n_months = len(rows[0][0]) and len(_model_clim(run, rows[0][0])[3])
    fig.suptitle(f"{run}: {n_months}-month climatology vs the ESMValTool "
                 f"reference datasets", fontsize=11)
    out = pathlib.Path(out_dir) / f"bias_maps_{run}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def ta_section(run, out_dir):
    """Pressure-latitude air-temperature bias vs ERA5.

    A surface map cannot show this error: the model is several K COLD through
    the lower free troposphere and up to +11 K WARM near 100 hPa, so any
    column-integrated or surface view averages the two together.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    mt = rb._load_model(run, "ta")
    if mt is None:
        print(f"  ta: not published by {run} -- skipped")
        return None
    months = rb._month_labels(mt)
    mlat = np.asarray(mt.lat, dtype=np.float64)
    mlon = np.asarray(mt.lon, dtype=np.float64) % 360.0
    plev_all = np.asarray(mt.plev, dtype=np.float64)
    keep = plev_all >= 10000.0           # above the lid CMOR clamps, not masks
    plev = plev_all[keep]
    Tm = np.asarray(mt["ta"]).mean(axis=0)[keep]

    fs = sorted(glob.glob(f"{ERA5}/ta/*.nc"))
    d = xr.open_dataset(fs[0])
    v = d["ta"].sel(time=slice("1979-01-01", None))
    clim = v.groupby("time.month").mean("time").sel(month=months).mean("month").load()
    ep = np.asarray(clim["plev"], dtype=np.float64)
    arr = np.asarray(clim.transpose("plev", "lat", "lon"), dtype=np.float64)
    rlat = np.asarray(clim["lat"], dtype=np.float64)
    rlon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    binned = np.stack([rb.bin_to_model(arr[k], rlat, rlon, mlat, mlon,
                                       label="ERA5 ta") for k in range(ep.size)])
    order = np.argsort(-ep)
    lep = np.log(ep[order])
    Te = np.empty_like(Tm)
    for j in range(mlat.size):
        for i in range(mlon.size):
            Te[:, j, i] = np.interp(np.log(plev)[::-1], lep[::-1],
                                    binned[order, j, i][::-1])[::-1]

    w = np.cos(np.deg2rad(mlat))
    zm = lambda f: (f * w[None, :, None]).sum(axis=(1, 2)) / (  # noqa: E731
        w.sum() * mlon.size)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for a, (fld, ttl, cmap, lim) in zip(ax, [
            (Tm.mean(axis=2), "legoESM ta [K]", "RdYlBu_r", None),
            ((Tm - Te).mean(axis=2), "legoESM - ERA5 [K]", "RdBu_r", 12.0)]):
        if lim is None:
            m = a.contourf(mlat, plev / 100.0, fld, 20, cmap=cmap)
        else:
            m = a.contourf(mlat, plev / 100.0, fld, np.linspace(-lim, lim, 25),
                           cmap=cmap, extend="both")
        a.invert_yaxis()
        a.set_yscale("log")
        a.set_yticks([1000, 850, 700, 500, 300, 200, 100])
        a.set_yticklabels([1000, 850, 700, 500, 300, 200, 100])
        a.set_xlabel("latitude")
        a.set_ylabel("pressure [hPa]")
        a.set_title(ttl, fontsize=9)
        fig.colorbar(m, ax=a, shrink=0.9)
    prof = zm(Tm - Te)
    fig.suptitle(f"{run}: zonal-mean air temperature vs ERA5 "
                 f"(global bias {prof.min():+.1f} K at "
                 f"{plev[int(np.argmin(prof))] / 100:.0f} hPa, "
                 f"{prof.max():+.1f} K at "
                 f"{plev[int(np.argmax(prof))] / 100:.0f} hPa)", fontsize=10)
    out = pathlib.Path(out_dir) / f"ta_section_{run}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--vars", nargs="+",
                    default=["albedo", "rsut", "rlut", "clt", "tas", "prw"])
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-section", action="store_true")
    args = ap.parse_args(argv)
    print(f"{args.run}:")
    maps(args.run, [v for v in args.vars if v in SPEC], args.out)
    if not args.no_section:
        ta_section(args.run, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
