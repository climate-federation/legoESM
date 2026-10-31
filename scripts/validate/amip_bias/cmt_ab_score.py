#!/usr/bin/env python3
"""Pre-registered score of the #2266 Emanuel momentum-transport A/B (off vs on).

    cmt_ab_score.py --off <run dir> --on <run dir> --era5-grib <ERA5 daily GRIB>
                    --start 1979-01-01 --ndays 30 --out <dir>

Both arms' CMOR Amon ``ua`` (the exact mean over the simulated window) against
the ERA5 daily mean over the same days, built here with cdo (never taken
prebuilt, so its window is always checked): first ``ndays`` daily means of the GRIB (reduced -> regular Gaussian,
``setgridtype,regular``), time mean, conservative remap
(``remapcon``) onto the arm's own CMOR grid; the remap is first checked on a
known field, sin(lat), whose exact cell mean on a regular lat band is
(sin(lat0) + sin(lat1)) / 2, and on the global mean of the remapped u.
Window checks (refused otherwise): the GRIB's selected dates are exactly
start .. start + ndays - 1; each arm's single Amon record starts in the start
month and the arm's newest checkpoint is day ``ndays`` (CMOR month bounds are
nominal, so the checkpoint is what proves the averaging window).

Metric (PREREGISTRATION.md, user-approved 2026-10-31): RMSE vs ERA5 of the
zonal-mean u profile u(lat) at 850 and 200 hPa, cos-lat weighted over the
5-degree bands, tropics (20S-20N) and global.  Delta = on - off; % of off.
Also printed, never substituted for the verdict: band-mean zonal-mean u, the
2-D (lat, lon) RMSE, tropical mean precipitation.  Figure: tropical and global
zonal-mean u profiles (off, on, ERA5) and the on-minus-off latitude-height
section of zonal-mean u, Okabe-Ito colours.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regional_bias as rb  # noqa: E402

BOXES = {"tropics": (-20, 20, 0, 360), "global": (-90, 90, 0, 360)}
LEVELS_PA = (85000.0, 20000.0)
OKABE_ITO = {"off": "#E69F00", "on": "#0072B2", "ERA5": "#000000"}
DAY_S = 86400.0


def big(x, frac):
    """max(x m/s, frac of the off-arm value): the pre-registered size."""
    return lambda off: max(x, frac * abs(off))


BIG, SMALL = big(0.5, 0.10), big(0.25, 0.05)


def prereg_verdict(rmse_off, rmse_on):
    """The approved rule.  ``rmse_*``: {(box, plev): RMSE}; plev in LEVELS_PA."""
    t = [(rmse_on[("tropics", p)] - rmse_off[("tropics", p)], rmse_off[("tropics", p)])
         for p in LEVELS_PA]
    g = [(rmse_on[("global", p)] - rmse_off[("global", p)], rmse_off[("global", p)])
         for p in LEVELS_PA]
    if any(d >= BIG(o) for d, o in t) or not any(-d >= SMALL(o) for d, o in t):
        return "REFUTE"
    for i in (0, 1):
        (d_i, o_i), (d_j, o_j) = t[i], t[1 - i]
        if (-d_i >= BIG(o_i) and d_j < SMALL(o_j)
                and all(d < SMALL(o) for d, o in g)):
            return "CONFIRM"
    return "INCONCLUSIVE"


def profile_rmse(zm_model, zm_ref, lat, box):
    """cos-lat weighted RMSE of a zonal-mean profile u(lat) over ``box``."""
    err2 = ((zm_model - zm_ref) ** 2)[:, None] * np.ones((1, 2))
    return float(np.sqrt(rb.region_mean(err2, lat, np.array([0.0, 180.0]), box)))


def field_rmse(m, r, lat, lon, box):
    return float(np.sqrt(rb.region_mean((m - r) ** 2, lat, lon, box)))


def band_mean(zm, lat, box):
    return rb.region_mean(zm[:, None] * np.ones((1, 2)), lat, np.array([0.0, 180.0]), box)


def _cdo(*args):
    r = subprocess.run(["cdo", "-s", "-L", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: cdo {' '.join(args)}\n{r.stderr}")
    return r.stdout


def _read(path, var):
    import xarray as xr
    d = xr.open_dataset(path, decode_times=False)
    a = np.asarray(d[var], dtype=np.float64)
    lat = np.asarray(d["lat"], dtype=np.float64)
    lon = np.asarray(d["lon"], dtype=np.float64)
    plev = np.asarray(d["plev"], dtype=np.float64) if "plev" in d.coords else None
    return a, lat, lon, plev


def check_dates(dates, start, ndays):
    """``dates`` (YYYY-MM-DD strings) must be the ``ndays`` consecutive days from ``start``."""
    want = [str(d) for d in np.arange(np.datetime64(start), np.datetime64(start) + ndays)]
    if list(dates) != want:
        raise SystemExit(f"FATAL: ERA5 dates {list(dates)[:3]}..{list(dates)[-1:]} "
                         f"are not {want[0]}..{want[-1]}")


def check_arm_window(run, start, ndays):
    """The arm stopped at day ``ndays`` and its Amon record is the start month."""
    import xarray as xr
    ck = sorted(glob.glob(f"{run}/checkpoint_day_*.npz"))
    last = int(Path(ck[-1]).stem.split("_")[-1]) if ck else None
    if last != ndays:
        raise SystemExit(f"FATAL: {run} newest checkpoint is day {last}, expected {ndays}")
    f = _one(run, "ua")
    t0 = xr.open_dataset(f).time_bnds.values[0][0]
    if (t0.year, t0.month, t0.day) != tuple(int(x) for x in start.split("-")[:2]) + (1,):
        raise SystemExit(f"FATAL: {f} starts {t0}, not in the month of {start}")


def build_era5(grb, target_nc, out_nc, ndays, work, start):
    """ERA5 ``ndays`` daily-mean u, time mean, remapcon onto ``target_nc``'s grid.
    Returns the remap check numbers (known field + global-mean)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    grid = work / "target_grid.txt"
    grid.write_text(_cdo("griddes", str(target_nc)))
    tm = work / "era5_ua_timmean_native.nc"
    # Reduced Gaussian -> regular Gaussian first (remapcon needs cell corners),
    # as snowheat_ab_score.py does for the same ERA5 pool.
    check_dates(_cdo("showdate", f"-seltimestep,1/{ndays}", str(grb)).split(), start, ndays)
    _cdo("-f", "nc4", "timmean", f"-seltimestep,1/{ndays}", "-setgridtype,regular", str(grb), str(tm))
    _cdo("-f", "nc4", f"remapcon,{grid}", str(tm), str(out_nc))
    # Known field: sin(lat) on the native grid, conservatively remapped.
    known = work / "known_sinlat_5deg.nc"
    _cdo("-f", "nc4", f"remapcon,{grid}", "-expr,k=sin(clat(var131)*M_PI/180)",
         f"-sellevel,85000", str(tm), str(known))
    k, lat, lon, _ = _read(known, "k")
    dlat = np.diff(lat).mean()
    exact = 0.5 * (np.sin(np.deg2rad(lat - dlat / 2)) + np.sin(np.deg2rad(lat + dlat / 2)))
    known_err = float(np.max(np.abs(np.squeeze(k) - exact[:, None])))
    # Global mean of u before / after the remap, per level.
    g_nat = np.array(_cdo("outputf,%.10g", "-fldmean", str(tm)).split(), dtype=float)
    g_5 = np.array(_cdo("outputf,%.10g", "-fldmean", str(out_nc)).split(), dtype=float)
    return {"known_field_max_abs_err": known_err,
            "global_mean_max_abs_diff_m_s": float(np.max(np.abs(g_nat - g_5))),
            "ndays": ndays, "start": start}


def _one(run, var):
    fs = sorted(glob.glob(f"{run}/cmor/Amon/{var}_Amon_*.nc"))
    if len(fs) != 1:
        raise SystemExit(f"FATAL: expected one Amon {var} file in {run}, got {fs}")
    return fs[0]


def _model(run, var):
    fs = [_one(run, var)]
    a, lat, lon, plev = _read(fs[0], var)
    if a.shape[0] != 1:
        raise SystemExit(f"FATAL: {fs[0]} has {a.shape[0]} time steps, expected 1")
    if not np.isfinite(a).all():
        raise SystemExit(f"FATAL: non-finite {var} in {fs[0]}")
    return a[0], lat, lon, plev


def score(off_dir, on_dir, era5_nc):
    u = {k: _model(d, "ua") for k, d in (("off", off_dir), ("on", on_dir))}
    _, lat, lon, plev = u["off"]
    for k in ("on",):
        if not (np.array_equal(u[k][1], lat) and np.array_equal(u[k][2], lon)
                and np.array_equal(u[k][3], plev)):
            raise SystemExit("FATAL: arms are on different grids")
    e, elat, elon, eplev = _read(era5_nc, "var131" if "var131" in _vars(era5_nc) else "u")
    e = np.squeeze(e, axis=0) if e.ndim == 4 else e
    if not (np.allclose(elat, lat) and np.allclose(elon, lon)):
        raise SystemExit("FATAL: ERA5 reference is not on the CMOR grid")
    if not np.isfinite(e).all():
        raise SystemExit("FATAL: non-finite ERA5 reference")
    out = {"rmse": {}, "rmse_2d": {}, "band_mean_u": {}, "pr_tropics_mm_day": {}}
    zm = {k: u[k][0].mean(-1) for k in u}
    zm["ERA5"] = np.stack([e[list(eplev).index(p)].mean(-1) for p in plev])
    for k in ("off", "on"):
        for box in BOXES:
            for p in LEVELS_PA:
                i = list(plev).index(p)
                out["rmse"][f"{k}|{box}|{int(p / 100)}"] = profile_rmse(
                    zm[k][i], zm["ERA5"][i], lat, BOXES[box])
                out["rmse_2d"][f"{k}|{box}|{int(p / 100)}"] = field_rmse(
                    u[k][0][i], e[list(eplev).index(p)], lat, lon, BOXES[box])
        pr, plat, plon, _ = _model(off_dir if k == "off" else on_dir, "pr")
        out["pr_tropics_mm_day"][k] = rb.region_mean(pr, plat, plon, BOXES["tropics"]) * DAY_S
    for k in ("off", "on", "ERA5"):
        for box in BOXES:
            for p in LEVELS_PA:
                i = list(plev).index(p)
                out["band_mean_u"][f"{k}|{box}|{int(p / 100)}"] = band_mean(zm[k][i], lat, BOXES[box])
    r = {k: {(b, p): out["rmse"][f"{k}|{b}|{int(p / 100)}"] for b in BOXES for p in LEVELS_PA}
         for k in ("off", "on")}
    out["verdict"] = prereg_verdict(r["off"], r["on"])
    return out, zm, lat, plev


def _vars(path):
    import xarray as xr
    return list(xr.open_dataset(path, decode_times=False).data_vars)


def figure(zm, lat, plev, path, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    p_hpa = plev / 100.0
    for a, box in zip(ax[:2], BOXES):
        for k in ("off", "on", "ERA5"):
            prof = [band_mean(zm[k][i], lat, BOXES[box]) for i in range(plev.size)]
            a.plot(prof, p_hpa, color=OKABE_ITO[k], lw=2.0 if k != "ERA5" else 1.5,
                   ls="-" if k != "ERA5" else "--", label=f"CMT {k}" if k != "ERA5" else "ERA5")
        a.axvline(0.0, color="0.6", lw=0.8)
        a.set_yscale("log"); a.invert_yaxis()
        a.set_yticks([1000, 850, 500, 200, 100, 10, 1]); a.set_yticklabels([1000, 850, 500, 200, 100, 10, 1])
        a.set_xlabel("zonal-mean u [m/s]"); a.set_ylabel("pressure [hPa]")
        a.set_title(f"{box} ({'20S-20N' if box == 'tropics' else '90S-90N'}), cos-lat mean")
        a.legend(frameon=False)
    d = zm["on"] - zm["off"]
    lim = max(float(np.max(np.abs(d))), 1e-6)
    h = ax[2].contourf(lat, p_hpa, d, levels=np.linspace(-lim, lim, 21), cmap="PuOr_r", extend="both")
    ax[2].contour(lat, p_hpa, zm["off"], levels=np.arange(-40, 81, 10), colors="0.35", linewidths=0.6)
    ax[2].set_yscale("log"); ax[2].invert_yaxis()
    ax[2].set_yticks([1000, 500, 200, 100, 10, 1]); ax[2].set_yticklabels([1000, 500, 200, 100, 10, 1])
    ax[2].set_xlabel("latitude"); ax[2].set_title("zonal-mean u, CMT on minus off [m/s]\n(contours: off arm, 10 m/s)")
    fig.colorbar(h, ax=ax[2])
    fig.suptitle(f"Emanuel convective momentum transport A/B, AMIP {window}, CMOR monthly ua vs ERA5 (#2266)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--era5-grib", required=True)
    ap.add_argument("--start", required=True, help="first simulated day, YYYY-MM-DD")
    ap.add_argument("--ndays", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in (a.off, a.on):
        check_arm_window(run, a.start, a.ndays)
    era5 = out / f"era5_ua_{a.start}_{a.ndays}d_cmorgrid.nc"
    rec = {"remap_check": build_era5(a.era5_grib, _one(a.off, "ua"), era5, a.ndays,
                                     out / "era5_work", a.start)}
    s, zm, lat, plev = score(a.off, a.on, era5)
    rec.update(s)
    end = str(np.datetime64(a.start) + a.ndays - 1)
    figure(zm, lat, plev, out / "cmt_ab_zonal_u.png", f"{a.start}..{end}")
    (out / "cmt_ab_score.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
