#!/usr/bin/env python3
"""legoESM AMIP bias against the CMIP6 AMIP multi-model ensemble.

Same observations, regridding, target grid, area weighting, TOA-albedo
definition and validity mask as ``bias_maps.py`` -- every one of those is
imported from there (and from ``regional_bias.py``), never re-derived.  What
this script adds is the ensemble:

* window   -- the (year, month) pairs the legoESM run actually published, read
              from its CMOR time axis.  Each CMIP6 model is averaged over the
              IDENTICAL months.  A window outside the CMIP6 AMIP period is a
              hard error; a model lacking any window month is left out and
              listed, never substituted.
* members  -- one per model: the lowest-f ``r1i1p1f*`` member, latest version
              directory.  Recorded in the CSV.
* spread   -- z = (legoESM bias) / (inter-model standard deviation of the
              CMIP6 biases), per cell, all against the same reference on the
              same grid.  |z| > 2: legoESM's bias exceeds twice the inter-model
              spread (not a min-max range test).

Every statistic is taken over ONE per-variable cell set: the cells where
legoESM, the reference and every included model all carry a value.  Only the
TOA albedo is ever masked (rsdt > threshold), so for every other variable this
is the whole globe.

Usage:
    cmip6_ensemble_bias.py --run logq30_base --out /path/figs
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import pathlib
import re
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bm", _DIR / "bias_maps.py")
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)
rb = bm.rb

CMIP6_AMIP = os.environ.get("CMIP6_AMIP_ROOT", "/pool/data/CMIP6/data/CMIP")
AMIP_FIRST, AMIP_LAST = (1979, 1), (2014, 12)   # CMIP6 amip protocol period
VARS = ["albedo", "rsut", "rlut", "clt", "tas", "prw", "evspsbl", "pr"]
# CMOR units each CMIP6 file must carry -- the legoESM side and the reference
# are compared in these units by bias_maps, so a model in any other unit would
# be a silent unit error, not a bias.
_UNITS = {"rsut": "W m-2", "rsdt": "W m-2", "rlut": "W m-2", "clt": "%",
          "tas": "K", "prw": "kg m-2", "evspsbl": "kg m-2 s-1",
          "pr": "kg m-2 s-1"}
Z_LIM = 3.0
# z is undefined where the models agree to within nothing: polar-night rsut
# (every model 0), desert pr/evspsbl.  Cells whose inter-model spread is below
# this fraction of the variable's median spread (median over the valid cells)
# are greyed out and left out of the |z| > 2 fraction.  User-approved
# 2026-10-15.
SPREAD_FLOOR_FRAC = 0.1
_MONTH = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


# ---------------------------------------------------------------- window ----
def run_window(run, var="rsut"):
    """(year, month) pairs the legoESM run published, from its CMOR time axis.

    ``bias_maps`` (via ``rb._load_model``) reads only the LAST CMOR file of a
    run, so the legoESM field covers that one file.  A run split over several
    files would make the window and the field disagree with the run's true
    coverage; refuse it rather than compare a truncated window.
    """
    import cftime
    n = len(glob.glob(f"{rb.ROOT}/{run}/cmor/Amon/"
                      f"{var}_Amon_legoESM-1-0_amip_r1i1p1f1_gn*.nc"))
    if n > 1:
        raise SystemExit(f"FATAL: {run} has {n} CMOR {var} files; bias_maps "
                         "reads only the last one -- extend it before use")
    d = rb._load_model(run, var)
    if d is None:
        raise SystemExit(f"{run}: no CMOR {var} -- cannot read the run window")
    t = d["time"]
    dates = cftime.num2date(np.asarray(t), t.attrs["units"],
                            t.attrs.get("calendar", "standard"))
    return sorted({(x.year, x.month) for x in dates})


def check_window(window):
    """Raise unless every window month lies inside the CMIP6 AMIP period."""
    bad = [ym for ym in window if not AMIP_FIRST <= ym <= AMIP_LAST]
    if bad:
        raise SystemExit(f"FATAL: run window months {bad} lie outside the CMIP6 "
                         f"AMIP period {AMIP_FIRST}-{AMIP_LAST}; no fallback")


def window_label(window):
    if len(window) == 1:
        y, m = window[0]
        return f"{_MONTH[m - 1]} {y} single month"
    (y0, m0), (y1, m1) = window[0], window[-1]
    return (f"{_MONTH[m0 - 1][:3]} {y0} - {_MONTH[m1 - 1][:3]} {y1} "
            f"({len(window)} months)")


# ----------------------------------------------------------- CMIP6 files ----
def _f_index(member):
    return int(re.fullmatch(r"r1i1p1f(\d+)", member).group(1))


def list_models(root=CMIP6_AMIP):
    """{model: (institute, member)} with the lowest-f r1i1p1f* member."""
    out = {}
    for d in sorted(glob.glob(f"{root}/*/*/amip")):
        inst, model = d.split("/")[-3:-1]
        mems = [m for m in os.listdir(d) if re.fullmatch(r"r1i1p1f\d+", m)]
        if mems:
            out[model] = (inst, min(mems, key=_f_index))
    return out


def model_files(root, inst, model, member, var):
    """(grid, version, files) for one variable; None if not published."""
    base = f"{root}/{inst}/{model}/amip/{member}/Amon/{var}"
    grids = sorted(os.listdir(base)) if os.path.isdir(base) else []
    if not grids:
        return None
    if len(grids) > 1:
        raise SystemExit(f"FATAL: {model} {var} has grids {grids}; "
                         "choosing one is a decision, not a default")
    vers = sorted(os.listdir(f"{base}/{grids[0]}"))
    # The standard CMIP6 filename only: GISS-E2-1-G also ships every file a
    # second time as "<var>_Amon_amip_<model>_...", so a bare *.nc glob reads
    # each month twice.
    files = sorted(glob.glob(f"{base}/{grids[0]}/{vers[-1]}/"
                             f"{var}_Amon_{model}_amip_{member}_{grids[0]}_*.nc"))
    return grids[0], vers[-1], files


def _overlaps(path, window):
    """Does a CMOR filename's YYYYMM-YYYYMM range touch the window?"""
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", path)
    if m is None:
        return True
    a, b = int(m.group(1)), int(m.group(2))
    return any(a <= y * 100 + mo <= b for y, mo in window)


def window_mean(files, var, window):
    """Mean over the window months of one CMIP6 variable on its native grid.

    Returns ``(field[lat, lon], lat, lon)``; raises ``LookupError`` with the
    reason when the model cannot be used (missing month, wrong unit, grid that
    is not a 1-D lat/lon product).
    """
    import xarray as xr
    if not files:
        raise LookupError("no CMOR-named file in the version directory")
    files = [f for f in files if _overlaps(f, window)]
    if not files:
        raise LookupError("no file covers the window")
    want = set(window)
    slices, seen, lat = [], [], None
    for path in files:              # no dask here: read only the window slices
        ds = xr.open_dataset(path, decode_times=xr.coders.CFDatetimeCoder(
            use_cftime=True))
        v = ds[var]
        units = v.attrs.get("units")
        if units != _UNITS[var]:
            raise LookupError(f"units {units!r}, expected {_UNITS[var]!r}")
        if "lat" not in v.dims or "lon" not in v.dims:
            raise LookupError(f"not a lat/lon grid (dims {v.dims})")
        ym = list(zip(v["time"].dt.year.values.tolist(),
                      v["time"].dt.month.values.tolist()))
        pick = [i for i, x in enumerate(ym) if x in want]
        if pick:
            seen += [ym[i] for i in pick]
            slices.append(np.asarray(v.isel(time=pick).transpose(
                "time", "lat", "lon"), dtype=np.float64))
            la = np.asarray(ds["lat"], dtype=np.float64)
            lo = np.asarray(ds["lon"], dtype=np.float64) % 360.0
            if lat is not None and not (np.array_equal(la, lat)
                                        and np.array_equal(lo, lon)):
                raise LookupError("lat/lon differ between files")
            lat, lon = la, lo
    if sorted(seen) != sorted(window):
        raise LookupError(f"window months missing or duplicated: got "
                          f"{sorted(seen)}, need {sorted(window)}")
    return np.concatenate(slices).mean(axis=0), lat, lon


def model_field(var, files_by_var, window, mlat, mlon):
    """One CMIP6 model's window-mean field on the legoESM grid.

    Uses the SAME reduction as the reference (``rb.bin_to_model``) and, for the
    albedo, the SAME order as ``bias_maps._ref_clim``: bin rsut and rsdt, then
    form the ratio with ``bias_maps._toa_albedo``.
    """
    def binned(v):
        f, lat, lon = window_mean(files_by_var[v], v, window)
        return rb.bin_to_model(f, lat, lon, mlat, mlon, label=v)
    if var == "albedo":
        return bm._toa_albedo(binned("rsut"), binned("rsdt"))[0]
    return binned(var)


# ------------------------------------------------------------ statistics ----
def area_mean(f, lat, valid):
    """cos-lat mean over ``valid`` (``rb.region_mean`` GLOBAL, renormalised).

    The GLOBAL box never reads longitude, so only its length matters here.
    """
    return rb.region_mean(f, lat, np.arange(f.shape[1]), rb.REGIONS["GLOBAL"],
                          valid=valid)


def area_rms(d, lat, valid):
    return float(np.sqrt(area_mean(d ** 2, lat, valid)))


def ensemble_stats(lego, ref, models):
    """Bias fields and the per-cell z score.

    ``lego`` and ``ref`` are (lat, lon); ``models`` is (n, lat, lon).  The
    returned ``valid`` mask is where ALL inputs are finite; every field is NaN
    outside it.  ``spread`` is the sample (ddof=1) standard deviation across
    models of their biases.  ``z`` is additionally NaN where ``z_ok`` is False:
    spread below ``SPREAD_FLOOR_FRAC`` x its median over the valid cells.
    """
    models = np.asarray(models, dtype=np.float64)
    valid = (np.isfinite(lego) & np.isfinite(ref)
             & np.isfinite(models).all(axis=0))
    nan = np.where(valid, 1.0, np.nan)
    b_lego = (lego - ref) * nan
    b_mod = (models - ref[None]) * nan[None]
    spread = np.std(b_mod, axis=0, ddof=1)
    z_ok = valid & (spread >= SPREAD_FLOOR_FRAC * np.median(spread[valid]))
    return {"valid": valid, "b_lego": b_lego, "b_mod": b_mod,
            "b_mmm": b_mod.mean(axis=0), "spread": spread,
            "z": np.where(z_ok, b_lego, np.nan) / spread, "z_ok": z_ok}


# ------------------------------------------------------------------ main ----
def collect(run, variables, root=CMIP6_AMIP):
    window = run_window(run)
    check_window(window)
    print(f"{run}: window {window_label(window)} {window}")
    models = list_models(root)
    rows, excluded, meta = [], [], {}
    for var in variables:
        for v in (["rsut", "rsdt"] if var == "albedo" else [var]):
            if run_window(run, v) != window:
                raise SystemExit(f"FATAL: legoESM {v} covers "
                                 f"{run_window(run, v)}, not {window}")
        got = bm._model_clim(run, var)
        if got is None:
            raise SystemExit(f"{run}: no model output for {var}")
        lego, mlat, mlon, months = got
        src, label, *_ = bm.SPEC[var]
        ref = bm._ref_clim(var, months, mlat, mlon, src)
        if ref is None:
            raise SystemExit(f"{var}: no {label} reference")
        scale = bm._DISPLAY_SCALE.get(var, 1.0)
        need = ["rsut", "rsdt"] if var == "albedo" else [var]
        names, fields = [], []
        for model, (inst, member) in models.items():
            fb, info = {}, []
            for v in need:
                got_f = model_files(root, inst, model, member, v)
                if got_f is None:
                    break
                fb[v] = got_f[2]
                info.append(f"{v}:{got_f[0]}/{got_f[1]}")
            if len(fb) < len(need):
                excluded.append((var, model, member, f"no Amon {v}"))
                continue
            try:
                f = model_field(var, fb, window, mlat, mlon)
            except LookupError as exc:
                excluded.append((var, model, member, str(exc)))
                continue
            names.append(model)
            fields.append(f * scale)
            meta[(var, model)] = (member, " ".join(info))
        if len(names) < 2:
            raise SystemExit(f"{var}: {len(names)} CMIP6 models -- no spread")
        st = ensemble_stats(np.asarray(lego, float) * scale,
                            np.asarray(ref, float) * scale, np.stack(fields))
        rows.append((var, label, mlat, mlon, names, st,
                     np.asarray(lego, float) * scale,
                     np.asarray(ref, float) * scale))
        print(f"  {var}: N={len(names)}  valid on "
              f"{st['valid'].mean():.0%} of cells")
    return window, rows, excluded, meta


def instrument_check(rows):
    """legoESM bias exactly as bias_maps titles it (legoESM ∩ obs mask)."""
    print("  instrument check (bias_maps formulas, legoESM ∩ obs mask):")
    for var, _label, lat, lon, _n, _st, lego, ref in rows:
        f, r = bm._common_mask(lego, ref)
        d = f - r
        print(f"    {var:<8} global {bm._gm(d, lat, lon):+.3g}  "
              f"RMS(unweighted) {np.sqrt(np.nanmean(d ** 2)):.3g}  "
              f"RMS(area) {area_rms(d, lat, np.isfinite(d)):.3g}")


def write_csv(rows, meta, excluded, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variable", "model", "member", "files", "status",
                    "global_mean_bias", "rms_bias_area_weighted"])
        for var, _l, lat, _lon, names, st, *_ in rows:
            v = st["valid"]
            w.writerow([var, "legoESM", "", "", "included",
                        f"{area_mean(st['b_lego'], lat, v):.6g}",
                        f"{area_rms(st['b_lego'], lat, v):.6g}"])
            for k, name in enumerate(names):
                b = np.nan_to_num(st["b_mod"][k])
                w.writerow([var, name, *meta[(var, name)], "included",
                            f"{area_mean(b, lat, v):.6g}",
                            f"{area_rms(b, lat, v):.6g}"])
        for var, model, member, why in excluded:
            w.writerow([var, model, member, "", f"excluded: {why}", "", ""])
    print(f"  wrote {path}")


def _save(fig, out):
    fig.savefig(out, dpi=130)
    from PIL import Image
    im = Image.open(out).convert("RGB")
    im.thumbnail((1400, 1400 * im.height // im.width))
    im.save(out.with_suffix(".jpg"), quality=70)
    print(f"  wrote {out} (+ .jpg)")


def figure_a(run, window, rows, out_dir):
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.0 * len(rows)),
                             subplot_kw={"projection":
                                         ccrs.Robinson(central_longitude=180)},
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, (var, label, lat, lon, names, st, *_x) in enumerate(rows):
        units, blim = bm.SPEC[var][4], bm.SPEC[var][3]
        v = st["valid"]
        name = "TOA albedo" if var == "albedo" else var
        n = len(names)
        for j, (key, who) in enumerate([("b_lego", "legoESM"),
                                        ("b_mmm", f"CMIP6 MMM (N={n})")]):
            d = st[key]
            m = bm._panel(axes[i, j], lon, lat, st[key],
                          f"{name} {who} - {label}\nmean "
                          f"{area_mean(d, lat, v):+.3g} {units}  RMS "
                          f"{area_rms(d, lat, v):.3g}", "RdBu_r", -blim, blim,
                          units)
        fig.colorbar(m, ax=axes[i, :2], shrink=0.85, pad=0.01)
        z = st["z"]
        zk = st["z_ok"]
        fz = area_mean((np.abs(np.nan_to_num(z)) > 2).astype(float), lat, zk)
        fmask = 1.0 - area_mean(zk.astype(float), lat, v)
        from matplotlib.colors import ListedColormap
        bm._panel(axes[i, 2], lon, lat, np.where(v & ~zk, 1.0, np.nan), "",
                  ListedColormap(["0.65"]), 0.0, 1.0, "-")
        m3 = bm._panel(axes[i, 2], lon, lat, z,
                       f"{name} legoESM bias / CMIP6 spread (N={n})\n"
                       f"|z|>2 on {fz:.0%} of unmasked area; grey = spread "
                       f"< {SPREAD_FLOOR_FRAC:g} x median ({fmask:.1%})",
                       "RdBu_r", -Z_LIM, Z_LIM, "-")
        fig.colorbar(m3, ax=axes[i, 2], shrink=0.85, pad=0.01)
    fig.suptitle(f"PROVISIONAL  {run} vs CMIP6 AMIP r1 members: "
                 f"{window_label(window)}, ESMValTool reference datasets "
                 f"(area-weighted mean and RMS)", fontsize=11)
    out = pathlib.Path(out_dir) / f"cmip6_bias_maps_{run}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, out)
    plt.close(fig)


def rank_of(lego_rms, model_rms):
    """1-based rank of legoESM among N+1 (1 = smallest RMS)."""
    return 1 + int(np.sum(np.asarray(model_rms) < lego_rms))


def figure_b(run, window, rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 4
    nrow = -(-len(rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 6.5 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    ranks = {}
    for ax, (var, label, lat, _lon, names, st, *_x) in zip(axes, rows):
        v = st["valid"]
        rms = [area_rms(b, lat, v) for b in st["b_mod"]]
        lr = area_rms(st["b_lego"], lat, v)
        allv = sorted(zip(rms + [lr], names + ["legoESM"]))
        k = rank_of(lr, rms)
        ranks[var] = (k, len(names) + 1)
        y = np.arange(len(allv))
        ax.barh(y, [a for a, _ in allv],
                color=["C3" if n == "legoESM" else "0.6" for _, n in allv])
        ax.set_yticks(y, [n for _, n in allv], fontsize=6)
        ax.invert_yaxis()
        name = "TOA albedo" if var == "albedo" else var
        ax.set_title(f"{name} vs {label}  [{bm.SPEC[var][4]}]\nlegoESM rank "
                     f"{k} of {len(names) + 1}", fontsize=9)
        ax.set_xlabel("area-weighted RMS bias", fontsize=8)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    fig.suptitle(f"PROVISIONAL  {run} vs CMIP6 AMIP r1 members: global RMS "
                 f"bias, {window_label(window)}", fontsize=11)
    out = pathlib.Path(out_dir) / f"cmip6_rms_rank_{run}.png"
    _save(fig, out)
    plt.close(fig)
    return ranks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    window, rows, excluded, meta = collect(args.run, VARS)
    instrument_check(rows)
    print("  spread percentiles (min, 1, 50) and z summary:")
    for var, _l, lat, _lo, names, st, *_ in rows:
        s = st["spread"][st["valid"]]
        print(f"    {var:<8} N={len(names):2d}  sd {s.min():.3g} "
              f"{np.percentile(s, 1):.3g} {np.median(s):.3g}")
    print("  excluded:")
    for e in excluded:
        print(f"    {e[0]:<8} {e[1]:<22} {e[2]:<10} {e[3]}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, meta, excluded, out / f"cmip6_rms_{args.run}.csv")
    figure_a(args.run, window, rows, out)
    ranks = figure_b(args.run, window, rows, out)
    for var, _l, lat, _lo, _n, st, *_ in rows:
        v, zk = st["valid"], st["z_ok"]
        print(f"    z {var:<8} masked {1 - area_mean(zk.astype(float), lat, v):.1%}"
              f" of valid area; |z|>2 on "
              f"{area_mean((np.abs(np.nan_to_num(st['z'])) > 2).astype(float), lat, zk):.1%}"
              " of unmasked area")
    for var, (k, n) in ranks.items():
        print(f"    rank {var:<8} {k} of {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
