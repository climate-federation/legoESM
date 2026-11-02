"""Evaluate the 3-D legoESM + MARBL run against observations, per basin.

Model input (one run directory, <output>/latlon/180x360):
  marbl_diag_dayNNNNNN.npz   2-D means over each --marbl-diag-days window (NPP, export,
                             air-sea CO2 flux, pCO2, pH, surface Chl)
  restart_dayNNNNNN.npz      snapshots -> surface NO3, PO4, SiO3, O2, DIC, ALK
Observations (monthly climatologies, found automatically under the ESMValTool OBS tree
or given explicitly):
  Chl     ESA-CCI ocean colour (chl)              --obs-chl
  fgco2   Landschuetzer SOM-FFN air-sea CO2 flux  --obs-fgco2
  spco2   Landschuetzer SOM-FFN surface pCO2      --obs-spco2
  no3/po4/si/o2  WOA monthly climatology          --obs-no3 --obs-po4 --obs-si --obs-o2

Outputs (--out):
  eval_maps.{png,pdf}                 annual mean: model | obs | model - obs
  eval_basin_timeseries.{png,pdf}     monthly NPP, export, CO2 flux, Chl per basin
  eval_seasonal_<var>.{png,pdf}       seasonal cycle per basin, model vs obs
  eval_stats.csv                      per basin: model, obs, bias, RMSE, pattern r

Basins: N/S Atlantic, N/S Pacific, Indian, Southern Ocean (< 35 S), Arctic (> 66 N),
built by flood fill on the model land mask.

Usage (repo root):
  python scripts/evaluate_marbl_3d.py --list-obs
  python scripts/evaluate_marbl_3d.py \
      --run-dir results/marbl_1deg_1yr_gpu/latlon/180x360 \
      --grid /work/bb1596/b384036/diffESM/data/marbl_grid_1deg.npz \
      --out results/paper_figs/eval_1yr
"""
import argparse
import csv
import glob
import sys
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_marbl_global_maps import cell_area, load_grid, to_physical   # noqa: E402

OBS_ROOT = "/work/bd0854/DATA/ESMValTool2/OBS"
SEC_PER_YEAR = 365.0 * 86400.0
OBS_PATTERNS = {   # variable -> (glob patterns under OBS_ROOT, netCDF variable name)
    "chl":   (["Tier*/ESACCI-OC/*Omon_chl_*.nc", "Tier*/ESACCI-OC/*chl*.nc"], "chl"),
    "fgco2": (["Tier*/Landschuetzer*/*Omon_fgco2_*.nc", "Tier*/Landschuetzer*/*fgco2*.nc"], "fgco2"),
    "spco2": (["Tier*/Landschuetzer*/*Omon_spco2_*.nc", "Tier*/Landschuetzer*/*spco2*.nc"], "spco2"),
    "no3":   (["Tier*/WOA/*2018*Omon_no3_*.nc", "Tier*/WOA/*Omon_no3_*.nc"], "no3"),
    "po4":   (["Tier*/WOA/*2018*Omon_po4_*.nc", "Tier*/WOA/*Omon_po4_*.nc"], "po4"),
    "si":    (["Tier*/WOA/*2018*Omon_si_*.nc", "Tier*/WOA/*Omon_si_*.nc"], "si"),
    "o2":    (["Tier*/WOA/*2018*Omon_o2_*.nc", "Tier*/WOA/*Omon_o2_*.nc"], "o2"),
}
REGIONS = ["N Atlantic", "S Atlantic", "N Pacific", "S Pacific", "Indian", "Southern Ocean", "Arctic"]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRIDC, LAND = "#1f1f1f", "#4a4a4a", "#8a8a85", "#e6e4df", "#d9d6cf"
MONTHS = "JFMAMJJASOND"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ basins
def _flood(allowed, seed):
    nlat, nlon = allowed.shape
    out = np.zeros_like(allowed)
    if not allowed[seed]:
        return out
    q = deque([seed]); out[seed] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, (j + dj) % nlon
            if 0 <= ii < nlat and allowed[ii, jj] and not out[ii, jj]:
                out[ii, jj] = True; q.append((ii, jj))
    return out


def basin_masks(latd, lond, ocean):
    """Basins by geography, no flood fill (a 1-degree land mask can leave narrow passages open,
    which made the flood fill leak from the Atlantic into the Pacific and Indian).
    Southern Ocean < 35 S, Arctic >= 66 N; in between, the Atlantic lies east of a dividing
    longitude that follows the coast of the Americas (Caribbean and Gulf of Mexico included) and
    west of 20 E (45 E north of 30 N: Mediterranean, Black Sea); the Indian lies between 20 E and
    147 E south of 31 N, except the Indonesian seas and South China Sea (east of 100 E, north of
    9 S), which belong to the Pacific; the rest is Pacific."""
    LA, LO = np.meshgrid(latd, np.mod(lond, 360.0), indexing="ij")
    so, arc = ocean & (LA < -35), ocean & (LA >= 66)
    mid = ocean & ~so & ~arc
    # Pacific | Atlantic dividing longitude [deg E] along the Americas, by latitude band
    bands = [(-35, 8, 290.0), (8, 9, 281.0), (9, 11, 276.0), (11, 14, 274.0), (14, 17, 271.0),
             (17, 19, 265.0), (19, 31, 262.0), (31, 66, 260.0)]
    div = np.full(LA.shape, 290.0)
    for lo, hi, d in bands:
        div = np.where((LA >= lo) & (LA < hi), d, div)
    east_lim = np.where(LA >= 30, 45.0, 20.0)
    atl = mid & ((LO >= div) | (LO < east_lim))
    ind = mid & ~atl & (LO >= 20) & (LO < 147) & (LA < 31) & ~((LO >= 100) & (LA > -9))
    pac = mid & ~atl & ~ind
    m = {"N Atlantic": atl & (LA >= 0), "S Atlantic": atl & (LA < 0),
         "N Pacific": pac & (LA >= 0), "S Pacific": pac & (LA < 0),
         "Indian": ind, "Southern Ocean": so, "Arctic": arc}
    m["Global"] = ocean
    return m


# ------------------------------------------------------------------ model
def _find(npz, name):
    for k in npz.files:
        kk = k.split("/")[-1].split(".")[-1]
        for pre in ("", "biogeo_", "bgc_", "biogeo.", "marbl_"):
            if kk == pre + name or k == pre + name:
                return k
    return None


def month_of(day_mid):
    return int((day_mid % 365.0) / 365.0 * 12) % 12


def load_model(run_dir, ocean, skip_days):
    files = sorted(glob.glob(str(Path(run_dir) / "marbl_diag_day*.npz")))
    if not files:
        raise SystemExit(f"no marbl_diag_day*.npz in {run_dir}")
    t, rows = [], []
    for f in files:
        d = np.load(f)
        mid = 0.5 * (float(d["day_start"]) + float(d["day_end"]))
        if mid < skip_days:
            continue
        p = to_physical(d)
        t.append(mid)
        rows.append({k: np.where(ocean, v, np.nan) for k, v in p.items()})
    sfc = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "restart_day*.npz"))):
        day = float(Path(f).stem.replace("restart_day", ""))
        if day - 15 < skip_days:
            continue
        r = np.load(f)
        out = {}
        for name, key in (("NO3", "NO3"), ("PO4", "PO4"), ("SiO3", "SiO3"), ("O2", "O2"), ("DIC", "DIC"), ("ALK", "ALK")):
            k = _find(r, key)
            if k is not None and np.asarray(r[k]).ndim == 3:
                out[name] = np.where(ocean, np.asarray(r[k])[..., 0], np.nan)
        if out:
            sfc[day - 15.0] = out                      # snapshot at the window end -> window centre
    print(f"model: {len(rows)} diagnostic windows (days {t[0]:.0f}..{t[-1]:.0f}), "
          f"{len(sfc)} restart snapshots with surface tracers {sorted(next(iter(sfc.values())).keys()) if sfc else []}")
    return np.array(t), rows, sfc


def monthly_climatology(t, fields, key):
    """12 monthly maps from windows (mean of windows falling in each month; NaN if none)."""
    acc = [[] for _ in range(12)]
    for ti, f in zip(t, fields):
        if key in f:
            acc[month_of(ti)].append(f[key])
    shape = next(f[key] for f in fields if key in f).shape
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.array([np.nanmean(a, axis=0) if a else np.full(shape, np.nan) for a in acc])


# ------------------------------------------------------------------ obs
def find_obs(var, explicit=None):
    if explicit:
        return sorted(glob.glob(explicit))
    pats, _ = OBS_PATTERNS[var]
    for p in pats:
        hits = sorted(glob.glob(f"{OBS_ROOT}/{p}"))
        if hits:
            return hits
    return []


def _to_model_units(var, a, units):
    u = (units or "").replace(" ", "").lower()
    if var == "chl":
        return a * 1e6 if ("kg" in u or np.nanmedian(a) < 1e-3) else a           # kg m-3 -> mg m-3
    if var == "fgco2":                                                            # kg C m-2 s-1 -> mol C m-2 yr-1
        return a / 12e-3 * SEC_PER_YEAR if ("kg" in u or np.nanmax(np.abs(a)) < 1e-5) else a
    if var == "spco2":                                                            # Pa -> uatm
        return a / 101325.0 * 1e6 if (u in ("pa",) or np.nanmedian(a) > 1000) else a
    if var in ("no3", "po4", "si", "o2"):                                         # mol m-3 -> mmol m-3
        if "mol" in u and "mmol" not in u and "umol" not in u and "µmol" not in u:
            return a * 1e3
        if "umol/kg" in u or "µmol/kg" in u or "umolkg-1" in u:
            return a * 1.025
        return a * 1e3 if np.nanmedian(a) < 0.1 else a
    return a


def load_obs(var, files, latd, lond):
    """Monthly climatology (12, nlat, nlon) of the surface field on the model grid."""
    import xarray as xr
    _, vname = OBS_PATTERNS[var]
    ds = xr.open_mfdataset(files, combine="by_coords", decode_times=True, chunks={"time": 12}) \
        if len(files) > 1 else xr.open_dataset(files[0], chunks={"time": 12})
    da = ds[vname]
    units = da.attrs.get("units", "")
    for d in list(da.dims):
        if d.lower() in ("lev", "depth", "olevel", "z", "plev") or d.lower().startswith("depth"):
            da = da.isel({d: 0})
    tdim = next(d for d in da.dims if d.lower().startswith("time"))
    if da.sizes[tdim] > 12:
        da = da.groupby(f"{tdim}.month").mean(tdim)
        mdim = "month"
    else:
        mdim = tdim
    latn = next(d for d in da.dims if d.lower().startswith("lat"))
    lonn = next(d for d in da.dims if d.lower().startswith("lon"))
    da = da.sortby(latn)
    da = da.assign_coords({lonn: np.mod(da[lonn].values, 360.0)}).sortby(lonn)
    dsrc = float(np.abs(np.diff(da[latn].values[:2]))[0])
    dmod = float(np.abs(np.diff(latd[:2]))[0])
    f = int(round(dmod / dsrc))
    if f >= 2:                                               # finer than the model grid: block mean first
        da = da.coarsen({latn: f, lonn: f}, boundary="trim").mean()
    first, last = da.isel({lonn: [0]}), da.isel({lonn: [-1]})
    da = xr.concat([last.assign_coords({lonn: last[lonn].values - 360.0}), da,
                    first.assign_coords({lonn: first[lonn].values + 360.0})], dim=lonn)
    lon_q = np.mod(lond, 360.0)
    order = np.argsort(lon_q)
    out = da.interp({latn: latd, lonn: lon_q[order]}, method="linear").transpose(mdim, latn, lonn).values
    out = out[:, :, np.argsort(order)]
    out = _to_model_units(var, np.asarray(out, float), units)
    if out.shape[0] == 1:                                    # annual climatology (e.g. WOA Oyr): same value every month
        out = np.repeat(out, 12, axis=0)
    print(f"obs {var:5s}: {len(files)} file(s) e.g. {Path(files[0]).name} [{units}] -> "
          f"median {np.nanmedian(out):.4g}")
    return out


# ------------------------------------------------------------------ stats
def wmean(x, w):
    ok = np.isfinite(x) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan


def stats(mod, obs, w):
    ok = np.isfinite(mod) & np.isfinite(obs) & (w > 0)
    if ok.sum() < 10:
        return dict(model=np.nan, obs=np.nan, bias=np.nan, rmse=np.nan, r=np.nan, n=int(ok.sum()))
    ww = w[ok] / w[ok].sum()
    m, o = mod[ok], obs[ok]
    mm, mo = np.sum(ww * m), np.sum(ww * o)
    cov = np.sum(ww * (m - mm) * (o - mo))
    r = cov / np.sqrt(np.sum(ww * (m - mm) ** 2) * np.sum(ww * (o - mo) ** 2))
    return dict(model=mm, obs=mo, bias=mm - mo, rmse=float(np.sqrt(np.sum(ww * (m - o) ** 2))), r=float(r),
                n=int(ok.sum()))


# ------------------------------------------------------------------ figures
def _edges(latd, lond):
    dlo, dla = np.median(np.diff(lond)), np.median(np.diff(latd))
    return (np.concatenate([lond - dlo / 2, [lond[-1] + dlo / 2]]),
            np.clip(np.concatenate([latd - dla / 2, [latd[-1] + dla / 2]]), -90, 90))


def fig_maps(pairs, latd, lond, out, title):
    lon_e, lat_e = _edges(latd, lond)
    n = len(pairs)
    fig, axs = plt.subplots(n, 3, figsize=(7.4, 1.75 * n + 0.5), constrained_layout=True, squeeze=False)
    for r, (label, unit, mod, obs, log) in enumerate(pairs):
        both = np.concatenate([mod[np.isfinite(mod)], obs[np.isfinite(obs)]])
        if log:
            norm = LogNorm(max(np.nanpercentile(both, 2), 1e-3), np.nanpercentile(both, 99))
            cmap = "Greens"
        else:
            lo, hi = np.nanpercentile(both, 2), np.nanpercentile(both, 98)
            if lo < 0 < hi:
                v = max(-lo, hi); norm, cmap = TwoSlopeNorm(0.0, -v, v), "RdBu_r"
            else:
                norm, cmap = plt.Normalize(lo, hi), "viridis"
        diff = (np.log10(mod) - np.log10(obs)) if log else (mod - obs)
        dl = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98) if np.isfinite(diff).any() else 1.0
        for c, (fld, ttl, nm, cm) in enumerate([
                (mod, f"{label}: model", norm, cmap), (obs, f"{label}: observed", norm, cmap),
                (diff, "log10(model/obs)" if log else "model − obs", TwoSlopeNorm(0.0, -dl, dl), "RdBu_r")]):
            ax = axs[r, c]
            cmo = plt.get_cmap(cm).copy(); cmo.set_bad(LAND)
            pc = ax.pcolormesh(lon_e, lat_e, fld, cmap=cmo, norm=nm, shading="flat", rasterized=True)
            ax.set_title(ttl, loc="left", fontsize=7.5)
            ax.set_xlim(lon_e[0], lon_e[-1]); ax.set_ylim(-80, 90)
            ax.set_xticks([0, 120, 240, 360]); ax.set_yticks([-60, 0, 60]); ax.tick_params(labelsize=6, length=2)
            cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01)
            cb.ax.tick_params(labelsize=6)
            if c == 1:
                cb.set_label(unit, fontsize=6.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"eval_maps.{ext}")
    plt.close(fig)


def fig_timeseries(t, series, out, title):
    keys = [("NPP", "NPP [Pg C yr$^{-1}$]"), ("POC", "Export at 100 m [Pg C yr$^{-1}$]"),
            ("fCO2", "Air-sea CO$_2$ flux [Pg C yr$^{-1}$, + into ocean]"), ("Chl", "Surface Chl [mg m$^{-3}$]")]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.0), constrained_layout=True, sharex=True)
    for ax, (k, lab) in zip(axs.flat, keys):
        for i, reg in enumerate(REGIONS):
            ax.plot(t, series[reg][k], color=C[i], lw=1.5, marker="o", ms=2.5, label=reg)
        if k == "fCO2":
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(lab, loc="left"); ax.grid(color=GRIDC, lw=0.6)
    for ax in axs[1]:
        ax.set_xlabel("model day")
    axs[0, 0].legend(fontsize=6.5, ncol=2, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"eval_basin_timeseries.{ext}")
    plt.close(fig)


def fig_seasonal(var, label, unit, mod12, obs12, masks, area, out, title):
    regs = REGIONS + ["Global"]
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 3.8), constrained_layout=True, sharex=True)
    mx = np.arange(12)
    for ax, reg in zip(axs.flat, regs):
        w = area * masks[reg]
        m = [wmean(mod12[i], w) for i in range(12)]
        ax.plot(mx, m, color=C[0], lw=1.6, marker="o", ms=2.5, label="model")
        if obs12 is not None:
            o = [wmean(obs12[i], w) for i in range(12)]
            ax.plot(mx, o, color=INK, lw=1.2, ls="--", label="observed")
        if var == "fgco2":
            ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_title(reg, loc="left"); ax.grid(color=GRIDC, lw=0.6)
        ax.set_xticks(mx, list(MONTHS), fontsize=6.5)
    axs[0, 0].legend(fontsize=6.5, frameon=False)
    for ax in axs[:, 0]:
        ax.set_ylabel(unit)
    fig.suptitle(f"{title}: seasonal cycle of {label}", x=0.01, ha="left", fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"eval_seasonal_{var}.{ext}")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--grid")
    ap.add_argument("--out", default="results/paper_figs/eval")
    ap.add_argument("--skip-days", type=float, default=0.0, help="ignore model windows centred before this day")
    ap.add_argument("--label", default="legoESM + MARBL, 1°, JRA55-do RYF")
    ap.add_argument("--list-obs", action="store_true", help="only print the observation files found")
    for v in OBS_PATTERNS:
        ap.add_argument(f"--obs-{v}", default=None, help=f"glob of {v} observation file(s)")
    a = ap.parse_args()

    obs_files = {v: find_obs(v, getattr(a, f"obs_{v}")) for v in OBS_PATTERNS}
    for v, fs in obs_files.items():
        print(f"  {v:5s}: {len(fs)} file(s)  {fs[0] if fs else '-- not found (pass --obs-' + v + ')'}")
    if a.list_obs:
        return
    if not (a.run_dir and a.grid):
        raise SystemExit("--run-dir and --grid are required")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    lat, lon, ocean = load_grid(a.grid)
    latd, lond = np.rad2deg(lat), np.rad2deg(lon)
    area = cell_area(lat, lon)
    masks = basin_masks(latd, lond, ocean)
    for k, m in masks.items():
        print(f"  basin {k:15s}: {int(m.sum()):6d} cells, {float(np.sum(area * m)) / 1e12:6.1f} million km2")

    t, fields, sfc = load_model(a.run_dir, ocean, a.skip_days)
    days_label = f"model days {t[0] - 15:.0f}–{t[-1] + 15:.0f}"
    title = f"{a.label} ({days_label}, not spun up)"

    # Per-basin monthly time series (integrals for fluxes, area means for Chl)
    series = {reg: {} for reg in REGIONS + ["Global"]}
    for reg, msk in masks.items():
        w = area * msk
        series[reg]["NPP"] = [np.nansum(f["NPP"] * w) * 1e-15 for f in fields]
        series[reg]["POC"] = [np.nansum(f["POC"] * w) * 1e-15 for f in fields]
        series[reg]["fCO2"] = [np.nansum(f["fCO2"] * w) * 12.0 * 1e-15 for f in fields]
        series[reg]["Chl"] = [wmean(f["Chl"], w) for f in fields]
    fig_timeseries(t, series, out, title)

    # Model monthly climatologies (from the windows)
    mod12 = {"chl": monthly_climatology(t, fields, "Chl"), "fgco2": monthly_climatology(t, fields, "fCO2"),
             "spco2": monthly_climatology(t, fields, "pCO2")}
    if sfc:
        ts, sf = np.array(sorted(sfc)), [sfc[k] for k in sorted(sfc)]
        for v, key in (("no3", "NO3"), ("po4", "PO4"), ("si", "SiO3"), ("o2", "O2")):
            if key in sf[0]:
                mod12[v] = monthly_climatology(ts, sf, key)
    months_ok = np.array([np.isfinite(mod12["chl"][i]).any() for i in range(12)])

    meta = {"chl": ("surface chlorophyll", "mg m$^{-3}$", True),
            "fgco2": ("air-sea CO$_2$ flux (+ into ocean)", "mol C m$^{-2}$ yr$^{-1}$", False),
            "spco2": ("surface pCO$_2$", "µatm", False),
            "no3": ("surface NO$_3$", "mmol m$^{-3}$", False), "po4": ("surface PO$_4$", "mmol m$^{-3}$", False),
            "si": ("surface SiO$_3$", "mmol m$^{-3}$", False), "o2": ("surface O$_2$", "mmol m$^{-3}$", False)}
    obs12, pairs, rows = {}, [], []
    for v in ("chl", "fgco2", "spco2", "no3", "po4", "si", "o2"):
        if v not in mod12:
            continue
        o = load_obs(v, obs_files[v], latd, lond) if obs_files[v] else None
        if o is not None:
            o = np.where(ocean[None], o, np.nan)
            obs12[v] = o
        label, unit, log = meta[v]
        # annual means over the months the model covers (same months for the obs)
        mod_ann = np.nanmean(mod12[v][months_ok], axis=0)
        if o is not None:
            obs_ann = np.nanmean(o[months_ok], axis=0)
            pairs.append((label, unit, mod_ann, obs_ann, log))
            for reg, msk in masks.items():
                s = stats(np.log10(mod_ann) if log else mod_ann, np.log10(obs_ann) if log else obs_ann, area * msk)
                rows.append(dict(variable=v + (" (log10)" if log else ""), region=reg, **s))
        fig_seasonal(v, label, unit.replace("$^{-3}$", "⁻³"), mod12[v], o, masks, area, out, a.label)

    if pairs:
        fig_maps(pairs, latd, lond, out, f"{a.label}: annual mean vs observations ({days_label})")
    with open(out / "eval_stats.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["variable", "region", "model", "obs", "bias", "rmse", "r", "n"])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n{'variable':18s} {'region':15s} {'model':>9s} {'obs':>9s} {'bias':>9s} {'rmse':>9s} {'r':>6s}")
    for r in rows:
        print(f"{r['variable']:18s} {r['region']:15s} {r['model']:9.3g} {r['obs']:9.3g} {r['bias']:9.3g} "
              f"{r['rmse']:9.3g} {r['r']:6.2f}")
    print(f"\nfigures and eval_stats.csv in {out}")


if __name__ == "__main__":
    main()
