#!/usr/bin/env python
"""Quick-look plots for the legoESM run v2_40yr_gpu_v8 (1 deg, NPZDv2).

Run on Levante:
    python /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/plot_v2_40yr_v8.py

Outputs PNGs to:
    /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/eval_omip_phys_1deg_overview/
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = "/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/results/omip_phys_1deg_jra55_ryf/latlon/180x360"
OUT = "/work/bb1596/b384036/diffESM/legoesm_ggn_newcopy/experiments/eval_omip_phys_1deg_overview"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(RUN, "restart_day*.npz")),
               key=lambda f: int(re.search(r"day(\d+)", f).group(1)))
days = np.array([int(re.search(r"day(\d+)", f).group(1)) for f in files])
print(f"{len(files)} snapshots, day {days[0]} -> {days[-1]} ({days[-1]/365:.1f} yr)")
gaps = np.unique(np.diff(days))
if len(gaps) > 1:
    print("  note: irregular snapshot spacing:", gaps)

last = np.load(files[-1])

# ---------------------------------------------------------------- inventory
print("\n=== Contents of", os.path.basename(files[-1]), "===")
for k in last.files:
    a = last[k]
    if a.dtype.kind in "fiub" and a.size > 1:
        print(f"{k:16s} {str(a.shape):18s} {str(a.dtype):8s} "
              f"min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} mean={np.nanmean(a):.4g}")
    else:
        print(f"{k:16s} {a}")

H = last["H_bathy"]
print(f"\nH_bathy: {len(np.unique(H))} unique values, range {H.min():.1f} - {H.max():.1f} m")
if len(np.unique(H)) < 10:
    print("  -> looks like a binary/flat land mask, NOT real bathymetry")

ny, nx = H.shape[-2:]
lat = np.linspace(-90 + 90 / ny, 90 - 90 / ny, ny)
lon = np.linspace(180 / nx, 360 - 180 / nx, nx)
extent = [0, 360, -90, 90]
w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nx))


def lev_first(a):
    """Return a 3D field as (level, lat, lon). Snapshots store (lat, lon, level)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 3 and a.shape[:2] == (ny, nx):
        a = np.moveaxis(a, -1, 0)
    return a


def ocean_mask_2d(z):
    """True over ocean. Uses land_mask if present (ocean = majority value), else H_bathy>0."""
    if "land_mask" in z.files:
        lm = lev_first(z["land_mask"])
        lm2 = lm[0] if lm.ndim == 3 else lm
        vals, counts = np.unique(lm2, return_counts=True)
        ocean_val = vals[np.argmax(counts)]  # ocean covers ~70% of the globe
        return lm2 == ocean_val
    return np.asarray(z["H_bathy"]) > 0


ocean = ocean_mask_2d(last)
print(f"ocean fraction of grid cells: land_mask-based {ocean.mean():.2f}, "
      f"H_bathy>0 {(np.asarray(H) > 0).mean():.2f}  (real 1-deg grid is ~0.67)")
print("land_mask values/counts:", np.unique(lev_first(last["land_mask"]), return_counts=True))


def surface(a):
    """Level 0 of a 3D field (assumed surface; check the T profile printed below)."""
    a = lev_first(a)
    return a[0] if a.ndim == 3 else a


def gmean(a2d, mask):
    m = mask & np.isfinite(a2d)
    return np.sum(a2d[m] * w2d[m]) / np.sum(w2d[m])


# Check vertical orientation using T
T = lev_first(last["T"])
if T.ndim == 3:
    prof = [gmean(T[k], ocean) for k in range(T.shape[0])]
    print("\nGlobal-mean T by level index (should decrease with depth if level 0 = surface):")
    print("  ", np.round(prof, 2))

# ---------------------------------------------------------- surface maps
fields = ["T", "S", "eta", "biogeo_NO3", "biogeo_PO4", "biogeo_Si", "biogeo_Fe",
          "biogeo_Chl_d", "biogeo_Chl_n", "biogeo_Pd", "biogeo_Pn", "biogeo_Zoo",
          "biogeo_DIC", "biogeo_ALK", "biogeo_POC", "biogeo_Det"]
fields = [f for f in fields if f in last.files]

ncol = 4
nrow = int(np.ceil(len(fields) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.8 * nrow), constrained_layout=True)
for ax, f in zip(axes.flat, fields):
    a = surface(last[f])
    if a.shape != ocean.shape:
        ax.set_title(f"{f}: shape {a.shape} (skipped)")
        ax.axis("off")
        continue
    a = np.where(ocean, a, np.nan)
    lo, hi = np.nanpercentile(a, [2, 98])
    im = ax.imshow(a, origin="lower", extent=extent, aspect="auto",
                   cmap="RdBu_r" if f == "eta" else "viridis", vmin=lo, vmax=hi)
    ax.set_title(f"{f} (surface)")
    plt.colorbar(im, ax=ax, shrink=0.8)
for ax in list(axes.flat)[len(fields):]:
    ax.axis("off")
fig.suptitle(f"omip_phys_1deg surface fields, day {days[-1]} (~year {days[-1]/365:.1f})")
fig.savefig(os.path.join(OUT, "surface_maps_last.png"), dpi=110)
plt.close(fig)

# ----------------------------------------------------- surface currents
for k in ["u", "v"]:
    if k in last.files:
        print(f"{k} shape: {last[k].shape}")
try:
    def sfc_stag(a):  # u/v may be staggered, e.g. (ny, nx+1, nz)
        a = np.asarray(a, dtype=float)
        if a.ndim == 3 and a.shape[-1] == T.shape[0]:
            a = a[..., 0]
        elif a.ndim == 3:
            a = a[0]
        return a[:ny, :nx]
    u = sfc_stag(last["u"])
    v = sfc_stag(last["v"])
    spd = np.where(ocean, np.hypot(u, v), np.nan)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(spd, origin="lower", extent=extent, aspect="auto", cmap="magma",
                   vmax=np.nanpercentile(spd, 99))
    plt.colorbar(im, ax=ax, label="m/s")
    ax.set_title(f"Surface speed, day {days[-1]}")
    fig.savefig(os.path.join(OUT, "surface_speed_last.png"), dpi=110)
    plt.close(fig)
except Exception as e:
    print("speed plot skipped:", e)

# ------------------------------------------------- zonal-mean sections
sec = [f for f in ["T", "S", "biogeo_NO3", "biogeo_PO4", "biogeo_DIC", "biogeo_Fe"] if f in last.files]
fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
for ax, f in zip(axes.flat, sec):
    a = lev_first(last[f])
    if a.ndim != 3:
        ax.axis("off")
        continue
    nz = a.shape[0]
    a = np.where(ocean[None], a, np.nan)
    a = np.where(a == 0, np.nan, a)  # cells below the sea floor are often exactly 0
    zm = np.nanmean(a, axis=2)
    lo, hi = np.nanpercentile(zm, [2, 98])
    im = ax.pcolormesh(lat, np.arange(nz), zm, shading="auto", vmin=lo, vmax=hi)
    ax.invert_yaxis()
    ax.set_xlabel("lat")
    ax.set_ylabel("level index (0 = top)")
    ax.set_title(f"{f} zonal mean")
    plt.colorbar(im, ax=ax)
for ax in list(axes.flat)[len(sec):]:
    ax.axis("off")
fig.savefig(os.path.join(OUT, "zonal_sections_last.png"), dpi=110)
plt.close(fig)

# ---------------------------------------- drift: one snapshot per year
ts_fields = [f for f in ["T", "S", "biogeo_NO3", "biogeo_Chl_d", "biogeo_Chl_n", "biogeo_DIC"] if f in last.files]
sel = list(range(0, len(files), 12)) + [len(files) - 1]
sel = sorted(set(sel))
rec = {f: {"sfc": [], "vol": []} for f in ts_fields}
yrs = []
for i in sel:
    z = np.load(files[i])
    yrs.append(days[i] / 365)
    for f in ts_fields:
        a = lev_first(z[f])
        rec[f]["sfc"].append(gmean(surface(a), ocean))
        if a.ndim == 3:
            b = np.where(ocean[None] & (a != 0), a, np.nan)
            rec[f]["vol"].append(np.nanmean(b))
        else:
            rec[f]["vol"].append(np.nan)
    print(f"  drift: year {yrs[-1]:5.1f} done")

fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
for ax, f in zip(axes.flat, ts_fields):
    ax.plot(yrs, rec[f]["sfc"], "o-", label="surface (area-wtd)")
    ax2 = ax.twinx()
    ax2.plot(yrs, rec[f]["vol"], "s--", color="C1", label="3D mean")
    ax.set_title(f)
    ax.set_xlabel("model year")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
for ax in list(axes.flat)[len(ts_fields):]:
    ax.axis("off")
fig.suptitle("Drift (one snapshot per year)")
fig.savefig(os.path.join(OUT, "drift_yearly.png"), dpi=110)
plt.close(fig)

# --------------------------------------------- seasonal cycle, last year
last_year = [i for i in range(len(files)) if days[i] > days[-1] - 365]
sst = [gmean(surface(np.load(files[i])["T"]), ocean) for i in last_year]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(days[last_year] - days[last_year][0], sst, "o-")
ax.set_xlabel("day within last year")
ax.set_ylabel("global-mean SST")
ax.set_title("Seasonal cycle check (flat = no seasonal forcing)")
fig.savefig(os.path.join(OUT, "sst_last_year.png"), dpi=110)
plt.close(fig)

# --------------------------------------------- run's own timeseries.csv
try:
    import pandas as pd
    ts = pd.read_csv(os.path.join(RUN, "timeseries.csv"))
    print("\ntimeseries.csv columns:", list(ts.columns))
    cols = [c for c in ts.columns if ts[c].dtype.kind in "fi"][1:9]
    xcol = ts.columns[0]
    fig, axes = plt.subplots(int(np.ceil(len(cols) / 2)), 2, figsize=(12, 2.5 * np.ceil(len(cols) / 2)),
                             constrained_layout=True, squeeze=False)
    for ax, c in zip(axes.flat, cols):
        ax.plot(ts[xcol], ts[c])
        ax.set_title(c)
    fig.savefig(os.path.join(OUT, "timeseries_csv.png"), dpi=110)
    plt.close(fig)
except Exception as e:
    print("timeseries.csv plot skipped:", e)

print("\nPlots written to", OUT)
for p in sorted(glob.glob(os.path.join(OUT, "*.png"))):
    print("  ", p)
