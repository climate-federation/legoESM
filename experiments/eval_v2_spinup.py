"""
Evaluate NPZDv2 5-year spinup from restart file.

Loads results/v2_5yr_gpu/latlon/180x360/restart_day007300.npz
and produces global maps + zonal sections for all 17 BGC tracers.

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python experiments/eval_v2_spinup.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star

# ── Config ────────────────────────────────────────────────────────────────────
RESTART  = "results/v2_40yr_nudged/latlon/180x360/restart_day007300.npz"
OUT_DIR  = "experiments/eval_v2_40yr_nudged"
N_LAT, N_LON = 180, 360
N_LEV    = 35
H_MAX    = 5500.0

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load restart ──────────────────────────────────────────────────────────────
print(f"Loading {RESTART}...")
d = np.load(RESTART, allow_pickle=True)
print(f"  Keys: {sorted(d.keys())}")

land_mask = d["land_mask"]
T         = d["T"]
S         = d["S"] if "S" in d else np.full((N_LAT, N_LON, N_LEV), 35.0)

def _bgc(key, scale=1e3):
    return d[f"biogeo_{key}"] * scale

NO3   = _bgc("NO3")
NH4   = _bgc("NH4")
PO4   = _bgc("PO4")
Si    = _bgc("Si")
Fe    = _bgc("Fe",    scale=1e6)   # mol/m³ → μmol/m³
Pd    = _bgc("Pd")
Pn    = _bgc("Pn")
Chl_d = _bgc("Chl_d", scale=1e6)  # kg/m³ → mg/m³
Chl_n = _bgc("Chl_n", scale=1e6)
Zoo   = _bgc("Zoo")
Det   = _bgc("Det")
DON   = _bgc("DON")
DOP   = _bgc("DOP")
POC   = _bgc("POC")
DOC   = _bgc("DOC")
DIC   = _bgc("DIC")
ALK   = _bgc("ALK")

Chl_tot = Chl_d + Chl_n
Phyto   = Pd + Pn

print(f"  NO3  surface mean: {np.nanmean(NO3[...,0]):.2f} mmol/m³")
print(f"  Chl  surface mean: {np.nanmean(Chl_tot[...,0]):.4f} mg/m³")
print(f"  PO4  surface mean: {np.nanmean(PO4[...,0]):.3f} mmol/m³")
print(f"  Si   surface mean: {np.nanmean(Si[...,0]):.2f} mmol/m³")
print(f"  DIC  surface mean: {np.nanmean(DIC[...,0]):.1f} mmol/m³")
print(f"  T    surface mean: {np.nanmean(T[...,0]):.1f} °C")

# ── Grid ──────────────────────────────────────────────────────────────────────
grid    = create_latlon_grid(N_LAT, N_LON)
z_coord = create_ocean_z_star(n_levels=N_LEV, H_max=H_MAX)
z_centers = np.abs(np.array(z_coord.z_full_ref))
lat_deg   = np.degrees(np.asarray(grid.lat))

# ── Land masking ──────────────────────────────────────────────────────────────
ocean_3d = land_mask[:, :, np.newaxis] * np.ones((1, 1, N_LEV))

def _mask(arr):
    out = arr.copy().astype(float)
    out[ocean_3d < 0.5] = np.nan
    return out

# ── Depth indices ─────────────────────────────────────────────────────────────
target_depths = [5, 25, 50, 100, 200, 500, 1000, 2000]
depth_idx  = [int(np.argmin(np.abs(z_centers - d))) for d in target_depths]
depth_labs = [f"{int(z_centers[k])} m" for k in depth_idx]

# ── Plot helpers ──────────────────────────────────────────────────────────────
proj = ccrs.PlateCarree()

def plot_tracer(data3d, title, cmap, unit, fname,
                diverging=False, vmin_fixed=None, vmax_fixed=None):
    data3d = _mask(data3d)
    data3d = np.roll(data3d, N_LON // 2, axis=1)  # centre on 0°

    fig, axes = plt.subplots(2, 4,
        figsize=(22, 8),
        subplot_kw=dict(projection=proj),
        gridspec_kw=dict(hspace=0.45, wspace=0.08),
    )
    for panel_i, (k, ax) in enumerate(zip(depth_idx, axes.flatten())):
        data_k = data3d[:, :, k]
        valid  = data_k[np.isfinite(data_k)]
        if valid.size == 0:
            ax.set_visible(False)
            continue
        if vmin_fixed is not None:
            vmin, vmax = vmin_fixed, vmax_fixed
        elif diverging:
            vabs = max(abs(np.nanpercentile(data_k, 2)),
                       abs(np.nanpercentile(data_k, 98)), 1e-6)
            vmin, vmax = -vabs, vabs
        else:
            vmin = 0.0
            vmax = max(np.nanpercentile(data_k, 90), 1e-6)

        im = ax.imshow(data_k, origin="lower",
                       extent=[-180, 180, -90, 90],
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=proj, interpolation="nearest")
        ax.add_feature(cfeature.LAND,      facecolor="#c8c8c8", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4,        zorder=3)
        ax.set_extent([-180, 180, -90, 90], crs=proj)
        ax.gridlines(linewidth=0.3, color="grey", alpha=0.5,
                     xlocs=range(-180, 181, 60),
                     ylocs=range(-60, 61, 30))
        ax.set_title(depth_labs[panel_i], fontsize=10,
                     fontweight="bold", pad=3)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.05, shrink=0.9, aspect=25)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(unit, fontsize=7)

    fig.suptitle(
        f"{title}\nlegoESM NPZDv2 — year-5 (180×360, 35 lev)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_zonal(data3d, title, cmap, unit, fname, diverging=False):
    data3d = _mask(data3d)
    zonal  = np.nanmean(data3d, axis=1)   # (nlat, nlev)
    p1  = np.nanpercentile(zonal, 2)
    p99 = np.nanpercentile(zonal, 98)
    if diverging:
        vabs = max(abs(p1), abs(p99), 1e-6)
        vmin, vmax = -vabs, vabs
    else:
        vmin, vmax = 0.0, max(p99, 1e-6)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.contourf(lat_deg, z_centers, zonal.T,
                     levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(z_centers.max(), 0)
    ax.set_xlabel("Latitude (°)", fontsize=11)
    ax.set_ylabel("Depth (m)",    fontsize=11)
    ax.set_title(f"{title} — zonal mean", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label=unit)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Generate all figures ──────────────────────────────────────────────────────
print("\nGenerating figures...")

# Global maps — fixed physically meaningful colorbar ranges
plot_tracer(NO3,                     "Nitrate (NO₃)",          "YlGn_r",  "mmol N m⁻³",  "NO3_map.png",  vmin_fixed=0,    vmax_fixed=40)
plot_tracer(NH4,                     "Ammonium (NH₄)",         "YlOrBr",  "mmol N m⁻³",  "NH4_map.png",  vmin_fixed=0,    vmax_fixed=0.5)
plot_tracer(PO4,                     "Phosphate (PO₄)",        "YlGn_r",  "mmol P m⁻³",  "PO4_map.png",  vmin_fixed=0,    vmax_fixed=3.0)
plot_tracer(Si,                      "Silicate (Si)",           "Blues",   "mmol Si m⁻³", "Si_map.png",   vmin_fixed=0,    vmax_fixed=150)
plot_tracer(Fe*1e3,                  "Dissolved Iron (Fe)",     "YlOrBr",  "nmol Fe m⁻³", "Fe_map.png",   vmin_fixed=0,    vmax_fixed=1.0)
plot_tracer(Chl_tot,                 "Total Chlorophyll",       "YlGn",    "mg Chl m⁻³",  "Chl_map.png",  vmin_fixed=0,    vmax_fixed=3.0)
plot_tracer(Pd,                      "Diatoms (P_d)",           "Greens",  "mmol N m⁻³",  "Pd_map.png",   vmin_fixed=0,    vmax_fixed=0.5)
plot_tracer(Pn,                      "Nanophyto (P_n)",         "YlGn",    "mmol N m⁻³",  "Pn_map.png",   vmin_fixed=0,    vmax_fixed=0.5)
plot_tracer(Zoo,                     "Zooplankton",             "Oranges", "mmol N m⁻³",  "Zoo_map.png",  vmin_fixed=0,    vmax_fixed=0.1)
plot_tracer(Det,                     "Detritus",                "YlOrRd",  "mmol N m⁻³",  "Det_map.png",  vmin_fixed=0,    vmax_fixed=0.5)
plot_tracer(DON,                     "Dissolved Organic N",     "PuBu",    "mmol N m⁻³",  "DON_map.png",  vmin_fixed=0,    vmax_fixed=0.05)
plot_tracer(POC,                     "Particulate Organic C",   "YlOrRd",  "mmol C m⁻³",  "POC_map.png",  vmin_fixed=0,    vmax_fixed=0.5)
plot_tracer(DIC - np.nanmean(DIC),  "DIC anomaly",             "RdBu_r",  "mmol C m⁻³",  "DIC_map.png",  diverging=True)
plot_tracer(ALK - np.nanmean(ALK),  "ALK anomaly",             "RdBu_r",  "mmol eq m⁻³", "ALK_map.png",  diverging=True)
plot_tracer(T,                       "Temperature",             "RdYlBu_r","°C",           "T_map.png",    vmin_fixed=-2,   vmax_fixed=30)

# Zonal sections
plot_zonal(NO3,                     "Nitrate",       "YlGn_r",   "mmol N m⁻³", "NO3_zonal.png")
plot_zonal(Chl_tot,                 "Chlorophyll",   "YlGn",     "mg Chl m⁻³", "Chl_zonal.png")
plot_zonal(PO4,                     "Phosphate",     "YlGn_r",   "mmol P m⁻³", "PO4_zonal.png")
plot_zonal(Si,                      "Silicate",      "Blues",     "mmol Si m⁻³","Si_zonal.png")
plot_zonal(T,                       "Temperature",   "RdYlBu_r", "°C",          "T_zonal.png")
plot_zonal(DIC - np.nanmean(DIC),  "DIC anomaly",   "RdBu_r",   "mmol C m⁻³", "DIC_zonal.png", diverging=True)

print(f"\nAll figures written to {OUT_DIR}/")
