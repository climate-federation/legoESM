"""Figures for a 1-D MARBL column run written by run_marbl_column.py.

Writes three PNGs next to the input file:
  <stem>_timeseries.png  surface/bulk diagnostics over the whole run
  <stem>_hovmoller.png   upper-ocean depth-time sections (last 2 years) + KPP h_bl
  <stem>_profiles.png    full-depth profiles: first day vs last day (drift check)

Usage:
    python scripts/plot_marbl_column.py results/marbl_column_bats/bats_5yr.nc
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PFT_COLORS = {"sp": "#2a78b5", "diat": "#d1772a", "diaz": "#5a9e45"}
INK, MUTED = "#222222", "#888888"
S2D = 86400.0
S2Y_GC = 86400 * 365 * 12e-3          # mmol C/m2/s -> g C/m2/yr

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
})


def chl_total(ds):
    return ds.spChl + ds.diatChl + ds.diazChl


def timeseries(ds, out):
    yr = ds.day.values / 365.0
    panels = [
        ("SST [°C]", [(ds.sst, None)], None),
        ("KPP boundary layer depth [m]", [(ds.h_bl, None)], "invert"),
        ("Surface Chl [mg/m³]", [(chl_total(ds).isel(z=0), None)], None),
        ("Surface NO₃ [mmol/m³]", [(ds.NO3.isel(z=0), None)], None),
        ("NPP [g C/m²/yr]", [(ds.NPP.sel(pft=p) * S2Y_GC, p) for p in PFT_COLORS], None),
        ("POC flux at 100 m [mmol C/m²/d]", [(ds.POC_flux_100m * S2D, None)], None),
        ("Surface pCO₂ [ppmv]  (dashed: atmosphere 410)", [(ds.pCO2surf, None)], "co2"),
        ("Air-sea CO₂ flux [mmol/m²/d], + into ocean", [(ds.flux_co2 * S2D, None)], "zero"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, constrained_layout=True)
    for ax, (title, series, extra) in zip(axes.T.ravel(), panels):
        for da, pft in series:
            ax.plot(yr, da.values, color=PFT_COLORS.get(pft, INK), label=pft)
        if any(p for _, p in series):
            ax.legend(loc="upper right", ncol=3)
        if extra == "invert":
            ax.invert_yaxis()
        elif extra == "co2":
            ax.axhline(410, color=MUTED, ls="--", lw=1)
        elif extra == "zero":
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("model year")
    fig.suptitle("MARBL column, BATS — surface and bulk diagnostics", x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def hovmoller(ds, out, zmax=300.0, years=2):
    sub = ds.isel(day=slice(-365 * years, None)).where(ds.z <= zmax, drop=True)
    yr = sub.day.values / 365.0
    fields = [
        ("Total Chl [mg/m³]", chl_total(sub), "Greens"),
        ("NO₃ [mmol/m³]", sub.NO3, "Blues"),
        ("O₂ [mmol/m³]", sub.O2, "Purples"),
        ("Temperature [°C]", sub.T, "Oranges"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for ax, (title, da, cmap) in zip(axes, fields):
        pc = ax.pcolormesh(yr, sub.z.values, da.values.T, cmap=cmap, shading="auto")
        ax.plot(yr, ds.h_bl.isel(day=slice(-365 * years, None)).values, color=INK, lw=1, label="KPP h_bl")
        ax.set_ylim(zmax, 0)
        ax.set_ylabel("depth [m]")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.grid(False)
        fig.colorbar(pc, ax=ax, pad=0.01)
    axes[0].legend(loc="lower right")
    axes[-1].set_xlabel("model year")
    fig.suptitle(f"MARBL column, BATS — upper {zmax:.0f} m, last {years} years", x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def profiles(ds, out):
    names = [("NO3", "NO₃"), ("PO4", "PO₄"), ("SiO3", "SiO₃"), ("Fe", "Fe"),
             ("O2", "O₂"), ("DIC", "DIC"), ("ALK", "ALK"), ("DOC", "DOC")]
    z = ds.z.values
    fig, axes = plt.subplots(2, 4, figsize=(11, 8), sharey=True, constrained_layout=True)
    for ax, (k, label) in zip(axes.ravel(), names):
        ax.plot(ds[k].isel(day=0).values, z, color=MUTED, ls="--", label="day 1")
        ax.plot(ds[k].isel(day=-1).values, z, color=INK, label=f"day {int(ds.day[-1])}")
        unit = "meq/m³" if k == "ALK" else "mmol/m³"
        ax.set_title(f"{label} [{unit}]", loc="left", fontsize=9, color=INK)
    axes[0, 0].invert_yaxis()
    for ax in axes[:, 0]:
        ax.set_ylabel("depth [m]")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("MARBL column, BATS — start vs end profiles (drift check)", x=0.01, ha="left")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/marbl_column_bats/bats_5yr.nc")
    ds = xr.open_dataset(path)
    stem = path.with_suffix("")
    outs = [Path(f"{stem}_timeseries.png"), Path(f"{stem}_hovmoller.png"), Path(f"{stem}_profiles.png")]
    timeseries(ds, outs[0])
    hovmoller(ds, outs[1])
    profiles(ds, outs[2])
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
