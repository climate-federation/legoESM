"""Global maps and time series from the 3-D MARBL run (run_omip.py --biogeo marbl).

Reads the 2-D diagnostics written every --marbl-diag-days
(<output>/latlon/<resolution>/marbl_diag_dayNNNNNN.npz) and the model grid
(.npz from --marbl-write-grid), and writes

  <out>/marbl_global_maps.{png,pdf}        6 maps, mean over the chosen files
  <out>/marbl_global_timeseries.{png,pdf}  global NPP, export and air-sea CO2 flux

Usage (repo root):
  python scripts/plot_marbl_global_maps.py \
      --run-dir results/marbl_1deg_1yr/latlon/180x360 \
      --grid $D/marbl_grid_1deg.npz --last 3 --out results/paper_figs
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

R_EARTH = 6.371e6
SEC_PER_YEAR = 365.0 * 86400.0
INK, INK2, MUTED = "#1f1f1f", "#4a4a4a", "#8a8a85"
LAND = "#d9d6cf"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titlecolor": INK, "savefig.dpi": 300,
})


def cell_area(lat, lon):
    dlat, dlon = np.abs(np.gradient(lat)), np.abs(np.gradient(lon))
    band = np.sin(np.minimum(lat + dlat / 2, np.pi / 2)) - np.sin(np.maximum(lat - dlat / 2, -np.pi / 2))
    return R_EARTH ** 2 * band[:, None] * dlon[None, :]


def load_grid(path):
    g = np.load(path)
    lat, lon = np.asarray(g["lat"], float), np.asarray(g["lon"], float)
    if np.abs(lat).max() > np.pi / 2 + 1e-6:
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return lat, lon, np.asarray(g["mask"]) > 0.5


def load_diags(run_dir):
    files = sorted(glob.glob(str(Path(run_dir) / "marbl_diag_day*.npz")))
    if not files:
        raise SystemExit(f"no marbl_diag_day*.npz in {run_dir}")
    return files


def to_physical(d):
    """Model units -> plotting units."""
    npp = np.asarray(d["NPP"])
    npp = npp.sum(0) if npp.ndim == 3 else npp
    return {
        "Chl": np.asarray(d["Chl_surf"]),                                  # mg Chl m-3
        "NPP": npp * 12e-3 * SEC_PER_YEAR,                                 # g C m-2 yr-1
        "POC": np.asarray(d["POC_flux_100m"]) * 12e-3 * SEC_PER_YEAR,      # g C m-2 yr-1
        "fCO2": np.asarray(d["flux_co2"]) * 1e-3 * SEC_PER_YEAR,           # mol C m-2 yr-1, + into ocean
        "pCO2": np.asarray(d["pCO2surf"]),                                 # uatm
        "pH": np.asarray(d["pH_surf"]),
    }


def global_totals(f, area, mask):
    a = area * mask
    pg = lambda x: float(np.nansum(x * a) * 1e-15)          # noqa: E731  g C yr-1 -> Pg C yr-1
    return pg(f["NPP"]), pg(f["POC"]), pg(f["fCO2"] * 12.0)


def robust(x, mask, lo=2, hi=98):
    v = x[mask & np.isfinite(x)]
    return np.percentile(v, lo), np.percentile(v, hi)


def maps(files, lat, lon, mask, area, out, label):
    acc = None
    for f in files:
        p = to_physical(np.load(f))
        acc = p if acc is None else {k: acc[k] + p[k] for k in acc}
    m = {k: v / len(files) for k, v in acc.items()}
    for k in m:
        m[k] = np.where(mask, m[k], np.nan)
    npp, poc, fco2 = global_totals({k: np.nan_to_num(v) for k, v in m.items()}, area, mask)

    latd, lond = np.rad2deg(lat), np.rad2deg(lon)
    dlo = np.median(np.diff(lond))
    dla = np.median(np.diff(latd))
    lon_e = np.concatenate([lond - dlo / 2, [lond[-1] + dlo / 2]])
    lat_e = np.clip(np.concatenate([latd - dla / 2, [latd[-1] + dla / 2]]), -90, 90)

    c_lo, c_hi = robust(m["fCO2"], mask)
    cl = max(abs(c_lo), abs(c_hi), 1e-3)
    p_lo, p_hi = robust(m["pCO2"], mask)
    h_lo, h_hi = robust(m["pH"], mask)
    panels = [
        ("Chl", "a  Surface chlorophyll", "mg m$^{-3}$", "Greens",
         LogNorm(vmin=0.02, vmax=max(0.05, robust(m["Chl"], mask, 1, 99.5)[1]))),
        ("NPP", f"b  NPP (global {npp:.1f} Pg C yr$^{{-1}}$)", "g C m$^{-2}$ yr$^{-1}$", "Greens",
         plt.Normalize(0, robust(m["NPP"], mask, 1, 99)[1])),
        ("POC", f"c  POC export at 100 m (global {poc:.2f} Pg C yr$^{{-1}}$)", "g C m$^{-2}$ yr$^{-1}$",
         "Blues", plt.Normalize(0, robust(m["POC"], mask, 1, 99)[1])),
        ("fCO2", f"d  Air-sea CO$_2$ flux (global {fco2:+.2f} Pg C yr$^{{-1}}$)",
         "mol C m$^{-2}$ yr$^{-1}$ (+ into ocean)", "RdBu_r", TwoSlopeNorm(0.0, -cl, cl)),
        ("pCO2", "e  Surface pCO$_2$", "µatm", "Purples", plt.Normalize(p_lo, p_hi)),
        ("pH", "f  Surface pH", "", "Purples_r", plt.Normalize(h_lo, h_hi)),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 7.4), constrained_layout=True)
    for ax, (k, title, unit, cmap, norm) in zip(axs.flat, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(LAND)
        pc = ax.pcolormesh(lon_e, lat_e, m[k], cmap=cm, norm=norm, shading="flat", rasterized=True)
        ax.set_title(title, loc="left")
        ax.set_xlim(lon_e[0], lon_e[-1]); ax.set_ylim(-80, 90)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360]); ax.set_yticks([-60, -30, 0, 30, 60, 90])
        ax.set_aspect("auto")
        ax.tick_params(labelsize=6.5, length=2)
        cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01,
                          extend="both" if k in ("fCO2", "pCO2", "pH") else "max")
        cb.set_label(unit, fontsize=7); cb.ax.tick_params(labelsize=6.5)
        cb.outline.set_edgecolor(MUTED)
    d0, d1 = np.load(files[0])["day_start"], np.load(files[-1])["day_end"]
    fig.suptitle(f"MARBL in legoESM, 1° JRA55-do RYF{label}: mean of model days {float(d0):.0f}–{float(d1):.0f}",
                 x=0.01, ha="left", fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"marbl_global_maps.{ext}")
    plt.close(fig)
    print(f"maps: days {float(d0):.0f}-{float(d1):.0f}: NPP {npp:.1f} PgC/yr, export {poc:.2f} PgC/yr, "
          f"air-sea CO2 {fco2:+.2f} PgC/yr -> {out}/marbl_global_maps.png")


def timeseries(files, area, mask, out):
    t, rows = [], []
    for f in files:
        d = np.load(f)
        p = {k: np.nan_to_num(v) for k, v in to_physical(d).items()}
        t.append(0.5 * (float(d["day_start"]) + float(d["day_end"])))
        rows.append(global_totals(p, area, mask))
    rows = np.array(rows)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), constrained_layout=True, sharex=True)
    spec = [("Global NPP", "Pg C yr$^{-1}$"), ("Export at 100 m", "Pg C yr$^{-1}$"),
            ("Air-sea CO$_2$ flux (+ into ocean)", "Pg C yr$^{-1}$")]
    for i, (ax, (title, unit)) in enumerate(zip(axs, spec)):
        ax.plot(t, rows[:, i], color="#2a78d6", lw=1.6, marker="o", ms=3)
        if i == 2:
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc="left"); ax.set_ylabel(unit); ax.set_xlabel("model day")
        ax.grid(color="#e6e4df", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"marbl_global_timeseries.{ext}")
    plt.close(fig)
    print("time series (day: NPP, export, CO2 flux in PgC/yr):")
    for ti, r in zip(t, rows):
        print(f"  {ti:7.1f}: {r[0]:6.1f} {r[1]:6.2f} {r[2]:+6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="directory with marbl_diag_day*.npz")
    ap.add_argument("--grid", required=True, help="model grid .npz (run_omip.py --marbl-write-grid)")
    ap.add_argument("--last", type=int, default=0, help="average the last N diag files (0 = all)")
    ap.add_argument("--out", default="results/paper_figs")
    ap.add_argument("--label", default="", help="extra text for the map title")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lat, lon, mask = load_grid(a.grid)
    area = cell_area(lat, lon)
    files = load_diags(a.run_dir)
    print(f"{len(files)} diag files: {Path(files[0]).name} .. {Path(files[-1]).name}")
    maps(files[-a.last:] if a.last else files, lat, lon, mask, area, out, a.label)
    timeseries(files, area, mask, out)


if __name__ == "__main__":
    main()
