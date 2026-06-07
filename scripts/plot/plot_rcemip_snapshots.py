"""Publication-quality surface + 4-level snapshots for the RCEMIP plane CRM.

Reads ``<output>/snapshots/sfc_NNN.npz`` (written by ``scripts/run/rce_snapshot.py``)
and renders, per frame, ``<output>/snapshots/sfc_NNN.png``:

* top row  — surface fields: precipitation rate, column water vapour, column-max
  |w| (convective cores), near-surface water vapour;
* lower rows — horizontal cross-sections of total condensate, vertical velocity w
  and water vapour at the four requested heights.

A ``<case>`` hero copy of the last frame is written at
``<output>/rcemip_surface_final.png``. 300-dpi serif scientific style.

Usage:  python scripts/plot/plot_rcemip_snapshots.py results/rcemip300_60day
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10,
    "axes.linewidth": 0.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def _panel(ax, fld, extent, cmap, label, vmin=None, vmax=None, quiver=None):
    im = ax.imshow(fld, origin="lower", extent=extent, cmap=cmap, aspect="equal",
                   vmin=vmin, vmax=vmax, interpolation="bilinear", rasterized=True)
    ax.set_title(label, fontsize=9)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    return im


def plot_frame(npz_path: Path, out_png: Path | None = None) -> Path:
    d = np.load(npz_path)
    day = float(d["day"])
    Lx, Ly = float(d["Lx"]) / 1e3, float(d["Ly"]) / 1e3
    extent = [0.0, Lx, 0.0, Ly]
    heights = np.asarray(d["heights"])
    # clip O(1e-12) round-off negatives in near-zero condensate to 0 (hydrometeors
    # are physically >= 0; the CRM transport is not strictly positivity-preserving).
    cond = np.maximum(np.asarray(d["cond_levels"]), 0.0) * 1e3   # g/kg
    w = np.asarray(d["w_levels"])
    qv = np.asarray(d["qv_levels"]) * 1e3       # g/kg
    nh = heights.shape[0]

    fig, axes = plt.subplots(4, nh, figsize=(3.1 * nh, 11.5),
                             squeeze=False, layout="constrained")
    # Row 0 — surface fields (first 4 columns; pad if nh!=4).
    surf = [("precip", "Blues", "precip [mm day$^{-1}$]", 0.0,
             float(np.nanpercentile(d["precip"], 99.5)) or 1.0),
            ("cwv", "viridis", "CWV [kg m$^{-2}$]", None, None),
            ("wcolmax", "magma", r"column-max $|w|$ [m s$^{-1}$]", 0.0, None),
            ("qv_sfc", "YlGnBu", r"$q_v$ surface [kg kg$^{-1}$]", None, None)]
    for c in range(nh):
        ax = axes[0][c]
        if c < len(surf):
            key, cmap, label, vmn, vmx = surf[c]
            _panel(ax, np.asarray(d[key]), extent, cmap, label, vmn, vmx)
        else:
            ax.axis("off")
    # Rows 1-3 — condensate / w / qv at the four heights.
    wlim = float(np.nanpercentile(np.abs(w), 99.0)) or 1.0
    rows = [(cond, "GnBu", "condensate [g kg$^{-1}$]", 0.0, None, False),
            (w, "RdBu_r", "w [m s$^{-1}$]", -wlim, wlim, False),
            (qv, "YlGnBu", r"$q_v$ [g kg$^{-1}$]", 0.0, None, False)]
    for r, (fld, cmap, label, vmn, vmx, _q) in enumerate(rows, start=1):
        for c in range(nh):
            ax = axes[r][c]
            _panel(ax, fld[c], extent, cmap,
                   f"{label}\nz = {heights[c]/1e3:.1f} km", vmn, vmx)
            if r == 3:
                ax.set_xlabel("x [km]")
            if c == 0:
                ax.set_ylabel("y [km]")
    fig.suptitle(f"RCEMIP-I (RCE300) — surface & level fields, day {day:.1f}",
                 fontsize=13)
    out_png = out_png or npz_path.with_suffix(".png")
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: plot_rcemip_snapshots.py <output_dir>")
        return 2
    out = Path(sys.argv[1])
    files = sorted((out / "snapshots").glob("sfc_*.npz"))
    if not files:
        print(f"no sfc_*.npz under {out}/snapshots")
        return 1
    for p in files:
        print("  wrote", plot_frame(p))
    plot_frame(files[-1], out / "rcemip_surface_final.png")
    print("  wrote", out / "rcemip_surface_final.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
