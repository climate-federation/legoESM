"""Render PNG per surface snapshot written by ``run_rce_mpi_long.py``.

Reads ``<output>/snapshots/snap_day_NNNN.npz`` and writes
``<output>/snapshots/snap_day_NNNN.png`` — 3x3 panel grid of the
surface variables (CWV, MSE, precip, T_sfc, q_v_sfc, q_c_sfc,
q_r_sfc, |U|_sfc, plus a u/v quiver overlay on |U|_sfc).

Usage
-----
.. code-block:: bash

   python scripts/plot_rce_surface_snapshots.py results/rce_smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("cwv",      "CWV [kg/m²]",        "viridis"),
    ("mse",      "MSE [J/m²]",         "magma"),
    ("precip",   "precip proxy [kg/m²/s]", "Blues"),
    ("T_sfc",    "T_sfc [K]",          "inferno"),
    ("qv_sfc",   "q_v sfc [kg/kg]",    "BrBG"),
    ("qc_sfc",   "q_c sfc [kg/kg]",    "Purples"),
    ("qr_sfc",   "q_r sfc [kg/kg]",    "Greens"),
    ("wind_sfc", "|U| sfc [m/s]",      "cividis"),
]


def plot_snapshot(npz_path: Path) -> Path:
    data = np.load(npz_path)
    day = float(data["day"])
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, (key, label, cmap) in zip(axes.flat, PANELS):
        field = data[key]
        im = ax.imshow(field, origin="lower", cmap=cmap, aspect="equal")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # last panel: wind quiver on |U|
    ax = axes.flat[8]
    u = data["u_sfc"]
    v = data["v_sfc"]
    wind = data["wind_sfc"]
    im = ax.imshow(wind, origin="lower", cmap="cividis", aspect="equal")
    step = max(1, wind.shape[0] // 16)
    ys, xs = np.mgrid[0:wind.shape[0], 0:wind.shape[1]]
    ax.quiver(
        xs[::step, ::step], ys[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        color="white", scale=120.0, width=0.003,
    )
    ax.set_title("|U| sfc + (u,v) quiver", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"RCE surface — day {day:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = npz_path.with_suffix(".png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return png_path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: plot_rce_surface_snapshots.py <output_dir>")
        return 2
    out_dir = Path(sys.argv[1])
    snap_dir = out_dir / "snapshots"
    if not snap_dir.is_dir():
        print(f"no snapshots dir at {snap_dir}")
        return 1
    npz_files = sorted(snap_dir.glob("snap_day_*.npz"))
    if not npz_files:
        print(f"no snap_day_*.npz under {snap_dir}")
        return 1
    for p in npz_files:
        png = plot_snapshot(p)
        print(f"  wrote {png}")
    print(f"done. {len(npz_files)} PNGs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
