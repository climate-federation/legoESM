"""Overlay every-5-day vertical profiles written by ``run_rce_mpi_long.py``.

Reads ``<output>/profiles/prof_day_NNNN.npz`` (one per profile-cadence
step), then writes ``<output>/profiles/profile_evolution.png`` —
a 2x3 panel of T(z), q_v(z), q_c(z), q_r(z), cloud_fraction(z),
w_variance(z), each as overlaid lines colored by simulation day.

Also writes one PNG per snapshot file at
``<output>/profiles/prof_day_NNNN.png`` for the single-day view.

Usage
-----
.. code-block:: bash

   python scripts/plot_rce_profile_evolution.py results/rce_smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("T",              "T [K]",                  "log_x=False"),
    ("qv",             "q_v [kg/kg]",            "log_x=True"),
    ("qc",             "q_c [kg/kg]",            "log_x=True"),
    ("qr",             "q_r [kg/kg]",            "log_x=True"),
    ("cloud_fraction", "cloud fraction [-]",     "log_x=False"),
    ("w_variance",     "var(w) [m²/s²]",         "log_x=False"),
]


def _safe_logx(ax, vals):
    arr = np.asarray(vals)
    if np.any(arr > 0):
        ax.set_xscale("symlog", linthresh=1e-10)


def plot_single_profile(npz_path: Path) -> Path:
    data = np.load(npz_path)
    z_km = data["z"] / 1000.0
    z_half_km = data["z_half"] / 1000.0 if "z_half" in data.files else z_km
    day = float(data["day"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, label, kw) in zip(axes.flat, PANELS):
        # Pick z-coord matching field length (some fields live on
        # half-levels, e.g. w_variance has shape nlev+1).
        zc = z_half_km if data[key].shape[0] == z_half_km.shape[0] else z_km
        ax.plot(data[key], zc, lw=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("z [km]")
        ax.grid(alpha=0.3)
        if "log_x=True" in kw:
            _safe_logx(ax, data[key])
    fig.suptitle(f"RCE profile — day {day:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = npz_path.with_suffix(".png")
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return png_path


def plot_evolution(npz_files, out_png: Path) -> Path:
    series = [np.load(p) for p in npz_files]
    days = np.array([float(d["day"]) for d in series])
    z_km = series[0]["z"] / 1000.0
    z_half_km = (series[0]["z_half"] / 1000.0
                 if "z_half" in series[0].files else z_km)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=days.min(), vmax=max(days.max(), days.min() + 1e-9))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, label, kw) in zip(axes.flat, PANELS):
        for d, data in zip(days, series):
            zc = (z_half_km if data[key].shape[0] == z_half_km.shape[0]
                  else z_km)
            ax.plot(data[key], zc, color=cmap(norm(d)), lw=1.3, alpha=0.85)
        ax.set_xlabel(label)
        ax.set_ylabel("z [km]")
        ax.grid(alpha=0.3)
        if "log_x=True" in kw:
            _safe_logx(ax, np.concatenate([d[key] for d in series]))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("day")
    fig.suptitle(
        f"RCE profile evolution — {len(days)} times, "
        f"days {days.min():.1f}…{days.max():.1f}",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: plot_rce_profile_evolution.py <output_dir>")
        return 2
    out_dir = Path(sys.argv[1])
    prof_dir = out_dir / "profiles"
    if not prof_dir.is_dir():
        print(f"no profiles dir at {prof_dir}")
        return 1
    npz_files = sorted(prof_dir.glob("prof_day_*.npz"))
    if not npz_files:
        print(f"no prof_day_*.npz under {prof_dir}")
        return 1
    for p in npz_files:
        png = plot_single_profile(p)
        print(f"  wrote {png}")
    evo_png = prof_dir / "profile_evolution.png"
    plot_evolution(npz_files, evo_png)
    print(f"  wrote {evo_png}")
    print(f"done. {len(npz_files)} per-day PNGs + 1 evolution PNG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
