"""Publication-quality 3D visualisation of RCEMIP condensate + moist static energy.

Reads the full-3D volume dumps ``<output>/snapshots3d/vol_NNN.npz`` (written by
``rce_snapshot.save_3d``) and renders, per dump (typically days 15/30/45/60):

* ``<output>/snapshots3d/vol3d_dayDD.png`` — a two-panel 3D view at a fixed
  oblique angle: (left) the condensate field as an alpha-blended point cloud
  (anvils aloft, convective cores below), (right) moist static energy as a stack
  of translucent horizontal slices through the troposphere.
* ``<output>/rcemip_condensate_3d_montage.png`` — the condensate panels for all
  available days in one 2x2 figure.

matplotlib mplot3d only (no skimage); 300-dpi serif scientific style.

Usage:  python scripts/plot/plot_rcemip_3d.py results/rcemip300_60day
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 11,
})

_ELEV, _AZIM = 22.0, -55.0          # oblique angle: most of the field visible
_Z_TOP_KM = 18.0                    # clip the (dry) stratosphere above the anvils
_COND_THR = 1.0e-5                  # condensate threshold [kg/kg] (0.01 g/kg)
_MAX_PTS = 60000                    # scatter budget (subsample beyond this)


def _coords(d):
    z = np.asarray(d["z"]) / 1e3                          # km
    Lx, Ly = float(d["Lx"]) / 1e3, float(d["Ly"]) / 1e3
    ny, nx = d["cond"].shape[0], d["cond"].shape[1]
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    return x, y, z, Lx, Ly


def _set_view(ax, Lx, Ly, ztop):
    ax.view_init(elev=_ELEV, azim=_AZIM)
    ax.set_box_aspect((Lx, Ly, ztop * 1.6))              # exaggerate z for clarity
    ax.set_xlabel("x [km]", labelpad=2)
    ax.set_ylabel("y [km]", labelpad=2)
    ax.set_zlabel("z [km]", labelpad=2)
    ax.set_zlim(0, ztop)
    ax.tick_params(labelsize=7)


def _scatter_condensate(ax, d, x, y, z, Lx, Ly):
    cond = np.asarray(d["cond"])                          # (ny,nx,nz) kg/kg
    ztop = min(_Z_TOP_KM, z.max())
    kmax = int(np.searchsorted(z, ztop)) + 1
    cond = cond[:, :, :kmax]
    zc = z[:kmax]
    jj, ii, kk = np.where(cond > _COND_THR)
    val = cond[jj, ii, kk] * 1e3                          # g/kg
    if val.size > _MAX_PTS:                               # subsample, keep densest
        sel = np.random.default_rng(0).choice(
            val.size, _MAX_PTS, replace=False, p=val / val.sum())
        jj, ii, kk, val = jj[sel], ii[sel], kk[sel], val[sel]
    if val.size == 0:
        return None
    sc = ax.scatter(x[ii], y[jj], zc[kk], c=val, cmap="GnBu",
                    s=2.0, alpha=0.18, vmin=0.0,
                    vmax=float(np.percentile(val, 99)), linewidths=0,
                    norm=None, depthshade=True)
    return sc


def _slices_mse(ax, d, x, y, z, Lx, Ly):
    mse = np.asarray(d["mse"])                            # (ny,nx,nz) kJ/kg
    ztop = min(_Z_TOP_KM, z.max())
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(float(np.percentile(mse, 2)),
                         float(np.percentile(mse, 98)), 24)
    sm = None
    for h in (0.1, 2.0, 5.0, 9.0, 13.0):                  # km slices through troposphere
        if h > ztop:
            continue
        k = int(np.argmin(np.abs(z - h)))
        sm = ax.contourf(X, Y, mse[:, :, k], zdir="z", offset=z[k],
                         levels=levels, cmap="inferno", alpha=0.55)
    return sm


def plot_volume(npz_path: Path, out_png: Path | None = None) -> Path:
    d = np.load(npz_path)
    day = float(d["day"])
    x, y, z, Lx, Ly = _coords(d)
    ztop = min(_Z_TOP_KM, z.max())
    fig = plt.figure(figsize=(15, 7.5))
    axc = fig.add_subplot(1, 2, 1, projection="3d")
    sc = _scatter_condensate(axc, d, x, y, z, Lx, Ly)
    _set_view(axc, Lx, Ly, ztop)
    axc.set_title(f"Condensate (cloud + precip)  —  day {day:.0f}")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axc, shrink=0.55, pad=0.08)
        cb.set_label("condensate [g kg$^{-1}$]"); cb.ax.tick_params(labelsize=7)
        cb.solids.set_alpha(1.0)
    axm = fig.add_subplot(1, 2, 2, projection="3d")
    sm = _slices_mse(axm, d, x, y, z, Lx, Ly)
    _set_view(axm, Lx, Ly, ztop)
    axm.set_title(f"Moist static energy slices  —  day {day:.0f}")
    if sm is not None:
        cb = fig.colorbar(sm, ax=axm, shrink=0.55, pad=0.08)
        cb.set_label("MSE [kJ kg$^{-1}$]"); cb.ax.tick_params(labelsize=7)
    fig.suptitle("RCEMIP-I (RCE300) 3D structure", fontsize=13)
    out_png = out_png or npz_path.with_name(f"vol3d_day{day:02.0f}.png")
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plot_montage(npz_files, out_png: Path) -> Path:
    n = len(npz_files)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(7.5 * ncol, 6.0 * nrow))
    for i, p in enumerate(npz_files):
        d = np.load(p)
        day = float(d["day"])
        x, y, z, Lx, Ly = _coords(d)
        ztop = min(_Z_TOP_KM, z.max())
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        _scatter_condensate(ax, d, x, y, z, Lx, Ly)
        _set_view(ax, Lx, Ly, ztop)
        ax.set_title(f"day {day:.0f}")
    fig.suptitle("RCEMIP-I condensate — 3D evolution", fontsize=13)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: plot_rcemip_3d.py <output_dir>")
        return 2
    out = Path(sys.argv[1])
    files = sorted((out / "snapshots3d").glob("vol_*.npz"))
    if not files:
        print(f"no vol_*.npz under {out}/snapshots3d")
        return 1
    for p in files:
        print("  wrote", plot_volume(p))
    print("  wrote", plot_montage(files, out / "rcemip_condensate_3d_montage.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
