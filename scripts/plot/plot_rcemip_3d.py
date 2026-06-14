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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure  # marching-cubes isosurfaces

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 11,
})

_ELEV, _AZIM = 18.0, -55.0          # oblique angle: most of the field visible
_Z_TOP_KM = 18.0                    # clip the (dry) stratosphere above the anvils
_Z_STRETCH = 6.0                    # vertical exaggeration of the z box-aspect
# 3D condensate ISOSURFACE levels [g/kg] (low = translucent outer cloud envelope,
# high = denser cores) with per-level opacity — true 3D contours via marching cubes.
_ISO_LEVELS = (0.02, 0.2, 1.0)
_ISO_ALPHA = (0.12, 0.30, 0.65)
_ISO_COLOR = ("#9ecae1", "#4292c6", "#08519c")   # light->dark blue (GnBu-like)


def _coords(d):
    z = np.asarray(d["z"]) / 1e3                          # km
    Lx, Ly = float(d["Lx"]) / 1e3, float(d["Ly"]) / 1e3
    ny, nx = d["cond"].shape[0], d["cond"].shape[1]
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    return x, y, z, Lx, Ly


def _set_view(ax, Lx, Ly, ztop):
    ax.view_init(elev=_ELEV, azim=_AZIM)
    ax.set_box_aspect((Lx, Ly, ztop * _Z_STRETCH))       # exaggerate z for clarity
    ax.set_xlabel("x [km]", labelpad=2)
    ax.set_ylabel("y [km]", labelpad=2)
    ax.set_zlabel("z [km]", labelpad=2)
    ax.set_zlim(0, ztop)
    ax.tick_params(labelsize=7)


def _contours_condensate(ax, d, x, y, z, Lx, Ly):
    """Condensate as true 3D CONTOURS (marching-cubes ISOSURFACES) at several
    mixing-ratio thresholds, nested + translucent (outer envelope light, dense
    cores dark) so the 3D shape — cores below, anvils aloft — is visible."""
    cond = np.asarray(d["cond"]) * 1e3                     # (ny,nx,nz) g/kg
    ztop = min(_Z_TOP_KM, z.max())
    kmax = int(np.searchsorted(z, ztop)) + 1
    cond = cond[:, :, :kmax]
    zc = z[:kmax]
    ny, nx, nz = cond.shape
    drew = False
    for lev, alpha, col in zip(_ISO_LEVELS, _ISO_ALPHA, _ISO_COLOR):
        if float(cond.max()) <= lev:
            continue
        try:
            verts, faces, _n, _v = measure.marching_cubes(cond, level=lev)
        except (ValueError, RuntimeError):
            continue
        # index-space verts (i=y, j=x, k=z) -> physical km
        vy = verts[:, 0] / max(ny - 1, 1) * Ly
        vx = verts[:, 1] / max(nx - 1, 1) * Lx
        vz = np.interp(verts[:, 2], np.arange(nz), zc)
        tri = np.stack([vx[faces], vy[faces], vz[faces]], axis=-1)
        mesh = Poly3DCollection(tri, alpha=alpha, facecolor=col,
                                edgecolor="none")
        ax.add_collection3d(mesh)
        drew = True
    # proxy mappable for a colorbar (levels legend)
    if drew:
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(
            cmap=mpl.colors.ListedColormap(list(_ISO_COLOR)),
            norm=mpl.colors.BoundaryNorm([0, *_ISO_LEVELS], len(_ISO_COLOR)))
        sm.set_array([])
        return sm
    return None


def _slices_mse(ax, d, x, y, z, Lx, Ly):
    """MSE as 3D CONTOURS (marching-cubes isosurfaces) at a few MSE thresholds
    (translucent nested shells) — consistent with the condensate contours, no
    horizontal slices."""
    import matplotlib as mpl
    mse = np.asarray(d["mse"])                            # (ny,nx,nz) kJ/kg
    ztop = min(_Z_TOP_KM, z.max())
    kmax = int(np.searchsorted(z, ztop)) + 1
    mse = mse[:, :, :kmax]
    zc = z[:kmax]
    ny, nx, nz = mse.shape
    lo, hi = float(np.percentile(mse, 8)), float(np.percentile(mse, 92))
    levels = np.linspace(lo, hi, 4)
    cmap = mpl.colormaps["inferno"]
    norm = mpl.colors.Normalize(lo, hi)
    for lev in levels:
        if not (mse.min() < lev < mse.max()):
            continue
        try:
            verts, faces, _n, _v = measure.marching_cubes(mse, level=lev)
        except (ValueError, RuntimeError):
            continue
        vy = verts[:, 0] / max(ny - 1, 1) * Ly
        vx = verts[:, 1] / max(nx - 1, 1) * Lx
        vz = np.interp(verts[:, 2], np.arange(nz), zc)
        tri = np.stack([vx[faces], vy[faces], vz[faces]], axis=-1)
        ax.add_collection3d(Poly3DCollection(
            tri, alpha=0.30, facecolor=cmap(norm(lev)), edgecolor="none"))
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    return sm


def plot_volume(npz_path: Path, out_png: Path | None = None) -> Path:
    d = np.load(npz_path)
    day = float(d["day"])
    x, y, z, Lx, Ly = _coords(d)
    ztop = min(_Z_TOP_KM, z.max())
    fig = plt.figure(figsize=(15, 7.5))
    axc = fig.add_subplot(1, 2, 1, projection="3d")
    sc = _contours_condensate(axc, d, x, y, z, Lx, Ly)
    _set_view(axc, Lx, Ly, ztop)
    axc.set_title(f"Condensate (cloud + precip)  —  day {day:.0f}")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axc, shrink=0.55, pad=0.08)
        cb.set_label("condensate [g kg$^{-1}$]"); cb.ax.tick_params(labelsize=7)
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
        _contours_condensate(ax, d, x, y, z, Lx, Ly)
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
