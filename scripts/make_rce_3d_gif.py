"""Make 3D-angled GIFs of MSE, q_v, T volumes over time.

Reads ``<output>/snapshots_3d/snap_hr_NNNN.npz`` (written by
``run_rce_mpi_long.py`` when ``--snapshot-3d-hours > 0``) and writes
three GIFs in the same directory:

* ``mse.gif`` — column-stacked horizontal slices of MSE [J/kg]
* ``qv.gif`` — same, q_v [g/kg]
* ``T.gif``  — same, T [K]

Each frame is a 3D matplotlib view rendered as a stack of horizontal
``contourf`` slices at chosen z-levels, viewed at an angle so the
whole 3D field is visible. Color limits are fixed across all frames
per variable (min/max scanned in one pass).

Usage
-----
.. code-block:: bash

   python scripts/make_rce_3d_gif.py results/rce_smoke
   python scripts/make_rce_3d_gif.py results/rce_smoke --duration 0.08
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

# Variables we render: (npz_key, gif_name, label, cmap, scale_fn)
VARS = [
    ("mse", "mse.gif", "MSE [kJ/kg]",   "magma",   lambda x: x / 1000.0),
    ("qv",  "qv.gif",  "q_v [g/kg]",    "BrBG",    lambda x: x * 1000.0),
    ("T",   "T.gif",   "T [K]",         "inferno", lambda x: x),
]

# Choose ~8 levels for the slice stack — too many = unreadable overdraw.
N_SLICES = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=str, help="run output dir")
    p.add_argument("--duration", type=float, default=0.08,
                   help="seconds per frame in GIF (default 0.08).")
    p.add_argument("--elev", type=float, default=25.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--max-frames", type=int, default=0,
                   help="If > 0, cap total frames per GIF (random "
                        "subsample). 0 = use all.")
    return p.parse_args()


def _scan_minmax(npz_files, key, scale_fn):
    vmin, vmax = +np.inf, -np.inf
    for p in npz_files:
        with np.load(p) as d:
            f = scale_fn(d[key])
            vmin = float(min(vmin, np.nanmin(f)))
            vmax = float(max(vmax, np.nanmax(f)))
    return vmin, vmax


def _select_levels(nlev):
    if nlev <= N_SLICES:
        return list(range(nlev))
    return [int(round(i * (nlev - 1) / (N_SLICES - 1)))
            for i in range(N_SLICES)]


def _render_frame(field_3d, z, kk, label, cmap, vmin, vmax,
                  hour, elev, azim):
    """Render one frame: stack of contourf slices in 3D."""
    ny, nx, nlev = field_3d.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    levels = np.linspace(vmin, vmax, 21)
    for k in kk:
        slice_2d = field_3d[:, :, k]
        z_km = z[k] / 1000.0
        # Use uniform alpha; layered contourf reads as volumetric.
        ax.contourf(
            X, Y, slice_2d, levels=levels,
            zdir="z", offset=z_km, cmap=cmap,
            vmin=vmin, vmax=vmax, alpha=0.55, antialiased=False,
        )
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_zlim(0, z[-1] / 1000.0)
    ax.set_xlabel("x (cell)")
    ax.set_ylabel("y (cell)")
    ax.set_zlabel("z [km]")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{label}  —  hour {hour:.1f}", fontsize=11)
    # Single colorbar reference (use one of the slices).
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, label=label)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def build_gif(npz_files, var_key, gif_name, label, cmap, scale_fn,
              out_dir, duration, elev, azim, max_frames):
    if max_frames and len(npz_files) > max_frames:
        idxs = np.linspace(0, len(npz_files) - 1, max_frames, dtype=int)
        files = [npz_files[i] for i in idxs]
    else:
        files = npz_files
    print(f"  scanning {var_key} min/max over {len(files)} frames...")
    vmin, vmax = _scan_minmax(files, var_key, scale_fn)
    print(f"    {var_key}: vmin={vmin:.4g}  vmax={vmax:.4g}")
    frames = []
    with np.load(files[0]) as d0:
        z = np.asarray(d0["z"], dtype=float)
        nlev = z.shape[0]
    kk = _select_levels(nlev)
    for i, p in enumerate(files):
        with np.load(p) as d:
            field = scale_fn(np.asarray(d[var_key], dtype=float))
            hour = float(d["hour"])
        frame = _render_frame(
            field, z, kk, label, cmap, vmin, vmax, hour, elev, azim,
        )
        frames.append(frame)
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            print(f"    {var_key}: frame {i+1}/{len(files)}")
    gif_path = out_dir / gif_name
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"  wrote {gif_path} ({len(frames)} frames)")
    return gif_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    snap3d_dir = out_dir / "snapshots_3d"
    if not snap3d_dir.is_dir():
        print(f"no snapshots_3d dir at {snap3d_dir}")
        return 1
    npz_files = sorted(snap3d_dir.glob("snap_hr_*.npz"))
    if not npz_files:
        print(f"no snap_hr_*.npz under {snap3d_dir}")
        return 1
    print(f"found {len(npz_files)} 3D snapshot files")
    for key, name, label, cmap, scale_fn in VARS:
        build_gif(
            npz_files, key, name, label, cmap, scale_fn,
            snap3d_dir, args.duration, args.elev, args.azim,
            args.max_frames,
        )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
