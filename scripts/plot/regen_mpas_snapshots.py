"""Regenerate MPAS 2D snapshot plots (SST, eta, speed_sfc, etc.) using the
native Voronoi PolyCollection path, from a run's ``snapshots_native.npz`` +
``snapshots_native.nc``.

Motivation: runs completed before the late-night plotting fixes sometimes
stored 2D snapshot PNGs that were actually rendered via the regrid+imshow
fallback (global 181×360 target, then cropped) — producing visible 1°×1°
rectangular pixels even on a 20 km Voronoi mesh. This script rebuilds the
Voronoi mesh from the same (lon_range, lat_range, resolution) used by the
run, matches its filtered cell set to the stored snapshot arrays, and
redraws the snapshots via ``_plot_voronoi_field`` for proper hexagon
rendering.

Usage
-----
    .venv/bin/python scripts/regen_mpas_snapshots.py <run_dir> [<run_dir>...]

Each run directory must contain ``snapshots_native.nc`` with ``lonCell`` /
``latCell`` coordinates and a stacked-per-field ``snapshots_native.npz``.
The domain bounds and resolution are read from the run's ``results.txt``
(``grid``, ``resolution``) plus ``snapshots_native.nc`` coordinate extent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matrix"))  # ocean_test_matrix package home
from ocean_test_matrix.regridding import _plot_voronoi_field  # noqa: E402


FIELD_SPECS: list[tuple[str, str, str, bool]] = [
    # (key, label, cmap, diverging)
    ("SST", "SST (degC)", "RdYlBu_r", False),
    ("eta", "SSH (m)", "RdBu_r", True),
    ("speed_sfc", "Surface Speed (m/s)", "plasma", False),
    ("w_sfc", "Surface w (m/s)", "RdBu_r", True),
    ("w_133m", "w at 133 m (m/s)", "RdBu_r", True),
]


def _resolution_to_km(tag: str) -> float:
    tag = tag.strip().lower().rstrip("km")
    try:
        return float(tag)
    except ValueError:
        return 20.0


def _build_matching_mesh(run_dir: Path, n_cells_target: int):
    """Rebuild a VoronoiMesh whose post-filter cell count matches the run."""
    from legoesm.grids.voronoi import create_regional_voronoi_mesh

    # Resolution from results.txt
    res_km = 20.0
    results_txt = run_dir / "results.txt"
    if results_txt.exists():
        for line in results_txt.read_text().splitlines():
            if line.startswith("resolution:"):
                res_km = _resolution_to_km(line.split(":", 1)[1])
    # Domain extent from the NC coords (already in degrees)
    with xr.open_dataset(run_dir / "snapshots_native.nc") as ds:
        lon_deg = np.asarray(ds.lonCell.values, dtype=np.float64)
        lat_deg = np.asarray(ds.latCell.values, dtype=np.float64)

    # Recover the nominal "outer" range by padding a bit — the stored
    # coordinates are only the cells that survived the run's trim.
    lon_lo = float(np.floor(lon_deg.min()))
    lon_hi = float(np.ceil(lon_deg.max()))
    lat_lo = float(np.floor(lat_deg.min()))
    lat_hi = float(np.ceil(lat_deg.max()))
    # Eady_uniform default is a 10x20° box with 0.5° margin.
    # Use hard bounds so the mesh matches regardless of sparsity in stored data.
    if abs(lon_hi - lon_lo - 10.0) <= 1.0:
        lon_lo, lon_hi = 0.0, 10.0
    if abs(lat_hi - lat_lo - 20.0) <= 1.0:
        lat_lo, lat_hi = 15.0, 35.0

    mesh = create_regional_voronoi_mesh(
        (lon_lo, lon_hi), (lat_lo, lat_hi),
        resolution_km=res_km, periodic_x=True,
    )
    return mesh, (lon_lo, lon_hi, lat_lo, lat_hi), lon_deg, lat_deg


def _map_run_cells_to_mesh(mesh, lon_run: np.ndarray, lat_run: np.ndarray
                           ) -> np.ndarray:
    """Build an index array of length mesh.nCells that places each run-cell
    value at its matching mesh position; remaining mesh cells receive NaN.
    Matching is nearest-neighbour in (lon, lat) on the unit sphere, with a
    strict distance cap derived from the mesh spacing.
    """
    from scipy.spatial import cKDTree

    lon_mesh = np.degrees(np.asarray(mesh.lonCell))
    lat_mesh = np.degrees(np.asarray(mesh.latCell))
    d2r = np.pi / 180.0
    mesh_xyz = np.column_stack([
        np.cos(lat_mesh * d2r) * np.cos(lon_mesh * d2r),
        np.cos(lat_mesh * d2r) * np.sin(lon_mesh * d2r),
        np.sin(lat_mesh * d2r),
    ])
    run_xyz = np.column_stack([
        np.cos(lat_run * d2r) * np.cos(lon_run * d2r),
        np.cos(lat_run * d2r) * np.sin(lon_run * d2r),
        np.sin(lat_run * d2r),
    ])
    tree = cKDTree(mesh_xyz)
    dist, idx = tree.query(run_xyz, k=1)
    # Reverse map: mesh_index → run_index (may have unfilled slots)
    mesh_to_run = np.full(mesh.nCells, -1, dtype=np.int64)
    mesh_to_run[idx] = np.arange(len(lon_run))
    return mesh_to_run


def _expand_to_mesh(field_run: np.ndarray, mesh_to_run: np.ndarray,
                    fill=np.nan) -> np.ndarray:
    out = np.full(mesh_to_run.shape, fill, dtype=np.float64)
    have = mesh_to_run >= 0
    out[have] = field_run[mesh_to_run[have]]
    return out


def regen_run(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    nc_path = run_dir / "snapshots_native.nc"
    npz_path = run_dir / "snapshots_native.npz"
    if not (nc_path.exists() and npz_path.exists()):
        print(f"[skip] {run_dir}: native snapshot files missing")
        return

    npz = np.load(npz_path, allow_pickle=True)
    steps = np.asarray(npz["steps"], dtype=np.int64)
    times_days = np.asarray(npz["times_days"], dtype=np.float64)

    mesh, extent, lon_run, lat_run = _build_matching_mesh(
        run_dir, n_cells_target=int(npz["SST"].shape[1]))
    print(f"[{run_dir.name}] mesh.nCells={mesh.nCells}, "
          f"run nCells={int(npz['SST'].shape[1])}, "
          f"domain={extent}")

    mesh_to_run = _map_run_cells_to_mesh(mesh, lon_run, lat_run)

    # Pick up to 8 evenly spaced snapshot indices.
    n_times = len(steps)
    if n_times > 8:
        sel = np.linspace(0, n_times - 1, 8).astype(int)
    else:
        sel = np.arange(n_times)

    for field_key, label, cmap, diverging in FIELD_SPECS:
        if field_key not in npz.files:
            continue
        data = np.asarray(npz[field_key])  # (n_times, nCells_run)
        n_cols = min(4, len(sel))
        n_rows = (len(sel) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 3.8 * n_rows))
        axes = np.atleast_2d(axes)

        # First pass: compute per-field shared color range on ocean cells.
        lm_series = npz.get("land_mask")  # (n_times, nCells_run)
        all_vals = []
        for ti in sel:
            field_run = data[ti]
            if lm_series is not None:
                m = np.asarray(lm_series[ti]) > 0.5
                all_vals.append(field_run[m])
            else:
                all_vals.append(field_run)
        finite = np.concatenate([v[np.isfinite(v)] for v in all_vals])
        vmin, vmax = (float(finite.min()), float(finite.max())) \
            if finite.size else (None, None)
        if diverging and vmin is not None:
            vlim = max(abs(vmin), abs(vmax))
            vmin, vmax = -vlim, vlim

        for ax_i, ti in enumerate(sel):
            r, c = divmod(ax_i, n_cols)
            ax = axes[r, c]
            field_run = data[ti]
            field_mesh = _expand_to_mesh(field_run, mesh_to_run)
            lm_mesh = (_expand_to_mesh(lm_series[ti], mesh_to_run, fill=0.0)
                       if lm_series is not None else None)
            im = _plot_voronoi_field(
                ax, mesh, field_mesh,
                land_mask=lm_mesh, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("auto")
            if im is not None:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"t={times_days[ti]:.1f} d", fontsize=9)
            if c == 0:
                ax.set_ylabel("Latitude")
            if r == n_rows - 1:
                ax.set_xlabel("Longitude")

        for ax_i in range(len(sel), n_rows * n_cols):
            r, c = divmod(ax_i, n_cols)
            axes[r, c].set_visible(False)

        fig.suptitle(f"{run_dir.name} — {field_key} ({label})", fontsize=11)
        fig.tight_layout()
        out = run_dir / f"snapshots_{field_key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [ok] wrote {out.name}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        regen_run(Path(d))


if __name__ == "__main__":
    main()
