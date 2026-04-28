"""Replot Eady channel runs with a sensible aspect ratio and correct domain.

The stock `snapshots_SST.png` / `snapshots_eta.png` produced by the
ocean test matrix have two problems on the 10° × 18° Eady channel:

1. MPAS snapshots show the regrid target range [0, 360] instead of the
   actual source domain [0, 10], so the signal is a tiny sliver.
2. Both lat-lon and MPAS panels use `aspect="auto"`, which stretches
   the narrow zonal extent to fill the panel width and flattens
   meridional structure.

This script reads ``snapshots_latlon.npz`` (lat-lon) or
``snapshots_native.npz`` (MPAS native Voronoi) and produces
multi-time-panel figures of SST, eta, and surface speed at a physical
aspect ratio, cropped to the domain. Shows BCI development clearly.

Usage
-----

    .venv/bin/python scripts/replot_eady_bci.py \\
        results/ocean/eady_uniform/latlon_channel/100x50/tvd_U02_Bh2.3e11_Cs0.0_600d \\
        results/ocean/eady_uniform/mpas_channel/20km/tvd_mpas_20km_U02_Bh2.3e11_Cs0.0_600d \\
        --out results/ocean/eady_uniform/bci_development

Each run directory gets a `bci_snapshots.png` inside the output dir,
named by the run basename.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

os.environ.setdefault("JAX_PLATFORMS", "cpu")


R_EARTH = 6371.0e3
TARGET_DAYS = [0.0, 60.0, 120.0, 240.0, 420.0, 600.0]


def _find_indices_for_days(times: np.ndarray, targets: list[float]) -> list[int]:
    """Pick snapshot indices closest to target days."""
    out = []
    for t in targets:
        out.append(int(np.argmin(np.abs(times - t))))
    # dedupe while preserving order
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def _physical_aspect(lon_range: tuple[float, float],
                     lat_range: tuple[float, float]) -> float:
    """Physical-distance aspect ratio (height/width) at domain mid-latitude."""
    dlon_deg = lon_range[1] - lon_range[0]
    dlat_deg = lat_range[1] - lat_range[0]
    lat_c = 0.5 * (lat_range[0] + lat_range[1])
    width_km = R_EARTH * np.radians(dlon_deg) * np.cos(np.radians(lat_c)) / 1e3
    height_km = R_EARTH * np.radians(dlat_deg) / 1e3
    return height_km / width_km


def _plot_latlon(run_dir: Path, out_png: Path) -> None:
    # For latlon the native output is on the (nlat+2, nlon) grid; regridded
    # file is easier to read. Go with regridded.
    npz = run_dir / "snapshots_latlon.npz"
    d = np.load(npz, allow_pickle=True)
    times = np.asarray(d["times_days"], dtype=np.float64)
    lon = np.asarray(d["lon"], dtype=np.float64)
    lat = np.asarray(d["lat"], dtype=np.float64)
    if "source_lon_range" in d.files:
        lon_rng = tuple(map(float, d["source_lon_range"]))
        lat_rng = tuple(map(float, d["source_lat_range"]))
    else:
        lon_rng = (float(lon.min()), float(lon.max()))
        lat_rng = (float(lat.min()), float(lat.max()))

    c0 = int(np.searchsorted(lon, lon_rng[0]))
    c1 = int(np.searchsorted(lon, lon_rng[1])) + 1
    r0 = int(np.searchsorted(lat, lat_rng[0]))
    r1 = int(np.searchsorted(lat, lat_rng[1])) + 1

    SST = np.asarray(d["SST"])[:, r0:r1, c0:c1]
    eta = np.asarray(d["eta"])[:, r0:r1, c0:c1]
    lm = np.asarray(d["land_mask"])[:, r0:r1, c0:c1]
    u = np.asarray(d["u_3d"])[:, r0:r1, c0:c1, 0]
    v = np.asarray(d["v_3d"])[:, r0:r1, c0:c1, 0]
    speed_sfc = np.sqrt(u * u + v * v)

    SST_m = np.where(lm > 0.5, SST, np.nan)
    eta_m = np.where(lm > 0.5, eta, np.nan)
    spd_m = np.where(lm > 0.5, speed_sfc, np.nan)

    # T anomaly = T - zonal mean(T) at each latitude row. Reveals BCI waves.
    zm = np.nanmean(SST_m, axis=2, keepdims=True)
    SST_anom = SST_m - zm

    idxs = _find_indices_for_days(times, TARGET_DAYS)

    _render_panels(
        out_png, run_dir.name, times[idxs],
        SST_m[idxs], SST_anom[idxs], eta_m[idxs], spd_m[idxs],
        lon_rng, lat_rng,
        extent=(lon_rng[0], lon_rng[1], lat_rng[0], lat_rng[1]),
    )


def _plot_mpas(run_dir: Path, out_png: Path) -> None:
    from legoesm.grids.voronoi import create_regional_voronoi_mesh
    from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig

    res_km = int(run_dir.parent.name.replace("km", ""))
    cfg = EadyUniformConfig()
    mesh = create_regional_voronoi_mesh(
        (cfg.lon_west, cfg.lon_east), (cfg.lat_south, cfg.lat_north),
        resolution_km=res_km, periodic_x=True,
    )
    lat_cell = np.degrees(np.asarray(mesh.latCell, dtype=np.float64))
    lon_cell = np.degrees(np.asarray(mesh.lonCell, dtype=np.float64))

    npz = run_dir / "snapshots_native.npz"
    d = np.load(npz, allow_pickle=True)
    times = np.asarray(d["times_days"], dtype=np.float64)
    SST = np.asarray(d["SST"])
    eta = np.asarray(d["eta"])
    speed_sfc = np.asarray(d["speed_3d"])[:, :, 0]
    lm = np.asarray(d["land_mask"])

    verts = _voronoi_polys(mesh)

    idxs = _find_indices_for_days(times, TARGET_DAYS)

    SST_m = np.where(lm > 0.5, SST, np.nan)
    eta_m = np.where(lm > 0.5, eta, np.nan)
    spd_m = np.where(lm > 0.5, speed_sfc, np.nan)

    # SST anomaly = SST - zonal mean at each latitude band. On the
    # unstructured mesh we bin by latitude, compute band means, then
    # subtract from each cell.
    n_bands = 60
    lat_edges = np.linspace(cfg.lat_south, cfg.lat_north, n_bands + 1)
    band_idx = np.clip(
        np.digitize(lat_cell, lat_edges) - 1, 0, n_bands - 1
    )
    SST_anom = np.empty_like(SST_m)
    for ti in range(SST_m.shape[0]):
        band_mean = np.full(n_bands, np.nan)
        for b in range(n_bands):
            sel = (band_idx == b) & np.isfinite(SST_m[ti])
            if sel.any():
                band_mean[b] = np.nanmean(SST_m[ti][sel])
        SST_anom[ti] = SST_m[ti] - band_mean[band_idx]

    lon_rng = (float(cfg.lon_west), float(cfg.lon_east))
    lat_rng = (float(cfg.lat_south), float(cfg.lat_north))

    _render_panels_voronoi(
        out_png, run_dir.name, times[idxs],
        SST_m[idxs], SST_anom[idxs], eta_m[idxs], spd_m[idxs],
        verts, lon_cell, lat_cell,
        lon_rng, lat_rng,
    )


def _voronoi_polys(mesh) -> list[np.ndarray]:
    """Build per-cell polygon vertex arrays (nCells,) of (k, 2) ndarrays."""
    # MPAS convention here: verticesOnCell has shape (maxEdges, nCells)
    verticesOnCell = np.asarray(mesh.verticesOnCell)  # (maxEdges, nCells)
    nEdgesOnCell = np.asarray(mesh.nEdgesOnCell)      # (nCells,)
    xVertex = np.degrees(np.asarray(mesh.lonVertex, dtype=np.float64))
    yVertex = np.degrees(np.asarray(mesh.latVertex, dtype=np.float64))

    xCell_deg = np.degrees(np.asarray(mesh.lonCell, dtype=np.float64))

    nCells = verticesOnCell.shape[1]
    polys: list[np.ndarray] = []
    for c in range(nCells):
        k = int(nEdgesOnCell[c])
        vids = verticesOnCell[:k, c]
        xv = xVertex[vids].copy()
        yv = yVertex[vids].copy()
        # Shift vertex x into ±180° window around cell center to avoid seam
        # wrap (a cell at lon=0 can reference a vertex at lon=359.9).
        dx = xv - xCell_deg[c]
        xv = xv - 360.0 * np.round(dx / 360.0)
        polys.append(np.column_stack([xv, yv]))
    return polys


def _render_panels(out_png: Path, tag: str,
                   times: np.ndarray,
                   SST: np.ndarray, SST_anom: np.ndarray,
                   eta: np.ndarray, speed: np.ndarray,
                   lon_rng: tuple[float, float], lat_rng: tuple[float, float],
                   extent: tuple[float, float, float, float]) -> None:
    """Lat-lon (imshow) rendering path."""
    _render_common(
        out_png, tag, times, [
            ("SST", SST, "RdBu_r", None, "SST [°C]"),
            ("SST − zonal mean", SST_anom, "RdBu_r", "symm", "SST′ [K]"),
            ("η", eta, "PuOr_r", "symm", "η [m]"),
            ("speed", speed, "viridis", "pos", "|u| [m/s]"),
        ],
        lon_rng, lat_rng,
        draw=lambda ax, data, vmin, vmax, cmap: ax.imshow(
            data, origin="lower", aspect="equal", cmap=cmap,
            extent=extent, vmin=vmin, vmax=vmax,
        ),
    )


def _render_panels_voronoi(out_png: Path, tag: str,
                           times: np.ndarray,
                           SST: np.ndarray, SST_anom: np.ndarray,
                           eta: np.ndarray, speed: np.ndarray,
                           polys: list[np.ndarray],
                           lon_cell: np.ndarray, lat_cell: np.ndarray,
                           lon_rng: tuple[float, float],
                           lat_rng: tuple[float, float]) -> None:
    """MPAS (PolyCollection) rendering path."""

    def _draw(ax, data, vmin, vmax, cmap):
        pc = PolyCollection(polys, array=data, cmap=cmap,
                            edgecolor="none", linewidth=0.0)
        if vmin is not None and vmax is not None:
            pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.set_xlim(lon_rng[0], lon_rng[1])
        ax.set_ylim(lat_rng[0], lat_rng[1])
        ax.set_aspect("equal")
        return pc

    _render_common(
        out_png, tag, times, [
            ("SST", SST, "RdBu_r", None, "SST [°C]"),
            ("SST − zonal mean", SST_anom, "RdBu_r", "symm", "SST′ [K]"),
            ("η", eta, "PuOr_r", "symm", "η [m]"),
            ("speed", speed, "viridis", "pos", "|u| [m/s]"),
        ],
        lon_rng, lat_rng, draw=_draw,
    )


def _render_common(out_png: Path, tag: str,
                   times: np.ndarray, rows: list[tuple],
                   lon_rng: tuple[float, float],
                   lat_rng: tuple[float, float], draw) -> None:
    n_rows = len(rows)
    n_cols = len(times)

    asp = _physical_aspect(lon_rng, lat_rng)
    panel_w = 3.0
    panel_h = panel_w * asp
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols + 1.4, panel_h * n_rows + 1.0),
        squeeze=False,
    )

    for i, (name, arr, cmap, mode, label) in enumerate(rows):
        # For the SST-anomaly row the IC perturbation at t=0 is much
        # larger than subsequent eddy amplitudes. Exclude t=0 from the
        # colour-range statistics so evolving structure is visible
        # (IC frame will saturate its colours — that's intentional).
        exclude_t0 = (mode == "symm" and "zonal mean" in name)
        src = arr[1:] if (exclude_t0 and arr.shape[0] > 1) else arr
        finite = src[np.isfinite(src)]
        if finite.size == 0:
            vmin = vmax = None
        else:
            if mode == "symm":
                v = float(np.nanpercentile(np.abs(finite), 99.0))
                vmin, vmax = -v, v
            elif mode == "pos":
                vmin, vmax = 0.0, float(np.nanpercentile(finite, 99.5))
            else:
                vmin = float(np.nanpercentile(finite, 0.5))
                vmax = float(np.nanpercentile(finite, 99.5))

        for j in range(n_cols):
            ax = axes[i, j]
            im = draw(ax, arr[j], vmin, vmax, cmap)
            ax.set_xlim(lon_rng[0], lon_rng[1])
            ax.set_ylim(lat_rng[0], lat_rng[1])
            if j == 0:
                ax.set_ylabel(f"{name}\nlat [°]")
            else:
                ax.set_yticklabels([])
            if i == n_rows - 1:
                ax.set_xlabel("lon [°]")
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(f"t = {times[j]:.0f} d", fontsize=10)

        # One colorbar per row on the right
        cbar = fig.colorbar(
            im, ax=axes[i, :].ravel().tolist(),
            fraction=0.025, pad=0.02,
        )
        cbar.set_label(label)

    fig.suptitle(tag, fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _dispatch(run_dir: Path) -> str:
    if (run_dir / "snapshots_native.npz").exists() and "mpas_channel" in str(run_dir):
        return "mpas"
    if (run_dir / "snapshots_latlon.npz").exists():
        return "latlon"
    raise FileNotFoundError(f"No recognised snapshots in {run_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="Directory for replots. Defaults to each run dir.")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        kind = _dispatch(run_dir)
        out_dir = args.out if args.out else run_dir
        out_png = out_dir / f"{run_dir.name}__bci_snapshots.png"
        print(f"{run_dir.name:60s}  ({kind})  →  {out_png}")
        if kind == "mpas":
            _plot_mpas(run_dir, out_png)
        else:
            _plot_latlon(run_dir, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
