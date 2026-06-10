#!/usr/bin/env python
"""Comparative plots across the 4 ETOPO + idealized-forcing ocean runs.

Runs compared:
  1. latlon_MRC_ETOPO1_10yr  -- Mercator 1deg, A_h=1e5, 10yr
  2. mpas_METOPO1_20yr       -- MPAS ico5,  A_h=1e5, 20yr
  3. mpas_METOPO2_Ah1e4      -- MPAS ico5,  A_h=1e4, 10yr
  4. tripole_orca1_etopo_20yr_production -- Tripolar ORCA1, 20yr

Fields plotted: SSH, SST, surface speed, depth-averaged speed.
Each field is binned to a common 1deg lat-lon grid so the four
panels share a colorbar and are directly comparable.

Usage:
    python compare_etopo_idealized_4runs.py              # final restart
    python compare_etopo_idealized_4runs.py --day 3650   # near year 10

Output: ~/saved_legoESM_data/comparison_etopo_idealized_<tag>/<field>.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ocean_test_matrix.regridding import _bin_to_latlon  # noqa: E402

from legoesm.grids.voronoi import create_voronoi_mesh  # noqa: E402
from legoesm.grids.latlon import create_mercator_grid  # noqa: E402
from legoesm.grids.tripole import create_tripole_grid  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402
from legoesm.ocean.init_mpas import reconstruct_cell_velocity  # noqa: E402

import jax.numpy as jnp  # noqa: E402

DATA_ROOT = Path.home() / "saved_legoESM_data"
COMP_ROOT = DATA_ROOT / "mpas_vs_mercator_comparison_2026_05"

# (label, grid_kind, restart_path) — see _runs_for(set_name) below.
RUN_SETS: dict[str, list] = {
    "idealized": [
        ("Mercator 1deg (A_h=1e5)", "latlon",
         COMP_ROOT / "latlon_MRC_ETOPO1_10yr"),
        ("MPAS ico5 (A_h=1e5)", "mpas",
         COMP_ROOT / "mpas_METOPO1_20yr"),
        ("MPAS ico5 (A_h=1e4)", "mpas",
         COMP_ROOT / "mpas_METOPO2_Ah1e4"),
        ("Tripolar ORCA1", "tripole",
         DATA_ROOT / "tripole_orca1_etopo_20yr_production"),
    ],
    "jra55": [
        ("MPAS ico5 (WOA-init, JRA55)", "mpas",
         DATA_ROOT / "mpas_jra55_etopo_100yr_woa_v5" / "mpas" / "ico5"),
        ("MPAS ico6 (from nudge, JRA55)", "mpas",
         DATA_ROOT / "mpas_jra55_etopo_100yr_ico6" / "mpas" / "ico6"),
    ],
    # 3-way intercomparison once the tripole JRA55 spinup completes
    # (Step 2 of docs/ocean_experiments/tripole_omip_plan.md). The
    # tripole entry resolves to results/omip/tripole/eorca1/ during
    # the run, then to ~/saved_legoESM_data/tripole_jra55_etopo_10yr_nudge/
    # after archival.
    "jra55_3way": [
        ("MPAS ico5 (WOA-init, JRA55)", "mpas",
         DATA_ROOT / "mpas_jra55_etopo_100yr_woa_v5" / "mpas" / "ico5"),
        ("MPAS ico6 (from nudge, JRA55)", "mpas",
         DATA_ROOT / "mpas_jra55_etopo_100yr_ico6" / "mpas" / "ico6"),
        ("Tripole eORCA1 (JRA55)", "tripole",
         DATA_ROOT / "tripole_jra55_etopo_10yr_nudge"),
    ],
}

# Common target grid for binning.
TARGET_LAT = np.linspace(-90.0, 90.0, 181)
TARGET_LON = np.linspace(-180.0, 180.0, 361)


def _restart_files(rundir: Path) -> list[Path]:
    """Tripole keeps restarts at top level; comparison runs use restarts/."""
    sub = rundir / "restarts"
    if sub.is_dir():
        files = sorted(sub.glob("restart_day*.npz"))
    else:
        files = sorted(rundir.glob("restart_day*.npz"))
    if not files:
        raise FileNotFoundError(f"No restart files in {rundir}")
    return files


def _pick_restart(rundir: Path, target_day: float | None) -> Path:
    """Pick latest if target_day is None, else closest to target_day."""
    files = _restart_files(rundir)
    if target_day is None:
        return files[-1]
    def _day(f: Path) -> int:
        return int(f.stem.split("day")[-1])
    return min(files, key=lambda f: abs(_day(f) - target_day))


def _z_star_dz(n_levels: int = 20, H_max: float = 5500.0,
               dz_surface: float = 20.0, dz_deep: float = 500.0) -> np.ndarray:
    """Layer thicknesses for the common z-star coordinate (m)."""
    z = create_ocean_z_star(n_levels=n_levels, H_max=H_max,
                            dz_surface=dz_surface, dz_deep=dz_deep)
    zhalf = np.asarray(z.z_half_ref)
    return -np.diff(zhalf)  # positive layer thicknesses


def _load_latlon(rundir: Path, target_day: float | None) -> dict:
    npz = np.load(_pick_restart(rundir, target_day), allow_pickle=True)
    eta = np.asarray(npz["eta"])                # (n_lat, n_lon)
    T = np.asarray(npz["T"])                    # (n_lat, n_lon, nlev)
    u = np.asarray(npz["u"])                    # (n_lat, n_lon+1, nlev)
    v = np.asarray(npz["v"])                    # (n_lat+1, n_lon, nlev)
    lm = np.asarray(npz["land_mask"]) > 0.5     # ocean = True
    n_lat, n_lon = T.shape[:2]
    grid = create_mercator_grid(n_lon=n_lon, lat_max_deg=80.0)
    lat = np.degrees(np.asarray(grid.lat2d))    # (n_lat, n_lon)
    lon = np.degrees(np.asarray(grid.lon2d))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    # C-grid edge averages.
    u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_c = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    return dict(label="latlon", lat=lat.ravel(), lon=lon.ravel(),
                ocean_mask=lm.ravel(),
                eta=eta.ravel(), sst=T[..., 0].ravel(),
                u_c=u_c.reshape(-1, u_c.shape[-1]),
                v_c=v_c.reshape(-1, v_c.shape[-1]),
                day=float(npz["time_days"]))


def _load_tripole(rundir: Path, target_day: float | None) -> dict:
    npz = np.load(_pick_restart(rundir, target_day), allow_pickle=True)
    eta = np.asarray(npz["eta"])
    T = np.asarray(npz["T"])
    u = np.asarray(npz["u"])
    v = np.asarray(npz["v"])
    lm = np.asarray(npz["land_mask"]) > 0.5
    geom = create_tripole_grid(str(REPO / "data/grids/eORCA1.2_mesh_mask.nc"))
    lat = np.degrees(np.asarray(geom.lat_T))   # (n_lat, n_lon)
    lon = np.degrees(np.asarray(geom.lon_T))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    # C-grid edge averages — these are grid-aligned (i/j) components.
    u_grid = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_grid = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    # Rotate to geographic (east/north) using T-point angles, formed
    # by averaging the u-point angles to T-points. Outside the bipolar
    # cap these reduce to (cos, sin) = (1, 0).
    cos_a_u = np.asarray(geom.cos_alpha_u)     # (n_lat, n_lon+1)
    sin_a_u = np.asarray(geom.sin_alpha_u)
    cos_a_T = 0.5 * (cos_a_u[:, :-1] + cos_a_u[:, 1:])  # (n_lat, n_lon)
    sin_a_T = 0.5 * (sin_a_u[:, :-1] + sin_a_u[:, 1:])
    u_east  = u_grid * cos_a_T[..., None] - v_grid * sin_a_T[..., None]
    v_north = u_grid * sin_a_T[..., None] + v_grid * cos_a_T[..., None]
    return dict(label="tripole", lat=lat.ravel(), lon=lon.ravel(),
                ocean_mask=lm.ravel(),
                eta=eta.ravel(), sst=T[..., 0].ravel(),
                u_c=u_east.reshape(-1, u_east.shape[-1]),
                v_c=v_north.reshape(-1, v_north.shape[-1]),
                day=float(npz["day"]))


def _mpas_subdivision(n_cells: int) -> int:
    """Infer subdivision level from cell count: nCells = 10·4^k + 2."""
    k = int(round(np.log2((n_cells - 2) / 10) / 2))
    expected = 10 * (4 ** k) + 2
    if expected != n_cells:
        raise ValueError(f"Cannot infer MPAS subdivision: {n_cells} cells")
    return k


def _load_mpas(rundir: Path, target_day: float | None) -> dict:
    npz = np.load(_pick_restart(rundir, target_day), allow_pickle=True)
    eta = np.asarray(npz["eta"])
    T = np.asarray(npz["T"])
    u_edges = np.asarray(npz["u"])
    lm = np.asarray(npz["land_mask"]) > 0.5
    # Prefer stored 'subdivision' field, but infer from cell count if missing.
    raw_sub = int(npz.get("subdivision", -1))
    sub_level = raw_sub if raw_sub > 0 else _mpas_subdivision(T.shape[0])
    mesh = create_voronoi_mesh(sub_level)
    u_east, v_north = reconstruct_cell_velocity(jnp.asarray(u_edges), mesh)
    u_east = np.asarray(u_east)
    v_north = np.asarray(v_north)
    lat = np.degrees(np.asarray(mesh.latCell))
    lon = np.degrees(np.asarray(mesh.lonCell))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return dict(label="mpas", lat=lat, lon=lon,
                ocean_mask=lm,
                eta=eta, sst=T[:, 0],
                u_c=u_east, v_c=v_north,
                day=float(npz["time_days"]))


def _load(run, target_day: float | None) -> dict:
    label, kind, path = run
    if kind == "latlon":
        data = _load_latlon(path, target_day)
    elif kind == "mpas":
        data = _load_mpas(path, target_day)
    elif kind == "tripole":
        data = _load_tripole(path, target_day)
    else:
        raise ValueError(kind)
    data["title"] = f"{label}\nday {data['day']:.0f}, year {data['day']/365:.1f}"
    return data


def _derive_fields(data: dict, dz: np.ndarray) -> dict:
    """Add surface/depth-averaged speed and (u, v) component fields."""
    u = data["u_c"]   # eastward
    v = data["v_c"]   # northward
    speed_sfc = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2)
    w = dz / dz.sum()
    u_bar = np.einsum("ck,k->c", u, w)
    v_bar = np.einsum("ck,k->c", v, w)
    speed_avg = np.sqrt(u_bar ** 2 + v_bar ** 2)
    data["sfc_speed"] = speed_sfc
    data["sfc_u"] = u[:, 0]
    data["sfc_v"] = v[:, 0]
    data["depth_speed"] = speed_avg
    data["depth_u"] = u_bar
    data["depth_v"] = v_bar
    return data


def _max_dist_for(data: dict) -> float:
    """Cartesian (unit-sphere) cutoff = 2x median cell spacing.

    Prevents the KDTree from painting continents (and polar caps that
    Mercator never reached) with the nearest ocean cell. Median spacing
    is estimated from ``sqrt(4*pi*r^2 / N)``-style argument: each cell
    covers ~ 4*pi/N sterad, so its linear extent ~ 2*sqrt(pi/N) rad.
    """
    n = data["lat"].size
    cell_rad = 2.0 * np.sqrt(np.pi / n)  # angular cell width
    return float(2.0 * np.sin(0.5 * 2.5 * cell_rad))  # 2.5x to be a bit lenient


def _bin(data: dict, key: str) -> np.ndarray:
    return _bin_to_latlon(
        data[key], data["lon"], data["lat"],
        target_lat=TARGET_LAT, target_lon=TARGET_LON,
        ocean_mask=data["ocean_mask"],
        max_dist=data["_max_dist"],
    )


def _bin_ocean_mask(data: dict) -> np.ndarray:
    """Fraction-of-ocean per target pixel; used to grey out land."""
    return _bin_to_latlon(
        data["ocean_mask"].astype(np.float64),
        data["lon"], data["lat"],
        target_lat=TARGET_LAT, target_lon=TARGET_LON,
        max_dist=data["_max_dist"],
    )


def _shared_range(grids: list[np.ndarray], diverging: bool, q: float = 0.99
                  ) -> tuple[float, float]:
    """Robust shared color range across panels."""
    flat = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    if diverging:
        vmax = float(np.quantile(np.abs(flat), q))
        return -vmax, vmax
    vmin = float(np.quantile(flat, 1 - q))
    vmax = float(np.quantile(flat, q))
    return vmin, vmax


def _nice_levels(vmin, vmax, target_n=10):
    rng = vmax - vmin
    raw = rng / target_n
    mag = 10 ** np.floor(np.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if rng / step <= target_n * 1.5:
            break
    start = np.ceil(vmin / step) * step
    stop = np.floor(vmax / step) * step
    n = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, max(n, 2)), step


def _grid_for(n: int) -> tuple[int, int]:
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    return ((n + 2) // 3, 3)


def _plot_field(field_key: str, title: str, units: str, cmap: str,
                diverging: bool, runs_data: list[dict], outdir: Path,
                tag: str, contour_step: float | None = None):
    binned = [_bin(d, field_key) for d in runs_data]
    vmin, vmax = _shared_range(binned, diverging)
    if contour_step is not None:
        cmin = np.ceil(vmin / contour_step) * contour_step
        cmax = np.floor(vmax / contour_step) * contour_step
        levels = np.arange(cmin, cmax + 0.5 * contour_step, contour_step)
        step = contour_step
    else:
        levels, step = _nice_levels(vmin, vmax)
    nrows, ncols = _grid_for(len(runs_data))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 4.25 * nrows),
                             constrained_layout=True, squeeze=False)
    extent = [TARGET_LON.min(), TARGET_LON.max(),
              TARGET_LAT.min(), TARGET_LAT.max()]
    LON2D, LAT2D = np.meshgrid(TARGET_LON, TARGET_LAT)
    for ax, d, b in zip(axes.flat, runs_data, binned):
        land_pixel = ~(d["_mask_bin"] > 0.5) | ~np.isfinite(b)
        bm = np.ma.masked_where(land_pixel, b)
        im = ax.imshow(bm, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        if np.isfinite(b).sum() > 4 and len(levels) >= 2:
            cs = ax.contour(LON2D, LAT2D, bm, levels=levels, colors="k",
                            linewidths=0.4, alpha=0.6)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.set_facecolor("#d9d9d9")
        ax.set_title(d["title"], fontsize=10)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xticklabels(["180W", "90W", "0", "90E", "180E"])
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_yticklabels(["90S", "45S", "0", "45N", "90N"])
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                        label=f"{title} [{units}]")
    fig.suptitle(f"{title} — ETOPO + idealized forcing, {tag}",
                 fontsize=13, fontweight="bold")
    out = outdir / f"compare_{field_key}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}  range=[{vmin:.3g}, {vmax:.3g}]  "
          f"contour step={step:.3g}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=float, default=None,
                   help="Target simulation day (e.g. 3650 for year 10). "
                        "Default: latest restart per run.")
    p.add_argument("--set", dest="set_name", default="idealized",
                   choices=sorted(RUN_SETS.keys()),
                   help="Which run set to compare (default: idealized).")
    args = p.parse_args()

    runs = RUN_SETS[args.set_name]
    tag = "final" if args.day is None else f"day{int(args.day)}"
    outdir = DATA_ROOT / f"comparison_{args.set_name}_{tag}"
    outdir.mkdir(exist_ok=True)
    dz = _z_star_dz()
    print(f"Set: {args.set_name} ({len(runs)} runs)")
    print(f"Target: {'latest' if args.day is None else f'day {args.day:.0f}'}")
    print(f"Layer thicknesses (m): {dz}")
    print()
    runs_data = []
    for run in runs:
        label = run[0]
        print(f"Loading {label} ...")
        d = _load(run, args.day)
        _derive_fields(d, dz)
        d["_max_dist"] = _max_dist_for(d)
        d["_mask_bin"] = _bin_ocean_mask(d)
        runs_data.append(d)
        print(f"  day={d['day']:.0f}  ncells={d['lat'].size}  "
              f"|U|_sfc range [{np.nanmin(d['sfc_speed']):.3f}, "
              f"{np.nanmax(d['sfc_speed']):.3f}] m/s  "
              f"max_dist={d['_max_dist']:.4f}  "
              f"ocean_pix={(d['_mask_bin']>0.5).sum()}")
    print()
    label_tag = "final snapshot" if args.day is None else f"day {int(args.day)}"
    _plot_field("eta",         "SSH",                  "m",   "RdBu_r",   True,  runs_data, outdir, label_tag, contour_step=0.25)
    _plot_field("sst",         "SST",                  "C",   "RdYlBu_r", False, runs_data, outdir, label_tag, contour_step=2.0)
    _plot_field("sfc_speed",   "Surface |U|",          "m/s", "magma",    False, runs_data, outdir, label_tag, contour_step=0.05)
    _plot_field("sfc_u",       "Surface u (eastward)", "m/s", "RdBu_r",   True,  runs_data, outdir, label_tag, contour_step=0.1)
    _plot_field("sfc_v",       "Surface v (northward)","m/s", "RdBu_r",   True,  runs_data, outdir, label_tag, contour_step=0.05)
    _plot_field("depth_speed", "Depth-averaged |U|",   "m/s", "magma",    False, runs_data, outdir, label_tag, contour_step=0.005)
    _plot_field("depth_u",     "Depth-avg u (eastward)","m/s","RdBu_r",   True,  runs_data, outdir, label_tag, contour_step=0.01)
    _plot_field("depth_v",     "Depth-avg v (northward)","m/s","RdBu_r",  True,  runs_data, outdir, label_tag, contour_step=0.005)
    print(f"\nDone. Plots in {outdir}/")


if __name__ == "__main__":
    main()
