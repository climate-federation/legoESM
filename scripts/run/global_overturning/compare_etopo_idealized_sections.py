#!/usr/bin/env python
"""Section plots across the 4 ETOPO + idealized-forcing ocean runs.

Sections produced (each a 4-panel figure with shared colorbar):
  - Zonal-mean T(lat, depth) [global]
  - Atlantic T(lat, depth)   ~ lon  -30 +- 5 deg
  - Pacific T(lat, depth)    ~ lon -150 +- 5 deg
  - Equatorial T(lon, depth) ~ lat   0 +- 2 deg
  - Drake T(lat, depth)      ~ lon  -65 +- 3 deg, lat in [-65, -55]
  - Drake U_east(lat, depth) ~ lon  -65 +- 3 deg, lat in [-65, -55]

Per-cell bottom topography is masked using the z-star reference grid
and each run's H_bathy.  Wet cells below the seafloor become NaN before
binning, so the seafloor shows up naturally in the panel.

Usage:
    python compare_etopo_idealized_sections.py --day 3650
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from legoesm.grids.voronoi import create_voronoi_mesh  # noqa: E402
from legoesm.grids.latlon import create_mercator_grid  # noqa: E402
from legoesm.grids.tripole import create_tripole_grid  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402
from legoesm.ocean.init_mpas import reconstruct_cell_velocity  # noqa: E402

import jax.numpy as jnp  # noqa: E402

DATA_ROOT = Path.home() / "saved_legoESM_data"
COMP_ROOT = DATA_ROOT / "mpas_vs_mercator_comparison_2026_05"

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
    "jra55_3way": [
        ("MPAS ico5 (WOA-init, JRA55)", "mpas",
         DATA_ROOT / "mpas_jra55_etopo_100yr_woa_v5" / "mpas" / "ico5"),
        ("MPAS ico6 (from nudge, JRA55)", "mpas",
         DATA_ROOT / "mpas_jra55_etopo_100yr_ico6" / "mpas" / "ico6"),
        ("Tripole eORCA1 (JRA55)", "tripole",
         DATA_ROOT / "tripole_jra55_etopo_10yr_nudge"),
    ],
}


def _restart_files(rundir):
    sub = rundir / "restarts"
    if sub.is_dir():
        files = sorted(sub.glob("restart_day*.npz"))
    else:
        files = sorted(rundir.glob("restart_day*.npz"))
    if not files:
        raise FileNotFoundError(f"No restart files in {rundir}")
    return files


def _pick_restart(rundir, target_day):
    files = _restart_files(rundir)
    if target_day is None:
        return files[-1]
    def _d(f):
        return int(f.stem.split("day")[-1])
    return min(files, key=lambda f: abs(_d(f) - target_day))


def _z_star():
    return create_ocean_z_star(n_levels=20, H_max=5500.0,
                               dz_surface=20.0, dz_deep=500.0)


def _load_latlon(rundir, target_day):
    npz = np.load(_pick_restart(rundir, target_day), allow_pickle=True)
    T = np.asarray(npz["T"])                  # (nlat, nlon, nlev)
    u = np.asarray(npz["u"])                  # (nlat, nlon+1, nlev)
    H = np.asarray(npz["H_bathy"])            # (nlat, nlon)
    lm = np.asarray(npz["land_mask"]) > 0.5   # ocean=True
    nlat, nlon = T.shape[:2]
    grid = create_mercator_grid(n_lon=nlon, lat_max_deg=80.0)
    lat = np.degrees(np.asarray(grid.lat2d))
    lon = np.degrees(np.asarray(grid.lon2d))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])  # cell-centered (nlat, nlon, nlev)
    return dict(T=T.reshape(-1, T.shape[-1]),
                U=u_c.reshape(-1, T.shape[-1]),
                lat=lat.ravel(), lon=lon.ravel(),
                H=H.ravel(), ocean=lm.ravel(),
                day=float(npz["time_days"]))


def _load_tripole(rundir, target_day):
    npz = np.load(_pick_restart(rundir, target_day), allow_pickle=True)
    T = np.asarray(npz["T"])
    u = np.asarray(npz["u"])
    H = np.asarray(npz["H_bathy"])
    lm = np.asarray(npz["land_mask"]) > 0.5
    geom = create_tripole_grid(str(REPO / "data/grids/eORCA1.2_mesh_mask.nc"))
    lat = np.degrees(np.asarray(geom.lat_T))
    lon = np.degrees(np.asarray(geom.lon_T))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    u_c = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    return dict(T=T.reshape(-1, T.shape[-1]),
                U=u_c.reshape(-1, T.shape[-1]),
                lat=lat.ravel(), lon=lon.ravel(),
                H=H.ravel(), ocean=lm.ravel(),
                day=float(npz["day"]))


def _mpas_subdivision(n_cells):
    k = int(round(np.log2((n_cells - 2) / 10) / 2))
    expected = 10 * (4 ** k) + 2
    if expected != n_cells:
        raise ValueError(f"Cannot infer MPAS subdivision: {n_cells} cells")
    return k


def _load_mpas(rundir, target_day):
    npz = np.load(_pick_restart(rundir, target_day), allow_pickle=True)
    T = np.asarray(npz["T"])                  # (nCells, nlev)
    u_edge = np.asarray(npz["u"])
    H = np.asarray(npz["H_bathy"])
    lm = np.asarray(npz["land_mask"]) > 0.5
    raw_sub = int(npz.get("subdivision", -1))
    sub_level = raw_sub if raw_sub > 0 else _mpas_subdivision(T.shape[0])
    mesh = create_voronoi_mesh(sub_level)
    u_east, _ = reconstruct_cell_velocity(jnp.asarray(u_edge), mesh)
    u_east = np.asarray(u_east)
    lat = np.degrees(np.asarray(mesh.latCell))
    lon = np.degrees(np.asarray(mesh.lonCell))
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return dict(T=T, U=u_east,
                lat=lat, lon=lon, H=H, ocean=lm,
                day=float(npz["time_days"]))


def _load(run, target_day):
    label, kind, path = run
    if kind == "latlon":
        d = _load_latlon(path, target_day)
    elif kind == "mpas":
        d = _load_mpas(path, target_day)
    elif kind == "tripole":
        d = _load_tripole(path, target_day)
    else:
        raise ValueError(kind)
    d["title"] = f"{label}\nday {d['day']:.0f}, year {d['day']/365:.1f}"
    return d


def _bin_axis(field, axis_vals, ocean, wet, axis_edges):
    """Bin (n_cells, n_lev) field along a per-cell axis (lat or lon).

    Returns (n_bins, n_lev) means over ocean+wet cells, NaN if empty.
    """
    n_bins = len(axis_edges) - 1
    n_lev = field.shape[1]
    out = np.full((n_bins, n_lev), np.nan)
    for b in range(n_bins):
        in_bin = ocean & (axis_vals >= axis_edges[b]) & (axis_vals < axis_edges[b + 1])
        if not in_bin.any():
            continue
        sub = field[in_bin, :]            # (n_in, n_lev)
        wet_sub = wet[in_bin, :]
        # nanmean over cells in this bin per level, ignoring dry cells.
        masked = np.where(wet_sub, sub, np.nan)
        with np.errstate(invalid="ignore"):
            mean_k = np.nanmean(masked, axis=0)
        out[b, :] = mean_k
    return out


def _build_wet(data, depth_pos):
    """Per-cell, per-level wet mask: True where cell is below sea surface
    and above local seafloor."""
    H = data["H"][:, None]               # (n_cells, 1)
    return depth_pos[None, :] <= H        # (n_cells, n_lev)


def _xyz(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def _cell_width_deg(data):
    """Approx mean cell width in degrees on the sphere."""
    n = int(data["ocean"].sum())
    return np.degrees(2.0 * np.sqrt(np.pi / max(n, 1)))


def _lon_diff(a, b):
    """Signed shortest-arc longitude difference, in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


def _sample_along_section(
    data, target_lat, target_lon, field,
    band_axis: str, band_center: float, band_halfwidth: float,
    k: int = 6, max_dist_deg: float | None = None,
):
    """Band-restricted K-NN IDW sample along an arbitrary section.

    ``band_axis`` is ``"lat"`` or ``"lon"``. Two-stage logic:

    1. **Land/ocean classification of target points**: nearest cell on
       the full sphere (including land cells) decides whether each
       target is "over land" — those become NaN unconditionally so
       coastlines render correctly. Without this step, the K-NN
       below would happily reach across a coastline up to
       ``max_dist_deg`` (≈ 2.5 native cell widths) and paint ocean
       values onto continents — particularly bad on coarse MPAS grids.

    2. **K-NN IDW over band-restricted ocean cells** for the surviving
       (ocean) targets. ``max_dist_deg`` still serves as a backstop
       for points in narrow ocean passages.
    """
    if max_dist_deg is None:
        max_dist_deg = 2.5 * _cell_width_deg(data)
    n_lev = field.shape[1]
    n_tgt = len(target_lat)
    out = np.full((n_tgt, n_lev), np.nan)

    # ---- Stage 1: land/ocean classification (full-sphere nearest cell) ----
    all_pts = _xyz(data["lat"], data["lon"])
    tgt_pts = _xyz(np.asarray(target_lat, dtype=float),
                   np.asarray(target_lon, dtype=float))
    all_tree = cKDTree(all_pts)
    _, near_idx = all_tree.query(tgt_pts, k=1)
    target_is_ocean = data["ocean"][near_idx]
    if not target_is_ocean.any():
        return out

    # ---- Stage 2: band-restricted K-NN IDW for ocean targets ----
    if band_axis == "lat":
        in_band = data["ocean"] & (np.abs(data["lat"] - band_center)
                                   <= band_halfwidth)
    elif band_axis == "lon":
        in_band = data["ocean"] & (np.abs(_lon_diff(data["lon"], band_center))
                                   <= band_halfwidth)
    else:
        raise ValueError(band_axis)
    sub_idx = np.where(in_band)[0]
    if sub_idx.size == 0:
        return out

    sub_pts = _xyz(data["lat"][sub_idx], data["lon"][sub_idx])
    tree = cKDTree(sub_pts)
    k_eff = int(min(k, sub_idx.size))
    dists, ki = tree.query(tgt_pts, k=k_eff)
    if k_eff == 1:
        dists = dists[:, None]
        ki = ki[:, None]
    max_chord = 2.0 * np.sin(np.radians(max_dist_deg) / 2.0)
    valid = dists <= max_chord                   # (n_tgt, k)
    full_idx = sub_idx[ki]                       # (n_tgt, k)
    wet = data["wet"][full_idx, :]               # (n_tgt, k, n_lev)
    vals = field[full_idx, :]                    # (n_tgt, k, n_lev)
    w = (1.0 / (dists + 1e-12)) * valid          # (n_tgt, k)
    w3 = w[..., None] * wet                      # (n_tgt, k, n_lev)
    num = np.sum(vals * w3, axis=1)              # (n_tgt, n_lev)
    den = np.sum(w3, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled = num / den
    sampled[den == 0] = np.nan

    # Apply the land mask: keep only ocean targets.
    out[target_is_ocean, :] = sampled[target_is_ocean, :]
    return out


def _select(data, lon_range=None, lat_range=None):
    """Boolean mask of cells passing lon/lat filter, AND in ocean."""
    sel = data["ocean"].copy()
    if lon_range is not None:
        lo, hi = lon_range
        if lo <= hi:
            sel &= (data["lon"] >= lo) & (data["lon"] <= hi)
        else:
            # wrap-around
            sel &= (data["lon"] >= lo) | (data["lon"] <= hi)
    if lat_range is not None:
        lo, hi = lat_range
        sel &= (data["lat"] >= lo) & (data["lat"] <= hi)
    return sel


def _shared_range(grids, diverging=False, q=0.99):
    flat = np.concatenate([g[np.isfinite(g)].ravel() for g in grids])
    if flat.size == 0:
        return 0.0, 1.0
    if diverging:
        vmax = float(np.quantile(np.abs(flat), q))
        return -vmax, vmax
    return float(np.quantile(flat, 1 - q)), float(np.quantile(flat, q))


def _nice_levels(vmin, vmax, target_n=10):
    """Round contour levels to a human-readable step."""
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
    return np.linspace(start, stop, max(n, 2))


def _plot_sections(runs_data, depth_pos, axis_centers, sections_by_run,
                   title, units, cmap, diverging, outpath,
                   xlabel, xlim=None, ylim=None,
                   contour_step=None):
    """sections_by_run: list of (n_bins, n_lev) arrays in same order as runs."""
    vmin, vmax = _shared_range(sections_by_run, diverging)
    if contour_step is not None:
        cmin = np.ceil(vmin / contour_step) * contour_step
        cmax = np.floor(vmax / contour_step) * contour_step
        levels = np.arange(cmin, cmax + 0.5 * contour_step, contour_step)
    else:
        levels = _nice_levels(vmin, vmax)
    n = len(runs_data)
    if n <= 2:
        nrows, ncols = 1, n
    elif n <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = (n + 2) // 3, 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6.5 * ncols, 4.25 * nrows),
                             constrained_layout=True, squeeze=False)
    X, Z = np.meshgrid(axis_centers, depth_pos, indexing="ij")
    for ax, d, S in zip(axes.flat, runs_data, sections_by_run):
        Sm = np.ma.masked_invalid(S)
        im = ax.pcolormesh(X, Z, Sm, shading="auto",
                           cmap=cmap, vmin=vmin, vmax=vmax)
        # Contour overlay (skip if too few finite cells)
        if np.isfinite(S).sum() > 4 and len(levels) >= 2:
            cs = ax.contour(X, Z, Sm, levels=levels, colors="k",
                            linewidths=0.5, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
        ax.invert_yaxis()
        ax.set_facecolor("#9a9a9a")
        ax.set_title(d["title"], fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("depth (m)")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, location="right",
                        label=f"{title} [{units}]")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"  wrote {outpath}  range=[{vmin:.3g}, {vmax:.3g}]  "
          f"contours @ {levels[0]:.3g}..{levels[-1]:.3g} step "
          f"{levels[1]-levels[0]:.3g}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--day", type=float, default=3650.0,
                   help="Target simulation day (default 3650 = year 10).")
    p.add_argument("--set", dest="set_name", default="idealized",
                   choices=sorted(RUN_SETS.keys()),
                   help="Which run set to compare (default: idealized).")
    p.add_argument("--out-suffix", default="",
                   help="Optional suffix appended to the output folder "
                        "name (use to keep prior runs side-by-side).")
    args = p.parse_args()

    runs = RUN_SETS[args.set_name]
    tag = f"day{int(args.day)}"
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    outdir = DATA_ROOT / f"comparison_{args.set_name}_sections_{tag}{suffix}"
    outdir.mkdir(exist_ok=True)

    z = _z_star()
    depth_pos = -np.asarray(z.z_full_ref)
    print(f"Target day: {args.day:.0f} (year {args.day/365:.1f})")
    print(f"Depth midpoints (m): {depth_pos.round(1)}")
    print()

    runs_data = []
    for run in runs:
        print(f"Loading {run[0]} ...")
        d = _load(run, args.day)
        d["wet"] = _build_wet(d, depth_pos)
        d["_cell_deg"] = _cell_width_deg(d)
        runs_data.append(d)
        print(f"  day={d['day']:.0f}  n_cells={d['lat'].size}  "
              f"ocean={d['ocean'].sum()}  cell~{d['_cell_deg']:.2f}°  "
              f"H range=[{d['H'][d['ocean']].min():.0f}, "
              f"{d['H'][d['ocean']].max():.0f}] m")
    print()

    # Bin edges
    lat_edges = np.linspace(-90.0, 90.0, 91)        # 2-deg lat
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_edges = np.linspace(-180.0, 180.0, 181)     # 2-deg lon
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Drake passage axis: 0.5-deg lat target points.
    drake_lat_centers = np.arange(-65.5, -54.0 + 1e-9, 0.5)

    # 1-deg target points for the fixed-lon (Atlantic/Pacific) and
    # fixed-lat (equator) sections.
    section_lat_centers = np.arange(-79.5, 79.5 + 1e-9, 1.0)
    section_lon_centers = np.arange(-179.5, 179.5 + 1e-9, 1.0)

    def _sample_at_lon(d, target_lon, target_lats, field="T",
                       band_halfwidth=None):
        tlat = np.asarray(target_lats, dtype=float)
        tlon = np.full_like(tlat, target_lon)
        # Band ~ 3 native cells wide. Use grid's own coarseness so MPAS
        # gets a wider band than Mercator/tripolar.
        bw = band_halfwidth if band_halfwidth is not None \
             else max(2.0, 1.5 * d["_cell_deg"])
        return _sample_along_section(
            d, tlat, tlon, d[field],
            band_axis="lon", band_center=target_lon, band_halfwidth=bw)

    def _sample_at_lat(d, target_lat, target_lons, field="T",
                       band_halfwidth=None):
        tlon = np.asarray(target_lons, dtype=float)
        tlat = np.full_like(tlon, target_lat)
        bw = band_halfwidth if band_halfwidth is not None \
             else max(2.0, 1.5 * d["_cell_deg"])
        return _sample_along_section(
            d, tlat, tlon, d[field],
            band_axis="lat", band_center=target_lat, band_halfwidth=bw)

    def _zonal_mean_T(d):
        return _bin_axis(d["T"], d["lat"], d["ocean"], d["wet"], lat_edges)

    # ------- Figures -------
    print("Computing sections ...")

    _plot_sections(
        runs_data, depth_pos, lat_centers,
        [_zonal_mean_T(d) for d in runs_data],
        title="Zonal-mean T", units="°C", cmap="RdYlBu_r",
        diverging=False, outpath=outdir / "section_zonal_mean_T.png",
        xlabel="latitude", xlim=(-80, 80), contour_step=2.0)

    _plot_sections(
        runs_data, depth_pos, section_lat_centers,
        [_sample_at_lon(d, -30.0, section_lat_centers) for d in runs_data],
        title="Atlantic T (lon = 30°W)", units="°C", cmap="RdYlBu_r",
        diverging=False, outpath=outdir / "section_atlantic_T.png",
        xlabel="latitude", xlim=(-80, 80), contour_step=2.0)

    _plot_sections(
        runs_data, depth_pos, section_lat_centers,
        [_sample_at_lon(d, -150.0, section_lat_centers) for d in runs_data],
        title="Pacific T (lon = 150°W)", units="°C", cmap="RdYlBu_r",
        diverging=False, outpath=outdir / "section_pacific_T.png",
        xlabel="latitude", xlim=(-80, 80), contour_step=2.0)

    _plot_sections(
        runs_data, depth_pos, section_lon_centers,
        [_sample_at_lat(d, 0.0, section_lon_centers) for d in runs_data],
        title="Equatorial T (lat = 0°)", units="°C", cmap="RdYlBu_r",
        diverging=False, outpath=outdir / "section_equator_T.png",
        xlabel="longitude", contour_step=2.0)

    _plot_sections(
        runs_data, depth_pos, drake_lat_centers,
        [_sample_at_lon(d, -65.0, drake_lat_centers) for d in runs_data],
        title="Drake Passage T (lon = 65°W)", units="°C",
        cmap="RdYlBu_r", diverging=False,
        outpath=outdir / "section_drake_T.png",
        xlabel="latitude", xlim=(-66, -54), contour_step=1.0)

    _plot_sections(
        runs_data, depth_pos, drake_lat_centers,
        [_sample_at_lon(d, -65.0, drake_lat_centers, field="U")
         for d in runs_data],
        title="Drake Passage U_east (lon = 65°W)", units="m/s",
        cmap="RdBu_r", diverging=True,
        outpath=outdir / "section_drake_Ueast.png",
        xlabel="latitude", xlim=(-66, -54), contour_step=0.02)

    print(f"\nDone. Plots in {outdir}/")


if __name__ == "__main__":
    main()
