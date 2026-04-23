"""Volumetric T-census diagnostic for advection-scheme comparison.

For each saved snapshot in an Eady uniform run, bin the total ocean volume
by temperature. A perfectly non-diffusive scheme preserves the T histogram
exactly; a diffusive scheme collapses volume from the tails toward the mean.

Usage
-----
Single run:

    python scripts/plot_T_volumetric_census.py \
        results/ocean/eady_uniform/latlon_channel/100x50/tvd_nosponge_200d

Comparison of multiple runs:

    python scripts/plot_T_volumetric_census.py \
        results/ocean/eady_uniform/latlon_channel/100x50/upwind_nosponge_200d \
        results/ocean/eady_uniform/latlon_channel/100x50/tvd_nosponge_200d \
        results/ocean/eady_uniform/latlon_channel/100x50/ppm_fct_nosponge_200d \
        results/ocean/eady_uniform/latlon_channel/100x50/som_nosponge_200d \
        --out results/ocean/eady_uniform/latlon_channel/100x50/T_census_compare

Notes
-----
Cell volumes are reconstructed from ``EadyUniformConfig`` (domain bounds,
H_max) and ``create_ocean_z_star`` (vertical levels). The z-star Jacobian
is ignored since eta/H ~ 3e-4 for this experiment — the resulting volume
error is <1 part in 3000.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Keep JAX off the GPU while running this analysis.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
from legoesm.ocean.vertical import create_ocean_z_star


R_EARTH = 6371.0e3


def _cell_volumes(cfg: EadyUniformConfig, nlat: int, nlon: int, nlev: int) -> np.ndarray:
    """Return reference cell volumes with shape (nlat+2, nlon, nlev).

    Leading dim matches the native snapshot layout: ``nlat+2`` latitude
    rows including two wall cells at the south/north boundaries; the
    middle dim is ``nlon`` periodic longitude cells; trailing is depth.
    Wall rows get the same area as their nearest interior row — the
    land_mask zeroes them out in the histogram anyway.
    """
    dlat_rad = np.radians((cfg.lat_north - cfg.lat_south) / nlat)
    dlon_rad = np.radians((cfg.lon_east - cfg.lon_west) / nlon)

    # Interior lat centers — with walls padded by dlat/2 outside domain
    lat_interior = cfg.lat_south + (np.arange(nlat) + 0.5) * (
        (cfg.lat_north - cfg.lat_south) / nlat
    )
    lat_full = np.concatenate([
        [cfg.lat_south - 0.5 * np.degrees(dlat_rad)],
        lat_interior,
        [cfg.lat_north + 0.5 * np.degrees(dlat_rad)],
    ])
    cos_lat = np.cos(np.radians(lat_full))  # (nlat+2,)

    area_col = R_EARTH**2 * cos_lat * dlat_rad * dlon_rad  # (nlat+2,)

    z_coord = create_ocean_z_star(n_levels=nlev, H_max=cfg.H_max)
    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)  # (nlev,)

    vol = np.broadcast_to(
        area_col[:, None, None] * dz[None, None, :], (nlat + 2, nlon, nlev)
    ).astype(np.float64)
    return vol


def _load_snapshots(run_dir: Path) -> dict[str, np.ndarray]:
    f = run_dir / "snapshots_native.npz"
    if not f.exists():
        raise FileNotFoundError(f"No snapshots_native.npz in {run_dir}")
    d = np.load(f, allow_pickle=True)
    return {
        "T_3d": d["T_3d"],
        "land_mask": d["land_mask"],
        "times_days": d["times_days"],
    }


def _compute_histogram(
    T_3d: np.ndarray,
    volume: np.ndarray,
    land_mask: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Return histogram of shape (nt, nbins) with units [m^3]."""
    nt = T_3d.shape[0]
    nbins = len(bin_edges) - 1
    H = np.zeros((nt, nbins), dtype=np.float64)
    # Broadcast the 3D volume over z using the (nlon+2, nlat, nlev) layout.
    # land_mask is (nt, nlon+2, nlat) — 0 at halos and land, 1 at ocean.
    for t in range(nt):
        T = T_3d[t]  # (nlon+2, nlat, nlev)
        w = volume * land_mask[t][..., None]  # (nlon+2, nlat, nlev)
        valid = w > 0
        H[t], _ = np.histogram(
            T[valid], bins=bin_edges, weights=w[valid]
        )
    return H


def _auto_bins(T_initial: np.ndarray, weights: np.ndarray, nbins: int = 60) -> np.ndarray:
    valid = weights > 0
    t0 = T_initial[valid]
    lo, hi = float(np.min(t0)), float(np.max(t0))
    pad = 0.05 * (hi - lo + 1e-9)
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def _zonal_mean(T_3d: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """Area-weighted zonal mean of T → shape (nt, nlat+2, nlev).

    Averages over the longitude axis (axis=2 of the native snapshot
    layout). The land_mask (0 at walls, 1 at ocean) weights each cell
    so wall rows contribute NaN (denom=0).
    """
    w = land_mask[..., None]  # (nt, nlat+2, nlon, 1)
    num = (T_3d * w).sum(axis=2)
    den = w.sum(axis=2)
    return np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)


def _volume_scalars(T_3d, volume, land_mask):
    """Volume-weighted <T>, <T^2>, Var(T), T_min, T_max per snapshot."""
    w = volume * land_mask[..., None]  # (nt, nx, ny, nz)
    W = w.sum(axis=(1, 2, 3))
    Tw = (T_3d * w).sum(axis=(1, 2, 3)) / W
    T2w = (T_3d**2 * w).sum(axis=(1, 2, 3)) / W
    var = T2w - Tw**2
    # Min/max over ocean cells only
    T_masked = np.where(land_mask[..., None] > 0, T_3d, np.nan)
    Tmin = np.nanmin(T_masked, axis=(1, 2, 3))
    Tmax = np.nanmax(T_masked, axis=(1, 2, 3))
    return dict(mean=Tw, meansq=T2w, var=var, Tmin=Tmin, Tmax=Tmax, V_total=W)


def process(run_dir: Path) -> dict:
    snaps = _load_snapshots(run_dir)
    T_3d = np.asarray(snaps["T_3d"], dtype=np.float64)
    land_mask = np.asarray(snaps["land_mask"], dtype=np.float64)
    times = np.asarray(snaps["times_days"], dtype=np.float64)

    # Native snapshot layout is (nt, nlat+2 walls, nlon periodic, nlev)
    nt, nlat_padded, nlon, nlev = T_3d.shape
    nlat = nlat_padded - 2

    cfg = EadyUniformConfig()
    volume = _cell_volumes(cfg, nlat=nlat, nlon=nlon, nlev=nlev)

    # Auto-bin from initial snapshot
    edges = _auto_bins(T_3d[0], volume * land_mask[0][..., None])
    H = _compute_histogram(T_3d, volume, land_mask, edges)

    scalars = _volume_scalars(T_3d, volume, land_mask)
    T_zm = _zonal_mean(T_3d, land_mask)  # (nt, nlat+2, nlev)

    # Full lat axis including wall cells (matches T_zm's leading spatial dim)
    lat_interior = cfg.lat_south + (np.arange(nlat) + 0.5) * (
        (cfg.lat_north - cfg.lat_south) / nlat
    )
    dlat_deg = (cfg.lat_north - cfg.lat_south) / nlat
    lat_centers = np.concatenate([
        [cfg.lat_south - 0.5 * dlat_deg],
        lat_interior,
        [cfg.lat_north + 0.5 * dlat_deg],
    ])
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=cfg.H_max)
    depth = -np.asarray(z_coord.z_full_ref, dtype=np.float64)

    return {
        "label": run_dir.name,
        "times": times,
        "edges": edges,
        "centers": 0.5 * (edges[:-1] + edges[1:]),
        "hist": H,
        "V_total": scalars["V_total"],
        "scalars": scalars,
        "T_zm": T_zm,
        "lat_centers": lat_centers,
        "depth": depth,
    }


def _plot_single(res: dict, out_png: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    centers = res["centers"]
    times = res["times"]
    H = res["hist"]
    # Bin width for density normalization (equal width)
    dT = centers[1] - centers[0]

    # Left: stacked histograms at each time, cooler→warmer colors
    cmap = plt.get_cmap("viridis")
    ax = axes[0]
    for i, t in enumerate(times):
        c = cmap(i / max(len(times) - 1, 1))
        ax.step(centers, H[i] / dT, where="mid", color=c,
                label=f"{t:.0f} d", lw=1.3)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Volume density  dV/dT  (m³/K)")
    ax.set_title(f"Volumetric T-census  {title_suffix}")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    # Right: anomaly H(T, t) - H(T, 0) showing mixing-induced redistribution
    anom = H - H[0:1]
    amax = float(np.abs(anom).max()) or 1.0
    im = axes[1].pcolormesh(
        centers, times, anom, cmap="RdBu_r",
        vmin=-amax, vmax=amax, shading="nearest"
    )
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Time (days)")
    axes[1].set_title("Anomaly  ΔV(T,t) = V(T,t) − V(T,0)   [m³]")
    fig.colorbar(im, ax=axes[1])

    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _redistribution_fraction(H: np.ndarray) -> np.ndarray:
    """Total-variation distance / 2 between each snapshot and H[0].

    0 means the histogram is unchanged (no mixing); 1 is maximal mixing.
    """
    H0 = H[0]
    denom = H0.sum() or 1.0
    return 0.5 * np.abs(H - H0[None]).sum(axis=1) / denom


def _plot_compare(results: list[dict], out_png: Path) -> None:
    n = len(results)
    # Row 0: per-run stacked histograms
    # Row 1: per-run anomaly maps
    # Row 2: single panel — redistribution fraction vs time for all runs
    fig = plt.figure(figsize=(4.2 * n, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.9])

    cmap = plt.get_cmap("viridis")
    # Top row: per-run stacked histograms so the shape evolution is visible
    vmax_anom = 0.0
    for j, res in enumerate(results):
        ax = fig.add_subplot(gs[0, j])
        centers = res["centers"]
        dT = centers[1] - centers[0]
        times = res["times"]
        for i, t in enumerate(times):
            ax.step(
                centers, res["hist"][i] / dT, where="mid",
                color=cmap(i / max(len(times) - 1, 1)), lw=1.0
            )
        ax.set_title(res["label"])
        ax.set_xlabel("T (°C)")
        if j == 0:
            ax.set_ylabel("dV/dT (m³/K)")
        ax.grid(alpha=0.3)
        vmax_anom = max(vmax_anom, float(np.abs(res["hist"] - res["hist"][0:1]).max()))

    # Shared color scale for anomaly row
    for j, res in enumerate(results):
        ax = fig.add_subplot(gs[1, j])
        anom = res["hist"] - res["hist"][0:1]
        pcm = ax.pcolormesh(
            res["centers"], res["times"], anom, cmap="RdBu_r",
            vmin=-vmax_anom, vmax=vmax_anom, shading="nearest"
        )
        ax.set_xlabel("T (°C)")
        if j == 0:
            ax.set_ylabel("Time (days)")
        ax.set_title("ΔV(T,t)")
        fig.colorbar(pcm, ax=ax)

    # Bottom row: single panel spanning all columns — mixing vs time
    ax = fig.add_subplot(gs[2, :])
    colors = plt.get_cmap("tab10")
    for i, res in enumerate(results):
        frac = _redistribution_fraction(res["hist"])
        ax.plot(res["times"], frac, "o-", color=colors(i),
                label=res["label"], lw=1.8, ms=5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Redistribution fraction  ½·Σ|ΔV|/V_total")
    ax.set_title("Cumulative T-histogram redistribution vs time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle("Volumetric T-census — advection scheme comparison")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_zonalmean_T_evolution(results: list[dict], out_png: Path,
                                n_times: int = 6) -> None:
    """Time series of zonal-mean T(lat, z) for each run.

    Rows = runs, cols = evenly-spaced times (including t=0 and t_final).
    Each panel shows zonal-mean T with the INITIAL isotherm contours
    overlaid as dashed gray — so deviation of current isotherms from
    dashed is slumping, while weakening of their slope is diffusion.

    Common T color scale across all panels so schemes are directly
    comparable.
    """
    n_runs = len(results)

    # Common T range across all runs and times
    T_all = np.concatenate([r["T_zm"].ravel() for r in results])
    T_all = T_all[np.isfinite(T_all)]
    vmin, vmax = float(np.nanmin(T_all)), float(np.nanmax(T_all))
    levels = np.linspace(vmin, vmax, 11)

    fig, axes = plt.subplots(
        n_runs, n_times,
        figsize=(2.6 * n_times + 0.8, 2.4 * n_runs + 0.4),
        constrained_layout=True, squeeze=False,
    )

    for i, res in enumerate(results):
        T_zm = res["T_zm"]
        times = res["times"]
        lat = res["lat_centers"]
        depth = res["depth"]
        T_init = T_zm[0]

        # Select n_times evenly spaced indices from available snapshots
        idxs = np.linspace(0, len(times) - 1, n_times).round().astype(int)
        idxs = np.unique(idxs)
        # Pad to requested n_times by repeating last index if needed
        while len(idxs) < n_times:
            idxs = np.append(idxs, idxs[-1])

        for j, ti in enumerate(idxs):
            ax = axes[i, j]
            pcm = ax.pcolormesh(
                lat, depth, T_zm[ti].T, cmap="RdYlBu_r",
                vmin=vmin, vmax=vmax, shading="nearest",
            )
            # Initial isotherms (dashed, light gray)
            ax.contour(lat, depth, T_init.T, levels=levels,
                       colors="gray", linestyles="--", linewidths=0.6)
            # Current isotherms (solid, black)
            ax.contour(lat, depth, T_zm[ti].T, levels=levels,
                       colors="black", linewidths=0.6)
            ax.invert_yaxis()
            if i == n_runs - 1:
                ax.set_xlabel("Lat (°N)")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{res['label']}\nDepth (m)", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_title(f"t = {times[ti]:.0f} d", fontsize=9)

    cbar = fig.colorbar(pcm, ax=axes, shrink=0.9, aspect=30)
    cbar.set_label("T (°C)")
    fig.suptitle(
        "Zonal-mean T(lat, z) time series — dashed = initial state",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_slumping_comparison(results: list[dict], out_png: Path,
                              stable_only: bool = True) -> None:
    """Comparison figure focused on the slumping-vs-diffusion question.

    For each scheme: zonal-mean T(lat, z) at its LAST STABLE snapshot, with
    the initial zonal-mean T overlaid as dashed contours so slumping (tilt
    of the current isotherms away from dashed) is visually distinct from
    diffusion (weakening of the current isotherms).

    Plus a bottom-row summary of the two non-diffusive invariants:
      * <T>(t)/<T>(0) drift  — scheme conservation error
      * Var(T)(t)/Var(T)(0)  — irreversible numerical mixing
    """
    n = len(results)
    fig = plt.figure(figsize=(4.2 * n, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 0.35, 0.9])

    # Common T scale across all zonal-mean plots
    T_all = np.concatenate([r["T_zm"][[0, -1]].ravel() for r in results])
    T_all = T_all[np.isfinite(T_all)]
    vmin, vmax = float(T_all.min()), float(T_all.max())

    for j, res in enumerate(results):
        ax = fig.add_subplot(gs[0, j])
        T_zm = res["T_zm"]
        lat = res["lat_centers"]
        depth = res["depth"]
        # Last snapshot — for schemes that blew up, this is the last stable one
        T_final = T_zm[-1]
        T_init = T_zm[0]
        pcm = ax.pcolormesh(
            lat, depth, T_final.T, cmap="RdYlBu_r",
            vmin=vmin, vmax=vmax, shading="nearest"
        )
        # Initial isotherms (dashed gray) and current isotherms (solid black)
        levels = np.linspace(vmin, vmax, 11)
        ax.contour(lat, depth, T_init.T, levels=levels,
                   colors="gray", linestyles="--", linewidths=0.8)
        ax.contour(lat, depth, T_final.T, levels=levels,
                   colors="black", linewidths=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Latitude (°N)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_title(
            f"{res['label']}\n"
            f"zonal-mean T at t={res['times'][-1]:.0f} d  "
            f"(dashed = t=0)"
        )
        if j == n - 1:
            fig.colorbar(pcm, ax=ax, label="T (°C)")

    # Middle row: per-scheme mini-scalar summary (variance / T range)
    for j, res in enumerate(results):
        ax = fig.add_subplot(gs[1, j])
        sc = res["scalars"]
        ax.plot(res["times"], sc["var"] / sc["var"][0], "-o", color="C0",
                label="Var(T)/Var(T)₀", ms=4)
        ax.plot(res["times"], (sc["Tmax"] - sc["Tmin"]) /
                (sc["Tmax"][0] - sc["Tmin"][0]), "-s", color="C3",
                label="(Tmax−Tmin)/₀", ms=4)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.5)
        ax.set_ylim(0.85, 1.02)
        ax.set_xlabel("Time (d)")
        if j == 0:
            ax.set_ylabel("Relative to t=0")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="lower left")

    # Bottom: overlay of redistribution-fraction (or equivalently, variance
    # loss) across all schemes
    ax = fig.add_subplot(gs[2, :])
    colors = plt.get_cmap("tab10")
    for i, res in enumerate(results):
        frac = _redistribution_fraction(res["hist"])
        ax.plot(res["times"], frac, "o-", color=colors(i),
                label=res["label"], lw=1.8, ms=5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Histogram redistribution ½·Σ|ΔV|/V  (= mixing)")
    ax.set_title("Cumulative mixing vs time  (zero = perfect slumping-only advection)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle(
        "Slumping vs diffusion — Eady 100×50, 20 lvl, no sponge, B_h=2.3e11"
    )
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _write_csv(res: dict, out_csv: Path) -> None:
    centers = res["centers"]
    times = res["times"]
    with open(out_csv, "w") as f:
        f.write("time_days," + ",".join(f"T={c:.3f}" for c in centers) + "\n")
        for i, t in enumerate(times):
            row = ",".join(f"{v:.6e}" for v in res["hist"][i])
            f.write(f"{t:.3f},{row}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="One or more Eady run directories.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory for comparison figure.")
    args = p.parse_args()

    results = [process(d) for d in args.run_dirs]

    for res, d in zip(results, args.run_dirs):
        out_png = d / "T_volumetric_census.png"
        _plot_single(res, out_png, title_suffix=f"({d.name})")
        _write_csv(res, d / "T_volumetric_census.csv")
        sc = res["scalars"]
        H0 = res["hist"][0]
        Hf = res["hist"][-1]
        mix_frac = 0.5 * np.abs(Hf - H0).sum() / H0.sum()
        dvar_rel = sc["var"][-1] / sc["var"][0] - 1
        dmean = sc["mean"][-1] - sc["mean"][0]
        drange_rel = ((sc["Tmax"][-1] - sc["Tmin"][-1]) /
                      (sc["Tmax"][0] - sc["Tmin"][0])) - 1
        print(
            f"{d.name:45s} | t_final={res['times'][-1]:5.0f} d "
            f"| <T> drift={dmean:+.2e} K "
            f"| Var(T) rel drift={dvar_rel:+.4f} "
            f"| (Tmax-Tmin) rel drift={drange_rel:+.4f} "
            f"| hist redistrib={mix_frac:.4f}"
        )

    if len(results) > 1:
        out_dir = args.out if args.out else args.run_dirs[0].parent
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_compare(results, out_dir / "T_volumetric_census_compare.png")
        _plot_slumping_comparison(
            results, out_dir / "slumping_vs_diffusion.png"
        )
        _plot_zonalmean_T_evolution(
            results, out_dir / "T_zonalmean_evolution.png"
        )
        print(f"Comparison figures written to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
