"""Rossby-number diagnostic for Eady uniform runs.

Computes local Ro = ζ/f from the cell-centered velocity fields saved in
``snapshots_native.npz``. The goal is to ask whether the simulated flow
sits in the balanced, adiabatic regime (Ro ≪ 1, typical of the ocean
interior) or in the ageostrophic, mixing-prone regime (Ro ~ 1 or higher,
typical of submesoscale fronts and unbalanced boundary-layer flow).

Usage
-----
    python scripts/plot_rossby_number.py \
        results/ocean/eady_uniform/latlon_channel/100x50/*_200d \
        --out results/ocean/eady_uniform/latlon_channel/100x50

Notes
-----
* Uses centered finite differences on the native (nlon+2, nlat) layout
  with periodic wrap in longitude (halo columns at [0] and [-1] reference
  columns [-2] and [1] already — they are written by the extract function).
* Spherical metric: ∂v/∂x = (R cos φ)^-1 ∂v/∂λ;  ∂u/∂y = R^-1 ∂u/∂φ.
* ζ = ∂v/∂x − ∂u/∂y.  The u tan(φ)/R metric term is neglected (error
  ~ 1% in our 18° domain at 25°N — order the grid anisotropy truncation).
* f(lat) is the local Coriolis; each cell is normalized by its own f.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from legoesm.ocean.experiments.eady_uniform import EadyUniformConfig
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.constants import Omega

R_EARTH = 6371.0e3


def _coriolis(lat_deg: np.ndarray) -> np.ndarray:
    return 2.0 * Omega * np.sin(np.radians(lat_deg))


def _vorticity_latlon(
    u_3d: np.ndarray, v_3d: np.ndarray,
    lat_deg_padded: np.ndarray, dlon_rad: float, dlat_rad: float,
) -> np.ndarray:
    """Relative vorticity on the cell-centered snapshot grid.

    Snapshot layout: (nlat+2 with walls, nlon periodic, nlev).
    ζ = ∂v/∂x − ∂u/∂y with
        ∂v/∂x = (R cos φ)^-1 ∂v/∂λ    (λ is axis 1, periodic)
        ∂u/∂y = R^-1 ∂u/∂φ             (φ is axis 0)

    Parameters
    ----------
    u_3d, v_3d : shape (nlat+2, nlon, nlev)
    lat_deg_padded : shape (nlat+2,)
        Cell-center latitudes including wall rows.
    dlon_rad, dlat_rad : floats

    Returns
    -------
    zeta : shape (nlat+2, nlon, nlev)
    """
    cos_lat = np.cos(np.radians(lat_deg_padded))  # (nlat+2,)

    # ∂v/∂λ — centered diff with periodic wrap on axis=1 (longitude)
    dv_dlam = 0.5 * (np.roll(v_3d, -1, axis=1) - np.roll(v_3d, 1, axis=1))

    # ∂u/∂φ — centered diff on axis=0 (latitude), one-sided at walls
    du_dphi = np.zeros_like(u_3d)
    du_dphi[1:-1, :, :] = 0.5 * (u_3d[2:, :, :] - u_3d[:-2, :, :])
    du_dphi[0, :, :] = u_3d[1, :, :] - u_3d[0, :, :]
    du_dphi[-1, :, :] = u_3d[-1, :, :] - u_3d[-2, :, :]

    dv_dx = dv_dlam / (R_EARTH * cos_lat[:, None, None] * dlon_rad)
    du_dy = du_dphi / (R_EARTH * dlat_rad)

    return dv_dx - du_dy


def _process_run(run_dir: Path) -> dict:
    snaps = np.load(run_dir / "snapshots_native.npz", allow_pickle=True)
    u = np.asarray(snaps["u_3d"], dtype=np.float64)
    v = np.asarray(snaps["v_3d"], dtype=np.float64)
    land_mask = np.asarray(snaps["land_mask"], dtype=np.float64)
    times = np.asarray(snaps["times_days"], dtype=np.float64)

    nt, nlat_padded, nlon, nlev = u.shape
    nlat = nlat_padded - 2

    cfg = EadyUniformConfig()
    dlat_rad = np.radians((cfg.lat_north - cfg.lat_south) / nlat)
    dlon_rad = np.radians((cfg.lon_east - cfg.lon_west) / nlon)

    lat_interior = cfg.lat_south + (np.arange(nlat) + 0.5) * (
        (cfg.lat_north - cfg.lat_south) / nlat
    )
    dlat_deg = (cfg.lat_north - cfg.lat_south) / nlat
    lat_centers = np.concatenate([
        [cfg.lat_south - 0.5 * dlat_deg],
        lat_interior,
        [cfg.lat_north + 0.5 * dlat_deg],
    ])  # (nlat+2,)
    f = _coriolis(lat_centers)  # (nlat+2,)

    z_coord = create_ocean_z_star(n_levels=nlev, H_max=cfg.H_max)
    depth = -np.asarray(z_coord.z_full_ref, dtype=np.float64)  # positive

    # Per-snapshot Ro field: (nt, nlat+2, nlon, nlev)
    Ro = np.zeros_like(u)
    for t in range(nt):
        zeta = _vorticity_latlon(u[t], v[t], lat_centers, dlon_rad, dlat_rad)
        Ro[t] = zeta / f[:, None, None]

    # Mask: ocean interior only; exclude the wall rows (first two and last
    # two lat rows) to avoid one-sided-stencil artifacts from the walls.
    mask = land_mask[..., None] > 0  # (nt, nlat+2, nlon, 1)
    mask = np.broadcast_to(mask, Ro.shape).copy()
    mask[:, :2, :, :] = False
    mask[:, -2:, :, :] = False

    # Summary stats per snapshot (surface / mid / deep)
    # Surface = level 0, Mid = approx 1000m (smallest k with depth>1000),
    # Deep = approx 3000m (smallest k with depth>3000)
    k_mid = int(np.argmax(depth > 1000.0))
    k_deep = int(np.argmax(depth > 3000.0))

    def stats_for_layer(k):
        m = mask[..., k]
        Rk = Ro[..., k]
        vals = [Rk[t][m[t]] for t in range(nt)]
        p95 = np.array([np.nanpercentile(np.abs(v), 95) for v in vals])
        p99 = np.array([np.nanpercentile(np.abs(v), 99) for v in vals])
        p50 = np.array([np.nanpercentile(np.abs(v), 50) for v in vals])
        rms = np.array([np.sqrt(np.nanmean(v**2)) for v in vals])
        mx = np.array([np.nanmax(np.abs(v)) for v in vals])
        return dict(p50=p50, p95=p95, p99=p99, rms=rms, max=mx, depth=depth[k])

    layer_stats = {
        "surface": stats_for_layer(0),
        f"mid ({depth[k_mid]:.0f}m)": stats_for_layer(k_mid),
        f"deep ({depth[k_deep]:.0f}m)": stats_for_layer(k_deep),
    }

    # Full-column aggregate stats for a general "how unbalanced" signal
    vals_all = [Ro[t][mask[t]] for t in range(nt)]
    col_stats = dict(
        p50=np.array([np.nanpercentile(np.abs(v), 50) for v in vals_all]),
        p95=np.array([np.nanpercentile(np.abs(v), 95) for v in vals_all]),
        p99=np.array([np.nanpercentile(np.abs(v), 99) for v in vals_all]),
        rms=np.array([np.sqrt(np.nanmean(v**2)) for v in vals_all]),
        max=np.array([np.nanmax(np.abs(v)) for v in vals_all]),
    )

    return {
        "label": run_dir.name,
        "times": times,
        "Ro": Ro,
        "mask": mask,
        "lat": lat_centers,
        "depth": depth,
        "layer_stats": layer_stats,
        "col_stats": col_stats,
    }


def _plot_run_summary(res: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    times = res["times"]

    # (0,0): RMS Ro per layer over time
    ax = axes[0, 0]
    for name, st in res["layer_stats"].items():
        ax.plot(times, st["rms"], "-o", label=f"{name} RMS", ms=4)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("Time (d)")
    ax.set_ylabel("|Ro|_rms")
    ax.set_title("Root-mean-square |Ro| by layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1): 95th percentile Ro per layer
    ax = axes[0, 1]
    for name, st in res["layer_stats"].items():
        ax.plot(times, st["p95"], "-o", label=f"{name} p95", ms=4)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("Time (d)")
    ax.set_ylabel("|Ro| 95th percentile")
    ax.set_title("95th-percentile |Ro| by layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0): full-column stats
    ax = axes[1, 0]
    st = res["col_stats"]
    ax.plot(times, st["p50"], "-o", label="|Ro| median", ms=4)
    ax.plot(times, st["p95"], "-s", label="|Ro| p95", ms=4)
    ax.plot(times, st["p99"], "-^", label="|Ro| p99", ms=4)
    ax.plot(times, st["max"], "-v", label="|Ro| max", ms=4)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("Time (d)")
    ax.set_ylabel("|Ro|")
    ax.set_title("Full-column |Ro| percentiles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    # (1,1): zonal-mean Ro contour (lat, depth) at final snapshot
    ax = axes[1, 1]
    Ro_final = res["Ro"][-1]  # (nlat+2, nlon, nlev)
    mask_final = res["mask"][-1]
    Ro_masked = np.where(mask_final, Ro_final, np.nan)
    Ro_zm = np.nanmean(Ro_masked, axis=1)  # zonal mean (nlat+2, nlev)
    vmax = np.nanpercentile(np.abs(Ro_zm), 95)
    pcm = ax.pcolormesh(
        res["lat"], res["depth"], Ro_zm.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"Zonal-mean Ro at t={res['times'][-1]:.0f} d")
    fig.colorbar(pcm, ax=ax, label="Ro")

    fig.suptitle(f"Rossby-number diagnostic — {res['label']}")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _plot_compare(results: list[dict], out_png: Path) -> None:
    n = len(results)
    fig = plt.figure(figsize=(4.3 * n, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[1, 0.9])

    # Row 0: zonal-mean Ro at final snapshot, per scheme
    Ro_for_scale = []
    for r in results:
        Ro_masked = np.where(r["mask"][-1], r["Ro"][-1], np.nan)
        Ro_for_scale.append(np.nanmean(Ro_masked, axis=1))
    vmax = np.nanpercentile([np.abs(v) for v in Ro_for_scale], 95)

    for j, res in enumerate(results):
        ax = fig.add_subplot(gs[0, j])
        Ro_zm = Ro_for_scale[j]
        pcm = ax.pcolormesh(
            res["lat"], res["depth"], Ro_zm.T,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest"
        )
        ax.invert_yaxis()
        ax.set_xlabel("Lat (°N)")
        if j == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_title(f"{res['label']}\nzonal-mean Ro @ t={res['times'][-1]:.0f} d")
        if j == n - 1:
            fig.colorbar(pcm, ax=ax, label="Ro")

    # Row 1: single panel — p95 of |Ro| vs time per layer, per scheme
    ax = fig.add_subplot(gs[1, :])
    colors = plt.get_cmap("tab10")
    linestyles = {"surface": "-", "mid": "--", "deep": ":"}
    for i, res in enumerate(results):
        for layer_key, ls in [("surface", "-")] + [
            (k, linestyles.get(k.split(" ")[0], "--"))
            for k in res["layer_stats"] if not k.startswith("surface")
        ]:
            if layer_key not in res["layer_stats"]:
                continue
            st = res["layer_stats"][layer_key]
            lab = f"{res['label']} {layer_key}" if layer_key == "surface" else None
            ax.plot(res["times"], st["p95"], color=colors(i), ls=ls, lw=1.5,
                    label=lab)
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    ax.axhline(0.1, color="k", lw=0.6, alpha=0.3, linestyle=":")
    ax.set_xlabel("Time (d)")
    ax.set_ylabel("|Ro| 95th percentile")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_title(
        "95th-percentile |Ro| vs time (solid=surface, dashed=~1000m, dotted=~3000m). "
        "Horizontal guides at Ro=1 (ageostrophic) and Ro=0.1 (balanced interior)."
    )
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Rossby-number comparison — Eady advection schemes")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = []
    for d in args.run_dirs:
        r = _process_run(d)
        results.append(r)
        _plot_run_summary(r, d / "rossby_number.png")
        # Print summary of final snapshot
        final_stats = {k: v["p95"][-1] for k, v in r["layer_stats"].items()}
        print(
            f"{d.name:55s} | t_final={r['times'][-1]:5.0f} d | "
            + " | ".join(f"{k} p95={v:.3f}" for k, v in final_stats.items())
        )

    if len(results) > 1:
        out_dir = args.out or args.run_dirs[0].parent
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_compare(results, out_dir / "rossby_compare.png")
        print(f"Comparison figure: {out_dir / 'rossby_compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
