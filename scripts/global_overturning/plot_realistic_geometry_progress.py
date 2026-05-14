"""Progress plots for the realistic-geometry GO spinup.

Reads restarts from
``results/ocean/global_overturning_realistic_geometry/`` (any number,
landed every 5 sim-yr by ``run_global_overturning_realistic_geometry.py``).

Produces 5 diagnostic plots in the same directory:

  - timeseries_progress.png:     scalar timeseries (max |u|, mean SST,
                                 mean T, mean η) — one point per restart.
  - snapshots_progress.png:      SSH / SST / surface-speed maps stacked
                                 vertically, one row per restart year.
  - T_zonal_mean_progress.png:   zonal-mean T(lat, depth) per restart year,
                                 side-by-side panels.
  - moc_progress.png:            **meridional overturning streamfunction**
                                 ψ(lat, depth) [Sv].  The headline AMOC
                                 diagnostic — does an overturning cell
                                 develop?  In which direction?
  - barotropic_streamfunction_progress.png:
                                 **barotropic streamfunction** ψ_bt(lat, lon)
                                 [Sv].  Shows wind-driven gyre patterns
                                 (subtropical, subpolar) develop.

Re-runnable: pick up new restarts as they land while the spinup is
still in flight.

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/plot_realistic_geometry_progress.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_layer_thickness,
)
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry


RUN_DIR = Path("results/ocean/global_overturning_realistic_geometry")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")
RHO_0 = constants.rho_ocean  # reference density [kg/m^3]


def _gather_restarts(d):
    pairs = []
    for p in d.glob("restart_day*.npz"):
        day = int(p.stem.removeprefix("restart_day"))
        pairs.append((day, p))
    pairs.sort()
    return pairs


def _record_diagnostics(d):
    eta = d["eta"]; T = d["T"]; u = d["u"]; v = d["v"]; mask = d["land_mask"]
    u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
    v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
    speed = np.sqrt(u_c ** 2 + v_c ** 2)
    ocean = mask > 0.5
    return {
        "max_speed": float(np.max(speed)),
        "mean_sst": float(np.mean(T[:, :, 0][ocean])) if ocean.any() else 0.0,
        "mean_T": float(np.mean(T[ocean])) if ocean.any() else 0.0,
        "T_deep": float(np.mean(T[:, :, -1][ocean])) if ocean.any() else 0.0,
        "mean_eta": float(np.mean(eta[ocean])) if ocean.any() else 0.0,
        "max_eta": float(np.max(np.abs(eta[ocean]))) if ocean.any() else 0.0,
    }


from legoesm.ocean.diagnostics_streamfunction import (
    barotropic_streamfunction as _barotropic_streamfunction,
    moc_streamfunction as _moc_streamfunction,
)


def main():
    pairs = _gather_restarts(RUN_DIR)
    if not pairs:
        print(f"No restarts found in {RUN_DIR}.")
        return
    print(f"Found {len(pairs)} restarts:")
    for day, p in pairs:
        print(f"  day {day:>6}  ({day/365:5.2f} yr)  -- {p.name}")

    # Reconstruct geometry consistent with the running script
    cfg = GlobalOverturningConfig(
        use_gm_redi=True,
        bottom_drag_coeff=2.5e-3,
        A_h=1.0e4,
        H_max=5000.0,
        dz_surface=20.0,
        kappa_GM=800.0,
        kappa_Redi=800.0,
    )
    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=cfg.n_levels, H_max=cfg.H_max,
        dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=cfg.H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True,
        r_factor_max=0.2,
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = np.asarray(H_bathy_jax)
    z_coord = create_partial_cell_coordinate(
        z_coord_base, jnp.asarray(H_bathy, dtype=jnp.float32)
    )
    h_partial_static = np.asarray(z_coord.h_partial)             # (36, 72, 20)

    z_full = -np.abs(np.asarray(z_coord.z_full_ref))             # depth, negative
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi
    lat_v = np.linspace(-90, 90, len(lat) + 1)                    # v-face latitudes

    diags = []
    snapshots = []
    for day, p in pairs:
        d = np.load(p, allow_pickle=False)
        rec = _record_diagnostics(d)
        rec["day"] = day
        diags.append(rec)
        eta = d["eta"]; T = d["T"]; u = d["u"]; v = d["v"]
        mask = d["land_mask"]
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        speed_sfc = np.sqrt(u_c ** 2 + v_c ** 2)
        ocean = mask > 0.5
        # Use the static h_partial for streamfunctions (eta is small relative
        # to h_partial in the abyss; surface wiggles don't matter for Sv).
        psi_moc = _moc_streamfunction(v, h_partial_static, eta, H_bathy, mask, grid)
        psi_bt = _barotropic_streamfunction(u, h_partial_static, mask, grid)
        snapshots.append({
            "day": day,
            "year": day / 365,
            "eta": np.where(ocean, eta, np.nan),
            "SST": np.where(ocean, T[:, :, 0], np.nan),
            "speed_sfc": np.where(ocean, speed_sfc, np.nan),
            "T_zonal_mean": np.nanmean(
                np.where(ocean[:, :, None], T, np.nan), axis=1,
            ),
            "psi_moc": psi_moc,
            "psi_bt": np.where(ocean, psi_bt, np.nan),
        })

    # ---- Timeseries ----
    days = np.array([r["day"] for r in diags])
    years = days / 365.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(years, [r["max_speed"] for r in diags], "o-", color="C0")
    axes[0].set_ylabel("Max surface speed (m/s)")
    axes[0].set_title(
        f"Realistic-geometry GO spinup ({RUN_DIR.name}) — {len(years)} restarts"
    )
    axes[0].grid(alpha=0.3)
    axes[1].plot(years, [r["mean_sst"] for r in diags], "o-", color="C3")
    axes[1].set_ylabel("Mean SST (°C)")
    axes[1].grid(alpha=0.3)
    axes[2].plot(years, [r["mean_T"] for r in diags], "o-", label="All-depth")
    axes[2].plot(years, [r["T_deep"] for r in diags], "s-", label="Bottom layer")
    axes[2].set_ylabel("Mean T (°C)")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[3].plot(years, [r["mean_eta"] for r in diags], "o-", color="C2",
                 label="mean η")
    axes[3].plot(years, [r["max_eta"] for r in diags], "^-", color="C4",
                 label="max |η|")
    axes[3].set_ylabel("η (m)")
    axes[3].set_xlabel("Sim year")
    axes[3].legend(); axes[3].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "timeseries_progress.png", dpi=130)
    plt.close()
    print(f"Saved {RUN_DIR / 'timeseries_progress.png'}")

    # ---- Snapshots: SSH / SST / surface speed ----
    n = len(snapshots)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, s in enumerate(snapshots):
        for j, (key, label, cmap, sym) in enumerate([
            ("eta", "SSH (m)", "RdBu_r", True),
            ("SST", "SST (°C)", "RdYlBu_r", False),
            ("speed_sfc", "Surface speed (m/s)", "magma", False),
        ]):
            ax = axes[i, j]
            field = s[key]
            if sym:
                vmax = float(np.nanmax(np.abs(field)))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1.0
                im = ax.pcolormesh(lon, lat, field, cmap=cmap,
                                    vmin=-vmax, vmax=vmax, shading="auto")
            else:
                im = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto")
            plt.colorbar(im, ax=ax, fraction=0.046)
            if i == 0:
                ax.set_title(label)
            ax.set_ylabel(f"Yr {s['year']:.0f}")
    plt.suptitle("Realistic-geometry GO spinup — surface evolution", y=1.0)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "snapshots_progress.png", dpi=110)
    plt.close()
    print(f"Saved {RUN_DIR / 'snapshots_progress.png'}")

    # ---- T zonal-mean ----
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    T_levels = np.arange(0, 25, 2)
    for i, s in enumerate(snapshots):
        T_zm = s["T_zonal_mean"]
        im = axes[i].pcolormesh(lat, z_full, T_zm.T, cmap="RdYlBu_r",
                                 shading="auto", vmin=0, vmax=24)
        cs = axes[i].contour(lat, z_full, T_zm.T, levels=T_levels,
                              colors="k", linewidths=0.5, alpha=0.6)
        axes[i].clabel(cs, inline=True, fontsize=6, fmt="%g")
        plt.colorbar(im, ax=axes[i], fraction=0.046,
                     label="T (°C)" if i == n - 1 else None)
        axes[i].set_title(f"Yr {s['year']:.0f}")
        axes[i].set_xlabel("Latitude (°)")
        if i == 0:
            axes[i].set_ylabel("Depth (m)")
    plt.suptitle("Zonal-mean T(lat, depth)", y=1.02)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "T_zonal_mean_progress.png", dpi=130)
    plt.close()
    print(f"Saved {RUN_DIR / 'T_zonal_mean_progress.png'}")

    # ---- MOC streamfunction ψ(lat, depth) [Sv] ----
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    psi_max = max(np.nanmax(np.abs(s["psi_moc"])) for s in snapshots)
    psi_max = max(psi_max, 1.0)
    psi_levels = np.linspace(-psi_max, psi_max, 21)
    for i, s in enumerate(snapshots):
        psi = s["psi_moc"]
        im = axes[i].contourf(lat_v, z_full, psi.T,
                               levels=psi_levels, cmap="RdBu_r", extend="both")
        cs = axes[i].contour(lat_v, z_full, psi.T, levels=psi_levels[::4],
                              colors="k", linewidths=0.4, alpha=0.6)
        axes[i].clabel(cs, inline=True, fontsize=6, fmt="%g")
        plt.colorbar(im, ax=axes[i], fraction=0.046,
                     label="ψ (Sv)" if i == n - 1 else None)
        axes[i].set_title(f"Yr {s['year']:.0f}")
        axes[i].set_xlabel("Latitude (°)")
        if i == 0:
            axes[i].set_ylabel("Depth (m)")
    plt.suptitle(
        "Meridional overturning streamfunction ψ(lat, z) [Sv] — "
        "Eulerian-mean (no GM bolus)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(RUN_DIR / "moc_progress.png", dpi=130)
    plt.close()
    print(f"Saved {RUN_DIR / 'moc_progress.png'}")

    # ---- Barotropic streamfunction (gyres) ----
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    psi_bt_max = max(np.nanmax(np.abs(s["psi_bt"])) for s in snapshots)
    psi_bt_max = max(psi_bt_max, 1.0)
    bt_levels = np.linspace(-psi_bt_max, psi_bt_max, 21)
    for i, s in enumerate(snapshots):
        psi_bt = s["psi_bt"]
        im = axes[i].contourf(lon, lat, psi_bt, levels=bt_levels,
                               cmap="RdBu_r", extend="both")
        cs = axes[i].contour(lon, lat, psi_bt, levels=bt_levels[::2],
                              colors="k", linewidths=0.4, alpha=0.6)
        plt.colorbar(im, ax=axes[i], fraction=0.025,
                     label="ψ_bt (Sv)")
        axes[i].set_title(f"Yr {s['year']:.0f}")
        axes[i].set_ylabel("Latitude (°)")
        if i == n - 1:
            axes[i].set_xlabel("Longitude (°)")
    plt.suptitle("Barotropic streamfunction ψ_bt(lat, lon) [Sv]", y=1.0)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "barotropic_streamfunction_progress.png", dpi=130)
    plt.close()
    print(f"Saved {RUN_DIR / 'barotropic_streamfunction_progress.png'}")

    # Summary table
    print("\n--- Diagnostics summary ---")
    print(f"{'Yr':>5} {'max|u|':>8} {'SST':>7} {'meanT':>7} "
          f"{'T_deep':>7} {'meanη':>11} {'max|η|':>8} "
          f"{'MOCmax':>8} {'BTmax':>8}")
    for r, s in zip(diags, snapshots):
        moc_max = float(np.nanmax(np.abs(s["psi_moc"])))
        bt_max = float(np.nanmax(np.abs(s["psi_bt"])))
        print(f"{r['day']/365:>5.1f} {r['max_speed']:>8.3f} "
              f"{r['mean_sst']:>7.2f} {r['mean_T']:>7.2f} "
              f"{r['T_deep']:>7.2f} {r['mean_eta']:>+11.3e} "
              f"{r['max_eta']:>8.3f} "
              f"{moc_max:>8.2f} {bt_max:>8.2f}")


if __name__ == "__main__":
    main()
