"""Polar-cap test: 2-yr spinup with cos²(lat) A_h=2e5 + north_cap_lat=80°.

Compares directly to the existing cos²(lat) A_h=2e5 result (uncapped)
in `results/ocean/cos2lat_sweep/run_Ah2e05/restart_yr2.npz`.

Question: does closing off lat > 80° eliminate the residual high-lat
noise that cos²(lat) scaling alone leaves under-damped?

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/diagnose_polar_cap.py
"""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.bathymetry import BathymetryConfig, init_ocean_bathymetry
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_centroid_depth,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_forcings, create_eos_config, create_gm_redi_config,
)


OUTPUT_DIR = Path("results/ocean/polar_cap_test")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")
NORTH_CAP_LAT = 80.0
A_H_GLOBAL = 2.0e5


def _save_restart(state, year, output_dir):
    npz = {"step": 0, "time_days": float(year * 365.0), "grid_type": "latlon"}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        npz[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_yr{int(round(year)):d}.npz"
    np.savez_compressed(fname, **npz)


def _make_step_block(model, dt):
    def scan_body(state, _):
        return model.step(state, dt), None
    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state
    return block_fn


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_for_geom = GlobalOverturningConfig(
        use_gm_redi=True, H_max=5000.0, dz_surface=20.0,
        kappa_GM=800.0, kappa_Redi=800.0,
    )
    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=cfg_for_geom.n_levels, H_max=cfg_for_geom.H_max,
        dz_surface=cfg_for_geom.dz_surface, dz_deep=cfg_for_geom.dz_deep,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=cfg_for_geom.H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True, r_factor_max=0.2,
        north_cap_lat=NORTH_CAP_LAT,                           # <-- new
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy_jax.astype(jnp.float32)
    ocean_mask = ocean_mask_jax.astype(jnp.float32)

    physics = create_forcings("latlon", grid, cfg_for_geom)
    eos_config = create_eos_config(cfg_for_geom)
    gm_redi_cfg = create_gm_redi_config(cfg_for_geom)

    config = GlobalOverturningConfig(
        use_gm_redi=True, bottom_drag_coeff=2.5e-3, A_h=A_H_GLOBAL,
        H_max=5000.0, dz_surface=20.0, kappa_GM=800.0, kappa_Redi=800.0,
        T_surface=20.0, T_deep=2.0, T_scale_depth=1000.0,
    )
    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_surface=config.T_surface, T_deep=config.T_surface,
        S_uniform=config.S_uniform,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )
    centroid = compute_centroid_depth(jnp.zeros_like(H_bathy), H_bathy, z_coord)
    T_per_cell = config.T_deep + (config.T_surface - config.T_deep) * jnp.exp(
        -centroid / config.T_scale_depth,
    )
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, config.T_deep)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)))

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30, physics=physics,
        A_h=A_H_GLOBAL,
        A_h_lat_scaling=True,
        B_h=5.0e9, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear", eos_linear=eos_config, gm_redi=gm_redi_cfg,
        pgf_scheme="smc03",
        momentum_advection="vector_invariant",
        barotropic_solver="implicit_cn",
    )

    n_ocean = int(np.sum(np.asarray(ocean_mask)))
    n_total = int(np.asarray(ocean_mask).size)
    print(f"=== Polar cap test ===")
    print(f"  Config: cos²(lat) + A_h_global=2e5 + north_cap_lat={NORTH_CAP_LAT}°")
    print(f"  Wet cells: {n_ocean}/{n_total} ({100.0*n_ocean/n_total:.1f}%)")

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    block_fn = _make_step_block(model, 600.0)

    total_years = 2.0
    dt = 600.0
    n_steps_total = int(total_years * 365.0 * 86400 / dt)
    block_size = 1000
    n_blocks = n_steps_total // block_size
    n_remainder = n_steps_total - n_blocks * block_size

    _save_restart(state, 0, OUTPUT_DIR)
    t0 = time.time()
    s = state
    steps_done = 0
    next_year_save = 1.0
    blew_up = False

    for b in range(n_blocks):
        s = block_fn(s, block_size)
        steps_done += block_size
        yr_done = steps_done * dt / (365.0 * 86400.0)
        if yr_done >= next_year_save:
            jax.block_until_ready(s.eta.data)
            _save_restart(s, next_year_save, OUTPUT_DIR)
            next_year_save += 1.0
        if (b + 1) % max(1, n_blocks // 8) == 0:
            jax.block_until_ready(s.eta.data)
            u_max = float(jnp.max(jnp.abs(s.u.data)))
            eta_max = float(jnp.max(jnp.abs(s.eta.data)))
            finite = bool(jnp.all(jnp.isfinite(s.u.data))
                          and jnp.all(jnp.isfinite(s.T.data)))
            print(f"    Yr {yr_done:5.2f}/{total_years:.0f} | "
                  f"|η|max={eta_max:.2e} | |u|max={u_max:.3f} | finite={finite}")
            if not finite:
                blew_up = True
                break

    if not blew_up and n_remainder > 0:
        s = block_fn(s, n_remainder)
        jax.block_until_ready(s.eta.data)
    if not blew_up:
        _save_restart(s, total_years, OUTPUT_DIR)
    wall = time.time() - t0
    final_yr = steps_done * dt / (365.0 * 86400.0)
    u_max = float(jnp.max(jnp.abs(s.u.data)))
    print(f"  Done in {wall:.0f}s. Final |u|max = {u_max:.3f} m/s at yr {final_yr:.2f}")

    with open(OUTPUT_DIR / "run.log", "w") as f:
        f.write(f"polar cap test, cos²(lat) + A_h_global={A_H_GLOBAL}, "
                f"north_cap_lat={NORTH_CAP_LAT}\n")
        f.write(f"final_yr = {final_yr}, final_umax = {u_max}, blew_up = {blew_up}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")

    # ---- Comparison plots ----
    print("\n=== Comparison vs uncapped cos²(lat) A_h=2e5 ===")
    H_for_contour = np.where(np.asarray(ocean_mask) > 0.5,
                              np.asarray(H_bathy), np.nan)
    isobath_levels = [200, 1000, 2000, 3000, 4000]
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    runs = [
        ("cos² A_h=2e5 (uncapped)",
         "results/ocean/cos2lat_sweep/run_Ah2e05/restart_yr2.npz"),
        (f"cos² A_h=2e5 + north_cap={NORTH_CAP_LAT}°",
         str(OUTPUT_DIR / f"restart_yr{int(round(total_years)):d}.npz")),
    ]

    fields = {}
    for label, path in runs:
        if not Path(path).exists():
            print(f"  Missing: {path}"); continue
        d = np.load(path, allow_pickle=False)
        u = d["u"]; v = d["v"]; mask = d["land_mask"]
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        ocean = mask > 0.5
        fields[label] = {
            "u": np.where(ocean, u_c, np.nan),
            "v": np.where(ocean, v_c, np.nan),
            "speed": np.where(ocean, np.sqrt(u_c ** 2 + v_c ** 2), np.nan),
        }

    # Speed comparison
    speed_p95 = max(np.nanpercentile(d["speed"], 95) for d in fields.values())
    fig, axes = plt.subplots(len(fields), 1, figsize=(11, 3 * len(fields)),
                              sharex=True, sharey=True)
    if len(fields) == 1:
        axes = [axes]
    for i, (label, d) in enumerate(fields.items()):
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, d["speed"], cmap="magma",
                            vmin=0, vmax=speed_p95, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="cyan", linewidths=0.4, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label="Surface speed (m/s)" if i == 0 else "")
        max_s = float(np.nanmax(d["speed"]))
        p95 = float(np.nanpercentile(d["speed"], 95))
        ax.set_title(f"{label}  |  yr 2  |  max = {max_s:.2f}, p95 = {p95:.2f}",
                      fontsize=10)
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle("Polar cap test — surface speed at yr 2", y=1.005, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "polar_cap_speed_comparison.png",
                 dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'polar_cap_speed_comparison.png'}")

    # u (zonal) comparison
    uv_max = max(np.nanpercentile(np.abs(d["u"]), 98) for d in fields.values())
    fig, axes = plt.subplots(len(fields), 1, figsize=(11, 3 * len(fields)),
                              sharex=True, sharey=True)
    if len(fields) == 1:
        axes = [axes]
    for i, (label, d) in enumerate(fields.items()):
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, d["u"], cmap="RdBu_r",
                            vmin=-uv_max, vmax=uv_max, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="k", linewidths=0.3, alpha=0.5)
        plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label="u (m/s)" if i == 0 else "")
        ax.set_title(f"{label}  |  yr 2", fontsize=10)
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle("Polar cap test — surface u (zonal)", y=1.005, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "polar_cap_u_comparison.png",
                 dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'polar_cap_u_comparison.png'}")

    # Numbers
    print(f"\n  {'config':<40} {'max|u|':>8} {'p95':>6}")
    for label, d in fields.items():
        max_u = float(np.nanmax(np.abs(d["u"])))
        p95 = float(np.nanpercentile(d["speed"], 95))
        print(f"  {label:<40} {max_u:>8.3f} {p95:>6.3f}")


if __name__ == "__main__":
    main()
