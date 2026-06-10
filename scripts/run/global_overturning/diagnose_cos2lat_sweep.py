"""cos²(lat) A_h scaling sweep on the realistic-geometry GO config.

Three 2-yr spinups with `A_h_lat_scaling=True` (the new flag added to
`LatLonCGridOceanConfig`), at A_h_global ∈ {1e5, 2e5, 5e5}.  Tests
whether the cos²(lat) scaling:
  (a) prevents the high-lat blow-up that killed A_h=2e5 unscaled
  (b) damps the basin-spanning N-S band envelope
  (c) reduces the residual equatorial 2Δy mode

All other params identical to ``run_global_overturning_realistic_geometry.py``
(SMC03 PGF, AL81 momentum advection, drag=2.5e-3, B_h=5e9, GM/Redi).

Outputs:
  results/ocean/cos2lat_sweep/
    run_Ah1e05/  restart_yr{0,1,2}.npz, run.log
    run_Ah2e05/
    run_Ah5e05/
    comparison_speed_yr2.png
    comparison_zonal_mean.png
    u_v_components_yr2.png

Usage:
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/diagnose_cos2lat_sweep.py
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
    GlobalOverturningConfig,
    create_forcings,
    create_eos_config,
    create_gm_redi_config,
)


OUTPUT_DIR = Path("results/ocean/cos2lat_sweep")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")
A_H_VALUES = [1.0e5, 2.0e5, 5.0e5]


def _save_restart(state, year, output_dir):
    npz = {"step": 0, "time_days": float(year * 365.0),
           "grid_type": "latlon"}
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


def _build_initial_state(grid, z_coord_base, H_bathy, ocean_mask, config):
    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=config.T_water_init_C, T_deep=config.T_water_init_C,
        S_uniform=config.S_uniform,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )
    centroid = compute_centroid_depth(jnp.zeros_like(H_bathy), H_bathy, z_coord)
    T_per_cell = config.T_deep_C + (config.T_water_init_C - config.T_deep_C) * jnp.exp(
        -centroid / config.T_scale_depth,
    )
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, config.T_deep_C)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
    )
    return state, z_coord


def _run_one_ah(A_h, grid, z_coord_base, H_bathy, ocean_mask,
                 gm_redi_cfg, eos_config, physics, total_years, dt, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"=== cos²(lat) sweep: A_h_global = {A_h:.1e} m^2/s ===")
    print(f"{'='*60}")

    config = GlobalOverturningConfig(
        use_gm_redi=True, bottom_drag_coeff=2.5e-3, A_h=A_h,
        H_max=5000.0, dz_surface=20.0, kappa_GM=800.0, kappa_Redi=800.0,
        T_water_init_C=20.0, T_deep_C=2.0, T_scale_depth=1000.0,
    )
    state, z_coord = _build_initial_state(grid, z_coord_base, H_bathy, ocean_mask, config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30, physics=physics,
        A_h=A_h,
        A_h_lat_scaling=True,                   # <-- the new flag
        B_h=5.0e9,
        A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear", eos_linear=eos_config, gm_redi=gm_redi_cfg,
        pgf_scheme="smc03",
        momentum_advection="vector_invariant",
        barotropic_solver="implicit_cn",
    )

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    block_fn = _make_step_block(model, dt)

    n_steps_total = int(total_years * 365.0 * 86400 / dt)
    block_size = 1000
    n_blocks = n_steps_total // block_size
    n_remainder = n_steps_total - n_blocks * block_size

    print(f"  A_h={A_h:.1e} (cos²(lat) scaled)")
    print(f"  dt={dt}s, n_steps={n_steps_total:,} ({total_years} yr)")
    _save_restart(state, 0, output_dir)

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
            _save_restart(s, next_year_save, output_dir)
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
                print("    BLEW UP — aborting this A_h")
                blew_up = True
                break

    if not blew_up and n_remainder > 0:
        s = block_fn(s, n_remainder)
        jax.block_until_ready(s.eta.data)

    final_yr = steps_done * dt / (365.0 * 86400.0)
    if not blew_up:
        _save_restart(s, total_years, output_dir)
    wall = time.time() - t0
    u_max = float(jnp.max(jnp.abs(s.u.data)))
    print(f"  Done in {wall:.0f}s. Final |u|max = {u_max:.3f} m/s at yr {final_yr:.2f}")

    with open(output_dir / "run.log", "w") as f:
        f.write(f"cos²(lat) sweep, A_h_global = {A_h}\n")
        f.write(f"A_h_lat_scaling = True\n")
        f.write(f"total_years = {total_years}, final_yr = {final_yr}\n")
        f.write(f"final_umax = {u_max}, blew_up = {blew_up}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")
    return s, blew_up


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
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy_jax.astype(jnp.float32)
    ocean_mask = ocean_mask_jax.astype(jnp.float32)

    physics = create_forcings("latlon", grid, cfg_for_geom)
    eos_config = create_eos_config(cfg_for_geom)
    gm_redi_cfg = create_gm_redi_config(cfg_for_geom)

    print(f"=== cos²(lat) A_h sweep ===")
    print(f"  Grid: 36×72 (5°), A_h_lat_scaling=True")
    print(f"  Sweep: A_h_global ∈ {A_H_VALUES}")

    sweep_results = {}
    for A_h in A_H_VALUES:
        run_dir = OUTPUT_DIR / f"run_Ah{A_h:.0e}".replace("+", "")
        s, blew_up = _run_one_ah(
            A_h, grid, z_coord_base, H_bathy, ocean_mask,
            gm_redi_cfg, eos_config, physics, total_years=2.0, dt=600.0,
            output_dir=run_dir,
        )
        sweep_results[A_h] = {"state": s, "blew_up": blew_up, "dir": run_dir}

    # ---- Comparison plots ----
    print("\n=== Comparison plots ===")
    H_for_contour = np.where(np.asarray(ocean_mask) > 0.5,
                              np.asarray(H_bathy), np.nan)
    isobath_levels = [200, 1000, 2000, 3000, 4000]
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    speeds, us, vs = {}, {}, {}
    for A_h in A_H_VALUES:
        s = sweep_results[A_h]["state"]
        u = np.asarray(s.u.data); v = np.asarray(s.v.data)
        mask = np.asarray(s.land_mask.data) > 0.5
        u_c = 0.5 * (u[:, :-1, 0] + u[:, 1:, 0])
        v_c = 0.5 * (v[:-1, :, 0] + v[1:, :, 0])
        speed = np.sqrt(u_c ** 2 + v_c ** 2)
        speeds[A_h] = np.where(mask, speed, np.nan)
        us[A_h] = np.where(mask, u_c, np.nan)
        vs[A_h] = np.where(mask, v_c, np.nan)

    speed_p95 = max(np.nanpercentile(speeds[A_h], 95) for A_h in A_H_VALUES
                     if not sweep_results[A_h]["blew_up"])
    speed_p95 = max(speed_p95, 0.1)

    fig, axes = plt.subplots(len(A_H_VALUES), 1,
                              figsize=(11, 3 * len(A_H_VALUES)),
                              sharex=True, sharey=True)
    if len(A_H_VALUES) == 1:
        axes = [axes]
    for i, A_h in enumerate(A_H_VALUES):
        ax = axes[i]
        im = ax.pcolormesh(lon, lat, speeds[A_h], cmap="magma",
                            vmin=0, vmax=speed_p95, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="cyan", linewidths=0.4, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label="Surface speed (m/s)" if i == 0 else "")
        max_speed = float(np.nanmax(speeds[A_h])) if not sweep_results[A_h]["blew_up"] else float("nan")
        p95 = float(np.nanpercentile(speeds[A_h], 95)) if not sweep_results[A_h]["blew_up"] else float("nan")
        nan_tag = "  ⚠ NaN" if sweep_results[A_h]["blew_up"] else ""
        ax.set_title(
            f"A_h_global = {A_h:.0e} (cos²-scaled)  |  yr 2  |  "
            f"max = {max_speed:.2f}, p95 = {p95:.2f}{nan_tag}",
            fontsize=10,
        )
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "cos²(lat) A_h sweep: surface speed at yr 2",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_speed_yr2.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'comparison_speed_yr2.png'}")

    # Zonal/meridional mean
    fig, (ax_z, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(A_H_VALUES)))
    for i, A_h in enumerate(A_H_VALUES):
        if sweep_results[A_h]["blew_up"]:
            continue
        speed = speeds[A_h]
        ax_z.plot(lat, np.nanmean(speed, axis=1), "o-",
                   color=cmap[i], ms=3, label=f"A_h={A_h:.0e}")
        ax_m.plot(lon, np.nanmean(speed, axis=0), "o-",
                   color=cmap[i], ms=3, label=f"A_h={A_h:.0e}")
    ax_z.set_xlabel("Latitude (°)"); ax_z.set_ylabel("Zonal-mean speed (m/s)")
    ax_z.set_title("cos²-scaled: zonal-mean speed(lat) — yr 2")
    ax_z.grid(alpha=0.3); ax_z.legend()
    ax_m.set_xlabel("Longitude (°)"); ax_m.set_ylabel("Meridional-mean speed (m/s)")
    ax_m.set_title("cos²-scaled: meridional-mean speed(lon) — yr 2")
    ax_m.grid(alpha=0.3); ax_m.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_zonal_mean.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'comparison_zonal_mean.png'}")

    # Summary
    print("\n=== cos²(lat) sweep summary ===")
    print(f"  {'A_h_global':>10}  {'max|u|':>8}  {'p95':>6}  {'eq.zm':>6}  {'NaN?':>5}")
    for A_h in A_H_VALUES:
        if sweep_results[A_h]["blew_up"]:
            print(f"  {A_h:>10.0e}  {'NaN':>8}  {'NaN':>6}  {'NaN':>6}  yes")
            continue
        speed = speeds[A_h]
        u_max = float(np.nanmax(np.abs(us[A_h])))
        p95 = float(np.nanpercentile(speed, 95))
        lat_eq = np.abs(lat) <= 5.0
        eq_zm = float(np.nanmean(speed[lat_eq, :]))
        print(f"  {A_h:>10.0e}  {u_max:>8.3f}  {p95:>6.3f}  {eq_zm:>6.3f}  no")


if __name__ == "__main__":
    main()
