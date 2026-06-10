"""Stage 0b' diagnostic: A_h sweep on the realistic-geometry GO config.

Runs three short (2 sim-yr) integrations from rest, identical to
``run_global_overturning_realistic_geometry.py`` except for A_h
(Laplacian momentum viscosity).  Goal: pin down whether the basin-
spanning N-S bands in surface speed (revealed by the Stage 0a overlay
to be a 2Δy zonal-jet computational mode peaking at the equator) are
caused by under-damping.

A_h sweep:
  - 1e4   (the realistic-geometry script's current value)
  - 5e4   (intermediate)
  - 2e5   (the idealised global overturning default that worked)

Damping rate of a 2Δy mode is A_h · π² / Δy²  →  e-folding time:
  - At Δy = 555 km (5° equator):
      A_h = 1e4 → τ ≈ 35 days
      A_h = 5e4 → τ ≈ 7 days
      A_h = 2e5 → τ ≈ 1.8 days

So if 2Δy under-damping is the cause, the bands should monotonically
shrink with A_h and be largely absent at 2e5.  If they persist at 2e5
the cause is elsewhere (likely AL81 q-stencil residual on partial
cells).

Outputs:
  - results/ocean/ah_sweep/run_Ah{value}/
      - restart_yr{0,1,2}.npz
      - run.log
  - results/ocean/ah_sweep/comparison_speed_yr2.png
  - results/ocean/ah_sweep/comparison_zonal_mean.png
  - results/ocean/ah_sweep/u_v_components_yr2.png

Usage:
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/diagnose_ah_sweep.py
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


OUTPUT_DIR = Path("results/ocean/ah_sweep")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")

# Sweep values
A_H_VALUES = [1.0e4, 5.0e4, 2.0e5]


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
    print(f"    Restart saved: {fname.name}")


def _make_step_block(model, dt):
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def _build_initial_state(grid, z_coord_base, H_bathy, ocean_mask, config):
    """Same initial state as run_global_overturning_realistic_geometry.py."""
    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=config.T_water_init_C, T_deep=config.T_water_init_C,
        S_uniform=config.S_uniform,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )
    centroid = compute_centroid_depth(
        jnp.zeros_like(H_bathy), H_bathy, z_coord,
    )
    T_per_cell = config.T_deep_C + (config.T_water_init_C - config.T_deep_C) * jnp.exp(
        -centroid / config.T_scale_depth,
    )
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, config.T_deep_C)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
    )
    return state, z_coord


def _run_one_ah(A_h, grid, z_coord_base, H_bathy, ocean_mask, gm_redi_cfg,
                eos_config, physics, total_years, dt, output_dir):
    """Run a 2-yr spinup at the given A_h.  Returns final state + diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"=== A_h = {A_h:.1e} m^2/s  →  output: {output_dir} ===")
    print(f"{'='*60}")

    config = GlobalOverturningConfig(
        use_gm_redi=True,
        bottom_drag_coeff=2.5e-3,
        A_h=A_h,                           # <-- swept
        H_max=5000.0,
        dz_surface=20.0,
        kappa_GM=800.0,
        kappa_Redi=800.0,
        T_water_init_C=20.0,
        T_deep_C=2.0,
        T_scale_depth=1000.0,
    )

    state, z_coord = _build_initial_state(
        grid, z_coord_base, H_bathy, ocean_mask, config,
    )

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=A_h,                           # <-- swept
        B_h=5.0e9,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear",
        eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
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
    n_steps_per_year = int(365.0 * 86400 / dt)

    print(f"  Initial state: rest, T∈[{float(jnp.min(jnp.where(state.land_mask.data[...,None]>0.5, state.T.data, jnp.inf))):.2f},"
          f" {float(jnp.max(jnp.where(state.land_mask.data[...,None]>0.5, state.T.data, -jnp.inf))):.2f}]°C")
    print(f"  dt={dt}s, n_steps={n_steps_total:,} ({total_years} yr), "
          f"{n_blocks} blocks × {block_size} steps + {n_remainder} remainder")

    _save_restart(state, 0.0, output_dir)

    times_yr = [0.0]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]

    t0 = time.time()
    s = state
    steps_done = 0
    last_print = t0
    next_year_save = 1.0
    blew_up = False

    for b in range(n_blocks):
        s = block_fn(s, block_size)
        steps_done += block_size

        # Save restart at integer-year boundaries
        yr_done = steps_done * dt / (365.0 * 86400.0)
        if yr_done >= next_year_save:
            jax.block_until_ready(s.eta.data)
            _save_restart(s, next_year_save, output_dir)
            next_year_save += 1.0

        if (b + 1) % max(1, n_blocks // 10) == 0:
            now = time.time()
            if now - last_print > 20:
                jax.block_until_ready(s.eta.data)
                u_max = float(jnp.max(jnp.abs(s.u.data)))
                eta_max = float(jnp.max(jnp.abs(s.eta.data)))
                _wet = s.land_mask.data[..., None] > 0.5
                T_min = float(jnp.min(jnp.where(_wet, s.T.data, jnp.inf)))
                T_max = float(jnp.max(jnp.where(_wet, s.T.data, -jnp.inf)))
                finite = bool(jnp.all(jnp.isfinite(s.u.data))
                              and jnp.all(jnp.isfinite(s.T.data)))
                print(f"    Yr {yr_done:5.2f}/{total_years:.0f} | "
                      f"|η|max={eta_max:.2e} | "
                      f"T∈[{T_min:.1f},{T_max:.1f}] | "
                      f"|u|max={u_max:.3f} | "
                      f"finite={finite}")
                times_yr.append(yr_done)
                umax_history.append(u_max)
                if not finite:
                    print("    BLEW UP — aborting this A_h")
                    blew_up = True
                    break
                last_print = now

    if not blew_up and n_remainder > 0:
        s = block_fn(s, n_remainder)
        jax.block_until_ready(s.eta.data)

    # Final restart at the actual year reached
    final_yr = steps_done * dt / (365.0 * 86400.0)
    if not blew_up:
        _save_restart(s, total_years, output_dir)

    wall = time.time() - t0
    u_max = float(jnp.max(jnp.abs(s.u.data)))
    print(f"  Done in {wall:.0f}s ({wall/60:.1f} min). "
          f"Final |u|max = {u_max:.3f} m/s at yr {final_yr:.2f}.")

    # Run log
    with open(output_dir / "run.log", "w") as f:
        f.write(f"A_h sweep, A_h = {A_h}\n")
        f.write(f"total_years = {total_years}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"final_yr = {final_yr}\n")
        f.write(f"final_umax = {u_max}\n")
        f.write(f"blew_up = {blew_up}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")

    return s, blew_up


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Configuration shared across the sweep ----
    total_years = float(os.environ.get("GO_RG_AH_YEARS", "2.0"))
    dt = 600.0

    cfg_for_geom = GlobalOverturningConfig(
        use_gm_redi=True,
        H_max=5000.0,
        dz_surface=20.0,
        kappa_GM=800.0,
        kappa_Redi=800.0,
    )
    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=cfg_for_geom.n_levels, H_max=cfg_for_geom.H_max,
        dz_surface=cfg_for_geom.dz_surface, dz_deep=cfg_for_geom.dz_deep,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=cfg_for_geom.H_max, H_min=50.0,
        smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True,
        r_factor_max=0.2,
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy_jax.astype(jnp.float32)
    ocean_mask = ocean_mask_jax.astype(jnp.float32)

    # Forcings + EOS + GM/Redi (built once; reused per sweep value)
    physics = create_forcings("latlon", grid, cfg_for_geom)
    eos_config = create_eos_config(cfg_for_geom)
    gm_redi_cfg = create_gm_redi_config(cfg_for_geom)

    print(f"=== A_h sweep on realistic-geometry GO ({total_years:.1f} yr each) ===")
    print(f"  Grid: 36×72 (5°), 20 levels, H_max=5000 m")
    print(f"  PGF: smc03 + AL81 + implicit-CN; drag=2.5e-3, B_h=5e9, "
          f"K_GM=K_Redi=800")
    print(f"  Sweep: A_h ∈ {A_H_VALUES}")

    sweep_results = {}
    for A_h in A_H_VALUES:
        run_dir = OUTPUT_DIR / f"run_Ah{A_h:.0e}".replace("+", "")
        s, blew_up = _run_one_ah(
            A_h, grid, z_coord_base, H_bathy, ocean_mask,
            gm_redi_cfg, eos_config, physics, total_years, dt, run_dir,
        )
        sweep_results[A_h] = {"state": s, "blew_up": blew_up, "dir": run_dir}

    # ============================================================
    # Comparison plots
    # ============================================================
    print("\n=== Building comparison plots ===")
    H_for_contour = np.where(np.asarray(ocean_mask) > 0.5,
                             np.asarray(H_bathy), np.nan)
    isobath_levels = [200, 1000, 2000, 3000, 4000]
    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    # Plot 1: Speed at year-N for each A_h, side by side
    fig, axes = plt.subplots(len(A_H_VALUES), 1,
                              figsize=(11, 3 * len(A_H_VALUES)),
                              sharex=True, sharey=True)
    if len(A_H_VALUES) == 1:
        axes = [axes]

    speeds = {}
    us = {}
    vs = {}
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

    speed_p95 = max(np.nanpercentile(speeds[A_h], 95) for A_h in A_H_VALUES)
    speed_p95 = max(speed_p95, 0.1)

    for i, A_h in enumerate(A_H_VALUES):
        ax = axes[i]
        im = ax.pcolormesh(
            lon, lat, speeds[A_h], cmap="magma",
            vmin=0, vmax=speed_p95, shading="auto",
        )
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="cyan", linewidths=0.4, alpha=0.7)
        cb = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
        cb.set_label("Surface speed (m/s)" if i == 0 else "", fontsize=8)
        max_speed = float(np.nanmax(speeds[A_h]))
        p95 = float(np.nanpercentile(speeds[A_h], 95))
        ax.set_title(
            f"A_h = {A_h:.0e} m²/s  |  yr {total_years:.0f}  |  "
            f"max speed = {max_speed:.2f} m/s, p95 = {p95:.2f}",
            fontsize=10,
        )
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "A_h sweep: surface speed at end of 2-yr spinup (cyan = isobaths)",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_speed_yr2.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'comparison_speed_yr2.png'}")

    # Plot 2: Zonal-mean and meridional-mean speed comparison
    fig, (ax_z, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(A_H_VALUES)))
    for i, A_h in enumerate(A_H_VALUES):
        speed = speeds[A_h]
        speed_zm = np.nanmean(speed, axis=1)
        speed_mm = np.nanmean(speed, axis=0)
        ax_z.plot(lat, speed_zm, "o-", color=cmap[i], ms=3,
                   label=f"A_h={A_h:.0e}")
        ax_m.plot(lon, speed_mm, "o-", color=cmap[i], ms=3,
                   label=f"A_h={A_h:.0e}")
    ax_z.set_xlabel("Latitude (°)")
    ax_z.set_ylabel("Zonal-mean surface speed (m/s)")
    ax_z.set_title(f"Zonal-mean speed(lat)  —  yr {total_years:.0f}")
    ax_z.grid(alpha=0.3); ax_z.legend(fontsize=9)
    ax_m.set_xlabel("Longitude (°)")
    ax_m.set_ylabel("Meridional-mean surface speed (m/s)")
    ax_m.set_title(f"Meridional-mean speed(lon)  —  yr {total_years:.0f}")
    ax_m.grid(alpha=0.3); ax_m.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_zonal_mean.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'comparison_zonal_mean.png'}")

    # Plot 3: u and v components side by side at yr-N
    fig, axes = plt.subplots(len(A_H_VALUES), 2,
                              figsize=(15, 3 * len(A_H_VALUES)),
                              sharex=True, sharey=True)
    if len(A_H_VALUES) == 1:
        axes = axes[np.newaxis, :]
    uv_max = max(
        max(np.nanpercentile(np.abs(us[A_h]), 98),
            np.nanpercentile(np.abs(vs[A_h]), 98))
        for A_h in A_H_VALUES
    )
    uv_max = max(uv_max, 0.1)
    for i, A_h in enumerate(A_H_VALUES):
        for j, (field, label) in enumerate([(us[A_h], "u (zonal)"),
                                              (vs[A_h], "v (meridional)")]):
            ax = axes[i, j]
            im = ax.pcolormesh(lon, lat, field, cmap="RdBu_r",
                                vmin=-uv_max, vmax=uv_max, shading="auto")
            ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                        colors="k", linewidths=0.3, alpha=0.5)
            cb = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
            cb.set_label(f"{label} (m/s)" if i == 0 else "", fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"A_h={A_h:.0e}")
        axes[-1, 0].set_xlabel("Longitude (°)")
        axes[-1, 1].set_xlabel("Longitude (°)")
    plt.suptitle(
        f"A_h sweep: surface u (zonal) and v (meridional) at yr {total_years:.0f}",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "u_v_components_yr2.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'u_v_components_yr2.png'}")

    # ============================================================
    # Summary table
    # ============================================================
    print("\n=== A_h sweep summary ===")
    print(f"{'A_h':>10}  {'max|u|':>8}  {'p95(speed)':>10}  "
          f"{'eq.speed':>9}  {'NaN?':>5}")
    for A_h in A_H_VALUES:
        speed = speeds[A_h]
        if np.all(np.isnan(speed)):
            print(f"{A_h:>10.0e}  {'NaN':>8}  {'NaN':>10}  {'NaN':>9}  yes")
            continue
        # Equator speed: average over latitude band -5° to +5°
        lat_eq = np.abs(lat) <= 5.0
        eq_speed = float(np.nanmean(speed[lat_eq, :]))
        max_u = float(np.nanmax(np.abs(us[A_h])))
        p95 = float(np.nanpercentile(speed, 95))
        nan_flag = "yes" if sweep_results[A_h]["blew_up"] else "no"
        print(f"{A_h:>10.0e}  {max_u:>8.3f}  {p95:>10.3f}  "
              f"{eq_speed:>9.3f}  {nan_flag:>5}")


if __name__ == "__main__":
    main()
