"""D1 + D2 diagnostics for the realistic-geometry GO sweep follow-up.

D1: Localize the A_h=2e5 blow-up.
    Run A_h=2e5 with fine diagnostic cadence, save state every ~30 days.
    Identify (lat, lon) of |η|max at each step → distinguishes localized
    coastal partial-cell stencil bug from basin-wide barotropic mode.

D2: WENO5 momentum advection vs AL81 at A_h=5e4.
    Run a 2-yr spinup with momentum_advection="weno5" + A_h=5e4.
    Compare residual 2Δy mode amplitude against the existing
    vector_invariant (AL81) + A_h=5e4 result from results/ocean/ah_sweep/.

Outputs:
  results/ocean/ah_diagnostics/
    ah2e5_blowup/
      restart_day{xxx}.npz   (every ~30 days)
      blowup_progression.png (max|η|, max|u| vs time + spatial map)
      blowup_log.txt
    ah5e4_weno5/
      restart_yr{0,1,2}.npz
      run.log
    weno5_vs_al81.png          (side-by-side speed comparison)
    weno5_vs_al81_zonal.png    (zonal-mean curves)

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/diagnose_blowup_and_weno.py
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


OUTPUT_DIR = Path("results/ocean/ah_diagnostics")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")


def _save_restart(state, day, output_dir, prefix="restart"):
    npz = {"step": 0, "time_days": float(day), "grid_type": "latlon"}
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        npz[f] = np.asarray(obj.data)
    fname = output_dir / f"{prefix}_day{int(round(day)):04d}.npz"
    np.savez_compressed(fname, **npz)
    return fname


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
    T_per_cell = config.T_deep + (config.T_water_init_C - config.T_deep) * jnp.exp(
        -centroid / config.T_scale_depth,
    )
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, config.T_deep)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
    )
    return state, z_coord


# ============================================================================
# D1: A_h=2e5 blow-up localization
# ============================================================================

def run_d1_blowup(grid, z_coord_base, H_bathy, ocean_mask,
                   gm_redi_cfg, eos_config, physics, dt, output_dir, lat, lon):
    """Run A_h=2e5 with fine diagnostic cadence; capture blow-up location."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"=== D1: A_h=2e5 blow-up localization ===")
    print(f"{'='*60}")

    A_h = 2.0e5
    config = GlobalOverturningConfig(
        use_gm_redi=True, bottom_drag_coeff=2.5e-3, A_h=A_h,
        H_max=5000.0, dz_surface=20.0, kappa_GM=800.0, kappa_Redi=800.0,
        T_water_init_C=20.0, T_deep_C=2.0, T_scale_depth=1000.0,
    )
    state, z_coord = _build_initial_state(
        grid, z_coord_base, H_bathy, ocean_mask, config,
    )
    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30, physics=physics,
        A_h=A_h, B_h=5.0e9, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear", eos_linear=eos_config, gm_redi=gm_redi_cfg,
        pgf_scheme="smc03", momentum_advection="vector_invariant",
        barotropic_solver="implicit_cn",
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)

    # Block size: 720 steps = 5 sim-days; coarse-but-finer than the sweep
    block_size = 720
    block_days = block_size * dt / 86400.0
    max_blocks = 70                   # 350 sim-days  > known blow-up at ~277 d
    block_fn = _make_step_block(model, dt)

    print(f"  dt={dt}s, block_size={block_size} ({block_days:.1f} days/block)")
    print(f"  Total potential: {max_blocks * block_days:.0f} days")
    print()

    history = []                      # list of dicts: day, eta_max, u_max, eta_argmax, u_argmax, finite
    saved_paths = []

    s = state
    last_finite_state = state
    last_finite_day = 0.0

    print(f"  {'day':>5}  {'|η|max':>10}  {'(lat,lon)':>14}  "
          f"{'|u|max':>9}  {'(lat,lon)':>14}  {'mean|η|':>10}  finite?")
    eta0 = float(jnp.max(jnp.abs(s.eta.data)))
    u0 = float(jnp.max(jnp.abs(s.u.data)))
    history.append({
        "day": 0.0, "eta_max": eta0, "u_max": u0,
        "eta_loc": (None, None), "u_loc": (None, None),
        "mean_eta": 0.0, "finite": True,
    })
    print(f"  {0:>5d}  {eta0:>10.3e}  {'(rest)':>14}  "
          f"{u0:>9.3e}  {'(rest)':>14}  {0:>10.3e}  True")

    for b in range(max_blocks):
        s = block_fn(s, block_size)
        jax.block_until_ready(s.eta.data)
        day = (b + 1) * block_days

        eta = np.asarray(s.eta.data)
        u_full = np.asarray(s.u.data)
        v_full = np.asarray(s.v.data)
        mask = np.asarray(s.land_mask.data) > 0.5

        # Stats on ocean cells only
        eta_masked = np.where(mask, eta, 0.0)
        finite_eta = np.isfinite(eta_masked)
        any_nan = not bool(np.all(np.isfinite(eta)) and np.all(np.isfinite(u_full)))

        if not any_nan:
            eta_abs = np.abs(eta_masked)
            i_eta, j_eta = np.unravel_index(np.argmax(eta_abs), eta.shape)
            eta_max = float(eta_abs[i_eta, j_eta])
            mean_eta = float(np.mean(eta[mask]))

            u_abs_sfc = np.abs(u_full[..., 0])
            i_u, j_u = np.unravel_index(np.argmax(u_abs_sfc), u_abs_sfc.shape)
            u_max = float(u_abs_sfc[i_u, j_u])
            j_u_lon = lon[j_u % len(lon)]

            history.append({
                "day": day, "eta_max": eta_max, "u_max": u_max,
                "eta_loc": (lat[i_eta], lon[j_eta]),
                "u_loc": (lat[i_u], j_u_lon),
                "mean_eta": mean_eta, "finite": True,
            })
            print(f"  {day:>5.0f}  {eta_max:>10.3e}  "
                  f"({lat[i_eta]:>+5.1f},{lon[j_eta]:>+6.1f})  "
                  f"{u_max:>9.3e}  ({lat[i_u]:>+5.1f},{j_u_lon:>+6.1f})  "
                  f"{mean_eta:>+10.3e}  True")
            last_finite_state = s
            last_finite_day = day

            # Save every 6 blocks (~30 days)
            if (b + 1) % 6 == 0:
                p = _save_restart(s, day, output_dir)
                saved_paths.append(p)
                print(f"    ... saved {p.name}")
        else:
            history.append({
                "day": day, "eta_max": np.nan, "u_max": np.nan,
                "eta_loc": (None, None), "u_loc": (None, None),
                "mean_eta": np.nan, "finite": False,
            })
            print(f"  {day:>5.0f}  NaN — last finite at day {last_finite_day:.0f}")
            break

    # Always save the last finite state as a "pre-NaN" diagnostic
    if last_finite_day > 0:
        p = _save_restart(last_finite_state, last_finite_day, output_dir,
                          prefix="last_finite")
        saved_paths.append(p)
        print(f"  Saved last finite state: {p.name} (day {last_finite_day:.0f})")

    # ---- Plot blow-up progression ----
    days = np.array([h["day"] for h in history])
    eta_max_arr = np.array([h["eta_max"] for h in history])
    u_max_arr = np.array([h["u_max"] for h in history])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Upper-left: |η|max over time (log scale)
    ax = axes[0, 0]
    ax.semilogy(days[:-1], np.abs(eta_max_arr[:-1]) + 1e-12, "o-", color="C0")
    ax.set_xlabel("Day")
    ax.set_ylabel("max |η| (m), log scale")
    ax.set_title("Growth of barotropic disturbance")
    ax.grid(alpha=0.3, which="both")

    # Upper-right: |u|max over time (log scale)
    ax = axes[0, 1]
    ax.semilogy(days[:-1], np.abs(u_max_arr[:-1]) + 1e-12, "o-", color="C1")
    ax.set_xlabel("Day")
    ax.set_ylabel("max |u| (m/s), log scale")
    ax.set_title("Growth of velocity")
    ax.grid(alpha=0.3, which="both")

    # Lower-left: track of |η|max location over time
    ax = axes[1, 0]
    H_for_contour = np.where(ocean_mask > 0.5, np.asarray(H_bathy), np.nan)
    ax.contourf(lon, lat, H_for_contour,
                 levels=[0, 200, 1000, 2000, 3000, 4000, 5000],
                 colors=["white", "lightgray", "gray", "darkgray",
                          "dimgray", "k"], alpha=0.5)
    eta_locs = [h["eta_loc"] for h in history if h["finite"] and h["eta_loc"][0] is not None]
    if eta_locs:
        eta_lats = [e[0] for e in eta_locs]
        eta_lons = [e[1] for e in eta_locs]
        sc = ax.scatter(eta_lons, eta_lats, c=range(len(eta_locs)),
                        cmap="plasma", s=50, edgecolor="k", linewidth=0.4)
        plt.colorbar(sc, ax=ax, label="time-order index")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Track of (lat, lon) of |η|max over time")
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)

    # Lower-right: η field at the LAST finite time (pre-NaN)
    ax = axes[1, 1]
    if last_finite_day > 0:
        eta_last = np.asarray(last_finite_state.eta.data)
        mask_last = np.asarray(last_finite_state.land_mask.data) > 0.5
        eta_plot = np.where(mask_last, eta_last, np.nan)
        vmax = float(np.nanmax(np.abs(eta_plot)))
        im = ax.pcolormesh(lon, lat, eta_plot, cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=[200, 1000, 3000],
                    colors="cyan", linewidths=0.4, alpha=0.7)
        plt.colorbar(im, ax=ax, label="η (m)")
        ax.set_title(f"η at day {last_finite_day:.0f} (last finite, pre-NaN)")
    else:
        ax.set_title("No pre-NaN snapshot")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    plt.suptitle("D1: A_h=2e5 blow-up — localization", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "blowup_progression.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_dir / 'blowup_progression.png'}")

    # Log
    with open(output_dir / "blowup_log.txt", "w") as f:
        f.write(f"D1: A_h={A_h:.0e} blow-up trace\n")
        f.write(f"dt={dt}, block_size={block_size}\n\n")
        f.write(f"{'day':>5}  {'|η|max':>10}  {'eta_loc':>16}  "
                f"{'|u|max':>9}  {'u_loc':>16}  finite?\n")
        for h in history:
            eta_loc_str = (f"({h['eta_loc'][0]:>+5.1f},{h['eta_loc'][1]:>+6.1f})"
                           if h["eta_loc"][0] is not None else "rest")
            u_loc_str = (f"({h['u_loc'][0]:>+5.1f},{h['u_loc'][1]:>+6.1f})"
                         if h["u_loc"][0] is not None else "rest")
            f.write(f"{h['day']:>5.0f}  {h['eta_max']:>10.3e}  "
                    f"{eta_loc_str:>16}  {h['u_max']:>9.3e}  "
                    f"{u_loc_str:>16}  {h['finite']}\n")
    print(f"  Saved {output_dir / 'blowup_log.txt'}")

    return history, last_finite_state, last_finite_day


# ============================================================================
# D2: WENO5 vs AL81 at A_h=5e4
# ============================================================================

def run_d2_weno5(grid, z_coord_base, H_bathy, ocean_mask,
                  gm_redi_cfg, eos_config, physics, dt,
                  total_years, output_dir):
    """2-yr spinup at A_h=5e4 with momentum_advection="weno5"."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"=== D2: WENO5 momentum advection at A_h=5e4 ===")
    print(f"{'='*60}")

    A_h = 5.0e4
    config = GlobalOverturningConfig(
        use_gm_redi=True, bottom_drag_coeff=2.5e-3, A_h=A_h,
        H_max=5000.0, dz_surface=20.0, kappa_GM=800.0, kappa_Redi=800.0,
        T_water_init_C=20.0, T_deep_C=2.0, T_scale_depth=1000.0,
    )
    state, z_coord = _build_initial_state(
        grid, z_coord_base, H_bathy, ocean_mask, config,
    )
    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30, physics=physics,
        A_h=A_h, B_h=5.0e9, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear", eos_linear=eos_config, gm_redi=gm_redi_cfg,
        pgf_scheme="smc03",
        momentum_advection="weno5",                 # <-- the change
        barotropic_solver="implicit_cn",
    )
    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    block_fn = _make_step_block(model, dt)

    n_steps_total = int(total_years * 365.0 * 86400 / dt)
    block_size = 1000
    n_blocks = n_steps_total // block_size
    n_remainder = n_steps_total - n_blocks * block_size

    print(f"  A_h={A_h:.1e}, momentum_advection=weno5")
    print(f"  dt={dt}s, n_steps={n_steps_total:,} ({total_years} yr)")

    _save_restart(state, 0, output_dir, prefix="restart_yr")

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
            _save_restart(s, int(round(next_year_save * 365.0)),
                           output_dir, prefix="restart_yr")
            next_year_save += 1.0

        if (b + 1) % max(1, n_blocks // 8) == 0:
            jax.block_until_ready(s.eta.data)
            u_max = float(jnp.max(jnp.abs(s.u.data)))
            eta_max = float(jnp.max(jnp.abs(s.eta.data)))
            finite = bool(jnp.all(jnp.isfinite(s.u.data))
                          and jnp.all(jnp.isfinite(s.T.data)))
            print(f"    Yr {yr_done:5.2f}/{total_years:.0f} | "
                  f"|η|max={eta_max:.2e} | "
                  f"|u|max={u_max:.3f} | finite={finite}")
            if not finite:
                print("    BLEW UP — aborting")
                blew_up = True
                break

    if not blew_up and n_remainder > 0:
        s = block_fn(s, n_remainder)
        jax.block_until_ready(s.eta.data)

    final_yr = steps_done * dt / (365.0 * 86400.0)
    if not blew_up:
        _save_restart(s, int(round(total_years * 365.0)),
                       output_dir, prefix="restart_yr")
    wall = time.time() - t0
    u_max = float(jnp.max(jnp.abs(s.u.data)))
    print(f"  Done in {wall:.0f}s. Final |u|max = {u_max:.3f} m/s at yr {final_yr:.2f}")

    with open(output_dir / "run.log", "w") as f:
        f.write(f"D2: WENO5 momentum advection, A_h={A_h}\n")
        f.write(f"total_years={total_years}, final_yr={final_yr}\n")
        f.write(f"final_umax={u_max}, blew_up={blew_up}\n")
        f.write(f"wall_time_s={wall:.1f}\n")

    return s, blew_up


# ============================================================================
# Comparison plots: WENO5 vs AL81 at A_h=5e4
# ============================================================================

def make_d2_comparison(state_weno5, ocean_mask, H_bathy, lat, lon, output_dir):
    """Compare WENO5 result against the existing AL81 + A_h=5e4 sweep result."""
    al81_restart = Path("results/ocean/ah_sweep/run_Ah5e04/restart_yr2.npz")
    if not al81_restart.exists():
        print(f"  WARNING: {al81_restart} missing — cannot make comparison plot")
        return

    d_al81 = np.load(al81_restart, allow_pickle=False)
    u_al = d_al81["u"]; v_al = d_al81["v"]; mask_al = d_al81["land_mask"]
    u_c_al = 0.5 * (u_al[:, :-1, 0] + u_al[:, 1:, 0])
    v_c_al = 0.5 * (v_al[:-1, :, 0] + v_al[1:, :, 0])
    speed_al = np.where(mask_al > 0.5,
                        np.sqrt(u_c_al ** 2 + v_c_al ** 2), np.nan)

    u_we = np.asarray(state_weno5.u.data)
    v_we = np.asarray(state_weno5.v.data)
    mask_we = np.asarray(state_weno5.land_mask.data)
    u_c_we = 0.5 * (u_we[:, :-1, 0] + u_we[:, 1:, 0])
    v_c_we = 0.5 * (v_we[:-1, :, 0] + v_we[1:, :, 0])
    speed_we = np.where(mask_we > 0.5,
                        np.sqrt(u_c_we ** 2 + v_c_we ** 2), np.nan)

    H_for_contour = np.where(ocean_mask > 0.5, np.asarray(H_bathy), np.nan)
    isobath_levels = [200, 1000, 2000, 3000, 4000]

    # Side-by-side speed
    speed_p95 = max(np.nanpercentile(speed_al, 95),
                     np.nanpercentile(speed_we, 95), 0.1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    for ax, speed, label, max_s in [
        (axes[0], speed_al, "vector_invariant (AL81)", float(np.nanmax(speed_al))),
        (axes[1], speed_we, "weno5", float(np.nanmax(speed_we))),
    ]:
        im = ax.pcolormesh(lon, lat, speed, cmap="magma",
                            vmin=0, vmax=speed_p95, shading="auto")
        ax.contour(lon, lat, H_for_contour, levels=isobath_levels,
                    colors="cyan", linewidths=0.4, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02,
                      label="Surface speed (m/s)")
        p95 = float(np.nanpercentile(speed, 95))
        ax.set_title(
            f"{label}  |  yr 2  |  max = {max_s:.2f} m/s  |  p95 = {p95:.2f}",
            fontsize=10,
        )
        ax.set_ylabel("Latitude (°)")
    axes[-1].set_xlabel("Longitude (°)")
    plt.suptitle(
        "WENO5 vs AL81 at A_h=5e4 — surface speed at yr 2",
        y=1.005, fontsize=11,
    )
    plt.tight_layout()
    out = output_dir.parent / "weno5_vs_al81.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # Zonal-mean and meridional-mean comparison
    fig, (ax_z, ax_m) = plt.subplots(1, 2, figsize=(14, 5))
    speed_zm_al = np.nanmean(speed_al, axis=1)
    speed_zm_we = np.nanmean(speed_we, axis=1)
    speed_mm_al = np.nanmean(speed_al, axis=0)
    speed_mm_we = np.nanmean(speed_we, axis=0)
    ax_z.plot(lat, speed_zm_al, "o-", color="C0", ms=3,
               label="AL81 (vector_invariant)")
    ax_z.plot(lat, speed_zm_we, "o-", color="C3", ms=3,
               label="WENO5")
    ax_z.set_xlabel("Latitude (°)")
    ax_z.set_ylabel("Zonal-mean surface speed (m/s)")
    ax_z.set_title("Zonal-mean speed(lat) — yr 2, A_h=5e4")
    ax_z.grid(alpha=0.3); ax_z.legend()
    ax_m.plot(lon, speed_mm_al, "o-", color="C0", ms=3, label="AL81")
    ax_m.plot(lon, speed_mm_we, "o-", color="C3", ms=3, label="WENO5")
    ax_m.set_xlabel("Longitude (°)")
    ax_m.set_ylabel("Meridional-mean surface speed (m/s)")
    ax_m.set_title("Meridional-mean speed(lon) — yr 2, A_h=5e4")
    ax_m.grid(alpha=0.3); ax_m.legend()
    plt.tight_layout()
    out2 = output_dir.parent / "weno5_vs_al81_zonal.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out2}")

    # Numbers
    print("\n  --- WENO5 vs AL81 numbers ---")
    print(f"  {'metric':<30} {'AL81':>10} {'WENO5':>10}")
    for name, val_al, val_we in [
        ("max |u| (m/s)", np.nanmax(np.abs(u_c_al)), np.nanmax(np.abs(u_c_we))),
        ("max |v| (m/s)", np.nanmax(np.abs(v_c_al)), np.nanmax(np.abs(v_c_we))),
        ("p95 surface speed (m/s)", np.nanpercentile(speed_al, 95),
         np.nanpercentile(speed_we, 95)),
        ("zonal-mean speed @ -2.5°lat",
         speed_zm_al[len(lat)//2 - 1] if len(lat) % 2 else speed_zm_al[len(lat)//2],
         speed_zm_we[len(lat)//2 - 1] if len(lat) % 2 else speed_zm_we[len(lat)//2]),
        ("meridional-mean speed peak", np.nanmax(speed_mm_al),
         np.nanmax(speed_mm_we)),
    ]:
        print(f"  {name:<30} {val_al:>10.4f} {val_we:>10.4f}")


# ============================================================================
# Main
# ============================================================================

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
        depth_is_negative=True,
        r_factor_max=0.2,
    )
    H_bathy_jax, ocean_mask_jax = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy_jax.astype(jnp.float32)
    ocean_mask = ocean_mask_jax.astype(jnp.float32)
    H_bathy_np = np.asarray(H_bathy)
    ocean_mask_np = np.asarray(ocean_mask)

    physics = create_forcings("latlon", grid, cfg_for_geom)
    eos_config = create_eos_config(cfg_for_geom)
    gm_redi_cfg = create_gm_redi_config(cfg_for_geom)

    lat = np.asarray(grid.lat) * 180 / np.pi
    lon = np.asarray(grid.lon) * 180 / np.pi

    dt = 600.0

    # ---- D1 ----
    d1_dir = OUTPUT_DIR / "ah2e5_blowup"
    history, last_finite_state, last_finite_day = run_d1_blowup(
        grid, z_coord_base, H_bathy, ocean_mask,
        gm_redi_cfg, eos_config, physics, dt, d1_dir, lat, lon,
    )

    # ---- D2 ----
    d2_dir = OUTPUT_DIR / "ah5e4_weno5"
    state_weno5, blew_up = run_d2_weno5(
        grid, z_coord_base, H_bathy, ocean_mask,
        gm_redi_cfg, eos_config, physics, dt,
        total_years=2.0, output_dir=d2_dir,
    )

    # ---- D2 comparison ----
    if not blew_up:
        print("\n=== Building D2 comparison vs existing A_h=5e4 sweep result ===")
        make_d2_comparison(
            state_weno5, ocean_mask_np, H_bathy_np, lat, lon, d2_dir,
        )
    else:
        print("\n  WENO5 run blew up — skipping comparison plots")


if __name__ == "__main__":
    main()
