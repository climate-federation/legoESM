#!/usr/bin/env python
"""Phase 4(c): Production 50-yr Wolfe-Cessi spinup on real ETOPO with
the full fix stack landed in the 2026-05-01 evening session.

Differences from `run_global_overturning_realistic_geometry.py`
(Phase 4(b)):

  - **A_h_lat_scaling = True**, A_h_global = 2e5
    (was uniform A_h = 1e4; cos²(lat) keeps viscous CFL constant
    and lets us use 20× more interior viscosity to damp the 2Δy
    zonal-jet mode).
  - **north_cap_lat = 80°** in BathymetryConfig
    (closes off Arctic ocean cells where cos²(lat) scaling leaves
    A_h_eff under-damped; matches idealized polar_cap_lat).
  - 50 sim-years, 5-yr restart cadence (matches the previous run's
    convention).
  - Auto-runs `plot_realistic_geometry_progress.py` at end (or after
    crash, on whatever restarts are saved).

Outputs (results/ocean/global_overturning_realistic_50yr_polar_cap/):
  - restart_dayNNNNNN.npz at every 5-yr mark + final
  - run.log
  - umax_timeseries.png
  - {timeseries,snapshots,T_zonal_mean,moc,barotropic_streamfunction}_progress.png

Wall time: ~4-5 hours on a MacBook P-cores at 5 min/sim-year.

Usage:
    JAX_ENABLE_X64=1 python scripts/global_overturning/run_global_overturning_realistic_50yr_polar_cap.py
"""

from __future__ import annotations

import os
import subprocess
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


OUTPUT_DIR = Path("results/ocean/global_overturning_realistic_50yr_polar_cap")
ETOPO_FILE = Path("data/bathymetry/etopo_1deg.nc")
ETOPO_URL = (
    "https://upwell.pfeg.noaa.gov/erddap/griddap/etopo180.nc?"
    "altitude%5B(-90):60:(90)%5D%5B(-180):60:(180)%5D"
)

# Production config (the headline change from Phase 4(b)):
A_H_GLOBAL = 2.0e5
A_H_LAT_SCALING = True
NORTH_CAP_LAT = 80.0


def _ensure_etopo():
    if ETOPO_FILE.exists():
        return
    ETOPO_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ETOPO → {ETOPO_FILE}")
    subprocess.run(
        ["curl", "-fsS", "-o", str(ETOPO_FILE), ETOPO_URL], check=True,
    )


def _save_restart(state, day, output_dir):
    npz = {
        "step": int(round(day * 86400 / 600)),
        "time_days": float(day),
        "grid_type": "latlon",
    }
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            continue
        npz[f] = np.asarray(obj.data)
    fname = output_dir / f"restart_day{int(round(day)):06d}.npz"
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


def _run_progress_plots():
    """Invoke plot_realistic_geometry_progress.py against this run dir.
    Patches the RUN_DIR via env var so the plotting script writes to
    OUTPUT_DIR rather than the original Phase 4(b) dir.
    """
    # The plotting script hardcodes RUN_DIR; the cleanest way is to copy
    # restarts to a temporary location, but that's wasteful.  Instead
    # we'll just call it with a wrapper-style path swap.
    plot_script = (
        Path(__file__).resolve().parent / "plot_realistic_geometry_progress.py"
    )
    if not plot_script.exists():
        print(f"  ⚠ plot script not found: {plot_script}")
        return
    # Read the script, replace RUN_DIR, write to a temp file, run that.
    src = plot_script.read_text()
    tmp = OUTPUT_DIR / "_plot_progress.py"
    src_patched = src.replace(
        'RUN_DIR = Path("results/ocean/global_overturning_realistic_geometry")',
        f'RUN_DIR = Path("{OUTPUT_DIR}")',
    )
    tmp.write_text(src_patched)
    try:
        env = os.environ.copy()
        env["JAX_ENABLE_X64"] = "1"
        subprocess.run(
            [sys.executable, str(tmp)],
            env=env,
            check=False,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print("  ⚠ progress plotting timed out")
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_etopo()

    # ---- Configuration ----
    total_years = float(os.environ.get("GO_RG_YEARS", "50.0"))
    dt = 600.0
    n_steps = int(total_years * 365.0 * 86400 / dt)
    block_size = 1000
    restart_every_years = 5.0
    n_steps_per_restart = int(restart_every_years * 365.0 * 86400 / dt)

    config = GlobalOverturningConfig(
        use_gm_redi=True,
        bottom_drag_coeff=2.5e-3,
        A_h=A_H_GLOBAL,                 # <-- 2e5 (was 1e4)
        H_max=5000.0,
        dz_surface=20.0,
        kappa_GM=800.0,
        kappa_Redi=800.0,
        T_surface=20.0,
        T_deep=2.0,
        T_scale_depth=1000.0,
    )

    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )

    bathy_cfg = BathymetryConfig(
        source="file",
        path=str(ETOPO_FILE),
        H_max=config.H_max,
        H_min=50.0,
        smoothing_passes=5,
        enforce_straits=True,
        fill_isolated_basins=True,
        depth_is_negative=True,
        r_factor_max=0.2,
        north_cap_lat=NORTH_CAP_LAT,    # <-- new
    )
    H_bathy, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy.astype(jnp.float32)
    ocean_mask = ocean_mask.astype(jnp.float32)

    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)

    n_ocean = int(np.sum(ocean_mask))
    n_total = int(ocean_mask.size)

    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_surface=config.T_surface, T_deep=config.T_surface,
        S_uniform=config.S_uniform,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )
    centroid = compute_centroid_depth(
        jnp.zeros_like(H_bathy), H_bathy, z_coord,
    )
    T_per_cell = config.T_deep + (config.T_surface - config.T_deep) * jnp.exp(
        -centroid / config.T_scale_depth,
    )
    T_per_cell = jnp.where(z_coord.is_active, T_per_cell, config.T_deep)
    T_per_cell = T_per_cell * state.land_mask.data[..., jnp.newaxis]
    state = state._replace(
        T=state.T.replace(data=T_per_cell.astype(state.T.data.dtype)),
    )

    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=A_H_GLOBAL,
        A_h_lat_scaling=A_H_LAT_SCALING,    # <-- new
        B_h=5.0e9,
        A_v=config.A_v,
        K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear",
        eos_linear=eos_config,
        gm_redi=gm_redi_cfg,
        pgf_scheme="adcroft",   # AC + h_actual; previously "smc03"
                                # (bit-equivalent on lat-lon, simpler;
                                # see plan-doc §8d)
        momentum_advection="vector_invariant",
        barotropic_solver="implicit_cn",
    )

    print("=== Phase 4(c): Production 50-yr Wolfe-Cessi spinup, polar-cap config ===")
    print(f"  Output:                {OUTPUT_DIR}")
    print(f"  Grid:                  36×72 (5°), 20 levels, H_max={config.H_max} m")
    print(f"  Coord:                 partial cells (z* + h_partial)")
    print(f"  PGF scheme:            {ocean_config.pgf_scheme}")
    print(f"  Momentum advection:    {ocean_config.momentum_advection} (AL81)")
    print(f"  Barotropic solver:     {ocean_config.barotropic_solver}")
    print(f"  Bottom drag:           r = {ocean_config.bottom_drag_r:.1e} 1/s")
    print(f"  B_h biharmonic:        {ocean_config.B_h:.1e} m⁴/s  (cos⁴ scaled)")
    print(f"  A_h Laplacian:         {ocean_config.A_h:.1e} m²/s  "
          f"(A_h_lat_scaling={ocean_config.A_h_lat_scaling}; cos²-scaled)")
    print(f"  GM/Redi:               K_GM = {gm_redi_cfg.kappa_GM:.0f} m²/s, "
          f"K_Redi = {gm_redi_cfg.kappa_Redi:.0f} m²/s")
    print(f"  Wet cells:             {n_ocean}/{n_total} "
          f"({100.0*n_ocean/n_total:.1f}%)")
    print(f"  H_bathy range:         "
          f"[{float(H_bathy[ocean_mask>0].min()):.0f}, "
          f"{float(H_bathy.max()):.0f}] m")
    print(f"  MEO r-factor cap:      {bathy_cfg.r_factor_max}")
    print(f"  North polar cap:       lat > {NORTH_CAP_LAT}° = land")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({total_years} sim-yr)")
    print(f"  Block size:            {block_size} steps  ({n_steps // block_size} blocks)")
    print(f"  Restart cadence:       every {restart_every_years} yr")
    print()
    _wet = state.land_mask.data[..., jnp.newaxis] > 0.5
    _T_min = float(jnp.min(jnp.where(_wet, state.T.data, jnp.inf)))
    _T_max = float(jnp.max(jnp.where(_wet, state.T.data, -jnp.inf)))
    print("  Initial state: rest, centroid-aware exp(z) T")
    print(f"    T range: [{_T_min:.2f}, {_T_max:.2f}] °C")
    print(f"    S uniform: {float(jnp.mean(state.S.data)):.2f} PSU")
    print(f"    η: {float(jnp.max(jnp.abs(state.eta.data))):.3e} m  (rest)")
    print(flush=True)

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    block_fn = _make_step_block(model, dt)

    _save_restart(state, 0.0, OUTPUT_DIR)

    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    print(f"Starting integration ({n_blocks} blocks × {block_size} steps + "
          f"{n_remainder} remainder)", flush=True)

    t0 = time.time()
    last_print = t0
    times_yr = [0.0]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]
    eta_max_history = [float(jnp.max(jnp.abs(state.eta.data)))]
    steps_done = 0
    last_restart_step = 0
    progress_every = max(1, n_blocks // 50)

    s = state
    blew_up = False
    for b in range(n_blocks):
        s = block_fn(s, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(s.eta.data)
            day = steps_done * dt / 86400.0
            _save_restart(s, day, OUTPUT_DIR)
            last_restart_step = steps_done
            # Generate progress plots after each restart so we have
            # intermediate diagnostics even if the run later crashes.
            try:
                _run_progress_plots()
            except Exception as e:
                print(f"  ⚠ progress plots failed: {e}", flush=True)

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(s.eta.data)
                yr = steps_done * dt / (365.0 * 86400.0)
                u_max = float(jnp.max(jnp.abs(s.u.data)))
                eta_max = float(jnp.max(jnp.abs(s.eta.data)))
                _wet = s.land_mask.data[..., jnp.newaxis] > 0.5
                T_min = float(jnp.min(jnp.where(_wet, s.T.data, jnp.inf)))
                T_max = float(jnp.max(jnp.where(_wet, s.T.data, -jnp.inf)))
                finite = bool(
                    jnp.all(jnp.isfinite(s.u.data))
                    and jnp.all(jnp.isfinite(s.T.data))
                )
                eta_remaining = (now - t0) / max(yr, 1e-3) * \
                    max(total_years - yr, 0.0) / 60.0
                print(
                    f"  Yr {yr:5.2f}/{total_years:.0f} | "
                    f"|η|max={eta_max:.2e} | "
                    f"T∈[{T_min:.1f},{T_max:.1f}] | "
                    f"|u|max={u_max:.3f} | "
                    f"finite={finite} | ETA {eta_remaining:.1f} min",
                    flush=True,
                )
                times_yr.append(yr)
                umax_history.append(u_max)
                eta_max_history.append(eta_max)
                if not finite:
                    print("  BLEW UP — saving last finite state and aborting",
                          flush=True)
                    blew_up = True
                    break
                last_print = now

    if not blew_up and n_remainder > 0:
        s = block_fn(s, n_remainder)
        jax.block_until_ready(s.eta.data)

    wall = time.time() - t0
    final_yr = steps_done * dt / (365.0 * 86400.0)
    print(f"\nIntegration complete in {wall:.0f}s ({wall/60:.1f} min, "
          f"{wall/3600:.2f} h)", flush=True)
    print(f"  Final state at sim year {final_yr:.2f}", flush=True)
    if not blew_up:
        u_max = float(jnp.max(jnp.abs(s.u.data)))
        eta_max = float(jnp.max(jnp.abs(s.eta.data)))
        _wet = s.land_mask.data[..., jnp.newaxis] > 0.5
        T_min = float(jnp.min(jnp.where(_wet, s.T.data, jnp.inf)))
        T_max = float(jnp.max(jnp.where(_wet, s.T.data, -jnp.inf)))
        print(f"  |η|max = {eta_max:.4e} m")
        print(f"  T range: [{T_min:.2f}, {T_max:.2f}] °C")
        print(f"  |u|max = {u_max:.4e} m/s")
        _save_restart(s, final_yr * 365.0, OUTPUT_DIR)

    times_yr_a = np.array(times_yr)
    umax_a = np.array(umax_history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_yr_a, umax_a, "o-", color="C0", ms=3)
    ax.set_xlabel("Sim year")
    ax.set_ylabel("max|u| (m/s)")
    ax.set_title(
        f"Phase 4(c) realistic-geometry GO 50-yr spinup\n"
        f"cos²(lat) A_h=2e5 + north_cap=80°  |  "
        f"final max|u| = {umax_a[-1]:.3f} m/s after {final_yr:.1f} yr  "
        f"({'PASS' if not blew_up else 'NaN'})",
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umax_timeseries.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'umax_timeseries.png'}", flush=True)

    with open(OUTPUT_DIR / "run.log", "w") as f:
        f.write("Phase 4(c) realistic-geometry GO 50-yr spinup, polar-cap config\n")
        f.write(f"grid = 36x72, n_levels = {config.n_levels}, H_max = {config.H_max}\n")
        f.write(f"coord = partial cells (ETOPO bathymetry)\n")
        f.write(f"pgf_scheme = smc03\n")
        f.write(f"momentum_advection = vector_invariant (AL81)\n")
        f.write(f"barotropic_solver = implicit_cn\n")
        f.write(f"bottom_drag_r = {ocean_config.bottom_drag_r}\n")
        f.write(f"B_h = {ocean_config.B_h}\n")
        f.write(f"A_h = {ocean_config.A_h}, A_h_lat_scaling = {ocean_config.A_h_lat_scaling}\n")
        f.write(f"K_GM = {gm_redi_cfg.kappa_GM}, K_Redi = {gm_redi_cfg.kappa_Redi}\n")
        f.write(f"north_cap_lat = {NORTH_CAP_LAT}\n")
        f.write(f"total_years = {total_years}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"final_year = {final_yr}\n")
        f.write(f"blew_up = {blew_up}\n")
        f.write(f"wall_time_s = {wall:.1f}\n")
        f.write(f"\nYear-by-year:\n")
        for yr, u, eta in zip(times_yr, umax_history, eta_max_history):
            f.write(f"  yr={yr:5.2f}  |u|max={u:.4f}  |eta|max={eta:.3e}\n")
    print(f"Saved {OUTPUT_DIR / 'run.log'}", flush=True)

    # Final progress plots — over all restarts saved.
    try:
        _run_progress_plots()
    except Exception as e:
        print(f"  ⚠ final progress plots failed: {e}", flush=True)


if __name__ == "__main__":
    main()
