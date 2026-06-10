#!/usr/bin/env python
"""Phase 4(c) extension: continue the 50-yr polar-cap spinup to 100 yr.

Loads the latest restart from
``results/ocean/global_overturning_realistic_50yr_polar_cap/`` (which
will be ``restart_day016438.npz`` — yr 45.04 — because the original
50-yr run hit a cosmetic post-completion crash that prevented the
yr-50 restart save), then integrates forward to total sim year 100.

Writes new restarts to the same output directory so the progress
plots produce a continuous yr 0-100 view.

Wall time: ~5 h on a MacBook P-cores (55 sim-yr × ~5 min/yr + JIT).

Usage:
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_realistic_100yr_continuation.py
"""

from __future__ import annotations

import os
import re
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
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

# Same production config as the 50-yr run
A_H_GLOBAL = 2.0e5
A_H_LAT_SCALING = True
NORTH_CAP_LAT = 80.0
TARGET_TOTAL_YEARS = 100.0


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
    print(f"    Restart saved: {fname.name}", flush=True)


def _make_step_block(model, dt):
    def scan_body(state, _):
        return model.step(state, dt), None

    @partial(jax.jit, static_argnames=("n_inner",))
    def block_fn(state, n_inner: int):
        state, _ = jax.lax.scan(scan_body, state, None, length=n_inner)
        return state

    return block_fn


def _find_latest_restart(output_dir):
    """Return (path, day) of the highest-numbered restart_dayXXXXXX.npz."""
    pat = re.compile(r"restart_day(\d+)\.npz$")
    candidates = []
    for p in output_dir.glob("restart_day*.npz"):
        m = pat.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No restart_day*.npz found in {output_dir}")
    candidates.sort()
    day, path = candidates[-1]
    return path, day


def _load_state_from_restart(path, fresh_state):
    """Replace data in `fresh_state` from the npz at `path`.

    Uses fresh_state as the structural template (correct pytree, dtypes,
    Field types).  For each field in the npz that exists in the state
    and has a `.data` attribute, replace its data with the saved values.
    """
    data = np.load(path, allow_pickle=False)
    print(f"  Loading restart: {path.name}", flush=True)
    print(f"    saved time_days = {float(data['time_days']):.2f}, "
          f"step = {int(data['step'])}")
    state = fresh_state
    restored = []
    skipped = []
    for f in state._fields:
        obj = getattr(state, f)
        if obj is None or not hasattr(obj, "data"):
            skipped.append(f"{f}(no data)")
            continue
        if f not in data.files:
            skipped.append(f"{f}(missing)")
            continue
        saved = np.asarray(data[f])
        if saved.shape != tuple(obj.data.shape):
            print(f"    WARNING: shape mismatch on {f}: "
                  f"saved {saved.shape} vs state {tuple(obj.data.shape)}")
            skipped.append(f"{f}(shape)")
            continue
        new_obj = obj.replace(data=jnp.asarray(saved, dtype=obj.data.dtype))
        state = state._replace(**{f: new_obj})
        restored.append(f)
    print(f"    Restored fields: {', '.join(restored)}")
    if skipped:
        print(f"    Skipped: {', '.join(skipped)}")
    return state


def _run_progress_plots():
    plot_script = (
        Path(__file__).resolve().parent / "plot_realistic_geometry_progress.py"
    )
    if not plot_script.exists():
        return
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
        subprocess.run([sys.executable, str(tmp)], env=env, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        print("  ⚠ progress plotting timed out", flush=True)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def main():
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Output dir {OUTPUT_DIR} does not exist — "
                                 "this script extends an existing 50-yr run.")

    # Find the latest restart to start from
    restart_path, restart_day = _find_latest_restart(OUTPUT_DIR)
    start_yr = restart_day / 365.0
    remaining_yr = TARGET_TOTAL_YEARS - start_yr
    if remaining_yr <= 0:
        print(f"Latest restart is already at yr {start_yr:.2f} ≥ "
              f"{TARGET_TOTAL_YEARS:.0f}. Nothing to do.")
        return

    print(f"=== Phase 4(c) extension: yr {start_yr:.2f} → "
          f"{TARGET_TOTAL_YEARS:.0f} ===")
    print(f"  Restart: {restart_path.name} (yr {start_yr:.2f})")
    print(f"  Remaining: {remaining_yr:.2f} yr")

    # ---- Build geometry exactly as in the 50-yr run ----
    config = GlobalOverturningConfig(
        use_gm_redi=True,
        bottom_drag_coeff=2.5e-3,
        A_h=A_H_GLOBAL,
        H_max=5000.0, dz_surface=20.0,
        kappa_GM=800.0, kappa_Redi=800.0,
        T_water_init_C=20.0, T_deep_C=2.0, T_scale_depth=1000.0,
    )
    grid = create_latlon_grid(36, 72)
    z_coord_base = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    bathy_cfg = BathymetryConfig(
        source="file", path=str(ETOPO_FILE),
        H_max=config.H_max, H_min=50.0, smoothing_passes=5,
        enforce_straits=True, fill_isolated_basins=True,
        depth_is_negative=True, r_factor_max=0.2,
        north_cap_lat=NORTH_CAP_LAT,
    )
    H_bathy, ocean_mask = init_ocean_bathymetry(grid, bathy_cfg)
    H_bathy = H_bathy.astype(jnp.float32)
    ocean_mask = ocean_mask.astype(jnp.float32)
    z_coord = create_partial_cell_coordinate(z_coord_base, H_bathy)

    # Build fresh state (structure template) and overwrite from restart
    fresh_state = rest_state_latlon_cgrid_ocean(
        grid, z_coord_base,
        T_water_init_C=config.T_water_init_C, T_deep=config.T_water_init_C,
        S_uniform=config.S_uniform,
        H_bathy_override=H_bathy,
        land_mask_override=ocean_mask,
    )
    state = _load_state_from_restart(restart_path, fresh_state)

    physics = create_forcings("latlon", grid, config)
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)

    ocean_config = LatLonCGridOceanConfig(
        n_barotropic_substeps=30,
        physics=physics,
        A_h=A_H_GLOBAL,
        A_h_lat_scaling=A_H_LAT_SCALING,
        B_h=5.0e9, A_v=config.A_v, K_v=config.K_v,
        bottom_drag_r=config.bottom_drag_coeff,
        bottom_drag_bbl_thickness=100.0,
        eos="linear", eos_linear=eos_config, gm_redi=gm_redi_cfg,
        pgf_scheme="adcroft",   # AC + h_actual; previously "smc03"
                                # (bit-equivalent on lat-lon, simpler;
                                # see plan-doc §8d)
        momentum_advection="vector_invariant",
        barotropic_solver="implicit_cn",
    )

    print(f"  Config: A_h={A_H_GLOBAL:.1e} (cos²-scaled), "
          f"north_cap={NORTH_CAP_LAT}°, drag={config.bottom_drag_coeff:.1e}",
          flush=True)

    # Verify state restoration
    _wet = state.land_mask.data[..., jnp.newaxis] > 0.5
    _T_min = float(jnp.min(jnp.where(_wet, state.T.data, jnp.inf)))
    _T_max = float(jnp.max(jnp.where(_wet, state.T.data, -jnp.inf)))
    print(f"  Loaded state: T∈[{_T_min:.2f}, {_T_max:.2f}] °C, "
          f"|u|max = {float(jnp.max(jnp.abs(state.u.data))):.3f} m/s, "
          f"|η|max = {float(jnp.max(jnp.abs(state.eta.data))):.3f} m",
          flush=True)

    # ---- Integrate ----
    dt = 600.0
    n_steps = int(remaining_yr * 365.0 * 86400 / dt)
    block_size = 1000
    restart_every_years = 5.0
    n_steps_per_restart = int(restart_every_years * 365.0 * 86400 / dt)

    n_blocks = n_steps // block_size
    n_remainder = n_steps - n_blocks * block_size
    print(f"  dt={dt}s, n_steps={n_steps:,} ({remaining_yr:.2f} yr remaining)")
    print(f"  Block size: {block_size} ({n_blocks} blocks + {n_remainder} remainder)")
    print(f"  Restart cadence: every {restart_every_years} yr")
    print()

    model = LatLonCGridOceanModel(grid, z_coord, ocean_config)
    block_fn = _make_step_block(model, dt)

    t0 = time.time()
    last_print = t0
    times_yr = [start_yr]
    umax_history = [float(jnp.max(jnp.abs(state.u.data)))]
    eta_max_history = [float(jnp.max(jnp.abs(state.eta.data)))]
    steps_done = 0
    last_restart_step = 0
    progress_every = max(1, n_blocks // 50)

    s = state
    blew_up = False

    # Save the starting state (current restart) again at the integer
    # year boundary if start_yr isn't already on a 5-yr mark.  The day
    # 16438 restart is at yr 45.04 (close enough to 45 — keep it as is).

    for b in range(n_blocks):
        s = block_fn(s, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(s.eta.data)
            day = restart_day + steps_done * dt / 86400.0
            _save_restart(s, day, OUTPUT_DIR)
            last_restart_step = steps_done
            try:
                _run_progress_plots()
            except Exception as e:
                print(f"  ⚠ progress plots failed: {e}", flush=True)

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(s.eta.data)
                yr_total = start_yr + steps_done * dt / (365.0 * 86400.0)
                u_max = float(jnp.max(jnp.abs(s.u.data)))
                eta_max = float(jnp.max(jnp.abs(s.eta.data)))
                _wet = s.land_mask.data[..., jnp.newaxis] > 0.5
                T_min = float(jnp.min(jnp.where(_wet, s.T.data, jnp.inf)))
                T_max = float(jnp.max(jnp.where(_wet, s.T.data, -jnp.inf)))
                finite = bool(
                    jnp.all(jnp.isfinite(s.u.data))
                    and jnp.all(jnp.isfinite(s.T.data))
                )
                eta_remaining = (now - t0) / max(yr_total - start_yr, 1e-3) * \
                    max(TARGET_TOTAL_YEARS - yr_total, 0.0) / 60.0
                print(
                    f"  Yr {yr_total:6.2f}/{TARGET_TOTAL_YEARS:.0f} | "
                    f"|η|max={eta_max:.2e} | "
                    f"T∈[{T_min:.1f},{T_max:.1f}] | "
                    f"|u|max={u_max:.3f} | "
                    f"finite={finite} | ETA {eta_remaining:.1f} min",
                    flush=True,
                )
                times_yr.append(yr_total)
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
    final_yr = start_yr + steps_done * dt / (365.0 * 86400.0)
    print(f"\nExtension complete in {wall:.0f}s ({wall/60:.1f} min, "
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
        # Save final restart at the actual final day
        _save_restart(s, restart_day + steps_done * dt / 86400.0, OUTPUT_DIR)

    # Run.log: append rather than overwrite if one exists
    log_path = OUTPUT_DIR / "run.log"
    mode = "a" if log_path.exists() else "w"
    with open(log_path, mode) as f:
        if mode == "w":
            f.write("Phase 4(c) realistic-geometry GO 50-yr + extension to 100 yr\n")
            f.write(f"grid = 36x72, n_levels = {config.n_levels}, H_max = {config.H_max}\n")
            f.write(f"coord = partial cells (ETOPO bathymetry)\n")
            f.write(f"pgf_scheme = smc03\n")
            f.write(f"momentum_advection = vector_invariant (AL81)\n")
            f.write(f"barotropic_solver = implicit_cn\n")
            f.write(f"bottom_drag_r = {ocean_config.bottom_drag_r}\n")
            f.write(f"B_h = {ocean_config.B_h}\n")
            f.write(f"A_h = {ocean_config.A_h}, A_h_lat_scaling = "
                    f"{ocean_config.A_h_lat_scaling}\n")
            f.write(f"K_GM = {gm_redi_cfg.kappa_GM}, K_Redi = "
                    f"{gm_redi_cfg.kappa_Redi}\n")
            f.write(f"north_cap_lat = {NORTH_CAP_LAT}\n")
        f.write(f"\n--- extension run from yr {start_yr:.2f} to {final_yr:.2f} "
                f"(wall {wall:.0f} s, blew_up = {blew_up}) ---\n")
        for yr, u, eta in zip(times_yr, umax_history, eta_max_history):
            f.write(f"  yr={yr:6.2f}  |u|max={u:.4f}  |eta|max={eta:.3e}\n")
    print(f"Updated {log_path}", flush=True)

    # Final umax timeseries plot
    times_yr_a = np.array(times_yr)
    umax_a = np.array(umax_history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times_yr_a, umax_a, "o-", color="C0", ms=3)
    ax.axvline(start_yr, color="C3", linestyle=":", alpha=0.6, label=f"continuation start (yr {start_yr:.0f})")
    ax.set_xlabel("Sim year")
    ax.set_ylabel("max|u| (m/s)")
    ax.set_title(
        f"Phase 4(c) realistic-geometry GO {TARGET_TOTAL_YEARS:.0f}-yr spinup "
        f"(continuation segment)\n"
        f"final max|u| = {umax_a[-1]:.3f} m/s after {final_yr:.1f} yr  "
        f"({'PASS' if not blew_up else 'NaN'})",
    )
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umax_timeseries_continuation.png", dpi=130)
    plt.close()
    print(f"Saved {OUTPUT_DIR / 'umax_timeseries_continuation.png'}", flush=True)

    # Final progress plots over all restarts (yr 0 through final_yr)
    try:
        _run_progress_plots()
    except Exception as e:
        print(f"  ⚠ final progress plots failed: {e}", flush=True)


if __name__ == "__main__":
    main()
