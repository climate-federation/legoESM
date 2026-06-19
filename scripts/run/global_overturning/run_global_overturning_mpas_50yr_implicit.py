#!/usr/bin/env python
"""MPAS analogue of the lat-lon ``run_global_overturning_implicit_spinup`` /
``run_global_overturning_50yr_implicit_continuation`` chain, with all
features lat-lon used now wired in on MPAS:

  - GM/Redi (centered + Visbeck adaptive)
  - Convective adjustment (enhanced_diffusion)
  - Implicit Crank-Nicolson barotropic solver
  - Same surface forcing, EOS, vertical levels, drag coefficient

Differences from lat-lon (geometry / numerical-stability / port-state
constraints, NOT knobs):
  - Grid: ico4 (~4°, 2562 cells) vs lat-lon 36×72 (5°)
  - A_h floor: 5e5 m²/s on TRiSK vs 2e5 on lat-lon
  - dt: configurable; lat-lon uses 600 s.  With implicit_cn the
    barotropic gravity-wave constraint is gone, so dt is bound only by
    3D CFL.  Smoke-test the largest stable dt before the long run.
  - GM/Redi slope_scheme = "centered" (lat-lon 50yr used default
    "triads"; MPAS triads is Phase 5 of the GM/Redi MPAS plan, not yet
    implemented).  Eady validation (docs/ocean_experiments/gm_redi_mpas_plan.md)
    found centered acceptable on weak-forcing physics, but expect more
    spurious diapycnal mixing in the deep ocean over 50 yr.

Outputs (results/ocean/global_overturning_mpas_50yr_implicit_dt{DT}/):
  - restart_day{000000,001825,...}.npz — every 5 sim-yr (matches lat-lon)
  - run.log

Usage:
    # Smoke test (30 days at dt=600)
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_mpas_50yr_implicit.py \\
        --years 0.082 --dt 600 --smoke-test

    # Full 50-yr run at chosen dt
    JAX_ENABLE_X64=1 python scripts/run/global_overturning/run_global_overturning_mpas_50yr_implicit.py \\
        --years 50 --dt 600
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "matrix"))
os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.experiments.global_overturning import (
    GlobalOverturningConfig, create_initial_conditions, create_forcings,
    create_eos_config, create_gm_redi_config,
    global_overturning_mpas_model_config,
)


MPAS_SUBDIVISION_LEVEL = 4   # ico4 ~ 4° (2562 cells) — closest match to lat-lon 5°
MPAS_A_H_FLOOR = 5.0e5       # TRiSK stability floor (matrix runs use this)


def _save_restart(state, day, output_dir, dt):
    npz = {"step": int(round(day * 86400 / dt)), "time_days": float(day),
           "grid_type": "mpas",
           "mpas_subdivision_level": MPAS_SUBDIVISION_LEVEL}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=float, default=50.0,
                        help="Total simulation years (default: 50)")
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Timestep in seconds (default: 600).  Lat-lon "
                             "uses 600s; implicit_cn removes the barotropic "
                             "constraint, so 600s should be within 3D CFL "
                             "on ico4.  Fall back to 300 or 200 if unstable.")
    parser.add_argument("--restart-every-years", type=float, default=5.0,
                        help="Restart save cadence (default: 5 yr)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tag output dir as smoke test (separate dir; "
                             "useful for stability sweeps).")
    parser.add_argument("--block-size", type=int, default=1000,
                        help="JIT scan block length (default: 1000 steps)")
    args = parser.parse_args()

    dt = args.dt
    total_years = args.years

    tag = "smoke" if args.smoke_test else "50yr"
    OUTPUT_DIR = Path(
        f"results/ocean/global_overturning_mpas_{tag}_implicit_dt{int(dt)}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Configuration: matches lat-lon GO+GM/Redi exactly via the
    #      shared GlobalOverturningConfig / create_forcings helpers ----
    config = GlobalOverturningConfig(use_gm_redi=True)
    z_coord = create_ocean_z_star(
        n_levels=config.n_levels, H_max=config.H_max,
        dz_surface=config.dz_surface, dz_deep=config.dz_deep,
    )
    mesh = create_voronoi_mesh(MPAS_SUBDIVISION_LEVEL)
    physics = create_forcings("mpas", mesh, config)   # includes enhanced_diffusion convection
    eos_config = create_eos_config(config)
    gm_redi_cfg = create_gm_redi_config(config)
    # Override slope_scheme: lat-lon 50yr used the default "triads"
    # (Griffies 1998); MPAS only has "centered" implemented (Phase 5
    # of the GM/Redi MPAS plan).  This is the one GM/Redi knob we
    # cannot match exactly.
    gm_redi_cfg = gm_redi_cfg._replace(slope_scheme="centered")

    a_h_used = max(config.A_h, MPAS_A_H_FLOOR)

    ocean_config = global_overturning_mpas_model_config(
        config, physics=physics, eos_config=eos_config, gm_redi_cfg=gm_redi_cfg,
        A_h=a_h_used,                                  # lifted to MPAS A_h floor
        barotropic_solver="implicit_cn",              # n_barotropic_substeps unused
    )

    n_steps = int(total_years * 365.0 * 86400 / dt)
    block_size = min(args.block_size, max(1, n_steps))
    n_steps_per_restart = int(args.restart_every_years * 365.0 * 86400 / dt)

    print(f"=== MPAS 50-yr GO+GM/Redi run (dt={dt:g}s, ico{MPAS_SUBDIVISION_LEVEL}) ===")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mesh: ico{MPAS_SUBDIVISION_LEVEL} "
          f"(nCells={int(mesh.lonCell.shape[0])})")
    print(f"  Vertical: {config.n_levels} levels, H_max={config.H_max} m")
    print(f"  dt = {dt} s, n_steps = {n_steps:,} ({total_years} sim-yr)")
    print(f"  Block size: {block_size}")
    print(f"  A_h = {a_h_used:.1e} m²/s (lat-lon ref={config.A_h:.1e}, "
          f"MPAS floor={MPAS_A_H_FLOOR:.0e})")
    print(f"  A_v = {config.A_v:.1e}, K_v = {config.K_v:.1e}, "
          f"r_bot = {config.bottom_drag_coeff:.4f}")
    print(f"  GM/Redi: ENABLED (slope_scheme=centered [lat-lon ref uses "
          f"triads, not ported to MPAS], Visbeck "
          f"α={config.visbeck_alpha}, κ∈[{config.visbeck_kappa_min:.0f},"
          f"{config.visbeck_kappa_max:.0f}])")
    print(f"  Convection: enhanced_diffusion (K_conv=1.0)")
    print(f"  Barotropic: implicit_cn θ={ocean_config.barotropic_implicit_theta_eta} "
          f"(PCG tol={ocean_config.barotropic_implicit_pcg_tol:.0e})")
    print(f"  Restart cadence: every {args.restart_every_years} yr")
    print()

    model = MPASOceanModel(mesh, z_coord, ocean_config)
    state = create_initial_conditions("mpas", mesh, z_coord, config)
    print(f"Initial state (rest, stratified):")
    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] °C")
    print(f"  S uniform: {float(jnp.mean(state.S.data)):.2f} PSU")
    print(f"  |η|max: {float(jnp.max(jnp.abs(state.eta.data))):.3e} m")
    print()

    block_fn = _make_step_block(model, dt)
    _save_restart(state, 0.0, OUTPUT_DIR, dt)

    n_blocks = max(1, n_steps // block_size)
    n_remainder = n_steps - n_blocks * block_size
    print(f"Starting integration ({n_blocks} blocks × {block_size} "
          f"+ {n_remainder} remainder)")
    t0 = time.time()
    last_print = t0
    steps_done = 0
    last_restart_step = 0
    progress_every = max(1, n_blocks // 50)

    for b in range(n_blocks):
        state = block_fn(state, block_size)
        steps_done += block_size

        if (steps_done - last_restart_step) >= n_steps_per_restart:
            jax.block_until_ready(state.eta.data)
            day = steps_done * dt / 86400.0
            _save_restart(state, day, OUTPUT_DIR, dt)
            last_restart_step = steps_done

        if (b + 1) % progress_every == 0 or (b + 1) == n_blocks:
            now = time.time()
            if now - last_print > 30 or (b + 1) == n_blocks:
                jax.block_until_ready(state.eta.data)
                yr = steps_done * dt / 86400.0 / 365.0
                eta_max = float(jnp.max(jnp.abs(state.eta.data)))
                T_max = float(jnp.max(state.T.data))
                T_min = float(jnp.min(state.T.data))
                u_max = float(jnp.max(jnp.abs(state.u.data)))
                elapsed = now - t0
                eta_s = elapsed / yr * total_years - elapsed if yr > 0 else 0
                eta_str = (f"{eta_s/60:.1f} min" if eta_s < 3600
                           else f"{eta_s/3600:.2f} h")
                print(f"  Yr {yr:6.3f}/{total_years:.1f} | "
                      f"|η|max={eta_max:.2e} | "
                      f"T∈[{T_min:.1f},{T_max:.1f}] | "
                      f"|u|max={u_max:.3f} | ETA {eta_str}", flush=True)
                last_print = now

                if not np.isfinite(eta_max) or eta_max > 100.0:
                    print(f"\n  BLOWUP at year {yr:.3f} (|η|max={eta_max}) "
                          f"— saving restart and aborting", flush=True)
                    final_day = steps_done * dt / 86400.0
                    _save_restart(state, final_day, OUTPUT_DIR, dt)
                    return 1

    if n_remainder > 0:
        state = block_fn(state, n_remainder)
        steps_done += n_remainder

    jax.block_until_ready(state.eta.data)
    wall = time.time() - t0
    final_day = steps_done * dt / 86400.0
    print(f"\nRun complete in {wall:.0f}s ({wall/60:.1f} min, {wall/3600:.2f} h)")
    print(f"  Final state: sim day {final_day:.2f} (year {final_day/365:.3f})")
    print(f"  |η|max = {float(jnp.max(jnp.abs(state.eta.data))):.4e} m")
    print(f"  T range: [{float(jnp.min(state.T.data)):.2f}, "
          f"{float(jnp.max(state.T.data)):.2f}] °C")
    print(f"  |u|max = {float(jnp.max(jnp.abs(state.u.data))):.4e} m/s")

    _save_restart(state, final_day, OUTPUT_DIR, dt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
