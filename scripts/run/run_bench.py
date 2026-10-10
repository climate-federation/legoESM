#!/usr/bin/env python
"""BENCH — NEMO-BENCH-inspired ocean performance benchmark (standalone runner).

Drives the ``bench`` experiment (Irrmann et al. 2022, GMD 15, 1567-1582,
Sect. 2.2) on the Mercator-truncated lat-lon or synthetic-tripole C-grid
and reports the production step cost: compile time, steady-state per-step
wall time, SYPD and Mcells/s, following the ``run_dino.py`` timing-JSONL
contract.

Zero input files by construction: the grid, bathymetry, initial conditions
and forcing are all analytic (see
``packages/ocean/legoesm/ocean/experiments/bench.py`` for the NEMO
provenance of every formula).

Quick start::

    # ORCA1-like global latlon (default 1000 steps, NEMO's nn_itend):
    JAX_ENABLE_X64=1 python scripts/run/run_bench.py

    # ORCA025-like preset (sizing from BENCH_PRESETS):
    JAX_ENABLE_X64=1 python scripts/run/run_bench.py --preset orca025_like

    # Synthetic tripole (active north fold, no mesh file; sizing keeps
    # NEMO's exact j-count per preset — see BENCH_PRESETS):
    JAX_ENABLE_X64=1 python scripts/run/run_bench.py --grid tripole

Timing contract (one JSON line appended per run, run_dino-compatible
fields plus BENCH provenance): ``compile_ms``, ``steady_mean_ms``,
``steady_std_ms``, ``n_warmup``, ``sypd``, ``mcells_s``,
``step_times_ms``, ``preset``, ``grid_type``, ``fold_active``,
``recipe``, ``precision``, ``n_devices``, ``n_processes``.

The default step count is 1000 with 5 warmup steps (NEMO BENCH namelist
``nn_itend = 1000``; run_dino's measured JIT warmup plateau is ~4-5
steps).  No snapshots are written (the paper: no I/O perturbing the
measurement); a final stability summary is printed and validated against
the pre-registered BENCH gates (finite fields, max|u| < 1 m/s,
max|eta| < 1 m — see ``bench.validate_results``).

Weak scaling (NEMO's negative ``nn_jsize`` semantics) is NOT wired here:
it requires actual SPMD sharding, which belongs to the runners-repo
SPMD harness (documented follow-up in
``docs/ocean/experiments/bench_plan.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import jax  # noqa: E402
from legoesm.ocean.experiments import bench as bench_exp  # noqa: E402


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="BENCH ocean performance benchmark (NEMO BENCH inspired)"
    )
    p.add_argument(
        "--grid",
        default="latlon",
        choices=["latlon", "tripole"],
        help="Horizontal grid (default: latlon). 'tripole' is the "
        "synthetic tripole — active north fold, no mesh file.",
    )
    p.add_argument(
        "--preset",
        default=None,
        choices=sorted(bench_exp.BENCH_PRESETS),
        help="Resolution preset (sizes + first-guess dt). Default: orca1_like.",
    )
    p.add_argument(
        "--n-lat",
        type=int,
        default=None,
        help="Latitude rows (tripole only; the latlon lane's row count is "
        "DERIVED from --n-lat... i.e. n_lon and --lat-max via the "
        "Mercator placement).",
    )
    p.add_argument("--n-lon", type=int, default=None, help="Longitude columns (overrides preset).")
    p.add_argument(
        "--lat-max",
        type=float,
        default=None,
        help="Latlon Mercator truncation latitude [deg] (default 80, "
        "preset). The full-sphere equirectangular latlon grid is "
        "barotropic-CFL-unstable at the pole row (measured — see "
        "bench_plan.md).",
    )
    p.add_argument(
        "--nlev",
        type=int,
        default=None,
        help="Vertical levels (overrides preset; NEMO BENCH default 75).",
    )
    p.add_argument("--dt", type=float, default=None, help="Time step [s] (overrides preset).")
    p.add_argument(
        "--steps", type=int, default=1000, help="Number of steps (NEMO nn_itend default 1000)."
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Warmup steps excluded from the steady-state mean.",
    )
    p.add_argument(
        "--H-max",
        type=float,
        default=None,
        help="Flat-bottom depth [m] (default: 5000, NEMO BENCH).",
    )
    p.add_argument(
        "--timing-jsonl",
        default="timing_bench.jsonl",
        help="Output JSONL timing record path (append mode).",
    )
    p.add_argument(
        "--no-x64",
        action="store_true",
        help="Run in float32 (default: float64 via JAX_ENABLE_X64).",
    )
    return p.parse_args(argv)


def _build_setup(args):
    """Return (grid, z_coord, state, model_config, dt, preset_name)."""
    preset_name = args.preset or "orca1_like"
    preset = bench_exp.BENCH_PRESETS[preset_name]

    n_lon = args.n_lon if args.n_lon is not None else preset["n_lon"]
    lat_max = args.lat_max if args.lat_max is not None else preset["lat_max_deg"]
    nlev = args.nlev if args.nlev is not None else preset["n_levels"]
    H_max = args.H_max if args.H_max is not None else 5000.0
    dt = args.dt if args.dt is not None else preset["dt_seconds"]

    if args.grid == "latlon":
        # Mercator-truncated isotropic grid (NEMO-BENCH never runs a
        # full-sphere equirectangular grid — see bench.py); n_lat is
        # DERIVED from (n_lon, lat_max_deg), read off the grid.
        grid = bench_exp.create_latlon_grid_for_preset(n_lon, lat_max)
        n_lat = len(np.asarray(grid.lat))
    else:
        n_lat = args.n_lat if args.n_lat is not None else preset["n_lat_tripole"]
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)

    z_coord = bench_exp.bench_uniform_z_star(nlev, H_max)

    cfg = bench_exp.BenchConfig(
        n_lat=n_lat, n_lon=n_lon, n_levels=nlev, H_max=H_max, lat_max_deg=lat_max
    )
    state = bench_exp.create_initial_conditions(args.grid, grid, z_coord, cfg)
    model_config = bench_exp.bench_model_config(cfg)
    return grid, z_coord, state, model_config, dt, preset_name


def main(argv=None):
    args = _parse_args(argv)

    if args.no_x64:
        jax.config.update("jax_enable_x64", False)
    else:
        jax.config.update("jax_enable_x64", True)

    grid, z_coord, state0, model_config, dt, preset_name = _build_setup(args)

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    model = LatLonCGridOceanModel(grid, z_coord, model_config)

    n_lat = int(state0.T.data.shape[0])
    n_lon = int(state0.T.data.shape[1])
    nlev = int(state0.T.data.shape[2])
    n_cells = n_lat * n_lon * nlev

    fold_active = bool(getattr(getattr(grid, "fold", None), "is_active", False))

    print(
        f"BENCH: grid={args.grid} preset={preset_name} "
        f"{n_lat}x{n_lon}x{nlev} dt={dt:.0f}s steps={args.steps}"
    )
    print(
        f"       devices={jax.device_count()} "
        f"processes={jax.process_count()} "
        f"fold_active={fold_active}"
    )

    # No outer jax.jit wrapper: ``model.step`` is itself jitted
    # (``_step_jitted``), so the loop measures the production per-step
    # cost INCLUDING the eager Python overhead of the step shim (cache
    # checks etc.) — the same path run_dino.py times.
    step_fn = lambda s: model.step(s, dt=dt)  # noqa: E731

    step_times_ms: list[float] = []
    state = state0
    t_wall_start = time.time()

    for k in range(args.steps):
        t_step = time.perf_counter()
        state = step_fn(state)
        jax.block_until_ready([leaf for leaf in jax.tree.leaves(state) if leaf is not None])
        ms = (time.perf_counter() - t_step) * 1000.0
        step_times_ms.append(ms)
        if k == 0:
            print(f"[step 1] compile+step = {ms:.1f} ms (JIT compile)")
        elif (k + 1) % 50 == 0 or k == args.steps - 1:
            print(f"[step {k + 1}] {ms:.1f} ms")

    wall = time.time() - t_wall_start

    # Steady-state statistics (run_dino contract).
    n_warmup = min(args.warmup_steps, len(step_times_ms))
    steady = step_times_ms[n_warmup:]
    steady_mean = float(np.mean(steady)) if steady else 0.0
    steady_std = float(np.std(steady)) if steady else 0.0
    compile_ms = step_times_ms[0] if step_times_ms else 0.0
    sypd = (dt / (steady_mean / 1000.0)) / 365.25 if steady_mean > 0 else 0.0
    mcells_s = (n_cells / (steady_mean / 1000.0) / 1e6) if steady_mean > 0 else 0.0

    # Final-state stability gate (pre-registered; see bench.validate_results).
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    eta = np.asarray(state.eta.data)
    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
    # Pointwise max speed on cell centers: interpolate the C-grid
    # staggered velocities to T-points, then max(sqrt(u^2 + v^2)).
    # (sqrt(max u^2 + max v^2) would overestimate by pairing maxima from
    # different points; max(max|u|, max|v|) would underestimate by
    # dropping the cross term.)
    u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    max_speed = float(np.max(np.sqrt(u_cc**2 + v_cc**2)))
    max_eta = float(np.max(np.abs(eta)))
    finite = bool(
        np.all(np.isfinite(u))
        and np.all(np.isfinite(v))
        and np.all(np.isfinite(eta))
        and np.all(np.isfinite(T))
        and np.all(np.isfinite(S))
    )
    gates_ok = finite and max_speed <= 1.0 and max_eta <= 1.0

    print()
    print("=== BENCH TIMING SUMMARY ===")
    print(f"  Steps:         {args.steps} (warmup {n_warmup})")
    print(f"  Compile (s1):  {compile_ms:.1f} ms")
    print(f"  Steady mean:   {steady_mean:.1f} ms (std {steady_std:.1f} ms)")
    print(f"  SYPD:          {sypd:.1f}")
    print(f"  Mcells/s:      {mcells_s:.1f}")
    print(f"  Wall:          {wall:.1f} s")
    print("=== BENCH STABILITY GATES ===")
    print(f"  finite:        {finite}")
    print(f"  max speed:     {max_speed:.4f} m/s (gate 1.0)")
    print(f"  max |eta|:     {max_eta:.4f} m (gate 1.0)")
    print(f"  GATES:         {'PASS' if gates_ok else 'FAIL'}")

    record = {
        "script": "run_bench",
        "preset": preset_name,
        "grid_type": args.grid,
        "fold_active": fold_active,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_cells": int(n_cells),
        "n_levels": nlev,
        "dt": dt,
        "steps": args.steps,
        "n_warmup": n_warmup,
        "n_devices": int(jax.device_count()),
        "n_processes": int(jax.process_count()),
        "precision": "float64" if jax.config.jax_enable_x64 else "float32",
        "recipe": "legoesm_nemo_like_v1",
        "compile_ms": compile_ms,
        "steady_mean_ms": steady_mean,
        "steady_std_ms": steady_std,
        "sypd": sypd,
        "mcells_s": mcells_s,
        "total_wall_s": wall,
        "step_times_ms": step_times_ms,
        "gates_pass": gates_ok,
        "max_speed_final": max_speed,
        "max_eta_final": max_eta,
    }
    out = Path(args.timing_jsonl)
    with out.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"Timing record appended to {out}")

    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
