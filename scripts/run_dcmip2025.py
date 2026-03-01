#!/usr/bin/env python
"""Run DCMIP-2025 non-hydrostatic test cases with CompressibleEulerModel.

Test cases:
  TC1: Mountain-triggered gravity waves (dry, ~3h simulated)
  TC2a: Gap flow through mountain chain (dry, small-Earth, ~6h simulated)
  TC3: Squall line (moist, small-Earth, ~2h simulated)

Usage:
    python scripts/run_dcmip2025.py --test tc1 --resolution 16 --levels 20
    python scripts/run_dcmip2025.py --test tc2a --resolution 16 --levels 20
    python scripts/run_dcmip2025.py --test tc3 --resolution 16 --levels 20
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def run_tc1(resolution, n_levels, dt, duration_hours, output_dir):
    """Run DCMIP-2025 TC1: Mountain gravity waves."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel,
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.dcmip2025 import dcmip25_tc1_init

    print("  Creating grid and initial conditions...")
    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric = dcmip25_tc1_init(
        grid, n_levels=n_levels,
    )

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=10000.0,
        sponge_coeff=0.05,
    )
    model = CompressibleEulerModel(grid, height_coord, terrain_metric, config)

    duration = duration_hours * 3600.0
    n_steps = int(duration / dt)
    diag_every = max(1, n_steps // 20)

    print(f"  Integrating for {n_steps} steps ({duration_hours}h)...")

    # JIT warmup
    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT compiled in {time.time() - t0:.1f}s")

    max_w_list = []
    t_start = time.time()

    for i in range(1, n_steps):
        state = model.step(state, dt)

        if (i + 1) % diag_every == 0:
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            max_w_list.append(w_max)
            t_sim = (i + 1) * dt / 3600.0
            print(f"    t={t_sim:.2f}h | max|w|={w_max:.4f} m/s")

            if not jnp.all(jnp.isfinite(state.w.data)) or w_max > 100:
                print(f"    *** BLOWUP at step {i+1} ***")
                break

    wall_time = time.time() - t_start

    # Final diagnostics
    w_max_final = float(jnp.max(jnp.abs(state.w.data)))
    u_max = float(jnp.max(jnp.abs(state.u.data)))
    stable = jnp.all(jnp.isfinite(state.u.data)) and u_max < 500

    results = {
        "test": "DCMIP-2025 TC1",
        "resolution": f"C{resolution}",
        "levels": n_levels,
        "duration_hours": duration_hours,
        "dt": dt,
        "wall_time": wall_time,
        "max_w": w_max_final,
        "max_u": u_max,
        "stable": bool(stable),
        "status": "PASS" if stable else "FAIL",
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results


def run_tc2a(resolution, n_levels, dt, duration_hours, output_dir):
    """Run DCMIP-2025 TC2a: Gap flow (small Earth)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel,
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.dcmip2025 import dcmip25_tc2_init

    print("  Creating grid and initial conditions...")
    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc2_init(
        grid, n_levels=n_levels, subcase="a",
    )

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=15000.0,
        sponge_coeff=1.0 / (0.1 * 86400.0),
        small_earth_factor=20.0,
    )
    model = CompressibleEulerModel(
        small_grid, height_coord, terrain_metric, config,
    )

    duration = duration_hours * 3600.0
    n_steps = int(duration / dt)
    diag_every = max(1, n_steps // 20)

    print(f"  Integrating for {n_steps} steps ({duration_hours}h)...")

    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT compiled in {time.time() - t0:.1f}s")

    t_start = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)

        if (i + 1) % diag_every == 0:
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            u_max = float(jnp.max(jnp.abs(state.u.data)))
            t_sim = (i + 1) * dt / 3600.0
            print(f"    t={t_sim:.2f}h | max|w|={w_max:.4f} | max|u|={u_max:.2f}")

            if not jnp.all(jnp.isfinite(state.u.data)) or u_max > 500:
                print(f"    *** BLOWUP at step {i+1} ***")
                break

    wall_time = time.time() - t_start
    u_max = float(jnp.max(jnp.abs(state.u.data)))
    stable = jnp.all(jnp.isfinite(state.u.data)) and u_max < 500

    results = {
        "test": "DCMIP-2025 TC2a",
        "resolution": f"C{resolution}",
        "levels": n_levels,
        "duration_hours": duration_hours,
        "dt": dt,
        "wall_time": wall_time,
        "max_w": float(jnp.max(jnp.abs(state.w.data))),
        "max_u": u_max,
        "stable": bool(stable),
        "status": "PASS" if stable else "FAIL",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results


def run_tc3(resolution, n_levels, dt, duration_hours, output_dir):
    """Run DCMIP-2025 TC3: Squall line (moist, small Earth)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.compressible_euler import (
        CompressibleEulerModel,
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.dcmip2025 import dcmip25_tc3_init

    print("  Creating grid and initial conditions...")
    grid = create_cubed_sphere(resolution)
    state, height_coord, terrain_metric, small_grid = dcmip25_tc3_init(
        grid, n_levels=n_levels,
    )

    config = CompressibleEulerConfig(
        n_acoustic_substeps=6,
        sponge_width=5000.0,
        sponge_coeff=0.05,
        small_earth_factor=60.0,
    )
    model = CompressibleEulerModel(
        small_grid, height_coord, terrain_metric, config,
    )

    duration = duration_hours * 3600.0
    n_steps = int(duration / dt)
    diag_every = max(1, n_steps // 20)

    print(f"  Integrating for {n_steps} steps ({duration_hours}h)...")

    t0 = time.time()
    state = model.step(state, dt)
    jax.block_until_ready(state.u.data)
    print(f"  JIT compiled in {time.time() - t0:.1f}s")

    t_start = time.time()
    for i in range(1, n_steps):
        state = model.step(state, dt)

        if (i + 1) % diag_every == 0:
            w_max = float(jnp.max(jnp.abs(state.w.data)))
            t_sim = (i + 1) * dt / 3600.0

            # Tracer diagnostics
            q = state.tracers.data
            q_rain_max = float(jnp.max(q[..., 2])) if q.shape[-1] >= 3 else 0.0

            print(f"    t={t_sim:.2f}h | max|w|={w_max:.4f} | max qr={q_rain_max:.6f}")

            if not jnp.all(jnp.isfinite(state.u.data)):
                print(f"    *** BLOWUP at step {i+1} ***")
                break

    wall_time = time.time() - t_start
    w_max_final = float(jnp.max(jnp.abs(state.w.data)))
    stable = bool(jnp.all(jnp.isfinite(state.u.data)))

    results = {
        "test": "DCMIP-2025 TC3",
        "resolution": f"C{resolution}",
        "levels": n_levels,
        "duration_hours": duration_hours,
        "dt": dt,
        "wall_time": wall_time,
        "max_w": w_max_final,
        "stable": stable,
        "status": "PASS" if stable else "FAIL",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="DCMIP-2025 test cases")
    parser.add_argument("--test", type=str, default="tc1",
                        choices=["tc1", "tc2a", "tc3"],
                        help="Test case (tc1, tc2a, tc3)")
    parser.add_argument("--resolution", "-n", type=int, default=16)
    parser.add_argument("--levels", "-l", type=int, default=20)
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument(
        "--x64",
        action="store_true",
        help="Enable float64 mode. Off by default for better Metal/FV performance.",
    )
    args = parser.parse_args()

    if args.x64:
        jax.config.update("jax_enable_x64", True)

    defaults = {"tc1": 3.0, "tc2a": 6.0, "tc3": 2.0}
    hours = args.hours or defaults[args.test]
    out = Path(args.output or f"results/dcmip2025_{args.test}_C{args.resolution}")

    print(f"DCMIP-2025 {args.test.upper()} | C{args.resolution} L{args.levels}")

    if args.test == "tc1":
        run_tc1(args.resolution, args.levels, args.dt, hours, out)
    elif args.test == "tc2a":
        run_tc2a(args.resolution, args.levels, args.dt, hours, out)
    elif args.test == "tc3":
        run_tc3(args.resolution, args.levels, args.dt, hours, out)


if __name__ == "__main__":
    main()
