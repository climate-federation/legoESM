#!/usr/bin/env python
"""Run DCMIP-2012 3D transport test cases.

Usage:
    python scripts/run_dcmip_transport.py --test 11 --resolution 16 --levels 30
    python scripts/run_dcmip_transport.py --test 12 --resolution 24 --levels 30 --dt 600
    python scripts/run_dcmip_transport.py --test 13 --resolution 16 --levels 30

Tests:
    11: 3D Deformational Flow (Nair-Lauritzen, 12 days)
    12: Hadley-like Circulation (1 day)
    13: Horizontal Advection over Orography (12 days)
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.tracer_transport import (
    TracerTransportModel,
    TracerTransportConfig,
)
from tests.test_cases.dcmip_transport import (
    dcmip11_wind,
    dcmip11_init,
    dcmip12_wind,
    dcmip12_init,
    dcmip13_wind,
    dcmip13_init,
    compute_tracer_error_norms,
    create_dcmip_sigma,
)


TEST_CONFIGS = {
    11: {
        "name": "3D Deformational Flow (Nair-Lauritzen)",
        "period_days": 12.0,
        "default_dt": 1800.0,
        "wind_fn": dcmip11_wind,
        "init_fn": dcmip11_init,
        "n_tracers": 4,
    },
    12: {
        "name": "Hadley-like Meridional Circulation",
        "period_days": 1.0,
        "default_dt": 600.0,
        "wind_fn": dcmip12_wind,
        "init_fn": dcmip12_init,
        "n_tracers": 1,
    },
    13: {
        "name": "Horizontal Advection over Orography",
        "period_days": 12.0,
        "default_dt": 1800.0,
        "wind_fn": dcmip13_wind,
        "init_fn": dcmip13_init,
        "n_tracers": 4,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run DCMIP-2012 3D transport test cases",
    )
    parser.add_argument("--test", type=int, required=True, choices=[11, 12, 13],
                        help="Test case number (11, 12, or 13)")
    parser.add_argument("--resolution", "-n", type=int, default=16,
                        help="Grid resolution (cells per face edge)")
    parser.add_argument("--levels", "-l", type=int, default=30,
                        help="Number of vertical levels")
    parser.add_argument("--dt", type=float, default=None,
                        help="Time step in seconds (default depends on test)")
    parser.add_argument("--hyperdiff", type=float, default=0.0,
                        help="Hyperdiffusion coefficient")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    tc = TEST_CONFIGS[args.test]
    dt = args.dt or tc["default_dt"]
    duration = tc["period_days"] * 86400.0
    n_steps = int(duration / dt)

    print(f"legoESM v0.1.0 | DCMIP Test 1-{args.test % 10}: {tc['name']}")
    print(f"  Resolution: C{args.resolution}")
    print(f"  Levels: {args.levels}")
    print(f"  Duration: {tc['period_days']} days ({n_steps} steps)")
    print(f"  Time step: {dt} s")
    print(f"  Tracers: {tc['n_tracers']}")
    print(f"  Hyperdiffusion: {args.hyperdiff}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print()

    # Create grids
    print("Creating grid...", end=" ", flush=True)
    grid = create_cubed_sphere(args.resolution)
    sigma_coord = create_dcmip_sigma(args.levels)
    print(f"done. ({grid.n_cells} cells x {args.levels} levels)")

    # Initial condition
    print("Initializing tracers...", end=" ", flush=True)
    state_init = tc["init_fn"](grid, sigma_coord)
    print("done.")

    # Print initial tracer stats
    q0 = state_init.tracers.data
    for i in range(tc["n_tracers"]):
        qi = q0[..., i]
        print(f"  q{i+1}: min={float(jnp.min(qi)):.4f}, "
              f"max={float(jnp.max(qi)):.4f}, "
              f"mean={float(jnp.mean(qi)):.4f}")

    # Create model
    config = TracerTransportConfig(hyperdiff_coeff=args.hyperdiff)
    model = TracerTransportModel(grid, sigma_coord, tc["wind_fn"], config)

    # Integrate
    diag_interval = max(1, n_steps // 20)
    print(f"\nIntegrating {n_steps} steps...")

    state = state_init
    t_start = time.time()

    for i in range(n_steps):
        state = model.step(state, dt)

        if (i + 1) % diag_interval == 0:
            q = state.tracers.data
            t_sim = float(state.time.data)
            progress = (i + 1) / n_steps * 100
            q1_min = float(jnp.min(q[..., 0]))
            q1_max = float(jnp.max(q[..., 0]))
            print(f"  Step {i+1:6d}/{n_steps} ({progress:5.1f}%) | "
                  f"t={t_sim/86400:.2f} days | "
                  f"q1: [{q1_min:.4f}, {q1_max:.4f}]")

    wall_time = time.time() - t_start
    print(f"\nCompleted in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")

    # Error norms (for flow-reversal tests, exact = initial condition)
    if args.test in (11, 12):
        norms = compute_tracer_error_norms(state, state_init, grid)
        print(f"\nError norms (exact = initial condition at t = T):")
        for i in range(tc["n_tracers"]):
            print(f"  q{i+1}: l1={float(norms['l1'][i]):.6e}, "
                  f"l2={float(norms['l2'][i]):.6e}, "
                  f"linf={float(norms['linf'][i]):.6e}")

    # Final tracer stats
    print(f"\nFinal tracer statistics:")
    qf = state.tracers.data
    for i in range(tc["n_tracers"]):
        qi = qf[..., i]
        print(f"  q{i+1}: min={float(jnp.min(qi)):.4f}, "
              f"max={float(jnp.max(qi)):.4f}, "
              f"mean={float(jnp.mean(qi)):.4f}")


if __name__ == "__main__":
    main()
