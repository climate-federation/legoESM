"""Command-line interface for legoESM.

Usage:
    legoesm run config.yaml
    legoesm test williamson --case 2 --resolution 48 --days 5
    legoesm benchmark --resolution 48 --n-steps 100
"""

from __future__ import annotations

import argparse
import sys
import time

import jax
import jax.numpy as jnp


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="legoesm",
        description="legoESM: A Differentiable Earth System Model",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run a simulation from config")
    run_parser.add_argument("config", help="Path to YAML configuration file")

    # --- test command ---
    test_parser = subparsers.add_parser("test", help="Run standard test cases")
    test_parser.add_argument("test_name", choices=["williamson"],
                             help="Test case name")
    test_parser.add_argument("--case", type=int, default=2,
                             help="Test case number (2 or 5)")
    test_parser.add_argument("--resolution", "-n", type=int, default=48,
                             help="Grid resolution (cells per face edge)")
    test_parser.add_argument("--days", type=float, default=5.0,
                             help="Integration time in days")
    test_parser.add_argument("--dt", type=float, default=600.0,
                             help="Time step in seconds")
    test_parser.add_argument("--output", "-o", type=str, default=None,
                             help="Output directory for plots")

    # --- benchmark command ---
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    bench_parser.add_argument("--resolution", "-n", type=int, default=48)
    bench_parser.add_argument("--n-steps", type=int, default=100)
    bench_parser.add_argument("--dt", type=float, default=600.0)
    bench_parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Shard across multiple devices (GPUs/TPUs) via face-parallel mesh",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


def cmd_run(args):
    """Run a simulation from a config file."""
    from legoesm.config import Config

    config = Config.from_yaml(args.config)
    print(f"legoESM v0.1.0 | Loaded config from {args.config}")
    print(f"  Model: {config.get('model.name')}")
    print(f"  Grid: {config.get('grid.type')} C{config.get('grid.resolution')}")
    print(f"  Duration: {config.get('time.duration_hours')} hours")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")

    # TODO: Build and run model from config
    print("\nFull run from config not yet implemented. Use 'legoesm test' for now.")


def cmd_test(args):
    """Run a Williamson test case."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import ShallowWaterModel, ShallowWaterConfig
    from legoesm.atmosphere.dynamics.williamson import (
        williamson_test2, williamson_test5, williamson_test2_exact,
        compute_error_norms,
    )
    from legoesm.core.conservation import compute_conservation_diagnostics

    print(f"legoESM v0.1.0 | Williamson Test Case {args.case}")
    print(f"  Resolution: C{args.resolution} (~{6.371229e3 / args.resolution:.0f} km)")
    print(f"  Duration: {args.days} days")
    print(f"  Time step: {args.dt} s")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print()

    # Create grid
    print("Creating cubed-sphere grid...", end=" ", flush=True)
    grid = create_cubed_sphere(args.resolution)
    print(f"done. ({grid.n_cells} cells)")

    # Create initial condition
    if args.case == 2:
        state = williamson_test2(grid)
    elif args.case == 5:
        state = williamson_test5(grid)
    else:
        print(f"Unknown test case: {args.case}")
        sys.exit(1)

    # Create model with hyperdiffusion for stability
    # Scale hyperdiffusion coefficient with grid spacing^4 for scale-selectivity
    mean_dx = float(jnp.mean(grid.dx))
    hyperdiff = 1e-4 * mean_dx**4 / args.dt  # CFL-scaled hyperdiffusion
    config = ShallowWaterConfig(
        hyperdiff_coeff=hyperdiff,
        use_conservation_fixer=True,
    )
    model = ShallowWaterModel(grid, config)

    # Integrate
    n_steps = int(args.days * 86400 / args.dt)
    diag_interval = max(1, n_steps // 20)  # ~20 diagnostic outputs

    print(f"Integrating {n_steps} steps...")
    diagnostics = [compute_conservation_diagnostics(state, grid)]

    t_start = time.time()
    for i in range(n_steps):
        state = model.step(state, args.dt)

        if (i + 1) % diag_interval == 0:
            diag = compute_conservation_diagnostics(state, grid)
            diagnostics.append(diag)
            progress = (i + 1) / n_steps * 100
            mass_err = abs(float(diag['total_mass'] - diagnostics[0]['total_mass']))
            energy_err = abs(float(diag['total_energy'] - diagnostics[0]['total_energy']))
            print(f"  Step {i+1:6d}/{n_steps} ({progress:5.1f}%) | "
                  f"Mass err: {mass_err:.2e} | Energy err: {energy_err:.2e}")

    wall_time = time.time() - t_start
    print(f"\nCompleted in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")

    # Error norms (Test 2 only)
    if args.case == 2:
        exact = williamson_test2_exact(grid, args.days * 86400)
        norms = compute_error_norms(state, exact, grid)
        print(f"\nError norms (height field):")
        print(f"  L1:   {norms['l1']:.6e}")
        print(f"  L2:   {norms['l2']:.6e}")
        print(f"  Linf: {norms['linf']:.6e}")

    # Conservation summary
    mass_0 = float(diagnostics[0]['total_mass'])
    mass_f = float(diagnostics[-1]['total_mass'])
    energy_0 = float(diagnostics[0]['total_energy'])
    energy_f = float(diagnostics[-1]['total_energy'])
    print(f"\nConservation:")
    print(f"  Mass:   initial={mass_0:.6e}, final={mass_f:.6e}, "
          f"relative change={(mass_f-mass_0)/mass_0:.2e}")
    print(f"  Energy: initial={energy_0:.6e}, final={energy_f:.6e}, "
          f"relative change={(energy_f-energy_0)/energy_0:.2e}")

    # Visualization
    if args.output:
        try:
            from legoesm.visualization.maps import (
                plot_global_field, plot_conservation_timeseries,
            )
            import os
            os.makedirs(args.output, exist_ok=True)

            plot_global_field(
                state.h, grid,
                title=f"Williamson Test {args.case}: Height field (Day {args.days})",
                cmap="RdYlBu_r",
                colorbar_label="h [m]",
                save_path=os.path.join(args.output, f"williamson{args.case}_height.png"),
            )
            plot_conservation_timeseries(
                diagnostics, dt=args.dt * diag_interval,
                title=f"Williamson Test {args.case}: Conservation",
                save_path=os.path.join(args.output, f"williamson{args.case}_conservation.png"),
            )
            print(f"\nPlots saved to {args.output}/")
        except ImportError:
            print("\nSkipping plots (matplotlib/cartopy not available)")


def cmd_benchmark(args):
    """Run a performance benchmark."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.shallow_water import ShallowWaterModel
    from legoesm.atmosphere.dynamics.williamson import williamson_test2

    print(f"legoESM v0.1.0 | Benchmark")
    print(f"  Resolution: C{args.resolution}")
    print(f"  Steps: {args.n_steps}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.devices()}")
    print()

    grid = create_cubed_sphere(args.resolution)
    model = ShallowWaterModel(grid)
    state = williamson_test2(grid)

    # Multi-GPU sharding
    if args.multi_gpu:
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree, replicate_pytree
        dev_config = create_device_mesh()
        print(f"  Sharding: {dev_config.n_devices} devices, "
              f"face-parallel on {dev_config.backend}")
        state = shard_pytree(state, dev_config)
        grid = replicate_pytree(grid, dev_config)

    # Warmup (JIT compilation)
    print("Warmup (JIT compilation)...", end=" ", flush=True)
    state_warm = model.step(state, args.dt)
    # Force evaluation
    jax.block_until_ready(state_warm.h.data)
    print("done.")

    # Timed run
    print(f"Running {args.n_steps} steps...", end=" ", flush=True)
    t_start = time.time()
    for i in range(args.n_steps):
        state = model.step(state, args.dt)
    jax.block_until_ready(state.h.data)
    wall_time = time.time() - t_start

    throughput = args.n_steps / wall_time
    sim_days_per_hour = throughput * args.dt / 86400 * 3600

    print(f"done.")
    print(f"\nResults:")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} steps/s")
    print(f"  Simulated days per wall-clock hour: {sim_days_per_hour:.1f}")
    print(f"  Time per step: {wall_time/args.n_steps*1000:.2f} ms")


if __name__ == "__main__":
    main()
