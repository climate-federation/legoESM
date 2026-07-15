"""Command-line interface for legoESM.

Usage:
    legoesm run config.yaml
    legoesm test williamson --case 2 --resolution 48 --days 5
    legoesm benchmark --resolution 48 --n-steps 100

Note: JAX is NOT imported at module scope.  The runtime bootstrap
(``legoesm.runtime.bootstrap``) is called inside each sub-command
*before* any JAX-heavy model code is imported, ensuring that X64,
backend, and precision policies are fully resolved first.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# Single-sourced version (importlib.metadata only — no JAX import at module scope).
from legoesm._version import __version__

logger = logging.getLogger("legoesm.cli")


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

    # --- reproduce command ---
    repro_parser = subparsers.add_parser(
        "reproduce",
        help="Re-run from a run manifest and optionally bit-check the result",
    )
    repro_parser.add_argument(
        "manifest",
        help="Path to a run_manifest.json (or the directory containing it)",
    )
    repro_parser.add_argument(
        "--check", action="store_true",
        help="Assert the rerun's final state_digest matches the manifest's "
             "(exit non-zero on mismatch)",
    )
    repro_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for the rerun (default: a fresh temp dir)",
    )

    # --- wizard command ---
    subparsers.add_parser(
        "wizard",
        help="Interactive wizard to configure and launch a run "
             "(needs the 'wizard' extra: pip install 'legoesm[wizard]')",
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
    elif args.command == "reproduce":
        cmd_reproduce(args)
    elif args.command == "wizard":
        cmd_wizard(args)


def cmd_wizard(args):
    """Launch the interactive configuration wizard.

    The wizard lives under ``scripts/experiment/`` (tooling, not installed
    source), so it is imported by path here — mirroring the ``cmd_test`` /
    ``init_experiment`` convention of inserting the repo dir onto ``sys.path``.
    """
    import pathlib
    exp_dir = pathlib.Path(__file__).parent.parent.parent / "scripts" / "experiment"
    if not exp_dir.is_dir():
        logger.error(
            "Cannot find scripts/experiment/ (run from a source checkout to use "
            "the wizard)."
        )
        sys.exit(1)
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    import wizard
    sys.exit(wizard.main())


def cmd_run(args):
    """Run a simulation from a config file."""
    from legoesm.config import Config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        config = Config.from_yaml(args.config)

        # --- Runtime bootstrap (before any JAX-heavy imports) ---
        from legoesm.runtime import bootstrap_from_yaml_config
        rc = bootstrap_from_yaml_config(config)

        logger.info(f"legoESM v{__version__} | Loaded config from {args.config}")
        logger.info(f"  Model: {config.get('model.name')}")
        logger.info(f"  Grid: {config.get('grid.type')} C{config.get('grid.resolution')}")
        logger.info(f"  Duration: {config.get('time.duration_hours')} hours")
        logger.info(f"  Backend: {rc.backend}")
        logger.info(f"  Devices: {rc.device_config.n_devices}")
        logger.info(f"  Distributed: {rc.distributed}")
        logger.info(f"  Precision: {rc.precision}")

        # Now safe to import JAX-heavy model code.
        from legoesm.driver.model_driver import ModelDriver
        from legoesm.driver.run_status import status_to_exit_code

        experiment_config = config.to_experiment_config()

        logger.info("Initializing model driver...")
        driver = ModelDriver(experiment_config)
        # The reproducibility run manifest (A1) is written inside driver.setup()
        # — rank-0 guarded and using the driver's *resolved* config (after
        # grid-type normalization and setup-time overrides), so it never races
        # under MPI and always matches the config the run actually uses.
        driver.setup()
        logger.info("Running simulation...")
        status = driver.run()
        logger.info(f"Simulation completed: {status}")
        # iter-109 (codex iter-104 MEDIUM-8): propagate
        # ModelDriver status to exit code so wrappers /
        # automation can detect BLOWUP via ``$?``.  Pre-iter-109
        # this CLI exited 0 even when ``status="BLOWUP at day 5"``.
        rc = status_to_exit_code(status)
        if rc != 0:
            logger.error(
                f"Simulation status '{status}' does not indicate "
                f"a clean run; exiting with code {rc}."
            )
            sys.exit(rc)
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        sys.exit(1)


def cmd_reproduce(args):
    """Re-run from a run manifest and optionally bit-check reproducibility."""
    import tempfile
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        from legoesm.driver.config import experiment_config_from_dict
        from legoesm.driver.restart import (
            read_run_manifest,
            recorded_state_digest,
            validate_run_manifest,
        )

        manifest = read_run_manifest(args.manifest)
        validate_run_manifest(manifest)

        # The reproduce rerun is wired for the atmosphere driver (ModelDriver).
        # An ocean manifest still validates (resolved_config reconstructs +
        # config_hash matches, checked above); its bit-identical rerun goes
        # through the ocean runner, not ModelDriver — say so plainly rather than
        # crashing in experiment_config_from_dict on ocean fields (#376).
        kind = manifest["config"].get("config_kind", "atmosphere")
        if kind != "atmosphere":
            logger.info(
                f"legoESM v{__version__} | manifest {args.manifest} VALIDATED "
                f"(config_kind={kind}; resolved_config reconstructs and "
                f"config_hash matches)."
            )
            logger.warning(
                "`legoesm reproduce` rerun supports the atmosphere driver only. "
                f"Re-run this {kind} manifest with its runner — e.g. "
                "`python scripts/run/run_omip_core2.py --config <config.yaml> "
                "--output <dir>` — then compare result.state_digest against the "
                "reference manifest."
            )
            # --check asks for a bit-identical rerun comparison. We did NOT rerun
            # or compare digests, so we must NOT exit 0 — a green here would let
            # CI treat an unperformed check as a passed check. Exit non-zero.
            if args.check:
                logger.error(
                    f"reproduce --check is not supported for {kind} manifests "
                    "(no driver rerun): the digest comparison was NOT performed. "
                    "Rerun via the component runner and compare result.state_digest."
                )
                sys.exit(2)
            return

        config = experiment_config_from_dict(manifest["config"]["resolved_config"])

        logger.info(f"legoESM v{__version__} | Reproducing run from {args.manifest}")
        logger.info(
            f"  Grid: {config.grid.grid_type} C{config.grid.resolution} | "
            f"days={config.days} | seed={config.seed}"
        )

        # The rerun records ITS digest into ITS output dir's manifest.  It must
        # not write into the reference run's directory, or it would overwrite the
        # reference digest we are checking against.
        # Resolve the manifest target FIRST (follows symlinks) so the reference
        # directory is the real run dir, not a symlink's containing dir.
        resolved_manifest = Path(args.manifest).resolve()
        reference_dir = (
            resolved_manifest if resolved_manifest.is_dir() else resolved_manifest.parent
        )
        out_dir = args.output or tempfile.mkdtemp(prefix="legoesm_reproduce_")
        if Path(out_dir).resolve() == reference_dir:
            logger.error(
                f"--output {out_dir} is the reference run's directory; the rerun "
                f"would overwrite the reference manifest. Use a fresh directory."
            )
            sys.exit(2)

        from legoesm.driver.model_driver import ModelDriver
        from legoesm.driver.run_status import status_to_exit_code

        driver = ModelDriver(config, output_dir=out_dir)
        driver.setup()
        status = driver.run()
        logger.info(f"  Rerun status: {status}")
        rc = status_to_exit_code(status)
        if rc != 0:
            logger.error(
                f"Rerun did not complete cleanly (status {status!r}); "
                f"cannot check reproducibility."
            )
            sys.exit(rc)

        if args.check:
            # --check needs both digests; missing either is a hard failure.
            reference = recorded_state_digest(manifest)
            rerun_digest = recorded_state_digest(read_run_manifest(driver.output_dir))
            if rerun_digest == reference:
                logger.info(
                    f"reproduce --check: MATCH — bit-identical "
                    f"(state_digest {reference[:16]}...)"
                )
            else:
                logger.error(
                    "reproduce --check: MISMATCH — run is NOT bit-reproducible\n"
                    f"  reference: {reference}\n"
                    f"  rerun:     {rerun_digest}"
                )
                sys.exit(1)
        else:
            # Plain reproduce just needs a clean rerun; the digest is a bonus
            # (best-effort recording, e.g. absent for some backends).
            try:
                rerun_digest = recorded_state_digest(read_run_manifest(driver.output_dir))
                logger.info(f"  Reproduced; final state_digest={rerun_digest}")
            except ValueError:
                logger.info(
                    "  Reproduced (no final state_digest was recorded for this run)."
                )
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Error during reproduce: {e}", exc_info=True)
        sys.exit(1)


def cmd_test(args):
    """Run a Williamson test case."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Bootstrap runtime before JAX-heavy imports.
    from legoesm.runtime import bootstrap
    rc = bootstrap(precision="fp32")

    # tests/ is not an installed package; add the project root so the import
    # works whether invoked via the console-script entry point or python -m.
    import pathlib
    _project_root = pathlib.Path(__file__).parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    import jax
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterModel as ShallowWaterModel,
        williamson_cli_calibration,
    )
    from tests.test_cases.williamson import (
        williamson_test2, williamson_test5, williamson_test2_exact,
        compute_error_norms,
    )
    from legoesm.core.conservation import compute_conservation_diagnostics
    from legoesm import constants

    _earth_radius_km = constants.R_earth / 1000.0
    logger.info(f"legoESM v{__version__} | Williamson Test Case {args.case}")
    logger.info(f"  Resolution: C{args.resolution} (~{_earth_radius_km / args.resolution:.0f} km)")
    logger.info(f"  Duration: {args.days} days")
    logger.info(f"  Time step: {args.dt} s")
    logger.info(f"  Backend: {rc.backend}")
    logger.info(f"  Devices: {jax.devices()}")

    # Create grid
    logger.info("Creating cubed-sphere grid...")
    grid = create_cubed_sphere(args.resolution)
    logger.info(f"done. ({grid.n_cells} cells)")

    # Create initial condition
    if args.case == 2:
        sw_state = williamson_test2(grid)
    elif args.case == 5:
        sw_state = williamson_test5(grid)
    else:
        logger.error(f"Unknown test case: {args.case}")
        sys.exit(1)

    # Convert cell-centre ShallowWaterState -> CDGridShallowWaterState
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterState, create_cubed_sphere_cdgrid,
    )
    _cdgrid_tmp = create_cubed_sphere_cdgrid(grid)
    u_cc = sw_state.u.data if hasattr(sw_state.u, 'data') else sw_state.u
    v_cc = sw_state.v.data if hasattr(sw_state.v, 'data') else sw_state.v
    h    = sw_state.h.data if hasattr(sw_state.h, 'data') else sw_state.h
    h_s  = sw_state.h_s.data if hasattr(sw_state.h_s, 'data') else sw_state.h_s
    u_d, v_d = center_to_dgrid_vector(u_cc, v_cc, _cdgrid_tmp)
    state = CDGridShallowWaterState(h=h, u_d=u_d, v_d=v_d, h_s=h_s)

    # Use the production-calibrated cubed-sphere shallow-water knobs
    # (see #269).  Earlier CLI defaults used a heuristic
    # ``1e-4 * mean_dx**4 / dt`` hyperdiffusion with no boundary fix,
    # divergence damping, or vorticity damping, which left visible
    # O(dx) v-wind streaks on one cube panel at C48 + 5 days even
    # though L2 error was small.  ``williamson_cli_calibration`` is
    # the single source of truth that the matrix runner and tests also
    # use, so future calibration updates flow through one place.
    config = williamson_cli_calibration(int(args.resolution))
    model = ShallowWaterModel(grid, config)

    # Integrate
    n_steps = int(args.days * 86400 / args.dt)
    diag_interval = max(1, n_steps // 20)  # ~20 diagnostic outputs

    logger.info(f"Integrating {n_steps} steps...")
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    from legoesm.core.state import ShallowWaterState
    from legoesm.core.field import Field as _Field

    def _to_sw_state(cdgrid_state):
        _h  = cdgrid_state.h.data  if hasattr(cdgrid_state.h,   'data') else cdgrid_state.h
        _hs = cdgrid_state.h_s.data if hasattr(cdgrid_state.h_s, 'data') else cdgrid_state.h_s
        _ud = cdgrid_state.u_d.data if hasattr(cdgrid_state.u_d, 'data') else cdgrid_state.u_d
        _vd = cdgrid_state.v_d.data if hasattr(cdgrid_state.v_d, 'data') else cdgrid_state.v_d
        u_cc, v_cc = dgrid_to_center_vector(_ud, _vd)
        dims = ("face", "x", "y")
        return ShallowWaterState(
            h=_Field(data=_h,   name="h",   dims=dims, units="m",   long_name="Fluid depth"),
            u=_Field(data=u_cc, name="u",   dims=dims, units="m/s", long_name="Zonal velocity"),
            v=_Field(data=v_cc, name="v",   dims=dims, units="m/s", long_name="Meridional velocity"),
            h_s=_Field(data=_hs, name="h_s", dims=dims, units="m",  long_name="Topography"),
        )

    diagnostics = [compute_conservation_diagnostics(_to_sw_state(state), grid)]

    t_start = time.time()
    for i in range(n_steps):
        state = model.step(state, args.dt)

        if (i + 1) % diag_interval == 0:
            diag = compute_conservation_diagnostics(_to_sw_state(state), grid)
            diagnostics.append(diag)
            progress = (i + 1) / n_steps * 100
            mass_err = abs(float(diag['total_mass'] - diagnostics[0]['total_mass']))
            energy_err = abs(float(diag['total_energy'] - diagnostics[0]['total_energy']))
            logger.info(f"  Step {i+1:6d}/{n_steps} ({progress:5.1f}%) | "
                  f"Mass err: {mass_err:.2e} | Energy err: {energy_err:.2e}")

    wall_time = time.time() - t_start
    logger.info(f"Completed in {wall_time:.1f}s ({n_steps/wall_time:.0f} steps/s)")

    # Error norms (Test 2 only)
    if args.case == 2:
        exact = williamson_test2_exact(grid, args.days * 86400)
        norms = compute_error_norms(_to_sw_state(state), exact, grid)
        logger.info("Error norms (height field):")
        logger.info(f"  L1:   {norms['l1']:.6e}")
        logger.info(f"  L2:   {norms['l2']:.6e}")
        logger.info(f"  Linf: {norms['linf']:.6e}")

    # Conservation summary
    mass_0 = float(diagnostics[0]['total_mass'])
    mass_f = float(diagnostics[-1]['total_mass'])
    energy_0 = float(diagnostics[0]['total_energy'])
    energy_f = float(diagnostics[-1]['total_energy'])
    logger.info("Conservation:")
    logger.info(f"  Mass:   initial={mass_0:.6e}, final={mass_f:.6e}, "
          f"relative change={(mass_f-mass_0)/mass_0:.2e}")
    logger.info(f"  Energy: initial={energy_0:.6e}, final={energy_f:.6e}, "
          f"relative change={(energy_f-energy_0)/energy_0:.2e}")

    # Visualization
    if args.output:
        try:
            from legoesm.visualization.maps import (
                plot_global_field, plot_conservation_timeseries,
                plot_dgrid_winds_per_tile,
            )
            import os
            import numpy as np
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

            # Issue #274: emit final winds.  The PlateCarree plots show
            # geographic east/north components, so they go through
            # ``dgrid_to_center_geographic`` (rotates BEFORE the corner
            # → cell-centre 4-point average).  Skipping the rotation —
            # e.g. via the face-local ``dgrid_to_center_vector`` /
            # ``_to_sw_state`` path — mislabels face-aligned components
            # as zonal/meridional and produces spurious O(25 m/s)
            # ``v_north`` for a purely zonal initial condition.  The
            # native D-grid arrays then go to a per-tile panel since
            # no canonical (lon, lat) is defined at the D-grid corners.
            from legoesm.core.operators_cdgrid import dgrid_to_center_geographic
            from legoesm.core.field import Field as _Field
            u_east, v_north = dgrid_to_center_geographic(
                state.u_d, state.v_d, _cdgrid_tmp,
            )
            u_east_field = _Field(
                data=u_east, name="u_east", dims=("face", "x", "y"),
                units="m/s", long_name="Zonal (eastward) wind",
            )
            v_north_field = _Field(
                data=v_north, name="v_north", dims=("face", "x", "y"),
                units="m/s", long_name="Meridional (northward) wind",
            )
            wind_lim = float(np.nanmax(np.abs(np.asarray(u_east))))
            wind_lim = max(
                wind_lim,
                float(np.nanmax(np.abs(np.asarray(v_north)))),
            )
            wind_lim = wind_lim if wind_lim > 0 else 1.0
            plot_global_field(
                u_east_field, grid,
                title=f"Williamson Test {args.case}: zonal wind u (Day {args.days})",
                cmap="RdBu_r",
                vmin=-wind_lim, vmax=wind_lim,
                projection="platecarree",
                colorbar_label="u [m/s]",
                save_path=os.path.join(args.output, f"williamson{args.case}_u_platecarree.png"),
            )
            plot_global_field(
                v_north_field, grid,
                title=f"Williamson Test {args.case}: meridional wind v (Day {args.days})",
                cmap="RdBu_r",
                vmin=-wind_lim, vmax=wind_lim,
                projection="platecarree",
                colorbar_label="v [m/s]",
                save_path=os.path.join(args.output, f"williamson{args.case}_v_platecarree.png"),
            )
            plot_dgrid_winds_per_tile(
                state.u_d, state.v_d,
                title=(
                    f"Williamson Test {args.case}: native D-grid winds "
                    f"(Day {args.days})"
                ),
                save_path=os.path.join(
                    args.output, f"williamson{args.case}_dgrid_winds.png"
                ),
            )
            logger.info(f"Plots saved to {args.output}/")
        except ImportError:
            logger.info("Skipping plots (matplotlib/cartopy not available)")


def cmd_benchmark(args):
    """Run a performance benchmark."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Bootstrap runtime before JAX-heavy imports.
    from legoesm.runtime import bootstrap
    rc = bootstrap(precision="fp32")

    import jax
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import CDGridShallowWaterModel as ShallowWaterModel
    from tests.test_cases.williamson import williamson_test2

    logger.info(f"legoESM v{__version__} | Benchmark")
    logger.info(f"  Resolution: C{args.resolution}")
    logger.info(f"  Steps: {args.n_steps}")
    logger.info(f"  Backend: {rc.backend}")
    logger.info(f"  Devices: {jax.devices()}")

    grid = create_cubed_sphere(args.resolution)
    model = ShallowWaterModel(grid)
    state = williamson_test2(grid)

    # Multi-GPU sharding
    if args.multi_gpu:
        from legoesm.parallel.mesh import create_device_mesh, shard_pytree, replicate_pytree
        dev_config = create_device_mesh()
        logger.info(f"  Sharding: {dev_config.n_devices} devices, "
              f"face-parallel on {dev_config.backend}")
        state = shard_pytree(state, dev_config)
        grid = replicate_pytree(grid, dev_config)

    # Warmup (JIT compilation)
    logger.info("Warmup (JIT compilation)...")
    state_warm = model.step(state, args.dt)
    # Force evaluation
    jax.block_until_ready(state_warm.h.data)
    logger.info("done.")

    # Timed run
    logger.info(f"Running {args.n_steps} steps...")
    t_start = time.time()
    for i in range(args.n_steps):
        state = model.step(state, args.dt)
    jax.block_until_ready(state.h.data)
    wall_time = time.time() - t_start

    throughput = args.n_steps / wall_time
    sim_days_per_hour = throughput * args.dt / 86400 * 3600

    logger.info("done.")
    logger.info("Results:")
    logger.info(f"  Wall time: {wall_time:.2f}s")
    logger.info(f"  Throughput: {throughput:.1f} steps/s")
    logger.info(f"  Simulated days per wall-clock hour: {sim_days_per_hour:.1f}")
    logger.info(f"  Time per step: {wall_time/args.n_steps*1000:.2f} ms")


if __name__ == "__main__":
    main()
