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
    repro_parser.add_argument(
        "--allow-version-drift", action="store_true",
        help="Rerun even when the imported legoESM code may differ from the "
             "code that wrote the manifest (other commit, uncommitted changes, "
             "or patches). Default: refuse.",
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
            RESOLVED_PHYSICS_KEYS,
            RUNNER_REPLAY_SPECS,
            config_drift_rows,
            format_drift_rows,
            manifest_version_drift,
            physics_config_from_record,
            read_run_manifest,
            recorded_state_digest,
            resolved_config_record,
            validate_run_manifest,
            version_drift_allowed,
        )

        manifest = read_run_manifest(args.manifest)
        validate_run_manifest(manifest)

        # Component runners (ocean / land / DINO / coupled): rerun the
        # recording script and verify it resolves to the same record.
        kind = manifest["config"].get("config_kind", "atmosphere")
        tag = (manifest.get("reproducibility") or {}).get("runner_tag", "")
        if kind == "atmosphere" and tag in RUNNER_REPLAY_SPECS:
            logger.error(f"Refusing to reproduce: an atmosphere manifest "
                         f"recorded by the {tag} runner is ambiguous.")
            sys.exit(2)
        if kind != "atmosphere" and tag not in RUNNER_REPLAY_SPECS:
            # No replay for this runner yet: validate only (unchanged).
            logger.info(
                f"legoESM v{__version__} | manifest {args.manifest} VALIDATED "
                f"(config_kind={kind}; resolved_config reconstructs and "
                f"config_hash matches)."
            )
            logger.warning(
                f"`legoesm reproduce` cannot rerun {tag or kind!r} manifests "
                f"yet (supported runners: {sorted(RUNNER_REPLAY_SPECS)}). "
                "Re-run it with its runner, then compare result.state_digest "
                "against the reference manifest."
            )
            # --check asks for a bit-identical rerun comparison. We did NOT
            # rerun or compare digests, so we must NOT exit 0.
            if args.check:
                logger.error(
                    f"reproduce --check is not supported for {tag or kind!r} "
                    "manifests (no rerun): the digest comparison was NOT "
                    "performed.")
                sys.exit(2)
            return
        if kind != "atmosphere":
            drift = manifest_version_drift(manifest)
            allowed = (getattr(args, "allow_version_drift", False)
                       or version_drift_allowed())
            if drift:
                logger.warning(
                    "Version drift: " + "; ".join(drift) + "\nThe rerun "
                    "refuses any configuration difference, but code that is "
                    "not a config field may have changed.")
                if not allowed:
                    logger.error("Refusing to reproduce across version drift; "
                                 "check out the recorded commit, or pass "
                                 "--allow-version-drift (or set "
                                 "LEGOESM_ALLOW_VERSION_DRIFT=1).")
                    sys.exit(2)
            sys.exit(_reproduce_runner(args, manifest, allowed))

        stored = manifest["config"]["resolved_config"]
        config = experiment_config_from_dict(stored)

        rows = config_drift_rows(stored, resolved_config_record(config))
        drift = manifest_version_drift(manifest)
        if drift:
            lines = [f"  {k}: stored={a!r} -> rebuilt={b!r}" for k, a, b in rows]
            logger.warning(
                "Version drift: " + "; ".join(drift) + "\n"
                f"Resolved-config fields that differ after rebuilding with "
                f"today's code ({len(rows)}):\n" + ("\n".join(lines) or "  (none)")
                + "\nAn empty table does NOT mean the same science: code that "
                "is not a config field may have changed."
            )
            if not (getattr(args, "allow_version_drift", False)
                    or version_drift_allowed()):
                logger.error("Refusing to reproduce across version drift; "
                             "check out the recorded commit, or pass "
                             "--allow-version-drift (or set "
                             "LEGOESM_ALLOW_VERSION_DRIFT=1).")
                sys.exit(2)

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
        if args.output and Path(args.output).resolve() == reference_dir:
            logger.error(
                f"--output {args.output} is the reference run's directory; the rerun "
                f"would overwrite the reference manifest. Use a fresh directory."
            )
            sys.exit(2)

        # Strict replay: today's code must rebuild the recorded configuration
        # exactly -- no field dropped, filled from a new default, or rewritten
        # -- even when version drift is allowed.
        if rows:
            logger.error(
                f"Refusing to reproduce: today's code does not rebuild the "
                f"recorded configuration exactly ({len(rows)} fields):\n"
                + format_drift_rows(rows))
            sys.exit(2)
        recorded_physics = manifest["config"].get("resolved_physics")
        if not (isinstance(recorded_physics, dict)
                and set(recorded_physics) == set(RESOLVED_PHYSICS_KEYS)):
            logger.error(
                "Refusing to reproduce: the manifest has no resolved-physics "
                "record (config.resolved_physics) with the expected entries, "
                "so the scheme parameters the run used cannot be checked.")
            sys.exit(2)
        # The full overrides are not kept in resolved_config; replay them from
        # the physics record (the closure check then covers them too).
        try:
            overrides = {k: recorded_physics[k] for k in
                         ("turbulence_override", "gravity_wave_drag_override")}
            for k, v in overrides.items():
                if not (v is None or (isinstance(v, dict) and "__type__" in v)):
                    raise ValueError(f"{k} is not a tagged config record")
            config = config._replace(**{
                k: physics_config_from_record(v) for k, v in overrides.items()})
        except (ValueError, TypeError, AttributeError, ImportError) as exc:
            logger.error(f"Refusing to reproduce: cannot replay a recorded "
                         f"override: {exc}")
            sys.exit(2)

        from legoesm.driver.model_driver import ModelDriver
        from legoesm.driver.run_status import status_to_exit_code

        out_dir = args.output or tempfile.mkdtemp(prefix="legoesm_reproduce_")
        driver = ModelDriver(config, output_dir=out_dir,
                             expected_resolved_physics=recorded_physics)
        driver.setup()
        # Pipeline lanes (rollout recorded): rerun in the recorded rollout and
        # reapply the scheme configs a driver replaced after setup.
        rollout = recorded_physics["rollout"]
        if rollout is not None:
            try:
                driver.apply_recorded_pipeline_physics(recorded_physics)
            except (ValueError, TypeError, AttributeError, ImportError) as exc:
                logger.error(f"Refusing to reproduce: cannot reapply the "
                             f"recorded pipeline schemes: {exc}")
                sys.exit(2)
        status = driver.run(compiled=rollout != "per_step")
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


def _reproduce_runner(args, manifest: dict, allow_drift: bool) -> int:
    """Rerun a component-runner manifest; return the exit code.

    The recorded command line is rerun from the recorded launch directory with
    only the output directory rewritten.  The rerun's own manifest writer
    refuses before stepping unless it resolves to the reference record
    (``legoesm.driver.restart._check_reproduce_reference``).
    """
    import os
    import shlex
    import subprocess
    import tempfile
    from pathlib import Path

    from legoesm.driver import restart as R

    spec = R.RUNNER_REPLAY_SPECS[manifest["reproducibility"]["runner_tag"]]
    resolved = manifest["config"].get("resolved_config") or {}

    def recorded(dotted):
        node = resolved
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        return node

    if args.check:
        no_digest = [k for k in spec.no_digest_keys if recorded(k)]
        if no_digest:
            logger.error(f"Refusing to reproduce --check: this run records no "
                         f"final digest ({', '.join(no_digest)} is set).")
            return 2
        try:
            reference = R.recorded_state_digest(manifest)
        except ValueError as exc:
            logger.error(f"Refusing to reproduce --check: {exc}")
            return 2

    run = manifest["run"]
    argv, cwd = run.get("argv"), run.get("cwd")
    if not (isinstance(argv, list) and argv
            and all(isinstance(a, str) for a in argv)):
        logger.error("Refusing to reproduce: the manifest does not record the "
                     "command line (run.argv), so the run cannot be replayed.")
        return 2
    if Path(argv[0]).name != spec.script:
        logger.error(f"Refusing to reproduce: the run was launched as "
                     f"{argv[0]!r}, not as the {spec.script} runner script, so "
                     f"rerunning that script would not repeat it.")
        return 2
    if not (isinstance(cwd, str) and Path(cwd).is_dir()):
        logger.error(f"Refusing to reproduce: the recorded launch directory "
                     f"{cwd!r} does not exist (relative paths in the command "
                     f"line resolve against it).")
        return 2
    try:
        script = R.runner_script_path(spec.script)
    except FileNotFoundError as exc:
        logger.error(f"Refusing to reproduce: {exc}")
        return 2
    # Run the recorded script itself when it is still there unchanged (same
    # import resolution); otherwise this checkout's copy, if identical.
    recorded_sha = run.get("script_sha256")
    if not recorded_sha:
        logger.error(f"Refusing to reproduce: the manifest does not record the "
                     f"launched script's content (run.script_sha256), so a "
                     f"rerun cannot be shown to repeat it.")
        return 2
    launched = Path(cwd) / argv[0]
    if launched.is_file() and R.file_sha256(launched) == recorded_sha:
        script = launched.resolve()
    elif R.file_sha256(script) != recorded_sha:
        logger.error(f"Refusing to reproduce: the recorded script "
                     f"({argv[0]}) differs from the {script} this rerun would "
                     f"run.")
        return 2
    ref = Path(args.manifest).resolve()
    if ref.is_dir():
        ref = ref / R.RUN_MANIFEST_FILENAME
    ref_root = ref.parent
    for _ in range(spec.manifest_depth):
        ref_root = ref_root.parent
    # Where the rerun writes its manifest: the runner's own file name, not the
    # reference's (which may be a renamed or archived copy).
    rel = ref.parent.relative_to(ref_root) / R.RUN_MANIFEST_FILENAME

    # The reference run's output: where its manifest sits now, and where the
    # run wrote it (a copied or archived manifest sits elsewhere).
    ref_roots = [ref_root]
    recorded_out = recorded(spec.rewritten_keys[0])
    if isinstance(recorded_out, str) and recorded_out:
        ref_roots.append((Path(cwd) / recorded_out).resolve())

    def _overlaps(new_root: Path) -> bool:
        for root in ref_roots:
            if (new_root == root or root in new_root.parents
                    or new_root in root.parents):
                logger.error(f"Refusing to reproduce: output {new_root} "
                             f"overlaps the reference run's output {root}; use "
                             f"a fresh directory outside it.")
                return True
        return False

    if args.output:
        new_root = Path(args.output).resolve()
        if _overlaps(new_root):
            return 2
        if (new_root / rel).exists():
            logger.error(f"Refusing to reproduce: --output {new_root} already "
                         f"holds a run manifest; use a fresh directory.")
            return 2

    def rewritten(out):
        new = R.rewrite_flag_argv(argv[1:], spec.output_flags, out)
        for extra_flags, key in spec.extra_rewrites:
            value = recorded(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"the manifest does not record {key}")
            new = R.rewrite_flag_argv(new, extra_flags, value)
        for out_flags, key in spec.relocated_outputs:
            value = recorded(key)
            if value:
                new = R.rewrite_flag_argv(new, out_flags,
                                          str(Path(out) / Path(value).name))
            elif any(str(t).split("=", 1)[0] in out_flags for t in argv[1:]):
                raise ValueError(f"the command line sets {out_flags[0]} but "
                                 f"the manifest does not record {key}")
        return new

    try:
        rewritten("<output>")
    except ValueError as exc:
        logger.error(f"Refusing to reproduce: {exc}")
        return 2
    n_proc = manifest["reproducibility"].get("n_processes")
    if n_proc is None:
        logger.error("Refusing to reproduce: the manifest does not record the "
                     "process count (reproducibility.n_processes).")
        return 2
    if n_proc != 1:
        out = str(Path(args.output).resolve()) if args.output else "OUT"
        cmd = [sys.executable, str(script), *rewritten(out)]
        logger.error(
            f"Refusing to reproduce: the run used {n_proc} processes and this "
            f"command reruns one. Launch it yourself on {n_proc} processes with "
            f"the original launcher (not recorded), from {cwd}"
            + ("" if args.output else ", with OUT a fresh directory")
            + f":\n  {R.REPRODUCE_REFERENCE_ENV}={shlex.quote(str(ref))} "
            f"{R.REPRODUCE_OUTPUT_ENV}={shlex.quote(out)} "
            + (f"{R.ALLOW_VERSION_DRIFT_ENV}=1 " if allow_drift else "")
            + shlex.join(cmd))
        return 2
    if not args.output:   # kept afterwards, for inspection after a mismatch
        new_root = Path(tempfile.mkdtemp(prefix="legoesm_reproduce_")).resolve()
        if _overlaps(new_root):      # TMPDIR inside the reference tree
            new_root.rmdir()
            return 2
    child = rewritten(str(new_root))
    env = dict(os.environ)
    env[R.REPRODUCE_REFERENCE_ENV] = str(ref)
    env[R.REPRODUCE_OUTPUT_ENV] = str(new_root)
    if allow_drift:
        env[R.ALLOW_VERSION_DRIFT_ENV] = "1"

    logger.info(f"legoESM v{__version__} | Reproducing "
                f"{manifest['reproducibility']['runner_tag']} run from {ref} "
                f"into {new_root}, running {script} with {sys.executable} "
                f"(recorded: {argv[0]})")
    rc = subprocess.run([sys.executable, str(script), *child], cwd=cwd,
                        env=env).returncode
    if rc != 0:
        logger.error(f"Rerun exited with {rc} (refused or failed; see its "
                     f"output above).")
        return rc if rc > 0 else 1
    rerun = new_root / rel
    if not rerun.is_file():
        logger.error(f"Rerun wrote no manifest at {rerun}.")
        return 1
    if not args.check:
        # Exit 0 is not completion (a wallclock-budget exit checkpoints and
        # exits 0 too); a completed run records its final digest.
        try:
            R.recorded_state_digest(R.read_run_manifest(rerun))
        except (ValueError, OSError):
            logger.error(f"Rerun recorded no final state digest in {rerun}: "
                         f"it did not complete (or its digest failed).")
            return 1
        logger.info(f"  Reproduced; rerun manifest {rerun}")
        return 0
    try:
        rerun_digest = R.recorded_state_digest(R.read_run_manifest(rerun))
    except (ValueError, OSError) as exc:
        logger.error(f"reproduce --check: the rerun recorded no digest ({exc}).")
        return 1
    if rerun_digest == reference:
        logger.info(f"reproduce --check: MATCH — bit-identical "
                    f"(state_digest {reference[:16]}...)")
        return 0
    logger.error("reproduce --check: MISMATCH — run is NOT bit-reproducible\n"
                 f"  reference: {reference}\n  rerun:     {rerun_digest}")
    return 1


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
