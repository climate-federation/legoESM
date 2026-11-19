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
        if kind == "atmosphere" and tag == "run_coupled":
            logger.error("Refusing to reproduce: this is the atmosphere record "
                         "of a coupled run; reproduce that run's coupled "
                         "record (coupled/run_manifest.json under its output "
                         "directory) instead.")
            sys.exit(2)
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
                "(and result.ice_state_digest, when recorded) against the "
                "reference manifest."
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

        # A coupled run's atmosphere manifest sits at its output root, beside
        # coupled/: rerunning the atmosphere alone would not repeat that run.
        coupled_beside = (Path(args.manifest).resolve()
                          if Path(args.manifest).is_dir()
                          else Path(args.manifest).resolve().parent)
        coupled_beside = coupled_beside / "coupled" / "run_manifest.json"
        if coupled_beside.is_file():
            logger.error(f"Refusing to reproduce: this is the atmosphere record "
                         f"of a coupled run; reproduce {coupled_beside} "
                         f"instead.")
            sys.exit(2)

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
    only the output directory and the spec's own rewrites changed.  The rerun's own manifest writer
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

    ref = Path(args.manifest).resolve()
    if ref.is_dir():
        ref = ref / R.RUN_MANIFEST_FILENAME
    ref_root = ref.parent
    for _ in range(spec.manifest_depth):
        ref_root = ref_root.parent
    # Where the rerun writes its manifest: the runner's own file name, not the
    # reference's (which may be a renamed or archived copy).
    rel = ref.parent.relative_to(ref_root) / R.RUN_MANIFEST_FILENAME
    # (relative path under the root, reference manifest) for every manifest
    # --check compares: this one plus the runner's companions.
    checked = [(rel, manifest)]
    for name, key in spec.companion_manifests:
        try:
            checked.append((Path(name), R.read_companion_manifest(
                ref, manifest, spec, name, key)[1]))
        except ValueError as exc:
            logger.error(f"Refusing to reproduce: no {name} of this run found "
                         f"({exc}).")
            return 2
    no_digest = [k for k in spec.no_digest_keys if recorded(k)]
    if no_digest:   # a plain reproduce needs the rerun's digest too
        logger.error(f"Refusing to reproduce: this run records no final "
                     f"digest ({', '.join(no_digest)} is set).")
        return 2
    if args.check:
        try:
            references = [(rel, R.recorded_state_digest(manifest))]
            # Companions: the digests this record paired with at its end, not
            # the companion files' (later runs may have updated those).
            paired = manifest["result"].get("companion_state_digests") or {}
            for path, _m in checked[1:]:
                if not paired.get(str(path)):
                    raise ValueError(f"the record pairs no final digest of "
                                     f"its {path}")
                references.append((path, paired[str(path)]))
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
    why = R.unreplayable_reasons(manifest, spec)
    if why:
        logger.error(f"Refusing to reproduce: the run was launched with "
                     f"{', '.join(why)}, which cannot be replayed into a fresh "
                     f"directory.")
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
    worst = 0
    for i, (path, reference) in enumerate(references):
        try:
            rerun_record = R.read_run_manifest(new_root / rel)
            if i == 0:
                rerun_digest = R.recorded_state_digest(rerun_record)
            else:
                # The digest the rerun's record paired, i.e. one this rerun
                # wrote, not whatever the output directory's file holds.
                rerun_digest = (rerun_record["result"].get(
                    "companion_state_digests") or {}).get(str(path))
                if not rerun_digest:
                    raise ValueError("its record pairs none")
        except (OSError, ValueError) as exc:
            logger.error(f"reproduce --check: the rerun's {path} recorded no "
                         f"digest ({exc}).")
            worst = 1
            continue
        if rerun_digest == reference:
            logger.info(f"reproduce --check: MATCH — bit-identical {path} "
                        f"(state_digest {reference[:16]}...)")
        else:
            logger.error(f"reproduce --check: MISMATCH — {path} is NOT "
                         f"bit-reproducible\n  reference: {reference}\n"
                         f"  rerun:     {rerun_digest}")
            worst = 1
    return worst


def cmd_test(args):
    raise ValueError('fv3_duo has no atmosphere shallow-water CLI or benchmark adapter.')


def cmd_benchmark(args):
    raise ValueError('fv3_duo has no atmosphere shallow-water CLI or benchmark adapter.')


if __name__ == "__main__":
    main()
