"""CLI parser, test filter, and main entry point for the ocean test matrix."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from ocean_test_matrix import config
from ocean_test_matrix.testcase import TestCase, TEST_MATRIX
from ocean_test_matrix.experiments import RUNNERS
from ocean_test_matrix.diagnostic_io import _ensure_required_artifacts
from ocean_test_matrix.postprocessing import (
    _check_and_generate_comparisons,
    _collect_grid_results,
    _create_cross_grid_comparisons,
    _create_rest_state_cross_variant_comparison,
    _run_replot,
)


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ocean test matrix for legoESM ocean dynamical cores.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--only", type=str, default="all",
        help="Run only cases matching this name "
             "(e.g. rest_state, barotropic_gyre, barotropic_double_gyre)")
    p.add_argument(
        "--grid", type=str, default="all",
        choices=["cubed_sphere", "latlon", "mpas",
                 "mpas_regional", "latlon_regional", "cs_regional",
                 "latlon_channel", "mpas_channel",
                 "all"],
        help="Run only a specific grid type (default: all)")
    p.add_argument(
        "--resolution", type=str, default=None,
        help="Override baseline resolution (e.g. C48, 72x144, ico4, T42)")
    p.add_argument(
        "--levels", type=int, default=config.DEFAULT_NLEV,
        help=f"Number of vertical levels (default: {config.DEFAULT_NLEV})")
    p.add_argument(
        "--dt", type=float, default=config.DEFAULT_DT,
        help=f"Time step in seconds (default: {config.DEFAULT_DT})")
    p.add_argument(
        "--output", "-o", type=str, default="results/ocean",
        help="Base output directory (default: results/ocean)")
    p.add_argument(
        "--quick", action="store_true",
        help="Use shorter durations for quick verification")
    p.add_argument(
        "--days", type=float, default=None,
        help="Override duration in days (overrides both normal and quick mode durations)")
    p.add_argument(
        "--tag", type=str, default=None,
        help="Append a tag to the output directory (e.g. --tag dst3_200d)")
    p.add_argument(
        "--list", action="store_true",
        help="List all test cases and exit")
    p.add_argument(
        "--replot", action="store_true",
        help="Skip simulations; regenerate all plots from existing NPZ data")
    p.add_argument(
        "--output-format", type=str, default="netcdf",
        choices=["netcdf", "zarr", "npz"],
        help="Snapshot output format: netcdf (default), zarr, or npz (legacy)")
    p.add_argument(
        "--tracer-advection", type=str, default=None,
        choices=["upwind", "tvd", "dst3", "dst3_multidim", "ppm", "ppm_fct", "som"],
        help="Override tracer advection scheme for Eady experiments")
    p.add_argument(
        "--no-sponge", action="store_true",
        help="Disable sponge relaxation in Eady experiments")
    p.add_argument(
        "--B-h", type=float, default=None,
        help="Override biharmonic viscosity [m^4/s]")
    p.add_argument(
        "--C-smag", type=float, default=None,
        help="Override Smagorinsky coefficient")
    p.add_argument(
        "--K-h", type=float, default=None,
        help="Override Laplacian tracer diffusivity [m^2/s]")
    p.add_argument(
        "--U-surface", type=float, default=None,
        help="Override Eady surface velocity [m/s] (reduces APE / slows BCI)")
    p.add_argument(
        "--barotropic-div-damp", type=float, default=None,
        help=(
            "Dimensionless divergence damping on barotropic velocity. "
            "Targets grid-scale compressible modes directly without "
            "smearing momentum.  Knob #3 in the issue #213 hi-res SOM "
            "Eady investigation recipe."
        ))
    return p


def filter_tests(tests: list[TestCase], args) -> list[TestCase]:
    filtered = tests
    if args.only != "all":
        # Support exact matching with "=" prefix (e.g., "=rest_state")
        if args.only.startswith("="):
            exact_name = args.only[1:]
            filtered = [t for t in filtered if t.case == exact_name]
        else:
            # Default substring matching
            filtered = [t for t in filtered if args.only in t.case]
    if args.grid != "all":
        filtered = [t for t in filtered if t.grid_type == args.grid]
    return filtered


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Use float64 precision for the validation test matrix.
    #
    # The default PrecisionPolicy is float32, which is faster on GPUs
    # and suitable for production runs.  However, float32 introduces
    # rounding noise (~1e-7 relative per step) that accumulates in
    # conservation diagnostics and masks real discretisation errors.
    # For example, the stratified rest-state test shows a spurious
    # temperature drift of ~3e-5 degC/day in float32 that vanishes
    # entirely in float64 — the vertical diffusion operator is in
    # fact perfectly conservative.
    #
    # Running the test matrix in float64 ensures that any drift we
    # detect is a genuine bug in the numerics, not arithmetic noise.
    # ---------------------------------------------------------------
    from legoesm.core.precision import set_policy, PrecisionPolicy
    set_policy(PrecisionPolicy.fp64())

    # Override global defaults if specified
    config.DEFAULT_NLEV = args.levels
    config.DEFAULT_DT = args.dt
    config.OUTPUT_FORMAT = args.output_format
    if args.tracer_advection:
        config.TRACER_ADVECTION_OVERRIDE = args.tracer_advection
    if args.no_sponge:
        config.NO_SPONGE = True
    if args.B_h is not None:
        config.B_H_OVERRIDE = args.B_h
    if args.C_smag is not None:
        config.C_SMAG_OVERRIDE = args.C_smag
    if args.K_h is not None:
        config.K_H_OVERRIDE = args.K_h
    if args.U_surface is not None:
        config.U_SURFACE_OVERRIDE = args.U_surface
    if args.barotropic_div_damp is not None:
        config.BAROTROPIC_DIV_DAMP_OVERRIDE = args.barotropic_div_damp

    tests = filter_tests(TEST_MATRIX, args)

    if args.list:
        print(f"{'#':>3}  {'Case':<22}  {'Grid':<14}  "
              f"{'Resolution':<10}  {'Days':>8}  {'Quick':>8}")
        print("-" * 78)
        for i, tc in enumerate(tests, 1):
            print(f"{i:3d}  {tc.case:<22}  {tc.grid_type:<14}  "
                  f"{tc.resolution:<10}  {tc.duration_days:8.2f}  "
                  f"{tc.quick_days:8.2f}")
        print(f"\nTotal: {len(tests)} test cases "
              f"(of {len(TEST_MATRIX)} in full matrix)")
        return

    if args.replot:
        _run_replot(args)
        return

    if not tests:
        print("No tests match the given filters.")
        return

    if args.resolution:
        tests = [TestCase(
            t.case, t.grid_type, args.resolution,
            t.duration_days, t.quick_days, t.run_kwargs)
            for t in tests]

    output_base = Path(args.output)

    print("=" * 78)
    print("  legoESM Ocean Test Matrix")
    print("=" * 78)
    from legoesm.core.precision import get_policy
    import jax
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  Precision:  {get_policy().storage.__name__}")
    print(f"  X64:        {jax.config.jax_enable_x64}")
    print(f"  Devices:    {jax.devices()}")
    print(f"  Output:     {output_base}")
    print(f"  Levels:     {config.DEFAULT_NLEV}")
    print(f"  dt:         {config.DEFAULT_DT}s")
    print(f"  Quick mode: {args.quick}")
    print(f"  Tests:      {len(tests)} / {len(TEST_MATRIX)}")
    print("=" * 78)
    print()

    t_start_all = time.time()

    # Keep track of completed test cases for cross-grid comparison
    completed_test_cases = set()

    for i, tc in enumerate(tests, 1):
        if args.days is not None:
            days = args.days
        else:
            days = tc.quick_days if args.quick else tc.duration_days
        out_dir = output_base / tc.output_path
        if args.tag:
            out_dir = out_dir / args.tag

        label = f"{tc.case}/{tc.grid_type}/{tc.resolution}"
        print(f"\n[{i}/{len(tests)}] {label} ({days:.4g} days)")
        print("-" * 60)

        runner = RUNNERS.get(tc.case)
        if runner is None:
            config.record(tc, "ERROR", 0, f"Unknown runner for case: {tc.case}")
            continue

        try:
            status, wall, notes = runner(tc, out_dir, days)
            config.record(tc, status, wall, notes)
        except NotImplementedError as e:
            config.record(tc, "SKIP", 0, str(e)[:120])
        except Exception as e:
            config.record(tc, "ERROR", 0, str(e)[:120])
            traceback.print_exc()
        finally:
            _ensure_required_artifacts(out_dir)

        # Check if this test case just completed across all its grids
        if tc.case not in completed_test_cases:
            # Find how many grids are supposed to run for this test case
            test_case_tests = [t for t in tests if t.case == tc.case]
            test_case_results = [r for r in config.ALL_RESULTS if r['test'] == tc.case]

            # If we have results for all grids of this test case, generate comparisons
            if len(test_case_results) >= len(test_case_tests):
                _check_and_generate_comparisons(output_base, tc.case, config.ALL_RESULTS)
                completed_test_cases.add(tc.case)

    total_wall = time.time() - t_start_all

    # --- Summary ---
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  {'Status':6}  {'Grid':<14}  {'Case':<22}  "
          f"{'Resolution':<10}  {'Time':>8}  Notes")
    print("-" * 90)

    n_pass = n_fail = n_error = n_skip = 0
    for r in config.ALL_RESULTS:
        icon = {"PASS": "  ", "FAIL": "**", "ERROR": "!!",
                "SKIP": "--"}[r["status"]]
        print(f"  {icon}{r['status']:5}  {r['grid']:<14}  "
              f"{r['test']:<22}  {r['resolution']:<10}  "
              f"{r['wall_time']:7.1f}s  {r['notes']}")
        if r["status"] == "PASS":
            n_pass += 1
        elif r["status"] == "FAIL":
            n_fail += 1
        elif r["status"] == "SKIP":
            n_skip += 1
        else:
            n_error += 1

    print("-" * 90)
    print(f"  Total: {len(config.ALL_RESULTS)} tests | "
          f"PASS: {n_pass} | FAIL: {n_fail} | SKIP: {n_skip} | "
          f"ERROR: {n_error} | Wall: {total_wall:.1f}s "
          f"({total_wall / 60:.1f} min)")
    print("=" * 78)

    # Save summary
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "summary.json", "w") as f:
        json.dump({
            "results": config.ALL_RESULTS, "total_wall_time": total_wall,
            "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
            "n_error": n_error, "quick_mode": args.quick,
            "levels": config.DEFAULT_NLEV, "dt": config.DEFAULT_DT,
        }, f, indent=2)
    with open(output_base / "summary.txt", "w") as f:
        f.write("legoESM Ocean Test Matrix Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total: {len(config.ALL_RESULTS)} tests | "
                f"PASS: {n_pass} | FAIL: {n_fail} | SKIP: {n_skip} | "
                f"ERROR: {n_error}\n")
        f.write(f"Wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)\n")
        f.write(f"Levels: {config.DEFAULT_NLEV}, dt: {config.DEFAULT_DT}s\n")
        f.write(f"Quick mode: {args.quick}\n\n")
        for r in config.ALL_RESULTS:
            f.write(f"{r['status']:5}  {r['grid']:<14}  "
                    f"{r['test']:<22}  {r['resolution']:<10}  "
                    f"{r['wall_time']:7.1f}s  {r['notes']}\n")

    print(f"\n  Summary: {output_base / 'summary.json'}")

    # Generate cross-grid comparison plots for any test cases that weren't completed during the run
    # (This handles cases where the run was filtered or interrupted)
    remaining_test_cases = set()
    test_cases = {}
    for result in config.ALL_RESULTS:
        test_name = result['test']
        if test_name not in test_cases:
            test_cases[test_name] = []
        test_cases[test_name].append(result)
        if test_name not in completed_test_cases:
            remaining_test_cases.add(test_name)

    if remaining_test_cases:
        print("\n" + "=" * 78)
        print("  GENERATING REMAINING CROSS-GRID COMPARISONS")
        print("=" * 78)

        # Create comparisons for test cases that weren't processed during the main loop
        for test_name in remaining_test_cases:
            results = test_cases[test_name]
            if len(results) > 1:  # Only create comparisons if multiple grids were run
                group = TestCase._CASE_TO_GROUP.get(test_name)
                if group is not None:
                    test_case_dir = output_base / group / test_name
                else:
                    test_case_dir = output_base / test_name
                if test_case_dir.exists():
                    grid_results = _collect_grid_results(test_case_dir)
                    if len(grid_results) > 1:
                        print(f"\n  Creating cross-grid comparisons for {test_name}...")
                        _create_cross_grid_comparisons(test_case_dir, grid_results)
                    else:
                        print(f"  Skipping {test_name}: insufficient grid data")
                else:
                    print(f"  Skipping {test_name}: directory not found")
            else:
                print(f"  Skipping {test_name}: only {len(results)} grid(s) run")
    else:
        print("\n" + "=" * 78)
        print("  ALL CROSS-GRID COMPARISONS COMPLETED DURING RUN")
        print("=" * 78)

    # Generate rest-state cross-variant comparison
    _create_rest_state_cross_variant_comparison(output_base)

    if n_fail > 0 or n_error > 0:
        sys.exit(1)
