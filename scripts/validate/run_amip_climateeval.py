#!/usr/bin/env python
"""Benchmark an AMIP CMOR output tree against ERA5 via ClimateEval.

Standalone consumer script: invoked as a subprocess under the *separate*
ClimateEval environment (iris + ESMValTool; never a legoESM dependency),
never imported into the legoESM/JAX process. See
``legoesm.driver.climateeval_hook.maybe_run_climateeval`` for the caller.

Runs one or more ClimateEval suites against the same CMOR tree and renders
them into a SINGLE combined portable HTML report (plus one ``.ddb`` per
suite). With no ``--suite``, runs ALL bundled suites (every tier); a suite
whose reference / model data is missing is **skipped and reported**, not
fatal, so the report always contains whatever could be scored.

Usage:
    <climateeval-env>/bin/python scripts/validate/run_amip_climateeval.py \\
        --cmor-dir /scratch/.../amip_run/cmor \\
        --model-id legoESM-1-0 \\
        --data-root-dir /work/bd1179/b309141/climateeval_input \\
        --output-dir /scratch/.../amip_run
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

DEFAULT_REPORT_NAME = "climateeval_report.html"

# CMOR MIP tables loaded for scoring: the monthly tables (atmosphere / ocean /
# sea-ice / land / Emon) + the fixed fields. ``day`` and sub-daily tables are
# EXCLUDED so a daily ``tas`` doesn't collide with the monthly ``tas`` a monthly
# suite expects. Loading Omon/SImon lets the ocean / sea-ice suites score.
_MONTHLY_CMOR_TABLES = ("Amon", "Omon", "SImon", "Lmon", "Emon", "fx")


def obs_only_suite_def(
    suite_def: list[Any], force_reference: str | None = None
) -> list[Any]:
    """Strip ``other_data`` (CMIP6 model comparisons), optionally forcing one
    reference dataset for every variable.

    With ``force_reference=None`` (the default) each variable keeps the
    reference the suite declares for it — CERES-EBAF for the TOA fluxes, GPCP
    for ``pr``, ESACCI-CLOUD for the cloud fields, HadCRUT5 for ``tas``, ERA5
    for the rest. That matters: forcing ERA5 everywhere silently DROPS every
    variable ERA5 does not carry, and the local ERA5 tree has no
    ``rsut``/``rlut``/``rsutcs``/``swcre``/``hfls``, i.e. the entire top-of-
    atmosphere radiation budget went unscored ("No reference data available
    for Variable(id='rsut'): skipping metrics calculation").

    Pass ``force_reference="climateeval.data.ERA5Monthly"`` to restore the
    single-reference behaviour when a like-for-like ERA5-only comparison is
    what is wanted.

    A ClimateEval suite YAML is a list of diagnostic blocks, each with a
    ``variables`` list of per-variable settings dicts (some diagnostics
    share the *same* list object via a YAML anchor/alias, e.g. ``map``
    and ``zonal_line`` both reusing ``&2d_variables`` — mutating one
    mutates all aliased diagnostics too, which is what we want here).

    Pure transform (no ClimateEval/iris import needed) so it is directly
    unit-testable without the ClimateEval environment installed.
    """
    for diagnostic_block in suite_def:
        if not isinstance(diagnostic_block, dict):
            continue
        variables = diagnostic_block.get("variables")
        if not isinstance(variables, list):
            continue
        for var_settings in variables:
            if isinstance(var_settings, dict):
                var_settings.pop("other_data", None)
                if force_reference and "reference_data" in var_settings:
                    var_settings["reference_data"] = force_reference
    return suite_def


def suite_db_path(output_dir: Path, suite: str) -> Path:
    """Canonical per-suite DuckDB path under the run's output directory."""
    return output_dir / f"climateeval_{suite}.ddb"


def cmor_nc_paths(cmor_dir: Path) -> list[Path]:
    """Resolve the CMOR ``*.nc`` files to load.

    If ``cmor_dir`` holds ``*.nc`` directly (e.g. a single ``…/cmor/Amon``
    table), load those (back-compat). Otherwise treat it as the ``cmor``
    root and load the monthly MIP tables + fx (see ``_MONTHLY_CMOR_TABLES``);
    ``day``/sub-daily tables are excluded to avoid frequency collisions.
    """
    direct = sorted(cmor_dir.glob("*.nc"))
    if direct:
        return direct
    return sorted(
        p for table in _MONTHLY_CMOR_TABLES for p in (cmor_dir / table).glob("*.nc")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmor-dir", required=True,
                         help="CMOR output root (…/cmor) or a single MIP-table "
                              "directory (…/cmor/Amon).")
    parser.add_argument("--suite", dest="suites", nargs="+", default=[],
                         help="ClimateEval suite names to render into a single "
                              "combined report. Default (empty) = ALL bundled "
                              "suites; a suite missing its data is skipped + "
                              "reported, not fatal.")
    parser.add_argument("--model-id", default="legoESM-1-0")
    parser.add_argument("--experiment-id", default="amip")
    parser.add_argument("--variant-id", default="r1i1p1f1")
    parser.add_argument("--data-root-dir", required=True,
                         help="ClimateEval reference-data root "
                              "(source_id/frequency/var layout).")
    parser.add_argument("--timerange", default="",
                         help="Variable timerange override, e.g. 19790101/19791231. "
                              "Empty = use the model output's own time span.")
    parser.add_argument(
        "--force-reference", default=None,
        help="Override EVERY variable's reference dataset with this "
             "climateeval.data class (e.g. climateeval.data.ERA5Monthly). "
             "Default: keep each variable's suite-declared reference, so the "
             "TOA fluxes score against CERES-EBAF instead of being dropped.")
    parser.add_argument("--fail-on-missing-data", action="store_true", default=False)
    parser.add_argument("--download-missing-data", action="store_true", default=False)
    parser.add_argument("--output-dir", required=True,
                         help="Directory to write per-suite .ddb files and the "
                              "combined HTML report into.")
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME,
                         help="Filename of the combined HTML report written "
                              f"under --output-dir (default: {DEFAULT_REPORT_NAME}).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Deferred: only the ClimateEval environment has these installed; keeping
    # them out of module scope lets era5_only_suite_def/suite_db_path/
    # build_arg_parser be imported and unit-tested from the legoESM env too.
    from importlib import resources

    import iris
    import yaml
    from climateeval.data import DataSourceInformation
    from climateeval.report import serve
    from climateeval.suites import Suite

    cmor_dir = Path(args.cmor_dir)
    paths = cmor_nc_paths(cmor_dir)
    if not paths:
        print(f"ERROR: no .nc files found under {cmor_dir}", file=sys.stderr)
        return 1
    print(f"Loading {len(paths)} CMOR files from {cmor_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in paths])
    print(f"Loaded {len(cubes)} cubes")

    data_info = DataSourceInformation(
        name=args.model_id,
        category="legoESM",
        institute="MPI-M",
        exp=args.experiment_id,
        variant=args.variant_id,
    )
    diagnostic_kwargs = {
        "data_root_dir": Path(args.data_root_dir),
        "fail_on_missing_data": args.fail_on_missing_data,
        "download_missing_data": args.download_missing_data,
    }
    variable_kwargs = {"timerange": args.timerange} if args.timerange else {}
    suite_dir = Path(str(resources.files("climateeval.suites")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # No --suite -> run ALL bundled suites (every tier), sorted for stable order.
    suites = args.suites or sorted(p.stem for p in suite_dir.glob("*.yml"))

    db_paths: list[Path] = []
    skipped: list[str] = []
    for suite in suites:
        db_path = suite_db_path(output_dir, suite)
        if db_path.is_file():
            db_path.unlink()
        tmp_yml = Path(tempfile.mktemp(suffix=".yml"))
        try:
            with (suite_dir / f"{suite}.yml").open() as f:
                suite_def = obs_only_suite_def(
                    yaml.safe_load(f), force_reference=args.force_reference
                )
            with tmp_yml.open("w") as f:
                yaml.dump(suite_def, f)
            Suite(
                tmp_yml,
                diagnostic_kwargs=diagnostic_kwargs,
                variable_kwargs=variable_kwargs,
            ).get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
        except Exception as exc:  # missing data / inapplicable suite -> skip + report
            db_path.unlink(missing_ok=True)
            skipped.append(f"{suite}: {type(exc).__name__}: {exc}")
            print(f"SKIPPED suite {suite}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        finally:
            tmp_yml.unlink(missing_ok=True)
        print(f"Database written to {db_path}")
        db_paths.append(db_path)

    if skipped:
        print(f"Skipped {len(skipped)}/{len(suites)} suite(s):")
        for s in skipped:
            print(f"  - {s}")
    if not db_paths:
        print("ERROR: no suite produced a database (all skipped)", file=sys.stderr)
        return 1

    # Single combined report over the suites that scored.
    html_path = output_dir / args.report_name
    serve(db_paths, model_name=args.model_id, save_html=html_path)
    print(f"Report covers {len(db_paths)}/{len(suites)} suites; "
          f"HTML report written to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
