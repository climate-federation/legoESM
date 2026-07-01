#!/usr/bin/env python
"""Benchmark an AMIP CMOR output tree against ERA5 via ClimateEval.

Standalone consumer script: invoked as a subprocess under the *separate*
ClimateEval environment (iris + ESMValTool; never a legoESM dependency),
never imported into the legoESM/JAX process. See
``legoesm.driver.climateeval_hook.maybe_run_climateeval`` for the caller.

Usage:
    <climateeval-env>/bin/python scripts/validate/run_amip_climateeval.py \\
        --cmor-dir /scratch/.../amip_run/cmor/Amon \\
        --suite Tier2_atmosphere_monthly --model-id legoESM-1-0 \\
        --data-root-dir /work/bd1179/b309141/climateeval_input \\
        --output-db amip_run.ddb --output-html amip_run_report.html
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any


def era5_only_suite_def(suite_def: list[Any]) -> list[Any]:
    """Strip ``other_data`` (CMIP6 model comparisons) and force every
    variable's ``reference_data`` to ``climateeval.data.ERA5Monthly``.

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
                if "reference_data" in var_settings:
                    var_settings["reference_data"] = "climateeval.data.ERA5Monthly"
    return suite_def


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmor-dir", required=True,
                         help="Directory of CMOR Amon *.nc files to load.")
    parser.add_argument("--suite", default="Tier1_sanity_checks",
                         help="ClimateEval suite name (e.g. Tier2_atmosphere_monthly).")
    parser.add_argument("--model-id", default="legoESM-1-0")
    parser.add_argument("--experiment-id", default="amip")
    parser.add_argument("--variant-id", default="r1i1p1f1")
    parser.add_argument("--data-root-dir", required=True,
                         help="ClimateEval reference-data root "
                              "(source_id/frequency/var layout).")
    parser.add_argument("--timerange", default="",
                         help="Variable timerange override, e.g. 19790101/19791231. "
                              "Empty = use the model output's own time span.")
    parser.add_argument("--fail-on-missing-data", action="store_true", default=False)
    parser.add_argument("--download-missing-data", action="store_true", default=False)
    parser.add_argument("--output-db", required=True,
                         help="Output DuckDB path for the suite database.")
    parser.add_argument("--output-html", required=True,
                         help="Output path for the portable HTML report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Deferred: only the ClimateEval environment has these installed; keeping
    # them out of module scope lets era5_only_suite_def/build_arg_parser be
    # imported and unit-tested from the legoESM environment too.
    from importlib import resources

    import iris
    import yaml
    from climateeval.data import DataSourceInformation
    from climateeval.report import serve
    from climateeval.suites import Suite

    cmor_dir = Path(args.cmor_dir)
    paths = sorted(cmor_dir.glob("*.nc"))
    if not paths:
        print(f"ERROR: no .nc files found in {cmor_dir}", file=sys.stderr)
        return 1
    print(f"Loading {len(paths)} CMOR files from {cmor_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in paths])
    print(f"Loaded {len(cubes)} cubes")

    suite_dir = Path(str(resources.files("climateeval.suites")))
    with (suite_dir / f"{args.suite}.yml").open() as f:
        suite_def = yaml.safe_load(f)
    suite_def = era5_only_suite_def(suite_def)

    tmp_yml = Path(tempfile.mktemp(suffix=".yml"))
    with tmp_yml.open("w") as f:
        yaml.dump(suite_def, f)

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

    db_path = Path(args.output_db)
    if db_path.is_file():
        db_path.unlink()
    try:
        suite = Suite(
            tmp_yml,
            diagnostic_kwargs=diagnostic_kwargs,
            variable_kwargs=variable_kwargs,
        )
        suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
    finally:
        tmp_yml.unlink(missing_ok=True)
    print(f"Database written to {db_path}")

    html_path = Path(args.output_html)
    serve([db_path], model_name=args.model_id, save_html=html_path)
    print(f"HTML report written to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
