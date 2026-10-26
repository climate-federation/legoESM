#!/usr/bin/env python
"""LES-truth suite test matrix — enumerable case selection + dispatch.

One entry point for every LES reference case in the suite catalog
(``docs/atmosphere/les_suite/LES_SUITE.md`` §5). Turns the scattered
``scripts/run/run_spectral_{cbl,sbl}.py`` / ``run_{bomex,dycoms}_les.py`` drivers
into an enumerable matrix, selected by case name and resolution-label grid through
the shared ``legoesm.core.setup_selector`` semantics (no re-implemented selector).

The LES runs themselves are GPU-gated (a 96^3 sim-hour is ~7 min on a V100S; hours
on CPU). By default this dispatcher only ENUMERATES and prints the per-case driver
command (a dry run); pass ``--run`` to actually launch each selected case's driver.

Usage::

    python scripts/matrix/run_les_suite_matrix.py --list
    python scripts/matrix/run_les_suite_matrix.py --only =cbl_nieuwstadt --grid 96x96x96
    python scripts/matrix/run_les_suite_matrix.py --grid 96x96x96          # dry run
    python scripts/matrix/run_les_suite_matrix.py --only =cbl_nieuwstadt --run

An exact ``--only =<name>`` (or a ``--grid``) that matches nothing exits non-zero
with a message — never a silent zero-case no-op (dispatch hardening).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from legoesm.atmosphere.les_suite.matrix import (
    MatrixCase,
    build_test_matrix,
    select_cases,
    valid_grid_labels,
)


# Re-exported for the setup-template test harness (mirrors the other components'
# ``run_<comp>_test_matrix._build_test_matrix``).
def _build_test_matrix() -> list[MatrixCase]:
    return build_test_matrix()


def _driver_command(case) -> list[str]:
    """The ``scripts/run/<driver>`` invocation for one LES case (dry-run text)."""
    cmd = ["python", f"scripts/run/{case.driver}"]
    if case.surface_theta_flux_K_m_s is not None:
        cmd += ["--Q0", str(case.surface_theta_flux_K_m_s)]
    if case.geostrophic_wind_m_s is not None:
        cmd += ["--Ug", str(case.geostrophic_wind_m_s)]
    cmd += ["--sgs", case.sgs_variants[0]]
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--only",
        default=None,
        help="case-name filter; prefix with '=' for EXACT match "
        "(e.g. --only =cbl_nieuwstadt). Bare value = substring.",
    )
    p.add_argument(
        "--grid",
        default=None,
        help=f"resolution-label filter; one of {list(valid_grid_labels())}",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list the selected (case, grid, regime, driver) rows and exit.",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="actually launch each selected case's driver (GPU-gated); "
        "default is a dry run that only prints the commands.",
    )
    args = p.parse_args(argv)

    # select_cases raises SystemExit on an empty exact/grid selection.
    cases = select_cases(only=args.only, grid=args.grid)

    if args.list:
        print(f"{'case':<20} {'grid':<12} {'regime':<16} driver")
        for c in cases:
            print(f"{c.name:<20} {c.grid.label:<12} {c.regime:<16} {c.driver}")
        return 0

    rc = 0
    for c in cases:
        cmd = _driver_command(c)
        if args.run:
            print(f"[run] {c.name}: {' '.join(cmd)}")
            result = subprocess.run(cmd)  # noqa: S603 (trusted, repo-internal)
            if result.returncode != 0:
                print(f"[fail] {c.name} exited {result.returncode}", file=sys.stderr)
                rc = result.returncode
        else:
            print(f"[dry-run] {c.name}: {' '.join(cmd)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
