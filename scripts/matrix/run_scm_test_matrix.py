"""Single-column model (SCM) test matrix dispatcher.

One entry point for every SCM case. Replaces the per-case scripts
``run_scm_{rce,gabls1,ekman,wangara}.py`` and ``run_jax_scm_oracle.py``.

Usage::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/run_scm_test_matrix.py <case> [options]

Cases:
    rce       Tropical RCE (gray rad + Louis + Kessler + mass-flux conv)
    gabls1    Stable BL (Cuxart et al. 2006, MYNN-2.5)
    ekman     Neutral Ekman BL (Andren et al. 1994, MYNN-2.5)
    wangara   Convective dry BL (Day 33, MYNN-2.5 + prescribed fluxes)
    oracle    Generate jax_scm reference NetCDFs (needs .venv-jax-scm)

Each subcommand exposes the same flags as the standalone script it
replaces; pass ``--help`` after the case name to see them. Example::

    .venv/bin/python scripts/run_scm_test_matrix.py rce --help
    .venv/bin/python scripts/run_scm_test_matrix.py rce --days 50
    .venv-jax-scm/bin/python scripts/run_scm_test_matrix.py oracle --cases gabls1
"""

from __future__ import annotations

import argparse
import os
import sys

# scripts/ (parent of this file's scripts/matrix/ dir) so `from matrix.scm
# import ...` resolves — the scm/ case modules live at scripts/matrix/scm/.
# Importing them as matrix.scm (not bare `scm`) keeps the external jax_scm
# `scm` package importable for the oracle case (scm/oracle.py).
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from matrix.scm import ekman, gabls1, oracle, rce, wangara  # noqa: E402

CASES = {
    "rce": rce,
    "gabls1": gabls1,
    "ekman": ekman,
    "wangara": wangara,
    "oracle": oracle,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="case", required=True, metavar="CASE")
    for name, mod in CASES.items():
        sp = sub.add_parser(
            name,
            description=mod.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            help=(mod.__doc__ or "").splitlines()[0] if mod.__doc__ else None,
        )
        mod.add_args(sp)
    args = p.parse_args(argv)
    return CASES[args.case].run(args)


if __name__ == "__main__":
    sys.exit(main())
