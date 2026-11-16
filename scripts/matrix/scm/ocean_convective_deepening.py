"""Ocean free-convective mixed-layer deepening on the ocean column model.

A resting, linearly stratified column cooled at the surface at a constant
rate; checked against the analytic oracle (heat conservation and the sqrt(t)
encroachment deepening law).  The case and its checks are the ones in
``scripts/validate/validate_ocean_scm.py`` (#335), reused unchanged; this
module only exposes them as an SCM-matrix case (#2103).

Usage:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \\
        .venv/bin/python scripts/matrix/run_scm_test_matrix.py \\
        ocean_convective_deepening
"""

from __future__ import annotations

import argparse
import os
import sys

import jax

# scripts/ on sys.path so ``validate.validate_ocean_scm`` resolves even when
# this module is imported without the dispatcher (which does the same).
_SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", action="store_true",
                        help="Save the profile / MLD figure under results/ocean_scm/.")


def run(args: argparse.Namespace) -> int:
    from validate.validate_ocean_scm import report, run_convective_deepening

    jax.config.update("jax_enable_x64", True)  # as validate_ocean_scm.main
    ok = report(run_convective_deepening(), plot=args.plot)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1
