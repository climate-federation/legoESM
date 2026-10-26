"""Exhaustive MIN/MAX/SIGN audit of the SI3 O10 salinity-4 object code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases import nemo_si3_o9_minmax_audit as common

BUILD = common.BUILD
TOOLCHAIN = common.TOOLCHAIN
ROUTINES = {"icethd_sal": ("ice_thd_sal",)}
CENSUS = {
    "ice_thd_sal": (
        "sz_i_1d", "zv_b", "zperm", "zra", "zrae", "zv", "zvold",
        "zvold2", "zs_min", "zdh",
    ),
}

TIE_LEMMAS = {
    "L0": (
        "MINSD/MAXSD returns its source operand on an equality or unordered "
        "comparison; equal finite nonzero operands have identical bits."
    ),
    "L1": (
        "The production deck has finite, nonnegative layer thickness, positive "
        "salinity limits, and positive ice salinity at the O10 entry."
    ),
    "L2": (
        "ABS produces +0 for either signed-zero input, and comparisons do not "
        "observe a zero sign."
    ),
}

# One verdict for every executing floating MIN/MAX source statement.  The four
# integer MAX statements (:447/:448/:607/:608) are censused separately because
# gfortran implements them with integer compares/conditional moves, not MINSD.
TIE_VERDICTS = {
    "sal:334": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi06 (L0)."},
    "sal:350": {"verdict": "MASKED", "proof": "MAX(0,zv_b) is consumed only by a comparison (L2)."},
    "sal:367": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi06 (L0)."},
    "sal:397": {"verdict": "MASKED", "proof": "A zero permeability is multiplied then bounded by later MAX sites."},
    "sal:412": {"verdict": "NO_TIE", "proof": "zRa-zRacrit is nonnegative under its enclosing guard (L1)."},
    "sal:444": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:482": {"verdict": "MASKED", "proof": "A zero permeability is multiplied then bounded by later MAX sites."},
    "sal:497": {"verdict": "NO_TIE", "proof": "zRa-zRacrit is nonnegative under its enclosing guard (L1)."},
    "sal:534": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:535": {"verdict": "NO_TIE", "proof": "zs_min starts +0 and deck salinities are positive (L1)."},
    "sal:538": {"verdict": "NO_TIE", "proof": "Both clamps use nonzero rn_simin/rn_simax (L0)."},
    "sal:569": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi06 (L0)."},
    "sal:605": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:619": {"verdict": "NO_TIE", "proof": "Layer brine volumes are nonnegative on the deck (L1)."},
    "sal:637": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:641": {"verdict": "NO_TIE", "proof": "Both salinity candidates are positive on the deck (L1)."},
    "sal:644": {"verdict": "NO_TIE", "proof": "Both clamps use nonzero rn_simin/rn_simax (L0)."},
    "sal:677": {"verdict": "NO_TIE", "proof": "Both clamps use nonzero rn_simin/rn_simax (L0)."},
}


def audit(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict:
    """Run the shared record-object audit for the O10 production routine."""
    return common.audit(
        build,
        toolchain,
        routines=ROUTINES,
        census=CENSUS,
        tie_lemmas=TIE_LEMMAS,
        tie_verdicts=TIE_VERDICTS,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--toolchain", type=Path, default=TOOLCHAIN)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.build, args.toolchain)
    rendered = json.dumps(report, indent=1, sort_keys=True)
    if args.check:
        same = json.loads(args.check.read_text()) == json.loads(rendered)
        print("O10_MINMAX_AUDIT", "MATCH" if same else "DIFFERS")
        return 0 if same else 1
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
