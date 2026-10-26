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
        "sz_i_1d", "zv_br", "zv_brmin", "zperm", "zperm_eff", "zra",
        "zrae", "zs_brmax", "zs_min", "zds", "zcfl", "zcfl_max", "zcfl_test",
    ),
}

TIE_LEMMAS = {
    "L0": (
        "MINSD/MAXSD returns its source operand on an equality or unordered "
        "comparison; equal finite nonzero operands have identical bits."
    ),
    "L1": (
        "The production deck has finite, nonnegative layer thickness and the "
        "O10 entry has sz_i >= rn_simin = 0.1; TEMP1 therefore has "
        "t_i-rt0 <= -rTmlt*rn_simin = -0.0054 K."
    ),
    "L2": (
        "ABS produces +0 for either signed-zero input, and comparisons do not "
        "observe a zero sign."
    ),
    "L3": (
        "The unrolled line-644 object emits MIN(0,zds) as (dst,src)=(zds,0) "
        "except in layer 9, and emits the outer MAX as (MIN,floor) except in "
        "layers 9 and 10; the committed bit-changing plants pin both splits."
    ),
}

# One verdict for every executing floating MIN/MAX source statement.  The four
# integer MAX statements (:447/:448/:607/:608) are censused separately because
# gfortran implements them with integer compares/conditional moves, not MINSD.
TIE_VERDICTS = {
    "sal:334": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi10 (L0)."},
    "sal:350": {"verdict": "MASKED", "proof": "The sign of zero zs_brmax is consumed only by > sss; the downstream comparison is checked (L2)."},
    "sal:367": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi06 (L0)."},
    "sal:397": {"verdict": "MASKED", "proof": "A signed-zero permeability reaches only the subcritical Rayleigh comparison; the downstream result is checked (L2)."},
    "sal:412": {"verdict": "NO_TIE", "proof": "zRa-zRacrit is nonnegative under its enclosing guard (L1)."},
    "sal:444": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:482": {"verdict": "MASKED", "proof": "A signed-zero permeability reaches only the subcritical Rayleigh comparison; the downstream result is checked (L2)."},
    "sal:497": {"verdict": "NO_TIE", "proof": "zRa-zRacrit is nonnegative under its enclosing guard (L1)."},
    "sal:534": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:535": {"verdict": "MASKED", "proof": "zs_min is read only under deck-disabled ln_sal_chk (census)."},
    "sal:538": {"verdict": "FOUND", "proof": "zds and the floor can tie as -0/+0; the flux-bit plant fires."},
    "sal:569": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi06 (L0)."},
    "sal:605": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:619": {"verdict": "NO_TIE", "proof": "L1 makes every zmsk=1 and every divisor nonzero on the deck."},
    "sal:637": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:641": {"verdict": "MASKED", "proof": "zs_min is read only under deck-disabled ln_sal_chk (census)."},
    "sal:644": {"verdict": "FOUND", "proof": "Both zero-valued clamps have layer-specific survivors pinned by L3."},
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
