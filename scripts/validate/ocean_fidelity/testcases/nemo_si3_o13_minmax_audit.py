"""Exhaustive MIN/MAX/SIGN/SQRT audit of the SI3 O13 remap object code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases import nemo_si3_o9_minmax_audit as common

BUILD = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/"
    "cfgs/ORCA2_SAS_ICE_O8O16_REF/BLD"
)
TOOLCHAIN = common.TOOLCHAIN
ROUTINES = {"iceitd": ("ice_itd_rem", "itd_glinear", "itd_shiftice")}
CENSUS = {
    "ice_itd_rem": (
        "zhbnew", "zdhice", "zetamin", "zetamax", "zda0", "zdamax",
        "zdaice", "zdvice", "jdonor",
    ),
    "itd_glinear": ("phL", "phR", "pg0", "pg1", "zh13", "zh23", "zdhr"),
    "itd_shiftice": (
        "zworka", "zworkv", "ztrans", "zaTsfn", "ztmp", "rn_amax_1d",
    ),
}

TIE_LEMMAS = {
    "L0": (
        "MINSD/MAXSD returns its source operand on an equality or unordered "
        "comparison; finite equal nonzero operands have identical bits."
    ),
    "L1": (
        "All nine executing O13 MIN/MAX sites have two finite nonnegative "
        "physical operands except the displaced-boundary upper expression; "
        "signed-zero distinctions are either impossible under the strict "
        "remap guards or overwritten by the subsequent extensive update."
    ),
}

# Populated from the record-object report below; every executing source site is
# classified explicitly.  Operand roles are independently pinned by the JSON
# report generated from the byte-identical -g recompile.
TIE_VERDICTS = {
    "itd:222": {"verdict": "NO_TIE", "proof": "hi_max(jpl-1) is strictly positive (L0)."},
    "itd:280": {"verdict": "NO_TIE", "proof": "The melt arm has -zdh0 strictly positive (L1)."},
    "itd:282": {"verdict": "NO_TIE", "proof": "A signed-zero tie cannot change the subsequent positive-eta guard (L1)."},
    "itd:289": {"verdict": "NO_TIE", "proof": "The loss arm has positive area and thickness loss (L1)."},
    "itd:298": {"verdict": "MASKED", "proof": "A zero sign is consumed only as a category boundary in finite subtraction/comparison (L1)."},
    "itd:326": {"verdict": "MASKED", "proof": "Equal bounds make eta_min zero; its sign is masked by the following interval arithmetic (L1)."},
    "itd:327": {"verdict": "MASKED", "proof": "Equal bounds make a zero-width transfer whose sign is masked by line 334 (L1)."},
    "itd:331": {"verdict": "MASKED", "proof": "Equal bounds make a zero-width transfer whose sign is masked by line 334 (L1)."},
    "itd:334": {"verdict": "MASKED", "proof": "A zero interval yields exactly zero area/volume transfer and cannot alter outputs (L1)."},
}


def audit(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict:
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.build, args.toolchain)
    rendered = json.dumps(report, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    if args.check:
        same = json.loads(args.check.read_text()) == json.loads(rendered)
        print("O13_MINMAX_AUDIT", "MATCH" if same else "DIFFERS")
        return 0 if same else 1
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
