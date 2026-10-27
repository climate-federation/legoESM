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
        "ice_cor(1) bounds every positive-volume production layer at "
        "sz_i >= rn_simin = 0.1, and the conservative remap preserves "
        "positive salinity and nonnegative enthalpy.  In TEMP1, sz_i > 0 "
        "makes the discriminant strictly larger than zbbb**2, so its positive "
        "root cannot cancel to rt0; only h_i <= 0 or sz_i == 0 can return rt0."
    ),
    "L2": (
        "ABS produces +0 for either signed-zero input, and comparisons do not "
        "observe a zero sign."
    ),
    "L3": (
        "The unrolled line-644 object emits MIN(0,zds) as (dst,src)=(zds,0) "
        "except in layer 9, and emits the outer MAX as (MIN,floor) except in "
        "layers 9 and 10; finite production-NML columns at the salinity floor "
        "make the layer-9 inner role and outer role flip the brine-flux zero."
    ),
}

# One verdict for every executing floating MIN/MAX source statement.  The four
# integer MAX statements (:447/:448/:607/:608) are censused separately because
# gfortran implements them with integer compares/conditional moves, not MINSD.
TIE_VERDICTS = {
    "sal:334": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi10 (L0)."},
    "sal:350": {"verdict": "MASKED", "proof": "The sign of zero zs_brmax is consumed only by > sss; the downstream comparison is checked (L2)."},
    "sal:367": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi10 (L0)."},
    "sal:397": {"verdict": "MASKED", "proof": "The MAX survivor can change only the zero sign of zRa; either zero is subcritical and the downstream result is checked (L2)."},
    "sal:412": {"verdict": "NO_TIE", "proof": "zRa-rn_Rc_RJW is nonnegative under its enclosing guard (L0)."},
    "sal:444": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:482": {"verdict": "MASKED", "proof": "The MAX survivor can change only the zero sign of zRa; either zero is subcritical and the downstream result is checked (L2)."},
    "sal:497": {"verdict": "NO_TIE", "proof": "zRa-rn_Rc_RJW is nonnegative under its enclosing guard (L0)."},
    "sal:534": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:535": {"verdict": "MASKED", "proof": "zs_min is read only under deck-disabled ln_sal_chk (census)."},
    "sal:538": {"verdict": "NO_TIE", "proof": "The former plant used an unordered NaN, not a finite signed-zero tie; it is removed (L0)."},
    "sal:569": {"verdict": "NO_TIE", "proof": "Other operand is nonzero epsi10 (L0)."},
    "sal:605": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:619": {"verdict": "NO_TIE", "proof": "The analytic production TEMP1 bound makes every flush-eligible zmsk and divisor nonzero (L1)."},
    "sal:637": {"verdict": "NO_TIE", "proof": "Both candidates are ABS results, hence never -0 (L2)."},
    "sal:641": {"verdict": "NO_TIE", "proof": "Production sz_i is positive; exact cancellation in sz_i+zds yields +0, not the former synthetic -0 operand (L1)."},
    "sal:644": {"verdict": "FOUND", "proof": "Finite production-NML columns hit a real layer-9 -0/+0 clamp tie and flip sfx_bri (L3)."},
    "sal:677": {"verdict": "NO_TIE", "proof": "The inner clamp uses nonzero rn_simin and the outer clamp uses positive rn_sinew*sss on the production deck (L0)."},
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
