"""Exhaustive MIN/MAX/SIGN/SQRT audit of the SI3 O12 melt-pond object code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.validate.ocean_fidelity.testcases import nemo_si3_o9_minmax_audit as common

BUILD = common.BUILD
TOOLCHAIN = common.TOOLCHAIN
ROUTINES = {"icethd_pnd": ("ice_thd_pnd", "pnd_LEV", "ice_perm_eff")}
CENSUS = {
    "ice_thd_pnd": (
        "at_i", "wfx_pnd", "a_ip", "v_ip", "v_il", "h_ip", "h_il", "a_ip_frac",
    ),
    "pnd_LEV": (
        "zdv_pnd", "a_ip_1d", "h_ip_1d", "h_il_1d", "v_ip_1d", "v_il_1d",
        "zdv_avail", "zfr_mlt", "zdv_mlt", "zv_ip_max", "zdT", "zdv_frz",
        "zhp", "zperm", "zdv_flush", "wfx_pnd_1d",
    ),
    "ice_perm_eff": (
        "zt1", "zt2", "zt3", "zs_br", "zv_br", "ztmp", "zperm", "ice_perm_eff",
    ),
}

TIE_LEMMAS = {
    "L0": (
        "MINSD/MAXSD returns its source operand on an equality or unordered "
        "comparison; equal finite nonzero operands have identical bits."
    ),
    "L1": (
        "MAXSD/MINSD with ZERO (+0.0) as src returns +0.0 on a signed-zero "
        "tie and otherwise the strictly winning VAR or +0.0; the result of a "
        "ZERO-src MAX(0, .) is >= +0.0 and never -0.0."
    ),
    "L2": (
        "Under IEEE-754 round-to-nearest-even, a + b is -0.0 only if a = -0.0 "
        "and b = -0.0; a - b is -0.0 only if a = -0.0 and b = +0.0; a square "
        "x * x is >= +0.0 and never -0.0; SQRT(x) on x >= +0.0 is >= +0.0."
    ),
    "L3": (
        "Under the pnd_LEV thick-ice guard (ELSE of h_i_1d < rn_himin .OR. "
        "a_i_1d < 0.01 at line 437), h_i_1d >= 0.1 > 0 and a_i_1d >= 0.01 > 0, "
        "so zv_ip_max at lines 464 and 471 and zhp at line 516 are strictly "
        "positive."
    ),
}

# One verdict for every floating MIN/MAX site in icethd_pnd.o.
# Note: gfortran -O3 fuses lines 522-524 (zdv_flush = -zperm*...;
# zdv_flush = MAX(zdv_flush, -v_ip_1d); v_ip_1d = v_ip_1d + zdv_flush) into
# positive pos_flush = +zperm*..., capped = minsd(dst=pos_flush, src=v_ip_1d)
# attributed in DWARF to line 522, and v_ip_1d = v_ip_1d - capped at line 524
# (leaving line 523 in token_statements_without_instruction and line 522 in
# instructions_without_token).
TIE_VERDICTS = {
    "pnd:456": {
        "verdict": "MASKED",
        "proof": (
            "When dh_i_sum = dh_s_sum = +0.0, zdv_avail = -0.0 and "
            "zfr_mlt * zdv_avail = -0.0 ties with ZERO (+0.0) at maxsd@456; "
            "any zero sign of zdv_mlt is overwritten at lines 465 and 472 "
            "whose outer maxsd has src = ZERO (L1)."
        ),
    },
    "pnd:465": {
        "verdict": "NO_TIE",
        "proof": (
            "In 465.min (dst = zv_ip_max - v_ip_1d, src = zdv_mlt), zdv_mlt "
            "is the ZERO-src maxsd@456 output (never -0.0, L1) and "
            "zv_ip_max > 0 (L3), so zv_ip_max - v_ip_1d is never -0.0 (L2); "
            "465.max then clamps with src = ZERO (L1)."
        ),
    },
    "pnd:472": {
        "verdict": "NO_TIE",
        "proof": (
            "In 472.min (dst = zdv_mlt, src = zv_ip_max - v_ip_1d), zdv_mlt "
            "is the ZERO-src maxsd@465 output (never -0.0, L1) and "
            "zv_ip_max > 0 (L3), so zv_ip_max - v_ip_1d is never -0.0 (L2); "
            "472.max then clamps with src = ZERO (L1)."
        ),
    },
    "pnd:478": {
        "verdict": "MASKED",
        "proof": (
            "Even if v_il_1d = -0.0 from line 447 ties with ZERO when "
            "zdv_mlt = +0.0, v_il_1d is read next only at lines 490 (squared, "
            "L2) and 494 (added to pos_frz >= +0.0 and clamped by maxsd@494 "
            "with src = ZERO, L1)."
        ),
    },
    "pnd:484": {
        "verdict": "NO_TIE",
        "proof": (
            "zTp + rt0 = 273.0 > 0, so (zTp + rt0) - t_su_1d is never -0.0 "
            "under round-to-nearest (L2)."
        ),
    },
    "pnd:490": {
        "verdict": "NO_TIE",
        "proof": (
            "The SQRT radicand is v_il_1d**2 plus a product of nonnegative "
            "terms with zdT from maxsd@484 (never -0.0, L1), so SQRT(rad) >= "
            "+0.0 and SQRT(rad) - v_il_1d is never -0.0 (L2)."
        ),
    },
    "pnd:494": {
        "verdict": "NO_TIE",
        "proof": (
            "v_il_1d from maxsd@478 (src = ZERO) and pos_frz = 0.5 * maxsd@490 "
            "(src = ZERO) are both >= +0.0 and never -0.0, so their sum at "
            "maxsd@494 is never -0.0 (L1, L2)."
        ),
    },
    "pnd:497": {
        "verdict": "NO_TIE",
        "proof": (
            "v_ip_1d at line 475 adds zdv_mlt from maxsd@472 (src = ZERO, "
            "never -0.0, L1), so v_ip_1d entering line 497 is never -0.0 and "
            "v_ip_1d - pos_frz is never -0.0 (L2)."
        ),
    },
    "pnd:502": {
        "verdict": "NOT_ON_DECK",
        "proof": (
            "Inside the ELSE arm of IF( ln_pnd_lids ); on the production "
            "ORCA2_SAS_ICE deck ln_pnd_lids = .true."
        ),
    },
    "pnd:508": {
        "verdict": "NO_TIE",
        "proof": (
            "In minsd(dst = sqrtsd@508, src = a_i_1d), a_i_1d >= 0.01 > 0 "
            "under line 437 (L0, L3)."
        ),
    },
    "pnd:522": {
        "verdict": "NO_TIE",
        "proof": (
            "Emitted minsd(dst = pos_flush, src = v_ip_1d) for source lines "
            "522-523: v_ip_1d from maxsd@497 is >= +0.0 (never -0.0, L1) and "
            "pos_flush >= +0.0 (never -0.0) because zperm >= +0.0, zhp > 0 "
            "(L3), h_i_1d >= 0.1 > 0, and a_ip_1d >= +0.0 (L2)."
        ),
    },
    "pnd:527": {
        "verdict": "NO_TIE",
        "proof": (
            "In minsd(dst = sqrtsd@527, src = a_i_1d), a_i_1d >= 0.01 > 0 "
            "under line 437 (L0, L3)."
        ),
    },
    "pnd:1398": {
        "verdict": "NO_TIE",
        "proof": "Other operand is the nonzero constant epsi10 = 1e-10 (L0).",
    },
}


def audit(build: Path = BUILD, toolchain: Path = TOOLCHAIN) -> dict:
    """Run the shared record-object audit for the O12 production routines."""
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
        print("O12_MINMAX_AUDIT", "MATCH" if same else "DIFFERS")
        return 0 if same else 1
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
