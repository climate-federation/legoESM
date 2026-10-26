"""Exhaustive MIN/MAX/SIGN/SQRT audit of the SI3 O9 routines' object code.

Recompiles each routine's ppsrc with the record build's flags plus ``-g``,
refuses unless the instruction stream equals the record object's, then lists
EVERY ``minsd``/``maxsd``/``sqrtsd`` instruction inside the O9 routines' source
ranges, mapped through the DWARF line table to its source statement, with the
kind of each operand resolved by reaching definitions over the function's
control-flow graph:

- ``ZERO`` (``pxor``/``xorpd`` of a register with itself), ``CONST:<hex>``
  (a ``.rodata`` load), ``VAR`` (anything else: loads, arithmetic, call
  clobbers), ``MIXED`` (different kinds reach on different paths).

``maxsd src, dst`` returns ``src`` on a tie or NaN, so for a site whose one
operand is ``ZERO`` the kind pair says exactly which operand survives a
signed-zero tie.  For two ``VAR`` operands each row also carries the nearest
non-copy defining instruction(s) of each operand (``src_defs``/``dst_defs``, as
``op@line``), which names the Fortran operand each register holds and exposes
emitted copies whose roles differ.

It also censuses the source in both directions: every ``MIN``/``MAX``/``SIGN``
token in the ranges and every instruction must meet, or the report lists the
mismatch.  Prints a JSON report; ``--check`` compares it to a committed one.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
BUILD = NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD"
TOOLCHAIN = Path("/home/dbalwada/miniconda3/envs/nemo-build")

# Object -> the O9 routines it holds (ppsrc SUBROUTINE/FUNCTION names).
# icethd.o: only ice_thd_temp; the rest of ice_thd is outside O9.
ROUTINES = {
    "icethd_zdf": ("ice_thd_zdf",),
    "icethd_zdf_bl99": ("ice_thd_zdf_BL99",),
    "icethd_dh": ("ice_thd_dh", "snw_ent"),
    "icethd": ("ice_thd_temp",),
    "icevar": ("ice_var_enthalpy", "ice_var_vremap", "ice_var_snwblow_1d"),
}
OPS = ("minsd", "maxsd", "sqrtsd")
# Variables whose every read/write statement the tie verdicts rest on, per routine.
CENSUS = {
    "ice_thd_dh": ("zq_top", "zq_bot", "h_s_1d", "zh_s", "zdeltah", "zdum", "h_i_1d", "zh_i", "zf_tt",
                   "zevap_rema", "zfmdt"),
    "snw_ent": ("zeh_cum0", "zeh_cum1"),
    "ice_var_vremap": ("zts_cum0", "zts_cum1"),
    "ice_thd_zdf_BL99": ("t_i_1d", "l_T_converged", "zdti_max", "ice_var_enthalpy"),
    "ice_var_enthalpy": ("t_i_1d",),
}
TOKEN = re.compile(r"\b(MIN|MAX|SIGN|SQRT)\s*\(", re.IGNORECASE)


# --------------------------------------------------------------------------- tie verdicts
# One verdict per executing MIN/MAX site tag (``obj:line`` or ``obj:line.<min|max|max1|max2>``):
# FOUND (a committed cell makes the surviving operand change output bits; tested),
# NO_TIE (no differently-signed zero can meet at the site), MASKED (the sign of a zero
# result never reaches an output), NOT_ON_DECK (the arm does not execute on this deck).
# Each proof is short and cites TIE_LEMMAS; its ``facts`` are the premises that
# tests/ice/unit/test_si3_o9_tie_survivors.py checks against this report (operand kinds,
# census lines, enclosing blocks, deck values) and, for MASKED, by flipping the result.
TIE_LEMMAS = {
    "L0": "Only differently-signed zeros (or a NaN) tie with a visible survivor: equal nonzero doubles are "
          "bit-identical.  Deck values are finite (NaN ties out of scope); results that need a subnormal "
          "operand or an underflow are under the stated D5 platform floor (XLA CPU flushes subnormals).",
    "L1": "maxsd/minsd with ZERO as src return +0 on a tie and otherwise the strictly winning VAR or +0: "
          "the result of a ZERO-src site is never -0 (and a MAX(0,.) result is >= +0).",
    "L2": "Round to nearest: a+b = -0 only if a = b = -0; a-b = -0 only if a = -0 and b = +0; a*b and a/b "
          "are -0 only with a zero operand of the other sign, or by underflow (subnormal scale, L0).",
    "L3": "EXP(+0) = EXP(-0) = 1 exactly; a comparison does not see the sign of a zero; MIN(0, x) with ZERO "
          "src is +0 for x = +0 and x = -0.",
}
_W = lambda routine, var, lines: ["writes", routine, var, lines]  # noqa: E731
_U = lambda routine, var, lines: ["uses", routine, var, lines]  # noqa: E731
_DH = "ice_thd_dh"
_ZQTOP = ("A zero zq_top keeps its sign only until the next surface layer that does not melt "
          "internally: there zdum = -(zq_top/zdE)*r1_rhoi is a zero (or the same 0/0 default NaN), "
          "MIN(0, MAX(zdum, -zh_i)) returns +0 for either sign (L3, dh:272.min ZERO src) and "
          "MAX(0, zq_top - (+0)*rhoi*zdE) returns +0 (L1).  zfmdt of 268 is read only at 270; the "
          "internal-melt arm and the snow loop (only under zq_top > 0, L3) do not see the sign; "
          "zq_top is dead after the layer loop.")
_ZQBOT = ("Same algebra as zq_top at 405/407/409/411 (dh:409.min ZERO src): a zero zq_bot gives +0 "
          "from the first basal non-internal-melt layer on, and is dead after the loop.")
_ZQTOP_FACTS = [_U(_DH, "zq_top", [128, 219, 221, 229, 268, 274]),
                _U(_DH, "zfmdt", [253, 257, 268, 270, 278, 280, 282, 283, 358, 360, 361, 392, 394, 405,
                                  407, 415, 417, 419, 420, 462, 464, 466, 467, 471, 484, 486]),
                ["encl", "dh:274", "ELSE of IF( t_i_1d(ji,jk) >= (ztmelts+rt0) )"],
                ["encl", "dh:229", "IF( zh_s(jk) > 0._wp .AND. zq_top > 0._wp )"],
                ["src_zero", "dh:272.min"], ["src_zero", "dh:274"], ["flip"]]
_ZQBOT_FACTS = [_U(_DH, "zq_bot", [130, 405, 411]),
                ["encl", "dh:411", "ELSE of IF( t_i_1d(ji,jk) >= (ztmelts+rt0) )"],
                ["src_zero", "dh:409.min"], ["src_zero", "dh:411"], ["flip"]]
_HS_W = _W(_DH, "h_s_1d", [170, 186, 206, 230, 445, 459, 539])
_ZHS_W = _W(_DH, "zh_s", [150, 153, 171, 179, 207, 231, 447, 491])
_ZDH_W = _W(_DH, "zdeltah", [194, 211, 481, 494])
_ZDUM_W = _W(_DH, "zdum", [200, 221, 222, 251, 270, 272, 300, 388, 407, 409, 490])
_ZHS0 = ("zh_s(0) is +0 (150) or, under sprecip > 0, zsnw*sprecip*rDt*r1_rhos/at_i (179) with "
         "zsnw = 1 - (1-at_i)**rn_snwblow and at_i in (0, 1]: >= +0 and never -0 (L2).")
_SUM_ARMS = ["encl", "dh:251", "IF( t_i_1d(ji,jk) >= (ztmelts+rt0) )"]
_BOM_ARMS = ["encl", "dh:388", "IF( t_i_1d(ji,jk) >= (ztmelts+rt0) )"]
_KELVIN_MAX = ("The other operand is rt0 - 100._wp = 173.15 (the statement's only subtraction, subsd@577 "
               "hoisted): nonzero, so equal operands are bit-identical (L0).")
_KELVIN_MIN = ("The MIN result is the other operand of MAX(., rt0 - 100._wp): a zero result returns "
               "173.15 for either sign, so no tie at the MIN reaches the output.")
_REMAP = ("VAR = c1(j) - c1(j-1) is -0 only if c1(j) = -0 (L2).  c1 is written as +0, as the last running "
          "sum c0 (a sum from +0, never -0, L2), or as (A*B + C*D)/E under the guard "
          "h1(j) <= h0(k) .AND. h1(j) > h0(k-1), where D = h1(j) - h0(k-1) > 0 and E = h0(k) - h0(k-1) > 0: "
          "C*D with C = c0(k) != -0 is not -0 (L2, underflow is L0), so the numerator is not -0 and the "
          "quotient by E > 0 is -0 only by underflow (L0).  No tie.")
TIE_VERDICTS = {
    # ---- icethd.o: ice_thd_temp
    "thd:285": {"verdict": "NO_TIE", "facts": [["src_zero", "thd:285"]],
                "proof": "VAR = zbbb*zbbb - 4*rcpi*rLfus*ztmelts; a square is never -0, so VAR is never -0 (L2)."},
    # ---- icethd_dh.o: ice_thd_dh
    "dh:128": {"verdict": "MASKED", "proof": _ZQTOP, "facts": [["src_zero", "dh:128"]] + _ZQTOP_FACTS},
    "dh:130": {"verdict": "MASKED", "proof": _ZQBOT, "facts": [["src_zero", "dh:130"], ["tie_masked"]] + _ZQBOT_FACTS},
    "dh:170": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:170"], _HS_W, _ZHS_W],
               "proof": "VAR = h_s_1d - zh_s(jk) is -0 only if h_s_1d = -0 and zh_s(jk) = +0 (L2).  170 is h_s_1d's "
                        "first write, so h_s_1d is the entry value or an earlier 170 result (never -0, L1); "
                        "zh_s(jk) is still h_s_entry*r1_nlay_s (153; 171 follows), which is -0 when h_s_entry is."},
    "dh:180": {"verdict": "FOUND", "facts": [["found"]], "proof": "Tie cell committed; survivor test."},
    "dh:194": {"verdict": "FOUND", "facts": [["found"], ["roles"]], "proof": "Tie cell committed; survivor test."},
    "dh:200": {"verdict": "FOUND", "facts": [["found"], ["roles"]], "proof": "Tie cell committed; survivor test."},
    "dh:206": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:206"], _HS_W, _ZHS_W, _ZDH_W, ["roles_of", "dh:194"],
                                              ["roles_of", "dh:200"], ["encl_only_do", "dh:206"]],
               "proof": "VAR = h_s_1d + zdum is -0 only if both are -0 (L2).  For jk >= 1 h_s_1d is the previous 206 "
                        "result (never -0, L1).  For jk = 0 h_s_1d = -0 only if it is the -0 entry value with neither "
                        "170 nor 186 run (186 adds zh_s(0) != -0). " + _ZHS0 + "  Then zh_s(0) = +0, "
                        "zdeltah = -MIN(A, h_s_1d) with the tie returning src h_s_1d = -0 (dh:194 roles) is >= +0, "
                        "and zdum = MAX(-0, zdeltah) returns src zdeltah (dh:200 roles) on a tie: never -0."},
    "dh:207": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:207"], ["src_zero", "dh:211"], _ZHS_W, _ZDH_W,
                                              ["encl_only_do", "dh:207"]],
               "proof": "VAR = zh_s(jk) + zdum is -0 only if both are -0 (L2). " + _ZHS0 + "  For jk >= 1, if "
                        "zh_s(jk) = -0 then zdum = MAX(-zh_s(jk), zdeltah) has operands +0 and zdeltah, a 211 result "
                        "(ZERO src, never -0, L1): zdum is never -0."},
    "dh:211": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:211"], _ZHS_W, _ZDH_W, ["roles_of", "dh:200"]],
               "proof": "VAR = zdeltah - zdum is -0 only if zdeltah = -0 and zdum = +0 (L2).  For jk >= 1 zdeltah is "
                        "the previous 211 result (never -0, L1).  For jk = 0, zdeltah = -0 gives zdum = MAX(-zh_s(0), -0) "
                        "with -zh_s(0) in {-0, negative} (zh_s(0) >= +0, never -0), which is -0, not +0."},
    "dh:221": {"verdict": "NO_TIE", "facts": [["const", "dh:221"]],
               "proof": "One operand is the nonzero constant epsi20 (L0)."},
    "dh:228": {"verdict": "NO_TIE", "facts": [["roles"], ["encl", "dh:228", "IF( zh_s(jk) > 0._wp .AND. zq_top > 0._wp )"], _ZHS_W],
               "proof": "Emitted MIN(zq_top/MAX(ze_s,epsi20), zh_s(jk)) of statement 222: zh_s(jk) (src, loaded at 219) "
                        "is > 0 under the enclosing IF and unwritten until 231, so equal operands are bit-identical (L0)."},
    "dh:229": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:229"], ["encl", "dh:229", "IF( zh_s(jk) > 0._wp .AND. zq_top > 0._wp )"],
                                              _W(_DH, "zq_top", [128, 229, 274])],
               "proof": "Under the enclosing IF zq_top > 0, unwritten between 219 and 229: VAR = zq_top + zdum*ze_s(jk) "
                        "has a positive addend and is never -0 (L2)."},
    "dh:230": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:230"], _HS_W, ["encl_only_do", "dh:206"]],
               "proof": "h_s_1d was last written by 206 (in the unconditional DO jk = 0, nlay_s, after 170/186) or by an "
                        "earlier 230: never -0 (L1), so VAR = h_s_1d + zdum is never -0 (L2)."},
    "dh:231": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:231"], ["encl", "dh:231", "IF( zh_s(jk) > 0._wp .AND. zq_top > 0._wp )"], _ZHS_W],
               "proof": "zh_s(jk) > 0 under the enclosing IF and unwritten between 219 and 231: VAR = zh_s(jk) + zdum is "
                        "never -0 (L2)."},
    "dh:251": {"verdict": "FOUND", "facts": [["found"]], "proof": "Tie cell committed; survivor test."},
    "dh:272.min": {"verdict": "FOUND", "facts": [["found"]], "proof": "Tie cell committed; survivor test."},
    "dh:272.max": {"verdict": "MASKED", "facts": [["roles"], ["tie_masked"], ["src_zero", "dh:272.min"], ["flip"]],
                   "proof": "The MAX result is consumed only by the enclosing MIN(0._wp, .) of the same statement, "
                            "whose ZERO src returns +0 for either zero sign (L3)."},
    "dh:274": {"verdict": "MASKED", "proof": _ZQTOP, "facts": _ZQTOP_FACTS},
    "dh:289": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:289"], _ZDUM_W, _SUM_ARMS, ["src_zero", "dh:251"],
                                              ["src_zero", "dh:272.min"]],
               "proof": "zdum was last written by 251 (internal-melt arm) or 272 (other arm), both MIN(0, .) with ZERO "
                        "src: never -0 (L1), so VAR = zh_i(jk) + zdum is never -0 (L2)."},
    "dh:290": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:290"], _ZDUM_W, _SUM_ARMS, ["src_zero", "dh:251"],
                                              ["src_zero", "dh:272.min"]],
               "proof": "Same zdum as dh:289 (never -0): VAR = h_i_1d + zdum is never -0 (L2)."},
    "dh:295": {"verdict": "FOUND", "facts": [["found"], ["roles"]],
               "proof": "Tie cell committed (every layer melts out, -0 evaporation): emitted -MIN(zevap_rema*r1_rhoi, zh_i) "
                        "of statement 300 changes wfx_err_sub; survivor test."},
    "dh:310": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:310"], _W(_DH, "zh_i", [141, 146, 289, 310, 366, 426, 480])],
               "proof": "zh_i(jk) was last written by 289 (never -0, L1): VAR = zh_i(jk) + zdum is never -0 (L2)."},
    "dh:311": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:311"], _W(_DH, "h_i_1d", [290, 311, 367, 427, 458])],
               "proof": "h_i_1d was last written by 290 (never -0, L1): VAR = h_i_1d + zdum is never -0 (L2)."},
    "dh:348.min": {"verdict": "NO_TIE", "facts": [["const", "dh:348.min"]],
                   "proof": "One operand is the nonzero constant -epsi10 (L0)."},
    "dh:348.max": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:348.max"]],
                   "proof": "VAR = 1.0 - ztmelts/MIN(.): x - y is -0 only if x = -0 (L2); x = 1.0."},
    "dh:355": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:355"], ["encl", "dh:355", "IF( zf_tt < 0._wp )"],
                                              _W(_DH, "zf_tt", [129]), ["deck", "nn_icesal", 4]],
               "proof": "Under IF(zf_tt < 0) VAR = zf_tt/(zdE*rhoi) with zf_tt negative: -0 only by underflow (L2). With "
                        "sss >= 0 (so ztmelts <= 0 and the latent MAX term is in [0, rLfus]) and |t_bo - rt0| < 1000 K, "
                        "|zdE*rhoi| < 2e10, so underflow needs |zf_tt| < 2e10 * 2^-1074: a subnormal zf_tt (L0)."},
    "dh:388": {"verdict": "FOUND", "facts": [["found"]], "proof": "Tie cell committed; survivor test."},
    "dh:409.min": {"verdict": "FOUND", "facts": [["found"]], "proof": "Tie cell committed; survivor test."},
    "dh:409.max": {"verdict": "MASKED", "facts": [["roles"], ["src_zero", "dh:409.min"], ["flip"]],
                   "proof": "The MAX result is consumed only by the enclosing MIN(0._wp, .) of the same statement, "
                            "whose ZERO src returns +0 for either zero sign (L3)."},
    "dh:411": {"verdict": "MASKED", "proof": _ZQBOT, "facts": _ZQBOT_FACTS},
    "dh:426": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:426"], _ZDUM_W, _BOM_ARMS, ["src_zero", "dh:388"],
                                              ["src_zero", "dh:409.min"]],
               "proof": "zdum was last written by 388 or 409 (MIN(0, .) with ZERO src, never -0, L1): VAR = zh_i(jk) + "
                        "zdum is never -0 (L2)."},
    "dh:427": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:427"], _ZDUM_W, _BOM_ARMS, ["src_zero", "dh:388"],
                                              ["src_zero", "dh:409.min"]],
               "proof": "Same zdum as dh:426 (never -0): VAR = h_i_1d + zdum is never -0 (L2)."},
    "dh:456": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:456"], _HS_W, ["stmt", _DH, "h_s_1d", 445, "h_s_1d(ji) = 0._wp"],
                                              ["encl_only_do", "dh:206"]],
               "proof": "h_s_1d was last written by 206/230 (MAX(0, .), >= +0, never -0, L1) or 445 (+0), so rhos*h_s_1d is "
                        "not -0 and the sum with (rhoi-rho0)*h_i_1d is not -0 (L2); times z1_rho > 0 it is -0 only by "
                        "underflow of a subnormal-scale sum (L0)."},
    "dh:490": {"verdict": "NO_TIE", "facts": [["roles"], _ZHS_W, _ZDH_W, ["encl_only_do", "dh:207"],
                                              ["stmt", _DH, "zdeltah", 481, "zdeltah = dh_snowice(ji)"],
                                              ["src_zero", "dh:456"], ["src_zero", "dh:494"], ["src_zero", "dh:491"]],
               "proof": "Neither operand is ever -0: zdeltah is the 456 result (481) or a 494 result (ZERO src, L1); "
                        "zh_s(jk) was rewritten for every jk by 207 (unconditional loop, L1) and afterwards only by 231, "
                        "491 (ZERO src) or 447 (+0).  A zero tie is +0 against +0 (L0)."},
    "dh:491": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:491"], _ZHS_W, ["encl_only_do", "dh:207"]],
               "proof": "VAR = zh_s(jk) - zdum is -0 only if zh_s(jk) = -0 (L2), excluded as in dh:490."},
    "dh:494": {"verdict": "NO_TIE", "facts": [["src_zero", "dh:494"], _ZDH_W, ["src_zero", "dh:456"]],
               "proof": "VAR = zdeltah - zdum is -0 only if zdeltah = -0 (L2); zdeltah is a 456/494 result (never -0, L1)."},
    "dh:525": {"verdict": "NOT_ON_DECK", "facts": [["encl", "dh:525", "IF( nn_icesal == 2 )"], ["deck", "nn_icesal", 4]],
               "proof": "Arm of nn_icesal == 2; the deck sets nn_icesal = 4."},
    # ---- icethd_dh.o: snw_ent
    "dh:619.max1": {"verdict": "NO_TIE", "proof": _REMAP + "  (c = zeh_cum, h = zh_cum; 584-615)",
                    "facts": [["src_zero", "dh:619.max1"], _W("snw_ent", "zeh_cum0", [584, 587]),
                              _W("snw_ent", "zeh_cum1", [603, 608, 615]),
                              ["census_encl", "snw_ent", "zeh_cum1", 608,
                               "IF( zh_cum1(jk1) <= zh_cum0(jk0) .AND. zh_cum1(jk1) > zh_cum0(jk0-1) )"]]},
    "dh:619.max2": {"verdict": "NO_TIE", "facts": [["const", "dh:619.max2"]],
                    "proof": "One operand is the nonzero constant epsi20 (L0)."},
    # ---- icethd_zdf_bl99.o: ice_thd_zdf_BL99
    **{f"zdf:{n}": {"verdict": "NO_TIE", "facts": [["const", f"zdf:{n}"]],
                    "proof": "One operand is a nonzero constant (zh_min, -epsi10, zkimin or epsi10) (L0)."}
       for n in (173, 181, 274, 276, 281, 292, 353)},
    "zdf:202": {"verdict": "NO_TIE", "facts": [["defs", "zdf:202", "src", "subsd@202"]],
                "proof": "The other operand rt0 - ztsu_err (the statement's only subtraction) is 273.14999 K: nonzero (L0)."},
    "zdf:219": {"verdict": "MASKED", "facts": [["src_zero", "zdf:219"], ["flip"],
                                               ["text", "zdf:219", "EXP( - zraext_s(ji) * MAX( 0._wp,"]],
                "proof": "The MAX result only multiplies -zraext_s inside EXP: EXP(+0) = EXP(-0) = 1 exactly (L3)."},
    "zdf:229": {"verdict": "MASKED", "facts": [["src_zero", "zdf:229"], ["flip"],
                                               ["text", "zdf:229", "EXP( - rn_kappa_i * MAX( 0._wp,"]],
                "proof": "Both MAX results only multiply -rn_kappa_i inside EXP: EXP(+0) = EXP(-0) = 1 exactly (L3)."},
    **{f"zdf:{n}.max": {"verdict": "NO_TIE", "facts": [["defs", f"zdf:{n}.max", "src", "subsd@577"],
                                                       ["text", f"zdf:{n}", "rt0 - 100._wp )"]],
                        "proof": _KELVIN_MAX} for n in (577, 582, 589)},
    **{f"zdf:{n}.min": {"verdict": "MASKED", "facts": [["defs", f"zdf:{n}.max", "src", "subsd@577"],
                                                       ["text", f"zdf:{n}", "MAX( MIN("]],
                        "proof": _KELVIN_MIN} for n in (577, 582, 589)},
    **{f"zdf:{n}": {"verdict": "NO_TIE", "facts": [["abs", f"zdf:{n}"]],
                    "proof": "Every operand is an ABS(.) result or a MAX of them, so a zero is +0 on both sides (L0)."}
       for n in (583, 590)},
    **{f"zdf:{n}": {"verdict": "NOT_ON_DECK", "facts": [["encl", f"zdf:{n}", "IF( ln_cndi_U64 )"], ["deck", "ln_cndi_u64", False]],
                    "proof": "Arm of ln_cndi_U64; the deck sets ln_cndi_U64 = .false."} for n in (261, 262, 266)},
    "zdf:309": {"verdict": "NOT_ON_DECK", "facts": [["encl", "zdf:309", "IF( ln_virtual_itd )"], ["deck", "ln_virtual_itd", False]],
                "proof": "Arm of ln_virtual_itd; the deck sets ln_virtual_itd = .false."},
    **{f"zdf:{n}": {"verdict": "NOT_ON_DECK", "facts": [["encl", f"zdf:{n}", "ELSEIF( k_cnd == np_cnd_ON )"],
                                                        ["deck", "ln_cndflx", False]],
                    "proof": "Arm of k_cnd == np_cnd_ON; with ln_cndflx = .false. ice_thd_zdf calls only "
                             "ice_thd_zdf_BL99( np_cnd_OFF ) (icethd_zdf.f90:115-116)."}
       for n in (747, 748, 754, 755, 800, 803)},
    # ---- icevar.o
    "var:955": {"verdict": "NO_TIE",
                "facts": [_U("ice_thd_zdf_BL99", "ice_var_enthalpy", [813]),
                          _W("ice_thd_zdf_BL99", "t_i_1d", [544, 548, 589, 719, 723, 754, 865]),
                          _W("ice_thd_zdf_BL99", "l_T_converged", [245, 599, 764]),
                          ["stmt", "ice_thd_zdf_BL99", "l_T_converged", 245, "l_T_converged(:) = .FALSE."],
                          ["defs", "zdf:589.max", "src", "subsd@577"]],
                "proof": "t_i_1d enters ice_var_enthalpy (called once, 813, after the iteration) as the 589 result of its "
                         "cell's last iteration (l_T_converged starts .FALSE., so 589 runs at least once and follows "
                         "544/548 in the same arm; 719/723/754/865 are np_cnd_ON/EMU): MAX(., 173.15) >= 173.15, "
                         "nonzero (L0)."},
    "var:957.min": {"verdict": "NO_TIE", "facts": [["const", "var:957.min"]],
                    "proof": "One operand is the nonzero constant -epsi10 (L0)."},
    "var:957.max": {"verdict": "NO_TIE", "facts": [["src_zero", "var:957.max"]],
                    "proof": "VAR = 1._wp - ztmelts/MIN(.): x - y is -0 only if x = -0 (L2); x = 1."},
    "var:1045.max1": {"verdict": "NO_TIE", "proof": _REMAP + "  (c = zts_cum, h = zh_cum; 1010-1041)",
                      "facts": [["src_zero", "var:1045.max1"], _W("ice_var_vremap", "zts_cum0", [1010, 1013]),
                                _W("ice_var_vremap", "zts_cum1", [1029, 1034, 1041]),
                                ["census_encl", "ice_var_vremap", "zts_cum1", 1034,
                                 "IF( zh_cum1(jk1) <= zh_cum0(jk0) .AND. zh_cum1(jk1) > zh_cum0(jk0-1) )"]]},
    "var:1045.max2": {"verdict": "NO_TIE", "facts": [["const", "var:1045.max2"]],
                      "proof": "One operand is the nonzero constant epsi20 (L0)."},
}


def _flags(build: Path) -> list[str]:
    text = (build / "arch_nemo.fcm").read_text()
    m = re.search(r"^%FCFLAGS\s+(.*)$", text, re.MULTILINE)
    if not m:
        raise SystemExit("no %FCFLAGS in arch_nemo.fcm")
    return m.group(1).split()


def _stream(obj: Path) -> str:
    """Instruction stream with addresses and symbolic targets removed."""
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)],
                         check=True, capture_output=True, text=True).stdout
    lines = []
    for line in out.splitlines():
        m = re.match(r"^\s+[0-9a-f]+:\t(.*)$", line)
        if m:
            lines.append(re.sub(r"[0-9a-f]+ <[^>]*>", "ADDR", m.group(1)).split("#")[0].rstrip())
    return "\n".join(lines)


def _rodata(obj: Path) -> dict[str, bytes]:
    out = subprocess.run(["objdump", "-s", str(obj)], check=True, capture_output=True,
                         text=True).stdout
    sections, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"Contents of section (\S+):", line)
        if m:
            cur = m.group(1) if m.group(1).startswith(".rodata") else None
            if cur:
                sections[cur] = bytearray()
            continue
        m = re.match(r"^ ([0-9a-f]{4,}) ((?:[0-9a-f]{2,8} ?){1,4})", line)
        if m and cur:
            sections[cur] += bytes.fromhex(m.group(2).replace(" ", ""))
    return {k: bytes(v) for k, v in sections.items()}


def _symbols(obj: Path) -> dict[str, tuple[str, int]]:
    """Local ``.LC*`` symbol -> (section, offset)."""
    out = subprocess.run(["objdump", "-t", str(obj)], check=True, capture_output=True,
                         text=True).stdout
    syms = {}
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]+)\s.*\s(\.rodata\S*)\s+[0-9a-f]+\s+(\S+)$", line)
        if m:
            syms[m.group(3)] = (m.group(2), int(m.group(1), 16))
    return syms


def _const(reloc: str | None, ro: dict[str, bytes], syms: dict) -> str | None:
    """Resolve a PC32 relocation (``.LC3-0x4`` or ``.rodata.cst8+0x10-0x4``) to the double's hex."""
    if not reloc:
        return None
    m = re.match(r"([.\w]+)((?:[+-]0x[0-9a-f]+)*)$", reloc)
    if not m:
        return None
    name, addend = m.group(1), sum(int(x, 16) for x in re.findall(r"[+-]0x[0-9a-f]+", m.group(2)))
    if name in syms:
        sec, base = syms[name]
    elif name in ro:
        sec, base = name, 0
    else:
        return None
    off = base + addend + 4
    data = ro[sec]
    if off < 0 or off + 8 > len(data):
        return None
    return float.hex(struct.unpack("<d", data[off:off + 8])[0])


def _split(args: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in args:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _parse(obj: Path, src_name: str):
    """Instructions per function: dicts with addr, op, args, reloc, line."""
    out = subprocess.run(["objdump", "-drl", "--no-show-raw-insn", str(obj)], check=True,
                         capture_output=True, text=True).stdout
    funcs, func, line = collections.OrderedDict(), None, None
    for raw in out.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", raw)
        if m:
            func = m.group(1)
            funcs[func] = []
            continue
        m = re.match(r"^/\S*/" + re.escape(src_name) + r":(\d+)", raw)
        if m:
            line = int(m.group(1))
            continue
        if re.match(r"^/\S*\.f90:\d+", raw):  # a line from another file (none expected)
            line = None
            continue
        m = re.match(r"^\s+[0-9a-f]+: (R_X86_64_\w+)\s+(\S+)", raw)
        if m and func and funcs[func]:
            funcs[func][-1]["reloc"] = m.group(2)
            continue
        m = re.match(r"^\s*([0-9a-f]+):\t(\S+)\s*(.*?)\s*$", raw)
        if m and func:
            funcs[func].append(dict(addr=int(m.group(1), 16), op=m.group(2),
                                    args=_split(m.group(3).split("#")[0].strip()),
                                    reloc=None, line=line))
    return funcs


_NOWRITE = ("ucomi", "comi", "cmp", "test", "j", "push", "ret", "nop", "endbr")


def _effect(ins, ro, syms):
    """(register written, kind or ('COPY', reg)) for an xmm write, else None; 'CALL' clobbers all."""
    op, a = ins["op"], ins["args"]
    if op.startswith("call"):
        return "CALL"
    if not a or not a[-1].startswith("%xmm") or op.startswith(_NOWRITE):
        return None
    dst = a[-1]
    if op in ("pxor", "xorpd", "xorps") and len(a) == 2 and a[0] == dst:
        return dst, "ZERO"
    if op.startswith(("movsd", "movq", "movapd", "movaps", "movupd")) and len(a) == 2:
        if "%rip" in a[0]:
            c = _const(ins["reloc"], ro, syms)
            return dst, f"CONST:{c}" if c else "VAR"
        if a[0].startswith("%xmm"):
            return dst, ("COPY", a[0])
        return dst, "VAR"
    return dst, "VAR"


def _cfg(insns):
    addrs = [i["addr"] for i in insns]
    index = {a: k for k, a in enumerate(addrs)}
    succ = []
    for k, ins in enumerate(insns):
        op, nxt = ins["op"], ([k + 1] if k + 1 < len(insns) else [])
        if op.startswith("j"):
            t = re.match(r"^([0-9a-f]+)", ins["args"][0]) if ins["args"] else None
            if not t or "*" in ins["args"][0]:
                raise SystemExit(f"indirect jump at {ins['addr']:#x}; CFG not exact")
            tgt = int(t.group(1), 16)
            targets = [index[tgt]] if tgt in index else []
            succ.append(targets if op in ("jmp", "jmpq") else targets + nxt)
        elif op.startswith("ret") or op.startswith("ud2"):
            succ.append([])
        else:
            succ.append(nxt)
    return succ


def _reaching(insns, ro, syms):
    """Per instruction: the definitions (instruction index or 'ENTRY') of each xmm register reaching it."""
    succ = _cfg(insns)
    regs = [f"%xmm{n}" for n in range(16)]
    effects = [_effect(i, ro, syms) for i in insns]
    inn = [None] * len(insns)
    inn[0] = {r: frozenset({"ENTRY"}) for r in regs}
    work = [0]
    while work:
        k = work.pop()
        state = dict(inn[k])
        e = effects[k]
        if e == "CALL":
            state = {r: frozenset({("CALL", k)}) for r in regs}
        elif e:
            state[e[0]] = frozenset({k})
        for s in succ[k]:
            if inn[s] is None:
                inn[s] = state
                work.append(s)
            else:
                merged = {r: inn[s][r] | state[r] for r in regs}
                if merged != inn[s]:
                    inn[s] = merged
                    work.append(s)
    return inn, effects


def _kinds(k, reg, inn, effects, seen=frozenset()):
    """Set of operand kinds reaching instruction ``k`` in ``reg``."""
    kinds = set()
    for d in inn[k][reg]:
        if d == "ENTRY" or isinstance(d, tuple):
            kinds.add("VAR")
            continue
        e = effects[d][1]
        if isinstance(e, tuple):  # register copy: kinds of the source at the copy
            if (d, e[1]) not in seen:
                kinds |= _kinds(d, e[1], inn, effects, seen | {(d, e[1])})
        else:
            kinds.add(e)
    return kinds


def _kind(k, reg, inn, effects):
    kinds = sorted(_kinds(k, reg, inn, effects))
    return kinds[0] if len(kinds) == 1 else "MIXED(" + "|".join(kinds) + ")"


def _defset(k, reg, insns, inn, effects, seen=frozenset()):
    out = set()
    for d in inn[k][reg]:
        if d == "ENTRY" or isinstance(d, tuple):
            out.add("ENTRY" if d == "ENTRY" else "call")
            continue
        e = effects[d][1]
        if isinstance(e, tuple):
            if (d, e[1]) not in seen:
                out |= _defset(d, e[1], insns, inn, effects, seen | {(d, e[1])})
            continue
        op = insns[d]["op"]
        out.add(f"{'load' if op.startswith(('movsd', 'movq')) else op}@{insns[d]['line']}")
    return out


def _defs(k, reg, insns, inn, effects):
    """Nearest non-copy definitions of ``reg`` at ``k`` as ``op@line`` (``load@line`` for memory loads)."""
    return "|".join(sorted(_defset(k, reg, insns, inn, effects)))


def _operand_defs(k, ins, insns, inn, effects):
    """(src defs, dst defs) for a minsd/maxsd; memory sources are ``load@line``."""
    s, d = ins["args"]
    sd = _defs(k, s, insns, inn, effects) if s.startswith("%xmm") else f"load@{ins['line']}"
    return sd, _defs(k, d, insns, inn, effects)


def _statements(path: Path):
    """Physical line -> (statement first line, statement text without comments)."""
    lines = path.read_text().splitlines()
    owner, text, k = {}, {}, 0
    while k < len(lines):
        start, parts = k + 1, []
        while True:
            code = re.sub(r"!.*$", "", lines[k]).rstrip()
            owner[k + 1] = start
            cont = code.endswith("&")
            parts.append(code.rstrip("&").strip().lstrip("&").strip())
            k += 1
            if not cont or k >= len(lines):
                break
        text[start] = " ".join(p for p in parts if p)
    return owner, text


def _strip_if(text: str):
    """(condition, rest) of a leading ``IF( cond )``/``WHERE( cond )``, else (None, text)."""
    m = re.match(r"^\s*(IF|WHERE)\s*\(", text, re.IGNORECASE)
    if not m:
        return None, text
    depth, k = 0, m.end() - 1
    for k in range(m.end() - 1, len(text)):
        depth += text[k] == "("
        depth -= text[k] == ")"
        if depth == 0:
            break
    return f"{m.group(1).upper()}{text[m.end() - 1:k + 1]}", text[k + 1:].strip()


def _pieces(text: str) -> list[str]:
    """The ``;``-separated statements of one statement line, whitespace-normalised."""
    return [" ".join(p.split()) for p in text.split(";") if p.strip()]


def _blocks(text: dict, ranges) -> dict[int, list[str]]:
    """Statement line -> enclosing DO/IF/ELSEIF/ELSE/CASE/WHERE headers, outermost first."""
    out = {}
    for b, e in ranges.values():
        stack = []
        for ln in range(b, e + 1):
            if ln not in text or not text[ln].strip():
                continue
            for k, t in enumerate(_pieces(text[ln])):
                u = t.upper()
                if re.match(r"^END\s*(IF|DO|SELECT|WHERE)\b", u):
                    out.setdefault(ln, list(stack))
                    stack.pop()
                    continue
                if (re.match(r"^ELSE\s*IF\b", u) or u.startswith("ELSEWHERE") or u == "ELSE"
                        or re.match(r"^CASE\b", u)):
                    prev = stack.pop()
                    out.setdefault(ln, list(stack))
                    arm = re.sub(r"\s*THEN$", "", t, flags=re.IGNORECASE)
                    stack.append(f"ELSE of {prev}" if u in ("ELSE", "ELSEWHERE") else arm)
                    continue
                cond, rest = _strip_if(t)
                out.setdefault(ln, list(stack) + ([cond] if cond and rest and rest.upper() != "THEN" else []))
                if cond and rest.upper() == "THEN":
                    stack.append(cond)
                elif cond and cond.startswith("WHERE") and not rest:
                    stack.append(cond)
                elif not cond and (re.match(r"^DO\b", u) or re.match(r"^SELECT\s+CASE\b", u)):
                    stack.append(t)
        if stack:
            raise SystemExit(f"unbalanced blocks in {b}-{e}: {stack}")
    return out


def _census(text: dict, ranges, blocks, census=CENSUS) -> dict:
    """Routine -> variable -> {statement line: [w|r|wr, statement, enclosing]} for the CENSUS variables."""
    out = {}
    for name, (b, e) in ranges.items():
        for var in census.get(name, ()):
            word = re.compile(rf"\b{var}\b", re.IGNORECASE)
            uses = {}
            for ln in range(b + 1, e):
                if ln not in text or not word.search(text[ln]) or "::" in text[ln]:
                    continue  # declarations are not uses
                flags = ""
                for piece in _pieces(text[ln]):
                    cond, rest = _strip_if(piece)
                    m = re.match(r"^(\w+)\s*(\((?:[^()]|\([^()]*\))*\))?\s*=(?!=)", rest)
                    w = bool(m) and not rest.upper().startswith("DO ") and m.group(1).lower() == var.lower()
                    seen = (cond or "") + " " + (rest[len(m.group(1)):] if w else rest)
                    flags += ("r" if word.search(seen) else "") + ("w" if w else "")
                uses[str(ln)] = ["".join(sorted(set(flags), reverse=True)), " ".join(text[ln].split()),
                                 blocks.get(ln, [])]
            out.setdefault(name, {})[var] = uses
    return out


def _ranges(path: Path, names):
    src = path.read_text().splitlines()
    out = {}
    for name in names:
        beg = next(n for n, l in enumerate(src, 1)
                   if re.match(rf"^\s*(SUBROUTINE|FUNCTION)\s+{name}\b", l, re.IGNORECASE))
        end = next(n for n, l in enumerate(src, 1)
                   if re.match(rf"^\s*END\s+(SUBROUTINE|FUNCTION)\s+{name}\b", l, re.IGNORECASE))
        out[name] = (beg, end)
    return out


def audit(
    build: Path = BUILD,
    toolchain: Path = TOOLCHAIN,
    *,
    routines=ROUTINES,
    census=CENSUS,
    tie_lemmas=TIE_LEMMAS,
    tie_verdicts=TIE_VERDICTS,
) -> dict:
    """Audit selected ppsrc routines against their record-build objects.

    Keyword arguments make the object-code machinery reusable by later SI3
    rungs while preserving the O9 command-line report byte for byte.
    """
    flags = _flags(build)
    env = dict(os.environ, PATH=f"{toolchain / 'bin'}:{os.environ['PATH']}")
    report = {"build": str(build), "flags": flags, "objects": {}, "tie_lemmas": tie_lemmas,
              "tie_verdicts": tie_verdicts}
    with tempfile.TemporaryDirectory() as tmp:
        for obj_name, names in routines.items():
            src = build / "ppsrc/nemo" / f"{obj_name}.f90"
            gobj = Path(tmp) / f"{obj_name}.o"
            subprocess.run(["mpif90", "-c", "-cpp", *flags, "-g", f"-I{build / 'inc'}",
                            f"-J{tmp}", f"-I{toolchain / 'include'}", str(src), "-o", str(gobj)],
                           check=True, env=env, capture_output=True)
            record = build / "obj" / f"{obj_name}.o"
            same = _stream(gobj) == _stream(record)
            if not same:
                raise SystemExit(f"{obj_name}: -g recompile differs from the record object")
            ro = _rodata(gobj)
            syms = _symbols(gobj)
            owner, text = _statements(src)
            ranges = _ranges(src, names)
            blocks = _blocks(text, ranges)
            in_scope = lambda ln: ln is not None and any(b <= ln <= e for b, e in ranges.values())
            sites = collections.defaultdict(lambda: collections.Counter())
            for func, insns in _parse(gobj, src.name).items():
                if not any(i["op"] in OPS and in_scope(i["line"]) for i in insns):
                    continue
                inn, effects = _reaching(insns, ro, syms)
                for k, ins in enumerate(insns):
                    if ins["op"] not in OPS or not in_scope(ins["line"]):
                        continue
                    stmt = owner[ins["line"]]
                    if ins["op"] == "sqrtsd":
                        sites[stmt][("sqrtsd", "-", "-", "-", "-")] += 1
                        continue
                    s, d = ins["args"]
                    if "%rip" in s:
                        c = _const(ins["reloc"], ro, syms)
                        sk = f"CONST:{c}" if c else "VAR"
                    elif s.startswith("%xmm"):
                        sk = _kind(k, s, inn, effects)
                    else:
                        sk = "VAR"
                    dk = _kind(k, d, inn, effects)
                    defs = (_operand_defs(k, ins, insns, inn, effects)
                            if sk == "VAR" and dk == "VAR" else ("-", "-"))
                    sites[stmt][(ins["op"], sk, dk, *defs)] += 1
            tokens = {}
            for name, (b, e) in ranges.items():
                for ln in range(b, e + 1):
                    if owner.get(ln) == ln:
                        found = [t.upper() for t in TOKEN.findall(text[ln])]
                        if found:
                            tokens[ln] = sorted(found)
            report["objects"][obj_name] = {
                "record_object_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "ppsrc_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "routines": {k: list(v) for k, v in ranges.items()},
                "census": _census(text, ranges, blocks, census),
                "sites": {str(ln): {"statement": text[ln], "enclosing": blocks[ln],
                                    "tokens": tokens.get(ln, []),
                                    "instructions": [dict(op=o, src=s, dst=d, count=n,
                                                          **({"src_defs": sd, "dst_defs": dd}
                                                             if sd != "-" else {}))
                                                     for (o, s, d, sd, dd), n in sorted(c.items())]}
                          for ln, c in sorted(sites.items())},
                "token_statements_without_instruction": {
                    str(ln): {"statement": text[ln], "tokens": t}
                    for ln, t in sorted(tokens.items())
                    if ln not in sites and any(x in ("MIN", "MAX", "SQRT") for x in t)},
                "instructions_without_token": sorted(
                    str(ln) for ln, c in sites.items()
                    if not any(t in ("MIN", "MAX", "SQRT") for t in tokens.get(ln, []))),
                "sign_tokens": sorted(str(ln) for ln, t in tokens.items() if "SIGN" in t),
            }
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--build", type=Path, default=BUILD)
    p.add_argument("--toolchain", type=Path, default=TOOLCHAIN)
    p.add_argument("--check", type=Path, help="committed report; exit 1 on any difference")
    args = p.parse_args(argv)
    rep = audit(args.build, args.toolchain)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if args.check:
        same = json.loads(args.check.read_text()) == json.loads(text)
        print("MINMAX_AUDIT", "MATCH" if same else "DIFFERS")
        return 0 if same else 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
