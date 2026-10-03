#!/usr/bin/env python3
"""Every ``file:line`` cited in the receipt must point at the symbol named.

Three rounds in a row produced a wrong-citation finding -- round 25 corrected
NEMO line numbers by hand, round 26 corrected round 25's, and round 27
corrected round 26's (``stp2d.F90:126/:129/:190/:279`` were a comment banner,
a blank line and two off-by-two references, and a code comment called
``stprk3_stg.F90:168-243`` "tracers" when it is the ``r3t/r3u/r3v`` block).
Hand-checking has now failed three times, so this gate replaces it.

How it works, and what it can and cannot see:

* it reads the receipt from a named heading to the end of the file, drops
  fenced code blocks, and walks the inline code spans IN ORDER;
* a span of the form ``<file>:<lines>`` is a citation and also sets the
  current file; a span that is a bare ``<file>`` sets the current file; a span
  of the form ``:<lines>`` is a citation against the current file.  That is
  how the prose reads, and binding continuations wrongly is itself a defect
  the gate would report;
* every extracted citation must appear in ``CITATION_MAP``, which pins the
  RESOLVED path (the same basename exists in ``src/OCE`` and in several
  ``MY_SRC`` overrides) and the symbol the prose names.  An unmapped citation
  is a FAILURE, so a new claim cannot enter the receipt uninspected;
* each mapped symbol must occur in the cited line range of the real file.

Blind spots, written down per Rule 2: it checks that the named symbol IS at
the cited line, not that the line means what the prose says about it; it
cannot see a citation the prose renders without backticks; and a mapped
symbol that is too generic (a bare ``!``) would pass vacuously, which is why
every entry below names an identifier or a distinctive fragment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from legoesm.ocean.fidelity.provenance import worktree_stamp

NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
REPO = Path(__file__).resolve().parents[4]
LOCK = NEMO / "tests/LOCK_EXCHANGE_OMIP_L1_P3"
OVERFLOW_RUN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10")

# path anchors, so one basename cannot silently resolve to a MY_SRC override
_OCE = NEMO / "src/OCE"
_DYN = _OCE / "DYN"
_R35 = NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R35TRAZDF/BLD/ppsrc/nemo"
_ORCA2_COMPILED = NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo"
_ORCA2_R79BZDF_COMPILED = (
    NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo")
_ORCA2_R84FRAMES_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo")
_ORCA2_R85FRAMEDEBUG_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo")
_ORCA2_R87FRAMES_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo")
_ORCA2_R88FRAMEDEBUG_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo")
_ORCA2_R90FRAMES_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo")
_ORCA2_R92RHS_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo")
_ORCA2_R93SLOW_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo")
_ORCA2_R96SPG_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo")
_ORCA2_R104EENACC_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo")
_ORCA2_R105EENACC_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo")
_ORCA2_R107EENSTEP_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo")
_ORCA2_R110EENFRAC_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo")
_ORCA2_R116EENUREC_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo")
_ORCA2_R118EENVREC_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R118EENVREC/BLD/ppsrc/nemo")
_ORCA2_R120EENVFRAC_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo")
_OVERFLOW_COMPILED = NEMO / "tests/OVERFLOW_OMIP_L1/BLD/ppsrc/nemo"
_OVERFLOW_P3_COMPILED = NEMO / "tests/OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo"
_OVERFLOW_R50PAIR_COMPILED = (
    NEMO / "tests/OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo")
_OVERFLOW_R56UP3_COMPILED = (
    NEMO / "tests/OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo")
_OVERFLOW_R62ZDF_COMPILED = (
    NEMO / "tests/OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo")
FILES = {
    "stprk3.F90": _OCE / "stprk3.F90",
    "stprk3_stg.F90": _OCE / "stprk3_stg.F90",
    "stp2d.F90": _OCE / "stp2d.F90",
    "oce.F90": _OCE / "oce.F90",
    "DOM/istate.F90": _OCE / "DOM/istate.F90",
    "domzgr_substitute.h90": _OCE / "DOM/domzgr_substitute.h90",
    "DOM/domzgr.F90": _OCE / "DOM/domzgr.F90",
    "dynspg_ts.F90": _DYN / "dynspg_ts.F90",
    "dynhpg.F90": _DYN / "dynhpg.F90",
    "dynadv.F90": _DYN / "dynadv.F90",
    "dynzdf.F90": _DYN / "dynzdf.F90",
    "dynldf.F90": _DYN / "dynldf.F90",
    "dynldf_lev_rot_scheme.h90": _DYN / "dynldf_lev_rot_scheme.h90",
    "vertical.py": REPO / "packages/ocean/legoesm/ocean/vertical.py",
    # GYRE's own run log and its own compiled branch, so a GYRE resolved value
    # cannot bind to OVERFLOW's ocean.output or to LOCK's ppsrc.
    "LDF/ldfslp.F90": _OCE / "LDF/ldfslp.F90",
    "domqco.F90": _OCE / "DOM/domqco.F90",
    # ORCA2 round 1 binds its first non-bit entry field to the compiled branch
    # that executes the category-summed initial snow/ice load adjustment.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90": (
        _ORCA2_COMPILED / "iceistate.f90"),
    # ORCA2 round 78 binds the turbulence mixing-length floor to the compiled
    # two-arm rmxl_min choice this deck's ln_zdfiwm selects.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90": (
        _ORCA2_COMPILED / "zdftke.f90"),
    # ORCA2 round 79b cites the vertical-physics chain and the internal-wave
    # arm that this deck runs and the card does not.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90": (
        _ORCA2_COMPILED / "zdfphy.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90": (
        _ORCA2_COMPILED / "zdfiwm.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90": (
        _ORCA2_COMPILED / "zdfddm.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90": (
        _ORCA2_COMPILED / "sbcmod.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_oce.f90": (
        _ORCA2_COMPILED / "sbc_oce.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90": (
        _ORCA2_COMPILED / "sbcrnf.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_ice.f90": (
        _ORCA2_COMPILED / "sbc_ice.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbini.f90": (
        _ORCA2_COMPILED / "icbini.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icb_oce.f90": (
        _ORCA2_COMPILED / "icb_oce.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90": (
        _ORCA2_COMPILED / "restart.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcflx.f90": (
        _ORCA2_COMPILED / "sbcflx.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R85FRAMEDEBUG_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R87FRAMES_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R87FRAMES_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/sbc_oce.f90": (
        _ORCA2_R87FRAMES_COMPILED / "sbc_oce.f90"),
    "ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R88FRAMEDEBUG_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/sbc_oce.f90": (
        _ORCA2_R88FRAMEDEBUG_COMPILED / "sbc_oce.f90"),
    "ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R88FRAMEDEBUG_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R90FRAMES_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R90FRAMES_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90": (
        _ORCA2_R90FRAMES_COMPILED / "zdfphy.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfdrg.f90": (
        _ORCA2_R90FRAMES_COMPILED / "zdfdrg.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/istate.f90": (
        _ORCA2_R90FRAMES_COMPILED / "istate.f90"),
    "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R92RHS_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R93SLOW_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R93SLOW_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R96SPG_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynvor.f90": (
        _ORCA2_R96SPG_COMPILED / "dynvor.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domzgr.f90": (
        _ORCA2_R96SPG_COMPILED / "domzgr.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R96SPG_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg.f90": (
        _ORCA2_R104EENACC_COMPILED / "dynspg.f90"),
    "ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R104EENACC_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R105EENACC_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R105EENACC_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R107EENSTEP_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/l4_r105_een_accum.f90": (
        _ORCA2_R107EENSTEP_COMPILED / "l4_r105_een_accum.f90"),
    "ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R110EENFRAC_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R110EENFRAC_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90": (
        _ORCA2_R110EENFRAC_COMPILED / "domhgr.f90"),
    "ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domzgr.f90": (
        _ORCA2_R110EENFRAC_COMPILED / "domzgr.f90"),
    "ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynvor.f90": (
        _ORCA2_R110EENFRAC_COMPILED / "dynvor.f90"),
    "ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90": (
        _ORCA2_R110EENFRAC_COMPILED / "lbclnk.f90"),
    "ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R116EENUREC_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R116EENUREC_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90": (
        _ORCA2_R116EENUREC_COMPILED / "lbclnk.f90"),
    "ORCA2_OMIP_L4_R118EENVREC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R118EENVREC_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R120EENVFRAC_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R120EENVFRAC_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domhgr.f90": (
        _ORCA2_R120EENVFRAC_COMPILED / "domhgr.f90"),
    "ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R120EENVFRAC_COMPILED / "lbcnfd.f90"),
    "ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90": (
        _ORCA2_R120EENVFRAC_COMPILED / "dynvor.f90"),
    "l4_r104_een_accum.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        "/nemo_testcase_l4_orca2_round104_een_accum_acquisition"
        "/l4_r104_een_accum.F90"),
    # Round 80 uses the compiled, instrumented branch that produced the
    # admitted per-step avt/avm record, rather than a nearby pristine deck.
    "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90": (
        _ORCA2_R79BZDF_COMPILED / "zdfphy.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90": (
        _ORCA2_R79BZDF_COMPILED / "zdfiwm.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90": (
        _ORCA2_R79BZDF_COMPILED / "trazdf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90": (
        _ORCA2_R79BZDF_COMPILED / "dynzdf.f90"),
    # Round 85 diagnoses the optimized frame-build crash against the exact
    # preprocessed source that produced it.  A debug build must resolve the
    # precise failing line before either statement can be repaired.
    "ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R84FRAMES_COMPILED / "stprk3.f90"),
    "OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _OVERFLOW_COMPILED / "stprk3_stg.f90"),
    "OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/dynhpg.f90": (
        _OVERFLOW_COMPILED / "dynhpg.f90"),
    "OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _OVERFLOW_P3_COMPILED / "stprk3_stg.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "stprk3_stg.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "stprk3.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "dynvor.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/eosbn2.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "eosbn2.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynhpg.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "dynhpg.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "dynadv.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "dynadv_up3.f90"),
    "OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90": (
        _OVERFLOW_R50PAIR_COMPILED / "dynzdf.f90"),
    "OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90": (
        _OVERFLOW_R56UP3_COMPILED / "dynadv_up3.f90"),
    # ORCA2 round 61: the two decks themselves, so the card-scoping claim
    # cites the switch each card resolves rather than describing it.
    "ORCA2_OMIP_L4/EXP00/namelist_cfg": (
        NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg"),
    "OVERFLOW_OMIP_L1/EXP00/namelist_cfg": (
        NEMO / "tests/OVERFLOW_OMIP_L1/EXP00/namelist_cfg"),
    # The namelist the CERTIFIED OVERFLOW reference run actually used, which
    # is the zps variant and not the deck's EXP00 copy.
    "overflow_kt1_10/namelist_cfg": OVERFLOW_RUN / "namelist_cfg",
    "OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90": (
        _OVERFLOW_R62ZDF_COMPILED / "dynzdf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/dynspg_ts.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/traadv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/traadv_cen.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/fldread.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo"
        "/restart.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dtatsd.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/eosbn2.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/istate.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/iceistate.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/mppini.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/mppini.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dynvor.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo"
        "/dynspg_ts.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/in_out_manager.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo"
        "/in_out_manager.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo"
        "/dynspg_ts.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo"
        "/dynspg_ts.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo"
        "/lbclnk.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/dynvor.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/domain.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/domqco.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/domhgr.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dommsk.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/dommsk.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/divhor.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/traadv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo"
        "/dynhpg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo"
        "/eosbn2.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo"
        "/dynhpg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/dynldf_lev.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo"
        "/ldfdyn.f90"),
    # --- round 40 paths: the stage-3 momentum operators and zdf_mxl ---
    "dynvor.F90": _DYN / "dynvor.F90",
    "dynkeg.F90": _DYN / "dynkeg.F90",
    "dynzad.F90": _DYN / "dynzad.F90",
    "ZDF/zdfmxl.F90": _OCE / "ZDF/zdfmxl.F90",
    # The configuration's OWN keys and its OWN compiled vorticity branch, so a
    # key_qco claim cannot bind to LOCK's ppsrc through the bare key above.
    "cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2"
        "/cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm"),
    "GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo"
        "/dynvor.f90"),
    # Round 41 cites the exact round-40 compiled GYRE branches it instruments.
    "GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynkeg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynkeg.f90"),
    "GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynzad.f90"),
    "GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sshwzv.f90"),
    "GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90"),
    # Round 42 cites its own compiled/instrumented build, including the call
    # site which binds dyn_adv's two formal time-level arguments to Kmm.
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynzad.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynkeg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynkeg.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90"),
    # Round 134 binds the reset contract to the compiled branch that produced
    # the admitted 360-day daily-restart record, rather than to its older
    # byte-identical source card.
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf_iso.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/sshwzv.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/traqsr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/traqsr.f90"),
    # Round 187 reconciles the exact compiled writer that produced the
    # developed shortwave record, including its no-halo surface allocation and
    # the persistent RK slot rotation at step 1080.
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/sbc_oce.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/sbc_oce.f90"),
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/mppini.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/mppini.f90"),
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/nemogcm.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/nemogcm.f90"),
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90"),
    "GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r54_tke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r54_tke.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r101_tke_walk.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r101_tke_walk.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/dommsk.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/dommsk.f90"),
    "GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/domqco.f90"),
    # Round 47 must bind to the actual widened build, not its R41 source.
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/l2_r46_stage.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/l2_r46_stage.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/usrdef_sbc.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/usrdef_sbc.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynhpg.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynvor.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdf_oce.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdf_oce.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/oce.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/oce.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dom_oce.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dom_oce.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc/do_loop_substitute.h90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc/do_loop_substitute.h90"),
    # Round 64 audits the rejected record against its producer's compiled
    # source rather than assuming the additive source patches compiled as read.
    "GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/traadv_fct.f90"),
    "GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R63KRHS/EXP00/namelist_cfg": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R63KRHS/EXP00/namelist_cfg"),
    # Round 65 consumes the admitted R64 rerun and therefore binds every
    # source claim to that rerun's own compiled branch.
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90"),
    # Round 112 consumes the admitted Round-111 FCT operand record and binds
    # every statement to that record producer's compiled source card.
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domain.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domhgr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domhgr.f90"),
    "GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90"),
    # Round 118 binds its recovered record and producer walk to the exact
    # compiled Round-117 build that wrote the admitted bytes.
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90"),
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/namelist_cfg": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/namelist_cfg"),
    "GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/ocean.output": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/ocean.output"),
    # Round 122 scores the admitted full-year restart series, so its state-level
    # claims bind to the exact compiled build that wrote that series.
    "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/restart.f90"),
    "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90"),
    "GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90"),
    # Round 92 binds the operand diagnosis to the exact acquired Round-90
    # compiled card, including its write-only recorder.
    "GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/domain.f90"),
    "GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/divhor.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynhpg.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90"),
    # Round 72 cites the exact acquired R71 compiled tracer-stage branch.
    "GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90"),
    # Round 73 consumes the admitted R72 record and binds the boundary and
    # next acquisition to that producer's own compiled branches.
    "GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90"),
    # Round 76 consumes the admitted R75 operand record and binds the next
    # producer acquisition to that record's own compiled branch.
    "GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90"),
    # Round 77 audits the failed admission against the exact compiled target
    # that wrote both the inherited transport-mean record and the new record.
    "GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90"),
    # Round 78 consumes the operator-admitted R77 acquisition and binds the
    # source walk to that record producer's exact compiled branch.
    "GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90"),
    "GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90"),
    # Round 138 consumes the operator-run developed-state record and binds
    # every statement to that record producer's exact compiled branch.
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/domqco.f90"),
    # Round 140 consumes the admitted Round-139 split and therefore binds the
    # operand verdict to that exact writer's compiled program.
    "GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90"),
    # Round 141 consumes the admitted Round-140 completed three-dimensional
    # RHS record and binds its program order to that exact compiled writer.
    "GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90"),
    "round64/oracle_krhs_split/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round64/"
        "oracle_krhs_split/ocean.output"),
    # Round 58 binds every TKE claim to the record producer's own compiled
    # branch, including the writer whose dummy-bound defect invalidated it.
    "GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90"),
    "GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdfphy.f90"),
    # Round 59 binds the rejected record to the R58 producer's own compiled
    # writer, closure, allocation, and pre-EVD transfer.
    "GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/l2_r54_tke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/l2_r54_tke.f90"),
    "GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdfphy.f90"),
    "GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdf_oce.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdf_oce.f90"),
    # Round 60's accepted record was emitted by the R59 compiled target.
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90"),
    # Round 104 walks the shear production, whose operands are built in
    # domqco/domhgr and masked in dommsk; cite the SAME compiled branch that
    # produced the record rather than the src/OCE originals, whose line
    # numbers differ from the preprocessed ones.
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domhgr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domhgr.f90"),
    "GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/dommsk.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/dommsk.f90"),
    # Round 102 consumes the operator-run Round-101 statement-boundary record
    # and binds every closure statement to that exact compiled target.
    "GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo"
        "/l2_r101_tke_walk.f90"),
    "GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90"),
    "GYRE_OMIP_L2_P3_SM_R101TKEW/EXP00/namelist_cfg": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R101TKEW/EXP00/namelist_cfg"),
    # Round 105 audits the shared shear route on DINO's compiled leapfrog card.
    "DINO/BLD/ppsrc/nemo/stpmlf.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/stpmlf.f90"),
    "DINO/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/zdfphy.f90"),
    "DINO/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/trazdf.f90"),
    "DINO/BLD/ppsrc/nemo/usrdef_zgr.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/usrdef_zgr.f90"),
    "DINO/BLD/ppsrc/nemo/zgr_lib.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/zgr_lib.f90"),
    "DINO/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/domain.f90"),
    "DINO/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/domqco.f90"),
    "DINO/BLD/ppsrc/nemo/lbclnk.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/lbclnk.f90"),
    "DINO/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "DINO/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/eosbn2.f90"),
    "DINO/BLD/ppsrc/nemo/ldfslp.f90": (
        NEMO / "cfgs/DINO/BLD/ppsrc/nemo/ldfslp.f90"),
    # Round 95 consumes the operator-run Round-94 stage-closure record and
    # therefore binds the transport boundary to that record's compiled card.
    "GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo"
        "/divhor.f90"),
    "GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo"
        "/sshwzv.f90"),
    # Round 99 consumes the admitted Round-98 direct-W record and binds every
    # statement to that record producer's exact compiled branch.
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/oce.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/oce.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/mppini.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/mppini.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domain.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/traadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90"),
    # Round 100 consumes the operator-admitted Round-99 same-call ratio
    # record, so its claims bind to that exact instrumented compiled target.
    "GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo"
        "/sshwzv.f90"),
    "GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo"
        "/traadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo"
        "/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo"
        "/dynzdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R21W/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R21W/BLD/ppsrc/nemo/traadv.f90"),
    "round33_lock_zdf_matrix/namelist_cfg": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
             "round33_lock_zdf_matrix/namelist_cfg")),
    "round33_overflow_zdf_matrix/namelist_cfg": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
             "round33_overflow_zdf_matrix/namelist_cfg")),
    "nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases"
        "/nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90"),
    "nemo_testcase_l2_gyre_round54_tke_operands.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases"
        "/nemo_testcase_l2_gyre_round54_tke_operands.py"),
    "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases"
        "/nemo_testcase_l2_gyre_round46_kt2_stage_gate.py"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdfmxl.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdfmxl.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/traldf_iso.f90"),
    "GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/ocean.output": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/ocean.output"),
    "nemo_testcase_l2_gyre_phase3_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases"
             / "nemo_testcase_l2_gyre_phase3_gate.py"),
    "fidelity/nemo_recipe.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py"),
    "GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "round38_oracle_trazdf_kt2/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
        "/round38_oracle_trazdf_kt2/ocean.output"),
    "round19_oracle_v2_external/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
        "/round19_oracle_v2_external/ocean.output"),
    "GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90":
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90",
    # PR #1802 final round: the barotropic velocity NEMO subtracts from the
    # 3-D velocity, and the surface mixing-length anchor, on the same build.
    "GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90":
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90",
    "GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90":
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90",
    "trabbl.F90": _OCE / "TRA/trabbl.F90",
    "MY_SRC/stprk3.F90": LOCK / "MY_SRC/stprk3.F90",
    "domain.F90": _OCE / "DOM/domain.F90",
    "usrdef_zgr.F90": _OCE / "USR/usrdef_zgr.F90",
    "domqco.F90": _OCE / "DOM/domqco.F90",
    "usrdef_hgr.F90": LOCK / "MY_SRC/usrdef_hgr.F90",
    "namelist_cfg": LOCK / "EXP00/namelist_cfg",
    # GYRE's own deck.  Without this key a GYRE premise written as a
    # bare "namelist_cfg" token binds to LOCK's namelist, where
    # ln_dynadv_vec is .false. -- the opposite of what it claims.
    "GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg":
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg",
    "namelist_ref": NEMO / "cfgs/SHARED/namelist_ref",
    "ocean.output": OVERFLOW_RUN / "ocean.output",
    # ORCA2 phase-2 run, cited for the deck's resolved dynzdf guard.
    "variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/runs"
        "/variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output"),
    "overflow_kt1_10/ocean.output": OVERFLOW_RUN / "ocean.output",
    "lock_kt1_10/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"
        "/ocean.output"),
    # The PREPROCESSED source LOCK compiles.  OVERFLOW's is the same body with
    # its own line offsets, which is why the eligibility gate greps both cards
    # programmatically instead of citing OVERFLOW's numbers here.
    "BLD/ppsrc/nemo/dynspg_ts.f90": LOCK / "BLD/ppsrc/nemo/dynspg_ts.f90",
    "BLD/ppsrc/nemo/dynhpg.f90": LOCK / "BLD/ppsrc/nemo/dynhpg.f90",
    "BLD/ppsrc/nemo/dynadv_up3.f90": LOCK / "BLD/ppsrc/nemo/dynadv_up3.f90",
    "BLD/ppsrc/nemo/dynvor.f90": LOCK / "BLD/ppsrc/nemo/dynvor.f90",
    "ocean_pe_latlon_cgrid.py":
        REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py",
    "ocean_model_latlon_cgrid.py":
        REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
    "barotropic_latlon_cgrid.py":
        REPO / "packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py",
    "eos.py": REPO / "packages/ocean/legoesm/ocean/eos.py",
    "state.py": REPO / "packages/ocean/legoesm/ocean/state.py",
    "provenance.py": REPO / "packages/ocean/legoesm/ocean/fidelity/provenance.py",
    "nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py"),
    # --- round 35 paths: the implicit vertical TRACER solve ---
    # GYRE's OWN run log for this round, so a GYRE resolved value cannot bind
    # to OVERFLOW's ocean.output through the bare key above.
    "round29_oracle_v2_zdf_matrix/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
        "/round29_oracle_v2_zdf_matrix/ocean.output"),
    "trazdf.F90": _OCE / "TRA/trazdf.F90",
    # --- round 38 paths: the isoneutral fold's producers ---
    "traldf_iso.F90": _OCE / "TRA/traldf_iso.F90",
    "dino.py": REPO / "packages/ocean/legoesm/ocean/experiments/dino.py",
    "nemo_testcase_l2_gyre_stage3_completion_gate.py":
        REPO / "scripts/validate/ocean_fidelity/testcases"
             / "nemo_testcase_l2_gyre_stage3_completion_gate.py",
    "ldftra.F90": _OCE / "LDF/ldftra.F90",
    # The PREPROCESSED body GYRE compiles: which branch runs, and how the
    # thickness macros expanded, are only visible here.
    "ppsrc/nemo/trazdf.f90":
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R29ZDF/BLD/ppsrc/nemo/trazdf.f90",
    # The two GYRE decks' compile keys.  GYRE_BARE is what the native demo
    # card reproduces and it is key_linssh; the oracle deck is key_qco, and a
    # bare "cpp" token could otherwise bind to either.
    "cpp_GYRE_BARE.fcm": NEMO / "cfgs/GYRE_BARE/cpp_GYRE_BARE.fcm",
    "cpp_GYRE_OMIP_L2_P3_SM.fcm":
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/cpp_GYRE_OMIP_L2_P3_SM.fcm",
    "nemo_testcase_recipe.py":
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py",
    "packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py",
    "tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py":
        REPO / "tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py",
    # Round 124 scores the exact instrumented translation units that produced
    # the admitted Round-123 process record, not an uninstrumented analogue.
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/dynspg_ts.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/traadv_fct.f90"),
    "GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo"
        "/traadv.f90"),
    "round123/oracle_process_budget/ocean.output": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round123"
             "/oracle_process_budget/ocean.output")),
    # Round 126 scores the compiled Round-125 acquisition branch itself.
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/eosbn2.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/zdfphy.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo"
        "/zdfevd.f90"),
    "GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg"),
    # Round 173 diagnoses the exact compiled reader that failed in the
    # operator's Round-172 directed acquisition.
    "GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo"
        "/trazdf.f90"),
    # Round 149 reads the exact preprocessed build that produced the admitted
    # Round-148 developed lateral-diffusion record.
    "GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo"
        "/dynldf_lev.f90"),
    "GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo"
        "/domqco.f90"),
    "GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo"
        "/dynvor.f90"),
    "GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo"
        "/traadv_fct.f90"),
    # Round 178 cites the acquired Round-177 build itself: the writer, active
    # stage call, and slope producer whose first returned output is non-bit.
    "GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo"
        "/traldf.f90"),
    "GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo"
        "/traldf_iso.f90"),
    "GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo"
        "/ldfslp.f90"),
    # Round 180 reads the exact passive build that produced its admitted
    # developed native-slope causal stream.
    "GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo"
        "/ldfslp.f90"),
    "GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo"
        "/zdfphy.f90"),
    "GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo"
        "/zdfmxl.f90"),
    "GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo"
        "/eosbn2.f90"),
    "GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/domzgr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo"
        "/domzgr.f90"),
    # Round 154 reads the exact instrumented build that produced the admitted
    # developed FCT record; its added observation calls shift FCT line numbers.
    "GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo"
        "/traadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo"
        "/traadv_fct.f90"),
    # Round 155 reads the exact compiled branch that produced the admitted
    # developed stage-3 transport operand record.
    "GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    # Round 157 reads the build that produced the admitted developed stage-2
    # record, whose line numbers are NOT R154TRPWALK's.
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    # Round 158 walks the two halves the vector-invariant dyn_adv calls, on
    # that same build.
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90"),
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90"),
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90"),
    # Round 159 walks the PRODUCER of the vertical velocity dynzad reads,
    # and the second continuity solve the tracer transport runs, on that same
    # build.
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90"),
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90"),
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90"),
    # Round 161 walks the free-surface ratio's OWN producer, on that same
    # build, because the ratio the velocity indicator reads is set by the
    # stage program rather than rebuilt from the stage sea surface height.
    "GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90"),
    # --- round 36 paths: THIS ROUND'S OWN BUILD ---
    # The R35TRAZDF card is the one that produced the round-35 record, and it
    # is the only build whose trazdf.f90 carries the instrument, so its line
    # numbers are not R29ZDF's.  Bare lowercase keys, distinct from the
    # shipped .F90 keys above: FILES is an EXACT-key lookup, so nothing here
    # can capture a citation that already resolves elsewhere.
    # --- ORCA2 round 13: the river runoff's water, and the emp it is not in
    # These are the ORCA2 record's OWN compiled branch, so a runoff citation
    # cannot bind to GYRE's ppsrc through a bare basename.
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/divhor.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcrnf.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcrnf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/trasbc.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dynadv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynkeg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dynkeg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dynzad.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynadv_round62_writer.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo"
        "/dynadv_round62_writer.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo"
        "/dom_oce.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo"
        "/dynadv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo"
        "/dynzad.f90"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round62_vector_advection_acquisition/"
    "dynadv_round62_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round62_vector_advection_acquisition/"
        "dynadv_round62_writer.F90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sshwzv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/traadv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/traadv_cen.f90"),
    # --- ORCA2 round 14: the barotropic slow forcing and its solver ---
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dynspg_ts.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/domain.f90"),
    "trazdf.f90": _R35 / "trazdf.f90",
    "zdf_oce.f90": _R35 / "zdf_oce.f90",
    "stprk3_stg.f90": _R35 / "stprk3_stg.f90",
    "domain.f90": _R35 / "domain.f90",
    # --- restored from the lane tip (980cc6369) after the merge conflict on
    # this file was resolved to HEAD's side, which silently dropped every
    # entry the VORTEX-card rounds had added.  Paths only; none of these
    # NEMO/VORTEX sources shifted in the merge. ---
    "VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "tests/VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "tests/VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90"),
    "VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "tests/VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90"),
    "VORTEX_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90": (
        NEMO / "tests/VORTEX_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90"),
    "VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "tests/VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90"),
    # --- The vector-EEN VORTEX card's own build (decision 73, round 3). ---
    "VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90"),
    "VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90"),
    # --- round 4: the record's OWN instrumented build, whose stp_2D is what
    # the kt=2 walk attributes the vector card's second step to. ---
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90"),
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90"),
    # Round 196: the split-explicit barotropic solve, read from the SAME
    # uninstrumented-dynspg build the rounds 192-195 citations use (that
    # build instruments stprk3_stg and dynadv only, so its dynspg_ts is the
    # shipped one).
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90"),
    # Round 197: dyn_vor_init builds e3f_0vor, the array dyn_cor_2D_init
    # divides ff_f by; same uninstrumented-dynamics build as the solve above.
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynzad.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynzad.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/divhor.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/divhor.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90"),
    "VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "tests/VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traadv.f90"),
    # --- round 6: the after-SSH slot the RK3 program leaves behind, in the
    # vector card's own build and in GYRE's (the statement is shared). ---
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90"),
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90"),
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90"),
    "GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/sshwzv.f90"),
    "vortex_round3/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round3/"
        "namelist_cfg"),
    "eosbn2.F90": _OCE / "TRA/eosbn2.F90",
    "restart.F90": _OCE / "IOM/restart.F90",
    "tests/VORTEX/MY_SRC/usrdef_hgr.F90": NEMO / "tests/VORTEX/MY_SRC/usrdef_hgr.F90",
    "tests/VORTEX/MY_SRC/usrdef_istate.F90": NEMO / "tests/VORTEX/MY_SRC/usrdef_istate.F90",
    "tests/VORTEX/MY_SRC/usrdef_nam.F90": NEMO / "tests/VORTEX/MY_SRC/usrdef_nam.F90",
    "tests/VORTEX/MY_SRC/usrdef_sbc.F90": NEMO / "tests/VORTEX/MY_SRC/usrdef_sbc.F90",
    "tests/VORTEX/MY_SRC/usrdef_zgr.F90": NEMO / "tests/VORTEX/MY_SRC/usrdef_zgr.F90",
    "vortex_round2/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round2/namelist_cfg"),
    # --- gating the DINO month-regression and PR-1802 review-fix receipts:
    # the DINO build's own compiled branch (bare and cfgs/-prefixed keys both
    # cited by those receipts), the shared LBC template, and the bridge. ---
    "cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90",
    "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90",
    "nemogcm.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90",
    "dynspg_ts.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90",
    "lbclnk.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/lbclnk.f90",
    "src/OCE/DOM/domqco.F90": _OCE / "DOM/domqco.F90",
    "lbc_lnk_pt2pt_generic.h90": _OCE / "LBC/lbc_lnk_pt2pt_generic.h90",
    "nemo_state_bridge.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py"),
    "zdftke.F90": _OCE / "ZDF/zdftke.F90",
    "domhgr.F90": _OCE / "DOM/domhgr.F90",
    "geo2ocean.F90": _OCE / "SBC/geo2ocean.F90",
    "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg": (
        NEMO / "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg"),
}

# citation -> the anchors that IDENTIFY its first and last line, plus the
# range LENGTH.  Three forms:
#
#   "symbol"                     a symbol unique in the file; both endpoints
#   [first, last, extent]        one anchor per endpoint, plus the line count
#   ("symbol", nth)              an anchor pinned to the nth occurrence, for a
#                                symbol that recurs (every terminal token does)
#
# Round 28 measured that 31 of 86 entries were anchored on a symbol occurring
# 2 to 59 times in its file, and that a recurring terminal token at a range's
# END is how a widened range slipped through: stprk3_stg.F90:309-334 still
# passed as :309-533 because line 334 and line 533 are both ENDIF.  A bare
# recurring symbol is refused now, and every multi-line citation states its
# length a SECOND time so widening the key without widening the extent fails.
CITATION_MAP = {
    # --- ORCA2 round 127: northern V recurrence prefix -------------------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348': [
        ('CALL r118_een_v_before(1, ji, jj, jk, zpvo_nw', 1),
        ('CALL r118_een_v_after(2, ji, jj, jk, ffv_ne(ji,jj))', 1), 10],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1360-1367': [
        ('r105_acc_v_nw(ji,jj) = ffv_nw(ji,jj)', 1),
        ('ffv_ne(ji,jj) = r1_12 * r1_e2v(ji,jj)', 1), 8],
    # --- ORCA2 round 126: complete northern frozen-mask association ------
    'barotropic_latlon_cgrid.py:1044-1050': [
        ('# dommsk.f90:232-258 applies the ordinary F-grid lateral boundary before', 1),
        ('fmask = fmask.at[-1].set(_nemo_een_north_f(fmask, grid)[-1])', 1), 7],
    # --- ORCA2 round 125: northeast northern frozen-mask association ------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258': [
        ("CALL lbc_lnk( 'dommsk', umask, 'U'", 1),
        ('fe3mask(:,:,:) = fmask(:,:,:)', 1), 27],
    # --- ORCA2 round 123: northeast northern e3f_0vor association ---------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937': [
        ('SELECT CASE( nn_e3f_typ )', 1),
        ('WHERE( e3f_0vor(:,:,:) == 0._wp )', 1), 26],
    # --- ORCA2 round 122: northern F-grid copy-fill association -----------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747': [
        ("CASE ( 'F' )", 5),
        ('END DO; END DO', 6), 26],
    'barotropic_latlon_cgrid.py:924-945': [
        ('def _nemo_een_north_f(field, grid):', 1),
        ('return north', 1), 22],
    'barotropic_latlon_cgrid.py:1082-1085': [
        ('q = b(ff[..., None] / e3f)', 1),
        ('q_north = b(ff_north[..., None] / e3f_north)', 1), 4],
    # --- ORCA2 round 121: northern V fraction walk -------------------------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328': [
        ('zpvo_ne = ff_f(ji  ,jj+1)', 1),
        ('& ff_f(ji-1,jj+1) /', 1), 6],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domhgr.f90:233-237': [
        ("IF(  iom_varid( inum, 'ff_f'", 1),
        ("CALL iom_get( inum, jpdom_global, 'ff_t'", 1), 5],
    # --- ORCA2 round 120: northern V three-fraction acquisition -----------
    'ORCA2_OMIP_L4_R118EENVREC/BLD/ppsrc/nemo/dynspg_ts.f90:1320-1325': [
        ('zpvo_ne = ff_f(ji  ,jj+1)', 1),
        ('& ff_f(ji-1,jj+1) /', 1), 6],
    # --- ORCA2 round 119: V recurrence source-ordered walk -----------------
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1317-1327': [
        ('zpvo_ne = ff_f(ji  ,jj+1)', 1),
        ('ffv_se(ji,jj) = ffv_se(ji,jj) +', 1), 11],
    # --- ORCA2 round 118: all four V EEN recurrences -----------------------
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1324-1327': [
        ('ffv_nw(ji,jj) = ffv_nw(ji,jj) +', 1),
        ('ffv_se(ji,jj) = ffv_se(ji,jj) +', 1), 4],
    # --- ORCA2 round 117: remaining U recurrence and south-mask owner -----
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1274-1288': [
        ('CALL r116_een_u_before(1, ji, jj, jk, zpvo_ne', 1),
        ('CALL r116_een_u_after(3, ji, jj, jk, ffu_se(ji,jj))', 1), 15],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dommsk.f90:211-232': [
        ('vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)', 1),
        ("CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 1), 22],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1816-1820': [
        ('zland = 0._wp', 3),
        ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), 5],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872': [
        ('! define ifill: which method should be used to fill each parts (sides+corners) of the halos', 3),
        ('ENDIF', 76), 9],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:2130-2136': [
        ('IF(     ifill(jn,jf) == jpfillcst ) THEN', 5),
        ('END DO   ;   END DO   ;   END DO   ;   END DO', 19), 7],
    # --- ORCA2 round 115: EEN stored product and recurrence signs ----------
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1238-1245': [
        ('ffu_nw(:,:) = 0._wp', 1),
        ('DO jk = 1, mbku(ji,jj)', 1), 8],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1263-1273': [
        ('r107_zpvo_nw(ji,jj,jk) = zpvo_nw', 1),
        ('ffu_se(ji,jj) = ffu_se(ji,jj) +', 1), 11],
    # --- ORCA2 round 114: southern frozen-mask association ----------------
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258': [
        ("CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 1),
        ('fe3mask(:,:,:) = fmask(:,:,:)', 1), 27],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1999-2004': [
        ('IF(     ifill(jn,jf) == jpfillcst ) THEN', 4),
        ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = zland', 4), 6],
    'barotropic_latlon_cgrid.py:919-921': [
        ('def _nemo_south_zero_fill(field):', 1),
        ('return jnp.concatenate([jnp.zeros_like(field[:1]), field[:-1]], axis=0)', 1), 3],
    # --- ORCA2 round 113: southern frozen vorticity thickness association ---
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937': [
        ('SELECT CASE( nn_e3f_typ )', 1),
        ('WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 1), 26],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1811-1820': [
        ('! take care of optional parameters', 3),
        ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), 10],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872': [
        ('! define ifill: which method should be used to fill each parts (sides+corners) of the halos', 3),
        ('ENDIF', 76), 9],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domzgr.f90:179-188': [
        ('ELSEIF( lk_vco_1d3d ) THEN', 1),
        ("CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1), 10],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:2021-2046': [
        ('ELSE                                                        ! southern/northern side', 4),
        ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = ptab(jf)%pt4d(ii2,ij2,jk,jl)', 4), 26],
    'barotropic_latlon_cgrid.py:914-916': [
        ('def _nemo_een_south_e3f0(e3f0, mesh_e3f0):', 1),
        ('return jnp.concatenate([mesh_e3f0[:1], e3f0[:-1]], axis=0)', 1), 3],
    'barotropic_latlon_cgrid.py:1104-1118': [
        ('# dyn_vor_init applies the default zero fill to e3f_0vor', 1),
        ('q_south = b(ff_south[..., None] / e3f_south)', 1), 15],
    # --- ORCA2 round 112: executed ff_f read/fill and literal EEN consumer ---
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:101-108': [
        ('IF( ln_read_cfg ) THEN', 1),
        ('&              iff   , ff_f  , ff_t', 1), 8],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:233-236': [
        ("IF(  iom_varid( inum, 'ff_f'", 1),
        ("CALL iom_get( inum, jpdom_global, 'ff_f'", 1), 4],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1199-1224': [
        ('ELSE                                                        ! southern/northern side', 1),
        ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = ptab(jf)%pt4d(ii2,ij2,jk,jl)', 1), 26],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1244-1248': [
        ('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 1),
        ('& ff_f(ji  ,jj-1) /', 1), 5],
    'barotropic_latlon_cgrid.py:909-911': [
        ('def _nemo_south_copy_fill(field):', 1),
        ('return jnp.concatenate([field[:1], field[:-1]], axis=0)', 1), 3],
    'barotropic_latlon_cgrid.py:1100-1103': [
        ('# ORCA2 reads ff_f through iom_get(..., kfill=jpfillcopy)', 1),
        ('ff_south = _nemo_south_copy_fill(ff)', 1), 4],
    # --- ORCA2 round 111: EEN frozen mask and next southern-halo owner ---
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1248': [
        ('SELECT CASE( nvor_scheme )', 1),
        ('& ff_f(ji  ,jj-1) /', 1), 8],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:252-258': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 2),
        ('fe3mask(:,:,:) = fmask(:,:,:)', 1), 7],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:135-143': [
        ('IF( iff == 0 ) THEN', 1),
        ("CALL lbc_lnk( 'dom_hgr', ff_t", 1), 9],
    # --- ORCA2 round 109: admitted EEN per-level recurrence ---
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1245': [
        ('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 1),
        ('& ff_f(ji  ,jj-1) /', 1), 5],
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1256-1263': [
        ('r107_zpvo_nw(ji,jj,jk) = zpvo_nw', 1),
        ('r107_acc_after(ji,jj,jk) = ffu_nw(ji,jj)', 1), 8],
    # --- ORCA2 round 108: inherited recorder initialization chain ---
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1103-1104': [
        ('CALL r105_een_accum_init', 1),
        ('CALL r107_een_step_init', 1), 2],
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/l4_r105_een_accum.f90:34-45': [
        ("GET_ENVIRONMENT_VARIABLE('ORCA2_R105_EEN_ACCUM_DIR'", 1),
        ("ORCA2_R105_EEN_ACCUM_INIT", 1), 12],
    # --- ORCA2 round 107: literal EEN vertical loop bounds ---
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1229-1230': [
        ('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 1),
        ('DO jk = 1, mbku(ji,jj)', 1), 2],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1262-1263': [
        ('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 2),
        ('DO jk = 1, mbkv(ji,jj)', 1), 2],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dommsk.f90:223-230': [
        'IF (jpk>2) THEN',
        'IF ( MAXVAL(fmask(ji,jj,:))/=0._wp )', 8],
    # --- ORCA2 round 106: admitted live EEN accumulator/scale producer ---
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1231-1244': [
        'zpvo_nw = ff_f(ji-1,jj  )',
        ('ffu_nw(ji,jj) = ffu_nw(ji,jj) +', 1), 14],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1277-1278': [
        ('ffv_nw(ji,jj) = ffv_nw(ji,jj) +', 1),
        ('ffv_ne(ji,jj) = ffv_ne(ji,jj) +', 1), 2],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1288-1289': [
        'ffv_nw(ji,jj) = r1_12 * r1_e2v',
        'ffv_ne(ji,jj) = r1_12 * r1_e2v', 2],
    # --- ORCA2 round 105: failed recorder and compiled call frequency ---
    'ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg.f90:300-303': [
        'SELECT CASE( nspg )',
        'CASE ( np_NO )', 4],
    'ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:302':
        'IF( kt == nit000 .OR. .NOT. lk_linssh )   CALL dyn_cor_2D_init( Kmm )',
    'ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1294-1297': [
        'CALL r104_een_accum_dump(r104_acc_u_nw',
        '& r104_scl_v_sw, r104_scl_v_se, r104_scl_v_nw, r104_scl_v_ne)', 4],
    'l4_r104_een_accum.F90:23-27': [
        "IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop",
        "IF(ios /= 0) CALL ctl_stop('round104: cannot open EEN operand record')", 5],
    # --- ORCA2 round 99: admitted frozen EEN coefficient discriminator ---
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:126-127': [
        'ALLOCATE( ffu_nw(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))',
        '&      ffv_nw(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))', 2],
    # --- ORCA2 round 97: admitted rung-0 split-explicit statement walk ---
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-666': [
        '!     Compute Sea Level at step jit+1',
        "CALL r95_spg_w2('cor_v', zv_trd)", 87],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265': [
        'CASE( np_EEN )',
        'ffv_se(ji,jj) = r1_12 * r1_e2v(ji,jj)', 53],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domzgr.f90:625,631': [
        'mbku(:,:) = MAX( NINT( zk(:,:) ), 1 )',
        'mbkv(:,:) = MAX( NINT( zk(:,:) ), 1 )', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dommsk.f90:224-225': [
        'IF (mbku(ji,jj)==1) umask(ji,jj,:) = 0._wp',
        'IF (mbkv(ji,jj)==1) vmask(ji,jj,:) = 0._wp', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1369-1392': [
        'SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )',
        ('END SUBROUTINE dyn_cor_2D', 2), 24],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynvor.f90:912-937': [
        'SELECT CASE( nn_e3f_typ )',
        'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 26],
    # --- ORCA2 round 95: independent rung-0 split-explicit substeps ---
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:301-316': [
        '! Lane-2 GYRE ENE operand instrument: write the eight frozen coefficient',
        ('CLOSE(l2_unit)', 1), 16],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:390-442': [
        '! Lane-2 GYRE causal instrument: open one write-only stream for the',
        'WRITE(l2_ord_unit) l2_magic, 2, kt, 2, jpi, jpj, STORAGE_SIZE(1._wp), rDt_e', 53],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:287-320': [
        'ssh_frc(:,:) = sshe_rhs(:,:)',
        'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', 34],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:339-381': [
        '! Initialize barotropic variables:',
        'vn_adv(:,:)     = 0._wp', 43],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:446': (
        'DO jn = 1, icycle', 1),
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:460-519': [
        '!* Set extrapolation coefficients for predictor step:',
        ('+ e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1), 60],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:530-557': [
        '! resulting flux at mid-step (not over the full domain)',
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 28],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:601-615': [
        '! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2',
        'zv_spg(ji,jj) = - zldg * ( zsshp2_e(ji,jj+1) - zsshp2_e(ji,jj) ) * r1_e2v(ji,jj)', 15],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:618-652': [
        '! Add Coriolis trend:',
        ("& l4_canon_2d(zu_trd,'U'), l4_canon_2d(zv_trd,'V')", 1), 35],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:655-708': [
        '! Set next velocities:',
        'va_e(ji,jj) =  va_e(ji,jj) / ( 1._wp - rDt_e * zCdU_v(ji,jj) * hvr_e(ji,jj) )', 54],
    # --- ORCA2 round 94: independent rung-0 slow/external boundary walk ---
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219': [
        '!*  vertical averaging  *!',
        "CALL r93_slow_put_pair( kt, 'depth', Ue_rhs, Ve_rhs )", 14],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:231-250': [
        '!* baroclinic drag forcing *!   (also provide the barotropic drag coeff.)',
        'CLOSE(l2_slow_unit)', 20],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:285': (
        "CALL r93_slow_put_pair( kt, 'final', Ue_rhs, Ve_rhs )", 1),
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:290-312': [
        ('!                 !=======================================!', 1),
        "IF( kt == nit000 ) CALL r93_slow_put2( 'ssh_rhs', sshe_rhs )", 23],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:317-326': [
        '!    using a split-explicit time integration in forward mode',
        'CALL r93_slow_finish( kt )', 10],
    # --- ORCA2 round 93: independent rung-0 stage-1 RHS walk ---
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175': [
        '!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST',
        'CALL r92_rhs_finish( kt )', 35],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215': [
        '!*  vertical averaging  *!',
        ('END SELECT', 2), 13],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:227-230': [
        '!* baroclinic drag forcing *!   (also provide the barotropic drag coeff.)',
        "l4_canon_2d(Ve_rhs,'V'), l4_canon_2d(CdU_u,'U'), l4_canon_2d(CdU_v,'V')", 4],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:232-245': [
        '!* wind forcing *!',
        ('ENDIF', 7), 14],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:247-277': [
        '!* atmospheric pressure forcing *!',
        ('ENDIF', 10), 31],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:283-300': [
        '!==   2D sea surface height forcing   ==!',
        ('ENDIF', 11), 18],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:305-315': [
        'Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)',
        'DEALLOCATE( sshe_rhs , Ue_rhs , Ve_rhs , CdU_u , CdU_v )', 11],
    # --- ORCA2 round 92: rung-0 card and source-order RHS acquisition ---
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:148-217': [
        '! Update external forcing (tides, open boundaries, ice shelf interaction and surface boundary condition (including sea-ice)',
        'CALL r84_dump_frame( kstp, 1, Naa )', 70],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:139-166': [
        '!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST',
        'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 28],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-228': [
        '!==  Background eddy viscosity and diffusivity  ==!',
        'avm_k(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk) =                avmb(jk) * wmask(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk)', 24],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90:262-284': [
        '!==  type of vertical turbulent closure  ==!',
        'IF( ln_zdfswm )   CALL zdf_swm_init       ! surface  wave-driven mixing', 23],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfdrg.f90:540-547': [
        'CASE( np_lin )             !==  linear friction  ==!   (pCdU = Cd0 * Uc0)',
        'CALL zdf_drg_lin( pCd0(:,:), pCdU(:,:) )  !  using a constant velocity', 8],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/istate.f90:143-162': [
        ('IF( .NOT. ln_rstart ) THEN', 1),
        'vv_b(:,:,Kmm)   = vv_b(:,:,Kbb)', 20],
    # --- ORCA2 round 91: admitted rung-0 entry/stage frames ---
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:92-108': [
        '! Lane-1 certified oracle: exact step-entry Nbb state.  This is a',
        'CALL r84_dump_frame( kstp, 0, Nbb )', 17],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:214-217': [
        '! Stage 1 :',
        'CALL r84_dump_frame( kstp, 1, Naa )', 4],
    # --- ORCA2 round 90: source-resolved OFF-runoff recorder fault ---
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/traadv.f90:287':
        "&       l4_canon_2d(rnf,'T')",
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/traadv.f90:357':
        'IF( llwet ) zfield(ji,jj) = pfield(ji,jj)',
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/sbc_oce.f90:201':
        'IF(ln_rnf)   ALLOCATE( rnf(jpi,jpj), rnf_b(jpi,jpj), STAT=ierr(3) )',
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/stprk3_stg.f90:555':
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
    # --- ORCA2 round 88: exact optimized branch and unresolved crash ---
    'ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/stprk3.f90:90-108': [
        "IF( ln_timing )   CALL timing_start('stp_RK3')",
        'CALL r84_dump_frame( kstp, 0, Nbb )', 19],
    'ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/traadv.f90:273-287': [
        ('IF( lwp .AND. kt == kit000 .AND. kstg == 1 ) THEN', 1),
        "&       l4_canon_2d(rnf,'T')", 15],
    'ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/sbc_oce.f90:195-216': [
        'ALLOCATE( utau(jpi,jpj) , utau_b(jpi,jpj) , utauU(jpi,jpj) , &',
        'wndm(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , taum (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , STAT=ierr(6) )', 22],
    # --- ORCA2 round 87: symbolized crashing debug branch ---
    'ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:149-154': [
        'IF( ln_tide    )   CALL tide_update( kstp )',
        'IF( kstp == nit000 )   CALL l4_dump_ocean_surface_input( kstp, Nbb )', 6],
    'ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:424-438': [
        "WRITE(iunit) l4_canon_2d(qsr,'T'), l4_canon_2d(qns,'T'), l4_canon_2d(qns_b,'T'), &",
        "WRITE(iunit) l4_canon_3d(rnf_tsc,'T'), l4_canon_3d(rnf_tsc_b,'T')", 15],
    'ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:443-473': [
        'FUNCTION l4_canon_2d( pfield, cdgrid ) RESULT( zfield )',
        'END FUNCTION l4_canon_2d', 31],
    # --- ORCA2 round 87: rung-0 disabled-owner surface fields ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_oce.f90:195-216': [
        'ALLOCATE( utau(jpi,jpj) , utau_b(jpi,jpj) , utauU(jpi,jpj) , &',
        'wndm(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , taum (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , STAT=ierr(6) )', 22],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:146-152': [
        'INTEGER FUNCTION sbc_rnf_alloc()',
        'rnf_tsc_b(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpts) , rnf_tsc (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpts) , STAT=sbc_rnf_alloc )', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbini.f90:133-140': [
        'CALL icb_nam               ! Read and print namelist parameters',
        "IF( icb_alloc() /= 0 )   CALL ctl_stop( 'STOP', 'icb_alloc : unable to allocate arrays' )", 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icb_oce.f90:161-174': [
        'INTEGER FUNCTION icb_alloc()',
        '&      berg_grid%tmp        (jpi,jpj) , STAT=ill)', 14],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:274-280': [
        'IF( nn_ice == 0 ) THEN        !* No sea-ice in the domain : ice fraction is always zero',
        'cloud_fra(:,:) = pp_cldf      !* cloud fraction over sea ice (used in si3)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:376-383': [
        ('IF( nn_ice == 0 ) THEN', 2),
        ('ENDIF', 21), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_ice.f90:91-121': [
        'INTEGER FUNCTION sbc_ice_alloc()',
        '&      emp_ice (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))     , sstfrz   (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))     , STAT= ierr(ii) )', 31],
    # --- ORCA2 round 83: rung-0 explicit restart-list repair ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-346': [
        'IF( ln_rst_list .OR. nn_stock /= -1 ) THEN',
        'IF( .NOT. ln_rst_list .AND. MOD( nn_stock, nn_fsbc) /= 0 ) THEN', 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119': [
        'IF( kt == nit000 ) THEN   ! default definitions',
        'IF( kt == nitrst - 1 .OR. nn_stock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN', 26],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202': [
        ('IF( kt == nitrst ) THEN', 1),
        ('ENDIF', 14), 15],
    # --- ORCA2 round 82: hierarchy rung-0 deck semantics ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:299-306': [
        'IF( ln_usr          ) THEN   ;   nsbc = jp_usr',
        "sbc_init : choose ONE and only ONE sbc option", 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:440-448': [
        ('SELECT CASE( nsbc )', 2),
        'IF( ll_opa    )       CALL sbc_cpl_rcv', 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:499-503': [
        ('SELECT CASE( nn_ice )', 2),
        ('END SELECT', 5), 5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcflx.f90:191-204': [
        'CALL fld_read( kt, nn_fsbc, sf )',
        'emp (ji,jj) =   sf(jp_emp )%fnow(ji,jj,1)', 14],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:206-228': [
        'IF( nn_avb == 0 ) THEN',
        'avm_k(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk)', 23],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:264-270': [
        'IF( ln_zdfcst ) THEN',
        'one and only one vertical diffusion option has to be defined', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:334-343': [
        'SELECT CASE ( nzdf_phy )',
        'END SELECT', 10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:738-740': [
        'IF( ln_trabbc  )   CALL tra_bbc',
        'IF( ln_tradmp  )   CALL tra_dmp', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-260': [
        'IF( ln_ldfeiv .AND. .NOT. ln_traldf_triad ) THEN',
        'IF( ln_mle    )   THEN', 6],
    # --- ORCA2 round 81: the recorded build's IWM addition and backgrounds ---
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:313-316': [
        '!* update momentum & tracer diffusivity with wave-driven mixing',
        'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + zav_wave(ji,jj)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:438-443': [
        '! This internal-wave-driven mixing parameterization elevates avt and avm in the interior, and',
        'avtb_2d(:,:) = 1._wp        ! uniform', 6],
    # --- ORCA2 round 80: admitted end-of-chain coefficients and consumers ---
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381': [
        '!                          !==  ocean Kz  ==!   (avt, avs, avm)',
        'CALL r79_write( kt, Kbb, Kmm )', 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:178-215': [
        '! vertical mixing coef.: avt for temperature, avs for salinity and passive tracers',
        ('ENDIF', 10), 38],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:191-205': [
        ('DO jk =    2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)            ! inner values', 1),
        ('zwd(ji,1) = 1._wp - zzws', 1), 15],
    # --- ORCA2 round 61: which card selects which momentum advection ---
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:346': 'ln_dynadv_vec = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:352': 'ln_dynvor_een = .true.',
    'OVERFLOW_OMIP_L1/EXP00/namelist_cfg:83': 'ln_dynadv_up3 = .true.',
    'OVERFLOW_OMIP_L1/EXP00/namelist_cfg:89': 'ln_dynvor_ens = .true.',
    'overflow_kt1_10/namelist_cfg:86': 'ln_dynadv_up3 = .true.',
    'overflow_kt1_10/namelist_cfg:92': 'ln_dynvor_ens = .true.',
    'nemo_testcase_recipe.py:2303':
        'if (cfg.momentum_advection != "vector_invariant"',
    'ocean_pe_latlon_cgrid.py:5360': ('if _mom_adv == "flux_form":', 2),
    'ocean_pe_latlon_cgrid.py:5378': (
        '_bc_horizontal_momentum_advection_flux_form(', 2),
    'nemo_testcase_recipe.py:2306':
        'requires ln_dynadv_vec=.true. with nn_dynkeg=0',
    # --- ORCA2 round 57: acquired OVERFLOW UP3 source-order walk ---
    'OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1),
        'CALL r56_up3_curv(ji,jj,jk,zlu_uu(ji,jj),zlv_vv(ji,jj),zlu_uv(ji,jj),zlv_vu(ji,jj))',
        10],
    'OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:182-192': [
        'DO jj = ntsj-( 1), ntej+(  0 ) ; DO ji = ntsi-( 1), ntei+(  0)',
        'CALL r56_up3_select_t(ji,jj,jk,zui,zvj,zl_u,zl_v)',
        11],
    'OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:194-195': [
        'zFu_t(ji+1,jj  ) = (  zFu(ji,jj) + zFu(ji+1,jj  )  ) * ( zui - gamma1 * zl_u )',
        'zFv_t(ji  ,jj+1) = (  zFv(ji,jj) + zFv(ji  ,jj+1)  ) * ( zvj - gamma1 * zl_v )',
        2],
    # --- ORCA2 round 56: admitted OVERFLOW stage-2 UP3 boundary ---
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv.f90:134-144': [
        ('SELECT CASE( n_dynadv )', 1),
        'CALL dyn_adv_up3     ( kt       , Kbb, Kmm, puu, pvv, Krhs, pau, pav, paw )',
        11],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:150-158': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1),
        '&             - ( pvv(ji  ,jj  ,jk,Kbb) - pvv(ji-1,jj  ,jk,Kbb) )    * fmask(ji-1,jj  ,jk)',
        9],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:174-219': [
        'DO jj = ntsj-( 1), ntej+(  0 ) ; DO ji = ntsi-( 1), ntei+(  0)',
        '&                                    / (e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))',
        46],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:278-359': [
        'DO jk = 1, jpk-2                    != divergence of advective fluxes =!',
        'pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - zFv_t(ji,jj) * r1_e1e2v(ji,jj) / (e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))',
        82],
    # --- ORCA2 round 52: admitted OVERFLOW stage-2 vorticity replay ---
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:343-358': [
        '!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST',
        ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2),
        16],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3.f90:200-207': [
        '! Stage 1 :',
        'CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )', 8],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:242-248': [
        ('CASE( np_ENS )                        !* enstrophy conserving scheme',
         1),
        ('CALL vor_ens( kt, Kmm, ntot, usd, vsd,', 1), 7],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:866-869': [
        'CASE( np_FLX_c2 , np_FLX_up3 )',
        'ntot = np_CME', 4],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:666': (
        'pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav *', 1),
    # --- ORCA2 round 51: admitted OVERFLOW pair record's executing build ---
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:327-363': [
        'CALL r50_mom_begin( kstp, kstg, Kbb, Kmm, Krhs, Kaa, ts, ssh, uu, vv )',
        "CALL r50_mom_uv( 'after_adv', uu, vv, Krhs )", 37],
    # --- ORCA2 round 60: controlled OVERFLOW walk after stage-2 ADV ---
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:386-403': [
        ('SELECT CASE( kstg )', 3),
        '&             /           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)',
        18],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:412-433': [
        ('CASE ( 3 )        !==  Stage 3  ==!', 1),
        "CALL r50_mom_uv( 'raw_kaa', uu, vv, Kaa )", 22],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:436-448': [
        '!==  All stages: correct the barotropic component ==!',
        'CALL r50_mom_finish( uu, vv, Kaa )', 13],
    # --- ORCA2 round 62: compiled dyn_zdf internal source order ---
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:139-151': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN   ! applied on velocity',
        '&              /          ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)',
        13],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:159-162': [
        'IF( ln_drgimp .AND. ln_dynspg_ts ) THEN',
        'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji,jj,jk)',
        4],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:164-170': [
        'DO ji = ntsi-( 0), ntei+( 0 )      ! Add bottom/top stress due to barotropic component only',
        ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1),
        7],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:188-350': [
        ('!**  tridiagonal matrix construction  **!    diagonal (zwd), lower (zwi), upper (zws)', 1),
        'puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - zws(ji,jk) * puu(ji,jj,jk+1,Kaa) ) / zwd(ji,jk)',
        163],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:357-518': [
        ('!**  tridiagonal matrix construction  **!    diagonal (zwd), lower (zwi), upper (zws)', 2),
        'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - zws(ji,jk) * pvv(ji,jj,jk+1,Kaa) ) / zwd(ji,jk)',
        162],
    # --- ORCA2 round 63: executing acquisition build, including its writer ---
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:140-159': [
        '!              !==  RHS : time-stepping of all trends but the implicit one  ==!',
        ('& pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 1), 20],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:165-171': [
        'IF( ln_drgimp .AND. ln_dynspg_ts ) THEN',
        ('& pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 2), 7],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:172-192': [
        'DO ji = ntsi-( 0), ntei+( 0 )      ! Add bottom/top stress due to barotropic component only',
        ('& pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 3), 21],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:344-362': [
        ('zwd(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwd(ji,jk-1)', 1),
        'CALL r62_zdf_u_solve(jj,puu(ntsi:ntei,jj,1:jpkm1,Kaa))', 19],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:513-531': [
        ('zwd(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwd(ji,jk-1)', 2),
        'CALL r62_zdf_v_solve(jj,pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 19],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:492-541': [
        '!                       !==  T-S Tracers  ==!',
        ('CALL r50_tra_finish( ts, ssh, Kaa )', 1), 50],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/eosbn2.f90:684-718': [
        ('SELECT CASE( neos )', 2),
        ('prd(ji,jj,jk) = (  zn * r1_rho0 - 1._wp  ) * ztm', 3), 35],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynhpg.f90:341-419': [
        'SUBROUTINE hpg_sco( kt, Kmm, puu, pvv, Krhs )',
        'END SUBROUTINE hpg_sco', 79],
    # --- ORCA2 round 50: executing OVERFLOW P3 kt=3 statement order ---
    'OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:327-354': [
        ('SELECT CASE( kstg )', 2), ('END SELECT', 6), 28],
    'OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:378-438': [
        ('SELECT CASE( kstg )', 3), ('DEALLOCATE( zub, zvb )', 2), 61],
    'OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:480-567': [
        '!                       !==  T-S Tracers  ==!',
        ('END SELECT', 8), 88],
    # --- ORCA2 round 49: held tracer statement and downstream OVERFLOW order ---
    'OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:501-508': [
        ('DO jn = 1, jpts', 2), ('END DO', 14), 8],
    'OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:311-338': [
        ('SELECT CASE( kstg )', 2), ('END SELECT', 6), 28],
    'OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/dynhpg.f90:341-414': [
        'SUBROUTINE hpg_sco( kt, Kmm, puu, pvv, Krhs )',
        'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 74],
    # --- ORCA2 card round 20: exact MPI transfer and slow-forcing owner ---
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:291-294': [
        '!                          ! set values computed in RK3_ssh',
        'zv_frc(:,:) =   Ve_rhs(:,:)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:325-330': [
        'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )',
        ('END DO   ;   END DO', 1), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:766-786': [
        ('IF( kt == nit000 .AND. jn <= 2 ) THEN', 3),
        "CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 21],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/lbclnk.f90:623-627': [
        ('IF( nn_comm <= 1 ) THEN', 4), ('ENDIF', 4), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/lbclnk.f90:1889-1909': [
        ('!                       !                       ________________________', 3),
        ('ishtRj(1:4,jf) = (/ ip1j, ip1j, ip0j, im0j /)', 3), 21],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/lbclnk.f90:1960-1979': [
        ('! ----------------------------------------------- !', 3),
        ('CALL MPI_ISEND( buffsnd_dp(ishtS(jn)+1)', 1), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/lbclnk.f90:2055-2073': [
        ('DO jn = 1, 2   ! next: do the MPI_RECV part', 2),
        ('ENDIF', 90), 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/stp2d.f90:189-210': [
        "WRITE(l2_slow_unit) l4_canon_3d(e3u_3d,'U')",
        "& l4_canon_2d(r1_hu_0,'U'), l4_canon_2d(r1_hv_0,'V')", 22],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo'
    '/stp2d.f90:218-233': [
        '!* baroclinic drag forcing *!',
        ("WRITE(l2_slow_unit) l4_canon_2d(Ue_rhs,'U'), l4_canon_2d(Ve_rhs,'V')", 2), 16],
    # --- ORCA2 card round 19: ranked histories and exchange-boundary walk ---
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:180-182': [
        'REAL(wp), DIMENSION(jpi,jpj) :: zu_trd, zu_spg',
        'REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ) :: zu_frc, zv_frc', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:458-467': [
        '! Round-18 WRITE-only per-rank stream:',
        'WRITE(l4_uhist_unit) l2_magic, 1, kt, 2, narea - 1, jpi, jpj, STORAGE_SIZE(1._wp)', 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:501-513': [
        '!* Extrapolate barotropic velocities at mid-step (jn+1/2)',
        'WRITE(l4_uhist_unit) un_e, ub_e, ubb_e, ua_e', 13],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:688-711': [
        '! Set next velocities:',
        '&   ) * ssvmask(ji,jj)', 24],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:754-770': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)',
        "CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 17],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:814-822': [
        ('IF( kt == nit000 .AND. jn <= 2 ) THEN', 3),
        'un_e   (:,:) = ua_e  (:,:)', 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:866-870': [
        ('IF( kt == nit000 ) THEN', 5),
        "WRITE(numout,*) 'LANE4_BT_UHIST_DUMP ', kt, 2, narea - 1, TRIM(l4_uhist_filename)", 5],
    # --- ORCA2 card round 18: ranked halo admission and operand walk ---
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:445-455': [
        '! Round-17 WRITE-only per-rank stream.',
        ('ENDIF', 14), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:539-544': [
        '! values of zhup2_e and zhvp2_e on the halo are not needed in bdy_vol2d',
        'zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)', 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:562-569': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1),
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 8],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:571-574': [
        'IF( kt == nit000 .AND. jn <= 2 ) THEN',
        ('ENDIF', 21), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:845-848': [
        ('IF( kt == nit000 ) THEN', 4),
        ('ENDIF', 41), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo'
    '/in_out_manager.f90:180-180':
        'lwp      = .FALSE.    !: boolean : true on the 1st processor only',
    # --- ORCA2 card round 15: source-ordered split-explicit solver walk ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:339-347': [
        '! Initialize barotropic variables:', ('ENDIF', 10), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:355-364': [
        ('IF( ln_bt_fw ) THEN', 2),
        'hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:460-493': [
        '!* Set extrapolation coefficients for predictor step:',
        'zsshp2_e(:,:) = za1 * sshn_e(:,:)', 34],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:505-536': [
        ('DO jj = ntsj-( 0), ntej+(  1 ) ; DO ji = ntsi-( 0), ntei+(  1)', 1),
        ('END DO   ;   END DO', 8), 32],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:550-558': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1),
        ('END DO   ;   END DO', 9), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:601-616': [
        '! Half-step back interpolation of SSH for surface pressure computation',
        ('END DO   ;   END DO', 14), 16],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:618-652': [
        '! Add Coriolis trend:',
        ('& l4_canon_2d(zu_trd,\'U\'), l4_canon_2d(zv_trd,\'V\')', 1), 35],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:666-679': [
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 1),
        ('END DO   ;   END DO', 17), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:755-779': [
        ('IF( lwp .AND. kt == nit000 .AND. jn <= 2 ) THEN', 3),
        ('& l4_canon_2d(ffv_nw,\'V\'), l4_canon_2d(ffv_ne,\'V\')', 2), 25],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:1533-1563': [
        'FUNCTION l4_canon_2d( pfield, cdgrid ) RESULT( zfield )',
        'END FUNCTION l4_canon_2d', 31],
    # --- ORCA2 card round 14: the barotropic slow forcing, in stp2d's order
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:139-147': [
        '!*  hydrostatic pressure gradient (HPG))  *!',
        'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )', 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:162-166': [
        'CASE( np_VEC_c2  )',
        'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:196-199': [
        'CASE( np_VEC_c2, np_LIN_dyn )',
        'Ve_rhs(ji,jj) = SUM( e3v_3d', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:218-221': [
        '!* baroclinic drag forcing *!',
        "l4_canon_2d(CdU_u,'U'), l4_canon_2d(CdU_v,'V')", 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:229-230': [
        'Ue_rhs(ji,jj) =  Ue_rhs(ji,jj) + r1_rho0 * utauU(ji,jj)',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:302-303': [
        'IF( ln_dynspg_ts )',
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )',
        2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:287-291': [
        'ssh_frc(:,:) = sshe_rhs(:,:)',
        'zCdU_v  (:,:) = CdU_v   (:,:)', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:320-324': [
        'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )',
        'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/dynspg_ts.f90:668-671': [
        'ua_e(ji,jj) = (                                 un_e(ji,jj)',
        '+ zu_frc(ji,jj) ) &', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domain.f90:199':
        'hu_0(:,:) = hu_0(:,:) + e3u_3d(:,:,jk) * umask(:,:,jk)',
    # --- ORCA2 card round 13: the river runoff's WATER and NEMO's emp ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/divhor.f90:142':
        '!==  + runoffs divergence  ==!',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcrnf.f90:279-282': [
        '!==   runoff put only at the surface   ==!',
        'phdivn(ji,jj,1) = phdivn(ji,jj,1) - rnf(ji,jj) * r1_rho0', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:278-281': [
        'sshe_rhs(:,:) =                 emp(:,:)',
        'sshe_rhs(:,:) = r1_rho0 * sshe_rhs(:,:)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:282-286': [
        'IF( .NOT.lk_linssh ) THEN           '
        '!* only heat and salt fluxes associated with mass fluxes',
        'pts(ji,jj,1,jp_sal,Krhs) = pts(ji,jj,1,jp_sal,Krhs) - '
        'emp(ji,jj)*pts(ji,jj,1,jp_sal,Kbb) * z1_rho0_e3t', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:318-324': [
        ('IF( ln_rnf ) THEN         '
         '! input of heat and salt due to river runoff', 2),
        'pts(ji,jj,jk,jp_sal,Krhs) = pts(ji,jj,jk,jp_sal,Krhs)  '
        '+ rnf_tsc(ji,jj,jp_sal) * zdep', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/stprk3_stg.f90:137,152-154': [
        '! save ssh, uu_b, vv_b at N+1  (computed in dynspg_ts)',
        ('CASE ( np_HYB )', 1),
        ('ssh (:,:,Kaa) = r2_3 * ssh (:,:,Kbb) + r1_3 * ssha(:,:)', 2), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo'
    '/stprk3_stg.f90:270-277': [
        ('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; '
         'DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 1),
        'zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0(ji,jj) '
        '/(1._wp+r3v(ji,jj,Kmm))) - vv_b(ji,jj,Kmm)', 8],
    # --- ORCA2 card round 78: the two rmxl_min arms, and which one this deck
    # takes.  The whole IF/ELSE/ENDIF is pinned as one span because the round's
    # claim is that the forced arm EXISTS and the derivation is not evaluated.
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-841': [
        'IF( ln_zdfiwm ) THEN',
        "minimum mixing length with your parameters rmxl_min = ", 7],
    # --- ORCA2 card round 79b: NEMO's vertical-physics chain, in the order
    # zdf_phy runs it, and the internal-wave arm the ORCA2 card does not
    # execute.  Each boundary is pinned on its own so a widened span cannot
    # silently swallow the neighbouring arm.
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:349-350': [
        'avt(ji,jj,jk) = avt_k(ji,jj,jk)',
        'avm(ji,jj,jk) = avm_k(ji,jj,jk)', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:355':
        'avt(ji,jj,jk) = avt(ji,jj,jk) + 2._wp * rn_avt_rnf '
        '* rnfmsk(ji,jj) * wmask(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359':
        'IF( ln_zdfevd )   CALL zdf_evd( kt, Kmm, Krhs, avm, avt )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363':
        'CALL zdf_ddm( kt, Kmm,  avm, avt, avs )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:372':
        'IF( ln_zdfiwm )   CALL zdf_iwm( kt, Kmm, avm, avt, avs )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:294':
        'zav_wave(ji,jj) = MIN( MAX( 1.4e-7_wp, zav_wave(ji,jj) ), '
        '1.e-2_wp ) * wmask(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:314-316': [
        'p_avs(ji,jj,jk) = p_avs(ji,jj,jk) + zav_wave(ji,jj) * zav_ratio(ji,jj)',
        'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + zav_wave(ji,jj)', 3],
    # The salt/heat split also adds to the momentum coefficient, which is why
    # the wave arm's momentum increment is measured after it and not before.
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:172':
        'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + MAX( zavft + zavdt, zavfs + zavds )',
    # --- ORCA2 card round 1: initial category-load SSH adjustment ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:442-459': [
        'snwice_mass  (:,:) = tmask(:,:,1) * SUM',
        'ssh(:,:,Kbb) = ssh(:,:,Kbb) - zsshadj', 18],
    # --- ORCA2 card round 4: returned surface record is root-rank only ---
    'ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE/BLD/ppsrc/nemo/stprk3.f90:396-400': [
        ('IF( .NOT.lwp ) RETURN', 5),
        "FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios", 5],
    # --- ORCA2 card round 5: step-entry state remains root-rank only ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:92-94': [
        'IF( lwp .AND. kstp >= nit000 .AND. kstp <= nit000 + 59 ) THEN',
        'WRITE(cl_traj,\'("oracle_step_entry_kt",I8.8,".bin")\') kstp', 3],
    # --- ORCA2 card round 69: surface operands are complete after sbc ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:151-152': [
        'CALL sbc        ( kstp, Nbb, Nbb )',
        'CALL l4_dump_ocean_surface_input( kstp, Nbb )', 2],
    # --- ORCA2 card round 71: independent-month clock and terminal state ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246': [
        'IF( sd(jf)%ln_tint ) THEN',
        'sd(jf)%fnow(:,:,:) = ztintb * sd(jf)%fdta(:,:,:,ibb) + ztinta * sd(jf)%fdta(:,:,:,iaa)',
        12],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184': [
        "CALL iom_rstput( kt, nitrst, numrow, 'rdt', rn_Dt )",
        "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )",
        15],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:223-272': [
        '! Stage 3 :',
        'IF( ln_sto_eos )   CALL sto_rst_write( kstp )', 50],
    # --- ORCA2 card round 72: independent temperature source walk ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:633-643': [
        ('IF( ln_tile )   CALL dom_tile_start', 1),
        ("& l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')", 4), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:645-651': [
        'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )',
        ("& l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')", 6), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:670-680': [
        ('SELECT CASE( kstg )', 4),
        '&                /          ( 1._wp + r3t(ji,jj,Kaa) )', 11],
    # --- ORCA2 card round 74: independent metric-U transport walk ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:45-49': [
        'INTEGER,  PUBLIC, PARAMETER ::   np_LIN = 0',
        'INTEGER  :: n_baro_upd =  np_HYB', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:274-284': [
        ('CASE ( np_LIN, np_HYB )', 2),
        ('END DO   ;   END DO   ;   END DO', 1), 11],
    # --- ORCA2 card round 76: independent external U-transport producer ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:530-536': [
        '! resulting flux at mid-step (not over the full domain)',
        ('END DO   ;   END DO', 8), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:563-580': [
        '! Sum over sub-time-steps to compute advective velocities',
        "& l4_canon_2d(vn_adv,'V')", 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:829-847': [
        '! Finalize sums:',
        "WRITE(l2_adv_unit) l4_canon_2d(un_adv,'U'), l4_canon_2d(vn_adv,'V')", 19],
    # --- ORCA2 card round 77: external U-transport operand pair ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:460-469': [
        '!* Set extrapolation coefficients for predictor step:',
        ('ENDIF', 14), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:476-485': [
        '!* Extrapolate barotropic velocities at mid-step (jn+1/2)',
        ('END DO   ;   END DO', 3), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:487-522': [
        ('IF( .NOT.lk_linssh ) THEN', 1),
        ('ENDIF', 17), 36],
    # --- ORCA2 card round 73: independent stage-1 CEN2 walk ---
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv.f90:491-535': [
        '! FCT at last stage only with RK3',
        ('ENDIF', 27), 45],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:149-155': [
        ('CASE(  2  )                         !* 2nd order centered', 1),
        ('END DO   ;   END DO', 1), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:157-161': [
        'DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)                     ! Horizontal divergence of advective fluxes',
        ('END DO   ;   END DO', 2), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:202-228': [
        'IF( lk_linssh ) THEN                !* top value   (linear free surf. only as zwz is multiplied by wmask)',
        ('END DO   ;   END DO', 9), 27],
    # --- ORCA2 card round 6: full-domain entry and first runtime stop ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:218-253': [
        'IF( cn_cfg == "orca" .OR. cn_cfg == "ORCA" ) THEN',
        'sf_tsd(jp_tem)%fnow( mi0(ii0,nn_hls):mi1(ii1,nn_hls) , mj0(ij0,nn_hls):mj1(ij1,nn_hls) , 14:20 ) = 6.0_wp',
        36],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1587-1647': [
        'SUBROUTINE bn2( pts, pab, pn2, Kmm, kbnd )',
        'END SUBROUTINE bn2_t', 61],
    # --- ORCA2 card round 7: the independent initial state and EOS-80 ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:117-118': [
        'IF( ln_tsd_init ) THEN',
        'CALL dta_tsd( nit000, ts(:,:,:,:,Kbb) )', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:308-309': [
        'ptsd(ji,jj,jk,jp_tem) = ptsd(ji,jj,jk,jp_tem) * tmask(ji,jj,jk)',
        'ptsd(ji,jj,jk,jp_sal) = ptsd(ji,jj,jk,jp_sal) * tmask(ji,jj,jk)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/mppini.f90:1587-1593': [
        'mi0(ji,jh) = MAX( 1 , MIN( ji - iimpp + 1, ipi+ishft+1 ) )',
        'mj1(jj,jh) = MAX( 0 , MIN( jj - ijmpp + 1, ipj+ishft   ) )', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:2284-2293': [
        'CASE( np_eos80 )                        '
        '!==  polynomial EOS-80 formulation  ==!',
        ('r1_Z0  = 1.e-4_wp', 2), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1281-1330': [
        ('CASE( np_teos10, np_eos80 )                '
         '!==  polynomial TEOS-10 / EOS-80 ==!', 7),
        'pab(ji,jj,jk,jp_sal) = zn / zs * r1_rho0 * ztm', 50],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:912-937': [
        'SELECT CASE( nn_e3f_typ )',
        'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)',
        26],
    # --- ORCA2 card round 62: executing vector-advection source order ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:402-420': [
        'CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )',
        'L2_RK_STAGE2_TERM_DUMP', 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynadv.f90:134-138': [
        ('SELECT CASE( n_dynadv )', 1),
        'CALL dyn_zad', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynkeg.f90:117-130': [
        'CASE ( nkeg_C2 )',
        ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) -', 1), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzad.f90:102-137': [
        'zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp',
        '&                                              * zWdzV(ji,jj)', 36],
    # --- ORCA2 card round 63: incomplete vector record and QCO repair ---
    'ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynadv_round62_writer.f90:91-92': [
        "WRITE(r62_unit) 'e3u_Kmm         '",
        "WRITE(r62_unit) 'e3v_Kmm         '", 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90:136-140': [
        'LOGICAL, PUBLIC, PARAMETER ::   lk_qco    = .TRUE.',
        'LOGICAL, PUBLIC, PARAMETER ::   lk_vco_1d3d = .TRUE.', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90:318-367': [
        'IF( lk_qco .OR. lk_linssh ) THEN',
        '&                                e3vw(jpi,jpj,jpk,jpt) ,   STAT=ierr(ii) )', 50],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_acquisition/dynadv_round62_writer.F90:79-89': [
        'ALLOCATE( r62_payload(jpi,jpj,jpk) )',
        "R62_3D('e3v_Kmm         ',r62_payload)", 11],
    # --- ORCA2 card round 64: admitted ZAD source-order replay ---
    'ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynadv.f90:136-143': [
        ('SELECT CASE( n_dynadv )', 1),
        'CALL dynadv_round62_after_zad( kt, Kmm, Krhs, nn_dynkeg, puu, pvv )', 8],
    'ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynzad.f90:102-137': [
        'zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp',
        '&                                              * zWdzV(ji,jj)', 36],
    # --- ORCA2 card round 21: merge-owner substitution ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937': [
        'SELECT CASE( nn_e3f_typ )',
        'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)',
        26],
    # --- ORCA2 card round 31: the two consumers' own frozen F thicknesses ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123':
        ('zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk) * (e3f_3d(ji-1,jj-1,jk)', 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:914-919': [
        ('DO jk =  1,  jpk  ; DO jj = ntsj-(  0), ntej+(   0) ; '
         'DO ji = ntsi-( 0), ntei+(   0)', 1),
        ('END DO   ;   END DO   ;   END DO', 2), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:935':
        ("CALL lbc_lnk( 'dynvor', e3f_0vor, 'F', 1._wp )", 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:937':
        ('WHERE( e3f_0vor(:,:,:) == 0._wp )   '
         'e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 1),
    'domzgr_substitute.h90:100':
        ('#     define  e3f_0(i,j,k)    e3f_3d(i,j,k)', 1),
    'DOM/domzgr.F90:173': (
        "CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', "
        "psgn = 1._wp, kfill = jpfillcopy )", 1),
    # --- ORCA2 card round 27: split the shared F-thickness consumers ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738': [
        ('DO jk = 1, jpkm1', 4),
        ('z1_e3f(ji,jj) = 1._wp / (e3f_0vor(ji,jj,jk)', 1),
        5],
    # --- ORCA2 card round 29: EEN numerator/denominator split ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:741-779': [
        ('SELECT CASE( kvor )                 !==  vorticity considered  ==!', 3),
        ("CALL ctl_stop('STOP','dyn_vor: wrong value for kvor'  )", 3),
        39],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:784-802': [
        ('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d(ji,jj,jk)', 3),
        ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2),
        19],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286': [
        ('IF( PRESENT( pr3f ) ) THEN             !==  ratio at f-point  ==!', 2),
        ('END DO   ;   END DO', 6), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157': [
        'e1e2t (:,:) = e1t(:,:) * e2t(:,:)',
        'IF( ie1e2u_v == 0 ) THEN', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-206': [
        ('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)', 1),
        "CALL lbc_lnk('domain', hf_0, 'F', 1._wp)", 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dommsk.f90:258':
        ('fe3mask(:,:,:) = fmask(:,:,:)', 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:139-166': [
        ('!*  hydrostatic pressure gradient (HPG))  *!', 1),
        ('CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 1), 28],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:189-204': [
        ("WRITE(l2_slow_unit) l4_canon_3d(e3u_3d,'U')", 1),
        ('Ve_rhs(ji,jj) = Ve_rhs(ji,jj) + SUM(', 1), 16],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:218-221': [
        ('!* baroclinic drag forcing *!', 1),
        ('l4_canon_2d(CdU_v', 1), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:225-235': [
        ('WRITE(l2_slow_unit) r1_rho0', 1),
        ('LANE2_SLOW_FORCING_DUMP', 1), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/stp2d.f90:155-187': [
        ('CALL eos    ( ts, Kbb, rhd )', 1),
        ('WRITE(r40_family_unit) l4_canon_3d(uu(:,:,:,Krhs)', 5), 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/dynhpg.f90:340-451': [
        ('SUBROUTINE hpg_sco( kt, Kmm, puu, pvv, Krhs )', 1),
        ('END SUBROUTINE hpg_sco', 1), 112],
    'ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/eosbn2.f90:810-842': [
        ('CASE( np_teos10, np_eos80 )', 3),
        ('prd(ji,jj,jk) = (  zn * r1_rho0 - 1._wp  ) * ztm', 3), 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/dynhpg.f90:403-458': [
        ('zhpi(ji,jj) = zcoef0 * r1_e1u(ji,jj)', 1),
        ('r41_sum_u (ji,jj,jk) = zhpi(ji,jj) + zuap', 1), 56],
    # --- ORCA2 card round 43: stage-1 tracer transport handoff ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:265-284': [
        ('ALLOCATE( zub(ntsi-', 1),
        ('END DO   ;   END DO   ;   END DO', 1), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:551-645': [
        ('IF( .NOT.ln_shuman ) THEN', 1),
        ('CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 1), 95],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv.f90:267-315': [
        ('IF( ll_Fw ) THEN', 1),
        ('CLOSE(993)', 1), 49],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sshwzv.f90:271-299': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 2),
        ('END DO   ;   END DO   ;   END DO', 6), 29],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160': [
        ('DO jn = 1, kjpt', 1),
        ('r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:202-227': [
        ('IF( lk_linssh ) THEN', 1),
        ('r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 4), 26],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:275-328': [
        ('!        EMP, SFX and QNS effects', 2),
        ('ENDIF', 24), 54],
    # --- ORCA2 card round 44: first downstream source-order debt ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:467-480': [
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 1),
        ('END DO   ;   END DO   ;   END DO', 6), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681': [
        ('SELECT CASE( kstg )', 4),
        ('END DO   ;   END DO   ;   END DO', 8), 12],
    # --- ORCA2 card round 66: independent from-rest entry ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140': [
        ('CALL dta_tsd_init', 1),
        ('vv    (:,:,:,Kmm) = vv   (:,:,:,Kbb)', 1), 48],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:217-254': [
        ('!                                   !==   ORCA_R2 configuration and T & S damping   ==!', 1),
        ('ENDIF', 9), 38],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:307-310': [
        ('DO jk =  1,  jpk', 2),
        ('END DO   ;   END DO   ;   END DO', 2), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465': [
        ('! 4) Adjust ssh and vertical scale factors to snow-ice mass', 1),
        ('CALL dom_qco_zgr( Kbb, Kmm )', 1), 26],
    # --- ORCA2 card round 45: stage-1 r3 interpolation owner ---
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179': [
        ('!                     !==  ssh/h0 ratio at Kaa  ==!', 1),
        ('r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 1), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:556':
        ('zwz(ji,jj) = zwz(ji,jj) / (e3f_0vor(ji,jj,jk) *(1._wp+r3f(ji,jj)*fe3mask(ji,jj,jk)))', 1),
    # --- ORCA2 card round 22: exact stage-to-entry transition and LDF owner ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3.f90:230-241': [
        'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 12],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:493-506': [
        'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )',
        'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )',
        14],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140': [
        ('zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk)', 1),
        ('+ ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 18],
    # --- ORCA2 card round 26: split the three live-thickness positions ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1),
        ('e1u(ji-1,jj  ) * pu(ji-1,jj  ,jk,Kbb)', 1), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:127-129': [
        ('zwt(ji,jj)     = ahmt(ji,jj,jk) * r1_e1e2t(ji,jj)', 1),
        ('e1v(ji,jj-1)*(e3v_3d(ji  ,jj-1,jk)', 1), 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:132-140': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 1),
        ('+ ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 9],
    # --- ORCA2 card round 24: held Decision-54 whole attribution ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353': [
        "CASE( -30  )",
        ('CALL iom_close( inum )', 2),
        6],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393': [
        'IF( .NOT.l_ldfdyn_time ) THEN',
        'ahmf(:,:,1:jpkm1) = SQRT( ahmf(:,:,1:jpkm1) ) * fmask(:,:,1:jpkm1)',
        7],
    # --- ORCA2 card round 23: Decision 58 second continuity solve ---
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:323-329': [
        ('IF( ln_dynadv_vec ) THEN', 1),
        'CALL wAimp( kstp, Kmm, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, wi, np_velocity, ld_diag=.TRUE. )',
        7],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/traadv.f90:296-300': [
        'CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )',
        'CALL wAimp( kt, Kmm, pFu, pFv, ww, wi, np_transport )',
        5],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/divhor.f90:123-130': [
        'SELECT CASE ( ik_ind )',
        (') * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1),
        8],
    # --- round 184: DINO developed-state closed V-face repair ---
    'DINO/BLD/ppsrc/nemo/domqco.f90:177-181': [
        'CALL dom_qco_r3c( ssh(:,:,Kbb), r3t(:,:,Kbb), r3u(:,:,Kbb), r3v(:,:,Kbb)           )',
        '&                         r3u(:,:,Kmm), \'U\', 1._wp, r3v(:,:,Kmm), \'V\', 1._wp, r3f(:,:), \'F\', 1._wp )', 5],
    'DINO/BLD/ppsrc/nemo/domqco.f90:211-215': [
        ('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 1),
        ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 5],
    'DINO/BLD/ppsrc/nemo/lbclnk.f90:1816-1820': [
        ('zland = 0._wp                                     ! land filling value: zero by default', 3),
        ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), 5],
    'DINO/BLD/ppsrc/nemo/lbclnk.f90:1866-1871': [
        ('DO jn = 1, 4   ! 4 sides', 2),
        ('ELSE                                ;   ifill(jn,jf) = jpfillcst       ! constant value (zland)', 3), 6],
    'DINO/BLD/ppsrc/nemo/lbclnk.f90:2130-2135': [
        ('IF(     ifill(jn,jf) == jpfillcst ) THEN', 5),
        ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = zland', 5), 6],
    'DINO/BLD/ppsrc/nemo/dynspg_ts.f90:484-487': [
        'hu_e  (:,:) =    (hu_0(:,:) *(1._wp+r3u(:,:,Kmm)))',
        'hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 4],
    # --- round 166: developed-state shear statement walk ---
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/stprk3.f90:167-168': [
        'CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )',
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 2],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90:317-320': [
        ('IF( l_zdfsh2 ) THEN', 1),
        '&                      sh2    )', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:97-103': [
        'ELSE',
        '&         * wumask(ji,jj,jk)', 7],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:104-108': [
        ('zsh2v(ji,jj) = ( p_avm(ji,jj+1,jk) + p_avm(ji,jj,jk) )', 2),
        '&         * wvmask(ji,jj,jk)', 5],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:111-114': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 1),
        ('END DO   ;   END DO', 3), 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:116-119': [
        'DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0) ! set p_sh2',
        ('END DO   ;   END DO', 4), 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/dommsk.f90:234-242': [
        'Ocean/land mask at wu-, wv- and w points',
        'wvmask(:,:,jk) = vmask(:,:,jk) * vmask(:,:,jk-1)', 9],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/domqco.f90:211-218': [
        ('ratio at u-,v-point', 1),
        ('END DO   ;   END DO', 2), 8],
    # --- round 165: developed-state TKE statement walk ---
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r54_tke.f90:128-130': [
        'WRITE(r54_unit) r54_magic',
        '& 1,ntei-ntsi+1,1,ntej-ntsj+1,STORAGE_SIZE(1._wp)', 3],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:65-67': [
        'WRITE(r101_unit) r101_magic',
        '& STORAGE_SIZE(1._wp)', 3],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:284':
        'en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )',
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:387':
        'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)',
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:433-435': [
        'zd_up(ji,jk) = zzd_up',
        'zdiag(ji,jk) = 1._wp - zzd_lw - zzd_up + zfact2 * dissl(ji,jj,jk) * wmask(ji,jj,jk)', 3],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:438-441': [
        'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * (  p_sh2(ji,jj,jk)',
        '&                                  ) * wmask(ji,jj,jk)', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:475-492': [
        'DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)                 ! First recurrence',
        'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:689-692': [
        'zemlm = MIN ( zmxld(ji,jk),  zmxlm(ji,jk) )',
        'zmxld(ji,jk) = zemlp', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:700-705': [
        'DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)   !* vertical eddy viscosity',
        'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:711':
        'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk)',
    # --- round 162: the T-point ratio, its statement and its two readers ---
    # The ratio at the T point is a single product of the stage sea surface
    # height, and it is the ONE operand of the velocity-indicator continuity
    # solve that rounds 159-161 never substituted.  The two compiled lines it
    # reaches are already mapped above (divhor.f90:126-130 divides by it and
    # :153 multiplies it back); what is new here is the statement that makes
    # it and legoESM's three answering lines.
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:257':
        ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 2),
    'ocean_pe_latlon_cgrid.py:1739':
        ('r3_now = jax.lax.optimization_barrier(eta_now * r1_h0)', 1),
    'ocean_pe_latlon_cgrid.py:1740':
        ('live_t = e3t0 * (1.0 + r3_now[..., None] * tmask) * tmask', 1),
    'ocean_pe_latlon_cgrid.py:1764':
        ('flux_u, west, flux_v, south, r1_area_t, live_t[..., jk],', 1),
    # Round 162's review disputed whether the multiply-back is on the
    # vertical-velocity path.  It is: the statement writes the routine's own
    # OUTPUT argument, and the vertical-velocity routine receives exactly that
    # array.  Those two lines are cited, and so is legoESM's answering one.
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:101':
        ('pe3divUh       ! e3t*div[Uh]', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:278':
        ('CALL div_hor( kt, Kbb, Kmm, pu, pv, ze3div )', 1),
    'ocean_pe_latlon_cgrid.py:1609':
        ('return jax.lax.optimization_barrier(live_e3t * hdiv) * tmask', 1),
    'ocean_pe_latlon_cgrid.py:1835-1836': [
        ('r3_after = jax.lax.optimization_barrier(eta_after * r1_h0)', 1),
        ('r3_before = jax.lax.optimization_barrier(eta_before * r1_h0)', 1), 2],
    # --- round 161: the stage face ratio, its producer and its composition ---
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:172':
        ('ssh (:,:,Kaa) = r2_3 * ssh (:,:,Kbb) + r1_3 * ssha(:,:)', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:185':
        ('CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:211':
        ('r3u(:,:,Kaa) = r2_3 * r3u(:,:,Kbb) + r1_3 * r3ua(:,:)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:255':
        ('r3u(:,:,Kaa) = r1_2 * ( r3u(:,:,Kbb) + r3ua(:,:) )', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:266-267': [
        ('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2),
        ('&                    + e1e2t(ji+1,jj) * pssh(ji+1,jj)  ) '
         '* r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)', 2), 2],
    'vertical.py:238-239': [
        ('weighted_eta = b(area_t * eta)', 1),
        ('num_u = b(half * b(weighted_eta + jnp.roll(weighted_eta, -1, '
         'axis=1)))', 1), 2],
    'vertical.py:247': ('r3u = b(b(num_u * r1_hu0) * r1_area_u)', 1),
    'vertical.py:251':
        ('e3u = b(e3u_0 * b(one + r3u[..., None] * umask3))', 1),
    # --- round 160: legoESM's own two-solve statements ---
    'ocean_model_latlon_cgrid.py:1733-1770': [
        ('def nemo_stage_momentum_wzv_executes(config, hooks=None) -> bool:', 1),
        ('return bool(config_split)', 1), 38],
    'ocean_model_latlon_cgrid.py:1894-1899': [
        ('w_momentum, _, _ = nemo_qco_wzv_operands(', 1),
        ('runoff_mass_flux=runoff_mass_flux)', 2), 6],
    'ocean_model_latlon_cgrid.py:7394-7400': [
        ('def _momentum_stage_w(geom):', 1),
        ('return geom[2] if geom[11] is None else geom[11]', 1), 7],
    # --- round 160: the stage clock, the stage after-level and the two
    # adaptive-implicit partitions, on the build that wrote the record ---
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:52':
        ('INTEGER  :: n_baro_upd =  np_HYB', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:221-222': [
        ('rDt = r1_2 * rn_Dt', 1),
        ('r1_Dt = 1._wp / rDt', 2), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:254':
        ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:265-266': [
        ('rDt = rn_Dt                   ! set time-step : rn_Dt', 1),
        ('r1_Dt = 1._wp / rDt', 3), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:268':
        ('ssh (:,:,Kaa) = ssha(:,:)     ! recover ssh and (uu_b,vv_b) at N + 1', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:362':
        ('CALL wAimp( kstp, Kmm, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, wi, np_velocity, ld_diag=.TRUE. )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:277':
        ('CALL wAimp( kt, Kmm, pFu, pFv, ww, wi, np_transport )', 1),
    # --- round 159: the two continuity call forms and the second solve the
    # tracer transport runs, on the build that wrote the admitted record ---
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:309':
        ('zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:315-316': [
        ('zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk))) * ( uu(ji,jj,jk,Kmm) + zub(ji,jj)*umask(ji,jj,jk) )', 1),
        ('zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk))) * ( vv(ji,jj,jk,Kmm) + zvb(ji,jj)*vmask(ji,jj,jk) )', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356':
        ('IF( ln_dynadv_vec ) THEN', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:358':
        ('IF( kstg /= 1 ) THEN', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360':
        ('CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:367':
        ('CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298': [
        ('pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)', 1),
        ('+ r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130': [
        ('hdiv(ji,jj,jk) = (  (  e2u(ji  ,jj) * (e3u_3d(ji  ,jj,jk) *(1._wp+r3u(ji  ,jj,Kmm)*umask(ji  ,jj,jk))) * pu(ji  ,jj,jk)', 1),
        ('&             ) * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:134-138': [
        ('hdiv(ji,jj,jk) = (  (  pu(ji  ,jj,jk)       &   ! add () for NP repro', 1),
        ('&             ) * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153':
        ('pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:195':
        ('IF( ln_dynadv_vec )   ll_Fw = .true.', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274':
        ('CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:280':
        ('pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 1),
    # --- round 158: the two compiled halves of the vector-invariant
    # dyn_adv, on the build that wrote the admitted record ---
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:153':
        ('CASE( np_VEC_c2  )                                                         != vector form =!', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:171':
        ('CALL dyn_keg     ( kt, nn_dynkeg     , Kmm, puu, pvv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:176':
        ('CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90:121-125': [
        'zu =    puu(ji-1,jj  ,jk,Kmm) * puu(ji-1,jj  ,jk,Kmm)   &',
        'zhke(ji,jj) = 0.25_wp * ( zv + zu )', 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90:129-130': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1),
        ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:102':
        ('zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp                  ! set surface (jk=1) vertical advection to zero', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:112-116': [
        'zWf  = e1e2t(ji  ,jj  ) *   ww(ji  ,jj  ,jk+1)',
        'zzWfu = zWfi + zWf                     ! averaging at uw- and vw-points (x2)', 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:119':
        ('zzWdzU = zzWfu * ( puu(ji,jj,jk,Kmm) - puu(ji,jj,jk+1,Kmm) )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:123-126': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk)))   &', 1),
        ('&                                            * ( zWdzV(ji,jj) + zzWdzV )', 1), 4],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:134-137': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk)))   &', 2),
        ('&                                              * zWdzV(ji,jj)', 1), 4],
    # --- round 157: developed stage-2 right-hand-side walk, on the build
    # that wrote the record it walks ---
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:462':
        ("CALL r156_stage2_pair3( 'rhs_entry       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:495':
        ("CALL r156_stage2_pair3( 'rhd_ww          ', rhd, ww )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:497':
        ('CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:499':
        ("CALL r156_stage2_pair3( 'after_hpg       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:510':
        ('CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:512':
        ("CALL r156_stage2_pair3( 'after_vor       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:523':
        ('IF( ln_dynadv_vec ) THEN                        ! VIF: only velocities used for momentum advection', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525':
        ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:531':
        ("CALL r156_stage2_pair3( 'after_adv       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:721':
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:723':
        ('uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:738':
        ("CALL r156_stage2_pair3( 'uu_vv_Kaa_raw   ', uu(:,:,:,Kaa), vv(:,:,:,Kaa) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:787':
        ('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:810':
        ('uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 1),
    # --- round 156: developed stage-2 velocity split ---
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:217-256': [
        'CASE ( 2 )           !==  Stage 2  ==!',
        ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 40],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:400-497': [
        'CASE ( 2 , 3 )    !==  Stage 2 & 3  ==!',
        'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 98],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:475':
        ('CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:485':
        ('CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:497':
        ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:697-699': [
        'uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) ) * uu(ji,jj,jk,Kbb )',
        '/           ( 1._wp + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)', 3],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:732':
        ('CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:745':
        ('IF( kstg == 3 )   CALL dyn_zdf', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:753':
        ('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:772':
        ('uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 1),
    # --- round 155: developed stage-3 U-transport operand walk ---
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:261-277': [
        'CASE ( 3 )           !==  Stage 3  ==!',
        'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 17],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:300-309': [
        ('SELECT CASE( n_baro_upd )', 4),
        'zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kmm))) - vv_b(ji,jj,Kmm)', 10],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:313-315': [
        ('DO jk =  1,  jpkm1', 1),
        'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 3],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:317-321': [
        'IF( lwp .AND. .NOT.ln_tile .AND. kstp == 1081 .AND. kstg == 3 )',
        '& un_adv, vn_adv, r1_hu_0, r1_hv_0, uu_b(:,:,Kmm), vv_b(:,:,Kmm) )', 5],
    # --- round 154: developed production FCT walk and inherited transport ---
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:260-314': [
        'CASE ( 3 )           !==  Stage 3  ==!',
        'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))',
        55],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:785':
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:879':
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )',
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90:195':
        'IF( ln_dynadv_vec )   ll_Fw = .true.',
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90:266-280': [
        'IF( ll_Fw ) THEN',
        'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 15],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:176-178': [
        "IF( cdtype == 'TRA' ) CALL r153_begin",
        'CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt', 3],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:515-553': [
        'ptFu(ji,jj,jk) = MAX( pU(ji,jj,jk) , 0._wp ) * pt_b(ji,jj,jk)',
        ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)', 2), 39],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:924-953': [
        'zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji+1,jj,ik) )',
        ('pcc(ji,jj,jk) = pcc(ji,jj,jk) * zcoef', 2), 30],
    # --- round 153: developed-state FCT operator acquisition ---
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:164-199': [
        'DO jn = 1, kjpt            !==  loop over the tracers  ==!',
        ('END DO   ;   END DO   ;   END DO', 4), 36],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:495-610': [
        '! *** 1st step',
        ('END DO   ;   END DO   ;   END DO', 28), 116],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:306-330': [
        '!        !==  monotonicity algorithm  ==!',
        ('END DO   ;   END DO   ;   END DO', 9), 25],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:849-883': [
        '!        !==  compute the beta term  ==!   (zbetup/do)',
        ('END DO   ;   END DO', 53), 35],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:886-936': [
        '!        !==  monotonic flux  ==!',
        'END DO    ! jk-loop', 51],
    # --- round 149: developed live lateral-diffusion geometry ---
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:111-140': [
        'r148_ldf_dump = lwp .AND. kt == 1081 .AND. Kbb == Kmm',
        ('ENDIF', 4), 30],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:142-197': [
        ('DO jk = 1, jpkm1', 1), ('ENDIF', 7), 56],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:204-217': [
        '! Round 140: repeat the established slow-forcing operand list',
        ('ENDIF', 6), 14],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:157-176': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1),
        20],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:256-286': [
        ('DO jj = ntsj-( nn_hls), ntej+(  nn_hls ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls)', 2),
        ('END DO   ;   END DO', 6), 31],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:273-286': [
        ('IF( PRESENT( pr3f ) ) THEN             !==  ratio at f-point  ==!', 2),
        ('END DO   ;   END DO', 6), 14],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynvor.f90:911-936': [
        'SELECT CASE( nn_e3f_typ )',
        'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)',
        26],
    'vertical.py:456-554': [
        'def nemo_qco_live_vorticity_e3f_cgrid(',
        'return jnp.concatenate([with_south[:, -1:], with_south], axis=1)',
        99],
    'ocean_model_latlon_cgrid.py:5568-5570': [
        '_ws_uses_nemo_ldf_e3 = (',
        'and _cfg_b.lateral_viscosity_e3_weighting == "nemo_e3")', 3],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:158-164': [
        'CALL eos    ( ts, Kbb, rhd )',
        ('IF( lwp .AND. kt == 1081 )   WRITE(r146_family_unit)', 2), 7],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stprk3_stg.f90:704-739': [
        ('!              !---------------!', 2),
        'IF( kstg == 3 )   CALL dyn_zdf(', 36],
    'ocean_model_latlon_cgrid.py:5617-5667': [
        'if not self._nemo_ws_test_hooks.legacy_hadv_min_face_thickness:',
        'ldf_thickness_operands=_ws_ldf_thickness_kbb,', 51],
    'ocean_model_latlon_cgrid.py:6177-6240': [
        '_stage_ldf_thickness = None', '_stage_ldf_thickness),', 64],
    # --- round 135: accepted-state swap before the daily tracer record ---
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:220-229': [
        'IF ( .NOT. l_perpetual_ts ) THEN', ('ENDIF', 8), 10],
    # --- round 134: compiled daily-restart schema and next-step SSH use ---
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:188-201': [
        '!  RK3 : single first external mode computation',
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:222-226': [
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:249-260': [
        'CALL dia_wri   ( kstp,      Nbb )',
        'CALL rst_write    ( kstp, Nbb, Nnn, Naa )', 12],
    # --- round 177: active developed tracer-LDF program before acquisition ---
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf.f90:69-110': [
        'SUBROUTINE tra_ldf( kt, Kbb, Kmm, pts, Krhs )',
        'CALL traldf_iso_lap  ( kt, Kbb, Kmm, pts, Krhs, l_ptr, l_hst )', 42],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf_iso.f90:154-300': [
        ('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1),
        ('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 2), 147],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:176-184': [
        "CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )",
        "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )",
        9],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:354-370': [
        '!                                     !*  RK3: Read ssh at Kbb',
        'ssh(:,:,Kaa) = ssh(:,:,Kbb)               ! no ssh variation in ww computation',
        17],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/dynspg_ts.f90:974-980': [
        ('IF( nn_bt_flt == 3 ) THEN', 2),
        "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'     ,    vb_e(:,:) )",
        7],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/zdftke.f90:895-901': [
        "ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN",
        "CALL iom_rstput( kt, nitrst, numrow, 'dissl', dissl )", 7],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stp2d.f90:150-166': [
        'IF( .NOT.lk_linssh ) THEN',
        'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 17],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/sshwzv.f90:293-299': [
        ('ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!', 2),
        ('END DO   ;   END DO   ;   END DO', 6), 7],
    # --- round 178: admitted developed tracer-LDF statement walk ---
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3.f90:173-180': [
        'IF( l_ldfslp ) THEN', ('ENDIF', 7), 8],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3_stg.f90:928-934': [
        ('IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000 .AND. kstg == 3 )', 5),
        'IF( ln_trabbl  )   CALL tra_bbl', 7],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf.f90:105-118': [
        ('SELECT CASE ( nldf_tra )', 2),
        'CALL traldf_iso_blp', 14],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:222-268': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4),
        '&                   * ( umask(ji,jj  ,jk) + umask(ji,jj  ,jk+1) ) * 0.5_wp',
        47],
    # --- round 179: causal inputs before the returned slope boundary ---
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:156-180': [
        ('! nmln calculation in zdfmxl is only on internal points', 1),
        'r1_hmlw(ji,jj) = 1._wp / MAX( hmlp(ji,jj) - ((gdepw_1d(mikt(ji,jj)) ) *(1._wp+r3t(ji,jj,Kmm))), 10._wp )',
        25],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:183-213': [
        ('iikm1 = 1   ;   iik = 2', 1),
        ('END DO   ;   END DO', 8), 31],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:262-268': [
        'uslp(ji,jj,jk) = z1_16',
        '&                   * ( umask(ji,jj  ,jk) + umask(ji,jj  ,jk+1) ) * 0.5_wp',
        7],
    # --- round 180: developed native-slope causal walk ---
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-180': [
        'IF( l_ldfslp ) THEN', ('ENDIF', 7), 8],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:187-229': [
        ('! nmln calculation in zdfmxl is only on internal points', 1),
        ('END DO   ;   END DO', 5), 43],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:231-261': [
        'iikm1 = 1   ;   iik = 2',
        ('END DO   ;   END DO', 8), 31],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:270-361': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4),
        ('END DO   ;   END DO', 10), 92],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfphy.f90:326-330': [
        '! Update top/bottom drag',
        'CALL zdf_mxl( kt, Kmm )                        !* mixed layer depth, and level',
        5],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124': [
        '! w-level of the mixing and mixed layers',
        ('END DO   ;   END DO', 3), 16],
    # --- round 181: developed mixed-layer carried-N2 causal walk ---
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:159-168': [
        'CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )',
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90:1609-1619': [
        'DO jk =  2,  jpkm1',
        ('END DO   ;   END DO   ;   END DO', 15), 11],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/domzgr.f90:382-386': [
        '! deepest/shallowest W level Above/Below ~10m',
        'nla10 = nlb10 - 1', 5],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:167': (
        'CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1),
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:215-250': [
        ('IF( jk == 1 ) THEN', 1),
        ('END DO   ;   END DO', 5), 36],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:272-299': [
        ('DO jj = ntsj-( 0+1), ntej+(  0 ) ; DO ji = ntsi-( 0+1), ntei+(  0 )', 1),
        ('END DO   ;   END DO', 7), 28],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:311-344': [
        ('IF( jk ==1 )   zfw', 1),
        ('ENDIF', 5), 34],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:346-367': [
        ('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1),
        ('&                 * r1_e1e2t(ji,jj) / (e3t_3d', 2), 22],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:403-419': [
        ('IF( lr177_write ) THEN', 2),
        'WRITE(r177_unit) e3w_1d', 17],
    # --- round 131: completed-step boundary and daily-reset state schema ---
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:222-226': [
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:249-260': [
        'CALL dia_wri   ( kstp,      Nbb )',
        'CALL rst_write    ( kstp, Nbb, Nnn, Naa )', 12],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90:176-184': [
        "CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )",
        "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )",
        9],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90:974-980': [
        ('IF( nn_bt_flt == 3 ) THEN', 2),
        "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'     ,    vb_e(:,:) )",
        7],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:895-901': [
        "ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN",
        "CALL iom_rstput( kt, nitrst, numrow, 'dissl', dissl )", 7],
    # --- round 127: compiled N2 inputs, assembly, and acquired state rows ---
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:159-168': [
        'CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )',
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1259-1308': [
        ('CASE( np_teos10, np_eos80 )', 7),
        'pab(ji,jj,jk,jp_sal) = zn / zs * r1_rho0 * ztm', 50],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1609-1618': [
        'DO jk =  2,  jpkm1',
        '/ (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm))) * wmask(ji,jj,jk)', 10],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/domqco.f90:208': (
        'pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:132-140': [
        'll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )',
        'IF( jpts /= 2 )', 9],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:217-221': [
        '!-- the tracer fields as tra_zdf RECEIVED them --!',
        'WRITE(il2_unit) pts(:,:,:,jp_sal,Kbb)', 5],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:332-337': [
        "WRITE(il2_unit) 'r3t_Kbb",
        'WRITE(il2_unit) r3t(:,:,Kaa)', 6],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:124-128': [
        '&nameos', 'ln_seos     = .false.', 5],
    # --- round 126: compiled vertical operator and closure response ---
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3_stg.f90:937-944': [
        'IF( ln_zdfmfc  )   CALL tra_mfc',
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 8],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-480': [
        "IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN",
        'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )',
        63],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-469': [
        'zzwi = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk  ) *(1._wp+r3t(ji,jj,Kmm)))',
        'zzws = - p2dt * zwt(ji,jk+1) / (e3w_1d(jk+1) *(1._wp+r3t(ji,jj,Kmm)))',
        2],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474': [
        'zzwi = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk  ) *(1._wp+r3t(ji,jj,Kmm)))',
        'zws(ji,jk) = zzws - p2dt *   MAX( wi(ji,jj,jk+1) , 0._wp )',
        7],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:470': (
        'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk)))',
        1),
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582': [
        '!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 56],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567': [
        '!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1',
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)',
        19],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-552': [
        '!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1',
        ('END DO', 23), 4],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:562-567': [
        ('DO jk =    2,  jpkm1,  1', 2),
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)',
        6],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:577-582': [
        '!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:136-150': [
        "INQUIRE( FILE='round172_e3t.arm', EXIST=ll_r172_e3t )",
        "CALL ctl_stop( 'tra_zdf: cannot read Round-172 solve-input frame' )",
        15],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:490-504': [
        'IF( ln_zad_Aimp ) THEN',
        'zws(ji,jk) = zzws - p2dt *   MAX( wi(ji,jj,jk+1) , 0._wp )',
        15],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:561-565': [
        '!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1',
        'zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)',
        5],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:583-609': [
        '!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1',
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)',
        27],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:619-624': [
        '!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359': [
        'SELECT CASE ( nzdf_phy )',
        'IF( ln_zdfevd )   CALL zdf_evd( kt, Kmm, Krhs, avm, avt )', 26],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692': [
        '!* vertical eddy viscosity & diffivity at w-points',
        'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk)', 12],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109': [
        ('IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )', 1),
        '&  p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 2],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:205-213': [
        '&namzdf        !   vertical physics',
        'rn_avt0     =   1.2e-5', 9],
    # --- round 124: admitted process writer and compiled process order ---
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830': [
        'lr123_write = lwp .AND. .NOT.ln_tile .AND. kstg == 3',
        'WRITE(r123_unit) r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa)', 13],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3.f90:188-201': [
        '!  RK3 : single first external mode computation',
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stp2d.f90:291-298': [
        '!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)',
        ('CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 1),
        8],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/dynspg_ts.f90:797-827': [
        '! Finalize sums:',
        'pssh (:,:,Kaa) = ssha_e(:,:)', 31],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:150-180': [
        'ssha(:,:) = ssh (:,:,Kaa)',
        'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 31],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:223-255': [
        ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 1),
        ('r3t(:,:,Kaa) = r3ta(:,:)', 3), 33],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/domqco.f90:237-257': [
        ('SUBROUTINE dom_qco_r3c_RK3( pssh, pr3t, pr3u, pr3v, pr3f )', 1),
        ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869': [
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )',
        ('IF( lr123_write )   WRITE(r123_unit) ts(:,:,:,jp_tem,Krhs)', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952': [
        'IF( ln_traqsr  ) THEN',
        ('IF( lr123_write )   WRITE(r123_unit) ts(:,:,:,jp_tem,Krhs)', 4), 23],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970': [
        '!== TRA time integration + ZDF',
        "CALL ctl_stop( 'stp_RK3_stg: cannot close round-123 process dump' )",
        7],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560': [
        '!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1',
        '& + p2dt * (e3t_3d(ji,jj,jk)', 16],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578': [
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk)',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 16],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:849-876': [
        '!==  compute the beta term  ==!',
        ('IF( zdo /=  zbig .AND. zneg /= 0._wp ) THEN', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:905-931': [
        'zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji+1,jj,ik) ),',
        ('pcc(ji,jj,jk) = pcc(ji,jj,jk) * zcoef', 2), 27],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv.f90:358-364': [
        ('SELECT CASE ( nadv )', 1),
        'CALL tra_adv_fct ( kt, nit000', 7],
    'round123/oracle_process_budget/ocean.output:756-764': [
        'Namelist namtra_adv : chose a advection scheme for tracers',
        'implicit   optimized(1)/accurate(2)    nn_fct_imp =            1', 9],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:473-474': [
        'Set the minimum value of tke',
        'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin )', 2],
    # --- round 122: per-step entry record and full-year restart semantics ---
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:89-99': [
        'IF( lwp .AND. kstp >= nit000 .AND. kstp <= nit000 + 59 ) THEN',
        "WRITE(numout,*) 'LANE1_STEP_ENTRY_DUMP '", 11],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:215-222': [
        'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )',
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 8],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:260':
        'CALL rst_write    ( kstp, Nbb, Nnn, Naa )',
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/restart.f90:176-180': [
        "CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )",
        "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )",
        5],
    # --- round 129: compiled infinitesimal-IC ensemble selector and write ---
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90:101-105': [
        'IF( nn_pert_seed /= 0 ) THEN',
        "IF(lwp) WRITE(numout,*) 'TINY PERTURBATION SEED = ', nn_pert_seed",
        5],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:42':
        'INTEGER, PUBLIC ::   nn_pert_seed = 0',
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:118-122': [
        'NAMELIST/namusr_def/ nn_GYRE, ln_bench, jpkglo, nn_pert_seed',
        'IF(lwm)   WRITE( numond, namusr_def )', 5],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:137-145': [
        "WRITE(numout,*) 'usr_def_nam  : read the user defined namelist (namusr_def) in namelist_cfg'",
        "WRITE(numout,*) '      from-rest ensemble perturbation seed        nn_pert_seed = ', nn_pert_seed",
        9],
    # --- round 123: stage-3 process order and implicit tracer content ---
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:812,843,849': [
        'ts(:,:,:,jn,Krhs) = 0._wp',
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )',
        'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )',
        3],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:915,930,944': [
        'CALL tra_qsr( kstp, Kmm, ts, Krhs )',
        'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )',
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 3],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:902,933,934,935,937,939,953': [
        'IF( ln_bdy     )   CALL bdy_tra_dmp',
        'IF( ln_trabbc  )   CALL tra_bbc',
        'IF( ln_trabbl  )   CALL tra_bbl',
        'IF( ln_tradmp  )   CALL tra_dmp',
        'IF( ln_zdfmfc  )   CALL tra_mfc',
        'CALL tra_osm( kstp,      Kmm, ts, Krhs )',
        'IF( ln_zdfnpc  )   CALL tra_npc', 7],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:546-560': [
        'pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1)',
        '& + p2dt * (e3t_3d(ji,jj,jk)', 15],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:563-578': [
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk)',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 16],
    # --- round 125: reused tra_zdf vertical-decomposition record ---
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:132-146': [
        'll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )',
        'zl2_srh(:,:,:) = pts(:,:,:,jp_sal,Krhs)', 15],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:160-335': [
        '!                    !--- round-35 instrument: the record ---!',
        'CLOSE(il2_unit)', 176],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:213-333': [
        '!              !-- the tracer fields as tra_zdf RECEIVED them --!',
        'WRITE(il2_unit) r3t(:,:,Kaa)', 121],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:414-450': [
        "IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN",
        'zwt(:,1) = 0._wp', 37],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:462-476': [
        'IF( ln_zad_Aimp ) THEN',
        'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )',
        15],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:523-527': [
        'DO ji = ntsi-( 0), ntei+( 0 )          !* 1st recurrence:',
        'zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)',
        5],
    # --- round 108: configured TEOS-10 producer and first bn2 statement ---
    'GYRE_OMIP_L2_P3_SM_R101TKEW/EXP00/namelist_cfg:126-128': [
        'ln_teos10   = .true.', 'ln_seos     = .false.', 3],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1257-1310': [
        ('SELECT CASE ( neos )', 5),
        ('END DO   ;   END DO   ;   END DO', 13), 54],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1610-1611': [
        'zrw =   ( ((gdepw_1d(jk  ) )',
        '&  / ( ((gdept_1d(jk-1) )', 2],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1613':
        'zaw = pab(ji,jj,jk,jp_tem)',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1614':
        'zbw = pab(ji,jj,jk,jp_sal)',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616':
        'pn2(ji,jj,jk) = grav',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1617':
        '- zbw * ( pts(ji,jj,jk-1,jp_sal)',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616-1617': [
        'pn2(ji,jj,jk) = grav', '- zbw * ( pts(ji,jj,jk-1,jp_sal)', 2],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616-1618': [
        'pn2(ji,jj,jk) = grav',
        '/ (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))', 3],
    # --- round 107: upstream producer of the non-bit TKE rn2 operand ---
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90:159-168': [
        'CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )',
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1609-1618': [
        'DO jk =  2,  jpkm1',
        '&            / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm))) * wmask(ji,jj,jk)',
        10],
    # --- round 106: compiled TKE RHS scalar, matrix predecessors, and RHS ---
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:261': (
        'zfact3  = 0.5_wp         * rn_ediss', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-436': [
        'zd_up(ji,jk) = zzd_up',
        'zdiag(ji,jk) = 1._wp - zzd_lw - zzd_up', 3],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:439-442': [
        ('en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt', 2),
        '&                                  ) * wmask(ji,jj,jk)', 4],
    # --- round 105: split shear/solver routes and the surviving JIT boundary ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3_stg.f90:240-256': [
        'CASE ( 3 )           !==  Stage 3  ==!   Kbb = N   ;   Kmm = N+1/2   ;   Kaa = N+1',
        'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 17],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/trazdf.f90:461-477': [
        '! Diagonal, lower (i), upper (s)',
        ('END DO   ;   END DO', 10), 17],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-442': [
        'zd_up(ji,jk) = zzd_up',
        '&                                  ) * wmask(ji,jj,jk)', 9],
    'DINO/BLD/ppsrc/nemo/stpmlf.f90:187-193': [
        'CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )',
        'CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )', 7],
    'DINO/BLD/ppsrc/nemo/zdfphy.f90:316-319': [
        ('IF( l_zdfsh2 ) THEN', 1),
        '&                      sh2    )     ! ==>> out : shear production', 4],
    'DINO/BLD/ppsrc/nemo/trazdf.f90:219-235': [
        '! Diagonal, lower (i), upper (s)',
        ('END DO   ;   END DO', 9), 17],
    # --- round 182: DINO raw geometry and the complete live-Kmm route ---
    'DINO/BLD/ppsrc/nemo/usrdef_zgr.f90:123-135': [
        'IF( ld_zco ) THEN',
        'CALL zgr_msk_top_bot( pdept_1d, zbathy, k_top, k_bot )', 13],
    'DINO/BLD/ppsrc/nemo/zgr_lib.f90:199-209': [
        ('!                       !==  t- and w- scale factors from depth  ==!', 3),
        '&                   pe3uw, pe3vw        )', 11],
    'DINO/BLD/ppsrc/nemo/domain.f90:194-216': [
        'ht_0(:,:) = 0._wp  ! Reference ocean thickness',
        'r1_hf_0(:,:) = ssfmask(:,:) / ( hf_0(:,:) + 1._wp -  ssfmask(:,:) )',
        23],
    'DINO/BLD/ppsrc/nemo/domqco.f90:186-207': [
        'SUBROUTINE dom_qco_r3c( pssh, pr3t, pr3u, pr3v, pr3f )',
        ('END DO   ;   END DO', 1), 22],
    'DINO/BLD/ppsrc/nemo/eosbn2.f90:1507-1517': [
        'DO jk =  2,  jpkm1',
        ('END DO   ;   END DO   ;   END DO', 15), 11],
    'DINO/BLD/ppsrc/nemo/stpmlf.f90:197-216': [
        'IF( l_ldfslp ) THEN',
        'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nnn )', 20],
    'DINO/BLD/ppsrc/nemo/ldfslp.f90:139-177': [
        'INTEGER , INTENT(in)                   ::   kt',
        'zhmlpt(ji,jj) = ((gdept_3d(ji,jj,nmln(ji,jj)-1)', 39],
    'DINO/BLD/ppsrc/nemo/ldfslp.f90:215-216': [
        ('DO jj = ntsj-( 1), ntej+(  1 )', 3),
        'r1_hmlw(ji,jj) = 1._wp / MAX(', 2],
    'DINO/BLD/ppsrc/nemo/ldfslp.f90:320-345': [
        ('DO jj = ntsj-( 1), ntej+(  1 )', 5),
        'zck = ( ((gdepw_3d(ji,jj,jk)', 26],
    # --- round 104: the shear-production routine and its operand builders ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:167-168': [
        '!!st                         CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )',
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:319-320': [
        'CALL zdf_sh2( Kbb, Kmm, avm_k,',
        '&                      sh2    )     ! ==>> out : shear production',
        2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83': [
        'DO jk = 2, jpkm1'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:97-109': [
        'ELSE', ('END DO   ;   END DO', 2), 13],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:99-103': [
        'zsh2u(ji,jj) = ( p_avm(ji+1,jj,jk) + p_avm(ji,jj,jk) ) &',
        '&         * wumask(ji,jj,jk)', 5],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:102': [
        '/ ( (e3w_1d(jk  ) *(1._wp+r3u(ji,jj,Kmm))) '
        '* (e3w_1d(jk) *(1._wp+r3u(ji,jj,Kbb))) )'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:104-108': [
        'zsh2v(ji,jj) = ( p_avm(ji,jj+1,jk) + p_avm(ji,jj,jk) ) &',
        '&         * wvmask(ji,jj,jk)', 5],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:107': [
        '/ ( (e3w_1d(jk  ) *(1._wp+r3v(ji,jj,Kmm))) '
        '* (e3w_1d(jk) *(1._wp+r3v(ji,jj,Kbb))) )'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:112-113': [
        'p_sh2(ji,jj,jk) = 0.25 * (',
        '&                       + ( zsh2v(ji,jj-1) + zsh2v(ji,jj) ) '
        '* ( 2. - vmask(ji,jj-1,jk) * vmask(ji,jj,jk) )   )', 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:116-119': [
        'set p_sh2 to 0 at the surface and bottom for output purpose',
        ('END DO   ;   END DO', 4), 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domqco.f90:266-267': [
        ('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2),
        ('&                    + e1e2t(ji+1,jj) * pssh(ji+1,jj)  ) '
         '* r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)', 2), 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domhgr.f90:170': [
        'r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/dommsk.f90:237-242': [
        'wumask(:,:,1) = umask(:,:,1)',
        'wvmask(:,:,jk) = vmask(:,:,jk) * vmask(:,:,jk-1)', 6],
    # --- round 100: admitted same-call ratio and production discriminator ---
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:140-150': [
        ('SELECT CASE( kstg )', 1),
        'ssha(:,:) = ssh (:,:,Kaa)', 11],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:150-192': [
        'ssha(:,:) = ssh (:,:,Kaa)',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:666-674': [
        ('SELECT CASE( kstg )', 3),
        'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )',
        9],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:283-291': [
        ('IF( lwp .AND. kt == nit000 .AND. PRESENT(k_ind) ) THEN', 1),
        'CLOSE(l99_unit)', 9],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:305-310': [
        ("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2),
        ('+ r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2),
        6],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/traadv.f90:266-280': [
        'IF( ll_Fw ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww', 15],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/domqco.f90:256-258': [
        ('DO jj = ntsj-( nn_hls)', 4), ('END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/dynzdf.f90:166-169': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN',
        'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kbb) + rDt * pvv(ji,jj,jk,Krhs) )',
        4],
    # --- round 99: admitted direct-W layout and paired stage program ---
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/oce.f90:99-104': [
        ('ALLOCATE( uu', 1), '&      rhd  (jpi,jpj,jpk)', 6],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:258-260': [
        ('INTEGER  ::   ji, jj, jk', 3),
        ('REAL(wp), DIMENSION(ntsi-(1):ntei+(1)', 1), 3],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/mppini.f90:1501-1516': [
        'Nis0 =   1+nn_hls', 'ntej = Nje0', 16],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:283-291': [
        ('IF( lwp .AND. kt == nit000 .AND. PRESENT(k_ind) ) THEN', 1),
        'WRITE(l98_unit) e3t_3d, r1_Dt', 9],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:305-317': [
        ("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2),
        'CLOSE(l98_unit)', 13],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:140-150': [
        ('SELECT CASE( kstg )', 1),
        'ssha(:,:) = ssh (:,:,Kaa)', 11],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:150-192': [
        'ssha(:,:) = ssh (:,:,Kaa)',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:198-202': [
        'CASE ( 2 )           !==  Stage 2',
        ('r1_Dt = 1._wp / rDt', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:242-246': [
        'CASE ( 3 )           !==  Stage 3',
        ('r1_Dt = 1._wp / rDt', 3), 5],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domqco.f90:256-258': [
        ('DO jj = ntsj-( nn_hls)', 4), ('END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domain.f90:198-212': [
        'ht_0(:,:) = ht_0(:,:) + e3t_3d(:,:,jk) * tmask(:,:,jk)',
        'r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:)', 15],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90:132-139': [
        'CASE ( np_transport )', ('END DO   ;   END DO   ;   END DO', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90:152-154': [
        ('DO jk =  1,  jpkm1', 4),
        ('END DO   ;   END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/traadv.f90:266-280': [
        'IF( ll_Fw ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww', 15],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176': [
        '!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 36],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:202-213': [
        '!*  vertical averaging  *!',
        ('END DO   ;   END DO', 3), 12],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90:188-202': [
        '!  RK3 : single first external mode computation',
        'IF( kstp == nit000 )   CALL l1_dump_stage( kstp, 1, Naa )', 15],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:326-347': [
        '!- vertical velocity and transport (ww,wi,zFw) -!',
        ('END DO   ;   END DO   ;   END DO', 3), 22],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:367-373': [
        ('SELECT CASE( kstg )', 2),
        'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv(', 7],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:668-674': [
        ('CASE ( 1 , 2 )    !==  Stage 1 & 2', 1),
        'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )',
        7],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:734-759': [
        '!==  All stages: correct the barotropic component ==!',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)',
        26],
    # --- parallel held candidate: source-ordered GYRE TKE K_H walk ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-201': [
        'IF( nn_pdl == 1 ) ALLOCATE( z_pdlr',
        'IF( nn_pdl == 1 ) DEALLOCATE( z_pdlr )', 12],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:589,612-620': [
        'IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )',
        ('IF( ln_mxl0 ) THEN', 1), ('END DO', 30), 10],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:627-630': [
        ('DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)', 10),
        ('END DO   ;   END DO', 14), 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:634-683': [
        'zmxld(:,1) = zmxlm(:,1)   ! surface set to the minimum value',
        ('END DO   ;   END DO', 21), 50],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:669-683': [
        'CASE ( 3 )           ! lup and ldown, |dk[xml]| bounded by e3t :',
        ('END DO   ;   END DO', 21), 15],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:690-695': [
        'DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)   !* vertical eddy viscosity & diffivity at w-points',
        'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:699-702': [
        'IF( nn_pdl == 1 ) THEN          !* Prandtl number case: update avt',
        ('END DO   ;   END DO', 23), 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:691': (
        'zsqen = SQRT( en(ji,jj,jk) )', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:692': (
        'zav   = rn_ediff * zmxlm(ji,jk) * zsqen', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:693': (
        'p_avm(ji,jj,jk) = MAX( zav,                  avmb(jk) )', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:694': (
        'p_avt(ji,jj,jk) = MAX( zav, avtb_2d(ji,jj) * avtb(jk) )', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:695': (
        'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:674-676': [
        'DO jk =  jpkm1,  2,  -1  ; DO ji = ntsi-( 0), ntei+(  0)   ! from the bottom to the surface : ldown',
        '&    MIN( zmxlm(ji,jk+1) + (e3t_3d(ji,jj,jk+1)', 3],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-413,699-702': [
        'IF( nn_pdl == 1 ) THEN          !* Prandtl number = F( Ri )',
        ('END DO   ;   END DO', 4),
        'IF( nn_pdl == 1 ) THEN          !* Prandtl number case: update avt',
        ('END DO   ;   END DO', 23), 24],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:154-168': [
        '! Ocean physics update',
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 15],
    # --- round 98: kt=1 stage-1 tracer-W compiled-order walk ---
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:331-339': [
        ('IF( ln_dynadv_vec ) THEN', 1), ('ENDIF', 8), 9],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:132-139': [
        'CASE ( np_transport )', ('END DO   ;   END DO   ;   END DO', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:152-154': [
        ('DO jk =  1,  jpkm1', 4),
        ('END DO   ;   END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/sshwzv.f90:293-300': [
        ("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2),
        ('ENDIF', 11), 8],
    'GYRE_OMIP_L2_P3_SM_R21W/BLD/ppsrc/nemo/traadv.f90:267-295': [
        'IF( ll_Fw ) THEN', ('ENDIF', 7), 29],
    # --- round 92: acquired correction record and live operand split ---
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:668-687': [
        ('SELECT CASE( kstg )', 3),
        ('ENDIF', 21), 20],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:728-765': [
        'DYN time integration + ZDF',
        "CALL r46_state( 'post_baro', uu, vv, Kaa )", 38],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/dynspg_ts.f90:763-804': [
        '!* Sum over whole bt loop (except in weight average)',
        'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 42],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/domain.f90:195-215': [
        'hv_0(:,:) = 0._wp',
        'r1_hf_0(:,:) = ssfmask(:,:) / ( hf_0(:,:) + 1._wp -  ssfmask(:,:) )', 21],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90:35-51': [
        'SUBROUTINE put2(name, value)',
        'END SUBROUTINE put3', 17],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90:69-103': [
        'SUBROUTINE r90_baro_begin(kt,kstg,Kaa,puu,pvv,pzub,pzvb)',
        'END SUBROUTINE r90_baro_finish', 35],
    # --- round 89: source-rounded RK3 assignment and ordered stage walk ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:140-148': [
        ('SELECT CASE( kstg )', 1),
        ('r1_Dt = 1._wp / rDt', 1), 9],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:140-170': [
        ('SELECT CASE( kstg )', 1),
        ('END SELECT', 1), 31],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:198-202': [
        'CASE ( 2 )           !==  Stage 2',
        ('r1_Dt = 1._wp / rDt', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:242-246': [
        'CASE ( 3 )           !==  Stage 3',
        ('r1_Dt = 1._wp / rDt', 3), 5],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:433-482': [
        'CALL    eos    (        ts, Kmm, rhd, rhop )',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 50],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:692-714': [
        ('CASE ( 3 )        !==  Stage 3', 1),
        "CALL r46_rhs( 'after_ldf', uu, vv, Krhs )", 23],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:728-759': [
        "CALL r46_rhs( 'pre_zdf_rhs', uu, vv, Krhs )",
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 32],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzdf.f90:163-170': [
        'RHS : time-stepping of all trends but the implicit one',
        ('END DO   ;   END DO', 1), 8],
    # --- round 88: carried pre-solve RK3 Kaa SSH lifecycle ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:188-201': [
        '!  RK3 : single first external mode computation',
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226': [
        'IF ( .NOT. l_perpetual_ts ) THEN',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 7],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:671-674': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN   !* applied on velocity',
        'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )', 4],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:176-184': [
        "CALL iom_rstput( kt, nitrst, numrow, 'sshn'",
        "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha'", 9],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:354-370': [
        '!*  RK3: Read ssh at Kbb',
        'ssh(:,:,Kaa) = ssh(:,:,Kbb)               ! no ssh variation in ww computation', 17],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:395-405': [
        'ELSE                                  !* no wet and dry',
        'ssh(:,:,Kaa) = ssh(:,:,Kbb)           !*  set ssh at Kaa (for AGRIF)', 11],
    # --- round 87: stage-1 WZV source order and Kaa scratch boundary ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:301-311': [
        '!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)',
        'DEALLOCATE( sshe_rhs , Ue_rhs , Ve_rhs , CdU_u , CdU_v )', 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/divhor.f90:123-153': [
        'SELECT CASE ( ik_ind )',
        ('pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk)', 2), 31],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:326-339': [
        '!              !- vertical velocity and transport (ww,wi,zFw) -!',
        ('ENDIF', 8), 14],
    # --- round 84: cumulative stage-1 HPG/LDF/ZAD walk ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynhpg.f90:378-435': [
        ('zcoef0 = - grav * 0.5_wp', 2), ('END DO', 5), 58],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf.f90:81-90': [
        'SELECT CASE ( nldf_dyn )',
        'CALL dynldf_lev_blp( kt, Kbb, Kmm, puu, pvv, Krhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 )', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1),
        20],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138': [
        'zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp',
        ('END DO   ;   END DO', 2), 37],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300': [
        ("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2),
        ('ENDIF', 11), 8],
    'round64/oracle_krhs_split/ocean.output:682-708': [
        'ldf_dyn : lateral momentum physics',
        'eddy viscosity. = constant', 27],
    # --- round 83: the admitted cumulative RHS and slow-forcing producer ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176': [
        '!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 36],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:187-199': [
        '! Lane-2 round-16 slow-forcing operand instrument.  WRITE-only:',
        'WRITE(l2_slow_unit) e3u_3d, uu(:,:,:,Krhs), umask, e3v_3d, vv(:,:,:,Krhs), vmask',
        13],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:202-207': [
        '!*  vertical averaging  *!',
        'Ve_rhs(ji,jj) = SUM( e3v_3d', 6],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:225-227': [
        '!* baroclinic drag forcing *!',
        'WRITE(l2_slow_unit) Ue_rhs, Ve_rhs, CdU_u, CdU_v', 3],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:229-235': [
        '!* wind forcing *!',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV', 7],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:291-327': [
        '! set values computed in RK3_ssh',
        ('END DO   ;   END DO', 1), 37],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1467-1526': [
        'SUBROUTINE dyn_drg_init( Kbb, Kmm, puu, pvv, puu_b ,pvv_b, pu_RHSi, pv_RHSi, pCdU_u, pCdU_v )',
        ('pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0', 1), 60],
    # --- round 82: kt=2 drag coefficient, entry inverse, and forcing owner ---
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-235': [
        'ELSE                                            !==  standard Cd  ==!',
        'pCdU(ji,jj) = - pCd0(ji,jj) * SQRT', 7],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1497-1500': [
        'ELSE                          ! bottom friction only',
        'pCdU_v(ji,jj) = r1_2*( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) )', 4],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:363-375': [
        'hu_e  (:,:) =    (hu_0(:,:) *(1._wp+r3u(:,:,Kmm)))',
        'hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kbb)))', 13],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:703-706': [
        'IF ( .NOT. ll_wd ) THEN ! Revert to explicit for bit comparison tests in non wad runs',
        'zv_trd(ji,jj) = zv_trd(ji,jj) + zCdU_v(ji,jj) * vn_e(ji,jj) * hvr_e(ji,jj)', 4],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:202-207': [
        '!                             !*  vertical averaging  *!',
        'Ve_rhs(ji,jj) = SUM( e3v_3d', 6],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:225-227': [
        '!                             !* baroclinic drag forcing *!',
        'IF( lwp .AND. kt == nit000 )   WRITE(l2_slow_unit) Ue_rhs, Ve_rhs, CdU_u, CdU_v', 3],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:229-235': [
        '!                             !* wind forcing *!',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV', 7],
    # --- round 81: the complete kt=2 split-explicit operand order ---
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:288-324': [
        '!                          ! set values computed in RK3_ssh',
        ('END DO   ;   END DO', 1), 37],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-505': [
        'IF ((jn<3).AND.ll_init) THEN', ('END DO   ;   END DO', 3), 25],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:512-518': [
        'IF( .NOT.lk_linssh ) THEN                        !* Update ocean depth',
        'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)',
        7],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:549-583': [
        '!                    !==  after SSH  ==!   (jn+1)',
        ('END DO   ;   END DO', 9), 35],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:627-642': [
        '! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2',
        ('END DO   ;   END DO', 14), 16],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:644-666': [
        '! Add Coriolis trend:', ('ENDIF', 29), 23],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:678-702': [
        '! Set next velocities:', ('END DO   ;   END DO', 17), 25],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:744-795': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth',
        'sshn_e (:,:) = ssha_e(:,:)', 52],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:127-214': [
        'ALLOCATE( sshe_rhs(jpi,jpj)', ('END SELECT', 2), 88],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:220-311': [
        ('!              !=====================================!', 1),
        'DEALLOCATE( sshe_rhs , Ue_rhs , Ve_rhs , CdU_u , CdU_v )', 92],
    # --- round 78: midpoint operands and persistent absolute histories ---
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378': [
        'IF( ll_init )THEN', 'pssh  (:,:,Kaa) = 0._wp', 40],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-509': [
        'IF ((jn<3).AND.ll_init) THEN',
        '& ubb_e(ntsi:ntei,ntsj:ntej), ua_e(ntsi:ntei,ntsj:ntej)', 29],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:783-795': [
        '!* Swap', 'sshn_e (:,:) = ssha_e(:,:)', 13],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:991-1018': [
        "IF( TRIM(cdrw) == 'READ' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'", 28],
    # --- round 77: valid midpoint record followed by an admission-parser stop ---
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:442-452': [
        '! Round-73 WRITE-only extension of the existing transport-mean stream.',
        ('WRITE(l2_adv_unit) r1_wgt2s, wgtbtp2(1:icycle), r1_e2u, r1_e1v', 2),
        11],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:454-462': [
        '! Round-76 WRITE-only source-order record for the kt=2 U midpoint.',
        '& ntsi, ntei, ntsj, ntej', 9],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:501-509': [
        'ua_e(ji,jj) = za1 * un_e(ji,jj) + za2 * ub_e(ji,jj) + za3 * ubb_e(ji,jj)',
        '& ubb_e(ntsi:ntei,ntsj:ntej), ua_e(ntsi:ntei,ntsj:ntej)', 9],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:588-606': [
        '! Sum over sub-time-steps to compute advective velocities',
        'WRITE(l2_adv_unit) zhup2_e, zhvp2_e, un_adv, vn_adv', 19],
    # --- round 73: external mean already differs at the stage-1 boundary ---
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:287-291': [
        ('CASE ( np_LIN, np_HYB )', 2),
        ('END DO   ;   END DO', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:295': (
        'zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:296': (
        'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:559-572': [
        '! Sum over sub-time-steps to compute advective velocities',
        '& l2_u_mid, l2_v_mid, zhup2_e, zhvp2_e, un_adv, vn_adv', 14],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:797-817': [
        '! Finalize sums:',
        "WRITE(numout,*) 'LANE2_BT_ADVMEAN_DUMP '", 21],
    # --- round 74: automatic external-step resolution in the R72 producer ---
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:1029-1045': [
        'IF( ln_bt_auto )   nn_e = CEILING( rn_Dt / rn_bt_cmax * zcmax)',
        'zcmax = zcmax * rDt_e', 17],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:1064-1067': [
        "WRITE(numout,*) '     Barotropic time steps => in seconds",
        "ELSE                  ; WRITE(numout,*) '        set      "
        "(ln_bt_auto=F) with the namelist parameter nn_e '", 4],
    # --- round 75: the failed R74 writer omitted two unallocated arrays ---
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:390-393': [
        ('IF( lwp .AND. kt == nit000 ) THEN', 2),
        '&      l2_u_cor(jpi,jpj), l2_v_cor(jpi,jpj) )', 4],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:442-450': [
        'IF( lwp .AND. kt == nit000 + 1 ) THEN',
        ('WRITE(l2_adv_unit) r1_wgt2s, wgtbtp2(1:icycle), r1_e2u, r1_e1v', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:503-506': [
        ('IF( lwp .AND. kt == nit000 ) THEN', 3),
        'l2_v_mid  (:,:) = va_e(:,:)', 4],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:572-579': [
        'za2 = wgtbtp2(jn)',
        'vn_adv(ji,jj) = vn_adv(ji,jj) + za2 * zhV(ji,jj) * r1_e1v(ji,jj)', 8],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:581-584': [
        ('IF( lwp .AND. kt <= nit000 + 1 ) THEN', 2),
        '& l2_u_mid, l2_v_mid, zhup2_e, zhvp2_e, un_adv, vn_adv', 4],
    # --- round 76: midpoint producer and downstream transport statements ---
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:484-493': [
        '!* Extrapolate barotropic velocities at mid-step (jn+1/2)',
        ('END DO   ;   END DO', 3), 10],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:538-544': [
        '! resulting flux at mid-step (not over the full domain)',
        ('END DO   ;   END DO', 8), 7],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:571-580': [
        '! Sum over sub-time-steps to compute advective velocities',
        ('END DO   ;   END DO', 10), 10],
    # --- round 72: stage-1 transport is earlier than the exact RK replay ---
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:289-290': [
        'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0',
        'zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0', 2],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:295': (
        'zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:296': (
        'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:843-853': [
        ('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) THEN', 1),
        ('WRITE(l1_unit) zFu, zFv, zFw', 3), 11],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:860,867': [
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs',
        'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs', 2],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:864-872': [
        ('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) &', 1),
        ('WRITE(l1_unit) ts(:,:,:,jp_tem,Krhs), ts(:,:,:,jp_sal,Krhs)', 3), 9],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:899-901': [
        'ts(ji,jj,jk,jn,Kaa) = (',
        '/          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:907-913': [
        ('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) THEN', 2),
        "WRITE(numout,*) 'L2_RK_TRACER_OPERAND_DUMP '", 7],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:299-321': [
        '! Round-13 WRITE-only stage-1 transport operand record.',
        "WRITE(numout,*) 'L2_RK_STAGE1_TRANSPORT_OPERAND_DUMP '", 23],
    # --- round 71: Kmm boundary and its producing stage-2 update ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:204-215': [
        ('Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs', 1),
        'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 12],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:899-901': [
        'ts(ji,jj,jk,jn,Kaa) = (',
        '/          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    # --- round 67: compiled tracer order and the held legoESM transcription ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965': [
        ('CASE ( 3 )        !==  Stage 3', 2),
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 49],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:416-443': [
        'IF( cdtype == \'TRA\' .AND. jn == jp_tem ) THEN',
        'zwt(ji,jk) = avs(ji,jj,jk) + ah_wslp2(ji,jj,jk)', 28],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:463-479': [
        '! Diagonal, lower (i), upper (s)',
        ('END DO   ;   END DO', 10), 17],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565': [
        ('DO ji = ntsi-( 0), ntei+( 0 )', 2),
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk)', 19],
    'ocean_model_latlon_cgrid.py:9427-9546': [
        'T=state_new.T.replace(data=_adv_content_T)',
        ('z_coord=z_coord, config=config, iwm_fields=iwm_fields)', 2), 120],
    'ocean_model_latlon_cgrid.py:11607-11613': [
        'K_v_cell = K_v_cell.astype(state.T.data.dtype)',
        'K_v_cell = K_v_cell + K33_iso.astype(state.T.data.dtype)', 7],
    'ocean_model_latlon_cgrid.py:12052-12081': [
        ('if do_tracers:', 2),
        'implicit_w=nemo_aimp_tracer_w, return_matrix_trace=return_tracer_solve_trace))', 30],
    # --- parallel LDF step-3 re-proof: current private arm and execution ---
    'ocean_model_latlon_cgrid.py:1480-1483': [
        '# Route the already-computed GM/Redi rate into the same stage-3 source',
        'route_gm_redi_stage3_source: bool = False', 4],
    'ocean_model_latlon_cgrid.py:8860-8868': [
        ('elif _tti == "rk3_ws":', 2),
        '_stage_source_rates[2][1] + dS_gm * active_3d,', 9],
    'nemo_testcase_recipe.py:587-674': [
        'return LatLonCGridOceanConfig.from_flat(',
        'gm_redi=None,', 88],
    # --- round 66: admitted content operands and Krhs/LDF walk ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868': [
        ('DO jn = 1, jpts', 1),
        "CALL r63_snapshot( 'after_sbc', ts, Krhs )", 42],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950': [
        ('DO jn = 1, jpts', 1),
        'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )', 124],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:548-562': [
        'pt(ji,jj,1,jn,Kaa) =',
        ('+ p2dt * (e3t_3d', 2), 15],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192': [
        ('zdit(ji,jj,ik  ) =', 1),
        ('zdkt(ji,jj,ikp1) =', 1), 10],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:230-246': [
        ('zA11 = e2_e1u(ji,jj)', 1),
        ('+ ( zdkt(ji,jj+1,ikp1)', 1), 17],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:287-305': [
        ('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1),
        ('END DO   ;   END DO', 8), 19],
    'round64/oracle_krhs_split/ocean.output:649-656': [
        ('no explicit diffusion', 1), 'iso-neutral triad operator', 8],
    'ocean_model_latlon_cgrid.py:2246-2309': [
        'h_one_third = h_k_old',
        '+ dt * h_one_half * stage_source_rates[2][1])', 64],
    # ORCA2 round 10 WIDENED this range rather than moving it: the RGB arm of
    # the stage-3 qsr seam was inserted INSIDE the cited block, so its first
    # anchor did not move while its last one shifted by 59.  Stated here
    # explicitly, with the new length, because a widening that is not declared
    # is exactly what this map exists to refuse.
    'ocean_model_latlon_cgrid.py:7199-7338': [
        '_stage3_T_rate = (',
        # 129 before round 12 wrapped the stage-3 pair in the river-runoff
        # deposit: the closing anchor now sits four lines inside the block.
        '/ jnp.maximum(_h_live_one_half, 1.0e-10),', 140],
    'ocean_model_latlon_cgrid.py:8309-8742': [
        'T_mid = state_new.T.data',
        'S_mid = S_mid + dt * dS_gm * active_3d', 434],
    'ocean_model_latlon_cgrid.py:8936-8977': [
        ('_nemo_ws_rk3_tracer_pair_step(', 3),
        'return_final_content=True,', 42],
    # --- round 65: admitted R64 Krhs/FCT/TKE walk ---
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-828': [
        ('DO jn = 1, jpts', 1), 'ts(:,:,:,jn,Krhs) = 0._wp', 2],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:860': (
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs', 1),
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:602-611': [
        ('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0)', 9),
        ('pt_up1(ji,jj,jk) = ( (e3t_3d', 3), 10],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:607': (
        'pt_rhs(ji,jj,jk) = pt_rhs(ji,jj,jk) + ztra', 2),
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-510': [
        'DO jk =  1,  jpkm1  ; DO jj = ntsj-(  2)',
        'ptFw(ji,jj,jk) = MAX( pW(ji,jj,jk)', 8],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:532-540': [
        ('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  1)', 5),
        ('pt_up1(ji,jj,jk) = ( (e3t_3d', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:570-580': [
        ('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  1)', 7),
        '&                                       +   MIN( pW(ji,jj,jk)', 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:197-201,264-269': [
        ('CASE(  2  )                   !- 2nd order centered', 1),
        ('END DO   ;   END DO   ;   END DO', 4),
        ('CASE(  2  )                   !- 2nd order centered', 2),
        ('END DO   ;   END DO   ;   END DO', 5), 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:798-933': [
        'z1_Dt = 1._wp / p2dt',
        ('pcc(ji,jj,jk) = pcc(ji,jj,jk) * zcoef', 2), 136],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:318-329': [
        'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb)',
        'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra', 12],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-506': [
        'DO jk =  1,  jpkm1  ; DO jj = ntsj-(  2)',
        ('END DO   ;   END DO   ;   END DO', 21), 4],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90:428-431': [
        ('en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt', 2),
        (') * wmask(ji,jj,jk)', 2), 4],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90:430': (
        '+ zfact3 * dissl(ji,jj,jk) * en(ji,jj,jk)', 1),
    # --- round 112: admitted R111 metric-FCT compiled-order walk ---
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:167-176': [
        'DO jn = 1, kjpt',
        'CALL fct_up1_2stp( Kbb, Kmm, Kaa', 10],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:503-533': [
        'zDt = 0.5_wp * pDt',
        'CALL r111_first_flux( ptFu, ptFv, ptFw )', 31],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:536-549': [
        '! -- middle time step tracer with upstream scheme',
        'CALL r111_midpoint( zr111, pt_up1 )', 14],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:572-608': [
        '! *** 2nd step',
        'CALL r111_average_flux( ptFu, ptFv, ptFw )', 37],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:611-623': [
        ('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0)', 9),
        'CALL r111_upstream( zr111, pt_rhs )', 13],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:322-333': [
        'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb)',
        'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra', 12],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:295-296': [
        'zFu(ji,jj,jk) = e2u(ji,jj)',
        'zFv(ji,jj,jk) = e1v(ji,jj)', 2],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:281-292': [
        ('SELECT CASE( n_baro_upd )', 4),
        ('END SELECT', 5), 12],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:52-54': [
        'INTEGER,  PUBLIC, PARAMETER ::   np_HYB = 2',
        'INTEGER  :: n_baro_upd =  np_HYB', 3],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:198-242': [
        'CASE ( 2 )           !==  Stage 2  ==!',
        'CASE ( 3 )           !==  Stage 3  ==!', 45],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:525-547': [
        "WRITE(l3_unit) 'uu_Kmm          '",
        ('WRITE(l3_unit) zl3_tmp', 3), 23],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90:266-268': [
        ('pr3u(ji,jj) = 0.5_wp *', 2),
        ('pr3v(ji,jj) = 0.5_wp *', 2), 3],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domain.f90:193-200': [
        'ht_0(:,:) = 0._wp  ! Reference ocean thickness',
        'hv_0(:,:) = hv_0(:,:) + e3v_3d(:,:,jk) * vmask(:,:,jk)', 8],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domain.f90:212-215': [
        'r1_ht_0(:,:) = ssmask (:,:) /',
        'r1_hf_0(:,:) = ssfmask(:,:) /', 4],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domhgr.f90:155-160': [
        'e1e2t (:,:) = e1t(:,:) * e2t(:,:)',
        'e1e2v (:,:) = e1v(:,:) * e2v(:,:)', 6],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domhgr.f90:170-171': [
        'r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)',
        'r1_e1e2v(:,:) = 1._wp / e1e2v(:,:)', 2],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3.f90:188-201': [
        '!  RK3 : single first external mode computation',
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:140-180': [
        ('SELECT CASE( kstg )', 1),
        'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 41],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:190-192': [
        'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 3],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:230-237': [
        ('ssh (:,:,Kaa) = r1_2 * ( ssh (:,:,Kbb) + ssha(:,:) )', 2),
        ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:242': (
        'CASE ( 3 )           !==  Stage 3  ==!', 1),
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:301-308': [
        '!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)',
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:202-207': [
        '!*  vertical averaging  *!',
        'Ve_rhs(ji,jj) = SUM( e3v_3d', 6],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:225-239': [
        '!* baroclinic drag forcing *!',
        'CLOSE(l2_slow_unit)', 15],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:326-346': [
        '!              !- vertical velocity and transport (ww,wi,zFw) -!',
        'zFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 21],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:860': (
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs', 1),
    # --- round 64: R46/R63 compiled-delta and run-horizon audit ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:21': (
        'nn_itend    =      10', 1),
    'GYRE_OMIP_L2_P3_SM_R63KRHS/EXP00/namelist_cfg:21': (
        'nn_itend    =      2', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:825-863': [
        ('DO jn = 1, jpts', 1),
        'CALL tra_sbc_RK3( kstp, Kbb, Kmm', 39],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:168-201': [
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )',
        'CALL stp_RK3_stg( 1, kstp', 34],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868': [
        ('DO jn = 1, jpts', 1),
        "CALL r63_snapshot( 'after_sbc'", 42],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:169-172': [
        '! -- Upstream fluxes',
        'output => ztFu(1,0,1,0)', 4],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/traadv_fct.f90:170-174': [
        '! -- Upstream fluxes',
        'output => ztFu(1,0,1,0)', 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:145-149': [
        'zl2_trh(:,:,:) = pts(:,:,:,jp_tem,Krhs)',
        'CALL tra_zdf_imp(', 5],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/trazdf.f90:146-151': [
        'zl2_trh(:,:,:) = pts(:,:,:,jp_tem,Krhs)',
        'CALL r63_content_finish(', 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:188-192': [
        'IF( nn_pdl == 1 ) ALLOCATE( z_pdlr',
        'CALL tke_avn(', 5],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90:190-196': [
        'IF( nn_pdl == 1 ) ALLOCATE( z_pdlr',
        'CALL tke_avn(', 7],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90:413-433': [
        'CALL r63_tke_before_row(',
        'CALL r63_tke_after_row(', 21],
    # --- round 63: compiled kt=2 stage-3 Krhs and RHS-product split ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:773-800': [
        ('IF( .NOT.ln_shuman ) THEN', 1),
        ('CALL tra_adv_trp( kstp, kstg, nit000', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:825-870': [
        ('DO jn = 1, jpts', 1),
        'IF( ln_isf )   CALL tra_isf', 46],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:886-908': [
        ('SELECT CASE( kstg )', 4),
        'L2_RK_TRACER_OPERAND_DUMP', 23],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:912-958': [
        ('CASE ( 3 )', 3),
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs', 47],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:94-111': [
        'INTEGER                                  , INTENT(in   ) ::   kt',
        ('REAL(wp), DIMENSION(ntsi-(nn_hls):ntei+(nn_hls)', 2), 18],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:164-198': [
        'DO jn = 1, kjpt',
        'ztFv(ji,jj,jk) = 0.5_wp * pV', 35],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:260-327': [
        'SELECT CASE( kn_fct_v )',
        'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra', 68],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:86-88': [
        'INTEGER                                  , INTENT(in)    :: kt',
        'REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:219-223': [
        'INTEGER                                   , INTENT(in   ) ::   Kbb, Kmm',
        ('REAL(wp), DIMENSION(ntsi-(0):ntei+(0)', 1), 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:409-427': [
        'Matrix and right hand side in en',
        ('END DO   ;   END DO', 5), 19],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdf_oce.f90:85-87': [
        ('ALLOCATE( avm', 1), 'avmb(jpk)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/oce.f90:40-42': [
        'DIMENSION(:,:,:,:,:) ::   ts',
        'DIMENSION(:,:,:)     ::   rn2b ,  rn2', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dom_oce.f90:170-180': [
        '::     e3t_3d',
        '::   r3t, r3u, r3v', 11],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc/do_loop_substitute.h90:72-89': [
        '!    A1Di(H) = Nis0-(H):Nie0+(H)',
        ('#define T2D(H) T1Di(H),T1Dj(H)', 1), 18],
    # --- round 62: tracer operand ladder on the R46 kt=2 producer ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:450': (
        'zwt(:,1) = 0._wp', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:461-477': [
        'Diagonal, lower (i), upper (s)',
        ('END DO   ;   END DO', 10), 17],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:523-528': [
        'DO ji = ntsi-( 0), ntei+( 0 )          !* 1st recurrence',
        ('END DO   ;   END DO', 13), 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:545-564': [
        'DO ji = ntsi-( 0), ntei+( 0 )             !* 2nd recurrence',
        ('END DO   ;   END DO', 16), 20],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:573-578': [
        'DO ji = ntsi-( 0), ntei+( 0 )             !* 3d recurrence',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 6],
    # --- decision 36: RK3 face-native shear on the GYRE identity card ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168': (
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 1),
    'nemo_testcase_recipe.py:437-439': [
        'tke_shear_production="nemo_face_native_now2"',
        ('tke_shear_metric_source="nemo_qco_live_face"', 2), 3],
    'ocean_model_latlon_cgrid.py:10459-10476': [
        'if shear_disc not in ("nemo_face_native", "nemo_face_native_now2"):',
        '_u_before, _v_before = _u_now, _v_now', 18],
    'ocean_model_latlon_cgrid.py:10497-10570': [
        'if metric_source == "nemo_qco_live_face":',
        'ref_v * (1.0 + r3vb[..., None]),', 74],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py:384-454': [
        ('wumask = u_mask[..., :-1] * u_mask[..., 1:]', 2),
        ('+ (zsh2v[:-1, :, :] + zsh2v[1:, :, :]) * coast_v', 2), 71],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:514-552': [
        'def _model_substitution_walk(',
        'require(jax.config.x64_enabled, "model substitution walk requires JAX fp64")',
        39],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1238-1425': [
        'def _bridge_kt2_production_entry(',
        'return state, audit', 188],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1174-1208': [
        'def _bridge_kt2_state(', ('return state', 1), 35],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:4627-4657': [
        'states = {1: _bridge_stage_context(',
        ('surface_forcing=surface))', 2), 31],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:546-563': [
        'observed_step = (header["kt"], header["stage"])',
        'f"Kmm={header[\'Kmm\']}",', 18],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1269-1273': [
        'step_entry = read_entry(year_entry_path)',
        'f"kt=2/Nbb=3, got kt={step_entry[\'kt\']}/Nbb={step_entry[\'Nbb\']}",',
        5],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:4470-4491': [
        'if production_tke_post_sweep is not None:',
        'tke_module._solve_tke_backward_euler = inject_recorded_post_sweep',
        22],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:225-239': [
        'header = struct.unpack("=13i", take(13 * 4))',
        'f"Kbb={head[\'Kbb\']}/Kmm={head[\'Kmm\']}",', 15],
    'ocean_model_latlon_cgrid.py:12312-12317': [
        'if _tke_coeff_new is not None:',
        'tke_avt=Field(data=_tke_coeff_new.K_H', 6],
    'ocean_model_latlon_cgrid.py:12517-12532': [
        '# ``step`` is the production-compiled entry point even when a caller',
        ('_nemo_stage1_zad_eta_after_override))', 1), 16],
    'state.py:593-597': [
        '# NEMO TKE-closure coefficient memory (avm_k/avt_k). These are the',
        'tke_avt: object = None', 5],
    'tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py:193-216': [
        'def test_rk3_card_steps_with_nemo_face_shear_and_live_face_metric(gate):',
        'assert all(np.all(np.isfinite(np.asarray(fields[k]))) for k in ("T", "S", "u", "v", "ssh"))',
        24],
    # --- round 61: split of the actual kt=2 carry and face-shear replay ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:317-337': [
        ('IF( l_zdfsh2 ) THEN', 1),
        'CASE( np_TKE )   ;   CALL zdf_tke( kt, Kbb, Kmm, sh2, avm_k, avt_k )',
        21],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83-114': [
        'DO jk = 2, jpkm1', ('END DO   ;   END DO', 3), 32],
    # --- round 60: accepted R59 record and source-ordered TKE walk ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-200': [
        'IF( nn_pdl == 1 ) ALLOCATE( z_pdlr',
        'CALL r54_tke_finish( p_avm, p_avt, dissl )', 11],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:277-281': [
        ('DO ji = ntsi-( 0), ntei+( 0 )', 1), 'zd_up(ji,1) = 0._wp', 5],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:292-300': [
        'IF( .NOT.ln_drg_OFF ) THEN',
        'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin )', 9],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:318-380': [
        'IF( ln_lc ) THEN',
        'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)', 63],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-433': [
        ('IF( nn_pdl == 1 ) THEN', 1),
        '&                                  ) * wmask(ji,jj,jk)', 40],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:466-483': [
        ('DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)'
         '                 ! First recurrence'),
        'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:589-701': [
        'IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )',
        'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk)', 113],
    # --- round 102: admitted R101 TKE statement-boundary walk ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:162-190': [
        'ALLOCATE(r54_zdiag(ntsi:ntei,ntsj:ntej,jpk)',
        'END SUBROUTINE r54_tke_matrix_row', 29],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:96-104': [
        'SUBROUTINE r101_rhs_row(jj,p_en)',
        'END SUBROUTINE r101_rhs_row', 9],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:116-128': [
        ('SUBROUTINE r101_tke_finish', 1),
        '& r101_post_sweep)', 13],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:196-207': [
        'CALL r101_tke_begin( kt, Kbb, Kmm )',
        'IF( nn_pdl == 1 ) DEALLOCATE( z_pdlr )', 12],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:257': (
        'zbbrau  = rn_ebb / rho0', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:285': (
        'en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:299-307': [
        'IF( .NOT.ln_drg_OFF ) THEN',
        'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin ) * ssmask(ji,jj)', 9],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:324': (
        'CALL r101_after_boundaries_row( jj, en )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:353': (
        'zWlc2(ji) = zcsd * taum(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:361-365': [
        'zpelc(ji,1) =  MAX( rn2b(ji,jj,1), 0._wp )',
        '&       MAX( rn2b(ji,jj,jk), 0._wp )', 5],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:369-388': [
        'imlc(:) = mbkt(ntsi-(0):ntei+(0),jj) + 1',
        'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)', 20],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:395': (
        'CALL r101_after_langmuir_row( jj, en )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:399-443': [
        '! Resolution of a tridiagonal linear system by a "methode de chasse"',
        ('END DO   ;   END DO', 5), 45],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:473': (
        'CALL r101_rhs_row( jj, en )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:475-495': [
        '!* Matrix inversion from level 2 (tke prescribed at level 1)',
        'CALL r101_post_sweep_row( jj, en )', 21],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354': [
        'start from turbulent closure values', 'CALL r54_zdfphy_finish', 7],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:277-281': [
        ('DO ji = ntsi-( 0), ntei+( 0 )', 1), 'zd_up(ji,1) = 0._wp', 5],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:318-380': [
        'IF( ln_lc ) THEN',
        'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)', 63],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:416-433': [
        'DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)    !* Matrix and right hand side in en',
        '&                                  ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:466-483': [
        ('DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)'
         '                 ! First recurrence'),
        'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:589-682': [
        'IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )',
        'zmxld(ji,jk) = zemlp', 94],
    'round33_lock_zdf_matrix/namelist_cfg:131': (
        'ln_zdfcst   = .true.', 1),
    'round33_overflow_zdf_matrix/namelist_cfg:128': (
        'ln_zdfcst   = .true.', 1),
    # --- round 59: R58 RHS-bound retraction and widened calibration ---
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:394-412': [
        ('IF( nn_pdl == 1 )', 3),
        'p_pdlr(ji,jj,jk) = MAX(  0.1_wp,  ri_cri / MAX( ri_cri , zri )  )', 19],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:691-701': [
        'zsqen = SQRT', 'p_avt(ji,jj,jk)   = MAX', 11],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354': [
        'start from turbulent closure values', 'CALL r54_zdfphy_finish', 7],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:466-483': [
        ('DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)'
         '                 ! First recurrence'),
        'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:589': (
        'IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )', 1),
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:601-603': [
        'initialisation of interior minimum value', 'zmxld(:,:)  = rmxl_min', 3],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:612-619': [
        ('IF( ln_mxl0 ) THEN', 1),
        'zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )', 8],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:628-629': [
        'zrn2 = MAX( rn2(ji,jj,jk), rsmall )',
        'zmxlm(ji,jk) = MAX(  rmxl_min,  SQRT( 2._wp * en(ji,jj,jk) / zrn2 )  )', 2],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:634': (
        'zmxld(:,1) = zmxlm(:,1)', 1),
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:669-682': [
        'CASE ( 3 )', 'zmxld(ji,jk) = zemlp', 14],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdf_oce.f90:85-87': [
        ('ALLOCATE( avm', 1), 'avmb(jpk)', 3],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:173-189': [
        ('SUBROUTINE r54_tke_matrix_row', 1), 'r54_pdlr(:,jj,2:jpkm1)', 17],
    'nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90:154-170': [
        ('SUBROUTINE r54_tke_matrix_row', 1), 'r54_pdlr(:,jj,2:jpkm1)', 17],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:116-199': [
        'def _calibrate_en_and_mixing', ('return counts', 2), 84],
    # --- round 58: fail-closed TKE operand acquisition retraction ---
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:106-110': [
        ('SUBROUTINE r54_tke_begin', 1),
        'REAL(wp), DIMENSION(:,:,:), INTENT(in) :: p_sh2,p_avm,p_avt,p_dissl', 5],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:150-155': [
        "CALL put3('avm_entry", "CALL put3('sh2", 6],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:128': (
        'ALLOCATE( htau', 1),
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:183-185': [
        ('REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ,jpk), '
         'INTENT(in   ) ::   p_sh2'),
        ('REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ,jpk), '
         'INTENT(inout) ::   p_avt'),
        3],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:394-412': [
        ('IF( nn_pdl == 1 )', 3),
        'p_pdlr(ji,jj,jk) = MAX(  0.1_wp,  ri_cri / MAX( ri_cri , zri )  )', 19],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:751': (
        'ri_cri   = 2._wp    / ( 2._wp + rn_ediss / rn_ediff )', 1),
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:691-701': [
        'zsqen = SQRT', 'p_avt(ji,jj,jk)   = MAX', 11],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354': [
        'start from turbulent closure values', 'CALL r54_zdfphy_finish', 7],
    'nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90:91-95': [
        'Match the executing dummies exactly',
        ('REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: p_avm', 1), 5],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:47-113': [
        'def _calibrate_closure', ('return counts', 1), 67],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:289-295': [
        'elif plant == "prandtl":', ('np.float64(np.inf))', 1), 7],
    # --- decision 35: the GYRE card drops fix_eta_drift (NEMO adds emp locally) ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:137': (
        'pssh(ji,jj,Kaa) = (  pssh(ji,jj,Kbb) - rDt * ( r1_rho0 * emp(ji,jj) + zhdiv(ji,jj) )  ) * ssmask(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:553': (
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/usrdef_sbc.f90:151-157': [
        "zsumemp = glob_2Dsum( 'usrdef_sbc', emp  (:,:)   )",
        'emp (ji,jj) = emp(ji,jj) - zsumemp * tmask(ji,jj,1)', 7],
    'ocean_model_latlon_cgrid.py:7863-7928': [
        'if _cfg_b.fix_eta_drift:',
        'eta=state_new.eta.replace(data=eta_fixed),', 66],
    # --- round 51: live WS-RK3 operand selection and history carry ---
    'ocean_model_latlon_cgrid.py:6124-6125': [
        ('u0 = state.u.data', 2), ('v0 = state.v.data', 2), 2],
    'ocean_model_latlon_cgrid.py:7559-7564': [
        '_p0_with_zub = _mom_pert_ws(',
        ('stage_face_thickness=_face_thickness_kbb, stage_index=1)', 2), 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215': [
        'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
        'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 26],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-146,196-200,236-244': [
        ('SELECT CASE( kstg )', 1), ('r1_Dt = 1._wp / rDt', 1),
        'CASE ( 2 )', ('r1_Dt = 1._wp / rDt', 2),
        ('ENDIF', 5), ('r1_Dt = 1._wp / rDt', 3), 23],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:308-310': [
        '! set current model timestep rDt = 2*rn_Dt if MLF or rDt = rn_Dt if RK3',
        'r1_Dt = 1._wp / rDt', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:277-298': [
        'IF( .NOT. PRESENT( k_ind ) ) THEN',
        '&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)',
        22],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:215-235': [
        ('CASE ( np_LIN )', 2),
        ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-489,749-761': [
        'Set extrapolation coefficients for predictor step:',
        'zsshp2_e(:,:) = za1 * sshn_e(:,:)',
        ('!* Swap', 1), 'sshn_e (:,:) = ssha_e(:,:)', 47],
    # --- round 49: compiled GYRE LDF/ENE statements and call order ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:165-167': [
        '&namdyn_vor', 'ln_dynvor_ene = .true.', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:431-480': [
        'CALL    eos    (        ts, Kmm, rhd, rhop )',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 50],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:690-712': [
        ('CASE ( 3 )', 2), "CALL r46_rhs( 'after_ldf', uu, vv, Krhs )", 23],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 )', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynvor.f90:536-573': [
        ('CASE ( np_CRV )', 2),
        'pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) - r1_4 * r1_e2v', 38],
    # --- ROUND 40: the stage-3 momentum operator table, ldf_slp, and r3t ---
    'stprk3_stg.F90:44': ('n_baro_upd =  np_HYB', 1),
    'stprk3_stg.F90:266-268': [
        ('DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )', 2),
        'zvb(ji,jj) = vn_adv(ji,jj)*r1_hv(ji,jj,Kmm) - vv_b(ji,jj,Kmm)', 3],
    'stprk3_stg.F90:273-274': [
        'zFu(ji,jj,jk) = e2u(ji,jj)*e3u(ji,jj,jk,Kmm)',
        'zFv(ji,jj,jk) = e1v(ji,jj)*e3v(ji,jj,jk,Kmm)', 2],
    'stprk3_stg.F90:290': (
        'CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )', 1),
    'stprk3_stg.F90:322': ('CALL    eos    (        ts, Kmm, rhd, rhop )', 1),
    'stprk3_stg.F90:202': (
        'r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 1),
    'stprk3_stg.F90:234': (
        'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 1),
    'stprk3_stg.F90:231': (
        'r3t(:,:,Kaa) = r3ta(:,:)                          ! at N+1   (Kaa)',
        1),
    'dynhpg.F90:359-363': [
        'puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap',
        'pvv(ji,jj,1,Krhs) = pvv(ji,jj,1,Krhs) + zhpj(ji,jj) + zvap', 5],
    'dynhpg.F90:383-387': [
        'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap',
        'pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) + zhpj(ji,jj) + zvap', 5],
    'dynhpg.F90:368-375': [
        'DO_2D( 0, 0, 0, 0 )    ! interior value (2=<jk=<jpkm1)',
        '&           - e3w(ji,jj  ,jk,Kmm) * ( rhd(ji,jj,  jk) + '
        'rhd(ji,jj  ,jk-1) )  )', 8],
    'dynkeg.F90:120-121': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - '
         'zhke(ji,jj) ) * r1_e1u(ji,jj)', 1),
        ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - '
         'zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2],
    'dynzad.F90:104-107': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * '
         'r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)', 1),
        '&                                            * ( zWdzV(ji,jj) + '
        'zzWdzV )', 4],
    'dynzad.F90:89-96': [
        'zWf  = e1e2t(ji  ,jj  ) * ( ww(ji  ,jj  ,jk+1) + '
        'wsd(ji  ,jj  ,jk+1) )',
        ('ENDIF', 4), 8],
    'dynvor.F90:493-516': [
        'SELECT CASE( nn_e3f_typ  )', ('END SELECT', 10), 24],
    'GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/dynvor.f90:556': (
        'zwz(ji,jj) = zwz(ji,jj) / (e3f_0vor(ji,jj,jk) *(1._wp+r3f(ji,jj)*fe3mask(ji,jj,jk)))', 1),
    # --- ROUND 41: exact compiled KEG/ZAD dispatch and consumed ww branch ---
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynadv.f90:134-138': [
        ('SELECT CASE( n_dynadv )', 1), 'CALL dyn_zad', 5],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/stprk3_stg.f90:326-332': [
        ('IF( ln_dynadv_vec ) THEN', 1),
        ('IF( ln_zad_Aimp .AND. kstg == 3 )   CALL wAimp', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynkeg.f90:117-130': [
        'CASE ( nkeg_C2 )',
        ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) -', 1), 14],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynzad.f90:105-137': [
        'DO jk =  1,  jpk-2',
        '&                                              * zWdzV(ji,jj)', 33],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sshwzv.f90:271-298': [
        ('DO jj = ntsj-( 1), ntej+(  1 )', 2),
        ('r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb)', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90:408-423': [
        'IF( .NOT. ln_wave ) THEN', 'RETURN', 16],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90:469-477': [
        ('IF( ln_sdw ) THEN', 1), 'wsd   (:,:,:) = 0._wp', 9],
    'ocean_pe_latlon_cgrid.py:2033-2067': [
        'elif config.ke_gradient_scheme == "c2":', 'dp_dy = _dKp_dy[..., 1]', 35],
    'ocean_pe_latlon_cgrid.py:3259-3285': [
        'area_w = jax.lax.optimization_barrier(',
        'diag_vertadv_v = jax.lax.optimization_barrier(diag_vertadv_v)', 27],
    'ocean_pe_latlon_cgrid.py:5207-5245': [
        'zad_w, zad_h_u, zad_h_v = w, h_u, h_v',
        ('zad_h_v = jax.lax.optimization_barrier(zad_h_v)', 1), 39],
    'ocean_model_latlon_cgrid.py:6935-6945': [
        '_freeze_hpg = self._nemo_ws_test_hooks.freeze_stage_hpg_operands',
        ('getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1), 11],
    'ocean_model_latlon_cgrid.py:8214-8231': [
        ('if (getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1),
        '/ jnp.maximum(_area_v, 1.0e-30))', 18],
    'ocean_model_latlon_cgrid.py:8242-8264': [
        '# 7b. Adaptive-implicit vertical momentum advection',
        'and _pflow is None):', 23],
    # --- ROUND 42: this round's compiled header and slope walk ---
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:466-472': [
        '!* advection (VIF or FF)',
        ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), 7],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333': [
        ('IF( ln_dynadv_vec ) THEN', 1),
        ('CALL wAimp', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:201-215': [
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )',
        'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 15],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stp2d.f90:141-166': [
        'CALL dyn_hpg', 'CALL dyn_zad', 26],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:447-469': [
        'CALL    dyn_hpg', 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 23],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:704-717': [
        'CALL dyn_ldf', 'CALL dyn_zdf', 14],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:724-743': [
        ('barotropic velocity correction', 4), 'corrected horizontal velocity', 20],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90:152-176': [
        ('SELECT CASE( n_dynadv )', 1),
        'CALL dyn_zad', 25],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynzad.f90:105-137': [
        'DO jk =  1,  jpk-2',
        '&                                              * zWdzV(ji,jj)', 33],
    # --- ROUND 47: exact widened compiled branch ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:292-316': [
        ('DO jk =  1,  jpkm1', 1),
        '&           vn_adv, (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm))), vv_b(:,:,Kmm)',
        25],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:350-357': [
        ('IF( lwp .AND. kstp == nit000 ) THEN', 1),
        ('WRITE(l1_unit) zFu, zFv, zFw', 1), 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:329-335': [
        ('IF( ln_dynadv_vec ) THEN', 1),
        ('IF( ln_zad_Aimp .AND. kstg == 3 )', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:105-137': [
        'DO jk =  1,  jpk-2',
        '&                                              * zWdzV(ji,jj)', 33],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-176': [
        '!*  hydrostatic pressure gradient (HPG))',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 36],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:206-207': [
        'Ue_rhs(ji,jj) = SUM(',
        'Ve_rhs(ji,jj) = SUM(', 2],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:365-371': [
        ('SELECT CASE( kstg )', 2),
        'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv', 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:664-672': [
        ('SELECT CASE( kstg )', 3),
        'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )',
        9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv.f90:248-250': [
        'pFu(ji,jj,jpk) = 0._wp', 'pFw(ji,jj,jpk) = 0._wp', 3],
    # --- ROUND 48: model-path first boundary and cross-step bt memory ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175': [
        '!*  hydrostatic pressure gradient (HPG))',
        "CALL r46_rhs( 'after_zad', uu, vv, Krhs )", 35],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-145': [
        'hydrostatic pressure gradient (HPG))',
        "CALL r46_rhs( 'after_hpg', uu, vv, Krhs )", 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:324-337': [
        'vertical velocity and transport (ww,wi,zFw)',
        ('ENDIF', 8), 14],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:472-480': [
        'advection (VIF or FF)',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:100-130': [
        ('vertical momentum advection', 2),
        'END DO   ;   END DO   ;   END DO', 31],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:123-126': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs)', 1),
        '&                                            * ( zWdzV(ji,jj) + zzWdzV )', 4],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:695-711': [
        'Round-29 L2 WRITE-only stage-3 pre-LDF momentum frame.',
        'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 17],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:64-74': [
        'SUBROUTINE dynldf_lev_lap( kt, Kbb, Kmm, pu, pv, Krhs )',
        'pu(Krhs), pv(Krhs) increased by the harmonic operator applied on pu(Kbb), pv(Kbb)', 11],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:123-127': [
        ('zwf(ji-1,jj-1) = ahmf', 1),
        ('zwt(ji,jj)     = ahmt', 1), 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynhpg.f90:400-427': [
        '!                                   ! add to the general momentum trend',
        'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:352-378': [
        'IF( ln_bt_fw ) THEN                 ! FORWARD integration: start from NOW fields',
        'vn_adv(:,:)     = 0._wp', 27],
    # --- PR #1802 final round: decisions 66 and 67 -----------------------
    # D67's own call site, and the DINO year screen's pre-existing refusal.
    'barotropic_latlon_cgrid.py:2852-2853': [
        'U_bar_corr, V_bar_corr = _depth_average_to_faces(',
        'u_corr, v_corr, _h_k_corr, min_water_col, mask, u_mask, v_mask, grid,',
        2],
    'eos.py:742':
        '"raw-mesh e3w_int must contain only finite values > 0",',
    # D67: the e1e2-weighted SSH-average face depth NEMO divides the
    # accumulated barotropic transport by, to form puu_b/pvv_b(Kaa).
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:835-842': [
        'zzsshu = r1_2 * r1_e1e2u(ji,jj) * ( e1e2t(ji  ,jj) * pssh(ji  ,jj,Kaa)',
        'pvv_b(ji,jj,Kaa) = pvv_b(ji,jj,Kaa) / ( hv_0(ji,jj) + zzsshv + 1._wp '
        '- ssvmask(ji,jj) )',
        8],
    # D67: the consumer -- the RK3 stage subtracts uu_b(Kmm), the velocity
    # the statement above produced, from the 3-D velocity.
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:276':
        'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) '
        '- uu_b(ji,jj,Kmm)',
    # The branch NEMO ACTUALLY runs on these cards (ln_dynadv_vec=T):
    # it sums the substep velocities and divides by the weight sum,
    # with no face depth -- which is what refutes decision 67's premise.
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:767-769': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN    ! Sum velocities',
        ('pvv_b  (:,:,Kaa) = pvv_b  (:,:,Kaa) + za1 * va_e  (:,:)', 1),
        3],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:802':
        'puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) / r1_wgt1s',
    # D66: the masked ln_mxl0 surface stress, and its floor.
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:606':
        'zmxlm(ji,1) =  zraug * taum(ji,jj) * tmask(ji,jj,1)',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:610':
        'zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )',
    # D66: ln_mxl0 overwrites the namelist rn_mxl0 with the derived rmxl_min,
    # so NEMO's own anchor floor is rmxl_min, not rn_mxl0=0.04.
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:831':
        ('rn_mxl0 = rmxl_min', 2),
    # D72: the whole guarded overwrite block -- the IF that decides whether
    # rn_mxl0 is replaced at all, and the assignment that replaces it -- and
    # the ln_zdfiwm arm that forces the rmxl_min it is replaced WITH.
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:828-831': [
        ('IF( ln_mxl0 ) THEN', 4),
        ('rn_mxl0 = rmxl_min', 2), 4],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:810-812': [
        'IF( ln_zdfiwm ) THEN          ! Internal wave-driven mixing',
        'rmxl_min = 1.e-03_wp             ! associated avt minimum = molecular salt diffusivity (10^-9 m2/s)',
        3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:824-827': [
        'IF(.NOT.ll_bt_av ) THEN', 'pssh (:,:,Kaa) = ssha_e(:,:)', 4],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-480': [
        'Set extrapolation coefficients for predictor step:',
        'va_e(ji,jj) = za1 * vn_e(ji,jj) + za2 * vb_e(ji,jj) + za3 * vbb_e(ji,jj)', 25],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:749-757': [
        '!* Swap', 'vn_e   (:,:) = va_e  (:,:)', 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:953-980': [
        "IF( TRIM(cdrw) == 'READ' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'", 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:483-489': [
        ('IF( .NOT.lk_linssh ) THEN', 1),
        'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)', 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:593-600': [
        'Half-step back interpolation of SSH',
        '&            + za2 *  sshb_e(ji,jj) + za3 *  sshbb_e(ji,jj)', 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:759-761': [
        'sshbb_e(:,:) = sshb_e(:,:)', 'sshn_e (:,:) = ssha_e(:,:)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:373-378': [
        '! Initialize sums:', 'vn_adv(:,:)     = 0._wp', 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:559-567': [
        '! Sum over sub-time-steps to compute advective velocities',
        'vn_adv(ji,jj) = vn_adv(ji,jj) + za2 * zhV(ji,jj) * r1_e1v(ji,jj)', 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:797-804': [
        '! Finalize sums:', 'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:858-863': [
        'END Phase 3 for RK3', "IF( lrst_oce )   CALL ts_rst( kt, 'WRITE' )", 6],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/ocean.output:535': (
        'Courant number targeted application   ln_zad_Aimp =  F', 1),
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90:176-185': [
        'CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )',
        '&           jpi, jpj, jpk, jpkm1, ntsi, ntei, ntsj, ntej, STORAGE_SIZE(1._wp)', 10],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynkeg.f90:117-130': [
        'CASE ( nkeg_C2 )',
        ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) -', 1), 14],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdfmxl.f90:109-123': [
        '! w-level of the mixing and mixed layers',
        'hmlp(ji,jj) = ((gdepw_1d(iik  ) ) *(1._wp+r3t(ji,jj,Kmm))) * ssmask(ji,jj)', 15],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:222-232': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4),
        'zbv = MIN(  zbv, - z1_slpmax * ABS( zav )', 11],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:235-275': [
        'iku = MAX( nmln(ji+1,jj), nmln(ji,jj) )',
        '* ( vmask(ji  ,jj,jk) + vmask(ji  ,jj,jk+1) ) * 0.5_wp', 41],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:284-333': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 5),
        '+ 4.*    zww(ji  ,jj  )                        ) * zcofw', 50],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/traldf_iso.f90:793-794': [
        'pah_wslp2(ji,jj,jk) = zahu_w * wslpi(ji,jj,jk) * wslpi(ji,jj,jk)',
        '&                + zahv_w * wslpj(ji,jj,jk) * wslpj(ji,jj,jk)', 2],
    'cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm:1': (
        'bld::tool::fppkeys key_qco key_vco_1d3d key_RK3', 1),
    # The statement is textually IDENTICAL at :160 (dom_qco_r3c) and :209
    # (dom_qco_r3c_RK3, the one the RK3 stage program calls), so the citation
    # spans the routine header that disambiguates it.
    'domqco.F90:189-209': [
        'SUBROUTINE dom_qco_r3c_RK3( pssh, pr3t, pr3u, pr3v, pr3f )',
        ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)   '
         '!==  ratio at t-point  ==!', 2), 21],
    'domain.F90:158': (
        'r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:) + 1._wp -  ssmask (:,:) )', 1),
    'ZDF/zdfmxl.F90:34': ('rho_c = 0.01_wp', 1),
    'ZDF/zdfmxl.F90:95': ('zN2_c = grav * rho_c * r1_rho0', 1),
    'ZDF/zdfmxl.F90:98': (
        'hmlp(ji,jj) =  hmlp(ji,jj) + MAX( rn2b(ji,jj,jk) , 0._wp ) * e3w(ji,jj,jk,Kmm)', 1),
    'ZDF/zdfmxl.F90:99': (
        'IF( hmlp(ji,jj) < zN2_c )   nmln(ji,jj) = MIN( jk , ikt ) + 1', 1),
    'ZDF/zdfmxl.F90:104': (
        'hmlp(ji,jj) = gdepw(ji,jj,iik  ,Kmm) * ssmask(ji,jj)', 1),
    'LDF/ldfslp.F90:143': (
        'zhmlpt(ji,jj) = gdept(ji,jj,nmln(ji,jj)-1,Kmm) * ssmask(ji,jj)', 1),
    'LDF/ldfslp.F90:161': (
        'r1_hmlw(ji,jj) = 1._wp / MAX( hmlp(ji,jj) - gdepw(ji,jj,mikt(ji,jj),Kmm), 10._wp )', 1),
    'LDF/ldfslp.F90:206': ('zau = zgru(ji,jj,iik) * r1_e1u(ji,jj)', 1),
    'LDF/ldfslp.F90:212-213': [
        'zbu = MIN(  zbu, - z1_slpmax * ABS( zau ) , '
        '-7.e+3_wp/e3u(ji,jj,jk,Kmm)* ABS( zau )  )',
        'zbv = MIN(  zbv, - z1_slpmax * ABS( zav ) , '
        '-7.e+3_wp/e3v(ji,jj,jk,Kmm)* ABS( zav )  )', 2],
    'LDF/ldfslp.F90:281-282': [
        'zbi = MIN( zbw ,- 100._wp* ABS( zai ) , '
        '-7.e+3_wp/e3w(ji,jj,jk,Kmm)* ABS( zai )  )',
        'zbj = MIN( zbw , -100._wp* ABS( zaj ) , '
        '-7.e+3_wp/e3w(ji,jj,jk,Kmm)* ABS( zaj )  )', 2],
    # Round 34's carried-hf_0 selector rigidly shifts this later site by ten.
    'vertical.py:1819': ('def compute_ocean_jacobian(', 1),
    'ocean_model_latlon_cgrid.py:6161-6165': [
        'transport_velocity = (', ('* _ws_stage_v_mask,', 1), 5],
    'ocean_model_latlon_cgrid.py:7694': ('_g2 = _nemo_ws_stage_transport(', 1),
    'ocean_model_latlon_cgrid.py:7713': (
        '_stage3_hpg_operands = _stage_hpg_operands(', 1),
    'nemo_testcase_l2_gyre_phase3_gate.py:323': (
        'require(magic == "NEMO_L2_RKTRM_1", f"{path}: bad magic")', 1),
    'round38_oracle_trazdf_kt2/ocean.output:799': (
        'with Hollingsworth scheme (=1) or not (=0)       nn_dynkeg   =', 1),
    'round38_oracle_trazdf_kt2/ocean.output:810': (
        'f-point energy conserving scheme               ln_dynvor_ene =  T', 1),
    'round38_oracle_trazdf_kt2/ocean.output:813': (
        'e3f = averaging /4 (=0) or /sum(tmask) (=1)    nn_e3f_typ =', 1),
    'round38_oracle_trazdf_kt2/ocean.output:834': (
        's-coord. (standard jacobian formulation)          ln_hpg_sco    =  T', 1),
    'round38_oracle_trazdf_kt2/ocean.output:846': (
        'Free surface with time splitting       ln_dynspg_ts  =  T', 1),
    'round38_oracle_trazdf_kt2/ocean.output:658': (
        'maximum isoppycnal slope             rn_slpmax', 1),
    # --- ROUND 39: the slopes are built ONCE PER STEP on the BEFORE state ---
    'stprk3.F90:173': ('CALL eos ( ts, Nbb, rhd )                   ! before in situ density', 1),
    'GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/stprk3.f90:178': (
        'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )   ! before slope for standard operator', 1),
    'LDF/ldfslp.F90:80': ('SUBROUTINE ldf_slp( kt, prd, pn2, Kbb, Kmm )', 1),
    'traldf_iso.F90:296-297': [
        'pah_wslp2(ji,jj,jk) = zahu_w * wslpi(ji,jj,jk) * wslpi(ji,jj,jk)   &',
        '&                + zahv_w * wslpj(ji,jj,jk) * wslpj(ji,jj,jk)', 2],
    'traldf_iso.F90:291-294': [
        'zahu_w = (  ( ahtu(ji  ,jj,jk-1) + ahtu(ji-1,jj,jk) )    &',
        '&      + ( ahtv(ji,jj-1,jk-1) + ahtv(ji,jj  ,jk) )  ) * zmskv', 4],
    'domqco.F90:160': (
        'pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)   !==  ratio at t-point  ==!', 1),
    'fidelity/nemo_recipe.py:978': ('def build_nemo_gyre_recipe(', 1),
    'ocean_model_latlon_cgrid.py:5732': (
        'T_new = state.T.data + dt * tend.dT_dt.data', 1),
    'round38_oracle_trazdf_kt2/ocean.output:798': (
        'Vector form: 2nd order centered scheme           ln_dynadv_vec  =  T', 1),
    'lock_kt1_10/ocean.output:578': (
        'no explicit diffusion                   ln_traldf_OFF   =  T', 1),
    'lock_kt1_10/ocean.output:584': (
        'iso-neutral Madec operator              ln_traldf_iso   =  F', 1),
    'overflow_kt1_10/ocean.output:690': (
        'no explicit diffusion                   ln_traldf_OFF   =  T', 1),
    'overflow_kt1_10/ocean.output:696': (
        'iso-neutral Madec operator              ln_traldf_iso   =  F', 1),
    # --- ROUND 38: the isoneutral fold, and where each side computes it ---
    # NEMO computes the slopes ONCE PER STEP, on the BEFORE state, outside the
    # RK3 stage loop; legoESM recomputes its K33 inside each stage.  These four
    # are the statements that difference rests on.
    'stprk3.F90:174': (
        'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )   ! before slope for standard operator', 1),
    # traldf_iso_a33 is called from BOTH the laplacian and the bilaplacian
    # routine; GYRE resolves ln_traldf_lap = T and its log says
    # "traldf_iso_lap", so the FIRST occurrence is the one that runs.
    'traldf_iso.F90:135': (
        'CALL traldf_iso_a33( Kmm, ah_wslp2, akz )   ! calculate  a33 element   (ah_wslp2 and akz)', 1),
    'trazdf.F90:173': ('zwt(ji,jk) = avt(ji,jj,jk) + ah_wslp2(ji,jj,jk)', 1),
    'ocean_model_latlon_cgrid.py:8333': (
        '_T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]', 1),
    # ROUND 39 moved these two: the before-state slope block added lines
    # above them, so the STATEMENT is unchanged and its line number is not.
    # Re-anchored rather than left to rot, which is what the gate exists for.
    'ocean_model_latlon_cgrid.py:8698': (
        'k33_implicit = compute_isoneutral_K33_latlon(', 1),
    # tracer_combine is READ by two step functions and SELECTED by a DINO
    # recipe -- the retraction of round 37's "a lever nothing selects".
    'dino.py:1751': ('"tracer_combine": "thickness_weighted",', 1),
    'dino.py:3967': ('tracer_combine=cfg.tracer_combine,', 1),
    # A bare '}' is the eighth-most-common line in that file, so the endpoint
    # is the last SUBSTANTIVE line of the reconstruction rather than its brace.
    'nemo_testcase_l2_gyre_stage3_completion_gate.py:147-156': [
        ('def _oracle_pre_zdf(record: dict, nlev: int, dt: float) -> dict[str, np.ndarray]:', 1),
        ('for name in ("T", "S")', 1),
        10],
    # --- round 35 citations: the implicit vertical TRACER solve ---
    # The build guard that makes salinity reuse temperature's matrix.
    'trazdf.F90:159-160': [
        "IF(  ( cdtype == 'TRA' .AND. ( jn == jp_tem .OR. ( jn == jp_sal .AND. ln_zdfddm ) ) ) .OR.   &",
        "& ( cdtype == 'TRC' .AND. jn == 1 )  )  THEN",
        2],
    # The ah_wslp2 arm.  DO_2Dik/END_2D repeat throughout the routine, so both
    # endpoints are pinned by occurrence.
    'trazdf.F90:172-174': [
        ('DO_2Dik( 0, 0,   2, jpk, 1 )', 2), ('END_2D', 2), 3],
    'trazdf.F90:204': 'zwt(:,1) = 0._wp',
    'trazdf.F90:219-220': [
        'zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / e3w(ji,jj,jk,Kmm)',
        'zws(ji,jk) = - p2dt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)',
        2],
    # The DRAKKAR negative-salinity clamp, guard to ENDIF.
    'trazdf.F90:89-91': [
        'IF ( .NOT.(ln_SEOS.AND.(rn_b0==0._wp)) ) THEN', ('ENDIF', 4), 3],
    'ldftra.F90:249': ('l_ldfslp = .TRUE.', 1),
    # The two thickness macros.  Each also appears once COMMENTED OUT further
    # down the header, which is why both are pinned to the first occurrence.
    'domzgr_substitute.h90:126': (
        '# define  e3t(i,j,k,t)      (E3t_0(i,j,k) Tmsk(r3t,tmask,i,j,k,t))', 1),
    'domzgr_substitute.h90:131': (
        '# define  e3w(i,j,k,t)      (E3w_0(i,j,k) Time(r3t,i,j,t))', 1),
    # The same two, as GYRE's build EXPANDED them: e3w loses its mask and its
    # third dimension, e3t keeps both.
    'ppsrc/nemo/trazdf.f90:231-233': [
        'zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))',
        'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )',
        3],
    'round29_oracle_v2_zdf_matrix/ocean.output:568': 'ln_zdfddm =  F',
    'round29_oracle_v2_zdf_matrix/ocean.output:667': (
        '==>>>   Rotated laplacian operator (standard)', 1),
    'round29_oracle_v2_zdf_matrix/ocean.output:657': 'ln_traldf_msc   =  F',
    'round29_oracle_v2_zdf_matrix/ocean.output:158': 'ln_SEOS   =  F',
    # --- round 36: the mislabelled record, and NEMO's own reciprocal ---
    # avt and avs are allocated over the INTERIOR box; avm, on the same
    # statement, is not.  That asymmetry is the whole finding, so both lines
    # are pinned.
    'zdf_oce.f90:85-86': [
        'ALLOCATE( avm (jpi,jpj,jpk), avm_k(jpi,jpj,jpk), avs(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) ,   &',
        '&      avt (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) , avt_k(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) , en (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) ,   &',
        2],
    # The two short WRITEs, which is where the record's headers stopped
    # describing its payload.
    'trazdf.f90:247': 'WRITE(il2_unit) avt',
    'trazdf.f90:249': 'WRITE(il2_unit) avs',
    # zwt(:,1) = 0 is why avt's surface plane is a slice no arm reads.
    'trazdf.f90:419': 'zwt(:,1) = 0._wp',
    # The first non-bit statements, in NEMO's own execution order: the two
    # face coefficients whose zeros are NEGATIVE, then the diagonal.
    'trazdf.f90:443-444': [
        'zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))',
        'zws(ji,jk) = - p2dt * zwt(ji,jk+1) / (e3w_1d(jk+1) *(1._wp+r3t(ji,jj,Kmm)))',
        2],
    'trazdf.f90:445':
        'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )',
    # The right-hand side, LEFT TO RIGHT: (p2dt*e3t_Kmm)*T_Krhs.
    'trazdf.f90:528-529': [
        'zrhs =       (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))) * pt(ji,jj,jk,jn,Kbb )   &',
        '& + p2dt * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))) * pt(ji,jj,jk,jn,Krhs)   ! zrhs=right hand side',
        2],
    # The MULTIPLY by a stored reciprocal, and where that reciprocal is made.
    'stprk3_stg.f90:522-523': [
        'zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)',
        'zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_3d(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)',
        2],
    'domain.f90:213':
        'r1_hu_0(:,:) = ssumask(:,:) / ( hu_0(:,:) + 1._wp -  ssumask(:,:) )',
    # NEMO adds the correction PER LEVEL, masked; it never masks uu itself.
    # This round first cited :526-527, which in THIS build is a comment --
    # the instrument shifts the line numbers, and the map caught it.
    'stprk3_stg.f90:541-542': [
        'uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)',
        2],
    # --- round 37 citations: the three recurrences, literally ---
    # ln_tile is F, which is what makes ntsi:ntei equal Nis0:Nie0 in the
    # round-37 instrument's staging of avt and avs.
    'round29_oracle_v2_zdf_matrix/ocean.output:247':
        'Tiling (T) or not (F)                ln_tile    =  F',
    # 1st recurrence: multiply, THEN divide, THEN subtract.  It ends in a
    # division, so a fused multiply-add cannot absorb it, which is why its
    # row was never over the bar.
    'trazdf.f90:493': 'zwt(ji,1) = zwd(ji,1)',
    'trazdf.f90:496':
        'zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)',
    # 2nd recurrence.  The surface row has no zrhs temporary.
    'trazdf.f90:515-516': [
        'pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,1))) * pt(ji,jj,1,jn,Kbb )    &',
        '&               + p2dt * (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1))) * pt(ji,jj,1,jn,Krhs)',
        2],
    # Divide, THEN multiply, THEN subtract -- the first of the two statements
    # XLA was contracting into a fused multiply-add.
    'trazdf.f90:532':
        'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)',
    # 3rd recurrence, and the tmask NEMO applies at EVERY level.
    'trazdf.f90:543':
        'pt(ji,jj,jpkm1,jn,Kaa) = pt(ji,jj,jpkm1,jn,Kaa) / zwt(ji,jpkm1) * tmask(ji,jj,jpkm1)',
    'trazdf.f90:546-547': [
        'pt(ji,jj,jk,jn,Kaa) = ( pt(ji,jj,jk,jn,Kaa) - zws(ji,jk) * pt(ji,jj,jk+1,jn,Kaa) )   &',
        '&             / zwt(ji,jk) * tmask(ji,jj,jk)',
        2],
    # The DO jk loop that APPLIES zub.  It is cited to say what it is NOT:
    # there is no DO jk loop that FORMS the column sum, only the SUM
    # intrinsic at :522-523, so the accumulation order is the compiler's.
    'stprk3_stg.f90:540-543': [
        'DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)   ! corrected horizontal velocity',
        ('END DO   ;   END DO   ;   END DO', 7),
        4],
    # --- round 32: the stage-3 ordering fix ---
    # --- round 33: the stage arm, and the tanks' reference geometry ---
    # NEMO's ONE selector, used at every stage, and both of its arms.
    'stprk3_stg.F90:365': 'IF( ln_dynadv_vec .OR. lk_linssh ) THEN   !* applied on velocity',
    'stprk3_stg.F90:366-369': [
        ('DO_3D( 0, 0, 0, 0, 1, jpkm1 )', 2), ('END_3D', 4), 4],
    'stprk3_stg.F90:370-388': [
        'ELSE                                      !* applied on thickness weighted velocity',
        ('ENDIF', 11), 19],
    'stprk3_stg.F90:395': ('CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping', 1),
    'stprk3_stg.F90:440,444-445': [
        'zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)',
        'uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)',
        3],
    'dynzdf.F90:127-132': [
        'puu(ji,jj,jk,Kaa) = (        ( 1._wp + r3u(ji,jj,Kbb) ) * puu(ji,jj,jk,Kbb )  &',
        '&              /          ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)',
        6],
    # e3u_0 is a MACRO, and which arm a card compiles is the whole point.
    'domzgr_substitute.h90:89': '#     define  e3u_0(i,j,k)    e3t_1d(k)',
    'domzgr_substitute.h90:98': ('#     define  e3u_0(i,j,k)    e3u_3d(i,j,k)', 1),
    # each tank's OWN resolved momentum-advection arm.
    'lock_kt1_10/ocean.output:705': 'ln_dynadv_vec  =  F',
    'overflow_kt1_10/ocean.output:822': 'ln_dynadv_vec  =  F',
    'ocean_model_latlon_cgrid.py:7078': '_vector_velocity_stage_update = (',
    'dynzdf.F90:119': 'IF( ln_dynadv_vec .OR. lk_linssh )',
    'dynzdf.F90:150-151': ['puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b',
                           'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b', 2],
    'round19_oracle_v2_external/ocean.output:798': 'ln_dynadv_vec  =  T',
    'variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output:1097':
        'ln_drgimp   =  T',
    'variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output:1363':
        'ln_dynspg_ts  =  T',
    'stprk3.F90:186': 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
    'stprk3.F90:195': 'stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )',
    'stprk3.F90:197': [('Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs', 1),
         ('Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs', 1),
         1],
    'stprk3.F90:200': 'stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )',
    'stprk3.F90:207': 'stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )',
    'MY_SRC/stprk3.F90:204': 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
    'MY_SRC/stprk3.F90:206': 'CALL l1_dump_rhs( kstp, Nrhs )',
    'MY_SRC/stprk3.F90:215': 'stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )',
    'stp2d.F90:126': 'hydrostatic pressure gradient (HPG))',
    'stp2d.F90:128': 'CALL dyn_hpg( kt, Kbb',
    'stp2d.F90:131': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )',
    'stp2d.F90:146': 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )',
    'stp2d.F90:172': 'CALL dyn_adv_up3 ( kt, Kbb, Kbb, uu, vv, Krhs, pUe=Ue_rhs',
    'stp2d.F90:185': 'Ue_rhs(ji,jj) + SUM( e3u_0(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)',
    'stp2d.F90:196': 'CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs',
    'stp2d.F90:199-202': [('DO_2D( 0, 0, 0, 0 )', 3), ('END_2D', 4), 4],
    'stp2d.F90:200': 'r1_rho0 * utauU(ji,jj) * r1_hu(ji,jj,Kbb)',
    'stp2d.F90:281': ('CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, '
         'Kaa )'),
    'stprk3_stg.F90:65': 'SUBROUTINE stp_RK3_stg( kstg, kstp, Kbb, Kmm, Krhs, Kaa )',
    'stprk3_stg.F90:262,267': ['zub(ji,jj) = r1_Dt * rn_Dt * un_adv(ji,jj)',
         'zub(ji,jj) = un_adv(ji,jj)*r1_hu(ji,jj,Kmm) - uu_b(ji,jj,Kmm)',
         2],
    'stprk3_stg.F90:309-334': [('SELECT CASE( kstg )', 2), ('ENDIF', 9), 26],
    'stprk3_stg.F90:315': ('IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, '
         'vv, Krhs, zFu'),
    'stprk3_stg.F90:315-430': ['IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, '
         'vv, Krhs, zFu',
         'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, '
         'Kaa  )',
         116],
    'stprk3_stg.F90:324': 'CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )',
    'stprk3_stg.F90:324-378': ['CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )',
         '/           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)',
         55],
    'stprk3_stg.F90:327': 'CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )',
    'stprk3_stg.F90:331': 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)',
    'stprk3_stg.F90:333': [('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )',
          2),
         ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )',
          2),
         1],
    'stprk3_stg.F90:367-368': ['uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * '
         'uu(ji,jj,jk,Krhs) )',
         'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * '
         'vv(ji,jj,jk,Krhs) )',
         2],
    'stprk3_stg.F90:373-378': ['uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) )',
         '/           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)',
         6],
    'stprk3_stg.F90:437-446': [('#endif', 6), ('END_3D', 6), 10],
    'stprk3_stg.F90:439-446': [
        ('DO_2D( 0, 0, 0, 0 )             ! barotropic velocity correction', 1),
        ('END_3D', 6), 8],
    # round 33: the two tanks' own resolved dynzdf guard, so the claim
    # that they EXECUTE the moved operator binds to their own run logs
    # and not to GYRE's or to each other's.
    'lock_kt1_10/ocean.output:560': 'ln_drgimp   =  T',
    'lock_kt1_10/ocean.output:752': 'ln_dynspg_ts  =  T',
    'overflow_kt1_10/ocean.output:672': 'ln_drgimp   =  T',
    'overflow_kt1_10/ocean.output:869': 'ln_dynspg_ts  =  T',
    'stprk3_stg.F90:453-598': ['Tracers : RHS computation + time-stepping',
         'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )',
         146],
    'stprk3_stg.F90:453-601': ['Tracers : RHS computation + time-stepping', ('END DO', 6), 149],
    'stprk3_stg.F90:463-599': [('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, '
          'zFu, zFv, zFw )',
          1),
         'IF( ln_zdfnpc  )   CALL tra_npc( kstp,      Kmm, Krhs, ts    , '
         'Kaa  )',
         137],
    'stprk3_stg.F90:655': 'END SUBROUTINE stp_RK3_stg',
    'stprk3_stg.F90:168-243': ['r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)',
         'Dynamic : RHS computation + time-stepping',
         76],
    'dynspg_ts.F90:282': 'zu_frc(:,:) =   Ue_rhs(:,:)',
    'dynspg_ts.F90:296': [('CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, '
          'zv_trd )',
          1),
         ('CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, '
          'zv_trd )',
          1),
         1],
    'dynspg_ts.F90:303': [('#else', 3), ('#else', 3), 1],
    'dynspg_ts.F90:344-345': ['DO_3D( 0, 0, 0, 0, 1, jpkm1 )',
         'puu(ji,jj,jk,Krhs) = ( puu(ji,jj,jk,Krhs) - zu_frc(ji,jj) )',
         2],
    'dynspg_ts.F90:487': 'un_e  (:,:) =    puu_b(:,:,Kmm)',
    'dynspg_ts.F90:735-761': [('DO_2D( 0, 0, 0, 0 )', 19), ('END_2D', 30), 27],
    'dynspg_ts.F90:750-753': ['z1_hv = ssvmask(ji,jj) / ( hv_0(ji,jj) + zsshv_a(ji,jj)',
         'rDt_e * (  zhu_bck        * zu_spg (ji,jj)',
         4],
    'dynspg_ts.F90:752-761': ['ua_e(ji,jj) = (               hu_e  (ji,jj) *   un_e (ji,jj)',
         ('END_2D', 30),
         10],
    'dynspg_ts.F90:825-847': ['ELSE                                       ! Sum transports',
         'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s',
         23],
    'dynspg_ts.F90:870-895': ['IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av ) '
         'THEN',
         "CALL lbc_lnk( 'dynspg_ts', puu_b, 'U', -1._wp, pvv_b, 'V', "
         '-1._wp )',
         26],
    'dynspg_ts.F90:910': [('#else', 7), ('#else', 7), 1],
    'dynspg_ts.F90:938-975': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 3),
         '* ( pvv_b(:,:,Kaa) - pvv_b(:,:,Kbb) * hv(:,:,Kbb) ) * r1_Dt',
         38],
    'dynhpg.F90:359': 'puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap',
    'dynhpg.F90:383': 'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap',
    'dynhpg.F90:359,383': ['puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap',
         'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap',
         2],
    'domzgr_substitute.h90:139': [('define  gdept(i,j,k,t)', 1), ('define  gdept(i,j,k,t)', 1), 1],
    'domzgr_substitute.h90:145': [('gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))', 1),
         ('gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))', 1),
         1],
    'domzgr_substitute.h90:139,145': [('define  gdept(i,j,k,t)', 1),
         ('gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))', 1),
         2],
    'oce.F90:39,99': ['ssh, uu_b,  vv_b',
         'ALLOCATE( ssh (jpi,jpj,jpt)  , uu_b(jpi,jpj,jpt)',
         2],
    'DOM/istate.F90:149-155': ['uu_b(:,:,Kbb) = 0._wp   ;   vv_b(:,:,Kbb) = 0._wp',
         'vv_b(:,:,Kbb) = vv_b(:,:,Kbb) * r1_hv(:,:,Kbb)',
         7],
    'usrdef_hgr.F90:103-104': ['pff_f(:,:) = 0._wp            ! here No earth rotation: f=0',
         'pff_t(:,:) = 0._wp',
         2],
    'trabbl.F90:519-527': [('DO_2D( 1, 0, 1, 0 )', 6), ('END_2D', 9), 9],
    'domain.F90:159': 'r1_hu_0(:,:) = ssumask(:,:) / ( hu_0(:,:) + 1._wp',
    'domzgr_substitute.h90:143': 'gdept_z0(i,j,k,t) gdept(i,j,k,t)',
    'stp2d.F90:163': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )',
    'stp2d.F90:165': 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )',
    'stp2d.F90:169': 'CALL dyn_adv_cen2( kt     , Kbb, uu, vv, Krhs, pUe=Ue_rhs',
    'stp2d.F90:177': [('SELECT CASE( n_dynadv )', 2),
                      ('SELECT CASE( n_dynadv )', 2), 1],
    'stp2d.F90:178': 'CASE( np_VEC_c2, np_LIN_dyn )',
    'stp2d.F90:183': 'CASE ( np_FLX_c2, np_FLX_up3 )',
    'namelist_cfg:91': 'ln_dynvor_ens = .true.',
    'lock_kt1_10/ocean.output:715': [('enstrophy conserving scheme', 1),
                                     ('enstrophy conserving scheme', 1), 1],
    'stp2d.F90:180': 'Ue_rhs(ji,jj) = SUM( e3u_0(ji,jj,1:jpkm1)',
    'stp2d.F90:207': 'grav * (  ssh_ib (ji+1,jj  ) - ssh_ib (ji,jj) )',
    'stp2d.F90:223': '( zpice(ji+1,jj) - zpice(ji,jj) ) * r1_e1u(ji,jj)',
    'stp2d.F90:235': '( bhd_wave(ji+1,jj) - bhd_wave(ji,jj) ) * r1_e1u(ji,jj)',
    'namelist_cfg:85': 'ln_dynadv_up3 = .true.',
    'namelist_cfg:105-106': ['nn_bt_flt     = 3', 'rn_bt_alpha   = 0.07', 2],
    'namelist_ref:1092': 'nn_bt_flt     = 1          ! Add dissipation',
    'namelist_ref:1177': 'ln_zad_Aimp = .false.',
    # --- round 29: the stage-3 momentum chain and dyn_zdf's matrix ---
    'stprk3_stg.F90:400': 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )     ! lateral mixing',
    'stprk3_stg.F90:430': 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration',
    'stprk3_stg.F90:156': 'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )',
    'stprk3_stg.F90:163': ('r3u(:,:,Kaa) = r3ua(:,:)', 1),
    'dynadv.F90:144': "CASE( np_VEC_c2  )   ;   WRITE(numout,*) '   ==>>>   vector form : keg + zad + vor is used'",
    'dynzdf.F90:121-122': ['puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kbb) + rDt * puu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kbb) + rDt * pvv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 2],
    'dynzdf.F90:156-159': ['puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * uu_b(ji,jj,Kaa)   &', ('&                                            / e3v(ji,jj,ikv,Kaa)', 1), 4],
    'dynzdf.F90:182-195': ['zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )     +  avm(ji,jj,jk  )     ) &', ('zwd(ji,1) = 1._wp - zzws', 1), 14],
    'dynldf.F90:70': 'CALL dynldf_lev_lap( kt, Kbb, Kmm, puu, pvv, Krhs )',
    'dynldf_lev_rot_scheme.h90:24-25': [
        'e2v(ji  ,jj-1) * pv_in(ji  ,jj-1,jk,Kbb)',
        'e1u(ji-1,jj  ) * pu_in(ji-1,jj  ,jk,Kbb)',
        2],
    'dynldf_lev_rot_scheme.h90:28-29': [
        'e2u(ji,jj)*e3u(ji,jj,jk,Kbb) * pu_in(ji,jj,jk,Kbb)',
        'e1v(ji,jj)*e3v(ji,jj,jk,Kbb) * pv_in(ji,jj,jk,Kbb)',
        2],
    'dynzdf.F90:296': 'zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) / e3u(ji,jj,iku,Kaa)',
    'dynzdf.F90:329-330': ['puu(ji,jj,1,Kaa) = puu(ji,jj,1,Kaa) + rDt * utauU(ji,jj)   &', '&                                    / ( e3u(ji,jj,1,Kaa) * rho0 ) * umask(ji,jj,1)', 2],
    'domqco.F90:166-169': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 1), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 4],
    'domqco.F90:219-222': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 2), 4],
    # ROUND 34: decision 17 added twelve lines above this one.
    # Round 34 inserted the ten-line carried-hf_0 selector before this site.
    'vertical.py:727': 'def nemo_qco_live_face_geometry_cgrid(',
    # --- round 31: the walk into dyn_zdf, and the stamp ---
    'dynzdf.F90:97': 'zDt_2 = rDt * 0.5_wp',
    'dynzdf.F90:148': 'IF( ln_drgimp .AND. ln_dynspg_ts ) THEN',
    'dynzdf.F90:149-150': [
        ('DO_2Dik( 0, 0,     1, jpkm1, 1 )      ! remove barotropic velocities', 1),
        ('puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)', 1),
        2],
    'dynzdf.F90:153': ('DO_1Di( 0, 0 )      ! Add bottom/top stress due to barotropic component only', 1),
    'stprk3_stg.F90:435': ('#if ! defined key_PSYCLONE_2p5p0', 5),
    'stprk3_stg.F90:440': 'zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)',
    'stprk3_stg.F90:446': ('END_3D', 6),
    'stp2d.F90:144-145': [
        ('!                             !*  COR + MET  *!   Flux Form        : Coriolis + Metric Term', 1),
        ('!                             !*     VOR     *!   Vector Inv. Form : Coriolis + relative Vorticity', 1),
        2],
    'usrdef_zgr.F90:133-137': [
        'zsur = -2033.194295283385_wp',
        'zacr =     5.0_wp',
        5],
    # ROUND 32 moved this site: stage 3 no longer corrects before the solve,
    # it defers the closure (stprk3_stg.F90:437-446 runs after :430).
    'ocean_model_latlon_cgrid.py:7768-7771': [
        ('u3_corr = u3_raw * _ws_stage_u_mask', 1),
        ('_replace_stage_mean, target_u, target_v)', 1),
        4],
    'ocean_model_latlon_cgrid.py:9572-9598': [
        ('if _ws_stage3_correction is not None:', 1),
        ('v=state_new.v.replace(data=_v_after),', 1),
        27],
    'stprk3_stg.F90:433': ('!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1', 1),
    'stprk3_stg.F90:440-441': [
        'zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)',
        'zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_0(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)',
        2],
    'stprk3_stg.F90:444-445': [
        'uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)',
        2],
    'ocean_model_latlon_cgrid.py:9407-9409': [
        '_nemo_ws_pre_implicit_state = (',
        'if self._nemo_ws_test_hooks.expose_pre_implicit_state else None)',
        3],
    'ocean_model_latlon_cgrid.py:11868': ('u_solve_in = u_solve_in - _u_bt_mean', 1),
    'ocean_model_latlon_cgrid.py:11987': ('u_solve_in = u_solve_in - (', 1),
    # ORCA2 round 61: private post-dyn_zdf/pre-barotropic raw-Kaa observer.
    'ocean_model_latlon_cgrid.py:1316': (
        'expose_stage3_raw_momentum: bool = False', 1),
    'ocean_model_latlon_cgrid.py:2818-2842': [
        '_stage3_rhs_hook = self._nemo_ws_test_hooks.expose_stage3_momentum_rhs',
        'expose_stage3_raw_momentum cannot be combined with another', 25],
    'ocean_model_latlon_cgrid.py:9565-9572': [
        'if self._nemo_ws_test_hooks.expose_stage3_raw_momentum:',
        'if _ws_stage3_correction is not None:', 8],
    'ocean_model_latlon_cgrid.py:9724-9728': [
        'if _nemo_ws_exposed_stage3_raw is not None:',
        ('v=state_new.v.replace(data=_raw_v),', 2), 5],
    # ORCA2 round 63: private write-only dyn_zdf seam observer.
    'ocean_model_latlon_cgrid.py:1317-1322': [
        '# WRITE-only observer for the four source-ordered stage-3 dyn_zdf',
        'zdf_momentum_observer: object = None', 6],
    'ocean_model_latlon_cgrid.py:2844-2848': [
        '_zdf_momentum_observer = (',
        'raise ValueError("zdf_momentum_observer must be callable or None")', 5],
    'ocean_model_latlon_cgrid.py:11185-11189': [
        ('_zc = self.z_coord if z_coord is None else z_coord', 11),
        '# Argument validation at ENTRY, not inside the drag branch below:', 5],
    'ocean_model_latlon_cgrid.py:11678-11681': [
        'u_new, v_new = state.u.data, state.v.data',
        'if do_momentum and getattr(_cfg_b, "surface_stress_implicit",', 4],
    'ocean_model_latlon_cgrid.py:11868-11871': [
        'u_solve_in = u_solve_in - _u_bt_mean',
        'u_solve_in, v_solve_in)', 4],
    'ocean_model_latlon_cgrid.py:11987-11993': [
        'u_solve_in = u_solve_in - (',
        '_zdf_baro_drag_u, _zdf_baro_drag_v = u_solve_in, v_solve_in', 7],
    'ocean_model_latlon_cgrid.py:12175-12182': [
        'if _zdf_momentum_observer is not None:',
        'ordered=True,', 8],
    'nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:171-230': [
        'def run(output: Path, expect_commit: str, entry_input: Path,',
        '"ordinary/observer commit mismatch")', 60],
    'nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:232-248': [
        'observer_rows = _noninterference(ordinary_arrays, observed)',
        'arrays[name] = R60._active(candidate, mask)', 17],
    'nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:280-292': [
        'planted = None',
        'R60.require(after == before + 1, "plant did not add one refusal")', 13],
    'ocean_pe_latlon_cgrid.py:3551-3554': [
        ('if not (getattr(grid, "dlon", 0.0) and grid.dlon > 0.0):', 1),
        ('"with a scalar dlon (got dlon<=0; tripolar unsupported)."', 1),
        4],
    # DECISION 17 moved the mask half of this guard below the early return,
    # so the round-31 anchors for it are re-anchored here rather than left to
    # resolve against lines that no longer say what the prose says.
    'vertical.py:66-71': [
        ('if e3t_0 is None:', 1),
        ('"literal NEMO QCO e3t requires explicit/reference nemo_e3t_0")', 1),
        6],
    'vertical.py:76-77': [
        ('if getattr(z_coord, "linear_free_surface", False):', 1),
        ('return e3t_0', 1),
        2],
    'vertical.py:85-89': [
        ('if active is None:', 1),
        ('tmask = jnp.asarray(active, dtype=dtype)', 1),
        5],
    # round 34
    'lock_kt1_10/ocean.output:556': 'ln_drg_OFF  =  T',
    'overflow_kt1_10/ocean.output:668': 'ln_drg_OFF  =  T',
    'stprk3_stg.F90:287':
        'zFw used in tracers only and computed in tra_adv_trp',
    'stprk3_stg.F90:295-301': [
        ('ELSE                                     !* Flux Form', 1),
        ('zFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 1),
        7],
    # the call occurs twice (the non-Shuman and Shuman arms); pin the first
    'stprk3_stg.F90:463': [
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
        1],
    # the field is named twice: GYRE's bundle at 226, the tanks' at 329.
    # ROUND (card reconciliation): the GYRE card's own freshwater selection
    # added eighteen lines above this anchor, so 311 became 329; decision 35
    # added nine lines above it and round 56 removed two; decision 36 added
    # fifteen more lines above it, so it is now 351.
    'nemo_testcase_recipe.py:655': [
        ('zdf_baroclinic_only=True,', 2), ('zdf_baroclinic_only=True,', 2), 1],
    'provenance.py:106': 'def git_sha(*, allow_dirty: bool = False, repo: str | Path | None = None) -> str:',
    'cpp_GYRE_BARE.fcm:1': 'key_linssh key_vco_1d  key_RK3',
    'cpp_GYRE_OMIP_L2_P3_SM.fcm:1': 'key_qco key_vco_1d3d key_RK3',
    'round19_oracle_v2_external/ocean.output:338': 'ice shelf cavities             ln_isfcav =  F',
    'round19_oracle_v2_external/ocean.output:553': 'Courant number targeted application   ln_zad_Aimp =  F',
    'round19_oracle_v2_external/ocean.output:629': 'implicit friction                         ln_drgimp   =  T',
    'round19_oracle_v2_external/ocean.output:630': 'implicit ice-ocean drag                   ln_drgice_imp  = F',
    'round19_oracle_v2_external/ocean.output:214':
        'single column domain (1x1pt)            ln_c1d      =  F',
    'round19_oracle_v2_external/ocean.output:546':
        'bdy_init : open boundaries not used (ln_bdy = F)',
    'round19_oracle_v2_external/ocean.output:559':
        'OSMOSIS-OBL closure (OSM)               ln_zdfosm =  F',
    'round19_oracle_v2_external/ocean.output:706': ('==>>>   iso-level laplacian operator', 1),
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:485': 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )     ! lateral mixing',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:498': 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration',
    # GYRE's own deck (round 29).  The round-28 premise was written as a bare
    # "namelist_cfg" token, which binds to LOCK's namelist -- where
    # ln_dynadv_vec is .false., the opposite of what the sentence claimed.
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:161': 'ln_dynadv_vec = .true.',
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:187-188':
        ['ln_dynldf_lap =  .true.', 'ln_dynldf_lev =  .true.', 2],
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:189-191':
        ['nn_ahm_ijk_t  = 0', 'rn_Lv      = 100.e+3', 3],
    'ocean.output:875': 'Barotropic time filter => nn_bt_flt',
    'lock_kt1_10/ocean.output:615': 'no explicit diffusion                ln_dynldf_OFF',
    'overflow_kt1_10/ocean.output:727': 'no explicit diffusion                ln_dynldf_OFF',
    'ocean_pe_latlon_cgrid.py:5640': 'rho_prime=rho_prime, h_k=h_k,',
    'ocean_pe_latlon_cgrid.py:5617-5618': [
        '_u_ldf_local = u if ldf_state is None else ldf_state[2]',
        '_v_ldf_local = v if ldf_state is None else ldf_state[3]',
        2],
    'ocean_pe_latlon_cgrid.py:2158-2159': [('z_coord,', 26), ('eta_safe,', 7), 2],
    'ocean_pe_latlon_cgrid.py:2199-2200': ['requires the raw NEMO',
         'nemo_e3w_0 mesh field; midpoint reconstruction on',
         2],
    # ROUND 34: decision 19 added seventeen lines above the last of these
    # three, so 914 became 931.  The two earlier anchors are unmoved.
    # ROUND (card reconciliation): the GYRE card's own freshwater selection
    # added eighteen lines above the SECOND and THIRD anchors, then decision
    # 36 added fifteen more, so the current lines are 314 and 971.  The first
    # (92) is above both edits and remains unmoved.  Round 7 of the ORCA2 lane
    # inserted the initial-state helper above the THIRD anchor only, moving it
    # 1411 -> 1493; later card additions moved all three to their current
    # merged locations without changing the anchor text.
    'nemo_testcase_recipe.py:370,615,2287': [('pgf_scheme="nemo_sco",', 1), ('pgf_scheme="nemo_sco",', 2), 'if cfg.pgf_scheme != "nemo_sco":', 3],
    'BLD/ppsrc/nemo/dynspg_ts.f90:1224': 'REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in   ) ::  puu, pvv',
    'BLD/ppsrc/nemo/dynhpg.f90:378,397': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  '
          '0)              ! Surface value',
          2),
         'DO jk= 2, jpkm1',
         2],
    'BLD/ppsrc/nemo/dynhpg.f90:393,412': ['puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap',
         'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap',
         2],
    'BLD/ppsrc/nemo/dynadv_up3.f90:138-141': ['IF( PRESENT( pUe ) ) THEN     ! 3D RHS cumulation : set 2D RHS '
         'to zero',
         ('END DO   ;   END DO', 1),
         4],
    'BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355': [('ELSE                           !-  added the 3D RHS  -!', 1), ('ELSE                           !-  added the 3D RHS  -!', 2), 'ELSE                                !-  added the 3D RHS  -!', 3],
    'BLD/ppsrc/nemo/dynadv_up3.f90:213,336,357': ['puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp', 'puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zFu_t(ji,jj) - zzFu_kp1 )', 'puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - zFu_t(ji,jj) * r1_e1e2u(ji,jj)', 3],
    'BLD/ppsrc/nemo/dynvor.f90:655-656': [('zwx(ji,jj) = e2u(ji,jj) * (e3t_1d(jk)', 2),
         ('zwy(ji,jj) = e1v(ji,jj) * (e3t_1d(jk)', 2),
         2],
    'BLD/ppsrc/nemo/dynvor.f90:665': ('pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav * ( zwz(ji  ,jj-1) '
         '+ zwz(ji,jj) )'),
    # --- round 94: consolidated stage-program contract ---
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168-190': [
        'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )',
        'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 23],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:195-222': [
        '!  RK3 time integration',
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 28],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:834-846': [
        '!* Swap',
        'sshn_e (:,:) = ssha_e(:,:)', 13],
    # --- round 116: admitted external-step writer and current executing loop ---
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:468-477': [
        'WRITE(l2_r81_filename',
        '& ssumask(ntsi:ntei,ntsj:ntej), ssvmask(ntsi:ntei,ntsj:ntej)', 10],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:522-572': [
        'WRITE(l2_uamid_unit) jn, za1, za2, za3',
        'WRITE(l2_r81_unit) zhup2_e(ntsi:ntei,ntsj:ntej), zhvp2_e(ntsi:ntei,ntsj:ntej)',
        51],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:603-615': [
        'zhdiv = (   ( zhU(ji,jj) - zhU(ji-1,jj) ) + ( zhV(ji,jj) - zhV(ji,jj-1) )  ) * r1_e1e2t(ji,jj)',
        '& ssha_e(ntsi:ntei,ntsj:ntej)', 13],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:670-713': [
        'WRITE(l2_r81_unit) za0, za1, za2, za3, zsshp2_e(ntsi:ntei,ntsj:ntej)',
        '& zu_frc(ntsi:ntei,ntsj:ntej), zv_frc(ntsi:ntei,ntsj:ntej)', 44],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:814-849': [
        'WRITE(l2_r81_unit) ua_e(ntsi:ntei,ntsj:ntej), va_e(ntsi:ntei,ntsj:ntej)',
        ('& sshn_e(ntsi:ntei,ntsj:ntej)', 2), 36],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:286-291': [
        '! set values computed in RK3_ssh',
        'zCdU_v  (:,:) = CdU_v   (:,:)', 6],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:298-321': [
        'CALL dyn_cor_2D_init( Kmm )',
        'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 24],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:456-531': [
        '!* Set extrapolation coefficients for predictor step:',
        'zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 76],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:542-553': [
        '!     Compute Sea Level at step jit+1',
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)',
        12],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:593-607': [
        '! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2',
        'zv_spg(ji,jj) = - zldg * ( zsshp2_e(ji,jj+1) - zsshp2_e(ji,jj) ) * r1_e2v(ji,jj)',
        15],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:611': [
        'CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )'],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:627-630': [
        'IF ( .NOT. ll_wd ) THEN',
        'zv_trd(ji,jj) = zv_trd(ji,jj) + zCdU_v(ji,jj) * vn_e(ji,jj) * hvr_e(ji,jj)',
        4],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:657-667': [
        'ua_e(ji,jj) = (                                 un_e(ji,jj)',
        '&   ) * ssvmask(ji,jj)', 11],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:701-730': [
        'IF( .NOT.lk_linssh ) THEN !* Update ocean depth',
        'CALL bdy_dyn2d( jn, ua_e, va_e, un_e, vn_e, hur_e, hvr_e, ssha_e )', 30],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:749-780': [
        '!* Swap',
        'pssh(:,:,Kaa) = pssh(:,:,Kaa) + za1 * ssha_e(:,:)', 32],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:797-804': [
        '! Finalize sums:',
        'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 8],
    # --- round 118: recovered direct producer record and complete source walk ---
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:147-179': [
        'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )',
        "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 33],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:222-230': [
        'CASE( np_VEC_c2, np_LIN_dyn )',
        'Ve_rhs(ji,jj) = Ve_rhs(ji,jj) + SUM(', 9],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:246-259': [
        'CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs, Ve_rhs, CdU_u, CdU_v )',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 14],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:290-345': [
        'ssh_frc(:,:) = sshe_rhs(:,:)',
        "WRITE(numout,*) 'ROUND117_PRELOOP_FORCING_DUMP '", 56],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:329-345': [
        'WRITE(l2_r117_unit) puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_frc, zv_frc',
        "WRITE(numout,*) 'ROUND117_PRELOOP_FORCING_DUMP '", 17],
    # --- round 119: stage-1 source order and first production-only boundary ---
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynldf_lev.f90:121-141': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 )', 1),
        ('END DO   ;   END DO', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:117-131': [
        'CASE ( nkeg_C2 )', ('END DO   ;   END DO', 2), 15],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:129-130': [
        ('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs)', 1),
        ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs)', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynadv.f90:339-345': [
        'IF( ln_dynadv_OFF  ) THEN',
        "IF( nn_dynkeg /= nkeg_C2 .AND. nn_dynkeg /= nkeg_HW )", 7],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/namelist_cfg:161-162': [
        'ln_dynadv_vec = .true.', 'nn_dynkeg     = 0', 2],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/ocean.output:780-781': [
        'Vector form: 2nd order centered scheme',
        'with Hollingsworth scheme (=1) or not (=0)', 2],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:278-324': [
        ('ALLOCATE( zub(', 1),
        ('DEALLOCATE( zub, zvb )', 1), 47],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:792-819': [
        ('IF( ln_shuman ) THEN', 1),
        ('WRITE(l1_unit) zFu, zFv, zFw', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:843-853': [
        ('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) THEN', 1),
        ('WRITE(l1_unit) zFu, zFv, zFw', 3), 11],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90:247-250': [
        ('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 )', 2),
        'pFw(ji,jj,jpk) = 0._wp', 4],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90:253-280': [
        'IF( kstg == 3 ) THEN',
        'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/l2_r46_stage.f90:120-147': [
        "IF(ios /= 0) CALL ctl_stop('round46: cannot open stage record')",
        "CALL put2('r3v_Kaa         ',r3v(:,:,Kaa))", 28],
    # --- round 95: complete the stage-program contract ---
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:792-834': [
        'IF( ln_shuman ) THEN                     ! Shuman averaging-',
        "WRITE(numout,*) 'ROUND94_STAGE_CLOSURE_DUMP '", 43],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:200-226': [
        '! Stage 1 :',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 27],
    # --- round 115: recorded/current compiled U-geometry boundary ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:193-200': [
        'ht_0(:,:) = 0._wp  ! Reference ocean thickness',
        'hv_0(:,:) = hv_0(:,:) + e3v_3d(:,:,jk) * vmask(:,:,jk)', 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:212-215': [
        'r1_ht_0(:,:) = ssmask (:,:) /',
        'r1_hf_0(:,:) = ssfmask(:,:) /', 4],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90:155-160': [
        'e1e2t (:,:) = e1t(:,:) * e2t(:,:)',
        'e1e2v (:,:) = e1v(:,:) * e2v(:,:)', 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90:170-171': [
        'r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)',
        'r1_e1e2v(:,:) = 1._wp / e1e2v(:,:)', 2],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domqco.f90:266-268': [
        ('pr3u(ji,jj) = 0.5_wp *', 2),
        ('pr3v(ji,jj) = 0.5_wp *', 2), 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:188-201': [
        '!  RK3 : single first external mode computation',
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-178': [
        ('SELECT CASE( kstg )', 1),
        'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 41],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:188-190': [
        'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:228-235': [
        ('ssh (:,:,Kaa) = r1_2 * ( ssh (:,:,Kbb) + ssha(:,:) )', 2),
        ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:240': (
        'CASE ( 3 )           !==  Stage 3  ==!', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:301-308': [
        '!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)',
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    # --- round 138: developed-state external solve and consumed QCO boundary ---
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90:291-298': [
        '!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)',
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325': [
        '! set values computed in RK3_ssh',
        ('END DO   ;   END DO', 1), 37],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:1336-1359': [
        ('SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )', 1),
        ('END SUBROUTINE dyn_cor_2D', 2), 24],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:462-591': [
        'DO jn = 1, icycle',
        '& ssha_e(ntsi:ntei,ntsj:ntej)', 130],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:631-684': [
        '! Half-step back interpolation of SSH for surface pressure',
        '& zu_frc(ntsi:ntei,ntsj:ntej), zv_frc(ntsi:ntei,ntsj:ntej)', 54],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:697-759': [
        '! Set next velocities:',
        'hvr_e(ji,jj) = ssvmask(ji,jj) / (  hv_e(ji,jj)', 63],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:811-844': [
        'vbb_e  (:,:) = vb_e  (:,:)',
        'END DO                                               !        end loop', 34],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:857-890': [
        '! Finalize sums:',
        'WRITE(r137_unit) pssh(ntsi:ntei,ntsj:ntej,Kaa)', 34],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90:176-205': [
        '!==  ssh/h0 ratio at Kaa  ==!',
        ('r3u(:,:,Kaa) = r3ua(:,:)', 1), 30],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/domqco.f90:237-258': [
        ('SUBROUTINE dom_qco_r3c_RK3', 1),
        ('END DO   ;   END DO', 4), 22],
    # --- round 140: admitted developed slow forcing and upstream source order ---
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:292-347': [
        '! set values computed in RK3_ssh',
        "WRITE(numout,*) 'ROUND139_SLOW_FORCING_DUMP '", 56],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:1359-1382': [
        ('SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )', 1),
        ('END SUBROUTINE dyn_cor_2D', 2), 24],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:139-228': [
        '!*  hydrostatic pressure gradient (HPG))  *!',
        ("WRITE(l2_slow_unit) Ue_rhs, Ve_rhs", 3), 90],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:297-298': [
        'IF( ln_dynspg_ts )',
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 2],
    # --- round 141: admitted completed 3-D RHS and its compiled consumers ---
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169': [
        '!*  hydrostatic pressure gradient (HPG))  *!',
        'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 28],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:182-192': [
        ('IF( lwp .AND. kt == 1081 ) THEN', 1),
        'WRITE(r140_slow_unit) e3u_3d, uu(:,:,:,Krhs), umask, e3v_3d, vv(:,:,:,Krhs), vmask', 11],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228': [
        '!*  vertical averaging  *!',
        'WRITE(r140_slow_unit) Ue_rhs, Ve_rhs, r1_hu_0, r1_hv_0', 19],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-260': [
        '!* baroclinic drag forcing *!',
        "WRITE(numout,*) 'ROUND140_DEVELOPED_RHS_DUMP '", 25],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-239': [
        '!* baroclinic drag forcing *!',
        'WRITE(r140_slow_unit) Ue_rhs, Ve_rhs, CdU_u, CdU_v', 4],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-260': [
        '!* wind forcing *!',
        "WRITE(numout,*) 'ROUND140_DEVELOPED_RHS_DUMP '", 20],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-246': [
        '!* wind forcing *!',
        'WRITE(r140_slow_unit) r1_rho0, utauU, vtauV', 6],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192': [
        '!*  hydrostatic pressure gradient (HPG))  *!',
        "WRITE(numout,*) 'ROUND146_DEVELOPED_RHS_FAMILIES_DUMP '", 48],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-130': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1),
        ('END DO   ;   END DO', 1), 10],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:132-140': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 9],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/dynspg_ts.f90:1460-1493': [
        'SUBROUTINE dyn_drg_init( Kbb, Kmm, puu, pvv, puu_b ,pvv_b, pu_RHSi, pv_RHSi, pCdU_u, pCdU_v )',
        'pCdU_v(ji,jj) = r1_2*( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) )', 34],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:248-251': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 3),
        ('END DO   ;   END DO', 4), 4],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:252-260': [
        ('IF( lwp .AND. kt == nit000 ) THEN', 4),
        "WRITE(numout,*) 'ROUND140_DEVELOPED_RHS_DUMP '", 9],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:249-250': [
        'Ue_rhs(ji,jj) =  Ue_rhs(ji,jj) + r1_rho0 * utauU(ji,jj)',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 2],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:180-192': [
        '! Round 140: repeat the established slow-forcing operand list at',
        'WRITE(r140_slow_unit) e3u_3d, uu(:,:,:,Krhs), umask, e3v_3d, vv(:,:,:,Krhs), vmask', 13],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-250': [
        '!* baroclinic drag forcing *!',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 15],
    # --- round 144: developed wind operand walk ---
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:160-182': [
        'SUBROUTINE dom_qco_zgr( Kbb, Kmm )',
        "CALL lbc_lnk( 'dom_qco_zgr', r3u(:,:,Kbb)", 23],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:188-218': [
        'SUBROUTINE dom_qco_r3c( pssh, pr3t, pr3u, pr3v, pr3f )',
        ('END DO   ;   END DO', 2), 31],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90:212-215': [
        'r1_ht_0(:,:) = ssmask',
        'r1_hf_0(:,:) = ssfmask', 4],
    'ocean_model_latlon_cgrid.py:5565-5620': [
        '_ws_face_thickness_kbb = None',
        ('_ws_v_live_mask, _grid)[:2]', 1), 56],
    'ocean_model_latlon_cgrid.py:5836-5877': [
        ('if _sfx is not None:', 1),
        ('F_slow_v + _wind_increment_v) * state.v_mask.data', 1), 42],
    # --- round 186: admitted pre-day-180 process stream and qsr promotion ---
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830': [
        'lr123_write = lwp .AND. .NOT.ln_tile .AND. kstg == 3 .AND.',
        'WRITE(r123_unit) r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa)', 13],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869': [
        ('CALL tra_adv', 3),
        ('WRITE(r123_unit) ts(:,:,:,jp_tem,Krhs)', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-970': [
        'IF( ln_traqsr  ) THEN',
        "CALL ctl_stop( 'stp_RK3_stg: cannot close round-185 process dump' )", 41],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-945': [
        'IF( ln_traqsr  ) THEN', 'DEALLOCATE( l2_qsr_before )', 16],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/traqsr.f90:615-645': [
        'zz0 =           rn_abs   * r1_rho0_rcp',
        ('zatt(ji,jj) = zzatt', 2), 31],
    # --- round 187: reconcile the acquired developed qsr record layout ---
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:925-931': [
        "r186_magic = 'NEMO_L2_R186QSR1'",
        '& ts(:,:,:,jp_tem,Krhs)-r186_qsr_before(:,:,:), r186_qsr_replay', 7],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/sbc_oce.f90:210-212': [
        'ALLOCATE( qns    (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))',
        '&      STAT=ierr(5) )', 3],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/mppini.f90:1501-1507': [
        'Nis0 =   1+nn_hls',
        'Nj_0 = Nje0 - Njs0 + 1', 7],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/nemogcm.f90:368':
        'Nbb = 1   ;   Nnn = 2   ;   Naa = 3   ;   Nrhs = Naa',
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3.f90:200-222': [
        '! Stage 1 :',
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 23],
    # --- round 188: compiled-order developed qsr statement walk ---
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645': [
        'zz0 =           rn_abs   * r1_rho0_rcp',
        ('pts(ji,jj,jk,jp_tem,Krhs) = pts(ji,jj,jk,jp_tem,Krhs)', 11), 30],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:1021-1070': [
        'FUNCTION qsr_ext_lev( pL, pfr ) RESULT( klev )',
        'END FUNCTION qsr_ext_lev', 50],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:1145-1213': [
        'r1_si0 = 1._wp / rn_si0',
        'nkV  = qsr_ext_lev( rn_si1, zVlp )', 69],
    # --- round 189: inherited stage-3 shortwave stretch attribution ---
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:172-192': [
        '!                     !==  ssh/h0 ratio at Kaa  ==!',
        ('END SELECT', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:227-237': [
        ('CASE ( np_HYB )', 2), ('ENDIF', 5), 11],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/domqco.f90:237-258': [
        'SUBROUTINE dom_qco_r3c_RK3( pssh, pr3t, pr3u, pr3v, pr3f )',
        ('END DO   ;   END DO', 4), 22],
    # --- round 190: split the production QSR observer association ---
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:911-950': [
        '!           !==  complete the tracers RHS  ==!   except ZDF (implicit)',
        'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )  ! lateral mixing', 40],
    'ocean_model_latlon_cgrid.py:7343-7391': [
        ('if _return_tracer_process_trace:', 1),
        'process_qsr_rate=_nemo_ws_process_qsr_rate,', 49],
    # --- ORCA2 round 101: sufficient owner of the merged GYRE year move ---
    'ocean_model_latlon_cgrid.py:7054-7064': [
        'nemo_r3t_rk3_stage1_stretch,',
        '_qt_13 = nemo_r3t_rk3_stage1_stretch(', 11],
    'eos.py:970-1001': [
        'def nemo_r3t_rk3_stage1_stretch(',
        'return jnp.where(wet, nemo_source_round(one + r3_stage), one)', 32],
    # --- restored from the lane tip (980cc6369) after the merge conflict on
    # this file was resolved to HEAD's side, which silently dropped every
    # entry the VORTEX-card rounds had added.  None of these NEMO/VORTEX
    # sources shifted in the merge, so the line numbers are unchanged.
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:359-361': [('zt  = pts  (ji,jj,jk,jp_tem,Knn) - rn_T0', 1), ('zh  = ((gdept_1d(jk) ) *(1._wp+r3t(ji,jj,Knn)))', 2), 3],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:364-366': [('zn =  - rn_a0 * ( 1._wp + 0.5_wp*rn_lambda1*zt + rn_mu1*zh ) * zt', 1), ('&  - rn_nu * zt * zs', 1), 3],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1217': ('zn  = rn_a0 * ( 1._wp + rn_lambda1*zt + rn_mu1*zh ) + rn_nu*zs', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1220': ('zn  = rn_b0 * ( 1._wp - rn_lambda2*zs - rn_mu2*zh ) - rn_nu*zt', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:100-101': ['REAL(wp), PUBLIC ::   rn_T0      = 10._wp', 'REAL(wp), PUBLIC ::   rn_S0      = 35._wp', 2],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:1938': ('NAMELIST/nameos/ ln_TEOS10, ln_EOS80, ln_SEOS', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:849': ('IF( ln_dynvor_een ) THEN', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:865-868': ['CASE( np_FLX_c2 , np_FLX_up3 )', 'ntot = np_CME', 4],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:256-257': [('CASE( np_EEN )', 3), ('CALL vor_een( kt, Kmm, ntot,', 1), 2],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:725-727': [('CASE ( np_COR )', 4), ('zwz(ji,jj) = ff_f(ji,jj) * z1_e3f(ji,jj)', 1), 3],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:755-758': [('CASE ( np_CME )', 4), ('dj_e1u_2e1e2f(ji,jj)   ) * z1_e3f(ji,jj)', 2), 4],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:882-883': ['di_e2v_2e1e2f(ji,jj) = ( e2v(ji+1,jj  ) - e2v(ji,jj) )', 'dj_e1u_2e1e2f(ji,jj) = ( e1u(ji  ,jj+1) - e1u(ji,jj) )', 2],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:720': ('z1_e3f(ji,jj) = 1._wp / (e3f_0vor(ji,jj,jk)', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:766-779': [('zwx(ji,jj) = e2u(ji,jj) *', 3), 'zua = + r1_12 * r1_e1u(ji,jj)', 14],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynspg_ts.f90:955-968': ['SELECT CASE( nvor_scheme )', ('& ff_f(ji-1,jj-1) / (e3f_0vor(ji-1,jj-1,jk)', 1), 14],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:158-161': ['pe1t(:,:) = rn_dy', 'pe1f(:,:) = rn_dy', 4],
    'vortex_round2/namelist_cfg:128-138': ['&nameos', 'rn_nu       =  0.', 11],
    # --- VORTEX round 3 (decision 73): the vector-EEN card, and the walk that
    # attributes the flux card's second step. ---
    'VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:184-190': [('ioptio = 0                      ! parameter control and set n_dynadv', 1), ("IF( ioptio /= 1 )   CALL ctl_stop( 'choose ONE and only ONE advection scheme' )", 1), 7],
    'VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:135-138': [('CASE( np_VEC_c2  )                                                         != vector form =!', 1), ('CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection', 1), 4],
    'VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:861-864': [('CASE( np_VEC_c2  )', 1), ('ntot = np_CRV        ! relative + planetary vorticity', 1), 4],
    # --- round 4: the boundary the kt=2 walk names ---
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:312-316': [
        'CASE ( 1 )        !==  Stage 1  ==!',
        'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 5],
    # --- VORTEX round 8: stage-2/3 compiled momentum execution order. ---
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:287-293': [
        'IF( ln_dynadv_vec ) THEN                 !* Vector invariant Form : no use of ww at stage 1 as 3D RHS computed in stp_2D',
        'IF( ln_zad_Aimp .AND. kstg == 3 )   CALL wAimp( kstp, Kmm, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, wi, np_velocity, ld_diag=.TRUE. )', 7],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:318-332': [
        '!                 !-------------------!   Flux Form        : HPG + VOR (COR+MET) + ADV',
        'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 15],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-138': [
        'SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!',
        'CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection', 5],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:363-370': [
        'CASE ( 1 , 2 )    !==  Stage 1 & 2  ==!   time stepping',
        ('END DO   ;   END DO   ;   END DO', 4), 8],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:387-405': [
        'CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping',
        'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration', 19],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:408-419': [
        '!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1',
        ('END DO   ;   END DO   ;   END DO', 6), 12],
    # --- round 193: exact lines in the acquired R8 build after its additive
    # writer shifted stprk3_stg; HPG is a write, then VOR and KEG/ZAD follow.
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:323-338': [
        'CASE ( 2 , 3 )    !==  Stage 2 & 3  ==!',
        'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 16],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:135-141': [
        'SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!',
        "CALL vortex_r8_stage_rhs( 'zad', Krhs, puu, pvv )", 7],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:320-334': [
        'puu(ji,jj,1,Krhs) = zhpi(ji,jj)     ! RK3 case: dyn_hpg always called first',
        ('pvv(ji,jj,jk,Krhs) = zhpj(ji,jj)', 1), 15],
    # --- round 196: the split-explicit barotropic sub-time-step loop ---
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:359':
        'DO jn = 1, icycle                             !  sub-time-step loop  !',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:389':
        'ua_e(ji,jj) = za1 * un_e(ji,jj) + za2 * ub_e(ji,jj) + za3 * ubb_e(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:401':
        'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:414':
        'zhup2_e(ji,jj) = hu_0(ji,jj) + r1_2 * r1_e1e2u(ji,jj)                        &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:434':
        'zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:454':
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:463':
        'un_adv(ji,jj) = un_adv(ji,jj) + za2 * zhU(ji,jj) * r1_e2u(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:478':
        'zsshu_a(ji,jj) = r1_2 * r1_e1e2u(ji,jj) * ( e1e2t(ji  ,jj  ) * ssha_e(ji  ,jj  )   &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:489':
        'CALL ts_bck_interp( jn, ll_init, za0, za1, za2, za3 )   ! coeficients of the interpolation',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:491':
        'zsshp2_e(ji,jj) = za0 *  ssha_e(ji,jj) + za1 *  sshn_e (ji,jj)   &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:498':
        'zu_spg(ji,jj) = - zldg * ( zsshp2_e(ji+1,jj) - zsshp2_e(ji,jj) ) * r1_e1u(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503':
        'CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )',
    # Round 197: the Coriolis routine itself, the routine that freezes its
    # coefficients, the statement that divides ff_f by e3f_0vor, and
    # e3f_0vor's own construction in dyn_vor_init.
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:933':
        'SUBROUTINE dyn_cor_2D_init( Kmm )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960':
        'zpvo_nw = ff_f(ji-1,jj  ) / (e3f_0vor(ji-1,jj  ,jk) '
        '*(1._wp+r3f(ji-1,jj  )*fe3mask(ji-1,jj  ,jk))) + &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1112':
        'SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:892':
        'ALLOCATE( e3f_0vor(jpi,jpj,jpk) )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1016':
        'ffu_nw(ji,jj) = ffu_nw(ji,jj) + (e3t_1d( jk) '
        '*(1._wp+r3u(ji  ,jj, Kmm)*umask(ji  ,jj, jk))) * (e3t_1d( jk) '
        '*(1._wp+r3v(ji  ,jj  , Kmm)*vmask(ji  ,jj  , jk))) '
        '* vmask(ji  ,jj  ,jk) / (e3f_0vor(ji,jj  ,jk) '
        '*(1._wp+r3f(ji,jj  )*fe3mask(ji,jj  ,jk)))',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1046':
        'zr1_e3f = ff_f(ji,jj) / (e3f_0vor(ji,jj  ,jk) '
        '*(1._wp+r3f(ji,jj  )*fe3mask(ji,jj  ,jk))) + ff_f(ji,jj-1) '
        '/ (e3f_0vor(ji,jj-1,jk) '
        '*(1._wp+r3f(ji,jj-1)*fe3mask(ji,jj-1,jk)))',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:890':
        'CASE( np_ENS , np_ENE , np_EEN , np_MIX )',
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:167':
        'ln_dynvor_ene = .true.  !  energy conserving scheme',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:897': (
        'e3f_0vor(ji,jj,jk) = (   ( e3t_1d(jk)*tmask(ji  ,jj+1,jk)     &   '
        '! need additional () for', 1),
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:920':
        'WHERE( e3f_0vor(:,:,jk) == 0._wp )   e3f_0vor(:,:,jk) = e3t_1d(jk)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:517':
        'zu_trd(ji,jj) = zu_trd(ji,jj) + zCdU_u(ji,jj) * un_e(ji,jj) * hur_e(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:535':
        'ua_e(ji,jj) = (                                 un_e(ji,jj)   &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:581':
        'hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:628':
        ('puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) + za1 * ua_e  (:,:)', 1),
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:648':
        'un_adv(:,:) = un_adv(:,:) / r1_wgt2s',
    # --- round 195: the barotropic operand the per-stage correction adds ---
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:133-135': [
        'ssha(:,:) = ssh (:,:,Kaa)     ! save ssh, uu_b, vv_b at N+1  '
        '(computed in dynspg_ts)',
        'va_b(:,:) = vv_b(:,:,Kaa)', 3],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:421-429': [
        'ALLOCATE( zub(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)), '
        'zvb(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) )',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 9],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:289-300': [
        ('IF( ln_dynadv_vec ) THEN', 1),
        'CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )', 12],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/divhor.f90:123-140': [
        'SELECT CASE ( ik_ind )', 'END SELECT', 18],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:271-299': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 2),
        ('END DO   ;   END DO   ;   END DO', 6), 29],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynzad.f90:105-137': [
        'DO jk =  1,  jpk-2',
        '&                                              * zWdzV(ji,jj)', 33],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traadv.f90:268-273': [
        'CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )',
        ('END DO   ;   END DO   ;   END DO', 3), 6],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:137-163': [
        'CALL eos    ( ts, Kbb, rhd )                          ! in situ density anomaly at Kbb',
        'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )                 !- vertical advection', 27],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:176-179': [
        'CASE( np_VEC_c2, np_LIN_dyn )       ! Vector Inv. Form   ==>> averaged 3D RHS only',
        'Ve_rhs(ji,jj) = SUM( e3t_1d(1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmask(ji,jj,1:jpkm1) ) * r1_hv_0(ji,jj)', 4],
    'vortex_round3/namelist_cfg:182-185': [('ln_dynadv_vec = .true.  !  decision 73: ORCA2/GYRE vector-invariant momentum', 1), ('ln_dynadv_up3 = .false. !  decision 73: exactly one advection form (dynadv.F90:190)', 1), 4],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:239': ('r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )   ! at N+1/2 (Kmm)', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:405': ('IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:412-418': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)             ! barotropic velocity correction', 1), ('vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 1), 7],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:143-147': [('ELSE                                      ! applied on thickness weighted velocity', 1), ('&              /          ( 1._wp + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)', 1), 5],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:159-161': [('DO jk =      1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)      ! remove barotropic velocities', 1), ('pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 1), 3],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:191-195': [('DO jk =    2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)            ! inner values', 1), ('&           / ( (e3t_1d(jk  ) *(1._wp+r3u(ji  ,jj,Kaa)*umask(ji  ,jj,jk  ))) * (e3w_1d(jk+1) *(1._wp+r3u(ji,jj,Kmm))) ) * wumask(ji,jj,jk+1)', 1), 5],
    'tests/VORTEX/MY_SRC/usrdef_nam.F90:96-97': [
        'kpi = NINT( 1800.e3  / rn_dx ) + 3',
        'kpj = NINT( 1800.e3  / rn_dy ) + 3',
        2],
    'tests/VORTEX/MY_SRC/usrdef_nam.F90:121': [
        'kpk = NINT( 5000._wp / rn_dz ) + 1'],
    'tests/VORTEX/MY_SRC/usrdef_hgr.F90:83-84': [
        'zroffsetx = (-REAL(Ni0glo-1, wp) + 1._wp) * 0.5_wp * 1.e-3 * rn_dx',
        'zroffsety = (-REAL(Nj0glo-1, wp) + 1._wp) * 0.5_wp * 1.e-3 * rn_dy',
        2],
    'tests/VORTEX/MY_SRC/usrdef_hgr.F90:174-177': [
        'zbeta = 2._wp * omega * COS( rad * rn_ppgphi0 ) / ra',
        'pff_t(:,:) = zf0 + zbeta * pphit(:,:) * 1.e+3',
        4],
    'tests/VORTEX/MY_SRC/usrdef_zgr.F90:126': [
        'zd = 5000._wp/REAL(jpkm1,wp)'],
    'tests/VORTEX/MY_SRC/usrdef_zgr.F90:187-193': [
        'z2d(:,:) = REAL( jpkm1 , wp )          ! flat bottom',
        'k_top(:,:) = MIN( 1 , k_bot(:,:) )     ! = 1    over the ocean point, =0 elsewhere',
        7],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:69-75': [
        ('zf0   = 2._wp * omega * SIN( rad * rn_ppgphi0 )', 1),
        ('zP0 = rho0 * zf0 * zumax * zlambda * SQRT(EXP(1._wp)/2._wp)', 1),
        7],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:83-88': [
        'zrho1 = rho0 * (1._wp + zn2*zdt/grav)',
        'pts(ji,jj,jk,jp_tem) = (20._wp + (rho0-zrho1) / rn_a0 ) * ptmask(ji,jj,jk)',
        6],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:101-105': [
        'zdu = 0.5_wp * (pdept(ji  ,jj,jk) + pdept(ji+1,jj,jk))',
        'pu(ji,jj,jk) = (za * zf * zy * EXP(-(zx**2+zy**2)/zlambda**2)) * ptmask(ji,jj,jk) * ptmask(ji+1,jj,jk)',
        5],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:177-182': [
        'za = -zP0 * (1._wp-EXP(-zH)) / (grav*(zH-1._wp + EXP(-zH)))',
        'pssh(ji,jj) = zP0 * EXP(-(zx**2+zy**2)/zlambda**2)/(zrho1*grav) * ptmask(ji,jj,1)',
        6],
    'tests/VORTEX/MY_SRC/usrdef_sbc.F90:60-68': [
        'utau(:,:) = 0._wp',
        'qsr (:,:) = 0._wp',
        9],
    'restart.F90:461': [
        ('CALL usr_def_istate_ssh( tmask, ssh(:,:,Kbb) )', 2)],
    'DOM/istate.F90:127-130': [
        'DO jk = 1, jpk',
        'CALL usr_def_istate( zgdept, tmask, ts(:,:,:,:,Kbb), uu(:,:,:,Kbb), vv(:,:,:,Kbb) )',
        4],
    'DOM/istate.F90:149-154': [
        'uu_b(:,:,Kbb) = 0._wp   ;   vv_b(:,:,Kbb) = 0._wp',
        'uu_b(:,:,Kbb) = uu_b(:,:,Kbb) * r1_hu(:,:,Kbb)',
        6],
    'eosbn2.F90:1890-1895': [
        'NAMELIST/nameos/ ln_TEOS10, ln_EOS80, ln_SEOS, rn_T0, rn_S0, rn_a0, rn_b0, rn_lambda1, rn_mu1, &',
        'READ_NML_CFG(numnam,nameos)',
        6],
    'dynvor.F90:874': [
        'IF( ln_dynvor_een ) THEN   ;   ioptio = ioptio + 1   ;   nvor_scheme = np_EEN   ;   ENDIF'],
    # --- gating the DINO month-regression and PR-1802 review-fix
    # receipts (both previously certified only by hand): every citation
    # verified against the real file, anchor = the exact statement text.
    'stprk3.F90:213': ('Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs    ! Swap: Nnn unchanged, Nbb <==> Naa', 1),
    'cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90:185': ('CALL stp_MLF( istp )', 1),
    'cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90:489-491': [('sshn_e(:,:) =    pssh (:,:,Kbb)', 1), ('vn_e  (:,:) =    pvv_b(:,:,Kbb)', 1), 3],
    'cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90:500-503': [('! DINO has ln_bt_fw=F -> the CENTRED branch above just ran, so this is', 1), ('IF( ll_spg_dump ) THEN', 6), 4],
    'src/OCE/DOM/domqco.F90:165-170': [('DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )', 1), ('END_2D', 2), 6],
    'src/OCE/DOM/domqco.F90:134-135': [("CALL lbc_lnk( 'dom_qco_zgr', r3u(:,:,Kbb), 'U', 1._wp, r3v(:,:,Kbb), 'V', 1._wp, &", 1), ("&                         r3u(:,:,Kmm), 'U', 1._wp, r3v(:,:,Kmm), 'V', 1._wp, r3f(:,:), 'F', 1._wp )", 1), 2],
    'lbc_lnk_pt2pt_generic.h90:49': ('zland = 0._wp                                     ! land filling value: zero by default', 1),
    'lbc_lnk_pt2pt_generic.h90:104': ('ELSE                                ;   ifill(jn,jf) = jpfillcst       ! constant value (zland)', 1),
    'lbc_lnk_pt2pt_generic.h90:305': ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = zland', 1),
    'DINO/BLD/ppsrc/nemo/domqco.f90:211-215,177-181': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 1), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), ('CALL dom_qco_r3c( ssh(:,:,Kbb), r3t(:,:,Kbb), r3u(:,:,Kbb), r3v(:,:,Kbb)           )', 1), ("&                         r3u(:,:,Kmm), 'U', 1._wp, r3v(:,:,Kmm), 'V', 1._wp, r3f(:,:), 'F', 1._wp )", 1), 10],
    'lbclnk.f90:1816-1820,1866-1871,2130-2135': [('zland = 0._wp                                     ! land filling value: zero by default', 3), ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), ('DO jn = 1, 4   ! 4 sides', 2), ('ELSE                                ;   ifill(jn,jf) = jpfillcst       ! constant value (zland)', 3), ('IF(     ifill(jn,jf) == jpfillcst ) THEN', 5), ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = zland', 5), 17],
    'nemo_state_bridge.py:275,917': [('nemo_prognostic_barotropic_velocity=True,', 1), ('nemo_prognostic_barotropic_velocity=True,', 2), 2],
    'nemogcm.f90:185': ('CALL stp_MLF( istp )', 1),
    'dynspg_ts.f90:489-491': [('sshn_e(:,:) =    pssh (:,:,Kbb)', 1), ('vn_e  (:,:) =    pvv_b(:,:,Kbb)', 1), 3],
    'zdftke.F90:845-846': [('ELSE                          ! standard case : associated avt minimum = molecular viscosity (10^-6 m2/s)', 1), ('rmxl_min = 1.e-6_wp / ( rn_ediff * SQRT( rn_emin ) )    ! resulting minimum length to recover molecular viscosity', 1), 2],
    'zdftke.F90:841-843': [('IF( ln_zdfiwm ) THEN          ! Internal wave-driven mixing', 1), ('rmxl_min = 1.e-03_wp             ! associated avt minimum = molecular salt diffusivity (10^-9 m2/s)', 1), 3],
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:396': ('ln_zdfiwm   = .true.       ! internal wave-induced mixing            (T =>   fill namzdf_iwm)', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:90-104': [
        ('! Lane-1 certified oracle: exact step-entry Nbb state.  This is a', 1),
        ("WRITE(numout,*) 'LANE1_STEP_ENTRY_DUMP ', kstp, Nbb, STORAGE_SIZE(1._wp), TRIM(cl_traj)", 1),
        15,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227': [
        ('! Stage 1 :', 1),
        ('IF( kstp <= nit000 + 9 )   CALL l1_dump_stage( kstp, 3, Naa )', 1),
        17,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:91-107': [
        ('! Lane-1 certified oracle: exact step-entry Nbb state.  This is a', 1),
        ('CALL r84_dump_frame( kstp, 0, Nbb )', 1),
        17,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:147-154': [
        ('! Update external forcing (tides, open boundaries, ice shelf interaction and surface boundary condition (including sea-ice)', 1),
        ('IF( kstp == nit000 )   CALL l4_dump_ocean_surface_input( kstp, Nbb )', 1),
        8,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:390-441': [
        ('SUBROUTINE l4_dump_ocean_surface_input( kstp, klevel )', 1),
        ('END SUBROUTINE l4_dump_ocean_surface_input', 1),
        52,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:443-473': [
        ('FUNCTION l4_canon_2d( pfield, cdgrid ) RESULT( zfield )', 1),
        ('END FUNCTION l4_canon_2d', 1),
        31,
    ],
    'namelist_ref:1200': ('ln_zdfiwm   = .false.      ! internal wave-induced mixing            (T =>   fill namzdf_iwm)', 1),
    'domhgr.F90:222-227': [("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1), ('kff = 1', 1), 6],
    'zdftke.F90:246,253-258': [('IF( nn_eice == 0 ) zice_fra(:) = 0._wp               ! No attenuation of TKE due to sea ice', 1), ('! ice fraction considered for attenuation of langmuir & wave breaking', 1), ('END SELECT', 1), 7],
    'zdftke.F90:828-835': [('SELECT CASE( nn_eice )', 1), ('END SELECT', 7), 8],
    'zdftke.F90:601-603': [('DO_1Di( 0, 0 )                  ! No sea-ice', 1), ('END_1D', 16), 3],
    'zdftke.F90:859-862': [('IF( ln_mxl0 ) THEN', 4), ('rn_mxl0 = rmxl_min', 2), 4],
    'geo2ocean.F90:168,259-260': [('REAL(wp) ::   zxffu, zyffu, znffu   ! x,y components and norm of the vector: between F points below and above a U point', 1), ('gsinu(ji,jj) = ( zxnpu*zyffu - zynpu*zxffu ) / znffu', 1), ('gcosu(ji,jj) = ( zxnpu*zxffu + zynpu*zyffu ) / znffu', 1), 3],
    # --- VORTEX round 6: NEMO's carried after-SSH slot ('ssha') ---
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225': [
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90:222-226': [
        'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs',
        'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:149':
        'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)',
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:153':
        'CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:152':
        'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:156':
        'CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )',
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:184':
        "CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )",
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:362-370': [
        "id1 = iom_varid( numror, 'ssha', ldstop = .FALSE. )",
        ('ssh(:,:,Kaa) = ssh(:,:,Kbb)', 1), 9],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:295-298': [
        ('integrate from the bottom the hor. divergence', 4),
        ('r1_Dt * e3t_1d(jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2), 4],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/sshwzv.f90:295-298': [
        ('integrate from the bottom the hor. divergence', 4),
        ('r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2), 4],
    'domhgr.F90:222-223': [("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1), ("& iom_varid( inum, 'ff_t', ldstop = .FALSE. ) > 0    ) THEN", 1), 2],
}


DEFAULT_RECEIPT = (
    REPO / "docs/ocean/fidelity/testcases"
    / "nemo_testcases_l2_gyre_phase3_round8_receipt.md")
DEFAULT_HEADING = "## Round 25 —"

# COVERAGE, stated so that silence cannot be read as a check.  The gate walks
# from --from-heading to the END of the receipt, so the default covers rounds
# 25 onward.  Rounds 1-24 are NOT audited: they cite far more file:line pairs
# than the map holds and the gate is fail-closed on an unmapped citation, so
# pointing it at them would fail on COVERAGE rather than on correctness.
# Extending it is a one-line change plus the map entries; until that is done,
# those rounds' citations rest on hand-checking, which is exactly what failed
# three rounds running.
AUDITED_ROUNDS = "25 and later (heading to end of file)"
UNAUDITED_ROUNDS = ("1-24 -- hand-checked only; their citations are not in "
                    "CITATION_MAP and this gate does not read them")

_NAME = (r"[A-Za-z0-9_./]+\.(?:F90|f90|h90|py|fcm)"
         r"|[A-Za-z0-9_./]*(?:namelist_cfg|namelist_ref|ocean\.output)")
_SPAN = re.compile(r"`([^`\n]+)`")
_FULL = re.compile(rf"^(?P<f>{_NAME}):(?P<l>\d[\d,\-]*)$")
_ONLY = re.compile(rf"^(?P<f>{_NAME})$")
_BARE = re.compile(r"^:(?P<l>\d[\d,\-]*)$")


def extract(text: str) -> list[str]:
    """Ordered, de-duplicated ``file:lines`` citations, continuations bound."""
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)
    current: str | None = None
    found: list[str] = []
    for match in _SPAN.finditer(text):
        span = match.group(1).strip()
        if (full := _FULL.match(span)):
            current = full.group("f")
            found.append(f"{current}:{full.group('l')}")
        elif (only := _ONLY.match(span)):
            current = only.group("f")
        elif (bare := _BARE.match(span)):
            if current is None:
                found.append(f"<UNBOUND>:{bare.group('l')}")
            else:
                found.append(f"{current}:{bare.group('l')}")
    return list(dict.fromkeys(found))


def line_numbers(spec: str) -> list[int]:
    """Cited line numbers, ascending.  An empty or reversed range RAISES.

    ``line_numbers("309-300")`` used to return ``[]``, and ``check`` then
    crashed on ``numbers[0]`` with an ``IndexError`` -- a gate that crashes on
    a malformed citation is a gate that has no verdict for it.
    """
    numbers: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            if int(last) < int(first):
                raise ValueError(f"reversed range {part!r}")
            numbers.extend(range(int(first), int(last) + 1))
        else:
            numbers.append(int(part))
    if not numbers:
        raise ValueError(f"empty citation {spec!r}")
    return numbers


# Terminal tokens recur so densely that they cannot IDENTIFY a line on their
# own; they are refused as bare anchors and must be pinned by occurrence.
TERMINAL_TOKENS = (
    "ENDIF", "END IF", "ENDDO", "END DO", "END_2D", "END_3D", "END SELECT",
    "#endif", "#else", "#if", "CONTINUE", "ELSE",
)


def _anchor_lines(body: list[str], symbol: str) -> list[int]:
    return [i + 1 for i, line in enumerate(body) if symbol in line]


def resolve_anchor(body: list[str], anchor) -> tuple[int | None, str]:
    """The ONE line an anchor identifies, or ``None`` with a reason.

    An anchor is either a symbol that occurs on exactly one line of the file,
    or ``(symbol, nth)`` naming which occurrence is meant.  A bare symbol that
    occurs more than once is REFUSED: round 28 measured that 31 of 86 map
    entries were anchored on a symbol occurring 2 to 59 times, and a recurring
    terminal token at a range's end is exactly how a widened range passed --
    ``stprk3_stg.F90:309-334`` still passed as ``:309-533`` because line 334
    and line 533 are both ``ENDIF``.
    """
    if isinstance(anchor, tuple):
        symbol, nth = anchor
        hits = _anchor_lines(body, symbol)
        if not 1 <= nth <= len(hits):
            return None, (f"occurrence {nth} of {symbol!r} does not exist "
                          f"({len(hits)} found)")
        return hits[nth - 1], ""
    hits = _anchor_lines(body, anchor)
    if not hits:
        return None, f"symbol {anchor!r} is not in the file"
    if len(hits) > 1:
        token = anchor.strip()
        kind = ("terminal token" if token in TERMINAL_TOKENS else "symbol")
        return None, (f"{kind} {anchor!r} occurs on {len(hits)} lines "
                      f"({hits[:6]}...); pin it as (symbol, nth)")
    return hits[0], ""


def group_endpoints(spec: str) -> list[int]:
    """One entry per cited ENDPOINT: a single line gives one, a range two.

    ``"211,317,355"`` gives ``[211, 317, 355]`` and ``"156,161-168"`` gives
    ``[156, 161, 168]``.  Pinning only the OUTER two endpoints left every
    interior comma group unvalidated: ``dynadv_up3.f90:211,317,355`` passed
    with its middle line changed to any number between 211 and 355, because
    the outer anchors and the length were both untouched.
    """
    endpoints: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            endpoints.extend((int(first), int(last)))
        else:
            endpoints.append(int(part))
    return endpoints


def parse_entry(value, n_endpoints: int = 2) -> tuple[list, int | None]:
    """``"sym"`` | ``[a, b, ...]`` | ``[a, b, ..., extent]``.

    Returns one anchor PER ENDPOINT.  A citation with more endpoints than the
    map pins is refused rather than partly checked.
    """
    if isinstance(value, (str, tuple)):
        # a bare string, or a (symbol, nth) TUPLE, is ONE anchor
        anchors, extent = [value], None
    else:
        anchors = list(value)
        extent = anchors.pop() if anchors and isinstance(anchors[-1], int) else None
    if len(anchors) == 1 and n_endpoints > 1:
        anchors = anchors * n_endpoints
    if n_endpoints == 1 and len(anchors) == 2:
        # legacy single-line form [first, last]: both anchors must identify
        # that one line, which is strictly stronger than pinning it once
        return anchors, extent
    if len(anchors) != n_endpoints:
        raise ValueError(
            f"cites {n_endpoints} endpoints, map pins {len(anchors)} anchors")
    return anchors, extent


def check(citation: str, value, shift: int = 0, shift_last_only: bool = False,
          shift_first_only: bool = False) -> dict:
    """Pin BOTH endpoints AND the range LENGTH.

    Three defects this replaces, all measured rather than argued.  (1) The
    first draft asked ``symbol in "\n".join(cited lines)``, so a range passed
    if ANY line held the symbol -- 36 of 78 entries survived a wrong line
    number.  (2) Round 27's fix pinned each endpoint's TEXT but the audit
    shifted every line together, so a changed range EXTENT was never tested
    and a recurring terminal token at the end let a widened range through.
    (3) A reversed range crashed instead of failing.
    """
    path_key, spec = citation.rsplit(":", 1)
    path = FILES.get(path_key)
    if path is None or not path.is_file():
        return {"citation": citation, "status": "UNRESOLVED",
                "detail": f"no readable file for {path_key!r}"}
    try:
        numbers = line_numbers(spec)
        endpoints = group_endpoints(spec)
        anchors, extent = parse_entry(value, len(endpoints))
    except ValueError as error:
        return {"citation": citation, "status": "BAD-CITATION",
                "detail": str(error)}
    body = path.read_text(errors="replace").splitlines()

    if extent is None and len(numbers) > 1:
        return {"citation": citation, "status": "EXTENT-NOT-PINNED",
                "detail": f"a {len(numbers)}-line citation must pin its "
                          "length as the map value's last element"}
    if extent is not None and extent != len(numbers):
        return {"citation": citation, "status": "EXTENT-MISMATCH",
                "detail": f"cites {len(numbers)} lines, map pins {extent}"}

    shifted = list(endpoints)
    if not shift_last_only:
        shifted[0] = endpoints[0] + shift
    if not shift_first_only:
        shifted[-1] = endpoints[-1] + shift
    if any(n < 1 or n > len(body) for n in shifted):
        return {"citation": citation, "status": "OUT-OF-RANGE",
                "detail": f"{path} has {len(body)} lines"}
    if len(anchors) == 2 and len(shifted) == 1:      # legacy single-line form
        shifted = shifted * 2
    names = (["first"] + ["interior"] * (len(shifted) - 2) + ["last"]
             if len(shifted) > 1 else ["first"])
    for anchor, line, end in zip(anchors, shifted, names):
        resolved, why = resolve_anchor(body, anchor)
        if resolved is None:
            return {"citation": citation, "status": "AMBIGUOUS-ANCHOR",
                    "endpoint": end, "detail": why}
        if resolved != line:
            return {"citation": citation, "status": "SYMBOL-NOT-AT-LINE",
                    "endpoint": end, "symbol": str(anchor), "line": line,
                    "detail": f"that symbol identifies line {resolved}"}
    return {"citation": citation, "status": "OK"}


def audit_map() -> list[dict]:
    """EVERY entry in the map must resolve cleanly, cited this round or not.

    Run on every invocation.  ``run`` only checks the citations the receipt
    actually contains, so an entry that has stopped identifying its line would
    sit unnoticed until the prose next used it.  This closes that gap, and it
    is what replaced round 27's shift audit: once an anchor must resolve to
    exactly ONE line, no shift of any size can pass, so a shift sweep could
    never report anything and would have been decoration.  Independent
    endpoint shifting is still exercised -- in :func:`self_test`, where it can
    actually fail.
    """
    return [row for row in (check(c, v) for c, v in CITATION_MAP.items())
            if row["status"] != "OK"]


def self_test() -> list[dict]:
    """The audit's own non-vacuity: three planted defects it MUST flag.

    A gate whose self-check cannot fail is decoration.  These are synthetic
    entries, evaluated directly rather than inserted into the shipped map.
    """
    good = CITATION_MAP["stprk3_stg.F90:309-334"]
    planted = [
        # a recurring terminal token as a BARE anchor, which is what let a
        # widened range through round 27's gate
        ("generic anchor", check("stprk3_stg.F90:309-334",
                                 ["SELECT CASE( kstg )", "ENDIF", 26])),
        # the range widened AND its pinned extent widened to match
        ("widened extent", check("stprk3_stg.F90:309-533",
                                 [good[0], good[1], 225])),
        # a reversed range, which used to raise IndexError instead of failing
        ("reversed range", check("stprk3_stg.F90:309-300", good)),
        # each endpoint shifted on its OWN, the case a uniform shift misses
        ("last endpoint shifted alone",
         check("stprk3_stg.F90:309-334", good, shift=2, shift_last_only=True)),
        ("first endpoint shifted alone",
         check("stprk3_stg.F90:309-334", good, shift=2, shift_first_only=True)),
        # a comma citation's INTERIOR group moved, with both outer endpoints
        # and the pinned length untouched.  Round 28 pinned only the outer two
        # endpoints, so this passed: the middle line of
        # dynadv_up3.f90:211,317,355 could be any number between 211 and 355.
        ("comma interior moved",
         check("BLD/ppsrc/nemo/dynadv_up3.f90:211,318,355",
               CITATION_MAP["BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355"])),
        # and a comma citation whose interior is not pinned at all
        ("comma interior not pinned",
         check("BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355",
               [CITATION_MAP["BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355"][0],
                CITATION_MAP["BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355"][-2],
                3])),
    ]
    # and the unplanted entry must still PASS, or the controls prove nothing
    baseline = check("stprk3_stg.F90:309-334", good)
    comma_baseline = check(
        "BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355",
        CITATION_MAP["BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355"])
    rows = [{"planted": name, "status": row["status"],
             "fired": row["status"] != "OK"} for name, row in planted]
    rows.append({"planted": "unplanted baseline must pass",
                 "status": baseline["status"],
                 "fired": baseline["status"] == "OK"})
    rows.append({"planted": "unplanted comma baseline must pass",
                 "status": comma_baseline["status"],
                 "fired": comma_baseline["status"] == "OK"})
    return rows


def run(receipt: Path, heading: str, *, plant: str | None = None) -> dict:
    document = receipt.read_text()
    if heading not in document:
        raise SystemExit(f"heading {heading!r} not in {receipt}")
    citations = extract(document[document.index(heading):])
    unmapped = [c for c in citations if c not in CITATION_MAP]
    rows = [check(c, CITATION_MAP[c], shift=2 if c == plant else 0)
            for c in citations if c in CITATION_MAP]
    bad = [row for row in rows if row["status"] != "OK"]
    blind = audit_map()
    controls = self_test()
    control_failures = [row for row in controls if not row["fired"]]
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-receipt-citation-gate-v2",
        "receipt": str(receipt),
        "from_heading": heading,
        "audited_rounds": AUDITED_ROUNDS,
        "unaudited_rounds": UNAUDITED_ROUNDS,
        "citations_found": len(citations),
        "unmapped_citations": unmapped,
        "failures": bad,
        "map_entries_failing_audit": blind,
        "self_test": controls,
        "planted_shift": plant,
        "status": ("PASS" if not unmapped and not bad and not blind
                   and not control_failures else "FAIL"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--from-heading", default=DEFAULT_HEADING)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plant", metavar="CITATION",
        help="shift this citation's line numbers by 2; the gate MUST fail, "
             "which is what proves it is not vacuous")
    args = parser.parse_args(argv)
    report = run(args.receipt, args.from_heading, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant:
        # A planted control exits NONZERO: 1 when it correctly made the gate
        # fail, 2 when it did not fire (which would mean the gate is vacuous).
        return 1 if report["status"] == "FAIL" else 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
