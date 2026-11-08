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
_ORCA2_R96SPG_OUTPUT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round96/"
    "acquisition/orca2_rung0_spgts_ranked_10step_np2/ocean.output")
_ORCA2_R141GROWTH_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo")
_ORCA2_R143EARLY_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo")
_ORCA2_R144INITIAL_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo")
_ORCA2_R166SPG8_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo")
_ORCA2_R169SLOW8_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo")
_ORCA2_R170RHS8_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo")
_ORCA2_R172HPG8_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo")
_ORCA2_R182HPGFOLD_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo")
_ORCA2_R210OMT1_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo")
_ORCA2_R213VECPRE_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo")
_ORCA2_R214VECPREV3_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo")
_ORCA2_R280STAGE1_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo")
_ORCA2_R297STAGE7_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo")
_ORCA2_R299W7_COMPILED = (
    NEMO / "cfgs/ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo")
_ORCA2_R229FOLDTRP = NEMO / "cfgs/ORCA2_OMIP_L4_R229FOLDTRP/BLD"
_ORCA2_R230FOLDTRP = NEMO / "cfgs/ORCA2_OMIP_L4_R230FOLDTRP/BLD"
_ORCA2G1_P3 = NEMO / "cfgs/ORCA2G1_OMIP_L1_P3/BLD"
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
_SWG_COMPILED = NEMO / "tests/SWG_OMIP_L1/BLD/ppsrc/nemo"
_SWG_DG_COMPILED = NEMO / "tests/SWG_DG/BLD/ppsrc/nemo"
_SWG_P3_COMPILED = NEMO / "tests/SWG_OMIP_L1_P3/BLD/ppsrc/nemo"
_SWG_P3D_COMPILED = NEMO / "tests/SWG_OMIP_L1_P3D/BLD/ppsrc/nemo"
_SWG_DG_P3_COMPILED = NEMO / "tests/SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo"
_SWG_DG_P3D_COMPILED = NEMO / "tests/SWG_DG_OMIP_L1_P3D/BLD/ppsrc/nemo"
FILES = {
    "tests/unit/test_transcendentals.py": (
        REPO / "tests/unit/test_transcendentals.py"),
    "tests/ocean/fidelity/test_orca2_round342_s4_year.py": (
        REPO / "tests/ocean/fidelity/test_orca2_round342_s4_year.py"),
    "tests/ocean/fidelity/test_orca2_round348_transcendentals.py": (
        REPO / "tests/ocean/fidelity/test_orca2_round348_transcendentals.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh"),
    "src/OCE/TRA/traldf_iso.F90": NEMO / "src/OCE/TRA/traldf_iso.F90",
    "src/OCE/SBC/sbcfwb.F90": NEMO / "src/OCE/SBC/sbcfwb.F90",
    "src/OCE/LBC/mpp_allreduce_generic.h90": (
        NEMO / "src/OCE/LBC/mpp_allreduce_generic.h90"),
    "src/OCE/IOM/iom.F90": NEMO / "src/OCE/IOM/iom.F90",
    # ORCA2X round 1: long-run design/receipt (DECISION 160).
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/zdftke.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/zdftke.f90'),
    'orca1ice_longrun_30day_np2/namelist_cfg': Path('/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/acquisition/orca1ice_longrun_30day_np2/namelist_cfg'),
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icestp.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icestp.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icethd.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icethd.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icedyn_adv_pra.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icedyn_adv_pra.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/stprk3.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/stprk3.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/restart.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/restart.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domzgr.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domzgr.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dommsk.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dommsk.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domutl.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domutl.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lib_fortran.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lib_fortran.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/sbcfwb.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/sbcfwb.f90'),
    'orca1ice_longrun_30day_np2/ocean.output': Path('/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2x_rounds/round1/acquisition/orca1ice_longrun_30day_np2/ocean.output'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domwri.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domwri.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/trdglo.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/trdglo.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domqco.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domqco.f90'),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domain.f90': NEMO / ('cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domain.f90'),
    # ORCA2 round 328 Resume-5: the unmodified round-326 record-build chain.
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfphy.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfevd.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfevd.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfddm.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfddm.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfiwm.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfiwm.f90"),
    # Round 336's admitted record was built from this instrumented target;
    # bind its statement walk to that producer rather than the parent build.
    "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdfphy.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdftke.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stp2d.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traadv.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90"),
    # Frozen round-331 resume prereg used the short basename; bind that one
    # explicitly to the producing record build rather than pristine source.
    "zdftke.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfdrg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfdrg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/istate.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/istate.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynvor.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynhpg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lbcnfd.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lbcnfd.f90"),
    "nemo_testcase_l4_orca2_round331_tke_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round331_tke_gate.py"),
    "tke_round331_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round331_tke_acquisition/"
        "tke_round331_writer.F90"),
    "zdftke_round326_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round326_hmi_acquisition/"
        "zdftke_round326_writer.F90"),
    "round332_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round332_source_order_acquisition/"
        "round332_writer.F90"),
    "zdftke_round331.patch": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round331_tke_acquisition/"
        "zdftke_round331.patch"),
    "zdftke_round332.patch": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round332_source_order_acquisition/"
        "zdftke_round332.patch"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round336_tke_acquisition/"
    "zdftke_round336.patch": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round336_tke_acquisition/"
        "zdftke_round336.patch"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round336_tke_acquisition/"
    "tke_round336_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round336_tke_acquisition/"
        "tke_round336_writer.F90"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round336_tke_acquisition/"
    "make_stub_records.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round336_tke_acquisition/"
        "make_stub_records.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round336_tke_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round336_tke_gate.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round341_rescore_walk_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round341_rescore_walk_gate.py"),
    "tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py": (
        REPO / "tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py"),
    "tests/ocean/unit/test_orca2_round341_stage2_thickness.py": (
        REPO / "tests/ocean/unit/test_orca2_round341_stage2_thickness.py"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/MY_SRC/stprk3.F90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/MY_SRC/stprk3.F90"),
    # ORCA2 round 311: G1b's build (P5) and G8's build (R31FULLENTRY_FIX1).
    "ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90": NEMO / "cfgs/ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90",
    "ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90": NEMO / "cfgs/ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90",
    "ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90": NEMO / "cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90",
    "ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90": NEMO / "cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90",
    # --- GRIDLADDER round 1 (G1 survey) BEGIN ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90': NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90",
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg': NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg",
    'cfgs/SHARED/namelist_ref': NEMO / "cfgs/SHARED/namelist_ref",
    'src/OCE/IOM/restart.F90': NEMO / "src/OCE/IOM/restart.F90",
    'cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm': NEMO / "cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm",
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90': NEMO / "tests/TSUNAMI/MY_SRC/usrdef_istate.F90",
    'tests/TSUNAMI/MY_SRC/usrdef_sbc.F90': NEMO / "tests/TSUNAMI/MY_SRC/usrdef_sbc.F90",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/istate.f90': NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/istate.f90",
    'tools/DOMAINcfg/src/domhgr.F90': NEMO / "tools/DOMAINcfg/src/domhgr.F90",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90': NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90",
    'tools/DOMAINcfg/src/domzgr.F90': NEMO / "tools/DOMAINcfg/src/domzgr.F90",
    'tools/DOMAINcfg/src/domain.F90': NEMO / "tools/DOMAINcfg/src/domain.F90",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90': NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domzgr.f90': NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/domzgr.f90",
    'tripole.py': REPO / "packages/core/legoesm/grids/tripole.py",
    # --- GRIDLADDER round 1 (G1 survey) END ---
    # --- GRIDLADDER round 2 (compiled G1 record build) ---
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/dynvor.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/dynhpg.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/dynzdf.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domzgr.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/domzgr.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domwri.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/domwri.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/dynspg_ts.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/dommsk.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/lbcnfd.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/domqco.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/stprk3_stg.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/trazdf.f90"),
    "ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2G1_P3 / "ppsrc/nemo/stp2d.f90"),
    "stprk3.F90": _OCE / "stprk3.F90",
    "phycst.F90": _OCE / "DOM/phycst.F90",
    "SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90": (
        _SWG_COMPILED / "usrdef_nam.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90": (
        _SWG_COMPILED / "usrdef_hgr.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90": (
        _SWG_COMPILED / "usrdef_zgr.f90"),
    "SWG_DG/BLD/ppsrc/nemo/usrdef_zgr.f90": (
        _SWG_DG_COMPILED / "usrdef_zgr.f90"),
    "SWG_DG/BLD/ppsrc/nemo/stprk3.f90": (
        _SWG_DG_COMPILED / "stprk3.f90"),
    "SWG_DG/BLD/ppsrc/nemo/divhor.f90": (
        _SWG_DG_COMPILED / "divhor.f90"),
    "SWG_DG/BLD/ppsrc/nemo/sshwzv.f90": (
        _SWG_DG_COMPILED / "sshwzv.f90"),
    "SWG_DG/BLD/ppsrc/nemo/diawri.f90": (
        _SWG_DG_COMPILED / "diawri.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90": (
        _SWG_COMPILED / "usrdef_sbc.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90": (
        _SWG_COMPILED / "dynvor.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/dynzad.f90": (
        _SWG_COMPILED / "dynzad.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90": (
        _SWG_COMPILED / "stprk3.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90": (
        _SWG_COMPILED / "nemogcm.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90": (
        _SWG_COMPILED / "domain.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/domhgr.f90": (
        _SWG_COMPILED / "domhgr.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/domzgr.f90": (
        _SWG_COMPILED / "domzgr.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/domtile.f90": (
        _SWG_COMPILED / "domtile.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/mppini.f90": (
        _SWG_COMPILED / "mppini.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/restart.f90": (
        _SWG_COMPILED / "restart.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/iom_nf90.f90": (
        _SWG_COMPILED / "iom_nf90.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/bdyini.f90": (
        _SWG_COMPILED / "bdyini.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/sshwzv.f90": (
        _SWG_COMPILED / "sshwzv.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/sbc_oce.f90": (
        _SWG_COMPILED / "sbc_oce.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/sbcmod.f90": (
        _SWG_COMPILED / "sbcmod.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90": (
        _SWG_COMPILED / "diawri.f90"),
    "ext/IOIPSL/src/mathelp.f90": (
        NEMO / "ext/IOIPSL/src/mathelp.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90": (
        _SWG_COMPILED / "dommsk.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/domqco.f90": (
        _SWG_COMPILED / "domqco.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90": (
        _SWG_COMPILED / "dynadv.f90"),
    "SWG_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90": (
        _SWG_COMPILED / "ldfdyn.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90": (
        _SWG_P3_COMPILED / "dynvor.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90": (
        _SWG_P3_COMPILED / "dynadv.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90": (
        _SWG_P3_COMPILED / "sshwzv.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90": (
        _SWG_P3_COMPILED / "domqco.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90": (
        _SWG_P3_COMPILED / "stprk3.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90": (
        _SWG_P3_COMPILED / "ldfdyn.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90": (
        _SWG_P3_COMPILED / "dynldf_lev.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf.f90": (
        _SWG_P3_COMPILED / "dynldf.f90"),
    "SWG_OMIP_L1_P3/BLD/ppsrc/nemo/sbcmod.f90": (
        _SWG_P3_COMPILED / "sbcmod.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90": (
        _SWG_P3D_COMPILED / "stprk3.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/usrdef_istate.f90": (
        _SWG_P3D_COMPILED / "usrdef_istate.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/sbcmod.f90": (
        _SWG_P3D_COMPILED / "sbcmod.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/domain.f90": (
        _SWG_P3D_COMPILED / "domain.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/nemogcm.f90": (
        _SWG_P3D_COMPILED / "nemogcm.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/restart.f90": (
        _SWG_P3D_COMPILED / "restart.f90"),
    "SWG_OMIP_L1_P3D/BLD/ppsrc/nemo/diawri.f90": (
        _SWG_P3D_COMPILED / "diawri.f90"),
    "SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90": (
        _SWG_DG_P3_COMPILED / "stprk3.f90"),
    "SWG_DG_OMIP_L1_P3D/BLD/ppsrc/nemo/stprk3.f90": (
        _SWG_DG_P3D_COMPILED / "stprk3.f90"),
    # SWG survey authorities.  Keep their qualified names: several basenames
    # also exist in OCE, SWE, compiled P3, and other testcase MY_SRC trees.
    "src/OCE/DYN/dynspg.F90": _DYN / "dynspg.F90",
    "src/OCE/DYN/dynadv.F90": _DYN / "dynadv.F90",
    "src/OCE/DYN/dynvor.F90": _DYN / "dynvor.F90",
    "src/OCE/DYN/sshwzv.F90": _DYN / "sshwzv.F90",
    "src/OCE/SBC/sbcmod.F90": _OCE / "SBC/sbcmod.F90",
    "src/SWE/stprk3.F90": NEMO / "src/SWE/stprk3.F90",
    "tests/SWG/MY_SRC/usrdef_istate.F90": (
        NEMO / "tests/SWG/MY_SRC/usrdef_istate.F90"),
    "tests/SWG/MY_SRC/usrdef_nam.F90": (
        NEMO / "tests/SWG/MY_SRC/usrdef_nam.F90"),
    "tests/SWG/MY_SRC/usrdef_hgr.F90": (
        NEMO / "tests/SWG/MY_SRC/usrdef_hgr.F90"),
    "tests/SWG/MY_SRC/usrdef_zgr.F90": (
        NEMO / "tests/SWG/MY_SRC/usrdef_zgr.F90"),
    "tests/SWG/MY_SRC/usrdef_sbc.F90": (
        NEMO / "tests/SWG/MY_SRC/usrdef_sbc.F90"),
    "nemo_recipe.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py"),
    # The frozen round-3 preregistration used these shortened names.  Pin
    # them to the admitted P3 build rather than silently resolving another
    # same-basename source tree.
    "dynadv.f90": _SWG_P3_COMPILED / "dynadv.f90",
    "domqco.f90": _SWG_P3_COMPILED / "domqco.f90",
    "stprk3.f90": _SWG_P3_COMPILED / "stprk3.f90",
    "sbcmod.f90": _SWG_P3_COMPILED / "sbcmod.f90",
    "dynldf.f90": _SWG_P3_COMPILED / "dynldf.f90",
    "dynldf_lev.f90": _SWG_P3_COMPILED / "dynldf_lev.f90",
    "stprk3_stg.F90": _OCE / "stprk3_stg.F90",
    "ORCA2_R96SPG/ocean.output": _ORCA2_R96SPG_OUTPUT,
    # round 202: the PRODUCTION GYRE build, so a GYRE resolved statement
    # cannot bind to one of the _SM_ probe builds.
    "GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/usrdef_sbc.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/usrdef_sbc.f90"),
    "GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/sbcmod.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/sbcmod.f90"),
    # High-resolution GYRE round 13 audits the uninstrumented production
    # program used by the operator's five-year vector-math reference.  Keep
    # these qualified so none can bind to a P3 writer override.
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynzdf.f90"),
    "GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynzdf.f90"),
    "GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90"),
    "GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/dynzdf.f90"),
    "GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/trdini.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/trdini.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/iom.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/iom.f90"),
    "GYRE_OMIP_L2/BLD/ppsrc/nemo/diawri.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/diawri.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/traqsr.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/traqsr.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfphy.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfphy.f90"),
    "GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfsh2.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfsh2.f90"),
    "GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90": (
        NEMO
        / "cfgs/GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90"),
    "GYRE_OMIP_L2_P3_DAILY/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_DAILY/BLD/ppsrc/nemo/stprk3.f90"),
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
    # ORCA2 round 324: the ice-ocean top-drag coefficient and its consumer.
    "ICE/iceupdate.F90": NEMO / "src/ICE/iceupdate.F90",
    "ZDF/zdfdrg.F90": _OCE / "ZDF/zdfdrg.F90",
    "do_loop_substitute.h90": _OCE / "do_loop_substitute.h90",
    "domqco.F90": _OCE / "DOM/domqco.F90",
    # ORCA2 round 1 binds its first non-bit entry field to the compiled branch
    # that executes the category-summed initial snow/ice load adjustment.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90": (
        _ORCA2_COMPILED / "iceistate.f90"),
    # ORCA2 round 327 binds the inherited entry owner to the exact build that
    # produced the admitted one-category trajectory.
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90"),
    # ORCA2 round 78 binds the turbulence mixing-length floor to the compiled
    # two-arm rmxl_min choice this deck's ln_zdfiwm selects.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90": (
        _ORCA2_COMPILED / "zdftke.f90"),
    # ORCA2 round 326 (DECISION 154): where SI3 sets hm_i, and when.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90": (
        _ORCA2_COMPILED / "icevar.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90": (
        _ORCA2_COMPILED / "icestp.f90"),
    # ORCA2 round 79b cites the vertical-physics chain and the internal-wave
    # arm that this deck runs and the card does not.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90": (
        _ORCA2_COMPILED / "zdfphy.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90": (
        _ORCA2_COMPILED / "zdfevd.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90": (
        _ORCA2_COMPILED / "zdfiwm.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90": (
        _ORCA2_COMPILED / "zdfddm.f90"),
    # ORCA2 round 310 imports rung G8's dyn_ldf_lev statements.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90": (
        _ORCA2_COMPILED / "dynldf_lev.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90": (
        _ORCA2_COMPILED / "dynkeg.f90"),
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
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90": (
        _ORCA2_COMPILED / "domain.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/in_out_manager.f90": (
        _ORCA2_COMPILED / "in_out_manager.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcflx.f90": (
        _ORCA2_COMPILED / "sbcflx.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90": (
        _ORCA2_COMPILED / "trasbc.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traqsr.f90": (
        _ORCA2_COMPILED / "traqsr.f90"),
    # ORCA2 round 278 cites the Decision-122 statements in the shipped build.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90": (
        _ORCA2_COMPILED / "dynzdf.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90": (
        _ORCA2_COMPILED / "domqco.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90": (
        _ORCA2_COMPILED / "eosbn2.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90": (
        _ORCA2_COMPILED / "zdfsh2.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90": (
        _ORCA2_COMPILED / "dynvor.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_COMPILED / "stp2d.f90"),
    # ORCA2 round 308 cites G6's wzv/zad statements in the shipped build.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90": (
        _ORCA2_COMPILED / "sshwzv.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90": (
        _ORCA2_COMPILED / "dynzad.f90"),
    # ORCA2 round 305 cites the G5/G7 drag statements in the shipped build.
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90": (
        _ORCA2_COMPILED / "stpctl.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90": (
        _ORCA2_R210OMT1_COMPILED / "stpctl.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R210OMT1_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90": (
        _ORCA2_R210OMT1_COMPILED / "dynzdf.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90": (
        _ORCA2_R210OMT1_COMPILED / "zdfdrg.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90": (
        _ORCA2_R210OMT1_COMPILED / "ldfdyn.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf.f90": (
        _ORCA2_R210OMT1_COMPILED / "dynldf.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf_lev.f90": (
        _ORCA2_R210OMT1_COMPILED / "dynldf_lev.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R210OMT1_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R280STAGE1_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R280STAGE1_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R280STAGE1_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R280STAGE1_COMPILED / "lbcnfd.f90"),
    "ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R297STAGE7_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R297STAGE7_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/divhor.f90": (
        _ORCA2_R297STAGE7_COMPILED / "divhor.f90"),
    "ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/sshwzv.f90": (
        _ORCA2_R297STAGE7_COMPILED / "sshwzv.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R299W7_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/divhor.f90": (
        _ORCA2_R299W7_COMPILED / "divhor.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/sshwzv.f90": (
        _ORCA2_R299W7_COMPILED / "sshwzv.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R299W7_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R299W7_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R299W7_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R299W7_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90": (
        _ORCA2_R299W7_COMPILED / "zdfevd.f90"),
    "ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R299W7_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R210OMT1_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_cen.f90": (
        _ORCA2_R210OMT1_COMPILED / "traadv_cen.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90": (
        _ORCA2_R210OMT1_COMPILED / "traadv_fct.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90": (
        _ORCA2_R210OMT1_COMPILED / "trazdf.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R210OMT1_COMPILED / "lbcnfd.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R210OMT1_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90": (
        _ORCA2_R210OMT1_COMPILED / "restart.f90"),
    "ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/daymod.f90": (
        _ORCA2_R210OMT1_COMPILED / "daymod.f90"),
    "ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R213VECPRE_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R213VECPRE_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R213VECPRE_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R214VECPREV3_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R214VECPREV3_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R214VECPREV3_COMPILED / "lbcnfd.f90"),
    "orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg"),
    "ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R229FOLDTRP / "ppsrc/nemo/stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R229FOLDTRP/BLD/inc/domzgr_substitute.h90": (
        _ORCA2_R229FOLDTRP / "inc/domzgr_substitute.h90"),
    "ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90": (
        _ORCA2_R229FOLDTRP / "ppsrc/nemo/dom_oce.f90"),
    "ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/traadv_fct.f90": (
        _ORCA2_R229FOLDTRP / "ppsrc/nemo/traadv_fct.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/dynhpg.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/dynvor.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/dynzdf.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/traadv_cen.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/domzgr.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/domqco.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domain.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/domain.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/zdfdrg.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/zdfdrg.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/stp2d.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/dommsk.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R230FOLDTRP / "ppsrc/nemo/lbcnfd.f90"),
    "ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90": (
        _ORCA2_R230FOLDTRP / "inc/domzgr_substitute.h90"),
    "ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R214VECPREV3_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90": (
        _ORCA2_COMPILED / "dynadv.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90": (
        _ORCA2_COMPILED / "ldfdyn.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90": (
        _ORCA2_COMPILED / "ldftra.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90": (
        _ORCA2_COMPILED / "zdfdrg.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90": (
        _ORCA2_COMPILED / "traadv_fct.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90": (
        _ORCA2_COMPILED / "trazdf.f90"),
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
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R90FRAMES_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R90FRAMES_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90": (
        _ORCA2_R90FRAMES_COMPILED / "zdfphy.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfdrg.f90": (
        _ORCA2_R90FRAMES_COMPILED / "zdfdrg.f90"),
    "ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/istate.f90": (
        _ORCA2_R90FRAMES_COMPILED / "istate.f90"),
    "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R92RHS_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90": (
        _ORCA2_R92RHS_COMPILED / "dynhpg.f90"),
    "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90": (
        _ORCA2_R92RHS_COMPILED / "domain.f90"),
    "ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dommsk.f90": (
        _ORCA2_R92RHS_COMPILED / "dommsk.f90"),
    "ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R93SLOW_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R93SLOW_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R96SPG_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domhgr.f90": (
        _ORCA2_R96SPG_COMPILED / "domhgr.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R96SPG_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R96SPG_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R141GROWTH_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R141GROWTH_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/stprk3.f90": (
        _ORCA2_R143EARLY_COMPILED / "stprk3.f90"),
    "ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/traadv.f90": (
        _ORCA2_R143EARLY_COMPILED / "traadv.f90"),
    "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R144INITIAL_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90": (
        _ORCA2_R144INITIAL_COMPILED / "stprk3_stg.f90"),
    "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/sshwzv.f90": (
        _ORCA2_R144INITIAL_COMPILED / "sshwzv.f90"),
    "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R144INITIAL_COMPILED / "lbcnfd.f90"),
    "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90": (
        _ORCA2_R144INITIAL_COMPILED / "lbclnk.f90"),
    "ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/mppini.f90": (
        _ORCA2_R144INITIAL_COMPILED / "mppini.f90"),
    "ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90": (
        _ORCA2_R166SPG8_COMPILED / "dynspg_ts.f90"),
    "ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R166SPG8_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R169SLOW8_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R170RHS8_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90": (
        _ORCA2_R170RHS8_COMPILED / "dynhpg.f90"),
    "ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/stp2d.f90": (
        _ORCA2_R172HPG8_COMPILED / "stp2d.f90"),
    "ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90": (
        _ORCA2_R172HPG8_COMPILED / "dynhpg.f90"),
    "ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90": (
        _ORCA2_R182HPGFOLD_COMPILED / "dynhpg.f90"),
    "ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/lbcnfd.f90": (
        _ORCA2_R96SPG_COMPILED / "lbcnfd.f90"),
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
    "ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90": (
        _ORCA2_R84FRAMES_COMPILED / "zdfphy.f90"),
    "ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90": (
        _ORCA2_R84FRAMES_COMPILED / "zdfevd.f90"),
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
    "ORCA2_OMIP_L4/EXP00/namelist_ref": (
        NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dom_oce.f90": (
        _ORCA2_COMPILED / "dom_oce.f90"),
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
    # GYRE-HR round 17 starts from the admitted round-14 budget producer's
    # developed restart, so bind the call order and ENE arithmetic to that
    # producer's exact compiled sources.
    "GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90"),
    # GYRE-HR round 18 consumes the round-17 common-state writer records, so
    # its source-ordered operand verdict is bound to that writer's program.
    "GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90"),
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
    "gm_redi_latlon_cgrid.py":
        REPO / "packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py",
    "advection.py": REPO / "packages/ocean/legoesm/ocean/advection.py",
    "implicit_solver.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py",
    "internal_wave_mixing.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py",
    "double_diffusion.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py",
    "k_profiles.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py",
    "tke.py": REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py",
    "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py",
    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py":
        REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py",
    "packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py":
        REPO / "packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py",
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py":
        REPO / "scripts/validate/ocean_fidelity/orca2_l4"
             / "nemo_testcase_l4_orca2_round336_tke_walk.py",
    "shortwave_penetration.py":
        REPO / "packages/ocean/legoesm/ocean/physics/shortwave_penetration.py",
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
    "nemo_testcase_v1_orca2tke/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/lanes/"
        "gridladder_vmix/scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_v1_orca2tke/namelist_cfg"),
    "gridladder_vmix/nemo_testcase_recipe.py": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/lanes/"
        "gridladder_vmix/packages/ocean/legoesm/ocean/fidelity/"
        "nemo_testcase_recipe.py"),
    "nemo_testcase_l4_orca2_round330_dino_probe.py":
        REPO / "scripts/validate/ocean_fidelity/orca2_l4"
             / "nemo_testcase_l4_orca2_round330_dino_probe.py",
    "nemo_testcase_l4_orca2_round226_fct_rhs_walk.py":
        REPO / "scripts/validate/ocean_fidelity/orca2_l4"
             / "nemo_testcase_l4_orca2_round226_fct_rhs_walk.py",
    "test_nemo_testcase_l4_orca2_round226_fct_rhs_walk.py":
        REPO / "tests/ocean/fidelity"
             / "test_nemo_testcase_l4_orca2_round226_fct_rhs_walk.py",
    "nemo_testcase_l4_orca2_round174_stage_growth_gate.py":
        REPO / "scripts/validate/ocean_fidelity/orca2_l4"
             / "nemo_testcase_l4_orca2_round174_stage_growth_gate.py",
    "nemo_testcase_l4_orca2_round166_external_substep_gate.py":
        REPO / "scripts/validate/ocean_fidelity/orca2_l4"
             / "nemo_testcase_l4_orca2_round166_external_substep_gate.py",
    "ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90": (
        NEMO / "cfgs/ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90"),
    "ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "round175/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round175"
        "/acquisition/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg"),
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
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/fldread.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssm.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcssm.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssr.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcssr.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcfwb.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcfwb.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/EXP00/namelist_cfg": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/EXP00/namelist_cfg"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/stp2d.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/trasbc.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcmod.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcblk.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcblk_algo_ncar.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbc_phy.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/stprk3_stg.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/dynzdf.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R328ZDF/BLD/ppsrc/nemo/dynzdf.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R328ZDF/BLD/ppsrc/nemo"
        "/dynzdf.f90"),
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
    # Round 214 / ORCA2 round 132: the exact compiled seamount domain-builder
    # branch whose face-thickness statement the shared qco repair transcribes.
    "VORTEX_SMT_R3_OMIP_L1_P3/BLD/ppsrc/nemo/usrdef_zgr.f90": (
        NEMO / "tests/VORTEX_SMT_R3_OMIP_L1_P3/BLD/ppsrc/nemo/usrdef_zgr.f90"),
    # --- round 6: the after-SSH slot the RK3 program leaves behind, in the
    # vector card's own build and in GYRE's (the statement is shared). ---
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90"),
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90"),
    "VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "tests/VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90"),
    # Round 241 binds the deferred SMT-1 daily-restart acquisition to the
    # exact admitted instrumented build whose binary the driver reuses.
    "VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "tests/VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90"),
    # Round 243 binds the deferred SMT-2 daily-restart acquisition to the
    # exact admitted instrumented build whose binary the driver reuses.
    "VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "tests/VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90"),
    # Round 245 scores the admitted SMT-3 daily trajectory from this exact
    # instrumented build; the trajectory was not reconstructed or interpolated.
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90"),
    # Round 246 scores the admitted SMT-4 daily trajectory from this exact
    # instrumented build; the trajectory was not reconstructed or interpolated.
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90"),
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
    "tests/TSUNAMI/MY_SRC/usrdef_nam.F90": NEMO / "tests/TSUNAMI/MY_SRC/usrdef_nam.F90",
    "tests/TSUNAMI/MY_SRC/usrdef_hgr.F90": NEMO / "tests/TSUNAMI/MY_SRC/usrdef_hgr.F90",
    "tests/TSUNAMI/MY_SRC/usrdef_zgr.F90": NEMO / "tests/TSUNAMI/MY_SRC/usrdef_zgr.F90",
    "tests/TSUNAMI/MY_SRC/usrdef_istate.F90": NEMO / "tests/TSUNAMI/MY_SRC/usrdef_istate.F90",
    "tests/TSUNAMI/MY_SRC/usrdef_sbc.F90": NEMO / "tests/TSUNAMI/MY_SRC/usrdef_sbc.F90",
    "tests/TSUNAMI/MY_SRC/stpmlf.F90": NEMO / "tests/TSUNAMI/MY_SRC/stpmlf.F90",
    "tests/TSUNAMI/MY_SRC/stprk3.F90": NEMO / "tests/TSUNAMI/MY_SRC/stprk3.F90",
    "tests/TSUNAMI/MY_SRC/diawri.F90": NEMO / "tests/TSUNAMI/MY_SRC/diawri.F90",
    "traadv.F90": _OCE / "TRA/traadv.F90",
    # --- TSUNAMI lane round 1: shared sources not mapped before ---
    "nemogcm.F90": _OCE / "nemogcm.F90",
    "depth_e3.F90": _OCE / "DOM/depth_e3.F90",
    "vortex_round2/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/vortex/round2/namelist_cfg"),
    # --- gating the DINO month-regression and PR-1802 review-fix receipts:
    # the DINO build's own compiled branch (bare and cfgs/-prefixed keys both
    # cited by those receipts), the shared LBC template, and the bridge. ---
    "cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90",
    "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90",
    "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90"),
    "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90"),
    "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90"),
    "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg_ts.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg_ts.f90"),
    "nemogcm.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/nemogcm.f90",
    "dynspg_ts.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/dynspg_ts.f90",
    "lbclnk.f90": NEMO / "cfgs/DINO/BLD/ppsrc/nemo/lbclnk.f90",
    "src/OCE/DOM/domqco.F90": _OCE / "DOM/domqco.F90",
    "lbc_lnk_pt2pt_generic.h90": _OCE / "LBC/lbc_lnk_pt2pt_generic.h90",
    "nemo_state_bridge.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_state_bridge.py"),
    "zdftke.F90": _OCE / "ZDF/zdftke.F90",
    "sbcmod.F90": _OCE / "SBC/sbcmod.F90",
    "iom.F90": _OCE / "IOM/iom.F90",
    "icestp.F90": NEMO / "src/ICE/icestp.F90",
    "ORCA2_OMIP_L4/EXP00/file_def_nemo-ice.xml": (
        NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/file_def_nemo-ice.xml"),
    "icbutl.F90": _OCE / "ICB/icbutl.F90",
    "sbcice_if.F90": _OCE / "SBC/sbcice_if.F90",
    "sbcfwb.F90": _OCE / "SBC/sbcfwb.F90",
    "diaar5.F90": _OCE / "DIA/diaar5.F90",
    "domhgr.F90": _OCE / "DOM/domhgr.F90",
    "geo2ocean.F90": _OCE / "SBC/geo2ocean.F90",
    "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg": (
        NEMO / "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg"),
    # Round 224 freezes SMT-3 from ORCA2 rung 0 and reads the executable
    # branch from the already-compiled SMT-2 base.  The new SMT-3 build is
    # intentionally absent until the operator runs the acquisition.
    "orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/"
        "acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg"),
    "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref": (
        NEMO / "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref"),
    "VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90": (
        NEMO / "tests/VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90"),
    "VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traldf.f90": (
        NEMO / "tests/VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traldf.f90"),
    # Round 226 cites the instrumented build that produced the admitted
    # SMT-3 record, not the nearby pristine or SMT-2 configurations.
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    # Round 237 freezes SMT-4 from ORCA2 rung 0 and reads the executable
    # momentum-LDF branch from the already-compiled pristine SMT-3 base. The
    # new SMT-4 target is intentionally absent until the operator acquisition.
    "VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90"),
    # Round 238 binds the admitted SMT-4 card and stage walk to the exact
    # compiled P3 target that produced their record.
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90"),
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf.f90"),
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90"),
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90"),
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90"),
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "tests/VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90"),
    # Round 228 binds the walk and its instrument retraction to the exact
    # compiled R15 branch that produced the admitted payload.
    "VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90"),
    "VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90"),
    "VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90"),
    "VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90"),
    # Round 229 binds the corrected tensor record and magnitude owner to the
    # exact R16 build that produced it.
    "VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90"),
    "VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90"),
    "VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90": (
        NEMO / "tests/VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90"),
    # Round 338: standalone-S3 sbc_ice_if sources (record build).
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/sbcice_if.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/stprk3.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lib_fortran.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/lib_fortran.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/fldread.f90"),
    "ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceupdate.f90": (
        NEMO / "cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
        "/iceupdate.f90"),
    # Round 340: the admitted K0 smoke build's exact preprocessed program.
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/sbcmod.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/stprk3.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/stprk3_stg.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/sbcice_if.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbc_oce.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/sbc_oce.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/nemogcm.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/nemogcm.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/lbcnfd.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/lbcnfd.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dommsk.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/dommsk.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/restart.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/ppsrc/nemo/restart.f90": (
        NEMO / "cfgs/ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/ppsrc/"
        "nemo/restart.f90"),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/istate.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/istate.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dtatsd.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/dtatsd.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/sbcblk.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/tradmp.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/tradmp.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/fldread.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/fldread.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfphy.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/zdfphy.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfdrg.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/zdfdrg.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/zdftke.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/stp2d.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dynspg_ts.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/dynspg_ts.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/trasbc.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/sbcssm.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/traqsr.f90")),
    "ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
             "round339/k0_smoke/orca2_k0_10step_np2/build_sources/ppsrc/"
             "nemo/stpctl.f90")),
    "round340_surface_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round340_s4_acquisition/"
        "round340_surface_writer.F90"),
    "nemo_testcase_l4_orca2_round340_standalone_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round340_standalone_gate.py"),
    "packages/ocean/legoesm/ocean/forcing/nemo_surface.py": (
        REPO / "packages/ocean/legoesm/ocean/forcing/nemo_surface.py"),
    "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py": (
        REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py"),
    "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py": (
        REPO / "packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py"),
    "packages/ocean/legoesm/ocean/bulk_flux_omip.py": (
        REPO / "packages/ocean/legoesm/ocean/bulk_flux_omip.py"),
    "packages/ocean/legoesm/ocean/state.py": (
        REPO / "packages/ocean/legoesm/ocean/state.py"),
    "packages/ocean/legoesm/ocean/physics/shortwave_penetration.py": (
        REPO / "packages/ocean/legoesm/ocean/physics/shortwave_penetration.py"),
    "packages/ocean/legoesm/ocean/eos.py": (
        REPO / "packages/ocean/legoesm/ocean/eos.py"),
    # Round 334 (ratchet cleanup): the NEMO statements the named constants cite.
    "trabbc.F90": _OCE / "TRA/trabbc.F90",
    "phycst.F90": _OCE / "DOM/phycst.F90",
    "tramle.F90": _OCE / "TRA/tramle.F90",
    "zdf_oce.F90": _OCE / "ZDF/zdf_oce.F90",
    "src/OCE/DOM/dommsk.F90": _OCE / "DOM/dommsk.F90",
    "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"),
    "tests/ocean/unit/test_nemo_sh2_fold_top_v.py": (
        REPO / "tests/ocean/unit/test_nemo_sh2_fold_top_v.py"),
    "tests/ocean/fidelity/test_orca2_round336_tke_walk.py": (
        REPO / "tests/ocean/fidelity/test_orca2_round336_tke_walk.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_round336_d122_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round336_d122_gate.py"),
    "tests/ocean/fidelity/test_orca2_round336_d122_gate.py": (
        REPO / "tests/ocean/fidelity/test_orca2_round336_d122_gate.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/"
    "nemo_testcase_l4_orca2_phase2t_tke_entry_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_phase2t_tke_entry_gate.py"),
    # Round 343: standalone ORCA2 K0 speed and 8-day accuracy benchmark.
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py"),
    "tests/ocean/fidelity/test_orca2_round343_speed_benchmark.py": (
        REPO / "tests/ocean/fidelity/test_orca2_round343_speed_benchmark.py"),
    "packages/core/legoesm/core/precision.py": (
        REPO / "packages/core/legoesm/core/precision.py"),
    "packages/core/legoesm/core/transcendentals.py": (
        REPO / "packages/core/legoesm/core/transcendentals.py"),
    "packages/core/legoesm/core/_scalar_libm.c": (
        REPO / "packages/core/legoesm/core/_scalar_libm.c"),
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round342_s4_year_driver.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/"
        "nemo_testcase_l4_orca2_round348_transcendentals_compare.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round16_discriminator.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/"
        "nemo_testcase_l2_gyre_round16_discriminator.py"),
    "packages/core/legoesm/core/source_rounding.py": (
        REPO / "packages/core/legoesm/core/source_rounding.py"),
    "packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py": (
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py"),
    "packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py": (
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py"),
    "packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py": (
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py"),
    "packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py": (
        REPO / "packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py"),
    "packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py": (
        REPO / "packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py"),
    "packages/ocean/legoesm/ocean/vertical.py": (
        REPO / "packages/ocean/legoesm/ocean/vertical.py"),
    "cfgs/GYRE_OMIP_L2/WORK/usrdef_nam.F90": (
        NEMO / "cfgs/GYRE_OMIP_L2/WORK/usrdef_nam.F90"),
    "cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90": (
        NEMO / "cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90"),
    "cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90": (
        NEMO / "cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90"),
    "cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90"),
    "cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90"),
    "cfgs/DINO/MY_SRC/zdftke.F90": (
        NEMO / "cfgs/DINO/MY_SRC/zdftke.F90"),
    "nemo_testcase_l2_gyre_gradient_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_gyre_gradient_gate.py"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/eosbn2.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/eosbn2.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynhpg.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynhpg.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/traadv.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/traadv.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynvor.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynvor.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90"),
    "TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/mppini.f90": (
        NEMO / "tests/TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/mppini.f90"),
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
    'src/OCE/SBC/sbcfwb.F90:342-365': ['! at the end of year n:', ('END SELECT', 5), 24],
    'src/OCE/LBC/mpp_allreduce_generic.h90:113-149': ['llisdelay = ln_mppdelay .AND. PRESENT(cdelay)', 'ALLOCATE(todelay(idvar)%DELAYBUFF/**/in(ipi), todelay(idvar)%DELAYBUFF/**/out(ipi))', 37],
    'src/OCE/IOM/iom.F90:1925-1966': ["IF( TRIM(cdaction) == 'READ' ) THEN   ! read restart related to mpp_delay !", ('ndelayid(ji) = ndlrstuse   ! set to ndlrstuse to specify that the value was read in the restart', 4), 42],
    'src/OCE/IOM/iom.F90:1972-1984': [('DO ji = 1, nbdelay', 2), "CALL iom_rstput( kt, nitrst, kncid, 'DELAY_im8_'//c_delaylist(ji), AIMAG(todelay(ji)%ydpbuffout      ), jp_r8 )", 13],
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/zdftke.f90:195': 'CALL orca2x_longrun_hmi_write( kt, nn_mxlice, ln_mxl0 )   ! ORCA2X write-only replay input',
    'orca1ice_longrun_30day_np2/namelist_cfg:21': 'nn_itend    =      240   !  Lane-4 registered ten-step endpoint',
    'orca1ice_longrun_30day_np2/namelist_cfg:23': 'nn_stock    =      8   !  Lane-4 registered terminal restart',
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icestp.f90:297': 'IF( kt > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer',
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icethd.f90:309,327,339': [('IF( kt > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer', 1), ('IF( kt > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer', 2), ('IF( kt > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer', 3), 3],
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/icedyn_adv_pra.f90:602': 'IF( kt > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer',
    'ORCA2_ORCA1ICE_OMIP_L4_ORCA2X_R1_LONGRUN/BLD/ppsrc/nemo/stprk3.f90:314': 'IF( kstp > nit000 + 9 ) RETURN   ! ORCA2X: cap inherited write-only observer',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )     ! before fields", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:330': 'CALL zdf_mxl( kt, Kmm )                        !* mixed layer depth, and level',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:175': 'CALL bn2    ( ts(:,:,:,:,Nbb), rab_b, rn2b, Nbb ) ! before Brunt-Vaisala frequency',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:45': 'REAL(wp), PUBLIC ::   rho_c = 0.01_wp    !: density criterion for mixed layer depth',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:114-118': ['zN2_c = grav * rho_c * r1_rho0      ! convert density criteria into N^2 criteria', 'IF( hmlp(ji,jj) < zN2_c )   nmln(ji,jj) = MIN( jk , ikt ) + 1   ! Mixed layer level', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:122-123': ['iik = nmln(ji,jj)', 'hmlp(ji,jj) = ((gdepw_1d(iik  ) ) *(1._wp+r3t(ji,jj,Kmm))) * ssmask(ji,jj)    ! Mixed layer depth', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domzgr.f90:384-386': ['zrefdep = 10._wp - 0.1_wp * MINVAL( e3w_1d )                   ! ref. depth with tolerance (10% of minimum layer thickness)', 'nla10 = nlb10 - 1                                              ! deepest    W level Above ~10m', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dommsk.f90:247,263-264': ['ssmask (:,:) = MAXVAL( tmask(:,:,:), DIM=3 )', "CALL dom_uniq( tmask_i, 'T' )", 'tmask_i (:,:) = ssmask(:,:) * tmask_i(:,:)', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domutl.f90:131-132': ["CALL lbc_lnk( 'domwri', puniq, cdgrd, 1._wp )   ! apply boundary conditions", 'lluniq(:,:,1) = puniq(:,:) == ztstref(:,:)      ! check which values have not been changed', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lib_fortran.f90:1508': ('zztmp = ptab(ji+iisht,jj+ijsht) * smask0_i(ji,jj)', 2),
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/sbcfwb.f90:212': "IF (lwp) WRITE(numout,*) 'Total Domain area (km**2):', area/1000._wp/1000._wp",
    'orca1ice_longrun_30day_np2/ocean.output:1662': 'Total Domain area (km**2):   357005237.10546613',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domwri.f90:118,120': ["CALL dom_uniq( zprw, 'U' )", 'zprt(ji,jj) = ssumask(ji,jj) * zprw(ji,jj)                        !    ! unique point mask', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/trdglo.f90:108': 'zvm = e1e2t(ji,jj) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))) * tmask(ji,jj,jk) * tmask_i(ji,jj)',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domqco.f90:257,266': [('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)   !==  ratio at t-point  ==!', 2), ('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2), 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/domain.f90:198,212': ['ht_0(:,:) = ht_0(:,:) + e3t_3d(:,:,jk) * tmask(:,:,jk)', 'r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:) + 1._wp -  ssmask (:,:) )', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dommsk.f90:210': 'umask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji+1,jj  ,jk)',
    'zdftke_round331.patch:8-24': ['@@ -254,0 +259,6 @@', '+         CALL r331_after_solve(jj,en(ntsi:ntei,jj,1:jpkm1))', 17],
    'zdftke_round332.patch:12-23': ['@@ -185,3 +187,5 @@', '+            & zmxld(ntsi:ntei,1:jpkm1),p_pdlr(ntsi:ntei,jj,2:jpkm1))', 12],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/zdftke_round336.patch:13-52': ['@@ -181,0 +191,5 @@', '+         r336_pre_pdlr_avt(:,jj,1:jpkm1)=p_avt(ntsi:ntei,jj,1:jpkm1) ! R336_TKE_COPYOUT', 40],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:80-128': ['SUBROUTINE r336_tke_begin(kt,Kbb,Kmm,pen,pavt,pavm,psh2,pdissl,prn2,prn2b, &', '& after_tke_en(A2D(0),jpk),after_tke_pdlr(A2D(0),jpk))', 49],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:236-288': ['WRITE(unit) magic,2,nkt,nkbb,nkmm,narea-1,jpi,jpj,jpk,ntsi,ntei,ntsj,ntej, &', "CALL put3(unit,'tke_wmask',entry_wmask,1,1,1)", 53],
    'ZDF/zdfdrg.F90:117-127': ['IF( ln_drgice_imp ) THEN', ('END_2D', 2), 11],
    'ZDF/zdfdrg.F90:171-190': ['IF( l_log_not_linssh ) THEN', ('END_2D', 4), 20],
    'ZDF/zdfdrg.F90:213-221': [('SELECT CASE( ndrg )', 1), ('END_2D', 6), 9],
    'do_loop_substitute.h90:117-125': ['#define DO_1Di(L, R) DO ji = ntsi-(L), ntei+(R)', '#define END_2D   END DO   ;   END DO', 9],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/tke_round336_writer.F90:116-135': ['ALLOCATE(entry_en(A2D(0),jpk),entry_avt(A2D(0),jpk), &', 'entry_drag_top=ptop(ntsi-1:ntei+1,ntsj-1:ntej+1)', 20],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:203-229': ['require(magic == MAGIC, f"magic {magic!r} != {MAGIC!r}")', 'f"{name}: wrong origin {(i0, j0, k0)}")', 27],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:319-324': ['drag_i0, drag_i1 = header["ntsi"] - 2, header["ntei"] + 1', '"bottom drag differs from -rn_Cd0*rn_Uc0*ssmask")', 6],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/make_stub_records.py:99-135': ['def write_record(path: Path, rank: int, tracer: dict, ldf: dict,', 'stream.write(values.tobytes(order="F"))', 37],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1019-1060': ['# The source configuration is an acquisition build with fifteen inherited', '"$(find -L "$R336_PRISTINE_BLD/obj" -maxdepth 1 -type f -name \'*.o\' | wc -l)"', 42],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1460-1472': ['compare_whole_executable_fp "$R336_PRISTINE_BLD" "$dry_bld" r336-whole \\', '"$r336_function_count" "$r336_instruction_count" "$r336_stage_count"', 13],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1473-1492': ['# Recompile the inherited round-326 stage observer for the plant.  The', 'cp "$scratch/stage.clean.o" "$dry_bld/obj/stprk3_stg.o"', 20],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1689-1737': ['while IFS= read -r -d \'\' source; do cp -a "$source" "$TARGET_ROOT/EXP00/$(basename "$source")"; done \\', 'patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$R336_TKE_PATCH"', 49],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1157-1233': ["command -v objdump >/dev/null || refuse 'no objdump for TKE arithmetic passivity proof'", "printf 'R336_HISTORICAL_TKE_RESULT admitted_rounds=331,332 arithmetic_differs_from_production=yes\\n'", 77],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:35-53': ['scratch=$(mktemp -d /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/.r336-tke-driver.XXXXXX)', 'trap cleanup EXIT', 19],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:131-205': [('if ! R336_TKE_STUB_FLOW_ROOT=$base/success R336_TKE_STUB_NEMO=$stub \\', 1), "printf 'R336_TKE_STUB_FLOW_PASS cases=success,in-place,refusal,impossible-output,retention,retained-admission\\n'", 75],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:92-119': ['READ_SET_FIELDS = {', '"h_i", "hsw", "phioc", "ut0sd", "vt0sd", "utau", "vtau",', 28],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:301-373': ['if plant == "source-values":', '"taum differs from admitted surface source")', 73],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/stprk3.f90:153-155': ['!  VERTICAL PHYSICS', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:199-204': ['CALL r336_tke_begin(kt,Kbb,Kmm,en,p_avt,p_avm,p_sh2,dissl,rn2,rn2b,', 'CALL tke_tke( Kbb, Kmm, p_sh2, p_avm, p_avt, z_pdlr )', 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:480-503': ['DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)                 ! First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1', 'r336_after_solve(:,jj,1:jpkm1)=en(ntsi:ntei,jj,1:jpkm1) ! R336_TKE_COPYOUT', 24],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1193-1224': ['terminal = work[..., jpkm1] / diagonal[..., jpkm1]', '* jnp.asarray(w_active, dtype=solved_prefix.dtype))', 32],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:136-238': ['def measure(record_root: Path, deck_root: Path,', '"after_solve": "actual production-card JIT output",', 103],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:245-317': ['def _instrumented_production_tke_solve_impl(', 'static_argnames=("source_rounded",),', 73],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_walk.py:333-547': ['def _evaluate_compiled_solver(', '"downstream": fixed_downstream,', 215],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1111-1224': ['def _nemo_literal_tke_solve(', '* jnp.asarray(w_active, dtype=solved_prefix.dtype))', 114],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:1826-1834': [('N = e_old.shape[-1]', 2), 'rhs = jnp.where(is_bottom, _e_bd[..., jnp.newaxis], rhs)', 9],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3588-3589': ['_statement_after_boundaries = jnp.concatenate(', ('[_surface_row, tke_curr], axis=-1)', 1), 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:303-309': ['DO ji = ntsi-( 0), ntei+( 0 )             ! bottom friction', 'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin ) * ssmask(ji,jj)', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TKE/BLD/ppsrc/nemo/zdftke.f90:431-436': ['zzd_up = zcof * MAX(  p_avm(ji,jj,jk+1) + p_avm(ji,jj,jk  ) , 2.e-5_wp  )         &  ! upper diagonal', 'zd_up(ji,jk) = zzd_up               ! Matrix (zdiag, zd_up, zd_lw)', 6],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:12719-12748': ['um3, vm3 = compute_face_masks_3d(_zc.is_active, _grid)', 'return ssmask * nemo_bottom_tke_dirichlet(r_t, u_sum, v_sum, vmix.tke)', 30],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13134-13151': ['fold_top_v = bool(getattr(tke_cfg, "nemo_sh2_fold_top_v", False))', 'jnp.asarray(_raw.vmask[-1], dtype=v_mask.dtype))', 18],
    'packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py:5967-5968': [('north = jnp.zeros_like(south)', 2), ('v_mask = jnp.concatenate([south, v_mask_interior, north], axis=0)', 2), 2],
    'src/OCE/DOM/dommsk.F90:146-172': ['!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)', "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )      ! Lateral boundary conditions", 27],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2470': 'nemo_sh2_fold_top_v=True,',
    'tests/ocean/unit/test_nemo_sh2_fold_top_v.py:118-311': ['def _bottom_bc_production_case(flag: bool, fold: bool = True):', '_bottom_bc_production_case(True, fold=False)', 194],
    'tests/ocean/fidelity/test_orca2_round336_tke_walk.py:126-207': ['def test_admitted_record_compiled_solver_and_bottom_dirichlet_fix() -> None:', 'assert fixed_score["max_abs"] == 0.0', 82],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_d122_gate.py:311-626': ['def measure(', '"d122_c_and_d": True,', 316],
    'tests/ocean/fidelity/test_orca2_round336_d122_gate.py:22-54': ['def test_d122_self_test_passes_and_plants_fire(tmp_path: Path) -> None:', 'assert d1["_max_index"] == (0, 0, 1)', 33],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:14200-14202': ['tke_bottom_dirichlet=self._tke_bottom_dirichlet(', 'u_now=u_now, v_now=v_now),', 3],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2t_tke_entry_gate.py:93': 'card_bottom = np.asarray(jax.jit(model._tke_bottom_dirichlet)(state))[:, :90]',
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_d122_gate.py:952-1411': ['def caller_audit_dump(', '"callers": callers_report,', 460],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:344-361': ['if source is not None:', '"TKE live e3w(Kmm) differs from admitted round-333 replay")', 18],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:362-373': ['if ice is not None:', '"taum differs from admitted surface source")', 12],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:440-483': ['def output_contract(same_program_path: Path, writer_source: Path) -> dict[str, object]:', '"files": sorted(actual)}', 44],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:593-608': ['compare_r336_twin() {', 'printf \'R336_OBSERVER_FREE_TWIN_PASS compared=%s observer_records=2\\n\' "$count"', 16],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1696-1699': ['if [[ "$ACQ_TARGET" == tke336 && -z ${R336_TKE_STUB_FLOW_ROOT:-} ]]; then', 'cp "$BINARY" "$r336_twin_binary"', 4],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1771-1814': ['same_program "$TARGET_ROOT/BLD" "$manifest/same_program.json"', ('admit', 16), 44],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:197-204': ['readonly R336_TRACER_REFERENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_tracer_stage3_kt2_np2', 'readonly R336_ICE_R1_SHA=4c69a6733b13a8f60d419e6cd1e42a1e0f15c902f9d7804c9431ef1d6d717243', 8],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_gate.py:337-430': ['expected_final = np.maximum(', 'ice=reference_digests[f"ice_r{rank:04d}"])', 94],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:84-97': ['for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \\', 'printf \'fabricated\\n\' >"$R336_TKE_STUB_FABRICATED_OUTPUT"', 14],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round336_tke_acquisition/run.sh:166-177': ['if R336_TKE_STUB_FLOW_ROOT=$base/fabricated R336_TKE_STUB_NEMO=$stub \\', "printf 'PREFLIGHT_PLANT_FIRED r336-tke-impossible-stub-output\\n'", 12],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:576-590': ['compare_deterministic() {', 'printf \'DETERMINISTIC_PASSIVITY_PASS %s files=%s\\n\' "$candidate" "$count"', 15],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1503-1535': ['admit() {', '[[ -z ${R336_TKE_STUB_FLOW_ROOT:-} ]] || r336_tke_stub_assert', 33],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:881-888': ["# Syntax proof uses the record program's stprk3_stg, not pristine NEMO.", 'patch -s --fuzz=0 -p0 -d "$dry" <"$STAGE_PATCH"', 8],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1691-1705': [('if [[ "$ACQ_TARGET" != tke336 ]]; then', 5), 'patch -s --fuzz=0 -p0 -d "$TARGET_ROOT/MY_SRC" <"$STAGE_PATCH"', 15],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh:1235-1310': ['# Audit every admitted ORCA2 recorder in the requested round set against an', "printf 'R336_HISTORICAL_OBSERVER_RESULT equal=r5,r326,r328-zdf,r333-ldf-arithmetic different=r328-dyn,r331,r332,r333-stage\\n'", 76],
    'zdf_oce.F90:70-72': ['ALLOCATE( avm (jpi,jpj,jpk), avm_k(jpi,jpj,jpk), avs(A2D(0),jpk)', '&      avmb(jpk)        , avtb(jpk)         , avtb_2d(A2D(0))', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3.f90:183': 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )',
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfphy.f90:365-392': ['!==  ocean Kz  ==!   (avt, avs, avm)', 'IF( ln_zdfiwm )   CALL zdf_iwm( kt, Kmm, avm, avt, avs )', 28],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfevd.f90:107-110': [('DO jk =  1,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 2), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfddm.f90:163-172': ['zavfs = rn_avts / ( 1. + zrr*zrr*zrr ) * zmsks * zmskf', 'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + MAX( zavft + zavdt, zavfs + zavds )', 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/zdfiwm.f90:293-317': ['DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)    ! Bound diffusivity by molecular value and 100 cm2/s', ('END DO   ;   END DO', 10), 25],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:180-216': ["IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN", ('zwt(:,1) = 0._wp', 1), 37],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:230-235': [('DO jk =    1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)', 2), ('ENDIF', 11), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:268-273': ['!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1', ('END DO   ;   END DO', 11), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:283-291': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', ('END DO   ;   END DO', 13), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/trazdf.f90:293-299': ['!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk', ('END DO   ;   END DO', 14), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdfphy.f90:369-404': ['!* start from turbulent closure values', ('& avt(ntsi:ntei,ntsj:ntej,2:jpkm1),avs(ntsi:ntei,ntsj:ntej,2:jpkm1)) ! R336_ZDFPHY_OBSERVER', 5), 36],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:185-240': ["IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN", ('ENDIF', 11), 56],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:273-278': ['!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1', ('END DO   ;   END DO', 11), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:288-296': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', ('END DO   ;   END DO', 13), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/trazdf.f90:298-304': ['!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk', ('END DO   ;   END DO', 14), 7],
    'implicit_solver.py:343-406': ['def pivot(d):', ('return solution', 2), 64],
    'implicit_solver.py:492-553': ['def implicit_vertical_diffusion_nemo_tracer_pair(', ('return out_1, out_2', 2), 62],
    'implicit_solver.py:556-655': ['def nemo_tracer_tridiagonal(', 'return lower, diagonal, upper', 100],
    'k_profiles.py:346-365': ['K_vmix, A_vmix, tke_new = _vmix_K_profiles(', 'A_v_total = A_v_total + A_vmix', 20],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:428': 'CALL sbc_ssm ( kt, Kbb, Kmm )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:440-449': [('SELECT CASE( nsbc )', 2), ('END SELECT', 4), 10],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:499-503': [('SELECT CASE( nn_ice )', 2), ('END SELECT', 5), 5],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:517-521': ['CALL sbc_rnf( kt )', 'CALL sbc_fwb( kt, nn_fwb, nn_fsbc, Kmm )', 5],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:565-570': ['DO jj = ntsj-( nn_hls-1)', ('END DO   ;   END DO', 5), 6],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:128': 'CALL sbc        ( kstp, Nbb, Nbb )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:185-197': ['CALL stp_RK3_stg( 1, kstp', 'CALL stp_RK3_stg( 3, kstp', 13],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/nemogcm.f90:368': 'Nbb = 1   ;   Nnn = 2   ;   Naa = 3',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:582-585': [('ENDIF', 21), 'DEALLOCATE( zFu, zFv, zFw )', 4],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:149': 'CALL fld_read( kt, nn_fsbc, sf_ice )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:158': 'CALL eos_fzp( sss_m(:,:), fr_i(:,:) )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:166': 'ts(ji,jj,1,jp_tem,Kmm) = MAX',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcice_if.f90:177': 'qsr(ji,jj) = ( 1. - zfr_obs ) * qsr(ji,jj)',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbc_oce.f90:195-216': ['ALLOCATE( utau(jpi,jpj)', 'wndm(Nis0-(0):Nie0+(0)', 22],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/lbcnfd.f90:588-606': [('DO jj = 1, ihls', 9), ('END DO', 85), 19],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/lbcnfd.f90:619-628': [('! line number ipj-ihls : right half', 5), ('END DO', 89), 10],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dommsk.f90:232': "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )",
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:444-521': ['CASE( jp_blk     )   ;   CALL sbc_blk       ( kt )', 'CALL sbc_fwb( kt, nn_fwb, nn_fsbc, Kmm )', 78],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:376-384': [('IF( nn_ice == 0 ) THEN', 2), 'IF( nn_ice == 3 )   CALL cice_sbc_init( nsbc, Kbb, Kmm )', 9],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:398,405,408': ['ssh(:,:,Kbb) = 0._wp', 'ssh(:,:,Kaa) = ssh(:,:,Kbb)           !*  set ssh at Kaa (for AGRIF)', 'ssh(:,:,Kmm) = ssh(:,:,Kbb)           !*  set now values from to before ones', 3],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/istate.f90:118': 'CALL dta_tsd( nit000, ts(:,:,:,:,Kbb) )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dtatsd.f90:308-309': ['ptsd(ji,jj,jk,jp_tem) = ptsd(ji,jj,jk,jp_tem) * tmask(ji,jj,jk)    ! Mask', 'ptsd(ji,jj,jk,jp_sal) = ptsd(ji,jj,jk,jp_sal) * tmask(ji,jj,jk)', 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90:656-657': ['zpre(:,:)         = pres_temp( q_air_zt(:,:), sf(jp_slp)%fnow(:,:,1), rn_zqt, pta=sf(jp_tair)%fnow(:,:,1) )', 'theta_air_zt(:,:) = theta_exner( sf(jp_tair)%fnow(:,:,1), zpre(:,:) )', 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90:740-876': ['ptsk(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) = pst(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) + rt0', 'pevp(:,:) = pevp(:,:) * smask0(:,:)', 137],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcblk.f90:925': 'ptsk(:,:) = ( ptsk(:,:) - rt0 ) * smask0(:,:)  ! Back to Celsius',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/tradmp.f90:181': 'CALL dta_tsd( kt, zts_dta )            ! read and interpolates T-S data at kt',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/fldread.f90:192,209-215': ['ll_firstcall = kt == nit000', 'IF( ll_firstcall ) THEN                      ! initialization', ('ENDIF', 3), 8],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:146-155': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:176': 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:482-495': ['IF( ln_icebergs ) THEN  ! save pure stresses (with no ice-ocean stress) for use by icebergs', ('ENDIF', 25), 14],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcmod.f90:507': 'IF( ln_icebergs    )   CALL icb_stp( kt, Kmm )',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfphy.f90:327-337': ["CALL zdf_drg( kt, Kmm, 'BOTTOM' )", 'CASE( np_GLS )   ;   CALL zdf_gls( kt, Kbb, Kmm, sh2, avm_k, avt_k )', 11],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdfdrg.f90:414-416': ['IF( ln_isfcav.OR.ln_drgice_imp ) THEN              ! Ocean cavities: top friction setting', ('ENDIF', 14), 3],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:260-275': ['SELECT CASE ( nn_eice )', ('END DO', 1), 16],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:365,501': ['zus3(ji) = MAX( 0._wp, 1._wp - zice_fra(ji) ) * zus * zus * zus * tmask(ji,jj,1)', ('&                                 * MAX( 0._wp, 1._wp - zice_fra(ji) ) * wmask(ji,jj,jk) * tmask(ji,jj,1)', 1), 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/zdftke.f90:608-638': ['CASE( 0 )                      ! No scaling under sea-ice', ('ELSE', 7), 31],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:137-200': ['CALL eos    ( ts, Kbb, rhd )', ('END DO   ;   END DO', 4), 64],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:198-199': ['Ue_rhs(ji,jj) =  Ue_rhs(ji,jj) + r1_rho0 * utauU(ji,jj) * (r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kbb)))', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj) * (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kbb)))', 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stp2d.f90:242-267': ['sshe_rhs(:,:) =                 emp(:,:)', '&   CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa ) ! time-splitting', 26],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dynspg_ts.f90:650-652': ['puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) / r1_wgt1s', 'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 3],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/dynspg_ts.f90:1246-1247,1277-1300': ['pCdU_v(ji,jj) = r1_2*( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) )', ('END DO   ;   END DO', 37), ('IF( ln_isfcav.OR.ln_drgice_imp ) THEN', 2), ('ENDIF', 53), 26],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:131-175': ['ssha(:,:) = ssh (:,:,Kaa)     ! save ssh, uu_b, vv_b at N+1  (computed in dynspg_ts)', '! r3f already properly set up                           ! at N     (Kmm)', 45],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:270-313': ['zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)', '!              !---------------!   Vector Inv. Form : 1st stage 3D RHS already entirely computed in stp_2D', 44],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:366-421': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN   !* applied on velocity', ('DEALLOCATE( zub, zvb )', 2), 56],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:431,474-507,572-573': [('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), 'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )', ('END DO', 14), ("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp   &", 2), "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 37],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:503-505': ['ts(ji,jj,jk,jn,Kaa) = (        ( 1._wp + r3t(ji,jj,Kbb) )*ts(ji,jj,jk,jn,Kbb )                      &', '&                /          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:405,538': ['IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/trasbc.f90:283-287,307-308,318-328': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 9), ('END DO   ;   END DO', 9), 'pts(ji,jj,1,jp_tem,Krhs) = pts(ji,jj,1,jp_tem,Krhs) +  r1_rcp * qns(ji,jj) * z1_rho0_e3t', 'pts(ji,jj,1,jp_sal,Krhs) = pts(ji,jj,1,jp_sal,Krhs) +           sfx(ji,jj) * z1_rho0_e3t', ('IF( ln_rnf ) THEN         ! input of heat and salt due to river runoff', 2), ('ENDIF', 24), 18],
    'packages/ocean/legoesm/ocean/forcing/nemo_surface.py:348-358': ['utau_u, vtau_v = nemo_t_stress_to_uv(', (')', 126), 11],
    'packages/ocean/legoesm/ocean/forcing/nemo_surface.py:406-414': ['q_solar=None,', 'result.runoff.salinity_content,', 9],
    'packages/ocean/legoesm/ocean/state.py:378-379': ['utau_u_native: object = None', 'vtau_v_native: object = None', 2],
    'packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4810-4835': ['_native_u = getattr(surface_forcing, "utau_u_native", None)', ('return tau_i_u, tau_j_v, dz_0_u, dz_0_v', 1), 26],
    'packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4837-4869': ['tau_i_T = jnp.asarray(_native_i, dtype=u_dtype)', ('return tau_i_u, tau_j_v, dz_0_u, dz_0_v', 2), 33],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7273-7290': ['_sfx = (surface_stress_faces(', '_tau_i_u, _tau_j_v, _, _ = _sfx', 18],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7349-7361': [('F_slow_u = nemo_source_round(', 1), ('_F_slow_wind_v = F_slow_v', 1), 13],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2923-2927': [('out = (', 1), ('return jnp.where(active_3d > 0.5, out, base)', 1), 5],
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1084-1088': ['if (int(nemo_kt) - 1) % int(s0.forcing_stride) == 0:', 'state = state._replace(T=state.T.replace(data=temperature))', 5],
    'round340_surface_writer.F90:23-25': ['IF(kid >= 1) WRITE(iu) sst_m(i0:i1,j0:j1),sss_m(i0:i1,j0:j1), &', '& e3t_m(i0:i1,j0:j1),frq_m(i0:i1,j0:j1)', 3],
    'nemo_testcase_l4_orca2_round340_standalone_gate.py:57-68': ['OUTER_FULL_FIELDS = (', '2: OUTER_FULL_FIELDS[:7],', 12],
    'nemo_testcase_l4_orca2_round340_standalone_gate.py:223-225': ['for name in OUTER_FIELDS_BY_KID[kid]:', ('fields[name] = _reduced(value, ni, nj)', 1), 3],
    'packages/ocean/legoesm/ocean/forcing/nemo_surface.py:27-34': ['class NemoSurfaceMeans(NamedTuple):', 'frq: jnp.ndarray', 8],
    'nemo_testcase_l4_orca2_round340_standalone_gate.py:935-940,1009-1011': ['means = surface.means', ('}', 101), ('for kid, boundary in OUTER_BOUNDARIES.items():', 2), 'OUTER_FIELDS_BY_KID[kid], outer_values[kid])', 9],
    'nemo_testcase_l4_orca2_round340_standalone_gate.py:1176-1185': ['require(tuple(actual[boundary]) == tuple(expected[boundary]),', 'rows[key] = row', 10],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:109-176': ['IF( kt == nit000 .AND. .NOT. l_ssm_mean ) THEN   !   Initialisation: 1st time-step, no input means !', ('ENDIF', 8), 68],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/sbcssm.f90:133-162': ['ELSEIF( MOD( kt - 2 , nn_fsbc ) == 0 ) THEN      !   Initialisation: New mean computation   !', 'frq_m(:,:) = frq_m(:,:) + fraqsr_1lev(:,:)', 30],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:1289-1293': ["IF( iom_varid( numror, 'fraqsr_1lev', ldstop = .FALSE. ) > 0 ) THEN", ('ENDIF', 24), 5],
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1046-1070': ['temperature = jnp.asarray(state.T.data, dtype=jnp.float64)', (')', 446), 25],
    'packages/ocean/legoesm/ocean/forcing/nemo_surface.py:143-187': ['first_from_rest = jnp.logical_and(', (')', 68), 45],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:215,525': [('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )   ! at N+1/2 (Kaa)', 2), 'IF( ln_traqsr  )   CALL tra_qsr( kstp,      Kmm, ts, Krhs )  ! penetrative solar radiation qsr', 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:327-402': [('zlogc = LOG(  MAX( 0.03_wp, MIN( sf_chl(1)%fnow(ji,jj,1) ,10._wp ) )  )  ! zlogc = log(zchl)   with 0.03 <= Chl >= 10.', 1), ('END WHERE', 2), 76],
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1116-1147': ['state_after = stepped.state_after if hasattr(stepped, "state_after") else stepped', 'shortwave_fraction_top=next_shortwave_fraction,', 32],
    'packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:1184-1207': ['def top_layer_absorbed_fraction(dz_top, chl_surface=None,', 'return frac_per_m[..., 0] * dz_live[..., 0]', 24],
    'packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:460-554': ['def _nemo_qsr_rgb_attenuation_coeffs(', 'return jnp.where(sw_down != 0.0, sr(one - sr(zzeT / safe_qsr)), one)', 95],
    'packages/ocean/legoesm/ocean/physics/shortwave_penetration.py:410-457': ['zlogc = precision_log(jnp.clip(chl_surface, c(_CHL_MIN), c(_CHL_MAX)))', 'return jnp.clip(itab - 1.0, 0.0, 60.0).astype(jnp.int32)', 48],
    'packages/ocean/legoesm/ocean/eos.py:1004-1035': ['def nemo_r3t_rk3_stage1_stretch(', ('return jnp.where(wet, nemo_source_round(one + r3_stage), one)', 1), 32],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/traqsr.f90:310': 'CALL fld_read( kt, 1, sf_chl )                                          ! Read Chl data and provides it at the current time step',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3_stg.f90:525': 'IF( ln_traqsr  )   CALL tra_qsr( kstp,      Kmm, ts, Krhs )  ! penetrative solar radiation qsr',
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1319-1391': ['update = (int(s0.kt) - 1) % int(s0.forcing_stride) == 0', 'ice_if=retained_iif,', 73],
    'tke.py:285-289': ['masked_taum = jnp.maximum(taum, 0.0) * surface_tmask', '* masked_taum)', 5],
    'trabbc.F90:179': '* 1.e-3 ! conversion in W/m2',
    'phycst.F90:26': 'rad      = 3.141592653589793_wp / 180._wp',
    'tramle.F90:51': 'r5_21 = 5.e0 / 21.e0',
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90:210-229': [('!                          ! set ssh and (uu_b,vv_b) at N+1/2  (Kaa)', 1), ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90:301-305': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1), ('* r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/traldf_iso.f90:318-322': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 2), ('* r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R333LDFI/BLD/ppsrc/nemo/stprk3_stg.f90:745-771': [('IF( .NOT.ln_tile .AND. kstp == nit000 + 1 .AND. kstg == 3 ) & ! R333_OBSERVER', 3), ('CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )  ! vertical mixing and after tracer fields', 1), 27],
    'gm_redi_latlon_cgrid.py:3314-3319': ['e3t_divisor = e3t if divisor_thickness is None else jnp.asarray(',
 'f"{q.shape}, got {e3t_divisor.shape}")',
 6],
    'gm_redi_latlon_cgrid.py:3338-3354': ['scaled_div = total_div * r1_e1e2t[:, :, jnp.newaxis]', 'tend = tend * act', 17],
    'ocean_model_latlon_cgrid.py:192-212': ['def _nemo_stage3_l8_krhs_walk(adv_t, adv_s, *term_groups):',
 'return t_value, s_value, tuple(boundaries)',
 21],
    'ocean_model_latlon_cgrid.py:215-231': ['def _nemo_safe_stored_ldf_divisor(stored_thickness, active_3d):',
 'jnp.ones_like(stored_thickness))',
 17],
    'ocean_model_latlon_cgrid.py:8948-8962': ["# Stage 3's Kmm: NEMO's stored e3t_0*(1+r3t(Kmm)), r3t(Kmm) being",
 '_h_ref_ws * _stretch_s3_kmm[..., None])',
 15],
    'ocean_model_latlon_cgrid.py:10821-10834': ['# Stage-3 tra_ldf receives the stage-2 output as Kmm',
 '# traldf_iso.f90:301-305/:318-322 reads the already-stored',
 14],
    'ocean_model_latlon_cgrid.py:10834-10850': ['# traldf_iso.f90:301-305/:318-322 reads the already-stored', ('else None)),', 3), 17],
    'ocean_model_latlon_cgrid.py:11898-11942': ['_tke_result = self._apply_implicit_vertical_mixing(',
 'state_new, tke_new, _nemo_ws_vertical_solve_trace = (_tke_result if _return_vertical_solve_trace '
 'else (*_tke_result, None))',
 45],
    'ocean_model_latlon_cgrid.py:15139-15168': [('from legoesm.ocean.physics.vertical_mixing import (', 3),
 'rounded_thickness=_zdf_tracer_rounded))',
 30],
    'src/OCE/TRA/traldf_iso.F90:145': [('#           define   INN                0', 1)],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:604': 'ts(:,:,:,jn,Krhs) = 0._wp',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:637': 'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:645': 'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:717': 'CALL tra_qsr( kstp, Kmm, ts, Krhs )',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:734': 'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:287-291': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1), ('* r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf.f90:105-122': [('SELECT CASE ( nldf_tra )   !* compute lateral mixing trend and add it to the general trend', 2), ('END SELECT', 2), 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:154-160': [('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1), ('DO jk = 1, jpkm1', 1), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:183-204': [('zdit(ji,jj,ik  ) = ( pt(ji+1,jj,jk  ,jn,Kbb) - pt(ji,jj,jk  ,jn,Kbb) ) * umask(ji,jj,jk  )', 1), ('zdkt(ji,jj,ikp1) = ( pt(ji,jj,jk,jn,Kbb) - pt(ji,jj,jk+1,jn,Kbb) ) * wmask(ji,jj,jk+1)', 2), 22],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:230-246': [('zA11 = e2_e1u(ji,jj) * (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk)))', 1), ('&                                             + ( zdkt(ji,jj+1,ikp1) + zdkt(ji,jj,ik  ) )  )   )', 1), 17],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:259-291': [('IF( jk ==1 )   zfw(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp', 1), ('* r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:297-304': [('ELSEIF( jk == jpkm1 ) THEN', 2), ('* r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 8],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traldf_iso.f90:335-341': [("IF( ll_ptr )  CALL dia_ptr_hst( jn, 'ldf', -zfv(:,:) )", 1), 'END SUBROUTINE traldf_iso_lap', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:738-740': ['IF( ln_trabbc  )   CALL tra_bbc( kstp,      Kmm, ts, Krhs )', 'IF( ln_tradmp  )   CALL tra_dmp( kstp, Kbb, Kmm, ts, Krhs )', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:749': 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )',
    'ocean_model_latlon_cgrid.py:2988-2989': ['fd2_a, fd2_b, rhs2_a, rhs2_b, fct_activity, fct_inputs = _flux_pair(', 'a2, b2, dt, 2, h_k_new)', 2],
    'ocean_model_latlon_cgrid.py:208-210': ['t_value = sr(t_value + term)', 's_value = sr(s_value + term)', 3],
    'ocean_model_latlon_cgrid.py:11239-11245': ['_stage3_l8_krhs_terms = (', '((_dmp_t,), (_dmp_s,)),', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stp2d.f90:139-147': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )', 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynvor.f90:796-802': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 4), ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2), 7],
    'ocean_pe_latlon_cgrid.py:3098': 'dv_dt = dv_dt + diag_vortcor_v',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:708-719': [('DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)', 2), 'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk), avtb_2d(ji,jj) * avtb(jk) ) * wmask(ji,jj,jk)', 12],
    'tke.py:2267-2292': ['kappa_convention = getattr(cfg, "kappa_convention", "gaspar_sqrt2e")', 'K_M = jnp.minimum(cfg.kappaM_max, K_M)', 26],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:795-796': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp", 2), "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/lbcnfd.f90:684-717': [("CASE ( 'V' )                               ! V-point", 5), ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = psgn(jf) * ptab(jf)%pt4d(ii2,ij2,jk,jl)', 57), 34],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynhpg.f90:402,427': ['pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap', 'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:467-470': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 4],
    'ocean_model_latlon_cgrid.py:18817-18838': ['def _rk3_fold_exchange_top_v(v, grid, v_mask_3d):', 'fold.vector_sign_v * v[-2][perm] * v_mask_3d[-2][perm].astype(v.dtype))', 22],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:687-700': ['CASE ( 3 )           ! lup and ldown, |dk[xml]| bounded by e3t :', 'zmxld(ji,jk) = zemlp', 14],
    'tke.py:904-1022': ['l_w = jnp.concatenate([l_sfc[..., None], l_int], axis=-1)  # (..., nlev)', 'l_eps = jnp.maximum(jnp.sqrt(lup * ldn), mxl_min)', 119],
    'tke.py:2267-2329': ['kappa_convention = getattr(cfg, "kappa_convention", "gaspar_sqrt2e")', 'K_H = jnp.maximum(cfg.kappaH_min, K_M / Pr_or_pdlr)', 63],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/dynspg_ts.f90:1574-1606': ['zfield(:,:,:) = 0._wp', 'IF( llwet ) zfield(ji,jj,jk) = pfield(ji,jj,jk)', 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:343-346': ['WRITE(iunit) magic, 1, kstp, kstage, klevel, jpi, jpj, jpk, jpts, STORAGE_SIZE(1._wp)', "& l4_canon_2d(ssh(:,:,klevel),'T')", 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:791-793': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp", 1), ("&                       , avm_k(:,:,:)        , 'W',  1._wp, ldfull=.TRUE. )", 2), 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:621-637': ['CASE( 2 )                      ! scaling with mean sea-ice thickness', 'zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )', 17],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:717-720': ['IF( nn_pdl == 1 ) THEN          !* Prandtl number case: update avt', ('END DO   ;   END DO', 23), 4],
    'tke.py:262-268': ['zraug = _NEMO_MXL0_VKARMN * _NEMO_MXL0_LENGTH_SCALE / (rho_0 * g)', 'jnp.asarray(_mxl0_anchor_floor(cfg), dtype=taum.dtype), raw)', 7],
    'tke.py:4219-4226': ['def _nemo_mxlice2_anchor_raw(fr_i, hm_i, taum, zraug, surface_tmask):', '* jnp.asarray(surface_tmask, dtype=taum.dtype))', 8],
    'tke.py:2306-2329': ['_nemo_literal_pdlr = (', 'K_H = jnp.maximum(cfg.kappaH_min, K_M / Pr_or_pdlr)', 24],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:183': 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )',
    'nemo_testcase_l4_orca2_round331_tke_gate.py:128-150': ['def _verify_source_order(record: dict) -> None:', 'require(bool(np.any(f["matrix_diag"] != 0.0)), "matrix channel is inactive")', 23],
    'tke_round331_writer.F90:51-75': ['SUBROUTINE r331_begin(kt,Kbb,Kmm,Krhs,pinterior)', 'END SUBROUTINE r331_begin', 25],
    'tke_round331_writer.F90:157-165': ['SUBROUTINE r331_finish(pen,pavm,pavt,pwmask)', 'IF(ninput/=ny .OR. nbc/=ny .OR. nmatrix/=ny .OR. nsolve/=ny) &', 9],
    'zdftke_round326_writer.F90:34-69': ['IF( kt < nit000 .OR. kt > nit000 + jp_r326_last - 1 )   RETURN', "R326_2D('tmask_1         ',zwrk)", 36],
    'round332_writer.F90:82-125': ['SUBROUTINE r332_tke_begin(kt,Kbb,Kmm)', 'END SUBROUTINE r332_tke_row', 44],
    'round332_writer.F90:127-145': ['SUBROUTINE write_tke2(kt,Kbb,Kmm)', 'END SUBROUTINE write_tke2', 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R336TZDF/BLD/ppsrc/nemo/zdftke.f90:197-200': ['CALL tke_tke( Kbb, Kmm, p_sh2, p_avm, p_avt, z_pdlr )   ! now tke (en)', 'CALL tke_avn( Kbb, Kmm,        p_avm, p_avt, z_pdlr )   ! now avt, avm, dissl', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/istate.f90:105-140': ['IF( ln_rstart ) THEN', 'vv    (:,:,:,Kmm) = vv   (:,:,:,Kbb)', 36],
    'zdftke.f90:291-293': ['zmskv = ( 2. - vmask(ji,jj-1,mbkt(ji,jj)) * vmask(ji,jj,mbkt(ji,jj)) )', 'zebot = - 0.001875_wp * rCdU_bot(ji,jj) * SQRT(', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:266': 'IF( ln_zdftke ) THEN   ;   ioptio = ioptio + 1   ;    nzdf_phy = np_TKE   ;   CALL zdf_tke_init          ;   ENDIF',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:275-278': ['!                                ! shear production term flag', ('ENDIF', 16), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90:83-119': ['DO jk = 2, jpkm1                 !* Shear production at uw- and vw-points (energy conserving form)', ('END DO   ;   END DO', 4), 37],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:316-336': [('IF( l_zdfsh2 ) THEN', 1), 'CASE( np_TKE )   ;   CALL zdf_tke( kt, Kbb, Kmm, sh2, avm_k, avt_k )', 21],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90:98-112': ['!* 2 x shear production at uw- and vw-points', 'p_sh2(ji,jj,jk) = 0.25 *', 15],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfdrg.f90:165-187': ['SELECT CASE( TRIM(cd_topbot) )', 'rCdU_top(ji,jj) = rCdU_ice(ji,jj)', 23],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:109-123': ['! w-level of the mixing and mixed layers', 'hmlp(ji,jj) = ((gdepw_1d(iik  ) ) *(1._wp+r3t(ji,jj,Kmm))) * ssmask(ji,jj)', 15],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:500-507': ['IF( nn_etau == 1 ) THEN', 'jk = nmln(ji,jj)', 8],
    'nemo_testcase_recipe.py:611,616-617': ['tke_surface_bc_level="nemo_z0",', 'tke_matrix_evaluation="nemo_literal",', 'tke_solver_evaluation="nemo_literal",', 3],
    'tke.py:1845-1857': ['if surface_bc_level not in ("interior_pinned", "nemo_z0"):', '# ---- NEMO z=0 surface row (zdftke.F90:264-269,403-410) ----', 13],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:222-240': ['REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ,jpk)          , INTENT(in   ) ::   p_sh2', 'REAL(wp), DIMENSION(ntsi-(0):ntei+(0),jpk) ::   zpelc, zdiag, zd_up, zd_lw', 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:273-276': ['en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )', 'zd_up(ji,1) = 0._wp', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:287-295': ['IF( .NOT.ln_drg_OFF ) THEN', 'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin ) * ssmask(ji,jj)', 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428': ['!* Matrix and right hand side in en', (') * wmask(ji,jj,jk)', 2), 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:459-477': ['!* Matrix inversion from level 2', 'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 19],
    'nemo_testcase_recipe.py:2213-2220': [('z_coord = create_partial_cell_coordinate(', 2), ('is_active=jnp.asarray(tmask),', 1), 8],
    'tke.py:1803-1834': ['if bottom_dirichlet is not None:', 'rhs = jnp.where(is_bottom, _e_bd[..., jnp.newaxis], rhs)', 32],
    'nemo_testcase_recipe.py:2449': 'bottom_tke_bc=True,',
    'nemo_testcase_v1_orca2tke/namelist_cfg:193': 'ln_drg_OFF  = .true.',
    'gridladder_vmix/nemo_testcase_recipe.py:5083,5099': ['bottom_tke_bc=False,', 't.bottom_tke_bc, t.tke_langmuir_evaluation, t.nemo_mxlice,', 2],
    'dino.py:1292': '"tke_bottom_bc": True,',
    'nemo_testcase_l4_orca2_round330_dino_probe.py:88-170': ['def _tke_probe():', 'json.dumps(result, indent=2, sort_keys=True) + "\\n", encoding="utf-8")', 83],
    'tke.py:1812-1817': ['if bottom_level is None:', 'rhs = rhs.at[..., -1].set(_e_bd)', 6],
    'tke.py:1664-1670': ['# zd_up(jpkm1) enters zdiag(jpkm1) (zdftke.f90:419) but NOT the', 'jnp.zeros_like(literal_up[..., :1])], axis=-1)', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:121': 'ALLOCATE( zFu(jpi,jpj,jpk), zFv(jpi,jpj,jpk), zFw(jpi,jpj,jpk) )',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:323-351': [('IF( ln_dynadv_vec ) THEN', 1), ("WRITE(l1_unit) l4_canon_3d(zFu,'U'), l4_canon_3d(zFv,'V'), l4_canon_3d(zFw,'T')", 1), 29],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traadv.f90:196': 'IF( ln_dynadv_vec )   ll_Fw = .true.',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/traadv.f90:296-312': ['CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', ('END DO   ;   END DO   ;   END DO', 4), 17],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:709-711': ['zsqen = SQRT( en(ji,jj,jk) )', 'p_avm(ji,jj,jk) = MAX( zav,                  avmb(jk) ) * wmask(ji,jj,jk)', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/MY_SRC/stprk3.F90:411-422': [('INTEGER, INTENT(in) :: kstp, klevel', 3), "& l4_canon_3d(avs,'T'), l4_canon_3d(en,'T')", 12],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:336': 'CALL zdf_tke( kt, Kbb, Kmm, sh2, avm_k, avt_k )',
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:348-351': [('DO jk =  2,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 1), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:362-368': [('IF( ln_zdfddm ) THEN', 2), ('ENDIF', 19), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:370-374': ['!* wave-induced mixing', 'CALL zdf_mxl_turb( kt, Kmm )', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:380': "CALL lbc_lnk( 'zdfphy', avm, 'W', 1.0_wp )",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:621-622': [('zmxlm(ji,1) = ( ( 1._wp - fr_i(ji,jj) ) * zraug * taum(ji,jj) + &', 1), '&                      fr_i(ji,jj)   * hm_i(ji,jj) * 2._wp ) * tmask(ji,jj,1)', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:186': ('hm_i(ji,jj) = vt_i(ji,jj) * z1_at_i', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186': 'IF( MOD( kt-1, nn_fsbc ) == 0 ) THEN   ! Ice time step',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:260': 'CALL ice_var_agg(2)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:151': 'CALL sbc        ( kstp, Nbb, Nbb )                ! Sea Boundary Condition (including sea-ice)',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:319': 'nn_aei_ijk_t  = 21',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:322-323': ['rn_Ue        = 0.03', 'rn_Le        = 200.e+3', 2],
    'ORCA2_OMIP_L4/EXP00/namelist_ref:980': 'ln_eke_equ    =.false.',
    'ICE/iceupdate.F90:369': 'tmod_io(ji,jj) = rho0 * drag_io(ji,jj) * SQRT( zmodt )',
    'ICE/iceupdate.F90:391': 'rCdU_ice(ji,jj) = -r1_rho0 * tmod_io(ji,jj) * at_i(ji,jj) * tmask(ji,jj,1)',
    'ZDF/zdfdrg.F90:126': 'rCdU_top(ji,jj) = rCdU_ice(ji,jj)',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:302': 'ln_traldf_msc   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:393': 'ln_zdfddm   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:394-395': ['rn_avts  =    1.e-4', 'rn_hsbfr =    1.6', 2],
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:396': 'ln_zdfiwm   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:423': 'ln_mevar    = .false.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:424': 'ln_tsdiff   = .true.',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:134': 'SUBROUTINE tra_zdf_imp(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:175': '( jn == jp_sal .AND. ln_zdfddm )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:185': 'zwt(ji,jk) = avt(ji,jj,jk) + akz(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:203': 'zwt(ji,jk) = avs(ji,jj,jk) + akz(ji,jj,jk)',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:474': 'ln_rnf_depth = .false.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:478': 'ln_rnf_depth_ini = .false.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:480': 'rn_dep_max  = 150.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1236': 'ln_mxl0     = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1243': 'rn_mxl0     =   0.04',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1245-1246': ['ln_lc       = .true.', 'rn_lc       =   0.15', 2],
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1252': 'nn_htau     =   1',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:406': 'nn_mxl      =   3',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:410': 'nn_etau     =   1',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:415': 'nn_mxlice = 2',
    'ocean_model_latlon_cgrid.py:2157': 'rhs = sr(jnp.where(active_t, sr(-horizontal_divergence), 0.0))',
    'ocean_model_latlon_cgrid.py:12687-12691': ['cd0=_cfg_b.bottom_drag.bottom_drag_cd0,', 'uc0=_cfg_b.bottom_drag.bottom_drag_uc0,', 5],
    'ocean_pe_latlon_cgrid.py:4483': [('cd0=config.bottom_drag.bottom_drag_cd0,', 1)],
    'ocean_model_latlon_cgrid.py:12687': ['cd0=_cfg_b.bottom_drag.bottom_drag_cd0,'],
    'ocean_model_latlon_cgrid.py:14200-14202': ['tke_bottom_dirichlet=self._tke_bottom_dirichlet(', 'u_now=u_now, v_now=v_now),', 3],
    'ocean_model_latlon_cgrid.py:17350': ('u_now=state.u.data, v_now=state.v.data,', 6),
    'ocean_model_latlon_cgrid.py:17726': ('u_now=state.u.data, v_now=state.v.data,', 7),
    'nemo_testcase_recipe.py:2552': ['2920,  # DECISION 142 (user, 2026-10-20): one year, deliberate'],
    'tke.py:2329': 'K_H = jnp.maximum(cfg.kappaH_min, K_M / Pr_or_pdlr)',
    'tke.py:2174': 'pdlr = jnp.maximum(0.1, ri_cri / jnp.maximum(ri_cri, zri))',
    'tke.py:2175': 'return pdlr if return_pdlr else 1.0 / pdlr',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:405': 'p_pdlr(ji,jj,jk) = MAX(  0.1_wp,  ri_cri / MAX( ri_cri , zri )  )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:708': 'zav   = rn_ediff * zmxlm(ji,jk) * zsqen',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:710': 'p_avt(ji,jj,jk) = MAX( zav, avtb_2d(ji,jj) * avtb(jk) ) * wmask(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:717': 'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk), avtb_2d(ji,jj) * avtb(jk) )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:442': 'avtb(:) = 1.e-10_wp',
    'ocean_model_latlon_cgrid.py:2171': 'return sr(jnp.where(active_t, combined, 0.0))',
    'ocean_model_latlon_cgrid.py:2919-2921': ['out = r(r(r(q_before[..., None] * base)', '/ q_after)', 3],
    'ocean_model_latlon_cgrid.py:2927': [('return jnp.where(active_3d > 0.5, out, base)', 1)],
    'ocean_model_latlon_cgrid.py:14617': 'dt_mom, _tau_j_v, _e3v1, _r0,',
    'ocean_pe_latlon_cgrid.py:4483-4487': [('cd0=config.bottom_drag.bottom_drag_cd0,', 1), ('uc0=config.bottom_drag.bottom_drag_uc0,', 1), 5],
    'ocean_pe_latlon_cgrid.py:4568-4572': [('cd0=config.bottom_drag.bottom_drag_cd0,', 2), ('uc0=config.bottom_drag.bottom_drag_uc0,', 2), 5],
    'implicit_solver.py:377-396': ['_round_the_multiply(quotient * previous))', 'numerator / pivot(current_diagonal))', 20],
    'implicit_solver.py:452-464': ['avm_sum = jnp.asarray(2.0, dtype=dtype) * avm_face', 'avm_product / lower_denominator) * interface_wet', 13],
    'internal_wave_mixing.py:439': 'n_pos = safe_sqrt(n2_pos)',
    'tke.py:1082': 'safe_sqrt((jnp.asarray(2.0, e.dtype) * e) / zrn2),',
    'tke.py:3662': '_dissl_positive = tke_curr > 0.0',
    'shortwave_penetration.py:653': '/ jnp.where((jk <= nkB) & (ze3t != 0.0), ze3t, c(1.0)))',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:21': 'nn_itend    =    5840',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:48': 'ln_tsd_dmp  = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:161': 'ln_rnf_mouth = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:206': 'ln_icebergs = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:256': 'ln_trabbc   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:304': 'nn_aht_ijk_t    = 20',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:306': 'rn_Ud        = 0.018',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:312': 'ln_mle      = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:317': 'ln_ldfeiv   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:328': 'ln_tradmp   =  .true.',
    'namelist_ref:818': 'ln_drgice_imp = .true.',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-836': ['IF( ln_zdfiwm ) THEN          ! Internal wave-driven mixing', 'rn_emin  = 1.e-10_wp             ! specific values of rn_emin & rmxl_min are used', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:238': 'SUBROUTINE sbc_rnf_div( phdivn, Kmm )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:289': 'SUBROUTINE sbc_rnf_init( Kmm )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:486-489': ['ELSE                                            ! runoffs applied at the surface', 'h_rnf (ji,jj) = (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:199': 'SUBROUTINE tke_tke( Kbb, Kmm, p_sh2, p_avm, p_avt, p_pdlr )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:531': 'SUBROUTINE tke_avn( Kbb, Kmm, p_avm, p_avt, p_pdlr )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:644-645': ['zrn2 = MAX( rn2(ji,jj,jk), rsmall )', 'zmxlm(ji,jk) = MAX(  rmxl_min,  SQRT( 2._wp * en(ji,jj,jk) / zrn2 )  )', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:690-692': ['DO jk =  jpkm1,  2,  -1  ; DO ji = ntsi-( 0), ntei+(  0)   ! from the bottom to the surface : ldown', '&    MIN( zmxlm(ji,jk+1) + (e3t_3d(ji,jj,jk+1)', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:354': 'imlc(:) = mbkt(ntsi-(0):ntei+(0),jj) + 1',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:61': 'SUBROUTINE dyn_keg(',
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:63': 'SUBROUTINE stp_2D(',
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:138': {'anchors': 'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:141': {'anchors': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:144': {'anchors': 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:161': {'anchors': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90:61': 'SUBROUTINE dyn_keg(',
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90:129-130': {'anchors': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1), ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2], 'routine': 'dyn_keg'},
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:63': 'SUBROUTINE stp_2D(',
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:141': {'anchors': 'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:144': {'anchors': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:147': {'anchors': 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:164': {'anchors': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90:61': 'SUBROUTINE dyn_keg(',
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90:129-130': {'anchors': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1), ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2], 'routine': 'dyn_keg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171': 'CALL bn2    ( ts(:,:,:,:,Nbb), rab_b, rn2b, Nbb ) ! before Brunt-Vaisala frequency',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:174': 'rn2   = rn2b',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:161': 'zrr = zrau / rn_hsbfr',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:202': 'CASE( np_EEN )           ;   CALL vor_een( kt, Kmm, ncor, puu(:,:,:,Kmm)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:334': 'CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:160-166': [('SELECT CASE( n_dynadv )', 1), 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:266': 'IF( ln_zdftke ) THEN   ;   ioptio = ioptio + 1   ;    nzdf_phy = np_TKE   ;   CALL zdf_tke_init          ;   ENDIF',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:318': 'CALL zdf_sh2( Kbb, Kmm, avm_k,   &     ! <<== in',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:336': 'CASE( np_TKE )   ;   CALL zdf_tke( kt, Kbb, Kmm, sh2, avm_k, avt_k )    ! TKE closure scheme for Kz',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1281-1285': [('CASE( np_teos10, np_eos80 )                !==  polynomial TEOS-10 / EOS-80 ==!', 7), ('zh  = ((gdept_1d(jk) ) *(1._wp+r3t(ji,jj,Kmm))) * r1_Z0                                ! depth', 1), 5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1631-1637': ['DO jk =  2,  jpkm1  ; DO jj = ntsj-(  kbnd), ntej+(   kbnd) ; DO ji = ntsi-( kbnd), ntei+(   kbnd)      ! interior points only (2=< jk =< jpkm1 ); surface and bottom value set to zero one for all in istate.F90', ('!', 780), 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179': 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )   ! vertical physics update (top/bot drag, avt, avs, avm + MLD)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:466-472': ['zd_lw(ji,jk) = en(ji,jj,jk) - zd_lw(ji,jk) / zdiag(ji,jk-1) * zd_lw(ji,jk-1)', 'en(ji,jj,jk) = ( zd_lw(ji,jk) - zd_up(ji,jk) * en(ji,jj,jk+1) ) / zdiag(ji,jk)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfsh2.f90:98-108': ['DO jj = ntsj-( 1), ntej+(  0 ) ; DO ji = ntsi-( 1), ntei+(  0)     !* 2 x shear production at uw- and vw-points (energy conserving form)', '&         * wvmask(ji,jj,jk)', 11],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:420-426': ['zdiag(ji,jk) = 1._wp - zzd_lw - zzd_up + zfact2 * dissl(ji,jj,jk) * wmask(ji,jj,jk)', '&                                  ) * wmask(ji,jj,jk)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:498-500': ['IF( nn_etau == 1 ) THEN           !* penetration below the mixed layer (rn_efr fraction)', ('en(ji,jj,jk) = en(ji,jj,jk) + rn_efr * en(ji,jj,1) * EXP( -((gdepw_1d(jk) ) *(1._wp+r3t(ji,jj,Kmm))) / htau(ji,jj) )   &', 1), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:864': 'htau(:,:) = MAX(  0.5_wp, MIN( 30._wp, 45._wp* ABS( SIN( rpi/180._wp * gphit(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) ) ) )   )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:648-651': {'anchors': [('CASE ( np_CRV )                           !* Coriolis + relative vorticity', 3), '&                        -( e1u(ji  ,jj+1) * pu(ji  ,jj+1,jk) - e1u(ji,jj) * pu(ji,jj,jk) )  ) * r1_e1e2f(ji,jj)', 4], 'routine': 'vor_ens'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:692-694': {'anchors': ['SUBROUTINE vor_een( kt, Kmm, kvor, pu, pv, pu_rhs, pv_rhs )', '!!                ***  ROUTINE vor_een  ***', 3], 'routine': 'vor_een'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766': {'anchors': [('CASE ( np_CRV )                           !* Coriolis + relative vorticity', 4), ') * r1_e1e2f(ji,jj)   ) * z1_e3f(ji,jj)', 5], 'routine': 'vor_een'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:155-156': ['e1e2t (:,:) = e1t(:,:) * e2t(:,:)   ;   r1_e1e2t(:,:) = 1._wp / e1e2t(:,:)', 'e1e2f (:,:) = e1f(:,:) * e2f(:,:)   ;   r1_e1e2f(:,:) = 1._wp / e1e2f(:,:)', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:509-519': ['!                          ! ocean u- and v-depth at mid-step   (separate DO-loops remove the need of a lbc_lnk)', ('&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1), 11],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:530-535': ['!                             ! resulting flux at mid-step (not over the full domain)', ('zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 1), 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:666-678': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', 1), ('&   ) * ssvmask(ji,jj)', 1), 13],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:721-730': [('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', 1), ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1), 10],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 4],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:521-528': ['!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1', ('END DO   ;   END DO', 9), 8],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:547-549': ['DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)   ! corrected horizontal velocity', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 3],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:565-592': ['IF( .NOT.ln_shuman ) THEN             ! No Shuman averaging- tra_adv_trp can be in tiling loop', ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 2), 28],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:617-662': ['!                       !==  T-S Tracers  ==!', "CALL r175_put3(kstp, kstg, 'adv_s', ts(:,:,:,jp_sal,Krhs))", 46],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:664-672': ['CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )   ! surface boundary condition', "CALL r175_put3(kstp, kstg, 'sbc_s', ts(:,:,:,jp_sal,Krhs))", 9],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:693-720': ['CASE ( 1 , 2 )    !==  Stage 1 & 2  ==!    time stepping', "CALL r175_put3(kstp, kstg, 'qco_s', ts(:,:,:,jp_sal,Kaa))", 28],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90:565-569': ['IF( .NOT.ln_shuman ) THEN             ! No Shuman averaging- tra_adv_trp can be in tiling loop', ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), 5],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90:599-601': ["CALL r175_put3(kstp, kstg, 'trp_u', zFu)", "CALL r175_put3(kstp, kstg, 'trp_w', zFw)", 3],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:248-251': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 2), 'pFw(ji,jj,jpk) = 0._wp', 4],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:267-302': ['IF( ll_Fw ) THEN', "IF( lwp .AND. kt == kit000 .AND. kstg == 3 )   WRITE(992) l4_canon_3d(ww,'T')", 36],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:314-315': ['DO jk =  1,  jpkm1  ; DO jj = ntsj-(  1), ntej+(   1) ; DO ji = ntsi-( 1), ntei+(   1)', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 2],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/divhor.f90:132-153': ['CASE ( np_transport )', ('pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 22],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/sshwzv.f90:277-298': ['IF( .NOT. PRESENT( k_ind ) ) THEN', '&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)', 22],
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90:248-252': {'anchors': [('DO jj = ntsj-( nn_hls)', 2), ('END DO   ;   END DO', 4), 5], 'routine': 'tra_adv_trp_t'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90:267-315': {'anchors': ['IF( ll_Fw ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 49], 'routine': 'tra_adv_trp_t'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/divhor.f90:132-139': {'anchors': ['CASE ( np_transport )', ('END DO   ;   END DO   ;   END DO', 2), 8], 'routine': 'div_hor_RK3_t'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/sshwzv.f90:271-298': {'anchors': [('DO jj = ntsj-( 1)', 2), '&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)', 28], 'routine': 'wzv_RK3_t'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:136-137': {'anchors': ['rDt = r1_3 * rn_Dt', ('r1_Dt = 1._wp / rDt', 1), 2], 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:51': 'INTEGER  :: n_baro_upd =  np_HYB',
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:179': {'anchors': 'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:290': {'anchors': 'zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:599': {'anchors': "CALL r175_put3(kstp, kstg, 'trp_u', zFu)", 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90:150-156': {'anchors': ['IF( .NOT.lk_linssh ) THEN', 'CALL wzv    ( kt, Kbb, Kbb, Kaa', 7], 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90:303': {'anchors': 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs', 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:164': {'anchors': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:152': {'anchors': 'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)', 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:156': {'anchors': 'CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )', 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297-298': {'anchors': ['pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)', ('&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2), 2], 'routine': 'wzv_RK3_t'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:112-126': {'anchors': ['zWf  = e1e2t(ji  ,jj  ) *   ww(ji  ,jj  ,jk+1)', '&                                            * ( zWdzV(ji,jj) + zzWdzV )', 15], 'routine': 'dyn_zad'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:188-189': {'anchors': ['rDt = r1_2 * rn_Dt            ! set time-step : rn_Dt/2', ('r1_Dt = 1._wp / rDt', 2), 2], 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221': {'anchors': ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )   ! at N+1/2 (Kaa)', 2), 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:327': {'anchors': 'CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:237': {'anchors': 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 'routine': 'stp_RK3'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283': {'anchors': 'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:211': {'anchors': 'vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)', 'routine': 'dom_msk'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:291': {'anchors': 'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:826-827': {'anchors': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp   &", 2), "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 2], 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90:211': {'anchors': 'vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)', 'routine': 'dom_msk'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90:232': {'anchors': "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 'routine': 'dom_msk'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121': {'anchors': 'IF( nn_evdm == 1 ) THEN', 'routine': 'zdf_evd'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:135': {'anchors': 'p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 'routine': 'zdf_evd'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90:121': {'anchors': 'IF( nn_evdm == 1 ) THEN', 'routine': 'zdf_evd'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90:135': {'anchors': 'p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 'routine': 'zdf_evd'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:284': {'anchors': 'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:276': {'anchors': 'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282': {'anchors': 'zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3.f90:215': {'anchors': 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 'routine': 'stp_RK3'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90:570': {'anchors': 'un_adv(ji,jj) = un_adv(ji,jj) + za2 * zhU(ji,jj) * r1_e2u(ji,jj)', 'routine': 'dyn_spg_ts'},
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90:831': {'anchors': 'un_adv(:,:) = un_adv(:,:) / r1_wgt2s', 'routine': 'dyn_spg_ts'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:570': {'anchors': 'un_adv(ji,jj) = un_adv(ji,jj) + za2 * zhU(ji,jj) * r1_e2u(ji,jj)', 'routine': 'dyn_spg_ts'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:831': {'anchors': 'un_adv(:,:) = un_adv(:,:) / r1_wgt2s', 'routine': 'dyn_spg_ts'},
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:857-860': ['IF(.NOT.ll_bt_av ) THEN                          !* Update Kaa barotropic external mode', 'pssh (:,:,Kaa) = ssha_e(:,:)', 4],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:517-519': ['zhvp2_e(ji,jj) = hv_0(ji,jj) + r1_2 * r1_e1e2v(ji,jj)', '&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 3],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:509-520': ['!                          ! ocean u- and v-depth at mid-step   (separate DO-loops remove the need of a lbc_lnk)', ('END DO   ;   END DO', 6), 12],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:666-678': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', '&   ) * ssvmask(ji,jj)', 13],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820': [('ELSE', 5), ('ENDIF', 35), 9],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/lbcnfd.f90:684-717': [("CASE ( 'V' )                               ! V-point", 5), ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = psgn(jf) * ptab(jf)%pt4d(ii2,ij2,jk,jl)', 57), 34],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:136': 'rDt = r1_3 * rn_Dt',
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:211': 'vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)',
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:225': 'IF (mbkv(ji,jj)==1) vmask(ji,jj,:) = 0._wp',
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:232': "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )",
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232': ['!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)', "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 27],
    'ocean_model_latlon_cgrid.py:6814-6818': [('if isinstance(_zc, OceanPartialCellCoordinate):', 1), '_ws_v_live_mask = _ws_v_live_mask.astype(state.eta.data.dtype)', 5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:192-198': ['zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )     +  avm(ji,jj,jk  )     ) &', 'zwd(ji,jk) = 1._wp - zzwi - zzws', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:361-367': ['zzwi = - zDt_2 * ( avm(ji,jj+1,jk  )     +  avm(ji,jj,jk  )     ) &', 'zwd(ji,jk) = 1._wp - zwi(ji,jk) - zws(ji,jk)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:231-233': ['zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm', ('zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))', 2), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:198-199': ['Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)*umas', 'Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmas', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223': [('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )   ! at N+1/2 (Kaa)', 2), ('r3v(:,:,Kaa) = r1_2 * ( r3v(:,:,Kbb) + r3va(:,:) )', 2), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:844': "CALL lbc_lnk( 'dynspg_ts', un_adv, 'U', -1._wp, vn_adv, 'V', -1._wp )",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:311': "CALL lbc_lnk( 'traadv_fct', ztFu, 'U', -1.0_wp, ztFv, 'V', -1.0_wp, zta_up1, 'T', 1.0_wp )",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177-179': ['r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:284-290': ['pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,1)))', 'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:169': "CALL lbc_lnk( 'stp_RK3_stg', r3ua, 'U', 1._wp, r3va, 'V', 1._wp, r3fa,",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:232': "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask,",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:258': 'fe3mask(:,:,:) = fmask(:,:,:)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:266-269': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,', 2), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:141': 'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )            ! horizontal gra',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:147': 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:474-479': ['uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) ) * uu(ji,jj,jk,', '&             /           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:514-515': ['zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) )', 'zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_3d(ji,jj,:)*vv(ji,jj,:,Kaa) )', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:536-537': ['uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:158-161': ['IF( ln_drgimp .AND. ln_dynspg_ts ) THEN', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:224': ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)   ! at N+1/3 (Kmm)', 2),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:245': 'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )   ! at N+1/2 (Kmm)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:178-179': ['r3u(:,:,Kaa) = r2_3 * r3u(:,:,Kbb) + r1_3 * r3ua(:,:)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:357': ('zh  = ((gdept_1d(jk) ) *(1._wp+r3t(ji,jj,Knn))) * r1_Z0', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:646': ('zh  = ((gdept_1d(jk) ) *(1._wp+r3t(ji,jj,Knn))) * r1_Z0', 2),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:745': 'zwz(ji,jj) = ff_f(ji,jj) * z1_e3f(ji,jj)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domqco.f90:257': ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 2),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:791-793': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:", 1), (", avm_k(:,:,:)        , 'W',  1._wp, ldfull=.TRUE. ) !  lbc_lnk needed for zdf_sh2", 2), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:", 2), ("&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,", 2), 2],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90:191-205': [('DO jk =    2,  jpkm1,  1', 1), ('zwd(ji,1) = 1._wp - zzws', 1), 15],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90:360-375': [('DO jk =     2,  jpkm1,  1', 2), ('zwd(ji,1) = 1._wp - zzws', 2), 16],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770': [('CASE ( np_CRV )                           !* Coriolis + relative vorticity', 4), ('zwz(ji,jj) = ( zwz(ji,jj) - ff_f(ji,jj) ) * fmask(ji,jj,jk) + ff_f(ji,jj)', 3), 9],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:784-785': [('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d(ji,jj,jk)', 3), ('zwy(ji,jj) = e1v(ji,jj) * (e3v_3d(ji,jj,jk)', 3), 2],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:790-802': [('ztne(ji,jj) = zwz(ji-1,jj  ) + zwz(ji  ,jj  ) + zwz(ji  ,jj-1)', 1), ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2), 13],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:797-802': [('zua = + r1_12 * r1_e1u(ji,jj)', 1), ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2), 6],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:361-364': [('hu_e  (:,:) =    (hu_0(:,:) *(1._wp+r3u(:,:,Kmm)))', 1), ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 1), 4],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:509-520': [('ocean u- and v-depth at mid-step', 1), ('END DO   ;   END DO', 6), 12],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1077': ['rDt_e = rn_Dt / REAL( nn_e , wp )'],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:550-558': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1), ('END DO   ;   END DO', 9), 9],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739': [('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', 1), ('ENDIF', 32), 19],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:781-793': [('!* Swap', 1), ('sshn_e (:,:) = ssha_e(:,:)', 1), 13],
    'barotropic_latlon_cgrid.py:2084-2130': [('if _continuity_evaluation == "nemo_literal":', 1),
 'eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, area)',
 47],
    'barotropic_latlon_cgrid.py:2384-2423': ['association_post = None', ('V_bar_new = association_post[1]', 2), 40],
    'barotropic_latlon_cgrid.py:2452-2457': [('if ab3_za is not None:', 5), 'U_bar_c, Ub_c, V_bar_c, Vb_c, eta_c, etab_c)', 6],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:728-730': [("CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp", 2), ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1), 3],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:584-638': [("CASE ( 'T' , 'W' )                         ! T-, W-point", 5), ('END DO   ;   END DO', 4), 55],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:791-793': [('sshbb_e(:,:) = sshb_e(:,:)', 1), ('sshn_e (:,:) = ssha_e(:,:)', 1), 3],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802': [('!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 4), 'ENDIF            ', 24],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:639-683': [("CASE ( 'U' )                               ! U-point", 5), ('END DO   ;   END DO', 5), 45],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721': [("CASE ( 'V' )                               ! V-point", 5), ('END DO   ;   END DO', 6), 38],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414': [('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1), ('zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_3d(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)', 1), 2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:417-418': [('uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 1), ('vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 1), 2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:138': ('CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )', 1),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:192-198': [('zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )', 1), ('zwd(ji,jk) = 1._wp - zzwi - zzws', 1), 7],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:361-367': [('zzwi = - zDt_2 * ( avm(ji,jj+1,jk  )', 1), ('zwd(ji,jk) = 1._wp - zwi(ji,jk) - zws(ji,jk)', 1), 7],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:231-233': [('zwi(ji,jk) = - p2dt * zwt(ji,jk  )', 1), ('zwd(ji,jk) = (e3t_3d(ji,jj,jk)', 2), 3],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179': [('Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)', 1), ('Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)', 1), 2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217': [('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 2), ('r3v(:,:,Kaa) = r1_2 * ( r3v(:,:,Kbb) + r3va(:,:) )', 2), 3],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:380-394': [('zhpi(ji,jj) = zcoef0 * r1_e1u(ji,jj)', 1), ('pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap', 1), 15],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:766-767': [('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk))) * pu(ji,jj,jk)', 3), ('zwy(ji,jj) = e1v(ji,jj) * (e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk))) * pv(ji,jj,jk)', 3), 2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:779-784': [('zua = + r1_12 * r1_e1u(ji,jj) * (  ( ztne(ji,jj  ) * zwy(ji  ,jj  ) + ztnw(ji+1,jj) * zwy(ji+1,jj  ) )', 1), ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2), 6],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:145-150': [('puu(ji,jj,jk,Kaa) = (        ( 1._wp + r3u(ji,jj,Kbb) ) * puu(ji,jj,jk,Kbb )', 1), ('/          ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 1), 6],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:160-161': [('puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)', 1), ('pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 1), 2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:218': ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:239': ('r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 1),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546': ('r3f(:,:) = r3fa(:,:)', 1),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:572-573': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp   &", 2), ("ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 2), 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1056-1080': [('zcmax = MAXVAL( zcu(Nis0:Nie0,Njs0:Nje0) )', 1), ("IF( zcmax>0.9_wp )   CALL ctl_stop( 'dynspg_ts ERROR: Maximum Courant number is greater than 0.9 => increase nn_e' )", 1), 25],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:28': ('rn_Dt       = 10800.     !  time step for the dynamics and tracer', 1),
    'cfgs/SHARED/namelist_ref:72': ('rn_Dt       = 5400.     !  time step for the dynamics and tracer', 1),
    'cfgs/SHARED/namelist_ref:1099-1101': [('ln_bt_auto    = .true.     ! Number of sub-step defined from:', 1), ('nn_e         =  30         ! =F : the number of sub-step in rn_Dt seconds', 1), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:395-400': [('ELSE                                  !* no wet and dry', 1), ('CALL usr_def_istate_ssh( tmask, ssh(:,:,Kbb) )', 2), 6],
    'src/OCE/IOM/restart.F90:457-461': [('IF( ln_read_cfg ) THEN                 ! read configuration : ssh is set to 0', 1), ('CALL usr_def_istate_ssh( tmask, ssh(:,:,Kbb) )', 2), 5],
    'cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm:1': ('bld::tool::fppkeys   key_si3   key_qco key_vco_1d3d key_RK3', 1),
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:365-366': [('nn_bt_flt     = 3          ! Time filtering with dissipative Forward Backward', 1), ('rn_bt_alpha   = 0.09       ! Dissipative Forward Backward parameter', 1), 2],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:65-68': [('pts(:,:,:,jp_tem) = 20._wp', 1), ('pv( :,:,:       ) =  0._wp', 1), 4],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:94-102': [('zdist(ji,jj) = SQRT( glamt(ji,jj)**2 + gphit(ji,jj)**2 )', 1), ('END_2D', 2), 9],
    'tests/TSUNAMI/MY_SRC/usrdef_sbc.F90:60-68': [('utau(:,:) = 0._wp', 1), ('qsr (:,:) = 0._wp', 1), 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:185': ('IF( ln_dynadv_OFF  ) THEN   ;   ioptio = ioptio + 1   ;   n_dynadv = np_LIN_dyn   ;   ENDIF', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:626': ('IF( ln_traadv_OFF ) THEN   ;   ioptio = ioptio + 1   ;   nadv = np_NO_adv   ;   ENDIF', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/istate.f90:129-133': [('IF( .NOT. ln_tsd_init .AND. .NOT. ln_uvd_init ) THEN', 1), ('CALL usr_def_istate( zgdept, tmask, ts(:,:,:,:,Kbb), uu(:,:,:,Kbb), vv(:,:,:,Kbb) )', 1), 5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:442': ('CASE( jp_usr     )   ;   CALL usrdef_sbc_oce( kt, Kbb )                        ! user defined formulation', 1),
    'tools/DOMAINcfg/src/domhgr.F90:484-502': [("CALL iom_get( inum, jpdom_global, 'glamt', glamt, cd_type = 'T', psgn = 1._wp )", 1), ("CALL iom_get( inum, jpdom_global, 'e2f'  , e2f , cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1), 19],
    'tools/DOMAINcfg/src/domhgr.F90:383-384': [('ff_f(:,:) = 2. * omega * SIN( rad * gphif(:,:) )', 1), ('ff_t(:,:) = 2. * omega * SIN( rad * gphit(:,:) )     !    -        -       - at t-point', 1), 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:233-237': [("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1), ("CALL iom_get( inum, jpdom_global, 'ff_t', pff_t, cd_type = 'T', psgn = 1._wp, kfill = jpfillcopy )", 1), 5],
    'tools/DOMAINcfg/src/domzgr.F90:425-437': [('IF( ppkth == 0._wp ) THEN            !  uniform vertical grid', 1), ('e3t_1d  (jk) =  za1', 1), 13],
    'tools/DOMAINcfg/src/domzgr.F90:961-984': [('SUBROUTINE zgr_zco', 1), ('END SUBROUTINE zgr_zco', 1), 24],
    'tools/DOMAINcfg/src/domzgr.F90:625-629': [('IF( ln_zco )   THEN                          ! zco : read level bathymetry', 1), ('mbathy(:,:) = INT( bathy(:,:) )', 1), 5],
    'tools/DOMAINcfg/src/domzgr.F90:633-652': [('IF( cp_cfg == "orca" .AND. jp_cfg == 2 ) THEN    ! ORCA R2 configuration', 1), ('END DO', 22), 20],
    'tools/DOMAINcfg/src/domzgr.F90:728-733': [('IF( rn_hmin < 0._wp ) THEN    ;   ik = - INT( rn_hmin )                                      ! from a nb of level', 1), ('ELSE WHERE ( risfdep == 0._wp );   bathy(:,:) = MAX(  zhmin , bathy(:,:)  )   ! min=zhmin over the oceans', 1), 6],
    'tools/DOMAINcfg/src/domain.F90:417': ("CALL iom_rstput( 0, 0, inum, 'jperio', REAL( jperio, wp), ktype = jp_i4 )", 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:595-604': [("IF( iatt == -999 .AND. catt == 'UNKNOWN' .AND. iom_varid( inum, 'jperio', ldstop = .FALSE. ) > 0 ) THEN", 1), ('ENDIF', 32), 10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:289-291': [("IF( iom_varid( inum, 'strait_shlat', ldstop = .FALSE. ) > 0 ) THEN", 1), ("CALL iom_get( inum, jpdom_global, 'strait_shlat', zstrshlat, cd_type = 'F', psgn = 1._wp, pval = -1._wp )", 1), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domzgr.f90:179-188': [('ELSEIF( lk_vco_1d3d ) THEN       !--  z-partial cells  --!   use 3D t-level e3', 1), ("CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1), 10],
    'tripole.py:453': 'def create_tripole_grid(',
    'nemo_testcase_recipe.py:1622': 'def _orca2_masks(bottom_level: np.ndarray, strait_shlat: np.ndarray):',
    'nemo_testcase_recipe.py:1603': 'def _orca2_depth_ladder(e3t_1d: np.ndarray, e3w_1d: np.ndarray):',
    'vertical.py:294': 'def nemo_fe3mask_from_tmask(tmask, *, grid=None):',
    'vertical.py:929': 'class NemoEENBarotropicOperands(NamedTuple):',
    'vertical.py:692': 'def nemo_qco_resolved_mesh_operands(',
    'ocean_model_latlon_cgrid.py:5081': '_nemo_filter_alpha = config.barotropic.nemo_barotropic_filter_alpha',
    'nemo_testcase_recipe.py:5649': 'def build_orca2g1_flat_card(deck_root: str | Path) -> NEMOTestcaseCard:',
    'nemo_testcase_recipe.py:5550': 'def read_orca2g1_deck(deck_root: str | Path) -> ORCA2G1Deck:',
    'nemo_testcase_recipe.py:5598': 'def orca2g1_initial_ssh(glamt_deg, gphit_deg, surface_tmask) -> np.ndarray:',
    'nemo_testcase_recipe.py:5863': 'def validate_orca2g1_flat_card(card: NEMOTestcaseCard, deck_root: str | Path) -> None:',
    'vertical.py:1134': 'nemo_e3w_source="mesh_reference", allow_single_level: bool = False,',
    'nemo_testcase_recipe.py:5712': 'allow_single_level=True,               # jpk = 2: one wet level',
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:489-493': [('Extrapolate Sea Level at step jit+0.5', 1), ('zsshp2_e(:,:) = za1 * sshn_e(:,:)', 1), 5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1496-1531': [('SUBROUTINE ts_bck_interp', 1), ('END SUBROUTINE ts_bck_interp', 1), 36],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:546-558': [('Compute Sea Level at step jit+1', 1), ('END DO   ;   END DO', 9), 13],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:655-718': [('Set next velocities:', 1), ('END DO   ;   END DO', 20), 64],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178': [('ssha(:,:) = ssh (:,:,Kaa)', 1), ('r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb)', 1), 41],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170': [('CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 1), ("r3va, 'V', 1._wp", 1), 3],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 )', 1), ('* r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-424': [('SELECT CASE( kstg )', 2), ('END SELECT', 6), 65],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:443-455': [('Round-11 L2 WRITE-only composition boundary', 1), ('L2_RK_STAGE2_PREUPDATE_DUMP', 1), 13],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:372-405': [('hydrostatic pressure gradient (HPG))', 1), ("l4_canon_3d(uu(:,:,:,Krhs),'U')", 2), 34],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:407-421': [('Coriolis / vorticity', 1), ('L2_RK_STAGE2_TERM_DUMP', 1), 15],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:395-406': [('zuap = -zcoef0', 1), ('l2_sum_u (ji,jj,1) = zhpi(ji,jj) + zuap', 1), 12],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:411-432': [('DO jk= 2, jpkm1', 1), ('l2_sum_v (ji,jj,jk) = zhpj(ji,jj) + zvap', 1), 22],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:401-402': [('puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap', 1), ('pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap', 1), 2],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:426-427': [('puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap', 1), ('pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 1), 2],
    'barotropic_latlon_cgrid.py:2040-2059': ['eta_mid = eta_c', 'za_i, eta_c, etab_c, etabb_c)', 20],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:198-230': [('Ue_rhs(ji,jj) = SUM( e3u_3d', 1), ('Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0', 1), 33],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1462-1463': [('pu_RHSi(ji,jj) = pu_RHSi(ji,jj) + (r1_hu_0', 1), ('pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0', 1), 2],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/zdfdrg.f90:224-249': [('REAL(wp), DIMENSION(Nis0-(2):Nie0+(2),Njs0-(2):Nje0+(2)), INTENT(inout) ::   pCdU', 1), ('pCdU(ji,jj) = - pCd0(ji,jj) * SQRT', 1), 26],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:194-230': [('!*  vertical averaging  *!', 1), ('Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0', 1), 37],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:286-324': [('! set values computed in RK3_ssh', 1), ('zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd', 1), 39],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463': [('SUBROUTINE dyn_drg_init', 1), ('pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0', 1), 60],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dommsk.f90:206-249': ['!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)', 'ssvmask(:,:) = MAXVAL( vmask(:,:,:), DIM=3 )', 44],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:945-958': [("IF( c_NFtype == 'T' ) THEN            ! *  North fold  T-point pivot", 4), ('ptab(ji,ipjm1) = psgn * ptab(ijt,ipjm1)', 3), 14],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:460-484': ['!* Set extrapolation coefficients for predictor step:', 'va_e(ji,jj) = za1 * vn_e(ji,jj) + za2 * vb_e(ji,jj) + za3 * vbb_e(ji,jj)', 25],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:487-519': ['IF( .NOT.lk_linssh ) THEN                        !* Update ocean depth (variable volume case only)', ('+ e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1), 33],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:530-535': ['! resulting flux at mid-step (not over the full domain)', 'zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 6],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:574-580': ['WRITE(l2_adv_unit) jn, za2', "& l4_canon_2d(vn_adv,'V')", 7],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:755-779': [('IF( lwp .AND. kt == nit000 .AND. jn <= 2 ) THEN', 3), ("& l4_canon_2d(ffv_nw,'V'), l4_canon_2d(ffv_ne,'V')", 2), 25],
    'barotropic_latlon_cgrid.py:799-819': ['def nemo_literal_metric_transports(', 'return zh_u, zh_v', 21],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:171-216': ['Matrix construction', 'zwt(:,1) = 0._wp', 46],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:218-235': ['Diagonal, lower (i), upper (s)', ('ENDIF', 11), 18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:249-273': ['Matrix inversion from the first level', ('END DO   ;   END DO', 11), 25],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:283-299': ['2nd recurrence', ('END DO   ;   END DO', 14), 17],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:278-281': ['sshe_rhs(:,:) =                 emp(:,:)', 'sshe_rhs(:,:) = r1_rho0 * sshe_rhs(:,:)', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:286-289': ['set values computed in RK3_ssh', 'zv_frc(:,:) =   Ve_rhs(:,:)', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:289,320-324': ['zv_frc(:,:) =   Ve_rhs(:,:)', 'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', 'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 6],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:712-741': ['IF( .NOT.lk_linssh ) THEN !* Update ocean depth', 'IF( ln_bdy )   CALL bdy_dyn2d', 30],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:532,535': ['zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)', 'zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 2],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:461-484': ['IF ((jn<3).AND.ll_init) THEN', 'va_e(ji,jj) = za1 * vn_e(ji,jj) + za2 * vb_e(ji,jj) + za3 * vbb_e(ji,jj)', 24],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:681-702': ['ELSE                           !* Flux form', ('END DO   ;   END DO', 18), 22],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:666-679': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:698-701': ['va_e(ji,jj) = (               hv_e  (ji,jj) *   vn_e (ji,jj)', '(hv_0(ji,jj) *(1._wp+r3v(ji,jj,Kmm))) * zv_frc (ji,jj)', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:781-789': ['!* Swap', 'vn_e   (:,:) = va_e  (:,:)', 9],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:512-519': ['zhup2_e(ji,jj) = hu_0(ji,jj)', ('e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)', 1), 8],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:721-730': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', "hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:162-190': ['NAMELIST/namdyn_adv/ ln_dynadv_OFF, ln_dynadv_vec', "CALL ctl_stop( 'choose ONE and only ONE advection scheme' )", 29],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:177-228': ['NAMELIST/namdyn_ldf/ ln_dynldf_OFF', 'IF(.NOT.ln_dynldf_OFF ) THEN', 52],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:586-633': ['NAMELIST/namtra_adv/ ln_traadv_OFF', "CALL ctl_stop( 'tra_adv_init: Choose ONE advection option in namelist namtra_adv' )", 48],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:214-268': ['NAMELIST/namtra_ldf/ ln_traldf_OFF', 'IF( .NOT.ln_traldf_OFF ) THEN', 55],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:371-401': ['NAMELIST/namdrg/ ln_drg_OFF, ln_lin', "CALL ctl_stop( 'zdf_drg_init: Choose ONE type of drag coef in namdrg' )", 31],
    'nemo_testcase_recipe.py:1754-1805': ['def apply_orca2_hand_alterations(temperature, salinity):', 'is what NEMO executes.', 52],
    'nemo_testcase_recipe.py:2228-2229': ['temperature, salinity = build_orca2_initial_ts(', 'temperature_path, salinity_path, tmask)', 2],
    'ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90:218-254': ['IF( cn_cfg == "orca" .OR. cn_cfg == "ORCA" ) THEN', ('ENDIF', 9), 37],
    'ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:139-181': ['ssha(:,:) = ssh (:,:,Kaa)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'round175/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg:51': 'ln_tsd_dmp  = .false.',
    'nemo_testcase_l4_orca2_round174_stage_growth_gate.py:224': 'card, state, freshwater, surface = r166._setup(deck_root, record_root)',
    'nemo_testcase_l4_orca2_round166_external_substep_gate.py:223-230': ['admission = rung0.frames.admit(record_root, ladder.EXPECTED_PRODUCER, None)', 'return card, initial, freshwater, surface', 8],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:137-179': ['ssha(:,:) = ssh (:,:,Kaa)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:469-479': ['uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) )', '&             /           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 11],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:513-537': ['DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)             ! barotropic velocity correction', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 25],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:555,578': [('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 2), 2],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:637,645': ['CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )', 'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 2],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:678-680': ['ts(ji,jj,jk,jn,Kaa) = (        ( 1._wp + r3t(ji,jj,Kbb) )*ts(ji,jj,jk,jn,Kbb )', '&                /          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:791-796': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp", 1), "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 6],
    'ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/stp2d.f90:145-149': [('!*  hydrostatic pressure gradient (HPG))  *!', 1), ("CALL r92_rhs_put_pair( kt, 'hpg'", 1), 5],
    'ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:402-429': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2), ('r172_sum_v (ji,jj,1) = zhpj(ji,jj) + zvap', 1), 28],
    'ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:433-460': [('DO jk= 2, jpkm1', 1), ('r172_sum_v (ji,jj,jk) = zhpj(ji,jj) + zvap', 1), 28],
    'ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2), ('&           - (e3w_1d(1) *(1._wp+r3t(ji  ,jj  ,Kmm))) * rhd(ji  ,jj  ,1)  )', 2), 8],
    'ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-453': ['DO jk= 2, jpkm1', '&           - (e3w_1d(jk) *(1._wp+r3t(ji,jj  ,Kmm))) * ( rhd(ji,jj,  jk) + rhd(ji,jj  ,jk-1) )  )', 9],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:193-200': ['ht_0(:,:) = 0._wp  ! Reference ocean thickness', 'hv_0(:,:) = hv_0(:,:) + e3v_3d(:,:,jk) * vmask(:,:,jk)', 8],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:212-215': ['r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:) + 1._wp -  ssmask (:,:) )', 'r1_hf_0(:,:) = ssfmask(:,:) / ( hf_0(:,:) + 1._wp -  ssfmask(:,:) )', 4],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dommsk.f90:206-232': ['!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)', "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 27],
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90:386-402': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2), 'pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap', 17],
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90:411-434': ['DO jk= 2, jpkm1', ('         END DO   ;   END DO', 1), 24],
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:145-157': [('!*  hydrostatic pressure gradient (HPG))  *!', 1), ("CALL r92_rhs_put_pair( kt, 'vor'", 1), 13],
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:168-179': [('!*  KEG + ZAD  *!', 1), ('CALL r92_rhs_finish( kt )', 1), 12],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:145-179': [('!*  hydrostatic pressure gradient (HPG))  *!', 1), ('CALL r92_rhs_finish( kt )', 1), 35],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:190-196': [('CALL r169_start( kt, Kbb, Kaa, Krhs )', 1), ("CALL r169_put3( kt, 'vmask', vmask )", 1), 7],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:215-227': [('!*  vertical averaging  *!', 1), ('END SELECT', 2), 13],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:244-274': [('!* baroclinic drag forcing *!', 1), ('CLOSE(l2_slow_unit)', 1), 31],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:278-306': [('!* atmospheric pressure forcing *!', 1), ('Ve_rhs(ji,jj) = Ve_rhs(ji,jj) + ( bhd_wave(ji,jj+1) - bhd_wave(ji,jj) ) * r1_e1u(ji,jj)', 1), 29],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:680-702': [('! Add bottom stresses:', 1), ("CALL r95_spg_w2('trd_v', zv_trd)", 1), 23],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:715-728': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', 1), ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:761-767': [('IF( .NOT.lk_linssh ) THEN !* Update ocean depth', 1), ('END DO   ;   END DO', 20), 7],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:207-218': [('SELECT CASE( n_dynadv )', 2), ('END SELECT', 2), 12],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:231-247': [('!* baroclinic drag forcing *!', 1), ("CALL r93_slow_put_pair( kt, 'wind', Ue_rhs, Ve_rhs )", 1), 17],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:254-285': [('!* atmospheric pressure forcing *!', 1), ("CALL r93_slow_put_pair( kt, 'final', Ue_rhs, Ve_rhs )", 1), 32],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:585-595': [('zhdiv = (   ( zhU(ji,jj) - zhU(ji-1,jj) ) + ( zhV(ji,jj) - zhV(ji,jj-1) )  ) * r1_e1e2t(ji,jj)', 1), ("CALL r95_spg_w2('ssha_e', ssha_e)", 1), 11],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:633-634': [('zsshu_a(ji,jj) = r1_2 * r1_e1e2u(ji,jj)', 1), ('&                                      + e1e2t(ji+1,jj  ) * ssha_e(ji+1,jj  ) ) * ssumask(ji,jj)', 1), 2],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:836-844': [('!* Swap', 1), ('vn_e   (:,:) = va_e  (:,:)', 1), 9],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:763-766': [('hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)', 1), ('hvr_e(ji,jj) = ssvmask(ji,jj) / (  hv_e(ji,jj) + 1._wp - ssvmask(ji,jj)  )', 1), 4],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:770-779': [('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', 1), ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1), 10],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:132-179': ['ALLOCATE( ssha(jpi,jpj) , ua_b(jpi,jpj) , va_b(jpi,jpj) )', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 48],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:185-226': ['CASE ( 2 )', ('END SELECT', 3), 42],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/sshwzv.f90:736-767': ['z2d(:,:) = 0._wp', ('pww(ji,jj,jk) = ( 1._wp - zcff ) * pww(ji,jj,jk)', 2), 32],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:171': ['CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt,', 'CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt,'],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316': ['CALL nonosc( Kaa, pt(:,:,:,jn,Kbb),', 'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb),'],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:816-845': [('! 2nd option', 1), ('END DO   ;   END DO', 52), 30],
    'advection.py:840-874': ['NEMO_FCT_STENCIL_TRACE_FIELDS = (', 'return_limiter_activity: bool = False,', 35],
    'advection.py:1392-1410': ['tr_west = jnp.roll(bnd_up, 1, axis=1)', 'base, q_td, wet, q_max,', 19],
    'advection.py:1576-1577': [('if return_nemo_stencil_trace:', 2), 'return div_h_fct, vert_div_fct, stencil_trace', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:800-845': ['!        !==  set slices and compute zbup/zbdo  ==!', ('END DO   ;   END DO', 52), 46],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-856': ['zup = MAX(  zbup(ji  ,jj  ,ik  ),', '&        zbup(ji  ,jj  ,ikm1), zbup(ji  ,jj  ,ikp1)  )', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:858-878': ['zdo = MIN(  zbdo(ji  ,jj  ,ik  ),', ('ENDIF', 33), 21],
    'advection.py:995-1010': [('raise ValueError(', 2), '"NEMO trace returns require the NEMO RK3 two-step predictor")', 16],
    'advection.py:1487-1502': ['alpha_u_full, alpha_v, alpha_vert_face = _zalesak_signsplit_face_alphas(', 'return_nemo_beta_trace=True,', 16],
    'advection.py:1529-1536': ['limited_u, limited_v = alpha_u_full * ad_flux_u, alpha_v * ad_flux_v', 'vert_div_fct = F_vert_fct[..., :-1] - F_vert_fct[..., 1:]', 8],
    'advection.py:2061-2084': [('if return_nemo_beta_trace:', 4), 'zbetup_literal, zbetdo_literal, R_in, R_out,', 24],
    'advection.py:2139-2140': [('if return_nemo_beta_trace:', 5), 'return alpha_u_full, alpha_v, alpha_vert_face, beta_trace', 2],
    'ocean_model_latlon_cgrid.py:15762-15779': ['if (self._nemo_ws_test_hooks.expose_stage3_fct_inputs', 'return state_after._replace(mass_flux_w=diagnostic)', 18],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:180-195': ["IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN", ('END DO   ;   END DO', 4), 16],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:218-235': ['! Diagonal, lower (i), upper (s)', ('ENDIF', 11), 18],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:268-273': ['!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1', ('END DO   ;   END DO', 11), 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:283-291': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', ('END DO   ;   END DO', 13), 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:293-299': ['!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk', ('END DO   ;   END DO', 14), 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749': ['CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 16],
    'ocean_model_latlon_cgrid.py:1696,1926': ['stage1_momentum_rhs_transform: object = None', ('tracer_zdf_trace: bool = False', 1), 2],
    'ocean_model_latlon_cgrid.py:9713-9717': ['if (self._nemo_ws_test_hooks.stage1_momentum_rhs_transform', '_u1_rhs, _v1_rhs))', 5],
    'ocean_model_latlon_cgrid.py:12303-12310': ['if _return_tracer_zdf_trace:', ('solve=_nemo_ws_vertical_solve_trace,', 2), 8],
    'ocean_model_latlon_cgrid.py:15749-15761': [('if self._nemo_ws_test_hooks.tracer_zdf_trace:', 1), ('return result._replace(state_after=state_after)', 4), 13],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:761-779': ['IF( .NOT.lk_linssh ) THEN !* Update ocean depth (variable volume case only)', "&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 19],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/lbcnfd.f90:684-721': [("CASE ( 'V' )                               ! V-point", 5), ('END DO   ;   END DO', 6), 38],
    'barotropic_latlon_cgrid.py:704-752': ['def _nemo_external_mode_boundary_association(', 'inverse_u_post, inverse_v_post, eta,', 49],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591': [('!                             ! resulting flux at mid-step (not over the full domain)', 1), ('ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 1), 28],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570': [('DO jj = ntsj-( 2), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)   ! not jpj-row', 2), ('END DO   ;   END DO', 8), 3],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:584-591': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1), ('ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 1), 8],
    'barotropic_latlon_cgrid.py:575-609': ['def _nemo_literal_reference_face_depths(z_coord, dtype):', 'return hu_compact, hv_compact', 35],
    'barotropic_latlon_cgrid.py:864-896': ['def nemo_literal_continuity_divergence(', 'return (du + dv) * (1.0 / grid.area)', 33],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545': ['zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)', '&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 27],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:584-615': [("CASE ( 'T' , 'W' )                         ! T-, W-point", 5), ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = psgn(jf) * ptab(jf)%pt4d(ii2,ij2,jk,jl)', 45), 32],
    'barotropic_latlon_cgrid.py:612-696': ['def _nemo_ssh_avg_apply(eta_dyn, u_mask, v_mask, grid, area, prep, *,', ('return H_u, H_v, r1_u_entry, r1_v_entry', 2), 85],
    'barotropic_latlon_cgrid.py:2611-2615': ['# Preserve associated halo operands in the private trace.', '"inverse_depth_v": r1_H_v,', 5],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:680-699': ['! Add bottom stresses:', ("& l4_canon_2d(zu_trd,'U'), l4_canon_2d(zv_trd,'V')", 1), 20],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:639-683': [("CASE ( 'U' )", 5), ('END DO   ;   END DO', 5), 45],
    'barotropic_latlon_cgrid.py:1099-1120': ['def _nemo_een_north_u(field, grid):', ('return north', 2), 22],
    'barotropic_latlon_cgrid.py:1225-1240': ['e3u = b(e3u0 * b(one + r3u[..., None] * umask))',
 'jnp.max(jnp.where(vmask > 0.0, levels, 0), axis=-1), 1)',
 16],
    'barotropic_latlon_cgrid.py:1343-1379': ['un = {', 'e3v, mbkv, neighbor, neighbor_mask, vq[corner],', 37],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348': [('CALL r118_een_v_before(1, ji, jj, jk, zpvo_nw', 1), ('CALL r118_een_v_after(2, ji, jj, jk, ffv_ne(ji,jj))', 1), 10],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1360-1367': [('r105_acc_v_nw(ji,jj) = ffv_nw(ji,jj)', 1), ('ffv_ne(ji,jj) = r1_12 * r1_e2v(ji,jj)', 1), 8],
    'barotropic_latlon_cgrid.py:1190-1196': ['# dommsk.f90:232-258 applies the ordinary F-grid lateral boundary before',
 'fmask = fmask.at[-1].set(_nemo_een_north_f(fmask, grid)[-1])',
 7],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258': [("CALL lbc_lnk( 'dommsk', umask, 'U'", 1), ('fe3mask(:,:,:) = fmask(:,:,:)', 1), 27],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937': [('SELECT CASE( nn_e3f_typ )', 1), ('WHERE( e3f_0vor(:,:,:) == 0._wp )', 1), 26],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:722-747': [("CASE ( 'F' )", 5), ('END DO; END DO', 6), 26],
    'barotropic_latlon_cgrid.py:1070-1091': ['def _nemo_een_north_f(field, grid):', ('return north', 1), 22],
    'barotropic_latlon_cgrid.py:1228-1231': ['q = b(ff[..., None] / e3f)', 'q_north = b(ff_north[..., None] / e3f_north)', 4],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328': [('zpvo_ne = ff_f(ji  ,jj+1)', 1), ('& ff_f(ji-1,jj+1) /', 1), 6],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/domhgr.f90:233-237': [("IF(  iom_varid( inum, 'ff_f'", 1), ("CALL iom_get( inum, jpdom_global, 'ff_t'", 1), 5],
    'ORCA2_OMIP_L4_R118EENVREC/BLD/ppsrc/nemo/dynspg_ts.f90:1320-1325': [('zpvo_ne = ff_f(ji  ,jj+1)', 1), ('& ff_f(ji-1,jj+1) /', 1), 6],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1317-1327': [('zpvo_ne = ff_f(ji  ,jj+1)', 1), ('ffv_se(ji,jj) = ffv_se(ji,jj) +', 1), 11],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1324-1327': [('ffv_nw(ji,jj) = ffv_nw(ji,jj) +', 1), ('ffv_se(ji,jj) = ffv_se(ji,jj) +', 1), 4],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1274-1288': [('CALL r116_een_u_before(1, ji, jj, jk, zpvo_ne', 1), ('CALL r116_een_u_after(3, ji, jj, jk, ffu_se(ji,jj))', 1), 15],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dommsk.f90:211-232': [('vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)', 1), ("CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 1), 22],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1816-1820': [('zland = 0._wp', 3), ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), 5],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872': [('! define ifill: which method should be used to fill each parts (sides+corners) of the halos', 3), ('ENDIF', 76), 9],
    'ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:2130-2136': [('IF(     ifill(jn,jf) == jpfillcst ) THEN', 5), ('END DO   ;   END DO   ;   END DO   ;   END DO', 19), 7],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1238-1245': [('ffu_nw(:,:) = 0._wp', 1), ('DO jk = 1, mbku(ji,jj)', 1), 8],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1263-1273': [('r107_zpvo_nw(ji,jj,jk) = zpvo_nw', 1), ('ffu_se(ji,jj) = ffu_se(ji,jj) +', 1), 11],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258': [("CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )", 1), ('fe3mask(:,:,:) = fmask(:,:,:)', 1), 27],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1999-2004': [('IF(     ifill(jn,jf) == jpfillcst ) THEN', 4), ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = zland', 4), 6],
    'barotropic_latlon_cgrid.py:1063-1067': ['def _nemo_south_zero_fill(field):',
 'return jnp.concatenate([jnp.zeros_like(field[:1]), field[:-1]], axis=0)',
 5],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynvor.f90:912-937': [('SELECT CASE( nn_e3f_typ )', 1), ('WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 1), 26],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1811-1820': [('! take care of optional parameters', 3), ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), 10],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872': [('! define ifill: which method should be used to fill each parts (sides+corners) of the halos', 3), ('ENDIF', 76), 9],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domzgr.f90:179-188': [('ELSEIF( lk_vco_1d3d ) THEN', 1), ("CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1), 10],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:2021-2046': [('ELSE                                                        ! southern/northern side', 4), ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = ptab(jf)%pt4d(ii2,ij2,jk,jl)', 4), 26],
    'barotropic_latlon_cgrid.py:1056-1060': ['def _nemo_een_south_e3f0(e3f0, mesh_e3f0):',
 'return jnp.concatenate([mesh_e3f0[:1], e3f0[:-1]], axis=0)',
 5],
    'barotropic_latlon_cgrid.py:1250-1264': ['# dyn_vor_init applies the default zero fill to e3f_0vor',
 'q_south = b(ff_south[..., None] / e3f_south)',
 15],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:101-108': [('IF( ln_read_cfg ) THEN', 1), ('&              iff   , ff_f  , ff_t', 1), 8],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:233-236': [("IF(  iom_varid( inum, 'ff_f'", 1), ("CALL iom_get( inum, jpdom_global, 'ff_f'", 1), 4],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/lbclnk.f90:1199-1224': [('ELSE                                                        ! southern/northern side', 1), ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = ptab(jf)%pt4d(ii2,ij2,jk,jl)', 1), 26],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1244-1248': [('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 1), ('& ff_f(ji  ,jj-1) /', 1), 5],
    'barotropic_latlon_cgrid.py:1049-1053': ['def _nemo_south_copy_fill(field):', 'return jnp.concatenate([field[:1], field[:-1]], axis=0)', 5],
    'barotropic_latlon_cgrid.py:1246-1249': ['# ORCA2 reads ff_f through iom_get(..., kfill=jpfillcopy)',
 'ff_south = _nemo_south_copy_fill(ff)',
 4],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1248': [('SELECT CASE( nvor_scheme )', 1), ('& ff_f(ji  ,jj-1) /', 1), 8],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dommsk.f90:252-258': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 2), ('fe3mask(:,:,:) = fmask(:,:,:)', 1), 7],
    'ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/domhgr.f90:135-143': [('IF( iff == 0 ) THEN', 1), ("CALL lbc_lnk( 'dom_hgr', ff_t", 1), 9],
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1245': [('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 1), ('& ff_f(ji  ,jj-1) /', 1), 5],
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1256-1263': [('r107_zpvo_nw(ji,jj,jk) = zpvo_nw', 1), ('r107_acc_after(ji,jj,jk) = ffu_nw(ji,jj)', 1), 8],
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1103-1104': [('CALL r105_een_accum_init', 1), ('CALL r107_een_step_init', 1), 2],
    'ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/l4_r105_een_accum.f90:34-45': [("GET_ENVIRONMENT_VARIABLE('ORCA2_R105_EEN_ACCUM_DIR'", 1), ('ORCA2_R105_EEN_ACCUM_INIT', 1), 12],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1229-1230': [('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 1), ('DO jk = 1, mbku(ji,jj)', 1), 2],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1262-1263': [('DO jj = ntsj-( 0), ntej+(  0) ; DO ji = ntsi-( 0), ntei+(  0)', 2), ('DO jk = 1, mbkv(ji,jj)', 1), 2],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dommsk.f90:223-230': ['IF (jpk>2) THEN', 'IF ( MAXVAL(fmask(ji,jj,:))/=0._wp )', 8],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1231-1244': ['zpvo_nw = ff_f(ji-1,jj  )', ('ffu_nw(ji,jj) = ffu_nw(ji,jj) +', 1), 14],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1277-1278': [('ffv_nw(ji,jj) = ffv_nw(ji,jj) +', 1), ('ffv_ne(ji,jj) = ffv_ne(ji,jj) +', 1), 2],
    'ORCA2_OMIP_L4_R105EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1288-1289': ['ffv_nw(ji,jj) = r1_12 * r1_e2v', 'ffv_ne(ji,jj) = r1_12 * r1_e2v', 2],
    'ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg.f90:300-303': ['SELECT CASE( nspg )', 'CASE ( np_NO )', 4],
    'ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:302': 'IF( kt == nit000 .OR. .NOT. lk_linssh )   CALL dyn_cor_2D_init( Kmm )',
    'ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1294-1297': ['CALL r104_een_accum_dump(r104_acc_u_nw', '& r104_scl_v_sw, r104_scl_v_se, r104_scl_v_nw, r104_scl_v_ne)', 4],
    'l4_r104_een_accum.F90:23-27': ['IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop', "IF(ios /= 0) CALL ctl_stop('round104: cannot open EEN operand record')", 5],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:126-127': ['ALLOCATE( ffu_nw(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))', '&      ffv_nw(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-666': ['!     Compute Sea Level at step jit+1', "CALL r95_spg_w2('cor_v', zv_trd)", 87],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:542-545': [('DO jj = ntsj-( 2), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 2), '&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 4],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-591': ['!     Compute Sea Level at step jit+1', 'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 12],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-660': ['! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2', "CALL r95_spg_w2('zv_spg', zv_spg)", 19],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-650': ['! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2', ('END DO   ;   END DO', 13), 9],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:652-660': ['! Surface pressure gradient', "CALL r95_spg_w2('zv_spg', zv_spg)", 9],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:663-670': ['! Add Coriolis trend:', ('ENDIF', 23), 8],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:680-702': ['! Add bottom stresses:', "CALL r95_spg_w2('trd_v', zv_trd)", 23],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:202-233': ['!  RK3 : single first external mode computation', 'CALL r84_dump_frame( kstp, 3, Naa )', 32],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-608': ['!                             ! Sum over sub-time-steps to compute advective velocities', 'vn_adv(ji,jj) = vn_adv(ji,jj) + za2 * zhV(ji,jj) * r1_e1v(ji,jj)', 9],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:502-511': ['!* Extrapolate barotropic velocities at mid-step (jn+1/2)', ('END DO   ;   END DO', 3), 10],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:486-511': ['!* Set extrapolation coefficients for predictor step:', ('END DO   ;   END DO', 3), 26],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', '&   ) * ssvmask(ji,jj)', 13],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:842-844': ['vbb_e  (:,:) = vb_e  (:,:)', 'vn_e   (:,:) = va_e  (:,:)', 3],
    'ORCA2_R96SPG/ocean.output:765': ('Vector form: 2nd order centered scheme           ln_dynadv_vec  =  T', 1),
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:564-570': ['resulting flux at mid-step (not over the full domain)', ('END DO   ;   END DO', 8), 7],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domhgr.f90:152': 'r1_e1v(:,:) = 1._wp / e1v(:,:)   ;   r1_e2v (:,:) = 1._wp / e2v(:,:)',
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:382-383': ['un_adv(:,:)     = 0._wp       ! Sum for now transport issued from ts loop', 'vn_adv(:,:)     = 0._wp', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:890-906': ['un_adv(:,:) = un_adv(:,:) / r1_wgt2s', "WRITE(l2_adv_unit) l4_canon_2d(un_adv,'U'), l4_canon_2d(vn_adv,'V')", 17],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', "&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 10],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:791-796': ["CALL r95_spg_w2('ua_new', ua_e)", ("CALL r95_spg_w2('hvr_e', hvr_e)", 2), 6],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265': ['CASE( np_EEN )', 'ffv_se(ji,jj) = r1_12 * r1_e2v(ji,jj)', 53],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domzgr.f90:625,631': ['mbku(:,:) = MAX( NINT( zk(:,:) ), 1 )', 'mbkv(:,:) = MAX( NINT( zk(:,:) ), 1 )', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dommsk.f90:224-225': ['IF (mbku(ji,jj)==1) umask(ji,jj,:) = 0._wp', 'IF (mbkv(ji,jj)==1) vmask(ji,jj,:) = 0._wp', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1369-1392': ['SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )', ('END SUBROUTINE dyn_cor_2D', 2), 24],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynvor.f90:912-937': ['SELECT CASE( nn_e3f_typ )', 'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 26],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:301-316': ['! Lane-2 GYRE ENE operand instrument: write the eight frozen coefficient', ('CLOSE(l2_unit)', 1), 16],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:390-442': ['! Lane-2 GYRE causal instrument: open one write-only stream for the', 'WRITE(l2_ord_unit) l2_magic, 2, kt, 2, jpi, jpj, STORAGE_SIZE(1._wp), rDt_e', 53],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:287-320': ['ssh_frc(:,:) = sshe_rhs(:,:)', 'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', 34],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:339-381': ['! Initialize barotropic variables:', 'vn_adv(:,:)     = 0._wp', 43],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:446': ('DO jn = 1, icycle', 1),
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:460-519': ['!* Set extrapolation coefficients for predictor step:', ('+ e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1), 60],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:530-557': ['! resulting flux at mid-step (not over the full domain)', 'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 28],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:601-615': ['! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2', 'zv_spg(ji,jj) = - zldg * ( zsshp2_e(ji,jj+1) - zsshp2_e(ji,jj) ) * r1_e2v(ji,jj)', 15],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:618-652': ['! Add Coriolis trend:', ("& l4_canon_2d(zu_trd,'U'), l4_canon_2d(zv_trd,'V')", 1), 35],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:655-708': ['! Set next velocities:', 'va_e(ji,jj) =  va_e(ji,jj) / ( 1._wp - rDt_e * zCdU_v(ji,jj) * hvr_e(ji,jj) )', 54],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219': ['!*  vertical averaging  *!', "CALL r93_slow_put_pair( kt, 'depth', Ue_rhs, Ve_rhs )", 14],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:231-250': ['!* baroclinic drag forcing *!   (also provide the barotropic drag coeff.)', 'CLOSE(l2_slow_unit)', 20],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:285': ("CALL r93_slow_put_pair( kt, 'final', Ue_rhs, Ve_rhs )", 1),
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:290-312': [('!                 !=======================================!', 1), "IF( kt == nit000 ) CALL r93_slow_put2( 'ssh_rhs', sshe_rhs )", 23],
    'ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:317-326': ['!    using a split-explicit time integration in forward mode', 'CALL r93_slow_finish( kt )', 10],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', 'CALL r92_rhs_finish( kt )', 35],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215': ['!*  vertical averaging  *!', ('END SELECT', 2), 13],
    'ocean_pe_latlon_cgrid.py:6484-6504': [('du_dt = du_dt * u_mask_3d', 1), 'dv_dt = jnp.where(_restore_hpg, hpg_tendency_v, dv_dt * v_mask_3d)', 21],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2), 'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 42],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:227-230': ['!* baroclinic drag forcing *!   (also provide the barotropic drag coeff.)', "l4_canon_2d(Ve_rhs,'V'), l4_canon_2d(CdU_u,'U'), l4_canon_2d(CdU_v,'V')", 4],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:232-245': ['!* wind forcing *!', ('ENDIF', 7), 14],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:247-277': ['!* atmospheric pressure forcing *!', ('ENDIF', 10), 31],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:283-300': ['!==   2D sea surface height forcing   ==!', ('ENDIF', 11), 18],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:305-315': ['Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', 'DEALLOCATE( sshe_rhs , Ue_rhs , Ve_rhs , CdU_u , CdU_v )', 11],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:148-217': ['! Update external forcing (tides, open boundaries, ice shelf interaction and surface boundary condition (including sea-ice)', 'CALL r84_dump_frame( kstp, 1, Naa )', 70],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:139-166': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 28],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-228': ['!==  Background eddy viscosity and diffusivity  ==!', 'avm_k(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk) =                avmb(jk) * wmask(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk)', 24],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfphy.f90:262-284': ['!==  type of vertical turbulent closure  ==!', 'IF( ln_zdfswm )   CALL zdf_swm_init       ! surface  wave-driven mixing', 23],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/zdfdrg.f90:540-547': ['CASE( np_lin )             !==  linear friction  ==!   (pCdU = Cd0 * Uc0)', 'CALL zdf_drg_lin( pCd0(:,:), pCdU(:,:) )  !  using a constant velocity', 8],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/istate.f90:143-162': [('IF( .NOT. ln_rstart ) THEN', 1), 'vv_b(:,:,Kmm)   = vv_b(:,:,Kbb)', 20],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:92-108': ['! Lane-1 certified oracle: exact step-entry Nbb state.  This is a', 'CALL r84_dump_frame( kstp, 0, Nbb )', 17],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:214-217': ['! Stage 1 :', 'CALL r84_dump_frame( kstp, 1, Naa )', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:221-233': ['! Stage 2 :', 'CALL r84_dump_frame( kstp, 3, Naa )', 13],
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/traadv.f90:287': "&       l4_canon_2d(rnf,'T')",
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/traadv.f90:357': 'IF( llwet ) zfield(ji,jj) = pfield(ji,jj)',
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/sbc_oce.f90:201': 'IF(ln_rnf)   ALLOCATE( rnf(jpi,jpj), rnf_b(jpi,jpj), STAT=ierr(3) )',
    'ORCA2_OMIP_L4_R88FRAMEDEBUG/BLD/ppsrc/nemo/stprk3_stg.f90:555': ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
    'ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/stprk3.f90:90-108': ["IF( ln_timing )   CALL timing_start('stp_RK3')", 'CALL r84_dump_frame( kstp, 0, Nbb )', 19],
    'ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/traadv.f90:273-287': [('IF( lwp .AND. kt == kit000 .AND. kstg == 1 ) THEN', 1), "&       l4_canon_2d(rnf,'T')", 15],
    'ORCA2_OMIP_L4_R87FRAMES/BLD/ppsrc/nemo/sbc_oce.f90:195-216': ['ALLOCATE( utau(jpi,jpj) , utau_b(jpi,jpj) , utauU(jpi,jpj) , &', 'wndm(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , taum (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , STAT=ierr(6) )', 22],
    'ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:149-154': ['IF( ln_tide    )   CALL tide_update( kstp )', 'IF( kstp == nit000 )   CALL l4_dump_ocean_surface_input( kstp, Nbb )', 6],
    'ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:424-438': ["WRITE(iunit) l4_canon_2d(qsr,'T'), l4_canon_2d(qns,'T'), l4_canon_2d(qns_b,'T'), &", "WRITE(iunit) l4_canon_3d(rnf_tsc,'T'), l4_canon_3d(rnf_tsc_b,'T')", 15],
    'ORCA2_OMIP_L4_R85FRAMEDEBUG/BLD/ppsrc/nemo/stprk3.f90:443-473': ['FUNCTION l4_canon_2d( pfield, cdgrid ) RESULT( zfield )', 'END FUNCTION l4_canon_2d', 31],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_oce.f90:195-216': ['ALLOCATE( utau(jpi,jpj) , utau_b(jpi,jpj) , utauU(jpi,jpj) , &', 'wndm(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , taum (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , STAT=ierr(6) )', 22],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:146-152': ['INTEGER FUNCTION sbc_rnf_alloc()', 'rnf_tsc_b(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpts) , rnf_tsc (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpts) , STAT=sbc_rnf_alloc )', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:259-282': ['IF( ln_rnf_depth .OR. ln_rnf_depth_ini ) THEN', 'phdivn(ji,jj,1) = phdivn(ji,jj,1) - rnf(ji,jj) * r1_rho0', 24],
    'cfgs/SHARED/namelist_ref:474,478,480': ['ln_rnf_depth = .false.', 'ln_rnf_depth_ini = .false.', 'rn_dep_max  = 150.', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-470': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:286-310': ['! Round-13 WRITE-only stage-1 transport operand record.', "l4_canon_2d((r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm))),'V'), l4_canon_2d(vv_b(:,:,Kmm),'V')", 25],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:513-537': ['DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)             ! barotropic velocity correction', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 25],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbini.f90:133-140': ['CALL icb_nam               ! Read and print namelist parameters', "IF( icb_alloc() /= 0 )   CALL ctl_stop( 'STOP', 'icb_alloc : unable to allocate arrays' )", 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icb_oce.f90:161-174': ['INTEGER FUNCTION icb_alloc()', '&      berg_grid%tmp        (jpi,jpj) , STAT=ill)', 14],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:274-280': ['IF( nn_ice == 0 ) THEN        !* No sea-ice in the domain : ice fraction is always zero', 'cloud_fra(:,:) = pp_cldf      !* cloud fraction over sea ice (used in si3)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:376-383': [('IF( nn_ice == 0 ) THEN', 2), ('ENDIF', 21), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbc_ice.f90:91-121': ['INTEGER FUNCTION sbc_ice_alloc()', '&      emp_ice (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))     , sstfrz   (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))     , STAT= ierr(ii) )', 31],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-346': ['IF( ln_rst_list .OR. nn_stock /= -1 ) THEN', 'IF( .NOT. ln_rst_list .AND. MOD( nn_stock, nn_fsbc) /= 0 ) THEN', 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119': ['IF( kt == nit000 ) THEN   ! default definitions', 'IF( kt == nitrst - 1 .OR. nn_stock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN', 26],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:120-146': ['IF( nitrst <= nitend .AND. nitrst > 0 ) THEN', 'lrst_oce = .TRUE.', 27],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202': [('IF( kt == nitrst ) THEN', 1), ('ENDIF', 14), 15],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:271-271': 'IF( lrst_oce   )   CALL rst_write    ( kstp, Nbb, Nnn, Naa )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:176-184': ['zmax(1) = MAXVAL( ABS( ssh(:,:,Kmm)           ), mask = llmsk(:,:,1) )', 'zmax(5) = MAXVAL(  ts(:,:,:,jp_sal,Kmm), mask = llmsk )', 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:243-250': ['IF(  zmax(1) >   20._wp .OR.', '& ABS(   SUM(zmax(1:jptst)) ) > HUGE(1._wp) ) THEN', 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:293-316': ["WRITE(ctmp1,*) ' stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  S >= 100  or  NaN encounter in the tests'", "CALL ctl_stop( 'STOP', ctmp1, ' ', ctmp2, ctmp3, ctmp4, ctmp5, ctmp6, ' ', ctmp7 )", 24],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-298': ["WRITE(ctmp1,*) ' stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  S >= 100  or  NaN encounter in the tests'", "CALL wrt_line( ctmp6, kt, 'Sal   max'", 6],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:243-250': ['IF(  zmax(1) >   20._wp .OR.', '& ABS(   SUM(zmax(1:jptst)) ) > HUGE(1._wp) ) THEN', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-310': ["WRITE(ctmp1,*) ' stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  S >= 100  or  NaN encounter in the tests'", 'IF(lwp) THEN   ;   CALL ctl_stop(', 18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:666-679': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:287-289': ['ssh_frc(:,:) = sshe_rhs(:,:)', 'zv_frc(:,:) =   Ve_rhs(:,:)', 3],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:674-678': ['va_e(ji,jj) = (                                 vn_e(ji,jj)', '&   ) * ssvmask(ji,jj)', 5],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-753': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', ("& l4_canon_2d(zhU,'U'), l4_canon_2d(zhV,'V')", 1), 33],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-739': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', ('ENDIF', 32), 19],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:258-284': ['SUBROUTINE zdf_drg_lin( pCd0, pCdU )', 'END SUBROUTINE zdf_drg_lin', 27],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:540-547': ['CASE( np_lin )             !==  linear friction  ==!', ('CALL zdf_drg_lin( pCd0(:,:), pCdU(:,:) )', 2), 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:158-169': ['IF( ln_drgimp .AND. ln_dynspg_ts ) THEN', ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1), 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:303-312': ['IF( ln_drgimp ) THEN      ! implicit bottom friction', 'zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) / (e3u_3d(ji,jj,iku) *(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))', 10],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:470-474': ['IF ( ln_drgimp ) THEN      ! implicit bottom friction', ('&                                   / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1), 5],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463': ['SUBROUTINE dyn_drg_init( Kbb, Kmm, puu, pvv, puu_b ,pvv_b, pu_RHSi, pv_RHSi, pCdU_u, pCdU_v )', 'pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kmm))) * r1_2*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) ) * zv_i(ji,jj)', 60],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-184': ['NAMELIST/namdyn_ldf/ ln_dynldf_OFF', 'READ(numnam_cfg(MAX(INDEX(numnam_cfg,"&namdyn_ldf "),1):),namdyn_ldf', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-228': ['nldf_dyn = np_ERROR', 'IF(.NOT.ln_dynldf_OFF ) THEN', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:250': ('IF( ln_dynldf_lev )   nldf_dyn = np_lap', 1),
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:297-305': [('IF( ln_dynldf_OFF ) THEN', 2), "IF( ierr /= 0 )   CALL ctl_stop( 'STOP', 'ldf_dyn_init: failed to allocate arrays')", 9],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:488-493': [('CASE ( 3 )        !==  Stage 3  ==!', 1), 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 6],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf.f90:81-85': ['SELECT CASE ( nldf_dyn )', 'CALL dynldf_lev_lap( kt, Kbb, Kmm, puu, pvv, Krhs )', 5],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:123-140': [('zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:592-602': ['NAMELIST/namtra_adv/ ln_traadv_OFF', 'READ(numnam_cfg(MAX(INDEX(numnam_cfg,"&namtra_adv "),1):),namtra_adv', 11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:630-647': ['!                                !==  Parameter control & set nadv ==!', "CALL ctl_stop( 'tra_adv_init: FCT scheme, choose 2nd or 4th order, and optimized or accurate treatment of implicit' )", 18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:668-674': [('SELECT CASE ( nadv )', 2), "' Implicit treatment: ', nn_fct_imp", 7],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540': ['! FCT at last stage only with RK3', 'CALL tra_adv_cen ( kt, nit000,', 44],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_cen.f90:149-160': [('CASE(  2  )                         !* 2nd order centered', 1), ('&                                        / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:191-199': ['! -- Anti-diffusive fluxes : high order minus low order', ('END DO   ;   END DO   ;   END DO', 4), 9],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:495-539': ['! *** 1st step', ('END DO   ;   END DO   ;   END DO', 23), 45],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:562-573': ['! *** 2nd step', ('END DO   ;   END DO   ;   END DO', 26), 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:306-316': ['!        !==  monotonicity algorithm  ==!', 'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb), ztFu, ztFv, ztFw, zta_up1, p2dt )', 11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:768-821': ['REAL(wp) ::   zpos, zneg, zbig, zsmall, zup, zdo, zbt, zland, z1_Dt', 'zbdo(ji,jj,ikp1) = MERGE( MIN( pbef(ji,jj,2), paft(ji,jj,2) ),  zbig, tmask(ji,jj,2) == 1._wp )', 54],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:798-861': [('DO jk = 1, jpkm1', 4), '&        zbdo(ji  ,jj  ,ikm1), zbdo(ji  ,jj  ,ikp1)  )', 64],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:862-878': ['!                                            ! positive part of the flux', ('ENDIF', 33), 17],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:888-915': [('DO jj = ntsj-( 1), ntej+(  0 )', 3), ('END DO   ;   END DO', 54), 28],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:322-329': [('DO jk =  1,  jpkm1', 5), 'zta_up1(ji,jj,jk) = zta_up1(ji,jj,jk) + p2Dt * ztra', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:581-638': [("IF( c_NFtype == 'T' ) THEN", 3), ('END DO   ;   END DO', 4), 58],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:633-643': [('IF( ln_tile )   CALL dom_tile_start', 1), ("& l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')", 4), 11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:598-609': [('! -- after tracer with upstream scheme', 2), ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)', 3), 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:318-329': ['!        !==  final trend with corrected fluxes  ==!', 'zta_up1(ji,jj,jk) = zta_up1(ji,jj,jk) + p2Dt * ztra /', 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:600-649': ['!                       !==  T-S Tracers  ==!', ('IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000 .AND. kstg <= 2 ) &', 2), 50],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:700-749': ['CASE ( 3 )        !==  Stage 3  ==!   add all left over RHS terms', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 50],
    'nemo_testcase_l4_orca2_round226_fct_rhs_walk.py:2-367': ['"""Compare OMT-4', 'raise SystemExit(main())', 366],
    'test_nemo_testcase_l4_orca2_round226_fct_rhs_walk.py:1-50': ['from __future__ import annotations', 'gate.classify(_report(), plant=plant)', 50],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:94-146': ['IF( kt == nit000 ) THEN   ! default definitions', 'lrst_oce = .TRUE.', 53],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:153-201': ['SUBROUTINE rst_write( kt, Kbb, Kmm, Kaa )', 'nitrst   = nn_stocklist( nrst_lst )', 49],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/daymod.f90:405-417': ["ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'ntime'  , REAL( nn_time0, wp) ) ! time", 13],
    'ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stprk3.f90:204-215': ['CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 12],
    'ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stp2d.f90:303-303': 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )',
    'ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/dynspg_ts.f90:714-726': ["r213_magic = 'NEMO_L4_R213VV1'", 'WRITE(r213_unit) va_e', 13],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:738-744': ['IF( .NOT.lk_linssh ) THEN !* Update ocean depth (variable volume case only)', ('END DO   ;   END DO', 20), 7],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', "&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 10],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:533-560': ['! resulting flux at mid-step (not over the full domain)', 'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 28],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-199': ['!*  vertical averaging  *!', 'Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmask(ji,jj,1:jpkm1) ) * r1_hv_0(ji,jj)', 6],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:219-230': ['CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs, Ve_rhs, CdU_u, CdU_v )', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj) * (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kbb)))', 12],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294': ['!                          ! set values computed in RK3_ssh', 'zCdU_v  (:,:) = CdU_v   (:,:)', 6],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:323-328': ['CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', ('END DO   ;   END DO', 1), 6],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:604-612': ['! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2', ('END DO   ;   END DO', 13), 9],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:1522-1557': ['SUBROUTINE ts_bck_interp( jn, ll_init,', 'END SUBROUTINE ts_bck_interp', 36],
    'barotropic_latlon_cgrid.py:115-127': ['# --- NEMO dynspg_ts nn_bt_flt=3 (Demange 2019 dissipative FB) coefficients ---', 'def nemo_ab3am4_coeff_arrays(n_loop: int, alpha: float,', 13],
    'barotropic_latlon_cgrid.py:3495-3496': [('_ab3_za, _ab3_zb = nemo_ab3am4_coeff_arrays(', 2), 'n_loop, alpha=_nemo_filter_alpha, ramp=_ab3_hist is None)', 2],
    'orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg:383-387': ['ln_dynspg_ts  = .true.', 'rn_bt_alpha   = 0.09', 5],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stprk3_stg.f90:383-402': ['CALL    eos    (        ts, Kmm, rhd, rhop )     ! Kmm in situ density anomaly for hpg computation', 'CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )  ! Hydrostratic Pressure Gradient (HPG)', 20],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718': [("CASE ( 'V' )                               ! V-point", 5), ('END DO', 105), 35],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:945-972': [("IF( c_NFtype == 'T' ) THEN            ! *  North fold  T-point pivot", 4), ('END DO', 149), 28],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:784-796': ["CALL lbc_lnk( 'stp_RK3_stg', ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp   &", "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 13],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:342-352': ['! Phase-3 read-only transport instrument.  This is exactly the zF', ('CLOSE(l1_unit)', 2), 11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:551-555': [('IF( .NOT.ln_shuman ) THEN', 1), ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), 5],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:973-980': [("CASE ( 'V' )                               ! V-point", 7), ('ptab(1,ipj+jh) = psgn * ptab(3,ipj-3-jh)', 2), 8],
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:558-558': 'e3t_3d(:,:,:) *(1._wp+r3t(:,:,Kmm)*tmask(:,:,:))',
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/inc/domzgr_substitute.h90:126-126': ('# define  e3t(i,j,k,t)      (E3t_0(i,j,k) Tmsk(r3t,tmask,i,j,k,t))', 1),
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90:170-170': 'REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:,:) ::     e3t_3d',
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90:180-180': 'REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:,:)   ::   r3t, r3u, r3v',
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/traadv_fct.f90:538-538': ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)', 2),
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285': [('DO jk =  1,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 1), 4],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90:150-160': [('DO jk = 1, jpkm1', 1), ('/ (e3t_3d(ji,jj,jk)', 1), 11],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279': [('SELECT CASE( n_baro_upd )', 4), ('END DO   ;   END DO', 2), 11],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:552-559': [('IF( .NOT.ln_shuman ) THEN', 1), '& r3t(:,:,Kmm), tmask(:,:,:) )', 8],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188': [('! t-level: 3D reference', 1), ("CALL iom_get( inum, jpdom_global, 'e3f_0'", 1), 5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90:114-121': [('#     define  E3t_0', 3), ('#     define  e3v_0', 3), 8],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:261-270': [('!==  ratio at u-,v-point  ==!', 2), ('END DO   ;   END DO', 5), 10],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domain.f90:193-214': [('ht_0(:,:) = 0._wp', 1), ('r1_hv_0(:,:) = ssvmask(:,:)', 1), 22],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:208-217': [('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0', 1), ('* r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 10],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:714-717': [('hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)', 1), ('hvr_e(ji,jj) = ssvmask(ji,jj)', 1), 4],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-213': [('Ocean/land mask at u-, v-, and f-points', 1), ('* tmask(ji+1,jj+1,jk)', 1), 8],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:228-232': [('IF ( MAXVAL(umask(ji,jj,:))/=0._wp )', 1), ("CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp", 1), 5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:973-980': [("CASE ( 'V' )", 7), ('ptab(1,ipj+jh) = psgn * ptab(3,ipj-3-jh)', 2), 8],
    'ocean_model_latlon_cgrid.py:2068-2078': [('nemo_qco_card_mesh_operands(', 1), 'include_reciprocals=include_reciprocals', 11],
    'vertical.py:666-673': [('geom_grid = ensure_geometry(grid)', 2), 'hv_0 = jnp.sum(e3v_0 * vmask3, axis=-1)', 8],
    'vertical.py:692-774': ['def nemo_qco_resolved_mesh_operands(', 'e2u=e2u, e1v=e1v)', 83],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/in_out_manager.f90:51-51': 'INTEGER, DIMENSION(10) :: nn_stocklist',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:323-325': ['READ(numnam_ref(MAX(INDEX(numnam_ref,"&namrun "),1):),namrun,IOSTAT=ios)', 'IF(lwm) WRITE ( numond, namrun )', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-348': ['IF( ln_rst_list .OR. nn_stock /= -1 ) THEN', ("CALL ctl_stop( ctmp1, 'Impossible to properly do model restart' )", 2), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:299-306': ['IF( ln_usr          ) THEN   ;   nsbc = jp_usr', 'sbc_init : choose ONE and only ONE sbc option', 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:440-448': [('SELECT CASE( nsbc )', 2), 'IF( ll_opa    )       CALL sbc_cpl_rcv', 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:499-503': [('SELECT CASE( nn_ice )', 2), ('END SELECT', 5), 5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcflx.f90:191-204': ['CALL fld_read( kt, nn_fsbc, sf )', 'emp (ji,jj) =   sf(jp_emp )%fnow(ji,jj,1)', 14],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:206-228': ['IF( nn_avb == 0 ) THEN', 'avm_k(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk)', 23],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:264-270': ['IF( ln_zdfcst ) THEN', 'one and only one vertical diffusion option has to be defined', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:334-343': ['SELECT CASE ( nzdf_phy )', 'END SELECT', 10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:738-740': ['IF( ln_trabbc  )   CALL tra_bbc', 'IF( ln_tradmp  )   CALL tra_dmp', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:954-955': ['pFu(ji,jj,jk) = pFu(ji,jj,jk) - ( zpsi_uw(ji,jj,1) - zpsi_uw(ji,jj,2) )', 'pFv(ji,jj,jk) = pFv(ji,jj,jk) - ( zpsi_vw(ji,jj,1) - zpsi_vw(ji,jj,2) )', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:354-357': ['zUfac = r1_2 *rn_Ud', 'aht0    = zUfac *    rn_Ld**inn', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296': 'CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:255-260': ['IF( ln_ldfeiv .AND. .NOT. ln_traldf_triad ) THEN', 'IF( ln_mle    )   THEN', 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:313-316': ['!* update momentum & tracer diffusivity with wave-driven mixing', 'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + zav_wave(ji,jj)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfiwm.f90:438-443': ['! This internal-wave-driven mixing parameterization elevates avt and avm in the interior, and', 'avtb_2d(:,:) = 1._wp        ! uniform', 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381': ['!                          !==  ocean Kz  ==!   (avt, avs, avm)', 'CALL r79_write( kt, Kbb, Kmm )', 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:178-215': ['! vertical mixing coef.: avt for temperature, avs for salinity and passive tracers', ('ENDIF', 10), 38],
    'ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:191-205': [('DO jk =    2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)            ! inner values', 1), ('zwd(ji,1) = 1._wp - zzws', 1), 15],
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:346': 'ln_dynadv_vec = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:352': 'ln_dynvor_een = .true.',
    'OVERFLOW_OMIP_L1/EXP00/namelist_cfg:83': 'ln_dynadv_up3 = .true.',
    'OVERFLOW_OMIP_L1/EXP00/namelist_cfg:89': 'ln_dynvor_ens = .true.',
    'overflow_kt1_10/namelist_cfg:86': 'ln_dynadv_up3 = .true.',
    'overflow_kt1_10/namelist_cfg:92': 'ln_dynvor_ens = .true.',
    'nemo_testcase_recipe.py:4290': 'if (cfg.momentum_advection != "vector_invariant"',
    'ocean_pe_latlon_cgrid.py:5828': ('if _mom_adv == "flux_form":', 2),
    'ocean_pe_latlon_cgrid.py:5846': ('_bc_horizontal_momentum_advection_flux_form(', 2),
    'nemo_testcase_recipe.py:4293': 'requires ln_dynadv_vec=.true. with nn_dynkeg=0',
    'OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1), 'CALL r56_up3_curv(ji,jj,jk,zlu_uu(ji,jj),zlv_vv(ji,jj),zlu_uv(ji,jj),zlv_vu(ji,jj))', 10],
    'OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:182-192': ['DO jj = ntsj-( 1), ntej+(  0 ) ; DO ji = ntsi-( 1), ntei+(  0)', 'CALL r56_up3_select_t(ji,jj,jk,zui,zvj,zl_u,zl_v)', 11],
    'OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:194-195': ['zFu_t(ji+1,jj  ) = (  zFu(ji,jj) + zFu(ji+1,jj  )  ) * ( zui - gamma1 * zl_u )', 'zFv_t(ji  ,jj+1) = (  zFv(ji,jj) + zFv(ji  ,jj+1)  ) * ( zvj - gamma1 * zl_v )', 2],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv.f90:134-144': [('SELECT CASE( n_dynadv )', 1), 'CALL dyn_adv_up3     ( kt       , Kbb, Kmm, puu, pvv, Krhs, pau, pav, paw )', 11],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:150-158': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1), '&             - ( pvv(ji  ,jj  ,jk,Kbb) - pvv(ji-1,jj  ,jk,Kbb) )    * fmask(ji-1,jj  ,jk)', 9],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:174-219': ['DO jj = ntsj-( 1), ntej+(  0 ) ; DO ji = ntsi-( 1), ntei+(  0)', '&                                    / (e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 46],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynadv_up3.f90:278-359': ['DO jk = 1, jpk-2                    != divergence of advective fluxes =!', 'pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - zFv_t(ji,jj) * r1_e1e2v(ji,jj) / (e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 82],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:343-358': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), 16],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3.f90:200-207': ['! Stage 1 :', 'CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )', 8],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:242-248': [('CASE( np_ENS )                        !* enstrophy conserving scheme', 1), ('CALL vor_ens( kt, Kmm, ntot, usd, vsd,', 1), 7],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:866-869': ['CASE( np_FLX_c2 , np_FLX_up3 )', 'ntot = np_CME', 4],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynvor.f90:666': ('pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav *', 1),
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:327-363': ['CALL r50_mom_begin( kstp, kstg, Kbb, Kmm, Krhs, Kaa, ts, ssh, uu, vv )', "CALL r50_mom_uv( 'after_adv', uu, vv, Krhs )", 37],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:386-403': [('SELECT CASE( kstg )', 3), '&             /           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 18],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:412-433': [('CASE ( 3 )        !==  Stage 3  ==!', 1), "CALL r50_mom_uv( 'raw_kaa', uu, vv, Kaa )", 22],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:436-448': ['!==  All stages: correct the barotropic component ==!', 'CALL r50_mom_finish( uu, vv, Kaa )', 13],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:139-151': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN   ! applied on velocity', '&              /          ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 13],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:159-162': ['IF( ln_drgimp .AND. ln_dynspg_ts ) THEN', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 4],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:164-170': ['DO ji = ntsi-( 0), ntei+( 0 )      ! Add bottom/top stress due to barotropic component only', ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1), 7],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:188-350': [('!**  tridiagonal matrix construction  **!    diagonal (zwd), lower (zwi), upper (zws)', 1), 'puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - zws(ji,jk) * puu(ji,jj,jk+1,Kaa) ) / zwd(ji,jk)', 163],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynzdf.f90:357-518': [('!**  tridiagonal matrix construction  **!    diagonal (zwd), lower (zwi), upper (zws)', 2), 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - zws(ji,jk) * pvv(ji,jj,jk+1,Kaa) ) / zwd(ji,jk)', 162],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:140-159': ['!              !==  RHS : time-stepping of all trends but the implicit one  ==!', ('& pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 1), 20],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:165-171': ['IF( ln_drgimp .AND. ln_dynspg_ts ) THEN', ('& pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 2), 7],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:172-192': ['DO ji = ntsi-( 0), ntei+( 0 )      ! Add bottom/top stress due to barotropic component only', ('& pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 3), 21],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:344-362': [('zwd(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwd(ji,jk-1)', 1), 'CALL r62_zdf_u_solve(jj,puu(ntsi:ntei,jj,1:jpkm1,Kaa))', 19],
    'OVERFLOW_OMIP_L1_P3_R62ZDF/BLD/ppsrc/nemo/dynzdf.f90:513-531': [('zwd(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwd(ji,jk-1)', 2), 'CALL r62_zdf_v_solve(jj,pvv(ntsi:ntei,jj,1:jpkm1,Kaa))', 19],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:492-541': ['!                       !==  T-S Tracers  ==!', ('CALL r50_tra_finish( ts, ssh, Kaa )', 1), 50],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/eosbn2.f90:684-718': [('SELECT CASE( neos )', 2), ('prd(ji,jj,jk) = (  zn * r1_rho0 - 1._wp  ) * ztm', 3), 35],
    'OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/dynhpg.f90:341-419': ['SUBROUTINE hpg_sco( kt, Kmm, puu, pvv, Krhs )', 'END SUBROUTINE hpg_sco', 79],
    'OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:327-354': [('SELECT CASE( kstg )', 2), ('END SELECT', 6), 28],
    'OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:378-438': [('SELECT CASE( kstg )', 3), ('DEALLOCATE( zub, zvb )', 2), 61],
    'OVERFLOW_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:480-567': ['!                       !==  T-S Tracers  ==!', ('END SELECT', 8), 88],
    'OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:501-508': [('DO jn = 1, jpts', 2), ('END DO', 14), 8],
    'OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:311-338': [('SELECT CASE( kstg )', 2), ('END SELECT', 6), 28],
    'OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/dynhpg.f90:341-414': ['SUBROUTINE hpg_sco( kt, Kmm, puu, pvv, Krhs )', 'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 74],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:291-294': ['!                          ! set values computed in RK3_ssh', 'zv_frc(:,:) =   Ve_rhs(:,:)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:325-330': ['CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', ('END DO   ;   END DO', 1), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:766-786': [('IF( kt == nit000 .AND. jn <= 2 ) THEN', 3), "CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 21],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:623-627': [('IF( nn_comm <= 1 ) THEN', 4), ('ENDIF', 4), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:1889-1909': [('!                       !                       ________________________', 3), ('ishtRj(1:4,jf) = (/ ip1j, ip1j, ip0j, im0j /)', 3), 21],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:1960-1979': [('! ----------------------------------------------- !', 3), ('CALL MPI_ISEND( buffsnd_dp(ishtS(jn)+1)', 1), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:2055-2073': [('DO jn = 1, 2   ! next: do the MPI_RECV part', 2), ('ENDIF', 90), 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/stp2d.f90:189-210': ["WRITE(l2_slow_unit) l4_canon_3d(e3u_3d,'U')", "& l4_canon_2d(r1_hu_0,'U'), l4_canon_2d(r1_hv_0,'V')", 22],
    'ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/stp2d.f90:218-233': ['!* baroclinic drag forcing *!', ("WRITE(l2_slow_unit) l4_canon_2d(Ue_rhs,'U'), l4_canon_2d(Ve_rhs,'V')", 2), 16],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:180-182': ['REAL(wp), DIMENSION(jpi,jpj) :: zu_trd, zu_spg', 'REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ) :: zu_frc, zv_frc', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:458-467': ['! Round-18 WRITE-only per-rank stream:', 'WRITE(l4_uhist_unit) l2_magic, 1, kt, 2, narea - 1, jpi, jpj, STORAGE_SIZE(1._wp)', 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:501-513': ['!* Extrapolate barotropic velocities at mid-step (jn+1/2)', 'WRITE(l4_uhist_unit) un_e, ub_e, ubb_e, ua_e', 13],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:688-711': ['! Set next velocities:', '&   ) * ssvmask(ji,jj)', 24],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:754-770': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', "CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 17],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:814-822': [('IF( kt == nit000 .AND. jn <= 2 ) THEN', 3), 'un_e   (:,:) = ua_e  (:,:)', 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R18UHIST/BLD/ppsrc/nemo/dynspg_ts.f90:866-870': [('IF( kt == nit000 ) THEN', 5), "WRITE(numout,*) 'LANE4_BT_UHIST_DUMP ', kt, 2, narea - 1, TRIM(l4_uhist_filename)", 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:445-455': ['! Round-17 WRITE-only per-rank stream.', ('ENDIF', 14), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:539-544': ['! values of zhup2_e and zhvp2_e on the halo are not needed in bdy_vol2d', 'zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)', 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:562-569': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1), 'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 8],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:571-574': ['IF( kt == nit000 .AND. jn <= 2 ) THEN', ('ENDIF', 21), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/dynspg_ts.f90:845-848': [('IF( kt == nit000 ) THEN', 4), ('ENDIF', 41), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R17HALO/BLD/ppsrc/nemo/in_out_manager.f90:180-180': 'lwp      = .FALSE.    !: boolean : true on the 1st processor only',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:339-347': ['! Initialize barotropic variables:', ('ENDIF', 10), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:355-364': [('IF( ln_bt_fw ) THEN', 2), 'hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:460-493': ['!* Set extrapolation coefficients for predictor step:', 'zsshp2_e(:,:) = za1 * sshn_e(:,:)', 34],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:505-536': [('DO jj = ntsj-( 0), ntej+(  1 ) ; DO ji = ntsi-( 0), ntei+(  1)', 1), ('END DO   ;   END DO', 8), 32],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:550-558': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1), ('END DO   ;   END DO', 9), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:601-616': ['! Half-step back interpolation of SSH for surface pressure computation', ('END DO   ;   END DO', 14), 16],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:618-652': ['! Add Coriolis trend:', ("& l4_canon_2d(zu_trd,'U'), l4_canon_2d(zv_trd,'V')", 1), 35],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:666-679': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 1), ('END DO   ;   END DO', 17), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:755-779': [('IF( lwp .AND. kt == nit000 .AND. jn <= 2 ) THEN', 3), ("& l4_canon_2d(ffv_nw,'V'), l4_canon_2d(ffv_ne,'V')", 2), 25],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:1533-1563': ['FUNCTION l4_canon_2d( pfield, cdgrid ) RESULT( zfield )', 'END FUNCTION l4_canon_2d', 31],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:139-147': ['!*  hydrostatic pressure gradient (HPG))  *!', 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )', 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:162-166': ['CASE( np_VEC_c2  )', 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:196-199': ['CASE( np_VEC_c2, np_LIN_dyn )', 'Ve_rhs(ji,jj) = SUM( e3v_3d', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:218-221': ['!* baroclinic drag forcing *!', "l4_canon_2d(CdU_u,'U'), l4_canon_2d(CdU_v,'V')", 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:229-230': ['Ue_rhs(ji,jj) =  Ue_rhs(ji,jj) + r1_rho0 * utauU(ji,jj)', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:302-303': ['IF( ln_dynspg_ts )', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:287-291': ['ssh_frc(:,:) = sshe_rhs(:,:)', 'zCdU_v  (:,:) = CdU_v   (:,:)', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:320-324': ['CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', 'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:668-671': ['ua_e(ji,jj) = (                                 un_e(ji,jj)', '+ zu_frc(ji,jj) ) &', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/domain.f90:199': 'hu_0(:,:) = hu_0(:,:) + e3u_3d(:,:,jk) * umask(:,:,jk)',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/divhor.f90:142': '!==  + runoffs divergence  ==!',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcrnf.f90:279-282': ['!==   runoff put only at the surface   ==!', 'phdivn(ji,jj,1) = phdivn(ji,jj,1) - rnf(ji,jj) * r1_rho0', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stp2d.f90:278-281': ['sshe_rhs(:,:) =                 emp(:,:)', 'sshe_rhs(:,:) = r1_rho0 * sshe_rhs(:,:)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:282-286': ['IF( .NOT.lk_linssh ) THEN           !* only heat and salt fluxes associated with mass fluxes', 'pts(ji,jj,1,jp_sal,Krhs) = pts(ji,jj,1,jp_sal,Krhs) - emp(ji,jj)*pts(ji,jj,1,jp_sal,Kbb) * z1_rho0_e3t', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:318-324': [('IF( ln_rnf ) THEN         ! input of heat and salt due to river runoff', 2), 'pts(ji,jj,jk,jp_sal,Krhs) = pts(ji,jj,jk,jp_sal,Krhs)  + rnf_tsc(ji,jj,jp_sal) * zdep', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:137,152-154': ['! save ssh, uu_b, vv_b at N+1  (computed in dynspg_ts)', ('CASE ( np_HYB )', 1), ('ssh (:,:,Kaa) = r2_3 * ssh (:,:,Kbb) + r1_3 * ssha(:,:)', 2), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:270-277': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 1), 'zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kmm))) - vv_b(ji,jj,Kmm)', 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-841': ['IF( ln_zdfiwm ) THEN', 'minimum mixing length with your parameters rmxl_min = ', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:349-350': ['avt(ji,jj,jk) = avt_k(ji,jj,jk)', 'avm(ji,jj,jk) = avm_k(ji,jj,jk)', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:355': 'avt(ji,jj,jk) = avt(ji,jj,jk) + 2._wp * rn_avt_rnf * rnfmsk(ji,jj) * wmask(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:359': 'IF( ln_zdfevd )   CALL zdf_evd( kt, Kmm, Krhs, avm, avt )',
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:371-373': ['ln_dynldf_lap =  .true.', 'nn_ahm_ijk_t  = -30', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:123': {'anchors': ('* (e3f_3d(ji-1,jj-1,jk) *(1._wp+r3f(ji-1,jj-1)*fe3mask(ji-1,jj-1,jk))) * r1_e1e2f(ji-1,jj-1)', 1), 'routine': 'dynldf_lev_lap'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:156': 'r1_e1e2f(:,:) = 1._wp / e1e2f(:,:)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:144': {'anchors': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )', 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:493': {'anchors': 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:129-130': {'anchors': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1), ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2], 'routine': 'dyn_keg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109': [('MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12', 1), ('p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 1), 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121-135': [('IF( nn_evdm == 1 ) THEN', 1), ('p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 1), 15],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:363': 'CALL zdf_ddm( kt, Kmm,  avm, avt, avs )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:372': 'IF( ln_zdfiwm )   CALL zdf_iwm( kt, Kmm, avm, avt, avs )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:294': 'zav_wave(ji,jj) = MIN( MAX( 1.4e-7_wp, zav_wave(ji,jj) ), 1.e-2_wp ) * wmask(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:314-316': ['p_avs(ji,jj,jk) = p_avs(ji,jj,jk) + zav_wave(ji,jj) * zav_ratio(ji,jj)', 'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + zav_wave(ji,jj)', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:172': 'p_avm(ji,jj,jk) = p_avm(ji,jj,jk) + MAX( zavft + zavdt, zavfs + zavds )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:442-459': ['snwice_mass  (:,:) = tmask(:,:,1) * SUM', 'ssh(:,:,Kbb) = ssh(:,:,Kbb) - zsshadj', 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:442-459': ['snwice_mass  (:,:) = tmask(:,:,1) * SUM', 'ssh(:,:,Kbb) = ssh(:,:,Kbb) - zsshadj', 18],
    'nemo_testcase_recipe.py:2228-2233': ['temperature, salinity = build_orca2_initial_ts(', 'metric["e1t"] * metric["e2t"], ORCA2_DECK_INITIAL_ICE)', 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:324': 'zapnd_ini(:,:) = rn_apd_ini_n * zmsk(:,:) * zat_i_ini(:,:)',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:343-346': ['IF ( .NOT.ln_pnd ) THEN', ('zhlid_ini(:,:) = 0._wp', 1), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE/BLD/ppsrc/nemo/stprk3.f90:396-400': [('IF( .NOT.lwp ) RETURN', 5), "FORM='UNFORMATTED', STATUS='NEW', ACTION='WRITE', IOSTAT=ios", 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:92-94': ['IF( lwp .AND. kstp >= nit000 .AND. kstp <= nit000 + 59 ) THEN', 'WRITE(cl_traj,\'("oracle_step_entry_kt",I8.8,".bin")\') kstp', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:151-152': ['CALL sbc        ( kstp, Nbb, Nbb )', 'CALL l4_dump_ocean_surface_input( kstp, Nbb )', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246': ['IF( sd(jf)%ln_tint ) THEN', 'sd(jf)%fnow(:,:,:) = ztintb * sd(jf)%fdta(:,:,:,ibb) + ztinta * sd(jf)%fdta(:,:,:,iaa)', 12],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184': ["CALL iom_rstput( kt, nitrst, numrow, 'rdt', rn_Dt )", "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )", 15],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:223-272': ['! Stage 3 :', 'IF( ln_sto_eos )   CALL sto_rst_write( kstp )', 50],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:633-643': [('IF( ln_tile )   CALL dom_tile_start', 1), ("& l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')", 4), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:645-651': ['CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', ("& l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')", 6), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:670-680': [('SELECT CASE( kstg )', 4), '&                /          ( 1._wp + r3t(ji,jj,Kaa) )', 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:45-49': ['INTEGER,  PUBLIC, PARAMETER ::   np_LIN = 0', 'INTEGER  :: n_baro_upd =  np_HYB', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:274-284': [('CASE ( np_LIN, np_HYB )', 2), ('END DO   ;   END DO   ;   END DO', 1), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:530-536': ['! resulting flux at mid-step (not over the full domain)', ('END DO   ;   END DO', 8), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:563-580': ['! Sum over sub-time-steps to compute advective velocities', "& l4_canon_2d(vn_adv,'V')", 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:829-847': ['! Finalize sums:', "WRITE(l2_adv_unit) l4_canon_2d(un_adv,'U'), l4_canon_2d(vn_adv,'V')", 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:460-469': ['!* Set extrapolation coefficients for predictor step:', ('ENDIF', 14), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:476-485': ['!* Extrapolate barotropic velocities at mid-step (jn+1/2)', ('END DO   ;   END DO', 3), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:487-522': [('IF( .NOT.lk_linssh ) THEN', 1), ('ENDIF', 17), 36],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv.f90:491-535': ['! FCT at last stage only with RK3', ('ENDIF', 27), 45],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:149-155': [('CASE(  2  )                         !* 2nd order centered', 1), ('END DO   ;   END DO', 1), 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:157-161': ['DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)                     ! Horizontal divergence of advective fluxes', ('END DO   ;   END DO', 2), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/traadv_cen.f90:202-228': ['IF( lk_linssh ) THEN                !* top value   (linear free surf. only as zwz is multiplied by wmask)', ('END DO   ;   END DO', 9), 27],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:218-253': ['IF( cn_cfg == "orca" .OR. cn_cfg == "ORCA" ) THEN', 'sf_tsd(jp_tem)%fnow( mi0(ii0,nn_hls):mi1(ii1,nn_hls) , mj0(ij0,nn_hls):mj1(ij1,nn_hls) , 14:20 ) = 6.0_wp', 36],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1587-1647': ['SUBROUTINE bn2( pts, pab, pn2, Kmm, kbnd )', 'END SUBROUTINE bn2_t', 61],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:117-118': ['IF( ln_tsd_init ) THEN', 'CALL dta_tsd( nit000, ts(:,:,:,:,Kbb) )', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:308-309': ['ptsd(ji,jj,jk,jp_tem) = ptsd(ji,jj,jk,jp_tem) * tmask(ji,jj,jk)', 'ptsd(ji,jj,jk,jp_sal) = ptsd(ji,jj,jk,jp_sal) * tmask(ji,jj,jk)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/mppini.f90:1587-1593': ['mi0(ji,jh) = MAX( 1 , MIN( ji - iimpp + 1, ipi+ishft+1 ) )', 'mj1(jj,jh) = MAX( 0 , MIN( jj - ijmpp + 1, ipj+ishft   ) )', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:2284-2293': ['CASE( np_eos80 )                        !==  polynomial EOS-80 formulation  ==!', ('r1_Z0  = 1.e-4_wp', 2), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1281-1330': [('CASE( np_teos10, np_eos80 )                !==  polynomial TEOS-10 / EOS-80 ==!', 7), 'pab(ji,jj,jk,jp_sal) = zn / zs * r1_rho0 * ztm', 50],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynvor.f90:912-937': ['SELECT CASE( nn_e3f_typ )', 'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 26],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:402-420': ['CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', 'L2_RK_STAGE2_TERM_DUMP', 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:488-555': [('CASE ( 3 )', 2), ('CALL tra_adv_trp', 1), 68],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:116-181': ['zDt_2 = rDt * 0.5_wp', ('ENDIF', 6), 66],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:187-349': [('!                    !**  tridiagonal matrix construction  **!    diagonal (zwd), lower (zwi), upper (zws)', 1), 'puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - zws(ji,jk) * puu(ji,jj,jk+1,Kaa) ) / zwd(ji,jk)', 163],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:193,195': [('(e3w_1d(jk  ) *(1._wp+r3u(ji,jj,Kmm)))', 1), ('(e3w_1d(jk+1) *(1._wp+r3u(ji,jj,Kmm)))', 1), 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:356-517': [('!                    !**  tridiagonal matrix construction  **!    diagonal (zwd), lower (zwi), upper (zws)', 2), 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - zws(ji,jk) * pvv(ji,jj,jk+1,Kaa) ) / zwd(ji,jk)', 162],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzdf.f90:362,364': [('(e3w_1d(jk  ) *(1._wp+r3v(ji,jj,Kmm)))', 1), ('(e3w_1d(jk+1) *(1._wp+r3v(ji,jj,Kmm)))', 1), 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R328ZDF/BLD/ppsrc/nemo/dynzdf.f90:206-210': ['zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )     +  avm(ji,jj,jk  )', ('zwi(ji,jk) = zzwi', 1), 5],
    'k_profiles.py:487-581': ['# NEMO-literal double diffusion (zdfddm)', 'cfg=iwm_cfg, rho_0=physics_config.constants.rho_0)', 95],
    'double_diffusion.py:330-357': ['zds = jnp.where(jnp.abs(zds) <= tiny, tiny, zds)', 'avm_new = r(avm + jnp.maximum(heat, salt))', 28],
    'internal_wave_mixing.py:430-481': ['hz = r(ht_0 * stretch)', 'avm = r(avm + zav_wave)', 52],
    'tke.py:3566-3656': ['for _ in range(max(1, int(n_iterations))):', ('prandtl_K_M=(preclosure_K_M if _carried_coeffs else None))', 2), 91],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:285-293': ['IF( .NOT.ln_drg_OFF ) THEN', 'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin ) * ssmask(ji,jj)', 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:706-711': [('DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)', 2), 'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:234-275': ['IF( (ht_0(ji,jj) *(1._wp+r3t(ji,jj,Kmm))) /= 0._wp ) THEN ; zfact1', 'zfact4(ji,jj) * SQRT( MAX( 0._wp, rn2(ji,jj,jk) ) )', 42],
    'internal_wave_mixing.py:430-466': ['hz = r(ht_0 * stretch)', '+ r(zfact4[..., None] * n_pos))', 37],
    'double_diffusion.py:357': 'avm_new = r(avm + jnp.maximum(heat, salt))',
    'ocean_model_latlon_cgrid.py:15242-15251': ['u_new = implicit_vertical_diffusion_ocean_momentum_dispatch(', ('implicit_w=nemo_aimp_momentum_w_v)', 1), 10],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynadv.f90:134-138': [('SELECT CASE( n_dynadv )', 1), 'CALL dyn_zad', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynkeg.f90:117-130': ['CASE ( nkeg_C2 )', ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) -', 1), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynzad.f90:102-137': ['zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp', '&                                              * zWdzV(ji,jj)', 36],
    'ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dynadv_round62_writer.f90:91-92': ["WRITE(r62_unit) 'e3u_Kmm         '", "WRITE(r62_unit) 'e3v_Kmm         '", 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90:136-140': ['LOGICAL, PUBLIC, PARAMETER ::   lk_qco    = .TRUE.', 'LOGICAL, PUBLIC, PARAMETER ::   lk_vco_1d3d = .TRUE.', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R62VADVSP/BLD/ppsrc/nemo/dom_oce.f90:318-367': ['IF( lk_qco .OR. lk_linssh ) THEN', '&                                e3vw(jpi,jpj,jpk,jpt) ,   STAT=ierr(ii) )', 50],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round62_vector_advection_acquisition/dynadv_round62_writer.F90:79-89': ['ALLOCATE( r62_payload(jpi,jpj,jpk) )', "R62_3D('e3v_Kmm         ',r62_payload)", 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynadv.f90:136-143': [('SELECT CASE( n_dynadv )', 1), 'CALL dynadv_round62_after_zad( kt, Kmm, Krhs, nn_dynkeg, puu, pvv )', 8],
    'ORCA2_ORCA1ICE_OMIP_L4_R63VADVSP/BLD/ppsrc/nemo/dynzad.f90:102-137': ['zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp', '&                                              * zWdzV(ji,jj)', 36],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:912-937': ['SELECT CASE( nn_e3f_typ )', 'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 26],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123': ('zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk) * (e3f_3d(ji-1,jj-1,jk)', 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:914-919': [('DO jk =  1,  jpk  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)', 1), ('END DO   ;   END DO   ;   END DO', 2), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:935': ("CALL lbc_lnk( 'dynvor', e3f_0vor, 'F', 1._wp )", 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:937': ('WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 1),
    'domzgr_substitute.h90:100': ('#     define  e3f_0(i,j,k)    e3f_3d(i,j,k)', 1),
    'DOM/domzgr.F90:173': ("CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:734-738': [('DO jk = 1, jpkm1', 4), ('z1_e3f(ji,jj) = 1._wp / (e3f_0vor(ji,jj,jk)', 1), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:741-779': [('SELECT CASE( kvor )                 !==  vorticity considered  ==!', 3), ("CALL ctl_stop('STOP','dyn_vor: wrong value for kvor'  )", 3), 39],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:784-802': [('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d(ji,jj,jk)', 3), ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2), 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domqco.f90:273-286': [('IF( PRESENT( pr3f ) ) THEN             !==  ratio at f-point  ==!', 2), ('END DO   ;   END DO', 6), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domhgr.f90:155-157': ['e1e2t (:,:) = e1t(:,:) * e2t(:,:)', 'IF( ie1e2u_v == 0 ) THEN', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/domain.f90:203-206': [('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)', 1), "CALL lbc_lnk('domain', hf_0, 'F', 1._wp)", 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dommsk.f90:258': ('fe3mask(:,:,:) = fmask(:,:,:)', 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:139-166': [('!*  hydrostatic pressure gradient (HPG))  *!', 1), ('CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 1), 28],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:189-204': [("WRITE(l2_slow_unit) l4_canon_3d(e3u_3d,'U')", 1), ('Ve_rhs(ji,jj) = Ve_rhs(ji,jj) + SUM(', 1), 16],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:218-221': [('!* baroclinic drag forcing *!', 1), ('l4_canon_2d(CdU_v', 1), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:225-235': [('WRITE(l2_slow_unit) r1_rho0', 1), ('LANE2_SLOW_FORCING_DUMP', 1), 11],
    'ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/stp2d.f90:155-187': [('CALL eos    ( ts, Kbb, rhd )', 1), ('WRITE(r40_family_unit) l4_canon_3d(uu(:,:,:,Krhs)', 5), 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R40RHSFAM/BLD/ppsrc/nemo/dynhpg.f90:340-451': [('SUBROUTINE hpg_sco( kt, Kmm, puu, pvv, Krhs )', 1), ('END SUBROUTINE hpg_sco', 1), 112],
    'ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/eosbn2.f90:810-842': [('CASE( np_teos10, np_eos80 )', 3), ('prd(ji,jj,jk) = (  zn * r1_rho0 - 1._wp  ) * ztm', 3), 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R41HPG1/BLD/ppsrc/nemo/dynhpg.f90:403-458': [('zhpi(ji,jj) = zcoef0 * r1_e1u(ji,jj)', 1), ('r41_sum_u (ji,jj,jk) = zhpi(ji,jj) + zuap', 1), 56],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:265-284': [('ALLOCATE( zub(ntsi-', 1), ('END DO   ;   END DO   ;   END DO', 1), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:551-645': [('IF( .NOT.ln_shuman ) THEN', 1), ('CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 1), 95],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv.f90:267-315': [('IF( ll_Fw ) THEN', 1), ('CLOSE(993)', 1), 49],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sshwzv.f90:271-299': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 2), ('END DO   ;   END DO   ;   END DO', 6), 29],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160': [('DO jn = 1, kjpt', 1), ('r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:202-227': [('IF( lk_linssh ) THEN', 1), ('r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 4), 26],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:275-328': [('!        EMP, SFX and QNS effects', 2), ('ENDIF', 24), 54],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:467-480': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 1), ('END DO   ;   END DO   ;   END DO', 6), 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681': [('SELECT CASE( kstg )', 4), ('END DO   ;   END DO   ;   END DO', 8), 12],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140': [('CALL dta_tsd_init', 1), ('vv    (:,:,:,Kmm) = vv   (:,:,:,Kbb)', 1), 48],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:217-254': [('!                                   !==   ORCA_R2 configuration and T & S damping   ==!', 1), ('ENDIF', 9), 38],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:307-310': [('DO jk =  1,  jpk', 2), ('END DO   ;   END DO   ;   END DO', 2), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465': [('! 4) Adjust ssh and vertical scale factors to snow-ice mass', 1), ('CALL dom_qco_zgr( Kbb, Kmm )', 1), 26],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:160-179': [('!                     !==  ssh/h0 ratio at Kaa  ==!', 1), ('r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 1), 20],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:556': ('zwz(ji,jj) = zwz(ji,jj) / (e3f_0vor(ji,jj,jk) *(1._wp+r3f(ji,jj)*fe3mask(ji,jj,jk)))', 1),
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3.f90:230-241': ['CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 12],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:493-506': ['CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 14],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140': [('zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk)', 1), ('+ ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 18],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1), ('e1u(ji-1,jj  ) * pu(ji-1,jj  ,jk,Kbb)', 1), 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:127-129': [('zwt(ji,jj)     = ahmt(ji,jj,jk) * r1_e1e2t(ji,jj)', 1), ('e1v(ji,jj-1)*(e3v_3d(ji  ,jj-1,jk)', 1), 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:132-140': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 1), ('+ ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 9],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353': ['CASE( -30  )', ('CALL iom_close( inum )', 2), 6],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:387-393': ['IF( .NOT.l_ldfdyn_time ) THEN', 'ahmf(:,:,1:jpkm1) = SQRT( ahmf(:,:,1:jpkm1) ) * fmask(:,:,1:jpkm1)', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:323-329': [('IF( ln_dynadv_vec ) THEN', 1), 'CALL wAimp( kstp, Kmm, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, wi, np_velocity, ld_diag=.TRUE. )', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/traadv.f90:296-300': ['CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 'CALL wAimp( kt, Kmm, pFu, pFv, ww, wi, np_transport )', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/divhor.f90:123-130': ['SELECT CASE ( ik_ind )', (') * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 8],
    'DINO/BLD/ppsrc/nemo/domqco.f90:177-181': ['CALL dom_qco_r3c( ssh(:,:,Kbb), r3t(:,:,Kbb), r3u(:,:,Kbb), r3v(:,:,Kbb)           )', "&                         r3u(:,:,Kmm), 'U', 1._wp, r3v(:,:,Kmm), 'V', 1._wp, r3f(:,:), 'F', 1._wp )", 5],
    'DINO/BLD/ppsrc/nemo/domqco.f90:211-215': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 1), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 5],
    'DINO/BLD/ppsrc/nemo/lbclnk.f90:1816-1820': [('zland = 0._wp                                     ! land filling value: zero by default', 3), ('IF( PRESENT(kfillmode) )   ifill_nfd = kfillmode', 3), 5],
    'DINO/BLD/ppsrc/nemo/lbclnk.f90:1866-1871': [('DO jn = 1, 4   ! 4 sides', 2), ('ELSE                                ;   ifill(jn,jf) = jpfillcst       ! constant value (zland)', 3), 6],
    'DINO/BLD/ppsrc/nemo/lbclnk.f90:2130-2135': [('IF(     ifill(jn,jf) == jpfillcst ) THEN', 5), ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = zland', 5), 6],
    'DINO/BLD/ppsrc/nemo/dynspg_ts.f90:484-487': ['hu_e  (:,:) =    (hu_0(:,:) *(1._wp+r3u(:,:,Kmm)))', 'hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/stprk3.f90:167-168': ['CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 2],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90:317-320': [('IF( l_zdfsh2 ) THEN', 1), '&                      sh2    )', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:97-103': ['ELSE', '&         * wumask(ji,jj,jk)', 7],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:104-108': [('zsh2v(ji,jj) = ( p_avm(ji,jj+1,jk) + p_avm(ji,jj,jk) )', 2), '&         * wvmask(ji,jj,jk)', 5],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:111-114': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 1), ('END DO   ;   END DO', 3), 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:116-119': ['DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0) ! set p_sh2', ('END DO   ;   END DO', 4), 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/dommsk.f90:234-242': ['Ocean/land mask at wu-, wv- and w points', 'wvmask(:,:,jk) = vmask(:,:,jk) * vmask(:,:,jk-1)', 9],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/domqco.f90:211-218': [('ratio at u-,v-point', 1), ('END DO   ;   END DO', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r54_tke.f90:128-130': ['WRITE(r54_unit) r54_magic', '& 1,ntei-ntsi+1,1,ntej-ntsj+1,STORAGE_SIZE(1._wp)', 3],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:65-67': ['WRITE(r101_unit) r101_magic', '& STORAGE_SIZE(1._wp)', 3],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:284': 'en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )',
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:387': 'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)',
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:433-435': ['zd_up(ji,jk) = zzd_up', 'zdiag(ji,jk) = 1._wp - zzd_lw - zzd_up + zfact2 * dissl(ji,jj,jk) * wmask(ji,jj,jk)', 3],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:438-441': ['en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * (  p_sh2(ji,jj,jk)', '&                                  ) * wmask(ji,jj,jk)', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:475-492': ['DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)                 ! First recurrence', 'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:689-692': ['zemlm = MIN ( zmxld(ji,jk),  zmxlm(ji,jk) )', 'zmxld(ji,jk) = zemlp', 4],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:700-705': ['DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)   !* vertical eddy viscosity', 'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:711': 'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk)',
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:257': ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 2),
    'ocean_pe_latlon_cgrid.py:1800': 'r3_now = jax.lax.optimization_barrier(eta_now * r1_h0)',
    'ocean_pe_latlon_cgrid.py:1807': 'live_t = e3t0 * (1.0 + r3_now[..., None] * tmask) * tmask',
    'ocean_pe_latlon_cgrid.py:1831': 'flux_u, west, flux_v, south, r1_area_t, live_t[..., jk],',
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:101': ('pe3divUh       ! e3t*div[Uh]', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:278': ('CALL div_hor( kt, Kbb, Kmm, pu, pv, ze3div )', 1),
    'ocean_pe_latlon_cgrid.py:1629': 'return jax.lax.optimization_barrier(live_e3t * hdiv) * tmask',
    'ocean_pe_latlon_cgrid.py:1902-1903': ['r3_after = jax.lax.optimization_barrier(eta_after * r1_h0)',
 'r3_before = jax.lax.optimization_barrier(eta_before * r1_h0)',
 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:172': ('ssh (:,:,Kaa) = r2_3 * ssh (:,:,Kbb) + r1_3 * ssha(:,:)', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:185': ('CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:211': ('r3u(:,:,Kaa) = r2_3 * r3u(:,:,Kbb) + r1_3 * r3ua(:,:)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:255': ('r3u(:,:,Kaa) = r1_2 * ( r3u(:,:,Kbb) + r3ua(:,:) )', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/domqco.f90:266-267': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2), ('&                    + e1e2t(ji+1,jj) * pssh(ji+1,jj)  ) * r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)', 2), 2],
    'vertical.py:240-241': ['weighted_eta = b(area_t * eta)',
 'num_u = b(half * b(weighted_eta + jnp.roll(weighted_eta, -1, axis=1)))',
 2],
    'vertical.py:256': 'r3u = b(b(num_u * r1_hu0) * r1_area_u)',
    'vertical.py:264': 'e3u = b(e3u_0 * b(one + r3u[..., None] * umask3))',
    'ocean_model_latlon_cgrid.py:2247-2284': ['def nemo_stage_momentum_wzv_executes(config, hooks=None) -> bool:',
 'return bool(config_split)',
 38],
    'ocean_model_latlon_cgrid.py:2432-2443': ['w_momentum, _, _ = nemo_qco_wzv_operands(', ('else 0))', 2), 12],
    'ocean_model_latlon_cgrid.py:9442-9448': ['def _momentum_stage_w(geom):', 'return geom[2] if geom[11] is None else geom[11]', 7],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:52': ('INTEGER  :: n_baro_upd =  np_HYB', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:221-222': [('rDt = r1_2 * rn_Dt', 1), ('r1_Dt = 1._wp / rDt', 2), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:254': ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:265-266': [('rDt = rn_Dt                   ! set time-step : rn_Dt', 1), ('r1_Dt = 1._wp / rDt', 3), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:268': ('ssh (:,:,Kaa) = ssha(:,:)     ! recover ssh and (uu_b,vv_b) at N + 1', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:362': ('CALL wAimp( kstp, Kmm, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, wi, np_velocity, ld_diag=.TRUE. )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:277': ('CALL wAimp( kt, Kmm, pFu, pFv, ww, wi, np_transport )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:309': ('zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:315-316': [('zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk))) * ( uu(ji,jj,jk,Kmm) + zub(ji,jj)*umask(ji,jj,jk) )', 1), ('zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk))) * ( vv(ji,jj,jk,Kmm) + zvb(ji,jj)*vmask(ji,jj,jk) )', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:356': ('IF( ln_dynadv_vec ) THEN', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:358': ('IF( kstg /= 1 ) THEN', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360': ('CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:367': ('CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:297-298': [('pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)', 1), ('+ r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130': [('hdiv(ji,jj,jk) = (  (  e2u(ji  ,jj) * (e3u_3d(ji  ,jj,jk) *(1._wp+r3u(ji  ,jj,Kmm)*umask(ji  ,jj,jk))) * pu(ji  ,jj,jk)', 1), ('&             ) * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:134-138': [('hdiv(ji,jj,jk) = (  (  pu(ji  ,jj,jk)       &   ! add () for NP repro', 1), ('&             ) * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:153': ('pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:195': ('IF( ln_dynadv_vec )   ll_Fw = .true.', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274': ('CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:280': ('pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:153': ('CASE( np_VEC_c2  )                                                         != vector form =!', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:171': ('CALL dyn_keg     ( kt, nn_dynkeg     , Kmm, puu, pvv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynadv.f90:176': ('CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90:121-125': ['zu =    puu(ji-1,jj  ,jk,Kmm) * puu(ji-1,jj  ,jk,Kmm)   &', 'zhke(ji,jj) = 0.25_wp * ( zv + zu )', 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynkeg.f90:129-130': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1), ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:102': ('zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp                  ! set surface (jk=1) vertical advection to zero', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:112-116': ['zWf  = e1e2t(ji  ,jj  ) *   ww(ji  ,jj  ,jk+1)', 'zzWfu = zWfi + zWf                     ! averaging at uw- and vw-points (x2)', 5],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:119': ('zzWdzU = zzWfu * ( puu(ji,jj,jk,Kmm) - puu(ji,jj,jk+1,Kmm) )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:123-126': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk)))   &', 1), ('&                                            * ( zWdzV(ji,jj) + zzWdzV )', 1), 4],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/dynzad.f90:134-137': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk)))   &', 2), ('&                                              * zWdzV(ji,jj)', 1), 4],
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:462': ("CALL r156_stage2_pair3( 'rhs_entry       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:495': ("CALL r156_stage2_pair3( 'rhd_ww          ', rhd, ww )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:497': ('CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:499': ("CALL r156_stage2_pair3( 'after_hpg       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:510': ('CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:512': ("CALL r156_stage2_pair3( 'after_vor       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:523': ('IF( ln_dynadv_vec ) THEN                        ! VIF: only velocities used for momentum advection', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:525': ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:531': ("CALL r156_stage2_pair3( 'after_adv       ', uu(:,:,:,Krhs), vv(:,:,:,Krhs) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:721': ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:723': ('uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:738': ("CALL r156_stage2_pair3( 'uu_vv_Kaa_raw   ', uu(:,:,:,Kaa), vv(:,:,:,Kaa) )", 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:787': ('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:810': ('uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:217-256': ['CASE ( 2 )           !==  Stage 2  ==!', ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 40],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:400-497': ['CASE ( 2 , 3 )    !==  Stage 2 & 3  ==!', 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 98],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:475': ('CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:485': ('CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:497': ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:697-699': ['uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) ) * uu(ji,jj,jk,Kbb )', '/           ( 1._wp + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)', 3],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:732': ('CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:745': ('IF( kstg == 3 )   CALL dyn_zdf', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:753': ('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:772': ('uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:261-277': ['CASE ( 3 )           !==  Stage 3  ==!', 'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 17],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:300-309': [('SELECT CASE( n_baro_upd )', 4), 'zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kmm))) - vv_b(ji,jj,Kmm)', 10],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:313-315': [('DO jk =  1,  jpkm1', 1), 'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 3],
    'GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:317-321': ['IF( lwp .AND. .NOT.ln_tile .AND. kstp == 1081 .AND. kstg == 3 )', '& un_adv, vn_adv, r1_hu_0, r1_hv_0, uu_b(:,:,Kmm), vv_b(:,:,Kmm) )', 5],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:260-314': ['CASE ( 3 )           !==  Stage 3  ==!', 'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))', 55],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:785': ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/stprk3_stg.f90:879': 'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )',
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90:195': 'IF( ln_dynadv_vec )   ll_Fw = .true.',
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv.f90:266-280': ['IF( ll_Fw ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 15],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:176-178': ["IF( cdtype == 'TRA' ) CALL r153_begin", 'CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt', 3],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:515-553': ['ptFu(ji,jj,jk) = MAX( pU(ji,jj,jk) , 0._wp ) * pt_b(ji,jj,jk)', ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)', 2), 39],
    'GYRE_OMIP_L2_P3_SM_R153FCTD/BLD/ppsrc/nemo/traadv_fct.f90:924-953': ['zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji+1,jj,ik) )', ('pcc(ji,jj,jk) = pcc(ji,jj,jk) * zcoef', 2), 30],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:164-199': ['DO jn = 1, kjpt            !==  loop over the tracers  ==!', ('END DO   ;   END DO   ;   END DO', 4), 36],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:495-610': ['! *** 1st step', ('END DO   ;   END DO   ;   END DO', 28), 116],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:306-330': ['!        !==  monotonicity algorithm  ==!', ('END DO   ;   END DO   ;   END DO', 9), 25],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:849-883': ['!        !==  compute the beta term  ==!   (zbetup/do)', ('END DO   ;   END DO', 53), 35],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/traadv_fct.f90:886-936': ['!        !==  monotonic flux  ==!', 'END DO    ! jk-loop', 51],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:111-140': ['r148_ldf_dump = lwp .AND. kt == 1081 .AND. Kbb == Kmm', ('ENDIF', 4), 30],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:142-197': [('DO jk = 1, jpkm1', 1), ('ENDIF', 7), 56],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:204-217': ['! Round 140: repeat the established slow-forcing operand list', ('ENDIF', 6), 14],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynldf_lev.f90:157-176': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:256-286': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls)', 2), ('END DO   ;   END DO', 6), 31],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/domqco.f90:273-286': [('IF( PRESENT( pr3f ) ) THEN             !==  ratio at f-point  ==!', 2), ('END DO   ;   END DO', 6), 14],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/dynvor.f90:911-936': ['SELECT CASE( nn_e3f_typ )', 'WHERE( e3f_0vor(:,:,:) == 0._wp )   e3f_0vor(:,:,:) = e3f_3d(:,:,:)', 26],
    'vertical.py:469-573': ['def nemo_qco_live_vorticity_e3f_cgrid(', ('return e3f_cgrid', 2), 105],
    'ocean_model_latlon_cgrid.py:6805-6807': ['_ws_uses_nemo_ldf_e3 = (', 'and _cfg_b.lateral_viscosity_e3_weighting == "nemo_e3")', 3],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:158-164': ['CALL eos    ( ts, Kbb, rhd )', ('IF( lwp .AND. kt == 1081 )   WRITE(r146_family_unit)', 2), 7],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stprk3_stg.f90:704-739': [('!              !---------------!', 2), 'IF( kstg == 3 )   CALL dyn_zdf(', 36],
    'ocean_model_latlon_cgrid.py:6896-6949': ['if not self._nemo_ws_test_hooks.legacy_hadv_min_face_thickness:', 'ldf_thickness_operands=_ws_ldf_thickness_kbb,', 54],
    'ocean_model_latlon_cgrid.py:7821-7886': ['_stage_ldf_thickness = None', '_stage_ldf_thickness),', 66],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:220-229': ['IF ( .NOT. l_perpetual_ts ) THEN', ('ENDIF', 8), 10],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:188-201': ['!  RK3 : single first external mode computation', 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:222-226': ['Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stprk3.f90:249-260': ['CALL dia_wri   ( kstp,      Nbb )', 'CALL rst_write    ( kstp, Nbb, Nnn, Naa )', 12],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf.f90:69-110': ['SUBROUTINE tra_ldf( kt, Kbb, Kmm, pts, Krhs )', 'CALL traldf_iso_lap  ( kt, Kbb, Kmm, pts, Krhs, l_ptr, l_hst )', 42],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/traldf_iso.f90:154-300': [('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1), ('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 2), 147],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:176-184': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )", 9],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/restart.f90:354-370': ['!                                     !*  RK3: Read ssh at Kbb', 'ssh(:,:,Kaa) = ssh(:,:,Kbb)               ! no ssh variation in ww computation', 17],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/dynspg_ts.f90:974-980': [('IF( nn_bt_flt == 3 ) THEN', 2), "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'     ,    vb_e(:,:) )", 7],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/zdftke.f90:895-901': ["ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'dissl', dissl )", 7],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/stp2d.f90:150-166': ['IF( .NOT.lk_linssh ) THEN', 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 17],
    'GYRE_OMIP_L2_P3_SM_R132DAILY/BLD/ppsrc/nemo/sshwzv.f90:293-299': [('ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!', 2), ('END DO   ;   END DO   ;   END DO', 6), 7],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3.f90:173-180': ['IF( l_ldfslp ) THEN', ('ENDIF', 7), 8],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/stprk3_stg.f90:928-934': [('IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000 .AND. kstg == 3 )', 5), 'IF( ln_trabbl  )   CALL tra_bbl', 7],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf.f90:105-118': [('SELECT CASE ( nldf_tra )', 2), 'CALL traldf_iso_blp', 14],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:222-268': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4), '&                   * ( umask(ji,jj  ,jk) + umask(ji,jj  ,jk+1) ) * 0.5_wp', 47],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:156-180': [('! nmln calculation in zdfmxl is only on internal points', 1), 'r1_hmlw(ji,jj) = 1._wp / MAX( hmlp(ji,jj) - ((gdepw_1d(mikt(ji,jj)) ) *(1._wp+r3t(ji,jj,Kmm))), 10._wp )', 25],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:183-213': [('iikm1 = 1   ;   iik = 2', 1), ('END DO   ;   END DO', 8), 31],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/ldfslp.f90:262-268': ['uslp(ji,jj,jk) = z1_16', '&                   * ( umask(ji,jj  ,jk) + umask(ji,jj  ,jk+1) ) * 0.5_wp', 7],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-180': ['IF( l_ldfslp ) THEN', ('ENDIF', 7), 8],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:187-229': [('! nmln calculation in zdfmxl is only on internal points', 1), ('END DO   ;   END DO', 5), 43],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:231-261': ['iikm1 = 1   ;   iik = 2', ('END DO   ;   END DO', 8), 31],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/ldfslp.f90:270-361': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4), ('END DO   ;   END DO', 10), 92],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfphy.f90:326-330': ['! Update top/bottom drag', 'CALL zdf_mxl( kt, Kmm )                        !* mixed layer depth, and level', 5],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124': ['! w-level of the mixing and mixed layers', ('END DO   ;   END DO', 3), 16],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:159-168': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90:1609-1619': ['DO jk =  2,  jpkm1', ('END DO   ;   END DO   ;   END DO', 15), 11],
    'GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/domzgr.f90:382-386': ['! deepest/shallowest W level Above/Below ~10m', 'nla10 = nlb10 - 1', 5],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:167': ('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1),
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:215-250': [('IF( jk == 1 ) THEN', 1), ('END DO   ;   END DO', 5), 36],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:272-299': [('DO jj = ntsj-( 0+1), ntej+(  0 ) ; DO ji = ntsi-( 0+1), ntei+(  0 )', 1), ('END DO   ;   END DO', 7), 28],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:311-344': [('IF( jk ==1 )   zfw', 1), ('ENDIF', 5), 34],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:346-367': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1), ('&                 * r1_e1e2t(ji,jj) / (e3t_3d', 2), 22],
    'GYRE_OMIP_L2_P3_SM_R177TRALDF/BLD/ppsrc/nemo/traldf_iso.f90:403-419': [('IF( lr177_write ) THEN', 2), 'WRITE(r177_unit) e3w_1d', 17],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:222-226': ['Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:249-260': ['CALL dia_wri   ( kstp,      Nbb )', 'CALL rst_write    ( kstp, Nbb, Nnn, Naa )', 12],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90:176-184': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )", 9],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90:974-980': [('IF( nn_bt_flt == 3 ) THEN', 2), "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'     ,    vb_e(:,:) )", 7],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:895-901': ["ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'dissl', dissl )", 7],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3.f90:159-168': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1259-1308': [('CASE( np_teos10, np_eos80 )', 7), 'pab(ji,jj,jk,jp_sal) = zn / zs * r1_rho0 * ztm', 50],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/eosbn2.f90:1609-1618': ['DO jk =  2,  jpkm1', '/ (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm))) * wmask(ji,jj,jk)', 10],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/domqco.f90:208': ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:132-140': ['ll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )', 'IF( jpts /= 2 )', 9],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:217-221': ['!-- the tracer fields as tra_zdf RECEIVED them --!', 'WRITE(il2_unit) pts(:,:,:,jp_sal,Kbb)', 5],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:332-337': ["WRITE(il2_unit) 'r3t_Kbb", 'WRITE(il2_unit) r3t(:,:,Kaa)', 6],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:124-128': ['&nameos', 'ln_seos     = .false.', 5],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/stprk3_stg.f90:937-944': ['IF( ln_zdfmfc  )   CALL tra_mfc', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 8],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:418-480': ["IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN", 'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )', 63],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-469': ['zzwi = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk  ) *(1._wp+r3t(ji,jj,Kmm)))', 'zzws = - p2dt * zwt(ji,jk+1) / (e3w_1d(jk+1) *(1._wp+r3t(ji,jj,Kmm)))', 2],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:468-474': ['zzwi = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk  ) *(1._wp+r3t(ji,jj,Kmm)))', 'zws(ji,jk) = zzws - p2dt *   MAX( wi(ji,jj,jk+1) , 0._wp )', 7],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:470': ('zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk)))', 1),
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:527-582': ['!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 56],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-567': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', 'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)', 19],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:549-552': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', ('END DO', 23), 4],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:562-567': [('DO jk =    2,  jpkm1,  1', 2), 'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)', 6],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/trazdf.f90:577-582': ['!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:136-150': ["INQUIRE( FILE='round172_e3t.arm', EXIST=ll_r172_e3t )", "CALL ctl_stop( 'tra_zdf: cannot read Round-172 solve-input frame' )", 15],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:490-504': ['IF( ln_zad_Aimp ) THEN', 'zws(ji,jk) = zzws - p2dt *   MAX( wi(ji,jj,jk+1) , 0._wp )', 15],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:561-565': ['!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1', 'zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)', 5],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:583-609': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', 'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)', 27],
    'GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:619-624': ['!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfphy.f90:334-359': ['SELECT CASE ( nzdf_phy )', 'IF( ln_zdfevd )   CALL zdf_evd( kt, Kmm, Krhs, avm, avt )', 26],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:681-692': ['!* vertical eddy viscosity & diffivity at w-points', 'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk)', 12],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdfevd.f90:108-109': [('IF(  MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12 )', 1), '&  p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 2],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/EXP00/namelist_cfg:205-213': ['&namzdf        !   vertical physics', 'rn_avt0     =   1.2e-5', 9],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830': ['lr123_write = lwp .AND. .NOT.ln_tile .AND. kstg == 3', 'WRITE(r123_unit) r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa)', 13],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3.f90:188-201': ['!  RK3 : single first external mode computation', 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stp2d.f90:291-298': ['!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', ('CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 1), 8],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/dynspg_ts.f90:797-827': ['! Finalize sums:', 'pssh (:,:,Kaa) = ssha_e(:,:)', 31],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:150-180': ['ssha(:,:) = ssh (:,:,Kaa)', 'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 31],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:223-255': [('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 1), ('r3t(:,:,Kaa) = r3ta(:,:)', 3), 33],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/domqco.f90:237-257': [('SUBROUTINE dom_qco_r3c_RK3( pssh, pr3t, pr3u, pr3v, pr3f )', 1), ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869': ['CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )', ('IF( lr123_write )   WRITE(r123_unit) ts(:,:,:,jp_tem,Krhs)', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952': ['IF( ln_traqsr  ) THEN', ('IF( lr123_write )   WRITE(r123_unit) ts(:,:,:,jp_tem,Krhs)', 4), 23],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970': ['!== TRA time integration + ZDF', "CALL ctl_stop( 'stp_RK3_stg: cannot close round-123 process dump' )", 7],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560': ['!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1', '& + p2dt * (e3t_3d(ji,jj,jk)', 16],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578': ['pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk)', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 16],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:849-876': ['!==  compute the beta term  ==!', ('IF( zdo /=  zbig .AND. zneg /= 0._wp ) THEN', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv_fct.f90:905-931': ['zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji+1,jj,ik) ),', ('pcc(ji,jj,jk) = pcc(ji,jj,jk) * zcoef', 2), 27],
    'GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/traadv.f90:358-364': [('SELECT CASE ( nadv )', 1), 'CALL tra_adv_fct ( kt, nit000', 7],
    'round123/oracle_process_budget/ocean.output:756-764': ['Namelist namtra_adv : chose a advection scheme for tracers', 'implicit   optimized(1)/accurate(2)    nn_fct_imp =            1', 9],
    'GYRE_OMIP_L2_P3_SM_R125ZDFMAG/BLD/ppsrc/nemo/zdftke.f90:473-474': ['Set the minimum value of tke', 'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin )', 2],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:89-99': ['IF( lwp .AND. kstp >= nit000 .AND. kstp <= nit000 + 59 ) THEN', "WRITE(numout,*) 'LANE1_STEP_ENTRY_DUMP '", 11],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:215-222': ['CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 8],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3.f90:260': 'CALL rst_write    ( kstp, Nbb, Nnn, Naa )',
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_istate.f90:101-105': ['IF( nn_pert_seed /= 0 ) THEN', "IF(lwp) WRITE(numout,*) 'TINY PERTURBATION SEED = ', nn_pert_seed", 5],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:42': 'INTEGER, PUBLIC ::   nn_pert_seed = 0',
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:118-122': ['NAMELIST/namusr_def/ nn_GYRE, ln_bench, jpkglo, nn_pert_seed', 'IF(lwm)   WRITE( numond, namusr_def )', 5],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/usrdef_nam.f90:137-145': ["WRITE(numout,*) 'usr_def_nam  : read the user defined namelist (namusr_def) in namelist_cfg'", "WRITE(numout,*) '      from-rest ensemble perturbation seed        nn_pert_seed = ', nn_pert_seed", 9],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:812,843,849': ['ts(:,:,:,jn,Krhs) = 0._wp', 'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )', 'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 3],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:915,930,944': ['CALL tra_qsr( kstp, Kmm, ts, Krhs )', 'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 3],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/stprk3_stg.f90:902,933,934,935,937,939,953': ['IF( ln_bdy     )   CALL bdy_tra_dmp', 'IF( ln_trabbc  )   CALL tra_bbc', 'IF( ln_trabbl  )   CALL tra_bbl', 'IF( ln_tradmp  )   CALL tra_dmp', 'IF( ln_zdfmfc  )   CALL tra_mfc', 'CALL tra_osm( kstp,      Kmm, ts, Krhs )', 'IF( ln_zdfnpc  )   CALL tra_npc', 7],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:546-560': ['pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1)', '& + p2dt * (e3t_3d(ji,jj,jk)', 15],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:563-578': ['pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk)', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 16],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:132-146': ['ll_l2_tra = ( lwp .AND. kt <= nit000 + 1 )', 'zl2_srh(:,:,:) = pts(:,:,:,jp_sal,Krhs)', 15],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:160-335': ['!                    !--- round-35 instrument: the record ---!', 'CLOSE(il2_unit)', 176],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:213-333': ['!              !-- the tracer fields as tra_zdf RECEIVED them --!', 'WRITE(il2_unit) r3t(:,:,Kaa)', 121],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:414-450': ["IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN", 'zwt(:,1) = 0._wp', 37],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:462-476': ['IF( ln_zad_Aimp ) THEN', 'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )', 15],
    'GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:523-527': ['DO ji = ntsi-( 0), ntei+( 0 )          !* 1st recurrence:', 'zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)', 5],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/EXP00/namelist_cfg:126-128': ['ln_teos10   = .true.', 'ln_seos     = .false.', 3],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1257-1310': [('SELECT CASE ( neos )', 5), ('END DO   ;   END DO   ;   END DO', 13), 54],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1610-1611': ['zrw =   ( ((gdepw_1d(jk  ) )', '&  / ( ((gdept_1d(jk-1) )', 2],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1613': 'zaw = pab(ji,jj,jk,jp_tem)',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1614': 'zbw = pab(ji,jj,jk,jp_sal)',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616': 'pn2(ji,jj,jk) = grav',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1617': '- zbw * ( pts(ji,jj,jk-1,jp_sal)',
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616-1617': ['pn2(ji,jj,jk) = grav', '- zbw * ( pts(ji,jj,jk-1,jp_sal)', 2],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1616-1618': ['pn2(ji,jj,jk) = grav', '/ (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))', 3],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/stprk3.f90:159-168': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/eosbn2.f90:1609-1618': ['DO jk =  2,  jpkm1', '&            / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm))) * wmask(ji,jj,jk)', 10],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:261': ('zfact3  = 0.5_wp         * rn_ediss', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-436': ['zd_up(ji,jk) = zzd_up', 'zdiag(ji,jk) = 1._wp - zzd_lw - zzd_up', 3],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:439-442': [('en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt', 2), '&                                  ) * wmask(ji,jj,jk)', 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3_stg.f90:240-256': ['CASE ( 3 )           !==  Stage 3  ==!   Kbb = N   ;   Kmm = N+1/2   ;   Kaa = N+1', 'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 17],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/trazdf.f90:461-477': ['! Diagonal, lower (i), upper (s)', ('END DO   ;   END DO', 10), 17],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:434-442': ['zd_up(ji,jk) = zzd_up', '&                                  ) * wmask(ji,jj,jk)', 9],
    'DINO/BLD/ppsrc/nemo/stpmlf.f90:187-193': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nnn )', 'CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )', 7],
    'DINO/BLD/ppsrc/nemo/zdfphy.f90:316-319': [('IF( l_zdfsh2 ) THEN', 1), '&                      sh2    )     ! ==>> out : shear production', 4],
    'DINO/BLD/ppsrc/nemo/trazdf.f90:219-235': ['! Diagonal, lower (i), upper (s)', ('END DO   ;   END DO', 9), 17],
    'DINO/BLD/ppsrc/nemo/usrdef_zgr.f90:123-135': ['IF( ld_zco ) THEN', 'CALL zgr_msk_top_bot( pdept_1d, zbathy, k_top, k_bot )', 13],
    'DINO/BLD/ppsrc/nemo/zgr_lib.f90:199-209': [('!                       !==  t- and w- scale factors from depth  ==!', 3), '&                   pe3uw, pe3vw        )', 11],
    'DINO/BLD/ppsrc/nemo/domain.f90:194-216': ['ht_0(:,:) = 0._wp  ! Reference ocean thickness', 'r1_hf_0(:,:) = ssfmask(:,:) / ( hf_0(:,:) + 1._wp -  ssfmask(:,:) )', 23],
    'DINO/BLD/ppsrc/nemo/domqco.f90:186-207': ['SUBROUTINE dom_qco_r3c( pssh, pr3t, pr3u, pr3v, pr3f )', ('END DO   ;   END DO', 1), 22],
    'DINO/BLD/ppsrc/nemo/eosbn2.f90:1507-1517': ['DO jk =  2,  jpkm1', ('END DO   ;   END DO   ;   END DO', 15), 11],
    'DINO/BLD/ppsrc/nemo/stpmlf.f90:197-216': ['IF( l_ldfslp ) THEN', 'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nnn )', 20],
    'DINO/BLD/ppsrc/nemo/ldfslp.f90:139-177': ['INTEGER , INTENT(in)                   ::   kt', 'zhmlpt(ji,jj) = ((gdept_3d(ji,jj,nmln(ji,jj)-1)', 39],
    'DINO/BLD/ppsrc/nemo/ldfslp.f90:215-216': [('DO jj = ntsj-( 1), ntej+(  1 )', 3), 'r1_hmlw(ji,jj) = 1._wp / MAX(', 2],
    'DINO/BLD/ppsrc/nemo/ldfslp.f90:320-345': [('DO jj = ntsj-( 1), ntej+(  1 )', 5), 'zck = ( ((gdepw_3d(ji,jj,jk)', 26],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:167-168': ['!!st                         CALL zdf_phy( kstp, Nbb, Nnn, Nrhs )', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:319-320': ['CALL zdf_sh2( Kbb, Kmm, avm_k,', '&                      sh2    )     ! ==>> out : shear production', 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83': ['DO jk = 2, jpkm1'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:97-109': ['ELSE', ('END DO   ;   END DO', 2), 13],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:99-103': ['zsh2u(ji,jj) = ( p_avm(ji+1,jj,jk) + p_avm(ji,jj,jk) ) &', '&         * wumask(ji,jj,jk)', 5],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:102': ['/ ( (e3w_1d(jk  ) *(1._wp+r3u(ji,jj,Kmm))) * (e3w_1d(jk) *(1._wp+r3u(ji,jj,Kbb))) )'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:104-108': ['zsh2v(ji,jj) = ( p_avm(ji,jj+1,jk) + p_avm(ji,jj,jk) ) &', '&         * wvmask(ji,jj,jk)', 5],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:107': ['/ ( (e3w_1d(jk  ) *(1._wp+r3v(ji,jj,Kmm))) * (e3w_1d(jk) *(1._wp+r3v(ji,jj,Kbb))) )'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:112-113': ['p_sh2(ji,jj,jk) = 0.25 * (', '&                       + ( zsh2v(ji,jj-1) + zsh2v(ji,jj) ) * ( 2. - vmask(ji,jj-1,jk) * vmask(ji,jj,jk) )   )', 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:116-119': ['set p_sh2 to 0 at the surface and bottom for output purpose', ('END DO   ;   END DO', 4), 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domqco.f90:266-267': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2), ('&                    + e1e2t(ji+1,jj) * pssh(ji+1,jj)  ) * r1_hu_0(ji,jj) * r1_e1e2u(ji,jj)', 2), 2],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/domhgr.f90:170': ['r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)'],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/dommsk.f90:237-242': ['wumask(:,:,1) = umask(:,:,1)', 'wvmask(:,:,jk) = vmask(:,:,jk) * vmask(:,:,jk-1)', 6],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:140-150': [('SELECT CASE( kstg )', 1), 'ssha(:,:) = ssh (:,:,Kaa)', 11],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:150-192': ['ssha(:,:) = ssh (:,:,Kaa)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:666-674': [('SELECT CASE( kstg )', 3), 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )', 9],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:283-291': [('IF( lwp .AND. kt == nit000 .AND. PRESENT(k_ind) ) THEN', 1), 'CLOSE(l99_unit)', 9],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:305-310': [("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2), ('+ r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2), 6],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/traadv.f90:266-280': ['IF( ll_Fw ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww', 15],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/domqco.f90:256-258': [('DO jj = ntsj-( nn_hls)', 4), ('END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/dynzdf.f90:166-169': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kbb) + rDt * pvv(ji,jj,jk,Krhs) )', 4],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/oce.f90:99-104': [('ALLOCATE( uu', 1), '&      rhd  (jpi,jpj,jpk)', 6],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:258-260': [('INTEGER  ::   ji, jj, jk', 3), ('REAL(wp), DIMENSION(ntsi-(1):ntei+(1)', 1), 3],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/mppini.f90:1501-1516': ['Nis0 =   1+nn_hls', 'ntej = Nje0', 16],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:283-291': [('IF( lwp .AND. kt == nit000 .AND. PRESENT(k_ind) ) THEN', 1), 'WRITE(l98_unit) e3t_3d, r1_Dt', 9],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:305-317': [("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2), 'CLOSE(l98_unit)', 13],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:140-150': [('SELECT CASE( kstg )', 1), 'ssha(:,:) = ssh (:,:,Kaa)', 11],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:150-192': ['ssha(:,:) = ssh (:,:,Kaa)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:198-202': ['CASE ( 2 )           !==  Stage 2', ('r1_Dt = 1._wp / rDt', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:242-246': ['CASE ( 3 )           !==  Stage 3', ('r1_Dt = 1._wp / rDt', 3), 5],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domqco.f90:256-258': [('DO jj = ntsj-( nn_hls)', 4), ('END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domain.f90:198-212': ['ht_0(:,:) = ht_0(:,:) + e3t_3d(:,:,jk) * tmask(:,:,jk)', 'r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:)', 15],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90:132-139': ['CASE ( np_transport )', ('END DO   ;   END DO   ;   END DO', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90:152-154': [('DO jk =  1,  jpkm1', 4), ('END DO   ;   END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/traadv.f90:266-280': ['IF( ll_Fw ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww', 15],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 36],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:202-213': ['!*  vertical averaging  *!', ('END DO   ;   END DO', 3), 12],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90:188-202': ['!  RK3 : single first external mode computation', 'IF( kstp == nit000 )   CALL l1_dump_stage( kstp, 1, Naa )', 15],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:326-347': ['!- vertical velocity and transport (ww,wi,zFw) -!', ('END DO   ;   END DO   ;   END DO', 3), 22],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:367-373': [('SELECT CASE( kstg )', 2), 'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv(', 7],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:668-674': [('CASE ( 1 , 2 )    !==  Stage 1 & 2', 1), 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )', 7],
    'GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:734-759': ['!==  All stages: correct the barotropic component ==!', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 26],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-201': ['IF( nn_pdl == 1 ) ALLOCATE( z_pdlr', 'IF( nn_pdl == 1 ) DEALLOCATE( z_pdlr )', 12],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:589,612-620': ['IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )', ('IF( ln_mxl0 ) THEN', 1), ('END DO', 30), 10],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:627-630': [('DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)', 10), ('END DO   ;   END DO', 14), 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:634-683': ['zmxld(:,1) = zmxlm(:,1)   ! surface set to the minimum value', ('END DO   ;   END DO', 21), 50],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:669-683': ['CASE ( 3 )           ! lup and ldown, |dk[xml]| bounded by e3t :', ('END DO   ;   END DO', 21), 15],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:690-695': ['DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)   !* vertical eddy viscosity & diffivity at w-points', 'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:699-702': ['IF( nn_pdl == 1 ) THEN          !* Prandtl number case: update avt', ('END DO   ;   END DO', 23), 4],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:691': ('zsqen = SQRT( en(ji,jj,jk) )', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:692': ('zav   = rn_ediff * zmxlm(ji,jk) * zsqen', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:693': ('p_avm(ji,jj,jk) = MAX( zav,                  avmb(jk) )', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:694': ('p_avt(ji,jj,jk) = MAX( zav, avtb_2d(ji,jj) * avtb(jk) )', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:695': ('dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:674-676': ['DO jk =  jpkm1,  2,  -1  ; DO ji = ntsi-( 0), ntei+(  0)   ! from the bottom to the surface : ldown', '&    MIN( zmxlm(ji,jk+1) + (e3t_3d(ji,jj,jk+1)', 3],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-413,699-702': ['IF( nn_pdl == 1 ) THEN          !* Prandtl number = F( Ri )', ('END DO   ;   END DO', 4), 'IF( nn_pdl == 1 ) THEN          !* Prandtl number case: update avt', ('END DO   ;   END DO', 23), 24],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:154-168': ['! Ocean physics update', 'CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 15],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:331-339': [('IF( ln_dynadv_vec ) THEN', 1), ('ENDIF', 8), 9],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:132-139': ['CASE ( np_transport )', ('END DO   ;   END DO   ;   END DO', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/divhor.f90:152-154': [('DO jk =  1,  jpkm1', 4), ('END DO   ;   END DO   ;   END DO', 4), 3],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/sshwzv.f90:293-300': [("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2), ('ENDIF', 11), 8],
    'GYRE_OMIP_L2_P3_SM_R21W/BLD/ppsrc/nemo/traadv.f90:267-295': ['IF( ll_Fw ) THEN', ('ENDIF', 7), 29],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:668-687': [('SELECT CASE( kstg )', 3), ('ENDIF', 21), 20],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:728-765': ['DYN time integration + ZDF', "CALL r46_state( 'post_baro', uu, vv, Kaa )", 38],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/dynspg_ts.f90:763-804': ['!* Sum over whole bt loop (except in weight average)', 'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 42],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/domain.f90:195-215': ['hv_0(:,:) = 0._wp', 'r1_hf_0(:,:) = ssfmask(:,:) / ( hf_0(:,:) + 1._wp -  ssfmask(:,:) )', 21],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90:35-51': ['SUBROUTINE put2(name, value)', 'END SUBROUTINE put3', 17],
    'GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90:69-103': ['SUBROUTINE r90_baro_begin(kt,kstg,Kaa,puu,pvv,pzub,pzvb)', 'END SUBROUTINE r90_baro_finish', 35],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:140-148': [('SELECT CASE( kstg )', 1), ('r1_Dt = 1._wp / rDt', 1), 9],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:140-170': [('SELECT CASE( kstg )', 1), ('END SELECT', 1), 31],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:198-202': ['CASE ( 2 )           !==  Stage 2', ('r1_Dt = 1._wp / rDt', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:242-246': ['CASE ( 3 )           !==  Stage 3', ('r1_Dt = 1._wp / rDt', 3), 5],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:433-482': ['CALL    eos    (        ts, Kmm, rhd, rhop )', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 50],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:692-714': [('CASE ( 3 )        !==  Stage 3', 1), "CALL r46_rhs( 'after_ldf', uu, vv, Krhs )", 23],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:728-759': ["CALL r46_rhs( 'pre_zdf_rhs', uu, vv, Krhs )", 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 32],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzdf.f90:163-170': ['RHS : time-stepping of all trends but the implicit one', ('END DO   ;   END DO', 1), 8],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:188-201': ['!  RK3 : single first external mode computation', 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226': ['IF ( .NOT. l_perpetual_ts ) THEN', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 7],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:671-674': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN   !* applied on velocity', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )', 4],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:176-184': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn'", "IF( PRESENT(Kaa) )   CALL iom_rstput( kt, nitrst, numrow, 'ssha'", 9],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:354-370': ['!*  RK3: Read ssh at Kbb', 'ssh(:,:,Kaa) = ssh(:,:,Kbb)               ! no ssh variation in ww computation', 17],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/restart.f90:395-405': ['ELSE                                  !* no wet and dry', 'ssh(:,:,Kaa) = ssh(:,:,Kbb)           !*  set ssh at Kaa (for AGRIF)', 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:301-311': ['!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', 'DEALLOCATE( sshe_rhs , Ue_rhs , Ve_rhs , CdU_u , CdU_v )', 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/divhor.f90:123-153': ['SELECT CASE ( ik_ind )', ('pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk)', 2), 31],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:326-339': ['!              !- vertical velocity and transport (ww,wi,zFw) -!', ('ENDIF', 8), 14],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynhpg.f90:378-435': [('zcoef0 = - grav * 0.5_wp', 2), ('END DO', 5), 58],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf.f90:81-90': ['SELECT CASE ( nldf_dyn )', 'CALL dynldf_lev_blp( kt, Kbb, Kmm, puu, pvv, Krhs )', 10],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [('DO jj = ntsj-( 0), ntej+(  0+1 )', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138': ['zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp', ('END DO   ;   END DO', 2), 37],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300': [("ELSE                                            !==  Quasi-Eulerian vertical coordinate  ==!   ('key_qco')", 2), ('ENDIF', 11), 8],
    'round64/oracle_krhs_split/ocean.output:682-708': ['ldf_dyn : lateral momentum physics', 'eddy viscosity. = constant', 27],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 36],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:187-199': ['! Lane-2 round-16 slow-forcing operand instrument.  WRITE-only:', 'WRITE(l2_slow_unit) e3u_3d, uu(:,:,:,Krhs), umask, e3v_3d, vv(:,:,:,Krhs), vmask', 13],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:202-207': ['!*  vertical averaging  *!', 'Ve_rhs(ji,jj) = SUM( e3v_3d', 6],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:225-227': ['!* baroclinic drag forcing *!', 'WRITE(l2_slow_unit) Ue_rhs, Ve_rhs, CdU_u, CdU_v', 3],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:229-235': ['!* wind forcing *!', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV', 7],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:291-327': ['! set values computed in RK3_ssh', ('END DO   ;   END DO', 1), 37],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1467-1526': ['SUBROUTINE dyn_drg_init( Kbb, Kmm, puu, pvv, puu_b ,pvv_b, pu_RHSi, pv_RHSi, pCdU_u, pCdU_v )', ('pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0', 1), 60],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-235': ['ELSE                                            !==  standard Cd  ==!', 'pCdU(ji,jj) = - pCd0(ji,jj) * SQRT', 7],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1497-1500': ['ELSE                          ! bottom friction only', 'pCdU_v(ji,jj) = r1_2*( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) )', 4],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:363-375': ['hu_e  (:,:) =    (hu_0(:,:) *(1._wp+r3u(:,:,Kmm)))', 'hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kbb)))', 13],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:703-706': ['IF ( .NOT. ll_wd ) THEN ! Revert to explicit for bit comparison tests in non wad runs', 'zv_trd(ji,jj) = zv_trd(ji,jj) + zCdU_v(ji,jj) * vn_e(ji,jj) * hvr_e(ji,jj)', 4],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:202-207': ['!                             !*  vertical averaging  *!', 'Ve_rhs(ji,jj) = SUM( e3v_3d', 6],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:225-227': ['!                             !* baroclinic drag forcing *!', 'IF( lwp .AND. kt == nit000 )   WRITE(l2_slow_unit) Ue_rhs, Ve_rhs, CdU_u, CdU_v', 3],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/stp2d.f90:229-235': ['!                             !* wind forcing *!', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV', 7],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:288-324': ['!                          ! set values computed in RK3_ssh', ('END DO   ;   END DO', 1), 37],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-505': ['IF ((jn<3).AND.ll_init) THEN', ('END DO   ;   END DO', 3), 25],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:512-518': ['IF( .NOT.lk_linssh ) THEN                        !* Update ocean depth', 'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)', 7],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:549-583': ['!                    !==  after SSH  ==!   (jn+1)', ('END DO   ;   END DO', 9), 35],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:627-642': ['! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2', ('END DO   ;   END DO', 14), 16],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:644-666': ['! Add Coriolis trend:', ('ENDIF', 29), 23],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:678-702': ['! Set next velocities:', ('END DO   ;   END DO', 17), 25],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:744-795': ['IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', 'sshn_e (:,:) = ssha_e(:,:)', 52],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:127-214': ['ALLOCATE( sshe_rhs(jpi,jpj)', ('END SELECT', 2), 88],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:220-311': [('!              !=====================================!', 1), 'DEALLOCATE( sshe_rhs , Ue_rhs , Ve_rhs , CdU_u , CdU_v )', 92],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378': ['IF( ll_init )THEN', 'pssh  (:,:,Kaa) = 0._wp', 40],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-509': ['IF ((jn<3).AND.ll_init) THEN', '& ubb_e(ntsi:ntei,ntsj:ntej), ua_e(ntsi:ntei,ntsj:ntej)', 29],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:783-795': ['!* Swap', 'sshn_e (:,:) = ssha_e(:,:)', 13],
    'GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:991-1018': ["IF( TRIM(cdrw) == 'READ' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'", 28],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:442-452': ['! Round-73 WRITE-only extension of the existing transport-mean stream.', ('WRITE(l2_adv_unit) r1_wgt2s, wgtbtp2(1:icycle), r1_e2u, r1_e1v', 2), 11],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:454-462': ['! Round-76 WRITE-only source-order record for the kt=2 U midpoint.', '& ntsi, ntei, ntsj, ntej', 9],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:501-509': ['ua_e(ji,jj) = za1 * un_e(ji,jj) + za2 * ub_e(ji,jj) + za3 * ubb_e(ji,jj)', '& ubb_e(ntsi:ntei,ntsj:ntej), ua_e(ntsi:ntei,ntsj:ntej)', 9],
    'GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:588-606': ['! Sum over sub-time-steps to compute advective velocities', 'WRITE(l2_adv_unit) zhup2_e, zhvp2_e, un_adv, vn_adv', 19],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:287-291': [('CASE ( np_LIN, np_HYB )', 2), ('END DO   ;   END DO', 2), 5],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:295': ('zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:296': ('zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:559-572': ['! Sum over sub-time-steps to compute advective velocities', '& l2_u_mid, l2_v_mid, zhup2_e, zhvp2_e, un_adv, vn_adv', 14],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:797-817': ['! Finalize sums:', "WRITE(numout,*) 'LANE2_BT_ADVMEAN_DUMP '", 21],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:1029-1045': ['IF( ln_bt_auto )   nn_e = CEILING( rn_Dt / rn_bt_cmax * zcmax)', 'zcmax = zcmax * rDt_e', 17],
    'GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:1064-1067': ["WRITE(numout,*) '     Barotropic time steps => in seconds", "ELSE                  ; WRITE(numout,*) '        set      (ln_bt_auto=F) with the namelist parameter nn_e '", 4],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:390-393': [('IF( lwp .AND. kt == nit000 ) THEN', 2), '&      l2_u_cor(jpi,jpj), l2_v_cor(jpi,jpj) )', 4],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:442-450': ['IF( lwp .AND. kt == nit000 + 1 ) THEN', ('WRITE(l2_adv_unit) r1_wgt2s, wgtbtp2(1:icycle), r1_e2u, r1_e1v', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:503-506': [('IF( lwp .AND. kt == nit000 ) THEN', 3), 'l2_v_mid  (:,:) = va_e(:,:)', 4],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:572-579': ['za2 = wgtbtp2(jn)', 'vn_adv(ji,jj) = vn_adv(ji,jj) + za2 * zhV(ji,jj) * r1_e1v(ji,jj)', 8],
    'GYRE_OMIP_L2_P3_SM_R74ADV2/BLD/ppsrc/nemo/dynspg_ts.f90:581-584': [('IF( lwp .AND. kt <= nit000 + 1 ) THEN', 2), '& l2_u_mid, l2_v_mid, zhup2_e, zhvp2_e, un_adv, vn_adv', 4],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:484-493': ['!* Extrapolate barotropic velocities at mid-step (jn+1/2)', ('END DO   ;   END DO', 3), 10],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:538-544': ['! resulting flux at mid-step (not over the full domain)', ('END DO   ;   END DO', 8), 7],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:571-580': ['! Sum over sub-time-steps to compute advective velocities', ('END DO   ;   END DO', 10), 10],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:289-290': ['zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0', 'zvb(ji,jj) = vn_adv(ji,jj)*(r1_hv_0', 2],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:295': ('zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:296': ('zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d', 1),
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:843-853': [('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) THEN', 1), ('WRITE(l1_unit) zFu, zFv, zFw', 3), 11],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:860,867': ['CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs', 'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs', 2],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:864-872': [('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) &', 1), ('WRITE(l1_unit) ts(:,:,:,jp_tem,Krhs), ts(:,:,:,jp_sal,Krhs)', 3), 9],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:899-901': ['ts(ji,jj,jk,jn,Kaa) = (', '/          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:907-913': [('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) THEN', 2), "WRITE(numout,*) 'L2_RK_TRACER_OPERAND_DUMP '", 7],
    'GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:299-321': ['! Round-13 WRITE-only stage-1 transport operand record.', "WRITE(numout,*) 'L2_RK_STAGE1_TRANSPORT_OPERAND_DUMP '", 23],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:204-215': [('Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs', 1), 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 12],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:899-901': ['ts(ji,jj,jk,jn,Kaa) = (', '/          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965': [('CASE ( 3 )        !==  Stage 3', 2), 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 49],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:416-443': ["IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN", 'zwt(ji,jk) = avs(ji,jj,jk) + ah_wslp2(ji,jj,jk)', 28],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:463-479': ['! Diagonal, lower (i), upper (s)', ('END DO   ;   END DO', 10), 17],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565': [('DO ji = ntsi-( 0), ntei+( 0 )', 2), 'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk)', 19],
    'ocean_model_latlon_cgrid.py:11833-11974': ['T=state_new.T.replace(data=_adv_content_T)', ('z_coord=z_coord, config=config, iwm_fields=iwm_fields)', 2), 142],
    'ocean_model_latlon_cgrid.py:14469-14486': ['K_v_cell = K_v_cell.astype(state.T.data.dtype)', 'K_v_cell = K_v_cell + K33_iso.astype(state.T.data.dtype)', 18],
    'ocean_model_latlon_cgrid.py:15132-15168': [('if do_tracers:', 2), 'rounded_thickness=_zdf_tracer_rounded))', 37],
    'ocean_model_latlon_cgrid.py:1895-1898': ['# Route the already-computed GM/Redi rate into the same stage-3 source', 'route_gm_redi_stage3_source: bool = False', 4],
    'ocean_model_latlon_cgrid.py:11147-11172': [('elif _tti == "rk3_ws":', 2), '_stage_source_rates[2][1] + dS_gm * active_3d,', 26],
    'nemo_testcase_recipe.py:811-911': ['return LatLonCGridOceanConfig.from_flat(', ('gm_redi=None,', 1), 101],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868': [('DO jn = 1, jpts', 1), "CALL r63_snapshot( 'after_sbc', ts, Krhs )", 42],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950': [('DO jn = 1, jpts', 1), 'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )', 124],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:548-562': ['pt(ji,jj,1,jn,Kaa) =', ('+ p2dt * (e3t_3d', 2), 15],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:183-192': [('zdit(ji,jj,ik  ) =', 1), ('zdkt(ji,jj,ikp1) =', 1), 10],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:230-246': [('zA11 = e2_e1u(ji,jj)', 1), ('+ ( zdkt(ji,jj+1,ikp1)', 1), 17],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traldf_iso.f90:287-305': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1), ('END DO   ;   END DO', 8), 19],
    'round64/oracle_krhs_split/ocean.output:649-656': [('no explicit diffusion', 1), 'iso-neutral triad operator', 8],
    'ocean_model_latlon_cgrid.py:2934-3047': ['h_one_third = h_k_old', '+ dt * h_one_half * stage_source_rates[2][1])', 114],
    'ocean_model_latlon_cgrid.py:9113-9386': ['_stage3_T_rate = (', '/ jnp.maximum(_h_s3_kmm, 1.0e-10),', 274],
    'ocean_model_latlon_cgrid.py:10450-10975': ['T_mid = state_new.T.data', 'S_mid = S_mid + dt * dS_gm * active_3d', 526],
    'ocean_model_latlon_cgrid.py:11315-11364': [('_nemo_ws_rk3_tracer_pair_step(', 3), 'return_final_content=True,', 50],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-828': [('DO jn = 1, jpts', 1), 'ts(:,:,:,jn,Krhs) = 0._wp', 2],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:860': ('CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs', 1),
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:602-611': [('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0)', 9), ('pt_up1(ji,jj,jk) = ( (e3t_3d', 3), 10],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:607': ('pt_rhs(ji,jj,jk) = pt_rhs(ji,jj,jk) + ztra', 2),
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-510': ['DO jk =  1,  jpkm1  ; DO jj = ntsj-(  2)', 'ptFw(ji,jj,jk) = MAX( pW(ji,jj,jk)', 8],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:532-540': [('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  1)', 5), ('pt_up1(ji,jj,jk) = ( (e3t_3d', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:570-580': [('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  1)', 7), '&                                       +   MIN( pW(ji,jj,jk)', 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:197-201,264-269': [('CASE(  2  )                   !- 2nd order centered', 1), ('END DO   ;   END DO   ;   END DO', 4), ('CASE(  2  )                   !- 2nd order centered', 2), ('END DO   ;   END DO   ;   END DO', 5), 11],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:798-933': ['z1_Dt = 1._wp / p2dt', ('pcc(ji,jj,jk) = pcc(ji,jj,jk) * zcoef', 2), 136],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:318-329': ['CALL nonosc( Kaa, pt(:,:,:,jn,Kbb)', 'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra', 12],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-506': ['DO jk =  1,  jpkm1  ; DO jj = ntsj-(  2)', ('END DO   ;   END DO   ;   END DO', 21), 4],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90:428-431': [('en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt', 2), (') * wmask(ji,jj,jk)', 2), 4],
    'GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/zdftke.f90:430': ('+ zfact3 * dissl(ji,jj,jk) * en(ji,jj,jk)', 1),
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:167-176': ['DO jn = 1, kjpt', 'CALL fct_up1_2stp( Kbb, Kmm, Kaa', 10],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:503-533': ['zDt = 0.5_wp * pDt', 'CALL r111_first_flux( ptFu, ptFv, ptFw )', 31],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:536-549': ['! -- middle time step tracer with upstream scheme', 'CALL r111_midpoint( zr111, pt_up1 )', 14],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:572-608': ['! *** 2nd step', 'CALL r111_average_flux( ptFu, ptFv, ptFw )', 37],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:611-623': [('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0)', 9), 'CALL r111_upstream( zr111, pt_rhs )', 13],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:322-333': ['CALL nonosc( Kaa, pt(:,:,:,jn,Kbb)', 'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra', 12],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:295-296': ['zFu(ji,jj,jk) = e2u(ji,jj)', 'zFv(ji,jj,jk) = e1v(ji,jj)', 2],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:281-292': [('SELECT CASE( n_baro_upd )', 4), ('END SELECT', 5), 12],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:52-54': ['INTEGER,  PUBLIC, PARAMETER ::   np_HYB = 2', 'INTEGER  :: n_baro_upd =  np_HYB', 3],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:198-242': ['CASE ( 2 )           !==  Stage 2  ==!', 'CASE ( 3 )           !==  Stage 3  ==!', 45],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:525-547': ["WRITE(l3_unit) 'uu_Kmm          '", ('WRITE(l3_unit) zl3_tmp', 3), 23],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domqco.f90:266-268': [('pr3u(ji,jj) = 0.5_wp *', 2), ('pr3v(ji,jj) = 0.5_wp *', 2), 3],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domain.f90:193-200': ['ht_0(:,:) = 0._wp  ! Reference ocean thickness', 'hv_0(:,:) = hv_0(:,:) + e3v_3d(:,:,jk) * vmask(:,:,jk)', 8],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domain.f90:212-215': ['r1_ht_0(:,:) = ssmask (:,:) /', 'r1_hf_0(:,:) = ssfmask(:,:) /', 4],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domhgr.f90:155-160': ['e1e2t (:,:) = e1t(:,:) * e2t(:,:)', 'e1e2v (:,:) = e1v(:,:) * e2v(:,:)', 6],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/domhgr.f90:170-171': ['r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)', 'r1_e1e2v(:,:) = 1._wp / e1e2v(:,:)', 2],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3.f90:188-201': ['!  RK3 : single first external mode computation', 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:140-180': [('SELECT CASE( kstg )', 1), 'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 41],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:190-192': ['r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 3],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:230-237': [('ssh (:,:,Kaa) = r1_2 * ( ssh (:,:,Kbb) + ssha(:,:) )', 2), ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:242': ('CASE ( 3 )           !==  Stage 3  ==!', 1),
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:301-308': ['!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:202-207': ['!*  vertical averaging  *!', 'Ve_rhs(ji,jj) = SUM( e3v_3d', 6],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:225-239': ['!* baroclinic drag forcing *!', 'CLOSE(l2_slow_unit)', 15],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:326-346': ['!              !- vertical velocity and transport (ww,wi,zFw) -!', 'zFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 21],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:860': ('CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:21': ('nn_itend    =      10', 1),
    'GYRE_OMIP_L2_P3_SM_R63KRHS/EXP00/namelist_cfg:21': ('nn_itend    =      2', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:825-863': [('DO jn = 1, jpts', 1), 'CALL tra_sbc_RK3( kstp, Kbb, Kmm', 39],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:168-201': ['CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 'CALL stp_RK3_stg( 1, kstp', 34],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-868': [('DO jn = 1, jpts', 1), "CALL r63_snapshot( 'after_sbc'", 42],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:169-172': ['! -- Upstream fluxes', 'output => ztFu(1,0,1,0)', 4],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/traadv_fct.f90:170-174': ['! -- Upstream fluxes', 'output => ztFu(1,0,1,0)', 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:145-149': ['zl2_trh(:,:,:) = pts(:,:,:,jp_tem,Krhs)', 'CALL tra_zdf_imp(', 5],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/trazdf.f90:146-151': ['zl2_trh(:,:,:) = pts(:,:,:,jp_tem,Krhs)', 'CALL r63_content_finish(', 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:188-192': ['IF( nn_pdl == 1 ) ALLOCATE( z_pdlr', 'CALL tke_avn(', 5],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90:190-196': ['IF( nn_pdl == 1 ) ALLOCATE( z_pdlr', 'CALL tke_avn(', 7],
    'GYRE_OMIP_L2_P3_SM_R63KRHS/BLD/ppsrc/nemo/zdftke.f90:413-433': ['CALL r63_tke_before_row(', 'CALL r63_tke_after_row(', 21],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:773-800': [('IF( .NOT.ln_shuman ) THEN', 1), ('CALL tra_adv_trp( kstp, kstg, nit000', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:825-870': [('DO jn = 1, jpts', 1), 'IF( ln_isf )   CALL tra_isf', 46],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:886-908': [('SELECT CASE( kstg )', 4), 'L2_RK_TRACER_OPERAND_DUMP', 23],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:912-958': [('CASE ( 3 )', 3), 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs', 47],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:94-111': ['INTEGER                                  , INTENT(in   ) ::   kt', ('REAL(wp), DIMENSION(ntsi-(nn_hls):ntei+(nn_hls)', 2), 18],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:164-198': ['DO jn = 1, kjpt', 'ztFv(ji,jj,jk) = 0.5_wp * pV', 35],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv_fct.f90:260-327': ['SELECT CASE( kn_fct_v )', 'pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) + ztra', 68],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:86-88': ['INTEGER                                  , INTENT(in)    :: kt', 'REAL(wp), DIMENSION(jpi,jpj,jpk,jpts,jpt)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:219-223': ['INTEGER                                   , INTENT(in   ) ::   Kbb, Kmm', ('REAL(wp), DIMENSION(ntsi-(0):ntei+(0)', 1), 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdftke.f90:409-427': ['Matrix and right hand side in en', ('END DO   ;   END DO', 5), 19],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/zdf_oce.f90:85-87': [('ALLOCATE( avm', 1), 'avmb(jpk)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/oce.f90:40-42': ['DIMENSION(:,:,:,:,:) ::   ts', 'DIMENSION(:,:,:)     ::   rn2b ,  rn2', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dom_oce.f90:170-180': ['::     e3t_3d', '::   r3t, r3u, r3v', 11],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/inc/do_loop_substitute.h90:72-89': ['!    A1Di(H) = Nis0-(H):Nie0+(H)', ('#define T2D(H) T1Di(H),T1Dj(H)', 1), 18],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:450': ('zwt(:,1) = 0._wp', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:461-477': ['Diagonal, lower (i), upper (s)', ('END DO   ;   END DO', 10), 17],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:523-528': ['DO ji = ntsi-( 0), ntei+( 0 )          !* 1st recurrence', ('END DO   ;   END DO', 13), 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:545-564': ['DO ji = ntsi-( 0), ntei+( 0 )             !* 2nd recurrence', ('END DO   ;   END DO', 16), 20],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:573-578': ['DO ji = ntsi-( 0), ntei+( 0 )             !* 3d recurrence', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 6],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168': ('CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 1),
    'nemo_testcase_recipe.py:642-644': ['tke_shear_production="nemo_face_native_now2"', ('tke_shear_metric_source="nemo_qco_live_face"', 2), 3],
    'ocean_model_latlon_cgrid.py:13102-13119': ['if shear_disc not in ("nemo_face_native", "nemo_face_native_now2"):', '_u_before, _v_before = _u_now, _v_now', 18],
    'ocean_model_latlon_cgrid.py:13174-13273': ['if metric_source == "nemo_qco_live_face":', 'ref_v * (1.0 + r3vb[..., None]),', 100],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py:397-470': [('wumask = u_mask[..., :-1] * u_mask[..., 1:]', 2), ('+ (zsh2v[:-1, :, :] + zsh2v[1:, :, :]) * coast_v', 2), 74],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:514-552': ['def _model_substitution_walk(', 'require(jax.config.x64_enabled, "model substitution walk requires JAX fp64")', 39],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1238-1425': ['def _bridge_kt2_production_entry(', 'return state, audit', 188],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1174-1208': ['def _bridge_kt2_state(', ('return state', 1), 35],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:4627-4657': ['states = {1: _bridge_stage_context(', ('surface_forcing=surface))', 2), 31],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:546-563': ['observed_step = (header["kt"], header["stage"])', 'f"Kmm={header[\'Kmm\']}",', 18],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:1269-1273': ['step_entry = read_entry(year_entry_path)', 'f"kt=2/Nbb=3, got kt={step_entry[\'kt\']}/Nbb={step_entry[\'Nbb\']}",', 5],
    'nemo_testcase_l2_gyre_round46_kt2_stage_gate.py:4470-4491': ['if production_tke_post_sweep is not None:', 'tke_module._solve_tke_backward_euler = inject_recorded_post_sweep', 22],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:225-239': ['header = struct.unpack("=13i", take(13 * 4))', 'f"Kbb={head[\'Kbb\']}/Kmm={head[\'Kmm\']}",', 15],
    'ocean_model_latlon_cgrid.py:15420-15425': ['if _tke_coeff_new is not None:', 'tke_avt=Field(data=_tke_coeff_new.K_H', 6],
    'ocean_model_latlon_cgrid.py:15665-15680': ['# ``step`` is the production-compiled entry point even when a caller', ('_nemo_stage1_zad_eta_after_override))', 1), 16],
    'state.py:629-633': ['# NEMO TKE-closure coefficient memory (avm_k/avt_k). These are the', 'tke_avt: object = None', 5],
    'tests/ocean/fidelity/test_nemo_testcase_l2_gyre_card_reconciliation.py:193-216': ['def test_rk3_card_steps_with_nemo_face_shear_and_live_face_metric(gate):', 'assert all(np.all(np.isfinite(np.asarray(fields[k]))) for k in ("T", "S", "u", "v", "ssh"))', 24],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:317-337': [('IF( l_zdfsh2 ) THEN', 1), 'CASE( np_TKE )   ;   CALL zdf_tke( kt, Kbb, Kmm, sh2, avm_k, avt_k )', 21],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83-114': ['DO jk = 2, jpkm1', ('END DO   ;   END DO', 3), 32],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:190-200': ['IF( nn_pdl == 1 ) ALLOCATE( z_pdlr', 'CALL r54_tke_finish( p_avm, p_avt, dissl )', 11],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:277-281': [('DO ji = ntsi-( 0), ntei+( 0 )', 1), 'zd_up(ji,1) = 0._wp', 5],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:292-300': ['IF( .NOT.ln_drg_OFF ) THEN', 'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin )', 9],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:318-380': ['IF( ln_lc ) THEN', 'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)', 63],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-433': [('IF( nn_pdl == 1 ) THEN', 1), '&                                  ) * wmask(ji,jj,jk)', 40],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:466-483': ['DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)                 ! First recurrence', 'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:589-701': ['IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )', 'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk)', 113],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:162-190': ['ALLOCATE(r54_zdiag(ntsi:ntei,ntsj:ntej,jpk)', 'END SUBROUTINE r54_tke_matrix_row', 29],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:96-104': ['SUBROUTINE r101_rhs_row(jj,p_en)', 'END SUBROUTINE r101_rhs_row', 9],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/l2_r101_tke_walk.f90:116-128': [('SUBROUTINE r101_tke_finish', 1), '& r101_post_sweep)', 13],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:196-207': ['CALL r101_tke_begin( kt, Kbb, Kmm )', 'IF( nn_pdl == 1 ) DEALLOCATE( z_pdlr )', 12],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:257': ('zbbrau  = rn_ebb / rho0', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:285': ('en(ji,jj,1) = MAX( rn_emin0, zbbrau * taum(ji,jj) )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:299-307': ['IF( .NOT.ln_drg_OFF ) THEN', 'en(ji,jj,mbkt(ji,jj)+1) = MAX( zebot, rn_emin ) * ssmask(ji,jj)', 9],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:324': ('CALL r101_after_boundaries_row( jj, en )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:353': ('zWlc2(ji) = zcsd * taum(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:361-365': ['zpelc(ji,1) =  MAX( rn2b(ji,jj,1), 0._wp )', '&       MAX( rn2b(ji,jj,jk), 0._wp )', 5],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:369-388': ['imlc(:) = mbkt(ntsi-(0):ntei+(0),jj) + 1', 'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)', 20],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:395': ('CALL r101_after_langmuir_row( jj, en )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:399-443': ['! Resolution of a tridiagonal linear system by a "methode de chasse"', ('END DO   ;   END DO', 5), 45],
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:473': ('CALL r101_rhs_row( jj, en )', 1),
    'GYRE_OMIP_L2_P3_SM_R101TKEW/BLD/ppsrc/nemo/zdftke.f90:475-495': ['!* Matrix inversion from level 2 (tke prescribed at level 1)', 'CALL r101_post_sweep_row( jj, en )', 21],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354': ['start from turbulent closure values', 'CALL r54_zdfphy_finish', 7],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:277-281': [('DO ji = ntsi-( 0), ntei+( 0 )', 1), 'zd_up(ji,1) = 0._wp', 5],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:318-380': ['IF( ln_lc ) THEN', 'en(ji,jj,jk) = en(ji,jj,jk) + rn_Dt * zus3(ji)', 63],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:416-433': ['DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)    !* Matrix and right hand side in en', '&                                  ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:466-483': ['DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)                 ! First recurrence', 'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:589-682': ['IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )', 'zmxld(ji,jk) = zemlp', 94],
    'round33_lock_zdf_matrix/namelist_cfg:131': ('ln_zdfcst   = .true.', 1),
    'round33_overflow_zdf_matrix/namelist_cfg:128': ('ln_zdfcst   = .true.', 1),
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:394-412': [('IF( nn_pdl == 1 )', 3), 'p_pdlr(ji,jj,jk) = MAX(  0.1_wp,  ri_cri / MAX( ri_cri , zri )  )', 19],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:691-701': ['zsqen = SQRT', 'p_avt(ji,jj,jk)   = MAX', 11],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354': ['start from turbulent closure values', 'CALL r54_zdfphy_finish', 7],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:466-483': ['DO jk =  2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)                 ! First recurrence', 'en(ji,jj,jk) = MAX( en(ji,jj,jk), rn_emin ) * wmask(ji,jj,jk)', 18],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:589': ('IF( ln_mxl0 ) zraug = vkarmn * 2.e5_wp / ( rho0 * grav )', 1),
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:601-603': ['initialisation of interior minimum value', 'zmxld(:,:)  = rmxl_min', 3],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:612-619': [('IF( ln_mxl0 ) THEN', 1), 'zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )', 8],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:628-629': ['zrn2 = MAX( rn2(ji,jj,jk), rsmall )', 'zmxlm(ji,jk) = MAX(  rmxl_min,  SQRT( 2._wp * en(ji,jj,jk) / zrn2 )  )', 2],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:634': ('zmxld(:,1) = zmxlm(:,1)', 1),
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdftke.f90:669-682': ['CASE ( 3 )', 'zmxld(ji,jk) = zemlp', 14],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/zdf_oce.f90:85-87': [('ALLOCATE( avm', 1), 'avmb(jpk)', 3],
    'GYRE_OMIP_L2_P3_SM_R58TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:173-189': [('SUBROUTINE r54_tke_matrix_row', 1), 'r54_pdlr(:,jj,2:jpkm1)', 17],
    'nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90:154-170': [('SUBROUTINE r54_tke_matrix_row', 1), 'r54_pdlr(:,jj,2:jpkm1)', 17],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:116-199': ['def _calibrate_en_and_mixing', ('return counts', 2), 84],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:106-110': [('SUBROUTINE r54_tke_begin', 1), 'REAL(wp), DIMENSION(:,:,:), INTENT(in) :: p_sh2,p_avm,p_avt,p_dissl', 5],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:150-155': ["CALL put3('avm_entry", "CALL put3('sh2", 6],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:128': ('ALLOCATE( htau', 1),
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:183-185': ['REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ,jpk), INTENT(in   ) ::   p_sh2', 'REAL(wp), DIMENSION(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0) ,jpk), INTENT(inout) ::   p_avt', 3],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:394-412': [('IF( nn_pdl == 1 )', 3), 'p_pdlr(ji,jj,jk) = MAX(  0.1_wp,  ri_cri / MAX( ri_cri , zri )  )', 19],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:751': ('ri_cri   = 2._wp    / ( 2._wp + rn_ediss / rn_ediff )', 1),
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:691-701': ['zsqen = SQRT', 'p_avt(ji,jj,jk)   = MAX', 11],
    'GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354': ['start from turbulent closure values', 'CALL r54_zdfphy_finish', 7],
    'nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90:91-95': ['Match the executing dummies exactly', ('REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT(in) :: p_avm', 1), 5],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:47-113': ['def _calibrate_closure', ('return counts', 1), 67],
    'nemo_testcase_l2_gyre_round54_tke_operands.py:289-295': ['elif plant == "prandtl":', ('np.float64(np.inf))', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:137': ('pssh(ji,jj,Kaa) = (  pssh(ji,jj,Kbb) - rDt * ( r1_rho0 * emp(ji,jj) + zhdiv(ji,jj) )  ) * ssmask(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:553': ('ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/usrdef_sbc.f90:151-157': ["zsumemp = glob_2Dsum( 'usrdef_sbc', emp  (:,:)   )", 'emp (ji,jj) = emp(ji,jj) - zsumemp * tmask(ji,jj,1)', 7],
    'ocean_model_latlon_cgrid.py:10004-10069': ['if _cfg_b.fix_eta_drift:', 'eta=state_new.eta.replace(data=eta_fixed),', 66],
    'ocean_model_latlon_cgrid.py:7725-7726': [('u0 = state.u.data', 2), ('v0 = state.v.data', 2), 2],
    'ocean_model_latlon_cgrid.py:9610-9615': ['_p0_with_zub = _mom_pert_ws(', ('stage_face_thickness=_face_thickness_kbb, stage_index=1)', 2), 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215': ['CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 26],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-146,196-200,236-244': [('SELECT CASE( kstg )', 1), ('r1_Dt = 1._wp / rDt', 1), 'CASE ( 2 )', ('r1_Dt = 1._wp / rDt', 2), ('ENDIF', 5), ('r1_Dt = 1._wp / rDt', 3), 23],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:308-310': ['! set current model timestep rDt = 2*rn_Dt if MLF or rDt = rn_Dt if RK3', 'r1_Dt = 1._wp / rDt', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/sshwzv.f90:277-298': ['IF( .NOT. PRESENT( k_ind ) ) THEN', '&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)', 22],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:215-235': [('CASE ( np_LIN )', 2), ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-489,749-761': ['Set extrapolation coefficients for predictor step:', 'zsshp2_e(:,:) = za1 * sshn_e(:,:)', ('!* Swap', 1), 'sshn_e (:,:) = ssha_e(:,:)', 47],
    'GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:165-167': ['&namdyn_vor', 'ln_dynvor_ene = .true.', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:431-480': ['CALL    eos    (        ts, Kmm, rhd, rhop )', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 50],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:690-712': [('CASE ( 3 )', 2), "CALL r46_rhs( 'after_ldf', uu, vv, Krhs )", 23],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [('DO jj = ntsj-( 0), ntej+(  0+1 )', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynvor.f90:536-573': [('CASE ( np_CRV )', 2), 'pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) - r1_4 * r1_e2v', 38],
    'stprk3_stg.F90:44': ('n_baro_upd =  np_HYB', 1),
    'stprk3_stg.F90:266-268': [('DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )', 2), 'zvb(ji,jj) = vn_adv(ji,jj)*r1_hv(ji,jj,Kmm) - vv_b(ji,jj,Kmm)', 3],
    'stprk3_stg.F90:273-274': ['zFu(ji,jj,jk) = e2u(ji,jj)*e3u(ji,jj,jk,Kmm)', 'zFv(ji,jj,jk) = e1v(ji,jj)*e3v(ji,jj,jk,Kmm)', 2],
    'stprk3_stg.F90:290': ('CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )', 1),
    'stprk3_stg.F90:322': ('CALL    eos    (        ts, Kmm, rhd, rhop )', 1),
    'stprk3_stg.F90:202': ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 1),
    'stprk3_stg.F90:234': ('r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 1),
    'stprk3_stg.F90:231': ('r3t(:,:,Kaa) = r3ta(:,:)                          ! at N+1   (Kaa)', 1),
    'dynhpg.F90:359-363': ['puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap', 'pvv(ji,jj,1,Krhs) = pvv(ji,jj,1,Krhs) + zhpj(ji,jj) + zvap', 5],
    'dynhpg.F90:383-387': ['puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap', 'pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) + zhpj(ji,jj) + zvap', 5],
    'dynhpg.F90:368-375': ['DO_2D( 0, 0, 0, 0 )    ! interior value (2=<jk=<jpkm1)', '&           - e3w(ji,jj  ,jk,Kmm) * ( rhd(ji,jj,  jk) + rhd(ji,jj  ,jk-1) )  )', 8],
    'dynkeg.F90:120-121': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1), ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1), 2],
    'dynzad.F90:104-107': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp * r1_e1e2u(ji,jj) / e3u(ji,jj,jk,Kmm)', 1), '&                                            * ( zWdzV(ji,jj) + zzWdzV )', 4],
    'dynzad.F90:89-96': ['zWf  = e1e2t(ji  ,jj  ) * ( ww(ji  ,jj  ,jk+1) + wsd(ji  ,jj  ,jk+1) )', ('ENDIF', 4), 8],
    'dynvor.F90:493-516': ['SELECT CASE( nn_e3f_typ  )', ('END SELECT', 10), 24],
    'GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/dynvor.f90:556': ('zwz(ji,jj) = zwz(ji,jj) / (e3f_0vor(ji,jj,jk) *(1._wp+r3f(ji,jj)*fe3mask(ji,jj,jk)))', 1),
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynadv.f90:134-138': [('SELECT CASE( n_dynadv )', 1), 'CALL dyn_zad', 5],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/stprk3_stg.f90:326-332': [('IF( ln_dynadv_vec ) THEN', 1), ('IF( ln_zad_Aimp .AND. kstg == 3 )   CALL wAimp', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynkeg.f90:117-130': ['CASE ( nkeg_C2 )', ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) -', 1), 14],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/dynzad.f90:105-137': ['DO jk =  1,  jpk-2', '&                                              * zWdzV(ji,jj)', 33],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sshwzv.f90:271-298': [('DO jj = ntsj-( 1), ntej+(  1 )', 2), ('r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb)', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90:408-423': ['IF( .NOT. ln_wave ) THEN', 'RETURN', 16],
    'GYRE_OMIP_L2_P3_SM_R40STG3TRM/BLD/ppsrc/nemo/sbcwave.f90:469-477': [('IF( ln_sdw ) THEN', 1), 'wsd   (:,:,:) = 0._wp', 9],
    'ocean_pe_latlon_cgrid.py:2119-2153': ['elif config.ke_gradient_scheme == "c2":', 'dp_dy = _dKp_dy[..., 1]', 35],
    'ocean_pe_latlon_cgrid.py:3411-3442': ['area_w = jax.lax.optimization_barrier(', 'diag_vertadv_v = jax.lax.optimization_barrier(diag_vertadv_v)', 32],
    'ocean_pe_latlon_cgrid.py:5643-5682': ['zad_w, zad_h_u, zad_h_v = w, h_u, h_v', 'zad_h_v = jax.lax.optimization_barrier(zad_h_v)', 40],
    'ocean_model_latlon_cgrid.py:8792-8802': ['_freeze_hpg = self._nemo_ws_test_hooks.freeze_stage_hpg_operands', ('getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1), 11],
    'ocean_model_latlon_cgrid.py:10355-10372': [('if (getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1), '/ jnp.maximum(_area_v, 1.0e-30))', 18],
    'ocean_model_latlon_cgrid.py:10383-10405': ['# 7b. Adaptive-implicit vertical momentum advection', 'and _pflow is None):', 23],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:466-472': ['!* advection (VIF or FF)', ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), 7],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:327-333': [('IF( ln_dynadv_vec ) THEN', 1), ('CALL wAimp', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:201-215': ['CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 15],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stp2d.f90:141-166': ['CALL dyn_hpg', 'CALL dyn_zad', 26],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:447-469': ['CALL    dyn_hpg', 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 23],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:704-717': ['CALL dyn_ldf', 'CALL dyn_zdf', 14],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3_stg.f90:724-743': [('barotropic velocity correction', 4), 'corrected horizontal velocity', 20],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90:152-176': [('SELECT CASE( n_dynadv )', 1), 'CALL dyn_zad', 25],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynzad.f90:105-137': ['DO jk =  1,  jpk-2', '&                                              * zWdzV(ji,jj)', 33],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:292-316': [('DO jk =  1,  jpkm1', 1), '&           vn_adv, (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm))), vv_b(:,:,Kmm)', 25],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:350-357': [('IF( lwp .AND. kstp == nit000 ) THEN', 1), ('WRITE(l1_unit) zFu, zFv, zFw', 1), 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:329-335': [('IF( ln_dynadv_vec ) THEN', 1), ('IF( ln_zad_Aimp .AND. kstg == 3 )', 1), 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:105-137': ['DO jk =  1,  jpk-2', '&                                              * zWdzV(ji,jj)', 33],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-176': ['!*  hydrostatic pressure gradient (HPG))', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 36],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:206-207': ['Ue_rhs(ji,jj) = SUM(', 'Ve_rhs(ji,jj) = SUM(', 2],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:365-371': [('SELECT CASE( kstg )', 2), 'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv', 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:664-672': [('SELECT CASE( kstg )', 3), 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )', 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/traadv.f90:248-250': ['pFu(ji,jj,jpk) = 0._wp', 'pFw(ji,jj,jpk) = 0._wp', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175': ['!*  hydrostatic pressure gradient (HPG))', "CALL r46_rhs( 'after_zad', uu, vv, Krhs )", 35],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-145': ['hydrostatic pressure gradient (HPG))', "CALL r46_rhs( 'after_hpg', uu, vv, Krhs )", 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:324-337': ['vertical velocity and transport (ww,wi,zFw)', ('ENDIF', 8), 14],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:472-480': ['advection (VIF or FF)', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:100-130': [('vertical momentum advection', 2), 'END DO   ;   END DO   ;   END DO', 31],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynzad.f90:123-126': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs)', 1), '&                                            * ( zWdzV(ji,jj) + zzWdzV )', 4],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:695-711': ['Round-29 L2 WRITE-only stage-3 pre-LDF momentum frame.', 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 17],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:64-74': ['SUBROUTINE dynldf_lev_lap( kt, Kbb, Kmm, pu, pv, Krhs )', 'pu(Krhs), pv(Krhs) increased by the harmonic operator applied on pu(Kbb), pv(Kbb)', 11],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynldf_lev.f90:123-127': [('zwf(ji-1,jj-1) = ahmf', 1), ('zwt(ji,jj)     = ahmt', 1), 5],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynhpg.f90:400-427': ['!                                   ! add to the general momentum trend', 'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:352-378': ['IF( ln_bt_fw ) THEN                 ! FORWARD integration: start from NOW fields', 'vn_adv(:,:)     = 0._wp', 27],
    'barotropic_latlon_cgrid.py:3223-3224': ['U_bar_corr, V_bar_corr = _depth_average_to_faces(', 'u_corr, v_corr, _h_k_corr, min_water_col, mask, u_mask, v_mask, grid,', 2],
    'eos.py:759': '"raw-mesh e3w_int must contain only finite values > 0",',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:835-842': ['zzsshu = r1_2 * r1_e1e2u(ji,jj) * ( e1e2t(ji  ,jj) * pssh(ji  ,jj,Kaa)', 'pvv_b(ji,jj,Kaa) = pvv_b(ji,jj,Kaa) / ( hv_0(ji,jj) + zzsshv + 1._wp - ssvmask(ji,jj) )', 8],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:276': 'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:767-769': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN    ! Sum velocities', ('pvv_b  (:,:,Kaa) = pvv_b  (:,:,Kaa) + za1 * va_e  (:,:)', 1), 3],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/dynspg_ts.f90:802': 'puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) / r1_wgt1s',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:606': 'zmxlm(ji,1) =  zraug * taum(ji,jj) * tmask(ji,jj,1)',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:610': 'zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:831': ('rn_mxl0 = rmxl_min', 2),
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:828-831': [('IF( ln_mxl0 ) THEN', 4), ('rn_mxl0 = rmxl_min', 2), 4],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/zdftke.f90:810-812': ['IF( ln_zdfiwm ) THEN          ! Internal wave-driven mixing', 'rmxl_min = 1.e-03_wp             ! associated avt minimum = molecular salt diffusivity (10^-9 m2/s)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:824-827': ['IF(.NOT.ll_bt_av ) THEN', 'pssh (:,:,Kaa) = ssha_e(:,:)', 4],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:456-480': ['Set extrapolation coefficients for predictor step:', 'va_e(ji,jj) = za1 * vn_e(ji,jj) + za2 * vb_e(ji,jj) + za3 * vbb_e(ji,jj)', 25],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:749-757': ['!* Swap', 'vn_e   (:,:) = va_e  (:,:)', 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:953-980': ["IF( TRIM(cdrw) == 'READ' ) THEN", "CALL iom_rstput( kt, nitrst, numrow, 'vb_e'", 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:483-489': [('IF( .NOT.lk_linssh ) THEN', 1), 'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)', 7],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:593-600': ['Half-step back interpolation of SSH', '&            + za2 *  sshb_e(ji,jj) + za3 *  sshbb_e(ji,jj)', 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:759-761': ['sshbb_e(:,:) = sshb_e(:,:)', 'sshn_e (:,:) = ssha_e(:,:)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:373-378': ['! Initialize sums:', 'vn_adv(:,:)     = 0._wp', 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:559-567': ['! Sum over sub-time-steps to compute advective velocities', 'vn_adv(ji,jj) = vn_adv(ji,jj) + za2 * zhV(ji,jj) * r1_e1v(ji,jj)', 9],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:797-804': ['! Finalize sums:', 'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/dynspg_ts.f90:858-863': ['END Phase 3 for RK3', "IF( lrst_oce )   CALL ts_rst( kt, 'WRITE' )", 6],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/EXP00/ocean.output:535': ('Courant number targeted application   ln_zad_Aimp =  F', 1),
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynadv.f90:176-185': ['CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )', '&           jpi, jpj, jpk, jpkm1, ntsi, ntei, ntsj, ntej, STORAGE_SIZE(1._wp)', 10],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynkeg.f90:117-130': ['CASE ( nkeg_C2 )', ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) -', 1), 14],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdfmxl.f90:109-123': ['! w-level of the mixing and mixed layers', 'hmlp(ji,jj) = ((gdepw_1d(iik  ) ) *(1._wp+r3t(ji,jj,Kmm))) * ssmask(ji,jj)', 15],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:222-232': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4), 'zbv = MIN(  zbv, - z1_slpmax * ABS( zav )', 11],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:235-275': ['iku = MAX( nmln(ji+1,jj), nmln(ji,jj) )', '* ( vmask(ji  ,jj,jk) + vmask(ji  ,jj,jk+1) ) * 0.5_wp', 41],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/ldfslp.f90:284-333': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 5), '+ 4.*    zww(ji  ,jj  )                        ) * zcofw', 50],
    'GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/traldf_iso.f90:793-794': ['pah_wslp2(ji,jj,jk) = zahu_w * wslpi(ji,jj,jk) * wslpi(ji,jj,jk)', '&                + zahv_w * wslpj(ji,jj,jk) * wslpj(ji,jj,jk)', 2],
    'cpp_GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2.fcm:1': ('bld::tool::fppkeys key_qco key_vco_1d3d key_RK3', 1),
    'domqco.F90:189-209': ['SUBROUTINE dom_qco_r3c_RK3( pssh, pr3t, pr3u, pr3v, pr3f )', ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)   !==  ratio at t-point  ==!', 2), 21],
    'domain.F90:158': ('r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:) + 1._wp -  ssmask (:,:) )', 1),
    'ZDF/zdfmxl.F90:34': ('rho_c = 0.01_wp', 1),
    'ZDF/zdfmxl.F90:95': ('zN2_c = grav * rho_c * r1_rho0', 1),
    'ZDF/zdfmxl.F90:98': ('hmlp(ji,jj) =  hmlp(ji,jj) + MAX( rn2b(ji,jj,jk) , 0._wp ) * e3w(ji,jj,jk,Kmm)', 1),
    'ZDF/zdfmxl.F90:99': ('IF( hmlp(ji,jj) < zN2_c )   nmln(ji,jj) = MIN( jk , ikt ) + 1', 1),
    'ZDF/zdfmxl.F90:104': ('hmlp(ji,jj) = gdepw(ji,jj,iik  ,Kmm) * ssmask(ji,jj)', 1),
    'LDF/ldfslp.F90:143': ('zhmlpt(ji,jj) = gdept(ji,jj,nmln(ji,jj)-1,Kmm) * ssmask(ji,jj)', 1),
    'LDF/ldfslp.F90:161': ('r1_hmlw(ji,jj) = 1._wp / MAX( hmlp(ji,jj) - gdepw(ji,jj,mikt(ji,jj),Kmm), 10._wp )', 1),
    'LDF/ldfslp.F90:206': ('zau = zgru(ji,jj,iik) * r1_e1u(ji,jj)', 1),
    'LDF/ldfslp.F90:212-213': ['zbu = MIN(  zbu, - z1_slpmax * ABS( zau ) , -7.e+3_wp/e3u(ji,jj,jk,Kmm)* ABS( zau )  )', 'zbv = MIN(  zbv, - z1_slpmax * ABS( zav ) , -7.e+3_wp/e3v(ji,jj,jk,Kmm)* ABS( zav )  )', 2],
    'LDF/ldfslp.F90:281-282': ['zbi = MIN( zbw ,- 100._wp* ABS( zai ) , -7.e+3_wp/e3w(ji,jj,jk,Kmm)* ABS( zai )  )', 'zbj = MIN( zbw , -100._wp* ABS( zaj ) , -7.e+3_wp/e3w(ji,jj,jk,Kmm)* ABS( zaj )  )', 2],
    'vertical.py:1980': 'def compute_ocean_jacobian(',
    'ocean_model_latlon_cgrid.py:7804-7808': ['transport_velocity = (', ('* _ws_stage_v_mask,', 1), 5],
    'ocean_model_latlon_cgrid.py:9832': '_g2 = _nemo_ws_stage_transport(',
    'ocean_model_latlon_cgrid.py:9854': '_stage3_hpg_operands = _stage_hpg_operands(',
    'nemo_testcase_l2_gyre_phase3_gate.py:333': 'require(magic == "NEMO_L2_RKTRM_1", f"{path}: bad magic")',
    'round38_oracle_trazdf_kt2/ocean.output:799': ('with Hollingsworth scheme (=1) or not (=0)       nn_dynkeg   =', 1),
    'round38_oracle_trazdf_kt2/ocean.output:810': ('f-point energy conserving scheme               ln_dynvor_ene =  T', 1),
    'round38_oracle_trazdf_kt2/ocean.output:813': ('e3f = averaging /4 (=0) or /sum(tmask) (=1)    nn_e3f_typ =', 1),
    'round38_oracle_trazdf_kt2/ocean.output:834': ('s-coord. (standard jacobian formulation)          ln_hpg_sco    =  T', 1),
    'round38_oracle_trazdf_kt2/ocean.output:846': ('Free surface with time splitting       ln_dynspg_ts  =  T', 1),
    'round38_oracle_trazdf_kt2/ocean.output:658': ('maximum isoppycnal slope             rn_slpmax', 1),
    'stprk3.F90:173': ('CALL eos ( ts, Nbb, rhd )                   ! before in situ density', 1),
    'GYRE_OMIP_L2_P3_SM_R38TRAZDFKT2/BLD/ppsrc/nemo/stprk3.f90:178': ('CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )   ! before slope for standard operator', 1),
    'LDF/ldfslp.F90:80': ('SUBROUTINE ldf_slp( kt, prd, pn2, Kbb, Kmm )', 1),
    'traldf_iso.F90:296-297': ['pah_wslp2(ji,jj,jk) = zahu_w * wslpi(ji,jj,jk) * wslpi(ji,jj,jk)   &', '&                + zahv_w * wslpj(ji,jj,jk) * wslpj(ji,jj,jk)', 2],
    'traldf_iso.F90:291-294': ['zahu_w = (  ( ahtu(ji  ,jj,jk-1) + ahtu(ji-1,jj,jk) )    &', '&      + ( ahtv(ji,jj-1,jk-1) + ahtv(ji,jj  ,jk) )  ) * zmskv', 4],
    'domqco.F90:160': ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0(ji,jj)   !==  ratio at t-point  ==!', 1),
    'fidelity/nemo_recipe.py:1014': 'def build_nemo_gyre_recipe(',
    'ocean_model_latlon_cgrid.py:7045': 'T_new = state.T.data + dt * tend.dT_dt.data',
    'round38_oracle_trazdf_kt2/ocean.output:798': ('Vector form: 2nd order centered scheme           ln_dynadv_vec  =  T', 1),
    'lock_kt1_10/ocean.output:578': ('no explicit diffusion                   ln_traldf_OFF   =  T', 1),
    'lock_kt1_10/ocean.output:584': ('iso-neutral Madec operator              ln_traldf_iso   =  F', 1),
    'overflow_kt1_10/ocean.output:690': ('no explicit diffusion                   ln_traldf_OFF   =  T', 1),
    'overflow_kt1_10/ocean.output:696': ('iso-neutral Madec operator              ln_traldf_iso   =  F', 1),
    'stprk3.F90:174': ('CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )   ! before slope for standard operator', 1),
    'traldf_iso.F90:135': ('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )   ! calculate  a33 element   (ah_wslp2 and akz)', 1),
    'trazdf.F90:173': ('zwt(ji,jk) = avt(ji,jj,jk) + ah_wslp2(ji,jj,jk)', 1),
    'ocean_model_latlon_cgrid.py:10474': '_T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]',
    'ocean_model_latlon_cgrid.py:10929': 'k33_implicit = compute_isoneutral_K33_latlon(',
    'dino.py:1775': '"tracer_combine": "thickness_weighted",',
    'dino.py:4005': 'tracer_combine=cfg.tracer_combine,',
    'nemo_testcase_l2_gyre_stage3_completion_gate.py:147-156': ['def _oracle_pre_zdf(record: dict, nlev: int, dt: float) -> dict[str, np.ndarray]:',
 ('for name in ("T", "S")', 1),
 10],
    'trazdf.F90:159-160': ["IF(  ( cdtype == 'TRA' .AND. ( jn == jp_tem .OR. ( jn == jp_sal .AND. ln_zdfddm ) ) ) .OR.   &", "& ( cdtype == 'TRC' .AND. jn == 1 )  )  THEN", 2],
    'trazdf.F90:172-174': [('DO_2Dik( 0, 0,   2, jpk, 1 )', 2), ('END_2D', 2), 3],
    'trazdf.F90:204': 'zwt(:,1) = 0._wp',
    'trazdf.F90:219-220': ['zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / e3w(ji,jj,jk,Kmm)', 'zws(ji,jk) = - p2dt * zwt(ji,jk+1) / e3w(ji,jj,jk+1,Kmm)', 2],
    'trazdf.F90:89-91': ['IF ( .NOT.(ln_SEOS.AND.(rn_b0==0._wp)) ) THEN', ('ENDIF', 4), 3],
    'ldftra.F90:249': ('l_ldfslp = .TRUE.', 1),
    'domzgr_substitute.h90:126': ('# define  e3t(i,j,k,t)      (E3t_0(i,j,k) Tmsk(r3t,tmask,i,j,k,t))', 1),
    'domzgr_substitute.h90:131': ('# define  e3w(i,j,k,t)      (E3w_0(i,j,k) Time(r3t,i,j,t))', 1),
    'ppsrc/nemo/trazdf.f90:231-233': ['zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))', 'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )', 3],
    'round29_oracle_v2_zdf_matrix/ocean.output:568': 'ln_zdfddm =  F',
    'round29_oracle_v2_zdf_matrix/ocean.output:667': ('==>>>   Rotated laplacian operator (standard)', 1),
    'round29_oracle_v2_zdf_matrix/ocean.output:657': 'ln_traldf_msc   =  F',
    'round29_oracle_v2_zdf_matrix/ocean.output:158': 'ln_SEOS   =  F',
    'zdf_oce.f90:85-86': ['ALLOCATE( avm (jpi,jpj,jpk), avm_k(jpi,jpj,jpk), avs(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) ,   &', '&      avt (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) , avt_k(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) , en (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpk) ,   &', 2],
    'trazdf.f90:247': 'WRITE(il2_unit) avt',
    'trazdf.f90:249': 'WRITE(il2_unit) avs',
    'trazdf.f90:419': 'zwt(:,1) = 0._wp',
    'trazdf.f90:443-444': ['zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm)))', 'zws(ji,jk) = - p2dt * zwt(ji,jk+1) / (e3w_1d(jk+1) *(1._wp+r3t(ji,jj,Kmm)))', 2],
    'trazdf.f90:445': 'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )',
    'trazdf.f90:528-529': ['zrhs =       (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))) * pt(ji,jj,jk,jn,Kbb )   &', '& + p2dt * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))) * pt(ji,jj,jk,jn,Krhs)   ! zrhs=right hand side', 2],
    'stprk3_stg.f90:522-523': ['zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 'zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_3d(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)', 2],
    'domain.f90:213': 'r1_hu_0(:,:) = ssumask(:,:) / ( hu_0(:,:) + 1._wp -  ssumask(:,:) )',
    'stprk3_stg.f90:541-542': ['uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 2],
    'round29_oracle_v2_zdf_matrix/ocean.output:247': 'Tiling (T) or not (F)                ln_tile    =  F',
    'trazdf.f90:493': 'zwt(ji,1) = zwd(ji,1)',
    'trazdf.f90:496': 'zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)',
    'trazdf.f90:515-516': ['pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,1))) * pt(ji,jj,1,jn,Kbb )    &', '&               + p2dt * (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1))) * pt(ji,jj,1,jn,Krhs)', 2],
    'trazdf.f90:532': 'pt(ji,jj,jk,jn,Kaa) = zrhs - zwi(ji,jk) / zwt(ji,jk-1) * pt(ji,jj,jk-1,jn,Kaa)',
    'trazdf.f90:543': 'pt(ji,jj,jpkm1,jn,Kaa) = pt(ji,jj,jpkm1,jn,Kaa) / zwt(ji,jpkm1) * tmask(ji,jj,jpkm1)',
    'trazdf.f90:546-547': ['pt(ji,jj,jk,jn,Kaa) = ( pt(ji,jj,jk,jn,Kaa) - zws(ji,jk) * pt(ji,jj,jk+1,jn,Kaa) )   &', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 2],
    'stprk3_stg.f90:540-543': ['DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)   ! corrected horizontal velocity', ('END DO   ;   END DO   ;   END DO', 7), 4],
    'stprk3_stg.F90:365': 'IF( ln_dynadv_vec .OR. lk_linssh ) THEN   !* applied on velocity',
    'stprk3_stg.F90:366-369': [('DO_3D( 0, 0, 0, 0, 1, jpkm1 )', 2), ('END_3D', 4), 4],
    'stprk3_stg.F90:370-388': ['ELSE                                      !* applied on thickness weighted velocity', ('ENDIF', 11), 19],
    'stprk3_stg.F90:395': ('CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping', 1),
    'stprk3_stg.F90:440,444-445': ['zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 'uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 3],
    'dynzdf.F90:127-132': ['puu(ji,jj,jk,Kaa) = (        ( 1._wp + r3u(ji,jj,Kbb) ) * puu(ji,jj,jk,Kbb )  &', '&              /          ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 6],
    'domzgr_substitute.h90:89': '#     define  e3u_0(i,j,k)    e3t_1d(k)',
    'domzgr_substitute.h90:98': ('#     define  e3u_0(i,j,k)    e3u_3d(i,j,k)', 1),
    'lock_kt1_10/ocean.output:705': 'ln_dynadv_vec  =  F',
    'overflow_kt1_10/ocean.output:822': 'ln_dynadv_vec  =  F',
    'ocean_model_latlon_cgrid.py:8992': '_vector_velocity_stage_update = (',
    'dynzdf.F90:119': 'IF( ln_dynadv_vec .OR. lk_linssh )',
    'dynzdf.F90:150-151': ['puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b', 2],
    'round19_oracle_v2_external/ocean.output:798': 'ln_dynadv_vec  =  T',
    'variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output:1097': 'ln_drgimp   =  T',
    'variant_icebergs_off_phase2v_tke_a_10step_np2/ocean.output:1363': 'ln_dynspg_ts  =  T',
    'stprk3.F90:186': 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
    'stprk3.F90:195': 'stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )',
    'stprk3.F90:197': [('Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs', 1), ('Nrhs = Nnn   ;   Nnn  = Naa   ;   Naa  = Nrhs', 1), 1],
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
    'stp2d.F90:281': 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )',
    'stprk3_stg.F90:65': 'SUBROUTINE stp_RK3_stg( kstg, kstp, Kbb, Kmm, Krhs, Kaa )',
    'stprk3_stg.F90:262,267': ['zub(ji,jj) = r1_Dt * rn_Dt * un_adv(ji,jj)', 'zub(ji,jj) = un_adv(ji,jj)*r1_hu(ji,jj,Kmm) - uu_b(ji,jj,Kmm)', 2],
    'stprk3_stg.F90:309-334': [('SELECT CASE( kstg )', 2), ('ENDIF', 9), 26],
    'stprk3_stg.F90:315': 'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu',
    'stprk3_stg.F90:315-430': ['IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu', 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 116],
    'stprk3_stg.F90:324': 'CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )',
    'stprk3_stg.F90:324-378': ['CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', '/           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 55],
    'stprk3_stg.F90:327': 'CALL    dyn_vor( kstp,      Kmm, uu, vv, Krhs )',
    'stprk3_stg.F90:331': 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)',
    'stprk3_stg.F90:333': [('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), 1],
    'stprk3_stg.F90:367-368': ['uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) )', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) )', 2],
    'stprk3_stg.F90:373-378': ['uu(ji,jj,jk,Kaa) = (         ( 1._wp + r3u(ji,jj,Kbb) )', '/           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 6],
    'stprk3_stg.F90:437-446': [('#endif', 6), ('END_3D', 6), 10],
    'stprk3_stg.F90:439-446': [('DO_2D( 0, 0, 0, 0 )             ! barotropic velocity correction', 1), ('END_3D', 6), 8],
    'lock_kt1_10/ocean.output:560': 'ln_drgimp   =  T',
    'lock_kt1_10/ocean.output:752': 'ln_dynspg_ts  =  T',
    'overflow_kt1_10/ocean.output:672': 'ln_drgimp   =  T',
    'overflow_kt1_10/ocean.output:869': 'ln_dynspg_ts  =  T',
    'stprk3_stg.F90:453-598': ['Tracers : RHS computation + time-stepping', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 146],
    'stprk3_stg.F90:453-601': ['Tracers : RHS computation + time-stepping', ('END DO', 6), 149],
    'stprk3_stg.F90:463-599': [('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), 'IF( ln_zdfnpc  )   CALL tra_npc( kstp,      Kmm, Krhs, ts    , Kaa  )', 137],
    'stprk3_stg.F90:655': 'END SUBROUTINE stp_RK3_stg',
    'stprk3_stg.F90:168-243': ['r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 'Dynamic : RHS computation + time-stepping', 76],
    'dynspg_ts.F90:282': 'zu_frc(:,:) =   Ue_rhs(:,:)',
    'dynspg_ts.F90:296': [('CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', 1), ('CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )', 1), 1],
    'dynspg_ts.F90:303': [('#else', 3), ('#else', 3), 1],
    'dynspg_ts.F90:344-345': ['DO_3D( 0, 0, 0, 0, 1, jpkm1 )', 'puu(ji,jj,jk,Krhs) = ( puu(ji,jj,jk,Krhs) - zu_frc(ji,jj) )', 2],
    'dynspg_ts.F90:487': 'un_e  (:,:) =    puu_b(:,:,Kmm)',
    'dynspg_ts.F90:735-761': [('DO_2D( 0, 0, 0, 0 )', 19), ('END_2D', 30), 27],
    'dynspg_ts.F90:750-753': ['z1_hv = ssvmask(ji,jj) / ( hv_0(ji,jj) + zsshv_a(ji,jj)', 'rDt_e * (  zhu_bck        * zu_spg (ji,jj)', 4],
    'dynspg_ts.F90:752-761': ['ua_e(ji,jj) = (               hu_e  (ji,jj) *   un_e (ji,jj)', ('END_2D', 30), 10],
    'dynspg_ts.F90:825-847': ['ELSE                                       ! Sum transports', 'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 23],
    'dynspg_ts.F90:870-895': ['IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av ) THEN', "CALL lbc_lnk( 'dynspg_ts', puu_b, 'U', -1._wp, pvv_b, 'V', -1._wp )", 26],
    'dynspg_ts.F90:910': [('#else', 7), ('#else', 7), 1],
    'dynspg_ts.F90:938-975': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 3), '* ( pvv_b(:,:,Kaa) - pvv_b(:,:,Kbb) * hv(:,:,Kbb) ) * r1_Dt', 38],
    'dynhpg.F90:359': 'puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap',
    'dynhpg.F90:383': 'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap',
    'dynhpg.F90:359,383': ['puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap', 'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap', 2],
    'domzgr_substitute.h90:139': [('define  gdept(i,j,k,t)', 1), ('define  gdept(i,j,k,t)', 1), 1],
    'domzgr_substitute.h90:145': [('gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))', 1), ('gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))', 1), 1],
    'domzgr_substitute.h90:139,145': [('define  gdept(i,j,k,t)', 1), ('gdept_z0(i,j,k,t) (gdept(i,j,k,t)-ssh(i,j,t))', 1), 2],
    'oce.F90:39,99': ['ssh, uu_b,  vv_b', 'ALLOCATE( ssh (jpi,jpj,jpt)  , uu_b(jpi,jpj,jpt)', 2],
    'DOM/istate.F90:149-155': ['uu_b(:,:,Kbb) = 0._wp   ;   vv_b(:,:,Kbb) = 0._wp', 'vv_b(:,:,Kbb) = vv_b(:,:,Kbb) * r1_hv(:,:,Kbb)', 7],
    'usrdef_hgr.F90:103-104': ['pff_f(:,:) = 0._wp            ! here No earth rotation: f=0', 'pff_t(:,:) = 0._wp', 2],
    'trabbl.F90:519-527': [('DO_2D( 1, 0, 1, 0 )', 6), ('END_2D', 9), 9],
    'domain.F90:159': 'r1_hu_0(:,:) = ssumask(:,:) / ( hu_0(:,:) + 1._wp',
    'domzgr_substitute.h90:143': 'gdept_z0(i,j,k,t) gdept(i,j,k,t)',
    'stp2d.F90:163': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )',
    'stp2d.F90:165': 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )',
    'stp2d.F90:169': 'CALL dyn_adv_cen2( kt     , Kbb, uu, vv, Krhs, pUe=Ue_rhs',
    'stp2d.F90:177': [('SELECT CASE( n_dynadv )', 2), ('SELECT CASE( n_dynadv )', 2), 1],
    'stp2d.F90:178': 'CASE( np_VEC_c2, np_LIN_dyn )',
    'stp2d.F90:183': 'CASE ( np_FLX_c2, np_FLX_up3 )',
    'namelist_cfg:91': 'ln_dynvor_ens = .true.',
    'lock_kt1_10/ocean.output:715': [('enstrophy conserving scheme', 1), ('enstrophy conserving scheme', 1), 1],
    'stp2d.F90:180': 'Ue_rhs(ji,jj) = SUM( e3u_0(ji,jj,1:jpkm1)',
    'stp2d.F90:207': 'grav * (  ssh_ib (ji+1,jj  ) - ssh_ib (ji,jj) )',
    'stp2d.F90:223': '( zpice(ji+1,jj) - zpice(ji,jj) ) * r1_e1u(ji,jj)',
    'stp2d.F90:235': '( bhd_wave(ji+1,jj) - bhd_wave(ji,jj) ) * r1_e1u(ji,jj)',
    'namelist_cfg:85': 'ln_dynadv_up3 = .true.',
    'namelist_cfg:105-106': ['nn_bt_flt     = 3', 'rn_bt_alpha   = 0.07', 2],
    'namelist_ref:1092': 'nn_bt_flt     = 1          ! Add dissipation',
    'namelist_ref:1177': 'ln_zad_Aimp = .false.',
    'stprk3_stg.F90:400': 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )     ! lateral mixing',
    'stprk3_stg.F90:430': 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration',
    'stprk3_stg.F90:156': 'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )',
    'stprk3_stg.F90:163': ('r3u(:,:,Kaa) = r3ua(:,:)', 1),
    'dynadv.F90:144': "CASE( np_VEC_c2  )   ;   WRITE(numout,*) '   ==>>>   vector form : keg + zad + vor is used'",
    'dynzdf.F90:121-122': ['puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kbb) + rDt * puu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kbb) + rDt * pvv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 2],
    'dynzdf.F90:156-159': ['puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * uu_b(ji,jj,Kaa)   &', ('&                                            / e3v(ji,jj,ikv,Kaa)', 1), 4],
    'dynzdf.F90:182-195': ['zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )     +  avm(ji,jj,jk  )     ) &', ('zwd(ji,1) = 1._wp - zzws', 1), 14],
    'dynldf.F90:70': 'CALL dynldf_lev_lap( kt, Kbb, Kmm, puu, pvv, Krhs )',
    'dynldf_lev_rot_scheme.h90:24-25': ['e2v(ji  ,jj-1) * pv_in(ji  ,jj-1,jk,Kbb)', 'e1u(ji-1,jj  ) * pu_in(ji-1,jj  ,jk,Kbb)', 2],
    'dynldf_lev_rot_scheme.h90:28-29': ['e2u(ji,jj)*e3u(ji,jj,jk,Kbb) * pu_in(ji,jj,jk,Kbb)', 'e1v(ji,jj)*e3v(ji,jj,jk,Kbb) * pv_in(ji,jj,jk,Kbb)', 2],
    'dynzdf.F90:296': 'zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) / e3u(ji,jj,iku,Kaa)',
    'dynzdf.F90:329-330': ['puu(ji,jj,1,Kaa) = puu(ji,jj,1,Kaa) + rDt * utauU(ji,jj)   &', '&                                    / ( e3u(ji,jj,1,Kaa) * rho0 ) * umask(ji,jj,1)', 2],
    'domqco.F90:166-169': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 1), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 4],
    'domqco.F90:219-222': [('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)  &', 2), ('&                    + e1e2t(ji,jj+1) * pssh(ji,jj+1)  ) * r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 2), 4],
    'vertical.py:798': 'def nemo_qco_live_face_geometry_cgrid(',
    'dynzdf.F90:97': 'zDt_2 = rDt * 0.5_wp',
    'dynzdf.F90:148': 'IF( ln_drgimp .AND. ln_dynspg_ts ) THEN',
    'dynzdf.F90:149-150': [('DO_2Dik( 0, 0,     1, jpkm1, 1 )      ! remove barotropic velocities', 1), ('puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)', 1), 2],
    'dynzdf.F90:153': ('DO_1Di( 0, 0 )      ! Add bottom/top stress due to barotropic component only', 1),
    'stprk3_stg.F90:435': ('#if ! defined key_PSYCLONE_2p5p0', 5),
    'stprk3_stg.F90:440': 'zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)',
    'stprk3_stg.F90:446': ('END_3D', 6),
    'stp2d.F90:144-145': [('!                             !*  COR + MET  *!   Flux Form        : Coriolis + Metric Term', 1), ('!                             !*     VOR     *!   Vector Inv. Form : Coriolis + relative Vorticity', 1), 2],
    'usrdef_zgr.F90:133-137': ['zsur = -2033.194295283385_wp', 'zacr =     5.0_wp', 5],
    'ocean_model_latlon_cgrid.py:9909-9912': ['u3_corr = u3_raw * _ws_stage_u_mask', '_replace_stage_mean, target_u, target_v)', 4],
    'ocean_model_latlon_cgrid.py:12006-12032': ['if _ws_stage3_correction is not None:', 'v=state_new.v.replace(data=_v_after),', 27],
    'stprk3_stg.F90:433': ('!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1', 1),
    'stprk3_stg.F90:440-441': ['zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 'zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_0(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)', 2],
    'stprk3_stg.F90:444-445': ['uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 2],
    'ocean_model_latlon_cgrid.py:11813-11815': ['_nemo_ws_pre_implicit_state = (', 'if self._nemo_ws_test_hooks.expose_pre_implicit_state else None)', 3],
    'ocean_model_latlon_cgrid.py:14809': 'u_solve_in = u_solve_in - _u_bt_mean',
    'ocean_model_latlon_cgrid.py:15008': 'u_solve_in = u_solve_in - (',
    'ocean_model_latlon_cgrid.py:1716': 'expose_stage3_raw_momentum: bool = False',
    'ocean_model_latlon_cgrid.py:3567-3659': ['_stage3_rhs_hook = self._nemo_ws_test_hooks.expose_stage3_momentum_rhs', 'expose_stage3_raw_momentum cannot be combined with another', 93],
    'ocean_model_latlon_cgrid.py:11999-12006': ['if self._nemo_ws_test_hooks.expose_stage3_raw_momentum:', 'if _ws_stage3_correction is not None:', 8],
    'ocean_model_latlon_cgrid.py:12178-12182': ['if _nemo_ws_exposed_stage3_raw is not None:', ('v=state_new.v.replace(data=_raw_v),', 3), 5],
    'ocean_model_latlon_cgrid.py:1717-1722': ['# WRITE-only observer for the four source-ordered stage-3 dyn_zdf', 'zdf_momentum_observer: object = None', 6],
    'ocean_model_latlon_cgrid.py:3673-3677': ['_zdf_momentum_observer = (', 'raise ValueError("zdf_momentum_observer must be callable or None")', 5],
    'ocean_model_latlon_cgrid.py:13892-13904': [('_zc = self.z_coord if z_coord is None else z_coord', 11), '# Argument validation at ENTRY, not inside the drag branch below:', 13],
    'ocean_model_latlon_cgrid.py:14557-14566': ['u_new, v_new = state.u.data, state.v.data', 'if do_momentum and getattr(_cfg_b, "surface_stress_implicit",', 10],
    'ocean_model_latlon_cgrid.py:14809-14814': ['u_solve_in = u_solve_in - _u_bt_mean', 'u_solve_in, v_solve_in)', 6],
    'ocean_model_latlon_cgrid.py:15008-15055': ['u_solve_in = u_solve_in - (', '_zdf_baro_drag_u, _zdf_baro_drag_v = u_solve_in, v_solve_in', 48],
    'ocean_model_latlon_cgrid.py:15278-15285': ['if _zdf_momentum_observer is not None:', 'ordered=True,', 8],
    'nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:171-230': ['def run(output: Path, expect_commit: str, entry_input: Path,', '"ordinary/observer commit mismatch")', 60],
    'nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:232-248': ['observer_rows = _noninterference(ordinary_arrays, observed)', 'arrays[name] = R60._active(candidate, mask)', 17],
    'nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py:280-292': ['planted = None', 'R60.require(after == before + 1, "plant did not add one refusal")', 13],
    'ocean_pe_latlon_cgrid.py:3814-3817': ['if not (getattr(grid, "dlon", 0.0) and grid.dlon > 0.0):',
 '"with a scalar dlon (got dlon<=0; tripolar unsupported)."',
 4],
    'vertical.py:67-72': [('if e3t_0 is None:', 1), '"literal NEMO QCO e3t requires explicit/reference nemo_e3t_0")', 6],
    'vertical.py:77-78': [('if getattr(z_coord, "linear_free_surface", False):', 1), 'return e3t_0', 2],
    'vertical.py:86-90': ['if active is None:', 'tmask = jnp.asarray(active, dtype=dtype)', 5],
    'lock_kt1_10/ocean.output:556': 'ln_drg_OFF  =  T',
    'overflow_kt1_10/ocean.output:668': 'ln_drg_OFF  =  T',
    'stprk3_stg.F90:287': 'zFw used in tracers only and computed in tra_adv_trp',
    'stprk3_stg.F90:295-301': [('ELSE                                     !* Flux Form', 1), ('zFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 1), 7],
    'stprk3_stg.F90:463': [('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1), 1],
    'nemo_testcase_recipe.py:892': [('zdf_baroclinic_only=True,', 2), ('zdf_baroclinic_only=True,', 2), 1],
    'provenance.py:149': 'def git_sha(*, allow_dirty: bool = False, repo: str | Path | None = None) -> str:',
    'cpp_GYRE_BARE.fcm:1': 'key_linssh key_vco_1d  key_RK3',
    'cpp_GYRE_OMIP_L2_P3_SM.fcm:1': 'key_qco key_vco_1d3d key_RK3',
    'round19_oracle_v2_external/ocean.output:338': 'ice shelf cavities             ln_isfcav =  F',
    'round19_oracle_v2_external/ocean.output:553': 'Courant number targeted application   ln_zad_Aimp =  F',
    'round19_oracle_v2_external/ocean.output:629': 'implicit friction                         ln_drgimp   =  T',
    'round19_oracle_v2_external/ocean.output:630': 'implicit ice-ocean drag                   ln_drgice_imp  = F',
    'round19_oracle_v2_external/ocean.output:214': 'single column domain (1x1pt)            ln_c1d      =  F',
    'round19_oracle_v2_external/ocean.output:546': 'bdy_init : open boundaries not used (ln_bdy = F)',
    'round19_oracle_v2_external/ocean.output:559': 'OSMOSIS-OBL closure (OSM)               ln_zdfosm =  F',
    'round19_oracle_v2_external/ocean.output:706': ('==>>>   iso-level laplacian operator', 1),
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:485': 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )     ! lateral mixing',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3_stg.f90:498': 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration',
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:161': 'ln_dynadv_vec = .true.',
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:187-188': ['ln_dynldf_lap =  .true.', 'ln_dynldf_lev =  .true.', 2],
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:189-191': ['nn_ahm_ijk_t  = 0', 'rn_Lv      = 100.e+3', 3],
    'ocean.output:875': 'Barotropic time filter => nn_bt_flt',
    'lock_kt1_10/ocean.output:615': 'no explicit diffusion                ln_dynldf_OFF',
    'overflow_kt1_10/ocean.output:727': 'no explicit diffusion                ln_dynldf_OFF',
    'ocean_pe_latlon_cgrid.py:6150': 'rho_prime=rho_prime, h_k=h_k,',
    'ocean_pe_latlon_cgrid.py:6127-6128': ['_u_ldf_local = u if ldf_state is None else ldf_state[2]', '_v_ldf_local = v if ldf_state is None else ldf_state[3]', 2],
    'ocean_pe_latlon_cgrid.py:2244-2245': [('z_coord,', 26), ('eta_safe,', 7), 2],
    'ocean_pe_latlon_cgrid.py:2285-2286': ['requires the raw NEMO', 'nemo_e3w_0 mesh field; midpoint reconstruction on', 2],
    'nemo_testcase_recipe.py:575,842,4274': [('pgf_scheme="nemo_sco",', 1), ('pgf_scheme="nemo_sco",', 2), 'if cfg.pgf_scheme != "nemo_sco":', 3],
    'BLD/ppsrc/nemo/dynspg_ts.f90:1224': 'REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in   ) ::  puu, pvv',
    'BLD/ppsrc/nemo/dynhpg.f90:378,397': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2), 'DO jk= 2, jpkm1', 2],
    'BLD/ppsrc/nemo/dynhpg.f90:393,412': ['puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap', 'puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap', 2],
    'BLD/ppsrc/nemo/dynadv_up3.f90:138-141': ['IF( PRESENT( pUe ) ) THEN     ! 3D RHS cumulation : set 2D RHS to zero', ('END DO   ;   END DO', 1), 4],
    'BLD/ppsrc/nemo/dynadv_up3.f90:211,317,355': [('ELSE                           !-  added the 3D RHS  -!', 1), ('ELSE                           !-  added the 3D RHS  -!', 2), 'ELSE                                !-  added the 3D RHS  -!', 3],
    'BLD/ppsrc/nemo/dynadv_up3.f90:213,336,357': ['puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - 0.25_wp', 'puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zFu_t(ji,jj) - zzFu_kp1 )', 'puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - zFu_t(ji,jj) * r1_e1e2u(ji,jj)', 3],
    'BLD/ppsrc/nemo/dynvor.f90:655-656': [('zwx(ji,jj) = e2u(ji,jj) * (e3t_1d(jk)', 2), ('zwy(ji,jj) = e1v(ji,jj) * (e3t_1d(jk)', 2), 2],
    'BLD/ppsrc/nemo/dynvor.f90:665': 'pu_rhs(ji,jj,jk) = pu_rhs(ji,jj,jk) + zuav * ( zwz(ji  ,jj-1) + zwz(ji,jj) )',
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168-190': ['CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 23],
    'GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:195-222': ['!  RK3 time integration', 'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 28],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:834-846': ['!* Swap', 'sshn_e (:,:) = ssha_e(:,:)', 13],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:468-477': ['WRITE(l2_r81_filename', '& ssumask(ntsi:ntei,ntsj:ntej), ssvmask(ntsi:ntei,ntsj:ntej)', 10],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:522-572': ['WRITE(l2_uamid_unit) jn, za1, za2, za3', 'WRITE(l2_r81_unit) zhup2_e(ntsi:ntei,ntsj:ntej), zhvp2_e(ntsi:ntei,ntsj:ntej)', 51],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:603-615': ['zhdiv = (   ( zhU(ji,jj) - zhU(ji-1,jj) ) + ( zhV(ji,jj) - zhV(ji,jj-1) )  ) * r1_e1e2t(ji,jj)', '& ssha_e(ntsi:ntei,ntsj:ntej)', 13],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:670-713': ['WRITE(l2_r81_unit) za0, za1, za2, za3, zsshp2_e(ntsi:ntei,ntsj:ntej)', '& zu_frc(ntsi:ntei,ntsj:ntej), zv_frc(ntsi:ntei,ntsj:ntej)', 44],
    'GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:814-849': ['WRITE(l2_r81_unit) ua_e(ntsi:ntei,ntsj:ntej), va_e(ntsi:ntei,ntsj:ntej)', ('& sshn_e(ntsi:ntei,ntsj:ntej)', 2), 36],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:286-291': ['! set values computed in RK3_ssh', 'zCdU_v  (:,:) = CdU_v   (:,:)', 6],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:298-321': ['CALL dyn_cor_2D_init( Kmm )', 'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 24],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:456-531': ['!* Set extrapolation coefficients for predictor step:', 'zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 76],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:542-553': ['!     Compute Sea Level at step jit+1', 'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 12],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:593-607': ['! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2', 'zv_spg(ji,jj) = - zldg * ( zsshp2_e(ji,jj+1) - zsshp2_e(ji,jj) ) * r1_e2v(ji,jj)', 15],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:611': ['CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )'],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:627-630': ['IF ( .NOT. ll_wd ) THEN', 'zv_trd(ji,jj) = zv_trd(ji,jj) + zCdU_v(ji,jj) * vn_e(ji,jj) * hvr_e(ji,jj)', 4],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:657-667': ['ua_e(ji,jj) = (                                 un_e(ji,jj)', '&   ) * ssvmask(ji,jj)', 11],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:701-730': ['IF( .NOT.lk_linssh ) THEN !* Update ocean depth', 'CALL bdy_dyn2d( jn, ua_e, va_e, un_e, vn_e, hur_e, hvr_e, ssha_e )', 30],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:749-780': ['!* Swap', 'pssh(:,:,Kaa) = pssh(:,:,Kaa) + za1 * ssha_e(:,:)', 32],
    'GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:797-804': ['! Finalize sums:', 'pssh   (:,:,Kaa) = pssh   (:,:,Kaa) / r1_wgt1s', 8],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:147-179': ['CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )', "CALL r46_rhs( 'after_adv', uu, vv, Krhs )", 33],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:222-230': ['CASE( np_VEC_c2, np_LIN_dyn )', 'Ve_rhs(ji,jj) = Ve_rhs(ji,jj) + SUM(', 9],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:246-259': ['CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs, Ve_rhs, CdU_u, CdU_v )', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 14],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:290-345': ['ssh_frc(:,:) = sshe_rhs(:,:)', "WRITE(numout,*) 'ROUND117_PRELOOP_FORCING_DUMP '", 56],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:329-345': ['WRITE(l2_r117_unit) puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_frc, zv_frc', "WRITE(numout,*) 'ROUND117_PRELOOP_FORCING_DUMP '", 17],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynldf_lev.f90:121-141': [('DO jj = ntsj-( 0), ntej+(  0+1 )', 1), ('END DO   ;   END DO', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:117-131': ['CASE ( nkeg_C2 )', ('END DO   ;   END DO', 2), 15],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynkeg.f90:129-130': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs)', 1), ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs)', 1), 2],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynadv.f90:339-345': ['IF( ln_dynadv_OFF  ) THEN', 'IF( nn_dynkeg /= nkeg_C2 .AND. nn_dynkeg /= nkeg_HW )', 7],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/namelist_cfg:161-162': ['ln_dynadv_vec = .true.', 'nn_dynkeg     = 0', 2],
    'GYRE_OMIP_L2_P3_SM_R117PRELOOP/EXP00/ocean.output:780-781': ['Vector form: 2nd order centered scheme', 'with Hollingsworth scheme (=1) or not (=0)', 2],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:278-324': [('ALLOCATE( zub(', 1), ('DEALLOCATE( zub, zvb )', 1), 47],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:792-819': [('IF( ln_shuman ) THEN', 1), ('WRITE(l1_unit) zFu, zFv, zFw', 2), 28],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:843-853': [('IF( lwp .AND. .NOT.ln_tile .AND. kstp <= nit000 + 1 .AND. kstg <= 2 ) THEN', 1), ('WRITE(l1_unit) zFu, zFv, zFw', 3), 11],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90:247-250': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 )', 2), 'pFw(ji,jj,jpk) = 0._wp', 4],
    'GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90:253-280': ['IF( kstg == 3 ) THEN', 'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 28],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/l2_r46_stage.f90:120-147': ["IF(ios /= 0) CALL ctl_stop('round46: cannot open stage record')", "CALL put2('r3v_Kaa         ',r3v(:,:,Kaa))", 28],
    'GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:792-834': ['IF( ln_shuman ) THEN                     ! Shuman averaging-', "WRITE(numout,*) 'ROUND94_STAGE_CLOSURE_DUMP '", 43],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:200-226': ['! Stage 1 :', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 27],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:193-200': ['ht_0(:,:) = 0._wp  ! Reference ocean thickness', 'hv_0(:,:) = hv_0(:,:) + e3v_3d(:,:,jk) * vmask(:,:,jk)', 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90:212-215': ['r1_ht_0(:,:) = ssmask (:,:) /', 'r1_hf_0(:,:) = ssfmask(:,:) /', 4],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90:155-160': ['e1e2t (:,:) = e1t(:,:) * e2t(:,:)', 'e1e2v (:,:) = e1v(:,:) * e2v(:,:)', 6],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domhgr.f90:170-171': ['r1_e1e2u(:,:) = 1._wp / e1e2u(:,:)', 'r1_e1e2v(:,:) = 1._wp / e1e2v(:,:)', 2],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domqco.f90:266-268': [('pr3u(ji,jj) = 0.5_wp *', 2), ('pr3v(ji,jj) = 0.5_wp *', 2), 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:188-201': ['!  RK3 : single first external mode computation', 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 14],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:138-178': [('SELECT CASE( kstg )', 1), 'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 41],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:188-190': ['r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:228-235': [('ssh (:,:,Kaa) = r1_2 * ( ssh (:,:,Kbb) + ssha(:,:) )', 2), ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 8],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:240': ('CASE ( 3 )           !==  Stage 3  ==!', 1),
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:301-308': ['!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stp2d.f90:291-298': ['!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:289-325': ['! set values computed in RK3_ssh', ('END DO   ;   END DO', 1), 37],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:1336-1359': [('SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )', 1), ('END SUBROUTINE dyn_cor_2D', 2), 24],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:462-591': ['DO jn = 1, icycle', '& ssha_e(ntsi:ntei,ntsj:ntej)', 130],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:631-684': ['! Half-step back interpolation of SSH for surface pressure', '& zu_frc(ntsi:ntei,ntsj:ntej), zv_frc(ntsi:ntei,ntsj:ntej)', 54],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:697-759': ['! Set next velocities:', 'hvr_e(ji,jj) = ssvmask(ji,jj) / (  hv_e(ji,jj)', 63],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:811-844': ['vbb_e  (:,:) = vb_e  (:,:)', 'END DO                                               !        end loop', 34],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/dynspg_ts.f90:857-890': ['! Finalize sums:', 'WRITE(r137_unit) pssh(ntsi:ntei,ntsj:ntej,Kaa)', 34],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/stprk3_stg.f90:176-205': ['!==  ssh/h0 ratio at Kaa  ==!', ('r3u(:,:,Kaa) = r3ua(:,:)', 1), 30],
    'GYRE_OMIP_L2_P3_SM_R137EXT/BLD/ppsrc/nemo/domqco.f90:237-258': [('SUBROUTINE dom_qco_r3c_RK3', 1), ('END DO   ;   END DO', 4), 22],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:292-347': ['! set values computed in RK3_ssh', "WRITE(numout,*) 'ROUND139_SLOW_FORCING_DUMP '", 56],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:1359-1382': [('SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )', 1), ('END SUBROUTINE dyn_cor_2D', 2), 24],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:139-228': ['!*  hydrostatic pressure gradient (HPG))  *!', ('WRITE(l2_slow_unit) Ue_rhs, Ve_rhs', 3), 90],
    'GYRE_OMIP_L2_P3_SM_R139SLOW/BLD/ppsrc/nemo/stp2d.f90:297-298': ['IF( ln_dynspg_ts )', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 2],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:142-169': ['!*  hydrostatic pressure gradient (HPG))  *!', 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 28],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:182-192': [('IF( lwp .AND. kt == 1081 ) THEN', 1), 'WRITE(r140_slow_unit) e3u_3d, uu(:,:,:,Krhs), umask, e3v_3d, vv(:,:,:,Krhs), vmask', 11],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:210-228': ['!*  vertical averaging  *!', 'WRITE(r140_slow_unit) Ue_rhs, Ve_rhs, r1_hu_0, r1_hv_0', 19],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-260': ['!* baroclinic drag forcing *!', "WRITE(numout,*) 'ROUND140_DEVELOPED_RHS_DUMP '", 25],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-239': ['!* baroclinic drag forcing *!', 'WRITE(r140_slow_unit) Ue_rhs, Ve_rhs, CdU_u, CdU_v', 4],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-260': ['!* wind forcing *!', "WRITE(numout,*) 'ROUND140_DEVELOPED_RHS_DUMP '", 20],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-246': ['!* wind forcing *!', 'WRITE(r140_slow_unit) r1_rho0, utauU, vtauV', 6],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192': ['!*  hydrostatic pressure gradient (HPG))  *!', "WRITE(numout,*) 'ROUND146_DEVELOPED_RHS_FAMILIES_DUMP '", 48],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-130': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1), ('END DO   ;   END DO', 1), 10],
    'GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:132-140': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 9],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/dynspg_ts.f90:1460-1493': ['SUBROUTINE dyn_drg_init( Kbb, Kmm, puu, pvv, puu_b ,pvv_b, pu_RHSi, pv_RHSi, pCdU_u, pCdU_v )', 'pCdU_v(ji,jj) = r1_2*( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) )', 34],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:248-251': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 3), ('END DO   ;   END DO', 4), 4],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:252-260': [('IF( lwp .AND. kt == nit000 ) THEN', 4), "WRITE(numout,*) 'ROUND140_DEVELOPED_RHS_DUMP '", 9],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:249-250': ['Ue_rhs(ji,jj) =  Ue_rhs(ji,jj) + r1_rho0 * utauU(ji,jj)', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 2],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:180-192': ['! Round 140: repeat the established slow-forcing operand list at', 'WRITE(r140_slow_unit) e3u_3d, uu(:,:,:,Krhs), umask, e3v_3d, vv(:,:,:,Krhs), vmask', 13],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-250': ['!* baroclinic drag forcing *!', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 15],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:160-182': ['SUBROUTINE dom_qco_zgr( Kbb, Kmm )', "CALL lbc_lnk( 'dom_qco_zgr', r3u(:,:,Kbb)", 23],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:188-218': ['SUBROUTINE dom_qco_r3c( pssh, pr3t, pr3u, pr3v, pr3f )', ('END DO   ;   END DO', 2), 31],
    'GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90:212-215': ['r1_ht_0(:,:) = ssmask', 'r1_hf_0(:,:) = ssfmask', 4],
    'ocean_model_latlon_cgrid.py:6801-6906': ['_ws_face_thickness_kbb = None', '_ws_v_live_mask, _grid, z_coord=_zc)[:2])', 106],
    'ocean_model_latlon_cgrid.py:7289-7352': [('if _sfx is not None:', 1), ('F_slow_v + _wind_increment_v) * _slow_surface_v_mask', 1), 64],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830': ['lr123_write = lwp .AND. .NOT.ln_tile .AND. kstg == 3 .AND.', 'WRITE(r123_unit) r3t(:,:,Kbb), r3t(:,:,Kmm), r3t(:,:,Kaa)', 13],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869': [('CALL tra_adv', 3), ('WRITE(r123_unit) ts(:,:,:,jp_tem,Krhs)', 2), 9],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-970': ['IF( ln_traqsr  ) THEN', "CALL ctl_stop( 'stp_RK3_stg: cannot close round-185 process dump' )", 41],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-945': ['IF( ln_traqsr  ) THEN', 'DEALLOCATE( l2_qsr_before )', 16],
    'GYRE_OMIP_L2_P3_SM_R185PREPROC/BLD/ppsrc/nemo/traqsr.f90:615-645': ['zz0 =           rn_abs   * r1_rho0_rcp', ('zatt(ji,jj) = zzatt', 2), 31],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:925-931': ["r186_magic = 'NEMO_L2_R186QSR1'", '& ts(:,:,:,jp_tem,Krhs)-r186_qsr_before(:,:,:), r186_qsr_replay', 7],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/sbc_oce.f90:210-212': ['ALLOCATE( qns    (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0))', '&      STAT=ierr(5) )', 3],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/mppini.f90:1501-1507': ['Nis0 =   1+nn_hls', 'Nj_0 = Nje0 - Njs0 + 1', 7],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/nemogcm.f90:368': 'Nbb = 1   ;   Nnn = 2   ;   Naa = 3   ;   Nrhs = Naa',
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3.f90:200-222': ['! Stage 1 :', 'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 23],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:616-645': ['zz0 =           rn_abs   * r1_rho0_rcp', ('pts(ji,jj,jk,jp_tem,Krhs) = pts(ji,jj,jk,jp_tem,Krhs)', 11), 30],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:1021-1070': ['FUNCTION qsr_ext_lev( pL, pfr ) RESULT( klev )', 'END FUNCTION qsr_ext_lev', 50],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/traqsr.f90:1145-1213': ['r1_si0 = 1._wp / rn_si0', 'nkV  = qsr_ext_lev( rn_si1, zVlp )', 69],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:172-192': ['!                     !==  ssh/h0 ratio at Kaa  ==!', ('END SELECT', 2), 21],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:227-237': [('CASE ( np_HYB )', 2), ('ENDIF', 5), 11],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/domqco.f90:237-258': ['SUBROUTINE dom_qco_r3c_RK3( pssh, pr3t, pr3u, pr3v, pr3f )', ('END DO   ;   END DO', 4), 22],
    'GYRE_OMIP_L2_P3_SM_R186QSRWALK/BLD/ppsrc/nemo/stprk3_stg.f90:911-950': ['!           !==  complete the tracers RHS  ==!   except ZDF (implicit)', 'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )  ! lateral mixing', 40],
    'ocean_model_latlon_cgrid.py:9391-9439': [('if _return_tracer_process_trace:', 1), 'process_qsr_rate=_nemo_ws_process_qsr_rate,', 49],
    'ocean_model_latlon_cgrid.py:8923-8934': ['nemo_r3t_rk3_stage1_stretch,', '_qt_13 = nemo_r3t_rk3_stage1_stretch(', 12],
    'eos.py:1004-1035': ['def nemo_r3t_rk3_stage1_stretch(', ('return jnp.where(wet, nemo_source_round(one + r3_stage), one)', 1), 32],
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
    'VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:184-190': [('ioptio = 0                      ! parameter control and set n_dynadv', 1), ("IF( ioptio /= 1 )   CALL ctl_stop( 'choose ONE and only ONE advection scheme' )", 1), 7],
    'VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:135-138': [('CASE( np_VEC_c2  )                                                         != vector form =!', 1), ('CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection', 1), 4],
    'VORTEX_VEC_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:861-864': [('CASE( np_VEC_c2  )', 1), ('ntot = np_CRV        ! relative + planetary vorticity', 1), 4],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:312-316': ['CASE ( 1 )        !==  Stage 1  ==!', 'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 5],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:287-293': ['IF( ln_dynadv_vec ) THEN                 !* Vector invariant Form : no use of ww at stage 1 as 3D RHS computed in stp_2D', 'IF( ln_zad_Aimp .AND. kstg == 3 )   CALL wAimp( kstp, Kmm, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, wi, np_velocity, ld_diag=.TRUE. )', 7],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:318-332': ['!                 !-------------------!   Flux Form        : HPG + VOR (COR+MET) + ADV', 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 15],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-138': ['SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!', 'CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection', 5],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:363-370': ['CASE ( 1 , 2 )    !==  Stage 1 & 2  ==!   time stepping', ('END DO   ;   END DO   ;   END DO', 4), 8],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:387-405': ['CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping', 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration', 19],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:408-419': ['!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1', ('END DO   ;   END DO   ;   END DO', 6), 12],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:323-338': ['CASE ( 2 , 3 )    !==  Stage 2 & 3  ==!', 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 16],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:135-141': ['SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!', "CALL vortex_r8_stage_rhs( 'zad', Krhs, puu, pvv )", 7],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:320-334': ['puu(ji,jj,1,Krhs) = zhpi(ji,jj)     ! RK3 case: dyn_hpg always called first', ('pvv(ji,jj,jk,Krhs) = zhpj(ji,jj)', 1), 15],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:359': 'DO jn = 1, icycle                             !  sub-time-step loop  !',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:389': 'ua_e(ji,jj) = za1 * un_e(ji,jj) + za2 * ub_e(ji,jj) + za3 * ubb_e(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:401': 'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:414': 'zhup2_e(ji,jj) = hu_0(ji,jj) + r1_2 * r1_e1e2u(ji,jj)                        &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:434': 'zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:454': 'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:463': 'un_adv(ji,jj) = un_adv(ji,jj) + za2 * zhU(ji,jj) * r1_e2u(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:478': 'zsshu_a(ji,jj) = r1_2 * r1_e1e2u(ji,jj) * ( e1e2t(ji  ,jj  ) * ssha_e(ji  ,jj  )   &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:489': 'CALL ts_bck_interp( jn, ll_init, za0, za1, za2, za3 )   ! coeficients of the interpolation',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:491': 'zsshp2_e(ji,jj) = za0 *  ssha_e(ji,jj) + za1 *  sshn_e (ji,jj)   &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:498': 'zu_spg(ji,jj) = - zldg * ( zsshp2_e(ji+1,jj) - zsshp2_e(ji,jj) ) * r1_e1u(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:503': 'CALL dyn_cor_2D( ua_e, va_e, zu_trd, zv_trd )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:933': 'SUBROUTINE dyn_cor_2D_init( Kmm )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960': 'zpvo_nw = ff_f(ji-1,jj  ) / (e3f_0vor(ji-1,jj  ,jk) *(1._wp+r3f(ji-1,jj  )*fe3mask(ji-1,jj  ,jk))) + &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1112': 'SUBROUTINE dyn_cor_2D( punb, pvnb, zu_trd, zv_trd   )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:892': 'ALLOCATE( e3f_0vor(jpi,jpj,jpk) )',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1016': 'ffu_nw(ji,jj) = ffu_nw(ji,jj) + (e3t_1d( jk) *(1._wp+r3u(ji  ,jj, Kmm)*umask(ji  ,jj, jk))) * (e3t_1d( jk) *(1._wp+r3v(ji  ,jj  , Kmm)*vmask(ji  ,jj  , jk))) * vmask(ji  ,jj  ,jk) / (e3f_0vor(ji,jj  ,jk) *(1._wp+r3f(ji,jj  )*fe3mask(ji,jj  ,jk)))',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1046': 'zr1_e3f = ff_f(ji,jj) / (e3f_0vor(ji,jj  ,jk) *(1._wp+r3f(ji,jj  )*fe3mask(ji,jj  ,jk))) + ff_f(ji,jj-1) / (e3f_0vor(ji,jj-1,jk) *(1._wp+r3f(ji,jj-1)*fe3mask(ji,jj-1,jk)))',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:890': 'CASE( np_ENS , np_ENE , np_EEN , np_MIX )',
    'GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:167': 'ln_dynvor_ene = .true.  !  energy conserving scheme',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:897': ('e3f_0vor(ji,jj,jk) = (   ( e3t_1d(jk)*tmask(ji  ,jj+1,jk)     &   ! need additional () for', 1),
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:920': 'WHERE( e3f_0vor(:,:,jk) == 0._wp )   e3f_0vor(:,:,jk) = e3t_1d(jk)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:517': 'zu_trd(ji,jj) = zu_trd(ji,jj) + zCdU_u(ji,jj) * un_e(ji,jj) * hur_e(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:535': 'ua_e(ji,jj) = (                                 un_e(ji,jj)   &',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:581': 'hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:628': ('puu_b  (:,:,Kaa) = puu_b  (:,:,Kaa) + za1 * ua_e  (:,:)', 1),
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:648': 'un_adv(:,:) = un_adv(:,:) / r1_wgt2s',
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:133-135': ['ssha(:,:) = ssh (:,:,Kaa)     ! save ssh, uu_b, vv_b at N+1  (computed in dynspg_ts)', 'va_b(:,:) = vv_b(:,:,Kaa)', 3],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:421-429': ['ALLOCATE( zub(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)), zvb(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) )', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 9],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:289-300': [('IF( ln_dynadv_vec ) THEN', 1), 'CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )', 12],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/divhor.f90:123-140': ['SELECT CASE ( ik_ind )', 'END SELECT', 18],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:271-299': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 2), ('END DO   ;   END DO   ;   END DO', 6), 29],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynzad.f90:105-137': ['DO jk =  1,  jpk-2', '&                                              * zWdzV(ji,jj)', 33],
    'VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traadv.f90:268-273': ['CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', ('END DO   ;   END DO   ;   END DO', 3), 6],
    'VORTEX_SMT_R3_OMIP_L1_P3/BLD/ppsrc/nemo/usrdef_zgr.f90:240-247': ['pe3u(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji+1,jj,jk) )', '&          kfillmode = jpfillcopy )', 8],
    'vertical.py:692-757': ['def nemo_qco_resolved_mesh_operands(', 'e3v0 = jnp.asarray(e3v0, dtype=dtype)[..., :nlev]', 66],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:137-163': ['CALL eos    ( ts, Kbb, rhd )                          ! in situ density anomaly at Kbb', 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )                 !- vertical advection', 27],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:176-179': ['CASE( np_VEC_c2, np_LIN_dyn )       ! Vector Inv. Form   ==>> averaged 3D RHS only', 'Ve_rhs(ji,jj) = SUM( e3t_1d(1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmask(ji,jj,1:jpkm1) ) * r1_hv_0(ji,jj)', 4],
    'vortex_round3/namelist_cfg:182-185': [('ln_dynadv_vec = .true.  !  decision 73: ORCA2/GYRE vector-invariant momentum', 1), ('ln_dynadv_up3 = .false. !  decision 73: exactly one advection form (dynadv.F90:190)', 1), 4],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:239': ('r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )   ! at N+1/2 (Kmm)', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:405': ('IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )  ! vertical diffusion and time integration', 1),
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:412-418': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)             ! barotropic velocity correction', 1), ('vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 1), 7],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:143-147': [('ELSE                                      ! applied on thickness weighted velocity', 1), ('&              /          ( 1._wp + r3u(ji,jj,Kaa) ) * umask(ji,jj,jk)', 1), 5],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:159-161': [('DO jk =      1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)      ! remove barotropic velocities', 1), ('pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 1), 3],
    'VORTEX_OMIP_L1/BLD/ppsrc/nemo/dynzdf.f90:191-195': [('DO jk =    2,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)            ! inner values', 1), ('&           / ( (e3t_1d(jk  ) *(1._wp+r3u(ji  ,jj,Kaa)*umask(ji  ,jj,jk  ))) * (e3w_1d(jk+1) *(1._wp+r3u(ji,jj,Kmm))) ) * wumask(ji,jj,jk+1)', 1), 5],
    'tests/VORTEX/MY_SRC/usrdef_nam.F90:96-97': ['kpi = NINT( 1800.e3  / rn_dx ) + 3', 'kpj = NINT( 1800.e3  / rn_dy ) + 3', 2],
    'tests/VORTEX/MY_SRC/usrdef_nam.F90:121': ['kpk = NINT( 5000._wp / rn_dz ) + 1'],
    'tests/VORTEX/MY_SRC/usrdef_hgr.F90:83-84': ['zroffsetx = (-REAL(Ni0glo-1, wp) + 1._wp) * 0.5_wp * 1.e-3 * rn_dx', 'zroffsety = (-REAL(Nj0glo-1, wp) + 1._wp) * 0.5_wp * 1.e-3 * rn_dy', 2],
    'tests/VORTEX/MY_SRC/usrdef_hgr.F90:174-177': ['zbeta = 2._wp * omega * COS( rad * rn_ppgphi0 ) / ra', 'pff_t(:,:) = zf0 + zbeta * pphit(:,:) * 1.e+3', 4],
    'tests/VORTEX/MY_SRC/usrdef_zgr.F90:126': ['zd = 5000._wp/REAL(jpkm1,wp)'],
    'tests/VORTEX/MY_SRC/usrdef_zgr.F90:187-193': ['z2d(:,:) = REAL( jpkm1 , wp )          ! flat bottom', 'k_top(:,:) = MIN( 1 , k_bot(:,:) )     ! = 1    over the ocean point, =0 elsewhere', 7],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:69-75': [('zf0   = 2._wp * omega * SIN( rad * rn_ppgphi0 )', 1), ('zP0 = rho0 * zf0 * zumax * zlambda * SQRT(EXP(1._wp)/2._wp)', 1), 7],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:83-88': ['zrho1 = rho0 * (1._wp + zn2*zdt/grav)', 'pts(ji,jj,jk,jp_tem) = (20._wp + (rho0-zrho1) / rn_a0 ) * ptmask(ji,jj,jk)', 6],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:101-105': ['zdu = 0.5_wp * (pdept(ji  ,jj,jk) + pdept(ji+1,jj,jk))', 'pu(ji,jj,jk) = (za * zf * zy * EXP(-(zx**2+zy**2)/zlambda**2)) * ptmask(ji,jj,jk) * ptmask(ji+1,jj,jk)', 5],
    'tests/VORTEX/MY_SRC/usrdef_istate.F90:177-182': ['za = -zP0 * (1._wp-EXP(-zH)) / (grav*(zH-1._wp + EXP(-zH)))', 'pssh(ji,jj) = zP0 * EXP(-(zx**2+zy**2)/zlambda**2)/(zrho1*grav) * ptmask(ji,jj,1)', 6],
    'tests/VORTEX/MY_SRC/usrdef_sbc.F90:60-68': ['utau(:,:) = 0._wp', 'qsr (:,:) = 0._wp', 9],
    'restart.F90:461': [('CALL usr_def_istate_ssh( tmask, ssh(:,:,Kbb) )', 2)],
    'DOM/istate.F90:127-130': ['DO jk = 1, jpk', 'CALL usr_def_istate( zgdept, tmask, ts(:,:,:,:,Kbb), uu(:,:,:,Kbb), vv(:,:,:,Kbb) )', 4],
    'DOM/istate.F90:149-154': ['uu_b(:,:,Kbb) = 0._wp   ;   vv_b(:,:,Kbb) = 0._wp', 'uu_b(:,:,Kbb) = uu_b(:,:,Kbb) * r1_hu(:,:,Kbb)', 6],
    'eosbn2.F90:1890-1895': ['NAMELIST/nameos/ ln_TEOS10, ln_EOS80, ln_SEOS, rn_T0, rn_S0, rn_a0, rn_b0, rn_lambda1, rn_mu1, &', 'READ_NML_CFG(numnam,nameos)', 6],
    'dynvor.F90:874': ['IF( ln_dynvor_een ) THEN   ;   ioptio = ioptio + 1   ;   nvor_scheme = np_EEN   ;   ENDIF'],
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
    'nemo_state_bridge.py:280,922': [('nemo_prognostic_barotropic_velocity=True,', 1), ('nemo_prognostic_barotropic_velocity=True,', 2), 2],
    'nemogcm.f90:185': ('CALL stp_MLF( istp )', 1),
    'dynspg_ts.f90:489-491': [('sshn_e(:,:) =    pssh (:,:,Kbb)', 1), ('vn_e  (:,:) =    pvv_b(:,:,Kbb)', 1), 3],
    'zdftke.F90:845-846': [('ELSE                          ! standard case : associated avt minimum = molecular viscosity (10^-6 m2/s)', 1), ('rmxl_min = 1.e-6_wp / ( rn_ediff * SQRT( rn_emin ) )    ! resulting minimum length to recover molecular viscosity', 1), 2],
    'zdftke.F90:841-843': [('IF( ln_zdfiwm ) THEN          ! Internal wave-driven mixing', 1), ('rmxl_min = 1.e-03_wp             ! associated avt minimum = molecular salt diffusivity (10^-9 m2/s)', 1), 3],
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:396': ('ln_zdfiwm   = .true.       ! internal wave-induced mixing            (T =>   fill namzdf_iwm)', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:90-104': [('! Lane-1 certified oracle: exact step-entry Nbb state.  This is a', 1), ("WRITE(numout,*) 'LANE1_STEP_ENTRY_DUMP ', kstp, Nbb, STORAGE_SIZE(1._wp), TRIM(cl_traj)", 1), 15],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:211-227': [('! Stage 1 :', 1), ('IF( kstp <= nit000 + 9 )   CALL l1_dump_stage( kstp, 3, Naa )', 1), 17],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282-283': [('zFu(ji,jj,jk) = e2u(ji,jj)', 1), ('zFv(ji,jj,jk) = e1v(ji,jj)', 1), 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:637-645': [('CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )', 1), ('CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 1), 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:529,531-532': [('CASE ( np_FCT )', 1), ('CALL tra_adv_fct ( kt, nit000,', 1), ('nn_fct_imp )', 1), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609': [('! *** 1st step', 1), ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))) * pt_b(ji,jj,jk) + zDt * ztra ) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) * tmask(ji,jj,jk) !!clem', 1), 115],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-197,260-266': [('!* horizontal anti-diffusive fluxes', 1), ('ztFu(ji,jj,jk) =', 1), ('!* vertical anti-diffusive fluxes', 1), ('- ztFw(ji,jj,jk)  ) * wmask(ji,jj,jk)', 1), 12],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316,743-938': [('CALL nonosc(', 1), ('SUBROUTINE nonosc(', 1), ('END SUBROUTINE nonosc', 2), 197],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:320-329': [('! -- Final trend with corrected fluxes', 1), ('zta_up1(ji,jj,jk) =', 1), 10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:194-200': [('!*  vertical averaging  *!', 1), ('END DO   ;   END DO', 2), 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:569-572': [('ptFu(ji,jj,jk) = 0.5_wp', 1), ('MIN( pV(ji,jj,jk) , 0._wp ) * pt_up1(ji,jj+1,jk)', 1), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:577-578': [('ptFw(ji,jj,jk) = 0.5_wp', 1), ('MIN( pW(ji,jj,jk) , 0._wp ) * pt_up1(ji,jj,jk-1)', 1), 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910-913': [('zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji,jj+1,ik) )', 1), ('pbb(ji,jj,jk) = pbb(ji,jj,jk) * zcoef', 2), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-876': [('zup = MAX(', 2), ('zbetdo(ji,jj,ik) = ( paft(ji,jj,jk) - zdo ) / zneg * zbt', 1), 24],
    'ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/stprk3.f90:202-221': [('CALL r141_growth_entry(', 1), ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1), 20],
    'ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/traadv.f90:299-320': [('CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 1), ('& pFw(ntsi:ntei,ntsj:ntej,:) )', 1), 22],
    'ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/stprk3.f90:202-221': [('CALL r143_growth_entry(', 1), ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1), 20],
    'ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/traadv.f90:299-320': [('CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 1), ('& pFw(ntsi:ntei,ntsj:ntej,:) )', 1), 22],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/stprk3.f90:204-215': [('CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 1), ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1), 12],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:359-375': [('sshn_e(:,:) =    pssh (:,:,Kmm)', 1), ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kbb)))', 1), 17],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:289-291': [('ssh_frc(:,:) = sshe_rhs(:,:)', 1), ('zv_frc(:,:) =   Ve_rhs(:,:)', 1), 3],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:357-375': [('IF( ln_bt_fw ) THEN', 2), ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kbb)))', 1), 19],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:357-366': [('IF( ln_bt_fw ) THEN', 2), ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 1), 10],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:220-223': [('ll_bt_av    = .TRUE.', 1), ('ll_init     = ll_bt_av', 1), 4],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779': [('IF( .NOT.lk_linssh ) THEN !* Update ocean depth', 1), ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp", 2), 19],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:507-545': [('ua_e(ji,jj) = za1 * un_e(ji,jj) + za2 * ub_e(ji,jj) + za3 * ubb_e(ji,jj)', 1), ('&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1), 39],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:566-569': [('zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)', 1), ('zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 1), 4],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:585-595': [('zhdiv = (   ( zhU(ji,jj) - zhU(ji-1,jj) )', 1), ("CALL r95_spg_w2('ssha_e', ssha_e)", 1), 11],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:715-757': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', 1), ('va_e(ji,jj) =  va_e(ji,jj) / ( 1._wp - rDt_e * zCdU_v(ji,jj) * hvr_e(ji,jj) )', 1), 43],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:763-766': [('hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)', 1), ('hvr_e(ji,jj) = ssvmask(ji,jj) / (  hv_e(ji,jj) + 1._wp - ssvmask(ji,jj)  )', 1), 4],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:770-779': [('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', 1), ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1), 10],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:854-856': [('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 2), ('pvv_b  (:,:,Kaa) = pvv_b  (:,:,Kaa) + za1 * va_e', 1), 3],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:924-937': [('IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av ) THEN', 1), ("CALL lbc_lnk( 'dynspg_ts', puu_b, 'U', -1._wp, pvv_b, 'V', -1._wp )", 1), 14],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:639-683': [("CASE ( 'U' )", 5), ('END DO   ;   END DO', 5), 45],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:684-721': [("CASE ( 'V' )", 5), ('END DO   ;   END DO', 6), 38],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979': [('3. Do east and west MPI_Isend if needed', 2), ('CALL MPI_ISEND( buffsnd_dp(ishtS(jn)+1)', 1), 19],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2060-2068': [('IF( ibufszR(jn) > 0 ) THEN', 3), ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = buffrcv_dp(idxr)', 1), 9],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2105-2113': [('6. north fold treatment', 2), ('CALL mpp_nfd(', 3), 9],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1561-1577': [('IF( ln_nnogather ) THEN', 2), ('l_full_nf_update = .TRUE.', 2), 17],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1629-1669': [('ALLOCATE(ipjfld(kfld))', 2), ('CALL MPI_Isend(', 2), 41],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1712-1738': [('Apply the North pole folding', 2), ('ptab(jf)%pt4d(ji,ij1,jk,jl) = psgn(jf) * zbufr(ii2,ij2,inei)', 3), 27],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1747-1766': [('IF( lnfd_same(ii1,iig) )', 2), ('ptab(jf)%pt4d(ji,ij1,jk,jl) = psgn(jf)', 4), 20],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/mppini.f90:1412-1440': [('ALLOCATE( nfd_rksnd(jpi,nn_hls+1,4)', 1), ('WHERE( lnfd_same )   nfd_jisnd(:,1,:) = HUGE(0)', 1), 29],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:890-960': [('un_adv(:,:) = un_adv(:,:) / r1_wgt2s', 1), ("CALL r95_spg_w2('ssh_aa', pssh(:,:,Kaa))", 1), 71],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90:170-235': [('SUBROUTINE tra_adv_trp_t(', 1), ('DEALLOCATE( zFu_cor, zFv_cor )', 1), 66],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90:267-315': [('IF( ll_Fw ) THEN', 1), ('pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 1), 49],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:91-107': [('! Lane-1 certified oracle: exact step-entry Nbb state.  This is a', 1), ('CALL r84_dump_frame( kstp, 0, Nbb )', 1), 17],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:147-154': [('! Update external forcing (tides, open boundaries, ice shelf interaction and surface boundary condition (including sea-ice)', 1), ('IF( kstp == nit000 )   CALL l4_dump_ocean_surface_input( kstp, Nbb )', 1), 8],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:390-441': [('SUBROUTINE l4_dump_ocean_surface_input( kstp, klevel )', 1), ('END SUBROUTINE l4_dump_ocean_surface_input', 1), 52],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:443-473': [('FUNCTION l4_canon_2d( pfield, cdgrid ) RESULT( zfield )', 1), ('END FUNCTION l4_canon_2d', 1), 31],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-228': [('Background eddy viscosity and diffusivity', 1), ('avm_k(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk)', 1), 24],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:347-359': [('start from turbulent closure values', 1), ('IF( ln_zdfevd )   CALL zdf_evd', 1), 13],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90:107-110': [('DO jk =  1,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 2), 4],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90:120-135': [('enhance momentum Kz', 1), ('p_avm(ji,jj,jk) = rn_evd * wmask', 1), 16],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:172-181': [('CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 1), ('CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 1), 10],
    'namelist_ref:1200': ('ln_zdfiwm   = .false.      ! internal wave-induced mixing            (T =>   fill namzdf_iwm)', 1),
    'domhgr.F90:222-227': [("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1), ('kff = 1', 1), 6],
    'zdftke.F90:246,253-258': [('IF( nn_eice == 0 ) zice_fra(:) = 0._wp               ! No attenuation of TKE due to sea ice', 1), ('! ice fraction considered for attenuation of langmuir & wave breaking', 1), ('END SELECT', 1), 7],
    'zdftke.F90:828-835': [('SELECT CASE( nn_eice )', 1), ('END SELECT', 7), 8],
    'zdftke.F90:601-603': [('DO_1Di( 0, 0 )                  ! No sea-ice', 1), ('END_1D', 16), 3],
    'zdftke.F90:859-862': [('IF( ln_mxl0 ) THEN', 4), ('rn_mxl0 = rmxl_min', 2), 4],
    'geo2ocean.F90:168,259-260': [('REAL(wp) ::   zxffu, zyffu, znffu   ! x,y components and norm of the vector: between F points below and above a U point', 1), ('gsinu(ji,jj) = ( zxnpu*zyffu - zynpu*zxffu ) / znffu', 1), ('gcosu(ji,jj) = ( zxnpu*zxffu + zynpu*zyffu ) / znffu', 1), 3],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:221-225': ['Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stprk3.f90:222-226': ['Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)', 5],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:149': 'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)',
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:153': 'CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:152': 'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)',
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/stp2d.f90:156': 'CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )',
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:184': "CALL iom_rstput( kt, nitrst, numrow, 'ssha', ssh(:,:,Kaa) )",
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:362-370': ["id1 = iom_varid( numror, 'ssha', ldstop = .FALSE. )", ('ssh(:,:,Kaa) = ssh(:,:,Kbb)', 1), 9],
    'VORTEX_VEC_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:295-298': [('integrate from the bottom the hor. divergence', 4), ('r1_Dt * e3t_1d(jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2), 4],
    'GYRE_OMIP_L2_P3_SM/BLD/ppsrc/nemo/sshwzv.f90:295-298': [('integrate from the bottom the hor. divergence', 4), ('r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2), 4],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:53-54': ['r1_3 = 1._wp / 3._wp', 'r2_3 = 2._wp / 3._wp', 2],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:160-179': ['!==  ssh/h0 ratio at Kaa  ==!', 'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 20],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:167': 'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )',
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:177': 'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)',
    'domhgr.F90:222-223': [("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1), ("& iom_varid( inum, 'ff_t', ldstop = .FALSE. ) > 0    ) THEN", 1), 2],
    'orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:319-326': ['ln_traldf_lap   = .true.', 'rn_Ld        = 200.e+3', 8],
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:918-933': ['ln_traldf_OFF   = .false.', 'ln_botmix_triad = .false.', 16],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:214-218': ['NAMELIST/namtra_ldf/ ln_traldf_OFF', '&                 nn_aht_ijk_t , rn_Ud          , rn_Ld', 5],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:231-233': ['READ(numnam_ref(MAX(INDEX(numnam_ref,"&namtra_ldf ")', 'IF(lwm) WRITE( numond, namtra_ldf )', 3],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:261-283': ['nldf_tra = np_ERROR', ('IF ( ln_traldf_triad )   nldf_tra = np_lap_it', 1), 23],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:354-390': ['IF(     ln_traldf_lap ) THEN   ;   zUfac = r1_2 *rn_Ud', ("CALL ldf_c2d( 'TRA', zUfac      , inn        , ahtu, ahtv )", 1), 37],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traldf.f90:105-122': [('SELECT CASE ( nldf_tra )', 2), ('END SELECT', 2), 18],
    'orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:388-392': ['&namdyn_ldf', 'nn_ahm_ijk_t  = -30', 5],
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:1104-1134': ['&namdyn_ldf', 'rn_ahm_b      = 0.0', 31],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-185': ['NAMELIST/namdyn_ldf/ ln_dynldf_OFF', 'IF(lwm) WRITE ( numond, namdyn_ldf )', 9],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:221-276': ['nldf_dyn = np_ERROR', 'IF( nldf_dyn == np_lap_i )   l_ldfslp = .TRUE.', 56],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:311-346': ['IF(     ln_dynldf_lap ) THEN   ;   zUfac = r1_2 *rn_Uv', ("CALL ldf_c2d( 'DYN', zUfac      , inn        , ahmt, ahmf )", 1), 36],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90:81-90': ['SELECT CASE ( nldf_dyn )', 'CALL dynldf_lev_blp( kt, Kbb, Kmm, puu, pvv, Krhs )', 10],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:387-405': [('CASE ( 3 )        !==  Stage 3  ==!', 1), 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 19],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:231-233': ['READ(numnam_ref(MAX(INDEX(numnam_ref,"&namtra_ldf ")', 'IF(lwm) WRITE( numond, namtra_ldf )', 3],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:261-283': ['nldf_tra = np_ERROR', ('IF ( ln_traldf_triad )   nldf_tra = np_lap_it', 1), 23],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:354-390': ['IF(     ln_traldf_lap ) THEN   ;   zUfac = r1_2 *rn_Ud', ("CALL ldf_c2d( 'TRA', zUfac      , inn        , ahtu, ahtv )", 1), 37],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110': [('SELECT CASE ( nldf_tra )', 2), 'CALL traldf_iso_lap  ( kt, Kbb, Kmm, pts, Krhs, l_ptr, l_hst )', 6],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )', 19],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:456-488': [('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 2), "CALL vortex_r18_tracer_rhs( 'adv', Krhs, ts )", 33],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:501-546': [('SELECT CASE( kstg )', 4), 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 46],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:532': 'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )',
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546': 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )',
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:154-304': [('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1), ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 151],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90:186-338': ['zgru(ji,jj,iikm1) = umask(ji,jj,jpkm1)', ("CALL lbc_lnk( 'ldfslp', uslp", 1), 153],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:780-829': [('DO jk =    2,  jpkm1', 1), 'pakz(ji,jj,jk) = MAX( zcoef0 - 0.5_wp , 0._wp )', 50],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )', 19],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110': [('SELECT CASE ( nldf_tra )', 2), 'CALL traldf_iso_lap  ( kt, Kbb, Kmm, pts, Krhs, l_ptr, l_hst )', 6],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90:226-356': [('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4), ('END DO   ;   END DO', 12), 131],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:822-837': [('DO jk =    2,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 1), 16],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:240-267': [('DO jj = ntsj-( 0+1), ntej+(  0 ) ; DO ji = ntsi-( 0+1), ntei+(  0 )', 1), 'r227_fu(ji,jj,jk) = zfu(ji,jj) ; r227_fv(ji,jj,jk) = zfv(ji,jj)', 28],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:282-333': [('IF( 1 <= jk .AND. jk <= jpk-2 ) THEN', 1), ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 52],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259': [('zA11 = e2_e1u(ji,jj) * (e3u_3d(ji,jj,jk)', 1), ('&                                             + ( zdkt(ji,jj+1,ikp1) + zdkt(ji,jj,ik  ) )  )   )', 1), 17],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259': [('zfu(ji,jj) =  ahtu(ji,jj,jk)', 1), ('&                                             + ( zdkt(ji,jj+1,ikp1) + zdkt(ji,jj,ik  ) )  )   )', 1), 6],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:260-266': [('IF( r227_iso_active .AND. jn == jp_tem ) THEN', 2), ('END DO   ;   END DO', 7), 7],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1), ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1), 5],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331': [('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 2), ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2), 5],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:225': 'CASE ( 3 )           !==  Stage 3  ==!   Kbb = N   ;   Kmm = N+1/2   ;   Kaa = N+1',
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:531-532': ['IF( ln_traqsr  )   CALL tra_qsr( kstp,      Kmm, ts, Krhs )', 'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )', 2],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:155': 'e1e2t (:,:) = e1t(:,:) * e2t(:,:)   ;   r1_e1e2t(:,:) = 1._wp / e1e2t(:,:)',
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90:149-151': ["CALL write3( unit, 'rhs_before', r227_rhs_before )", "CALL write3( unit, 'rhs_increment', pt(:,:,:,jp_tem,Krhs) - r227_rhs_before )", 3],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-185': ['NAMELIST/namdyn_ldf/ ln_dynldf_OFF', 'IF(lwm) WRITE ( numond, namdyn_ldf )', 9],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-278': ['nldf_dyn = np_ERROR', ('ENDIF', 11), 58],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:311-346': ['IF(     ln_dynldf_lap ) THEN   ;   zUfac = r1_2 *rn_Uv', ("CALL ldf_c2d( 'DYN', zUfac", 1), 36],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf.f90:81-90': ['SELECT CASE ( nldf_dyn )', 'CALL dynldf_lev_blp( kt, Kbb, Kmm, puu, pvv, Krhs )', 10],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344': ['!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST', ("CALL vortex_r16_stage_rhs( 'adv', Krhs, uu, vv )", 2), 17],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:398-404': ['CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping', "CALL vortex_r16_stage_rhs( 'ldf', Krhs, uu, vv )", 7],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:415-417': ['!*  DYN time integration + ZDF  *!   ∆t = rDt', 'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 3],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:539-554': ['!==  complete the tracers RHS  ==!   except ZDF (implicit)', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 16],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:193-200': ['!* baroclinic drag forcing *!   (also provide the barotropic drag coeff.)', ('END DO   ;   END DO', 4), 8],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:260-267': ['Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa ) ! time-splitting', 8],
    'VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119': ['IF( .NOT. ln_rst_list .AND. nn_stock == -1 )   RETURN', 'IF( kt == nitrst - 1 .OR. nn_stock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN', 16],
    'VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119': ['IF( .NOT. ln_rst_list .AND. nn_stock == -1 )   RETURN', 'IF( kt == nitrst - 1 .OR. nn_stock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN', 16],
    'VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119': ['IF( .NOT. ln_rst_list .AND. nn_stock == -1 )   RETURN', 'IF( kt == nitrst - 1 .OR. nn_stock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN', 16],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119': ['IF( .NOT. ln_rst_list .AND. nn_stock == -1 )   RETURN', 'IF( kt == nitrst - 1 .OR. nn_stock == 1 .OR. ( kt == nitend .AND. .NOT. lrst_oce ) ) THEN', 16],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:314-335': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 1), ('END DO   ;   END DO   ;   END DO', 1), 22],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-138': ['SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!', 'CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection', 5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:225-230': [('WRITE(l2_slow_unit) r1_rho0', 1), ('Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0', 1), 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:553-569': [('!       (in addition to snwice_mass)', 1), ('&                      ( 2. - vmask(ji,jj,1) )', 1), 17],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:336-339': [('DO ji = ntsi-( 0), ntei+( 0 )                    !==  second recurrence', 1), ('/ ( (e3u_3d(ji,jj,1) *(1._wp+r3u(ji,jj,Kaa)', 2), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:678-680': [('ts(ji,jj,jk,jn,Kaa) = (', 1), ('/          ( 1._wp + r3t(ji,jj,Kaa) )', 1), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:217-224': [('ssh (:,:,Kaa) = r1_2 * ( ssh (:,:,Kbb) + ssha(:,:) )', 2), ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307': ('pts(ji,jj,1,jp_tem,Krhs) = pts(ji,jj,1,jp_tem,Krhs) +  r1_rcp * qns(ji,jj) * z1_rho0_e3t', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:279-282': [('ELSE                       !==   runoff put only at the surface', 1), ('phdivn(ji,jj,1) = phdivn(ji,jj,1) - rnf(ji,jj)', 1), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traqsr.f90:387-394': [('!                                      !- fluxes at jk+1 w-level', 1), ('pts(ji,jj,jk,jp_tem,Krhs) = pts(ji,jj,jk,jp_tem,Krhs) + r1_rho0_rcp * ( zeT(ji,jj) - zzeT ) / ze3t', 1), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288-289': [('zrhs =       (e3t_3d(ji,jj,jk)', 1), ('& + p2dt * (e3t_3d(ji,jj,jk)', 1), 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:220': 'SUBROUTINE tra_sbc_RK3 (',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:282': {'anchors': 'IF( .NOT.lk_linssh ) THEN           !* only heat and salt fluxes', 'routine': 'tra_sbc_RK3'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:284-286': {'anchors': [('z1_rho0_e3t = r1_rho0 / (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))', 1), '- emp(ji,jj)*pts(ji,jj,1,jp_sal,Kbb) * z1_rho0_e3t', 3], 'routine': 'tra_sbc_RK3'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:292': {'anchors': ('IF( lk_linssh ) THEN                !* linear free surface', 2), 'routine': 'tra_sbc_RK3'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:308': {'anchors': '+           sfx(ji,jj) * z1_rho0_e3t', 'routine': 'tra_sbc_RK3'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:49': 'INTEGER  :: n_baro_upd =  np_HYB',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:78': 'SUBROUTINE stp_RK3_stg(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177': {'anchors': 'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)   ! at N+1/3 (Kaa)', 'routine': 'stp_RK3_stg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dom_oce.f90:137': 'lk_linssh = .FALSE.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:85': 'nn_ice      = 2',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:251': 'ln_lin = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:817': 'ln_drgimp   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:818': 'ln_drgice_imp = .true. ! implicit ice-ocean drag',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1404': 'SUBROUTINE dyn_drg_init(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1431-1432': {'anchors': ['pCdU_u(ji,jj) = r1_2*( ( rCdU_bot(ji+1,jj) + rCdU_bot(ji,jj) ) + ( rCdU_top(ji+1,jj) + rCdU_top(ji,jj) ) )', 'pCdU_v(ji,jj) = r1_2*( ( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) ) + ( rCdU_top(ji,jj+1) + rCdU_top(ji,jj) ) )', 2], 'routine': 'dyn_drg_init'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1447-1448': {'anchors': ['zu_i(ji,jj) = puu(ji,jj,ikbu,Kmm) - puu_b(ji,jj,Kmm)', ('zv_i(ji,jj) = pvv(ji,jj,ikbv,Kmm) - pvv_b(ji,jj,Kmm)', 1), 2], 'routine': 'dyn_drg_init'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1462-1463': {'anchors': ['* r1_2*( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * zu_i(ji,jj)', '* r1_2*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) ) * zv_i(ji,jj)', 2], 'routine': 'dyn_drg_init'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1487-1488': {'anchors': ['* r1_2*( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) * zu_i(ji,jj)', '* r1_2*( rCdU_top(ji,jj+1)+rCdU_top(ji,jj) ) * zv_i(ji,jj)', 2], 'routine': 'dyn_drg_init'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:63': 'SUBROUTINE stp_2D(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:219': {'anchors': 'CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs, Ve_rhs, CdU_u, CdU_v )', 'routine': 'stp_2D'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:67': 'SUBROUTINE dyn_zdf(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:166-169': {'anchors': ['puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * uu_b(ji,jj,Kaa)', ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1), 4], 'routine': 'dyn_zdf'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:175-178': {'anchors': ['puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) * uu_b(ji,jj,Kaa)', ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 2), 4], 'routine': 'dyn_zdf'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:187': {'anchors': 'rCdU_top(ji,jj) = rCdU_ice(ji,jj)', 'routine': 'zdf_drg'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:258': 'SUBROUTINE zdf_drg_lin(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:276': {'anchors': 'pCdU(ji,jj) = - pCd0(ji,jj) * rn_Uc0   ! linear friction', 'routine': 'zdf_drg_lin'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:525': {'anchors': 'IF(ll_bot)   zmsk_boost(:,:) = zmsk_boost(:,:) * ssmask(:,:)', 'routine': 'drg_init'},
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:546': {'anchors': ('pCd0(:,:) = rn_Cd0 * zmsk_boost(:,:)', 1), 'routine': 'drg_init'},
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:205': 'isecsbc = nsec_year + nsec1jan000 + NINT( ( 0.5*REAL(kn_fsbc-1,wp) + zt_offset ) * rn_Dt )',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:217': 'IF( MOD( kt-1, kn_fsbc ) == 0 ) THEN',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:244-246': ['ztinta =  REAL( isecsbc - sd(jf)%nrec(2,ibb), wp ) / REAL( sd(jf)%nrec(2,iaa) - sd(jf)%nrec(2,ibb), wp )', 'sd(jf)%fnow(:,:,:) = ztintb * sd(jf)%fdta(:,:,:,ibb) + ztinta * sd(jf)%fdta(:,:,:,iaa)', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:309-366': ['IF( ksecsbc > sdjf%nrec(2,iaa) ) THEN', 'END SUBROUTINE fld_update', 58],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:932-936': ['IF( sdjf%ln_tint ) THEN   ! record time defined in the middle of the record, computed using an implementation', '& MAX( MOD( sdjf%nrecsec(0:sdjf%nreclast-1), 2 ), MOD( sdjf%nrecsec(1:sdjf%nreclast), 2 ) ) + irecshft', 5],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssm.f90:109-174': ['IF( kt == nit000 .AND. .NOT. l_ssm_mean ) THEN', 'frq_m(:,:) = frq_m(:,:) * zcoef', 66],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcrnf.f90:185-227': ['IF( .NOT. l_rnfcpl )  THEN', 'rnf_tsc(:,:,jp_sal) = zrnf_sal * rnf(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) * r1_rho0', 43],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcssr.f90:147-205': ['IF( nn_sstr + nn_sssr /= 0 ) THEN', "IF( nn_sssr == 2 )   CALL iom_put( 'vflx_ssr_cea', -erp(:,:) )", 59],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcfwb.f90:277-332': ['CASE ( 2 )                             !==  set volume annual trend  ==!', ("IF( iom_use('vflx_fwb_cea') )  CALL iom_put( 'vflx_fwb_cea', -emp_corr                       * smask0(:,:) )", 2), 56],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/EXP00/namelist_cfg:92': 'nn_fwb      = 2         !  FreshWater Budget:',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/EXP00/namelist_cfg:153-156': ['nn_sssr     =     2     !  add a damping term to the surface freshwater flux', 'rn_sssr_bnd =   4.e0    !  ABS(Max/Min) value of the damping erp term [mm/day]', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/EXP00/namelist_cfg:164': 'rn_rfact    =   1.e0    !  multiplicative factor for runoff',
    'cfgs/SHARED/namelist_ref:474-476': ['ln_rnf_depth = .false.   !  read in depth information for runoff', 'ln_rnf_sal   = .false.   !  read in salinity information for runoff', 3],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:428-568': ['IF( .NOT.ll_sas )   CALL sbc_ssm', 'vtauV  (ji,jj) = 0.5_wp * ( vtau(ji,jj) + vtau(ji,jj+1) )', 141],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:640-678': ['q_air_zt(:,:) = sf(jp_humi )%fnow(:,:,1)', '&                zsen, zlat, zevp, zqlwn )', 39],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:791-948': ['ptsk(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) = pst', ("CALL lbc_lnk( 'sbcblk', utau, 'T', -1._wp", 1), 158],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:976': 'ptsk(:,:) = ( ptsk(:,:) - rt0 ) * smask0(:,:)',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:1019-1045': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 7), 'qns(ji,jj) = qns(ji,jj) * smask0(ji,jj)', 27],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk.f90:835-838': [('CASE( np_NCAR      )', 2), '&                nb_iter=nn_iter_algo )', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:125-238': ['nbit = nb_iter0', 'END SUBROUTINE turb_ncar', 114],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90:241-381': ['FUNCTION cd_n10_ncar( pw10 )', 'END FUNCTION psi_h_ncar', 141],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:250-409': ['impure ELEMENTAL FUNCTION pres_temp_sclr', 'END FUNCTION rho_air_sclr', 160],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:443-506': ['FUNCTION L_vap_vctr( psst )', 'END FUNCTION cp_air_sclr', 64],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:548-581': ['FUNCTION One_on_L( ptha, pqa, pus, pts, pqs )', 'END FUNCTION One_on_L', 34],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:645-766': ['ELEMENTAL FUNCTION e_sat_sclr( ptak )', 'END FUNCTION q_sat_sclr', 122],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:898-940': ['ELEMENTAL SUBROUTINE BULK_FORMULA_SCLR', 'END SUBROUTINE BULK_FORMULA_SCLR', 43],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:1019-1042': ['ELEMENTAL FUNCTION qlw_net_sclr( pdwlw, pts,  l_ice )', 'END FUNCTION qlw_net_sclr', 24],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:1062-1080': ['FUNCTION z0_from_Cd( pzu, pCd,  ppsi )', 'END FUNCTION z0_from_Cd', 19],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbc_phy.f90:1198-1213': ['FUNCTION UN10_from_CD( pzu, pUb, pCd, ppsi )', 'END FUNCTION UN10_from_CD', 16],
    'sbcmod.F90:331-340': [('IF( nn_ice == 0 ) THEN', 2), 'CALL ice_init( Kbb, Kmm, Kaa )', 10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:380': {'anchors': 'IF ( ln_drgice_imp .AND.   nn_ice /= 2  )   ln_drgice_imp = .FALSE.', 'routine': 'zdf_drg_init'},
    'zdftke.F90:622': 'fr_i(ji,jj)   * hm_i(ji,jj) * 2._wp',
    'zdftke.F90:794-795': ['ELSEIF ( nn_mxlice > 1 .AND. nn_ice == 1 ) THEN', "CALL ctl_stop( 'zdf_tke_init: with no ice model, nn_mxlice must be 0 or 1')", 2],
    'icbutl.F90:89-93': ['zssh_lead_m(:,:) = ice_var_sshdyn(ssh_m, snwice_mass, snwice_mass_b)', 'hi_icb(:,:) = vt_i(:,:) / at_i(:,:)', 5],
    'icbutl.F90:174-181': ['IF ( PRESENT(pui) ) pui = icb_utl_bilin_h( u_ice', 'IF ( PRESENT(phi) ) phi = 0._wp', 8],
    'sbcice_if.F90:105': 'IF( ln_cpl )   a_i(:,:,1) = fr_i(:,:)',
    'sbcfwb.F90:119': 'IF ( nn_ice/=2 ) nn_fwb_voltype = 2',
    'stp2d.F90:213-220': ['IF( ln_ice_embd ) THEN', 'zpice(ji,jj) = ( zintp * snwice_mass(ji,jj)', 8],
    'diaar5.F90:197-200': ["IF( iom_use( 'sshice' ) .AND. nn_ice /= 0 ) THEN", 'z2d(ji,jj) = ztf * snwice_mass(ji,jj)', 4],
    'iom.F90:116-123': [('#if defined key_xios', 4), 'LOGICAL           :: llrst_context', 8],
    'iom.F90:241-245': ['# if defined key_si3', "CALL iom_duplicate_grid( 'grid_T_nlayice'", 5],
    'icestp.F90:241-263': ['SUBROUTINE ice_init( Kbb, Kmm, Kaa )', 'CALL par_init', 23],
    'icestp.F90:341-346': ['NAMELIST/nampar/ jpl, nlay_i, nlay_s', 'READ_NML_CFG(numnam_ice,nampar)', 6],
    'ORCA2_OMIP_L4/EXP00/file_def_nemo-ice.xml:13-14': ['<file_group id="ice_5d"', '<file id="file101"', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:158-159': ['CALL eos_fzp( sss_m(:,:), fr_i(:,:) )', 'fr_i(:,:) = fr_i(:,:) * tmask(:,:,1)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:164-167': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls )', 1), ('END DO   ;   END DO', 1), 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:182-185': ['zqri = ztrp * ( ts(ji,jj,1,jp_tem,Kbb) - ( zt_fzp - 1.) )', '&                 +      fr_i(ji,jj)   * zqrj ) ) * smask0(ji,jj)', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:192-195': ['zqi = -3. + SIGN( 1._wp, ff_f(ji,jj) )', '&       + zqrp', 4],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/eosbn2.f90:1867-1868': [('ptf(ji,jj) = ( - 0.0575_wp + 1.710523e-3_wp * SQRT( psal(ji,jj) )', 1), '- 2.154996e-4_wp *       psal(ji,jj)   ) * psal(ji,jj)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/lib_fortran.f90:1815-1816': ['IF ( pb >= 0._wp ) THEN   ;   SIGN_SCALAR =  ABS(pa)', 'ELSE                      ;   SIGN_SCALAR = -ABS(pa)', 2],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:155': 'CALL sbc        ( kstp, Nbb, Nbb )',
    'cfgs/SHARED/namelist_ref:418': "sn_ice      ='ice_cover_clim.nc'",
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/fldread.f90:885': 'IF    ( NINT(sdjf%freqh) == -12 ) THEN            ;   ireclast = 1',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:501': 'CALL ice_stp  ( kt, Kbb, Kmm, nsbc )',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceupdate.f90:212': 'fr_i(ji,jj) = at_i(ji,jj)',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcmod.f90:500': 'CALL sbc_ice_if   ( kt, Kbb, Kmm )',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/sbcice_if.f90:149': 'CALL fld_read( kt, nn_fsbc, sf_ice )',
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:216-222': ['CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 'CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )', 7],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:479-525': ['FUNCTION l4_canon_3d( pfield, cdgrid ) RESULT( zfield )', 'END FUNCTION l4_canon_3d', 47],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3.f90:488-520': ['zfield(:,:,:) = 0._wp', 'IF( llwet ) zfield(ji,jj,jk) = pfield(ji,jj,jk)', 33],
    'ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3_stg.f90:846-892': ['FUNCTION l4_canon_3d( pfield, cdgrid ) RESULT( zfield )', 'END FUNCTION l4_canon_3d', 47],
    'ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:177': 'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)',
    'ocean_model_latlon_cgrid.py:2295': 'nemo_stage_pair=None, stored_h_stage=False,',
    'ocean_model_latlon_cgrid.py:2319-2323': ['if stored_h_stage and stage_index != 1: raise ValueError("stored_h_stage is Stage-2 (stage_index == 1) only")', 'e3t_0=h_ref)', 5],
    'ocean_model_latlon_cgrid.py:2321-2323': ['h_stage = jnp.asarray(h_stage) if stored_h_stage else nemo_qco_live_t_thickness(', 'e3t_0=h_ref)', 3],
    'ocean_model_latlon_cgrid.py:2707-2708': [('(mf_u, mf_v, w_stage, h_stage, hu_stage, hv_stage,', 4), 'wi_stage, zfu_stage, zfv_stage, _, _, _) = stage_geom', 2],
    'ocean_model_latlon_cgrid.py:2747-2748': ['rhs = _nemo_cen2_tracer_rhs(', 'tracer, p_u, p_v, p_w, h_stage, active_3d, grid)', 2],
    'ocean_model_latlon_cgrid.py:8985-8988': ['_h_s2_kmm = (', 'else _h_live_one_third)', 4],
    'ocean_model_latlon_cgrid.py:9354-9361': ['_stage_tracer_source_terms(h_k_old),', '_stage_tracer_sources(_h_s2_kmm),', 8],
    'ocean_model_latlon_cgrid.py:9532-9545': ['_g0 = _nemo_ws_stage_transport(', ('**_stage_transport_kw)', 1), 14],
    'ocean_model_latlon_cgrid.py:9761-9763': ['_g1 = _nemo_ws_stage_transport(', 'eta_stage=_eta_live_one_third,', 3],
    'ocean_model_latlon_cgrid.py:9832-9849': ['_g2 = _nemo_ws_stage_transport(', ('**_stage_transport_kw)', 3), 18],
    'ocean_model_latlon_cgrid.py:10335-10353': ['_nemo_ws_stage_transport_geometry = tuple(', 'strict=True)))', 19],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:480-602': ['def measure(', 'return result', 123],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:605-755': ['def aggregate(', 'str(rung10_table_out) if rung10_table_out is not None else None', 151],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:760-1411': ['def walk_source_order(', '"per_kt_stage2_summary": per_kt_stage2_summary,', 652],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:1423-1669': ['def classify(', '"d122_c_and_d": True,', 247],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round341_rescore_walk_gate.py:1995-2479': ['def caller_audit_dump(', '"callers": callers_report,', 485],
    'tests/ocean/fidelity/test_orca2_round341_rescore_walk_gate.py:22-124': ['def test_round341_self_test_passes_and_plants_fire() -> None:', 'gate.walk_source_order(r5_root, r326_root, plant=plant_name)', 103],
    'tests/ocean/unit/test_orca2_round341_stage2_thickness.py:168-604': ['def test_stage2_transport_uses_stored_h_s2_kmm_and_fails_on_revert() -> None:',
 'np.asarray(h_rev, dtype=np.float64).view(np.uint64),',
 437],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:382-3478': ['def build_orca2_exact_vs_normal_switch_inventory() -> tuple[SwitchSpec, ...]:', 'raise SystemExit(main())', 3097],
    'tests/ocean/fidelity/test_orca2_round343_speed_benchmark.py:32-203': ['def test_switch_inventory_citations_and_class_defaults() -> None:', 'r343.validate_benchmark_report(bad_days)', 172],
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:894-967': ['next_s2: NemoS2Inputs', ('),', 33), 74],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2268-2531': [('bbl_aht_m2_s=1000.0,', 2), ('1,', 51), 264],
    'packages/core/legoesm/core/precision.py:117-125': ['def fp64(*, transcendentals: str = "native") -> PrecisionPolicy:', (')', 24), 9],
    'packages/core/legoesm/core/transcendentals.py:286-300': ['def _scalar_libm_reference(name: str, values: np.ndarray) -> np.ndarray:', ('return result', 2), 15],
    'packages/core/legoesm/core/transcendentals.py:343-364': ['def _scalar_libm_binary_reference(', ('return result', 4), 22],
    # ORCA2 round 348: compiled exact-libm loops and native-vs-libm arm tool.
    'packages/core/legoesm/core/transcendentals.py:36-239': ['_LIBM_SONAME = "libm.so.6"', '"glibc_version": _glibc_version(),', 204],
    'packages/core/legoesm/core/_scalar_libm.c:4-38': ['#define DEFINE_UNARY(NAME)', ('}', 8), 35],
    'packages/core/legoesm/core/transcendentals.py:249-494': ['def _fast_function(name: str, dtype: np.dtype, *, binary: bool):', 'return result, tangent', 246],
    'packages/core/legoesm/core/transcendentals.py:91-239': ['def _glibc_version() -> str:', '"glibc_version": _glibc_version(),', 149],
    'packages/core/legoesm/core/transcendentals.py:367-379': ['def _callback_binary(name: str, left: jax.Array, right: jax.Array) -> jax.Array:', ('vmap_method="broadcast_all",', 2), 13],
    'packages/core/legoesm/core/transcendentals.py:431-556': ['def exp(value) -> jax.Array:', 'return _libm_cos(array)', 126],
    'packages/core/legoesm/core/transcendentals.py:521-532': ['def pow(left, right) -> jax.Array:', 'return _libm_pow(left_array, right_array)', 12],
    'packages/core/legoesm/core/transcendentals.py:474-532': ['def _libm_pow(left: jax.Array, right: jax.Array) -> jax.Array:', 'return _libm_pow(left_array, right_array)', 59],
    'packages/core/legoesm/core/transcendentals.py:316-364': ['def _scalar_libm_binary(', ('return result', 4), 49],
    'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round16_discriminator.py:353-407': ['class _PoisonCompiledFunction:', 'and any(changed[name] for name in ("qsr", "qns", "emp", "utau", "vtau"))', 55],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:234-299': ['def _resolve_transcendentals(requested: str) -> str:', 'return manifest', 66],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:234-323': ['def _resolve_transcendentals(requested: str) -> str:', 'require(save_stride > 0, f"save_stride must be positive, got {save_stride}")', 90],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:302-560': ['def run_standalone_k0_driver(', 'worktree=initial_worktree,', 259],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:302-312': ['def run_standalone_k0_driver(', (') -> dict[str, Any]:', 3), 11],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:56-223': ['def compare_snapshot_data(', '"daily_t3d_rms_plant": float(planted["t3d_wet_rms_k"]),', 168],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:155-223': ['def compare_daily_temperature(', '"daily_t3d_rms_plant": float(planted["t3d_wet_rms_k"]),', 69],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:249-312': ['def compare_manifest_timing(', '"candidate_transcendentals": manifests[1]["transcendentals"],', 64],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:249-402': ['def compare_manifest_timing(', 'return 0', 154],
    'tests/ocean/fidelity/test_orca2_round348_transcendentals.py:227-261': ['def test_manifest_timing_accepts_copy_of_real_pre_field_r342_manifest(', 'in capsys.readouterr().out', 35],
    'tests/ocean/fidelity/test_orca2_round348_transcendentals.py:276-308': ['def test_manifest_timing_refuses_identical_paths_and_wrong_roots(', ('candidate_root=candidate_root,', 3), 33],
    'tests/unit/test_transcendentals.py:147-219': ['def test_pow_libm_callback_dtype_matches_jax_result_type_and_native_lane(', 'np.testing.assert_array_equal(observed, np.asarray([8.0], dtype=np.float32))', 73],
    'tests/unit/test_transcendentals.py:216-219': ['def test_direct_pow_callback_preserves_float32_with_python_scalar(function) -> None:', 'np.testing.assert_array_equal(observed, np.asarray([8.0], dtype=np.float32))', 4],
    'tests/unit/test_transcendentals.py:20-25': ['tx.glibc_libm()', 'allow_module_level=True,', 6],
    'tests/ocean/fidelity/test_orca2_round342_s4_year.py:35-39': ['@pytest.fixture(autouse=True)', 'set_policy(previous)', 5],
    'tests/ocean/fidelity/test_orca2_round342_s4_year.py:180-448': ['def test_driver_manifest_records_resolved_transcendentals(', 'assert not (output_dir / ".log").exists()', 269],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py:585-588': ['"--transcendentals",', 'help="Transcendental backend (default: certified libm)",', 4],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh:27': 'TRANSCENDENTALS="libm"',
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round348_transcendentals_compare.py:282-300': ['mode = payload.get("transcendentals")', 'payload["transcendentals"] = mode', 19],
    'tests/ocean/fidelity/test_orca2_round342_s4_year.py:265-360': ['def test_driver_default_transcendentals_is_libm_and_native_plant_fails(', 'require_certified_default(run_default(tmp_path / "plant", plant_native=True))', 96],
    'tests/ocean/fidelity/test_orca2_round342_s4_year.py:710-727': ['def test_callable_default_transcendentals_is_libm_and_native_plant_fails(', ('require_certified_default()', 3), 18],
    'tests/ocean/fidelity/test_orca2_round348_transcendentals.py:312-356': ['def test_manifest_timing_legacy_field_guards_and_plants_fail(', ('candidate_root=candidate_root,', 5), 45],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh:58-108': ['if [[ "${OUTPUT_DIR}" != "/" ]]; then', '>"${LOG_FILE}" 2>&1 &', 51],
    'packages/core/legoesm/core/precision.py:103-114': [('transcendentals: str = "native"', 1), (')', 22), 12],
    'packages/core/legoesm/core/source_rounding.py:14-35': ['def nemo_source_round(value: jax.Array) -> jax.Array:', 'def nemo_source_store(fn, *operands):', 22],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:1024-1086': ['def verify_switch_inventory_against_card(', ('}', 39), 63],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:1108-1113': ['tke = vm.tke._replace(', 'iwm = vm.iwm._replace(form="generic", tsdiff=False)', 6],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:66': 'transcendentals: str = "libm"',
    'packages/core/legoesm/core/precision.py:103': ('transcendentals: str = "native"', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2482': 'tke_htau_evaluation="nemo_scalar_libm",',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:706': 'tke_htau_evaluation: str = "jax_expression"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2483': 'tke_etau_exponential_evaluation="nemo_scalar_libm",',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:698': 'tke_etau_exponential_evaluation: str = "jax_expression"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2504-2505': ['ddm_config = model_config.physics.vertical_mixing.ddm._replace(', 'enabled=True, form="nemo_literal", rn_avts=1.0e-4, rn_hsbfr=1.6)', 2],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:245-251': ['enabled: bool = False', 'form: str = "hybrid"', 7],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2497-2502': ['iwm_config = model_config.physics.vertical_mixing.iwm._replace(', 'require_forcing_maps=True,', 6],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:187-203': ['tsdiff: bool = False      # ln_tsdiff (ORCA1: .false. — avs == avt)', 'form: str = "generic"', 17],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2467': 'nemo_bn2_source_rounded=True,',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:723': ('nemo_bn2_source_rounded: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2469': 'nemo_tke_solve_source_rounded=True,',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:732': ('nemo_tke_solve_source_rounded: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2471': 'nemo_tke_matrix_rhs_source_rounded=True,',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:739': ('nemo_tke_matrix_rhs_source_rounded: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2472': 'nemo_sh2_r3_source_rounded=True,',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:743': ('nemo_sh2_r3_source_rounded: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2473': 'nemo_sh2_r3_stored_reciprocal=True,',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:747': ('nemo_sh2_r3_stored_reciprocal: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2486': 'tke_mxl_raw_evaluation="nemo_literal",',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:710': 'tke_mxl_raw_evaluation: str = "factored"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:617': 'tke_solver_evaluation="nemo_literal",',
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:690': 'tke_solver_evaluation: str = "shared_thomas"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2272': ('tke_langmuir_evaluation="nemo_literal",', 2),
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:752': 'tke_langmuir_evaluation: str = "vectorized"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3266': ('redi_flux_geometry_stage="rk3_kmm", slope_uv_source_rounding=True,', 1),
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:417': ('slope_uv_source_rounding: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3267': ('redi_a33_source_ordering=True,', 1),
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:495': ('redi_a33_source_ordering: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3244': 'redi_a33_evaluation="nemo_literal",',
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:491': 'redi_a33_evaluation: str = "normalized_square"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3243': 'redi_vertical_skew_evaluation="nemo_literal",',
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:486': 'redi_vertical_skew_evaluation: str = "normalized_sums"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3229': 'slope_metric_evaluation="nemo_reciprocal",',
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:412': 'slope_metric_evaluation: str = "division"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3322': 'through_fct=True, nemo_association=True)',
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:121': ('nemo_association: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2539': 'z_coord=z_coord._replace(nemo_r3t_stored_product=True,',
    'packages/ocean/legoesm/ocean/vertical.py:1541': 'nemo_ldf_ahmf: jnp.ndarray | None = None; nemo_r3t_stored_product: bool = False  # card-selected, see _nemo_r3t_product  # noqa: E501,E702',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2313': ('nemo_eos_gdept_rounded_product=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3263': 'nemo_eos_gdept_rounded_product: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2306': 'zdf_tracer_nemo_rounded_thickness=True,',
    'packages/ocean/legoesm/ocean/state.py:3272': 'zdf_tracer_nemo_rounded_thickness: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2307': 'barotropic_rhs_ascending_level_sum=True,',
    'packages/ocean/legoesm/ocean/state.py:3276': 'barotropic_rhs_ascending_level_sum: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2308': ('rk3_stage1_rhs_nemo_literal=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3218': 'rk3_stage1_rhs_nemo_literal: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2309': ('rk3_stage_velocity_written_order=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3223': 'rk3_stage_velocity_written_order: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2321': 'rk3_stage_tracer_written_order=True,',
    'packages/ocean/legoesm/ocean/state.py:3303': 'rk3_stage_tracer_written_order: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2311': ('rk3_stage_r3f_nemo_literal=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3253': 'rk3_stage_r3f_nemo_literal: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2312': ('rk3_stage_r3t_nemo_literal=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3258': 'rk3_stage_r3t_nemo_literal: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2314': ('rk3_stage_r3uv_nemo_literal=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3268': 'rk3_stage_r3uv_nemo_literal: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2332': 'nemo_bottom_drag_written_statements=True,',
    'packages/ocean/legoesm/ocean/state.py:3340': ('nemo_bottom_drag_written_statements: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2344': 'rk3_vector_rhs_nemo_order=True,',
    'packages/ocean/legoesm/ocean/state.py:3381': ('rk3_vector_rhs_nemo_order: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2345': ('een_vorticity_written_order=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:3229': 'een_vorticity_written_order: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2347': 'nemo_wzv_stored_r3_ratios=True,',
    'packages/ocean/legoesm/ocean/state.py:3360': ('nemo_wzv_stored_r3_ratios: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2348': 'nemo_zad_written_association=True,',
    'packages/ocean/legoesm/ocean/state.py:3366': ('nemo_zad_written_association: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2349': 'nemo_stage2_momentum_wzv_half_step=True,',
    'packages/ocean/legoesm/ocean/state.py:3372': ('nemo_stage2_momentum_wzv_half_step: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2350': 'nemo_stage_momentum_wzv_stored_kmm_ratios=True,',
    'packages/ocean/legoesm/ocean/state.py:3377': ('nemo_stage_momentum_wzv_stored_kmm_ratios: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2356': 'nemo_ldf_stored_f_area=True,',
    'packages/ocean/legoesm/ocean/state.py:3387': ('nemo_ldf_stored_f_area: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2357': 'nemo_ldf_stage_r3f=True,',
    'packages/ocean/legoesm/ocean/state.py:3406': ('nemo_ldf_stage_r3f: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2358': 'rk3_stage1_ldf_source_order=True,',
    'packages/ocean/legoesm/ocean/state.py:3400': ('rk3_stage1_ldf_source_order: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2287': ('barotropic_substep_dt_nemo_literal=True,', 1),
    'packages/ocean/legoesm/ocean/state.py:1654': 'barotropic_substep_dt_nemo_literal: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2383': 'barotropic_een_ratio_written_order=True,',
    'packages/ocean/legoesm/ocean/state.py:1675': 'barotropic_een_ratio_written_order: bool = False',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:747': ('zdf_implicit_solver_evaluation="nemo_literal",', 1),
    'packages/ocean/legoesm/ocean/state.py:3202': 'zdf_implicit_solver_evaluation: str = "shared_thomas"',
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3286-3306': ['def orca2_l8_krhs_selection() -> dict[str, object]:', '"bbl_aht_m2_s": 1000.0, "bbl_gamma_s": 10.0,', 21],
    'packages/ocean/legoesm/ocean/state.py:3462': ('rk3_stage3_damping_after_geothermal_nemo_order: bool = False', 1),
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3241': 'kappa_redi_horizontal_evaluation="nemo_metric_literal",',
    'packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:479': 'kappa_redi_horizontal_evaluation: str = "cosine_scaled"',
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1023-1158': ['def nemo_orca2_k0_ocean_step(', 'surface=surface, ocean=stepped, next_s0=next_s0, next_s2=next_s2)', 136],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:15867-15876': [('@partial(jax.jit, static_argnums=(0,))', 2), (') -> LatLonCGridOceanState:', 4), 10],
    'packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:16060-16081': ['def _carry_nemo_rk3_after_ssh(self, entry_state, new_state):', 'data=2.0 * new_state.eta.data - entry_state.eta.data))', 22],
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1259-1499': ['def nemo_blk_oce_ncar(', 'means, bulk, tail, forcing, freshwater, bulk_before_ice, iif)', 241],
    'packages/ocean/legoesm/ocean/bulk_flux_omip.py:1320-1392': ['if not update:', (')', 573), 73],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py:1186-1233': ['def build_fast_literal_mode_orca2_k0_card(base_card: Any) -> Any:', 'return {"changed_fields": ["ocean.transcendentals"], "retained_inventory_entries": retained}', 48],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:309-358': ['def nemo_zdf_ddm(avt, avm, zdt, zds, rn2, wmask, cfg: DoubleDiffusionConfig):', 'return avt_new, avs, avm_new', 50],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:388-482': ['def nemo_zdf_iwm(avt, avs, avm, forcing: IWMForcing, rn2, stretch, ht_0,', 'return avt, avs, avm', 95],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2708-2811': ['def _nemo_etau_htau(', 'profile = _nemo_glibc234_vector_exp(-depth_w / htau)', 104],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:107-115': ['IF( .NOT. ln_rst_list ) THEN', ('ENDIF', 4), 9],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/restart.f90:176-180': ["CALL iom_rstput( kt, nitrst, numrow, 'sshn', ssh(:,:        ,Kbb) )", "CALL iom_rstput( kt, nitrst, numrow, 'sn'  , ts(:,:,:,jp_sal,Kbb) )", 5],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y/BLD/ppsrc/nemo/restart.f90:291-296,311-316': ['!*  Read Kbb fields   (NB: in RK3 Kmm = Kbb = Nbb)', "CALL iom_get( numror, jpdom_auto, 'sn'   , ts(:,:,:,jp_sal,Kbb) )", 'uu(:,:,:  ,Kmm) = uu(:,:,:  ,Kbb)', 'vv_b(:,:,Kmm)   = vv_b(:,:,Kbb)', 12],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:203': 'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs    ! Swap: Nnn unchanged, Nbb <==> Naa',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stprk3.f90:241': 'IF( lrst_oce   )   CALL rst_write    ( kstp, Nbb, Nnn, Naa )   ! write output ocean restart file',
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/nemogcm.f90:237-238': ['IF( nstop == 0 ) THEN   ;   STOP 0', 'ELSE                    ;   STOP 123', 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90:106-107': ['!                                ! open time.step    ascii file, done only by 1st subdomain', "IF( lwm )   CALL ctl_opn( numstp, 'time.step', 'REPLACE', 'FORMATTED', 'SEQUENTIAL', -1, numout, lwp, narea )", 2],
    'ORCA2_STANDALONE_IIF_OMIP_L4_R339K0/BLD/ppsrc/nemo/stpctl.f90:160-161': ["WRITE ( numstp, '(1x, i8)' )   kt", 'REWIND( numstp )', 2],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stp2d.f90:136-163': ['!*  hydrostatic pressure gradient (HPG))  *!', 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 28],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stp2d.f90:266-267': ['IF( ln_dynspg_ts )', 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs', 2],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:482-485': ['IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb)', 4],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:521': ('IF( kstg == 3 )   CALL dyn_zdf', 1),
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:524-549': ['!==  All stages: correct the barotropic component ==!', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb', 26],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3.f90:201-215': ['CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 15],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3.f90:207-215': ['CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )', 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 9],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/stprk3_stg.f90:404-430': ['CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), 27],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:235': ('CALL vor_ene( kt, Kmm, ntot, puu(:,:,:,Kmm)', 1),
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:552-572': [('CASE ( np_CRV )', 2), ('zwz(ji,jj) = zwz(ji,jj) / (e3f_0vor', 1), 21],
    'GYRE_OMIP_L2_P3_R17VORS/BLD/ppsrc/nemo/dynvor.f90:576-591': [('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d', 1), 'pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) - r1_4 * r1_e2v', 16],
    'GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/dynzdf.f90:159-168': ['CALL r14_zdf_pre_row( jj, pvv(:,jj,:,Kaa) )', 'pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b', 10],
    'GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/stprk3_stg.f90:504,532,534': ['CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb', 'CALL r14_finish( kstp, rDt, vv(:,:,1,Kaa) )', 3],
    'GYRE_OMIP_L2_P3_R15BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90:161-184': ['zr14_vertical(:,kj) = pv_no_stress(:,1) - zr14_pre_zdf(:,kj)', 'zr14_sum(:,:,8) = zr14_sum(:,:,8) + pv_final(:,:) - zr14_entry(:,:)', 24],
    'ocean_model_latlon_cgrid.py:15046-15295': [('if _zdf_baroclinic_only:', 3), 'v_no_stress = v_no_stress + _v_bt_mean', 250],
    'ocean_model_latlon_cgrid.py:12254-12271': ['return _NEMOWSMomentumBudgetTrace(', 'total_increment=_surface(state_new.v.data - v0)', 18],
    'GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynzdf.f90:100-105': ['REAL(wp) ::   zWu , zWv', 'REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   ztrdu, ztrdv', 6],
    'GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90:153-163': ['SUBROUTINE r14_zdf_row( kj, pv_no_stress, pv_full )', 'zr14_after_zdf(:,kj) = pv_full(:,1)', 11],
    'GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90:231-247': ['CASE ( 3 )           !==  Stage 3  ==!', 'r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 17],
    'GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/stprk3_stg.f90:399-410': ['CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', 'CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs)', 12],
    'GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90:235': ('CALL vor_ene( kt, Kmm, ntot, puu(:,:,:,Kmm)', 1),
    'GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynvor.f90:536-573': [('CASE ( np_CRV )', 2), 'pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) - r1_4 * r1_e2v', 38],
    'dynhpg.F90:276-295': [('puu(ji,jj,1,Krhs) = zhpi(ji,jj)', 1), ('pvv(ji,jj,jk,Krhs) = zhpj(ji,jj)', 1), 20],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:183-210': ['IF( l_ldfdyn_time                    )   CALL ldf_dyn( kstp, Nbb )', 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 28],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:319-334': ['CASE ( 2 , 3 )    !==  Stage 2 & 3  ==!', ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2), 16],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:387-418': ['CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 32],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/dynzdf.f90:500-528': [('!==  First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1', 2), 'DEALLOCATE( ztrdu, ztrdv )', 29],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90:260-267': ['!             Compute ssh and (uu_b,vv_b)  at N+1  (Kaa)', '&   CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg.f90:231-240': ['SELECT CASE ( nspg )', 'DEALLOCATE( ztrdu , ztrdv )', 10],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/trdini.f90:127-138': ['l_trdtra = .FALSE.', "CALL ctl_warn('The trends diagnostics are a work in progress: they are not yet fully tested or functional')", 12],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/iom.f90:1468-1506': ['SUBROUTINE iom_p2d_sp( cdname, pfield2d )', 'END SUBROUTINE iom_p3d_dp', 39],
    'GYRE_OMIP_L2/BLD/ppsrc/nemo/diawri.f90:524-539': ['CALL histwrite( nid_T, "sozotaux", kt, utau', 'CALL histwrite( nid_W, "votkeavm", kt, avm', 16],
    'GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:431-457': ['zwt(:,1) = 0._wp', 'zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) - ( zwi(ji,jk) + zws(ji,jk) )', 27],
    'GYRE_OMIP_L2_P3_SMW_R7STEP2/BLD/ppsrc/nemo/trazdf.f90:527-559': ['pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,1))) * pt(ji,jj,1,jn,Kbb )', '&             / zwt(ji,jk) * tmask(ji,jj,jk)', 33],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90:92-106': ['zyydd = REAL(nyear_len(1),wp)', '&      - (nyear  - 1) * rjjhh * zyydd', 15],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90:121-145': ['zcos_sais1 = COS( (ztime - ztimemax1) / (ztimemin1 - ztimemax1) * rpi )', '&         * ( 1 - zemp_sais / zemp_N * zcos_sais1 )', 25],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/usrdef_sbc.f90:176-191': ['ztime = REAL( kt, wp ) * rn_Dt / (rmmss * rhhmm)', 'vtau(ji,jj) =   ztaun * SIN( rpi * (gphit(ji,jj) - 15.) / (29.-15.) )', 16],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3.f90:168-216': ['CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 'IF( kstp == nit000 )   CALL l1_dump_stage( kstp, 3, Naa )', 49],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfphy.f90:316-336': [('IF( l_zdfsh2 ) THEN', 1), 'CASE( np_TKE )   ;   CALL zdf_tke( kt, Kbb, Kmm, sh2, avm_k, avt_k )', 21],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfsh2.f90:98-108': [('DO jj = ntsj-( 1), ntej+(  0 ) ; DO ji = ntsi-( 1), ntei+(  0)', 2), '&         * wvmask(ji,jj,jk)', 11],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:215-221': [('CASE ( np_HYB )', 2), ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 2), 7],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:604-634': ['Phase-3 L2 WRITE-only tracer-stage operand stream', ('& WRITE(l1_unit) ts(:,:,:,jp_tem,Krhs), ts(:,:,:,jp_sal,Krhs)', 2), 31],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:396-412': ['CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )', "WRITE(numout,*) 'L2_RK_STAGE2_TERM_DUMP ', kstp, Kmm, Krhs, TRIM(l1_filename)", 17],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:461-462': ['uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) ) * umask(ji,jj,jk)', 'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 2],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:506-526': ['zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 21],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:691-725': ['IF( ln_traqsr  ) THEN', 'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 35],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/trazdf.f90:284-289': ['pt(ji,jj,1,jn,Kaa) =       (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,1))) * pt(ji,jj,1,jn,Kbb )', '& + p2dt * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk))) * pt(ji,jj,jk,jn,Krhs)', 6],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/traqsr.f90:615-645': ['zz0 =           rn_abs   * r1_rho0_rcp       ! surface equi-partition in 2-bands', 'zatt(ji,jj) = zzatt                       ! save for the next level computation', 31],
    'GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90:681-692': ['DO jk =  1,  jpkm1,  1  ; DO ji = ntsi-( 0), ntei+(  0)   !* vertical eddy viscosity & diffivity at w-points', 'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk), avtb_2d(ji,jj) * avtb(jk) ) * wmask(ji,jj,jk)', 12],
    'GYRE_OMIP_L2_P3_DAILY/BLD/ppsrc/nemo/stprk3.f90:90-99': ['IF( lwp .AND. kstp >= nit000 .AND. kstp <= nit000 + 720 .AND.', 'WRITE(itraj) ts(:,:,:,:,Nbb), uu(:,:,:,Nbb), vv(:,:,:,Nbb), ssh(:,:,Nbb)', 10],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3.f90:184-190': ['IF( l_ldfdyn_time                    )   CALL ldf_dyn( kstp, Nbb )', 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 7],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stp2d.f90:178-199': ['Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)*umask(ji,jj,1:jpkm1) ) * r1_hu_0(ji,jj)', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj) * (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kbb)))', 22],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/usrdef_sbc.f90:176-191': ['ztime = REAL( kt, wp ) * rn_Dt / (rmmss * rhhmm)', 'vtau(ji,jj) =   ztaun * SIN( rpi * (gphit(ji,jj) - 15.) / (29.-15.) )', 16],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/sbcmod.f90:550-559': ["CALL lbc_lnk( 'sbcmod', utau, 'T', -1.0_wp, vtau  , 'T', -1.0_wp, emp, 'T', 1.0_wp, qns, 'T',  1.0_wp, ldfull=.TRUE. )", '&                      ( 2. - vmask(ji,jj,1) ) * MAX( tmask(ji,jj,1), tmask(ji,jj+1,1) )', 10],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_nam.F90:64-80': ['NAMELIST/namusr_def/ nn_GYRE, ln_bench, jpkglo', 'kpj = 20 * nn_GYRE + 2', 17],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:84-94': ['zlam1 = -85._wp', 'zphi0 = zphi1 + zsin_alpha * ze1deg * REAL( Nj0glo - 2, wp )', 11],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:119-150': ['DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )', 'pe1f(:,:) =  ze1     ;      pe2f(:,:) = ze1', 32],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_hgr.F90:161-168': ['zbeta = 2. * omega * COS( rad * zphi1 ) / ra', 'pff_t(:,:) = ( zf0 + zbeta * ABS( pphit(:,:) - zphi0 ) * rad * ra )', 8],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90:84-107': ['zyear0     =   ndate0 / 10000', 'zcos_sais2 = COS( (ztime - ztimemax2) / (ztimemax2 - ztimemin2) * rpi )', 24],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90:115-144': ['t_star = zTstar * ( 1. + 1. / 50. * zcos_sais2 )', 'qns (ji,jj) = qns(ji,jj) - emp(ji,jj) * sst_m(ji,jj) * rcp', 30],
    'cfgs/GYRE_OMIP_L2/WORK/usrdef_sbc.F90:161-176': ['ztime = REAL( kt, wp ) * rn_Dt / (rmmss * rhhmm)', 'vtau(ji,jj) =   ztaun * SIN( rpi * (gphit(ji,jj) - 15.) / (29.-15.) )', 16],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:368-380': [('CASE(  32  )       !==  time varying 3D field  ==!', 1), 'esqf(ji,jj) = ( 2._wp * e1e2f(ji,jj) / ( e1f(ji,jj) + e2f(ji,jj) ) )**2', 13],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:387-393': ['IF( .NOT.l_ldfdyn_time ) THEN', 'ahmf(:,:,1:jpkm1) = SQRT( ahmf(:,:,1:jpkm1) ) * fmask(:,:,1:jpkm1)', 7],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:462-528': ['CASE(  32  )       !==  time varying 3D field  ==!   = F( local deformation rate and gridscale ) (Smagorinsky)', 'ahmf(ji,jj,jk) = SQRT( r1_8 * esqf(ji,jj) * ahmf(ji,jj,jk) )', 67],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/ldfdyn.f90:512-534': ['ahmf(ji,jj,jk) = MAX( ahmf(ji,jj,jk), SQRT( (zu2pv2_ij + zu2pv2_ij_p1) * zdelta * zstabf_lo ) )', "CALL lbc_lnk( 'ldfdyn', ahmt, 'T', 1.0_wp , ahmf, 'F', 1.0_wp )", 23],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:159-183': ['CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 'IF( l_ldfdyn_time                    )   CALL ldf_dyn( kstp, Nbb )', 25],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3.f90:195-216': ['CALL rk3_dia( 0 )', 'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs', 22],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:260-302': ['!- horizontal transport (zFu,zFv) -!', 'zFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 43],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:453-476': [('!* Update or Compute (VIF) advective transport', 2), 'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 24],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stp2d.f90:174-199': ['!*  vertical averaging  *!', 'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj)', 26],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynspg_ts.f90:284-292': ['CALL dyn_cor_2D_init( Kmm )', 'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 9],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/stprk3_stg.f90:387-398': [('CASE ( 3 )        !==  Stage 3  ==!', 1), 'CALL dyn_dmp( kstp, Kbb, Kmm', 12],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:604-634': ['Phase-3 L2 WRITE-only tracer-stage operand stream', ('& WRITE(l1_unit) ts(:,:,:,jp_tem,Krhs), ts(:,:,:,jp_sal,Krhs)', 2), 31],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:650-675': ['stage 3     : time-stepping with left over RHS trends', "WRITE(numout,*) 'L2_RK_TRACER_OPERAND_DUMP '", 26],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:218-229': ['SUBROUTINE dynldf_lev_blp( kt, Kbb, Kmm, pu, pv, Krhs )', '!! ** Action :   pt(:,:,:,:,Krhs)   updated with the before rotated bilaplacian diffusion', 12],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:271-290': [('DO jj = ntsj-( 1), ntej+(  1+1 ) ; DO ji = ntsi-( 1), ntei+(  1+1)', 1), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)                      )', 2), 20],
    'cfgs/GYRE_OMIP_L2/BLD/ppsrc/nemo/dynldf_lev.f90:311-330': [('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 4), ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)                      )', 3), 20],
    'nemo_testcase_recipe.py:4693': 'def build_nemo_testcase_card(',
    'ocean_model_latlon_cgrid.py:15868': 'def _step_jitted(',
    'nemo_testcase_l2_gyre_gradient_gate.py:292': 'def build_context(card: str, *, transcendentals: str = "libm") -> GateContext:',
    'nemo_testcase_l2_gyre_gradient_gate.py:342,356,365': ['def _production_step(context, state, kt: int):', 'def gradient_rollout(context: GateContext, perturbation, n_steps: int):', 'def scheduled_rollout(context: GateContext, perturbation, n_steps: int,', 3],
    'nemo_testcase_l2_gyre_gradient_gate.py:540': 'def explicit_stepwise_directional_derivatives(',
    'nemo_testcase_l2_gyre_gradient_gate.py:1408': 'def measure_resources(card: str, n_steps: int, schedule: str, *,',
    'nemo_testcase_l2_gyre_gradient_gate.py:1230': 'def classify_resource_row(row: dict) -> str:',
    'cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:295-298': ['! FCT at last stage only with RK3', ('ENDIF', 9), 4],
    'cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:370-380': ['SELECT CASE ( nadv )                      !==  compute advection trend and add it to general trend  ==!', ('ENDIF', 15), 11],
    'cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:137-149,203-216': [('CASE(  2  )                         !* 2nd order centered', 1), ('END_2D', 2), ('CASE(  2  )                         !* 2nd order centered', 2), ('&                                        / e3t(ji,jj,jk,Kmm)', 4), 27],
    'cfgs/DINO/MY_SRC/zdftke.F90:832-837': ['DO_2Dik( 0, 0, 1, jpkm1, 1 )   !* vertical eddy viscosity & diffivity at w-points', 'dissl(ji,jj,jk) = zsqen / zmxld(ji,jk)', 6],
    'phycst.F90:29': 'rday     = 24.*60.*60.',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:199-201': ['NAMELIST/namctl/ sn_cfctl', 'NAMELIST/namcfg/ ln_read_cfg', 3],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:292-296': ['IF( ln_read_cfg ) THEN', ('ENDIF', 6), 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:402': 'IF( .NOT.ln_read_cfg )   ln_closea = .false.',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:332-352': ['CALL     dom_init( Nbb, Nnn, Naa )', 'CALL dyn_ldf_init', 21],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:282-288': ['NAMELIST/namrun/ cn_ocerst_indir', 'NAMELIST/namtile/ ln_tile, nn_ltile_i, nn_ltile_j', 7],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:282-289': ['NAMELIST/namrun/ cn_ocerst_indir', 'NAMELIST/namnc4/ nn_nchunks_i, nn_nchunks_j, nn_nchunks_k, ln_nc4zip', 8],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:354-358': ['IF( ln_rst_list ) THEN', ('ENDIF', 10), 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:253': 'IF( ln_write_cfg   )   CALL cfg_write',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:196': 'IF( ln_closea ) CALL dom_clo',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:371': 'IF( ln_top )    ln_top = .FALSE.',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:446-472': ['IF( ln_tile ) THEN', ('ENDIF', 24), 27],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:493-496': ['snc4set%ni   = nn_nchunks_i', 'snc4set%luse = ln_nc4zip', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domhgr.f90:101-117': ['IF( ln_read_cfg ) THEN        !==  read in domcfg file  ==!', 'CALL usr_def_hgr( glamt , glamu , glamv , glamf', 17],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domzgr.f90:111-117': ['IF( ln_read_cfg ) THEN        !==  read in domcfg.nc file  ==!', 'CALL iom_open( cn_domcfg, inum )    ! open domcfg file', 7],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domtile.f90:101-119': ['IF( ln_tile ) THEN            ! Calculate number of tiles', 'IF( ln_tile ) THEN            ! Calculate tile domain indices', 19],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/restart.f90:103-110': ['IF( kt == nit000 ) THEN   ! default definitions', ('ENDIF', 1), 8],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/restart.f90:197-211': ['IF( kt == nitrst ) THEN', ('ENDIF', 14), 15],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:332-333': ['READ(numnam_ref(MAX(INDEX(numnam_ref,"&namrun ")', 'READ(numnam_cfg(MAX(INDEX(numnam_cfg,"&namrun ")', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:374-377': ['cexper = cn_exp', 'nitend = nn_itend', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:416-426': ['SELECT CASE ( nleapy )', 'END SELECT', 11],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/nemogcm.f90:147-160': ['istp = nit000', 'END DO', 14],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:123-126': ['!! ** Method  :   At the beginning of the first time step', '!!      Each nn_write time step, output the instantaneous or mean fields', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:156-160': ['! Define frequency of output and means', 'zout = REAL( nn_write, wp ) * rn_Dt', 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:258-259': ['CALL histdef( nid_T, "sossheig"', ('jpi, jpj, nh_T, 1  , 1, 1  , -99 , 32, clop, zsto, zout )', 3), 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:363-374': ['CALL histdef( nid_U, "vozocrtx"', ('jpi, jpj, nh_V, jpk, 1, jpk, nz_V, 32, clop, zsto, zout )', 1), 12],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:457': 'CALL histwrite( nid_T, "sossheig", kt, ssh(:,:,Kmm)',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/diawri.f90:536-538': ['CALL histwrite( nid_U, "vozocrtx", kt, uu(:,:,:,Kmm)', 'CALL histwrite( nid_V, "vomecrty", kt, vv(:,:,:,Kmm)', 3],
    'ext/IOIPSL/src/mathelp.f90:3086-3119': ['SUBROUTINE moycum (opp,np,px,py,pwx)', 'END SUBROUTINE moycum', 34],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:308-321': [('!                                 !==  Swap time levels  ==!', 3), 'IF( lrst_oce   )   CALL rst_write', 14],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/restart.f90:103-129': ['IF( kt == nit000 ) THEN', 'IF( nitrst <= nitend .AND. nitrst > 0 ) THEN', 27],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/iom_nf90.f90:684-705': ['lchunk = .false.', 'CALL iom_nf90_check(SET_NF90_DEF_VAR_CHUNKING', 22],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/mppini.f90:200-205': ['! do we need to take into account bdy_msk?', 'IF( ln_bdy .AND. ln_mask_file ) CALL iom_open( cn_mask_file, numbdy )', 6],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/mppini.f90:204-205': ['IF(               ln_read_cfg ) CALL iom_open( cn_domcfg,    numbot )', 'IF( ln_bdy .AND. ln_mask_file ) CALL iom_open( cn_mask_file, numbdy )', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/bdyini.f90:164-189': ['IF( nb_bdy == 0 ) ln_bdy = .FALSE.', ('ENDIF', 1), 26],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sshwzv.f90:152-156': ['IF ( .NOT.ln_dynspg_ts ) THEN', ('ENDIF', 3), 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sshwzv.f90:221-225': [('IF( ln_bdy ) THEN', 2), ('ENDIF', 7), 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sshwzv.f90:311-315': [('IF( ln_bdy ) THEN', 3), ('ENDIF', 12), 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sbcmod.f90:157-163': ['NAMELIST/namsbc/ nn_fsbc', '&             ln_wave  , nn_lsm', 7],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:149': 'NAMELIST/namlbc/ rn_shlat, ln_vorlat',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:149-155': ['NAMELIST/namlbc/ rn_shlat, ln_vorlat', '&             ln_vol, nn_volctl, nn_rimwidth', 7],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:192-203': ['! Mask corrections for bdy (read in mppini2)', ('ENDIF', 4), 12],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dommsk.f90:286-303': ['IF( ln_read_cfg ) THEN        !==  read in domcfg file  ==!', ("CALL lbc_lnk( 'dommsk', fmask, 'F', 1._wp )", 2), 18],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sbc_oce.f90:116-117': ['utauU  , utau_b', 'vtauV  , vtau_b', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sbcmod.f90:567-568': ["'utau_b', utau_b, cd_type = 'U'", "'vtau_b', vtau_b, cd_type = 'V'", 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domqco.f90:303-306': ['NAMELIST/nam_vvl/ ln_vvl_zstar', '&              nn_vvl_interp', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynadv.f90:162': 'NAMELIST/namdyn_adv/ ln_dynadv_OFF',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:813-814': ['NAMELIST/namdyn_vor/ ln_dynvor_ens', '&                 ln_dynvor_een, nn_e3f_typ', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-180': ['NAMELIST/namdyn_ldf/ ln_dynldf_OFF', '&                 rn_csmc      , rn_minfac    , rn_maxfac', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-166': ['ze1 =  rn_dx / REAL(nn_SWG, wp)', 'ldIperio = .FALSE.   ;   ldJperio = .FALSE.', 13],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:56,143': ['REAL(wp), PUBLIC ::   rn_abp', 'rn_abp, nn_cnp, rn_fsp', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:57,143': ['INTEGER , PUBLIC ::   nn_cnp', 'rn_abp, nn_cnp, rn_fsp', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:58,143': ['REAL(wp), PUBLIC ::   rn_fsp', 'rn_abp, nn_cnp, rn_fsp', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:123-168': ['DO jj = 1, jpj', 'pff_t(:,:) =  REAL( rn_f0, wp ) + REAL( rn_beta, wp ) * ABS( pphit(:,:) )', 46],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:158-168': ['pdepw_1d(1) =   0._wp', 'CALL e3_to_depth( pe3t_1d, pe3w_1d, pdept_1d, pdepw_1d )', 11],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:206-235': ['z2d(:,:) = REAL( jpkm1 , wp )', ('k_bot(ji,jj) = 0', 2), 30],
    'SWG_DG/BLD/ppsrc/nemo/usrdef_zgr.f90:211,213,220-221': ['zylim1 = rn_domsiz + 10000._wp', 'zxlim1 = rn_domsiz + 10000._wp', 'IF ( gphiv(ji,jj) > zylim0', '& glamu(ji,jj) > zxlim0', 4],
    'SWG_DG/BLD/ppsrc/nemo/stprk3.f90:308-321': [('!                                 !==  Swap time levels  ==!', 3), 'IF( lrst_oce   )   CALL rst_write', 14],
    'SWG_DG/BLD/ppsrc/nemo/stprk3.f90:140-171': ['rDt   = rn_Dt / 3._wp', ('vv(ji,jj,jk,Naa) = vv(ji,jj,jk,Nbb) + rDt * vv(ji,jj,jk,Nrhs)', 1), 32],
    'SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-210': [("CALL lbc_lnk( 'stp_RK3', uu (:,:,:,Naa)", 1), ('Naa = Nrhs', 1), 7],
    'SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:220-235': [('IF( kstp <= nit000 + 9 ) THEN', 2), ('CALL swg_close( kstp, 2 )', 1), 16],
    'SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:238-255': [('uu(:,:,:,Nrhs) = 0._wp', 2), ('CALL ssh_nxt( kstp, Nbb, Nnn, ssh, Naa )', 1), 18],
    'SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:360,366,376': [("CALL lbc_lnk( 'stp_RK3', uu (:,:,:,Naa)", 3), 'Nbb = Naa', 'CALL rst_write    ( kstp, Nbb, Nbb )', 3],
    'SWG_DG/BLD/ppsrc/nemo/divhor.f90:134-140': [('DO jk =  1,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 1), 7],
    'SWG_DG/BLD/ppsrc/nemo/sshwzv.f90:135-149': ['CALL div_hor( kt, Kbb, Kmm )', "CALL lbc_lnk( 'sshwzv', pssh(:,:,Kaa)", 15],
    'SWG_DG/BLD/ppsrc/nemo/diawri.f90:457,536-538': ['CALL histwrite( nid_T, "sossheig", kt, ssh(:,:,Kmm)', 'CALL histwrite( nid_U, "vozocrtx", kt, uu(:,:,:,Kmm)', 'CALL histwrite( nid_V, "vomecrty", kt, vv(:,:,:,Kmm)', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:84-87': ['emp (:,:) = 0._wp', 'qsr (:,:) = 0._wp', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_sbc.f90:99-106': ['ztauu =   REAL( rn_tau, wp ) * COS( rn_theta * rad )', 'vtau(ji,jj) = - ztauv * COS( rpi * gphit(ji,jj) / 2000000._wp)', 8],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:250-251': [('CASE( np_ENS )', 2), 'CALL vor_ens( kt, Kmm, ntot, puu(:,:,:,Kmm)', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675': [('CASE ( np_CRV )', 3), 'pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zvau', 37],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynzad.f90:111-146': ['zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp', '&                                              * zWdzV(ji,jj)', 36],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140,196,252': ['rDt   = rn_Dt / 3._wp', 'rDt   = rn_Dt / 2._wp', ('rDt   = rn_Dt', 3), 3],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:148-149': ['CALL dyn_adv( kstp, Nbb, Nbb, uu, vv, Nrhs )', 'CALL dyn_vor( kstp,      Nbb, uu, vv, Nrhs )', 2],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140-171': ['rDt   = rn_Dt / 3._wp', ('vv(ji,jj,jk,Naa) = vv(ji,jj,jk,Nbb) + rDt', 1), 32],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273': [('CALL dyn_adv( kstp, Nbb, Nnn, uu, vv, Nrhs )', 2), '&            - rn_rfr   * vv(ji,jj,jk,Nbb)', 14],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:281-283': [('CALL ssh_nxt( kstp, Nbb, Nnn, ssh, Naa )', 2), ('CALL dom_qco_r3c( ssh(:,:,Naa), r3t(:,:,Naa), r3u(:,:,Naa), r3v(:,:,Naa), r3f(:,:) )', 3), 3],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321': ['CALL sbc     ( kstp, Nbb, Nnn )', 'CALL rst_write    ( kstp, Nbb, Nbb )', 200],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:154-160': ['ze1 =  rn_dx / REAL(nn_SWG, wp)', 'kpj = ceiling(zbl / ze1 )', 7],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:95,150-156': ['ze1 =  rn_dx / REAL(nn_SWG, wp)', '!                       !== Horizontal scale factors ==! (in meters)', 'pe1f(:,:) =  ze1     ;      pe2f(:,:) = ze1', 8],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:123-146': ['DO jj = 1, jpj', 'pphif(ji,jj) = + zim1  * ze1 * zsin_theta', 24],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_hgr.f90:164-171': ['!                       !==  Coriolis parameter  ==!', "beta-plane used. beta = ', rn_beta", 8],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:210-228': ['zylim0 =   10000._wp', ('END DO   ;   END DO', 1), 19],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:229-235': ['! mask the lonely corners', ('k_bot(ji,jj) = 0', 2), 7],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:157-168': ['! depth at T and W-points', 'CALL e3_to_depth( pe3t_1d, pe3w_1d, pdept_1d, pdepw_1d )', 12],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_zgr.f90:206-208': ['z2d(:,:) = REAL( jpkm1 , wp )', "CALL lbc_lnk( 'usrdef_zgr', z2d, 'T', 1._wp )", 3],
    'src/OCE/DYN/dynspg.F90:238-239': ['IF( ln_dynspg_exp .AND. lk_RK3 )', "CALL ctl_stop( ' dynspg_exp not implemented with RK3", 2],
    'tests/SWG/MY_SRC/usrdef_istate.F90:59-61': ['pu  (:,:,:)   = 0._wp', 'pts (:,:,:,:) = 0._wp', 3],
    'tests/SWG/MY_SRC/usrdef_istate.F90:69-87': ['SUBROUTINE usr_def_istate_ssh( ptmask, pssh )', 'END SUBROUTINE usr_def_istate_ssh', 19],
    'src/SWE/stprk3.F90:112': 'rDt   = rn_Dt / 3._wp',
    'src/SWE/stprk3.F90:178': 'rDt   = rn_Dt / 2._wp',
    'src/SWE/stprk3.F90:244': ('rDt   = rn_Dt', 3),
    'src/SWE/stprk3.F90:120': 'CALL dyn_adv( kstp, Nbb, Nbb, uu, vv, Nrhs )',
    'src/SWE/stprk3.F90:185': ('CALL dyn_adv( kstp, Nbb, Nnn, uu, vv, Nrhs )  ! advection (VF or FF)', 1),
    'src/SWE/stprk3.F90:252': ('CALL dyn_adv( kstp, Nbb, Nnn, uu, vv, Nrhs )  ! advection (VF or FF)', 2),
    'src/OCE/DYN/dynadv.F90:79-82': [('CASE( np_VEC_c2  )', 1), 'CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynzad.f90:111-114': ['zWdzU(ntsi-(0):ntei+(0),ntsj-(0):ntej+(0)) = 0._wp', 'DO jk =  1,  jpk-2', 4],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynzad.f90:141-146': ['jk = jpkm1', '&                                              * zWdzV(ji,jj)', 6],
    'src/OCE/DYN/dynvor.F90:72,135': ['INTEGER, PUBLIC, PARAMETER ::   np_ENS = 0', 'CASE( np_ENS )           ;   CALL vor_ens', 2],
    'src/OCE/DYN/dynvor.F90:539-669': ['SUBROUTINE vor_ens( kt, Kmm, kvor, pu, pv, pu_rhs, pv_rhs )', 'END SUBROUTINE vor_ens', 131],
    'src/OCE/DYN/dynvor.F90:67': 'LOGICAL, PUBLIC ::   ln_dynvor_msk',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:149,204,261': ['CALL dyn_vor( kstp,      Nbb, uu, vv, Nrhs )', ('CALL dyn_vor( kstp,      Nnn, uu, vv, Nrhs )', 1), ('CALL dyn_vor( kstp,      Nnn, uu, vv, Nrhs )', 2), 3],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:201-228': ['IF( l_trddyn ) THEN', 'CALL trd_dyn( ztrdu, ztrdv, jpdyn_rvo, kt, Kmm )', 28],
    'src/SWE/stprk3.F90:254': 'CALL dyn_ldf( kstp, Nbb, Nnn, uu, vv, Nrhs )',
    'src/SWE/stprk3.F90:122-124': [('#if defined key_RK3all', 1), ('#endif', 1), 3],
    'src/SWE/stprk3.F90:187-189': [('#if defined key_RK3all', 3), ('#endif', 3), 3],
    'src/SWE/stprk3.F90:128-129': ['zrhs_u =        - grav    * ( ssh(ji+1,jj,Nbb)', 'zrhs_v =        - grav    * ( ssh(ji,jj+1,Nbb)', 2],
    'src/SWE/stprk3.F90:194-195': ['zrhs_u =        - grav    * ( ssh(ji+1,jj,Nnn)', 'zrhs_v =        - grav    * ( ssh(ji,jj+1,Nnn)', 2],
    'src/SWE/stprk3.F90:259-260': ['zrhs_u =        - grav * ( ssh(ji+1,jj,Nnn)', 'zrhs_v =        - grav * ( ssh(ji,jj+1,Nnn)', 2],
    'tests/SWG/MY_SRC/usrdef_nam.F90:109': 'IF( rn_modified_grav /= 0._wp) grav = rn_modified_grav',
    'src/SWE/stprk3.F90:262-265': ['zrhs_u = zrhs_u + z1_2rho0 * ( utau_b(ji,jj) + utauU(ji,jj) )', '&            - rn_rfr   * vv(ji,jj,jk,Nbb)', 4],
    'src/OCE/SBC/sbcmod.F90:570': ('utau_b(:,:) = utauU(:,:)', 2),
    'src/SWE/stprk3.F90:138-139': [('uu(ji,jj,jk,Nrhs) = uu(ji,jj,jk,Nrhs) + zrhs_u', 1), ('vv(ji,jj,jk,Nrhs) = vv(ji,jj,jk,Nrhs) + zrhs_v', 1), 2],
    'src/SWE/stprk3.F90:144': 'CALL ssh_nxt( kstp, Nbb, Nbb, ssh, Naa )',
    'src/SWE/stprk3.F90:210': ('CALL ssh_nxt( kstp, Nbb, Nnn, ssh, Naa )', 1),
    'src/SWE/stprk3.F90:273': ('CALL ssh_nxt( kstp, Nbb, Nnn, ssh, Naa )', 2),
    'src/OCE/DYN/sshwzv.F90:75-147': ['SUBROUTINE ssh_nxt( kt, Kbb, Kmm, pssh, Kaa )', 'END SUBROUTINE ssh_nxt', 73],
    'src/OCE/DYN/sshwzv.F90:123': ('pssh(ji,jj,Kaa) = (  pssh(ji,jj,Kbb) - rDt', 1),
    'SWG_OMIP_L1/BLD/ppsrc/nemo/sshwzv.f90:145-147': ['DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', ('END DO   ;   END DO', 2), 3],
    'src/SWE/stprk3.F90:146': ('CALL dom_qco_r3c( ssh(:,:,Naa)', 1),
    'src/SWE/stprk3.F90:212': ('CALL dom_qco_r3c( ssh(:,:,Naa)', 2),
    'src/SWE/stprk3.F90:275': ('CALL dom_qco_r3c( ssh(:,:,Naa)', 3),
    'src/SWE/stprk3.F90:150-153': [('IF( ln_dynadv_vec ) THEN', 1), ('vv(ji,jj,jk,Naa) = vv(ji,jj,jk,Nbb) + rDt', 1), 4],
    'src/SWE/stprk3.F90:167-168': [("CALL lbc_lnk( 'stp_RK3', uu (:,:,:,Naa)", 1), ("CALL lbc_lnk( 'stp_RK3', r3u(:,:,  Naa)", 1), 2],
    'tests/SWG/MY_SRC/usrdef_nam.F90:82-87': ['NAMELIST/namusr_def/ nn_SWG, rn_theta, jpkglo', '&                 rn_abp, nn_cnp, rn_fsp', 6],
    'tests/SWG/MY_SRC/usrdef_nam.F90:101': 'ze1 =  rn_dx / REAL(nn_SWG, wp)',
    'tests/SWG/MY_SRC/usrdef_nam.F90:117': 'r1_abp = 1._wp / rn_abp',
    'SWG_OMIP_L1/BLD/ppsrc/nemo/usrdef_nam.f90:56-60,143': ['REAL(wp), PUBLIC ::   rn_abp', 'REAL(wp), PUBLIC ::   r1_abp', '&                 rn_abp, nn_cnp, rn_fsp', 6],
    'tests/SWG/MY_SRC/usrdef_hgr.F90:84-145': ['ze1 =  rn_dx / REAL(nn_SWG, wp)', 'pe1f(:,:) =  ze1     ;      pe2f(:,:) = ze1', 62],
    'tests/SWG/MY_SRC/usrdef_hgr.F90:154-157': ['kff = 1', 'pff_t(:,:) =  REAL( rn_f0, wp ) + REAL( rn_beta, wp )', 4],
    'tests/SWG/MY_SRC/usrdef_hgr.F90:100-105': ['zlam1 =  zbl * COS((rn_theta + 45 )* rad )', 'zphi0 = REAl( anint( zphi1 / ze1 ), wp ) * ze1', 6],
    'tests/SWG/MY_SRC/usrdef_zgr.F90:143-147': ['pdepw_1d(1) =   0._wp', 'pdept_1d(2) = 750._wp', 5],
    'tests/SWG/MY_SRC/usrdef_zgr.F90:191': 'z2d(:,:) = REAL( jpkm1 , wp )',
    'tests/SWG/MY_SRC/usrdef_zgr.F90:195-206': ['zylim0 =   10000._wp', '& glamu(ji,jj) > zxlim0 .AND. glamu(ji,jj) < zxlim1', 12],
    'tests/SWG/MY_SRC/usrdef_zgr.F90:214-222': ['! mask the lonely corners', ('END_2D', 2), 9],
    'tests/SWG/MY_SRC/usrdef_sbc.F90:84-106': ['ztauu =   REAL( rn_tau, wp ) * COS( rn_theta * rad )', 'END SUBROUTINE usrdef_sbc_ice_tau', 23],
    'ocean_pe_latlon_cgrid.py:2714-2718': ['if vorticity_scheme not in (', 'f"\'al81\', \'ene\', \'ene_total\', \'ens_total\', \'een_total\', or \'een_planetary\'"', 5],
    'nemo_recipe.py:58': 'NEMO_CONSTANTS_CONFIG = _CANONICAL_NEMO_CONSTANTS',
    'nemo_testcase_recipe.py:3775-4090': ['class TsunamiResolvedNamelist(NamedTuple):', ('return card', 7), 316],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:282-286': ['NAMELIST/namrun/ cn_ocerst_indir', '&             ln_cfmeta, ln_xios_read, nn_wxios, ln_top', 5],
    'SWG_OMIP_L1/BLD/ppsrc/nemo/domain.f90:288': 'NAMELIST/namtile/ ln_tile, nn_ltile_i, nn_ltile_j',
    'SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-139': [('SELECT CASE( n_dynadv )', 1), ('         !', 13), 6],
    'SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:207-234': [('DO jj = ntsj-( nn_hls), ntej+(  nn_hls ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls)', 1), ('END SUBROUTINE dom_qco_r3c', 1), 28],
    'SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:170-176': [('DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)', 1), ('vv(ji,jj,jk,Nrhs) = vv(ji,jj,jk,Nrhs) + zrhs_v', 1), 7],
    'tests/TSUNAMI/MY_SRC/usrdef_nam.F90:92-93': ['kpi = NINT( rn_domszx / rn_dx ) + 1', 'kpj = NINT( rn_domszy / rn_dy ) + 1', 2],
    'tests/TSUNAMI/MY_SRC/usrdef_nam.F90:98': 'kpk = 2',
    'tests/TSUNAMI/MY_SRC/usrdef_hgr.F90:79-80': ['ii0 = NINT( REAL(Ni0glo, wp) * rn_0xratio )', 'ij0 = NINT( REAL(Nj0glo, wp) * rn_0yratio )', 2],
    'tests/TSUNAMI/MY_SRC/usrdef_hgr.F90:93-94': ['plamt(ji,jj) = rn_dx *   zti', 'plamu(ji,jj) = rn_dx * ( zti + 0.5_wp )', 2],
    'tests/TSUNAMI/MY_SRC/usrdef_hgr.F90:121-122': ['CASE(0)', ('zf0   = 2._wp * omega * SIN( rad * rn_ppgphi0 )', 1), 2],
    'tests/TSUNAMI/MY_SRC/usrdef_zgr.F90:128': 'zd = rn_domszz/REAL(jpkm1,wp)',
    'tests/TSUNAMI/MY_SRC/usrdef_zgr.F90:187-189': ['k_bot(:,:) = jpkm1', 'k_top(:,:) = 1', 3],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:65-66': ['pts(:,:,:,jp_tem) = 20._wp', 'pts(:,:,:,jp_sal) = 30._wp', 2],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:94-101': ['zdist(ji,jj) = SQRT( glamt(ji,jj)**2 + gphit(ji,jj)**2 )', 'pssh(ji,jj) = 0.1 * COS( zdist(ji,jj) / zmax * rpi * 0.5_wp )', 8],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:96': 'zmax = MAXVAL( zdist ) / 20._wp',
    'tests/TSUNAMI/MY_SRC/usrdef_sbc.F90:60-65': ['utau(:,:) = 0._wp', 'emp (:,:) = 0._wp', 6],
    'tests/TSUNAMI/MY_SRC/stpmlf.F90:111': 'CALL sbc     ( kstp, Nbb, Nnn )',
    'tests/TSUNAMI/MY_SRC/stpmlf.F90:118': 'CALL dom_qco_r3c( ssh(:,:,Naa), r3t(:,:,Naa)',
    'tests/TSUNAMI/MY_SRC/stpmlf.F90:126': 'CALL dyn_spg( kstp, Nbb, Nnn, Nrhs, uu, vv, ssh, uu_b, vv_b, Naa )',
    'tests/TSUNAMI/MY_SRC/stpmlf.F90:131-134': ['Nrhs = Nbb', 'Naa = Nrhs', 4],
    'nemogcm.F90:186': 'CALL stp_MLF( istp )',
    'dynspg_ts.F90:202': 'll_init     = ll_bt_av',
    'dynspg_ts.F90:355': ('CALL dyn_cor_2D_init( Kmm )', 2),
    'dynspg_ts.F90:486': 'sshn_e(:,:) =    pssh (:,:,Kmm)',
    'dynspg_ts.F90:490': 'hu_e  (:,:) =    hu(:,:,Kmm)',
    'dynspg_ts.F90:920': 'un_adv(ji,jj) = r1_2 * ( ub2_b(ji,jj) + zun_save - rn_atfp * un_bf(ji,jj) )',
    'dynspg_ts.F90:1240': 'nn_e = CEILING( rn_Dt / rn_bt_cmax * zcmax)',
    'domhgr.F90:114': "CALL lbc_lnk( 'dom_hgr', glamt, 'T'",
    'depth_e3.F90:68': 'pe3w_1d( 1 ) = 2._wp * ( pdept_1d(1) - pdepw_1d(1) )',
    'depth_e3.F90:73': 'pe3t_1d(jpk) = 2._wp * ( pdept_1d(jpk) - pdepw_1d(jpk) )',
    'restart.F90:466': 'ssh(:,:,Kaa) = ssh(:,:,Kbb)           !*  set ssh at Kaa',
    'restart.F90:481': 'ssh(:,:,Kmm) = ssh(:,:,Kbb)           !*  set now',
    'domqco.F90:126': 'r3t(:,:,Kmm) = r3t(:,:,Kbb)',
    'domqco.F90:99': 'r3t(:,:,Kaa) = r3t(:,:,Kmm)',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:98': 'CALL sbc        ( kstp, Nbb, Nbb )',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:107': 'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:113': 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:118': 'CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:123': 'CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:125': 'Nrhs = Nbb   ;   Nbb  = Naa   ;   Naa  = Nrhs',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:129': 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)',
    'tests/TSUNAMI/MY_SRC/stprk3.F90:136': 'CALL dia_wri   ( kstp,      Nbb )',
    'tests/TSUNAMI/MY_SRC/stpmlf.F90:37': '#if ! defined key_RK3',
    'tests/TSUNAMI/MY_SRC/diawri.F90:626': ('-99 , 32, clop, zsto, zout', 1),
    'tests/TSUNAMI/MY_SRC/diawri.F90:655': 'CALL histwrite( nid_V, "sovvbaro", kt, vv_b(:,:,Kmm)',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:137-138': [('CALL eos    ( ts, Kbb, rhd )', 1), ('CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )', 1), 2],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/eosbn2.f90:361': ('zh  = ((gdept_1d(jk) ) *(1._wp+r3t(ji,jj,Knn)))', 2),
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynhpg.f90:387': ('zuap = -zcoef0 * ( rhd     (ji+1,jj,1)     + rhd     (ji,jj,1) )   &', 1),
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stp2d.f90:176': 'CASE( np_VEC_c2, np_LIN_dyn )',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynadv.f90:185': 'n_dynadv = np_LIN_dyn',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/traadv.f90:444': 'nadv = np_NO_adv',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414': [('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3t_1d(:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1), ('zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3t_1d(:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)', 1), 2],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:316': 'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:334': ('CALL dyn_adv( kstp, Kmm, Kmm, uu, vv, Krhs, zFu, zFv, zFw )', 2),
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:467': 'ts(:,:,:,jn,Krhs) = 0._wp',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/dynvor.f90:857-860': [('CASE( np_LIN_dyn )', 1), ('ntot = np_COR', 1), 4],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/stprk3_stg.f90:503-505': [('ts(ji,jj,jk,jn,Kaa) = (        ( 1._wp + r3t(ji,jj,Kbb) )*ts(ji,jj,jk,jn,Kbb )', 1), ('/          ( 1._wp + r3t(ji,jj,Kaa) )', 1), 3],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:1868': ('ELSEIF(    l_SelfPerio(jn) ) THEN   ;   ifill(jn,jf) = jpfillperio', 3),
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2026-2033': [('IF(     jn == jpso .OR. jn == jpsw .OR. jn == jpse ) THEN   ! southern side', 4), ('IF( ifill(jn,jf) == jpfillperio ) THEN   ;   ishtj2 = ishtSj(jpso,jf)       ;   isgnj2 =  1', 4), 8],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2041-2046': [('DO jl = 1, ipl(jf)  ;  DO jk = 1, ipk(jf)  ;  DO jj = 1,isizej(jn,jf)  ;  DO ji = 1,isizei(jn,jf)', 13), ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = ptab(jf)%pt4d(ii2,ij2,jk,jl)', 4), 6],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbclnk.f90:2111-2112': [('IF( l_IdoNFold ) THEN', 3), ('CALL lbc_nfd(         ptab, cd_nat, psgn', 3), 2],
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/mppini.f90:432': 'l_SelfPerio(3:4) = l_Jperio .AND. jpnj == 1',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/mppini.f90:582': 'l_IdoNFold = ijn(narea) == jpnj .AND. l_NFold',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:561': 'SUBROUTINE lbc_nfd_dp( ptab, cd_nat, psgn, kfld )',
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:584': ("CASE ( 'T' , 'W' )", 5),
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:639': ("CASE ( 'U' )", 5),
    'TSUNAMI_OMIP_L1_RK3/BLD/ppsrc/nemo/lbcnfd.f90:684': ("CASE ( 'V' )", 5),
    'nemogcm.F90:166': ('CALL stp_RK3', 1),
    'stprk3_stg.F90:143-145': [('ssh (:,:,Kaa) = r2_3 * ssh (:,:,Kbb) + r1_3 * ssha(:,:)', 2), ('vv_b(:,:,Kaa) = va_b(:,:)', 2), 3],
    'stprk3_stg.F90:158': "CALL lbc_lnk( 'stp_RK3_stg', r3ua, 'U', 1._wp, r3va, 'V', 1._wp, r3fa, 'F', 1._wp )",
    'stprk3_stg.F90:224-226': [('ssh (:,:,Kaa) = ssha(:,:)', 3), ('vv_b(:,:,Kaa) = va_b(:,:)', 5), 3],
    'stprk3_stg.F90:552-554': ['( 1._wp + r3t(ji,jj,Kbb) )*ts(ji,jj,jk,jn,Kbb )', '/          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'stprk3_stg.F90:636': ("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp", 2),
    'stp2d.F90:127': 'CALL eos    ( ts, Kbb, rhd )',
    'stp2d.F90:151': 'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)',
    'dynadv.F90:129': 'n_dynadv = np_LIN_dyn',
    'traadv.F90:461': 'nadv = np_NO_adv',
    'dynspg_ts.F90:245': ('CALL ts_wgt( ll_bt_av, ll_fw_start, icycle, wgtbtp1, wgtbtp2 )', 1),
    'dynspg_ts.F90:787-789': [("CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp", 2), ("ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1), 3],
    'dynspg_ts.F90:854': "CALL lbc_lnk( 'dynspg_ts', un_adv, 'U', -1._wp, vn_adv, 'V', -1._wp )",
    'dynspg_ts.F90:895': "CALL lbc_lnk( 'dynspg_ts', puu_b, 'U', -1._wp, pvv_b, 'V', -1._wp )",
    'barotropic_latlon_cgrid.py:1246-1264': ['# ORCA2 reads ff_f through iom_get(..., kfill=jpfillcopy)',
 'q_south = b(ff_south[..., None] / e3f_south)',
 19],
    'ocean_model_latlon_cgrid.py:2432-2437': ['w_momentum, _, _ = nemo_qco_wzv_operands(',
 'fold_exchange_ratios=bool(config.rk3_vector_raw_vmask), runoff_mass_flux=runoff_mass_flux,',
 6],
    'ocean_model_latlon_cgrid.py:9113-9115': ['_stage3_T_rate = (', ('/ jnp.maximum(_h_live_one_half, 1.0e-10))', 1), 3],
    'ocean_model_latlon_cgrid.py:6801-6899': ['_ws_face_thickness_kbb = None', ('_ws_v_live_mask, _grid, z_coord=_zc)[:2]', 1), 99],
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

_NAME = (r"[A-Za-z0-9_./]+\.(?:F90|f90|h90|py|fcm|xml|sh|patch|c)"
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
          shift_first_only: bool = False, routine_plant: bool = False) -> dict:
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
    routine = None
    if isinstance(value, dict):
        routine = value.get("routine")
        value = value.get("anchors")
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
    if routine is not None:
        expected = "vor_ens" if routine_plant else str(routine)
        enclosing = None
        for line in range(shifted[0] - 1, -1, -1):
            match = re.match(
                r"\s*SUBROUTINE\s+([A-Za-z0-9_]+)\b", body[line], re.I)
            if match:
                enclosing = match.group(1)
                break
            if re.match(r"\s*END\s+SUBROUTINE\b", body[line], re.I):
                break
        if enclosing is None or enclosing.lower() != expected.lower():
            return {
                "citation": citation,
                "status": "ROUTINE-CONTEXT-MISMATCH",
                "expected_routine": expected,
                "enclosing_routine": enclosing,
            }
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


def run(receipt: Path, heading: str, *, plant: str | None = None,
        context_plant: str | None = None) -> dict:
    document = receipt.read_text()
    if heading not in document:
        raise SystemExit(f"heading {heading!r} not in {receipt}")
    citations = extract(document[document.index(heading):])
    unmapped = [c for c in citations if c not in CITATION_MAP]
    rows = [check(c, CITATION_MAP[c], shift=2 if c == plant else 0,
                  routine_plant=(c == context_plant))
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
        "planted_context": context_plant,
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
    parser.add_argument(
        "--plant-context", metavar="CITATION",
        help="replace this citation's enclosing-routine expectation with "
             "vor_ens; the gate MUST fail")
    args = parser.parse_args(argv)
    if args.plant and args.plant_context:
        parser.error("choose only one citation plant")
    report = run(args.receipt, args.from_heading, plant=args.plant,
                 context_plant=args.plant_context)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    if args.plant or args.plant_context:
        # A planted control exits NONZERO: 1 when it correctly made the gate
        # fail, 2 when it did not fire (which would mean the gate is vacuous).
        return 1 if report["status"] == "FAIL" else 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
