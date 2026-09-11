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
FILES = {
    "stprk3.F90": _OCE / "stprk3.F90",
    "stprk3_stg.F90": _OCE / "stprk3_stg.F90",
    "stp2d.F90": _OCE / "stp2d.F90",
    "oce.F90": _OCE / "oce.F90",
    "DOM/istate.F90": _OCE / "DOM/istate.F90",
    "domzgr_substitute.h90": _OCE / "DOM/domzgr_substitute.h90",
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
    # Round 47 must bind to the actual widened build, not its R41 source.
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90"),
    "GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/domain.f90"),
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
    "state.py": REPO / "packages/ocean/legoesm/ocean/state.py",
    "provenance.py": REPO / "packages/ocean/legoesm/ocean/fidelity/provenance.py",
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
    # --- round 36 paths: THIS ROUND'S OWN BUILD ---
    # The R35TRAZDF card is the one that produced the round-35 record, and it
    # is the only build whose trazdf.f90 carries the instrument, so its line
    # numbers are not R29ZDF's.  Bare lowercase keys, distinct from the
    # shipped .F90 keys above: FILES is an EXACT-key lookup, so nothing here
    # can capture a citation that already resolves elsewhere.
    "trazdf.f90": _R35 / "trazdf.f90",
    "zdf_oce.f90": _R35 / "zdf_oce.f90",
    "stprk3_stg.f90": _R35 / "stprk3_stg.f90",
    "domain.f90": _R35 / "domain.f90",
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
    # --- round 51: live WS-RK3 operand selection and history carry ---
    'ocean_model_latlon_cgrid.py:5280-5281': [
        ('u0 = state.u.data', 2), ('v0 = state.v.data', 2), 2],
    'ocean_model_latlon_cgrid.py:6317-6322': [
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
    'barotropic_latlon_cgrid.py:2063-2085': [
        'elif ab3_hist is not None:', 'etabb0 = eta - deta_bb', 23],
    'barotropic_latlon_cgrid.py:2890-2901': [
        'if _ab3 and not _boxcar_ab3 and hasattr(state, "bt_hist"):',
        '_finals[0] - _finals[12], _finals[0] - _finals[13],', 12],
    # --- round 49: compiled GYRE LDF/ENE statements and call order ---
    'GYRE_OMIP_L2_P3_SM_R46KT2/EXP00/namelist_cfg:165-167': [
        '&namdyn_vor', 'ln_dynvor_ene = .true.', 3],
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-175': [
        'hydrostatic pressure gradient (HPG))',
        "CALL r46_rhs( 'after_zad', uu, vv, Krhs )", 35],
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
    'ocean_pe_latlon_cgrid.py:1884-1918': [
        'elif config.ke_gradient_scheme == "c2":', 'dp_dy = _dKp_dy[..., 1]', 35],
    'ocean_pe_latlon_cgrid.py:3041-3067': [
        'area_w = jax.lax.optimization_barrier(',
        'diag_vertadv_v = jax.lax.optimization_barrier(diag_vertadv_v)', 27],
    'ocean_pe_latlon_cgrid.py:4752-4773': [
        'zad_w, zad_h_u, zad_h_v = w, h_u, h_v',
        'zad_h_v = jax.lax.optimization_barrier(zad_h_v)', 22],
    'ocean_model_latlon_cgrid.py:5976-5986': [
        '_freeze_hpg = self._nemo_ws_test_hooks.freeze_stage_hpg_operands',
        ('getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1), 11],
    'ocean_model_latlon_cgrid.py:6938-6955': [
        ('if (getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1),
        '/ jnp.maximum(_area_v, 1.0e-30))', 18],
    'ocean_model_latlon_cgrid.py:6966-6988': [
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
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:365-371': [
        ('SELECT CASE( kstg )', 2),
        'IF( .NOT.ln_dynadv_vec )   CALL dyn_adv', 7],
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
    'GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:301-308': [
        'Compute ssh and (uu_b,vv_b)  at N+1',
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )', 8],
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
    'state.py:648-666': [
        '# Cross-window barotropic AB3/AM4 substep histories for',
        'bt_hist: object = None', 19],
    'barotropic_latlon_cgrid.py:2054-2085': [
        ('if ab3_za is not None:', 7), 'etabb0 = eta - deta_bb', 32],
    'barotropic_latlon_cgrid.py:2877-2888': [
        ('state_new = state._replace(', 1),
        ('vv_b=state.vv_b.replace(data=V_bar_avg),', 1), 12],
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
    'vertical.py:1565': ('def compute_ocean_jacobian(', 1),
    'ocean_model_latlon_cgrid.py:5316-5320': [
        'transport_velocity = (', ('* _ws_stage_v_mask,', 1), 5],
    'ocean_model_latlon_cgrid.py:6422': ('_g2 = _nemo_ws_stage_transport(', 1),
    'ocean_model_latlon_cgrid.py:6450': (
        '_stage3_hpg_operands = _stage_hpg_operands(', 1),
    'nemo_testcase_l2_gyre_phase3_gate.py:315': (
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
    'stprk3.F90:195': ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1),
    'stprk3.F90:200': ('CALL stp_RK3_stg( 2, kstp, Nbb, Nnn, Nrhs, Naa )', 1),
    'stprk3.F90:207': ('CALL stp_RK3_stg( 3, kstp, Nbb, Nnn, Nrhs, Naa )', 1),
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
    'fidelity/nemo_recipe.py:941': ('def build_nemo_gyre_recipe(', 1),
    'ocean_model_latlon_cgrid.py:4949': (
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
    'ocean_model_latlon_cgrid.py:7057': (
        '_T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]', 1),
    # ROUND 39 moved these two: the before-state slope block added lines
    # above them, so the STATEMENT is unchanged and its line number is not.
    # Re-anchored rather than left to rot, which is what the gate exists for.
    'ocean_model_latlon_cgrid.py:7422': (
        'k33_implicit = compute_isoneutral_K33_latlon(', 1),
    # tracer_combine is READ by two step functions and SELECTED by a DINO
    # recipe -- the retraction of round 37's "a lever nothing selects".
    'dino.py:1677': ('"tracer_combine": "thickness_weighted",', 1),
    'dino.py:3605': ('tracer_combine=cfg.tracer_combine,', 1),
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
    'ocean_model_latlon_cgrid.py:6093': '_vector_velocity_stage_update = (',
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
    'dynadv.F90:144': 'vector form : keg + zad + vor is used',
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
    'domqco.F90:166-169': [
        ('pr3u(ji,jj) = 0.5_wp * (  e1e2t(ji  ,jj) * pssh(ji  ,jj)', 1),
        ('r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1), 4],
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
    'vertical.py:482': 'def nemo_qco_live_face_geometry_cgrid(',
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
    'ocean_model_latlon_cgrid.py:6501-6504': [
        ('u3_corr = u3_raw * _ws_stage_u_mask', 1),
        ('_replace_stage_mean, target_u, target_v)', 1),
        4],
    'ocean_model_latlon_cgrid.py:8151-8177': [
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
    'ocean_model_latlon_cgrid.py:8003-8005': [
        '_nemo_ws_pre_implicit_state = (',
        'if self._nemo_ws_test_hooks.expose_pre_implicit_state else None)',
        3],
    'ocean_model_latlon_cgrid.py:10226': ('u_solve_in = u_solve_in - _u_bt_mean', 1),
    'ocean_model_latlon_cgrid.py:10343': ('u_solve_in = u_solve_in - (', 1),
    'ocean_pe_latlon_cgrid.py:3276-3279': [
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
    # the field is named twice: GYRE's bundle at 226, the tanks' at 311
    'nemo_testcase_recipe.py:311': [
        ('zdf_baroclinic_only=True,', 2), ('zdf_baroclinic_only=True,', 2), 1],
    'provenance.py:96': 'def git_sha(*, allow_dirty: bool = False, repo: str | Path | None = None) -> str:',
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
    'ocean_pe_latlon_cgrid.py:5083': 'rho_prime=rho_prime, h_k=h_k,',
    'ocean_pe_latlon_cgrid.py:5060-5061': [
        '_u_ldf_local = u if ldf_state is None else ldf_state[2]',
        '_v_ldf_local = v if ldf_state is None else ldf_state[3]',
        2],
    'ocean_pe_latlon_cgrid.py:2009-2010': [('z_coord,', 26), ('eta_safe,', 7), 2],
    'ocean_pe_latlon_cgrid.py:2049-2050': ['requires the raw NEMO',
         'nemo_e3w_0 mesh field; midpoint reconstruction on',
         2],
    # ROUND 34: decision 19 added seventeen lines above the last of these
    # three, so 914 became 931.  The two earlier anchors are unmoved.
    'nemo_testcase_recipe.py:92,274,931': [('pgf_scheme="nemo_sco",', 1), ('pgf_scheme="nemo_sco",', 2), 'if cfg.pgf_scheme != "nemo_sco":', 3],
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
