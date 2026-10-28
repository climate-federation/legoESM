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
FILES = {
    # --- SI3 ladder round 1 survey/design BEGIN ---
    "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg": (
        NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg"),
    "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref": (
        NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref"),
    "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref": (
        NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref"),
    "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ice_cfg": (
        NEMO / "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ice_cfg"),
    "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg": (
        NEMO / "cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg"),
    "cfgs/ORCA2_ICE_PISCES/cpp_ORCA2_ICE_PISCES.fcm": (
        NEMO / "cfgs/ORCA2_ICE_PISCES/cpp_ORCA2_ICE_PISCES.fcm"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90": _ORCA2_COMPILED / "icestp.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn.f90": _ORCA2_COMPILED / "icedyn.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90": _ORCA2_COMPILED / "icesbc.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90": _ORCA2_COMPILED / "iceistate.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcssr.f90": _ORCA2_COMPILED / "sbcssr.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbutl.f90": _ORCA2_COMPILED / "icbutl.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_adv_pra.f90": (
        _ORCA2_COMPILED / "icedyn_adv_pra.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90": (
        _ORCA2_COMPILED / "icedyn_rdgrft.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rhg_evp.f90": (
        _ORCA2_COMPILED / "icedyn_rhg_evp.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90": _ORCA2_COMPILED / "icethd.f90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_sal.f90": (
        _ORCA2_COMPILED / "icethd_sal.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_pnd.f90": (
        _ORCA2_COMPILED / "icethd_pnd.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90": (
        _ORCA2_COMPILED / "iceupdate.f90"),
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90": _ORCA2_COMPILED / "icevar.f90",
    "packages/ice/legoesm/ice/state.py": REPO / "packages/ice/legoesm/ice/state.py",
    "packages/ice/legoesm/ice/itd.py": REPO / "packages/ice/legoesm/ice/itd.py",
    "packages/ice/legoesm/ice/dynamics.py": REPO / "packages/ice/legoesm/ice/dynamics.py",
    "packages/ice/legoesm/ice/transport.py": REPO / "packages/ice/legoesm/ice/transport.py",
    "packages/ice/legoesm/ice/ridging.py": REPO / "packages/ice/legoesm/ice/ridging.py",
    "packages/ice/legoesm/ice/config.py": REPO / "packages/ice/legoesm/ice/config.py",
    "packages/ice/legoesm/ice/sea_ice.py": REPO / "packages/ice/legoesm/ice/sea_ice.py",
    "packages/ice/legoesm/ice/experiment_config.py": (
        REPO / "packages/ice/legoesm/ice/experiment_config.py"),
    "packages/ocean/legoesm/ocean/recipes.py": (
        REPO / "packages/ocean/legoesm/ocean/recipes.py"),
    "packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py"),
    "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py": (
        REPO / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"),
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py"),
    "tests/ice/fidelity/test_nemo_si3_phase2_gate.py": (
        REPO / "tests/ice/fidelity/test_nemo_si3_phase2_gate.py"),
    "tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py": (
        REPO / "tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py"),
    "tests/ice/fidelity/test_nemo_si3_phase2_rung33_kernel.py": (
        REPO / "tests/ice/fidelity/test_nemo_si3_phase2_rung33_kernel.py"),
    "tests/ice/fidelity/test_si3_jpl1_ridging.py": (
        REPO / "tests/ice/fidelity/test_si3_jpl1_ridging.py"),
    "tests/ice/unit/test_si3_landfast_l16.py": (
        REPO / "tests/ice/unit/test_si3_landfast_l16.py"),
    "cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ref": (
        NEMO / "cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ref"),
    "cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg": (
        NEMO / "cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg"),
    "sette/input_ORCA2_SAS_ICE.cfg": NEMO / "sette/input_ORCA2_SAS_ICE.cfg",
    "sette/sette_fetch_inputs.sh": NEMO / "sette/sette_fetch_inputs.sh",
    "docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md": (
        REPO / "docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md"),
    "docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md": (
        REPO / "docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md"),
    "docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md": (
        REPO / "docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md"),
    "docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2_manifest.md": (
        REPO / "docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2_manifest.md"),
    "src/SAS/sbcssm.F90": NEMO / "src/SAS/sbcssm.F90",
    "src/OCE/SBC/sbcssm.F90": NEMO / "src/OCE/SBC/sbcssm.F90",
    # --- SI3 ladder round 1 survey/design END ---
    # --- SI3 ladder round 2, round 3 & round 5 (D5) BEGIN ---
    "docs/ice/fidelity/si3_ladder_design.md": (
        REPO / "docs/ice/fidelity/si3_ladder_design.md"),
    "docs/ice/fidelity/si3_r2_d5_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r2_d5_preregister.md"),
    "docs/ice/fidelity/si3_r2_d5_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r2_d5_receipt.md"),
    "docs/ice/fidelity/si3_r3_d5_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r3_d5_preregister.md"),
    "docs/ice/fidelity/si3_r3_d5_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r3_d5_receipt.md"),
    "docs/ice/fidelity/si3_r5_d5_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r5_d5_preregister.md"),
    "docs/ice/fidelity/si3_r5_d5_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r5_d5_receipt.md"),
    "docs/ice/fidelity/si3_r6_d5_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r6_d5_preregister.md"),
    "docs/ice/fidelity/si3_r6_d5_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r6_d5_receipt.md"),
    "packages/core/legoesm/core/source_rounding.py": (
        REPO / "packages/core/legoesm/core/source_rounding.py"),
    "packages/core/legoesm/timestepping/tridiagonal.py": (
        REPO / "packages/core/legoesm/timestepping/tridiagonal.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90"),
    "C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90"),
    "packages/ice/legoesm/ice/bitz_lipscomb.py": (
        REPO / "packages/ice/legoesm/ice/bitz_lipscomb.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py"),
    "src/ICE/icevar.F90": NEMO / "src/ICE/icevar.F90",
    "ORCA2_SAS_ICE_O1_REC/BLD/ppsrc/nemo/icevar.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REC/BLD/ppsrc/nemo/icevar.f90"),
    "C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90"),
    "icevar.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90"),
    "src/ICE/icethd.F90": NEMO / "src/ICE/icethd.F90",
    "src/ICE/icethd_dh.F90": NEMO / "src/ICE/icethd_dh.F90",
    "C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90"),
    "icethd_dh.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90"),
    "tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py": (
        REPO / "tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py"),
    "tests/ocean/fidelity/test_nemo_si3thd_phase2_gate.py": (
        REPO / "tests/ocean/fidelity/test_nemo_si3thd_phase2_gate.py"),
    "tests/ocean/fidelity/test_nemo_si3thd_phase2b_year_gate.py": (
        REPO / "tests/ocean/fidelity/test_nemo_si3thd_phase2b_year_gate.py"),
    "C1D_OMIP_L3/BLD/ppsrc/nemo/sbcssm.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/sbcssm.f90"),
    "sbcssm.f90": (
        NEMO / "cfgs/C1D_OMIP_L3/BLD/ppsrc/nemo/sbcssm.f90"),
    # --- SI3 ladder round 2, round 3 & round 5 (D5) END ---
    # --- SI3 ladder round 4 (O1) BEGIN ---
    "docs/ice/fidelity/si3_r4_o1_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r4_o1_preregister.md"),
    "docs/ice/fidelity/si3_r4_o1_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r4_o1_receipt.md"),
    "packages/ice/legoesm/ice/si3_orca2_state.py": (
        REPO / "packages/ice/legoesm/ice/si3_orca2_state.py"),
    "packages/ice/legoesm/ice/__init__.py": (
        REPO / "packages/ice/legoesm/ice/__init__.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_fixture_stubs.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_fixture_stubs.F90"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_layout_fixture.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_layout_fixture.F90"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py"),
    "tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py": (
        REPO / "tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/namelist_cfg": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/namelist_cfg"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_nemo_restart_fixture.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_nemo_restart_fixture.F90"),
    "src/OCE/IOM/iom.F90": NEMO / "src/OCE/IOM/iom.F90",
    "src/OCE/SBC/sbcmod.F90": NEMO / "src/OCE/SBC/sbcmod.F90",
    "src/OCE/DOM/daymod.F90": NEMO / "src/OCE/DOM/daymod.F90",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid/run.sh": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid/run.sh"),
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2_tke_boundary_acquisition/run.sh": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2_tke_boundary_acquisition/run.sh"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_adv2d_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_adv2d_gate.py"),
    "cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm": (
        NEMO / "cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm"),
    "cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg": (
        NEMO / "cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg"),
    "cfgs/SHARED/namelist_ice_ref": NEMO / "cfgs/SHARED/namelist_ice_ref",
    "cfgs/ref_cfgs.txt": NEMO / "cfgs/ref_cfgs.txt",
    "src/ICE/icedyn_adv_pra.F90": NEMO / "src/ICE/icedyn_adv_pra.F90",
    "src/ICE/iceitd.F90": NEMO / "src/ICE/iceitd.F90",
    "src/ICE/icerst.F90": NEMO / "src/ICE/icerst.F90",
    "src/ICE/icestp.F90": NEMO / "src/ICE/icestp.F90",
    "src/ICE/iceupdate.F90": NEMO / "src/ICE/iceupdate.F90",
    "src/OCE/LBC/lbc_lnk_pt2pt_generic.h90": (
        NEMO / "src/OCE/LBC/lbc_lnk_pt2pt_generic.h90"),
    "src/OCE/LBC/lbc_nfd_generic.h90": (
        NEMO / "src/OCE/LBC/lbc_nfd_generic.h90"),
    "src/ICE/iceistate.F90": NEMO / "src/ICE/iceistate.F90",
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round326_hmi_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round326_hmi_gate.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/ocean.output.excerpt": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/ocean.output.excerpt"),
    # --- SI3 ladder round 4 (O1) END ---
    # --- SI3 ladder round 7 (O2) BEGIN ---
    "docs/ice/fidelity/si3_r7_o2_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r7_o2_preregister.md"),
    "docs/ice/fidelity/si3_r7_o2_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r7_o2_receipt.md"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_state_writer.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_state_writer.F90"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_fixture_stubs.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_fixture_stubs.F90"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_layout_fixture.F90": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_layout_fixture.F90"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/ocean.output.excerpt": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/ocean.output.excerpt"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py"),
    "tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py": (
        REPO / "tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py"),
    "src/OCE/DOM/domutl.F90": NEMO / "src/OCE/DOM/domutl.F90",
    "src/OCE/DOM/domain.F90": NEMO / "src/OCE/DOM/domain.F90",
    # --- SI3 ladder round 10 (O5) ---
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbcnfd.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbcnfd.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbclnk.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbclnk.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90",
    "ORCA2_SAS_ICE_O5_REC/BLD/ppsrc/nemo/icedyn_adv_pra.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O5_REC/BLD/ppsrc/nemo/icedyn_adv_pra.f90",
    "tests/ice/unit/test_si3_o5_strength_aevp.py": REPO / "tests/ice/unit/test_si3_o5_strength_aevp.py",
    # --- SI3 ladder round 9 (O3) ---
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/arch_nemo.fcm": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/arch_nemo.fcm",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/par_ice.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/par_ice.f90",
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py": REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py",
    "tests/ice/unit/test_si3_o3_itd_fit.py": REPO / "tests/ice/unit/test_si3_o3_itd_fit.py",
    "docs/ice/fidelity/si3_r9_o3_receipt.md": REPO / "docs/ice/fidelity/si3_r9_o3_receipt.md",
    "docs/ice/fidelity/si3_r10_o5_receipt.md": REPO / "docs/ice/fidelity/si3_r10_o5_receipt.md",
    "docs/ice/fidelity/si3_r11_o7_receipt.md": REPO / "docs/ice/fidelity/si3_r11_o7_receipt.md",
    "docs/ice/fidelity/si3_r12_o5_acquisition_receipt.md": REPO / "docs/ice/fidelity/si3_r12_o5_acquisition_receipt.md",
    "docs/ice/fidelity/si3_r13_o7_acquisition_receipt.md": REPO / "docs/ice/fidelity/si3_r13_o7_acquisition_receipt.md",
    "docs/ice/fidelity/si3_r13_o7_admission_receipt.md": REPO / "docs/ice/fidelity/si3_r13_o7_admission_receipt.md",
    "ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icesbc.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icesbc.f90",
    "ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceupdate.f90": NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceupdate.f90",
    "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_eos_gate.py": REPO / "scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_eos_gate.py",
    "packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py": REPO / "packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py",
    "src/OCE/DOM/dommsk.F90": NEMO / "src/OCE/DOM/dommsk.F90",
    "src/OCE/SBC/sbcmod.F90": NEMO / "src/OCE/SBC/sbcmod.F90",
    "src/OCE/TRA/eosbn2.F90": NEMO / "src/OCE/TRA/eosbn2.F90",
    "src/OCE/LBC/lib_mpp.F90": NEMO / "src/OCE/LBC/lib_mpp.F90",
    "src/SAS/nemogcm.F90": NEMO / "src/SAS/nemogcm.F90",
    "ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90": (
        NEMO / "cfgs/ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90"),
    # --- SI3 ladder round 7 (O2) END ---
    # --- SI3 ladder round 8 (O4 model prep) BEGIN ---
    "docs/ice/fidelity/si3_r8_o4_preregister.md": (
        REPO / "docs/ice/fidelity/si3_r8_o4_preregister.md"),
    "docs/ice/fidelity/si3_r8_o4_receipt.md": (
        REPO / "docs/ice/fidelity/si3_r8_o4_receipt.md"),
    "tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py": (
        REPO / "tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py"),
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_lane3_prather_dump.py": (
        REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_lane3_prather_dump.py"),
    "packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py": (
        REPO / "packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iom.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iom.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/ice.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/ice.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90"),
    "ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90"),
    "ORCA2_SAS_ICE_O1_REF/EXP00/namelist_ref": (
        NEMO / "cfgs/ORCA2_SAS_ICE_O1_REF/EXP00/namelist_ref"),
    "si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784/ref_2step/ocean.output": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784/ref_2step/ocean.output"),
    # --- SI3 ladder round 8 (O4 model prep) END ---
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
    "stprk3_stg.F90": _OCE / "stprk3_stg.F90",
    "ORCA2_R96SPG/ocean.output": _ORCA2_R96SPG_OUTPUT,
    # round 202: the PRODUCTION GYRE build, so a GYRE resolved statement
    # cannot bind to one of the _SM_ probe builds.
    "GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90": (
        NEMO / "cfgs/GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90"),
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
    "advection.py": REPO / "packages/ocean/legoesm/ocean/advection.py",
    "implicit_solver.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py",
    "internal_wave_mixing.py":
        REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py",
    "tke.py": REPO / "packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py",
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
    # --- SI3 ladder round 11 (O7) ---
    'packages/ice/legoesm/ice/si3_surface.py': REPO / "packages/ice/legoesm/ice/si3_surface.py",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icethd_pnd.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icethd_pnd.f90",
    'packages/core/legoesm/core/transcendentals.py': REPO / "packages/core/legoesm/core/transcendentals.py",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90",
    'tests/ice/unit/test_si3_surface.py': REPO / "tests/ice/unit/test_si3_surface.py",
    'packages/core/legoesm/thermo.py': REPO / "packages/core/legoesm/thermo.py",
    'packages/core/legoesm/core/bulk_flux.py': REPO / "packages/core/legoesm/core/bulk_flux.py",
    'packages/core/legoesm/core/nemo_namelist.py': REPO / "packages/core/legoesm/core/nemo_namelist.py",
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_bulk_flux_gate.py': REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_bulk_flux_gate.py",
    # --- SI3 ladder round 15 (O8) ---
    'ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/ice.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/ice.f90",
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref': REPO / "tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref",
    'docs/ice/fidelity/si3_r15_o8_receipt.md': REPO / "docs/ice/fidelity/si3_r15_o8_receipt.md",
    'docs/ice/fidelity/si3_r15_o8_preregister.md': REPO / "docs/ice/fidelity/si3_r15_o8_preregister.md",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90",
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py': REPO / "tests/ice/unit/test_si3_o9_zdf_dh_temp.py",
    'docs/ice/fidelity/si3_r16_o9_receipt.md': REPO / "docs/ice/fidelity/si3_r16_o9_receipt.md",
    'docs/ice/fidelity/si3_r16_o9_preregister.md': REPO / "docs/ice/fidelity/si3_r16_o9_preregister.md",
    'scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg': REPO / "scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg",
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py': REPO / "tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py",
    # --- SI3 ladder round 19 (O10) ---
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90",
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py': REPO / "scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py",
    'tests/ocean/fidelity/test_nemo_si3_o10_minmax_audit.py': REPO / "tests/ocean/fidelity/test_nemo_si3_o10_minmax_audit.py",
    'tests/ice/unit/test_si3_o10_salinity.py': REPO / "tests/ice/unit/test_si3_o10_salinity.py",
    'scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py': REPO / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py",
    'tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py': REPO / "tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py",
    'docs/ice/fidelity/si3_r19_o10_preregister.md': REPO / "docs/ice/fidelity/si3_r19_o10_preregister.md",
    'docs/ice/fidelity/si3_r19_o10_receipt.md': REPO / "docs/ice/fidelity/si3_r19_o10_receipt.md",
    # --- SI3 ladder round 17 (O11) ---
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/nemogcm.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/nemogcm.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/zdfdrg.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/zdfdrg.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90",
    'packages/ice/legoesm/ice/si3_thd_da.py': REPO / "packages/ice/legoesm/ice/si3_thd_da.py",
    'docs/ice/fidelity/si3_r17_o11_receipt.md': REPO / "docs/ice/fidelity/si3_r17_o11_receipt.md",
    'docs/ice/fidelity/si3_r18_o8_o16_receipt.md': REPO / "docs/ice/fidelity/si3_r18_o8_o16_receipt.md",
    'docs/ice/fidelity/si3_r17_o11_preregister.md': REPO / "docs/ice/fidelity/si3_r17_o11_preregister.md",
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
    # --- SI3 ladder round 1 survey/design BEGIN ---
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:80':
        'nn_fsbc     = 2',
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:28':
        'rn_Dt       = 10800.',
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:80':
        'nn_fsbc     = 2',
    'cfgs/ORCA2_ICE_PISCES/cpp_ORCA2_ICE_PISCES.fcm:1':
        'key_si3 key_top key_xios key_qco key_vco_1d3d key_RK3',
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ice_cfg:22-84': [
        '&nampar', ('/', 20), 63],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:263':
        'ln_Cx_ice_frm = .false.',
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:724':
        'ln_bdy         = .false.',
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:24-32': [
        'jpl              =   5', 'rn_amax_s        =   0.997', 9],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:41-46': [
        'ln_cat_hfn       = .true.', 'rn_himax         =  99.0', 6],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:51-58': [
        'ln_dynALL        = .true.', 'ln_landfast_L16  = .false.', 8],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:76-103': [
        'ln_str_H79       = .true.', 'rn_fpndrft    =   1.0', 28],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:108-131': [
        'ln_rhg_EVP       = .true.', 'nn_UMx        =   5', 24],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:136-150': [
        'rn_Cd_io         =   5.0e-03', 'nn_qtrice        =   0', 15],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:157-174': [
        'ln_icedH         = .true.', 'ln_zdf_chkcvg    = .false.', 18],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:179-194': [
        'rn_beta          =   1.0', 'rn_Cfraz      =   5.0', 16],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:199-236': [
        'nn_icesal        =   4', 'ln_sal_chk       = .FALSE.', 38],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:241-257': [
        'ln_pnd            = .true.', 'rn_pnd_hl_max  =  0.015', 17],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:262-286': [
        'ln_iceini        = .true.', 'rn_hld_ini_s     =   0.0', 25],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:305-311': [
        'rn_alb_sdry      =   0.85', 'rn_alb_hpiv      =   1.00', 7],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:316-320': [
        'ln_icediachk     = .false.', 'ln_icectl        = .false.', 5],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg:22-84': [
        '&nampar', ('/', 20), 63],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186':
        'IF( MOD( kt-1, nn_fsbc ) == 0 ) THEN',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:188-214': [
        'kt_ice = kt', 'CALL ice_rst_opn( kt )', 27],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn.f90:187-194': [
        ('SELECT CASE( nice_dyn )', 1), 'CALL ice_cor       ( kt , 1 )', 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_adv_pra.f90:324-366': [
        'CALL adv_y( ihls, jl, zdt , zvdx , 1._wp , zarea , z0ice',
        'CALL adv_x( ihls, jl, zdt , zudy , 0._wp , zarea , z0vl', 43],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:220-233': [
        'CALL diag_trends( 1 )', ('CALL store_fields', 2), 14],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:181-187': [
        ('IF( vt_i(ji,jj) > epsi20 )', 1),
        ('hm_s(ji,jj) = vt_s(ji,jj) * z1_at_i', 1), 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90:182-245': [
        'CALL ice_thd_frazil', 'CALL ice_cor( kt , 2 )', 64],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:250-275': [
        'CALL ice_sbc_flx( kt, ksbc )', 'CALL ice_ctl( kt )', 26],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:282':
        'CALL ice_update_tau( kt, uu(:,:,1,Kbb), vv(:,:,1,Kbb) )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:373-418': [
        'IF( MOD( kt-1, nn_fsbc ) == 0 ) THEN',
        'vtau(ji,jj) = ( 1._wp - at_i(ji,jj) ) * vtau_oce(ji,jj)', 46],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:145-216': [
        'zqsr = ( 1._wp - at_i_b(ji,jj) ) * qsr_oce(ji,jj)',
        'tn_ice(ji,jj,jl) = t_su(ji,jj,jl)', 72],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:269-283': [
        '! Ice strength',
        'zsshdyn(:,:) = ice_var_sshdyn( ssh_m, snwice_mass, snwice_mass_b)', 15],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:1075-1090': [
        'IF( ln_ice_embd ) THEN',
        ('ice_var_sshdyn(:,:) = pssh(:,:)', 2), 16],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:247-258': [
        'IF( ln_ice_embd ) THEN',
        'DEALLOCATE( zpice )', 12],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:606-629': [
        ('SELECT CASE( nn_mxlice )', 1), 'fr_i(ji,jj)   * zmaxice', 24],
    'packages/ice/legoesm/ice/state.py:33-75': [
        'class SI3ColumnState(NamedTuple):', 'pond_depth: Field', 43],
    'packages/ice/legoesm/ice/itd.py:1-18': [
        '"""Ice Thickness Distribution (ITD)', 'both the area and volume moments are exact', 18],
    'packages/ice/legoesm/ice/dynamics.py:153-209': [
        'class SI3CGridMetrics(NamedTuple):', ('stress12_f: jnp.ndarray', 1), 57],
    'packages/ice/legoesm/ice/transport.py:69-104': [
        "SI3's Prather program carries these five", 'return nemo_source_round(intensives * cell_area', 36],
    'packages/ice/legoesm/ice/ridging.py:84-100': [
        'Selectable SI3 ORCA1/jpl=1 arm', 'first call.  These constants name', 17],
    'packages/ice/legoesm/ice/config.py:291-338': [
        'class SI3ThermoConfig(NamedTuple):', 'invalid SI3 C1D identity', 48],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py:27-52': [
        'NX, NY, NTR, NCLASSES, HALO =', '("rnf_tsc_b", "reduced3d")', 26],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1622-1629': [
        'def build_orca2_zps_card', 'Phase-2 gate stop before their first consumer', 8],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1676-1689': [
        ('grid = create_tripole_grid(', 1), ('native_lat_T_deg=jnp.asarray', 2), 14],
    'packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2091-2107': [
        ('card = NEMOTestcaseCard(', 4), '"si3_jpl5_layered_prather_state"', 17],
    'packages/ice/legoesm/ice/sea_ice.py:89-121': [
        'def grid_supports_ice_dynamics', 'return isinstance(grid, LatLonCGridGeometry)', 33],
    'packages/ice/legoesm/ice/sea_ice.py:329-362': [
        'Grid-TYPE guards: dynamics', 'thermodynamics-only on this grid', 34],
    'packages/ocean/legoesm/ocean/recipes.py:272-377': [
        'def list_recipes(', 'return cfg', 106],
    'packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py:1-7': [
        '"""Reusable NEMO ocean numerics recipe card', 'different domains.', 7],
    'packages/ice/legoesm/ice/experiment_config.py:1-14': [
        '"""YAML experiment adapter', 'shared component-agnostic mechanics', 14],
    'packages/ice/legoesm/ice/experiment_config.py:34-46': [
        'def _ice_matrix_spec()', 'Setup-only YAML adapter', 13],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90:187-195': [
        'npti = 0 ; nptidx(:) = 0', 'IF( npti > 0 ) THEN', 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:268-274': [
        'at_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) = SUM',
        ('nptidx( npti ) = (jj - 1) * jpi + ji', 1), 7],
    'packages/ice/legoesm/ice/dynamics.py:239-264': [
        '@jax.custom_jvp', 'return root, derivative', 26],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:291-312': [
        ('DO ii = 1, npti', 1), 'IF( SUM( apartf(ii,1:jpl) ) > 0._wp', 22],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:341-384': [
        'DO WHILE( iterate_ridging > 0 .AND. iter < jp_itermax )',
        'iter = iter + 1', 44],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1077-1114': [
        'CASE ( np_strh79 )',
        ("CALL lbc_lnk( 'icedyn_rdgrft', strength, 'T', 1.0_wp )", 2), 38],
    'src/SAS/sbcssm.F90:87-121': [
        'IF ( l_sasread ) THEN',
        'ssh_m(:,:) = sf_ssm_2d(jf_ssh)%fnow(:,:,1) * tmask(:,:,1)', 35],
    'src/OCE/SBC/sbcssm.F90:128-173': [
        'ELSEIF( MOD( kt - 2 , nn_fsbc ) == 0 ) THEN',
        'frq_m(:,:) = frq_m(:,:) * zcoef', 46],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_sal.f90:357-414': [
        ('IF( h_i_1d(ji) >= rn_sal_himin', 1), ('jc = jk', 1), 58],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_pnd.f90:203-237': [
        'at_i(:,:) = SUM( a_i, dim=3 )', 'CASE (np_pndLEV)   ;   CALL pnd_LEV', 35],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_pnd.f90:329-381': [
        ('SUBROUTINE pnd_LEV', 1), 'Hunke et al        (OM 2012)', 53],
    'tests/ice/fidelity/test_nemo_si3_phase2_gate.py:58-71': [
        'def test_complete_card_step_jits_and_has_nonzero_finite_reverse_mode_gradient',
        'assert float(jnp.max(jnp.abs(gradient))) > 0.0', 14],
    'tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py:64-74': [
        'def test_complete_card_step_jits_and_has_a_finite_nonzero_gradient',
        'assert float(jnp.max(jnp.abs(gradient))) > 0.0', 11],
    'tests/ice/fidelity/test_nemo_si3_phase2_rung33_kernel.py:233-238': [
        'compiled = jax.jit(objective)',
        'assert float(jnp.max(jnp.abs(gradient))) > 0.0', 6],
    'tests/ice/fidelity/test_si3_jpl1_ridging.py:221-250': [
        'def test_si3_jpl1_jit_and_gradient_are_finite',
        'assert bool(jnp.all(jnp.isfinite(ice_free_gradient)))', 30],
    'tests/ice/unit/test_si3_landfast_l16.py:191-207': [
        'def test_landfast_depth_path_has_finite_nonzero_gradient',
        'jnp.asarray(_PLANT_DEPTH_M, dtype=jnp.float64)', 17],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:199':
        'ln_ice_embd = .false.',
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:259-262': [
        'ln_Cx_ice_cst = .true.', 'rn_Ch_ia  = 1.4e-3', 4],
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ref:398-404': [
        "sn_usp      = 'sas_grid_U'", "sn_frq      = 'sas_grid_T'", 7],
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:28':
        'rn_Dt       = 10800.',
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:61':
        'nn_fsbc     = 1',
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:151-156': [
        '&namsbc_ssr', 'rn_sssr_bnd =   4.e0', 6],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:446-458': [
        '&namsbc_ssr', '>1 = enhanced restoration under ice', 13],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcssr.f90:166-174': [
        'IF( nn_sssr /= 0 .AND. nn_sssr_ice /= 1 ) THEN',
        ('END DO   ;   END DO', 2), 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbutl.f90:168-173': [
        'IF ( PRESENT(pssu) )', 'IF ( PRESENT(pff ) )', 6],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:204-206': [
        '&namberg', 'ln_icebergs = .true.', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:203-250': [
        'Dynamical coupling with the atmosphere', 'CALL ice_sbc_flx( kt, ksbc )', 48],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90:139-166': [
        ('SELECT CASE( ksbc )', 1), 'vtau_ice(ji,jj) = vtau_ice(ji,jj)', 28],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90:201-264': [
        'INTEGER, INTENT(in) ::   kt', 'END SUBROUTINE ice_sbc_flx', 64],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90:440-570': [
        ('SUBROUTINE ice_flx_other', 1), 'END SUBROUTINE ice_flx_other', 131],
    'packages/ice/legoesm/ice/dynamics.py:101-103': [
        '_SI3_REQUIRED_HALO = 2', '_SI3_REQUIRED_RN_ISHLAT = 2.0', 3],
    'packages/ice/legoesm/ice/dynamics.py:809-822': [
        'def _si3_periodic_halo', 'mode="wrap"', 14],
    'packages/ice/legoesm/ice/transport.py:1829-1834': [
        'def _si3_periodic_halo_xy', 'mode="wrap"', 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:309-338': [
        'no ice if (sst - Tfreez) >= thresold', ('END WHERE', 3), 30],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:440-460': [
        'Adjust ssh and vertical scale factors to snow-ice mass', 'ENDWHERE', 21],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:354-364': [
        '3) fill fields', ('ENDDO', 3), 11],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn.f90:187-201': [
        ('SELECT CASE( nice_dyn )', 1), ('CALL ice_var_zapsmall', 1), 15],
    'sette/input_ORCA2_SAS_ICE.cfg:1-2': [
        'ORCA2_ICE_v5.0.0.tar.gz', 'SAS_v5.0.0.tar.gz', 2],
    'sette/sette_fetch_inputs.sh:39-46': [
        'suff="5.0.0"', 'else', 8],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:856-866': [
        'The committed first-divergence sweep advances the full dynALL card',
        'measured attribution.', 11],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:1629-1649': [
        'CPU/fp64/scalar-libm production JIT is byte-exact',
        'mechanism claim.', 21],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:872-884': [
        '### Round-8 active-redistribution measurement',
        'completed step 8 is first.', 13],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:194-209': [
        '| exact-entry statistic | Phase 5 before | Phase 6 after |',
        'histograms by step, sub-call, and variable.', 16],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:180-208': [
        '## Full-year exact-entry result',
        'NEMO-exactly-zero rows.', 29],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-36': [
        'Rung 3.5b remains **AT-BAR, not bit-identical**.',
        'disclosed at bar.', 7],
    'docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-43': [
        'Rung 3.5b remains **AT-BAR, not bit-identical**.',
        'not accepted or silently compared here.', 14],
    'docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2_manifest.md:136-143': [
        '## Phase-2c icebergs-off VARIANT handoff', 'labelled `VARIANT`.', 8],
    # --- SI3 ladder round 1 survey/design END ---
    # --- SI3 ladder round 2, round 3, round 5 & round 6 (D5) BEGIN ---
    'docs/ice/fidelity/si3_ladder_design.md:354-365': ['| Decision | Numbered options | Pick |', '| 10. Atmospheric forcing boundary | **1)** replay completed surface p', 12],
    'docs/ice/fidelity/si3_ladder_design.md:220': '| D5 | `cfgs/C1D/EXP_SASICE`, `key_si3 key_RK3 key_vco_1d3d key_linssh',
    'docs/ice/fidelity/si3_ladder_design.md:344-350': ["**Denormals are platform-specific.** D5's CPU result cannot be inherit", 'workaround is allowed.', 7],
    'docs/ice/fidelity/si3_r2_d5_preregister.md:1-74': ['# SI3 Ladder Round 2 (`si3_r2`) — Rung D5 Preregistration', '- **REFUTE P2**: Either (a) GPU flushes `0x0000000000000001` (`h_s > 0', 74],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:144-252': ['def run_stage_a_subnormal_probe(*, plant_zero_input: bool = False) -> ', ('}', 4), 109],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:131-141': ['def _jit_snow_pred(h_s: jax.Array) -> jax.Array:', (')', 57), 11],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:172-252': ['expected_bits = "0x0000000000000001"', ('}', 4), 81],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:170-186': ['IF( h_i_1d(ji) > 0._wp ) THEN', ('ENDIF', 3), 17],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:172-188': ['IF( h_i_1d(ji) > 0._wp ) THEN', ('ENDIF', 3), 17],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:217-230': ['zradtr_s(ji,jk) = zradtr_s(ji,0) * EXP( - zraext_s(ji) * MAX( 0._wp, z', ('&                                       * EXP( - rn_kappa_i * MAX( 0._', 2), 14],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:219-232': ['zradtr_s(ji,jk) = zradtr_s(ji,0) * EXP( - zraext_s(ji) * MAX( 0._wp, z', ('&                                       * EXP( - rn_kappa_i * MAX( 0._', 2), 14],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:326-348': [('IF ( .NOT. l_T_converged(ji) ) THEN', 1), ('ENDIF', 10), 23],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:321-343': [('IF ( .NOT. l_T_converged(ji) ) THEN', 1), ('ENDIF', 9), 23],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:366-376': ['zeta_i(ji,jk) = rDt_ice * r1_rhoi * z1_h_i(ji) / zcpi', ('END DO', 22), 11],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:353-364': ['zcpi = rcpi + zgamma * sz_i_1d(ji,jk) / MAX( ( t_i_1d(ji,jk) - rt0 ) *', ('END DO', 22), 12],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:821-823': [('qcn_ice_bot_1d(ji) = - zkappa_i(ji,nlay_i) * zg1 * ( t_bo_1d(ji ) - t_', 1), '&                 - ( 1._wp - isnow(ji) ) * zkappa_i(ji,0) * zg1  * ( ', 3],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:782-784': [('qcn_ice_bot_1d(ji) = - zkappa_i(ji,nlay_i) * zg1 * ( t_bo_1d(ji ) - t_', 1), '&                 - ( 1._wp - isnow(ji) ) * zkappa_i(ji,0) * zg1  * ( ', 3],
    'tests/ocean/fidelity/test_nemo_si3thd_phase2_gate.py:208-219': ['result = gate_module.run()', 'assert result["zdf_iterations"]["oracle"] == 2', 12],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1836': ('step = jax.jit(lambda state, forcing: si3_column_step_arrays(', 1),
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:404-699': ['def run_step1_reviewer_reproduction(', ('}', 20), 296],
    'docs/ice/fidelity/si3_r3_d5_receipt.md:36-65': ['`run_step1_reviewer_reproduction` (`scripts/validate/ocean_fidelity/te', '### 3.1 Step 2: Complete Harness Entry Conversion & Option-2 Salinity ', 30],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-286': ['def _nemo_salprof(', (')', 148), 86],
    'src/ICE/icevar.F90:390': 'ze_i             =   e_i (ji,jj,jk,jl) / v_i(ji,jj,jl) * zlay_i       ',
    'src/ICE/icevar.F90:412-413': ['t_s(ji,jj,jk,jl) = rt0 + MAX( -100._wp ,  &', '&               MIN( r1_rcpi*( -r1_rhos*( e_s(ji,jj,jk,jl) / v_s(ji,jj', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404': 'ze_i             =   e_i (ji,jj,jk,jl) / v_i(ji,jj,jl) * zlay_i       ',
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:426-427': ['t_s(ji,jj,jk,jl) = rt0 + MAX( -100._wp ,  &', '&               MIN( r1_rcpi*( -r1_rhos*( e_s(ji,jj,jk,jl) / v_s(ji,jj', 2],
    'src/ICE/icevar.F90:488-506': [('z1_dS = 1._wp / ( zsi1 - zsi0 )', 1), ('!  => mean(sz_i(jk)) = s_i', 1), 19],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:502-520': [('z1_dS = 1._wp / ( zsi1 - zsi0 )', 1), ('!  => mean(sz_i(jk)) = s_i', 1), 19],
    'src/ICE/icethd_dh.F90:498-503': ['! recalculate t_s_1d from e_s_1d', ('ELSE', 6), 6],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:508-513': ['! recalculate t_s_1d from e_s_1d', ('ELSE', 6), 6],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-313': ['def test_d5_zdf_layer_reciprocals_statement_is_bit_exact_and_fails_whe', 'assert rev_norm_5860 > probe_module.phase2.BAR', 253],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:316-472': ['def test_d5_zdf_gradients_are_finite_and_reverse_equals_forward_with_l', 'assert float(jnp.linalg.norm(planted_grad)) > 0.0', 157],
    'packages/core/legoesm/core/source_rounding.py:14-27': ['def nemo_source_round(value: jax.Array) -> jax.Array:', (')', 5), 14],
    'docs/ice/fidelity/si3_r3_d5_preregister.md:1-54': ['# SI3 Ladder Round 3 (`si3_r3`) — Rung D5 Continuation Preregistration', '- **REFUTE**: CPU production JIT (`XLA_FLAGS=""`) still disagrees with', 54],
    'docs/ice/fidelity/si3_r2_d5_receipt.md:194-203': ['### Operator note on reviewer-measured numbers', '* **Measured impact:** Recorded ZDF-entry snow temperature is reproduc', 10],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:717-732': ['def _exact_1d_entry(', (')', 361), 16],
    'src/ICE/icevar.F90:354-360': ['s_i(ji,jj,jl) = sv_i(ji,jj,jl) / v_i(ji,jj,jl)', ('CALL ice_var_salprof   ! salinity profile', 2), 7],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:368-374': ['s_i(ji,jj,jl) = sv_i(ji,jj,jl) / v_i(ji,jj,jl)', ('CALL ice_var_salprof   ! salinity profile', 2), 7],
    'packages/core/legoesm/timestepping/tridiagonal.py:137-179': ['if operation_order in ("nemo_unnormalised", "nemo_unnormalised_unround', 'return jax.lax.convert_element_type(solution, out_dtype)', 43],
    'docs/ice/fidelity/si3_r3_d5_preregister.md:40-42': ['1. **P1 (Step 1 — Reviewer reproduction)**:', '- **REFUTE**: Any of the counts (`1,811`, `7,385`, `26,202`, `1,625 ->', 3],
    'src/ICE/icevar.F90:362-379': ['ELSEIF( nn_icesal == 4 ) THEN', ('ENDIF', 18), 18],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-249': ['def _nemo_salprof(', (')', 128), 49],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:252-286': ['def _entry_arrays(', (')', 148), 35],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:571-595': [('t_i_1d(ji,nlay_i) = zindtbis(jm_max) / zdiagbis(jm_max)', 1), ('ENDIF', 22), 25],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:544-567': [('t_i_1d(ji,nlay_i) = zindtbis(jm_max) / zdiagbis(jm_max)', 1), '&          ( isnow(ji) * t_s_1d(ji,1) + ( 1._wp - isnow(ji) ) * t_i_1d', 24],
    'packages/core/legoesm/timestepping/tridiagonal.py:169-174': ['def unnormalised_backward(reverse_k, current):', (')', 53), 6],
    'src/ICE/icevar.F90:390-395': ['ze_i             =   e_i (ji,jj,jk,jl) / v_i(ji,jj,jl) * zlay_i       ', 't_i(ji,jj,jk,jl) = MAX( -100._wp , MIN( -( zbbb + zccc ) * 0.5_wp * r1', 6],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:404-409': ['ze_i             =   e_i (ji,jj,jk,jl) / v_i(ji,jj,jl) * zlay_i       ', 't_i(ji,jj,jk,jl) = MAX( -100._wp , MIN( -( zbbb + zccc ) * 0.5_wp * r1', 6],
    'src/ICE/icevar.F90:938-953': ['DO jk = 1, nlay_i             ! Sea ice energy of melting', ('END DO', 25), 16],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:952-967': ['DO jk = 1, nlay_i             ! Sea ice energy of melting', ('END DO', 58), 16],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:813': 'CALL ice_var_enthalpy',
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:276-290': ['ELSEIF( ln_cndi_P07 ) THEN     !-- Pringle et al formula: k = k0 + bet', ('END DO', 13), 15],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:271-285': ['ELSEIF( ln_cndi_P07 ) THEN     !-- Pringle et al formula: k = k0 + bet', ('END DO', 13), 15],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:334-335': ['zkappa_s(ji,nlay_s) = isnow(ji) * zghe(ji) * rcnd_s * ztcond_i(ji,0) &', '&                            / ( 0.5_wp * ( ztcond_i(ji,0) * zh_s(ji) ', 2],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:365-366': ['zcpi = rcpi + zgamma * sz_i_1d(ji,jk) / MAX( ( t_i_1d(ji,jk) - rt0 ) *', 'zeta_i(ji,jk) = rDt_ice * r1_rhoi * z1_h_i(ji) / zcpi', 2],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:399-529': ['qns_ice_1d(ji) = qns_ice_1d(ji) + dqns_ice_1d(ji) * ( t_su_1d(ji) - zt', 'zindterm(jm_min)   = ztiold(ji,1) + zeta_i(ji,1) * (zradab_i(ji,1) + z', 131],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:620': ('ztmelts       = -rTmlt * sz_i_1d(ji,jk) + rt0', 1),
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:329-330': ['zkappa_s(ji,nlay_s) = isnow(ji) * zghe(ji) * rcnd_s * ztcond_i(ji,0) &', '&                            / ( 0.5_wp * ( ztcond_i(ji,0) * zh_s(ji) ', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:353-354': ['zcpi = rcpi + zgamma * sz_i_1d(ji,jk) / MAX( ( t_i_1d(ji,jk) - rt0 ) *', 'zeta_i(ji,jk) = rDt_ice * r1_rhoi * z1_h_i(ji) / zcpi', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:387-512': ['qns_ice_1d(ji) = qns_ice_1d(ji) + dqns_ice_1d(ji) * ( t_su_1d(ji) - zt', 'zindterm(jm_min)   = ztiold(ji,1) + zeta_i(ji,1) * (zradab_i(ji,1) + z', 126],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:588': ('ztmelts       = -rTmlt * sz_i_1d(ji,jk) + rt0', 1),
    'src/ICE/icevar.F90:575-576': ['sz_i_1d(ji,jk) =          zalpha   * s_i_1d(ji) * 2._wp * ( REAL(jk,wp', '&          + ( 1._wp - zalpha ) * s_i_1d(ji)', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:589-590': ['sz_i_1d(ji,jk) =          zalpha   * s_i_1d(ji) * 2._wp * ( REAL(jk,wp', '&          + ( 1._wp - zalpha ) * s_i_1d(ji)', 2],
    'src/ICE/icevar.F90:1019-1023': ['IF( zh_cum1(jk1) <= zh_cum0(jk0) .AND. zh_cum1(jk1) > zh_cum0(jk0-1) )', ('ENDIF', 43), 5],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:1033-1037': ['IF( zh_cum1(jk1) <= zh_cum0(jk0) .AND. zh_cum1(jk1) > zh_cum0(jk0-1) )', ('ENDIF', 43), 5],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:486': 'ze_i_old(0) = ze_i_old(0) + zfmdt * zEw           ! 1st part (sea wate',
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:492': 'ze_i_old(0) = ze_i_old(0) + zdum * ze_s(jk)    ! 2nd part (snow enthal',
    'src/ICE/icethd_dh.F90:577': 'zeh_cum0(jk0) = zeh_cum0(jk0-1) + pe_old(jk0-1) * ph_old(jk0-1)',
    'src/ICE/icethd_dh.F90:598-600': ['zeh_cum1(jk1) = ( zeh_cum0(jk0-1) * ( zh_cum0(jk0) - zh_cum1(jk1  ) ) ', '&            / ( zh_cum0(jk0) - zh_cum0(jk0-1) )', 3],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:587': 'zeh_cum0(jk0) = zeh_cum0(jk0-1) + pe_old(jk0-1) * ph_old(jk0-1)',
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:608-610': ['zeh_cum1(jk1) = ( zeh_cum0(jk0-1) * ( zh_cum0(jk0) - zh_cum1(jk1  ) ) ', '&            / ( zh_cum0(jk0) - zh_cum0(jk0-1) )', 3],
    'src/ICE/icethd_dh.F90:219': 'zq_top       = MAX( 0._wp , zq_top       + zdum * ze_s(jk) )',
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:229': 'zq_top       = MAX( 0._wp , zq_top       + zdum * ze_s(jk) )',
    'src/ICE/icethd_dh.F90:254-256': [('zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enth', 2), 'zdE            =    zEi - zEw                          ! Specific enth', 3],
    'src/ICE/icethd_dh.F90:391-393': ['zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Spe', 'zdE            = zEi - zEw                                       ! Spe', 3],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264-266': [('zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enth', 2), 'zdE            =    zEi - zEw                          ! Specific enth', 3],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:401-403': ['zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Spe', 'zdE            = zEi - zEw                                       ! Spe', 3],
    'docs/ice/fidelity/si3_r3_d5_receipt.md:247-266': ['## Operator amendments after independent review (Claude LAND, codex HO', 'refusal.', 20],
    'src/ICE/icevar.F90:957-1034': ['SUBROUTINE ice_var_vremap( ph_old, pts_old, pts_i )', 'END SUBROUTINE ice_var_vremap', 78],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:971-1048': ['SUBROUTINE ice_var_vremap( ph_old, pts_old, pts_i )', 'END SUBROUTINE ice_var_vremap', 78],
    'src/ICE/icethd_dh.F90:511': 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )',
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:521': 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )',
    'src/ICE/icethd_dh.F90:254,391': [('zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enth', 2), 'zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Spe', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264,401': [('zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enth', 2), 'zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Spe', 2],
    'docs/ice/fidelity/si3_r3_d5_receipt.md:109-110': ['12. **Surface & basal ice melt specific enthalpy difference `zdE = zEi', '* NEMO source: `src/ICE/icethd_dh.F90:254-256` and `src/ICE/icethd_dh.', 2],
    'src/ICE/icevar.F90:504-505,575-576': ['sz_i(ji,jj,jk,jl) =             zalpha   * s_i(ji,jj,jl) * 2._wp * ( R', '&                + ( 1._wp - zalpha ) * s_i(ji,jj,jl)', 'sz_i_1d(ji,jk) =          zalpha   * s_i_1d(ji) * 2._wp * ( REAL(jk,wp', '&          + ( 1._wp - zalpha ) * s_i_1d(ji)', 4],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icevar.f90:518-519,589-590': ['sz_i(ji,jj,jk,jl) =             zalpha   * s_i(ji,jj,jl) * 2._wp * ( R', '&                + ( 1._wp - zalpha ) * s_i(ji,jj,jl)', 'sz_i_1d(ji,jk) =          zalpha   * s_i_1d(ji) * 2._wp * ( REAL(jk,wp', '&          + ( 1._wp - zalpha ) * s_i_1d(ji)', 4],
    'src/ICE/icevar.F90:504-505': ['sz_i(ji,jj,jk,jl) =             zalpha   * s_i(ji,jj,jl) * 2._wp * ( R', '&                + ( 1._wp - zalpha ) * s_i(ji,jj,jl)', 2],
    'src/ICE/icevar.F90:499-503': ['IF( s_i(ji,jj,jl) >= ( 0.5_wp * rn_sinew * sss_m(ji,jj) ) )           ', ('! linear profile with 0 surface value', 1), 5],
    'src/ICE/icevar.F90:471-472,546-547': [('REAL(wp), PARAMETER ::   zsi0 = 3.5_wp', 1), ('REAL(wp), PARAMETER ::   zsi1 = 4.5_wp', 1), ('REAL(wp), PARAMETER ::   zsi0 = 3.5_wp', 2), ('REAL(wp), PARAMETER ::   zsi1 = 4.5_wp', 2), 4],
    'src/ICE/icevar.F90:499-500': ['IF( s_i(ji,jj,jl) >= ( 0.5_wp * rn_sinew * sss_m(ji,jj) ) )           ', 'IF( s_i(ji,jj,jl) <= ( REAL( nlay_i , wp ) * MIN( rn_simin, rn_sinew *', 2],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:289-301': ['def _forcing(exact):', 'return SI3SurfaceForcing(*(val(exact[name]) for name in ZDF_INPUT_NAME', 13],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:705-714': ['def _forcing(values, dtype):', ('))', 29), 10],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:1802,1898-1904,2161-2179': ['plant_missing_sss=False) -> dict[str, object]:', 'if plant_missing_sss and kt == 1:', (')', 773), 'parser.add_argument("--plant-missing-sss", action="store_true")', 'plant_missing_sss=args.plant_missing_sss)', 27],
    'src/ICE/icethd_dh.F90:254-256,391-393': [('zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enth', 2), 'zdE            =    zEi - zEw                          ! Specific enth', 'zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Spe', 'zdE            = zEi - zEw                                       ! Spe', 6],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:264-266,401-403': [('zEi            = - e_i_1d(ji,jk) * r1_rhoi             ! Specific enth', 2), 'zdE            =    zEi - zEw                          ! Specific enth', 'zEi            = - e_i_1d(ji,jk) * r1_rhoi                       ! Spe', 'zdE            = zEi - zEw                                       ! Spe', 6],
    'src/ICE/icevar.F90:471-472,499-500,546-547': [('REAL(wp), PARAMETER ::   zsi0 = 3.5_wp', 1), ('REAL(wp), PARAMETER ::   zsi1 = 4.5_wp', 1), 'IF( s_i(ji,jj,jl) >= ( 0.5_wp * rn_sinew * sss_m(ji,jj) ) )           ', 'IF( s_i(ji,jj,jl) <= ( REAL( nlay_i , wp ) * MIN( rn_simin, rn_sinew *', ('REAL(wp), PARAMETER ::   zsi0 = 3.5_wp', 2), ('REAL(wp), PARAMETER ::   zsi1 = 4.5_wp', 2), 6],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2_gate.py:201-301': ['def _nemo_salprof(', 'return SI3SurfaceForcing(*(val(exact[name]) for name in ZDF_INPUT_NAME', 101],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_phase2b_year_gate.py:705-752,1802,1898-1904,2161-2179': ['def _forcing(values, dtype):', '"exact-entry 1-D thickness bridge",', 'plant_missing_sss=False) -> dict[str, object]:', 'if plant_missing_sss and kt == 1:', (')', 773), 'parser.add_argument("--plant-missing-sss", action="store_true")', 'plant_missing_sss=args.plant_missing_sss)', 75],
    'src/ICE/icethd_dh.F90:134,284,306,361,421,476,482': ['ze_i_old(jk) = h_i_1d(ji) * r1_nlay_i * e_i_1d(ji,jk)', ('ze_i_old(jk) = ze_i_old(jk) + zdum * e_i_1d(ji,jk)', 1), ('ze_i_old(jk) = ze_i_old(jk) + zdum * e_i_1d(ji,jk)', 2), 'ze_i_old(nlay_i+1) = ze_i_old(nlay_i+1) + dh_i_bog(ji) * (-zEi * rhoi)', ('ze_i_old(jk) = ze_i_old(jk) + zdum * e_i_1d(ji,jk)', 3), 'ze_i_old(0) = ze_i_old(0) + zfmdt * zEw           ! 1st part (sea wate', 'ze_i_old(0) = ze_i_old(0) + zdum * ze_s(jk)    ! 2nd part (snow enthal', 7],
    'src/ICE/icethd.F90:445': 'e_s_1d(1:npti,jk) = e_s_1d(1:npti,jk) * h_s_1d(1:npti) * a_i_1d(1:npti',
    'src/ICE/icethd.F90:442': 'e_i_1d(1:npti,jk) = e_i_1d(1:npti,jk) * h_i_1d(1:npti) * a_i_1d(1:npti',
    'src/ICE/icethd.F90:442,445': ['e_i_1d(1:npti,jk) = e_i_1d(1:npti,jk) * h_i_1d(1:npti) * a_i_1d(1:npti', 'e_s_1d(1:npti,jk) = e_s_1d(1:npti,jk) * h_s_1d(1:npti) * a_i_1d(1:npti', 2],
    'src/ICE/icethd_dh.F90:204-225,441-496': ['! Snow melting', ('END DO', 5), '! Snow-Ice formation', 'e_s_1d(ji,:) = snw_ent( zh_s, ze_s )', 78],
    'packages/ice/legoesm/ice/sea_ice.py:293-305,508-525': [('if config.thermo_scheme == "si3_bl99":', 1), ('return new_state, response', 1), 'def _si3_step_with_trace(state: SI3ColumnState,', 'trace = si3_column_step_arrays(arrays, forcing, dt, config.ice_constan', 31],
    'packages/ice/legoesm/ice/sea_ice.py:304,508-525': ['new_state, response, _ = _si3_step_with_trace(state, forcing, config, ', 'def _si3_step_with_trace(state: SI3ColumnState,', 'trace = si3_column_step_arrays(arrays, forcing, dt, config.ice_constan', 19],
    'tests/ocean/fidelity/test_nemo_si3thd_phase2b_year_gate.py:64-85': ['def test_snow_free_entry_uses_nemo_zero_volume_convention(gate) -> Non', 'assert found, "seasonal C1D stream never exercised its snow-free state', 22],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:144-2072': ['def run_stage_a_subnormal_probe(*, plant_zero_input: bool = False) -> ', ('}', 102), 1929],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:1985-2008': ['"predicate_true_count": pred_true_count,', ('zdf_layer_reciprocals=False,', 2), 24],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1026-1074': ['def test_d5_ftz_daz_classification_distinguishes_ftz_only_from_daz(', 'assert classified_ftz_daz["ftz_daz_mode"] == "FTZ_DAZ_FLUSH_TO_ZERO"', 49],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:806-1883': ['def run_year_rescore(', ('}', 94), 1078],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:475-1023': ['def _parent_commit_bitz_lipscomb_conduction_step(', ('_verify_si3_dispatcher_vs_parent(current_mods)', 4), 549],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1077-1232': ['def test_d5_platform_probe_and_planted_controls(probe_module) -> None:', 'assert s3_s4["identical_full_year_output_sha256"] is True', 156],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3thd_d5_platform_probe.py:1745-1758': ['for arm_name, acc in arm_accs.items():', '"changed_field_rows": step3_vs_step4_changed_rows,', 14],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1235-1358': ['def test_r3_step2_harness_entry_conversion_is_bit_exact_and_fails_when', (')', 569), 124],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1361-1458': ['def test_r3_step3_snow_temperature_from_enthalpy_is_bit_exact_and_fail', 'assert norm_rev_1334 > probe_module.phase2.BAR', 98],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1461-1599': ['def test_r3_step4_cpu_fusion_prevention_is_bit_exact_and_fails_when_re', 'assert probe_module._ulp_diff(arr_r, arr_u) >= 1', 139],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1602-1737': ['def test_r3_all_touched_functions_gradients_and_forward_bits() -> None', (')', 762), 136],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1740-1808': ['def test_r5_step1_nemo_ice_enthalpy_remap_is_bit_exact_and_fails_when_', ('assert norm_rev > probe_module.phase2.BAR', 1), 69],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1811-1900': ['def test_r5_step2_ice_density_reciprocal_in_dh_step_is_bit_exact_and_f', 'assert probe_module._ulp_diff(arr_r, arr_d) >= 1', 90],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1903-1964': ['def test_r5_step3_option2_salinity_profile_factoring_is_bit_exact_and_', (')', 858), 62],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:1967-2044': ['def test_r5_step4_harness_salprof_zsi1_zsi0_and_missing_sss_refusal(', (')', 894), 78],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2047-2209': ['def test_r5_all_touched_functions_gradients_and_forward_bits(', 'assert float(jnp.linalg.norm(planted_g)) > 0.0', 163],
    'docs/ice/fidelity/si3_r5_d5_receipt.md:248-258': ['### 5.3 Retraction of Open-Water Growth Attribution & Recorded Next D5', ('', 258), 11],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:480-495': ['zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 16],
    'src/ICE/icethd_dh.F90:168-177,184-185,221': ['IF( sprecip_1d(ji) > 0._wp ) THEN', ('ENDIF', 4), 'zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) ', 'zevap_rema =        evap_ice_1d(ji)           * rDt_ice + zdeltah * rh', 'zh_s    (jk) = MAX( 0._wp , zh_s    (jk) + zdum )', 13],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:178-187,194-195,231': ['IF( sprecip_1d(ji) > 0._wp ) THEN', ('ENDIF', 4), 'zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) ', 'zevap_rema =        evap_ice_1d(ji)           * rDt_ice + zdeltah * rh', 'zh_s    (jk) = MAX( 0._wp , zh_s    (jk) + zdum )', 13],
    'src/ICE/icethd_dh.F90:168-177': ['IF( sprecip_1d(ji) > 0._wp ) THEN', ('ENDIF', 4), 10],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:178-187': ['IF( sprecip_1d(ji) > 0._wp ) THEN', ('ENDIF', 4), 10],
    'icethd_dh.f90:138-147': ['zs_i_old(0:nlay_i+1) = 0._wp', ('END DO', 1), 10],
    'icethd_dh.f90:243-325': [('DO jk = 1, nlay_i', 2), ('END DO', 6), 83],
    'icethd_dh.f90:378-434': ['DO jk = nlay_i, 1, -1', ('END DO', 7), 57],
    'icethd_dh.f90:521': 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )',
    'icevar.f90:971-1048': ['SUBROUTINE ice_var_vremap( ph_old, pts_old, pts_i )', 'END SUBROUTINE ice_var_vremap', 78],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:194-195': ['zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) ', 'zevap_rema =        evap_ice_1d(ji)           * rDt_ice + zdeltah * rh', 2],
    'src/ICE/icethd_dh.F90:184-185': ['zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) ', 'zevap_rema =        evap_ice_1d(ji)           * rDt_ice + zdeltah * rh', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:231': 'zh_s    (jk) = MAX( 0._wp , zh_s    (jk) + zdum )',
    'src/ICE/icethd_dh.F90:221': 'zh_s    (jk) = MAX( 0._wp , zh_s    (jk) + zdum )',
    'docs/ice/fidelity/si3_r6_d5_preregister.md:32,36': ['- **CONFIRM**: Enabling `_nemo_dh_ice_content_accumulation=True` (`r6_', '- **CONFIRM**: Enabling `_nemo_dh_snow_update=True` on top of Statemen', 2],
    'docs/ice/fidelity/si3_r5_d5_receipt.md:215,251,257': ['| 6 | `POST_DH.e_s` | `2` | `2` | `137` | `129` | `2,563,146,750,041,3', '* Attributing the remaining `34` normal-step over-bar rows (`13` at `P', '* The remaining normal-step snow-enthalpy over-bar rows (`2` steps `kt', 3],
    'icethd_dh.f90:179': 'zh_s(0) = zsnw(ji) * sprecip_1d(ji) * rDt_ice * r1_rhos / at_i_1d(ji) ',
    'icethd_dh.f90:194': 'zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) ',
    'icethd_dh.f90:178-187,194-195,231': ['IF( sprecip_1d(ji) > 0._wp ) THEN', ('ENDIF', 4), 'zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) ', 'zevap_rema =        evap_ice_1d(ji)           * rDt_ice + zdeltah * rh', 'zh_s    (jk) = MAX( 0._wp , zh_s    (jk) + zdum )', 13],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2212-2352': ['def test_r6_step1_dh_ice_content_accumulation_is_bit_exact_and_fails_w', (')', 1060), 141],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2355-2484': ['def test_r6_step2_dh_snow_update_is_bit_exact_and_fails_when_reverted(', 'assert probe_module._ulp_diff(got_es, oracle_es) == 0', 130],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2487-2656': ['def test_r6_all_touched_functions_gradients_and_forward_bits(', 'assert float(jnp.linalg.norm(planted_g_rc)) > 0.0', 170],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:61-2776': ['def test_d5_zdf_layer_reciprocals_statement_is_bit_exact_and_fails_whe', (')', 1293), 2716],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1655-1691': ['def si3_column_step_arrays(state: SI3ColumnArrays,', 'e_snow=zdf.e_snow)', 37],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:432-478': ['def _radiation(', 'return ab_s, ab_i, tr_i[..., -1]', 47],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:481-499': ['def _si3_zdf_bl99_step(', ') -> SI3ZDFResult:', 19],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:557-647': ['dz_i = jnp.maximum(h_ice, _H_MIN) / 3.0', 'eta_s = dt / (constants.rho_snow * dz_s[..., None] * constants.c_ice)', 91],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:794-888': ['Tsu = jnp.where(active, Tsu_candidate, Tsu)', 'qcn_bottom = 2.0 * k_base_i * (Ti[..., -1] - T_bottom)', 95],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:321-351': ['def bitz_lipscomb_conduction_step(q_layers, h, S, F_top, T_bottom_K, d', 'return q_new, T_new, F_bottom', 31],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:402-411': ['if _nemo_order:', 'return jnp.clip(unbounded, constants.T0 - 100.0, constants.T0)', 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:523-529': ['T_s_old = snow_temperature_from_enthalpy(', (')', 153), 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:369-411': ['def snow_temperature_from_enthalpy(', 'return jnp.clip(unbounded, constants.T0 - 100.0, constants.T0)', 43],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:148-196': ['def option2_salinity_profile(', 'return S_bulk[..., None] * (sr(alpha[..., None] * z) + (1.0 - alpha[..', 49],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:199-232': ['def ice_enthalpy_from_temperature(', 'return sr(constants.rho_ice * sr(sr(sensible + latent) - ocean))', 34],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:235-296': ['def ice_temperature_from_enthalpy(', (')', 73), 62],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:354-366': ['def snow_enthalpy_from_temperature(', (')', 104), 13],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:414-429': ['def p07_conductivity(', (')', 122), 16],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:481-891': ['def _si3_zdf_bl99_step(', (')', 329), 411],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:894-914': ['def _piecewise_remap(', 'return jnp.where(dz_new[..., None] > _EPS10, content / dz_new[..., Non', 21],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:917-976': ['def _nemo_snow_enthalpy_remap(', ('], axis=-1)', 9), 60],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1040-1652': ['def _dh_step(state: SI3ColumnArrays, zdf: SI3ZDFResult,', 'S_bulk, state.S_layers, state.age_volume)', 613],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1655-1773': ['def si3_column_step_arrays(state: SI3ColumnArrays,', 'post_temp2, post_do, exit_state)', 119],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:587-611': ['# P07 conductivity at the four ice interfaces (top, 2 interior, base).', '# (compiled `icethd_zdf_bl99.f90:331-346`): layer thicknesses use', 25],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:624-862': ['k_i = tcond_i / dz_i[..., None]', ('else:', 5), 239],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1727-1732': ['S_layers = option2_salinity_profile(', (')', 689), 6],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1620-1636': [('e_i = (', 2), (')', 671), 17],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1630-1636': ['e_s = (', (')', 671), 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1201-1203': ['surface_energy = jnp.maximum(', (')', 472), 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1225-1234': [('internal_melt = _surface_melt & (', 1), ('energy_difference = sr(ice_specific - water_specific)', 1), 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1331-1337': [('ice_specific = (', 2), ('energy_difference = sr(ice_specific - water_specific)', 2), 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1374-1383': [('bottom_specific = (', 1), ('bottom_difference = bottom_specific - bottom_water', 1), 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:979-1037': ['def _nemo_ice_enthalpy_remap(', ('], axis=-1)', 10), 59],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1620-1629': [('e_i = (', 2), (')', 668), 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1206,1225-1234,1331-1337,1374-1383': ['inverse_ice_density = 1.0 / constants.rho_ice', ('internal_melt = _surface_melt & (', 1), ('energy_difference = sr(ice_specific - water_specific)', 1), ('ice_specific = (', 2), ('energy_difference = sr(ice_specific - water_specific)', 2), ('bottom_specific = (', 1), ('bottom_difference = bottom_specific - bottom_water', 1), 28],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1206': 'inverse_ice_density = 1.0 / constants.rho_ice',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:989': 'pts_old = content if content is not None else sr(thickness * enthalpy)',
    'src/ICE/icethd_dh.F90:118-120,128-137,233-315,330-364,368-424,452-453,470-485,511': ['zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )', 'zq_bot = MAX( 0._wp, zf_tt * rDt_ice )', 'zs_i_old(0:nlay_i+1) = 0._wp', ('END DO', 1), ('DO jk = 1, nlay_i', 2), ('END DO', 6), 'IF(  zf_tt < 0._wp  ) THEN', ('ENDIF', 8), 'DO jk = nlay_i, 1, -1', ('END DO', 7), 'zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0', 'zEw            = rcp * sst_1d(ji)', 'zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )', 207],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:128-130,138-147,243-325,340-374,378-434,462-463,480-495,521': ['zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )', 'zq_bot = MAX( 0._wp, zf_tt * rDt_ice )', 'zs_i_old(0:nlay_i+1) = 0._wp', ('END DO', 1), ('DO jk = 1, nlay_i', 2), ('END DO', 6), 'IF(  zf_tt < 0._wp  ) THEN', ('ENDIF', 8), 'DO jk = nlay_i, 1, -1', ('END DO', 7), 'zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0', 'zEw            = rcp * sst_1d(ji)', 'zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )', 207],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1210-1402,1578-1629': [('if _nemo_dh_ice_content_accumulation:', 1), 'old_h = jnp.stack(zh_i_old_layers, axis=-1)', ('if _nemo_dh_ice_content_accumulation:', 2), (')', 668), 245],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1655-1707': ['def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-24', 53],
    'icethd_dh.f90:128-130,340-374': ['zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )', 'zq_bot = MAX( 0._wp, zf_tt * rDt_ice )', 'IF(  zf_tt < 0._wp  ) THEN', ('ENDIF', 8), 38],
    'icethd_dh.f90:462-463,480-495': ['zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0', 'zEw            = rcp * sst_1d(ji)', 'zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 18],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1068-1085,1120-1136,1195-1199': [('if _nemo_dh_snow_update:', 1), (')', 417), ('if _nemo_dh_snow_update:', 2), 'remaining_mass = evaporation * dt + snow_delta * constants.rho_snow', 'melted_snow.append(', (')', 469), 40],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1672,1704': [('_nemo_dh_snow_update: bool = True,', 2), '_nemo_dh_snow_update=_nemo_dh_snow_update,', 2],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:128-130,138-147,243-253,262-274,288-295,300,309-317,322-325,340-355,365-374,378-389,399-411,425-434,462-463,480-495,521': ['zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )', 'zq_bot = MAX( 0._wp, zf_tt * rDt_ice )', 'zs_i_old(0:nlay_i+1) = 0._wp', ('END DO', 1), ('DO jk = 1, nlay_i', 2), 'zfmdt          = - zdum * rhoi                         ! Recompute mas', 'ELSE                                        !-- Surface melting', 'zq_top         = MAX( 0._wp , zq_top - zdum * rhoi * zdE ) ! update av', ('! update thickness', 3), ('zh_i_old(jk) = zh_i_old(jk) + zdum', 1), 'zdum               = MAX( - zh_i(jk) , - zevap_rema * r1_rhoi )', 'zevap_rema   = zevap_rema + zdum * rhoi', ('zh_i_old(jk) = zh_i_old(jk) + zdum', 2), 'IF( zh_i(jk) > 0._wp ) THEN ; icount(jk) = 0', ('END DO', 6), 'IF(  zf_tt < 0._wp  ) THEN', 'dh_i_bog(ji)   = rDt_ice * MAX( 0._wp , zf_tt / ( zdE * rhoi ) )', ('! update thickness', 4), ('ENDIF', 8), 'DO jk = nlay_i, 1, -1', '!                                                  this should normall', 'ELSE                                        !-- Basal melting', 'zq_bot         = MAX( 0._wp , zq_bot - zdum * rhoi * zdE )       ! upd', ('! update thickness', 5), ('END DO', 7), 'zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0', 'zEw            = rcp * sst_1d(ji)', 'zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )', 139],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/sbcssm.f90:180-184': ['IF( ln_read_frq ) THEN', ('ENDIF', 4), 5],
    'src/SAS/sbcssm.F90:122-126': ['IF( ln_read_frq ) THEN', ('ENDIF', 4), 5],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:128-130,138-147': ['zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )', 'zq_bot = MAX( 0._wp, zf_tt * rDt_ice )', 'zs_i_old(0:nlay_i+1) = 0._wp', ('END DO', 1), 13],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:243-253,262-274,288-295,300,309-317,322-325': [('DO jk = 1, nlay_i', 2), 'zfmdt          = - zdum * rhoi                         ! Recompute mas', 'ELSE                                        !-- Surface melting', 'zq_top         = MAX( 0._wp , zq_top - zdum * rhoi * zdE ) ! update av', ('! update thickness', 3), ('zh_i_old(jk) = zh_i_old(jk) + zdum', 1), 'zdum               = MAX( - zh_i(jk) , - zevap_rema * r1_rhoi )', 'zevap_rema   = zevap_rema + zdum * rhoi', ('zh_i_old(jk) = zh_i_old(jk) + zdum', 2), 'IF( zh_i(jk) > 0._wp ) THEN ; icount(jk) = 0', ('END DO', 6), 46],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:340-355,365-374': ['IF(  zf_tt < 0._wp  ) THEN', 'dh_i_bog(ji)   = rDt_ice * MAX( 0._wp , zf_tt / ( zdE * rhoi ) )', ('! update thickness', 4), ('ENDIF', 8), 26],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:378-389,399-411,425-434': ['DO jk = nlay_i, 1, -1', '!                                                  this should normall', 'ELSE                                        !-- Basal melting', 'zq_bot         = MAX( 0._wp , zq_bot - zdum * rhoi * zdE )       ! upd', ('! update thickness', 5), ('END DO', 7), 35],
    'C1D_OMIP_L3/BLD/ppsrc/nemo/icethd_dh.f90:462-463,480-495,521': ['zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0', 'zEw            = rcp * sst_1d(ji)', 'zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )', 19],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:40,979-1037,1210-1402,1578-1629,1655-1707': ['frq_m: jax.Array | float = 1.0', 'def _nemo_ice_enthalpy_remap(', ('], axis=-1)', 10), ('if _nemo_dh_ice_content_accumulation:', 1), 'old_h = jnp.stack(zh_i_old_layers, axis=-1)', ('if _nemo_dh_ice_content_accumulation:', 2), (')', 668), 'def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-24', 358],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:979-989': ['def _nemo_ice_enthalpy_remap(', 'pts_old = content if content is not None else sr(thickness * enthalpy)', 11],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:40,1281-1285': ['frq_m: jax.Array | float = 1.0', 'frq_m = getattr(forcing, "frq_m", 1.0)', (')', 516), 6],
    'icethd_dh.f90:129': 'zf_tt  = qcn_ice_bot_1d(ji) + qsb_ice_bot_1d(ji) + fhld_1d(ji) + qtr_i',
    'sbcssm.f90:183': ('frq_m(:,:) = 1._wp', 1),
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1235': 'zfmdt = sr(-surface_energy / energy_difference)',
    'icethd_dh.f90:268': 'zfmdt          = - zq_top / zdE                        ! Mass flux to ',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1338': 'zfmdt = sr(-basal_energy / energy_difference)',
    'icethd_dh.f90:405': 'zfmdt          = - zq_bot / zdE                                  ! Mas',
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2659-2776': ['def test_r6_literal_denominators_never_zero_under_record_and_frq_m_sca', (')', 1293), 118],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1608-1617': ['ze_i_old = jnp.stack(', (')', 665), 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1621-1626': [('_nemo_ice_enthalpy_remap(', 2), (')', 666), 6],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1068-1085,1120-1136,1195-1199,1672,1704': [('if _nemo_dh_snow_update:', 1), (')', 417), ('if _nemo_dh_snow_update:', 2), 'remaining_mass = evaporation * dt + snow_delta * constants.rho_snow', 'melted_snow.append(', (')', 469), ('_nemo_dh_snow_update: bool = True,', 2), '_nemo_dh_snow_update=_nemo_dh_snow_update,', 42],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1073-1077': ['h_precip_raw = (', (')', 414), 5],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1078-1080': ['h_precip = jnp.where(', (')', 415), 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1124-1130': [('snow_delta = jnp.maximum(', 1), (')', 438), 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1195-1199': ['melted_snow.append(', (')', 469), 5],
    'icethd_dh.f90:128-130,138-147,243-325,340-374,378-434,462-463,480-495,521': ['zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )', 'zq_bot = MAX( 0._wp, zf_tt * rDt_ice )', 'zs_i_old(0:nlay_i+1) = 0._wp', ('END DO', 1), ('DO jk = 1, nlay_i', 2), ('END DO', 6), 'IF(  zf_tt < 0._wp  ) THEN', ('ENDIF', 8), 'DO jk = nlay_i, 1, -1', ('END DO', 7), 'zfmdt          = ( rhos - rhoi ) * dh_snowice(ji)    ! <0', 'zEw            = rcp * sst_1d(ji)', 'zh_i(0) = zh_i(0) + dh_snowice(ji)', ('END DO', 9), 'CALL ice_var_vremap( zh_i_old, ze_i_old, e_i_1d (ji,:) )', 207],
    'sbcssm.f90:180-184': ['IF( ln_read_frq ) THEN', ('ENDIF', 4), 5],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1076,1235,1338': [('/ a', 2), 'zfmdt = sr(-surface_energy / energy_difference)', 'zfmdt = sr(-basal_energy / energy_difference)', 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:40,1281-1285,1465': ['frq_m: jax.Array | float = 1.0', 'frq_m = getattr(forcing, "frq_m", 1.0)', (')', 516), '+ zdf.qtr_ice_bottom * getattr(forcing, "frq_m", 1.0)', 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1052-1053,1671-1672': [('_nemo_dh_ice_content_accumulation: bool = True,', 1), ('_nemo_dh_snow_update: bool = True,', 1), ('_nemo_dh_ice_content_accumulation: bool = True,', 2), ('_nemo_dh_snow_update: bool = True,', 2), 4],
    # --- SI3 ladder round 2, round 3, round 5 & round 6 (D5) END ---
    # --- SI3 ladder round 4 (O1) BEGIN ---
    'docs/ice/fidelity/si3_ladder_design.md:221': '| O1 | ORCA2_SAS_ICE, dynamics/thermo off, exact NEMO state injected. ',
    'cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm:1': 'bld::tool::fppkeys   key_si3 key_linssh key_vco_1d3d key_xios key_RK3',
    'docs/ice/fidelity/si3_r4_o1_preregister.md:1-76': ['# SI3 Ladder Rung O1 (`si3_r4`) — Preregistration', '- Item 7 (whitespace-tolerant log check & `EXIT` trap): planted `E R R', 76],
    'packages/ice/legoesm/ice/state.py:338-345': ['from legoesm.ice.si3_orca2_state import (  # noqa: E402,F401', (')', 115), 8],
    'packages/ice/legoesm/ice/config.py:524-575': ['# Optional ORCA2 5-category 10+5-layer SI3 state configuration (`therm', (')', 156), 52],
    'packages/ice/legoesm/ice/sea_ice.py:364-365': ['if config.thermo_scheme == "si3_orca2_o1":', 'return _si3_orca2_o1_step_with_trace(state, forcing, config, dt)', 2],
    'packages/ice/legoesm/ice/itd.py:1078-1080': ['from legoesm.ice.si3_orca2_state import (  # noqa: E402,F401', 'si3_hfn_category_bounds,', 3],
    'packages/ice/legoesm/ice/__init__.py:201-227': [']', (')', 17), 27],
    'scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid/run.sh:1-382': ['#!/usr/bin/env bash', ('admit', 18), 382],
    'scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_orca2_tke_boundary_acquisition/run.sh:1-903': ['#!/usr/bin/env bash', ('finalize_existing', 3), 903],
    'scripts/validate/ocean_fidelity/testcases/nemo502_si3thd_MY_SRC/icethd_zdf_bl99.F90:1-947': [('MODULE icethd_zdf_BL99', 1), 'END MODULE icethd_zdf_BL99', 947],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_adv2d_gate.py:1-120': ['#!/usr/bin/env python3', (')', 43), 120],
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:1-84': [('!!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 1), ('/', 20), 84],
    'cfgs/SHARED/namelist_ice_ref:22-258': ['&nampar         !   Generic parameters', ('/', 40), 237],
    'packages/ice/legoesm/ice/config.py:524-525': ['# Optional ORCA2 5-category 10+5-layer SI3 state configuration (`therm', ('si3_orca2: SI3ORCA2StateConfig | None = None', 1), 2],
    'packages/ice/legoesm/ice/config.py:566-575': ['def build_si3_orca2_o1_config(', (')', 156), 10],
    'packages/ice/legoesm/ice/config.py:531-563': ['def _validate_si3_orca2_o1_config(config: "SeaIceConfig") -> None:', 'raise ValueError("invalid SI3 ORCA2 O1 identity: " + "; ".join(conflic', 33],
    'src/ICE/icedyn_adv_pra.F90:36-46': ['REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxice, syice, sxx', 'REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   sxvl , syvl , sxx', 11],
    'src/ICE/icedyn_adv_pra.F90:1145-1200': [('SUBROUTINE adv_pra_init', 1), 'END SUBROUTINE adv_pra_init', 56],
    'src/ICE/iceitd.F90:715-769': ['IF(lwp) THEN                  ! control print', 'END SUBROUTINE ice_itd_init', 55],
    'src/ICE/iceitd.F90:736-761': ['SELECT CASE( nice_catbnd )', 'hi_max(jpl) = rn_himax        ! set to a big value to ensure that all ', 26],
    'src/ICE/iceitd.F90:757-759': ['DO jl = 1, jpl                ! mean thickness by category', ('END DO', 23), 3],
    'src/ICE/iceitd.F90:761': 'hi_max(jpl) = rn_himax        ! set to a big value to ensure that all ',
    'src/ICE/icevar.F90:255-428': ['SUBROUTINE ice_var_glo2eqv( kn )', 'END SUBROUTINE ice_var_glo2eqv', 174],
    'src/ICE/icevar.F90:454-534': [('SUBROUTINE ice_var_salprof', 1), ('END SUBROUTINE ice_var_salprof', 1), 81],
    'src/ICE/icevar.F90:1564-1587': ['SUBROUTINE ice_var_snwfra_3d( ph_s, pa_s_fra )', 'END SUBROUTINE ice_var_snwfra_3d', 24],
    'src/ICE/icevar.F90:290-303': [('DO jl = 1, jpl', 4), ('END_2D', 8), 14],
    'src/ICE/icevar.F90:344-379': ['IF( nn_icesal == 1 .OR. nn_icesal == 3 ) THEN', ('ENDIF', 18), 36],
    'src/ICE/icevar.F90:384-402': [('zlay_i = REAL( nlay_i , wp )    ! number of layers', 2), ('END DO', 8), 19],
    'src/ICE/icevar.F90:407-419': ['zlay_s = REAL( nlay_s , wp )', ('END DO', 10), 13],
    'src/ICE/icevar.F90:315-339': ['h_ip(ji,jj,jl) = v_ip(ji,jj,jl) * z1_a_ip', ('ENDIF', 15), 25],
    'src/ICE/icevar.F90:105-252': ['SUBROUTINE ice_var_agg( kn )', 'END SUBROUTINE ice_var_agg', 148],
    'src/ICE/iceupdate.F90:189-194': [('DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )', 1), ('END_2D', 5), 6],
    'src/ICE/iceupdate.F90:459-472': ['IF( id1 > 0 ) THEN                       ! fields exist', ('ENDIF', 13), 14],
    'src/OCE/LBC/lbc_lnk_pt2pt_generic.h90:99-243': ['DO jn = 1, 4   ! 4 sides', ('ENDIF', 12), 145],
    'src/OCE/LBC/lbc_nfd_generic.h90:22-191': ["IF( c_NFtype == 'T' ) THEN            ! *  North fold  T-point pivot", "ENDIF   ! c_NFtype == 'T'", 170],
    'src/OCE/LBC/lbc_nfd_generic.h90:25-79': [("CASE ( 'T' , 'W' )                         ! T-, W-point", 1), ('END DO   ;   END DO', 1), 55],
    'src/OCE/LBC/lbc_nfd_generic.h90:80-124': [("CASE ( 'U' )                               ! U-point", 1), ('END DO   ;   END DO', 2), 45],
    'src/OCE/LBC/lbc_nfd_generic.h90:125-162': [("CASE ( 'V' )                               ! V-point", 1), ('END DO   ;   END DO', 3), 38],
    'src/ICE/icedyn_adv_pra.F90:438-482': [("&                          , sxxice, 'T', 1._wp, syyice, 'T',  1._wp, ", 1), ('!', 211), 45],
    'src/ICE/icedyn_adv_pra.F90:1246-1356': ["CALL iom_get( numrir, jpdom_auto, 'sxice' , sxice , psgn = -1._wp )", ('ELSE', 29), 111],
    'src/ICE/icerst.F90:281-282': ["CALL iom_get( numrir, jpdom_auto, 'u_ice', u_ice, cd_type = 'U', psgn ", "CALL iom_get( numrir, jpdom_auto, 'v_ice', v_ice, cd_type = 'V', psgn ", 2],
    'src/ICE/icestp.F90:96-238': ['SUBROUTINE ice_stp( kt, Kbb, Kmm, ksbc )', ('END SUBROUTINE ice_stp', 1), 143],
    'src/ICE/icestp.F90:167': 'IF( ln_icedyn .AND. .NOT.ln_c1d )   &',
    'src/ICE/icestp.F90:205': ('!----------------------------!', 2),
    'src/ICE/icestp.F90:164': 'CALL diag_set0                ! set diag of mass, heat and salt fluxes',
    'src/ICE/icestp.F90:488': 't_si       (ji,jj,jl) = rt0     ! temp at the ice-snow interface',
    'src/ICE/icestp.F90:182': 'CALL ice_var_glo2eqv(2)       ! h_i and h_s for ice albedo calculation',
    'src/ICE/icestp.F90:183': 'CALL ice_var_agg(1)           ! at_i for coupling',
    'src/ICE/icestp.F90:184': ('CALL store_fields             ! Store now ice values', 2),
    'src/ICE/icestp.F90:388-426': [('SUBROUTINE store_fields', 1), 'END SUBROUTINE store_fields', 39],
    'src/ICE/icestp.F90:218': 'CALL ice_dia( kt )            ! -- Diagnostics outputs',
    'src/ICE/icestp.F90:219': ('!', 183),
    'src/ICE/icestp.F90:221': ('!', 185),
    'src/ICE/icerst.F90:111-197': ['SUBROUTINE ice_rst_write( kt )', 'END SUBROUTINE ice_rst_write', 87],
    'src/ICE/icerst.F90:200-378': ['SUBROUTINE ice_rst_read( Kbb, Kmm, Kaa )', ('ENDIF', 25), 179],
    'src/ICE/icedyn_adv_pra.F90:487': "IF( lrst_ice )   CALL adv_pra_rst( 'WRITE', kt )   !* write Prather fi",
    'src/ICE/icedyn_adv_pra.F90:1203-1503': ['SUBROUTINE adv_pra_rst( cdrw, kt )', 'END SUBROUTINE adv_pra_rst', 301],
    'src/ICE/iceupdate.F90:438-484': ['SUBROUTINE update_rst( cdrw, kt )', 'END SUBROUTINE update_rst', 47],
    'src/ICE/icestp.F90:281': 'CALL diag_set0                   ! set diag of mass, heat and salt flu',
    'src/ICE/icerst.F90:241-244': ['id_nlay=0', ("znam = 'e_s'//'_l'//zchar1", 2), 4],
    'src/ICE/icerst.F90:369': "IF(lwp) WRITE(numout,*) '  SAS: default initialisation of ss[st]_m arr",
    'src/ICE/icerst.F90:245-255': [('IF( iom_varid( numrir, znam , ldstop = .FALSE. ) > 0 )   id_nlay = id_', 1), 'IF( id0 > 0 .AND. id_nlay == (nlay_s+nlay_i) ) THEN   ! == case of a n', 11],
    'src/ICE/icerst.F90:358-360': [('!                 ! ---------------------------------- !', 2), ('!', 127), 3],
    'src/ICE/icerst.F90:351': 'szv_i(:,:,jk,:) = sv_i(:,:,:) * r1_nlay_i',
    'src/ICE/icestp.F90:382': 'r1_nlay_i = 1._wp / REAL( nlay_i, wp )   !--- inverse of nlay_i and nl',
    'src/ICE/icedyn_adv_pra.F90:1226': "IF( ln_rstart ) THEN   ;   id1 = iom_varid( numrir, 'sxice' , ldstop =",
    'cfgs/ref_cfgs.txt:8': 'ORCA2_SAS_ICE OCE ICE NST SAS',
    'src/ICE/icestp.F90:341-342': ['NAMELIST/nampar/ jpl, nlay_i, nlay_s, ln_virtual_itd, ln_icedyn, ln_ic', '&             cn_icerst_in, cn_icerst_indir, cn_icerst_out, cn_icerst_', 2],
    'src/ICE/iceistate.F90:465-470': ['NAMELIST/namini/ ln_iceini, nn_iceini_file, rn_thres_sst, &', '&             sn_hti, sn_hts, sn_ati, sn_tsu, sn_tmi, sn_smi, sn_tms, ', 6],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90:1-251': [('MODULE o1_si3_state_writer', 1), 'END MODULE o1_si3_state_writer', 251],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90:6': 'USE, INTRINSIC :: iso_fortran_env, ONLY : int32, real64',
    'src/ICE/icestp.F90:116-124': ['INTEGER, INTENT(in) ::   kt       ! ocean time step', "IF( ln_timing )   CALL timing_start('icestp')", 9],
    'src/ICE/icestp.F90:220-228': ['IF( ln_icediachk )             CALL ice_drift_wri( kt )      ! -- Diag', 'ENDIF   ! End sea-ice time step only', 9],
    'src/ICE/icestp.F90:295-300': ['CALL ice_var_glo2eqv(1)', 'CALL ice_update_init             ! ice surface boundary condition', 6],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_fixture_stubs.F90:1-115': [('MODULE par_oce', 1), 'END MODULE lib_mpp', 115],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_layout_fixture.F90:1-271': [('PROGRAM o1_si3_layout_fixture', 1), 'END PROGRAM o1_si3_layout_fixture', 271],
    'src/ICE/icerst.F90:241-255': ['id_nlay=0', 'IF( id0 > 0 .AND. id_nlay == (nlay_s+nlay_i) ) THEN   ! == case of a n', 15],
    "src/OCE/IOM/iom.F90:1347-1431": [
        "! we do not read the overlap and the extra-halos",
        ("CALL lbc_lnk( 'iom', pvdp3d, cl_type, zsgn_dp", 1),
        85,
    ],
    "src/ICE/icerst.F90:275-282": [
        "! --- mandatory fields --- !",
        "CALL iom_get( numrir, jpdom_auto, 'v_ice', v_ice",
        8,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_nemo_restart_fixture.F90:7-34": [
        "INTEGER, PARAMETER :: nx=180, ny=148, nc=5, n3=34, n2=4",
        "CALL ck(nf90_def_var(ncid, TRIM(names2(n)), NF90_DOUBLE, dims2",
        28,
    ],
    "src/ICE/icerst.F90:318-324": [
        "! melt pond lids",
        "v_il(:,:,:) = 0._wp",
        7,
    ],
    "src/ICE/iceupdate.F90:455-461": [
        "IF( ln_rstart ) THEN                   !* Read the restart file",
        "CALL iom_get( numrir, jpdom_auto, 'snwice_mass_b', snwice_mass_b )",
        7,
    ],
    "packages/ice/legoesm/ice/sea_ice.py:590-610": [
        "if config.si3_orca2 is None:",
        '"heat/freshwater/salt TileResponse contract is implemented"',
        21,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90:94-97": [
        "IF( TRIM(cd_stage) == 'entry' ) THEN",
        "WRITE(cl_fname, '(\"o1_si3_records/o1_si3_state_step\", I4.4, \"_rank\", I4.4, \".bin\")') kt, mpprank",
        4,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1778-1843": [
        "# 5. Score rst_1step (Item 5 & Item 6): NetCDF restart read + 1-step trajectory vs rec_2step step 2",
        'rst_sss_step2 = jnp.asarray(rst_snap_step2["fields"]["sss_m"], dtype=jnp.float64)',
        66,
    ],
    "tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1231-1252": [
        "# Write rst_1step binary records and restart",
        'ev / "rst_1step/o1_si3_records/o1_si3_state_step0002_rank0000.bin",',
        22,
    ],
    "src/OCE/SBC/sbcmod.F90:397-403": [
        "! most of the following routines update fields only in the interior",
        "IF( .NOT.ll_sas )   CALL sbc_ssm ( kt, Kbb, Kmm )",
        7,
    ],
    "src/OCE/SBC/sbcmod.F90:474-477": [
        ("SELECT CASE( nn_ice )", 2),
        "CASE(  2 )   ;         CALL ice_stp  ( kt, Kbb, Kmm, nsbc )",
        4,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1755-1787": [
        "state_entry, cfg = o1_snapshot_to_si3_state(snap_entry)",
        "state_kt2 = o1_step_jit(state_kt1, tmask, sss_step2)",
        33,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1830-1852": [
        "rst_snap_entry = stitch_o1_binary_snapshots(",
        "lego_rst_step2 = o1_step_jit(lego_rst_entry, tmask, rst_sss_step2)",
        23,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1955-1962": [
        '"sss_m_kt1_sha256": hashlib.sha256(sss_step1_np.tobytes()).hexdigest(),',
        '"sss_m_kt2_plant_branch_point": cfg.rn_simin / cfg.rn_sinew,',
        8,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1767-1787": [
        "o1_step_jit = jax.jit(",
        "state_kt2 = o1_step_jit(state_kt1, tmask, sss_step2)",
        21,
    ],
    "src/ICE/icevar.F90:362-375": {
        "anchors": [
            "ELSEIF( nn_icesal == 4 ) THEN",
            "s_i(ji,jj,jl) = s_i(ji,jj,jl) + sz_i(ji,jj,jk,jl) * r1_nlay_i",
            14,
        ],
        "routine": "ice_var_glo2eqv",
    },
    "ORCA2_SAS_ICE_O1_REC/BLD/ppsrc/nemo/icevar.f90:376-389": {
        "anchors": [
            "ELSEIF( nn_icesal == 4 ) THEN",
            "s_i(ji,jj,jl) = s_i(ji,jj,jl) + sz_i(ji,jj,jk,jl) * r1_nlay_i",
            14,
        ],
        "routine": "ice_var_glo2eqv",
    },
    "src/ICE/icevar.F90:368-375": {
        "anchors": [
            ("IF ( v_i(ji,jj,jl) > epsi20 ) THEN     !--- icy area", 1),
            "s_i(ji,jj,jl) = s_i(ji,jj,jl) + sz_i(ji,jj,jk,jl) * r1_nlay_i",
            8,
        ],
        "routine": "ice_var_glo2eqv",
    },
    "ORCA2_SAS_ICE_O1_REC/BLD/ppsrc/nemo/icevar.f90:382-389": {
        "anchors": [
            ("IF ( v_i(ji,jj,jl) > epsi20 ) THEN     !--- icy area", 1),
            "s_i(ji,jj,jl) = s_i(ji,jj,jl) + sz_i(ji,jj,jk,jl) * r1_nlay_i",
            8,
        ],
        "routine": "ice_var_glo2eqv",
    },
    "tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1112-1122": [
        "# Make one wet category ice-free so the synthetic kt=2 salinity change is",
        "state2 = si3_o1_step(state1, tmask=tmask, sss_m=sss_step2, config=cfg)",
        11,
    ],
    "tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:1301-1325": [
        "adm = gate.admit_evidence_directory(ev, expect_commit=commit_hex)",
        "gate.admit_evidence_directory(ev, plant=plant_name, expect_commit=commit_hex)",
        25,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1287-1383": [
        "def _compare_si3_states_wet_and_full(",
        ('f"full_ULP={ulp_f}, wet_norm={norm_w:.3e}"', 3),
        97,
    ],
    "tests/ocean/fidelity/test_nemo_si3_o1_orca2_sas_gate.py:822-879": [
        "def test_o1_scorer_includes_defined_seam_and_fold_halos_only() -> None:",
        "equivalent=state.equivalent._replace(a_ip_frac=jnp.asarray(a_ip_frac_changed))",
        58,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1055-1061": [
        "prove_real_same_program() {",
        "printf 'REFUSE: real reference/instrumented same-program comparison failed before NEMO launch\\n' >&2",
        7,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1204-1212": [
        ('build_target "$TARGET_REF_CFG" 0', 2),
        ('stage_and_run "$TARGET_REC_CFG" "$EVIDENCE/rst_1step" 1', 2),
        9,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:199-230": [
        "check_operator_run_order() {",
        '"proof must precede every NEMO launch"',
        32,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:905-913": [
        "# The operator path builds both real executables, proves they are the same",
        'grep -F "STATUS PLANT-FIRED" "$scratch/run_order_plant.log" >/dev/null',
        9,
    ],
    "src/OCE/DOM/daymod.F90:276-277": [
        "IF( .NOT. l_offline ) CALL rst_opn( kt )",
        "IF( lrst_oce         ) CALL day_rst( kt, 'WRITE' )",
        2,
    ],
    "src/OCE/IOM/restart.F90:81-145": [
        "IF( kt == nit000 ) THEN   ! default definitions",
        "lrst_oce = .TRUE.",
        65,
    ],
    "src/OCE/SBC/sbcssm.F90:177-191": [
        "IF( lrst_oce ) THEN",
        "CALL iom_rstput( kt, nitrst, numrow, 'frq_m'  , frq_m  )",
        15,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/namelist_cfg:21-24": [
        "nn_it000    =       1",
        "nn_write    =      -1",
        4,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:180-194": [
        "require_rec_2step_restarts() {",
        '"$ocean_restart" >&2',
        15,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:915-926": [
        "# rec_2step must produce both files consumed by rst_1step.  In particular,",
        ('require_rec_2step_restarts "$scratch/restart_dependency"', 2),
        12,
    ],
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/run.sh:1173-1181": [
        'if [[ "$is_rst_run" -eq 1 ]]; then',
        '"$run_dir/input_restart/ORCA2_SAS_00000001_restart.nc"',
        9,
    ],
    # --- SI3 ladder round 4 (O1) END ---
    # --- SI3 ladder round 7 (O2) BEGIN ---
    'docs/ice/fidelity/si3_ladder_design.md:222': '| O2 | Same diagnostic deck, initialized from the recorded SST/SSS and',
    'src/ICE/iceistate.F90:124-126': ['! basal temperature (considered at freezing point)   [Kelvin]', 't_bo(:,:) = ( t_bo(:,:) + rt0 ) * smask0(:,:)', 3],
    'src/ICE/iceistate.F90:203-207': [('!', 121), 'zht_i_ini(:,:) = si(jp_hti)%fnow(:,:,1) * tmask(:,:,1)', 5],
    'src/ICE/iceistate.F90:262-304': ['! no ice if (sst - Tfreez) >= thresold', ('ENDIF', 4), 43],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:174-175': ['! basal temperature (considered at freezing point)   [Kelvin]', 'CALL eos_fzp( sss_m(:,:), t_bo(:,:), kbnd=0 )', 2],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90:174-175': ['CALL eos_fzp( sss_m(:,:), t_bo(:,:), kbnd=0 )', 't_bo(:,:) = ( t_bo(:,:) + rt0 ) * smask0(:,:)', 2],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90:308-337': ['! no ice if (sst - Tfreez) >= thresold', ('END WHERE', 3), 30],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90:441-459': ['snwice_mass  (:,:) = tmask(:,:,1) * SUM(', 'ENDWHERE', 19],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iceistate.f90:456-459': ['WHERE( ssmask(:,:) == 1._wp )', 'ENDWHERE', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:252-256': [('!', 122), 'zat_i_ini(:,:) = si(jp_ati)%fnow(:,:,1) * tmask(:,:,1)', 5],
    'src/ICE/iceistate.F90:311-316': ['CALL ice_var_itd( zht_i_ini(:,jj)     , zht_s_ini(:,jj)     , zat_i_in', '&                a_ip   (:,jj,:)   ,   h_ip   (:,jj,:)   ,   h_il   (:', 6],
    'src/ICE/iceistate.F90:398-430': ['! 4) Adjust ssh and vertical scale factors to snow-ice mass', ('!', 197), 33],
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:146': 'ln_teos10    = .true.         !  = Use TEOS-10 equation of state',
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:9-50': ['!!              7 - Ice surface boundary conditions    (namsbc)', '&namthd         !   Ice thermodynamics', 42],
    'docs/ice/fidelity/si3_r7_o2_preregister.md:1-143': ['# SI3 Ladder Rung O2 Preregistration — Round 7 (`si3_r7`): Scalar Init', '| **P7** | Preflight checklist items 1–7 and admission provenance gate', 143],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2105-2661': ['class SI3O2InitConfig(NamedTuple):', 'return new_state, o2_istate', 557],
    'packages/ice/legoesm/ice/config.py:528-587': ['from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig  # noqa: E', (')', 159), 60],
    'packages/ice/legoesm/ice/sea_ice.py:364-367': ['if config.thermo_scheme == "si3_orca2_o1":', 'return _si3_orca2_o2_step_with_trace(state, forcing, config, dt)', 4],
    'packages/ice/legoesm/ice/sea_ice.py:643-697': ['def _si3_orca2_o2_step_with_trace(', 'return out_state, response', 55],
    'packages/ice/legoesm/ice/__init__.py:227-251': ['from legoesm.ice.config import build_si3_orca2_o2_config  # noqa: E402', (')', 20), 25],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:1-1073': ['#!/usr/bin/env bash', 'printf \'ACQUISITION_DONE %s\\n\' "$EVIDENCE"', 1073],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1-1599': ['#!/usr/bin/env python3', 'raise SystemExit(main())', 1599],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:309-460': ['! no ice if (sst - Tfreez) >= thresold', 'ENDWHERE', 152],
    'src/OCE/DOM/domutl.F90:91-121': ['SUBROUTINE dom_uniq( puniq, cdgrd )', 'END SUBROUTINE dom_uniq', 31],
    'src/OCE/LBC/lib_mpp.F90:1244-1267': ['!!               scalar to add to the summ yddb', ('END SUBROUTINE DDPDD', 2), 24],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2105-2210': ['class SI3O2InitConfig(NamedTuple):', ('apply_halos: bool = False', 3), 106],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2105-2156': ['class SI3O2InitConfig(NamedTuple):', ('return base._replace(**overrides) if overrides else base', 4), 52],
    'src/ICE/iceistate.F90:453-533': ['!!              values called at the first timestep (nit000)', 'END SUBROUTINE ice_istate_init', 81],
    'cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:14-37': ['!!             12 - Ice salinity                       (namthd_sal)', ('!---------------------------------------------------------------------', 9), 24],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2159-2193': ['class SI3O2ScalarInitialState(NamedTuple):', 'ssh_kmm: Field', 35],
    'packages/ice/legoesm/ice/config.py:578-587': ['def build_si3_orca2_o2_config(', (')', 159), 10],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2196-2210': ['class SI3ORCA2O2Forcing(NamedTuple):', ('apply_halos: bool = False', 3), 15],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2214-2370': ['def _safe_sqrt(x: Field) -> Field:', 'return pfzp', 157],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2318-2370': ['def si3_eos_fzp(', 'return pfzp', 53],
    'src/OCE/TRA/eosbn2.F90:1650-1710': ['SUBROUTINE eos_fzp_2d_t( psal, ktsal, ptf, kttf, pdep, ktdep, kbnd )', 'END SUBROUTINE eos_fzp_2d_t', 61],
    'src/OCE/TRA/eosbn2.F90:69': 'PUBLIC   eos_rab        ! called by ldfslp, zdfddm, trabbl',
    'src/OCE/TRA/eosbn2.F90:1674-1683': [('CASE ( np_teos10, np_seos )      !==  CT,SA (TEOS-10 and S-EOS formula', 3), ('END_2D', 8), 10],
    'src/OCE/TRA/eosbn2.F90:1691-1696': [('CASE ( np_eos80 )                !==  PT,SP (UNESCO formulation)  ==!', 3), ('END_2D', 10), 6],
    'src/ICE/iceistate.F90:125': 'CALL eos_fzp( sss_m(:,:), t_bo(:,:), kbnd=0 )',
    'src/ICE/iceistate.F90:263': 'CALL eos_fzp( sss_m(:,:), ztfrz(:,:), kbnd=nn_hls )',
    'packages/ice/legoesm/ice/si3_orca2_state.py:2229-2257': ['def si3_dom_uniq_t_orca2(', 'return jnp.where((interior > 0.0) & (z2d_linked == ztst), 1.0, 0.0)', 29],
    'src/OCE/DOM/dommsk.F90:203-205': ["CALL dom_uniq( tmask_i, 'T' )", 'smask0_i(:,:) = tmask_i(A2D(0))', 3],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2261-2315': ['def _si3_glob_2dsum_ddpdd_core(field_int: Field, mask_int: Field) -> F', (')', 833), 55],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2285-2293': ['def _si3_glob_2dsum_ddpdd_jvp(', ('return y, dy', 2), 9],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2373-2661': ['def si3_o2_scalar_initial_state(', 'return new_state, o2_istate', 289],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2373-2621': ['def si3_o2_scalar_initial_state(', (')', 980), 249],
    'src/ICE/iceistate.F90:125-126': ['CALL eos_fzp( sss_m(:,:), t_bo(:,:), kbnd=0 )', 't_bo(:,:) = ( t_bo(:,:) + rt0 ) * smask0(:,:)', 2],
    'src/ICE/iceistate.F90:263-266': ['CALL eos_fzp( sss_m(:,:), ztfrz(:,:), kbnd=nn_hls )', ('END WHERE', 2), 4],
    'src/ICE/iceistate.F90:275-303': ['zt_su_ini(:,:) = rn_tsu_ini_n * zmsk(:,:)', ('zhlid_ini(:,:) = 0._wp', 2), 29],
    'src/ICE/iceistate.F90:295-304': ['! make sure ponds = 0 if no ponds scheme', ('ENDIF', 4), 10],
    'src/ICE/iceistate.F90:388-427': ['! ice concentration should not exceed amax', 'ENDWHERE', 40],
    'src/ICE/iceistate.F90:309-348': ['! distribute 1-cat into jpl-cat:', ('END DO', 2), 40],
    'src/ICE/icevar.F90:1211-1272': [('! == thickness and concentration == !', 3), ('END DO', 32), 62],
    'src/ICE/iceistate.F90:388-400': ['! ice concentration should not exceed amax', 'snwice_mass  (:,:) = tmask(:,:,1) * SUM(', 13],
    'src/ICE/icevar.F90:122-130': ['! --- integrated values', 'vt_il    (ji,jj) = SUM( v_il    (ji,jj,:) )', 9],
    'src/ICE/iceistate.F90:407': ('!', 190),
    'src/ICE/iceistate.F90:412-427': [('#if defined key_agrif', 4), 'ENDWHERE', 16],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2601-2615': ['zsm_i_ini=zsm_i_ini,', ('snwice_mass=snwice_mass,', 4), 15],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2489-2675': ['fitted_v_i: Field,', 'snwice_mass = sr(tmask_2d * fortran_sum_categories(mass_by_category, axis=-1))', 187],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:74-82': ['EVIDENCE="$arg2"', ('', 82), 9],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:578-927': ['run_preflight() {', ('}', 92), 350],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:972-1073': [('}', 95), 'printf \'ACQUISITION_DONE %s\\n\' "$EVIDENCE"', 102],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:964-970': ['printf \'ACQUISITION_ADMITTED %s\\n\' "$EVIDENCE"', "printf 'REFUSE: --run requires a clean committed git worktree\\n' >&2", 7],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:929-962': ['admit_all() {', 'existing_commit=$(tr -d \'[:space:]\' < "$EVIDENCE/manifest/producer_com', 34],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:581-583': ['ACTIVE_SCRATCH_DIR=$(mktemp -d "$EVIDENCE_ROOT/.o2-scratch.XXXXXX")', 'export TMPDIR=$scratch', 3],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:186-199': ['verify_work_cfgs_guard() {', ('}', 30), 14],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:125-133': ['check_free_ram_gb() {', ('return 0', 1), 9],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:136-151': ['check_free_disk_kb() {', ('}', 22), 16],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:92-123': ['cleanup_on_exit() {', ('trap cleanup_on_exit EXIT', 1), 32],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:201-389': ['render_rst_namelists() {', ('}', 63), 189],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:391-468': ['verify_excerpt_provenance_and_deck_independence() {', ('}', 68), 78],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:470-576': ['build_orca2_sas_ice_syntax_mods() {', ('}', 72), 107],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_state_writer.F90:1-433': [('MODULE o2_si3_state_writer', 1), 'END MODULE o2_si3_state_writer', 433],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_state_writer.F90:6': 'USE, INTRINSIC :: iso_fortran_env, ONLY : int32, real64',
    'src/ICE/iceistate.F90:306': ('!----------------!', 1),
    'src/ICE/iceistate.F90:438': ('!', 200),
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_fixture_stubs.F90:1-129': [('MODULE par_oce', 1), 'END MODULE lib_mpp', 129],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_layout_fixture.F90:1-316': [('PROGRAM o2_si3_layout_fixture', 1), 'END PROGRAM o2_si3_layout_fixture', 316],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1-1111': ['"""Fidelity and invariance tests for SI3 ladder rung O2 (`si3_r7`, `DE', 'assert body == [src_lines[n - 1] for n in nums]', 1111],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:163-354': ['def test_o2_parent_vs_current_all_callers_bit_identical(monkeypatch) -', '_assert_zero_ulp(p_recipe, planted_rec, "planted_recipe_ssh")', 192],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:119-160': ['def _load_parent_commit_modules(', 'return loaded', 42],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:357-397': ['def test_o2_eos_fzp_teos10_and_eos80_zero_ulp_vs_nemo502() -> None:', (')', 171), 41],
    'src/OCE/TRA/eosbn2.F90:1674-1696': [('CASE ( np_teos10, np_seos )      !==  CT,SA (TEOS-10 and S-EOS formula', 3), ('END_2D', 10), 23],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:400-432': ['def test_o2_dom_uniq_t_and_glob_2dsum_ddpdd_zero_ulp() -> None:', '_assert_zero_ulp([zssh_mul], [o2_state.zsshadj], "zsshadj_r1_rho0_ddpd', 33],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:435-461': ['def test_o2_self_test_orca2_initial_ice_ssh_reuse_and_all_plants_fire(', 'gate.run_self_test(plant=plant_name)', 27],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:921-1210': ['def run_self_test(plant: str | None = None) -> dict[str, Any]:', ('}', 62), 290],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:464-512': ['def test_o2_compiled_gfortran_layout_fixture_and_plants(tmp_path: Path', 'gate.verify_layout_fixture(records_dir, plant=plant_name)', 49],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:515-533': ['def test_o2_ocean_output_log_verifier_and_plants(tmp_path: Path) -> No', 'gate.verify_ocean_output_log(excerpt, plant=plant_name)', 19],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:536-649': ['def test_o2_jit_grad_jvp_and_stop_gradient_plant() -> None:', 'assert float(jnp.linalg.norm(g)) == 0.0', 114],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:652-720': ['def test_o2_production_step_sea_ice_selects_o2_and_validates_config() ', 'step_sea_ice(state0, None, z2, z2, z2, sea_cfg, 0.1, cfg.rDt_ice)', 69],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:723-789': ['def test_o2_same_program_and_objects_equal_plants(tmp_path: Path) -> N', 'gate.same_program(n_b, r_b, pra_patch, ist_patch, stp_patch)', 67],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:359-476': ['def same_program(', ('}', 32), 118],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:792-1085': ['def _write_o2_istate_binary_snapshot(', 'gate.admit_evidence_directory(ev, plant=plant_name, expect_commit=comm', 294],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1213-1521': ['def admit_evidence_directory(', ('}', 74), 309],
    'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1088-1111': ['def test_o2_committed_ocean_output_excerpt_provenance() -> None:', 'assert body == [src_lines[n - 1] for n in nums]', 24],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/ocean.output.excerpt:1-111': ['# Verbatim selected lines from the real NEMO 5.0.2 ORCA2 record output', 'iom_close ~~~ close file: ./geothermal_heating.nc ok', 111],
    # --- SI3 ladder round 7 (O2) END ---
    # ORCA2 round 324 (DECISION 150): rCdU_ice -> rCdU_top.
    'ICE/iceupdate.F90:369': 'tmod_io(ji,jj) = rho0 * drag_io(ji,jj) * SQRT( zmodt )',
    'ICE/iceupdate.F90:391': 'rCdU_ice(ji,jj) = -r1_rho0 * tmod_io(ji,jj) * at_i(ji,jj) * tmask(ji,jj,1)',
    'ZDF/zdfdrg.F90:126': 'rCdU_top(ji,jj) = rCdU_ice(ji,jj)',
    # ORCA2 round 313: V4's internal-wave mixing and V3's double diffusion (D122).
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:302': 'ln_traldf_msc   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:393': 'ln_zdfddm   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:394-395': [
        'rn_avts  =    1.e-4', 'rn_hsbfr =    1.6', 2],
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:396': 'ln_zdfiwm   = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:423': 'ln_mevar    = .false.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:424': 'ln_tsdiff   = .true.',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:134': 'SUBROUTINE tra_zdf_imp(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:175': "( jn == jp_sal .AND. ln_zdfddm )",
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:185': 'zwt(ji,jk) = avt(ji,jj,jk) + akz(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:203': 'zwt(ji,jk) = avs(ji,jj,jk) + akz(ji,jj,jk)',
    # ORCA2 round 312: surface-only runoff (D126) and V1's TKE forms (D135).
    'ORCA2_OMIP_L4/EXP00/namelist_ref:474': 'ln_rnf_depth = .false.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:478': 'ln_rnf_depth_ini = .false.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:480': 'rn_dep_max  = 150.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1236': 'ln_mxl0     = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1243': 'rn_mxl0     =   0.04',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1245-1246': [
        'ln_lc       = .true.', 'rn_lc       =   0.15', 2],
    'ORCA2_OMIP_L4/EXP00/namelist_ref:1252': 'nn_htau     =   1',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:406': 'nn_mxl      =   3',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:410': 'nn_etau     =   1',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:415': 'nn_mxlice = 2',
    # round 315 differentiability probe (legoESM statements, measurement only)
    'ocean_model_latlon_cgrid.py:2111': 'rhs = sr(-sr(horizontal_scaled / e3t_div))',
    # round 317: the drag arguments of the TKE bottom boundary condition
    # (their float() casts were removed in round 319)
    'ocean_model_latlon_cgrid.py:12359-12363': [
        'cd0=_cfg_b.bottom_drag.bottom_drag_cd0,',
        'uc0=_cfg_b.bottom_drag.bottom_drag_uc0,', 5],
    'ocean_pe_latlon_cgrid.py:4303': [('cd0=config.bottom_drag.bottom_drag_cd0,', 1)],
    # round 319: the cast-free cd0 argument; DECISION 142 at the card run length
    'ocean_model_latlon_cgrid.py:12359': [('cd0=_cfg_b.bottom_drag.bottom_drag_cd0,', 1)],
    'nemo_testcase_recipe.py:2095': [('2920,  # DECISION 142 (user, 2026-10-20): one year, deliberate', 1)],
    # round 319 review: the background diffusivity floor against NEMO's avt statement
    'tke.py:2309': 'K_H = jnp.maximum(cfg.kappaH_min, K_M / Pr)',
    'tke.py:2165': 'pdlr = jnp.maximum(0.1, ri_cri / jnp.maximum(ri_cri, zri))',
    'tke.py:2166': 'return 1.0 / pdlr',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:405':
        'p_pdlr(ji,jj,jk) = MAX(  0.1_wp,  ri_cri / MAX( ri_cri , zri )  )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:708': 'zav   = rn_ediff * zmxlm(ji,jk) * zsqen',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:710':
        'p_avt(ji,jj,jk) = MAX( zav, avtb_2d(ji,jj) * avtb(jk) ) * wmask(ji,jj,jk)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:717':
        'p_avt(ji,jj,jk)   = MAX( p_pdlr(ji,jj,jk) * p_avt(ji,jj,jk), avtb_2d(ji,jj) * avtb(jk) )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfiwm.f90:442': 'avtb(:) = 1.e-10_wp',
    'ocean_model_latlon_cgrid.py:2124': 'return sr(rhs - sr(vertical_scaled / e3t_div))',
    'ocean_model_latlon_cgrid.py:2858-2860': [
        'out = r(r(r(q_before[..., None] * base)', '/ q_after)', 3],
    'ocean_model_latlon_cgrid.py:2866': [('return jnp.where(active_3d > 0.5, out, base)', 1)],
    'ocean_model_latlon_cgrid.py:14243': 'dt_mom, _tau_j_v, _e3v1, _r0,',  # lane B's top-stress helper call (round 323)
    'ocean_pe_latlon_cgrid.py:4303-4307': [
        ('cd0=config.bottom_drag.bottom_drag_cd0,', 1),
        ('uc0=config.bottom_drag.bottom_drag_uc0,', 1), 5],
    'ocean_pe_latlon_cgrid.py:4384-4388': [
        ('cd0=config.bottom_drag.bottom_drag_cd0,', 2),
        ('uc0=config.bottom_drag.bottom_drag_uc0,', 2), 5],
    'implicit_solver.py:363-382': [
        '_round_the_multiply(quotient * previous))', 'numerator / pivot(current_diagonal))', 20],
    'internal_wave_mixing.py:444': 'n_pos = safe_sqrt(n2_pos)',
    'tke.py:1082': 'safe_sqrt((jnp.asarray(2.0, e.dtype) * e) / zrn2),',
    'tke.py:3647': 'dissl_new = safe_sqrt(tke_curr) / l_eps_final',
    'shortwave_penetration.py:561': '/ jnp.where((jk <= nkB) & (ze3t != 0.0), ze3t, c(1.0)))',
    # round 314 namelist audit
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
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:835-836': [
        'IF( ln_zdfiwm ) THEN          ! Internal wave-driven mixing',
        'rn_emin  = 1.e-10_wp             ! specific values of rn_emin & rmxl_min are used', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:238': 'SUBROUTINE sbc_rnf_div( phdivn, Kmm )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:289': 'SUBROUTINE sbc_rnf_init( Kmm )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:486-489': [
        'ELSE                                            ! runoffs applied at the surface',
        'h_rnf (ji,jj) = (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:199': 'SUBROUTINE tke_tke( Kbb, Kmm, p_sh2, p_avm, p_avt, p_pdlr )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:531': 'SUBROUTINE tke_avn( Kbb, Kmm, p_avm, p_avt, p_pdlr )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:644-645': [
        'zrn2 = MAX( rn2(ji,jj,jk), rsmall )',
        'zmxlm(ji,jk) = MAX(  rmxl_min,  SQRT( 2._wp * en(ji,jj,jk) / zrn2 )  )', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:690-692': [
        'DO jk =  jpkm1,  2,  -1  ; DO ji = ntsi-( 0), ntei+(  0)   ! from the bottom to the surface : ldown',
        '&    MIN( zmxlm(ji,jk+1) + (e3t_3d(ji,jj,jk+1)', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:354': 'imlc(:) = mbkt(ntsi-(0):ntei+(0),jj) + 1',
    # ORCA2 round 311: stage-1 order and dyn_keg in the ORCA2, G1b and G8 builds.
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:61': 'SUBROUTINE dyn_keg(',
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:63': 'SUBROUTINE stp_2D(',
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:138': {
        'anchors': 'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:141': {
        'anchors': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:144': {
        'anchors': 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/stp2d.f90:161': {
        'anchors': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90:61': 'SUBROUTINE dyn_keg(',
    'ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/dynkeg.f90:129-130': {
        'anchors': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1),
                    ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1),
                    2],
        'routine': 'dyn_keg',
    },
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:63': 'SUBROUTINE stp_2D(',
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:141': {
        'anchors': 'CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:144': {
        'anchors': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:147': {
        'anchors': 'CALL dyn_vor( kt,      Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/stp2d.f90:164': {
        'anchors': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90:61': 'SUBROUTINE dyn_keg(',
    'ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynkeg.f90:129-130': {
        'anchors': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1),
                    ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1),
                    2],
        'routine': 'dyn_keg',
    },
    # --- ORCA2 round 309: OMT-0's np_LIN_dyn lines; V3's EVD trigger and DDM ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171': 'CALL bn2    ( ts(:,:,:,:,Nbb), rab_b, rn2b, Nbb ) ! before Brunt-Vaisala frequency',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:174': 'rn2   = rn2b',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:161': 'zrr = zrau / rn_hsbfr',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:202': 'CASE( np_EEN )           ;   CALL vor_een( kt, Kmm, ncor, puu(:,:,:,Kmm)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:334': 'CALL wzv( kstp, Kbb, Kmm, Kaa, zFu, zFv, ww, np_transport )',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:160-166': [('SELECT CASE( n_dynadv )', 1), 'CALL dyn_zad( kt, Kbb, uu, vv, Krhs )', 7],
    # --- ORCA2 round 307: V1 vertical-mixing statements on rung 10 ---
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
    # --- ORCA2 round 282: G1b statements measured on ORCA2 --------
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:648-651': {
        'anchors': [
            ('CASE ( np_CRV )                           !* Coriolis + relative vorticity', 3),
            '&                        -( e1u(ji  ,jj+1) * pu(ji  ,jj+1,jk) - e1u(ji,jj) * pu(ji,jj,jk) )  ) * r1_e1e2f(ji,jj)',
            4],
        'routine': 'vor_ens',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:692-694': {
        'anchors': [
            'SUBROUTINE vor_een( kt, Kmm, kvor, pu, pv, pu_rhs, pv_rhs )',
            '!!                ***  ROUTINE vor_een  ***',
            3],
        'routine': 'vor_een',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766': {
        'anchors': [
            ('CASE ( np_CRV )                           !* Coriolis + relative vorticity', 4),
            ') * r1_e1e2f(ji,jj)   ) * z1_e3f(ji,jj)',
            5],
        'routine': 'vor_een',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:155-156': [
        'e1e2t (:,:) = e1t(:,:) * e2t(:,:)   ;   r1_e1e2t(:,:) = 1._wp / e1e2t(:,:)',
        'e1e2f (:,:) = e1f(:,:) * e2f(:,:)   ;   r1_e1e2f(:,:) = 1._wp / e1e2f(:,:)',
        2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:509-519': [
        "!                          ! ocean u- and v-depth at mid-step   (separate DO-loops remove the need of a lbc_lnk)",
        ('&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1),
        11],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:530-535': [
        "!                             ! resulting flux at mid-step (not over the full domain)",
        ('zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 1),
        6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:666-678': [
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', 1),
        ('&   ) * ssvmask(ji,jj)', 1),
        13],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:721-730': [
        ('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', 1),
        ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1),
        10],
    # --- ORCA2 round 281: admitted stage-1 source split ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN',
        'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)',
        4,
    ],
    # --- ORCA2 round 290: matched barotropic-correction boundary ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:521-528': [
        '!                 !==  All stages: correct the barotropic component ==!   at Kaa = N+1/3, N+1/2 or N+1',
        ('END DO   ;   END DO', 9),
        8,
    ],
    # --- ORCA2 round 297: kt=7 stage-1 source-order record ----------
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:547-549': [
        'DO jk =  1,  jpkm1  ; DO jj = ntsj-(  0), ntej+(   0) ; DO ji = ntsi-( 0), ntei+(   0)   ! corrected horizontal velocity',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)',
        3,
    ],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:565-592': [
        'IF( .NOT.ln_shuman ) THEN             ! No Shuman averaging- tra_adv_trp can be in tiling loop',
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 2),
        28,
    ],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:617-662': [
        '!                       !==  T-S Tracers  ==!',
        "CALL r175_put3(kstp, kstg, 'adv_s', ts(:,:,:,jp_sal,Krhs))",
        46,
    ],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:664-672': [
        'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )   ! surface boundary condition',
        "CALL r175_put3(kstp, kstg, 'sbc_s', ts(:,:,:,jp_sal,Krhs))",
        9,
    ],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:693-720': [
        'CASE ( 1 , 2 )    !==  Stage 1 & 2  ==!    time stepping',
        "CALL r175_put3(kstp, kstg, 'qco_s', ts(:,:,:,jp_sal,Kaa))",
        28,
    ],
    # --- ORCA2 round 299: kt=7 stage-1 W-transport split ----------
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90:565-569': [
        'IF( .NOT.ln_shuman ) THEN             ! No Shuman averaging- tra_adv_trp can be in tiling loop',
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
        5,
    ],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/stprk3_stg.f90:599-601': [
        "CALL r175_put3(kstp, kstg, 'trp_u', zFu)",
        "CALL r175_put3(kstp, kstg, 'trp_w', zFw)",
        3,
    ],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:248-251': [
        ('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 ) ; DO ji = ntsi-( nn_hls), ntei+(  nn_hls-1)', 2),
        'pFw(ji,jj,jpk) = 0._wp',
        4,
    ],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:267-302': [
        'IF( ll_Fw ) THEN',
        "IF( lwp .AND. kt == kit000 .AND. kstg == 3 )   WRITE(992) l4_canon_3d(ww,'T')",
        36,
    ],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/traadv.f90:314-315': [
        'DO jk =  1,  jpkm1  ; DO jj = ntsj-(  1), ntej+(   1) ; DO ji = ntsi-( 1), ntei+(   1)',
        'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)',
        2,
    ],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/divhor.f90:132-153': [
        'CASE ( np_transport )',
        ('pe3divUh(ji,jj,jk) = hdiv(ji,jj,jk) * (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2),
        22,
    ],
    'ORCA2_OMIP_L4_R297STAGE7/BLD/ppsrc/nemo/sshwzv.f90:277-298': [
        'IF( .NOT. PRESENT( k_ind ) ) THEN',
        '&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)',
        22,
    ],
    # --- ORCA2 round 300: admitted kt=7 W leaf record ------------
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90:248-252': {
        'anchors': [
            ('DO jj = ntsj-( nn_hls)', 2), ('END DO   ;   END DO', 4), 5],
        'routine': 'tra_adv_trp_t',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90:267-315': {
        'anchors': [
            'IF( ll_Fw ) THEN',
            'pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)',
            49],
        'routine': 'tra_adv_trp_t',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/divhor.f90:132-139': {
        'anchors': [
            'CASE ( np_transport )',
            ('END DO   ;   END DO   ;   END DO', 2),
            8],
        'routine': 'div_hor_RK3_t',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/sshwzv.f90:271-298': {
        'anchors': [
            ('DO jj = ntsj-( 1)', 2),
            '&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )  ) * tmask(ji,jj,jk)',
            28],
        'routine': 'wzv_RK3_t',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:136-137': {
        'anchors': ['rDt = r1_3 * rn_Dt', ('r1_Dt = 1._wp / rDt', 1), 2],
        'routine': 'stp_RK3_stg',
    },
    # --- ORCA2 round 300 review: stage-1 Kaa ratio and U transport ---
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:51':
        'INTEGER  :: n_baro_upd =  np_HYB',
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:179': {
        'anchors': 'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:290': {
        'anchors': 'zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:599': {
        'anchors': "CALL r175_put3(kstp, kstg, 'trp_u', zFu)",
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90:150-156': {
        'anchors': ['IF( .NOT.lk_linssh ) THEN',
                    'CALL wzv    ( kt, Kbb, Kbb, Kaa', 7],
        'routine': 'stp_2D',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90:303': {
        'anchors': 'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs',
        'routine': 'stp_2D',
    },
    # --- ORCA2 round 303: fold-band stage-1 V transport mask + top-V exchange ---
    # ln_shuman=F and ln_zdfcst=T (l_zdfsh2=F) on this run: the exchange that
    # runs is the ELSE/ELSE lbc_lnk after the tracer update, not the Shuman one.
    # ORCA2 round 308: G6's vector Krhs order and wzv/zad statements.
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:164': {
        'anchors': 'CALL dyn_keg( kt, nn_dynkeg, Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:152': {
        'anchors': 'r3t(ji,jj,Kaa) =  ssh(ji,jj,Kaa) * r1_ht_0(ji,jj)',
        'routine': 'stp_2D',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:156': {
        'anchors': 'CALL wzv    ( kt, Kbb, Kbb, Kaa , uu(:,:,:,Kbb), vv(:,:,:,Kbb), ww, np_velocity )',
        'routine': 'stp_2D',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297-298': {
        'anchors': ['pww(ji,jj,jk) = pww(ji,jj,jk+1) - (  ze3div(ji,jj,jk)',
                    ('&                               + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) )', 2),
                    2],
        'routine': 'wzv_RK3_t',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzad.f90:112-126': {
        'anchors': ['zWf  = e1e2t(ji  ,jj  ) *   ww(ji  ,jj  ,jk+1)',
                    '&                                            * ( zWdzV(ji,jj) + zzWdzV )',
                    15],
        'routine': 'dyn_zad',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:188-189': {
        'anchors': ['rDt = r1_2 * rn_Dt            ! set time-step : rn_Dt/2',
                    ('r1_Dt = 1._wp / rDt', 2), 2],
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221': {
        'anchors': ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )   ! at N+1/2 (Kaa)', 2),
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:327': {
        'anchors': 'CALL wzv( kstp, Kbb, Kmm, Kaa, uu(:,:,:,Kmm), vv(:,:,:,Kmm), ww, np_velocity )',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:237': {
        'anchors': 'ssh(:,:,Naa) = 2*ssh(:,:,Nbb) - ssh(:,:,Naa)',
        'routine': 'stp_RK3',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:283': {
        'anchors': 'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:211': {
        'anchors': 'vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)',
        'routine': 'dom_msk',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:291': {
        'anchors': 'zFv(ji,jj,jk) = e1v(ji,jj)*(e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk)))',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:826-827': {
        'anchors': [
            ("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp   &", 2),
            "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )",
            2],
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90:211': {
        'anchors': 'vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)',
        'routine': 'dom_msk',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dommsk.f90:232': {
        'anchors': "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )",
        'routine': 'dom_msk',
    },
    # --- ORCA2 round 302: tracer-only EVD (nn_evdm = 0) ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121': {
        'anchors': 'IF( nn_evdm == 1 ) THEN',
        'routine': 'zdf_evd',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:135': {
        'anchors': 'p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)',
        'routine': 'zdf_evd',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90:121': {
        'anchors': 'IF( nn_evdm == 1 ) THEN',
        'routine': 'zdf_evd',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/zdfevd.f90:135': {
        'anchors': 'p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)',
        'routine': 'zdf_evd',
    },
    # --- ORCA2 round 301: stage-1 U-transport factors and un_adv ---
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:284': {
        'anchors': 'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:276': {
        'anchors': 'zub(ji,jj) = un_adv(ji,jj)*(r1_hu_0(ji,jj) /(1._wp+r3u(ji,jj,Kmm))) - uu_b(ji,jj,Kmm)',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282': {
        'anchors': 'zFu(ji,jj,jk) = e2u(ji,jj)*(e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3.f90:215': {
        'anchors': 'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )',
        'routine': 'stp_RK3',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90:570': {
        'anchors': 'un_adv(ji,jj) = un_adv(ji,jj) + za2 * zhU(ji,jj) * r1_e2u(ji,jj)',
        'routine': 'dyn_spg_ts',
    },
    'ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/dynspg_ts.f90:831': {
        'anchors': 'un_adv(:,:) = un_adv(:,:) / r1_wgt2s',
        'routine': 'dyn_spg_ts',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:570': {
        'anchors': 'un_adv(ji,jj) = un_adv(ji,jj) + za2 * zhU(ji,jj) * r1_e2u(ji,jj)',
        'routine': 'dyn_spg_ts',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:831': {
        'anchors': 'un_adv(:,:) = un_adv(:,:) / r1_wgt2s',
        'routine': 'dyn_spg_ts',
    },
    # --- ORCA2 round 291: carried external-mode V target boundary ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:857-860': [
        'IF(.NOT.ll_bt_av ) THEN                          !* Update Kaa barotropic external mode',
        'pssh (:,:,Kaa) = ssha_e(:,:)',
        4,
    ],
    # --- ORCA2 round 292: first valued external-V-target producer ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:517-519': [
        'zhvp2_e(ji,jj) = hv_0(ji,jj) + r1_2 * r1_e1e2v(ji,jj)',
        '&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)',
        3,
    ],
    # --- ORCA2 round 294: raw-hv continuation to the vector V update ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:509-520': [
        '!                          ! ocean u- and v-depth at mid-step   (separate DO-loops remove the need of a lbc_lnk)',
        ('END DO   ;   END DO', 6),
        12,
    ],
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:666-678': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form',
        '&   ) * ssvmask(ji,jj)',
        13,
    ],
    # --- ORCA2 round 288: held completed-stage V-fold partner ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820': [
        ('ELSE', 5),
        ('ENDIF', 35),
        9,
    ],
    # --- ORCA2 round 289: completed-stage V association split ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/lbcnfd.f90:684-717': [
        ("CASE ( 'V' )                               ! V-point", 5),
        ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = psgn(jf) * ptab(jf)%pt4d(ii2,ij2,jk,jl)', 57),
        34,
    ],
    # --- ORCA2 round 283: stage-1 V-mask operand owner ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:136':
        'rDt = r1_3 * rn_Dt',
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:211':
        'vmask(ji,jj,jk) = tmask(ji,jj  ,jk) * tmask(ji  ,jj+1,jk)',
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:225':
        'IF (mbkv(ji,jj)==1) vmask(ji,jj,:) = 0._wp',
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:232':
        "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )",
    # --- ORCA2 round 286: held resolved stage-mask statement ---
    'ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232': [
        '!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)',
        "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )",
        27,
    ],
    'ocean_model_latlon_cgrid.py:6624-6628': [
        ('if isinstance(_zc, OceanPartialCellCoordinate):', 1),
        '_ws_v_live_mask = _ws_v_live_mask.astype(state.eta.data.dtype)',
        5,
    ],
    # --- ORCA2 round 278: Decision-122 lines in the shipped ORCA2 build ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:192-198': ['zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )     +  avm(ji,jj,jk  )     ) &', 'zwd(ji,jk) = 1._wp - zzwi - zzws', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:361-367': ['zzwi = - zDt_2 * ( avm(ji,jj+1,jk  )     +  avm(ji,jj,jk  )     ) &', 'zwd(ji,jk) = 1._wp - zwi(ji,jk) - zws(ji,jk)', 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:231-233': ['zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_1d(jk) *(1._wp+r3t(ji,jj,Kmm', ('zwd(ji,jk) = (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))', 2), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:198-199': ['Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)*uu(ji,jj,1:jpkm1,Krhs)*umas', 'Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmas', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:221-223': [('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )   ! at N+1/2 (Kaa)', 2), ('r3v(:,:,Kaa) = r1_2 * ( r3v(:,:,Kbb) + r3va(:,:) )', 2), 3],
    # --- ORCA2 round 318: north-fold set under Decision 122 ---
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
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:791-793': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:", 1), (', avm_k(:,:,:)        , \'W\',  1._wp, ldfull=.TRUE. ) !  lbc_lnk needed for zdf_sh2', 2), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:795-796': [("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:", 2), ("&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,", 2), 2],
    # --- ORCA2 round 275: G2 live momentum-thickness divisors -------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90:191-205': [
        ('DO jk =    2,  jpkm1,  1', 1),
        ('zwd(ji,1) = 1._wp - zzws', 1),
        15],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90:360-375': [
        ('DO jk =     2,  jpkm1,  1', 2),
        ('zwd(ji,1) = 1._wp - zzws', 2),
        16],
    # --- ORCA2 round 268: transferred G1 EEN statement and remaining row ---
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770': [
        ('CASE ( np_CRV )                           !* Coriolis + relative vorticity', 4),
        ('zwz(ji,jj) = ( zwz(ji,jj) - ff_f(ji,jj) ) * fmask(ji,jj,jk) + ff_f(ji,jj)', 3),
        9],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:784-785': [
        ('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d(ji,jj,jk)', 3),
        ('zwy(ji,jj) = e1v(ji,jj) * (e3v_3d(ji,jj,jk)', 3),
        2],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:790-802': [
        ('ztne(ji,jj) = zwz(ji-1,jj  ) + zwz(ji  ,jj  ) + zwz(ji  ,jj-1)', 1),
        ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2),
        13],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:797-802': [
        ('zua = + r1_12 * r1_e1u(ji,jj)', 1),
        ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2),
        6],
    # --- ORCA2 round 269: remaining transferred G1 statements -------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:361-364': [
        ('hu_e  (:,:) =    (hu_0(:,:) *(1._wp+r3u(:,:,Kmm)))', 1),
        ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 1),
        4],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:509-520': [
        ('ocean u- and v-depth at mid-step', 1),
        ('END DO   ;   END DO', 6),
        12],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1077': [
        'rDt_e = rn_Dt / REAL( nn_e , wp )'],
    # --- ORCA2 round 255: substep-2 SSH writer boundary split -------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:550-558': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1),
        ('END DO   ;   END DO', 9),
        9],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739': [
        ('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)', 1),
        ('ENDIF', 32),
        19],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:781-793': [
        ('!* Swap', 1),
        ('sshn_e (:,:) = ssha_e(:,:)', 1),
        13],
    'barotropic_latlon_cgrid.py:2069-2115': [
        ('if _continuity_evaluation == "nemo_literal":', 1),
        ('eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, area)', 1),
        47],
    'barotropic_latlon_cgrid.py:2369-2408': [
        ('association_post = None', 1),
        ('V_bar_new = association_post[1]', 2),
        40],
    'barotropic_latlon_cgrid.py:2437-2442': [
        ('if ab3_za is not None:', 5),
        ('U_bar_c, Ub_c, V_bar_c, Vb_c, eta_c, etab_c)', 1),
        6],
    # --- ORCA2 round 254: south-row audit and midpoint operand split ---
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:728-730': [
        ("CALL lbc_lnk( 'dynspg_ts', ua_e , 'U', -1._wp, va_e , 'V', -1._wp", 2),
        ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1),
        3],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:584-638': [
        ("CASE ( 'T' , 'W' )                         ! T-, W-point", 5),
        ('END DO   ;   END DO', 4),
        55],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:791-793': [
        ('sshbb_e(:,:) = sshb_e(:,:)', 1),
        ('sshn_e (:,:) = ssha_e(:,:)', 1),
        3],
    # --- ORCA2 round 256: completed-stage fold association ----------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802': [
        ('!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', 4),
        'ENDIF            ',
        24],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:639-683': [
        ("CASE ( 'U' )                               ! U-point", 5),
        ('END DO   ;   END DO', 5),
        45],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721': [
        ("CASE ( 'V' )                               ! V-point", 5),
        ('END DO   ;   END DO', 6),
        38],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414': [
        ('zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_3d(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)', 1),
        ('zvb(ji,jj) = vv_b(ji,jj,Kaa) - SUM( e3v_3d(ji,jj,:)*vv(ji,jj,:,Kaa) ) * r1_hv_0(ji,jj)', 1),
        2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:417-418': [
        ('uu(ji,jj,jk,Kaa) = uu(ji,jj,jk,Kaa) + zub(ji,jj)*umask(ji,jj,jk)', 1),
        ('vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 1),
        2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:138': ('CALL dyn_hpg( kt, Kbb     , uu, vv, Krhs )', 1),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:192-198': [
        ('zzwi = - zDt_2 * ( avm(ji+1,jj,jk  )', 1),
        ('zwd(ji,jk) = 1._wp - zzwi - zzws', 1),
        7],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:361-367': [
        ('zzwi = - zDt_2 * ( avm(ji,jj+1,jk  )', 1),
        ('zwd(ji,jk) = 1._wp - zwi(ji,jk) - zws(ji,jk)', 1),
        7],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:231-233': [
        ('zwi(ji,jk) = - p2dt * zwt(ji,jk  )', 1),
        ('zwd(ji,jk) = (e3t_3d(ji,jj,jk)', 2),
        3],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179': [
        ('Ue_rhs(ji,jj) = SUM( e3u_3d(ji,jj,1:jpkm1)', 1),
        ('Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)', 1),
        2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217': [
        ('r3t(:,:,Kaa) = r1_2 * ( r3t(:,:,Kbb) + r3ta(:,:) )', 2),
        ('r3v(:,:,Kaa) = r1_2 * ( r3v(:,:,Kbb) + r3va(:,:) )', 2),
        3],
    # --- GRIDLADDER round 6 END ---
    # --- GRIDLADDER round 7 (vorticity order, stage 3, stage F ratio, V exchange) ---
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:380-394': [
        ('zhpi(ji,jj) = zcoef0 * r1_e1u(ji,jj)', 1),
        ('pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap', 1),
        15],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:766-767': [
        ('zwx(ji,jj) = e2u(ji,jj) * (e3u_3d(ji,jj,jk) *(1._wp+r3u(ji,jj,Kmm)*umask(ji,jj,jk))) * pu(ji,jj,jk)', 3),
        ('zwy(ji,jj) = e1v(ji,jj) * (e3v_3d(ji,jj,jk) *(1._wp+r3v(ji,jj,Kmm)*vmask(ji,jj,jk))) * pv(ji,jj,jk)', 3),
        2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:779-784': [
        ('zua = + r1_12 * r1_e1u(ji,jj) * (  ( ztne(ji,jj  ) * zwy(ji  ,jj  ) + ztnw(ji+1,jj) * zwy(ji+1,jj  ) )', 1),
        ('pv_rhs(ji,jj,jk) = pv_rhs(ji,jj,jk) + zva', 2),
        6],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:145-150': [
        ('puu(ji,jj,jk,Kaa) = (        ( 1._wp + r3u(ji,jj,Kbb) ) * puu(ji,jj,jk,Kbb )', 1),
        ('/          ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 1),
        6],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:160-161': [
        ('puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b(ji,jj,Kaa) ) * umask(ji,jj,jk)', 1),
        ('pvv(ji,jj,jk,Kaa) = ( pvv(ji,jj,jk,Kaa) - vv_b(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 1),
        2],
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:218': ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:239': ('r3f(:,:    ) = r1_2 * ( r3fb(:,:) + r3fa(:,:) )', 1),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546': ('r3f(:,:) = r3fa(:,:)', 1),
    'ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:572-573': [
        ("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp   &", 2),
        ("ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 2),
        2],
    # --- GRIDLADDER round 7 END ---
    # --- GRIDLADDER round 1 (G1 survey) BEGIN ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1056-1080': [
        ('zcmax = MAXVAL( zcu(Nis0:Nie0,Njs0:Nje0) )', 1),
        ("IF( zcmax>0.9_wp )   CALL ctl_stop( 'dynspg_ts ERROR: Maximum Courant number is greater than 0.9 => increase nn_e' )", 1),
        25],
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:28': ('rn_Dt       = 10800.     !  time step for the dynamics and tracer', 1),
    'cfgs/SHARED/namelist_ref:72': ('rn_Dt       = 5400.     !  time step for the dynamics and tracer', 1),
    'cfgs/SHARED/namelist_ref:1099-1101': [
        ('ln_bt_auto    = .true.     ! Number of sub-step defined from:', 1),
        ('nn_e         =  30         ! =F : the number of sub-step in rn_Dt seconds', 1),
        3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:395-400': [
        ('ELSE                                  !* no wet and dry', 1),
        ('CALL usr_def_istate_ssh( tmask, ssh(:,:,Kbb) )', 2),
        6],
    'src/OCE/IOM/restart.F90:457-461': [
        ('IF( ln_read_cfg ) THEN                 ! read configuration : ssh is set to 0', 1),
        ('CALL usr_def_istate_ssh( tmask, ssh(:,:,Kbb) )', 2),
        5],
    'cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm:1': ('bld::tool::fppkeys   key_si3   key_qco key_vco_1d3d key_RK3', 1),
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:365-366': [
        ('nn_bt_flt     = 3          ! Time filtering with dissipative Forward Backward', 1),
        ('rn_bt_alpha   = 0.09       ! Dissipative Forward Backward parameter', 1),
        2],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:65-68': [
        ('pts(:,:,:,jp_tem) = 20._wp', 1),
        ('pv( :,:,:       ) =  0._wp', 1),
        4],
    'tests/TSUNAMI/MY_SRC/usrdef_istate.F90:94-102': [
        ('zdist(ji,jj) = SQRT( glamt(ji,jj)**2 + gphit(ji,jj)**2 )', 1),
        ('END_2D', 2),
        9],
    'tests/TSUNAMI/MY_SRC/usrdef_sbc.F90:60-68': [
        ('utau(:,:) = 0._wp', 1),
        ('qsr (:,:) = 0._wp', 1),
        9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:185': ('IF( ln_dynadv_OFF  ) THEN   ;   ioptio = ioptio + 1   ;   n_dynadv = np_LIN_dyn   ;   ENDIF', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:626': ('IF( ln_traadv_OFF ) THEN   ;   ioptio = ioptio + 1   ;   nadv = np_NO_adv   ;   ENDIF', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/istate.f90:129-133': [
        ('IF( .NOT. ln_tsd_init .AND. .NOT. ln_uvd_init ) THEN', 1),
        ('CALL usr_def_istate( zgdept, tmask, ts(:,:,:,:,Kbb), uu(:,:,:,Kbb), vv(:,:,:,Kbb) )', 1),
        5],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:442': ('CASE( jp_usr     )   ;   CALL usrdef_sbc_oce( kt, Kbb )                        ! user defined formulation', 1),
    'tools/DOMAINcfg/src/domhgr.F90:484-502': [
        ("CALL iom_get( inum, jpdom_global, 'glamt', glamt, cd_type = 'T', psgn = 1._wp )", 1),
        ("CALL iom_get( inum, jpdom_global, 'e2f'  , e2f , cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1),
        19],
    'tools/DOMAINcfg/src/domhgr.F90:383-384': [
        ('ff_f(:,:) = 2. * omega * SIN( rad * gphif(:,:) )', 1),
        ('ff_t(:,:) = 2. * omega * SIN( rad * gphit(:,:) )     !    -        -       - at t-point', 1),
        2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:233-237': [
        ("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1),
        ("CALL iom_get( inum, jpdom_global, 'ff_t', pff_t, cd_type = 'T', psgn = 1._wp, kfill = jpfillcopy )", 1),
        5],
    'tools/DOMAINcfg/src/domzgr.F90:425-437': [
        ('IF( ppkth == 0._wp ) THEN            !  uniform vertical grid', 1),
        ('e3t_1d  (jk) =  za1', 1),
        13],
    'tools/DOMAINcfg/src/domzgr.F90:961-984': [
        ('SUBROUTINE zgr_zco', 1),
        ('END SUBROUTINE zgr_zco', 1),
        24],
    'tools/DOMAINcfg/src/domzgr.F90:625-629': [
        ('IF( ln_zco )   THEN                          ! zco : read level bathymetry', 1),
        ('mbathy(:,:) = INT( bathy(:,:) )', 1),
        5],
    'tools/DOMAINcfg/src/domzgr.F90:633-652': [
        ('IF( cp_cfg == "orca" .AND. jp_cfg == 2 ) THEN    ! ORCA R2 configuration', 1),
        ('END DO', 22),
        20],
    'tools/DOMAINcfg/src/domzgr.F90:728-733': [
        ('IF( rn_hmin < 0._wp ) THEN    ;   ik = - INT( rn_hmin )                                      ! from a nb of level', 1),
        ('ELSE WHERE ( risfdep == 0._wp );   bathy(:,:) = MAX(  zhmin , bathy(:,:)  )   ! min=zhmin over the oceans', 1),
        6],
    'tools/DOMAINcfg/src/domain.F90:417': ("CALL iom_rstput( 0, 0, inum, 'jperio', REAL( jperio, wp), ktype = jp_i4 )", 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:595-604': [
        ("IF( iatt == -999 .AND. catt == 'UNKNOWN' .AND. iom_varid( inum, 'jperio', ldstop = .FALSE. ) > 0 ) THEN", 1),
        ('ENDIF', 32),
        10],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dommsk.f90:289-291': [
        ("IF( iom_varid( inum, 'strait_shlat', ldstop = .FALSE. ) > 0 ) THEN", 1),
        ("CALL iom_get( inum, jpdom_global, 'strait_shlat', zstrshlat, cd_type = 'F', psgn = 1._wp, pval = -1._wp )", 1),
        3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domzgr.f90:179-188': [
        ('ELSEIF( lk_vco_1d3d ) THEN       !--  z-partial cells  --!   use 3D t-level e3', 1),
        ("CALL iom_get( inum, jpdom_global, 'e3f_0'  , e3f_3d, cd_type = 'F', psgn = 1._wp, kfill = jpfillcopy )", 1),
        10],
    'tripole.py:453': ('def create_tripole_grid(', 1),
    'nemo_testcase_recipe.py:1296': ('def _orca2_masks(bottom_level: np.ndarray, strait_shlat: np.ndarray):', 1),
    'nemo_testcase_recipe.py:1277': ('def _orca2_depth_ladder(e3t_1d: np.ndarray, e3w_1d: np.ndarray):', 1),
    'vertical.py:293': ('def nemo_fe3mask_from_tmask(tmask, *, grid=None):', 1),
    'vertical.py:922': ('class NemoEENBarotropicOperands(NamedTuple):', 1),
    'vertical.py:685': ('def nemo_qco_resolved_mesh_operands(', 1),
    'ocean_model_latlon_cgrid.py:4929': ('_nemo_filter_alpha = config.barotropic.nemo_barotropic_filter_alpha', 1),
    'nemo_testcase_recipe.py:4050': ('def build_orca2g1_flat_card(deck_root: str | Path) -> NEMOTestcaseCard:', 1),
    'nemo_testcase_recipe.py:3951': ('def read_orca2g1_deck(deck_root: str | Path) -> ORCA2G1Deck:', 1),
    'nemo_testcase_recipe.py:3999': ('def orca2g1_initial_ssh(glamt_deg, gphit_deg, surface_tmask) -> np.ndarray:', 1),
    'nemo_testcase_recipe.py:4263': ('def validate_orca2g1_flat_card(card: NEMOTestcaseCard, deck_root: str | Path) -> None:', 1),
    'vertical.py:1127': ('nemo_e3w_source="mesh_reference", allow_single_level: bool = False,', 1),
    'nemo_testcase_recipe.py:4113': ('allow_single_level=True,               # jpk = 2: one wet level', 1),
    # --- GRIDLADDER round 1 (G1 survey) END ---
    # --- ORCA2 round 253: kt=8 midpoint-SSH failure boundary -------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:489-493': [
        ('Extrapolate Sea Level at step jit+0.5', 1),
        ('zsshp2_e(:,:) = za1 * sshn_e(:,:)', 1),
        5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1496-1531': [
        ('SUBROUTINE ts_bck_interp', 1),
        ('END SUBROUTINE ts_bck_interp', 1),
        36],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:546-558': [
        ('Compute Sea Level at step jit+1', 1),
        ('END DO   ;   END DO', 9),
        13],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:655-718': [
        ('Set next velocities:', 1),
        ('END DO   ;   END DO', 20),
        64],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178': [
        ('ssha(:,:) = ssh (:,:,Kaa)', 1),
        ('r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb)', 1),
        41],
    # --- ORCA2 round 261: stage-2 V-transport operand split ---------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170': [
        ('CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )', 1),
        ("r3va, 'V', 1._wp", 1),
        3],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217': [
        ('DO jj = ntsj-( nn_hls), ntej+(  nn_hls-1 )', 1),
        ('* r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1),
        5],
    # --- ORCA2 round 265: matched stage-2 U RHS boundary -----------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:360-424': [
        ('SELECT CASE( kstg )', 2),
        ('END SELECT', 6),
        65],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:443-455': [
        ('Round-11 L2 WRITE-only composition boundary', 1),
        ('L2_RK_STAGE2_PREUPDATE_DUMP', 1),
        13],
    # --- ORCA2 round 266: stage-2 HPG accumulator assignment ------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:372-405': [
        ('hydrostatic pressure gradient (HPG))', 1),
        ("l4_canon_3d(uu(:,:,:,Krhs),'U')", 2),
        34],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:407-421': [
        ('Coriolis / vorticity', 1),
        ('L2_RK_STAGE2_TERM_DUMP', 1),
        15],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:395-406': [
        ('zuap = -zcoef0', 1),
        ('l2_sum_u (ji,jj,1) = zhpi(ji,jj) + zuap', 1),
        12],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:411-432': [
        ('DO jk= 2, jpkm1', 1),
        ('l2_sum_v (ji,jj,jk) = zhpj(ji,jj) + zvap', 1),
        22],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:401-402': [
        ('puu(ji,jj,1,Krhs) = zhpi(ji,jj) + zuap', 1),
        ('pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap', 1),
        2],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:426-427': [
        ('puu(ji,jj,jk,Krhs) = zhpi(ji,jj) + zuap', 1),
        ('pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 1),
        2],
    'barotropic_latlon_cgrid.py:2025-2044': [
        ('eta_mid = eta_c', 1),
        ('za_i, eta_c, etab_c, etabb_c)', 1),
        20],
    # --- ORCA2 round 252: direct post-drag surface-mask boundary ----
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:198-230': [
        ('Ue_rhs(ji,jj) = SUM( e3u_3d', 1),
        ('Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0', 1),
        33],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1462-1463': [
        ('pu_RHSi(ji,jj) = pu_RHSi(ji,jj) + (r1_hu_0', 1),
        ('pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0', 1),
        2],
    # --- ORCA2 round 250: literal drag replay retracts round 249 -----
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/zdfdrg.f90:224-249': [
        ('REAL(wp), DIMENSION(Nis0-(2):Nie0+(2),Njs0-(2):Nje0+(2)), INTENT(inout) ::   pCdU', 1),
        ('pCdU(ji,jj) = - pCd0(ji,jj) * SQRT', 1),
        26],
    # --- ORCA2 round 249: first private-unit growth boundary ---------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:194-230': [
        ('!*  vertical averaging  *!', 1),
        ('Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0', 1),
        37],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:286-324': [
        ('! set values computed in RK3_ssh', 1),
        ('zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd', 1),
        39],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463': [
        ('SUBROUTINE dyn_drg_init', 1),
        ('pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0', 1),
        60],
    # --- ORCA2 round 245: OMT-4 midpoint V-depth cancelling pair ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dommsk.f90:206-249': [
        '!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)',
        'ssvmask(:,:) = MAXVAL( vmask(:,:,:), DIM=3 )',
        44],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:945-958': [
        ("IF( c_NFtype == 'T' ) THEN            ! *  North fold  T-point pivot", 4),
        ('ptab(ji,ipjm1) = psgn * ptab(ijt,ipjm1)', 3),
        14],
    # --- ORCA2 round 243: OMT-4 substep-6 V-transport operand split ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:460-484': [
        '!* Set extrapolation coefficients for predictor step:',
        'va_e(ji,jj) = za1 * vn_e(ji,jj) + za2 * vb_e(ji,jj) + za3 * vbb_e(ji,jj)',
        25],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:487-519': [
        'IF( .NOT.lk_linssh ) THEN                        !* Update ocean depth (variable volume case only)',
        ('+ e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1),
        33],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:530-535': [
        '! resulting flux at mid-step (not over the full domain)',
        'zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)',
        6],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:574-580': [
        'WRITE(l2_adv_unit) jn, za2',
        "& l4_canon_2d(vn_adv,'V')", 7],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:755-779': [
        ('IF( lwp .AND. kt == nit000 .AND. jn <= 2 ) THEN', 3),
        ("& l4_canon_2d(ffv_nw,'V'), l4_canon_2d(ffv_ne,'V')", 2),
        25],
    'barotropic_latlon_cgrid.py:789-809': [
        'def nemo_literal_metric_transports(', 'return zh_u, zh_v', 21],
    # --- ORCA2 round 227: OMT-4 stage-3 implicit tracer solve ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:171-216': [
        'Matrix construction', 'zwt(:,1) = 0._wp', 46],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:218-235': [
        'Diagonal, lower (i), upper (s)', ('ENDIF', 11), 18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:249-273': [
        'Matrix inversion from the first level',
        ('END DO   ;   END DO', 11),
        25],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/trazdf.f90:283-299': [
        '2nd recurrence', ('END DO   ;   END DO', 14), 17],
    # --- ORCA2 round 205: OMT-0 split-explicit statement walk ---
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stp2d.f90:278-281': [
        'sshe_rhs(:,:) =                 emp(:,:)',
        'sshe_rhs(:,:) = r1_rho0 * sshe_rhs(:,:)', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:286-289': [
        'set values computed in RK3_ssh',
        'zv_frc(:,:) =   Ve_rhs(:,:)', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:289,320-324': [
        'zv_frc(:,:) =   Ve_rhs(:,:)',
        'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )',
        'zv_frc(ji,jj) = zv_frc(ji,jj) - zv_trd(ji,jj) * ssvmask(ji,jj)', 6],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:712-741': [
        'IF( .NOT.lk_linssh ) THEN !* Update ocean depth',
        'IF( ln_bdy )   CALL bdy_dyn2d', 30],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:532,535': [
        'zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)',
        'zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 2],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:461-484': [
        'IF ((jn<3).AND.ll_init) THEN',
        'va_e(ji,jj) = za1 * vn_e(ji,jj) + za2 * vb_e(ji,jj) + za3 * vbb_e(ji,jj)', 24],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:681-702': [
        'ELSE                           !* Flux form',
        ('END DO   ;   END DO', 18), 22],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:666-679': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form',
        ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:698-701': [
        'va_e(ji,jj) = (               hv_e  (ji,jj) *   vn_e (ji,jj)',
        '(hv_0(ji,jj) *(1._wp+r3v(ji,jj,Kmm))) * zv_frc (ji,jj)', 4],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:781-789': [
        '!* Swap',
        'vn_e   (:,:) = va_e  (:,:)', 9],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:512-519': [
        'zhup2_e(ji,jj) = hu_0(ji,jj)',
        ('e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)', 1), 8],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:721-730': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth',
        "hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 10],
    # --- ORCA2 round 199: Decision-103 OMT-0 module selections ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:162-190': [
        'NAMELIST/namdyn_adv/ ln_dynadv_OFF, ln_dynadv_vec',
        "CALL ctl_stop( 'choose ONE and only ONE advection scheme' )", 29],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:177-228': [
        'NAMELIST/namdyn_ldf/ ln_dynldf_OFF',
        'IF(.NOT.ln_dynldf_OFF ) THEN', 52],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:586-633': [
        'NAMELIST/namtra_adv/ ln_traadv_OFF',
        "CALL ctl_stop( 'tra_adv_init: Choose ONE advection option in namelist namtra_adv' )", 48],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:214-268': [
        'NAMELIST/namtra_ldf/ ln_traldf_OFF',
        'IF( .NOT.ln_traldf_OFF ) THEN', 55],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:371-401': [
        'NAMELIST/namdrg/ ln_drg_OFF, ln_lin',
        "CALL ctl_stop( 'zdf_drg_init: Choose ONE type of drag coef in namdrg' )", 31],
    # --- ORCA2 round 176: the independent rung-0 entry guard ---
    'nemo_testcase_recipe.py:1428-1479': [
        'def apply_orca2_hand_alterations(temperature, salinity):',
        'is what NEMO executes.', 52],
    'nemo_testcase_recipe.py:1762-1763': [
        'temperature, salinity = build_orca2_initial_ts(',
        'temperature_path, salinity_path, tmask)', 2],
    'ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/dtatsd.f90:218-254': [
        'IF( cn_cfg == "orca" .OR. cn_cfg == "ORCA" ) THEN',
        ('ENDIF', 9), 37],
    'ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:139-181': [
        'ssha(:,:) = ssh (:,:,Kaa)',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'round175/orca2_rung0_stage1_ranked_10step_np2/namelist_cfg:51':
        'ln_tsd_dmp  = .false.',
    'nemo_testcase_l4_orca2_round174_stage_growth_gate.py:224':
        'card, state, freshwater, surface = r166._setup(deck_root, record_root)',
    'nemo_testcase_l4_orca2_round166_external_substep_gate.py:223-230': [
        'admission = rung0.frames.admit(record_root, ladder.EXPECTED_PRODUCER, None)',
        'return card, initial, freshwater, surface', 8],
    # --- ORCA2 round 175: independent kt=1 stage-1 record boundary ---
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:137-179': [
        'ssha(:,:) = ssh (:,:,Kaa)',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 43],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:469-479': [
        'uu(ji,jj,jk,Kaa) = ( uu(ji,jj,jk,Kbb) + rDt * uu(ji,jj,jk,Krhs) )',
        '&             /           ( 1._wp + r3v(ji,jj,Kaa) ) * vmask(ji,jj,jk)', 11],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:513-537': [
        'DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)             ! barotropic velocity correction',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 25],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:555,578': [
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 2), 2],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:637,645': [
        'CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )',
        'CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 2],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:678-680': [
        'ts(ji,jj,jk,jn,Kaa) = (        ( 1._wp + r3t(ji,jj,Kbb) )*ts(ji,jj,jk,jn,Kbb )',
        '&                /          ( 1._wp + r3t(ji,jj,Kaa) )', 3],
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:791-796': [
        ("CALL lbc_lnk( 'stp_RK3_stg', uu(:,:,:,       Kaa), 'U', -1._wp, vv(:,:,:       ,Kaa), 'V', -1._wp", 1),
        "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )", 6],
    # --- ORCA2 round 173: kt=8 HPG operand instrument hold -------
    'ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/stp2d.f90:145-149': [
        ('!*  hydrostatic pressure gradient (HPG))  *!', 1),
        ("CALL r92_rhs_put_pair( kt, 'hpg'", 1),
        5,
    ],
    'ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:402-429': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2),
        ('r172_sum_v (ji,jj,1) = zhpj(ji,jj) + zvap', 1),
        28,
    ],
    'ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:433-460': [
        ('DO jk= 2, jpkm1', 1),
        ('r172_sum_v (ji,jj,jk) = zhpj(ji,jj) + zvap', 1),
        28,
    ],
    # --- ORCA2 round 183: independent HPG fold-operand walk -------
    'ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:409-416': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2),
        ('&           - (e3w_1d(1) *(1._wp+r3t(ji  ,jj  ,Kmm))) * rhd(ji  ,jj  ,1)  )', 2),
        8,
    ],
    'ORCA2_OMIP_L4_R182HPGFOLD/BLD/ppsrc/nemo/dynhpg.f90:445-453': [
        'DO jk= 2, jpkm1',
        '&           - (e3w_1d(jk) *(1._wp+r3t(ji,jj  ,Kmm))) * ( rhd(ji,jj,  jk) + rhd(ji,jj  ,jk-1) )  )',
        9,
    ],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:193-200': [
        'ht_0(:,:) = 0._wp  ! Reference ocean thickness',
        'hv_0(:,:) = hv_0(:,:) + e3v_3d(:,:,jk) * vmask(:,:,jk)',
        8,
    ],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/domain.f90:212-215': [
        'r1_ht_0(:,:) = ssmask (:,:) / ( ht_0(:,:) + 1._wp -  ssmask (:,:) )',
        'r1_hf_0(:,:) = ssfmask(:,:) / ( hf_0(:,:) + 1._wp -  ssfmask(:,:) )',
        4,
    ],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dommsk.f90:206-232': [
        '!  Ocean/land mask at u-, v-, and f-points   (computed from tmask)',
        "CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp, vmask, 'V', 1.0_wp, fmask, 'F', 1.0_wp )",
        27,
    ],
    # --- ORCA2 round 172: kt=8 passive HPG owner ------------------
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90:386-402': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2),
        'pvv(ji,jj,1,Krhs) = zhpj(ji,jj) + zvap',
        17,
    ],
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/dynhpg.f90:411-434': [
        'DO jk= 2, jpkm1',
        ('         END DO   ;   END DO', 1),
        24,
    ],
    # --- ORCA2 round 171: kt=8 RHS observer hold ------------------
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:145-157': [
        ('!*  hydrostatic pressure gradient (HPG))  *!', 1),
        ("CALL r92_rhs_put_pair( kt, 'vor'", 1),
        13,
    ],
    'ORCA2_OMIP_L4_R170RHS8/BLD/ppsrc/nemo/stp2d.f90:168-179': [
        ('!*  KEG + ZAD  *!', 1),
        ('CALL r92_rhs_finish( kt )', 1),
        12,
    ],
    # --- ORCA2 round 170: kt=8 slow-forcing producer walk ---------
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:145-179': [
        ('!*  hydrostatic pressure gradient (HPG))  *!', 1),
        ('CALL r92_rhs_finish( kt )', 1),
        35,
    ],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:190-196': [
        ('CALL r169_start( kt, Kbb, Kaa, Krhs )', 1),
        ("CALL r169_put3( kt, 'vmask', vmask )", 1),
        7,
    ],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:215-227': [
        ('!*  vertical averaging  *!', 1),
        ('END SELECT', 2),
        13,
    ],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:244-274': [
        ('!* baroclinic drag forcing *!', 1),
        ('CLOSE(l2_slow_unit)', 1),
        31,
    ],
    'ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:278-306': [
        ('!* atmospheric pressure forcing *!', 1),
        ('Ve_rhs(ji,jj) = Ve_rhs(ji,jj) + ( bhd_wave(ji,jj+1) - bhd_wave(ji,jj) ) * r1_e1u(ji,jj)', 1),
        29,
    ],
    # --- ORCA2 round 169: kt=8 substep-one and slow forcing -------
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:680-702': [
        ('! Add bottom stresses:', 1),
        ("CALL r95_spg_w2('trd_v', zv_trd)", 1),
        23,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:715-728': [
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', 1),
        ('END DO   ;   END DO', 17),
        14,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:761-767': [
        ('IF( .NOT.lk_linssh ) THEN !* Update ocean depth', 1),
        ('END DO   ;   END DO', 20),
        7,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:207-218': [
        ('SELECT CASE( n_dynadv )', 2),
        ('END SELECT', 2),
        12,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:231-247': [
        ('!* baroclinic drag forcing *!', 1),
        ("CALL r93_slow_put_pair( kt, 'wind', Ue_rhs, Ve_rhs )", 1),
        17,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/stp2d.f90:254-285': [
        ('!* atmospheric pressure forcing *!', 1),
        ("CALL r93_slow_put_pair( kt, 'final', Ue_rhs, Ve_rhs )", 1),
        32,
    ],
    # --- ORCA2 round 168: exit-depth operand split -----------------
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:585-595': [
        ('zhdiv = (   ( zhU(ji,jj) - zhU(ji-1,jj) ) + ( zhV(ji,jj) - zhV(ji,jj-1) )  ) * r1_e1e2t(ji,jj)', 1),
        ("CALL r95_spg_w2('ssha_e', ssha_e)", 1),
        11,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:633-634': [
        ('zsshu_a(ji,jj) = r1_2 * r1_e1e2u(ji,jj)', 1),
        ('&                                      + e1e2t(ji+1,jj  ) * ssha_e(ji+1,jj  ) ) * ssumask(ji,jj)', 1),
        2,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:836-844': [
        ('!* Swap', 1),
        ('vn_e   (:,:) = va_e  (:,:)', 1),
        9,
    ],
    # --- ORCA2 round 167: admitted kt=8 exit-depth walk ------------
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:763-766': [
        ('hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)', 1),
        ('hvr_e(ji,jj) = ssvmask(ji,jj) / (  hv_e(ji,jj) + 1._wp - ssvmask(ji,jj)  )', 1),
        4,
    ],
    'ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:770-779': [
        ('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', 1),
        ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1),
        10,
    ],
    # --- ORCA2 round 165: kt=8 vertical-coordinate boundary ------------
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:132-179': [
        'ALLOCATE( ssha(jpi,jpj) , ua_b(jpi,jpj) , va_b(jpi,jpj) )',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 48],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:185-226': [
        'CASE ( 2 )',
        ('END SELECT', 3), 42],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/sshwzv.f90:736-767': [
        'z2d(:,:) = 0._wp',
        ('pww(ji,jj,jk) = ( 1._wp - zcff ) * pww(ji,jj,jk)', 2), 32],
    # --- ORCA2 round 139: passive FCT bound-input walk ------------
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:171': [
        'CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt,',
        'CALL fct_up1_2stp( Kbb, Kmm, Kaa, p2dt,'],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316': [
        'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb),',
        'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb),'],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:816-845': [
        ('! 2nd option', 1), ('END DO   ;   END DO', 52), 30],
    'advection.py:840-874': [
        'NEMO_FCT_STENCIL_TRACE_FIELDS = (',
        'return_limiter_activity: bool = False,', 35],
    # Round 318 (G9 import, advection.py = gridladder tip): lane A's entries.
    'advection.py:1392-1410': [
        'tr_west = jnp.roll(bnd_up, 1, axis=1)', 'base, q_td, wet, q_max,', 19],
    'advection.py:1576-1577': [
        ('if return_nemo_stencil_trace:', 2),
        'return div_h_fct, vert_div_fct, stencil_trace', 2],
    # --- ORCA2 round 138: passive FCT beta-operand walk ------------
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:800-845': [
        '!        !==  set slices and compute zbup/zbdo  ==!',
        ('END DO   ;   END DO', 52), 46],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-856': [
        'zup = MAX(  zbup(ji  ,jj  ,ik  ),',
        '&        zbup(ji  ,jj  ,ikm1), zbup(ji  ,jj  ,ikp1)  )', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:858-878': [
        'zdo = MIN(  zbdo(ji  ,jj  ,ik  ),', ('ENDIF', 33), 21],
    'advection.py:995-1010': [
        ('raise ValueError(', 2),
        '"NEMO trace returns require the NEMO RK3 two-step predictor")', 16],
    'advection.py:1487-1502': [
        'alpha_u_full, alpha_v, alpha_vert_face = _zalesak_signsplit_face_alphas(',
        'return_nemo_beta_trace=True,', 16],
    'advection.py:1529-1536': [
        'limited_u, limited_v = alpha_u_full * ad_flux_u, alpha_v * ad_flux_v',
        'vert_div_fct = F_vert_fct[..., :-1] - F_vert_fct[..., 1:]', 8],
    'advection.py:2061-2084': [
        ('if return_nemo_beta_trace:', 4),
        'zbetup_literal, zbetdo_literal, R_in, R_out,', 24],
    'advection.py:2139-2140': [
        ('if return_nemo_beta_trace:', 5),
        'return alpha_u_full, alpha_v, alpha_vert_face, beta_trace', 2],
    'ocean_model_latlon_cgrid.py:15312-15329': [
        'if (self._nemo_ws_test_hooks.expose_stage3_fct_inputs',
        'return state_after._replace(mass_flux_w=diagnostic)', 18],
    # --- ORCA2 round 136: passive literal tracer-ZDF source walk --------
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:180-195': [
        "IF( cdtype == 'TRA' .AND. jn == jp_tem ) THEN",
        ('END DO   ;   END DO', 4), 16],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:218-235': [
        '! Diagonal, lower (i), upper (s)', ('ENDIF', 11), 18],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:268-273': [
        '!* 1st recurrence:   Tk = Dk - Ik Sk-1 / Tk-1',
        ('END DO   ;   END DO', 11), 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:283-291': [
        '!* 2nd recurrence:    Zk = Yk - Ik / Tk-1  Zk-1',
        ('END DO   ;   END DO', 13), 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:293-299': [
        '!* 3d recurrence:    Xk = (Zk - Sk Xk+1 ) / Tk',
        ('END DO   ;   END DO', 14), 7],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-749': [
        'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )',
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 16],
    'ocean_model_latlon_cgrid.py:1660,1885': [
        'stage1_momentum_rhs_transform: object = None',
        ('tracer_zdf_trace: bool = False', 1), 2],
    'ocean_model_latlon_cgrid.py:9472-9476': [
        'if (self._nemo_ws_test_hooks.stage1_momentum_rhs_transform',
        '_u1_rhs, _v1_rhs))', 5],
    'ocean_model_latlon_cgrid.py:11984-11991': [
        'if _return_tracer_zdf_trace:',
        ('solve=_nemo_ws_vertical_solve_trace,', 2), 8],
    'ocean_model_latlon_cgrid.py:15299-15311': [
        ('if self._nemo_ws_test_hooks.tracer_zdf_trace:', 1),
        ('return result._replace(state_after=state_after)', 4), 13],
    # --- ORCA2 round 129: split-explicit boundary association walk -------
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:761-779': [
        'IF( .NOT.lk_linssh ) THEN !* Update ocean depth (variable volume case only)',
        "&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )",
        19],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/lbcnfd.f90:684-721': [
        ("CASE ( 'V' )                               ! V-point", 5),
        ('END DO   ;   END DO', 6), 38],
    # --- ORCA2 round 147: exact seven-field T-pivot association ----------
    'barotropic_latlon_cgrid.py:694-742': [
        ('def _nemo_external_mode_boundary_association(', 1),
        ('inverse_u_post, inverse_v_post, eta,', 1), 49],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591': [
        ('!                             ! resulting flux at mid-step (not over the full domain)', 1),
        ('ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 1),
        28],
    # ORCA2 round 154: the separately materialized V-transport statement and
    # its later continuity consumer.  These narrow anchors distinguish the
    # source boundary proved by the private round-154 arm.
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570': [
        ('DO jj = ntsj-( 2), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)   ! not jpj-row', 2),
        ('END DO   ;   END DO', 8), 3],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:584-591': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 1),
        ('ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)', 1),
        8],
    # ORCA2 round 155: promote the four separately proved operands as one
    # source program on the production NEMO-literal path.
    'barotropic_latlon_cgrid.py:570-599': [
        ('def _nemo_literal_reference_face_depths(z_coord, dtype):', 1),
        ('return hu_compact, hv_compact', 1), 30],
    'barotropic_latlon_cgrid.py:854-886': [
        ('def nemo_literal_continuity_divergence(', 1),
        ('return (du + dv) * (1.0 / grid.area)', 1), 33],
    # --- ORCA2 round 148: V-face north-neighbour association arm --------
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545': [
        'zsshp2_e(:,:) = za1 * sshn_e(:,:)  + za2 * sshb_e(:,:) + za3 * sshbb_e(:,:)',
        '&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)',
        27],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:584-615': [
        ("CASE ( 'T' , 'W' )                         ! T-, W-point", 5),
        ('ptab(jf)%pt4d(ii1,ij1,jk,jl) = psgn(jf) * ptab(jf)%pt4d(ii2,ij2,jk,jl)', 45),
        32],
    'barotropic_latlon_cgrid.py:602-686': [
        'def _nemo_ssh_avg_apply(eta_dyn, u_mask, v_mask, grid, area, prep, *,',
        ('return H_u, H_v, r1_u_entry, r1_v_entry', 2), 85],
    # --- ORCA2 round 149: carried V reciprocal retraction ---------------
    'barotropic_latlon_cgrid.py:2596-2600': [
        '# Preserve associated halo operands in the private trace.',
        '"inverse_depth_v": r1_H_v,', 5],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:680-699': [
        '! Add bottom stresses:',
        ("& l4_canon_2d(zu_trd,'U'), l4_canon_2d(zv_trd,'V')", 1),
        20],
    # --- ORCA2 round 128: complete literal-EEN source program ------------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:639-683': [
        ("CASE ( 'U' )", 5),
        ('END DO   ;   END DO', 5), 45],
    'barotropic_latlon_cgrid.py:1083-1104': [
        ('def _nemo_een_north_u(field, grid):', 1),
        ('return north', 2), 22],
    'barotropic_latlon_cgrid.py:1209-1224': [
        ('e3u = b(e3u0 * b(one + r3u[..., None] * umask))', 1),
        ('jnp.max(jnp.where(vmask > 0.0, levels, 0), axis=-1), 1)', 1), 16],
    'barotropic_latlon_cgrid.py:1327-1363': [
        ('un = {', 1),
        ('e3v, mbkv, neighbor, neighbor_mask, vq[corner],', 1), 37],
    # --- ORCA2 round 127: northern V recurrence prefix -------------------
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348': [
        ('CALL r118_een_v_before(1, ji, jj, jk, zpvo_nw', 1),
        ('CALL r118_een_v_after(2, ji, jj, jk, ffv_ne(ji,jj))', 1), 10],
    'ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1360-1367': [
        ('r105_acc_v_nw(ji,jj) = ffv_nw(ji,jj)', 1),
        ('ffv_ne(ji,jj) = r1_12 * r1_e2v(ji,jj)', 1), 8],
    # --- ORCA2 round 126: complete northern frozen-mask association ------
    'barotropic_latlon_cgrid.py:1174-1180': [
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
    'barotropic_latlon_cgrid.py:1054-1075': [
        ('def _nemo_een_north_f(field, grid):', 1),
        ('return north', 1), 22],
    'barotropic_latlon_cgrid.py:1212-1215': [
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
    'barotropic_latlon_cgrid.py:1049-1051': [
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
    'barotropic_latlon_cgrid.py:1044-1046': [
        ('def _nemo_een_south_e3f0(e3f0, mesh_e3f0):', 1),
        ('return jnp.concatenate([mesh_e3f0[:1], e3f0[:-1]], axis=0)', 1), 3],
    'barotropic_latlon_cgrid.py:1234-1248': [
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
    'barotropic_latlon_cgrid.py:1039-1041': [
        ('def _nemo_south_copy_fill(field):', 1),
        ('return jnp.concatenate([field[:1], field[:-1]], axis=0)', 1), 3],
    'barotropic_latlon_cgrid.py:1230-1233': [
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
    # --- ORCA2 round 191: exact-forcing midpoint V-depth boundary ---
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:542-545': [
        ('DO jj = ntsj-( 2), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 2),
        '&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)',
        4],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-591': [
        '!     Compute Sea Level at step jit+1',
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)',
        12],
    # --- ORCA2 round 198: pressure-gradient and explicit-drag split -----
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-660': [
        '! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2',
        "CALL r95_spg_w2('zv_spg', zv_spg)",
        19],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:642-650': [
        '! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2',
        ('END DO   ;   END DO', 13),
        9],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:652-660': [
        '! Surface pressure gradient',
        "CALL r95_spg_w2('zv_spg', zv_spg)",
        9],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:663-670': [
        '! Add Coriolis trend:',
        ('ENDIF', 23),
        8],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:680-702': [
        '! Add bottom stresses:',
        "CALL r95_spg_w2('trd_v', zv_trd)",
        23],
    # --- ORCA2 round 193: corrected-entry growth and external replay ------
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:202-233': [
        '!  RK3 : single first external mode computation',
        'CALL r84_dump_frame( kstp, 3, Naa )', 32],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-608': [
        '!                             ! Sum over sub-time-steps to compute advective velocities',
        'vn_adv(ji,jj) = vn_adv(ji,jj) + za2 * zhV(ji,jj) * r1_e1v(ji,jj)', 9],
    # --- ORCA2 round 195: substep-3 V-transport operand split -----------
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:502-511': [
        '!* Extrapolate barotropic velocities at mid-step (jn+1/2)',
        ('END DO   ;   END DO', 3),
        10],
    # --- ORCA2 round 196: substep-3 AB3 midpoint-V input split ---------
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:486-511': [
        '!* Set extrapolation coefficients for predictor step:',
        ('END DO   ;   END DO', 3),
        26],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form',
        '&   ) * ssvmask(ji,jj)',
        13],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:842-844': [
        'vbb_e  (:,:) = vb_e  (:,:)',
        'vn_e   (:,:) = va_e  (:,:)',
        3],
    'ORCA2_R96SPG/ocean.output:765': (
        'Vector form: 2nd order centered scheme           ln_dynadv_vec  =  T',
        1),
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:564-570': [
        'resulting flux at mid-step (not over the full domain)',
        ('END DO   ;   END DO', 8),
        7],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/domhgr.f90:152':
        'r1_e1v(:,:) = 1._wp / e1v(:,:)   ;   r1_e2v (:,:) = 1._wp / e2v(:,:)',
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:382-383': [
        'un_adv(:,:)     = 0._wp       ! Sum for now transport issued from ts loop',
        'vn_adv(:,:)     = 0._wp', 2],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:890-906': [
        'un_adv(:,:) = un_adv(:,:) / r1_wgt2s',
        "WRITE(l2_adv_unit) l4_canon_2d(un_adv,'U'), l4_canon_2d(vn_adv,'V')", 17],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth',
        "&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 10],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:791-796': [
        "CALL r95_spg_w2('ua_new', ua_e)",
        ("CALL r95_spg_w2('hvr_e', hvr_e)", 2), 6],
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
    'ocean_pe_latlon_cgrid.py:6236-6255': [
        ('du_dt = du_dt * u_mask_3d', 1),
        'dv_dt = jnp.where(_restore_hpg, hpg_tendency_v, dv_dt * v_mask_3d)',
        20],
    'ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 2),
        'pvv(ji,jj,jk,Krhs) = zhpj(ji,jj) + zvap', 42],
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
    'ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:221-233': [
        '! Stage 2 :',
        'CALL r84_dump_frame( kstp, 3, Naa )', 13],
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
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:259-282': [
        'IF( ln_rnf_depth .OR. ln_rnf_depth_ini ) THEN',
        'phdivn(ji,jj,1) = phdivn(ji,jj,1) - rnf(ji,jj) * r1_rho0', 24],
    'cfgs/SHARED/namelist_ref:474,478,480': [
        'ln_rnf_depth = .false.', 'ln_rnf_depth_ini = .false.',
        'rn_dep_max  = 150.', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-470': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN',
        'vv(ji,jj,jk,Kaa) = ( vv(ji,jj,jk,Kbb) + rDt * vv(ji,jj,jk,Krhs) ) * vmask(ji,jj,jk)', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:286-310': [
        '! Round-13 WRITE-only stage-1 transport operand record.',
        "l4_canon_2d((r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm))),'V'), l4_canon_2d(vv_b(:,:,Kmm),'V')", 25],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:513-537': [
        'DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)             ! barotropic velocity correction',
        'vv(ji,jj,jk,Kaa) = vv(ji,jj,jk,Kaa) + zvb(ji,jj)*vmask(ji,jj,jk)', 25],
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
    # --- ORCA2 round 187: terminal list entry is reopened for write ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:120-146': [
        'IF( nitrst <= nitend .AND. nitrst > 0 ) THEN',
        'lrst_oce = .TRUE.', 27],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202': [
        ('IF( kt == nitrst ) THEN', 1),
        ('ENDIF', 14), 15],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:271-271':
        'IF( lrst_oce   )   CALL rst_write    ( kstp, Nbb, Nnn, Naa )',
    # --- ORCA2 round 200: OMT-0's own first safety boundary ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:176-184': [
        'zmax(1) = MAXVAL( ABS( ssh(:,:,Kmm)           ), mask = llmsk(:,:,1) )',
        'zmax(5) = MAXVAL(  ts(:,:,:,jp_sal,Kmm), mask = llmsk )', 9],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:243-250': [
        'IF(  zmax(1) >   20._wp .OR.',
        '& ABS(   SUM(zmax(1:jptst)) ) > HUGE(1._wp) ) THEN', 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stpctl.f90:293-316': [
        "WRITE(ctmp1,*) ' stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  S >= 100  or  NaN encounter in the tests'",
        "CALL ctl_stop( 'STOP', ctmp1, ' ', ctmp2, ctmp3, ctmp4, ctmp5, ctmp6, ' ', ctmp7 )", 24],
    # --- ORCA2 round 211: OMT-1's compiled vector arm and safety boundary ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-298': [
        "WRITE(ctmp1,*) ' stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  S >= 100  or  NaN encounter in the tests'",
        "CALL wrt_line( ctmp6, kt, 'Sal   max'", 6],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:243-250': [
        'IF(  zmax(1) >   20._wp .OR.',
        '& ABS(   SUM(zmax(1:jptst)) ) > HUGE(1._wp) ) THEN', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:293-310': [
        "WRITE(ctmp1,*) ' stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0  or  S >= 100  or  NaN encounter in the tests'",
        'IF(lwp) THEN   ;   CALL ctl_stop(', 18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:666-679': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form',
        ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:287-289': [
        'ssh_frc(:,:) = sshe_rhs(:,:)',
        'zv_frc(:,:) =   Ve_rhs(:,:)', 3],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:674-678': [
        'va_e(ji,jj) = (                                 vn_e(ji,jj)',
        '&   ) * ssvmask(ji,jj)', 5],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-753': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)',
        ("& l4_canon_2d(zhU,'U'), l4_canon_2d(zhV,'V')", 1), 33],
    # --- ORCA2 round 244: in-loop seven-field fold association ----------
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:721-739': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)',
        ('ENDIF', 32), 19],
    # --- ORCA2 round 218: OMT-2 linear implicit bottom drag ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:258-284': [
        'SUBROUTINE zdf_drg_lin( pCd0, pCdU )',
        'END SUBROUTINE zdf_drg_lin', 27],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:540-547': [
        'CASE( np_lin )             !==  linear friction  ==!',
        ('CALL zdf_drg_lin( pCd0(:,:), pCdU(:,:) )', 2), 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:158-169': [
        'IF( ln_drgimp .AND. ln_dynspg_ts ) THEN',
        ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1),
        12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:303-312': [
        'IF( ln_drgimp ) THEN      ! implicit bottom friction',
        'zwd(ji,iku) = zwd(ji,iku) - zDt_2 *( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) / (e3u_3d(ji,jj,iku) *(1._wp+r3u(ji,jj,Kaa)*umask(ji,jj,iku)))',
        10],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:470-474': [
        'IF ( ln_drgimp ) THEN      ! implicit bottom friction',
        ('&                                   / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1),
        5],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:1404-1463': [
        'SUBROUTINE dyn_drg_init( Kbb, Kmm, puu, pvv, puu_b ,pvv_b, pu_RHSi, pv_RHSi, pCdU_u, pCdU_v )',
        'pv_RHSi(ji,jj) = pv_RHSi(ji,jj) + (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kmm))) * r1_2*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) ) * zv_i(ji,jj)',
        60],
    # --- ORCA2 round 220: OMT-3 lateral momentum diffusion ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-184': [
        'NAMELIST/namdyn_ldf/ ln_dynldf_OFF',
        'READ(numnam_cfg(MAX(INDEX(numnam_cfg,"&namdyn_ldf "),1):),namdyn_ldf', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-228': [
        'nldf_dyn = np_ERROR',
        'IF(.NOT.ln_dynldf_OFF ) THEN', 8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:250':
        ('IF( ln_dynldf_lev )   nldf_dyn = np_lap', 1),
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:297-305': [
        ('IF( ln_dynldf_OFF ) THEN', 2),
        "IF( ierr /= 0 )   CALL ctl_stop( 'STOP', 'ldf_dyn_init: failed to allocate arrays')", 9],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:488-493': [
        ('CASE ( 3 )        !==  Stage 3  ==!', 1),
        'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )', 6],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf.f90:81-85': [
        'SELECT CASE ( nldf_dyn )',
        'CALL dynldf_lev_lap( kt, Kbb, Kmm, puu, pvv, Krhs )', 5],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:123-140': [
        ('zwf(ji-1,jj-1) = ahmf(ji-1,jj-1,jk)', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1),
        18],
    # --- ORCA2 round 222: OMT-4 FCT tracer advection ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:592-602': [
        'NAMELIST/namtra_adv/ ln_traadv_OFF',
        'READ(numnam_cfg(MAX(INDEX(numnam_cfg,"&namtra_adv "),1):),namtra_adv', 11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:630-647': [
        '!                                !==  Parameter control & set nadv ==!',
        "CALL ctl_stop( 'tra_adv_init: FCT scheme, choose 2nd or 4th order, and optimized or accurate treatment of implicit' )",
        18],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:668-674': [
        ('SELECT CASE ( nadv )', 2),
        "' Implicit treatment: ', nn_fct_imp", 7],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:497-540': [
        '! FCT at last stage only with RK3',
        'CALL tra_adv_cen ( kt, nit000,', 44],
    # --- ORCA2 round 224: source-ordered northern-fold tracer walk ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_cen.f90:149-160': [
        ('CASE(  2  )                         !* 2nd order centered', 1),
        ('&                                        / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1),
        12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:191-199': [
        '! -- Anti-diffusive fluxes : high order minus low order',
        ('END DO   ;   END DO   ;   END DO', 4), 9],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:495-539': [
        '! *** 1st step',
        ('END DO   ;   END DO   ;   END DO', 23), 45],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:562-573': [
        '! *** 2nd step',
        ('END DO   ;   END DO   ;   END DO', 26), 12],
    # --- ORCA2 round 225: live nonosc fold walk and next boundary ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:306-316': [
        '!        !==  monotonicity algorithm  ==!',
        'CALL nonosc( Kaa, pt(:,:,:,jn,Kbb), ztFu, ztFv, ztFw, zta_up1, p2dt )',
        11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:768-821': [
        'REAL(wp) ::   zpos, zneg, zbig, zsmall, zup, zdo, zbt, zland, z1_Dt',
        'zbdo(ji,jj,ikp1) = MERGE( MIN( pbef(ji,jj,2), paft(ji,jj,2) ),  zbig, tmask(ji,jj,2) == 1._wp )',
        54],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:798-861': [
        ('DO jk = 1, jpkm1', 4),
        '&        zbdo(ji  ,jj  ,ikm1), zbdo(ji  ,jj  ,ikp1)  )',
        64],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:862-878': [
        '!                                            ! positive part of the flux',
        ('ENDIF', 33),
        17],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:888-915': [
        ('DO jj = ntsj-( 1), ntej+(  0 )', 3),
        ('END DO   ;   END DO', 54),
        28],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:322-329': [
        ('DO jk =  1,  jpkm1', 5),
        'zta_up1(ji,jj,jk) = zta_up1(ji,jj,jk) + p2Dt * ztra',
        8],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:581-638': [
        ("IF( c_NFtype == 'T' ) THEN", 3),
        ('END DO   ;   END DO', 4),
        58],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:633-643': [
        ('IF( ln_tile )   CALL dom_tile_start', 1),
        ("& l4_canon_3d(ts(:,:,:,jp_sal,Krhs),'T')", 4), 11],
    # --- ORCA2 round 226: final FCT RHS association and refutation ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:598-609': [
        ('! -- after tracer with upstream scheme', 2),
        ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)', 3), 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:318-329': [
        '!        !==  final trend with corrected fluxes  ==!',
        'zta_up1(ji,jj,jk) = zta_up1(ji,jj,jk) + p2Dt * ztra /', 12],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:600-649': [
        '!                       !==  T-S Tracers  ==!',
        ('IF( lwp .AND. .NOT.ln_tile .AND. kstp == nit000 .AND. kstg <= 2 ) &', 2),
        50],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:700-749': [
        'CASE ( 3 )        !==  Stage 3  ==!   add all left over RHS terms',
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 50],
    'nemo_testcase_l4_orca2_round226_fct_rhs_walk.py:2-367': [
        '"""Compare OMT-4',
        'raise SystemExit(main())', 366],
    'test_nemo_testcase_l4_orca2_round226_fct_rhs_walk.py:1-50': [
        'from __future__ import annotations',
        'gate.classify(_report(), plant=plant)', 50],
    # --- ORCA2 round 223: completed-month terminal restart overwrite ---
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:94-146': [
        'IF( kt == nit000 ) THEN   ! default definitions',
        'lrst_oce = .TRUE.', 53],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/restart.f90:153-201': [
        'SUBROUTINE rst_write( kt, Kbb, Kmm, Kaa )',
        'nitrst   = nn_stocklist( nrst_lst )', 49],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/daymod.f90:405-417': [
        "ELSEIF( TRIM(cdrw) == 'WRITE' ) THEN",
        "CALL iom_rstput( kt, nitrst, numrow, 'ntime'  , REAL( nn_time0, wp) ) ! time", 13],
    # --- ORCA2 round 214: live RK indices and malformed round-213 schema ---
    'ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stprk3.f90:204-215': [
        'CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )',
        'CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 12],
    'ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/stp2d.f90:303-303':
        'CALL dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ssh, uu_b, vv_b, Kaa )',
    'ORCA2_OMIP_L4_R213VECPRE/BLD/ppsrc/nemo/dynspg_ts.f90:714-726': [
        "r213_magic = 'NEMO_L4_R213VV1'",
        'WRITE(r213_unit) va_e', 13],
    # --- ORCA2 round 215: vector update and T-pivot association -----------
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:669-682': [
        'IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form',
        ('END DO   ;   END DO', 17), 14],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:738-744': [
        'IF( .NOT.lk_linssh ) THEN !* Update ocean depth (variable volume case only)',
        ('END DO   ;   END DO', 20), 7],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756': [
        'IF( .NOT.lk_linssh ) THEN   !* Update ocean depth (variable volume case only)',
        "&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )",
        10],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:533-560': [
        '! resulting flux at mid-step (not over the full domain)',
        'ssha_e(ji,jj) = (  sshn_e(ji,jj) - rDt_e * ( ssh_frc(ji,jj) + zhdiv )  ) * ssmask(ji,jj)',
        28],
    # --- ORCA2 round 236: OMT-4 fold-local slow-V operand split ---
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-199': [
        '!*  vertical averaging  *!',
        'Ve_rhs(ji,jj) = SUM( e3v_3d(ji,jj,1:jpkm1)*vv(ji,jj,1:jpkm1,Krhs)*vmask(ji,jj,1:jpkm1) ) * r1_hv_0(ji,jj)',
        6],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:219-230': [
        'CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs, Ve_rhs, CdU_u, CdU_v )',
        'Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0 * vtauV(ji,jj) * (r1_hv_0(ji,jj) /(1._wp+r3v(ji,jj,Kbb)))',
        12],
    # --- ORCA2 round 235: OMT-4 global back-interpolation owner ---
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294': [
        '!                          ! set values computed in RK3_ssh',
        'zCdU_v  (:,:) = CdU_v   (:,:)', 6],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:323-328': [
        'CALL dyn_cor_2D( puu_b(:,:,Kmm), pvv_b(:,:,Kmm), zu_trd, zv_trd )',
        ('END DO   ;   END DO', 1), 6],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:604-612': [
        '! Half-step back interpolation of SSH for surface pressure computation at step jit+1/2',
        ('END DO   ;   END DO', 13), 9],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:1522-1557': [
        'SUBROUTINE ts_bck_interp( jn, ll_init,',
        'END SUBROUTINE ts_bck_interp', 36],
    'barotropic_latlon_cgrid.py:113-125': [
        '# --- NEMO dynspg_ts nn_bt_flt=3 (Demange 2019 dissipative FB) coefficients ---',
        'def nemo_ab3am4_coeff_arrays(n_loop: int, alpha: float,',
        13],
    'barotropic_latlon_cgrid.py:3469-3470': [
        ('_ab3_za, _ab3_zb = nemo_ab3am4_coeff_arrays(', 2),
        'n_loop, alpha=_nemo_filter_alpha, ramp=_ab3_hist is None)', 2],
    'orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg:383-387': [
        'ln_dynspg_ts  = .true.',
        'rn_bt_alpha   = 0.09', 5],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stprk3_stg.f90:383-402': [
        'CALL    eos    (        ts, Kmm, rhd, rhop )     ! Kmm in situ density anomaly for hpg computation',
        'CALL    dyn_hpg( kstp,      Kmm, uu, vv, Krhs )  ! Hydrostratic Pressure Gradient (HPG)',
        20],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:684-718': [
        ("CASE ( 'V' )                               ! V-point", 5),
        ('END DO', 105), 35],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:945-972': [
        ("IF( c_NFtype == 'T' ) THEN            ! *  North fold  T-point pivot", 4),
        ('END DO', 149), 28],
    # --- ORCA2 round 228: post-stage four-field association -------------
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:784-796': [
        "CALL lbc_lnk( 'stp_RK3_stg', ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp   &",
        "&                       , ts(:,:,:,jp_tem,Kaa), 'T',  1._wp, ts(:,:,:,jp_sal,Kaa), 'T',  1._wp, ldfull=.TRUE. )",
        13],
    # --- ORCA2 round 229: rank-complete tracer-consumer record gap -------
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:342-352': [
        '! Phase-3 read-only transport instrument.  This is exactly the zF',
        ('CLOSE(l1_unit)', 2),
        11],
    'ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:551-555': [
        ('IF( .NOT.ln_shuman ) THEN', 1),
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 1),
        5],
    'ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:973-980': [
        ("CASE ( 'V' )                               ! V-point", 7),
        ('ptab(1,ipj+jh) = psgn * ptab(3,ipj-3-jh)', 2),
        8],
    # --- ORCA2 round 230: repair the failed fold-operand writer ----------
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:558-558':
        'e3t_3d(:,:,:) *(1._wp+r3t(:,:,Kmm)*tmask(:,:,:))',
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/inc/domzgr_substitute.h90:126-126':
        ('# define  e3t(i,j,k,t)      (E3t_0(i,j,k) Tmsk(r3t,tmask,i,j,k,t))', 1),
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90:170-170':
        'REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:,:) ::     e3t_3d',
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/dom_oce.f90:180-180':
        'REAL(wp), PUBLIC, ALLOCATABLE, DIMENSION(:,:,:)   ::   r3t, r3u, r3v',
    'ORCA2_OMIP_L4_R229FOLDTRP/BLD/ppsrc/nemo/traadv_fct.f90:538-538':
        ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk)', 2),
    # --- ORCA2 round 231: admitted fold record and V-transport boundary ---
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285': [
        ('DO jk =  1,  jpkm1', 1),
        ('END DO   ;   END DO   ;   END DO', 1),
        4],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/traadv_cen.f90:150-160': [
        ('DO jk = 1, jpkm1', 1),
        ('/ (e3t_3d(ji,jj,jk)', 1),
        11],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279': [
        ('SELECT CASE( n_baro_upd )', 4),
        ('END DO   ;   END DO', 2),
        11],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:552-559': [
        ('IF( .NOT.ln_shuman ) THEN', 1),
        '& r3t(:,:,Kmm), tmask(:,:,:) )',
        8],
    # --- ORCA2 round 232: stage-1 V-transport operand split -----------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domzgr.f90:184-188': [
        ('! t-level: 3D reference', 1),
        ("CALL iom_get( inum, jpdom_global, 'e3f_0'", 1),
        5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/inc/domzgr_substitute.h90:114-121': [
        ('#     define  E3t_0', 3),
        ('#     define  e3v_0', 3),
        8],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:261-270': [
        ('!==  ratio at u-,v-point  ==!', 2),
        ('END DO   ;   END DO', 5),
        10],
    # --- ORCA2 round 248: kt=5 land-support and exit-depth writer ------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domain.f90:193-214': [
        ('ht_0(:,:) = 0._wp', 1),
        ('r1_hv_0(:,:) = ssvmask(:,:)', 1),
        22],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:208-217': [
        ('pr3t(ji,jj) = pssh(ji,jj) * r1_ht_0', 1),
        ('* r1_hv_0(ji,jj) * r1_e1e2v(ji,jj)', 1),
        10],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:714-717': [
        ('hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)', 1),
        ('hvr_e(ji,jj) = ssvmask(ji,jj)', 1),
        4],
    # --- ORCA2 round 233: atomic stage-transport geometry unit -------
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:206-213': [
        ('Ocean/land mask at u-, v-, and f-points', 1),
        ('* tmask(ji+1,jj+1,jk)', 1),
        8],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dommsk.f90:228-232': [
        ('IF ( MAXVAL(umask(ji,jj,:))/=0._wp )', 1),
        ("CALL lbc_lnk( 'dommsk', umask, 'U', 1.0_wp", 1),
        5],
    'ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:973-980': [
        ("CASE ( 'V' )", 7),
        ('ptab(1,ipj+jh) = psgn * ptab(3,ipj-3-jh)', 2),
        8],
    'ocean_model_latlon_cgrid.py:2026-2036': [
        ('nemo_qco_card_mesh_operands(', 1),
        ('include_reciprocals=include_reciprocals', 1),
        11],
    'vertical.py:659-666': [
        ('geom_grid = ensure_geometry(grid)', 2),
        ('hv_0 = jnp.sum(e3v_0 * vmask3, axis=-1)', 1),
        8],
    'vertical.py:685-767': [
        ('def nemo_qco_resolved_mesh_operands(', 1),
        ('e2u=e2u, e1v=e1v)', 1),
        83],
    # --- ORCA2 round 189: bounded replacement restart list ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/in_out_manager.f90:51-51':
        'INTEGER, DIMENSION(10) :: nn_stocklist',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:323-325': [
        'READ(numnam_ref(MAX(INDEX(numnam_ref,"&namrun "),1):),namrun,IOSTAT=ios)',
        'IF(lwm) WRITE ( numond, namrun )', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-348': [
        'IF( ln_rst_list .OR. nn_stock /= -1 ) THEN',
        ("CALL ctl_stop( ctmp1, 'Impossible to properly do model restart' )", 2), 8],
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
    # --- ORCA2 round 321: L3's bolus statements in the ORCA2 build ---
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:954-955': [
        'pFu(ji,jj,jk) = pFu(ji,jj,jk) - ( zpsi_uw(ji,jj,1) - zpsi_uw(ji,jj,2) )',
        'pFv(ji,jj,jk) = pFv(ji,jj,jk) - ( zpsi_vw(ji,jj,1) - zpsi_vw(ji,jj,2) )', 2],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:354-357': [
        'zUfac = r1_2 *rn_Ud', 'aht0    = zUfac *    rn_Ld**inn', 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296':
        'CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )',
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
    'nemo_testcase_recipe.py:3424':
        'if (cfg.momentum_advection != "vector_invariant"',
    'ocean_pe_latlon_cgrid.py:5598': ('if _mom_adv == "flux_form":', 2),
    'ocean_pe_latlon_cgrid.py:5616': (
        '_bc_horizontal_momentum_advection_flux_form(', 2),
    'nemo_testcase_recipe.py:3427':
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
    # ORCA2 round 310 (G8 import under Decision 122).
    'cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:371-373': [
        'ln_dynldf_lap =  .true.', 'nn_ahm_ijk_t  = -30', 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynldf_lev.f90:123': {
        'anchors': ('* (e3f_3d(ji-1,jj-1,jk) *(1._wp+r3f(ji-1,jj-1)*fe3mask(ji-1,jj-1,jk))) * r1_e1e2f(ji-1,jj-1)', 1),
        'routine': 'dynldf_lev_lap',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:156':
        'r1_e1e2f(:,:) = 1._wp / e1e2f(:,:)',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:144': {
        'anchors': 'CALL dyn_ldf( kt, Kbb, Kbb, uu, vv, Krhs )',
        'routine': 'stp_2D',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:493': {
        'anchors': 'CALL dyn_ldf( kstp, Kbb, Kmm, uu, vv, Krhs )',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynkeg.f90:129-130': {
        'anchors': [('puu(ji,jj,jk,Krhs) = puu(ji,jj,jk,Krhs) - ( zhke(ji+1,jj  ) - zhke(ji,jj) ) * r1_e1u(ji,jj)', 1),
                    ('pvv(ji,jj,jk,Krhs) = pvv(ji,jj,jk,Krhs) - ( zhke(ji  ,jj+1) - zhke(ji,jj) ) * r1_e2v(ji,jj)', 1),
                    2],
        'routine': 'dyn_keg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109': [
        ('MIN( rn2(ji,jj,jk), rn2b(ji,jj,jk) ) <= -1.e-12', 1),
        ('p_avt(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 1),
        2,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:121-135': [
        ('IF( nn_evdm == 1 ) THEN', 1),
        ('p_avm(ji,jj,jk) = rn_evd * wmask(ji,jj,jk)', 1),
        15,
    ],
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
    'ocean_pe_latlon_cgrid.py:1788':
        ('r3_now = jax.lax.optimization_barrier(eta_now * r1_h0)', 1),
    'ocean_pe_latlon_cgrid.py:1795':
        ('live_t = e3t0 * (1.0 + r3_now[..., None] * tmask) * tmask', 1),
    'ocean_pe_latlon_cgrid.py:1819':
        ('flux_u, west, flux_v, south, r1_area_t, live_t[..., jk],', 1),
    # Round 162's review disputed whether the multiply-back is on the
    # vertical-velocity path.  It is: the statement writes the routine's own
    # OUTPUT argument, and the vertical-velocity routine receives exactly that
    # array.  Those two lines are cited, and so is legoESM's answering one.
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:101':
        ('pe3divUh       ! e3t*div[Uh]', 2),
    'GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/sshwzv.f90:278':
        ('CALL div_hor( kt, Kbb, Kmm, pu, pv, ze3div )', 1),
    'ocean_pe_latlon_cgrid.py:1617':
        ('return jax.lax.optimization_barrier(live_e3t * hdiv) * tmask', 1),
    'ocean_pe_latlon_cgrid.py:1890-1891': [
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
    'vertical.py:239-240': [
        ('weighted_eta = b(area_t * eta)', 1),
        ('num_u = b(half * b(weighted_eta + jnp.roll(weighted_eta, -1, '
         'axis=1)))', 1), 2],
    'vertical.py:255': ('r3u = b(b(num_u * r1_hu0) * r1_area_u)', 1),
    'vertical.py:263':
        ('e3u = b(e3u_0 * b(one + r3u[..., None] * umask3))', 1),
    # --- round 160: legoESM's own two-solve statements ---
    'ocean_model_latlon_cgrid.py:2200-2237': [
        ('def nemo_stage_momentum_wzv_executes(config, hooks=None) -> bool:', 1),
        ('return bool(config_split)', 1), 38],
    'ocean_model_latlon_cgrid.py:2385-2396': [
        ('w_momentum, _, _ = nemo_qco_wzv_operands(', 1),
        ('else 0))', 2), 12],
    'ocean_model_latlon_cgrid.py:9201-9207': [
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
    'vertical.py:468-566': [
        'def nemo_qco_live_vorticity_e3f_cgrid(',
        'return jnp.concatenate([with_south[:, -1:], with_south], axis=1)',
        99],
    'ocean_model_latlon_cgrid.py:6615-6617': [
        '_ws_uses_nemo_ldf_e3 = (',
        'and _cfg_b.lateral_viscosity_e3_weighting == "nemo_e3")', 3],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stp2d.f90:158-164': [
        'CALL eos    ( ts, Kbb, rhd )',
        ('IF( lwp .AND. kt == 1081 )   WRITE(r146_family_unit)', 2), 7],
    'GYRE_OMIP_L2_P3_SM_R148LDF/BLD/ppsrc/nemo/stprk3_stg.f90:704-739': [
        ('!              !---------------!', 2),
        'IF( kstg == 3 )   CALL dyn_zdf(', 36],
    'ocean_model_latlon_cgrid.py:6706-6758': [
        'if not self._nemo_ws_test_hooks.legacy_hadv_min_face_thickness:',
        'ldf_thickness_operands=_ws_ldf_thickness_kbb,', 53],
    'ocean_model_latlon_cgrid.py:7623-7688': [
        '_stage_ldf_thickness = None', '_stage_ldf_thickness),', 66],
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
    'ocean_model_latlon_cgrid.py:11575-11698': [
        'T=state_new.T.replace(data=_adv_content_T)',
        ('z_coord=z_coord, config=config, iwm_fields=iwm_fields)', 2), 124],
    'ocean_model_latlon_cgrid.py:14105-14122': [
        'K_v_cell = K_v_cell.astype(state.T.data.dtype)',
        'K_v_cell = K_v_cell + K33_iso.astype(state.T.data.dtype)', 18],
    'ocean_model_latlon_cgrid.py:14740-14772': [
        ('if do_tracers:', 2),
        'rounded_thickness=_zdf_tracer_rounded))', 33],
    # --- parallel LDF step-3 re-proof: current private arm and execution ---
    'ocean_model_latlon_cgrid.py:1854-1857': [
        '# Route the already-computed GM/Redi rate into the same stage-3 source',
        'route_gm_redi_stage3_source: bool = False', 4],
    'ocean_model_latlon_cgrid.py:10890-10915': [
        ('elif _tti == "rk3_ws":', 2),
        '_stage_source_rates[2][1] + dS_gm * active_3d,', 26],
    # Decision 90 added nine stated lines INSIDE this block (the slow-forcing
    # depth average, which this call now states rather than defaulting), so
    # the pinned extent grows with it: 88 -> 98, same two symbols.
    'nemo_testcase_recipe.py:608-705': [
        'return LatLonCGridOceanConfig.from_flat(',
        'gm_redi=None,', 98],
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
    'ocean_model_latlon_cgrid.py:2873-2986': [
        'h_one_third = h_k_old',
        '+ dt * h_one_half * stage_source_rates[2][1])', 114],
    # Extent 174 -> 199 at ORCA2 round 304 (G5 round 4 import): the stage-3
    # salt block (trasbc.f90:308) was inserted between the same two pinned
    # endpoint statements.
    'ocean_model_latlon_cgrid.py:8898-9145': [
        '_stage3_T_rate = (',
        '/ jnp.maximum(_h_s3_kmm, 1.0e-10),', 248],
    # Extent 433 -> 434 at the 2026-09-25 merge: main added one line inside
    # this span (``omega=_cfg_b.omega,`` on the isoneutral K33 call), so the
    # two pinned endpoint statements are one line further apart than before.
    'ocean_model_latlon_cgrid.py:10209-10718': [
        'T_mid = state_new.T.data',
        'S_mid = S_mid + dt * dS_gm * active_3d', 510],
    'ocean_model_latlon_cgrid.py:11058-11107': [
        ('_nemo_ws_rk3_tracer_pair_step(', 3),
        'return_final_content=True,', 50],
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
    'nemo_testcase_recipe.py:447-449': [
        'tke_shear_production="nemo_face_native_now2"',
        ('tke_shear_metric_source="nemo_qco_live_face"', 2), 3],
    'ocean_model_latlon_cgrid.py:12765-12782': [
        'if shear_disc not in ("nemo_face_native", "nemo_face_native_now2"):',
        '_u_before, _v_before = _u_now, _v_now', 18],
    'ocean_model_latlon_cgrid.py:12837-12936': [
        'if metric_source == "nemo_qco_live_face":',
        'ref_v * (1.0 + r3vb[..., None]),', 100],
    'packages/ocean/legoesm/ocean/physics/vertical_mixing/_shared.py:387-460': [
        ('wumask = u_mask[..., :-1] * u_mask[..., 1:]', 2),
        ('+ (zsh2v[:-1, :, :] + zsh2v[1:, :, :]) * coast_v', 2), 74],
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
    'ocean_model_latlon_cgrid.py:15007-15012': [
        'if _tke_coeff_new is not None:',
        'tke_avt=Field(data=_tke_coeff_new.K_H', 6],
    'ocean_model_latlon_cgrid.py:15236-15251': [
        '# ``step`` is the production-compiled entry point even when a caller',
        ('_nemo_stage1_zad_eta_after_override))', 1), 16],
    'state.py:621-625': [
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
    'ocean_model_latlon_cgrid.py:9763-9828': [
        'if _cfg_b.fix_eta_drift:',
        'eta=state_new.eta.replace(data=eta_fixed),', 66],
    # --- round 51: live WS-RK3 operand selection and history carry ---
    'ocean_model_latlon_cgrid.py:7530-7531': [
        ('u0 = state.u.data', 2), ('v0 = state.v.data', 2), 2],
    'ocean_model_latlon_cgrid.py:9369-9374': [
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
    'ocean_pe_latlon_cgrid.py:2107-2141': [
        'elif config.ke_gradient_scheme == "c2":', 'dp_dy = _dKp_dy[..., 1]', 35],
    'ocean_pe_latlon_cgrid.py:3360-3391': [
        'area_w = jax.lax.optimization_barrier(',
        'diag_vertadv_v = jax.lax.optimization_barrier(diag_vertadv_v)', 32],
    'ocean_pe_latlon_cgrid.py:5427-5466': [
        'zad_w, zad_h_u, zad_h_v = w, h_u, h_v',
        ('zad_h_v = jax.lax.optimization_barrier(zad_h_v)', 1), 40],
    'ocean_model_latlon_cgrid.py:8578-8588': [
        '_freeze_hpg = self._nemo_ws_test_hooks.freeze_stage_hpg_operands',
        ('getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1), 11],
    'ocean_model_latlon_cgrid.py:10114-10131': [
        ('if (getattr(_cfg_b, "adaptive_implicit_vertadv", False)', 1),
        '/ jnp.maximum(_area_v, 1.0e-30))', 18],
    'ocean_model_latlon_cgrid.py:10142-10164': [
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
    'barotropic_latlon_cgrid.py:3197-3198': [
        'U_bar_corr, V_bar_corr = _depth_average_to_faces(',
        'u_corr, v_corr, _h_k_corr, min_water_col, mask, u_mask, v_mask, grid,',
        2],
    'eos.py:759':
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
    'vertical.py:1963': ('def compute_ocean_jacobian(', 1),
    'ocean_model_latlon_cgrid.py:7607-7611': [
        'transport_velocity = (', ('* _ws_stage_v_mask,', 1), 5],
    'ocean_model_latlon_cgrid.py:9591': ('_g2 = _nemo_ws_stage_transport(', 1),
    'ocean_model_latlon_cgrid.py:9613': (
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
    'fidelity/nemo_recipe.py:998': ('def build_nemo_gyre_recipe(', 1),
    'ocean_model_latlon_cgrid.py:6854': (
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
    'ocean_model_latlon_cgrid.py:10233': (
        '_T_gm_in = T_mid if _ldf_state is None else _ldf_state[0]', 1),
    # ROUND 39 moved these two: the before-state slope block added lines
    # above them, so the STATEMENT is unchanged and its line number is not.
    # Re-anchored rather than left to rot, which is what the gate exists for.
    'ocean_model_latlon_cgrid.py:10672': (
        'k33_implicit = compute_isoneutral_K33_latlon(', 1),
    # tracer_combine is READ by two step functions and SELECTED by a DINO
    # recipe -- the retraction of round 37's "a lever nothing selects".
    'dino.py:1769': ('"tracer_combine": "thickness_weighted",', 1),
    'dino.py:3988': ('tracer_combine=cfg.tracer_combine,', 1),
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
    'ocean_model_latlon_cgrid.py:8777': '_vector_velocity_stage_update = (',
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
    'vertical.py:791': 'def nemo_qco_live_face_geometry_cgrid(',
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
    'ocean_model_latlon_cgrid.py:9668-9671': [
        ('u3_corr = u3_raw * _ws_stage_u_mask', 1),
        ('_replace_stage_mean, target_u, target_v)', 1),
        4],
    'ocean_model_latlon_cgrid.py:11724-11750': [
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
    'ocean_model_latlon_cgrid.py:11555-11557': [
        '_nemo_ws_pre_implicit_state = (',
        'if self._nemo_ws_test_hooks.expose_pre_implicit_state else None)',
        3],
    'ocean_model_latlon_cgrid.py:14435': ('u_solve_in = u_solve_in - _u_bt_mean', 1),
    'ocean_model_latlon_cgrid.py:14627': ('u_solve_in = u_solve_in - (', 1),
    # ORCA2 round 61: private post-dyn_zdf/pre-barotropic raw-Kaa observer.
    'ocean_model_latlon_cgrid.py:1675': (
        'expose_stage3_raw_momentum: bool = False', 1),
    'ocean_model_latlon_cgrid.py:3506-3598': [
        '_stage3_rhs_hook = self._nemo_ws_test_hooks.expose_stage3_momentum_rhs',
        'expose_stage3_raw_momentum cannot be combined with another', 93],
    'ocean_model_latlon_cgrid.py:11717-11724': [
        'if self._nemo_ws_test_hooks.expose_stage3_raw_momentum:',
        'if _ws_stage3_correction is not None:', 8],
    # Merge re-anchor (round 207): the GYRE side adds a third occurrence of
    # this fragment, so the ORCA2 block's own copy is now the THIRD, not the
    # second.  Pinning the stale occurrence resolved the range BACKWARDS.
    'ocean_model_latlon_cgrid.py:11896-11900': [
        'if _nemo_ws_exposed_stage3_raw is not None:',
        ('v=state_new.v.replace(data=_raw_v),', 3), 5],
    # ORCA2 round 63: private write-only dyn_zdf seam observer.
    'ocean_model_latlon_cgrid.py:1676-1681': [
        '# WRITE-only observer for the four source-ordered stage-3 dyn_zdf',
        'zdf_momentum_observer: object = None', 6],
    'ocean_model_latlon_cgrid.py:3600-3604': [
        '_zdf_momentum_observer = (',
        'raise ValueError("zdf_momentum_observer must be callable or None")', 5],
    'ocean_model_latlon_cgrid.py:13553-13565': [
        ('_zc = self.z_coord if z_coord is None else z_coord', 11),
        '# Argument validation at ENTRY, not inside the drag branch below:', 13],
    'ocean_model_latlon_cgrid.py:14188-14192': [
        'u_new, v_new = state.u.data, state.v.data',
        'if do_momentum and getattr(_cfg_b, "surface_stress_implicit",', 5],
    'ocean_model_latlon_cgrid.py:14435-14438': [
        'u_solve_in = u_solve_in - _u_bt_mean',
        'u_solve_in, v_solve_in)', 4],
    # Round 305: the G7 dyn_zdf bottom statement now sits inside this span.
    'ocean_model_latlon_cgrid.py:14627-14669': [
        'u_solve_in = u_solve_in - (',
        '_zdf_baro_drag_u, _zdf_baro_drag_v = u_solve_in, v_solve_in', 43],
    'ocean_model_latlon_cgrid.py:14870-14877': [
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
    'ocean_pe_latlon_cgrid.py:3657-3660': [
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
    'nemo_testcase_recipe.py:686': [
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
    'ocean_pe_latlon_cgrid.py:5910': 'rho_prime=rho_prime, h_k=h_k,',
    'ocean_pe_latlon_cgrid.py:5887-5888': [
        '_u_ldf_local = u if ldf_state is None else ldf_state[2]',
        '_v_ldf_local = v if ldf_state is None else ldf_state[3]',
        2],
    'ocean_pe_latlon_cgrid.py:2232-2233': [('z_coord,', 26), ('eta_safe,', 7), 2],
    'ocean_pe_latlon_cgrid.py:2273-2274': ['requires the raw NEMO',
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
    'nemo_testcase_recipe.py:380,636,3451': [('pgf_scheme="nemo_sco",', 1), ('pgf_scheme="nemo_sco",', 2), 'if cfg.pgf_scheme != "nemo_sco":', 3],
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
    'ocean_model_latlon_cgrid.py:6611-6716': [
        '_ws_face_thickness_kbb = None',
        ('_ws_v_live_mask, _grid, z_coord=_zc)[:2])', 1), 106],
    'ocean_model_latlon_cgrid.py:7098-7161': [
        ('if _sfx is not None:', 1),
        ('F_slow_v + _wind_increment_v) * _slow_surface_v_mask', 1), 64],
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
    'ocean_model_latlon_cgrid.py:9150-9198': [
        ('if _return_tracer_process_trace:', 1),
        'process_qsr_rate=_nemo_ws_process_qsr_rate,', 49],
    # --- ORCA2 round 101: sufficient owner of the merged GYRE year move ---
    'ocean_model_latlon_cgrid.py:8709-8719': [
        'nemo_r3t_rk3_stage1_stretch,',
        '_qt_13 = nemo_r3t_rk3_stage1_stretch(', 11],
    'eos.py:1004-1035': [
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
    # Round 214 / ORCA2 round 132: compiled face-thickness construction and
    # its paired U/V halo exchange, not the differently numbered MY_SRC file.
    'VORTEX_SMT_R3_OMIP_L1_P3/BLD/ppsrc/nemo/usrdef_zgr.f90:240-247': [
        'pe3u(ji,jj,jk) = MIN( pe3t(ji,jj,jk), pe3t(ji+1,jj,jk) )',
        '&          kfillmode = jpfillcopy )',
        8],
    'vertical.py:685-750': [
        'def nemo_qco_resolved_mesh_operands(',
        'e3v0 = jnp.asarray(e3v0, dtype=dtype)[..., :nlev]',
        66],
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
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:282-283': [
        ('zFu(ji,jj,jk) = e2u(ji,jj)', 1),
        ('zFv(ji,jj,jk) = e1v(ji,jj)', 1),
        2,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:637-645': [
        ('CALL tra_adv    ( kstp, Kbb, Kmm, Kaa, ts, Krhs, zFu, zFv, zFw, kstg )', 1),
        ('CALL tra_sbc_RK3( kstp, Kbb, Kmm,      ts, Krhs,                kstg )', 1),
        9,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:529,531-532': [
        ('CASE ( np_FCT )', 1),
        ('CALL tra_adv_fct ( kt, nit000,', 1),
        ('nn_fct_imp )', 1),
        3,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609': [
        ('! *** 1st step', 1),
        ('pt_up1(ji,jj,jk) = ( (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kbb)*tmask(ji,jj,jk))) * pt_b(ji,jj,jk) + zDt * ztra ) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kaa)*tmask(ji,jj,jk))) * tmask(ji,jj,jk) !!clem', 1),
        115,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-197,260-266': [
        ('!* horizontal anti-diffusive fluxes', 1),
        ('ztFu(ji,jj,jk) =', 1),
        ('!* vertical anti-diffusive fluxes', 1),
        ('- ztFw(ji,jj,jk)  ) * wmask(ji,jj,jk)', 1),
        12,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316,743-938': [
        ('CALL nonosc(', 1),
        ('SUBROUTINE nonosc(', 1),
        ('END SUBROUTINE nonosc', 2),
        197,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:320-329': [
        ('! -- Final trend with corrected fluxes', 1),
        ('zta_up1(ji,jj,jk) =', 1),
        10,
    ],
    # --- ORCA2 card round 135: incoming slow forcing and passive FCT walk
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:194-200': [
        ('!*  vertical averaging  *!', 1),
        ('END DO   ;   END DO', 2),
        7,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:569-572': [
        ('ptFu(ji,jj,jk) = 0.5_wp', 1),
        ('MIN( pV(ji,jj,jk) , 0._wp ) * pt_up1(ji,jj+1,jk)', 1),
        4,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:577-578': [
        ('ptFw(ji,jj,jk) = 0.5_wp', 1),
        ('MIN( pW(ji,jj,jk) , 0._wp ) * pt_up1(ji,jj,jk-1)', 1),
        2,
    ],
    # --- ORCA2 card round 137: level-3 FCT limiter-coefficient walk
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:910-913': [
        ('zcoef = MERGE( MIN( 1._wp, zbetdo(ji,jj,ik), zbetup(ji,jj+1,ik) )', 1),
        ('pbb(ji,jj,jk) = pbb(ji,jj,jk) * zcoef', 2),
        4,
    ],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:853-876': [
        ('zup = MAX(', 2),
        ('zbetdo(ji,jj,ik) = ( paft(ji,jj,jk) - zdo ) / zneg * zbt', 1),
        24,
    ],
    # --- ORCA2 card round 141: independent finite-growth acquisition
    'ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/stprk3.f90:202-221': [
        ('CALL r141_growth_entry(', 1),
        ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1),
        20,
    ],
    'ORCA2_OMIP_L4_R141GROWTH/BLD/ppsrc/nemo/traadv.f90:299-320': [
        ('CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 1),
        ('& pFw(ntsi:ntei,ntsj:ntej,:) )', 1),
        22,
    ],
    # --- ORCA2 card round 144: independent early-growth bracket
    'ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/stprk3.f90:202-221': [
        ('CALL r143_growth_entry(', 1),
        ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1),
        20,
    ],
    'ORCA2_OMIP_L4_R143EARLY/BLD/ppsrc/nemo/traadv.f90:299-320': [
        ('CALL wzv( kt, Kbb, Kmm, Kaa, pFu, pFv, ww, np_transport )', 1),
        ('& pFw(ntsi:ntei,ntsj:ntej,:) )', 1),
        22,
    ],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/stprk3.f90:204-215': [
        ('CALL stp_2D( kstp, Nbb, Nbb, Naa, Nrhs )', 1),
        ('CALL stp_RK3_stg( 1, kstp, Nbb, Nbb, Nrhs, Naa )', 1),
        12,
    ],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:359-375': [
        ('sshn_e(:,:) =    pssh (:,:,Kmm)', 1),
        ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kbb)))', 1),
        17,
    ],
    # ORCA2 round 178: direct association of the completed slow RHS.
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:289-291': [
        ('ssh_frc(:,:) = sshe_rhs(:,:)', 1),
        ('zv_frc(:,:) =   Ve_rhs(:,:)', 1),
        3,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:357-375': [
        ('IF( ln_bt_fw ) THEN', 2),
        ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kbb)))', 1),
        19,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:357-366': [
        ('IF( ln_bt_fw ) THEN', 2),
        ('hvr_e (:,:) = (r1_hv_0(:,:) /(1._wp+r3v(:,:,Kmm)))', 1),
        10,
    ],
    # ORCA2 round 163: this deck's nn_bt_flt=3 disables the weighted
    # external-mode accumulator before the vector-invariant arm below.
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:220-223': [
        ('ll_bt_av    = .TRUE.', 1),
        ('ll_init     = ll_bt_av', 1),
        4,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779': [
        ('IF( .NOT.lk_linssh ) THEN !* Update ocean depth', 1),
        ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp", 2),
        19,
    ],
    # ORCA2 round 166: source-ordered external-substep localization.  Keep
    # each prose span narrow so the gate proves the stated program boundary.
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:507-545': [
        ('ua_e(ji,jj) = za1 * un_e(ji,jj) + za2 * ub_e(ji,jj) + za3 * ubb_e(ji,jj)', 1),
        ('&                                 + e1e2t(ji,jj+1) * zsshp2_e(ji,jj+1)  ) * ssvmask(ji,jj)', 1),
        39,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:566-569': [
        ('zhU(ji,jj) = e2u(ji,jj) * ua_e(ji,jj) * zhup2_e(ji,jj)', 1),
        ('zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)', 1),
        4,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:585-595': [
        ('zhdiv = (   ( zhU(ji,jj) - zhU(ji-1,jj) )', 1),
        ("CALL r95_spg_w2('ssha_e', ssha_e)", 1),
        11,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:715-757': [
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN      !* Vector form', 1),
        ('va_e(ji,jj) =  va_e(ji,jj) / ( 1._wp - rDt_e * zCdU_v(ji,jj) * hvr_e(ji,jj) )', 1),
        43,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:763-766': [
        ('hu_e (ji,jj) =    hu_0(ji,jj) + zsshu_a(ji,jj)', 1),
        ('hvr_e(ji,jj) = ssvmask(ji,jj) / (  hv_e(ji,jj) + 1._wp - ssvmask(ji,jj)  )', 1),
        4,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:770-779': [
        ('IF( .NOT.lk_linssh ) THEN   !* Update ocean depth', 1),
        ("&                   , hur_e, 'U',  1._wp, hvr_e, 'V',  1._wp  , ssha_e, 'T', 1._wp, ldfull=.TRUE. )", 1),
        10,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:854-856': [
        ('IF( ln_dynadv_vec .OR. lk_linssh ) THEN', 2),
        ('pvv_b  (:,:,Kaa) = pvv_b  (:,:,Kaa) + za1 * va_e', 1),
        3,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:924-937': [
        ('IF( (.NOT.(ln_dynadv_vec .OR. lk_linssh)) .AND. ll_bt_av ) THEN', 1),
        ("CALL lbc_lnk( 'dynspg_ts', puu_b, 'U', -1._wp, pvv_b, 'V', -1._wp )", 1),
        14,
    ],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:639-683': [
        ("CASE ( 'U' )", 5),
        ('END DO   ;   END DO', 5), 45],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:684-721': [
        ("CASE ( 'V' )", 5),
        ('END DO   ;   END DO', 6), 38],
    # --- ORCA2 round 157: MPI cyclic exchange / T-pivot split -----------
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:1961-1979': [
        ('3. Do east and west MPI_Isend if needed', 2),
        ('CALL MPI_ISEND( buffsnd_dp(ishtS(jn)+1)', 1), 19],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2060-2068': [
        ('IF( ibufszR(jn) > 0 ) THEN', 3),
        ('ptab(jf)%pt4d(ishti+ji,ishtj+jj,jk,jl) = buffrcv_dp(idxr)', 1), 9],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbclnk.f90:2105-2113': [
        ('6. north fold treatment', 2),
        ('CALL mpp_nfd(', 3), 9],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1561-1577': [
        ('IF( ln_nnogather ) THEN', 2),
        ('l_full_nf_update = .TRUE.', 2), 17],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1629-1669': [
        ('ALLOCATE(ipjfld(kfld))', 2),
        ('CALL MPI_Isend(', 2), 41],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1712-1738': [
        ('Apply the North pole folding', 2),
        ('ptab(jf)%pt4d(ji,ij1,jk,jl) = psgn(jf) * zbufr(ii2,ij2,inei)', 3), 27],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:1747-1766': [
        ('IF( lnfd_same(ii1,iig) )', 2),
        ('ptab(jf)%pt4d(ji,ij1,jk,jl) = psgn(jf)', 4), 20],
    'ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/mppini.f90:1412-1440': [
        ('ALLOCATE( nfd_rksnd(jpi,nn_hls+1,4)', 1),
        ('WHERE( lnfd_same )   nfd_jisnd(:,1,:) = HUGE(0)', 1), 29],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:890-960': [
        ('un_adv(:,:) = un_adv(:,:) / r1_wgt2s', 1),
        ("CALL r95_spg_w2('ssh_aa', pssh(:,:,Kaa))", 1),
        71,
    ],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90:170-235': [
        ('SUBROUTINE tra_adv_trp_t(', 1),
        ('DEALLOCATE( zFu_cor, zFv_cor )', 1),
        66,
    ],
    'ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90:267-315': [
        ('IF( ll_Fw ) THEN', 1),
        ('pFw(ji,jj,jk) = e1e2t(ji,jj) * ww(ji,jj,jk)', 1),
        49,
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
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:205-228': [
        ('Background eddy viscosity and diffusivity', 1),
        ('avm_k(Nis0-(1):Nie0+(1),Njs0-(1):Nje0+(1),jk)', 1),
        24,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfphy.f90:347-359': [
        ('start from turbulent closure values', 1),
        ('IF( ln_zdfevd )   CALL zdf_evd', 1),
        13,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90:107-110': [
        ('DO jk =  1,  jpkm1', 1),
        ('END DO   ;   END DO   ;   END DO', 2),
        4,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/zdfevd.f90:120-135': [
        ('enhance momentum Kz', 1),
        ('p_avm(ji,jj,jk) = rn_evd * wmask', 1),
        16,
    ],
    'ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:172-181': [
        ('CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )', 1),
        ('CALL zdf_phy( kstp, Nbb, Nbb, Nrhs )', 1),
        10,
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
    # --- round 202: shared stage-one tracer-thickness ratio order ---
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:53-54': [
        'r1_3 = 1._wp / 3._wp', 'r2_3 = 2._wp / 3._wp', 2],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:160-179': [
        '!==  ssh/h0 ratio at Kaa  ==!',
        'r3v(:,:,Kaa) = r2_3 * r3v(:,:,Kbb) + r1_3 * r3va(:,:)', 20],
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:167':
        'CALL dom_qco_r3c_RK3( ssha, r3ta, r3ua, r3va, r3fa )',
    'GYRE_OMIP_L2_P3/BLD/ppsrc/nemo/stprk3_stg.f90:177':
        'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)',
    'domhgr.F90:222-223': [("IF(  iom_varid( inum, 'ff_f', ldstop = .FALSE. ) > 0  .AND.  &", 1), ("& iom_varid( inum, 'ff_t', ldstop = .FALSE. ) > 0    ) THEN", 1), 2],
    # --- round 224: SMT-3 deck provenance and compiled branch ---
    'orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:319-326': [
        'ln_traldf_lap   = .true.', 'rn_Ld        = 200.e+3', 8],
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:918-933': [
        'ln_traldf_OFF   = .false.', 'ln_botmix_triad = .false.', 16],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:214-218': [
        'NAMELIST/namtra_ldf/ ln_traldf_OFF',
        '&                 nn_aht_ijk_t , rn_Ud          , rn_Ld', 5],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:231-233': [
        'READ(numnam_ref(MAX(INDEX(numnam_ref,"&namtra_ldf ")',
        'IF(lwm) WRITE( numond, namtra_ldf )', 3],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:261-283': [
        'nldf_tra = np_ERROR',
        ('IF ( ln_traldf_triad )   nldf_tra = np_lap_it', 1), 23],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:354-390': [
        'IF(     ln_traldf_lap ) THEN   ;   zUfac = r1_2 *rn_Ud',
        ("CALL ldf_c2d( 'TRA', zUfac      , inn        , ahtu, ahtv )", 1), 37],
    'VORTEX_SMT2_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traldf.f90:105-122': [
        ('SELECT CASE ( nldf_tra )', 2), ('END SELECT', 2), 18],
    # --- round 237: SMT-4 deck provenance and compiled momentum-LDF branch ---
    'orca2_rounds/round83/acquisition/orca2_rung0_restart_list_10step_a_np2/namelist_cfg:388-392': [
        '&namdyn_ldf', 'nn_ahm_ijk_t  = -30', 5],
    'cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ref:1104-1134': [
        '&namdyn_ldf', 'rn_ahm_b      = 0.0', 31],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:177-185': [
        'NAMELIST/namdyn_ldf/ ln_dynldf_OFF',
        'IF(lwm) WRITE ( numond, namdyn_ldf )', 9],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:221-276': [
        'nldf_dyn = np_ERROR',
        'IF( nldf_dyn == np_lap_i )   l_ldfslp = .TRUE.', 56],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldfdyn.f90:311-346': [
        'IF(     ln_dynldf_lap ) THEN   ;   zUfac = r1_2 *rn_Uv',
        ("CALL ldf_c2d( 'DYN', zUfac      , inn        , ahmt, ahmf )", 1), 36],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf.f90:81-90': [
        'SELECT CASE ( nldf_dyn )',
        'CALL dynldf_lev_blp( kt, Kbb, Kmm, puu, pvv, Krhs )', 10],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'VORTEX_SMT3_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:387-405': [
        ('CASE ( 3 )        !==  Stage 3  ==!', 1),
        'IF( kstg == 3 )   CALL dyn_zdf( kstp, Kbb, Kmm, Krhs, uu, vv, Kaa  )', 19],
    # --- round 226: admitted SMT-3 compiled branch and magnitude boundary ---
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:231-233': [
        'READ(numnam_ref(MAX(INDEX(numnam_ref,"&namtra_ldf ")',
        'IF(lwm) WRITE( numond, namtra_ldf )', 3],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:261-283': [
        'nldf_tra = np_ERROR',
        ('IF ( ln_traldf_triad )   nldf_tra = np_lap_it', 1), 23],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldftra.f90:354-390': [
        'IF(     ln_traldf_lap ) THEN   ;   zUfac = r1_2 *rn_Ud',
        ("CALL ldf_c2d( 'TRA', zUfac      , inn        , ahtu, ahtv )", 1), 37],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110': [
        ('SELECT CASE ( nldf_tra )', 2),
        'CALL traldf_iso_lap  ( kt, Kbb, Kmm, pts, Krhs, l_ptr, l_hst )', 6],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177': [
        'CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )',
        'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )', 19],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:456-488': [
        ('CALL tra_adv_trp( kstp, kstg, nit000, Kbb, Kmm, Kaa, Krhs, zFu, zFv, zFw )', 2),
        "CALL vortex_r18_tracer_rhs( 'adv', Krhs, ts )", 33],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:501-546': [
        ('SELECT CASE( kstg )', 4),
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )', 46],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:532':
        'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )',
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546':
        'CALL tra_zdf( kstp, Kbb, Kmm, Krhs, ts    , Kaa  )',
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:154-304': [
        ('CALL traldf_iso_a33( Kmm, ah_wslp2, akz )', 1),
        ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2),
        151],
    # --- round 227: internal SMT-3 LDF acquisition seams ---
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90:186-338': [
        'zgru(ji,jj,iikm1) = umask(ji,jj,jpkm1)',
        ("CALL lbc_lnk( 'ldfslp', uslp", 1), 153],
    'VORTEX_SMT3_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:780-829': [
        ('DO jk =    2,  jpkm1', 1),
        'pakz(ji,jj,jk) = MAX( zcoef0 - 0.5_wp , 0._wp )', 50],
    # --- round 228: admitted R15 walk and the scalar-writer retraction ---
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:159-177': [
        'CALL eos_rab( ts(:,:,:,:,Nbb), rab_b, Nbb )',
        'CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )', 19],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf.f90:105-110': [
        ('SELECT CASE ( nldf_tra )', 2),
        'CALL traldf_iso_lap  ( kt, Kbb, Kmm, pts, Krhs, l_ptr, l_hst )', 6],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/ldfslp.f90:226-356': [
        ('DO jj = ntsj-( 1), ntej+(  1 ) ; DO ji = ntsi-( 1), ntei+(  1)', 4),
        ('END DO   ;   END DO', 12), 131],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:822-837': [
        ('DO jk =    2,  jpkm1', 1), ('END DO   ;   END DO   ;   END DO', 1), 16],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:240-267': [
        ('DO jj = ntsj-( 0+1), ntej+(  0 ) ; DO ji = ntsi-( 0+1), ntei+(  0 )', 1),
        'r227_fu(ji,jj,jk) = zfu(ji,jj) ; r227_fv(ji,jj,jk) = zfv(ji,jj)', 28],
    'VORTEX_SMT3_VEC_R15_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:282-333': [
        ('IF( 1 <= jk .AND. jk <= jpk-2 ) THEN', 1),
        ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2),
        52],
    # --- round 229: corrected tensor factors and final volume divisor ---
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:243-259': [
        ('zA11 = e2_e1u(ji,jj) * (e3u_3d(ji,jj,jk)', 1),
        ('&                                             + ( zdkt(ji,jj+1,ikp1) + zdkt(ji,jj,ik  ) )  )   )', 1),
        17],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:254-259': [
        ('zfu(ji,jj) =  ahtu(ji,jj,jk)', 1),
        ('&                                             + ( zdkt(ji,jj+1,ikp1) + zdkt(ji,jj,ik  ) )  )   )', 1),
        6],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:260-266': [
        ('IF( r227_iso_active .AND. jn == jp_tem ) THEN', 2),
        ('END DO   ;   END DO', 7), 7],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:306-310': [
        ('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 1),
        ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 1),
        5],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/traldf_iso.f90:327-331': [
        ('pt(ji,jj,jk,jn,Krhs) = pt(ji,jj,jk,jn,Krhs) +', 2),
        ('&                 * r1_e1e2t(ji,jj) / (e3t_3d(ji,jj,jk) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,jk)))', 2),
        5],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:225':
        'CASE ( 3 )           !==  Stage 3  ==!   Kbb = N   ;   Kmm = N+1/2   ;   Kaa = N+1',
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:531-532': [
        'IF( ln_traqsr  )   CALL tra_qsr( kstp,      Kmm, ts, Krhs )',
        'CALL tra_ldf( kstp, Kbb, Kmm, ts, Krhs )', 2],
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:155':
        'e1e2t (:,:) = e1t(:,:) * e2t(:,:)   ;   r1_e1e2t(:,:) = 1._wp / e1e2t(:,:)',
    'VORTEX_SMT3_VEC_R16_OMIP_L1_P3/BLD/ppsrc/nemo/vortex_r23_ldf_terms.f90:149-151': [
        "CALL write3( unit, 'rhs_before', r227_rhs_before )",
        "CALL write3( unit, 'rhs_increment', pt(:,:,:,jp_tem,Krhs) - r227_rhs_before )",
        3],
    # --- round 238: admitted SMT-4 selection, operator and stage order ---
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-185': [
        'NAMELIST/namdyn_ldf/ ln_dynldf_OFF',
        'IF(lwm) WRITE ( numond, namdyn_ldf )', 9],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-278': [
        'nldf_dyn = np_ERROR',
        ('ENDIF', 11), 58],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:311-346': [
        'IF(     ln_dynldf_lap ) THEN   ;   zUfac = r1_2 *rn_Uv',
        ("CALL ldf_c2d( 'DYN', zUfac", 1), 36],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf.f90:81-90': [
        'SELECT CASE ( nldf_dyn )',
        'CALL dynldf_lev_blp( kt, Kbb, Kmm, puu, pvv, Krhs )', 10],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:121-140': [
        ('DO jj = ntsj-( 0), ntej+(  0+1 ) ; DO ji = ntsi-( 0), ntei+(  0+1)', 1),
        ('&              + ( zwt(ji,jj+1) - zwt(ji  ,jj) ) * r1_e2v(ji,jj)', 1), 20],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344': [
        '!*  hydrostatic pressure gradient (HPG))  *!   always called FIRST',
        ("CALL vortex_r16_stage_rhs( 'adv', Krhs, uu, vv )", 2), 17],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:398-404': [
        'CASE ( 3 )        !==  Stage 3  ==!   add left over RHS terms + time stepping',
        "CALL vortex_r16_stage_rhs( 'ldf', Krhs, uu, vv )", 7],
    # --- round 239: locally exact but trajectory-refused source order ---
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:314-335': [
        ('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)              ! Surface value', 1),
        ('END DO   ;   END DO   ;   END DO', 1), 22],
    'VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynadv.f90:134-138': [
        'SELECT CASE( n_dynadv )    !==  compute advection trend and add it to general trend  ==!',
        'CALL dyn_zad     ( kt                , Kmm, puu, pvv, Krhs )                  !* vertical advection', 5],
    # --- ORCA2 round 285: G4 ladder-certified statement import -------
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:225-230': [
        ('WRITE(l2_slow_unit) r1_rho0', 1),
        ('Ve_rhs(ji,jj) =  Ve_rhs(ji,jj) + r1_rho0', 1), 6],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:553-569': [
        ('!       (in addition to snwice_mass)', 1),
        ('&                      ( 2. - vmask(ji,jj,1) )', 1), 17],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:336-339': [
        ('DO ji = ntsi-( 0), ntei+( 0 )                    !==  second recurrence', 1),
        ('/ ( (e3u_3d(ji,jj,1) *(1._wp+r3u(ji,jj,Kaa)', 2), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:678-680': [
        ('ts(ji,jj,jk,jn,Kaa) = (', 1),
        ('/          ( 1._wp + r3t(ji,jj,Kaa) )', 1), 3],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:217-224': [
        ('ssh (:,:,Kaa) = r1_2 * ( ssh (:,:,Kbb) + ssha(:,:) )', 2),
        ('r3f(:,:)     = r2_3 * r3fb(:,:) + r1_3 * r3fa(:,:)', 2), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:307': (
        'pts(ji,jj,1,jp_tem,Krhs) = pts(ji,jj,1,jp_tem,Krhs) +  r1_rcp * qns(ji,jj) * z1_rho0_e3t', 1),
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:279-282': [
        ('ELSE                       !==   runoff put only at the surface', 1),
        ('phdivn(ji,jj,1) = phdivn(ji,jj,1) - rnf(ji,jj)', 1), 4],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/traqsr.f90:387-394': [
        ('!                                      !- fluxes at jk+1 w-level', 1),
        ('pts(ji,jj,jk,jp_tem,Krhs) = pts(ji,jj,jk,jp_tem,Krhs) + r1_rho0_rcp * ( zeT(ji,jj) - zzeT ) / ze3t', 1), 8],
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trazdf.f90:288-289': [
        ('zrhs =       (e3t_3d(ji,jj,jk)', 1),
        ('& + p2dt * (e3t_3d(ji,jj,jk)', 1), 2],
    # --- ORCA2 round 304: G5 tracer-surface statements (Decision 122) ---
    # Both run because key_linssh is absent from the build (lk_linssh=.FALSE.);
    # sfx at :308 is SI3's (nn_ice = 2).
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:220': 'SUBROUTINE tra_sbc_RK3 (',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:282': {
        'anchors': 'IF( .NOT.lk_linssh ) THEN           !* only heat and salt fluxes',
        'routine': 'tra_sbc_RK3',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:284-286': {
        'anchors': [
            ('z1_rho0_e3t = r1_rho0 / (e3t_3d(ji,jj,1) *(1._wp+r3t(ji,jj,Kmm)*tmask(ji,jj,1)))', 1),
            '- emp(ji,jj)*pts(ji,jj,1,jp_sal,Kbb) * z1_rho0_e3t', 3],
        'routine': 'tra_sbc_RK3',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:292': {
        'anchors': ('IF( lk_linssh ) THEN                !* linear free surface', 2),
        'routine': 'tra_sbc_RK3',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:308': {
        'anchors': '+           sfx(ji,jj) * z1_rho0_e3t',
        'routine': 'tra_sbc_RK3',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:49': 'INTEGER  :: n_baro_upd =  np_HYB',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:78': 'SUBROUTINE stp_RK3_stg(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:177': {
        'anchors': 'r3t(:,:,Kaa) = r2_3 * r3t(:,:,Kbb) + r1_3 * r3ta(:,:)   ! at N+1/3 (Kaa)',
        'routine': 'stp_RK3_stg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dom_oce.f90:137': 'lk_linssh = .FALSE.',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:85': 'nn_ice      = 2',
    'ORCA2_OMIP_L4/EXP00/namelist_cfg:251': 'ln_lin = .true.',
    'ORCA2_OMIP_L4/EXP00/namelist_ref:817': 'ln_drgimp   = .true.',
    # --- ORCA2 round 305: G5 top drag + G7 bottom drag (Decision 122) ---
    'ORCA2_OMIP_L4/EXP00/namelist_ref:818': 'ln_drgice_imp = .true. ! implicit ice-ocean drag',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1404': 'SUBROUTINE dyn_drg_init(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1431-1432': {
        'anchors': ['pCdU_u(ji,jj) = r1_2*( ( rCdU_bot(ji+1,jj) + rCdU_bot(ji,jj) ) + ( rCdU_top(ji+1,jj) + rCdU_top(ji,jj) ) )', 'pCdU_v(ji,jj) = r1_2*( ( rCdU_bot(ji,jj+1) + rCdU_bot(ji,jj) ) + ( rCdU_top(ji,jj+1) + rCdU_top(ji,jj) ) )', 2],
        'routine': 'dyn_drg_init',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1447-1448': {
        'anchors': ['zu_i(ji,jj) = puu(ji,jj,ikbu,Kmm) - puu_b(ji,jj,Kmm)', ('zv_i(ji,jj) = pvv(ji,jj,ikbv,Kmm) - pvv_b(ji,jj,Kmm)', 1), 2],
        'routine': 'dyn_drg_init',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1462-1463': {
        'anchors': ['* r1_2*( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * zu_i(ji,jj)', '* r1_2*( rCdU_bot(ji,jj+1)+rCdU_bot(ji,jj) ) * zv_i(ji,jj)', 2],
        'routine': 'dyn_drg_init',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1487-1488': {
        'anchors': ['* r1_2*( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) * zu_i(ji,jj)', '* r1_2*( rCdU_top(ji,jj+1)+rCdU_top(ji,jj) ) * zv_i(ji,jj)', 2],
        'routine': 'dyn_drg_init',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:63': 'SUBROUTINE stp_2D(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:219': {
        'anchors': 'CALL dyn_drg_init( Kbb, Kbb, uu, vv, uu_b, vv_b, Ue_rhs, Ve_rhs, CdU_u, CdU_v )',
        'routine': 'stp_2D',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:67': 'SUBROUTINE dyn_zdf(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:166-169': {
        'anchors': ['puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_bot(ji+1,jj)+rCdU_bot(ji,jj) ) * uu_b(ji,jj,Kaa)', ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 1), 4],
        'routine': 'dyn_zdf',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:175-178': {
        'anchors': ['puu(ji,jj,iku,Kaa) = puu(ji,jj,iku,Kaa) + zDt_2 * ( rCdU_top(ji+1,jj)+rCdU_top(ji,jj) ) * uu_b(ji,jj,Kaa)', ('&                                            / (e3v_3d(ji,jj,ikv) *(1._wp+r3v(ji,jj,Kaa)*vmask(ji,jj,ikv)))', 2), 4],
        'routine': 'dyn_zdf',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:187': {
        'anchors': 'rCdU_top(ji,jj) = rCdU_ice(ji,jj)',
        'routine': 'zdf_drg',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:258': 'SUBROUTINE zdf_drg_lin(',
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:276': {
        'anchors': 'pCdU(ji,jj) = - pCd0(ji,jj) * rn_Uc0   ! linear friction',
        'routine': 'zdf_drg_lin',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:380': {
        'anchors': 'IF ( ln_drgice_imp .AND.   nn_ice /= 2  )   ln_drgice_imp = .FALSE.',
        'routine': 'zdf_drg_init',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:525': {
        'anchors': 'IF(ll_bot)   zmsk_boost(:,:) = zmsk_boost(:,:) * ssmask(:,:)',
        'routine': 'drg_init',
    },
    'ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:546': {
        'anchors': ('pCd0(:,:) = rn_Cd0 * zmsk_boost(:,:)', 1),
        'routine': 'drg_init',
    },
    # --- SI3 ladder round 9 (O3) receipt citations ---
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:95': 'MODULE PROCEDURE ice_var_itd_1c1c, ice_var_itd_Nc1c, ice_var_itd_1cMc, ice_var_itd_NcMc',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:357-362': ['CALL ice_var_itd( zht_i_ini(:,jj)     , zht_s_ini(:,jj)     , zat_i_ini(:,jj)   ,                     & ! <<= in', '&                a_ip   (:,jj,:)   ,   h_ip   (:,jj,:)   ,   h_il   (:,jj,:) )                       ! =>> out', 6],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1203-1204': ['REAL(wp), DIMENSION(:),   INTENT(in)    ::   phti, phts, pati    ! input  ice/snow variables', ('REAL(wp), DIMENSION(:,:), INTENT(inout) ::   ph_i, ph_s, pa_i    ! output ice/snow variables', 1), 2],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1182': 'SUBROUTINE ice_var_itd_1cMc( phti, phts, pati ,                             ph_i, ph_s, pa_i, &',
    'ORCA2_SAS_ICE_O2_REF/BLD/arch_nemo.fcm:27': '%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize',
    'packages/ice/legoesm/ice/si3_orca2_state.py:2529-2638': ['def si3_var_itd_1cmc(', 'return SI3O3ITDFit(', 110],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1221': 'WHERE( phti(:) /= 0._wp )   ;   z1_hti(:) = 1._wp / phti(:)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1232': 'pa_i(ji,jl) = pati(ji) * z1_hti(ji) * (  ( phti(ji) + 2.*hi_max(jl-1) ) * EXP( -2.*hi_max(jl-1)*z1_hti(ji) ) &',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1257': 'pa_i(ji,jpl) = pati(ji) * z1_hti(ji) * ( phti(ji) + 2.*hi_max(jpl-1) ) * EXP( -2.*hi_max(jpl-1)*z1_hti(ji) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1280-1283': ['zdh = MAX( 0._wp, ( rhos * ph_s(ji,jl) + ( rhoi - rho0 ) * ph_i(ji,jl) ) * r1_rho0 )', 'ph_s(ji,jl) = MAX( 0._wp, ph_s(ji,jl) - zdh * rhoi * r1_rhos )', 4],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1294': 'pt_su(:,jl) = ptmsu(:)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1301': 'WHERE( pati(:) /= 0._wp )   ;   zfra(:) = patip(:) / pati(:)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1321': ('WHERE( pa_ip(:,jl) /= 0._wp )   ;   ph_il(:,jl) = zfra(:) * ( ph_i(:,jl) * pa_i(:,jl) ) / pa_ip(:,jl)', 1),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/par_ice.f90:134-135': ['REAL(wp), PUBLIC, PARAMETER ::   epsi06 = 1.e-06_wp  !: small number', 'REAL(wp), PUBLIC, PARAMETER ::   epsi10 = 1.e-10_wp  !: small number', 2],
    'packages/ice/legoesm/ice/si3_orca2_state.py:2556': 'exp = transcendentals.exp',
    'packages/ice/legoesm/ice/si3_orca2_state.py:2645-2759': ['def si3_o3_istate_fill_fields(', 'zswitch=zswitch, at_i=at_i,', 115],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:376': 'WHERE( SUM(a_i(:,:,:),dim=3) > 0._wp )   ;   zswitch(:,:) = tmask(:,:,1)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:383': 's_i(ji,jj,jl) = MIN( MAX( rn_simin , s_i(ji,jj,jl) ) , rn_sinew * sss_m(ji,jj) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:386': 'CALL ice_var_salprof ! for sz_i',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:390-392': ['v_i (ji,jj,jl) = h_i(ji,jj,jl) * a_i(ji,jj,jl)', 'sv_i(ji,jj,jl) = s_i(ji,jj,jl) * v_i(ji,jj,jl)', 3],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:398-399': ['e_s(ji,jj,jk,jl) = zswitch(ji,jj) * v_s(ji,jj,jl) * r1_nlay_s * &', '&               rhos * ( rcpi * ( rt0 - t_s(ji,jj,jk,jl) ) + rLfus )', 2],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:409-412': ['e_i(ji,jj,jk,jl) = zswitch(ji,jj) * v_i(ji,jj,jl) * r1_nlay_i * &', '&                       - rcp   * ( ztmelts - rt0 ) )', 4],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:406': 'szv_i(ji,jj,jk,jl) = sz_i(ji,jj,jk,jl) * v_i(ji,jj,jl) * r1_nlay_i',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:420-421': ['v_ip(:,:,:) = h_ip(:,:,:) * a_ip(:,:,:)', 'v_il(:,:,:) = h_il(:,:,:) * a_ip(:,:,:)', 2],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:426': 't1_ice(ji,jj,:) = t_i (ji,jj,1,:)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:432': 'WHERE( at_i(:,:) > rn_amax_2d(:,:) )   a_i(:,:,jl) = a_i(:,:,jl) * rn_amax_2d(:,:) / at_i(:,:)',
    'packages/ice/legoesm/ice/si3_orca2_state.py:2927-2934': ['# 5. Rung-O3 fit and fill (`iceistate.f90:356-434`).', 'zhlid_ini=zhlid_ini, tmask=tmask_2d, sss_m=sss, rn_amax_2d=rn_amax_2d,', 8],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:153-221': ['def score_o3(evidence_dir: str | Path = O2_RECORD, plant: str | None = None) -> dict[str, Any]:', '"rows": rows,', 69],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90:336-337': ['CALL ice_var_glo2eqv(1)', ('CALL ice_var_agg(1)', 2), 2],
    'tests/ice/unit/test_si3_o3_itd_fit.py:104-150': ['def test_snow_transfer_branch_live_and_mass_conserving():', 'assert np.array_equal(np.asarray(good9.a_i), np.asarray(bad9.a_i))  # ORCA2 deck: cap never binds', 47],
    'tests/ice/unit/test_si3_o3_itd_fit.py:93-101': ['def test_fit_bit_exact_vs_independent_oracle_and_jit_equals_eager():', 'assert np.array_equal(got.view(np.int64), np.asarray(ref).view(np.int64)), (name, c, got, ref)', 9],
    'packages/ice/legoesm/ice/si3_orca2_state.py:3039-3046': ['new_state = si3_o1_step(', (')', 1123), 8],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:225-277': ['def o2_admission_fit_dependence(evidence_dir: str | Path = O2_RECORD) -> dict[str, Any]:', ('return out', 2), 53],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:74-91': ['def _fit_perturbed(fill_fn):', 'return perturbed', 18],
    'tests/ice/unit/test_si3_o3_itd_fit.py:191-202': [('@pytest.mark.skipif(not (_O2_RECORD / "o2_admission.json").is_file(), reason="admitted O2 record absent")', 2), 'assert [k for k, r in dep.items() if r["depends_on_fit"]] == ["o2_istate"]', 12],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1386': 'state_kt1, _ = o2_step_jit(state_entry, sss_entry)',
    'docs/ice/fidelity/si3_ladder_design.md:195-197': ['The O rungs separate **operator admission** from **composition**. O1-O16 replay', 'hide a first owner; only O17 advances the complete live deck. O1-O6 replay', 3],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o3_orca2_sas_gate.py:280-344': ['def score_o3_entry_steps(evidence_dir: str | Path = O2_RECORD, *, bump_fit: bool = False) -> dict[str, Any]:', ('return out', 3), 65],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceupdate.f90:482-483': [('snwice_mass  (:,:) = tmask(:,:,1) * ( rhos * vt_s(:,:) + rhoi * vt_i(:,:) + rhow * (vt_ip(:,:) + vt_il(:,:)) )', 2), ('snwice_mass_b(:,:) = snwice_mass(:,:)', 2), 2],
    'packages/ice/legoesm/ice/si3_orca2_state.py:1219-1245': ['def si3_update_snwice_mass(', 'return new_mass, snwice_mass_prev, fmass', 27],
    'tests/ice/unit/test_si3_o3_itd_fit.py:205-212': [('@pytest.mark.skipif(not (_O2_RECORD / "o2_admission.json").is_file(), reason="admitted O2 record absent")', 3), 'score_o3_entry_steps(_O2_RECORD, bump_fit=True)', 8],
    'scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_eos_gate.py:108': 'policy = PrecisionPolicy.fp64(transcendentals="libm")',
    'packages/ice/legoesm/ice/fidelity/nemo_adv2d_rhg_testcase_recipe.py:152': 'oracle_policy = PrecisionPolicy.fp64(transcendentals="libm")',
    # --- SI3 ladder round 10 (O5) receipt citations ---
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:973': 'at_i(:,:) = SUM( a_i, dim=3 )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1081': 'strength(ji,jj) = rn_pstar * SUM( v_i(ji,jj,:) ) * EXP( -rn_crhg * ( 1._wp - at_i(ji,jj) ) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1102-1105': ['zworka(ji,jj) = ( 4._wp * strength(ji,jj)              &', '&            ) / ( 4._wp + tmask(ji-1,jj,1) + tmask(ji+1,jj,1) + tmask(ji,jj-1,1) + tmask(ji,jj+1,1) )', 4],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1114': ("CALL lbc_lnk( 'icedyn_rdgrft', strength, 'T', 1.0_wp )", 2),
    'packages/ice/legoesm/ice/dynamics.py:1898': 'def si3_h79_strength_raw(',
    'packages/ice/legoesm/ice/dynamics.py:1989': 'def si3_orca2_ice_strength(',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:268': 'CALL ice_strength',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:281': 'zsshdyn(:,:) = ice_var_sshdyn( ssh_m, snwice_mass, snwice_mass_b)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1090': ('ice_var_sshdyn(:,:) = pssh(:,:)', 2),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:238': "CALL lbc_lnk( 'icedyn_rhg_evp', fimask, 'F', 1._wp )",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:440': "CALL lbc_lnk( 'icedyn_rhg_evp', zdelta, 'T', 1.0_wp, zp_delt, 'T', 1.0_wp )",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:647': ("CALL lbc_lnk( 'icedyn_rhg_evp', u_ice, 'U', -1.0_wp, v_ice, 'V', -1.0_wp, ldfull=.true. )", 1),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:754': ("CALL lbc_lnk( 'icedyn_rhg_evp', u_ice, 'U', -1.0_wp, v_ice, 'V', -1.0_wp, ldfull=.true. )", 2),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:845': "CALL lbc_lnk( 'icedyn_rhg_evp', zs1   , 'T', 1._wp, zs2   , 'T', 1._wp, zs12  , 'F', 1._wp )",
    'packages/ice/legoesm/ice/dynamics.py:1255': 'def _si3_cgrid_aevp_core(',
    'packages/ice/legoesm/ice/dynamics.py:1203': 'def si3_cgrid_aevp_solver(',
    'packages/ice/legoesm/ice/dynamics.py:825': 'def _si3_lane3_halo(label, value, nature, sign, ldfull):',
    'packages/ice/legoesm/ice/dynamics.py:2087': 'def si3_orca2_aevp_solver(',
    'packages/ice/legoesm/ice/dynamics.py:2066': 'def si3_var_sshdyn(',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbcnfd.f90:722': ("CASE ( 'F' )                               ! F-point", 5),
    'packages/ice/legoesm/ice/si3_orca2_state.py:1383': 'elif cd_nat == "F":',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbclnk.f90:2112': ('IF( jpni == 1 )  THEN   ;   CALL lbc_nfd(         ptab, cd_nat, psgn                  , kfld         )   ! self NFold', 3),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/lbcnfd.f90:1617': ('llfullappl = ( llfullprsc .OR. ncom_stp <= nit000 ) .AND. .NOT. ( llfullarg .AND. .NOT. llfullprsc )', 2),
    'packages/ice/legoesm/ice/dynamics.py:1930': 'class SI3ORCA2FoldProgram(NamedTuple):',
    'tests/ice/unit/test_si3_o5_strength_aevp.py:172': 'def test_f_fold_matches_lbcnfd_index_formulas():',
    'tests/ice/unit/test_si3_o5_strength_aevp.py:198': 'def test_strength_matches_independent_numpy_transcription():',
    'tests/ice/unit/test_si3_o5_strength_aevp.py:226': 'def test_strength_inexact_value_pins_and_plants():',
    'tests/ice/unit/test_si3_o5_strength_aevp.py:280': 'def test_aevp_wrong_fold_sign_plant_fires(label):',
    'tests/ice/unit/test_si3_o5_strength_aevp.py:291': 'def test_aevp_exchange_program_site_by_site(monkeypatch):',
    'tests/ice/unit/test_si3_o5_strength_aevp.py:317': 'def test_partial_vs_full_pivot_line_fold():',
    # --- SI3 round 10 (O5) review fixes ---
    'tests/ice/unit/test_si3_o5_strength_aevp.py:400': 'def test_aevp_reads_recorded_previous_velocity_not_entry_state():',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:574': ('v_ice(ji,jj) = ( zmV_t(ji,jj) * ( zbetav * v_ice(ji,jj) + v_ice_b(ji,jj) ) + zRHS + zTauO * v_ice(ji,jj) ) &', 1),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:625': ('u_ice(ji,jj) = ( zmU_t(ji,jj) * ( zbetau * u_ice(ji,jj) + u_ice_b(ji,jj) ) + zRHS + zTauO * u_ice(ji,jj) ) &', 1),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:560-562': [('zCorV(ji,jj)  = - 0.25_wp * r1_e2v(ji,jj) *  &', 1), ('&    + zmf(ji,jj+1) * ( e2u(ji,jj+1) * u_ice(ji,jj+1) + e2u(ji-1,jj+1) * u_ice(ji-1,jj+1) ) )', 1), 3],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:611-613': [('zCorU(ji,jj)  =   0.25_wp * r1_e1u(ji,jj) *  &', 1), ('&    + zmf(ji+1,jj) * ( e1v(ji+1,jj) * v_ice(ji+1,jj) + e1v(ji+1,jj-1) * v_ice(ji+1,jj-1) ) )', 1), 3],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:666-668': [('zCorU(ji,jj)  =   0.25_wp * r1_e1u(ji,jj) *  &', 2), ('&    + zmf(ji+1,jj) * ( e1v(ji+1,jj) * v_ice(ji+1,jj) + e1v(ji+1,jj-1) * v_ice(ji+1,jj-1) ) )', 2), 3],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:718-720': [('zCorV(ji,jj)  = - 0.25_wp * r1_e2v(ji,jj) *  &', 2), ('&    + zmf(ji,jj+1) * ( e2u(ji,jj+1) * u_ice(ji,jj+1) + e2u(ji-1,jj+1) * u_ice(ji-1,jj+1) ) )', 2), 3],
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90:463': 'u_ice_b(ji,jj) = u_ice(ji,jj)',
    # --- SI3 ladder round 11 (O7) receipt citations ---
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1481': "IF( ln_pnd_alb  ) THEN ; ln_pnd_alb  = .FALSE. ; CALL ctl_warn( 'ln_pnd_alb=false when no ponds' )  ; ENDIF",
    'packages/core/legoesm/core/transcendentals.py:151': 'def pow(base, exponent) -> jax.Array:  # noqa: A001 - mirrors the C name',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1082': 'wndm_ice(ji,jj) = SQRT( pwndi(ji,jj) * pwndi(ji,jj) + pwndj(ji,jj) * pwndj(ji,jj) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1093': 'Cd_ice(:,:) = rn_Cd_ia',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1137': 'zztmp         = rhoa(ji,jj) * Cd_ice(ji,jj) * wndm_ice(ji,jj)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1138': 'putaui(ji,jj) = zztmp * pwndi(ji,jj)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:167': "CALL lbc_lnk( 'icesbc', utau_ice, 'T', -1.0_wp, vtau_ice, 'T', -1.0_wp )",
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:184': 'z1_c1 = 1._wp / ( LOG(rn_alb_hpiv) - LOG(0.05_wp) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:201': 'zafrac_pnd = MIN( pafrac_pnd(ji,jj,jl), 1._wp - zafrac_snw ) ! make sure (a_ip_eff + a_s_fra) <= 1',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:218': 'IF( 0.05 < ph_ice(ji,jj,jl) .AND. ph_ice(ji,jj,jl) <= rn_alb_hpiv ) THEN      ! 5cm < hi < 100cm',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:231': 'IF( ph_lid(ji,jj,jl) <= 0.02 ) THEN ! do not change pond albedo if the lid is < 2cm',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:244': 'palb_ice(ji,jj,jl) = ( 1._wp - pcloud_fra(ji,jj) ) * zalb_cs + pcloud_fra(ji,jj) * zalb_os',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1597': 'pa_s_fra(ji,jj,jl) = ph_s(ji,jj,jl) / ( ph_s(ji,jj,jl) + 0.02_wp )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1231': 'qsr_ice(ji,jj,jl) = zztmp * ( 1. - palb(ji,jj,jl) ) * qsr(ji,jj)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1255': 'qla_ice (ji,jj,jl) = zztmp1 * (zsq - q_zu_i(ji,jj))',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1290': 'emp_ice(ji,jj) = SUM( a_i_b(ji,jj,:) * evap_ice(ji,jj,:) ) - sprecip(ji,jj) * zsnw(ji,jj)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1320': 'ztri(:,:) = 0.18 * ( 1.0 - cloud_fra(:,:) ) + 0.35 * cloud_fra(:,:)  ! surface transmission when hi>10cm',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1636': ('pout = ( 1._wp - ( pin )**rn_snwblow )', 1),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:692': 'zle  = rAg_i*(ztmp - 1._wp) + rBg_i*LOG10(ztmp) + rCg_i*(1._wp - zta/rtt0) + rDg_i',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:694': 'e_sat_ice_sclr = 100._wp * 10._wp**zle',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:724': 'zde = -(rAg_i*rtt0)/(zta*zta) - rBg_i/(zta*LOG(10._wp)) - rCg_i/rtt0',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:764': 'q_sat_sclr = reps0*ze_s/(ppa - (1._wp - reps0)*ze_s)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:803': 'dq_sat_dt_ice_sclr = reps0*ppa*zde_s_dt / ( ztmp*ztmp )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:350': 'theta_exner_sclr = pta * ( rpref / ppa ) ** rgamma_dry',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:90': '&      rDg_i = LOG10(6.1071_wp)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:396': 'zfric(ji,jj) = zdrag * ( 0.5_wp * (  ( zu_io*zu_io + zu_iom1*zu_iom1 )   &   ! add () for NP repro',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:403': 'zfric(ji,jj) = r1_rho0 * SQRT( utau(ji,jj)*utau(ji,jj) + vtau(ji,jj)*vtau(ji,jj) ) * smask0(ji,jj)',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:436': 'qsb_ice_bot(ji,jj) = zswitch * MIN( qsb_ice_bot(ji,jj), - zqfr_neg * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:450': 'IF( ( zqld - zqfr ) < 0._wp ) THEN',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:472': 'IF( zvel(ji,jj) <= 5.e-03_wp .AND. at_i(ji,jj) >= (rn_amax_2d(ji,jj)-0.001_wp) .AND. vt_i(ji,jj) >= 3._wp )   qlead(ji,jj) = 0._wp',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90:418': 'rDt_ice   = REAL(nn_fsbc) * rn_Dt          !--- sea-ice timestep and its inverse',
    'src/ICE/iceupdate.F90:208': 'CALL ice_alb( ln_pnd_alb, t_su(:,:,:), h_i(:,:,:), h_s(:,:,:), a_ip_eff(:,:,:), &',
    # --- SI3 round 11 (O7) codex HOLD fixes ---
    'packages/core/legoesm/thermo.py:385': 'zle = sr(zle + sr(constants.goff_ice_C_nemo * sr(1.0 - sr(zta / sr(jnp.asarray(rtt0))))))',
    'packages/ice/legoesm/ice/si3_surface.py:256': 'acc = sr(jnp.zeros(terms.shape[:-1], dtype=terms.dtype))',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1086': 'zsipt(:,:) = theta_exner( ptsui(:,:), pslp(:,:) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1102': 'CALL turb_ice_an05( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice,       &',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1107': 'CALL turb_ice_lu12( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice, fr_i, &',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1112': 'CALL turb_ice_lg15( rn_zqt, rn_zu, zsipt, ptair, ztmp, pqair, wndm_ice, fr_i, &',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1225': 'zsipt = theta_exner( zst, pslp(ji,jj) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1280': 'evap_ice (ji,jj,jl) = rn_efac * qla_ice (ji,jj,jl) * z1_rLsub    ! sublimation',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:463': 'fhld (ji,jj) = zswitch * MAX( 0._wp, ( zqld - zqfr_pos ) * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) )',
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:203': ('zafrac_pnd = 0._wp', 1),
    'ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:214': 'IF( ph_snw(ji,jj,jl) == 0._wp .AND. pt_su(ji,jj,jl) >= rt0 ) THEN   ;   zalb_ice = rn_alb_imlt',
    'packages/core/legoesm/core/bulk_flux.py:1646': 'zsipt = nemo_theta_exner(ptsui, pslp)',
    'packages/core/legoesm/core/bulk_flux.py:1636': 'def nemo_si3_blk_ice_1_cst(',
    'packages/core/legoesm/core/bulk_flux.py:1668': 'def nemo_si3_ice_turbulent(',
    'packages/core/legoesm/thermo.py:410': 'def nemo_si3_saturation_over_ice(',
    'packages/core/legoesm/core/nemo_namelist.py:80': 'def parse_namelist(path: Path, ns: str)',
    'packages/ice/legoesm/ice/si3_surface.py:146': 'def from_input_namelists(',
    'packages/ice/legoesm/ice/si3_surface.py:262': 'def _ieee_is_zero(value):',
    'packages/ice/legoesm/ice/si3_surface.py:268': 'def _ieee_le_zero(value):',
    'packages/ice/legoesm/ice/si3_surface.py:300': 'def si3_ice_alb(',
    'packages/ice/legoesm/ice/si3_surface.py:384': 'def si3_blk_ice_2(',
    'packages/ice/legoesm/ice/si3_surface.py:467': 'def si3_ice_flx_other(',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_bulk_flux_gate.py:411': 'cfg = SI3SurfaceConfig.from_resolved_namelists(',
    'tests/ice/unit/test_si3_surface.py:395': 'def test_production_config_is_read_from_the_shipped_deck():',
    'tests/ice/unit/test_si3_surface.py:442': 'def test_jit_equals_independent_glibc_transcription_on_every_output(deck):',
    'tests/ice/unit/test_si3_surface.py:478': 'def test_blk_ice_1_zsipt_is_transcribed_and_has_no_consumer_on_this_arm():',
    'tests/ice/unit/test_si3_surface.py:483': 'def test_zsipt_consumer_plant_is_caught(monkeypatch):',
    'tests/ice/unit/test_si3_surface.py:324': 'def lbc_t_oracle(interior, psgn):',
    'tests/ice/unit/test_si3_surface.py:514': 'def test_production_path_plants_fire(plant, monkeypatch):',
    # --- SI3 round 12 (O5) admitted-record result ---
    'ORCA2_SAS_ICE_O5_REC/BLD/ppsrc/nemo/icedyn_adv_pra.f90:267': 'odd ice time step:  adv_x then adv_y',
    'ORCA2_SAS_ICE_O5_REC/BLD/ppsrc/nemo/icedyn_adv_pra.f90:317': 'even ice time step:  adv_y then adv_x',
    # --- SI3 round 13 (O7) retained-record admission ---
    'ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icesbc.f90:435-436': ['zqfr_neg = MIN( zqfr , 0._wp )', 'zqfr_pos = MAX( zqfr , 0._wp )', 2],
    'ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icesbc.f90:447': 'qsb_ice_bot(ji,jj) = zswitch * MIN( qsb_ice_bot(ji,jj), - zqfr_neg * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) )',
    'packages/ice/legoesm/ice/si3_surface.py:505-506': ['zqfr_neg = _fmin(zqfr, 0.0)', 'zqfr_pos = _fmax(zqfr, 0.0)', 2],
    'packages/ice/legoesm/ice/si3_surface.py:510': 'qsb = sr(zswitch * jnp.where(cap <= qsb, cap, qsb))',
    'tests/ice/unit/test_si3_surface.py:615': 'def test_production_surface_step_keeps_qsb_dry_negative_zero():',
    # --- SI3 round 15 (O8) frazil collection bookkeeping ---
    'ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27': '%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:113': '_NEW_ICE_THICKNESS = 0.05  # C1D card rn_hinew, c1d_omip_l3_namelist_ice_cfg:113',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:174': 'CALL ice_thd_frazil             !--- frazil ice: collection thickness (ht_i_new) & fraction of frazil (fraz_frac)',
    # --- SI3 round 18 (O8--O16 master acquisition boundary inventory) ---
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:174-229': ['CALL ice_thd_frazil', 'CALL ice_cor( kt , 2 )', 56],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:393-477': [('SUBROUTINE ice_thd_frazil', 1), 'ht_i_new(ji,jj) = 0._wp', 85],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:112-139': ['DIMENSION(jpij,0:nlay_s) ::   zradtr_s', 'DIMENSION(nlay_i+nlay_s+1,3) ::   ztrid', 28],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:387-577': ['qns_ice_1d(ji) = qns_ice_1d(ji) + dqns_ice_1d(ji)', 't_su_1d(ji) = MAX( MIN( t_su_1d(ji) , rt0 )', 191],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:115-687': [('SUBROUTINE ice_thd_sal', 1), ('END SUBROUTINE ice_thd_sal', 1), 573],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-210': [('CALL ice_thd_temp', 1), ('CALL ice_thd_da', 1), 10],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:162-264': ['SUBROUTINE ice_thd_pnd( kt )', ('END SUBROUTINE ice_thd_pnd', 1), 103],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:175-232': ['h_ib_2d(1:npti,:), h_i_b', 'h_ib_2d(ii,1) > ( hi_max(1)', 58],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:464': 'rn_amax_1d(1:npti), rn_amax_2d',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:196-206': ['qlead_1d     (1:npti), qlead', 'sss_1d    (1:npti), sss_m', 11],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:304': 'zh_newice(ii) > hi_max(jl-1)',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:94-118': ['WHERE( at_i', 'rn_sinew*sss_m(ji,jj) * v_i', 25],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:632-668': ['v_i(ji,jj,jl) > (a_i(ji,jj,jl) * hi_max(jl))', 'v_i(ji,jj,jl+1) <= (a_i(ji,jj,jl+1) * hi_max(jl))', 37],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:254-257': [('CALL ice_var_glo2eqv(2)', 2), 'CALL ice_update_flx( kt )', 4],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:105-111': [('INTERFACE ice_var_snwfra', 1), ('END INTERFACE', 3), 7],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:1578-1643': [('SUBROUTINE ice_var_snwfra_3d', 1), 'END SUBROUTINE ice_var_snwblow_1d', 66],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:185-190': [('! mean ice/snow thickness', 1), ('tm_si(ji,jj) = SUM( t_si', 1), 6],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:296-299': ['! bound h_i by hi_max', 'z1_hmax =  1._wp / hi_max(jpl)', 4],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:362-387': ['ELSEIF( nn_icesal == 2 ) THEN', 'sz_i(ji,jj,jk,jl) = MIN( rn_simin, rn_sinew * sss_m(ji,jj) )', 26],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:505-514': [('DO jk = 1, nlay_i', 3), 'IF( s_i(ji,jj,jl) <= ( REAL( nlay_i , wp ) * MIN( rn_simin, rn_sinew * sss_m(ji,jj) ) )', 10],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:119-223': ['qt_atm_oi(ji,jj) = ( 1._wp - at_i_b(ji,jj) )', 'h_ip(:,:,:), h_il(:,:,:), cloud_fra(:,:)', 105],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:332-423': [('SUBROUTINE ice_update_tau', 1), ('END SUBROUTINE ice_update_tau', 1), 92],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:387-484': [('CASE( 1 )', 1), ('CASE( 2 )', 1), 98],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:484-567': [('CASE( 2 )', 1), 'CALL tab_1d_2d( npti, nptidx(1:npti), qml_ice_1d', 84],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:119-175': ['qt_atm_oi(ji,jj) = ( 1._wp - at_i_b(ji,jj) )', 'qns(ji,jj) = qt_oce_ai(ji,jj) - qsr(ji,jj)', 57],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceupdate.f90:400-418': ['IF ( ln_drgice_imp ) THEN', 'vtau(ji,jj) = ( 1._wp - at_i(ji,jj) )', 19],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/zdfdrg.f90:61': 'LOGICAL , PUBLIC ::   ln_drgice_imp = .FALSE.',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/nemogcm.f90:138-161': ['CALL nemo_init', 'CALL stp( istp )', 24],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/nemogcm.f90:355-391': ['CALL nemo_ctl', 'END SUBROUTINE nemo_init', 37],
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:190-191': ['rn_hinew         =   0.1', 'ln_frazil        = .false.', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:419': 'fraz_frac(:,:) = 0._wp',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:422': 'WHERE( qlead(:,:) < 0._wp ) ! cooling',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:423': 'ht_i_new(:,:) = rn_hinew',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:425': 'ht_i_new(:,:) = 0._wp',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:428': 'IF( ln_frazil ) THEN',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/ice.f90:445': 'ALLOCATE( ht_i_new (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , fraz_frac (Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , STAT=ierr(ii) )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:474': 'ht_i_new(ji,jj) = MAX( 0.01_wp, MIN( ht_i_new(ji,jj), hi_max(jpl) ) )',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1799': 'def si3_ice_thd_frazil(qlead, *, rn_hinew, ln_frazil) -> SI3Frazil:',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1787': 'def _ieee_lt_zero(value):',
    # --- SI3 round 15 resume 1 (O8 production wiring) ---
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:182': 'IF ( qlead(ji,jj)  <  0._wp ) THEN',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:200': 'CALL tab_2d_1d( npti, nptidx(1:npti), zh_newice    (1:npti), ht_i_new   )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_do.f90:283': 'za_newice = zv_newice / zh_newice(ii)',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1750': 'frazil = si3_ice_thd_frazil(forcing.qlead, rn_hinew=_NEW_ICE_THICKNESS, ln_frazil=False)',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1751': 'cooling = _ieee_lt_zero(forcing.qlead)',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1759': 'vnew / jnp.where(cooling, frazil.ht_i_new, 1.0),',
    'scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:113-114': ['rn_hinew         = 0.05', 'ln_frazil         = .false.', 2],
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2808': 'def test_o8_production_c1d_step_calls_shared_frazil(probe_module, monkeypatch):',
    'tests/ocean/fidelity/test_nemo_si3thd_d5_platform_probe.py:2822': 'def test_o8_production_c1d_plants_fire(probe_module, monkeypatch, plant):',
    # --- SI3 round 17 (O11) lateral melt ice_thd_da ---
    'scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:91': 'ln_icedA         = .false.    ! ORCA1 ice cfg:87',
    'packages/ice/legoesm/ice/config.py:308': 'lateral_melt: bool = False',
    'packages/ice/legoesm/ice/si3_surface.py:243': 'def _fmin(a, b):',
    'packages/ice/legoesm/ice/si3_surface.py:253': 'def _sum0(terms):',
    'packages/ice/legoesm/ice/si3_surface.py:590-593': ['fortran_min = _fmin', 'ieee_is_zero = _ieee_is_zero', 4],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1822': 'ieee_lt_zero = _ieee_lt_zero',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:210': 'IF( ln_icedA )    CALL ice_thd_da                       ! --- Lateral ',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:179': 'rn_beta          =   1.0           !  coef. beta for lateral melting p',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:183': 'rn_dmin          =   8.            !  minimum floe diameter for latera',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:199': 'nn_icesal        =   4             !  ice salinity option',
    'packages/ice/legoesm/ice/si3_thd_da.py:25-28': ['_DA_ZDMAX_M = 300.0  # zdmax, maximum floe diameter', '_DA_ZM2 = 1.36  # zm2, melt-speed exponent', 4],
    'packages/ice/legoesm/ice/si3_thd_da.py:65': 'def si3_ice_thd_da(',
    'packages/ice/legoesm/ice/si3_thd_da.py:143': 'active = _ieee_gt_zero(a_i)',
    'packages/ice/legoesm/ice/si3_thd_da.py:149': 'zdt = fortran_max(jnp.zeros_like(sst), sr(sst - sr(t_bo - rt0)))',
    'packages/ice/legoesm/ice/si3_thd_da.py:152': 'zda_tot = fortran_min(at_i, sr(sr(zwlat * zperi) * rdt))',
    'packages/ice/legoesm/ice/si3_thd_da.py:155': 'zda = fortran_min(a_i, sr(sr(zda_tot * a_i) / at_div))',
    'packages/ice/legoesm/ice/si3_thd_da.py:170': 'emptied = ieee_is_zero(a_new)',
    'packages/ice/legoesm/ice/si3_thd_da.py:50': 'zdmax = sr(jnp.float64(_DA_ZDMAX_M))',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:419': 'r1_Dt_ice = 1._wp / rDt_ice',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icestp.f90:423': 'r1_nlay_i = 1._wp / REAL( nlay_i, wp )   !--- inverse of nlay_i and nl',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:36': 'REAL(wp), PUBLIC ::   rpi      = 3.141592653589793_wp             !: p',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:50': 'REAL(wp), PUBLIC ::   rt0      = 273.15_wp         !: freezing point o',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:68-69': ['REAL(wp), PUBLIC ::   rhos     =  330._wp          !: volumic mass of ', 'REAL(wp), PUBLIC ::   rhoi     =  917._wp          !: volumic mass of ', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:167-170': ['REAL(wp), PARAMETER ::   zdmax = 300._wp', 'REAL(wp), PARAMETER ::   zm2   = 1.36_wp', 4],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:174': 'zastar = 1._wp / ( 1._wp - (rn_dmin / zdmax)**(1._wp/rn_beta) )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:178': 'IF( nn_icesal == 4 ) THEN   ;   zs_i(:) = sz_i_1d(ji,:)  ! use layer s',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:179': 'ELSE                        ;   zs_i(:) = s_i_1d (ji)    !     bulk sa',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:183': 'zdfloe = rn_dmin * ( zastar / ( zastar - at_i_1d(ji) ) )**rn_beta     ',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:185': 'zperi  = at_i_1d(ji) * rpi / ( zcs * zdfloe )                         ',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:187': 'zwlat  = zm1 * ( MAX( 0._wp, sst_1d(ji) - ( t_bo_1d(ji) - rt0 ) ) )**z',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:189': 'zda_tot = MIN( zwlat * zperi * rDt_ice, at_i_1d(ji) )                 ',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:192': 'IF( a_i_1d(ji) > 0._wp ) THEN',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:195': 'zda = MIN( a_i_1d(ji), zda_tot * a_i_1d(ji) / at_i_1d(ji) )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:198': 'sfx_lam_1d(ji) = sfx_lam_1d(ji) + rhoi * zda * r1_Dt_ice * h_i_1d(ji) ',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:201-202': ['hfx_thd_1d(ji) = hfx_thd_1d(ji) - zda * r1_Dt_ice * ( h_i_1d(ji) * r1_', '+ h_s_1d(ji) * r1_nlay_s * SUM( e_s_1d(ji,1:nlay_s) ) )', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:205': 'wfx_lam_1d(ji) =  wfx_lam_1d(ji) + zda * r1_Dt_ice * ( rhoi * h_i_1d(j',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:208': 'a_i_1d(ji) = a_i_1d(ji) - zda',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_da.f90:211-213': ['IF( a_i_1d(ji) == 0._wp ) THEN', 'h_s_1d(ji) = 0._wp', 3],
    # --- SI3 round 17 resume 1 (O11 production wiring) ---
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:181': 'IF ( a_i(ji,jj,jl) > epsi10 ) THEN',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:205': ('CALL ice_thd_temp', 2),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:227': 'IF( ln_icedO )          CALL ice_thd_do',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:41': 'sst_m: jax.Array | None = None',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1677': 'si3=None) -> SI3StepTrace:',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1735': 'post_da = _si3_column_thd_da(state, post_temp2, forcing, si3, dt, constants)',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1848': 'def _si3_column_thd_da(entry, post_temp2, forcing, si3, dt, constants):',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1880': 'gathered = entry.concentration > _EPS10',
    'packages/ice/legoesm/ice/sea_ice.py:525': 'trace = si3_column_step_arrays(arrays, forcing, dt, config.ice_constants, si3=config.si3)',
    'packages/ice/legoesm/ice/config.py:309': 'namthd_da: SI3LateralMeltConfig | None = None',
    'packages/ice/legoesm/ice/config.py:316': 'wanted = SI3ThermoConfig(lateral_melt=config.si3.lateral_melt, namthd_da=config.si3.namthd_da)',
    'packages/ice/legoesm/ice/config.py:597-601': ['class SI3LateralMeltConfig(NamedTuple):', 'rn_dmin: float', 5],
    # --- SI3 round 16 (O9) ZDF/DH/TEMP generalisation ---
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:197': 'CALL ice_thd_zdf                      ! --- Ice-Snow temperature --- !',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:199': 'IF( ln_icedH )    CALL ice_thd_dh                       ! --- Growing/Melting --- !',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201': ('CALL ice_thd_temp                     ! --- Temperature update --- !', 1),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:203': 'CALL ice_thd_sal                      ! --- Ice salinity --- !',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:532': 'def _si3_zdf_bl99_core(',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:923': 'def si3_ice_thd_zdf_bl99(',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1251': 'def _si3_ice_thd_dh_core(',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1524': 'def si3_ice_thd_dh(*, fluxes: SI3DHFluxes, rDt_ice, rn_snwblow, rn_sinew,',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1216': 'def _cumulative_remap(h_old, content_old, nlay, sr):',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:522': 'IF( nn_icesal == 4 )   CALL ice_var_vremap( zh_i_old, zs_i_old, sz_i_1d(ji,:) )',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1508': 'def si3_ice_thd_temp(*, h_i, e_i, sz_i, constants: IceConstantsConfig):',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1558': 'def si3_ice_thd_category_stages(*, entry: dict, fluxes: SI3DHFluxes, namelist: dict,',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:439': 'def _maxsd(dst, src):',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:444': 'def _minsd(dst, src):',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:128': 'zq_top = MAX( 0._wp, qml_ice_1d(ji) * rDt_ice )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:180': 'ze_s(0) = MAX( 0._wp, - qprec_ice_1d(ji) )                              ! enthalpy of the precip (>0, J.m-3)',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:589': ('t_i_1d(ji,jk) =  MAX( MIN( t_i_1d(ji,jk), ztmelts ), rt0 - 100._wp )', 1),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:202': 't_su_1d    (1:npti) = MIN( t_su_1d(1:npti), rt0 - ztsu_err )   ! required to leave the choice between melting or not',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:221': 'zdum = - zq_top / MAX( ze_s(jk), epsi20 )   ! thickness change',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:194': 'zdeltah    = MAX( - evap_ice_1d(ji) * r1_rhos * rDt_ice, - h_s_1d(ji) )   ! amount of snw that sublimates (<0) or deposition (>0)',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:300': 'zdum               = MAX( - zh_i(jk) , - zevap_rema * r1_rhoi )',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1334': 'zdeltah = -_minsd(sr(sr(evap_ice * r1_rhos) * dt), h_s)',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:490': 'zdum        = MIN( zdeltah, zh_s(jk) )         ! amount of snw that floods, > 0',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1473': 'zdum = _minsd(zh_s[jk], zdeltah)  # :490 emitted `minsd zdeltah, zh_s`',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:955': 't_i_1d(ji,jk) = MIN( t_i_1d(ji,jk), ztmelts + rt0 ) ! Force t_i_1d to be lower than melting point => likely conservation issue',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:285': 'zccc          = SQRT( MAX( zbbb * zbbb - 4._wp * rcpi * rLfus * ztmelts, 0._wp ) )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:191': 'isnow_comb(ji) = h_s_1d(ji) / zh_min',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:537-538': [('zdiagbis(jm) = ztrid   (jm,2) - ztrid(jm,1) * ztrid   (jm-1,3) / zdiagbis(jm-1)', 1), ('zindtbis(jm) = zindterm(jm  ) - ztrid(jm,1) * zindtbis(jm-1  ) / zdiagbis(jm-1)', 1), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_zdf_bl99.f90:857': 'cnd_ice_1d(ji) = 2._wp * ztcond_i(ji,0) / zhi_ssl ! cnd_ice is capped by: cond_i/zhi_ssl',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:449': 'def _div_const(x, c):',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:590': 'isnow_comb = jnp.where(h_s < _H_MIN, _div_const(h_s, _H_MIN), 1.0)  # :190-194',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:995': '_div_const(2.0 * core.tcond_i[..., 0], _ICE_SSL_DEPTH),',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1293': "zero = sr(jnp.zeros_like(h_i))  # materialized: NEMO's 0._wp + (-0) is +0, XLA's folded 0 + x is not",
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:670': 'def test_five_categories_bit_identical_to_independent_transcription():',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:693': 'def test_eager_equals_jit_every_bit(jl):',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:706': 'def test_inexact_value_pins():',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:721': 'def test_production_path_plants_fire(plant, monkeypatch):',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:751': 'def test_signed_zero_and_nan_sites_follow_object_code():',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:766': 'def test_c1d_card_routes_its_dh_through_the_shared_core(monkeypatch):',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:783': 'def test_gradients_finite_and_central_difference():',
    'tests/ice/unit/test_si3_o9_zdf_dh_temp.py:802': 'def test_no_defaults_and_refused_arms():',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_dh.f90:463': 'zEw            = rcp * sst_1d(ji)',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1612': 'sst=forcing.t_bottom - constants.T0,',
    # Landed O1 anchors consumed by the cross-rung re-anchor chain below.
    "packages/ice/legoesm/ice/si3_orca2_state.py:51-62": [
        "# Fixed NEMO 5.0.2 SI3 coefficients.",
        "_ORCA2_RN_HIMAX_M = 99.0",
        12,
    ],
    "packages/ice/legoesm/ice/si3_orca2_state.py:2037-2047": [
        'if ln_rstart and "snwice_mass" in rst_full and "snwice_mass_b" in rst_full:',
        "snwice_mass_b = si3_lbc_lnk_orca2(",
        11,
    ],
    # --- SI3 ladder round 8 (O4 model prep) BEGIN ---
    'docs/ice/fidelity/si3_r5_d5_preregister.md:1-48': ['# SI3 Ladder Round 5 (`si3_r5`) — Rung D5 Debts Preregistration', '- **REFUTE**: Omitting `sss_m` does not raise `GateError`, CPU production JIT and `--xla_cpu_max_isa=AVX` differ in full-year SHA-256, any non-SI3 caller changes by `>= 1` ULP, or `jax.grad` / `jax.jvp` fails.', 48],
    'docs/ice/fidelity/si3_r6_d5_preregister.md:1-37': ['# SI3 Ladder Round 6 (`si3_r6`) — Rung D5 Remaining Debts Preregistration', '- **REFUTE**: Any normal-step (`kt != 5842..5867`) over-bar row remains (`> 0`) on CPU or any over-bar row (`> 0`) remains on GPU, CPU production JIT and `--xla_cpu_max_isa=AVX` differ in full-year SHA-256, any non-SI3 caller changes by `>= 1` ULP vs `071dfa844369`, or `jax.grad` / `jax.jvp` fails.', 37],
    'docs/ice/fidelity/si3_ladder_design.md:224': '| O4 | ORCA2_SAS_ICE with Prather only; dynamics/thermo otherwise off. | Add **five-category Prather across the cyclic seam and north fold**, replacing the periodic arm-local halo. D1 and D2 gate composition, but exact-input seam/fold admission can proceed. | Both sweep orders, fold/seam exchanges and every moment; 10–20 GiB. |',
    'docs/ice/fidelity/si3_ladder_design.md:223': '| O3 | Same diagnostic deck, O2 scalar state as exact input. | Add the **initial HFN thickness-distribution fit** (`ice_var_itd`) separately from state construction (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:354-364`). | Category fit inputs/outputs, layer initialization and conservation; about 1–2 GiB. |',
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:136': "CALL mpp_max( 'icedyn_adv_pra', zcfl, cdelay = 'cflice' )",
    'si3_rounds/round4/refused/oracle_o1_orca2_sas_ice_admission_refused_20261024T161410Z_1010784/ref_2step/ocean.output:43': 'use delayed global communications?             ln_mppdelay =  T',
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:789': ('llisdelay = ln_mppdelay .AND. PRESENT(cdelay)', 3),
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:823-828': [('IF( ndelayid(idvar) == ndlrstoff ) THEN    ! first call without restart: define zdpbuffout from ptab(:) with a blocking allreduce', 1), ('ENDIF', 17), 6],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lib_mpp.f90:834-840': [('todelay(idvar)%zdpbuffin(:) = ptab(:)', 1), 'CALL mpi_iallreduce( todelay(idvar)%zdpbuffin, todelay(idvar)%zdpbuffout, ipi, MPI_DOUBLE_PRECISION, MPI_MAX, ilocalcomm, ndelayid(idvar), ierr )', 7],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/iom.f90:1396-1400': ["idvar = iom_varid( kncid, 'DELAY__r8_'   //c_delaylist(ji), kdimsz = isz, ldstop = .FALSE. )", ('ndelayid(ji) = ndlrstuse   ! set to ndlrstuse to specify that the value was read in the restart', 3), 5],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90:138': "CALL iom_delay_rst( 'WRITE', numriw, iter )   ! save all delayed global communication variables",
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icerst.f90:353': "CALL iom_delay_rst( 'READ', numrir )   ! read all delayed global communication variables (if not already done)",
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:138-141': ['IF    ( zcfl > 1.5 ) THEN   ;   icycle = 3', ('ENDIF', 2), 4],
    'packages/ice/legoesm/ice/si3_orca2_state.py:363': 'cflice_delayed: Field | None = None',
    'packages/ice/legoesm/ice/si3_orca2_state.py:3223-3230': ['cfl_u = jnp.max(sr(sr(jnp.abs(u_ice) * rDt_ice) * grid.r1_e1u))', 'cflice_delayed_out = zcfl_curr', 8],
    'packages/ice/legoesm/ice/si3_orca2_state.py:1663-1665': ['if do_write_prather and cfg.ln_mppdelay and state.cflice_delayed is not None:', 'out["DELAY__r8_cflice"] = np.asarray(state.cflice_delayed, dtype=np.float64)', 3],
    'packages/ice/legoesm/ice/si3_orca2_state.py:3034-3116': ['def _si3_pra_subcycle_cond_dispatch(', (')', 1088), 83],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:821-1122': ['def test_o4_a2_odd_even_sweep_order_and_cfl_subcycling_pins_and_plant() -> None:', (')', 400), 302],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2150-2241': ['def test_o4_d_jit_primal_zero_ulp_and_delayed_cfl_selection() -> None:', '_assert_zero_ulp(e_st.prather.ice.sx, j_st.prather.ice.sx, f"jit_delayed_icycle{want}.sx")', 92],
    'packages/ice/legoesm/ice/config.py:609-623': ['def build_si3_orca2_o4_config(', 'return cfg', 15],
    'packages/ice/legoesm/ice/si3_orca2_state.py:3119-3350': ['def si3_dyn_adv_pra_orca2(', (')', 1155), 232],
    'packages/ice/legoesm/ice/si3_orca2_state.py:3353-3783': ['def si3_dyn_orca2(', 'return state_dyn, diagnostics', 431],
    'packages/ice/legoesm/ice/si3_orca2_state.py:3786-3939': ['def si3_o4_step(', 'return out_state, diagnostics', 154],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:485': 'sfx_sub(ji,jj) = 0._wp   ;   sfx_res(ji,jj) = 0._wp',
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:491': 'wfx_sub(ji,jj) = 0._wp   ;   wfx_res(ji,jj) = 0._wp',
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:502': 'hfx_sub(ji,jj) = 0._wp   ;   hfx_res(ji,jj) = 0._wp',
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:521-523': ['diag_adv_mass(ji,jj) = 0._wp', 'diag_adv_heat(ji,jj) = 0._wp', 3],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1125-1468': ['def test_o4_a3_icemax_hbig_hsnow_zapneg_zapsmall_pato_diag_pins_and_plants() -> None:', 'build_si3_orca2_o4_config(None)  # type: ignore[arg-type]', 344],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1471-1671': ['def test_o4_a4_orca2_seam_and_north_fold_subcycle_lbc_lnk_pins_and_plants() -> None:', (')', 616), 201],
    'packages/ice/legoesm/ice/transport.py:858-1348': ['def _si3_prather_x_substep(', ('return content, moments, area', 1), 491],
    'packages/ice/legoesm/ice/transport.py:1351-1826': ['def _si3_prather_y_substep(', ('return content, moments, area', 2), 476],
    'packages/ice/legoesm/ice/transport.py:2233-2275': ['def si3_adv_x(', ('return area, content_out, moments_out', 1), 43],
    'packages/ice/legoesm/ice/transport.py:2278-2318': ['def si3_adv_y(', ('return area, content_out, moments_out', 2), 41],
    'packages/ice/legoesm/ice/fidelity/nemo_adv2d_testcase_recipe.py:388-613': ['def apply_si3_prather_source_corrections(', 'return _periodic_halo(contents, halo_width)', 226],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_lane3_prather_dump.py:59-150': ['def main(argv: list[str]) -> None:', 'np.savez(out_path, **out)', 92],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1713-1744': ['def test_o4_b_lane3_parent_vs_current_zero_ulp(tmp_path: Path) -> None:', 'assert {"corr_int", "corr_ext"} <= differing', 32],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:536-614': ['def _ref_nemo_lbc_lnk_orca2_t_point(', 'return arr_3d.reshape(orig_shape)', 79],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90:96-150': [("CASE ( 'T' , 'W' )                         ! T-, W-point", 1), ('END DO   ;   END DO', 1), 55],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90:1987': ('!     4. Fill east and west halos     !', 2),
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90:2105': ('!     6. north fold treatment     !', 2),
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbclnk.f90:2118': ('!           7. Fill south and north halos          !', 2),
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:451-452': [("CALL lbc_lnk( 'icedyn_adv_pra', pv_i  , 'T', 1._wp, sxice , 'T', -1._wp, syice , 'T', -1._wp  & ! ice volume", 1), ("&                          , sxxice, 'T', 1._wp, syyice, 'T',  1._wp, sxyice, 'T',  1._wp  &", 1), 2],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2244-2438': ['def _o4_ad_checks() -> None:', 'print(f"O4_AD_CHECKS_PASS fd_u_ratio={fd_u_ratio!r} fd_vi_ratio={fd_vi_ratio!r} rel_ad_diff={rel_ad_diff!r}")', 195],
    'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2441-2466': ['def test_o4_d_ad_reverse_forward_and_central_fd() -> None:', 'print(result.stdout.strip().splitlines()[-1])', 26],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/ice.f90:465': 'ALLOCATE( t_si(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpl) , tm_si(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) , qcn_ice_bot(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpl) , qcn_ice_top(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),jpl) , STAT = ierr(ii) )',
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:529': 't_si       (ji,jj,jl) = rt0     ! temp at the ice-snow interface',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_acquisition/o1_si3_state_writer.F90:35-40': ['SUBROUTINE o1_pad_a2d_3d( psrc, pdst )', 'END SUBROUTINE o1_pad_a2d_3d', 6],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:226-227': ['CALL ice_var_glo2eqv(2)       ! h_i and h_s for ice albedo calculation', 'CALL ice_var_agg(1)           ! at_i for coupling', 2],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:254-255': [('CALL ice_var_glo2eqv(2)', 2), 'CALL ice_var_agg(2)', 2],
    'ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:245': 'CALL ice_sbc_flx( kt, ksbc )',
    'ORCA2_SAS_ICE_O1_REF/EXP00/namelist_ref:1515': 'ln_mppdelay  =  .true.  !  activate delayed global communications to go faster',
    # --- SI3 ladder round 8 (O4 model prep) END ---
}


# Cross-lane additions move source without changing the receipts' claims.
# Keep the historical keys above reviewable, then replace every moved key with
# its current, independently anchored location before the mandatory full-map
# audit runs.  Extents are explicit: none is inferred from the new key.
def _reanchor(old: str, new: str, *, extent: int | None = None, anchors=None) -> None:
    value = CITATION_MAP.pop(old)
    if anchors is not None:
        value = anchors
    elif extent is not None:
        if isinstance(value, dict):
            value = {**value, "anchors": [*value["anchors"][:-1], extent]}
        else:
            value = [*value[:-1], extent]
    if new in CITATION_MAP:
        raise RuntimeError(f"duplicate re-anchored citation {new}")
    CITATION_MAP[new] = value


_reanchor('packages/ice/legoesm/ice/config.py:566-575', 'packages/ice/legoesm/ice/config.py:571-579', anchors=['def build_si3_orca2_o1_config(', ('si3_orca2=cfg,', 1), 9])
_reanchor('packages/ice/legoesm/ice/config.py:531-563', 'packages/ice/legoesm/ice/config.py:531-568', extent=38)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:51-62', 'packages/ice/legoesm/ice/si3_orca2_state.py:54-65')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2037-2047', 'packages/ice/legoesm/ice/si3_orca2_state.py:2040-2050')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:590-610', 'packages/ice/legoesm/ice/sea_ice.py:592-612', anchors=[('if config.si3_orca2 is None:', 1), ('"heat/freshwater/salt TileResponse contract is implemented"', 1), 21])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1778-1843', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1804-1869')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1830-1852', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1856-1878')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2105-2661', 'packages/ice/legoesm/ice/si3_orca2_state.py:2261-2768', extent=508)
_reanchor('packages/ice/legoesm/ice/sea_ice.py:643-697', 'packages/ice/legoesm/ice/sea_ice.py:621-655', anchors=['def _si3_orca2_o2_step_with_trace(', ('"heat/freshwater/salt TileResponse contract is implemented"', 2), 35])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:1-1073', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:1-1221', extent=1221)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1-1599', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1-1718', extent=1718)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2105-2210', 'packages/ice/legoesm/ice/si3_orca2_state.py:2261-2367', anchors=['class SI3O2InitConfig(NamedTuple):', ('apply_halos: bool', 5), 107])
# Round 11 removed one import line near the top of sea_ice.py: every later line moved up by one.
_reanchor('packages/ice/legoesm/ice/sea_ice.py:89-121', 'packages/ice/legoesm/ice/sea_ice.py:90-122')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:329-362', 'packages/ice/legoesm/ice/sea_ice.py:335-368')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:364-365', 'packages/ice/legoesm/ice/sea_ice.py:370-371')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:364-367', 'packages/ice/legoesm/ice/sea_ice.py:370-373')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:592-612', 'packages/ice/legoesm/ice/sea_ice.py:599-619')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:621-655', 'packages/ice/legoesm/ice/sea_ice.py:628-662')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2105-2156', 'packages/ice/legoesm/ice/si3_orca2_state.py:2261-2314', extent=54)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2159-2193', 'packages/ice/legoesm/ice/si3_orca2_state.py:2317-2351')
_reanchor('packages/ice/legoesm/ice/config.py:578-587', 'packages/ice/legoesm/ice/config.py:583-593', anchors=['def build_si3_orca2_o2_config(', ('si3_orca2=cfg,', 2), 11])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2196-2210', 'packages/ice/legoesm/ice/si3_orca2_state.py:2354-2367', anchors=['class SI3ORCA2O2Forcing(NamedTuple):', ('apply_halos: bool', 5), 14])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2214-2370', 'packages/ice/legoesm/ice/si3_orca2_state.py:2371-2474', extent=104)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2318-2370', 'packages/ice/legoesm/ice/si3_orca2_state.py:2439-2474', extent=36)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2229-2257', 'packages/ice/legoesm/ice/si3_orca2_state.py:2386-2414')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2261-2315', 'packages/ice/legoesm/ice/si3_orca2_state.py:2417-2435', anchors=['def si3_glob_2dsum_ddpdd(', 'jnp.transpose(smask0_i[nis0:nie0, njs0:nje0]),', 19])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2285-2293', 'packages/ice/legoesm/ice/si3_orca2_state.py:42', anchors='from legoesm.ocean.fidelity.nemo_recipe import nemo_ddpdd_sum_2d')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2373-2661', 'packages/ice/legoesm/ice/si3_orca2_state.py:2477-2768', extent=292)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2373-2621', 'packages/ice/legoesm/ice/si3_orca2_state.py:2477-2665', anchors=['def si3_o2_scalar_initial_state(', 'vt_i = fortran_sum_categories(v_i_fit, axis=-1)', 189])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2601-2615', 'packages/ice/legoesm/ice/si3_orca2_state.py:2706-2720')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:74-82', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:75-82', anchors=['EVIDENCE="$arg2"', 'esac', 8])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:578-927', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:645-1052', anchors=['run_preflight() {', "printf 'PREFLIGHT_OK: checklist items 1-7 and all plants passed\\n'", 408])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:972-1073', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:1055-1221', anchors=['prove_real_same_program() {', 'printf \'ACQUISITION_DONE %s\\n\' "$EVIDENCE"', 167])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:964-970', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:1101-1107')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:929-962', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:1066-1099', extent=34)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:581-583', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:648-650')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:186-199', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:229-241', anchors=['verify_work_cfgs_guard() {', ('return 0', 5), 13])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:125-133', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:127-135')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:136-151', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:138-153')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:92-123', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:93-124')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:201-389', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:244-432')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:391-468', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:464-532', anchors=['verify_excerpt_provenance_and_deck_independence() {', 'raise SystemExit(f"STATUS PLANT-FIRED: missing source anchor {token!r} in {label}")', 69])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:470-576', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/run.sh:537-642', anchors=['build_orca2_sas_ice_syntax_mods() {', ('PY', 12), 106])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_layout_fixture.F90:1-316', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/o2_si3_layout_fixture.F90:1-320', extent=320)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1-1111', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1-1214', anchors=['"""Fidelity and invariance tests for SI3 ladder rung O2 (`si3_r7`, `DECISION 153`, `DECISION 159`).', 'assert body == [src_lines[n - 1].rstrip() for n in nums]', 1214])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:163-354', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:152-307', anchors=['def test_o2_parent_vs_current_all_callers_bit_identical(monkeypatch) -> None:', '_assert_zero_ulp(p_o1_st.v_i, planted_o1.v_i, "planted_o1_v_i")', 156])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:119-160', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:113-149', extent=37)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:357-397', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:309-348', anchors=['def test_o2_eos_fzp_teos10_and_eos80_zero_ulp_vs_nemo502() -> None:', '"eos_fzp_eos80",', 40])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:400-432', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:382-424', extent=43)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:435-461', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:427-471', anchors=['def test_o2_self_test_exact_category_inputs_and_all_plants_fire(monkeypatch) -> None:', 'assert switch_by_plant[plant_name] in selected_switches', 45])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:921-1210', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:837-1160', anchors=['def run_self_test(plant: str | None = None) -> dict[str, Any]:', '"ad_grad_hti_abs": grad_hti_val,', 324])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:464-512', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:474-522')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:515-533', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:525-543')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:536-649', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:546-660', anchors=['def test_o2_jit_grad_jvp_and_stop_gradient_plant() -> None:', 'assert float(jnp.linalg.norm(g)) == 0.0', 115])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:652-720', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:663-757', anchors=['def test_o2_production_step_sea_ice_selects_o2_and_validates_config(', 'step_sea_ice(state0, None, z2, z2, z2, sea_cfg, 0.1, cfg.rDt_ice)', 95])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:723-789', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:760-826', extent=67)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:359-476', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:352-384', anchors=['def same_program(', '"new_bld": str(new_bld),', 33])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:792-1085', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:829-1188', extent=360)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1213-1521', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1164-1636', anchors=['def admit_evidence_directory(', '"max_norm_rst_step1": cmp_rst_step1["max_norm_wet"],', 473])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1088-1111', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1191-1214', anchors=['def test_o2_committed_ocean_output_excerpt_provenance() -> None:', 'assert body == [src_lines[n - 1].rstrip() for n in nums]', 24])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/ocean.output.excerpt:1-111', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_acquisition/ocean.output.excerpt:1-71', anchors=['# Selected lines from the admitted NEMO 5.0.2 ORCA2_SAS_ICE O1 run matching this deck; trailing whitespace removed.', ('AAAAAAAA', 2), 71])
_reanchor('nemo_testcase_recipe.py:380,636,3451', 'nemo_testcase_recipe.py:380,636,3408')
# --- SI3 ladder round 9 (O3): the O3 fit inserted above O2 in si3_orca2_state.py ---
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:54-65', 'packages/ice/legoesm/ice/si3_orca2_state.py:55-66')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2040-2050', 'packages/ice/legoesm/ice/si3_orca2_state.py:2041-2051')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2261-2768', 'packages/ice/legoesm/ice/si3_orca2_state.py:2262-3047', extent=786)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1-1718', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1-1681', extent=1681)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2261-2367', 'packages/ice/legoesm/ice/si3_orca2_state.py:2262-2369', extent=108)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2261-2314', 'packages/ice/legoesm/ice/si3_orca2_state.py:2262-2315')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2317-2351', 'packages/ice/legoesm/ice/si3_orca2_state.py:2318-2352')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2354-2367', 'packages/ice/legoesm/ice/si3_orca2_state.py:2356-2369')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2371-2474', 'packages/ice/legoesm/ice/si3_orca2_state.py:2373-2476')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2439-2474', 'packages/ice/legoesm/ice/si3_orca2_state.py:2441-2476')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2386-2414', 'packages/ice/legoesm/ice/si3_orca2_state.py:2388-2416')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2417-2435', 'packages/ice/legoesm/ice/si3_orca2_state.py:2419-2437')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:42', 'packages/ice/legoesm/ice/si3_orca2_state.py:43')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2477-2768', 'packages/ice/legoesm/ice/si3_orca2_state.py:2763-3047', extent=285)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2477-2665', 'packages/ice/legoesm/ice/si3_orca2_state.py:2763-2948', extent=186)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1-1214', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1-1202', extent=1202)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:382-424', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:380-417', extent=38)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:427-471', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:420-464')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:837-1160', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:839-1135', extent=297)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:474-522', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:467-515')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:525-543', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:518-536')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:546-660', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:539-648', extent=110)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:663-757', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:651-745')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:760-826', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:748-814')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:352-384', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:353-385')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:829-1188', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:817-1176')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1164-1636', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o2_orca2_sas_gate.py:1139-1599', extent=461)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1191-1214', 'tests/ocean/fidelity/test_nemo_si3_o2_orca2_sas_gate.py:1179-1202')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2489-2675', 'packages/ice/legoesm/ice/si3_orca2_state.py:2777-2958', anchors=['_control_exact_fitted: tuple[Field, Field, Field, Field, Field, Field] | None = None,', 'snwice_mass = sr(tmask_2d * fortran_sum_categories(mass_by_category, axis=-1))', 182])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2706-2720', 'packages/ice/legoesm/ice/si3_orca2_state.py:2989-3003', anchors=[('zsm_i_ini=zsm_i_ini,', 2), ('snwice_mass=snwice_mass,', 4), 15])
# --- SI3 ladder round 10 (O5): F-point fold (+17 lines) inserted in si3_lbc_lnk_orca2 ---
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2041-2051', 'packages/ice/legoesm/ice/si3_orca2_state.py:2058-2068')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2262-3047', 'packages/ice/legoesm/ice/si3_orca2_state.py:2279-3064')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2262-2369', 'packages/ice/legoesm/ice/si3_orca2_state.py:2279-2386')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2262-2315', 'packages/ice/legoesm/ice/si3_orca2_state.py:2279-2332')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2318-2352', 'packages/ice/legoesm/ice/si3_orca2_state.py:2335-2369')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2356-2369', 'packages/ice/legoesm/ice/si3_orca2_state.py:2373-2386')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2373-2476', 'packages/ice/legoesm/ice/si3_orca2_state.py:2390-2493')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2441-2476', 'packages/ice/legoesm/ice/si3_orca2_state.py:2458-2493')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2388-2416', 'packages/ice/legoesm/ice/si3_orca2_state.py:2405-2433')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2419-2437', 'packages/ice/legoesm/ice/si3_orca2_state.py:2436-2454')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2763-3047', 'packages/ice/legoesm/ice/si3_orca2_state.py:2780-3064')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2763-2948', 'packages/ice/legoesm/ice/si3_orca2_state.py:2780-2965')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2989-3003', 'packages/ice/legoesm/ice/si3_orca2_state.py:3006-3020')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2777-2958', 'packages/ice/legoesm/ice/si3_orca2_state.py:2794-2975')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2529-2638', 'packages/ice/legoesm/ice/si3_orca2_state.py:2546-2655')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2556', 'packages/ice/legoesm/ice/si3_orca2_state.py:2573')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2645-2759', 'packages/ice/legoesm/ice/si3_orca2_state.py:2662-2776')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2927-2934', 'packages/ice/legoesm/ice/si3_orca2_state.py:2944-2951')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3039-3046', 'packages/ice/legoesm/ice/si3_orca2_state.py:3056-3063', anchors=['new_state = si3_o1_step(', (')', 1131), 8])
# SI3 round 16 (O9) generalised bitz_lipscomb.py to n layers; historical citations move to the
# same statements in the current file (anchors recomputed from a parent->current line diff).
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:432-478', 'packages/ice/legoesm/ice/bitz_lipscomb.py:462-515', anchors=['def _radiation(', 'return ab_s, ab_i, tr_i[..., -1]', 54])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:794-888', 'packages/ice/legoesm/ice/bitz_lipscomb.py:746-874', anchors=['Tsu = jnp.where(active, Tsu_candidate, Tsu)', 'qcn_bottom = 2.0 * k_base_i * (Ti[..., -1] - T_bottom)', 129])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:523-529', 'packages/ice/legoesm/ice/bitz_lipscomb.py:816-821', anchors=['T_s_old = snow_temperature_from_enthalpy(', (')', 319), 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:199-232', 'packages/ice/legoesm/ice/bitz_lipscomb.py:201-233', anchors=['def ice_enthalpy_from_temperature(', 'return sr(constants.rho_ice * sr(sr(sensible + latent) - ocean))', 33])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:235-296', 'packages/ice/legoesm/ice/bitz_lipscomb.py:236-298', anchors=['def ice_temperature_from_enthalpy(', (')', 76), 63])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:354-366', 'packages/ice/legoesm/ice/bitz_lipscomb.py:356-368', anchors=['def snow_enthalpy_from_temperature(', (')', 107), 13])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:414-429', 'packages/ice/legoesm/ice/bitz_lipscomb.py:416-429', anchors=['def p07_conductivity(', (')', 122), 14])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:481-891', 'packages/ice/legoesm/ice/bitz_lipscomb.py:780-878', anchors=['def _si3_zdf_bl99_step(', (')', 341), 99])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:917-976', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1035-1094', anchors=['def _nemo_snow_enthalpy_remap(', ('], axis=-1)', 11), 60])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1040-1652', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1628-2250', anchors=['def _dh_step(state: SI3ColumnArrays, zdf: SI3ZDFResult,', 'S_bulk, state.S_layers, state.age_volume)', 623])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:624-862', 'packages/ice/legoesm/ice/bitz_lipscomb.py:623-858', anchors=['k_i = tcond_i / dz_i[..., None]', ('else:', 5), 236])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1727-1732', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2325-2330', anchors=['S_layers = option2_salinity_profile(', (')', 1002), 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1620-1636', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2218-2234', anchors=['e_i = (', (')', 984), 17])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1630-1636', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2228-2234', anchors=['e_s = (', (')', 984), 7])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1201-1203', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1799-1801', anchors=['surface_energy = jnp.maximum(', (')', 785), 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:979-1037', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1097-1155', anchors=['def _nemo_ice_enthalpy_remap(', ('], axis=-1)', 12), 59])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1620-1629', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2218-2227', anchors=['e_i = (', (')', 981), 10])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1210-1402,1578-1629', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1808-2000,2176-2227', anchors=[('if _nemo_dh_ice_content_accumulation:', 1), 'old_h = jnp.stack(zh_i_old_layers, axis=-1)', ('if _nemo_dh_ice_content_accumulation:', 2), (')', 981), 245])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1068-1085,1120-1136,1195-1199', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1666-1683,1718-1734,1793-1797', anchors=[('if _nemo_dh_snow_update:', 1), (')', 730), ('if _nemo_dh_snow_update:', 2), 'remaining_mass = evaporation * dt + snow_delta * constants.rho_snow', 'melted_snow.append(', (')', 782), 40])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:40,979-1037,1210-1402,1578-1629,1655-1707', 'packages/ice/legoesm/ice/bitz_lipscomb.py:41,1097-1155,1808-2000,2176-2227,2253-2305', anchors=['frq_m: jax.Array | float = 1.0', 'def _nemo_ice_enthalpy_remap(', ('], axis=-1)', 12), ('if _nemo_dh_ice_content_accumulation:', 1), 'old_h = jnp.stack(zh_i_old_layers, axis=-1)', ('if _nemo_dh_ice_content_accumulation:', 2), (')', 981), 'def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-24', 358])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:40,1281-1285', 'packages/ice/legoesm/ice/bitz_lipscomb.py:41,1879-1883', anchors=['frq_m: jax.Array | float = 1.0', 'frq_m = getattr(forcing, "frq_m", 1.0)', (')', 829), 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1608-1617', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2206-2215', anchors=['ze_i_old = jnp.stack(', (')', 978), 10])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1621-1626', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2219-2224', anchors=[('_nemo_ice_enthalpy_remap(', 2), (')', 979), 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1068-1085,1120-1136,1195-1199,1672,1704', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1666-1683,1718-1734,1793-1797,2270,2302', anchors=[('if _nemo_dh_snow_update:', 1), (')', 730), ('if _nemo_dh_snow_update:', 2), 'remaining_mass = evaporation * dt + snow_delta * constants.rho_snow', 'melted_snow.append(', (')', 782), ('_nemo_dh_snow_update: bool = True,', 2), '_nemo_dh_snow_update=_nemo_dh_snow_update,', 42])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1073-1077', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1671-1675', anchors=['h_precip_raw = (', (')', 727), 5])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1078-1080', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1676-1678', anchors=['h_precip = jnp.where(', (')', 728), 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1124-1130', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1722-1728', anchors=[('snow_delta = jnp.maximum(', 1), (')', 751), 7])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1195-1199', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1793-1797', anchors=['melted_snow.append(', (')', 782), 5])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1076,1235,1338', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1674,1833,1936', anchors=[('/ a', 3), 'zfmdt = sr(-surface_energy / energy_difference)', 'zfmdt = sr(-basal_energy / energy_difference)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:40,1281-1285,1465', 'packages/ice/legoesm/ice/bitz_lipscomb.py:41,1879-1883,2063', anchors=['frq_m: jax.Array | float = 1.0', 'frq_m = getattr(forcing, "frq_m", 1.0)', (')', 829), '+ zdf.qtr_ice_bottom * getattr(forcing, "frq_m", 1.0)', 7])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1052-1053,1671-1672', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1640-1641,2269-2270', anchors=[('_nemo_dh_ice_content_accumulation: bool = True,', 1), ('_nemo_dh_snow_update: bool = True,', 1), ('_nemo_dh_ice_content_accumulation: bool = True,', 2), ('_nemo_dh_snow_update: bool = True,', 2), 4])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1655-1691', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2253-2289', anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', 'e_snow=zdf.e_snow)', 37])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:481-499', 'packages/ice/legoesm/ice/bitz_lipscomb.py:780-798', anchors=['def _si3_zdf_bl99_step(', ') -> SI3ZDFResult:', 19])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:321-351', 'packages/ice/legoesm/ice/bitz_lipscomb.py:323-353', anchors=['def bitz_lipscomb_conduction_step(q_layers, h, S, F_top, T_bottom_K, d', 'return q_new, T_new, F_bottom', 31])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:402-411', 'packages/ice/legoesm/ice/bitz_lipscomb.py:404-413', anchors=['if _nemo_order:', 'return jnp.clip(unbounded, constants.T0 - 100.0, constants.T0)', 10])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:369-411', 'packages/ice/legoesm/ice/bitz_lipscomb.py:371-413', anchors=['def snow_temperature_from_enthalpy(', 'return jnp.clip(unbounded, constants.T0 - 100.0, constants.T0)', 43])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:148-196', 'packages/ice/legoesm/ice/bitz_lipscomb.py:150-198', anchors=['def option2_salinity_profile(', 'return S_bulk[..., None] * (sr(alpha[..., None] * z) + (1.0 - alpha[..', 49])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:894-914', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1012-1032', anchors=['def _piecewise_remap(', 'return jnp.where(dz_new[..., None] > _EPS10, content / dz_new[..., Non', 21])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1655-1773', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2253-2371', anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', 'post_temp2, post_do, exit_state)', 119])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1225-1234', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1823-1832', anchors=[('internal_melt = _surface_melt & (', 1), ('energy_difference = sr(ice_specific - water_specific)', 1), 10])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1331-1337', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1929-1935', anchors=[('ice_specific = (', 2), ('energy_difference = sr(ice_specific - water_specific)', 2), 7])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1374-1383', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1972-1981', anchors=[('bottom_specific = (', 1), ('bottom_difference = bottom_specific - bottom_water', 1), 10])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1206,1225-1234,1331-1337,1374-1383', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1804,1823-1832,1929-1935,1972-1981', anchors=['inverse_ice_density = 1.0 / constants.rho_ice', ('internal_melt = _surface_melt & (', 1), ('energy_difference = sr(ice_specific - water_specific)', 1), ('ice_specific = (', 2), ('energy_difference = sr(ice_specific - water_specific)', 2), ('bottom_specific = (', 1), ('bottom_difference = bottom_specific - bottom_water', 1), 28])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1206', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1804', anchors='inverse_ice_density = 1.0 / constants.rho_ice')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:989', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1107', anchors='pts_old = content if content is not None else sr(thickness * enthalpy)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1655-1707', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2253-2305', anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-24', 53])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1672,1704', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2270,2302', anchors=[('_nemo_dh_snow_update: bool = True,', 2), '_nemo_dh_snow_update=_nemo_dh_snow_update,', 2])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:979-989', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1097-1107', anchors=['def _nemo_ice_enthalpy_remap(', 'pts_old = content if content is not None else sr(thickness * enthalpy)', 11])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1235', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1833', anchors='zfmdt = sr(-surface_energy / energy_difference)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1338', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1936', anchors='zfmdt = sr(-basal_energy / energy_difference)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:113', 'packages/ice/legoesm/ice/bitz_lipscomb.py:115', anchors='_NEW_ICE_THICKNESS = 0.05  # C1D card rn_hinew, c1d_omip_l3_namelist_ice_cfg:113')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1799', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2397', anchors='def si3_ice_thd_frazil(qlead, *, rn_hinew, ln_frazil) -> SI3Frazil:')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1787', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2385', anchors='def _ieee_lt_zero(value):')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1750', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2348', anchors='frazil = si3_ice_thd_frazil(forcing.qlead, rn_hinew=_NEW_ICE_THICKNESS, ln_frazil=False)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1751', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2349', anchors='cooling = _ieee_lt_zero(forcing.qlead)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1759', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2357', anchors='vnew / jnp.where(cooling, frazil.ht_i_new, 1.0),')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:557-647', 'packages/ice/legoesm/ice/bitz_lipscomb.py:579-648', anchors=['dz_i = jnp.maximum(h_i, _H_MIN) / nlay_i', 'eta_s = dt / (constants.rho_snow * dz_s[..., None] * constants.c_ice)', 70])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:587-611', 'packages/ice/legoesm/ice/bitz_lipscomb.py:600-611', anchors=['# P07 conductivity at the nlay_i+1 interfaces (:273-285), floor :293.', '# :326-341, zghe = 1 (ln_virtual_itd = F).', 12])
# SI3 round 16 resume 4: O11's citations into bitz_lipscomb.py after rebasing O9 onto O11.
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:41', 'packages/ice/legoesm/ice/bitz_lipscomb.py:42')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1677', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2275')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1735', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2333')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1822', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2420')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1848', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2446')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1880', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2478')

# SI3 round 12 adds the O5 trace API and exposes dynamics-on log expectations.
# Re-anchor every affected historical receipt citation before the full-map audit.
_reanchor('packages/ice/legoesm/ice/dynamics.py:239-264', 'packages/ice/legoesm/ice/dynamics.py:257-282')
_reanchor('packages/ice/legoesm/ice/dynamics.py:809-822', 'packages/ice/legoesm/ice/dynamics.py:827-840')
_reanchor('packages/ice/legoesm/ice/dynamics.py:1203', 'packages/ice/legoesm/ice/dynamics.py:1221')
_reanchor('packages/ice/legoesm/ice/dynamics.py:1255', 'packages/ice/legoesm/ice/dynamics.py:1273')
_reanchor('packages/ice/legoesm/ice/dynamics.py:825', 'packages/ice/legoesm/ice/dynamics.py:843')
_reanchor('packages/ice/legoesm/ice/dynamics.py:1898', 'packages/ice/legoesm/ice/dynamics.py:1938')
_reanchor('packages/ice/legoesm/ice/dynamics.py:1930', 'packages/ice/legoesm/ice/dynamics.py:1970')
_reanchor('packages/ice/legoesm/ice/dynamics.py:1989', 'packages/ice/legoesm/ice/dynamics.py:2029')
_reanchor('packages/ice/legoesm/ice/dynamics.py:2066', 'packages/ice/legoesm/ice/dynamics.py:2106')
_reanchor('packages/ice/legoesm/ice/dynamics.py:2087', 'packages/ice/legoesm/ice/dynamics.py:2127')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1755-1787', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1757-1789')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1767-1787', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1769-1789')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1955-1962', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1957-1964')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1804-1869', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1806-1871')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1856-1878', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1858-1880')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1287-1383', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1300-1396')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1757-1789', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1770-1802')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1769-1789', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1782-1802')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1957-1964', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1970-1977')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1806-1871', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1819-1884')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1858-1880', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o1_orca2_sas_gate.py:1871-1893')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:172', 'tests/ice/unit/test_si3_o5_strength_aevp.py:173')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:198', 'tests/ice/unit/test_si3_o5_strength_aevp.py:199')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:226', 'tests/ice/unit/test_si3_o5_strength_aevp.py:227')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:280', 'tests/ice/unit/test_si3_o5_strength_aevp.py:281')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:291', 'tests/ice/unit/test_si3_o5_strength_aevp.py:310')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:317', 'tests/ice/unit/test_si3_o5_strength_aevp.py:336')
_reanchor('tests/ice/unit/test_si3_o5_strength_aevp.py:400', 'tests/ice/unit/test_si3_o5_strength_aevp.py:419')
# The O4 Prather implementation was prepared before O2/O3/O5/O7 landed.  The
# composed rebase retains every rung and therefore moves all later O4/O2/O3
# anchors.  Re-anchor the historical keys only after the landed-rung chain
# above has finished so the mandatory full-map audit sees the composed tree.
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:55-66', 'packages/ice/legoesm/ice/si3_orca2_state.py:56-67')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2058-2068', 'packages/ice/legoesm/ice/si3_orca2_state.py:2078-2088')
_reanchor('packages/ice/legoesm/ice/__init__.py:227-251', 'packages/ice/legoesm/ice/__init__.py:266-290', anchors=['from legoesm.ice.config import build_si3_orca2_o2_config  # noqa: E402', (')', 24), 25])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:1219-1245', 'packages/ice/legoesm/ice/si3_orca2_state.py:1222-1248')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:1383', 'packages/ice/legoesm/ice/si3_orca2_state.py:1389')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:363', 'packages/ice/legoesm/ice/si3_orca2_state.py:367')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3223-3230', 'packages/ice/legoesm/ice/si3_orca2_state.py:3244-3251')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:1663-1665', 'packages/ice/legoesm/ice/si3_orca2_state.py:1684-1686')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3034-3116', 'packages/ice/legoesm/ice/si3_orca2_state.py:3055-3125', extent=71)
_reanchor('packages/ice/legoesm/ice/config.py:609-623', 'packages/ice/legoesm/ice/config.py:628-642')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3119-3350', 'packages/ice/legoesm/ice/si3_orca2_state.py:3140-3317', extent=178)
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3353-3783', 'packages/ice/legoesm/ice/si3_orca2_state.py:3374-3804')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3786-3939', 'packages/ice/legoesm/ice/si3_orca2_state.py:3807-3960')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:43', 'packages/ice/legoesm/ice/si3_orca2_state.py:44')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2279-3064', 'packages/ice/legoesm/ice/si3_orca2_state.py:4027-4812')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2279-2386', 'packages/ice/legoesm/ice/si3_orca2_state.py:4027-4134', anchors=['class SI3O2InitConfig(NamedTuple):', ('apply_halos: bool', 6), 108])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2279-2332', 'packages/ice/legoesm/ice/si3_orca2_state.py:4027-4080')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2335-2369', 'packages/ice/legoesm/ice/si3_orca2_state.py:4083-4117')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2373-2386', 'packages/ice/legoesm/ice/si3_orca2_state.py:4121-4134', anchors=['class SI3ORCA2O2Forcing(NamedTuple):', ('apply_halos: bool', 6), 14])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2390-2493', 'packages/ice/legoesm/ice/si3_orca2_state.py:4138-4241')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2458-2493', 'packages/ice/legoesm/ice/si3_orca2_state.py:4206-4241')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2405-2433', 'packages/ice/legoesm/ice/si3_orca2_state.py:4153-4181')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2436-2454', 'packages/ice/legoesm/ice/si3_orca2_state.py:4184-4202')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2780-3064', 'packages/ice/legoesm/ice/si3_orca2_state.py:4528-4812')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2780-2965', 'packages/ice/legoesm/ice/si3_orca2_state.py:4528-4713')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3006-3020', 'packages/ice/legoesm/ice/si3_orca2_state.py:4754-4768', anchors=[('zsm_i_ini=zsm_i_ini,', 2), ('snwice_mass=snwice_mass,', 5), 15])
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2794-2975', 'packages/ice/legoesm/ice/si3_orca2_state.py:4542-4723')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2546-2655', 'packages/ice/legoesm/ice/si3_orca2_state.py:4294-4403')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2573', 'packages/ice/legoesm/ice/si3_orca2_state.py:4321')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2662-2776', 'packages/ice/legoesm/ice/si3_orca2_state.py:4410-4524')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:2944-2951', 'packages/ice/legoesm/ice/si3_orca2_state.py:4692-4699')
_reanchor('packages/ice/legoesm/ice/si3_orca2_state.py:3056-3063', 'packages/ice/legoesm/ice/si3_orca2_state.py:4804-4811', anchors=['new_state = si3_o1_step(', (')', 1619), 8])
# O4 resume 6: the limiter tie fix, the shared-sum imports and the new tests move these O4 keys,
# and O11's lateral-melt config moves the O4 builder.
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:821-1122', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:834-1135', anchors=['def test_o4_a2_odd_even_sweep_order_and_cfl_subcycling_pins_and_plant() -> None:', (')', 406), 302])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2150-2241', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2214-2312', anchors=['def test_o4_d_jit_primal_zero_ulp_and_delayed_cfl_selection() -> None:', '_assert_same_bits(e_leaf, j_leaf, f"jit_delayed_icycle{want}")', 99])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1125-1468', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1138-1481', anchors=['def test_o4_a3_icemax_hbig_hsnow_zapneg_zapsmall_pato_diag_pins_and_plants() -> None:', 'build_si3_orca2_o4_config(None)  # type: ignore[arg-type]', 344])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1471-1671', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1535-1735', anchors=['def test_o4_a4_orca2_seam_and_north_fold_subcycle_lbc_lnk_pins_and_plants() -> None:', (')', 651), 201])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1713-1744', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:1777-1808', anchors=['def test_o4_b_lane3_parent_vs_current_zero_ulp(tmp_path: Path) -> None:', 'assert {"corr_int", "corr_ext"} <= differing', 32])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:536-614', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:549-627', anchors=['def _ref_nemo_lbc_lnk_orca2_t_point(', 'return arr_3d.reshape(orig_shape)', 79])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2244-2438', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2315-2509', anchors=['def _o4_ad_checks() -> None:', 'print(f"O4_AD_CHECKS_PASS fd_u_ratio={fd_u_ratio!r} fd_vi_ratio={fd_vi_ratio!r} rel_ad_diff={rel_ad_diff!r}")', 195])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2441-2466', 'tests/ocean/fidelity/test_nemo_si3_o4_orca2_prather.py:2512-2537', anchors=['def test_o4_d_ad_reverse_forward_and_central_fd() -> None:', 'print(result.stdout.strip().splitlines()[-1])', 26])
_reanchor('packages/ice/legoesm/ice/transport.py:858-1348', 'packages/ice/legoesm/ice/transport.py:860-1350', anchors=['def _si3_prather_x_substep(', ('return content, moments, area', 1), 491])
_reanchor('packages/ice/legoesm/ice/transport.py:1351-1826', 'packages/ice/legoesm/ice/transport.py:1353-1828', anchors=['def _si3_prather_y_substep(', ('return content, moments, area', 2), 476])
_reanchor('packages/ice/legoesm/ice/transport.py:2233-2275', 'packages/ice/legoesm/ice/transport.py:2235-2277', anchors=['def si3_adv_x(', ('return area, content_out, moments_out', 1), 43])
_reanchor('packages/ice/legoesm/ice/transport.py:2278-2318', 'packages/ice/legoesm/ice/transport.py:2280-2320', anchors=['def si3_adv_y(', ('return area, content_out, moments_out', 2), 41])
_reanchor('packages/ice/legoesm/ice/transport.py:1829-1834', 'packages/ice/legoesm/ice/transport.py:1831-1836', anchors=['def _si3_periodic_halo_xy', 'mode="wrap"', 6])
_reanchor('packages/ice/legoesm/ice/config.py:628-642', 'packages/ice/legoesm/ice/config.py:635-649', anchors=['def build_si3_orca2_o4_config(', 'return cfg', 15])

# O10 adds the salinity-4 constants, stage carriers and implementation in three
# deliberate blocks of bitz_lipscomb.py.  Re-anchor every historical citation
# mechanically, preserving comma/range structure; the full-map audit below
# still verifies every translated endpoint against its independently pinned
# source symbol and rejects any wrong shift.
def _o10_bitz_line(line: int) -> int:
    if line <= 115:
        return line
    if line < 1555:
        return line + 13
    if line < 1597:
        return line + 35
    if line < 2422:
        return line + 425
    return line + 429


def _o10_bitz_spec(spec: str) -> str:
    pieces = []
    for piece in spec.split(","):
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            pieces.append(f"{_o10_bitz_line(first)}-{_o10_bitz_line(last)}")
        else:
            pieces.append(str(_o10_bitz_line(int(piece))))
    return ",".join(pieces)


# The inserted O10 expressions also change occurrence numbers for otherwise
# anonymous closing-parenthesis anchors.  Supply the moved anchors during the
# one delete-and-reinsert reanchor operation; never overwrite a live map key.
_O10_MOVED_ANCHORS = {
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2091-2108,2143-2159,2218-2222': [('if _nemo_dh_snow_update:', 1), (')', 947), ('if _nemo_dh_snow_update:', 2), 'remaining_mass = evaporation * dt + snow_delta * constants.rho_snow', 'melted_snow.append(', (')', 999), 40],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2091-2108,2143-2159,2218-2222,2695,2727': [('if _nemo_dh_snow_update:', 1), (')', 947), ('if _nemo_dh_snow_update:', 2), 'remaining_mass = evaporation * dt + snow_delta * constants.rho_snow', 'melted_snow.append(', (')', 999), ('_nemo_dh_snow_update: bool = True,', 2), '_nemo_dh_snow_update=_nemo_dh_snow_update,', 42],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2096-2100': ['h_precip_raw = (', (')', 944), 5],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2101-2103': ['h_precip = jnp.where(', (')', 945), 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2147-2153': [('snow_delta = jnp.maximum(', 1), (')', 968), 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2218-2222': ['melted_snow.append(', (')', 999), 5],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2224-2226': ['surface_energy = jnp.maximum(', (')', 1002), 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2233-2425,2601-2652': [('if _nemo_dh_ice_content_accumulation:', 1), 'old_h = jnp.stack(zh_i_old_layers, axis=-1)', ('if _nemo_dh_ice_content_accumulation:', 2), (')', 1198), 245],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:249-311': ['def ice_temperature_from_enthalpy(', (')', 77), 63],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2631-2640': ['ze_i_old = jnp.stack(', (')', 1195), 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2643-2652': ['e_i = (', (')', 1198), 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2643-2659': ['e_i = (', (')', 1201), 17],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2644-2649': [('_nemo_ice_enthalpy_remap(', 2), (')', 1196), 6],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2653-2659': ['e_s = (', (')', 1201), 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2750-2755': ['S_layers = option2_salinity_profile(', (')', 1219), 6],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:369-381': ['def snow_enthalpy_from_temperature(', (')', 108), 13],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:41,1110-1168,2233-2425,2601-2652,2678-2730': ['frq_m: jax.Array | float = 1.0', 'def _nemo_ice_enthalpy_remap(', ('], axis=-1)', 12), ('if _nemo_dh_ice_content_accumulation:', 1), 'old_h = jnp.stack(zh_i_old_layers, axis=-1)', ('if _nemo_dh_ice_content_accumulation:', 2), (')', 1198), 'def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).', 358],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:41,2304-2308': ['frq_m: jax.Array | float = 1.0', 'frq_m = getattr(forcing, "frq_m", 1.0)', (')', 1046), 6],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:41,2304-2308,2488': ['frq_m: jax.Array | float = 1.0', 'frq_m = getattr(forcing, "frq_m", 1.0)', (')', 1046), '+ zdf.qtr_ice_bottom * getattr(forcing, "frq_m", 1.0)', 7],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:429-442': ['def p07_conductivity(', ('sr = nemo_source_round if _nemo_source_round else (lambda x: x)', 6), 14],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:793-891': ['def _si3_zdf_bl99_step(', (')', 342), 99],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:829-834': ['T_s_old = snow_temperature_from_enthalpy(', (')', 320), 6],
}

for _old in [key for key in CITATION_MAP
             if key.startswith("packages/ice/legoesm/ice/bitz_lipscomb.py:")]:
    _prefix, _spec = _old.rsplit(":", 1)
    _new = f"{_prefix}:{_o10_bitz_spec(_spec)}"
    if _new != _old:
        _reanchor(_old, _new, anchors=_O10_MOVED_ANCHORS.get(_new))

_missing_o10_anchors = sorted(set(_O10_MOVED_ANCHORS) - set(CITATION_MAP))
if _missing_o10_anchors:
    raise RuntimeError(f"unapplied O10 citation anchors: {_missing_o10_anchors}")


# --- SI3 round 19 (O10) salinity-4 model preparation ---
CITATION_MAP.update({
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-205': [('CALL ice_thd_temp                     ! --- Temperature update --- !', 1), ('CALL ice_thd_temp                     ! --- Temperature update --- !', 2), 5],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311-683': ['CASE( 4 )        ! Gravity Drainage and Flushing  !', ('ENDIF', 19), 373],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:311': 'CASE( 4 )        ! Gravity Drainage and Flushing  !',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:317-353': [('DO jk = 1, nlay_i', 1), ('zs_br(nlay_i+1) = sss_1d(ji)', 1), 37],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:345': ('zs_br(jk) = -18.7_wp * zt1 - 0.519_wp * zt2 - 0.00535_wp * zt3', 1),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:326-560': [('IF( ln_drainage ) THEN', 2), ('ENDIF', 12), 235],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:377-390': [('ELSEIF( np_perm_eff == 2 ) THEN ! Harmonic Mean', 1), ('END DO', 5), 14],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:397-398': [('zRa(jk) = zcp_br * grav * zbeta * MAX( 0._wp, zs_br(jk) - sss_1d(ji)) * zperm_eff(jk) &', 1), ('&                            *  h_i_1d(ji) * ( 1._wp - z_mid(jk) ) * ( z1_cnd_br * z1_visc )', 1), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:403-422': [('IF ( nn_sal_scheme == 1 ) THEN     ! *** RJW 2014 ***', 1), ('END DO', 8), 20],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:447-548': ['ndrainage = MIN( nn_drainage, CEILING(zcfl_test*2._wp) )', ('END DO', 21), 102],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:541-544': [('sz_i_1d(ji,jk) = sz_i_1d(ji,jk) + zds(jk)', 1), ('sfx_bri_1d(ji) = sfx_bri_1d(ji) - rhoi * a_i_1d(ji) * h_i_1d(ji) * r1_nlay_i * zds(jk) * r1_Dt_ice ! r1_Dt_ice is ok', 1), 4],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:562-669': [('! ========', 10), ('ENDIF', 18), 108],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:601-602': [('zw_br(jk) = -rn_flushrate * ( dh_i_sum(ji)*rhoi + dh_s_sum(ji)*rhos )  &', 1), ('&                 / ( rhow * ( 1._wp + 0.8e-3_wp * zs_br(jk) ) ) * r1_Dt_ice ! r1_Dt_ice is ok', 1), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:607-608': ['nflushing = MIN( nn_flushing, CEILING(zcfl_test*2._wp) )', 'zdt   = rDt_ice / REAL( MAX( 1, nflushing ) )', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:616-622': ['zv_brmin = 1.e+20_wp', 'IF( zv_brmin >= rn_vbrc ) THEN            ! Flushing if brine volume fraction exceeds a certain treshold', 7],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:637-651': [('zcfl_max = MAX( zcfl_max, ABS(zcfl) )', 2), ('sfx_bri_1d(ji) = sfx_bri_1d(ji) - rhoi * a_i_1d(ji) * h_i_1d(ji) * r1_nlay_i * zds(jk) * r1_Dt_ice ! r1_Dt_ice is ok', 2), 15],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:671-683': [('! --- salinity must stay inbounds --- !', 2), ('ENDIF', 19), 13],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:677-679': ['sz_i_1d(ji,jk) = MIN( MAX( rn_simin, sz_i_1d(ji,jk) ), rn_sinew*sss_1d(ji) )', 'sfx_res_1d(ji) = sfx_res_1d(ji) - rhoi * a_i_1d(ji) * h_i_1d(ji) * r1_nlay_i * ( sz_i_1d(ji,jk) - zsal ) * r1_Dt_ice', 3],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:277-292': ['zds(1) = - MAX( s_i_1d(ji) - rn_sal_fl , 0._wp ) * z1_time_fl', 's_i_1d(ji) = MIN( MAX( rn_simin, s_i_1d(ji) ), rn_sinew*sss_1d(ji) )', 16],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:424-435': [('ELSEIF ( nn_sal_scheme == 2 ) THEN    ! *** GN 2013 ***', 1), ('END DO', 10), 12],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:509-521': [('ELSEIF ( nn_sal_scheme == 2 ) THEN    ! *** GN 2013 ***', 2), ('END DO', 19), 13],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:455-474': ['IF( iter > 1 ) THEN', ('zperm_eff(jk) = REAL( nlay_i-jk+1, wp ) / ztmp', 2), 20],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:593-644': [('WHERE( (t_i_1d(ji,:)-rt0) <  - epsi06 ) ; zmsk(:) = 1._wp', 2), 'zds(jk) = MAX( MIN( 0._wp, zds(jk) ), -sz_i_1d(ji,jk) + MIN( rn_simin, rn_sinew*sss_1d(ji) ) )', 52],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:113-121': [('ELSEIF ( nn_icesal == 4 ) THEN', 1), ('END DO   ;   END DO   ;   END DO', 1), 9],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:952-960': [('DO jk = 1, nlay_i             ! Sea ice energy of melting', 1), ('END DO', 55), 9],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:1040-1046': [('! to ensure that total heat/salt content is strictly conserved, set:', 1), ('END DO', 63), 7],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icevar.f90:1045': 'pts_i(jk1) = MAX( 0._wp, zts_cum1(jk1) - zts_cum1(jk1-1) ) / MAX( zhnew, epsi20 ) ! max for roundoff error',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:279-291': [('DO jk = 1, nlay_i', 1), ('END DO', 5), 13],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90:569-576': [('z1_h_i = 1._wp / MAX( h_i_1d(ji) * r1_nlay_i, epsi10 )', 1), ('IF( h_i_1d(ji) >= rn_sal_himin .AND. zhmelt < 0._wp ) THEN     ! Flushing if  surface melting', 1), 8],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1634-1643': ['def _sal4_vc19_brine(t_i, T0):', 'return nemo_source_round(nemo_source_round(linear - quadratic) - cubic)', 10],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1646-1672': ['def _sal4_effective_permeability(zv_br, *, initial):', 'return jnp.stack(out, axis=-1)', 27],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1653-1672': [('for jk in range(nlay):', 1), 'return jnp.stack(out, axis=-1)', 20],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1675-1729': ['def _sal4_rjw_velocity(sz_i, zs_br, zmsk, sss, h_i, z_mid, z_edge, z1_h_i,', 'return jnp.stack(velocity, axis=-1)', 55],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1682-1689': ['divisor = (', 'zperm_eff = _sal4_effective_permeability(zv_br, initial=initial)', 8],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1687-1689': ['volume_numerator = nemo_source_round(zmsk * sz_i)', 'zperm_eff = _sal4_effective_permeability(zv_br, initial=initial)', 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1691-1693': ['z1_cp_br = nemo_source_round(_div_const(one, _SAL4_CP_BRINE))', 'z1_visc = nemo_source_round(_div_const(one, _SAL4_VISCOSITY_BRINE))', 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1740-1843': ['def _si3_ice_thd_sal4_core(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i,', ('(sz_i, sfx_bri, zero, zero),', 1), 104],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1846-1917': ['# Flushing uses upward meltwater flow and the same positive-ocean salt flux.', ('(sz_i, sfx_bri, zero, zero),', 2), 72],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1874-1882': ['def flushing_body(iteration, carry):', 'active = flush_cell & (iteration < nflush) & (zv_brmin >= rn_vbrc)', 9],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1891': 'sal_min_next = _minsd(sal_min, raw_salinity)',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1895-1897': [('# -O3 line-644 roles:', 1), '_maxsd(nonpositive, floor_delta)', 3],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1921-1937': ['zsal = sz_i[..., jk]', 'cfl_drain, cfl_flush, smin_drain, smin_flush,', 17],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1949-1986': ['def si3_ice_thd_sal(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i, dh_i_sum,', 'raise ValueError("SI3 O10 requires the production 10 ice layers")', 38],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1997-2020': ['def si3_ice_thd_o10_category_stages(*, entry: dict, fluxes: SI3DHFluxes,', 'return SI3O10CategoryStages(o9, sal, t_i)', 24],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o10_minmax_audit.py:12-100': ['BUILD = common.BUILD', 'sys.exit(main())', 89],
    'tests/ocean/fidelity/test_nemo_si3_o10_minmax_audit.py:21-169': ['SITE_SIGNATURES = {', 'assert (max_floor_dst, max_min_dst) == (2, 2)', 149],
    'tests/ice/unit/test_si3_o10_salinity.py:442-452': ['def test_salinity4_is_bit_exact_to_independent_scalar_under_jit_and_eager():', ('np.testing.assert_array_equal(_bits(got), _bits(want))', 1), 11],
    'tests/ice/unit/test_si3_o10_salinity.py:455-512': [('@pytest.mark.parametrize(', 1), 'for got, want in zip(stripped, eager)', 58],
    'tests/ice/unit/test_si3_o10_salinity.py:516-548': ['def test_zero_permeability_guard_changes_no_forward_bits_on_fixture_and_fuzz():', 'np.testing.assert_array_equal(_bits(new), _bits(old))', 33],
    'tests/ice/unit/test_si3_o10_salinity.py:551-559': ['def test_salinity4_inexact_inputs_and_output_digest_are_pinned():', 'assert digest.hexdigest() == "1d77233ee682c9e47b266ed263db0b4be98a13d4dcf329d6a6fe80a9dd27a674"', 9],
    'tests/ice/unit/test_si3_o10_salinity.py:562-599': ['def test_salinity4_branch_and_subcycle_coverage_is_real():', '"flush_iterations_after_first": 9,', 38],
    'tests/ice/unit/test_si3_o10_salinity.py:603-668': ['def _reachable_flush_role_fixture():', 'assert changed == [0, 1, 0, 0, 0, 0, 0, 0, 0], site', 66],
    'tests/ice/unit/test_si3_o10_salinity.py:671-704': ['def test_masked_signed_zero_paths_are_checked_through_the_salinity_model():', 'np.testing.assert_array_equal(values[0], values[1])', 34],
    'tests/ice/unit/test_si3_o10_salinity.py:707-737': ['def test_temp1_two_million_state_scan_excludes_t0_for_positive_salinity():', 'np.testing.assert_array_equal(boundary, NEMO_SI3_CONSTANTS_CONFIG.T0)', 31],
    'tests/ice/unit/test_si3_o10_salinity.py:740-824': ['def test_five_category_o9_sal_temp2_composition_matches_scalar_oracle():', 'assert (thin_positive, zero_salt_residue, excluded_from_flush) == (5, 3, 8)', 85],
    'tests/ice/unit/test_si3_o10_salinity.py:827-845': ['def test_salinity4_gradient_is_finite_nonzero_and_matches_central_difference():', 'np.testing.assert_allclose(analytic, finite, rtol=1.0e-5, atol=1.0e-7)', 19],
    'tests/ice/unit/test_si3_o10_salinity.py:848-864': ['def test_melted_h_i_zero_column_has_finite_temperature_and_salinity_gradients():', 'np.testing.assert_array_equal(grad_s[-1], jnp.ones(10, dtype=jnp.float64))', 17],
    'tests/ice/unit/test_si3_o10_salinity.py:867-895': [('@pytest.mark.parametrize(', 3), ('bl.si3_ice_thd_sal(**kwargs)', 2), 29],
    'tests/ice/unit/test_si3_o10_salinity.py:898-964': ['def test_o10_production_composer_calls_salinity_then_second_temperature(monkeypatch):', 'np.testing.assert_array_equal(result.t_i, sal_token + 1.0)', 67],
    'scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py:10272-10309': [('# The inserted O10 expressions also change occurrence numbers for otherwise', 1), ('raise RuntimeError(f"unapplied O10 citation anchors: {_missing_o10_anchors}")', 1), 38],
    'tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py:183-234': ['def test_no_citation_map_key_is_written_twice():', 'assert len(written) == len(gate.CITATION_MAP)', 52],
    'docs/ice/fidelity/si3_r19_o10_preregister.md:34-68': ['## 2. Registered master-record projection (GIVEN-NEMO-ENTRY only)', 'compiled-source line before another change.', 35],
})

# --- SI3 round 20 (O12) melt-pond model preparation ---
FILES.update({
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/par_ice.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/par_ice.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90",
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90': NEMO / "cfgs/ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90",
    'packages/ice/legoesm/ice/ponds.py': REPO / 'packages/ice/legoesm/ice/ponds.py',
    'packages/ice/legoesm/ice/si3_thd_pnd.py': REPO / 'packages/ice/legoesm/ice/si3_thd_pnd.py',
    'packages/ocean/legoesm/ocean/restart.py': REPO / 'packages/ocean/legoesm/ocean/restart.py',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o12_minmax_audit.py': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o12_minmax_audit.py',
    'tests/ocean/fidelity/test_nemo_si3_o12_minmax_audit.py': REPO / 'tests/ocean/fidelity/test_nemo_si3_o12_minmax_audit.py',
    'tests/ice/unit/test_si3_o12_pnd.py': REPO / 'tests/ice/unit/test_si3_o12_pnd.py',
    'docs/ice/fidelity/si3_r20_o12_preregister.md': REPO / 'docs/ice/fidelity/si3_r20_o12_preregister.md',
    'docs/ice/fidelity/si3_r20_o12_receipt.md': REPO / 'docs/ice/fidelity/si3_r20_o12_receipt.md',
})

CITATION_MAP.update({
    'docs/ice/fidelity/si3_ladder_design.md:232': '| O12 | Exact-input pond stage replay with production pond selectors. | Add **level ponds, rain, lids and pond albedo state changes**. | Pond/lid growth, drainage/flush/freeze and conservation; 10–20 GiB. |',
    'docs/ice/fidelity/si3_r20_o12_preregister.md:1-77': ['# SI3 Rung O12 (`ice_thd_pnd` / `pnd_LEV` / `ice_perm_eff`) — Model-Preparation Preregistration (`si3_r20`)', '- **REFUTE**: Any hash or bit change in any existing caller.', 77],
    'docs/ice/fidelity/si3_ladder_design.md:205-209': ['O8-O16 are admitted from **one** instrumented 100-step ORCA2_SAS_ICE run with', 'one record, not the cost of a different deck or another NEMO run. In', 5],
    'scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_ice_cfg:132': 'ln_pnd           = .false.    ! ORCA1 ice cfg:151',
    'packages/ice/legoesm/ice/si3_surface.py:300-347': ['def si3_ice_alb(*, t_su, h_i, h_s, a_ip_eff, h_ip, h_il, cloud_fra, smask0, config):', 'return sr(sr(sr(1.0 - cloud) * zalb_cs) + sr(cloud * zalb_os))                              # :244', 48],
    'packages/ice/legoesm/ice/si3_orca2_state.py:941-965': ['if kn == 1 or cfg.ln_pnd:', (')', 330), 25],
    'packages/ice/legoesm/ice/si3_orca2_state.py:3450-3500': ['aip_ge_eps20 = state.a_ip >= epsi20', 'a_ip=state.a_ip,', 51],
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:45': 'rn_himin         =   0.1           !  minimum ice thickness (m) allowed',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:157': 'ln_icedH         = .true.          !  activate ice thickness change from growing/melting (T) or not (F)',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:241-257': ['ln_pnd            = .true.         !  activate melt ponds or not', 'rn_pnd_hl_max  =  0.015      !    pond lid thickness above which ponds disappear from albedo calculation', 17],
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:241': 'ln_pnd            = .true.         !  activate melt ponds or not',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:243': 'rn_apnd_min    =  0.15          !  minimum meltwater fraction contributing to pond growth (TOPO and LEV)',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:244': 'rn_apnd_max    =  0.80          !  maximum meltwater fraction contributing to pond growth (TOPO and LEV)',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:246': 'ln_pnd_TOPO       = .false.     !  topographic melt ponds',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:247': 'ln_pnd_rain    = .true.      !    rain added to melt ponds',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:248': 'ln_pnd_LEV        = .true.      !  level ice melt ponds',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:249': 'rn_pnd_flush   =  0.1        !    pond flushing efficiency (tuning parameter)',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:250': 'ln_pnd_lids    = .true.      !    frozen lids on top of the ponds',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:251': 'ln_pnd_CST        = .false.     !  constant  melt ponds',
    'tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:255': 'ln_pnd_alb        = .true.      !  effect of melt ponds on ice albedo',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1424-1487': [('SUBROUTINE ice_thd_pnd_init', 1), 'END SUBROUTINE ice_thd_pnd_init', 64],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:52,1474': ['INTEGER, PARAMETER ::   np_pndLEV  = 2   ! Level ice pond scheme', 'IF( ln_pnd_LEV  ) THEN   ;   ioptio = ioptio + 1   ;   nice_pnd = np_pndLEV    ;   ENDIF', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:176-223': ['DO jl = 1, jpl', '&                    CALL ice_thd_pnd( kt )                ! --- Melt ponds --- !', 48],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:212': 'CALL ice_thd_1d2d( jl, 2 )            ! --- Change units of e_i, e_s from J/m3 to J/m2 --- !',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:484-588': ['CASE( 2 )            !==  from 1D to 2D  ==!', 'END SUBROUTINE ice_thd_1d2d', 105],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:488-501': ['e_i_1d(1:npti,jk) = e_i_1d(1:npti,jk) * h_i_1d(1:npti) * a_i_1d(1:npti) * r1_nlay_i', ('ENDDO', 1), 14],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:222-223': ['IF ( ln_pnd .AND. ln_icedH ) &', '&                    CALL ice_thd_pnd( kt )                ! --- Melt ponds --- !', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1441': '&                          ln_pnd_lids, ln_pnd_rain, ln_pnd_alb,  &',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:702': 'IF( ln_pnd_rain )    zv_mlt = zv_mlt + ( tprecip(ji,jj) - sprecip(ji,jj) ) * z1_rhow * a_i(ji,jj,jl) * rDt_ice',
    'packages/ice/legoesm/ice/si3_thd_pnd.py:575-763': ['def si3_ice_thd_pnd(', (')', 315), 189],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:203': 'at_i(:,:) = SUM( a_i, dim=3 )',
    'packages/ice/legoesm/ice/si3_thd_pnd.py:683': 'at_i = fortran_sum0(arr["a_i"])',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:205-217': [('DO jl = 1, jpl', 1), ('END DO', 2), 13],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:208': 'wfx_pnd  (ji,jj)    = wfx_pnd(ji,jj) + ( v_ip(ji,jj,jl) + v_il(ji,jj,jl) ) * rhow * r1_Dt_ice',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:209-214': ['a_ip     (ji,jj,jl) = 0._wp', 'a_ip_frac(ji,jj,jl) = 0._wp', 6],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:699-717': [('for jl in range(jpl):', 2), 'a_ip_frac_int = jnp.stack(a_ip_frac_clean, axis=-1)', 19],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:222-240': ['npti = 0   ;   nptidx(:) = 0', ('ENDIF', 6), 19],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:720-763': ['has_ice = at_int >= epsi10', (')', 315), 44],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:329-582': [('SUBROUTINE pnd_LEV', 1), 'END SUBROUTINE pnd_LEV', 254],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:235-572': ['def _pnd_lev_single_category(', (')', 242), 338],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:385-387': ['REAL(wp), PARAMETER ::   zaspect =  0.8_wp      ! pond aspect ratio', 'REAL(wp), PARAMETER ::   zvisc   =  1.79e-3_wp  ! water viscosity', 3],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/par_ice.f90:134-135': ['REAL(wp), PUBLIC, PARAMETER ::   epsi06 = 1.e-06_wp  !: small number', 'REAL(wp), PUBLIC, PARAMETER ::   epsi10 = 1.e-10_wp  !: small number', 2],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:32-46': ['_PND_ZASPECT = 0.8  # zaspect, pond aspect ratio (icethd_pnd.f90:385)', '_EPSI10 = 1.0e-10  # epsi10 (par_ice.f90:52)', 15],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:401-403': [('z1_rhow   = 1._wp / rhow', 1), 'z1_Tp     = 1._wp / zTp', 3],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:493-495': ['z1_rhow = sr(one / scl["rho_w"])', 'z1_Tp = sr(one / sr(jnp.float64(_PND_ZTP_C)))', 3],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:412-435': [('ENDIF', 9), ('zdv_pnd = ( h_ip_1d(ii) + h_il_1d(ii) ) * a_ip_1d(ii)', 3), 24],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:286,530-563': ['zdv_pnd_old = sr(sr(h_ip + h_il) * a_ip)', ('for jl in range(jpl):', 1), 'v_il_list.append(cat_out.v_il)', 35],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:437-442': ['IF( h_i_1d(ii) < rn_himin .OR. a_i_1d(ii) < 0.01_wp ) THEN   ! Case ice thickness < rn_himin or tiny ice fraction !', ('h_il_1d(ii) = 0._wp', 1), 6],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:289,406-408': ['thin_or_tiny = (h_i < rn_himin) | (a_i < tiny_ai)', 'a_ip_out = jnp.where(thin_or_tiny, zero, a_ip_thick)', 'h_il_out = jnp.where(thin_or_tiny, zero, h_il_thick)', 4],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:446-478': ['v_ip_1d(ii) = h_ip_1d(ii) * a_ip_1d(ii)   ! retrieve volume from thickness', 'IF( ln_pnd_lids )   v_il_1d(ii) = MAX( 0._wp, v_il_1d(ii) - zdv_mlt ) ! must be bounded by 0', 33],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:464-465': ['zv_ip_max = zfr_mlt * zfr_mlt * a_i_1d(ii) * zaspect', ('zdv_mlt   = MAX( 0._wp, MIN( zdv_mlt, zv_ip_max - v_ip_1d(ii) ) )', 1), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:471-472': ['zv_ip_max = z1_aspect * a_i_1d(ii) * 0.25 * h_i_1d(ii) * h_i_1d(ii)', ('zdv_mlt   = MAX( 0._wp, MIN( zdv_mlt, zv_ip_max - v_ip_1d(ii) ) )', 2), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:475': 'v_ip_1d(ii) = v_ip_1d(ii) + zdv_mlt',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:478': 'IF( ln_pnd_lids )   v_il_1d(ii) = MAX( 0._wp, v_il_1d(ii) - zdv_mlt ) ! must be bounded by 0',
    'packages/ice/legoesm/ice/si3_thd_pnd.py:294-323': ['v_ip = sr(h_ip * a_ip)', 'v_il = _maxsd(sr(v_il - zdv_mlt), zero)', 30],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:484-503': ['zdT = MAX( zTp+rt0 - t_su_1d(ii), 0._wp )', ('ENDIF', 10), 20],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:490-497': ['zdv_frz = - 0.5_wp * MAX( 0._wp, -v_il_1d(ii) + & ! Flocco 2010 (eq. 5) solved implicitly as aH**2 + bH + c = 0', ('v_ip_1d(ii) = MAX( 0._wp, v_ip_1d(ii) + zdv_frz )', 1), 8],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:500-502': ['zdv_frz = v_ip_1d(ii) * ( EXP( 0.01_wp * zdT * z1_Tp ) - 1._wp )  ! Holland 2012 (eq. 6)', ('v_ip_1d(ii) = MAX( 0._wp, v_ip_1d(ii) + zdv_frz )', 2), 3],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:326-354': ['zdT = _maxsd(sr(sr(zTp + rt0) - t_su), zero)', 'v_ip = _maxsd(sr(v_ip + zdv_frz), zero)', 29],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:508-528': [('a_ip_1d(ii) = MIN( a_i_1d(ii), SQRT( v_ip_1d(ii) * z1_aspect * a_i_1d(ii) ) ) ! make sure a_ip < a_i', 1), ('h_ip_1d(ii) = zaspect * a_ip_1d(ii) / a_i_1d(ii)', 2), 21],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:508-509': [('a_ip_1d(ii) = MIN( a_i_1d(ii), SQRT( v_ip_1d(ii) * z1_aspect * a_i_1d(ii) ) ) ! make sure a_ip < a_i', 1), ('h_ip_1d(ii) = zaspect * a_ip_1d(ii) / a_i_1d(ii)', 1), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:516': 'zhp = ( h_i_1d(ii) * ( rho0 - rhoi ) + h_ip_1d(ii) * ( rho0 - rhow * a_ip_1d(ii) / a_i_1d(ii) ) ) * r1_rho0',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/phycst.f90:52-53': ['REAL(wp), PUBLIC ::   rho0                         !: volumic mass of reference     [kg/m3]', 'REAL(wp), PUBLIC ::   r1_rho0                      !: = 1. / rho0                   [m3/kg]', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:519': 'zperm = ice_perm_eff( t_i_1d(ii,:), sz_i_1d(ii,:) )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:522-524': ['zdv_flush   = -zperm * rho0 * grav * zhp * rDt_ice / (zvisc * h_i_1d(ii)) * a_ip_1d(ii) * rn_pnd_flush ! rn_pnd_flush comes from Hunke et al. (2012)', 'v_ip_1d(ii) = v_ip_1d(ii) + zdv_flush', 3],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:527-528': [('a_ip_1d(ii) = MIN( a_i_1d(ii), SQRT( v_ip_1d(ii) * z1_aspect * a_i_1d(ii) ) ) ! make sure a_ip < a_i', 2), ('h_ip_1d(ii) = zaspect * a_ip_1d(ii) / a_i_1d(ii)', 2), 2],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:358-387': ['rad_1 = sr(sr(v_ip * z1_aspect) * a_i)', 'h_ip_2 = sr(sr(zaspect * a_ip_2) / a_i_safe)', 30],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:531-559': [('IF( ln_pnd_lids ) THEN', 2), ('!', 287), 29],  # const-ok: occurrence count of '!' in icethd_pnd.f90
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:533-535': ['IF( a_ip_1d(ii) > 0.01_wp ) THEN   ;   h_il_1d(ii) = v_il_1d(ii) / a_ip_1d(ii)', ('ENDIF', 11), 3],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:537-541': ['IF ( h_il_1d(ii) > h_ip_1d(ii) * 10._wp ) THEN', ('ENDIF', 12), 5],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:554-555': [('v_ip_1d(ii) = h_ip_1d(ii) * a_ip_1d(ii)', 3), ('v_il_1d(ii) = h_il_1d(ii) * a_ip_1d(ii)', 3), 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:557-558': [('zdv_pnd = ( h_ip_1d(ii) + h_il_1d(ii) ) * a_ip_1d(ii) - zdv_pnd', 2), ('wfx_pnd_1d(ii) = wfx_pnd_1d(ii) - zdv_pnd * rhow * r1_Dt_ice', 2), 2],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:390-414': [('if ln_pnd_lids:', 3), 'wfx_pnd_out = sr(wfx_pnd - sr(sr(zdv_pnd * rhow) * r1_Dt_ice))', 25],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1372-1422': ['FUNCTION ice_perm_eff( zti, zsi )', 'END FUNCTION ice_perm_eff', 51],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:168-232': ['def si3_ice_perm_eff(', 'return jnp.where(valid, sr(nlay_f / ztmp_safe), zero)', 65],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1389-1401': [('DO jk = 1, nlay_i', 2), ('ENDDO', 3), 13],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1393': 'zs_br = -18.7_wp * zt1 - 0.519_wp * zt2 - 0.00535_wp * zt3',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1398': 'IF( (zti(jk)-rt0) <  - epsi06 ) THEN   ;   zv_br(jk) = zsi(jk) / MAX( zs_br, epsi10 )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1399': 'ELSE                                   ;   zv_br(jk) = 0._wp',
    'packages/ice/legoesm/ice/si3_thd_pnd.py:200-211': ['zt1 = sr(t_i - rt0)', 'zv_br = jnp.where(cold, sr(sz_i / denom), jnp.zeros_like(sz_i))', 12],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1408-1420': ['ELSEIF( np_perm_eff == 2 ) THEN ! Harmonic Mean', ('END IF', 18), 13],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1412': 'IF( zv_br(jk) < epsi06 ) THEN',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1416': 'zperm = 3.e-8_wp * zv_br(jk)*zv_br(jk)*zv_br(jk)',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1417': 'ztmp = ztmp + 1._wp / zperm',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1419': 'IF( ztmp > 0._wp )   ice_perm_eff = REAL( nlay_i, wp ) / ztmp',
    'packages/ice/legoesm/ice/si3_thd_pnd.py:214-232': ['cell_shape = t_i.shape[:-1]', 'return jnp.where(valid, sr(nlay_f / ztmp_safe), zero)', 19],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:766-954': ['def si3_ice_thd_o12_stages(', 'return SI3O12Stages(categories=tuple(cat_outputs), pnd=pnd_out)', 189],
    'packages/ice/legoesm/ice/__init__.py:292-316': ['from legoesm.ice.config import SI3PondConfig  # noqa: E402', (')', 27), 25],
    'packages/ice/legoesm/ice/config.py:267-288,311-315,652-727': ['class MeltPondConfig(NamedTuple):', 'namthd_pnd: SI3PondConfig | None = None', 'def validate_si3_thermo_config(config: "SeaIceConfig") -> None:', 'return _validate_si3_orca2_o1_config(config) or _validate_si3_orca2_o4_config(config) or _validate_si3_lev_pond_config(config)', 'class SI3PondConfig(NamedTuple):', (')', 209), 103],
    'packages/ice/legoesm/ice/config.py:311-315,662-727': ['def validate_si3_thermo_config(config: "SeaIceConfig") -> None:', 'return _validate_si3_orca2_o1_config(config) or _validate_si3_orca2_o4_config(config) or _validate_si3_lev_pond_config(config)', 'def _validate_si3_lev_pond_config(config: "SeaIceConfig") -> None:', (')', 209), 71],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:255-261,308-315,358-365,391-397,421-425,435,1004-1007,1146-1147': [('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 2), ('ENDIF', 8), ('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 3), ('ENDIF', 10), ('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 4), ('ENDIF', 12), ('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 5), ('ENDIF', 15), 'CALL Hbig_pra( ihls, jl, rDt_ice, zhi_max, zhs_max, zhip_max, zsi_max, zes_max, zei_max, zszi_max, &', 'CALL Hsnow_pra( ihls, jl, rDt_ice, pv_i, pv_s, pa_i, pa_ip, pe_s )', 'CALL ice_var_zapneg( ihls, rDt_ice, pv_i, pv_s, psv_i, poa_i, pa_i, pa_ip, pv_ip, pv_il, pe_s, pe_i, pszv_i )', 'IF( ln_pnd_LEV .OR. ln_pnd_TOPO .AND. pv_ip(ji,jj,jcat) > 0._wp ) THEN', 'pa_ip(ji,jj,jcat) = pv_ip(ji,jj,jcat) / phip_max(ji,jj)', '!-- correct pond concentration to avoid a_ip > a_i -- !', 'pa_ip(ji,jj,jcat) = MIN( pa_ip(ji,jj,jcat), pa_i(ji,jj,jcat) )', 42],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icecor.f90:84-95': ['IF( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', ('END DO', 1), 12],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/iceitd.f90:456-458,514-528,589-591': ['CALL tab_3d_2d( npti, nptidx(1:npti), a_ip_2d (1:npti,:)  , a_ip  )', 'CALL tab_3d_2d( npti, nptidx(1:npti), v_il_2d (1:npti,:)  , v_il  )', 'IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', ('ENDIF', 19), 'CALL tab_2d_3d( npti, nptidx(1:npti), a_ip_2d (1:npti,:)  , a_ip  )', 'CALL tab_2d_3d( npti, nptidx(1:npti), v_il_2d (1:npti,:)  , v_il  )', 21],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icedyn_rdgrft.f90:764-773,802-809,824-828,919-925': [('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 1), ('ENDIF', 24), ('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 2), ('ENDIF', 27), ('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 3), ('ENDIF', 29), ('IF ( ln_pnd_LEV .OR. ln_pnd_TOPO ) THEN', 4), ('ENDIF', 36), 30],
    'packages/ice/legoesm/ice/sea_ice.py:1717-1795,2992-3000,3228-3387': ['def _cap_multicat_concentration(conc, h, h_snow=None, pond_depth=None, *, h_il=None):', 'return area, depth', ('if config.ponds.enabled:', 3), 'jnp.maximum(pond_thick, 0.0) * conc, conc, config)', 'if is_multicat and config.itd_remap != "none":', '+ ridge_result["pond_to_ocean"]', 248],
    'packages/ice/legoesm/ice/itd.py:263-336,635-636': ['def linear_remap(', 'a_deficit = a_new * deficit_frac', 'pond_factor = jnp.where(V_new_cat > 1e-30, V_pond_new / V_safe, 0.0)', 'V_pond_remap = jnp.sum(int_vol * pond_factor[None, :], axis=1)', 76],
    'packages/ice/legoesm/ice/ridging.py:422-423,459-559': ['# Donor pond water is drained (not redistributed into the ridge).', 'Vpond_new = jnp.maximum(V_pond_cat - dVpond_per_cat, 0.0)', 'def apply_ridging(', ('}', 1), 103],
    'packages/ice/legoesm/ice/state.py:52-76,89-177,213-258,315-335': ['class DynamicSeaIceState(NamedTuple):', 'h_il: Field | None = None  # Mean melt pond lid thickness [m]', 'def init_dynamic_ice_state(', (')', 55), 'def slab_to_dynamic(', (')', 82), 'return state._replace(', (')', 114), 181],
    'packages/ocean/legoesm/ocean/restart.py:322-329,382-392,859-861': ['# --- sea ice (SeaIceState + the 13-field DynamicSeaIceState) -----------', '"h_il": _SLOT_PROGNOSTIC,', 'def _iter_state_fields(state) -> list[str]:', ('', 392), 'declared = getattr(state, "_fields", None) or _iter_state_fields(state)', 'if n not in _SLOT_POLICY)', 22],
    'packages/ice/legoesm/ice/ponds.py:34-212': ['def step_ponds(', (')', 55), 179],
    'packages/ice/legoesm/ice/sea_ice.py:427-440,482,1745-1779,2396-2421,2477-2543,2829-2877,3014-3094,3228-3349,3394-3396,3513-3531': ['if not (config.ponds.enabled and config.ponds.scheme == "si3_lev") and config.itd_remap not in ("simple", "lipscomb2001"):', 'if config.n_categories > 1 and not (config.ponds.enabled and config.ponds.scheme == "si3_lev"):', 'state = slab_to_dynamic(state, pond_scheme=config.ponds.scheme if config.ponds.enabled else "cesm")', ('if h_il is not None:', 1), (')', 575), ('if not config.ponds.enabled and config.ponds.scheme != "cesm":', 3), 'h_il_dil = h_il * dilute if _use_si3_lev else None', 'if config.ponds.scheme == "si3_lev":', ('h_il_new = h_il_dil', 1), ('if not config.ponds.enabled and config.ponds.scheme != "cesm":', 4), ('conc, h, h_snow, pond_depth, h_il=h_il)', 1), ('if is_multicat and config.itd_remap != "none" and not exact_si3_itd:', 1), 'h_il = jnp.stack(hil_list, axis=-1) if h_il is not None else None', 'if is_multicat and config.itd_remap != "none":', ('conc, h, h_snow, pond_depth, h_il=h_il)', 4), ('if is_multicat and config.itd_remap != "none" and not exact_si3_itd:', 2), ('conc, h, h_snow, pond_depth, h_il=h_il)', 5), ('new_state = DynamicSeaIceState(', 2), ('    )', 193), 417],
    'packages/ice/legoesm/ice/sea_ice.py:427-440,1745-1779,2829-2877,3014-3094,3228-3349,3394-3396': ['if not (config.ponds.enabled and config.ponds.scheme == "si3_lev") and config.itd_remap not in ("simple", "lipscomb2001"):', 'if config.n_categories > 1 and not (config.ponds.enabled and config.ponds.scheme == "si3_lev"):', ('if h_il is not None:', 1), (')', 575), ('if not config.ponds.enabled and config.ponds.scheme != "cesm":', 4), ('conc, h, h_snow, pond_depth, h_il=h_il)', 1), ('if is_multicat and config.itd_remap != "none" and not exact_si3_itd:', 1), 'h_il = jnp.stack(hil_list, axis=-1) if h_il is not None else None', 'if is_multicat and config.itd_remap != "none":', ('conc, h, h_snow, pond_depth, h_il=h_il)', 4), ('if is_multicat and config.itd_remap != "none" and not exact_si3_itd:', 2), ('conc, h, h_snow, pond_depth, h_il=h_il)', 5), 304],
    'packages/ice/legoesm/ice/sea_ice.py:3026-3065': [('if config.ponds.enabled and config.ponds.scheme == "si3_lev":', 3), ('            )', 33), 40],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o9_minmax_audit.py:644-656': ['flags = _flags(build)', 'same = _stream(gobj) == _stream(record)', 13],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o12_minmax_audit.py:1-193': ['"""Exhaustive MIN/MAX/SIGN/SQRT audit of the SI3 O12 melt-pond object code."""', 'sys.exit(main())', 193],
    'tests/ocean/fidelity/test_nemo_si3_o12_minmax_audit.py:1-161': ['"""Pins every MIN/MAX/SIGN/SQRT site in O12\'s compiled production routines."""', 'assert (min_522["dst_defs"], min_522["src_defs"]) == ("mulsd@522", "load@519")', 161],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:500': 'zdv_frz = v_ip_1d(ii) * ( EXP( 0.01_wp * zdT * z1_Tp ) - 1._wp )  ! Holland 2012 (eq. 6)',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:456': 'zdv_mlt   = MAX( 0._wp, zfr_mlt * zdv_avail ) ! max for roundoff errors?',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:465': ('zdv_mlt   = MAX( 0._wp, MIN( zdv_mlt, zv_ip_max - v_ip_1d(ii) ) )', 1),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:472': ('zdv_mlt   = MAX( 0._wp, MIN( zdv_mlt, zv_ip_max - v_ip_1d(ii) ) )', 2),
    'packages/ice/legoesm/ice/si3_thd_pnd.py:311,316': ['zdv_mlt = _maxsd(_minsd(sr(zv_ip_max_area - v_ip), zdv_mlt), zero)', 'zdv_mlt = _maxsd(_minsd(zdv_mlt, sr(zv_ip_max_depth - v_ip)), zero)', 2],
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:490': 'zdv_frz = - 0.5_wp * MAX( 0._wp, -v_il_1d(ii) + & ! Flocco 2010 (eq. 5) solved implicitly as aH**2 + bH + c = 0',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:494': 'v_il_1d(ii) = MAX( 0._wp, v_il_1d(ii) - zdv_frz )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:484': 'zdT = MAX( zTp+rt0 - t_su_1d(ii), 0._wp )',
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:497': ('v_ip_1d(ii) = MAX( 0._wp, v_ip_1d(ii) + zdv_frz )', 1),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:502': ('v_ip_1d(ii) = MAX( 0._wp, v_ip_1d(ii) + zdv_frz )', 2),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:508': ('a_ip_1d(ii) = MIN( a_i_1d(ii), SQRT( v_ip_1d(ii) * z1_aspect * a_i_1d(ii) ) ) ! make sure a_ip < a_i', 1),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:527': ('a_ip_1d(ii) = MIN( a_i_1d(ii), SQRT( v_ip_1d(ii) * z1_aspect * a_i_1d(ii) ) ) ! make sure a_ip < a_i', 2),
    'ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_pnd.f90:522': 'zdv_flush   = -zperm * rho0 * grav * zhp * rDt_ice / (zvisc * h_i_1d(ii)) * a_ip_1d(ii) * rn_pnd_flush ! rn_pnd_flush comes from Hunke et al. (2012)',
    'packages/ice/legoesm/ice/si3_thd_pnd.py:373-382': ['q1 = sr(zperm * rho0)', 'v_ip = sr(v_ip - capped_flush)', 10],
    'tests/ice/unit/test_si3_o12_pnd.py:35-54': ['@pytest.fixture(autouse=True)', 'jax.default_backend = orig_backend', 20],
    'tests/ice/unit/test_si3_o12_pnd.py:62-123': ['# Independent oracle physical constants transcribed directly from the NEMO 5.0.2', 'DECK_PND_NML = _parse_independent_pnd_namelist()', 62],
    'tests/ice/unit/test_si3_o12_pnd.py:140-179': ['def _oracle_ice_perm_eff(', 'return 0.0', 40],
    'tests/ice/unit/test_si3_o12_pnd.py:747-793': ['def test_ice_perm_eff_is_bit_exact_to_independent_scalar_under_jit_and_eager():', 'np.testing.assert_array_equal(_bits(compiled), _bits(oracle))', 47],
    'tests/ice/unit/test_si3_o12_pnd.py:182-421': ['def _oracle_pnd_lev_category(', 'return a_ip_out, h_ip_out, h_il_out, v_ip_out, v_il_out, wfx_out', 240],
    'tests/ice/unit/test_si3_o12_pnd.py:796-917': ['@pytest.mark.parametrize("ln_pnd_lids", [True, False])', 'assert events["lev_nolid_exp_shrink"] >= 1', 122],
    'tests/ice/unit/test_si3_o12_pnd.py:424-520': ['def _oracle_ice_thd_pnd(', 'return at_i_w, a_ip_w, v_ip_w, h_ip_w, v_il_w, h_il_w, a_ip_frac_w, wfx_w', 97],
    'tests/ice/unit/test_si3_o12_pnd.py:920-976': ['def test_ice_thd_pnd_is_bit_exact_to_independent_scalar_and_preserves_halo():', 'np.testing.assert_array_equal(_bits(got[:, -2:]), _bits(entry[:, -2:]))', 57],
    'tests/ice/unit/test_si3_o12_pnd.py:979-1003': ['def test_o12_inexact_constants_and_output_digest_are_pinned():', (')', 298), 25],
    'tests/ice/unit/test_si3_o12_pnd.py:1006-1239': ['_CORE_SINGLE_LINE_SITES: dict[str, tuple[str, str]] = {', (') > 0', 3), 234],
    'tests/ice/unit/test_si3_o12_pnd.py:1242-1301': [('@pytest.mark.parametrize(', 3), 'assert changed > 0, f"plant ({target_name}, {old!r} -> {new!r}) failed to change output"', 60],
    'tests/ice/unit/test_si3_o12_pnd.py:1304-1323': ['def test_ice_perm_eff_epsi06_early_exit_boundary_plant():', 'assert float(planted[0]) > 0.0', 20],
    'tests/ice/unit/test_si3_o12_pnd.py:1326-1421': ['def test_masked_and_signed_zero_paths_are_verified_through_pnd_lev(monkeypatch):', ('np.testing.assert_array_equal(_bits(got), _bits(want))', 5), 96],
    'tests/ice/unit/test_si3_o12_pnd.py:1424-1550': ['def test_five_category_o9_o10_o11_1d2d_o12_composition_matches_scalar_oracle(monkeypatch):', ('np.testing.assert_array_equal(_bits(got), _bits(want))', 6), 127],
    'tests/ice/unit/test_si3_o12_pnd.py:1553-1648': ['def test_o12_composer_calls_o10_then_da_then_1d2d_then_pnd(monkeypatch):', 'np.testing.assert_array_equal(res.pnd.wfx_pnd, one_2d * 42.0)', 96],
    'tests/ice/unit/test_si3_o12_pnd.py:1651-1785': ['def test_pnd_lev_and_ice_thd_pnd_gradients_are_finite_nonzero_and_match_central_difference():', 'assert np.all(np.asarray(grads_planted[1]) == 0.0)', 135],
    'packages/ice/legoesm/ice/si3_thd_pnd.py:90-104': ['@jax.custom_jvp', 'return y, ydot', 15],
    'tests/ice/unit/test_si3_o12_pnd.py:1788-1807': ['def test_edge_case_zero_pond_thin_ice_and_vanished_ice_have_finite_gradients():', 'assert np.all(np.isfinite(np.asarray(grad))), f"NaN/Inf gradient in {name}"', 20],
    'tests/ice/unit/test_si3_o12_pnd.py:1810-1837': [('@pytest.mark.parametrize(', 4), ('pnd_mod.si3_ice_thd_pnd(**kwargs)', 1), 28],
    'tests/ice/unit/test_si3_o12_pnd.py:1840-1867': [('@pytest.mark.parametrize(', 5), ('pnd_mod.si3_ice_thd_pnd(**kwargs)', 2), 28],
    'tests/ice/unit/test_si3_o12_pnd.py:1870-1911': [('@pytest.mark.parametrize(', 6), 'assert mismatches > 0, f"constant defect in {const_field} was not caught by independent oracle"', 42],
    'tests/ice/unit/test_si3_o12_pnd.py:1914-2183': ['def test_production_step_sea_ice_and_step_ponds_wire_si3_lev_and_preserve_cesm(monkeypatch):', (')', 818), 270],
    'tests/ice/unit/test_si3_o12_pnd.py:2186-2407': ['def test_lid_thickness_h_il_persists_across_steps_and_restart_round_trip(tmp_path, monkeypatch):', 'assert np.count_nonzero(_bits(rsp2_reset.freshwater_flux) != _bits(rsp2_jit.freshwater_flux)) == shape[0] * shape[1]', 222],
    'tests/ice/unit/test_si3_o12_pnd.py:2410-2610': ['def test_multicategory_jpl5_step_sea_ice_sums_at_i_over_categories(monkeypatch):', 'assert np.count_nonzero(_bits(rsp_spatial_bug.freshwater_flux) != _bits(rsp_jit.freshwater_flux)) == math.prod(spatial)', 201],
    'tests/ice/unit/test_si3_o12_pnd.py:2613-2860': ['def test_m6_parent_commit_vs_current_across_all_other_callers(monkeypatch):', 'assert np.count_nonzero(_bits(perturbed_st.pond_area.data) != _bits(want_v2_1_st.pond_area.data)) > 0', 248],
    'tests/ice/unit/test_si3_o12_pnd.py:2863-3027': ['def test_decision_184a_default_cesm_run_state_pytree_structure_and_leaves_equal_parent():', 'dynamic_to_slab(init_dynamic_ice_state(shape_2d, pond_scheme="si3_lev"))', 165],
    'tests/ice/unit/test_si3_o12_pnd.py:3030-3200': ['def test_decision_184b_default_run_restarts_are_byte_identical_to_parent(tmp_path):', 'assert si3_rst.read_bytes() != par_rst_1.read_bytes()', 171],
    'tests/ice/unit/test_si3_o12_pnd.py:3203-3333': ['def test_decision_184c_si3_lev_run_carries_and_restarts_h_il(tmp_path):', 'load_run_restart(rst_cesm_path, ocn_state, ice_template=template_mc_si3, grid_type="latlon")', 131],
    'tests/ice/unit/test_si3_o12_pnd.py:3336-3793': ['def test_si3_lev_lids_refuses_advection_concentration_cap_itd_remap_and_ridging_while_thermo_only_runs(', ('assert jnp.all(jnp.isfinite(rsp_out_5.freshwater_flux))', 2), 458],
    'docs/ice/fidelity/si3_r20_o12_preregister.md:66-74': ['- **P1 (Object-code MIN/MAX/SQRT/comparison audit)**: Disassembly of `ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_pnd.o` maps every `minsd`, `maxsd`, `sqrtsd`, and `comisd`/`ucomisd` in `ice_thd_pnd`, `pnd_LEV`, and `ice_perm_eff`. In particular, gfortran `-O3` fuses lines `522-524` (`zdv_flush = -zperm * ...`, `zdv_flush = MAX( zdv_flush, -v_ip_1d(ii) )`, `v_ip_1d(ii) = v_ip_1d(ii) + zdv_flush`) into positive `pos_flush = +zperm * ...`, `capped = minsd(dst=pos_flush, src=v_ip_1d)`, `v_ip_1d = v_ip_1d - capped` (`instructions_without_token = [522]`, `token_statements_without_instruction = [523]`), and lines `465` and `472` have opposite `minsd` `dst`/`src` operand roles (`465`: `dst = zv_ip_max - v_ip_1d, src = zdv_mlt`; `472`: `dst = zdv_mlt, src = zv_ip_max - v_ip_1d`).', '- **REFUTE**: Any `NaN`/`Inf` gradient or FD mismatch.', 9],
})


# --- SI3 round 21 (O8-O12 master-record scoring) ---
FILES.update({
    'docs/ice/fidelity/si3_r21_o8_o12_score_preregister.md': REPO / 'docs/ice/fidelity/si3_r21_o8_o12_score_preregister.md',
    'docs/ice/fidelity/si3_r21_o8_o12_score_receipt.md': REPO / 'docs/ice/fidelity/si3_r21_o8_o12_score_receipt.md',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py': REPO / 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py',
})

CITATION_MAP.update({
    'docs/ice/fidelity/si3_r21_o8_o12_score_preregister.md:1-51': ['# SI3 Ladder Round 21 (`si3_r21`) — O8, O9, O10, O11, O12 Master-Record Scoring Preregistration', '- A 1-ULP change planted in one exit field of each rung (`O8` `ht_i_new`, `O9` `t_su`/`h_i`/`t_i`, `O10` `sz_i`/`t_i`, `O11` `a_i`, `O12` `v_ip`) is caught by the scorer (`unequal >= 1`, `max_ulp >= 1`, nonzero exit status), and parameter/checksum/swapped-field plants fail closed.', 51],
    'docs/ice/fidelity/si3_r18_o8_o16_receipt.md:154-180': ['`STATUS: ADMITTED`', '| O16 | 18, 15, 16, 19, 20 | 500 | `6a3f9ed6f9330af1c65b603726c2ba407a28683f2de1b2a9dcaf9fd7dd3dc5e0` |', 27],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-1862': ['#!/usr/bin/env python3', 'raise SystemExit(main())', 1862],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-438': ['"""Direct unit and master-record tests for the SI3 O8--O12 0-ULP scoring driver."""', 'np.asarray(rev_state_pnd.pond_depth.data).view(np.uint64),', 438],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:308-363': ['def verify_master_admission_manifests(', ('}', 22), 56],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:267-305': ['def parse_resolved_ice_namelist(', 'return parsed', 39],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:40-55': ['SCALAR_NAMES = (', (')', 5), 16],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:467-523': ['def _verify_record_scalars_against_namelist(', 'f"scalar header {sname}={got!r} disagrees', 57],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:641-944': ['@functools.lru_cache(maxsize=8)', '"jit_o13_remap": jit_o13_remap,', 304],
    'docs/ice/fidelity/si3_r15_o8_receipt.md:1-195': ['# SI3 round 15 receipt — O8 model preparation: frazil collection bookkeeping', 'Reviews: Argon LAND; codex HOLD (production wiring) then LAND after resume 1. DECISION 173 (user: "Your pick"): O8 is admitted from the ONE shared 100-step ORCA2_SAS_ICE master record for O8-O16 (si3_ladder_design.md:205-209); no separate O8 run. The O8 acquisition therefore waits for the master-record tool.', 195],
    'docs/ice/fidelity/si3_r16_o9_receipt.md:1-336': ['# SI3 round 16 receipt — O9 model preparation: ZDF, DH and TEMP at jpl = 5, 10 + 5 layers', 'default citation-gate test `33 passed`; private-import ratchet `6 passed`. Review of the rebase: UNREVIEWED.', 336],
    'docs/ice/fidelity/si3_r19_o10_receipt.md:1-245': ['# SI3 Ladder Round 19 (`si3_r19`) — O10 model-preparation receipt', 'lines produced `SYMBOL-NOT-AT-LINE`, gate status `FAIL`, and exit status 1.', 245],
    'docs/ice/fidelity/si3_r17_o11_receipt.md:1-251': ['# SI3 round 17 receipt — O11 model preparation: lateral melt (ice_thd_da)', 'Review: `codex review: operator`; GLM: not run (NO GATE).', 251],
    'docs/ice/fidelity/si3_r20_o12_receipt.md:1-188': ['# SI3 Ladder Round 20 (`si3_r20`) — O12 model-preparation receipt', '`======================= 165 passed in 1097.07s (0:18:17) =======================`', 188],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1084-1101': ['# 1. Rung O8: Kind 1 -> Kind 2 (si3_ice_thd_frazil)', 'o8_acc.update("fraz_frac", out_o8.fraz_frac, f2["fraz_frac"], kt=kt, category=0)', 18],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1259-1492': ['# Check Kind 21 -> Kind 3 gather for intensive/state fields', 'picard_iconv_matches += 1', 234],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1381-1406': ['# Spot-check unpadded (npti,) JIT execution on kt == steps[0]', 'unpadded_spot_checks += 1', 26],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1494-1513': ['# 3. Rung O10: Kind 6 -> Kind 7 (si3_ice_thd_sal) and Kind 7 -> Kind 8 (si3_ice_thd_temp)', 'spec.name, r7["fields"][spec.name], f8[spec.name], kt=kt, category=jl', 20],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1516-1526': ['# 4. Rung O11: Kind 8 -> Kind 9 (si3_ice_thd_da)', 'spec.name, r8["fields"][spec.name], f9[spec.name], kt=kt, category=jl', 11],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1529-1605': ['# Production entry point: si3_column_step_arrays (Kind 3 -> Kind 4, 5, 6, 7, 8, 9)', ('            )', 58), 77],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1607-1639': ["# 5. Rung O12: Kind 10 -> Kind 11 (si3_ice_thd_pnd and step_ponds(scheme='si3_lev'))", ('                )', 49), 33],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:21-53': ['def test_field_score_accumulator_catches_1ulp_and_signed_zero() -> None:', 'assert acc.summary()["status"] == "NON_BIT_EXACT"', 33],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:534-624': ['class FieldScoreAccumulator:', ('}', 44), 91],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:56-83': ['def test_master_admission_manifests_and_plants() -> None:', ('verify_manifests=False,', 1), 28],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:87-135': ['def test_score_o8_to_o12_representative_steps_0_ulp() -> None:', 'assert rungs["O12"]["production_step_ponds_si3_lev"]["first_non_bit"] is None', 49],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1033-1768': ['def score_o8_to_o12(', '"rungs": rungs_report,', 736],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:142-160': ['@pytest.mark.parametrize(', (')', 48), 19],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:170-438': ['def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(', 'np.asarray(rev_state_pnd.pond_depth.data).view(np.uint64),', 269],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2024-2042': ['def si3_ice_thd_frazil(qlead, *, rn_hinew, ln_frazil) -> SI3Frazil:', 'return SI3Frazil(ht_i_new=ht_i_new, fraz_frac=zero)', 19],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:834-919': ['def si3_ice_thd_zdf_bl99(', 'hfx_dif, cnd_ice, t1_ice, t_si, core.iterations)', 86],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1288-1311': ['def si3_ice_thd_dh(*, fluxes: SI3DHFluxes, rDt_ice, rn_snwblow, rn_sinew,', (')', 587), 24],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1272-1277': ['def si3_ice_thd_temp(*, h_i, e_i, sz_i, constants: IceConstantsConfig):', 'return jnp.where((h_i > 0.0)[..., None], t, constants.T0)', 6],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1344-1382': ['def si3_ice_thd_category_stages(*, entry: dict, fluxes: SI3DHFluxes, namelist: dict,', 'return SI3CategoryStages(zdf, dh, t_i)', 39],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1782': ['def si3_ice_thd_sal(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i, dh_i_sum,', (')', 834), 78],
    'packages/ice/legoesm/ice/si3_thd_da.py:65-185': ['def si3_ice_thd_da(', (')', 77), 121],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:701-741': ['def admit(root: Path, producer: str | None = None) -> dict[str, Any]:', 'return {"status": "ADMITTED", "manifest": manifest, "projections": list(PROJECTIONS)}', 41],
})

# --- SI3 round 21 Resume 2 (DECISION 186 Option A re-anchors) ---
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:936', 'packages/ice/legoesm/ice/bitz_lipscomb.py:834', anchors='def si3_ice_thd_zdf_bl99(')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1264', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1015', anchors='def _si3_ice_thd_dh_core(')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1537', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1288', anchors='def si3_ice_thd_dh(*, fluxes: SI3DHFluxes, rDt_ice, rn_snwblow, rn_sinew,')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1229', 'packages/ice/legoesm/ice/bitz_lipscomb.py:980', anchors='def _cumulative_remap(h_old, content_old, nlay, sr):')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1521', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1272', anchors='def si3_ice_thd_temp(*, h_i, e_i, sz_i, constants: IceConstantsConfig):')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1593', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1344', anchors='def si3_ice_thd_category_stages(*, entry: dict, fluxes: SI3DHFluxes, namelist: dict,')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1347', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1098', anchors='zdeltah = -_minsd(sr(sr(evap_ice * r1_rhos) * dt), h_s)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1486', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1237', anchors='zdum = _minsd(zh_s[jk], zdeltah)  # :490 emitted `minsd zdeltah, zh_s`')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1008', 'packages/ice/legoesm/ice/bitz_lipscomb.py:906', anchors='_div_const(2.0 * core.tcond_i[..., 0], _ICE_SSL_DEPTH),')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1306', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1057', anchors="zero = sr(jnp.zeros_like(h_i))  # materialized: NEMO's 0._wp + (-0) is +0, XLA's folded 0 + x is not")
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2037', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1879', anchors='sst_dh = t_bo_arr - constants.T0')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2822', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2024', anchors='def si3_ice_thd_frazil(qlead, *, rn_hinew, ln_frazil) -> SI3Frazil:')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2810', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2012', anchors='def _ieee_lt_zero(value):')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2773', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1968', anchors='frazil = si3_ice_thd_frazil(')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2774', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1973', anchors='cooling = _ieee_lt_zero(forcing.qlead)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2782', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1981', anchors='vnew / jnp.where(cooling, frazil.ht_i_new, 1.0),')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2700', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1815', anchors='si3=None,')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2758', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1952', anchors='post_da = _si3_column_thd_da(state, post_temp2, forcing, si3, dt, constants)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2845', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2047', anchors='ieee_lt_zero = _ieee_lt_zero')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2875', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2077', anchors='def _si3_column_thd_da(entry, post_temp2, forcing, si3, dt, constants, *,')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2907', 'packages/ice/legoesm/ice/bitz_lipscomb.py:2116', anchors='gathered = entry.concentration > _EPS10')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1634-1643', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1385-1394', anchors=['def _sal4_vc19_brine(t_i, T0):', 'return nemo_source_round(nemo_source_round(linear - quadratic) - cubic)', 10])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1646-1672', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1397-1423', anchors=['def _sal4_effective_permeability(zv_br, *, initial):', 'return jnp.stack(out, axis=-1)', 27])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1653-1672', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1404-1423', anchors=[('for jk in range(nlay):', 1), 'return jnp.stack(out, axis=-1)', 20])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1675-1729', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1426-1480', anchors=['def _sal4_rjw_velocity(sz_i, zs_br, zmsk, sss, h_i, z_mid, z_edge, z1_h_i,', 'return jnp.stack(velocity, axis=-1)', 55])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1682-1689', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1433-1440', anchors=['divisor = (', 'zperm_eff = _sal4_effective_permeability(zv_br, initial=initial)', 8])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1687-1689', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1438-1440', anchors=['volume_numerator = nemo_source_round(zmsk * sz_i)', 'zperm_eff = _sal4_effective_permeability(zv_br, initial=initial)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1691-1693', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1442-1444', anchors=['z1_cp_br = nemo_source_round(_div_const(one, _SAL4_CP_BRINE))', 'z1_visc = nemo_source_round(_div_const(one, _SAL4_VISCOSITY_BRINE))', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1740-1843', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1491-1594', anchors=['def _si3_ice_thd_sal4_core(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i,', ('(sz_i, sfx_bri, zero, zero),', 1), 104])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1846-1917', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1597-1668', anchors=['# Flushing uses upward meltwater flow and the same positive-ocean salt flux.', ('(sz_i, sfx_bri, zero, zero),', 2), 72])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1874-1882', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1625-1633', anchors=['def flushing_body(iteration, carry):', 'active = flush_cell & (iteration < nflush) & (zv_brmin >= rn_vbrc)', 9])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1891', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1642', anchors='sal_min_next = _minsd(sal_min, raw_salinity)')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1895-1897', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1646-1648', anchors=['# -O3 line-644 roles: inner dst=0 only at layer 9; outer dst=floor at 9/10.', 'zds = _maxsd(floor_delta, nonpositive) if jk in (8, 9) else _maxsd(nonpositive, floor_delta)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1921-1937', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1672-1688', anchors=['zsal = sz_i[..., jk]', 'cfl_drain, cfl_flush, smin_drain, smin_flush,', 17])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1949-1986', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1705-1774', anchors=['def si3_ice_thd_sal(*, h_i, t_su, t_i, sz_i, t_bo, sss, a_i, dh_i_sum,', 'raise ValueError("SI3 O10 requires the production 10 ice layers")', 70])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1997-2020', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1785-1808', anchors=['def si3_ice_thd_o10_category_stages(*, entry: dict, fluxes: SI3DHFluxes,', 'return SI3O10CategoryStages(o9, sal, t_i)', 24])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:759-887', 'packages/ice/legoesm/ice/bitz_lipscomb.py:759-804', anchors=['Tsu = jnp.where(active, Tsu_candidate, Tsu)', 'return qcn_bottom, qcn_top', 46])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:829-834', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1841-1843', anchors=['T_s_old = snow_temperature_from_enthalpy(e_snow, constants)', 'T_s_old = jnp.where((h_snow > 0.0)[..., None], T_s_old, constants.T0)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:793-811', 'packages/ice/legoesm/ice/bitz_lipscomb.py:545-566', anchors=['def _si3_zdf_bl99_core(', ') -> _SI3ZDFCore:', 22])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:793-891', 'packages/ice/legoesm/ice/bitz_lipscomb.py:545-804', anchors=['def _si3_zdf_bl99_core(', 'return qcn_bottom, qcn_top', 260])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:636-871', 'packages/ice/legoesm/ice/bitz_lipscomb.py:636-789', anchors=['k_i = tcond_i / dz_i[..., None]', 'comb_frozen, tcond_frozen, isnow, iterations)', 154])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1025-1045', 'packages/ice/legoesm/ice/bitz_lipscomb.py:980-1012', anchors=['def _cumulative_remap(h_old, content_old, nlay, sr):', ('], axis=-1)', 10), 33])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1048-1107', 'packages/ice/legoesm/ice/bitz_lipscomb.py:988-1012', anchors=['zero = nemo_source_round(jnp.zeros_like(h_old[0]))', ('], axis=-1)', 10), 25])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1110-1168', 'packages/ice/legoesm/ice/bitz_lipscomb.py:980-1011', anchors=['def _cumulative_remap(h_old, content_old, nlay, sr):', '_maxsd(e_cum1[k] - e_cum1[k - 1], 0.0) / divisor for k in range(1, nlay + 1)', 32])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1110-1120', 'packages/ice/legoesm/ice/bitz_lipscomb.py:980-992', anchors=['def _cumulative_remap(h_old, content_old, nlay, sr):', 'e_cum0.append(e_cum0[-1] + e)', 13])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:1120', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1074', anchors='ze_i_old = [zero] + [sr(dz0 * e_i[..., k]) for k in range(nlay_i)] + [zero]')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2053-2675', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1015-1269', anchors=['def _si3_ice_thd_dh_core(', (')', 567), 255])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2750-2755', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1750-1755', anchors=['s_i_new = jnp.clip(', 'sz_i_new = option2_salinity_profile(s_i_new, arrays["sss"])', 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2643-2659', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1243-1259', anchors=['e_s_new = _cumulative_remap(zh_s, [sr(e * h) for e, h in zip(ze_s, zh_s)], nlay_s, sr)', 's_i_new = s_i + zds', 17])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2653-2659', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1243-1248', anchors=['e_s_new = _cumulative_remap(zh_s, [sr(e * h) for e, h in zip(ze_s, zh_s)], nlay_s, sr)', (')', 556), 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2224-2226', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1063-1066', anchors=['qml = jnp.where(', 'zq_top = _maxsd(sr(qml * dt), 0.0)', 4])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2643-2652', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1249-1252', anchors=['e_i_new = _cumulative_remap(zh_i_old, ze_i_old, nlay_i, sr)', ('s_i_new = s_i', 1), 4])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2631-2640', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1071-1077', anchors=['dz0 = sr(h_i * r1_nlay_i)', 'ze_s = [zero] + [e_s[..., k] for k in range(nlay_s)]', 7])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2644-2649', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1249-1251', anchors=['e_i_new = _cumulative_remap(zh_i_old, ze_i_old, nlay_i, sr)', 'sz_i_new = _cumulative_remap(zh_i_old, zs_i_old, nlay_i, sr)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2233-2425,2601-2652', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1121-1209,1231-1252', anchors=['def melt_layer(jk, heat, internal_melt, active, kind):', 'zq_bot = melt_layer(jk, zq_bot, internal_melt, active, "bom")', 'zh_i[0] = zh_i[0] + dh_sni', ('s_i_new = s_i', 1), 111])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2091-2108,2143-2159,2218-2222', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1095,1098-1108,1111-1119', anchors=['falling = sprecip > 0.0', 'h_s = jnp.where(falling, h_s + zh_s[0], h_s)', 'zdeltah = -_minsd(sr(sr(evap_ice * r1_rhos) * dt), h_s)', 'zdeltah = _minsd(zdeltah - zdum, 0.0)', ('for jk in range(nlay_s + 1):', 2), 'zh_s[jk] = jnp.where(melt, _maxsd(zh_s[jk] + zdum, 0.0), zh_s[jk])', 26])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:41,1110-1168,2233-2425,2601-2652,2678-2730', 'packages/ice/legoesm/ice/bitz_lipscomb.py:41,980-1011,1121-1209,1231-1252,1811-1939', anchors=['frq_m: jax.Array | float = 1.0', 'def _cumulative_remap(h_old, content_old, nlay, sr):', '_maxsd(e_cum1[k] - e_cum1[k - 1], 0.0) / divisor for k in range(1, nlay + 1)', 'def melt_layer(jk, heat, internal_melt, active, kind):', 'zq_bot = melt_layer(jk, zq_bot, internal_melt, active, "bom")', 'zh_i[0] = zh_i[0] + dh_sni', ('s_i_new = s_i', 1), 'def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).', 273])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:41,2304-2308', 'packages/ice/legoesm/ice/bitz_lipscomb.py:41,1067-1068', anchors=['frq_m: jax.Array | float = 1.0', 'zf_tt = sr(sr(qcn_ice_bot + qsb_ice_bot) + fhld) + sr(qtr_ice_bot * frq_m)', 'zq_bot = _maxsd(sr(zf_tt * dt), 0.0)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:41,2304-2308,2488', 'packages/ice/legoesm/ice/bitz_lipscomb.py:41,1067-1068,1901', anchors=['frq_m: jax.Array | float = 1.0', 'zf_tt = sr(sr(qcn_ice_bot + qsb_ice_bot) + fhld) + sr(qtr_ice_bot * frq_m)', 'zq_bot = _maxsd(sr(zf_tt * dt), 0.0)', '"frq_m": jnp.asarray(forcing.frq_m, dtype=dtype),', 4])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2091-2108,2143-2159,2218-2222,2695,2727', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1095,1098-1108,1111-1119,1902,1904', anchors=['falling = sprecip > 0.0', 'h_s = jnp.where(falling, h_s + zh_s[0], h_s)', 'zdeltah = -_minsd(sr(sr(evap_ice * r1_rhos) * dt), h_s)', 'zdeltah = _minsd(zdeltah - zdum, 0.0)', ('for jk in range(nlay_s + 1):', 2), 'zh_s[jk] = jnp.where(melt, _maxsd(zh_s[jk] + zdum, 0.0), zh_s[jk])', '"sprecip": jnp.asarray(forcing.snow_precipitation, dtype=dtype),', '"evap_ice": jnp.asarray(forcing.evaporation, dtype=dtype),', 28])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2096-2100', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1090-1091', anchors=['falling = sprecip > 0.0', 'zh_s[0] = jnp.where(falling, sr(sr(sr(zsnw * sprecip) * dt) * r1_rhos) / at_i, 0.0)', 2])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2101-2103', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1091-1092', anchors=['zh_s[0] = jnp.where(falling, sr(sr(sr(zsnw * sprecip) * dt) * r1_rhos) / at_i, 0.0)', 'ze_s[0] = jnp.where(falling, _maxsd(-qprec_ice, 0.0), 0.0)', 2])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2147-2153', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1098-1103', anchors=['zdeltah = -_minsd(sr(sr(evap_ice * r1_rhos) * dt), h_s)', 'zdum = _maxsd(-zh_s[jk], zdeltah)  # :200', 6])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2218-2222', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1112-1119', anchors=['melt = (zh_s[jk] > 0.0) & (zq_top > 0.0)', 'zh_s[jk] = jnp.where(melt, _maxsd(zh_s[jk] + zdum, 0.0), zh_s[jk])', 8])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2099,2258,2361', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1091,1129,1130', anchors=['zh_s[0] = jnp.where(falling, sr(sr(sr(zsnw * sprecip) * dt) * r1_rhos) / at_i, 0.0)', 'zfmdt_gross = -heat / zdE', 'zdum_melt = _minsd(_maxsd(sr(-zfmdt_gross * r1_rhoi), -zh_i[jk]), 0.0)', 3])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2065-2066,2694-2695', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1021-1022,1902-1904', anchors=['rDt_ice: float, rn_snwblow: float, rn_sinew: float, nn_icesal: int,', ('constants: IceConstantsConfig,', 7), '"sprecip": jnp.asarray(forcing.snow_precipitation, dtype=dtype),', '"evap_ice": jnp.asarray(forcing.evaporation, dtype=dtype),', 5])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2678-2714', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1927', anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', (')', 880), 117])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2678-2796', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1998', anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', ('return trace', 2), 188])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2678-2730', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1811-1939', anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).', 129])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2695,2727', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1902,1904', anchors=['"sprecip": jnp.asarray(forcing.snow_precipitation, dtype=dtype),', '"evap_ice": jnp.asarray(forcing.evaporation, dtype=dtype),', 2])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2248-2257', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1125-1132', anchors=['ztmelts = sr(-rTmlt * sz_i[..., jk - 1])', 'zdum = jnp.where(internal_melt, zdum_itm, zdum_melt)', 8])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2354-2360', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1205-1209', anchors=['for jk in range(nlay_i, 0, -1):', 'zq_bot = melt_layer(jk, zq_bot, internal_melt, active, "bom")', 5])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2397-2406', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1182-1190', anchors=['ztmelts = sr(-rTmlt * zs_i_new)', 'dh_bog = jnp.where(growing, sr(dt * _maxsd(sr(zf_tt / sr(zdE * rhoi)), 0.0)), 0.0)', 9])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2229,2248-2257,2354-2360,2397-2406', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1039,1125-1132,1182-1190,1205-1209', anchors=['r1_rhoi, r1_rhos = 1.0 / rhoi, 1.0 / rhos', 'ztmelts = sr(-rTmlt * sz_i[..., jk - 1])', 'zdum = jnp.where(internal_melt, zdum_itm, zdum_melt)', 'ztmelts = sr(-rTmlt * zs_i_new)', 'dh_bog = jnp.where(growing, sr(dt * _maxsd(sr(zf_tt / sr(zdE * rhoi)), 0.0)), 0.0)', 'for jk in range(nlay_i, 0, -1):', 'zq_bot = melt_layer(jk, zq_bot, internal_melt, active, "bom")', 23])
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2229', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1039', anchors='r1_rhoi, r1_rhos = 1.0 / rhoi, 1.0 / rhos')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2258', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1129', anchors='zfmdt_gross = -heat / zdE')
_reanchor('packages/ice/legoesm/ice/bitz_lipscomb.py:2361', 'packages/ice/legoesm/ice/bitz_lipscomb.py:1130', anchors='zdum_melt = _minsd(_maxsd(sr(-zfmdt_gross * r1_rhoi), -zh_i[jk]), 0.0)')
_reanchor('packages/ice/legoesm/ice/si3_thd_pnd.py:575-763', 'packages/ice/legoesm/ice/si3_thd_pnd.py:575-781', anchors=['def si3_ice_thd_pnd(', (')', 322), 207])
_reanchor('packages/ice/legoesm/ice/si3_thd_pnd.py:683', 'packages/ice/legoesm/ice/si3_thd_pnd.py:701', anchors='at_i = fortran_sum0(arr["a_i"]) if _at_i_override is None else _required("at_i", _at_i_override)')
_reanchor('packages/ice/legoesm/ice/si3_thd_pnd.py:699-717', 'packages/ice/legoesm/ice/si3_thd_pnd.py:717-735', anchors=[('for jl in range(jpl):', 2), 'a_ip_frac_int = jnp.stack(a_ip_frac_clean, axis=-1)', 19])
_reanchor('packages/ice/legoesm/ice/si3_thd_pnd.py:720-763', 'packages/ice/legoesm/ice/si3_thd_pnd.py:738-781', anchors=['has_ice = at_int >= epsi10', (')', 322), 44])
_reanchor('packages/ice/legoesm/ice/si3_thd_pnd.py:766-954', 'packages/ice/legoesm/ice/si3_thd_pnd.py:784-972', anchors=['def si3_ice_thd_o12_stages(', 'return SI3O12Stages(categories=tuple(cat_outputs), pnd=pnd_out)', 189])
_reanchor('packages/ice/legoesm/ice/ponds.py:34-212', 'packages/ice/legoesm/ice/ponds.py:34-311', anchors=['def step_ponds(', (')', 70), 278])


# --- SI3 round 23 (O14) production/scorer insertions and receipt ---
# Re-anchor every pre-existing citation shifted by the single production
# implementation and scorer extension.  The two deleted inline expressions
# move to their exact equivalents inside ``si3_ice_thd_do``.
def _o14_bitz_line(line: int) -> int:
    if line <= 1810:
        return line
    if line <= 1862:
        return line + 248
    if line <= 1953:
        return line + 250
    if line == 1968:
        return 2205
    if line == 1973:
        return 1895
    if line == 1981:
        return 1950
    if line == 1998:
        return 2249
    if 1999 <= line <= 2068:
        return line + 251
    if line > 2068:
        return line + 253
    raise RuntimeError(f"O14 needs an explicit anchor for replaced bitz_lipscomb line {line}")


def _o14_spec(spec: str, mapper) -> str:
    pieces = []
    for piece in spec.split(","):
        if "-" in piece:
            first, last = map(int, piece.split("-", 1))
            pieces.append(f"{mapper(first)}-{mapper(last)}")
        else:
            pieces.append(str(mapper(int(piece))))
    return ",".join(pieces)


for _old in [key for key in CITATION_MAP
             if key.startswith("packages/ice/legoesm/ice/bitz_lipscomb.py:")]:
    _prefix, _spec = _old.rsplit(":", 1)
    _new = f"{_prefix}:{_o14_spec(_spec, _o14_bitz_line)}"
    if _new != _old:
        _reanchor(_old, _new)


def _o14_scorer_line(line: int) -> int:
    if line == 1:
        return 1
    if line <= 928:
        return line + 14
    return line + 133


for _old in [key for key in CITATION_MAP if key.startswith(
        "scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:")]:
    _prefix, _spec = _old.rsplit(":", 1)
    _new = f"{_prefix}:{_o14_spec(_spec, _o14_scorer_line)}"
    if _new != _old:
        _reanchor(_old, _new)

_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-438',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-424',
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:21-53',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:22-54',
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:56-83',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:57-85',
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:87-135',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:88-136',
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:142-160',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:161-184',
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:170-438',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:187-424',
)

FILES.update({
    'docs/ice/fidelity/si3_r23_o14_score_preregister.md': REPO / 'docs/ice/fidelity/si3_r23_o14_score_preregister.md',
    'docs/ice/fidelity/si3_r23_o14_score_receipt.md': REPO / 'docs/ice/fidelity/si3_r23_o14_score_receipt.md',
    'tests/ice/unit/test_si3_thd_do.py': REPO / 'tests/ice/unit/test_si3_thd_do.py',
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90': Path('/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90'),
})

CITATION_MAP.update({
    'docs/ice/fidelity/si3_r23_o14_score_preregister.md:1-51': ['# SI3 Ladder Round 23 (`si3_r23`) — O14 Model Preparation and Scoring Preregistration', 'No configuration, threshold, cadence, state schema, or scoring choice is made', 51],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:177-219': ['at_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) = SUM(', 'END DO', 43],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:225-273': ['! --- Salinity of new ice --- !', 'sfx_opw_1d(ii) = sfx_opw_1d(ii) - zv_newice * rhoi * zs_newice(ii) * r1_Dt_ice', 49],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:275-320': ['! A fraction fraz_frac of frazil ice is accreted at the ice bottom', 'ENDIF', 46],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:322-375': ['! --- bottom ice growth + ice enthalpy remapping --- !', 'CALL tab_1d_2d( npti, nptidx(1:npti), hfx_opw_1d(1:npti), hfx_opw )', 54],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1811-2056': ['class SI3DOOutputs(NamedTuple):', (')', 930), 246],
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2204-2249': ['# `ice_thd_frazil` supplies ht_i_new', ('return trace', 2), 46],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:816-825': ['@jax.jit', ('}', 38), 10],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:957-1061': ['def score_o14(', ('}', 52), 105],
    'tests/ice/unit/test_si3_thd_do.py:82-143': ['def _scalar_oracle(x):', 'return out', 62],
    'tests/ice/unit/test_si3_thd_do.py:157-295': ['def test_scalar_oracle_category_choice_fluxes_and_all_state_fields():', 'assert np.all(swapped != got)', 139],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:139-158': ['def test_score_o14_representative_steps_0_ulp_and_plants() -> None:', (')', 48), 20],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:187-360': ['def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(', (')', 141), 174],
})


def _repin(citation: str, anchors) -> None:
    if citation not in CITATION_MAP:
        raise RuntimeError(f"cannot repin missing citation {citation}")
    CITATION_MAP[citation] = anchors


_repin('packages/ice/legoesm/ice/bitz_lipscomb.py:1895', 'cooling = _ieee_lt_zero(qlead)')
_repin('packages/ice/legoesm/ice/bitz_lipscomb.py:1950', 'za_newice = sr(zv_newice / safe_h_new)')
_repin('packages/ice/legoesm/ice/bitz_lipscomb.py:41,980-1011,1121-1209,1231-1252,2059-2189', [
    'frq_m: jax.Array | float = 1.0', 'def _cumulative_remap(h_old, content_old, nlay, sr):',
    '_maxsd(e_cum1[k] - e_cum1[k - 1], 0.0) / divisor for k in range(1, nlay + 1)',
    'def melt_layer(jk, heat, internal_melt, active, kind):',
    'zq_bot = melt_layer(jk, zq_bot, internal_melt, active, "bom")',
    'zh_i[0] = zh_i[0] + dh_sni', ('s_i_new = s_i', 1),
    'def si3_column_step_arrays(state: SI3ColumnArrays,',
    '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).', 275])
_reanchor(
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2059-2177',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:2059-2176',
    anchors=['def si3_column_step_arrays(state: SI3ColumnArrays,', 'e_snow=o10.o9.zdf.e_s,', 118],
)
_repin('packages/ice/legoesm/ice/bitz_lipscomb.py:2059-2249', [
    'def si3_column_step_arrays(state: SI3ColumnArrays,', ('return trace', 2), 191])
_repin('packages/ice/legoesm/ice/bitz_lipscomb.py:2059-2189', [
    'def si3_column_step_arrays(state: SI3ColumnArrays,',
    '# `ice_thd_temp` is diagnostic because enthalpy is prognostic (:221-247).', 131])

_reanchor(
    'docs/ice/fidelity/si3_r23_o14_score_preregister.md:1-51',
    'docs/ice/fidelity/si3_r23_o14_score_preregister.md:1-52',
    anchors=['# SI3 Ladder Round 23 (`si3_r23`) — O14 Model Preparation and Scoring Preregistration',
             'No configuration, threshold, cadence, state schema, or scoring choice is made', 52],
)
_repin('ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:177-219', [
    'at_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0)) = SUM(', ('END DO', 3), 43])
_repin('ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:225-273', [
    'CASE ( 2 , 4 )', 'sfx_opw_1d(ii) = sfx_opw_1d(ii) - zv_newice * rhoi * zs_newice(ii) * r1_Dt_ice', 49])
_repin('ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:275-320', [
    '! A fraction fraz_frac of frazil ice is accreted at the ice bottom', ('ENDIF', 5), 46])
_repin('ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icethd_do.f90:322-375', [
    '! --- bottom ice growth + ice enthalpy remapping --- !',
    'CALL tab_3d_4d( npti, nptidx(1:npti), szv_i_2d(1:npti,:,:), szv_i )', 54])
_reanchor(
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1811-2056',
    'packages/ice/legoesm/ice/bitz_lipscomb.py:1811-2055',
    anchors=['class SI3DOOutputs(NamedTuple):', 'hfx_opw=jnp.where(cooling, hfx_opw_new, hfx_opw),', 245],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:816-825',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:816-823',
    anchors=[('@jax.jit', 11), ('r1_nlay_i=r1_nlay_i,', 3), 8],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:957-1061',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:957-1060',
    anchors=['def score_o14(', ('"resolved_namelist": namelist,', 1), 104],
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:139-158',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:139-157',
    anchors=['def test_score_o14_representative_steps_0_ulp_and_plants() -> None:',
             ('verify_manifests=False', 3), 19],
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:187-360',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:187-359',
    anchors=['def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(',
             'np.asarray(rev_do_state.h_ice.data).view(np.uint64),', 173],
)
# --- SI3 round 24 (O15/O16 stopped score) ---
FILES.update({
    'docs/ice/fidelity/si3_r24_o15_o16_receipt.md': REPO / 'docs/ice/fidelity/si3_r24_o15_o16_receipt.md',
    'icecor.f90': Path('/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REC/BLD/ppsrc/nemo/icecor.f90'),
    'iceitd.f90': Path('/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REC/BLD/ppsrc/nemo/iceitd.f90'),
    'iceupdate.f90': Path('/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/refused/builds_20261026T210237Z_2/ORCA2_SAS_ICE_O8O16_REC/BLD/ppsrc/nemo/iceupdate.f90'),
    'sbc_phy.F90': NEMO / 'src/OCE/SBC/sbc_phy.F90',
    'packages/ice/legoesm/ice/si3_finalize.py': REPO / 'packages/ice/legoesm/ice/si3_finalize.py',
    'nemo_si3_o8_o16_master_gate.py': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py',
    'o8_o16_master_writer.F90': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_acquisition/o8_o16_master_writer.F90',
})

CITATION_MAP.update({
    'iceitd.f90:496-500': ['a_i_2d(ii,jl1) = a_i_2d(ii,jl1) - pdaice(ii,jl)', 'v_i_2d(ii,jl2) = v_i_2d(ii,jl2) + pdvice(ii,jl)', 5],
    'iceitd.f90:575-595': ['WHERE( a_i_2d(1:npti,:) >= epsi20 )', 'CALL tab_3d_4d( npti, nptidx(1:npti), szv_i_2d(1:npti,:,:), szv_i )', 21],
    'icecor.f90:81-99': ['WHERE( a_i(Nis0-(0):Nie0+(0),Njs0-(0):Nje0+(0),:) >= epsi20 )', 'IF( jpl > 1 )   CALL ice_itd_reb( kt )', 19],
    'packages/ice/legoesm/ice/si3_finalize.py:259-318': [('_materialize_transfer_statements: bool = True,', 1), 'sv_i = transfer3(sv_i, zworkv)', 60],
    'packages/ice/legoesm/ice/si3_finalize.py:470-534': ['def si3_ice_cor(', '_materialize_transfer_statements=not _plant_strip_transfer_barriers,', 65],
    'packages/ice/legoesm/ice/si3_finalize.py:254-311': [('e_i: Field,', 2), 'sv_i = transfer3(sv_i, zworkv)', 58],
    'packages/ice/legoesm/ice/si3_finalize.py:286-290': ['a_i = a_i.at[..., donor].set(', 'jnp.where(active, sr(a_i[..., receiver] + pdaice), a_i[..., receiver])', 5],
    'iceupdate.f90:373-418': ['IF( MOD( kt-1, nn_fsbc ) == 0 ) THEN', 'vtau(ji,jj) = ( 1._wp - at_i(ji,jj) ) * vtau_oce(ji,jj) + at_i(ji,jj) * zvtau_ice', 46],
    'nemo_si3_o8_o16_master_gate.py:171-180': ['15: Kind(', '+ _fields("snwice_mass snwice_mass_b", "F2"),', 10],
    'o8_o16_master_writer.F90:446-457': ['ELSEIF(kind==15) THEN', ('CALL put_f2(u,snwice_mass); CALL put_f2(u,snwice_mass_b)', 1), 12],
    'o8_o16_master_writer.F90:476-486': ['IF(kind==19) THEN', ('CALL put_f2(u,utau); CALL put_f2(u,vtau); CALL put_i2(u,taum)', 2), 11],
    'iceupdate.f90:410-419': [('DO jj = ntsj-( 0), ntej+(  0 ) ; DO ji = ntsi-( 0), ntei+(  0)', 7), ('END DO   ;   END DO', 11), 10],
    'sbc_phy.F90:69': 'pp_cldf = 0.81',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1778-1845': ['steps: Sequence[int] = (1, 2, 100),', '"translation_applied": False', 68],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1849-1875': ['def score_o15_o16(', 'for kind in PROJECTIONS["O15"] + (19, 20):', 27],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1927-1941': ['owned = (', 'o16_tau.update(spec.name, got, expected, kt=kt, category=0)', 15],
    'iceitd.f90:497': 'a_i_2d(ii,jl2) = a_i_2d(ii,jl2) + pdaice(ii,jl)',
    'packages/ice/legoesm/ice/si3_finalize.py:289-290': ['a_i = a_i.at[..., receiver].set(', 'jnp.where(active, sr(a_i[..., receiver] + pdaice), a_i[..., receiver])', 2],
})

_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-1995',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-1990',
    anchors=['#!/usr/bin/env python3', 'raise SystemExit(main())', 1990],
)

# Round 24 adds O15/O16 score controls and their tests. Re-anchor every
# affected round-21 and stopped-round citation before the mandatory map audit.
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-424',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-846',
    anchors=['"""Direct unit and master-record tests for the SI3 O8--O12 0-ULP scoring driver."""',
             'assert calls == ["_cumulative_remap", "_cumulative_remap"]', 846],
)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:22-54',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:23-55')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:57-85',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:58-85')
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:161-184',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:182-204',
    anchors=[('@pytest.mark.parametrize(', 2), ('verify_manifests=False,', 2), 23],
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:187-424',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:208-846',
    anchors=['def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(',
             'assert calls == ["_cumulative_remap", "_cumulative_remap"]', 639],
)
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:254-311',
          'packages/ice/legoesm/ice/si3_finalize.py:255-318',
          anchors=[('e_i: Field,', 2), 'sv_i = transfer3(sv_i, zworkv)', 64])
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:286-290',
          'packages/ice/legoesm/ice/si3_finalize.py:290-294',
          anchors=['a_i = a_i.at[..., donor].set(',
                   'a_i = a_i.at[..., receiver].set(', 5])
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:289-290',
          'packages/ice/legoesm/ice/si3_finalize.py:294-295',
          anchors=['a_i = a_i.at[..., receiver].set(',
                   'jnp.where(active, a_receiver_new, a_i[..., receiver])', 2])
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-1990',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2053',
    anchors=['#!/usr/bin/env python3', 'raise SystemExit(main())', 2053],
)

# --- SI3 round 24 Resume 2 (Decision 187 versioned kind-15 translation) ---
# The reader helper shifts the established O8--O12 scorer anchors; keep the
# complete citation map audited, then register the new translation/O16 spans.
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:281-319',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:297-335')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:322-377',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:338-392',
          anchors=['def verify_master_admission_manifests(',
                   '"rung_record_digests": rung_digests,', 55])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:481-537',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:506-562',
          anchors=['def _verify_record_scalars_against_namelist(',
                   'f"scalar header {sname}={got!r} disagrees with resolved namelist {expected!r} in kind {kind}"', 57])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:548-638',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:573-662',
          anchors=['class FieldScoreAccumulator:', '"fields": self.stats,', 90])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:655-1077',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:680-973',
          anchors=['@functools.lru_cache(maxsize=8)',
                   '"jit_prod_step_ponds": jit_prod_step_ponds,', 294])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1166-1901',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:977-1697',
          anchors=['def score_o8_to_o12(', '"rungs": rungs_report,', 721])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1217-1234',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1027-1044')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1392-1625',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1201-1434')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1514-1539',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1323-1348')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1627-1646',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1436-1455',
          anchors=['# 3. Rung O10: Kind 6 -> Kind 7 (si3_ice_thd_sal) and Kind 7 -> Kind 8 (si3_ice_thd_temp)',
                   'spec.name, r7["fields"][spec.name], f8[spec.name], kt=kt, category=jl', 20])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1649-1659',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1458-1468',
          anchors=['# 4. Rung O11: Kind 8 -> Kind 9 (si3_ice_thd_da)',
                   'spec.name, r8["fields"][spec.name], f9[spec.name], kt=kt, category=jl', 11])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1662-1738',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1471-1543',
          anchors=['# Production entry point: si3_column_step_arrays (Kind 3 -> Kind 4, 5, 6, 7, 8, 9)',
                   'prod_trace.post_sal.S_layers[:npti],', 73])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1740-1772',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1549-1580',
          anchors=["# 5. Rung O12: Kind 10 -> Kind 11 (si3_ice_thd_pnd and step_ponds(scheme='si3_lev'))",
                   ('spec.name, f10[spec.name], f11[spec.name], kt=kt, category=0', 2), 32])
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1778-1845',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1826-1899',
    anchors=['def verify_kind15_translation_evidence(', '"translation_kind": 15,', 74],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1849-1875',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1903-2050',
    anchors=['def score_o15_o16(', '"scores": summaries,', 148],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1927-1941',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2015-2030',
    anchors=['tau_meta = records[20]["meta"]',
             'o16_tau.update(spec.name, got, expected, kt=kt, category=0)', 16],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2053',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2147',
    anchors=['#!/usr/bin/env python3', 'raise SystemExit(main())', 2147],
)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-846',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-858',
    anchors=['"""Direct unit and master-record tests for the SI3 O8--O12 0-ULP scoring driver."""',
             'assert calls == ["_cumulative_remap", "_cumulative_remap"]', 858],
)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:182-204',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216')
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:208-846',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:220-858',
    anchors=['def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(',
             'assert calls == ["_cumulative_remap", "_cumulative_remap"]', 639],
)

CITATION_MAP.update({
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:236-269': [
        '_KIND15_V1_SCHEMA_LABELS = ("cloud_fra", "qsr", "sfx")',
        '"writer_labels": list(writer_labels),', 34],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:396-502': [
        'def read_record_verified(', '"reader_translation": reader_translation,', 107],
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:170-191': [
        'def test_kind15_versioned_translation_evidence_and_wrong_permutation_plant() -> None:',
        'score_o15_o16(DEFAULT_MASTER_ROOT, steps=(1,), plant="o16-model-mutation")', 22],
    'packages/ice/legoesm/ice/si3_finalize.py:825-882': [
        'def si3_o16_step(', 'return SI3O16Result(state=state, fluxes=fluxes_out, outputs=outputs)', 58],
})

# Resume 3 rebases O14 and wires the late stages into ``step_sea_ice``.  Keep
# the mandatory full-map audit anchored to the combined source tree.
_reanchor('packages/ice/legoesm/ice/config.py:524-575',
          'packages/ice/legoesm/ice/config.py:524-574')
_reanchor('packages/ice/legoesm/ice/config.py:528-587',
          'packages/ice/legoesm/ice/config.py:528-586')
_reanchor('packages/ice/legoesm/ice/config.py:597-601',
          'packages/ice/legoesm/ice/config.py:603-607')
_reanchor('packages/ice/legoesm/ice/config.py:571-579',
          'packages/ice/legoesm/ice/config.py:577-585')
_reanchor('packages/ice/legoesm/ice/config.py:531-568',
          'packages/ice/legoesm/ice/config.py:531-574')
_reanchor('packages/ice/legoesm/ice/config.py:583-593',
          'packages/ice/legoesm/ice/config.py:589-600')
_reanchor('packages/ice/legoesm/ice/config.py:635-649',
          'packages/ice/legoesm/ice/config.py:641-655')
_reanchor('packages/ice/legoesm/ice/config.py:267-288,311-315,652-727',
          'packages/ice/legoesm/ice/config.py:267-288,311-315,658-730')
_reanchor('packages/ice/legoesm/ice/config.py:311-315,662-727',
          'packages/ice/legoesm/ice/config.py:311-315,668-730')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:628-662',
          'packages/ice/legoesm/ice/sea_ice.py:627-700')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:1717-1795,2992-3000,3228-3387',
          'packages/ice/legoesm/ice/sea_ice.py:1754-1832,3005-3013,3241-3334')
_reanchor(
    'packages/ice/legoesm/ice/sea_ice.py:427-440,482,1745-1779,2396-2421,2477-2543,2829-2877,3014-3094,3228-3349,3394-3396,3513-3531',
    'packages/ice/legoesm/ice/sea_ice.py:420-433,475,1782-1817,2433-2458,2514-2580,2865-2890,3027-3107,3241-3295,3340-3342,3459-3477',
)
_reanchor(
    'packages/ice/legoesm/ice/sea_ice.py:427-440,1745-1779,2829-2877,3014-3094,3228-3349,3394-3396',
    'packages/ice/legoesm/ice/sea_ice.py:420-433,1782-1817,2865-2890,3027-3107,3241-3295,3340-3342',
)
_reanchor('packages/ice/legoesm/ice/sea_ice.py:3026-3065',
          'packages/ice/legoesm/ice/sea_ice.py:3039-3078')
_reanchor('packages/ice/legoesm/ice/__init__.py:266-290',
          'packages/ice/legoesm/ice/__init__.py:264-293')
_reanchor('packages/ice/legoesm/ice/__init__.py:292-316',
          'packages/ice/legoesm/ice/__init__.py:296-320')

_reanchor('packages/ice/legoesm/ice/si3_finalize.py:259-318',
          'packages/ice/legoesm/ice/si3_finalize.py:325-384')
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:470-534',
          'packages/ice/legoesm/ice/si3_finalize.py:536-601')
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:255-318',
          'packages/ice/legoesm/ice/si3_finalize.py:321-384')
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:290-294',
          'packages/ice/legoesm/ice/si3_finalize.py:356-360')
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:294-295',
          'packages/ice/legoesm/ice/si3_finalize.py:360-361')
_reanchor('packages/ice/legoesm/ice/si3_finalize.py:825-882',
          'packages/ice/legoesm/ice/si3_finalize.py:894-954')

_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:297-335',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:321-359')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:338-392',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:362-416')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:506-562',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:533-589')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:573-662',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:600-689')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:680-973',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:707-1000')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:977-1697',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1123-1843')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1027-1044',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1173-1190')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1201-1434',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1347-1580')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1323-1348',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1469-1494')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1436-1455',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1582-1601')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1458-1468',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1604-1614')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1471-1543',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1617-1689')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1549-1580',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1695-1726')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1826-1899',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1973-2046')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1903-2050',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2112-2289')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2015-2030',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2254-2269')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:236-269',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:250-293')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:396-502',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:420-529')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2147',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2391')

_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:23-55',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:25-57')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:58-85',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:60-87')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:88-136',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:91-139')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:139-157',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:142-160')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:187-359',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:261-433')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:194-216',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:235-257')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:220-858',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:261-932')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:170-191',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:195-232')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-858',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-932')

# Pin the resulting combined-tree extents as well as their independently
# resolved endpoints.  Several of the production-wiring changes add code
# inside an already-cited routine, so a line translation alone is not enough.
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:816-823',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:875-882',
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:957-1060',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1016-1119',
)

_repin('packages/ice/legoesm/ice/config.py:524-574', [
    '# Optional ORCA2 5-category 10+5-layer SI3 state configuration (`therm',
    (')', 156), 51])
_repin('packages/ice/legoesm/ice/config.py:528-586', [
    'from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig  # noqa: E',
    (')', 159), 59])
_repin('packages/ice/legoesm/ice/config.py:531-574', [
    'def _validate_si3_orca2_o1_config(config: "SeaIceConfig") -> None:',
    'raise ValueError("invalid SI3 ORCA2 O1 identity: " + "; ".join(conflic',
    44])
_reanchor('packages/ice/legoesm/ice/config.py:589-600',
          'packages/ice/legoesm/ice/config.py:589-599')
_reanchor('packages/ice/legoesm/ice/config.py:267-288,311-315,658-730',
          'packages/ice/legoesm/ice/config.py:267-288,311-315,658-727',
          extent=97)
_reanchor('packages/ice/legoesm/ice/config.py:311-315,668-730',
          'packages/ice/legoesm/ice/config.py:311-315,668-727', extent=65)

_reanchor('packages/ice/legoesm/ice/sea_ice.py:627-700',
          'packages/ice/legoesm/ice/sea_ice.py:627-699', extent=73)
_reanchor('packages/ice/legoesm/ice/sea_ice.py:1754-1832,3005-3013,3241-3334',
          'packages/ice/legoesm/ice/sea_ice.py:1754-1832,3005-3013,3241-3314',
          extent=162)
_reanchor(
    'packages/ice/legoesm/ice/sea_ice.py:420-433,475,1782-1817,2433-2458,2514-2580,2865-2890,3027-3107,3241-3295,3340-3342,3459-3477',
    'packages/ice/legoesm/ice/sea_ice.py:420-433,475,1782-1789,2433-2458,2514-2580,2865-2890,3027-3107,3241-3295,3340-3342,3459-3466',
    extent=289,
)
_reanchor(
    'packages/ice/legoesm/ice/sea_ice.py:420-433,1782-1817,2865-2890,3027-3107,3241-3295,3340-3342',
    'packages/ice/legoesm/ice/sea_ice.py:420-433,1782-1789,2865-2890,3027-3107,3241-3295,3340-3342',
    extent=187,
)
_repin('packages/ice/legoesm/ice/sea_ice.py:3039-3078', [
    ('if config.ponds.enabled and config.ponds.scheme == "si3_lev":', 3),
    ('            )', 31), 40])

_repin('packages/ice/legoesm/ice/__init__.py:264-293', [
    'from legoesm.ice.config import build_si3_orca2_o2_config  # noqa: E402',
    (')', 24), 30])
_reanchor('packages/ice/legoesm/ice/__init__.py:296-320',
          'packages/ice/legoesm/ice/__init__.py:296-319', extent=24)

_repin('packages/ice/legoesm/ice/si3_finalize.py:536-601', [
    'def si3_ice_cor(',
    '_materialize_transfer_statements=not _plant_strip_transfer_barriers,',
    66])
_repin('packages/ice/legoesm/ice/si3_finalize.py:894-954', [
    'def si3_o16_step(',
    'return SI3O16Result(state=state, fluxes=fluxes_out, outputs=outputs)',
    61])

_repin(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:533-589',
    ['def _verify_record_scalars_against_namelist(',
     'f"record identity {sname}={got!r} disagrees with resolved/preregistered value {expected!r} in kind {kind}"',
     57],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:707-1000',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:707-1012',
    extent=306,
)
_repin(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2112-2289',
    ['def score_o15_o16(', '"scores": summaries,', 178],
)
_repin(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:250-293',
    ['_KIND15_V1_SCHEMA_LABELS = ("cloud_fra", "qsr", "sfx")',
     '"writer_labels": list(writer_labels),', 44],
)
_repin(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:420-529',
    ['def read_record_verified(', '"reader_translation": reader_translation,', 110],
)
_repin(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2391',
    ['#!/usr/bin/env python3', 'raise SystemExit(main())', 2391],
)

_repin('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:235-257', [
    ('@pytest.mark.parametrize(', 2), ('verify_manifests=False,', 3), 23])
_repin('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:261-932', [
    'def test_production_call_sites_route_through_o8_o12_and_fail_when_reverted(',
    'assert calls == ["_cumulative_remap", "_cumulative_remap"]', 672])
_repin('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:195-232', [
    'def test_kind15_versioned_translation_evidence_and_wrong_permutation_plant() -> None:',
    'score_o15_o16(DEFAULT_MASTER_ROOT, steps=(1,), plant="o16-model-mutation")',
    38])
_repin('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-932', [
    '"""Direct unit and master-record tests for the SI3 O8--O12 0-ULP scoring driver."""',
    'assert calls == ["_cumulative_remap", "_cumulative_remap"]', 932])

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
# --- SI3 round 22 (O13 exact ITD remap); appended for merge isolation ---
FILES.update({
    'docs/ice/fidelity/si3_r22_o13_score_preregister.md': REPO / 'docs/ice/fidelity/si3_r22_o13_score_preregister.md',
    'docs/ice/fidelity/si3_r22_o13_receipt.md': REPO / 'docs/ice/fidelity/si3_r22_o13_receipt.md',
    'packages/ice/legoesm/ice/itd.py': REPO / 'packages/ice/legoesm/ice/itd.py',
    'packages/ice/legoesm/ice/sea_ice.py': REPO / 'packages/ice/legoesm/ice/sea_ice.py',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o13_minmax_audit.py': REPO / 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o13_minmax_audit.py',
    'tests/ice/unit/test_si3_o13_itd_remap.py': REPO / 'tests/ice/unit/test_si3_o13_itd_remap.py',
    'tests/ocean/fidelity/test_nemo_si3_o13_minmax_audit.py': REPO / 'tests/ocean/fidelity/test_nemo_si3_o13_minmax_audit.py',
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90': NEMO / 'cfgs/ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90',
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icevar.f90': NEMO / 'cfgs/ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icevar.f90',
})

CITATION_MAP.update({
    'docs/ice/fidelity/si3_r22_o13_score_preregister.md:1-65': ['# SI3 round 22 — O13 model preparation and master-record scoring preregistration', 'the admitted record, the field set, or the equality bar.', 65],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90:118-365': ['SUBROUTINE ice_itd_rem( kt )', 'END SUBROUTINE ice_itd_rem', 248],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90:368-429': ['SUBROUTINE itd_glinear( HbL, Hbr, phice, paice, pg0, pg1, phL, phR )', 'END SUBROUTINE itd_glinear', 62],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90:432-597': ['SUBROUTINE itd_shiftice( kdonor, pdaice, pdvice )', 'END SUBROUTINE itd_shiftice', 166],
    'ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/icevar.f90:856-894': ['SUBROUTINE ice_var_roundoff( pa_i, pv_i, pv_s, psv_i, poa_i, pa_ip, pv_ip, pv_il, pe_s, pe_i, pszv_i )', 'END SUBROUTINE ice_var_roundoff', 39],
    'packages/ice/legoesm/ice/sea_ice.py:2847-2868': ['exact_si3_itd = (', 'raise ValueError("rn_amax_2d must have the spatial (no-category) state shape")', 22],
    'packages/ice/legoesm/ice/itd.py:835-1099': ['def si3_ice_itd_rem(', ('return result', 2), 265],
    'packages/ice/legoesm/ice/itd.py:802-831': ['def _si3_itd_glinear(', 'jnp.where(active, h_r, 0.0),', 30],
    'packages/ice/legoesm/ice/itd.py:790-799': ['def _si3_o13_fused_melt_min(', 'return _nemo_minsd(_nemo_minsd(hi_max, h_r), neg_dh0)', 10],
    'packages/ice/legoesm/ice/sea_ice.py:3235-3307': [('if not exact_si3_itd:', 1), 'h_il = jnp.where(remap["a_ip"] > 1.0e-20, remap["v_il"] / aip_safe, 0.0)', 73],
    'packages/ice/legoesm/ice/sea_ice.py:3254-3286': ['remap = si3_ice_itd_rem(', 'rt0=si3_cfg.ice_constants.T0,', 33],
    'tests/ice/unit/test_si3_o13_itd_remap.py:294-409': ['def test_o13_production_sea_ice_call_site_fires(monkeypatch) -> None:', ('grid=grid, rn_amax_2d=jnp.full(shape, 0.997),', 3), 116],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o13_minmax_audit.py:1-89': ['"""Exhaustive MIN/MAX/SIGN/SQRT audit of the SI3 O13 remap object code."""', 'sys.exit(main())', 89],
    'tests/ocean/fidelity/test_nemo_si3_o13_minmax_audit.py:65-109': ['def test_committed_o13_object_audit_and_tie_verdicts() -> None:', 'assert all(census[routine][name] for name in names)', 45],
    'tests/ice/unit/test_si3_o13_itd_remap.py:242-257': ['def test_o13_independent_scalar_oracle_true_eager_jit_and_conservation() -> None:', 'np.testing.assert_allclose(after, before, rtol=3.0e-15, atol=3.0e-15)', 16],
    'tests/ice/unit/test_si3_o13_itd_remap.py:260-291': ['def test_o13_jit_arguments_gradient_central_difference_and_zero_edges() -> None:', 'assert bool(jnp.all(jnp.isfinite(edge_grad)))', 32],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:955-985': ['def _score_o13_step(', 'return conservation', 31],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1052-1104': ['def score_o13(', '"rungs": {"O13": {', 53],
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1721-1725': ['# 6. Rung O13: Kind 11 -> Kind 12 (production si3_ice_itd_rem).', 'jit_o13_remap=jit_o13_remap, accumulator=o13_remap_acc, plant=plant,', 5],
})

# Round 22 added the O13 kernel and its conservation scorer.  Re-anchor every
# earlier receipt span whose source moved, preserving independently pinned
# symbols and explicit extents.
_reanchor('packages/ice/legoesm/ice/itd.py:1078-1080', 'packages/ice/legoesm/ice/itd.py:1082-1084')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:1754-1832,3005-3013,3241-3314', 'packages/ice/legoesm/ice/sea_ice.py:1754-1832,3005-3013,3241-3316', extent=164)
_reanchor('packages/ice/legoesm/ice/sea_ice.py:420-433,475,1782-1789,2433-2458,2514-2580,2865-2890,3027-3107,3241-3295,3340-3342,3459-3466', 'packages/ice/legoesm/ice/sea_ice.py:420-433,475,1782-1789,2433-2458,2514-2580,2865-2890,3027-3107,3241-3297,3342-3344,3461-3468', extent=291)
_reanchor('packages/ice/legoesm/ice/sea_ice.py:420-433,1782-1789,2865-2890,3027-3107,3241-3295,3340-3342', 'packages/ice/legoesm/ice/sea_ice.py:420-433,1782-1789,2865-2890,3027-3107,3241-3297,3342-3344', extent=189)
# Current-round anchors are kept separate so the O13 receipt remains auditable
# without weakening any historical entry.
_reanchor('packages/ice/legoesm/ice/itd.py:835-1099', 'packages/ice/legoesm/ice/itd.py:823-1079', extent=257)
_reanchor('packages/ice/legoesm/ice/itd.py:802-831', 'packages/ice/legoesm/ice/itd.py:790-819')
_reanchor('packages/ice/legoesm/ice/itd.py:790-799', 'packages/ice/legoesm/ice/itd.py:1088-1092', extent=5)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:955-985', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1036-1066')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1052-1104', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1069-1121')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1721-1725', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1845-1849')

# The lane landed O14 before O13 was replayed.  Re-anchor the shared scorer
# and its tests to the combined O8--O14 layout; each rung keeps its own exact
# entry points and the historical O8--O12 spans still identify the same code.
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2391',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1-2588',
    anchors=['#!/usr/bin/env python3', 'raise SystemExit(main())', 2588],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:533-589',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:540-596',
    anchors=['def _verify_record_scalars_against_namelist(',
             'f"record identity {sname}={got!r} disagrees', 57],
)
_reanchor(
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:707-1012',
    'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:714-1032',
    anchors=['@functools.lru_cache(maxsize=8)',
             '"jit_prod_step_ponds": jit_prod_step_ponds,', 319],
)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1173-1190', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1287-1304')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1347-1580', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1463-1696')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1469-1494', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1585-1610')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1582-1601', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1698-1718')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1604-1614', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1720-1730')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1617-1689', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1733-1805')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1695-1726', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1811-1842')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1123-1843', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1236-1973', extent=738)
_reanchor(
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-932',
    'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:1-1151',
    anchors=['"""Direct unit and master-record tests for the SI3 O8--O14 0-ULP scoring driver."""',
             'assert cited <= planted', 1151],
)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:25-57', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:27-59')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:60-87', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:62-91', anchors=['def test_master_admission_manifests_and_plants() -> None:', ('verify_manifests=False,', 1), 30])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:91-139', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:94-149', anchors=['def test_score_o8_to_o12_representative_steps_0_ulp() -> None:', 'assert set(rungs["O13"]["conservation"]["fields"]) == set(O13_CONSERVED_FIELDS)', 56])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:235-257', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:245-271', anchors=[('@pytest.mark.parametrize(', 2), ('verify_manifests=False,', 3), 27])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:261-932', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:274-964', extent=691)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:875-882', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:882-890', anchors=[('@jax.jit', 11), ('constants=constants,', 9), 9])
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1016-1119', 'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1129-1232', anchors=['def score_o14(', ('"resolved_namelist": namelist,', 2), 104])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:142-160', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:152-170', extent=19)
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:261-433', 'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:274-446', extent=173)

# O13 and O14 add source inside several spans that O15/O16 had already
# re-anchored.  Resolve those composed translations against the final lane
# tree, retaining exact endpoints and expanding only spans that now enclose
# the newly landed rung implementation.
_reanchor('packages/ice/legoesm/ice/sea_ice.py:3039-3078',
          'packages/ice/legoesm/ice/sea_ice.py:3071-3110',
          anchors=[
              ('if config.ponds.enabled and config.ponds.scheme == "si3_lev":', 3),
              ('            )', 33), 40])
_reanchor('packages/ice/legoesm/ice/__init__.py:264-293',
          'packages/ice/legoesm/ice/__init__.py:266-295')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:321-359',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:328-366')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:362-416',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:369-423')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:600-689',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:607-696')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1973-2046',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2159-2232')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2112-2289',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2298-2475')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2254-2269',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:2440-2455')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:250-293',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:257-300')
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:420-529',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:427-536')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:195-232',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:205-242')
_reanchor('packages/ice/legoesm/ice/config.py:267-288,311-315,658-727',
          'packages/ice/legoesm/ice/config.py:267-288,311-315,658-730', extent=100)
_reanchor('packages/ice/legoesm/ice/config.py:311-315,668-727',
          'packages/ice/legoesm/ice/config.py:311-315,668-730', extent=68)
_reanchor('packages/ice/legoesm/ice/sea_ice.py:627-699',
          'packages/ice/legoesm/ice/sea_ice.py:635-707')
_reanchor('packages/ice/legoesm/ice/__init__.py:296-319',
          'packages/ice/legoesm/ice/__init__.py:298-321')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:2847-2868',
          'packages/ice/legoesm/ice/sea_ice.py:2892-2913')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:3235-3307',
          'packages/ice/legoesm/ice/sea_ice.py:3280-3352')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:3254-3286',
          'packages/ice/legoesm/ice/sea_ice.py:3299-3331')
_reanchor('packages/ice/legoesm/ice/sea_ice.py:1754-1832,3005-3013,3241-3316',
          'packages/ice/legoesm/ice/sea_ice.py:1762-1840,3037-3045,3273-3434', extent=250)
_reanchor(
    'packages/ice/legoesm/ice/sea_ice.py:420-433,475,1782-1789,2433-2458,2514-2580,2865-2890,3027-3107,3241-3297,3342-3344,3461-3468',
    'packages/ice/legoesm/ice/sea_ice.py:427-440,482,1790-1796,2441-2466,2522-2588,2874-2922,3059-3139,3273-3396,3441-3443,3560-3578',
    anchors=[
        'if not (config.ponds.enabled and config.ponds.scheme == "si3_lev") and config.itd_remap not in ("simple", "lipscomb2001"):',
        'if config.n_categories > 1 and not (config.ponds.enabled and config.ponds.scheme == "si3_lev"):',
        'state = slab_to_dynamic(state, pond_scheme=config.ponds.scheme if config.ponds.enabled else "cesm")',
        ('if h_il is not None:', 1), (')', 575),
        ('if not config.ponds.enabled and config.ponds.scheme != "cesm":', 3),
        'h_il_dil = h_il * dilute if _use_si3_lev else None',
        'if config.ponds.scheme == "si3_lev":', ('h_il_new = h_il_dil', 1),
        ('if not config.ponds.enabled and config.ponds.scheme != "cesm":', 4),
        ('conc, h, h_snow, pond_depth, h_il=h_il)', 1),
        ('if is_multicat and config.itd_remap != "none" and not exact_si3_itd:', 1),
        'h_il = jnp.stack(hil_list, axis=-1) if h_il is not None else None',
        'if is_multicat and config.itd_remap != "none":',
        ('conc, h, h_snow, pond_depth, h_il=h_il)', 4),
        ('if is_multicat and config.itd_remap != "none" and not exact_si3_itd:', 2),
        ('conc, h, h_snow, pond_depth, h_il=h_il)', 5),
        ('new_state = DynamicSeaIceState(', 2), ('    )', 198), 391],
)
_reanchor(
    'packages/ice/legoesm/ice/sea_ice.py:420-433,1782-1789,2865-2890,3027-3107,3241-3297,3342-3344',
    'packages/ice/legoesm/ice/sea_ice.py:427-440,1790-1796,2874-2922,3059-3139,3273-3396,3441-3443',
    extent=278,
)
_reanchor('scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1698-1718',
          'scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py:1698-1717')
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:62-91',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:62-90',
          anchors=['def test_master_admission_manifests_and_plants() -> None:',
                   ('verify_manifests=False,', 1), 29])
_reanchor('tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:245-271',
          'tests/ocean/fidelity/test_nemo_si3_o8_o12_master_scorer.py:245-270',
          anchors=[('@pytest.mark.parametrize(', 2),
                   ('verify_manifests=False,', 3), 26])

UNAUDITED_ROUNDS = ("1-24 -- hand-checked only; their citations are not in "
                    "CITATION_MAP and this gate does not read them")

_NAME = (r"[A-Za-z0-9_./]+\.(?:F90|f90|h90|py|fcm)"
         r"|[A-Za-z0-9_./]*(?:namelist_cfg|namelist_ref|namelist_ice_cfg|namelist_ice_ref|ocean\.output\.excerpt|ocean\.output)"
         r"|cfgs/ref_cfgs\.txt"
         r"|sette/(?:input_ORCA2_SAS_ICE\.cfg|sette_fetch_inputs\.sh)"
         r"|scripts/validate/ocean_fidelity/testcases/[A-Za-z0-9_./]+/run\.sh"
         r"|docs/ocean/fidelity/testcases/(?:nemo_testcases_l3dyn_phase2_receipt|nemo_testcases_l3thd_phase6_receipt|nemo_testcases_l3thd_scalarmath_v2_receipt|nemo_testcases_l4_orca2_phase2_manifest)\.md"
         r"|docs/ice/fidelity/(?:si3_ladder_design|si3_r\d+_[A-Za-z0-9_]+_(?:preregister|receipt))\.md")
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
