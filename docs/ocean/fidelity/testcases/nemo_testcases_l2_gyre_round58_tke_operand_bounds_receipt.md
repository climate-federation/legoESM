# NEMO testcase L2 GYRE round 58 — TKE operand acquisition retraction

Date: 2026-09-11. Frozen measurement parent `8524c91dbcf0`; supplied
record producer `21e85d25284de001c575e23f351a45c426ba5e15`; CPU/fp64.
**CALIBRATION FAILED; ACQUISITION RETRACTED; PHYSICS AND TRAJECTORIES
UNMEASURED.** Round 57's preregistration remains the governing contract.

## Calibration-first verdict

The narrow terminal calibration passes: rebuilding the recorded closure
outputs from recorded NEMO `en_post_sweep`, mixing lengths, floors, masks and
`pdlr` gives **avm 0, avt 0, dissl 0 unequal cells** over 18,000 consumed wet
closure cells. Those are NEMO's viscosity/diffusivity/dissipation statements
at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:691-701`; the subsequent
pre-EVD transfer is the literal copy at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdfphy.f90:348-354`.

The wider operand calibration fails before the cumulative model-path walk.
Rebuilding NEMO's Prandtl factor from the record's `rn2b`, `avm_entry`, `sh2`,
`rn_bshear`, `rn_ediff` and `rn_ediss` gives **635 unequal cells** over 17,400
consumed wet Prandtl cells. The reconstructed association and exact-zero
branch are NEMO's own
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:394-412`.
Its critical factor is defined at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:751`.
The repaired gate therefore exits 1 with all four counts:
`prandtl=635, avm=0, avt=0, dissl=0`.

## Root cause and repair

The record's own compiled writer declares all four entry operands
assumed-shape at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:106-110`, then slices
them with global `ntsi:ntei,ntsj:ntej` indices at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/l2_r54_tke.f90:150-155`.
Assumed-shape rebases each dummy lower bound to one. That is wrong for the
reduced arrays: compiled NEMO declares `p_sh2` and `p_avt` on its explicit
no-halo domain while `p_avm` is full-domain at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:183-185`, and allocates
`dissl` on the no-halo domain at
`GYRE_OMIP_L2_P3_SM_R56TKE/BLD/ppsrc/nemo/zdftke.f90:128`.
The old writer consequently recorded misindexed/out-of-bounds `sh2`, `avt`
and `dissl` entry fields. Its row recorder had already preserved explicit
bounds, so its correct `pdlr` exposed the inconsistency.

The writer repair now gives `p_sh2`, `p_avt` and `p_dissl` explicit `A2D(0)`
bounds and keeps `p_avm(jpi,jpj,jpk)` at
`nemo_testcase_l2_gyre_round54_tke_operands/l2_r54_tke.F90:91-95`.
The reader now reconstructs and bit-scores Prandtl plus `avm/avt/dissl` at
`nemo_testcase_l2_gyre_round54_tke_operands.py:47-113`; a one-ULP mutation of
one wet recorded `pdlr` cell is the non-vacuity plant at
`nemo_testcase_l2_gyre_round54_tke_operands.py:289-295`. The corrected writer
passes preprocessing and `gfortran -fsyntax-only`; the gate unit file passes
9/9 tests.

## Preregistered decisions

No source-ordered substitution result and no first exact `avt` statement is
reported: the frozen record failed calibration. Therefore no shared TKE
physics statement was landed, P1 is neither confirmed nor refuted, and P2's
kt3 T/S predictions remain unscored. The decision-35 recorded after arm
remains the only valid before arm, but neither the kt=1..10 ladder nor the
days 1--30 comparison was run. The combined decision-35-to-round-56 floor
attribution remains unmeasured; its one-ULP round-55-to-56 component remains
sub-resolution by itself.

Rule 12 disposition is unchanged: GYRE is blocked on reacquisition;
`nemo_v1`, DINO and ORCA2 remain UNMEASURED-with-spec; LOCK_EXCHANGE and
OVERFLOW do not execute TKE. No configuration, model physics, state,
freshwater pair, #1484 guard, NEMO source, or evidence file was changed.

ASKED: read/review, calibration, confirmed writer/gate repair, tests, receipt.
UNASKED: NEMO acquisition/execution, physics landing, ladder/month scoring,
and cross-card claims. Next admissible action is an operator-produced clean,
commit-stamped record from the repaired writer, followed by this same
calibration before any substitution.
