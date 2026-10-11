# NEMO SWG level-1 round 2b — review-fix receipt

Date: 2026-10-11. Base: `27e7d4f2708c584d026ed7388bb4b01ea81f9656`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round2b`.

## Preregistration

This section is committed before running a post-fix numerical measurement.
Round 2b fixes only the five reproduced blockers in the operator's independent
review of round 2; it does not start the SWE stage-program arm.

Frozen predictions and falsifiers:

1. The previous 13 stage-1 rows are withdrawn as **REFUTED** because they copied
   NEMO's recorded stage-1 entry and discarded legoESM's tendency instead of
   executing a production stage.  Until the `swe_rk3` production stage exists,
   the ladder will report every stage-1 output field **UNMEASURED**, never
   AT-BAR.  Any fidelity row reported AT-BAR or DEBT without a JIT production
   stage output **REFUTES** this repair.
2. The ladder control will move one recorded stage-entry input by exactly one
   `nextafter` step.  Separate zero and one controls will both be classified
   DEBT by the bit-for-bit ladder gate.  A post-computation perturbation, a
   larger move, or either control remaining AT-BAR **REFUTES** the instrument.
3. The ENS production path will use NEMO's direct division by live F-thickness
   at `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:659-660`, without the shared
   ENE thickness floor.  A tiny-positive-thickness plant must distinguish the
   direct result from the guarded result; equality with the guarded result or
   non-finite output on the positive plant **REFUTES** the transcription.
4. `ln_dynvor_msk` will be an explicit required ENS production choice.  The
   SWG card will state `true`, matching the executed compiled branch at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:639-646`; absent, non-boolean, or
   unknown values must raise before array evaluation.  A silent fallback to a
   vertex mask **REFUTES** the configuration boundary.
5. Every SWG receipt citation accepted by the citation gate will resolve under
   the admitted `SWG_OMIP_L1_P3/BLD/ppsrc/nemo` tree.  The shifted-citation
   plant must exit nonzero.  In particular, stage-1 velocity advance must bind
   to `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:187-190`, and stage-3 forcing
   and updates to `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:317-338`.
6. No certified non-SWG card selects ENS.  TSUNAMI, VORTEX, VORTEX_VEC, GYRE,
   LOCK_EXCHANGE, OVERFLOW, DINO, and ORCA2 must retain their previous selector
   and card hashes or focused-test disposition.  Any previously AT-BAR row
   moving, or any first-over-bar boundary becoming earlier, **REFUTES** Rule 12
   eligibility.

No configuration or carried-state choice is introduced: the required mask
value is the shipped SWG deck's already-stated `ln_dynvor_msk = .true.`.

## Measurement

Pending.

## Review findings and fixes

The operator's independent round-2 review verdict was:

> DO NOT SHIP — The claimed fidelity rows do not measure a JIT production replay, and the required one-ULP control cannot fail the bar.

All five reproduced blockers are fixed as preregistered: the copied rows are
withdrawn as REFUTED and replaced by UNMEASURED output rows; the controls move
stage-entry inputs by one ULP at zero and one; ENS uses direct F-thickness
division; its required mask switch has no default; and every SWG citation in
this receipt resolves under the admitted P3 compiled tree.

## Independent review

Independent review run by the operator.

## OPEN

Pending completion of this bounded fix round.  No later SWG arm is started here.
