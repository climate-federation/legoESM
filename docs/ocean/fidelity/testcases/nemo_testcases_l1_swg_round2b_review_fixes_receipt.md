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

Preregistration commit: `de0047cb11b65ebd4461224cd08f378ae7f9f689`.
Clean measured implementation: `636d428294716af04da0d9f5f3fa8d611b0caf3f`.

1. **CONFIRMED.**  The repaired `ens-stage1` arm reports all 13 recorded
   output groups UNMEASURED, records `production_stage_executed=false`, and
   exits 1.  It creates no candidate by copying a NEMO entry.  Final artifact
   `stage1_unmeasured_final.json` has SHA-256
   `a00ae148bbd1ff8cd900ef7b9c83f56a146dd11f48d1da6fdf34fa1be72997be`.
2. **CONFIRMED.**  The zero input control moves `0.0` to `5e-324`; the one
   input control moves `1.0` by `2.220446049250313e-16`.  Each changes one
   wet U point at `[1,1]`, is non-bit, and is classified DEBT by the
   bit-for-bit SWG ladder.  The underlying relative-only score is retained as
   `numerical_status=AT-BAR`; it cannot override the requested bit endpoint.
   Both control CLIs exit 1.  Their final artifact SHA-256 values are
   `7f5d4aebac7255344fc385874e6591c79bf628cdad3fdc514da9147ad0167086`
   and
   `3e51aad73a4f804bca66bc4bf4d007ec6c5a061991f66a83f19d3f88dc0ad8df`.
3. **CONFIRMED.**  The ENS call explicitly selects direct thickness division.
   The committed tiny-positive-thickness plant produces finite output and
   differs from the historical guarded ENE arm; removing the direct selector
   makes the test red.  The JIT/autodiff ENS test remains finite.
4. **CONFIRMED.**  `ENSVorticityConfig` has two required fields and no
   defaults.  The SWG card supplies `scheme="ens_total"` and
   `ln_dynvor_msk=true`; missing, string, and integer mask selections raise
   before array evaluation.  A second branch test proves `false` is consumed
   rather than silently replaced by the vertex mask.
5. **CONFIRMED.**  The amended round-2 receipt and this round-2b receipt pass
   against the admitted P3 source with respectively five and four mapped
   citations, no unmapped citations, and an empty map audit.  Shifting the
   direct-division citation by two lines exits 1 with
   `SYMBOL-NOT-AT-LINE`.  The final clean and plant artifact SHA-256 values
   are `a3cb144c5a30fca57a5775912b1e1dfa34aa4d854d262eadcee370a6d7c95215`
   and `0a9e9bb2208410d869f094e61483a6651b205624bf9b42fa64332829834c6716`.
6. **CONFIRMED.**  The exclusive ENS configuration object occurs only in the
   SWG builder.  DINO and ORCA2 remain on EEN, GYRE remains on ENE-total, and
   the other certified cards remain on their previous selectors.

## First non-bit statement

No NEMO-versus-legoESM stage output is scored in this fix round.  The first
stage statement that remains unavailable is the production stage-1 velocity
advance at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:187-190`; its 13 target fields are
therefore UNMEASURED, not AT-BAR or DEBT.  The round-2 manufactured ENS
eager-versus-JIT last-bit difference remains a retained REFUTATION, not an
oracle mismatch.

## Rule 12 disposition

The new direct division and required mask selection are reachable only through
SWG's `ENSVorticityConfig`.  The fp64 card hashes remain TSUNAMI
`8bb5ed6c1bc031bd`, VORTEX `f7489467bc156303`, VORTEX_VEC
`5041b91c67bd1443`, GYRE `f02d2f42a570f413`, LOCK_EXCHANGE
`d5dfdf83b3a1e0ab`, and OVERFLOW `83de0010e1170032`, identical to the
operator's pre-fix values.  DINO and ORCA2 use EEN and cannot enter ENS.
No certified AT-BAR row moved and no first-over-bar boundary became earlier.

## Gates

The final CPU battery collected 364 tests: 361 passed.  The only three failures
are the pre-existing stale expected-digest pins for GYRE, LOCK_EXCHANGE, and
OVERFLOW; their actual hashes are the unchanged values above.  Dispatch
hardening contributes 24 green tests.  Geometry remains 34/34 bit-identical;
the signed-y Coriolis, wet-cell, and layer-thickness plants each exit 1 and
change respectively 84, one, and one rows.  Both one-ULP input controls, the
UNMEASURED ladder, and the shifted-citation plant exit nonzero.

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

1. The operator runs the independent re-review outside the sandbox.  Round 2b
   remains HELD until that verdict is SHIP.
2. Only after SHIP, round 3 may implement the complete SWE three-stage
   explicit-free-surface program and replay the stage-1 production output.
3. Stage-3-only forcing, the chained ten-step ladder, 30 days, and shipped-run
   statistics remain unmeasured.

No acquisition or configuration/carried-state decision is requested.
