# NEMO SWG level-1 round 2 — ENS vorticity and stage-1 boundary receipt

Date: 2026-10-11. Base: `8e0c896f0b1b3573e57b4c173d50ebccc5e60721`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round2`.

## Preregistration

This section is committed before reading a numerical value from the admitted
stage-entry payload or running a model measurement.

Round 2 lands only the first source-order missing operator: NEMO's total-ENS
vorticity term with `ln_dynvor_msk = T`.  The compiled five-argument stage-1
call dispatches to `vor_ens(..., ntot, ...)` at
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:250-251`.  The operator builds relative
vorticity, masks only that relative part, adds `ff_f`, divides by the live F
thickness, and applies the Sadourny enstrophy stencil at
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.  The card will select this as
one guarded shared-lane option; an unknown vorticity selection must still
raise.  The other two `SWG_UNMEASURED` entries remain refused.

Frozen predictions and falsifiers:

1. A source-literal manufactured operator case will reproduce the compiled
   ENS U and V expressions bit for bit.  Replacing ENS by the existing ENE or
   EEN stencil, masking the planetary term, or assigning the wrong sign to V
   **REFUTES** the prediction.
2. Eager and JIT execution will be bit-identical in fp64, and gradients through
   both velocity inputs will be finite.  A JIT difference or non-finite
   gradient **REFUTES** the lane requirement.
3. NEMO stage 1 of `kt=1` starts from the shipped zero SSH and zero velocity;
   stage-3-only wind, drag, and viscosity are absent from this stage
   (`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140-173`, versus the stage-3 terms
   at `:260-283`).  Therefore the admitted stage-2 entry for the same step is
   predicted bit-identical to the stage-1 entry for `ssh_k`, `u_k`, `v_k`,
   `r3t_k`, `r3u_k`, `r3v_k`, and `r3f`.  Any unequal wet value or any row over
   the `1e-15` relative bar **REFUTES** the prediction.  A planted wet U value
   must leave the bar and make the gate exit nonzero.
4. No existing certified card selects the new ENS option.  Their focused
   hashes and tests must remain unchanged; an AT-BAR row leaving the bar or an
   earlier first-over-bar boundary **REFUTES** Rule 12 eligibility.

No configuration or carried-state choice is made: `ln_dynvor_ens = T` and
`ln_dynvor_msk = T` are shipped SWG deck statements already frozen on the
round-1 card.

## Measurement

Pending preregistration commit.

## First non-bit statement

Pending measurement.

## Rule 12 disposition

Pending measurement and regression gates.

## Independent review

Independent review run by the operator.

## OPEN

Pending round-2 measurement.
