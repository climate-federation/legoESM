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
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:250-251`.  The operator builds relative
vorticity, masks only that relative part, adds `ff_f`, divides by the live F
thickness, and applies the Sadourny enstrophy stencil at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:639-675`.  The card will select this as
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
   (`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:164-190`, versus the stage-3 terms
   at `:317-338`).  Therefore the admitted stage-2 entry for the same step is
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

The preregistration is commit
`3f12f91347e9aa31b86df70aa8b109337450e969`; the clean measured implementation
is commit `1816a3724f218c8ba696c3bb73a10cfaed6ad224`.  The admitted round-1
acquisition supplied all 30 ordered stage-entry frames and had already passed
its passive-output check.

The shared `pv_flux_ens` operator transcribes the compiled `np_CRV` construction
and mask ordering, the live F-thickness division, metric transports, and ENS
U/V sums at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:639-675`.  It reuses the existing ENE
operand builder through an optional pre-associated total-vorticity input; the
ordinary ENE call with that input omitted is bit-identical to the historical
path.  The guarded shared dispatcher accepts `ens_total`; unknown selections
continue to raise.  The SWG card now selects that arm and removes only ENS from
`SWG_UNMEASURED`.  The complete SWE RK3 program and stage-3 forcing remain
listed and the independent card still refuses execution.

Frozen prediction results:

1. **CONFIRMED.**  The manufactured source-literal U and V expressions are
   bit-identical.  The mask plant proves that `ff_f` survives where relative
   vorticity is masked, and ENS differs from ENE.
2. **REFUTED and retained.**  Autodiff gradients are finite and eager/JIT are
   within `1.0e-14` relative, but they are not bit-identical on the nonzero
   manufactured case.  U has 29 unequal values and V has 16; both maximum
   absolute differences are `1.6940658945086007e-21` (maximum-relative
   `1.322847903309725e-16` and `1.122867863194352e-16`).  Routing the curl
   through the NEMO source-ordered implementation reduced the discrepancy but
   did not eliminate XLA's last-bit reassociation.  No claim was relaxed: this
   prediction is red, while the NEMO-versus-legoESM record bar remains exact.
3. **REFUTED and withdrawn.**  The formal `ens-stage1` arm evaluated an isolated
   ENS tendency, discarded it, copied NEMO's recorded stage-1 entry, and scored
   that NEMO copy against NEMO's recorded stage-2 entry.  No recorded input
   entered a JIT production stage.  The 13 reported AT-BAR rows and their
   `ens_stage1.json` artifact are therefore not legoESM fidelity evidence.
   Round 2b changes the arm to report all 13 outputs UNMEASURED until the SWE
   RK3 production stage exists.  Its controls move a stage-entry input by one
   `nextafter` step at zero and at one; both are DEBT under the bit-for-bit
   ladder gate.  The old `+0.001` post-computation plant is retired.
4. **CONFIRMED.**  The 160-test focused implementation battery passed before
   the measurement commit.  The post-commit certified-card battery reports 117
   passes and the same three pre-existing stale digest assertions for GYRE,
   LOCK_EXCHANGE, and OVERFLOW: actual digests `f02d2f42a570f413`,
   `d5dfdf83b3a1e0ab`, and `83de0010e1170032`.  This round adds no configuration
   field, changes no non-SWG card builder, and repository selection inspection
   shows no other card chooses `ens_total`.

The compiled P3 branch advances the stage-1 velocity at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:187-190` and applies stage-3
viscosity, forcing, surface, and QCO updates at `:317-338`.  Because legoESM's
SWE production stage remains unbuilt, round 2 makes no stage-boundary fidelity
claim.

## First non-bit statement

No NEMO-versus-legoESM stage row was measured: the 13 former rows are withdrawn
and now UNMEASURED.  The first non-bit result exposed this round is the
internal eager-versus-JIT evaluation of the nonzero manufactured ENS U/V
updates—the final `pu_rhs`/`pv_rhs` statements in
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:639-675`.  It is the retained refutation
above, not evidence of an oracle mismatch.  The first unlanded statement on the
oracle ladder remains the coupled SWE stage program surrounding the now-landed
vorticity call: the compiled dispatcher calls ENS total vorticity at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:250-251`, but the card still refuses the
full stage.

## Rule 12 disposition

| Landed statement | Shared site | Cards that can execute it | Disposition |
|---|---|---|---|
| ENS total-vorticity stencil | `pv_flux_ens` plus guarded `ens_total` dispatch | SWG only | Every other card selects AL81, ENE, ENE-total, EEN-total, or EEN-planetary; no other selector reaches the new branch. |
| Source-associated total-vorticity operand seam | Optional argument on shared `pv_flux_ene` | ENE cards use the omitted/default arm | A direct test proves omitted/default versus the explicit historical `zeta+f` input is byte-identical.  All pre-existing ENE operator tests pass. |
| kt=1 stage-1 boundary | SWG-named ladder arm | SWG CLI only | REFUTED as a replay; all 13 outputs are UNMEASURED until a JIT production SWE stage exists. |

The 13 alleged AT-BAR stage rows are withdrawn rather than moved; no certified
row moved.  TSUNAMI and VORTEX focused tests
plus the cross-card constructibility suite produced the 117 green results
reported above; the three digest failures exactly reproduce the stale base
pins and are not a new digest.  DINO and ORCA2 select EEN-total, GYRE selects
ENE-total, and the tank/VORTEX families do not select ENS, so none executes the
landed numerical statement.

## Gates

The final focused run combines the ENS/operator, SWG card and ladder, and
citation-gate suites: 177 tests pass.  The clean geometry rerun remains 34/34
bit-identical; its signed-y Coriolis, wet-cell, and layer-thickness plants each
exit 1 and identify respectively 84, one, and one changed values.  The receipt
citation gate finds five mapped compiled-source citations, no failures, no
unmapped citations, and an empty map audit.  Shifting the ENS citation by two
lines exits 1 with `SYMBOL-NOT-AT-LINE`.  `git diff --check` passes and the
tracked worktree is clean.

## Review findings and fixes

The operator's independent round-2 review verdict was:

> DO NOT SHIP — The claimed fidelity rows do not measure a JIT production replay, and the required one-ULP control cannot fail the bar.

Round 2b fixes all five reproduced blockers:

1. The 13 copied NEMO-to-NEMO rows are withdrawn and the ladder now reports
   them UNMEASURED until the JIT production SWE stage exists.
2. The old post-computation `+0.001` plant is replaced by separate one-ULP
   stage-entry input controls at zero and one; the bit-for-bit ladder marks
   both DEBT while retaining their `1e-15` numerical score as metadata.
3. ENS takes the compiled direct F-thickness division; the ENE floor remains
   only on its historical arm, and a tiny-positive-thickness plant separates
   the two by more than nine orders of magnitude.
4. The SWG card selects an `ENSVorticityConfig` whose `ln_dynvor_msk` field
   has no default.  Missing or non-boolean values raise before array evaluation.
5. SWG citations now bind to the admitted P3 compiled tree.  The P3 stage
   statements are re-anchored rather than borrowing line numbers from the
   uninstrumented build.

No later SWE program or stage-3 forcing arm is started in this fix round.

## Independent review

Independent review run by the operator.

## OPEN

1. The operator runs the independent review outside the sandbox.  This receipt
   intentionally records “independent review run by the operator”; no
   in-sandbox service retry was attempted.
2. Round 3 implements the complete SWE three-stage explicit-free-surface
   program as the next single source-order arm, including the `rn_Dt/3` stage-1
   SSH and QCO ratio updates, then replays kt=1/stage 1 against the admitted
   stage-2 entry.
3. Stage-3-only viscosity, mean wind, and before-velocity drag stay unmeasured
   until their source-order round.  The chained ten-step ladder, 30-day run,
   and shipped-run statistics therefore remain closed.

No acquisition or configuration/carried-state decision is requested in this
round.
