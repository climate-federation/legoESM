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
   (`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140-171`, versus the stage-3 terms
   at `:260-273` and `:281-283`).  Therefore the admitted stage-2 entry for the same step is
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
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.  It reuses the existing ENE
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
3. **CONFIRMED.**  The formal `ens-stage1` arm executes the production ENS
   dispatcher on the zero kt=1/stage-1 card state, requires both resulting
   tendencies to be bit-zero, carries that isolated source result forward, and
   scores it against NEMO's recorded stage-2 entry.  All 13 stored groups are
   bit-identical and AT-BAR, not only the seven frozen state/ratio groups.
   `ens_stage1.json` records `first_kt_over_bar=null` and commit
   `1816a3724f218c8ba696c3bb73a10cfaed6ad224`; its SHA-256 is
   `c0c121878feb6e24a1425ab3955d5f3caccafea30b9a60d5de9a81efbf07632b`.
   The wet-U plant exits 1, reports DEBT at kt=1, one unequal cell `[1,1]`, and
   `max_abs=0.001`; its JSON SHA-256 is
   `e36edd81b606e4d773b3830a2c3c6b3fddc0cc199cfb06af4b054981517a8293`.
4. **CONFIRMED.**  The 160-test focused implementation battery passed before
   the measurement commit.  The post-commit certified-card battery reports 117
   passes and the same three pre-existing stale digest assertions for GYRE,
   LOCK_EXCHANGE, and OVERFLOW: actual digests `f02d2f42a570f413`,
   `d5dfdf83b3a1e0ab`, and `83de0010e1170032`.  This round adds no configuration
   field, changes no non-SWG card builder, and repository selection inspection
   shows no other card chooses `ens_total`.

The stage-1 zero result follows the compiled order: stage 1 zeros and evaluates
the momentum RHS and advances U/V over `rn_Dt/3` at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:140-171`; the wind, drag, and viscosity
terms first appear only in stage 3 at `:260-273`, before that stage's surface
and ratio updates at `:281-283`.  Therefore this is a certified isolated ENS
source replay, not a claim that the complete stage program has landed.

## First non-bit statement

No NEMO-versus-legoESM row in the measured zero-state arm is non-bit: every
record row is identical.  The first non-bit result exposed this round is the
internal eager-versus-JIT evaluation of the nonzero manufactured ENS U/V
updates—the final `pu_rhs`/`pv_rhs` statements in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.  It is the retained refutation
above, not evidence of an oracle mismatch.  The first unlanded statement on the
oracle ladder remains the coupled SWE stage program surrounding the now-landed
vorticity call: the compiled dispatcher calls ENS total vorticity at
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:250-251`, but the card still refuses the
full stage.

## Rule 12 disposition

| Landed statement | Shared site | Cards that can execute it | Disposition |
|---|---|---|---|
| ENS total-vorticity stencil | `pv_flux_ens` plus guarded `ens_total` dispatch | SWG only | Every other card selects AL81, ENE, ENE-total, EEN-total, or EEN-planetary; no other selector reaches the new branch. |
| Source-associated total-vorticity operand seam | Optional argument on shared `pv_flux_ene` | ENE cards use the omitted/default arm | A direct test proves omitted/default versus the explicit historical `zeta+f` input is byte-identical.  All pre-existing ENE operator tests pass. |
| kt=1 stage-1 isolated replay | SWG-named ladder arm | SWG CLI only | It cannot be imported or selected by another card and it refuses to certify the complete RK3 program. |

No previously AT-BAR row moved, the new clean arm has no over-bar row, and its
plant makes the first-over-bar boundary kt=1.  TSUNAMI and VORTEX focused tests
plus the cross-card constructibility suite produced the 117 green results
reported above; the three digest failures exactly reproduce the stale base
pins and are not a new digest.  DINO and ORCA2 select EEN-total, GYRE selects
ENE-total, and the tank/VORTEX families do not select ENS, so none executes the
landed numerical statement.

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
