# Preregistration: ORCA2 round 228 — fold-invariant audit

Date: 2026-10-10. Frozen base: `b6daf6603`. Scope: Decision 113's
measurement-only fold-invariant audit of the complete round-217 vector unit on
independent rung 0 and OMT-4, each under the independent and given-NEMO-entry
labels. This file is committed before any round-228 trajectory or invariant
measurement.

No model statement, configuration, carried state, stabiliser, threshold,
sea-ice selector, or `unmeasured_features` entry may change. OMT-5 remains
blocked. Execution is CPU, production JIT, fp64/libm. The audit reuses the
already-certified live-stage side output; no new in-executable observer is
permitted.

## Source boundary and audit scope

NEMO associates `ua`, `va`, `hu`, `hv`, `hur`, `hvr`, and `ssha` in one
post-substep call at compiled
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:747-756`.
The active T-pivot fold rules are the T/W, U, V, and F cases at compiled
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/lbcnfd.f90:619-718,945-993`.
NEMO then exchanges stage `u`, `v`, `T`, and `S` after every RK3 stage at
compiled `ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:775-798`
before rebuilding the next stage's transports at `stprk3_stg.f90:261-284`.

The current compact-grid association folds U velocity and V velocity/depth/
inverse depth, but only cyclically closes U depth/inverse depth and returns SSH
unchanged at `barotropic_latlon_cgrid.py:689-744`. The current RK3 program uses
the stage-corrected velocities and tracers directly at
`ocean_model_latlon_cgrid.py:8446-8478,8524-8543,8655-8660`. These are
inspection findings, not trajectory attribution.

At every stage of kt=1..7 and stages 1-2 of kt=8, the committed audit will
score only the northern fold band (pivot/right half and the three southern
neighbour rows). For `eta`, `r3t`, `e3t`, `e3w`, `T`, `S`, `u`, `v`, `H_u`,
`H_v`, `r1_H_u`, `r1_H_v`, `zFv`, and `ww`, it records exact fold-identity
violations, maximum absolute residual, argmax, first non-finite, and
`D = max|unit-ON - NEMO| - max|unit-OFF - NEMO|`. No global RMS or off-fold
aggregate may support the verdict.

## Frozen predictions and falsifiers

1. **R228-P1 — NEMO guard is valid.** CONFIRM: every admitted NEMO rung-0 and
   OMT-4 stage through kt=10 has finite SSH-derived `r3t`/`e3w`, with
   `min(e3w_int) > 0`. REFUTE: any admitted NEMO stage is non-finite or has
   non-positive `e3w_int`; then the guard, not the unit, is the finding.
2. **R228-P2 — incomplete post-substep association is visible at its first
   acting boundary.** CONFIRM: NEMO and unit-OFF have zero relevant T/U/V/F
   fold residual at the first boundary, while unit-ON has a nonzero pivot/right-
   half residual in `eta`, `H_u`, or `r1_H_u`, and that residual grows
   monotonically toward kt=8. REFUTE: every listed unit-ON fold residual stays
   bit-exact through kt=8 stage 2, or NEMO/unit-OFF already violates the same
   identity.
3. **R228-P3 — missing per-stage exchange is separately visible.** CONFIRM:
   after a completed stage, unit-ON first introduces a nonzero fold residual in
   `u`, `v`, `T`, or `S`, followed at the next consumer by the same-location
   residual in `zFv`, `ww`, `r3t`, `e3t`, or `e3w`; NEMO's corresponding
   residual is zero. REFUTE: the stage fields and all named consumers remain
   fold-exact under unit-ON through kt=8 stage 2.
4. **R228-P4 — classification.** If P2 confirms at the first acting boundary
   and its maximum grows monotonically, classify the missing association as
   **CONFIRMED OWNER CANDIDATE** for round 229, not as a landed fix. If unit-ON
   is worse than unit-OFF immediately but the exact missing-association
   signature is absent, classify the round-217 unit as wrong-from-first-step
   and diagnose its four parts only on kt=1. Otherwise classify the result
   **HELD / UNRESOLVED**.
5. **R228-P5 — controls.** Plants for the NEMO guard, fold permutation/sign,
   first-boundary ordering, monotonic-growth verdict, and label/card coverage
   must each refuse. The live-stage trace must reproduce an ordinary run's
   completed states bit-for-bit before any trace field is read.

## Stop conditions

Any missing admitted frame or non-passive trace yields
`STOPPED_FOR_RECORD`; values are never inferred from another deck or label.
Regardless of outcome, round 228 makes no package change and ends `HELD` with
the measured table and a round-229 landing plan only if the preregistered owner
predicate confirms.

ASKED choices: Decisions 103, 109, 113, and standing Decision 96. UNASKED
choices: empty.
