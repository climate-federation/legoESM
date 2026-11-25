# Preregistration — ORCA2 round 165 kt=8 vertical-coordinate boundary

Date: 2026-10-07. Frozen base: `f589ec773`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round165/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number will be mixed into the table. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen source order and protocol

Round 164 proved that the complete private V-transport source unit reaches
kt=8 after stages 1 and 2, then the ordinary stage-3 call refuses because a
raw-mesh `e3w_int` operand is non-finite or non-positive. Production was
restored before the round-164 landing.

The compiled rung-0 oracle constructs stage-3 Kmm as the half-step state at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/stprk3_stg.f90:215-245`, then the
vector-invariant program calls `wzv` and the adaptive split at
`stprk3_stg.f90:323-330`. The split divides by the live
`e3w_1d*(1+r3t(Kmm))` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/sshwzv.f90:736-767`.

This round will reuse the existing round-81 scalar-only bad-`e3w` callback
pattern. It observes the already-computed operand without replacing, clipping
or feeding it back. The complete four-arm private candidate remains exactly:
raw reference face depth, seven-array external-mode association, unmasked V
transport and materialised completed `zhV`.

1. Reproduce the unobserved complete-arm refusal through kt=8 and record its
   exact terminal text.
2. Repeat with the passive observer. Require the same last completed
   checkpoint and refusal class, then report the first invalid consumer,
   index, value, invalid count and the source operands that produced it.
3. Split only the first finite-to-invalid producer in compiled order. No
   production physics lands in this round unless that one statement makes the
   complete arm finish and passes the full Decision-96/shared-card predicate.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R165-P1 | The passive observer reproduces round 164's terminal boundary. | Both runs complete kt=7 and kt=8 stages 1-2, then refuse on the same raw-mesh `e3w_int` invariant before stage 3 returns. | The observer advances/retards the failure, changes its class, or the unobserved run no longer reproduces. |
| R165-P2 | The first invalid geometry belongs to the stage-3 Kmm half-step free-surface path, not the finite kt=8 step-entry state. | Step-entry geometry is finite; the callback first reports a non-finite/non-positive live Kmm `e3w` whose source is the half-step ssh/r3t construction. | Step-entry geometry is already invalid, or another earlier producer owns the first invalid value. |
| R165-P3 | One column crosses `1+r3t(Kmm) <= 0`; the raw mesh `e3w_0` remains finite and positive. | The first cell's raw `e3w_0` is positive while the recorded stretch is non-positive/non-finite, and their source-ordered product reproduces the bad value. | Raw `e3w_0` is itself invalid, the product does not reproduce, or the invalidity comes from another operand. |
| R165-P4 | No production or configuration change is eligible from localization alone. | The round ends HELD with package behavior identical to `f589ec773` and names the next single producer boundary. | A cited one-statement correction completes rung 0 and passes every landing gate in this round. |

## Terminal rule

An observer that moves the refusal is rejected as an instrument. Localization
alone does not authorize a fix. No stabiliser, clip, configuration choice,
carried-state change, sea-ice change, bar relaxation or partial source-unit
landing is permitted. Failed predictions remain in the receipt.

ASKED choices: continue round 164's OPEN vertical-coordinate walk. UNASKED
choices: empty.
