# Preregistration: NEMO-testcases L2 GYRE round 109 stage-one handoff walk

Date: 2026-09-18. Frozen at incoming lane tip
`5ffef4983dda47b51423fc0c3441de5fe15eaa1c`, before any round-109
measurement or numerical edit. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round109/`.

## Ordered statement and compiled source

Round 108 leaves kt=1 stage-1 U as the first stage output to resume. The
individual HPG, LDF, vorticity, KEG, and ZAD rows are already BIT when driven
from NEMO's recorded entry. This round therefore starts at the handoff after
that operator program and does not continue the downstream TKE/`bn2` walk.

The record-producing compiled program accumulates HPG, LDF, VOR, KEG, and ZAD
into one unchanged `Krhs` slot at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176`, diagnoses its
depth mean into separate `Ue_rhs`/`Ve_rhs` fields without overwriting `Krhs`
at `GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:202-213`, and passes
that `Nrhs` slot to stage 1 at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90:188-202`.

The instantiated GYRE card prints `momentum_advection=vector_invariant`,
`momentum_time_integrator=rk3_ws`, `adaptive_implicit_vertadv=False`, and
binary64 state. In the matching compiled vector branch, stage 1 does not add a
new momentum-advection term: the source says the 3-D RHS was already completed
in `stp_2D`, and the only conditional call at that site belongs to the
non-vector arm
(`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:363-374`). The
stage transport and W constructed immediately before that branch are tracer
inputs in this vector program; the source explicitly says stage-1 momentum
does not use W at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:326-347`.

The next momentum statement that actually consumes the carried RHS is the
vector RK assignment at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:661-675`, followed
by the all-stage reference-depth barotropic correction at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:731-760`.

The existing consolidated round-46/51 stage gate will be extended in place.
No second harness, public selector, configuration choice, or NEMO acquisition
is created.

## Immutable trajectory arm and stage order

The immutable before arm is
`phase3/merge_main_2026-09-17/after/{ladder.json,day_gap.json}`. Its headline
rows are kt2 T/S/U/V `1.4210854715202004e-14` /
`2.1316282072803006e-14` / `2.7377110452773967e-12` /
`3.2849219221489645e-12`, kt3 T/S `1.627497246303733e-4` /
`6.327735185607253e-6`, and day-30 T RMS
`1.2397011296352737e-2` K.

The restored implementation's given-entry kt1 stage-1 U/V rows are 7,620 and
8,460 unequal cells, each with maximum `5.421010862427522e-20`; the consumed
RHS is already unequal in all 17,400 U and 17,100 V wet cells at
`2.0121494123449567e-8`. The held round-97 full-RHS patch made U/V BIT but
failed Rule 12 in 56 rows. This round does not repeat that one-line candidate.
It discriminates the first post-operator transport/W/RK boundary and, only if
the compiled vector branch is confirmed, tests the narrower program that both
carries the full source and omits post-external momentum recomputation that
the compiled stage-1 vector branch does not execute.

## Frozen predictions and falsifiers

**P1 — reproduced baseline boundary.** The unchanged production-JIT arm will
reproduce BIT HPG/LDF/VOR/KEG/ZAD accumulator rows, the 17,400/17,100-cell
consumed-RHS discrepancy at `2.0121494123449567e-8`, and stage-1 U/V counts
7,620/8,460 at `5.421010862427522e-20`. REFUTED if any count or maximum moves.

**P2 — one-boundary handoff walk.** Separate arms will expose, one selected
pair at a time, the full pre-external operator accumulator, the projected
accumulator, the value after the model's post-external transport replacement,
the value after its W/ZAD replacement, the final RK input, the raw RK write,
and the barotropically corrected U/V output. The full accumulator is predicted
BIT. The projected accumulator is predicted to be the first non-bit value in
the current path, with every later current-path RHS boundary retaining the
17,400/17,100-cell `2.0121494123449567e-8` discrepancy. REFUTED if the full
accumulator is non-bit, an earlier selected input is non-bit, or a later
boundary removes that discrepancy.

Each selected boundary will be measured through the production JIT and the
complete production closure with JIT disabled. The RK write will additionally
be measured under an isolated JIT of the shared RK update/correction helpers,
explicitly labelled `isolated-closure JIT`. The production and eager walks are
predicted to name the same first boundary; isolated JIT is predicted BIT when
fed NEMO's recorded full RHS, before velocity, clock, masks, external target,
and reference-depth correction geometry. REFUTED if production eager and JIT
name different boundaries or if the isolated source statement is non-bit.

**P3 — compiled-branch discriminator.** In the vector configuration, the
stage transport/W arrays may feed tracers but cannot be classified as inputs
to the stage-1 momentum write. The gate must print the instantiated selector
and record this source disposition. REFUTED if the resolved card is not the
vector branch or if a compiled call between the completed `stp_2D` RHS and the
RK assignment mutates momentum `Krhs` at stage 1. In that case no candidate is
measured without a frozen addendum.

**P4 — conditional narrower candidate.** Only if P1-P3 hold, the registered
candidate will make the shared vector stage-1 handoff carry the already
computed full momentum source directly to the RK write and will not apply the
model-only post-external transport or W/ZAD momentum replacements in that
branch. The flux-form branch, stage-2/3 programs, tracer transports, W field,
external target, slow forcing, and barotropic correction remain unchanged.

Given NEMO's kt1 stage entry, the candidate is predicted to make the selected
RHS, raw RK, and corrected U/V rows BIT through production JIT and production
eager, with isolated-JIT parity. It is also predicted to leave every unrelated
kt1 stage-1 output bitwise unchanged. REFUTED by one unequal cell in a target
row, by any moved unrelated output, or by a production/eager disagreement. A
failed local proof is restored and retained as a held manifest, not sent to
the trajectory gate.

**P5 — production plant.** A one-ULP change to one finite, nonzero, previously
equal U reference at the first exact production target must add exactly one
unequal cell, print `STATUS PLANT-FIRED`, and exit nonzero. REFUTED if it
perturbs zero, changes no cell or more than one cell, prints PASS, or exits
zero. The plant must run through the complete production step, not only the
isolated closure.

**P6 — Rule 12 and magnitude.** If P4 closes locally, the candidate will run
the complete 954-row ladder and days 1-30 against the immutable before arm.
It is predicted to retain the exact headline kt2 T/S/U/V and kt3 T/S maxima
listed above, keep first-over-bar at kt2 U/V, and improve on the held round-97
candidate's 56 violating rows by reaching zero Rule-12 violations. The
day-30 change is predicted smaller than `1e-10 K`; no direction is claimed
and the candidate is not claimed to own the `1.2397011296352737e-2 K` gap.
REFUTED by one AT-BAR row leaving the bar, an earlier first-over-bar, one
unregistered moved row, one Rule-12 violation, any headline movement, or a
day-30 change at least `1e-10 K`. A refuted candidate remains HELD.

**P7 — other configurations.** DINO selects the shared vector WS-RK3 program,
so a candidate requires its cheapest committed executing-stage gate before
and after; every moved bit is registered and the 96--98% regional-cancellation
warning remains explicit. LOCK_EXCHANGE and OVERFLOW must be checked from
their resolved cards: if either executes this vector handoff, its committed
gate must pass before landing. ORCA2 remains **UNMEASURED-WITH-SPEC**: resolve
its integrator/advection branch, then compare a one-step production stage-1
RHS, raw write, corrected output, transports, W, and next consumed state
before/after from identical fp64 input. No absent-execution arm is called a
pass.

**P8 — instrumentation neutrality.** With the private boundary selector empty,
the restored canonical stage twin must reproduce round 108's complete row
arrays, not only its counts. REFUTED by any default-path bit movement.

## Measurement order and controls

1. Commit this preregistration.
2. Extend the existing consolidated stage gate and existing private live-stage
   trace with a one-boundary selector, production-eager driver, isolated-JIT
   discriminator, and fail-closed production plant. Add direct tests.
3. Reproduce P1 before interpreting a new row. Run the selected boundaries in
   compiled order and stop at the first production non-bit value.
4. If P1-P3 hold, commit a frozen candidate addendum before changing the shared
   numerical branch. Prove P4 and P5; restore immediately if either fails.
5. If the local proof closes, run the canonical ladder, days 1-30, and every
   executing shared-card gate. Land only on a complete Rule-12 pass.
6. Restore the default selector, rerun the canonical stage twin, then run the
   separate read-only Codex refutation pass, citation gate and shifted-citation
   plant, focused tests, and the complete ocean fidelity/unit trees. Quote
   terminal summaries rather than shell status.

## Scope limits

CPU only, fp64, `JAX_ENABLE_X64=1`. No NEMO source is modified and neither
`makenemo` nor `mpirun` is run. No public configuration, default, coefficient,
threshold, carried state, restart schema, year harness, reconciliation gate,
freshwater pair, #1484 guard, or immutable trajectory baseline changes. No
stabilizer is introduced. The downstream TKE/`bn2` diagnostic and every held
manifest stay held unless the registered same-stage candidate is explicitly
reapplied for comparison.
