# Lane 3b round 20 preregistration — OVERFLOW producer and ORCA2 SI3 inputs

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `e98d8e642effaf83ae46d69d630246ac6dcdcd4c`

State: **PREREGISTERED; NO ROUND-20 NUMERICAL MEASUREMENT HAS RUN.**

## A. OVERFLOW finite/non-finite history discriminator

The frozen observable is the faithful OVERFLOW `kt=1`, WS-RK3 stage-1
velocity score, run with the complete `flux_form/upwind3/nemo_up3` program.
For each integration-line revision between lane-3b round 18
`8b224b0b276` and reconciled tip `03c6e8d96ff7`, the discriminator records
whether all scored and dry/workspace values entering the next UP3 stencil are
finite.  The first transition is then reduced to the first changed source
statement; commit subject alone is not an owner.

Ranked prediction, frozen from source history before the run:

1. the transition is introduced by the integration line's 3-D stage-face
   `umask` identity, because NEMO multiplies the WS-RK3 stage velocity by the
   3-D mask at `stprk3_stg.F90:367,375,382,444` and `dynadv_up3.F90:142-143,
   160,166-176` reads the resulting dry neighbour;
2. if the first finite revision predates that statement, the next candidate
   is the source-rounded QCO face-depth reciprocal guard in the stage
   transport builder;
3. otherwise the result is REFUTED and the exact earlier producer statement
   is reported without restoring the round-18 consumer-side `where` mask.

CONFIRM requires a predecessor/revision pair under the same card, inputs and
gate where the predecessor is non-finite, the revision is finite, and the
diff names a producer statement.  A source-only inference is PLAUSIBLE, not
CONFIRMED.  No new public selector or numerical arm is authorized.

## B. Rule-11 stage-clock retraction

The ownership label in the round-19 receipt is withdrawn independently of
this round's measurements.  GYRE Round 21 measured the tracer-consumed
post-`tra_adv_trp` `ww`: production is AT_BAR at stages 1--3, while changing
only the denominator clock is red at all stages.  The remaining OVERFLOW
stage-2 instantaneous-u row (`1.5711182355104825e-12`) is preregistered as a
boundary with **owner unassigned**.  Round 22 is checked for row identity;
Round 23 is cited only if an artifact exists on the fetched branch.

## C. Combined bottom-plus-top drag association

The frozen source identity is NEMO's single half of the combined neighbour
sums:

`r1_2*((bot_e+bot)+(top_e+top))`

at `dynspg_ts.F90:1611-1612`, with each written assignment guarded at its
statement boundary by `nemo_source_round`.  The accepted lane-16 register
reads the first step having non-zero top drag and compares that literal
association with the NEMO `PRE_DYN_SPG_TS` operand.  Prediction: the merged
tree retains the literal nesting and the focused row remains AT_BAR; bit
identity is reported separately.  The separately halved expression remains
a private discriminator only.

## D. ORCA2 SI3 exact-input rung

The admitted candidate root is
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2q_zdf_schemafix_a_10step_np2`.
Before model execution, every SI3 record is schema-checked against its writer
and the ORCA2 phase-2 manifests.  Required entry operands are the resolved
five-category prognostic state and every atmosphere/ocean exchange operand
consumed by `ice_thd`/`ice_sbc`, with their NEMO time levels registered.  The
score uses the existing shared SI3 thermodynamic column and exchange
operators, cellwise binary64, `0 / n` non-bit target, and stops at the first
non-bit operand/boundary.

Prediction: the retained `oracle_si3_zdf_inputs.bin` and restart shards are
insufficient for the full `ice_thd` entry-to-exit walk unless their schema
also supplies the category state, snow/ice layer temperatures or enthalpies,
salinity, and all surface/basal exchange inputs at the call's actual time
level.  A malformed header, absent field, or unresolved rank-local mapping
fails closed.  In that case no partial thermodynamic verdict is issued; the
result is an exact config-local WRITE-only frame handoff to the ORCA2 lane,
not a new ORCA2 build in lane 3b.  A row plant is required for every admitted
scored boundary and must exit nonzero.

## E. ASKED / UNASKED

| choice or action | status | preregistered disposition |
|---|---|---|
| locate the historical OVERFLOW NaN producer | ASKED | finite/non-finite revision discriminator above |
| retract the stage-clock owner label | ASKED / Rule 11 | owner becomes unassigned; no clock change |
| verify the merged combined-drag association | ASKED | source audit plus focused accepted-row rerun |
| score ORCA2 SI3 from exact retained inputs | ASKED | conditional on complete, valid schemas |
| request exact ORCA2 WRITE-only frames if incomplete | ASKED conditional | handoff only; lane 3b will not instrument ORCA2 |
| implement prognostic `uu_b/vv_b` state | UNASKED / user decision pending | not implemented |
| extend scalar-libm to LOG/LOG10/POW | UNASKED / user decision pending | policy unchanged; hazard retained |
| modify shipped NEMO, build an ORCA2 instrument, delete evidence, GPU, `mpirun`, or push | UNASKED / forbidden | not done |

All measurements initially produced by this round are Codex-internal unless
a branch artifact records an independent reviewer, reviewed commit and
verdict.
