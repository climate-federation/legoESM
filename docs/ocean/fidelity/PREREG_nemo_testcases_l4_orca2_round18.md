# NEMO testcase Lane 4 — ORCA2 card round 18 preregistration

Date: 2026-09-25

Parent: `75d1ac4ad64b98b9d7d13ac35839135492af85b6`

Status: **PREREGISTERED BEFORE ROUND-18 SCIENTIFIC SCORING.**

Round 17 stopped because its admission required two ranked dump markers in
the single `ocean.output`.  The operator's completed run instead has one
rank-0 marker there and one rank-1 marker in `run.user.stdout.log`, while both
rank-tagged binary streams exist at the expected byte count.  The file census
and compiled-writer reconciliation were explicitly required by the handoff
and precede this preregistration; no ocean operand has yet been scored.

Every solver number in this round is labelled **given NEMO's entry**.  The
whole-card ladder remains separately labelled **independent with Decision-52
SSH**.  The six sea-ice selectors and the card's `unmeasured_features` tuple
are frozen.  Decisions 54, 57, and 58 remain pending and untouched.

## Compiled statements and record

The record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round17/acquisition/orca1ice_bt_halo_ranked_np2`.
The executing writer is the record's compiled
`compiled_dynspg_ts.f90:445-455,562-574,845-848`: each MPI rank opens a file
named with `narea - 1`, writes direct `zhU`, `e2u`, `ua_e`, `zhup2_e`, and
`ssumask` arrays for the first two substeps, closes it, and prints its marker
to `numout`.  The record's compiled output manager declares that `lwp` is true
only on the first processor unless all-rank output is selected at
`in_out_manager.f90:180`; rank 1 therefore retains unit 6 and its unguarded
marker is captured by `run.user.stdout.log`, not the rank-0 `ocean.output`.

The arithmetic walk follows the compiled source order:
`dynspg_ts.f90:539-544` forms the U-face depth and
`dynspg_ts.f90:562-569` forms `zhU`, subtracts its left face, and updates sea
surface height.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R18-P1 | The existing run is admissible without rebuild or rerun. | `STOP 0`, `MPIRUN_RC=0`, one correctly ranked marker in each of `ocean.output` and `run.user.stdout.log`, two distinct expected-size streams, headers naming ranks 0 and 1, and both first-two-substep payloads present. | any failed run stamp, duplicate/missing rank, wrong header/size, missing substep, or marker/filename disagreement. |
| R18-P2 | The inherited boundary reproduces before the new record is used. | substep 1 remains bit-exact; substep-2 `continuity_du` remains first non-bit at 64 / 8,794 wet T cells with maximum `7.705384632572532e-07`. | any changed order, count, or maximum; stop for instrument drift. |
| R18-P3 | The direct rank-0 `zhU` halo closes the inferred-operand uncertainty. | the recorded direct right-minus-left `zhU` replays the recorded `continuity_du` bit-exactly on all 8,794 wet T cells; the 64 visible differences are the west-edge population already localized in round 17. | direct replay is non-bit or the support/count changes. |
| R18-P4 | Static multiplicands do not own the west-edge transport difference. | rank-0 `e2u` and `ssumask` are bit-exact against the candidate on all directly comparable local cells, including the scored halo support. | either first differs on the 64-cell support; name it as the first non-bit operand and stop the later attribution. |
| R18-P5 | The first dynamic non-bit operand is `ua_e`, before multiplication by face depth and width. | `ua_e` differs on the 64-cell support; replaying the compiled multiplication with recorded `ua_e` removes that support, while substituting only later operands does not. | `ua_e` is bit-exact there, or its isolated substitution does not remove the support; continue in compiled order to `zhup2_e`. |
| R18-P6 | A planted rank/header or one-ULP payload violation fires. | the admission gate refuses a swapped rank header and the arithmetic gate refuses a one-ULP change reaching direct `zhU`. | either planted violation passes. |
| R18-P7 | No production statement lands unless one cited shared-model statement alone becomes bit-exact and passes every landing gate. | absent such proof, no `packages/` diff; ORCA2 remains `LADDER_MEASURED`, kt=10 entry T maximum `3.9430791763114783` on 430,552 cells. | any ungated model change, earlier ladder boundary, moved unregistered row, or changed GYRE output. |

Failed predictions remain **REFUTED** in the receipt.  If R18-P5 is refuted,
the walk continues only in the already-preregistered compiled order and names
the first non-bit multiplicand; it does not rewrite the prediction.

## Controls and stop rules

- Admission is against the existing run only.  No rebuild, rerun, or repair of
  an artifact is allowed.
- Rank-local arrays must be mapped from recorded layout metadata; no longitude
  wrap or halo convention is inferred from output values.
- Every multiplication boundary is replayed in the compiled association before
  attribution.  A directly recorded operand outranks round 17's inversion.
- Any new production change requires the ORCA2 kt=1..10 ladder and both GYRE
  trajectory proofs from the campaign brief.  Otherwise this is a
  measurement-only HELD round.
- The receipt citation gate must pass, and a rigid two-line shift of a rendered
  compiled citation must fail.
- No configuration choice, carried-state change, stabilizer, sea-ice change,
  or action on pending Decisions 54, 57, or 58 is allowed.

## Choices

ASKED: reconcile the failed ranked-halo acquisition, admit the existing record
if the writer produced what it intended, then walk `zhU`, `e2u`, `ua_e`,
`zhup2_e`, and `ssumask` in compiled order.

UNASKED: none.
