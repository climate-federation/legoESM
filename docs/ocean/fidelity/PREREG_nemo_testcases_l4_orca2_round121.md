# Preregistration — ORCA2 round 121 northern-V EEN fraction walk

Date: 2026-10-03. Base: `74fd18c2bd1d377e5ca6192291cbaa4f0192d1dd`.
Every ocean number in this hierarchy-rung-0 walk is **independent** because
rung 0 starts from NEMO's own from-rest state. No model physics, card field,
configuration value, carried state, stabilizer, sea-ice selector, or
`unmeasured_features` entry may change in this round.

## Admitted boundary

Round 119 found that northwest and northeast V first differ in 1,431 completed
`zpvo` magnitudes on the northern-fold row. Round 120 committed an additions-
only, two-rank record of the three source-ordered fractions and their `ff_f`,
`e3f_0vor`, `r3f`, and `fe3mask` operands. The record now exists but has not
yet been admitted or interpreted.

The compiled rung-0 program forms northeast fraction 1 from `(ji,jj+1)`, then
fractions 2 and 3 from the current row; northwest uses the current row for
fractions 1 and 2 and `(ji-1,jj+1)` for fraction 3
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1323-1328`).
The walk therefore scores NE and NW separately in literal fraction order,
then scores the first unequal fraction's operands in the recorded expression
order. It does not infer a fold permutation from the final sum.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R121-P1 | The round-120 recorder is observationally passive. | All twenty kt=1..10 restart shards and every inherited round-105/107/110/116/118 stream are byte-identical to round 118. | Any moved byte rejects the record; quote no fraction. |
| R121-P2 | The stream is rank-complete and arithmetically sufficient. | Both ranks cover the `148 x 180` domain exactly once; denominators, quotients, ordered sums, and admitted final sums replay bitwise; every runtime plant fires. | Any missing rank, malformed field, arithmetic failure, or surviving plant rejects the record. |
| R121-P3 | NE first differs at fraction 1 and NW first differs at fraction 3, the two fractions that read `jj+1`. | Fractions before those boundaries are bit-exact and each first magnitude difference is confined to global `j=147`. | Preserve the failed half as **REFUTED** and name the earlier observed fraction. |
| R121-P4 | The first unequal fraction's first unequal recorded operand is confined to the northern-fold row. | Scoring `ff_f`, `e3f_0vor`, `r3f`, then `fe3mask` finds no non-fold magnitude support before the fraction. | Any non-fold support stops fold interpretation and becomes the first boundary. |
| R121-P5 | No rank seam owns the difference. | Unequal support crosses the owned-rank boundary and is not confined to either rank edge. | A rank-edge-only support stops interpretation and requires a decomposition audit. |
| R121-P6 | No production statement lands from a measurement-only walk. | No `packages/` file changes and no trajectory moves. | Any model change requires every shared landing gate or is reverted before close. |

## Measurement and landing bar

Admission runs the committed round-120 checker and all of its header, schema,
arithmetic, inherited-stream, and restart plants. The round-121 walk must use
that self-describing parser, assemble each rank exactly once, set fp64/libm on
CPU with production JIT, refuse a dirty or wrongly stamped worktree, lock the
resolved card census, and carry independent one-bit plants for the oracle and
candidate paths plus a scope-route plant.

This is a measurement-only round. After the first unequal operand is named,
the next round walks its compiled northern-fold association. The northern
accumulator/final-scale cancelling pair and later 68-cell substep-2 U residual
remain downstream.
