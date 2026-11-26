# Preregistration — ORCA2 round 120 northern-V EEN fraction acquisition

Date: 2026-10-03. Base: `a83abe5510481c8855017b86489d015c38563fd6`.
Every eventual ocean number from this hierarchy-rung-0 record is
**independent** because rung 0 starts from NEMO's own from-rest state. No model
physics, card field, configuration value, carried state, stabilizer, sea-ice
selector, or `unmeasured_features` entry may change in this round.

## Admitted boundary

Round 119 admitted and walked all four V EEN recurrences. Southwest and
southeast close under ordinary IEEE-zero addition. Northwest and northeast
first differ in 1,431 `zpvo` magnitudes, all on the northern-fold row. The
admitted stream contains only each completed `zpvo`, not its three fractions.

The compiled rung-0 program evaluates southeast, southwest, northeast, then
northwest `zpvo`; each is an ordered sum of three `ff_f / (e3f_0vor *
(1 + r3f * fe3mask))` fractions before the NW/NE recurrence statements
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1317-1325`).
Therefore this round records the three northeast and three northwest fractions,
their numerator and denominator operands, partial sums, and final sums on both
ranks. It does not infer a fold permutation from the completed sum.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R120-P1 | The additions-only recorder is observationally passive. | All twenty kt=1..10 restart shards and every inherited round-105/107/110/116/118 stream are byte-identical to round 118. | Any moved byte rejects the record; quote no fraction. |
| R120-P2 | The new stream is rank-complete and arithmetically sufficient. | Both ranks cover the 148 x 180 domain exactly once; all named denominators, quotients, partial sums, and final sums replay bitwise; final sums equal admitted NW/NE `zpvo`. | Any missing rank, malformed self-described field, outside-loop write, or replay failure rejects the record. |
| R120-P3 | NW and NE first differ in one of their three source-ordered fractions on the northern-fold row. | Earlier `mbkv` is exact and the first unequal fraction has magnitude differences confined to global `j=147`. | Any earlier bottom-index difference or non-fold support owns the walk instead; preserve this prediction as `REFUTED`. |
| R120-P4 | No rank seam owns the 1,431-cell differences. | The global support is continuous across the owned-rank boundary and neither first boundary is confined to a rank edge. | A rank-edge-only support stops interpretation and requires an association audit. |
| R120-P5 | No production statement lands from an acquisition alone. | No `packages/` file changes and no trajectory claim is made. | If a model file changes, run every shared landing gate or revert it before the round closes. |

## Acquisition and admission bar

The writer must be additions-only against the admitted round-118 compiled
source, use a new target and absolute pre-created per-rank output path, and
emit a self-describing header whose field names and payload lengths drive the
checker. Admission requires `STOP 0`, exact restart and inherited-stream
identity, exact arithmetic replay, exact two-rank coverage, and firing plants
for header, schema, truncation, duplicate rank, bottom index, denominator,
quotient, ordered sum, inherited final sum, inherited stream, and restart
identity.

If the operator record is absent, the round ends `STOPPED_FOR_RECORD` with the
committed `run.sh`. Once admitted, the next round walks NW and NE independently
through the three fractions and then their first unequal fraction's four
operands. The accumulator/final-scale cancelling pair and later 68-cell
substep-2 U residual remain downstream.
