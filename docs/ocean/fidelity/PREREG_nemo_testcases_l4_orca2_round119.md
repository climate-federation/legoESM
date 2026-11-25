# Preregistration — ORCA2 round 119 V EEN recurrence walk

Date: 2026-10-03. Base: `ec024f3ca832988cc3ee2593b0ba138d2d1f79f1`.
All ocean numbers in this round are **independent**: hierarchy rung 0 starts
from NEMO's own from-rest state. No model physics, card field, configuration
value, carried state, threshold, stabilizer, sea-ice selector, or
`unmeasured_features` entry may change before the source-ordered walk names a
bit-exact candidate.

## Admitted boundary

Rounds 115--117 completed the four U-grid EEN recurrences. Northwest and
northeast differ only in exact-zero accumulator signs; southwest and southeast
first differ in 68 southern-row neighboring V-mask magnitudes and then in
exact-zero accumulator signs. NEMO's constant-zero southern V-mask association
plus ordinary IEEE zero addition closes every recorded U field.

Round 118 committed the rank-complete recorder for all four V recurrences. The
operator reports `PASS_R118_EEN_V_ADMISSION`; this round independently reruns
that admission before interpreting the record. The compiled rung-0 source
evaluates northwest, northeast, southwest, then southeast V. Each recurrence
adds live V thickness, neighboring live U thickness, the neighboring U mask,
and the source-ordered `zpvo` coefficient
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1324-1327`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R119-P1 | The round-118 instrument is passive and complete. | Admission reports exactly-once two-rank coverage; bit-exact product, recurrence, terminal accumulator, inherited streams, and twenty restart shards; every runtime plant fires. | Any moved byte, missing rank, replay failure, or plant staying green rejects the record; quote no operand number. |
| R119-P2 | Northwest V first differs in the source-ordered `zpvo` operand on the northern fold, before thickness, mask, product, or recurrence. | Earlier `mbkv` is exact; `zpvo_nw` has magnitude differences confined to the fold row. | The earliest unequal field in `mbkv`, `zpvo`, live V thickness, neighboring U thickness, or U mask owns the walk instead; preserve this prediction as `REFUTED`. |
| R119-P3 | Northeast V shares northwest's first northern-fold `zpvo` boundary. | Its first unequal field and support match P2's class. | Any earlier or different field/support owns NE independently; do not combine the two northern paths. |
| R119-P4 | Southwest and southeast V have bit-exact operands and stored products; their first differences are exact-zero carried accumulator signs. | `mbkv`, `zpvo`, both thicknesses, mask, and product are bit-exact; `before`/`after` differ only by zero sign. | Any upstream unequal bit owns that path. Any magnitude movement forbids the IEEE-zero arm. |
| R119-P5 | Ordinary host IEEE zero addition closes every recurrence whose earlier operands and product are exact, without moving a nonzero value. | Candidate recurrence fields are bit-exact and candidate-vs-baseline movements are signed-zero-only. | Any remaining bit or nonzero movement refutes arithmetic sufficiency and requires another source-side discriminator. |
| R119-P6 | The northern V magnitude debt remains a measured accumulator/final-scale cancelling pair and does not land with the zero-recurrence statement. | The walk leaves NW/NE magnitude debt before recurrence; no partial production edit is made. | If a single source-exact association closes the complete NW/NE coefficient without worsening a certified row, register it for the full landing gates rather than asserting a pair. |

## Measurement and landing bar

Assemble the self-describing rank shards over the 148 x 180 global domain and
compare fields in compiled order: `mbkv`, `zpvo`, live V thickness, neighboring
live U thickness, neighboring U mask, stored product, accumulator before, and
accumulator after. Count unequal bit patterns separately from unequal numeric
magnitudes and print the first global `(j,i,k)` for each first boundary.

The measurement uses CPU production JIT, fp64 policy, x64, and libm. Its gate
locks every census and carries oracle-bit, candidate-bit, and scope-route
plants. No production statement lands unless the complete candidate is
bit-exact and the ORCA2 rung-0/rung-7, GYRE, DINO, tank, generic-card,
citation, and push gates all pass. Otherwise the round is `HELD` at the first
non-bit statement and leaves the later 68-cell substep-2 U residual open.
