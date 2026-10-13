# PREREGISTRATION — grid ladder rung G1, round 6 (gridladder_r6)

Branch tip b0fa6a3f5. Record: the operator's round-4 acquisition,
.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step (stage frames at steps
1 to 50; substep streams at steps 1, 2, 17, 31, 41). Nothing was measured
before this freeze: only NEMO's source, the round-5 receipt and the held
round-5 patch were read.

Decision 119 is pending. The G1 card's entry-depth selection is not touched.
Every headline is reported as measured, with round 5's deselected-arm reading
next to it.

## Measures (rounds 4 and 5's, unchanged)

Bar 1e-15 normalised (shared trajectory gate). "Beyond rounding" for a region
row: largest error over NEMO's largest value on the same wet points above
1e-12. Labels never mixed: GIVEN-NEMO-ENTRY and INDEPENDENT. Same program
point on both sides; magnitudes, not only identities.

## Item 1 — the held fold operand and its partner

Read off NEMO before measuring (a reading, to be tested, not a result):
the thickness ratios the solver's Coriolis coefficients are stretched with
are not the bare stencil of `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:268-269`
and `:281-285`. They pass through the north-fold exchange at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:163` (U, V and F points,
sign +1), whose T-pivot rules
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/lbcnfd.f90:639-749`) replace the whole
pivot row of the V and F ratios by the mirrored row below, and the second
half of the U ratio's pivot row by its mirrored first half. Round 5's held
operand (the stencil reading the fold halo as the north neighbour of the top
row) is the half that NEMO computes and then overwrites.

Leaf table, offline numpy from NEMO's record alone, steps 31 and 41, pivot
row and the row below, against NEMO's recorded entry ratios:
L0 the stencil with the cyclic wrap (the card); L1 = L0 with the fold halo
read (round 5's held operand); L2 = L1 with the exchange; L3 = L0 with the
exchange.

| id | prediction | falsifier |
|---|---|---|
| F-P1 | L0 is unequal on the pivot row of the V and F ratios at both steps; L1 leaves a handful unequal (round 5 saw 5 to 8 coefficient faces) | L0 or L1 already 0 unequal |
| F-P2 | L2 is 0 unequal on both rows of all three ratios at both steps | any unequal point |
| F-P3 | L3 equals L2 bit for bit: once exchanged, the halo read is inert | any differing point |
| F-P4 | in production, with the exchange selected, the eight coefficient sets are 0 unequal at steps 31 and 41 (GIVEN-NEMO-ENTRY) | any unequal face |
| F-P5 | loop-exit sea level in the fold band falls at least 10-fold at step 41 (from 2.9e-12 abs) and the 100-step record's step-100 sea level falls at least 10-fold (from 5.30e-08) | either gets worse: HOLD and name the next row |
| F-P6 | the record does not reach step 100 at the bar from this pair alone; the next fold row is the entry face depth on the fold-image V row (round 5's review finding) | the record is at the bar, or another row is first |

Landing rule: the exchange lands, selected by the G1 card only, if F-P2 and
F-P4 hold and no AT-BAR row leaves the bar and no first-over-bar step moves
earlier. If the halo read is inert (F-P3) it is not added: no inert code.

## Item 2 — the step-17 event

Walk from step 1, where the solver loop is already bit-identical and the
stage velocities are not (about 590 faces, round 5). Compiled order, stage by
stage, then the stage momentum statement replayed offline from NEMO's frames.

| id | prediction | falsifier |
|---|---|---|
| L-P1 | the first unequal row at step 1 is a 3-D stage velocity (not sea level, T or S), GIVEN-NEMO-ENTRY, at most 2 last-place units | first unequal row is another field, or larger |
| L-P2 | handed NEMO's stage entry and solver output, that stage's inputs are bit-identical and the difference is made inside the stage momentum update | an input row is already unequal |
| L-P3 | the cause is a written-order difference in the card (a transcription defect), not compiler rounding of an identical order | NEMO's order replayed in numpy does not reproduce NEMO, or the card's order equals NEMO's |
| L-P4 | if landed, the first sea-level step over the bar moves later than 17 (INDEPENDENT) | 17 or earlier |

No stabiliser, no fix that NEMO's source does not state. If every operand is
bit-identical and the written orders agree, the row is registered as a
rounding row with its census and nothing is changed.

## Item 3

Only if the 100-step record is at the bar at every step: the certification
receipt. Predicted NOT reached this round (F-P6).
