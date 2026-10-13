# PREREGISTRATION — grid ladder rung G1, round 5 (gridladder_r5)

Branch tip 67f1edd54. Record: the operator's round-4 acquisition,
.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step (substep streams at kt
1, 2, 17, 31, 41; 13 substeps each).

## What was done BEFORE this freeze (stated, not hidden)

This file is NOT written before all measurement. Done first, uncommitted:
the record was re-admitted (PASS_G1_RECORD); a per-substep arm and an offline
replay arm were added to the ladder instrument and run at kt = 1 ONLY. They
showed: every operand of substep 1 bit-identical up to the velocity update;
the update unequal on 169 U and 160 V faces, all interior, at most 4.3e-19;
NEMO's own statement replays bit for bit from NEMO's record (0 unequal over
13 substeps); the same bracket DIVIDED by the exit depth is unequal on 4,395
face-substeps. So the kt = 1 owner is a diagnosis, not a prediction.
Frozen here, before they are measured: the landing's effect, and everything
at kt 2, 17, 31, 41 and on the 100-step record.

## Measures (round 4's, unchanged)

Bar 1e-15 normalised (shared trajectory gate). "Beyond rounding" for a region
row: max abs error over NEMO's max abs on the same wet points above 1e-12.
Substep rows are scored on NEMO's wet points of the operand's own stagger, in
the compiled order of the VORTEX round-196 walk's table, split fold band
(row by row) / seam / south / interior. Labels never mixed: GIVEN-NEMO-ENTRY
(NEMO's entry state; past kt 1 the six histories are the record's own
loop-entry frame) and INDEPENDENT (the card's own chain).

## Statement to land (item 1)

NEMO's flux-form external velocity update,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:544-562`: the bracket times
the inverse exit depth z1_hu / z1_hv (:550-551), every product and sum
rounded in the written order. A new BarotropicConfig option, default False,
selected by the G1 card only.

| id | prediction | falsifier |
|---|---|---|
| S-P1 | with the option, the kt = 1 velocity-update rows (new.u, new.v) are bit-identical at all 13 substeps, GIVEN-NEMO-ENTRY | any unequal face at any substep |
| S-P2 | the kt = 1 handoff (exit frame) falls from 1,070 unequal sea-level cells to fewer than 100; the remainder is the single-face depth rows seen at substeps 6 to 9 (1.4e-14 m on a 100 m depth) | 100 or more, or a first unequal row that is not a face depth |
| S-P3 | INDEPENDENT 100-step record: the first step where sea level leaves the bar moves from 17 to 29 or later (the fold's arrival) | it stays before 29 |
| S-P4 | the tracer flips of round 4's census (T, S cells differing after the step) fall to zero through step 16 on the INDEPENDENT chain | any step up to 16 with a differing T or S cell |
| S-P5 | kt = 2 and kt = 17, GIVEN-NEMO-ENTRY: no substep row beyond rounding in any region; the first unequal row is at last-place size | a row beyond rounding |

## The fold residual (item 2), kt = 31 and 41

| id | prediction | falsifier |
|---|---|---|
| F-P8 | GIVEN-NEMO-ENTRY, the first substep row beyond rounding lies in the fold band, at substep 1 or in the loop-entry frame, and the seam, south and interior rows are at rounding at that row | first row beyond rounding outside the fold band, or after substep 1 |
| F-P9 | that first row is one of the loop-entry EEN coefficient sets on the pivot row or the row below (round 3's held operand): the coefficients are bit-identical at kt = 1 because the sea level is zero there | all eight coefficient sets bit-identical on the fold rows at kt = 31 |
| F-P10 | if F-P9 fails, the first row is an operand the ORCA2 lane already named: the in-loop fold copy of sea level / velocity (:569-578), the surface V mask with the north-neighbour pair, or the V-point ratio | a fold-band operand none of those covers |
| F-P11 | landing the named fold statement takes the kt = 31 handoff pivot row below 1e-13 and the step-100 sea level error down 100-fold | either not met (then HELD, operand named, nothing landed) |

One cited statement per landing; the 100-step record re-scored after each.

## Erratum (added after the freeze; no prediction changed)

The statement's line numbers above are wrong: I read them off a listing that
started at line 505. In `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90` the
inverse exit depths are at 557 and 558 and the two assignments run from 560 to 568.
