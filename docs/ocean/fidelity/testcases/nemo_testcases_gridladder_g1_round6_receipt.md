# Grid ladder rung G1 — round 6 receipt (gridladder_r6)

Status: **LANDED** (four cited statements, G1-card-selected). The rung is
**NOT certified**: the 100-step record is not at the bar. Sea level first
leaves it at step 23 (was 17); the largest error anywhere in 100 steps is
1.5e-14 (was 9.8e-08).
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round6.md
(commit 89eb06cc0, before any measurement). Code: f04fbf0fd (options,
instrument, tests), 3f5198b53 (a rename), dc5c03769 (comment-only, a cited
line), 4af3a4e4a (review fixes). Evidence:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round6/final/.
Record: the operator's round-4 acquisition,
.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step. NEMO was not re-run.

Decision 119 is pending. The card's entry-depth selection was not touched;
every headline below carries the deselected arm's reading next to it.

## Take-aways

1. **The fold is closed.** Round 5's held operand had a partner, and it was
   not the one I preregistered. The pair is (a) NEMO's fold exchange of the
   thickness ratios and (b) the pivot row of one mask, which the shared code
   was overwriting with the wrong row. Together: 0 unequal coefficient faces
   at steps 31 and 41; either alone leaves 5 to 132.
2. **Step-100 sea level: 5.30e-08 -> 1.32e-14.** The fold band's own error
   at step 40 goes from 1.4e-12 to 3.0e-19.
3. **The step-17 event had two transcription causes, both in stage 1 of the
   3-D velocity.** The card handed stage 1 a right-hand side with the
   barotropic part removed, where NEMO hands it whole. And the card's
   compiled update fused a multiply into an add, where NEMO rounds each.
4. **Headline: sea level 17 -> 23, velocities 27 -> 54.** Deselected arm
   (Decision 119): 23, and 59 / 58. Round 5 read 17 and 27 for the two arms.
5. **What still feeds step 23:** stage 1 of step 1 is now NEMO's bit for
   bit. The first unequal rows are stage 2 of step 1 (123 U, 238 V faces,
   last place) and, from step 2, the 3-D right-hand side itself (618 U, 766
   V faces, last place). Neither is replayed yet; no rounding row is
   registered, because no statement is named with bit-equal operands.
6. **One row is large and is not an error in the solution:** the stored top
   row of the 3-D V velocity (4.6e-03 at step 100). Written as NEMO's fold
   image between steps, it goes to 2.6e-16 and no other field moves by one
   bit. Open since round 2; not fixed here.

## 0. Admission and controls

- No new record. Geometry 36 of 36 rows bit-identical (geometry.json).
- The before arm (all four new options off) reproduces round 5's card to
  every printed digit: 17 / 27 / 27 / 19 / 27, step 100 at 5.305e-08.
- Score plant exits 1 (plant_score.log).

## 1. Item 1: the fold pair

### 1.1 Leaf table (offline, NEMO's record alone)

The three ratios on the fold rows, from NEMO's entry sea level, against
NEMO's recorded entry ratios. Unequal points / largest error (NEMO's largest
value on the row).

| variant | V ratio, pivot row, step 31 | step 41 | F ratio, pivot row, step 31 | step 41 |
|---|---|---|---|---|
| L0 bare stencil, cyclic wrap (the card) | 68 / 8.8e-09 (1.7e-08) | 68 / 4.2e-07 (8.2e-07) | 66 / 8.0e-09 (9.3e-09) | 66 / 3.3e-07 (7.5e-07) |
| L1 = L0 reading the fold halo (round 5's held operand) | 20 / 1.7e-24 | 16 / 5.3e-23 | 25 / 1.7e-24 | 21 / 1.1e-22 |
| L2 = L1 then the exchange | 20 / 1.7e-24 | 16 / 5.3e-23 | 25 / 1.7e-24 | 21 / 1.1e-22 |
| L3 = L0 then the exchange | equal to L2 bit for bit | | | |
| L4 = L3 in NEMO's written order | **0** | **0** | **0** | **0** |

- L4 is 0 unequal on every row of all three ratios, about 26,600 points per
  ratio. That is the instrument's control.
- The 16 to 25 left by L1 to L3 are the instrument's own: those variants
  divide by the face area where NEMO multiplies by its inverse
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:268-269`,
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:281-285`). The same order
  leaves about 3,900 last-place points on the rows below. The production
  builder has that order too; it does not reach the coefficients (0 unequal
  below), because the ratio is added to one.
- So at the leaf the halo read and the exchange are not told apart. What
  NEMO executes is the exchange: `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:163`,
  with the T-pivot rules of `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/lbcnfd.f90:639-748`.
  It overwrites the pivot row of the V and F ratios entirely. L3 equals L2,
  so the halo read is inert once exchanged and is NOT added.

### 1.2 The coefficients, four ways (offline, the production builder)

The production builder run on NEMO's entry sea level against NEMO's eight
recorded sets. Unequal wet faces, range over the eight sets; largest error
(NEMO about 6.0e-05). All unequal faces are on the pivot row or the row
below.

| ratio exchange | mask pivot row | step 31 | step 41 |
|---|---|---|---|
| off | off (round 5's card, the live row) | 65 to 132 / 2.7e-13 | 65 to 132 / 1.7e-11 |
| on | off | 5 to 13 / 1.2e-13 | 5 to 13 / 1.7e-11 |
| off | on | 65 to 132 / 3.2e-13 | 65 to 132 / 1.3e-11 |
| **on** | **on** | **0** | **0** |

- The first row equals the live loop table's coefficient rows of round 5 and
  of this round's before arm (67 / 67 / 65 / 65 and 131 / 132 / 67 / 67).
- The faces the exchange alone leaves are columns 58 to 63, 149 and 150:
  round 5's "5 to 8 faces". At exactly those columns the shared builder had
  replaced the pivot row of the mask by the mirrored row two below.

### 1.3 The two statements

| # | NEMO statement | what the shared code did | option |
|---|---|---|---|
| D | the ratios pass through the fold exchange, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:163`: V and F pivot rows are the mirrored row below, the U pivot row's second half its mirrored first half | used the bare stencil on the pivot row, with a cyclic wrap to the southernmost row as north neighbour | barotropic_een_ratio_fold_exchange |
| E | the mask that stretches the reference thickness is the exchanged F mask, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90:232`, frozen at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dommsk.f90:258`: its pivot row is the mirrored row below, which the mesh array already holds | overwrote that row with the halo row's source, two rows below | barotropic_een_fe3mask_pivot_row |

The solver refuses one without the other. Both are BarotropicConfig fields,
default False, appended last.

### 1.4 The loop and the record with the pair

Solver loop, GIVEN-NEMO-ENTRY, NEMO's own histories. Fold band, largest
absolute error:

| row | step 31 before -> after | step 41 before -> after |
|---|---|---|
| eight coefficient sets, unequal faces | 65 to 132 -> 0 | 65 to 132 -> 0 |
| loop exit, sea level, fold band | 1.0e-15 (round 5) -> 1.7e-21 | 2.9e-12 -> 5.4e-20 |
| loop exit, u / v, pivot row | not re-run -> 8.5e-22 / 4.2e-22 | 2.7e-12 / 1.1e-12 -> 1.0e-20 / 8.5e-21 |

The step-41 "before" column is this round's before arm
(sub_kt41_before.json); the step-31 one is round 5's receipt.

Still unequal on the fold-image V row, 68 faces, and not consumed: the
handed V forcing (3.0e-11 of 1.4e-08 at step 31; 1.7e-09 of 1.0e-08 at step
41), the pressure gradient (2.8e-10 of 4.7e-07; 6.5e-09 of 5.7e-07) and the
Coriolis trend (4.4e-11; 2.7e-09). The velocity NEMO builds from them on
that row is overwritten by the fold image inside the loop; the new V on the
pivot row differs on 2 and 3 faces by 2.6e-23 and 3.4e-21.

100-step record, INDEPENDENT, sea level in the fold band (absolute):

| step | before | card |
|---|---|---|
| 31 | 1.0e-15 | 5.1e-21 |
| 40 | 1.4e-12 | 3.0e-19 |
| 41 | 2.7e-12 | 3.4e-19 |
| 100 | 5.3e-08 | 1.2e-15 |

## 2. Item 2: the step-17 event

### 2.1 The first unequal row, compiled order

GIVEN-NEMO-ENTRY, each stage handed NEMO's solver output and NEMO's stage
entry. Unequal interior faces, u / v (round 5's card):

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| stage 1 | 2 / 0 | 65 / 50 | 120 / 145 |
| stage 2 | 194 / 303 | 610 / 805 | 1,141 / 1,439 |
| stage 3 | 586 / 550 | 1,220 / 1,144 | 2,082 / 1,990 |

Sea level, T and S are 0 in every cell of the table. Largest difference
3.5e-18. Counts are interior faces, coast-adjacent plus open (fold, seam and
south are 0).
First row: the stage-1 U velocity of step 1, two open-ocean faces, one
last-place unit. L-P1 HOLDS.

### 2.2 The replay and its control

NEMO's stage-1 statements re-run in numpy from NEMO's recorded operands: the
update `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:373-378`, then the
barotropic correction `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`
and `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:417-418`.

- In the written order, each product and sum rounded once: **0 unequal**
  against NEMO's stage-1 frame, u and v, steps 1, 2 and 3. NEMO's binary
  runs the statements unfused.
- The after ratio is not the cause: taken as the ratio of the interpolated
  sea level instead of the interpolated ratios
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:171-173`) it differs on
  684 points and the velocity is still 0 unequal. My first guess, refuted.

### 2.3 The card's own operands

Write-only exposures of the card's stage 1 from the same entry.

| measure, step 1, u / v | round 5's card | card |
|---|---|---|
| stage-1 right-hand side unequal to NEMO's | 226 / 210, largest 6.6e-09 of 6.6e-09 | 0 / 0 |
| raw stage velocity, before the correction, unequal | 226 / 210, 2.4e-05 of 2.4e-05 | 0 / 0 |
| stage-1 velocity unequal | 2 / 0 | 0 / 0 |

- **Statement F.** NEMO's flux-form stage 1 consumes the 3-D right-hand side
  that the 2-D step left in place: the pressure gradient
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:138`) and the vorticity term
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:144`), whole. The card
  consumed it with the slow barotropic forcing subtracted. On one level that
  is the whole thing: the card's raw velocity was 3.0e-21 where NEMO's is
  2.4e-05. The correction then replaces the depth mean, so the two are equal
  as real numbers and differ in the last place.
- **Statement G.** Handed NEMO's own right-hand side, the card still
  differed (1 / 0, 47 / 34, 93 / 115 faces at steps 1 to 3). Candidate
  replays with one product fused into the following sum moved those counts
  (fused correction: 0 / 0, 20 / 19, 40 / 40) and none reproduced them
  exactly. With every product and sum guarded, the card's stage-1 V raw
  velocity equals the written update on its own right-hand side (0 unequal,
  steps 1 to 3). Which products the compiler fused is not established; that
  the unguarded expression is not NEMO's order is.
- L-P2 REFUTED as written: an input row was already unequal (the right-hand
  side). L-P3 HOLDS: transcription, not rounding.

### 2.4 Landed, and each alone

| # | NEMO statement | option |
|---|---|---|
| F | stage 1 consumes the whole right-hand side of the 2-D step | rk3_stage1_rhs_nemo_literal |
| G | update and correction, every product and sum rounded once | rk3_stage_velocity_written_order |

Both are LatLonCGridOceanConfig fields, default False, appended last.

Stage table, GIVEN-NEMO-ENTRY, unequal interior faces u / v:

| arm | stage 1, steps 1 / 2 / 3 | stage 2, step 1 | stage 3, step 1 |
|---|---|---|---|
| neither (round 5's card) | 2/0, 65/50, 120/145 | 194 / 303 | 586 / 550 |
| F only | 1/0, 46/33, 99/116 | 194 / 303 | 586 / 550 |
| G only | 2/0, 57/47, 106/102 | 123 / 238 | 542 / 462 |
| **F and G (the card)** | **0/0**, 6/4, 13/9 | 123 / 238 | 542 / 462 |

- F acts on stage 1 only; G on all three stages.
- What is left at stage 1 of steps 2 and 3 enters through the right-hand
  side: it differs from NEMO's on 618 / 766 faces at step 2 (largest 7.9e-23
  of 2.1e-07) and 1,578 / 1,765 at step 3. That is round 4's open row.
- One more open row: with G on, the card's stage-1 U raw velocity is not the
  written update of its own right-hand side on 17 and 63 faces (steps 2, 3;
  1.1e-19, 4.3e-19) when the replay uses NEMO's ratios. V is (0). Not
  attributed; the card's U ratio is the untested operand.

## 3. The 100-step record (INDEPENDENT)

First step leaving the 1e-15 bar; largest normalised error over 100 steps.

| card | ssh | u | uu_b | vv_b | v | worst ssh | worst u |
|---|---|---|---|---|---|---|---|
| before (round 5's card) | 17 | 27 | 27 | 27 | 19 | 9.85e-08 | 1.25e-07 |
| + fold pair (D, E) | 17 | 27 | 27 | 27 | 19 | 1.47e-14 | 2.34e-15 |
| + fold pair + F | 17 | 27 | 27 | 27 | 19 | 1.57e-14 | 2.27e-15 |
| + fold pair + G | 23 | 54 | 54 | 54 | 19 | 1.44e-14 | 2.00e-15 |
| **+ all four (the card)** | **23** | **54** | **54** | **54** | 19 | 1.48e-14 | 2.18e-15 |
| the card with the entry depth deselected (Decision 119) | 23 | 59 | 59 | 58 | 19 | 1.38e-14 | 1.81e-15 |

- Round 5 read 17 for the card and 27 for the deselected arm. Both read 23
  now. The deselected arm is 5 steps later on the velocities; that is a
  draw, as in round 5.
- F does not move the headline alone or on top of G. It is kept because its
  own row is bit-proven with G (2 / 0 -> 0 / 0). Listed in section 7.
- The V column is the stored top row (take-away 6). With the fold image
  written into that row between steps (a measurement arm, not the card): V
  first leaves the bar at 54, its worst is 2.1e-15, and every other number
  in the table is unchanged.
- 82 of 100 steps still have a row over the bar (84 before). Sea level is
  over the bar at steps 23, 24 and from 27 on.

Sea level, normalised error / cells not bit-equal:

| step | before | card |
|---|---|---|
| 1 | 0 / 0 | 0 / 0 |
| 2 | 2.08e-17 / 714 | 2.08e-17 / 742 |
| 10 | 4.16e-17 / 10955 | 5.03e-17 / 10687 |
| 16 | 5.20e-17 / 12161 | 5.03e-17 / 11980 |
| 17 | 8.73e-15 | 4.62e-17 / 12190 |
| 25 | 3.58e-15 | 4.61e-16 |
| 27 | 1.14e-14 | 5.75e-15 |
| 40 | 1.39e-12 | 5.01e-15 |
| 60 | - | 1.16e-14 |
| 100 | 5.30e-08 | 1.32e-14 |

T / S cells that differ after the step, before -> card: step 11: 1/1 -> 3/3;
16: 2/2 -> 2/2; 20: 4/3 -> 3/2; 25: 29/15 -> 5/3; 30: 58/24 -> 6/4; 40:
724/608 -> 49/17; 49: 1771/1454 -> 151/64. The tracer flip is still there;
it is the same carrier as rounds 4 and 5.

The step-23 event: one cell, inner-domain (74, 84), 5.7e-15 (NEMO's largest
3.3e-02). The step-17 cell (94, 41) is at the bar.

Ladders, steps 1 to 10, the card (max abs / RMS / cells; all AT-BAR, and no
field leaves the bar in either label):

| step | INDEPENDENT ssh | u | v | GIVEN-NEMO-ENTRY ssh | u | v |
|---|---|---|---|---|---|---|
| 1 | 0 / 0 / 0 | 1.73e-18 | 1.73e-18 | 0 / 0 / 0 | 1.73e-18 | 1.73e-18 |
| 2 | 2.08e-17 / 3.48e-19 / 742 | 6.94e-18 | 5.20e-18 | 2.08e-17 / 3.66e-19 / 564 | 6.94e-18 | 5.20e-18 |
| 3 | 2.43e-17 / 6.88e-19 / 2185 | 8.67e-18 | 5.64e-18 | 1.73e-17 / 4.77e-19 / 1451 | 6.94e-18 | 5.20e-18 |
| 4 | 2.08e-17 / 8.74e-19 / 3694 | 1.04e-17 | 8.24e-18 | 1.73e-17 / 5.73e-19 / 2328 | 6.94e-18 | 4.77e-18 |
| 5 | 3.47e-17 / 1.09e-18 / 5390 | 1.13e-17 | 7.81e-18 | 1.39e-17 / 5.57e-19 / 3328 | 6.94e-18 | 4.34e-18 |
| 6 | 2.26e-17 / 1.14e-18 / 6639 | 1.26e-17 | 7.81e-18 | 1.39e-17 / 6.05e-19 / 3906 | 5.20e-18 | 3.52e-18 |
| 7 | 4.34e-17 / 1.30e-18 / 7404 | 1.47e-17 | 1.04e-17 | 1.39e-17 / 5.68e-19 / 4492 | 6.94e-18 | 4.34e-18 |
| 8 | 3.99e-17 / 1.36e-18 / 8265 | 1.39e-17 | 8.67e-18 | 1.39e-17 / 5.45e-19 / 4887 | 4.34e-18 | 5.20e-18 |
| 9 | 5.29e-17 / 1.43e-18 / 9636 | 1.21e-17 | 7.81e-18 | 1.39e-17 / 5.68e-19 / 5881 | 5.20e-18 | 4.34e-18 |
| 10 | 5.03e-17 / 1.45e-18 / 10687 | 1.34e-17 | 9.11e-18 | 1.21e-17 / 5.49e-19 / 6534 | 6.94e-18 | 4.34e-18 |

GIVEN-NEMO-ENTRY was not run past step 10 this round.

## 4. Gates for the landing

- No AT-BAR row leaves the bar; no first-over-bar step moves earlier (17 ->
  23, 27 -> 54, 19 -> 19).
- Each option is default False and selected only by the G1 card. Caller grep
  after the change: each name appears in its field, its one use, the G1 card
  (selection and validator), the ladder's before arms and the tests.
- Searched before building (grep of the solver and the grid operators for
  fold helpers and for the stage statements): the fold permutations, the
  north-neighbour helpers, the stage update and the correction operator
  already existed and are reused; nothing was re-implemented.
- No cited line of the model, solver or recipe files moves: in-place lines
  keep their count; helpers sit at the ends of the files.
- Closed cards (GYRE, VORTEX, TSUNAMI, ORCA2) were NOT re-scored. The
  default expressions of the two shared stage functions were re-typed on one
  line each with the same operation order; that is argued, not measured.

## 5. ORCA2 pointer

| statement | G1 | ORCA2 lane has it? | does ORCA2's deck reach it? |
|---|---|---|---|
| D, ratio fold exchange in the solver coefficients | landed, selected | its rounds 261 to 264 hold the V-ratio association for the stage transport (HELD there); not this consumer | yes: shared builder, any step with sea level at the fold |
| E, mask pivot row | landed, selected | no. The overwrite was written in its rounds 125 to 126 on a record whose F ratio was zero, where it cannot show | yes: shared builder. Every ORCA2 run with sea level at the fold reads the overwritten row. UNMEASURED there; reported as a finding |
| F, stage 1 consumes the whole right-hand side | landed, selected | n/a | NO as read: ORCA2's deck is vector form, and that arm already hands stage 1 the whole right-hand side. The flux-form mini-ladder rungs reach it. UNMEASURED |
| G, written order of the stage update and correction | landed, selected | no | the correction is shared by both forms; the vector-form update is a different expression and is NOT guarded by this option. UNMEASURED |

## 6. Predictions

| id | result |
|---|---|
| F-P1 bare stencil unequal on the V and F pivot rows; the halo read leaves a handful | first half HOLDS (68 / 66 points, up to half of NEMO's value). Second half REFUTED at the leaf: the halo read leaves only the instrument's residue; the "handful" lives in the coefficients and is the mask |
| F-P2 halo read plus exchange is 0 unequal | HOLDS only in NEMO's written order (L4); L2 as built carries the instrument's residue |
| F-P3 exchange alone equals halo read plus exchange | HOLDS, bit for bit |
| F-P4 production coefficients 0 unequal with the exchange | **REFUTED**: 5 to 13 faces remain. 0 only with the mask row too |
| F-P5 fold-band exit sea level falls 10-fold at step 41; step 100 falls 10-fold | HOLDS: 2.9e-12 -> 3.0e-20; 5.30e-08 -> 1.32e-14 |
| F-P6 record not at the bar; next fold row is the entry depth on the fold-image V row | first half HOLDS. Second half NOT OBSERVED: no fold row is first any more |
| L-P1 first unequal row at step 1 is a 3-D stage velocity | HOLDS |
| L-P2 that stage's inputs are bit-identical | **REFUTED**: its right-hand side was not NEMO's |
| L-P3 a written-order difference, not compiler rounding of the same order | HOLDS for F. For G the written order is the same and the compiled evaluation is not; landed with the shared rounding guard |
| L-P4 first sea-level step over the bar moves later than 17 | HOLDS: 23 |

## 7. Corrections to my own work, and choices

Corrections:
1. The preregistered partner was wrong. I named the exchange as the partner
   of the halo read; the exchange replaces the halo read, and the partner is
   the mask row.
2. My first leaf instrument was not bit-comparable (it divided by the face
   area). Found from its own rows below the fold; L4 added as the control.
3. My first stage hypothesis (the after ratio) was refuted by the replay.
4. Commit f04fbf0fd's message and comments cite the mask exchange at line
   233 and the fold rules at 639-749. They are 232 and 639-748. Comments
   fixed in dc5c03769 and the working tree; the message stands as written.
   The preregistration also carries 639-749.

ASKED (round brief): split the held operand with its partner; land the pair
if it closes its boundary; re-score the fold band and the record; walk the
step-17 event to its first statement and land a transcription defect;
receipt; tests; one review.

UNASKED, each offered for revert:
1. F stays selected although it moves no headline step. My pick: keep
   (bit-proven at its own row).
2. Two fold options that the solver refuses apart, rather than one option.
3. The held halo read is not added (inert once exchanged).
4. Three older tests pinned option positions in the config; re-pinned.
5. A measurement arm that hands stage 1 NEMO's right-hand side; the
   fold-image arm run for the V row.

Left alone on purpose: the production ratio's divide-by-area order (it
reaches no measured row); the stored top V row; the fold-image V row's
unconsumed operands.

## 8. Tests and review

Single review (codex): **findings**, two CONFIRMED, both acted on. Log:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round6/codex_review_r6.log.
- "The "written-order" barotropic correction is not NEMO's written
  statement. NEMO computes `field + zub*mask` ... while
  `_rk3_stage_correction_written_order` computes `(field+zub)*mask`. The
  test repeats the incorrect association ... so it cannot detect this."
  Fixed in 4af3a4e4a: the correction adds the masked difference, the update
  masks its quotient as the statement does, and the test now has a dry input
  the two associations treat differently. Wet faces are bit-identical either
  way: the 100-step record re-run at 4af3a4e4a differs from the earlier run
  in 0 of 598 rows (recheck_record100_3_card.json).
- "The offline `fused_multiply_add` is not an IEEE FMA for non-finite
  operands ... `0*Inf+1` incorrectly return zero instead of NaN". Fixed: the
  helper refuses a non-finite operand; tested.
"VERDICT: findings". No second reviewer: a single review, not a dual one.
Claim review before code: NOT done (the lane's rule is one review of the
diff).

Focused tests and the citation gate: counts are in the final commit message
of this round and in tests_g1.log, tests_ratchets.log, citation_receipt.json
under the evidence directory.

Compliance, stated because no gate enforces it: search-before-build (section
4); one variable per arm (four-way coefficient table; nested record arms;
one before arm per option); instruments checked before quoting (L4 and the
written-order replay are 0 unequal against NEMO; the first leaf instrument
failed its own control and was corrected); non-vacuity (each half and each
stage option off turns its record test red; score plant exits 1).

Stamps: the record arms are stamped 3f5198b53, the fold and stage tables
dc5c03769, the recheck 4af3a4e4a. dc5c03769 is comment-only. The stage
tables were NOT re-run after the review fix; the record recheck and the
stage-1 record test (run on the final tree) cover it.

UNVERIFIED:
- that no closed card moves (default paths argued, not run);
- which products the compiler fuses in the unguarded stage expressions;
- the ORCA2 column of section 5 (read, not run);
- that the halo read and the exchange differ in any bit (not separated at
  the leaf);
- the stage tables at the final tree (see Stamps).

## OPEN for round 7

1. The 3-D right-hand side at step 2 (618 U, 766 V faces, last place): the
   pressure gradient and the vorticity term, replayed offline from NEMO's
   entry frame one at a time. This is the first unequal row of the INDEPENDENT
   chain now.
2. Stage 2 of step 1 (123 U, 238 V faces): its right-hand side is not in the
   record, but every operand of it is (the stage-1 frame). Replay the
   pressure gradient and the vorticity term from that frame; then stage 3.
   If that cannot separate them, the acquisition is three more frames per
   stage (the right-hand side after each stage's last call).
3. The stage-1 U raw velocity's 17 and 63 faces (section 2.4): test the
   card's U ratio operand, then the after ratio in NEMO's interpolation.
4. The stored top V row of the 3-D velocity: name the statement (NEMO's
   exchange after each stage) and land it; it is the whole V column of the
   headline table.
5. Only then the step-23 event, and the rounding-row question for whatever
   is left.
