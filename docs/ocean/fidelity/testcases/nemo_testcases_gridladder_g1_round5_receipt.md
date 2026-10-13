# Grid ladder rung G1 — round 5 receipt (gridladder_r5)

Status: **LANDED** (three cited solver statements, G1-card-selected), with the
headline NOT moved and one fold statement HELD. The 100-step record is not at
the bar: sea level first leaves it at step 17, before and after.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round5.md
(commit 9beabde3e; it states that the step-1 table was run before the
freeze). Code: 0a933b862 (options, instrument, tests), fc59a99ef (cited-line
corrections, plant fix, test re-pin, citation map), 7dd990a4b (an anchored
line restored; no arithmetic change). Evidence:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round5/final/
(json stamped 0a933b862; the plant and the three step-17 cell runs are
stamped dirty, run between the two commits). Record: the operator's round-4
acquisition, .../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step.

## Take-aways

1. **Step 1 of the solver is now NEMO's bit for bit.** All 13 substeps and
   the handoff (sea level, both barotropic velocities): 0 unequal points,
   from 1,070 / 1,103 / 1,124.
2. **Three transcription differences, all inside the velocity update.**
   The substep length (one last-place unit off), the last operation of the
   update (divide, where NEMO multiplies by an inverse), and the entry depth.
3. **The headline did not move: step 17, before and after.** The step-17
   event is the same tracer-carried one-cell flip as round 4, now at another
   cell. Which cell flips is a lottery on last-place differences; an
   intermediate arm reads step 27 and is NOT progress (section 3).
4. **What still feeds it:** from step 2 the loop is HANDED a slow forcing that
   differs in the last place (round 4's open right-hand-side row), and the
   stage velocities differ on about 590 faces at step 1. Both are outside the
   loop.
5. **Fold: first row named, not landed.** At steps 31 and 41 the first rows
   beyond rounding are the loop-entry Coriolis coefficients on the pivot row.
   One operand found (the north neighbour of the top row wraps to the south
   instead of reading the fold halo); alone it makes the record slightly
   worse, so it is HELD with its partner unnamed.
6. **Instrument limit:** the model's substep trace is bit-passive at step 1
   only. Past step 1 it moves about 1,400 faces by one last-place unit, so
   last-place claims there rest on the record-only replay, not on the trace.

## 0. Admission

- check_records, 100 steps, 50 stage steps, twin, reference, substep streams
  at 1, 2, 17, 31, 41: PASS_G1_RECORD (admission_r5.json). Every key equals
  the operator's admission file except the restart list I passed. No
  self-check refused; nothing re-admitted; NEMO not re-run.
- Each stream: 13 substeps, 450 arrays, 32 per substep. Substep length in the
  record 830.7692307692307 s; cold start flagged at step 1 only.
- Geometry: 36 of 36 rows bit-identical.

## 1. Step 1, every substep, compiled order (item 1)

GIVEN-NEMO-ENTRY (at step 1 this is also the INDEPENDENT state). Operands in
the order of the loop that starts at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:359`; unequal wet points,
card as it entered the round. Fold band, seam and south are 0 in every row
(the bump has not reached them); every nonzero count is interior.

| operand (substep 1) | unequal | largest |
|---|---|---|
| loop entry: slow forcing u, v; sea-level forcing | 0, 0, 0 | |
| loop entry: the eight Coriolis coefficient sets | 0 each | |
| midpoint weights and filter weights (91 scalars, alpha 0.09) | 0 | third substep on: 0.7125337174, 0.2525882038999999, 0.03722244, -0.0023443612999999985 |
| midpoint u, v, sea level | 0, 0, 0 | |
| midpoint face depths u, v | 0, 0 | |
| transports u, v | 0, 0 | |
| after sea level; back-interpolated sea level | 0, 0 | |
| pressure gradient u, v; Coriolis u, v; trend u, v | 0 each | |
| **new velocity u, v** | **169, 160** | 4.3e-19, 2.2e-19 (NEMO 1.1e-03) |
| exit face depths | 0, 0 | |

First non-bit statement: the flux-form velocity update,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:560-568`. Its first non-bit
operand is the substep length.

The loop-entry history frame (the two older sea levels) reads 201 unequal at
step 1. Not an owner: NEMO's frame holds zeros there on a cold start, the
first substep's weights are (1, 0, 0) and the compiled swap fills both
before they are read. Stated because the table prints it as its first row.

Record-only replay of the update (numpy, NEMO's frames and mesh file; no
model step), unequal face-substeps out of about 412,000 per step:

| step | NEMO's order | bracket divided by the exit depth | entry depth as rest depth plus mean sea level |
|---|---|---|---|
| 1 | 0 | 4,395 | 4 |
| 2 | 0 | 10,960 | 292 |
| 17 | 0 | 84,487 | 3,524 |
| 31 | 0 | 103,285 | 5,721 |
| 41 | 0 | 110,991 | 6,498 |

Largest difference 1.7e-18 to 3.5e-18 in every non-zero cell. The top V row
is not replayed (its north neighbour is outside the mesh file).

## 2. The three statements (landed, G1 card only)

| # | NEMO statement | what the shared code did | option |
|---|---|---|---|
| A | substep length is the quotient, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:886` | under JIT the division by the constant count 13 compiles to a product with 1/13: 830.7692307692308, one last-place unit high | barotropic_substep_dt_nemo_literal |
| B | bracket TIMES the inverse exit depth, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:557-558`, in the update at `:560-568` | bracket divided by the exit depth | barotropic_flux_update_nemo_literal |
| C | loop-entry face depth is rest depth times (1 + ratio), `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:329-330`; ratio from `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:268-269` | rest depth plus face-mean sea level, at the first substep and in the forcing product | barotropic_entry_depth_nemo_literal |

Step 1, nested arms, one option added each time (unequal points):

| card | update faces, substep 1 | update faces, all 13 | handoff sea level / u / v |
|---|---|---|---|
| before | 329 | 15,556 | 1,070 / 1,103 / 1,124 |
| + A | 126 | 12,272 | 701 / 855 / 941 |
| + A + B | 2 | 8 | 2 / 0 / 2 |
| + A + B + C (the card) | **0** | **0** | **0 / 0 / 0** |

The 126 of the second row is the replay's "divided" column at substep 1
(59 + 65 = 124, plus the 2 of the entry depth); the 2 of the third row is
the replay's entry-depth count at substep 1. A was found by elimination:
the update run standalone on the recorded operands gave NEMO's bits, and
gave the card's bits (0 of 15,853 unequal) only with the product-form length.

All three are BarotropicConfig fields, default False, appended after the
last existing field. Searched first (grep for the update helper, the
entry-inverse helper, any guard on a constant divisor): the update helper
and the ratio already existed and are reused; no divisor guard existed, the
shared rounding guard is used. Caller grep afterwards: each name appears in
the field, its one use, the G1 card (selection and validator) and the
tests. No other card selects them; the default path is unchanged (the
"before" arm above is that path and reproduces round 4's numbers to every
printed digit). No cited line of the solver file moves: new helpers sit at
the end of the file.

## 3. The 100-step record after each landing (INDEPENDENT)

First step leaving the 1e-15 bar:

| card | ssh | u | uu_b | vv_b | v |
|---|---|---|---|---|---|
| before | 17 | 32 | 32 | 23 | 19 |
| + A | 17 | 27 | 27 | 27 | 19 |
| + A + B | 27 | 32 | 32 | 32 | 19 |
| + A + B + C (the card) | **17** | 27 | 27 | 27 | 19 |

Sea level, normalised error / cells not bit-equal:

| step | before | + A | + A + B | card |
|---|---|---|---|---|
| 1 | 4.16e-17 / 1070 | 2.78e-17 / 701 | 6.94e-18 / 2 | 0 / 0 |
| 2 | 4.86e-17 / 2711 | 3.47e-17 / 2089 | 2.78e-17 / 1073 | 2.08e-17 / 714 |
| 10 | 4.34e-17 / 15429 | 3.99e-17 / 14039 | 5.03e-17 / 11365 | 4.16e-17 / 10955 |
| 16 | 6.98e-17 / 16347 | 5.20e-17 / 14856 | 5.38e-17 / 12475 | 5.20e-17 / 12161 |
| 17 | 5.75e-15 | 8.72e-15 | 3.86e-17 | 8.73e-15 |
| 25 | 4.71e-15 | 3.58e-15 | 4.16e-17 | 3.58e-15 |
| 30 | 8.04e-15 | 7.30e-15 | 6.17e-15 | 7.30e-15 |
| 40 | 1.39e-12 | 1.39e-12 | 1.39e-12 | 1.39e-12 |
| 100 | 5.30e-08 | 5.30e-08 | 5.30e-08 | 5.30e-08 |

- The third arm reaching step 27 is NOT a result. Each arm has last-place
  sea-level differences from step 2 (1,073 and 714 cells); the thickness
  quotient turns one of them into a tracer flip at about one cell per step,
  and the flip that crosses the bar is a matter of which cell. Arm three
  happened to draw none until 27.
- The card keeps all three because each is bit-proven at its own row. The
  headline is the same step. This choice is listed in section 8.
- Cells whose T / S differ after the step, before -> card: step 8: 2 / 1 ->
  0 / 0; 11: 6 / 4 -> 1 / 1; 16: 6 / 2 -> 2 / 2; 20: 13 / 6 -> 4 / 3; 25:
  49 / 29 -> 29 / 15; 30: 97 / 34 -> 58 / 24; 40: 778 / 631 -> 724 / 608.
- From step 32 the fold dominates and nothing moves.

Ladders, steps 1 to 10, card (max abs / RMS / cells; all AT-BAR):

| step | INDEPENDENT ssh | u | v | GIVEN-NEMO-ENTRY ssh | u | v |
|---|---|---|---|---|---|---|
| 1 | 0 / 0 / 0 | 1.73e-18 | 1.73e-18 | 0 / 0 / 0 | 1.73e-18 | 1.73e-18 |
| 2 | 2.08e-17 / 3.48e-19 / 714 | 6.94e-18 | 5.20e-18 | 2.08e-17 / 3.66e-19 / 564 | 6.94e-18 | 5.20e-18 |
| 3 | 2.60e-17 / 6.95e-19 / 2200 | 8.67e-18 | 6.94e-18 | 1.73e-17 / 4.77e-19 / 1434 | 6.94e-18 | 5.20e-18 |
| 4 | 2.08e-17 / 8.82e-19 / 3742 | 1.04e-17 | 8.67e-18 | 1.73e-17 / 5.88e-19 / 2404 | 6.94e-18 | 4.77e-18 |
| 5 | 3.47e-17 / 1.08e-18 / 5417 | 1.13e-17 | 8.24e-18 | 1.39e-17 / 5.49e-19 / 3333 | 6.94e-18 | 5.20e-18 |
| 6 | 1.91e-17 / 1.14e-18 / 6699 | 1.34e-17 | 1.08e-17 | 1.39e-17 / 6.10e-19 / 3839 | 5.20e-18 | 5.15e-18 |
| 7 | 6.07e-17 / 1.31e-18 / 7486 | 1.39e-17 | 9.11e-18 | 1.39e-17 / 5.81e-19 / 4346 | 7.81e-18 | 4.07e-18 |
| 8 | 3.47e-17 / 1.36e-18 / 8445 | 1.30e-17 | 1.19e-17 | 1.39e-17 / 5.54e-19 / 4902 | 6.07e-18 | 5.20e-18 |
| 9 | 5.29e-17 / 1.44e-18 / 9843 | 1.47e-17 | 9.54e-18 | 1.04e-17 / 5.49e-19 / 5825 | 5.20e-18 | 4.00e-18 |
| 10 | 4.16e-17 / 1.48e-18 / 10955 | 1.47e-17 | 9.97e-18 | 1.39e-17 / 5.61e-19 / 6534 | 6.94e-18 | 3.47e-18 |

Round 4's step-10 cells were 15,429 and 13,000.

## 4. Step 17 (item 2, the interior departure)

- INDEPENDENT, card: 1 cell over the bar, inner-domain (94, 41), -8.73e-15
  (NEMO -1.89e-03), 11 cells from land, 53 rows from the fold. Round 4's
  cell (72, 70) is at the bar now.
- Same carrier as round 4, re-measured at the new cell: one step from NEMO's
  step-17 entry with NEMO's T and S is at the bar (largest 1.04e-17, 2.2e-19
  at the cell); with the card's own T and S it is -8.71e-15 at the cell.
- Substep table at step 17, GIVEN-NEMO-ENTRY: no loop row beyond rounding in
  any region (largest region ratio 6e-16). The first unequal row is the slow
  forcing the loop is handed: 7,928 U and 8,338 V faces, largest 7.9e-23 and
  1.1e-22 on values near 1.2e-08 (ratio 6e-15 and 9e-15: the size of the last
  place of the pressure gradient it is the residue of). That is round 4's
  open right-hand-side row, upstream of the loop.
- INDEPENDENT at step 17: the handed forcing differs by 2.7e-19 (ratio
  2e-11) at the event; the loop's exit carries 8.7e-15. The loop adds
  nothing of its own above rounding.
- Step 2: the same picture, 546 and 692 forcing faces, all interior.

So the loop is closed as far as this record can see: its one statement that
was not NEMO's replays bit for bit, and what remains at step 17 enters
through the right-hand side (from step 2) and the stage velocities (about
590 faces at step 1), then the tracer quotient. No statement is named there.

## 5. Steps 31 and 41 (item 2, the fold residual)

GIVEN-NEMO-ENTRY, NEMO's own histories (the record's loop-entry frame, which
equals the restart's). Error over NEMO's maximum on the same row; seam, south
and interior are at rounding in every row listed (at most 3e-14).

| row, compiled order | step 31: pivot row / row below | step 41: pivot row / row below |
|---|---|---|
| handed forcing u | 5.8e-08 / 1.1e-14 | 1.6e-05 / 1.5e-14 |
| handed forcing v | 8.6 / 1.5e-08 | 18 / 3.8e-06 |
| Coriolis coefficients, U sets (4) | 1.9e-09 to 6.3e-09 (65 to 67 faces) / 0 | 1.5e-07 to 3.0e-07 / 0 |
| Coriolis coefficients, V north sets (2) | 3.6e-09, 3.8e-09 / 1.9e-09, 2.5e-09 | 3.5e-07 / 1.5e-07, 2.0e-07 |
| substep 1 pressure gradient v | 1.0 (68 faces) / 0 | 1.0 / 0 |
| substep 1 Coriolis v | 0.65 / 7.9e-10 | 0.63 / 8.3e-08 |
| substep 1 new u, v | 7.4e-12, 1.5e-11 / 9.9e-17, 1.5e-11 | 2.1e-10, 9.7e-11 / 1.9e-16, 9.7e-11 |
| loop exit: u, v, sea level | 1.9e-10, 2.1e-10, 1.2e-10 | 1.8e-07, 6.2e-08, 6.7e-08 |

The exit row reproduces round 4's handoff numbers. The coefficients are the
blocks at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960-981` and
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:987-1008`; they are
bit-identical at steps 1, 2 and 17 (no sea level at the fold yet).

Operand found by reading, then tested as one variable: the coefficient
builder takes the north neighbour of the TOP T row by a cyclic wrap to the
southernmost row. NEMO reads the fold halo there, in the V-point ratio
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:268-269`) and the F-point
ratio (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:281-285`).

| measure, with the fold halo read | without | with |
|---|---|---|
| pivot-row coefficient faces unequal, step 31 | 65 to 67 | 5 to 8 (columns 58 to 63, 149, 150) |
| their largest ratio, step 31 / 41 | 6.3e-09 / 3.8e-07 | 2.7e-09 / 3.8e-07 |
| loop-exit sea level, fold band, step 31 / 41 (abs) | 1.0e-15 / 2.9e-12 | 2.4e-15 / 5.0e-12 |
| 100-step record, sea level, step 40 / 100 | 1.39e-12 / 5.30e-08 | 2.55e-12 / 7.52e-08 |

**HELD.** It is a real NEMO read and it repairs 59 of 67 faces, but the
record gets worse, so a partner is missing. Not selected, and its code is
NOT on the branch (a knob nothing selects is a defect); the tested patch is
kept as held_fold_halo_plus_landings.diff. Candidates for the partner, none
tested: the 5 to 8 faces that stay unequal; the pressure gradient on the
fold-image V row, which the card leaves at zero (ratio 1.0 above); the
handed V forcing on the same row. The with-halo runs are uncommitted-tree
runs (dirty stamp).

## 6. ORCA2 pointer

| statement | G1 | ORCA2 lane has it? | does ORCA2's deck reach it? |
|---|---|---|---|
| A, substep length as a true quotient | landed, selected | no | shared line. ORCA2's deck is also 10800 s; for a count of 65 (the count in the recipe's ORCA2 builder; I did not confirm it is the resolved one) the product and the quotient are the same number, so it would be inert there. UNMEASURED |
| B, times the inverse exit depth | landed, selected | no | NO as read: ORCA2's deck sets the vector form (ORCA2_OMIP_L4/EXP00/namelist_cfg line 346), and this statement is the flux-form branch. G1 runs the flux form |
| C, entry depth as rest depth times (1 + ratio) | landed, selected | no | the forcing-product part is flux form only; the loop-entry depth is shared by both forms and feeds the drag product. UNMEASURED |
| fold halo in the coefficient ratios | HELD | no (grep of the branch: the builder wraps) | yes, shared builder, any step with sea level at the fold. Round 3's pointer stands |
| prior art reused | per-substep table with the fold / interior split (round 235's layout, the VORTEX walk's table); matched program point (round 263) | | the in-loop fold copy, the raw V mask pair and the V-ratio association were NOT re-tested here: no G1 row asked for them before the coefficients |

## 7. Predictions

| id | result |
|---|---|
| S-P1 velocity update bit-identical at all 13 substeps with the option | HOLDS only with all three statements; REFUTED as written (the one option moves 169 to 159 on its own) |
| S-P2 step-1 handoff below 100 unequal cells, remainder a face depth | HOLDS on the count (0); the predicted remainder was wrong (the single-face depth rows were downstream and vanished) |
| S-P3 first sea-level step over the bar moves to 29 or later | **REFUTED**: 17 |
| S-P4 no T or S cell differs through step 16 | **REFUTED**: 1 / 1 at step 11, 2 / 2 at step 16 |
| S-P5 no substep row beyond rounding at steps 2 and 17 | HOLDS |
| F-P8 first fold row beyond rounding is in the loop-entry frame, other regions at rounding | HOLDS |
| F-P9 it is the Coriolis coefficient sets | HOLDS (the handed forcing on the same rows is listed first by the table and is their consumer's output) |
| F-P10 | not reached |
| F-P11 landing the fold statement takes step 31 below 1e-13 and step 100 down 100-fold | NOT MET: held |

## 8. Corrections to my own work, and choices

Corrections:
1. I cited the update at lines 544 to 562 in code comments, the
   preregistration and the first commit message. It is 557 to 568. Fixed in
   fc59a99ef; the first commit's message still carries the wrong numbers.
2. My first fix (B alone) was aimed at the wrong owner. The standalone run
   showed the operand, not the operation, was off.
3. The substep table's plant could not fire (it compared against the table's
   first unequal row, which is the unread history frame). Fixed; it exits 1.
4. The fold option did nothing on its first run: the coefficients are built
   on a second path I had not plumbed. Stated because an unchanged number
   could have been read as a null.

ASKED (round brief): admit; the step-1 table; land a transcription defect
selected by the G1 card; the step-17 and fold walks; re-score after each
landing; receipt; tests; one review.
UNASKED, offered for revert:
1. Keeping statement C selected although the arm without it reads step 27.
   My pick: keep (bit-proven at its row; the 27 is a draw, not fidelity).
2. Three options rather than one. Each is its own cited statement and its
   own before-arm.
3. Statement C is read only when B is selected; the solver refuses C alone.
4. Two round-3/4 tests pinned the step-17 event to cell (72, 70); re-pinned
   to (94, 41) and one cell, the re-measured fact.
5. The two new arms and the replay arm of the instrument; the unused
   history frame left in the table as its first row.

## 9. Tests and review

Single review (codex): **findings**, one CONFIRMED, measured, not acted on.
Log: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round5/codex_review_r5.log.
- "zeros `r3_v` on the top/fold V row and never applies the folded
  north-neighbor calculation ... `_nemo_entry_face_depths` ... supplies
  `hv_0`, not NEMO's `hv_0*(1+r3v(Kmm))` ... for that wet image row."
  True as read: with statement C the entry depth on the fold-image V row is
  the rest depth. Measured before deciding (step 41, with and without C,
  fold rows): the V update and the handoff agree to 8 digits (pivot-row V
  1.1013429e-12 both; sea level 2.9430665e-12 and 2.9430664e-12), and the
  pivot row equals the row below in both, so that row is set by the fold
  image and its own depth does not bind here. Not acted on; it belongs to
  the held fold unit (the V-point ratio on that row is the ORCA2 lane's
  rounds 261 to 264 statement). Registered open.
- "Coverage misses this: ... explicitly excludes the top V row; the unit
  test ... injects an already-correct `r3v` instead of testing its
  producer." True. The replay's exclusion is stated in section 1; the
  producer on that row is untested. No change.
- "I confirmed the remaining requested mappings: signs/staggering,
  first-substep-only entry depth, later `hu_e/hv_e`, east/north rolls,
  guarded-divisor behavior, legacy return modes, cold-start frame handling,
  and default-false paths. Tests could not run".
"VERDICT: findings". After the review nothing in the packages changed. No
second reviewer: a single review, not a dual one.

Focused tests, synchronous, final tree:
- The three G1 files: **118 passed** (tests_g1.log). New: each option's
  plant (the step-1 loop is bit-identical with all three; 2 or more unequal
  faces with any one off), the replay's own control (0 unequal in NEMO's
  order, over 10,000 and over 1,000 in the other two), the written order
  against numpy, the refusals, the guarded quotient, the table's plant.
- Six ratchets: 6153 passed, 2 skipped, 4 failed (tests_ratchets.log). The
  same four as rounds 1 to 4, none in a file this round touched: the
  slab-land dispatch guard, the jra55 literal, the land restart
  coefficients, the shortwave parameter spec.
- NOT RUN: any closed-card ladder (GYRE, VORTEX, TSUNAMI, ORCA2), the wider
  barotropic unit selection, and the inline-import budget test. The options
  are default-off and the "before" arm reproduces round 4, but no closed
  card was re-scored.

Citation gate on this receipt: PASS, 9 citations, 0 unmapped, whole-map
audit clean; the shifted-citation plant exits 1 (citation_receipt.json,
citation_plant.json under final/). The gate first FAILED the audit: my edit
had rewritten a line that an earlier round's citation is anchored on (the
ratio helper's return). Fixed in 7dd990a4b by restoring that line; the
100-step record and the step-1 table re-run at 7dd990a4b, the final code:
0 of 598 record rows differ from the 0a933b862 run, and the step-1 loop is
still bit-identical (recheck_record100.json, recheck_sub_kt1.json).

Compliance, stated because no gate enforces it: search-before-build
(section 2); one variable per arm (nested arms; the fold arm adds one
option); instruments checked before quoting (the replay reproduces NEMO
with 0 unequal; the trace's passivity is measured and its limit stated);
non-vacuity (option plants, score and table plants exit 1). Claim review
before code: NOT done (the lane's rule is one review of the diff). Dual
review: NOT met.

UNVERIFIED:
- that the compiled product-for-quotient rewrite is the only place the
  substep length is formed for this card (shown: the option makes the
  update's bits NEMO's; not shown: every other consumer of the length);
- the ORCA2 column of section 6 (read off the deck and the recipe, no run);
- why the trace is not bit-passive past step 1;
- the partner of the held fold operand;
- that no closed card moves (default path argued, not run).

## 10. Evidence hashes (SHA-256 prefixes)

final/: geometry.json=66888058ddc9b0c9, given_entry_10.json=d1525135bd90ea70, interior_events_17.json=a8fd6c53dce928ee, kt17_given_entry_chainTS.json=c82af7d479e42467, kt17_given_entry.json=92c0f496c1f27728, plant_score.json=06a1ba53f3d3dfc1, record100_0_before.json=cdef22c540a154c2, record100_1_dt.json=80f53e90fe660444, record100_2_dt_update.json=c8210c06980c45f3, record100_3_all.json=f993726ec3c62fc2, replay_kt1.json=01f9ee77d1f7254f, replay_kt2.json=dae4ab9a265dd41e, replay_kt17.json=9863b0c164ad8264, replay_kt31.json=01813733d92a4428, replay_kt41.json=ffc35bd047d00aa8, sub_kt1_0_before.json=3136ef358ece4656, sub_kt1_1_dt.json=4ad74922458375ad, sub_kt1_2_dt_update.json=25478c3384b1c619, sub_kt1.json=75aa599e5491d0f4, sub_kt1_plant.json=6776d3be58f32919, sub_kt2.json=06bba99b1ec88128, sub_kt17.json=648139bcdacc8e73, sub_kt17_independent.json=41deab6867c5fea2, sub_kt31.json=0d734eb46b108ab4, sub_kt41.json=0739662eaa8330a2, sub_kt41_without_entry_depth.json=ac6d2218259a9d5c, recheck_record100.json=e2e4300258d273a4, recheck_sub_kt1.json=89471148a69cc99e.
One level up: codex_review_r5.log=8c1e3386d313eb9f, admission_r5.json=cc734db89cb9fec8, tables5.txt=ce6e980cc09b9c10, held_fold_halo_plus_landings.diff=ccf215e9352f45db, record100_ABCD_held_dirty.json=33ca1626f0d82cb9, sub_kt31_ABCD_dirty.json=03f8b6fff28c9322, sub_kt41_ABCD_dirty.json=c4ce2da0b4fbd4c3.

## OPEN for round 6

1. The right-hand side's last-place row from step 2 and the stage
   velocities' row at step 1: the two feeds of the step-17 event.
2. The fold: the partner of the held halo read (the 5 to 8 faces, the
   fold-image V row's pressure gradient and handed forcing), then land the
   pair.
3. Why the substep trace moves bits past step 1.
4. The stored top V row of the 3-D velocity; the stage thickness that still
   interpolates the legacy way (round 4's review finding 1); the entry depth
   on the fold-image V row (this round's review finding).
