# PREREGISTRATION — grid ladder rung G1, round 4 (gridladder_r4)

Committed BEFORE any round-4 measurement. Done before this freeze, and nothing
else: the standing note and the round-3 receipt were read; the operator's
acquisition log was read to its verdict lines (PASS_G1_RECORD, ACQUISITION_DONE)
and the record directory was listed (stage frames at kt 1..50, after-step
frames to kt 100). No record was opened, no step was run. Branch tip
43f4f070c. Instruments: round 3's
(scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid_ladder.py),
extended only so that the stage-frame arms accept the steps the new record
holds. Baseline: the tables of the round-3 receipt, both labels.

## 0. Measures (frozen; round 3's unless stated)

- Bar: 1e-15 normalised (the shared trajectory gate's). "Beyond rounding" for
  a region row: max abs error over NEMO's max abs on the same cells above
  1e-12, with NEMO's row nonzero there ("no signal" otherwise, never a pass).
- Program point: NEMO's recorded frame (after its boundary exchange) against
  the card's value at the same point. Operands from the card's production
  operand route.
- Compiled order inside a step: right-hand side -> external-mode handoff ->
  stage 1 -> stage 2 -> stage 3 -> after-step frame.
- The stage-frame arms at kt > 1 take the unrecorded carry (the six
  barotropic histories, w) from the card's own chain at that kt, as in rounds
  1 to 3. Where a restart exists (kt 11, 21, 31, 41) the histories are NEMO's.
- New record admission: check_records with 50 stage steps; and the after-step
  frames of the new record must equal round 1's record byte for byte at every
  step, or the round-3 baseline does not carry and is re-measured first.

## 1. Item 1 — the step-17 interior departure

Cell: the first open-ocean sea-level cell over the bar inside one step (round
3: inner-domain (72, 70) at kt 17, -5.75e-15 m; re-located from the new
record, not assumed). Walk at that step, GIVEN-NEMO-ENTRY, compiled order:
right-hand side, external handoff (sea level, barotropic U, V), stage 1, 2, 3
(each from NEMO's handoff and NEMO's stage entry), whole step. Reported at
the cell and its eight neighbours, absolute error and NEMO's magnitude, plus
the whole-domain row. The INDEPENDENT label is the census of round 3 re-run on
the new record. The same walk is repeated at kt 20 and kt 22 (the next two
fresh events) to check that the named statement is the same.

| id | prediction | falsifier |
|---|---|---|
| N-P0 | the new record's after-step frames equal round 1's byte for byte, kt 1..100 | any differing byte |
| I-P3 | the first row over the bar at the event cell, in compiled order, is a STAGE sea level (the stage's own continuity sum over the solver's advective transport), with the right-hand side and the external handoff at rounding at that cell | the handoff sea level is already over the bar at the cell, or the right-hand side is |
| I-P4 | given NEMO's frames, every operand of that statement at the cell is NEMO's bit for bit and the difference is the order of a sum: a rounding row, no fix | an operand differs from NEMO's beyond one unit in the last place, or an offline replay of NEMO's cited statement in NEMO's order does not reproduce NEMO's recorded value bit for bit at the cell |
| I-P5 | the same statement owns the kt 20 and kt 22 events | a different first row at either |

If the first row is the external handoff and the solver's inside cannot be
split without a substep record, the round writes that acquisition and stops
item 1 there rather than guess.

## 2. Item 2 — the remaining fold residual

First, round 3's open instrument question: the stage arm's pivot-row U reads
0.09 / 0.20 while the whole step is exact there.

| id | prediction | falsifier |
|---|---|---|
| F-P5 | that disagreement is a fault of the stage arm's feed on the fold rows (the fed solver output / stage entry lacks the fold image the card's own chain carries), not a stage defect | the feed is shown equal to what the card's own whole step hands its stages on that row, and the row still differs |
| F-P6 | at kt 31..41, with the round-3 statements in place, the first fold-band row beyond rounding in compiled order is the external handoff, not the right-hand side and not a stage | the right-hand side's fold rows are beyond rounding first, or the handoff's are at rounding and a stage's are not |
| F-P7 | the statement is a fold-image operand of the solver that the card leaves at zero or reads from the wrong row; with it (and the held Coriolis-coefficient operand re-tested on top) one step from NEMO's restart at kt 31 leaves the fold band below 1e-13 of its maximum (from 7e-11), and the kt-100 sea-level error of the 100-step record falls at least 100-fold from 5.30e-08 | either number not reached |

## 3. Headline (frozen form)

The step at which each field of the INDEPENDENT 100-step record first leaves
the bar, before -> after each landing. Before (round 3): ssh 17, u 32,
uu_b 32, vv_b 23, v 19. Both labels and the three-way split are re-printed
after each landing.

## 4. Landing rule (unchanged)

Only a statement cited from the compiled source lands, in the one shared
implementation, selected explicitly by the G1 card (library default and every
other card unchanged, caller grep in the receipt); a statement ORCA2 shares
gets the ORCA2 pointer. One statement per landing, the record re-scored after
each. A rounding row (all operands bit-exact, order of a sum only) is
registered with its census and statement, not fixed.
