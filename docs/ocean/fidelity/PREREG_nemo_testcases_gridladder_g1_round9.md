# Grid ladder rung G1 — round 9 preregistration (gridladder_r9)

Base: df3bb2f14 (rounds 1-8). Record: the operator's round-4 acquisition
(`.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step`); NEMO is not re-run.
Evidence: `.../gridladder_rounds/round9/`. The Decision-96 ruling of
2026-10-13 is in force: nothing selected is re-asked.

## What was read BEFORE this file (nothing measured this round)

- Round 8's receipt and its step-40 tables (stage_replay_kt40, stages_kt40).
- NEMO's stage statements `stprk3_stg.f90:171-173` and `:215-217`.
- The card's stage code: stage 1's T-point weight is already the mean of the
  two stored ratios (an existing helper, unconditional); stage 2's weight and
  the pressure gradient's stretch both take the ratio of the stage's sea level.

## Item 1 — the stage-2 T-point ratio (step 40)

Candidate statement: `r3t(Kaa) = r1_2 * ( r3t(Kbb) + r3ta )`
(`stprk3_stg.f90:215`). One new option, default False, selected by the G1
card only. It is handed to (a) stage 2's tracer weight and (b) the stretch
the stage-3 pressure gradient reads. Stage 1's twin (`:171`) as the stretch
of the stage-2 pressure gradient is NOT in the candidate; it is counted
offline (R-P5) and reported.

- R-P1 (control). NEMO's stage-2 ratio and stage-2 tracers replayed in NEMO's
  order are 0 unequal at step 40 (round 8's arm, re-run on this tree).
  Falsifier: unequal; no attribution then.
- R-P2. With the option, GIVEN-NEMO-ENTRY at step 40: stage-2 T and S are 0
  unequal, and the stage-3 right-hand-side faces next to (55, 83) are 0.
  Falsifier: either nonzero.
- R-P3. The second cell, (101, 65), is NOT removed by the option: its four
  faces stay. It is then walked on its own (term split). Falsifier: 0 there.
- R-P4. The INDEPENDENT chain is bit-identical on T and S through step 100
  (unchanged) and on sea level through step 40 apart from the tiny-number
  steps 8 to 11. No direction is predicted for the first step over the bar.
  Falsifier: an unequal T or S cell, or an unequal sea-level cell at or
  before step 40 outside steps 8 to 11.
- R-P5 (count, no landing). Stage 1's twin: one plus the mean of the stored
  ratios against one plus the ratio of the stage-1 sea level differs on
  fewer than ten T cells per step. Falsifier: ten or more.

## Item 2 — the step-57 sea-level event at (67, 76)

Instrument: the whole-step GIVEN-NEMO-ENTRY arm at step 57, then the stage
and substep tables if the step itself makes the event.

- S-P1 (a guess, labelled so). One step from NEMO's step-57 entry leaves
  (67, 76) below the bar: the event is carried in by the chain's earlier
  last-place differences, not made at step 57. Falsifier: the cell is over
  the bar after one given-entry step; then the first non-bit statement inside
  step 57 is walked.
- S-P2. If S-P1 holds, the INDEPENDENT chain's first sea-level difference
  after item 1 lands is the owner and is named instead.

## Item 3 — the stage-3 right-hand-side faces (steps 5 and 21)

Instrument: the term table (rhs_terms) at stage 3 with its offline replay.

- V-P1 (control). The replay's sum equals NEMO's recorded stage-3 right-hand
  side, 0 unequal. Falsifier: unequal; no attribution then.
- V-P2. The unequal term is the pressure gradient, not the vorticity term.
  Falsifier: the pressure-gradient term is 0 unequal at those faces.
- V-P3. Item 1's option does not remove them (step 5 and 21 differ from step
  40's pattern only in which cell). Falsifier: 0 unequal with the option.
  (If it does remove them, items 1 and 3 are one statement.)

## Item 4 — the tiny-number class

Measured and registered, not asked.

- N-P1. Every NEMO value in the class is subnormal (below 2.2251e-308,
  nonzero); the card's value there is exactly zero. Falsifier: a normal
  number in the class.
- N-P2. NEMO's build does not flush: the arch file has no fast-math or
  flush flag. Falsifier: such a flag.
- N-P3. The card's compiled executable flushes: one compiled product of two
  normal numbers whose exact result is subnormal returns zero, and the same
  product in numpy does not. Falsifier: the compiled product is nonzero.
- N-P4. Every operand of the producing statement is bit-equal between the
  two sides at the first step of the class. Falsifier: an unequal operand;
  then no rounding row is registered.

## Item 5 — ORCA2 note

A reading, no prediction: which domain-file arrays statement F needs on
ORCA2, and whether the raw-mesh builder already reads them.

## Gates for any landing

Cited statement; default False, appended last, selected by the G1 card only
(caller grep); the 100-step record re-scored after it (first step over the
1e-15 bar for sea level, velocities, T and S, both labels); every moved row
registered; an earlier first step over the bar is registered under Decision
96, not reverted; focused tests with the option off as the plant; one
read-only codex review; citation gate on the receipt.
