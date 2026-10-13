# PREREGISTRATION — grid ladder rung G1, round 3 (gridladder_r3)

Committed BEFORE any round-3 measurement. Done before this freeze, and nothing
else: the standing note and the round-2 receipt were read; the record's field
names were listed (stage frames at kt 1..10, after-step frames only at
kt 11..100, restarts every 10 steps). No round-2 evidence file was re-opened
and no step was run. Branch tip e579a1a2e. Record and instruments: round 2's,
unchanged (scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid_ladder.py).
Baseline: the tables of the round-2 receipt, both labels.

## 0. Measures (frozen)

- "Beyond rounding" for a row restricted to a region: max abs error divided by
  NEMO's max abs on the same cells above 1e-12, AND NEMO's row nonzero there
  (a row where NEMO is zero is reported "no signal", never as a pass: a zero
  row satisfies the fold's antisymmetric identity trivially).
- Program point: every comparison is against NEMO's recorded frame, which is
  written after NEMO's boundary exchange; the card's value is taken at the
  same point (after its own fold and seam fill).
- Operands come from the card's production operand route (the raw-mesh
  operands the card carries), never a rebuilt mesh.
- Regions: round 2's (fold = top three carried rows; seam; south; interior
  split coast / open).
- Compiled order inside a step: right-hand side -> external-mode handoff ->
  stage 1 -> stage 2 -> stage 3 -> after-step frame.

## 1. Item 1 — the second fold statement

Walk: GIVEN-NEMO-ENTRY, kt 1..10 (the only steps with stage frames), arms
rhs, handoff, stages, whole step; fold band reported row by row (pivot row,
the row below, the third row) and face by face, both labels for the whole
step. Then the INDEPENDENT 100-step record and the restart-entry arm before
and after any landing.

| id | prediction | falsifier |
|---|---|---|
| F-P0 | kt 1..4: fold-band rows have 0 unequal cells in every arm | any unequal fold cell before kt 5 |
| F-P1 | the first fold-band row beyond rounding, in compiled order, is the external-mode handoff (barotropic U on the pivot row), not the right-hand side and not a stage | the handoff's fold rows are at rounding at kt 5..10, or the right-hand side is beyond rounding first |
| F-P2 | its statement is the substep Coriolis trend of the split-explicit solver reading the V transport on the fold-image row (the same omission as round 2, in the solver's own call) | an offline replay of that statement with NEMO's operands does not reproduce NEMO, or the card with the image row supplied does not move the row to rounding |
| F-P3 | given NEMO's handoff and stage entries, every stage row in the fold band is at rounding, except the stored top V row (zero against NEMO's image; known) | a stage row beyond rounding other than that row |
| F-P4 | after the landing: one step from NEMO's restart at kt 31 leaves a fold sea-level error below 1e-13 of the band's maximum (from 1e-7); the 100-step record's kt-100 sea-level error falls at least 100-fold from 1.39e-3 | either number not reached: a third statement is open and is named by the same walk |

If the walk cannot split the external mode further without a substep record,
the round writes the acquisition and stops for it rather than guess.

## 2. Item 2 — the kt-17 interior departure (PREREGISTERED PREDICTION, brief's)

G1 differs from the certified TSUNAMI program in three things: the fold, the
seam / south wall (exact to rounding, round 2), and ORCA2's coastline
(Decision 118), which doubly periodic TSUNAMI never exercised.

| id | prediction | falsifier |
|---|---|---|
| I-P1 | the first over-bar interior cell at kt 17 traces back to a COASTAL first non-bit statement — the lateral boundary treatment of the EEN vorticity at coasts (the f-point thickness masking the deck selects) or the lateral slip condition — not to the fold | (a) the trace leads to the fold; or (b) every operand of the right-hand side and of the stages is bit-exact at coastal faces at kt 1..10 and the excursion is reproduced by the card's own one-step response to a last-place perturbation of its entry at that cell: then it is registered as a rounding-accumulation row with its census, and I-P1 is REFUTED |
| I-P2 (mine, secondary) | the excursion is made inside step 17: one card step from NEMO's kt-16 after-frame gives at least half of 5.75e-15 at the same cell | less than half: it accumulated over earlier steps |

Instrument limits, stated before measuring: no stage frame exists at kt 17,
so a statement-by-statement split against NEMO is possible only at kt 1..10.
At kt 17 the instruments are (i) the error map of the independent chain at
kt 12..18 around the cell with distances to the nearest coast and to the
fold, (ii) the card's one-step sensitivity (two entries differing in the last
place, split at the handoff and stage exposures), (iii) the kt 1..10 operand
table restricted to coastal faces. If these cannot decide, the round asks for
a stage record around kt 17 and says so.

## 3. Landing rule

Only a statement cited from the compiled source lands, in the one shared
implementation, selected explicitly by the G1 card (library default and every
other card unchanged, caller grep in the receipt); an ORCA2-shared statement
goes in the shared card machinery with the ORCA2 pointer. One statement per
landing, the 100-step record re-scored after each.
