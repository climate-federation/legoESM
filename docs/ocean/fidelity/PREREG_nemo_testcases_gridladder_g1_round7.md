# Grid ladder rung G1 — round 7 preregistration (gridladder_r7)

Written before any measurement of this round. Base: 35cec53b7 (rounds 1-6).
Record: the operator's round-4 acquisition
(`.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step`); NEMO is not re-run.
Decision 119 is pending: the entry-depth selection is not touched, and every
headline carries the deselected arm's reading.

Work, in the order of round 6's open list.

## Item 1 — the 3-D right-hand side at step 2 (618 U, 766 V faces)

Instrument: an offline numpy replay, from NEMO's step-entry frame and mesh
file, of the two statements that build the array NEMO records after its 2-D
step: the pressure gradient (`dynhpg.f90:379-394`, one level) and the
planetary vorticity term (`dynvor.f90:720`, `:727`, `:767-768`, `:772-776`,
`:779-785`). The density anomaly is the shared library function on NEMO's
recorded T and S. The card's two terms come from the model's existing
write-only per-term observer.

- R-P1 (control). pressure gradient + vorticity term, as replayed, equals
  NEMO's recorded right-hand side: 0 unequal wet faces, U and V, steps 1 to 3.
  Falsifier: any unequal face; the replay is then not an instrument and no
  attribution is made from it.
- R-P2. The card's pressure-gradient term equals the replayed one bit for bit;
  the vorticity term is the first non-bit term (it is zero at step 1, where
  the right-hand side is already bit-equal). Falsifier: the pressure gradient
  is unequal on any wet face at step 2.
- R-P3. The unequal vorticity faces come from an operand, not the triad
  arithmetic: the card's thickness ratios at U, V or F points (round 6 saw
  about 3,900 last-place points from the divide-by-area order). Falsifier:
  the card's term stays unequal when handed NEMO's recorded ratios.

## Item 2 — stage 2 of step 1 (123 U, 238 V faces), then stage 3

Instrument: the same two statements replayed from NEMO's stage-1 frame, then
round 6's written-order stage update; scored against NEMO's stage-2 frame.

- S-P1 (control). The replayed stage-2 velocity is 0 unequal against NEMO's
  stage-2 frame at step 1. Falsifier: unequal; then the operands in the
  record do not determine the stage (an unrecorded operand), and the round
  returns ACQUISITION_NEEDED for the right-hand side after each stage.
- S-P2. The owner at stage 2 is the same term and operand as item 1.

## Item 3 — the stage-1 U raw velocity, 17 and 63 faces

- U-P1. Replayed with the card's own U ratio in place of NEMO's, the card's
  stage-1 U raw velocity is the written update of its own right-hand side
  (0 unequal, steps 2 and 3): the U ratio is the unequal operand.
  Falsifier: still unequal; then the after ratio's interpolation is tested.

## Item 4 — the stored top V row

- V-P1. NEMO's exchange after each stage writes the top V row as the fold
  image of the row below. Landing it (selected by the G1 card only) moves V's
  first step over the bar from 19 to 54 and its worst from 4.6e-03 to about
  2.1e-15, and moves no other row of the 100-step record by one bit.
  Falsifier: any other field's row changes.

## Headline

- H-P1. After the landings the 100-step record is NOT at the bar; sea level's
  first step over the bar is 23 or later, never earlier. G1 is not certified
  this round. Falsifier (welcome): the record is at the bar.
- H-P2. No rounding row is registered unless a named statement with bit-equal
  operands is shown.

Gates for any landing: cited statement; option default off, selected only by
the G1 card; no AT-BAR row leaves the bar; no first-over-bar step earlier.
