# Grid ladder rung G1 — round 8 preregistration (gridladder_r8)

Base: 1ce119a43 (rounds 1-7). Record: the operator's round-4 acquisition
(`.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step`); NEMO is not re-run.
Operator ruling of 2026-10-13 13:10 (Decision 96): round 7's stage statements
and the entry-depth statement stay selected; neither is re-asked.

## What was measured BEFORE this file (the locating walk of item 1)

Stated so the frozen part below is honest. Uncommitted probes under
`.../gridladder_rounds/round8/` (probe2 to probe4), no landing:

- Step 2's substep table from the card's own chain: every operand, every
  substep, 0 unequal. The six histories the card carries into step 3 equal
  NEMO's (rebuilt from the step-2 substep record): 0 unequal.
- The solver's exit at step 3: sea level and U 0 unequal, V one face,
  inner-domain (77, 52), one last place.
- NEMO's velocity update replayed in numpy from the card's own step-3
  operands reproduces the card at every face of every substep (control), and
  turns into NEMO's value at that face when one of four operands moves one
  last place (the Coriolis trend among them).
- NEMO's solver Coriolis coefficients (`dynspg_ts.f90:987-1008`) replayed in
  numpy from NEMO's recorded entry ratios: 0 unequal against NEMO's recorded
  coefficients at steps 1 and 2 on 12,332 open V faces (control); against the
  card's step-3 coefficients 3 entries unequal, all three reading F point
  (78, 51).
- Read, not measured: the card's coefficient builder divides the three
  thickness ratios by the face area; NEMO multiplies by the stored reciprocal
  (`domqco.f90:266-269`, `:281-285`).

## Item 1 — frozen predictions

Candidate statement: the solver coefficient builder's U, V and F ratios in
NEMO's written order (times the reciprocal face area). One new option,
default False, selected by the G1 card only.

- C-P1 (control). NEMO's three ratio statements replayed in numpy from NEMO's
  entry sea level equal NEMO's recorded entry ratios, 0 unequal off the top
  row, steps 2 to 5. Falsifier: any unequal point; no attribution then.
- C-P2. The divide-by-area order differs from NEMO's on more than a tenth of
  the points, and (1 + ratio) differs on fewer than ten points per step.
  Falsifier: either count outside its range.
- C-P3. With the option the card's step-3 coefficients are 0 unequal against
  the replay, and the V face after step 3 is 0. Falsifier: either nonzero.
- C-P4. The INDEPENDENT chain is then bit-identical on every scored field
  through step 4. Falsifier: any unequal scored cell at step 4 or earlier.
  If it holds, no acquisition is needed for item 1.
- C-P5. The 100-step record is NOT at the bar afterwards (the tracer carrier
  is untouched). No direction is predicted for the first step over the bar.
  Falsifier: the record at the bar.

## Item 2 — the two V faces of the stage-3 right-hand side at step 5

Instrument: round 7's term table (rhs_terms, stage 3) and its replay.

- V-P1. With item 1's option the two faces are still there (they are
  GIVEN-NEMO-ENTRY rows; the solver's coefficients do not enter the 3-D
  right-hand side). Falsifier: 0 unequal.
- V-P2. The unequal term is the vorticity term and its operand is the stage
  F thickness or a face thickness, not the triad order. Falsifier: every
  operand bit-equal.

## Item 3 — the tracer carrier

Instrument: the existing stage_replay arm (NEMO's frames only) and the
card's stage tracer update handed NEMO's frames.

- T-P1 (control). The stage tracer update replayed in NEMO's order is 0
  unequal at steps 19 to 21. Falsifier: unequal.
- T-P2. The first T cell that differs in the INDEPENDENT chain (step 20)
  differs because the chain's sea level already differs there in the last
  place (from step 4 on in round 7's card): given NEMO's stage frames the
  card's own tracer update is bit-equal at that cell. Falsifier: the card's
  update is unequal from NEMO's frames; then the statement is the tracer
  update's and is named.
- T-P3. If C-P4 holds and the chain stays bit-identical longer, the first
  differing T cell moves later than step 20. Falsifier: step 20 or earlier.

## Item 4 — the rounding-row question

- Q-P1. No rounding row is registered this round unless a named statement
  with bit-equal operands is shown to differ. Falsifier: a row registered
  without one.

## Gates for any landing

Cited statement; default False, appended last, selected by the G1 card only
(caller grep); the 100-step record re-scored after it, both labels; every
moved row registered; an earlier first step over the bar is registered under
Decision 96, not reverted; focused tests with the option off as the plant;
one read-only codex review; citation gate on the receipt.
