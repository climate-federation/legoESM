# Grid ladder rung G2 — round 2 preregistration (gridladder_r11)

Base: 9e0cebd1b (G1 certified; G2 round 1's deck, acquisition, card).
Record: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/oracle_g2_r10
(the operator's run of round 1's script; the brief's `oracle_g2_r1` is the
same directory misspelt). Evidence for this round:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round11/.
NEMO citations are to the record binary's compiled sources,
ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo.

## 0. Already measured before this file (not predictions)

Stated so nothing below can pass as a prediction after the fact.

- NEMO finished all four runs with no error line. Its log: 31 levels, 65
  substeps, Courant number 0.79801811104755460, bump centre (51,74) at
  longitude 179.99992123688389, latitude 0. Round-1 G-P2 and G-P3 HOLD.
- The acquisition refused once: the kt = 2 solver substep stream differs
  between the twins. Mapped this round through the stream's own header
  (round11/twin_kt2_substep_diff_map.log): 3,384 words (14,219 bytes) in 15
  arrays over the entry frame and the 65 substep frames; **every one lies in
  the 2-cell halo** (record rows 0-1, columns 0-1 and 182-183; one array also
  the west halo of rows 2-7); none in the inner domain. The values are
  storage garbage (about 6.9e-310, and up to 1.0 in the unused drag array).
  The kt = 1 substep stream and every step stream kt = 1..10 are
  byte-identical. Round-1 G-P4 is REFUTED as worded (byte identity) and holds
  on the inner domain.
- The only reader of these streams cuts every full plane to the inner domain
  (the ladder's substep reader). So no differing word is consumed.

Nothing of the card has been compared with NEMO's record yet.

## 1. What this round does

1. Twin check: full planes are compared on the inner domain, every other
   array whole; the count of differing halo words is reported. A planted
   inner-domain bit must refuse. Then re-admission with --admit-existing. NEMO
   is not re-run.
2. Geometry identity, every level.
3. Ladders kt = 1..10, both labels, split by region and by depth class.
4. The 100-step record.
5. The first row that is not bit-identical, with its NEMO statement.
6. Landing only of a cited statement; re-score after it.

## 2. Limits of the record, fixed before measuring

- Stage and step frames hold 2-D fields and level 1 of U, V, T, S. Levels 2 to
  30 exist only in restarts: steps 2, 4, 6, 8, 10 (twin b), 10, 20, .., 100
  (twin a).
- INDEPENDENT rows: level 1 and 2-D fields at every step; all levels at steps
  2, 4, 6, 8, 10 and every tenth step after.
- GIVEN-NEMO-ENTRY needs all 30 levels at entry, so entries are kt = 1 (the
  initial state, equal to the independent row) and kt = 3, 5, 7, 9 (NEMO's
  restart of the step before: all prognostics and the six barotropic
  histories). Scored: level 1 and 2-D after one step; all levels after two
  steps (at kt = 4, 6, 8, 10), labelled as two-step rows. Entries at even kt
  are not possible with this record and are not reported.
- Depth classes, by level, at U, V and T points separately: FULL = wet cell
  whose rest thickness equals the 1-D ladder's; PARTIAL = wet cell whose rest
  thickness differs from it (a bottom cell, or a face next to one).

## 3. Predictions (frozen)

- R2-P1 (geometry). The 36 rows of the G1 gate, now compared on all levels
  (30 card levels against NEMO's levels 1-30, level 31 required dry):
  **36 of 36 bit-identical.** Falsifier: any unequal row. Most exposed: the
  vorticity mask at the two straits, the face thicknesses at partial cells.
  Plant: one moved element of the card's vorticity mask turns that row red.
- R2-P2 (step 1, level 1). No row is bit-identical; every row is at the
  1e-15 bar. Falsifier: either half.
- R2-P3 (first row not bit-identical) is round 1's G-P7, unchanged: the
  solver's entry forcing at step 1 on faces inside the bump (PLAUSIBLE).
  Falsifier: that frame is bit-identical.
- R2-P4 (depth class, PLAUSIBLE). At the all-level steps the PARTIAL class of
  U and V has a larger error relative to NEMO's own maximum than the FULL
  class at the same level. Falsifier: FULL is larger at most levels.
- R2-P5 (regions). At kt = 1 fold, seam and south hold no unequal cell (the
  signal has not arrived). Falsifier: an unequal cell there at kt = 1.
- R2-P6 (100 steps). Some field leaves the 1e-15 bar before step 100; which
  step is not predicted. Falsifier: every row at the bar through step 100.
- R2-P7 (tiny numbers) is round 1's G-P9, unchanged.
- R2-P8 (G1 unchanged). No G1 row moves: nothing landed this round is
  selected by default, and the G1 certification test passes on the final
  tree. Falsifier: a moved G1 row.

Not predicted: the size of any error; whether any statement lands.

## 4. Choices

ASKED (round brief): map the refusal first; narrow the twin check to consumed
support and re-admit without re-running NEMO; geometry identity with a
predicted count and a plant; both ladders with the region split and a depth
class split; the 100-step record; first non-bit statement cited; land only
cited statements; one read-only review.

UNASKED, each offered for revert:
1. "Consumed support" is taken as the inner domain of every full plane (what
   the one reader uses), not a per-array wet mask.
2. Given-entry at odd kt only, with two-step all-level rows (section 2).
3. The depth-class definition of section 2.
4. A new G2 ladder script beside G1's instead of level arguments threaded
   through G1's 2,300-line script (G1's script and its certified numbers stay
   untouched).
