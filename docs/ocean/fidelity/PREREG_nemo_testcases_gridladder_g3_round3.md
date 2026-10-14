# Grid ladder rung G3 — round 3 preregistration (gridladder_r16)

Base: f797baf65 (G1, G2 certified; G3 rounds 1-2). Written BEFORE any card
number is compared with the round-15 G3 record
(gridladder_rounds/round15/oracle_g3_r15: step frames kt = 1..10 from the
record binary, twins b/c frames identical, restarts at steps 2..10).
Decision 120 pending: as-built deck; hand-edit cells reported separately.

## Measurements

1. INDEPENDENT chain kt = 1..10: level-1 and 2-D rows at every step from
   the frames; all levels where a restart exists (2..10), by depth class
   and by region (fold / seam / south / coast / open).
2. GIVEN-NEMO-ENTRY: NEMO's restart of kt - 1 as entry, one and two steps,
   at kt = 3..9 (entry restarts 2..8 exist; kt = 1 is the initial state).
3. STAGES at the first step k* where a u or v row is unequal: stage-local
   frames (3-D RHS, stage 1, 2, 3) to place the stage.

## Frozen predictions

- G3-R3-P1. kt = 1 every frame row (all stages) bit-identical.
  Falsifier: one unequal cell at kt = 1.
- G3-R3-P2. The first unequal u/v bit in the INDEPENDENT chain appears at
  some k* in 3..10 and is at the bar there; the bar is crossed later
  because nothing dissipates and the flow is approaching NEMO's own
  kt = 11 instability (growth, not a new statement, after k*). PLAUSIBLE.
  Falsifier: u or v over the bar at the first unequal step.
- G3-R3-P3. GIVEN-NEMO-ENTRY at k* reproduces an unequal cell in one step
  (a statement acting on NEMO's own entry). Falsifier: given-entry
  bit-identical at every kt = 3..9 while the independent chain is not
  (then the first difference is accumulated, not a single-step statement,
  and its first appearance step owns the walk).
- G3-R3-P4. The first unequal cells are in the pressure-gradient / density
  path (they need the real T/S, which is G3's only addition over the
  certified G2). Falsifier: the first unequal stage is one whose operands
  do not involve density.
- Hand-edit cells: unequal counts reported inside them; no prediction.

## What lands

Only a cited NEMO statement whose offline replay on the card's operands
differs from the card and equals NEMO, selected by the G3 card (shared
with ORCA2 where ORCA2 compiles the same statement). Re-score kt = 1..10
after each landing.
