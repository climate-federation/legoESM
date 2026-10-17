# PREREG — ORCA2 round 308: rung G6's certified statements under Decision 122

Frozen before any round-308 measurement.  Base `8007f03c5`; import commit
`a4a1bd214` (gridladder package hunks, every selector default-off).

## What is measured

Gate `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round308_g6_d122_gate.py`,
15 arms (control; VECORD, EENORD, TOPF, FOLDUNIT, TOPV, RAWV, HPGF, WZVR,
ZADA, CLOCK, KMM, CARRY singly; FOLD = FOLDUNIT+TOPV+RAWV+HPGF; ALL) on the
seven D96 pairs (rung 0, OMT-1..4 independent; rung 10 independent and
given NEMO's entry), production JIT, CPU, fp64, libm.  Records as round 307.
D122 (c) no exact row lost and (d) first over-bar row unchanged or toward,
scored by the round-276 classifier against `control` on the same tree.

Not measured and stays off: the stage-1 3-D V transport's associated V mask
(round 303), which no rung certifies.  Not imported: G1b's subnormal-KEG
workaround (an XLA flush workaround; Decision 127 already scores flushed
subnormals as matching).

## Predictions (frozen)

- R308-P1: every rounding-form arm (VECORD, EENORD, WZVR, ZADA, CLOCK, KMM)
  loses 0 exact rows and leaves the first over-bar row unchanged on all
  seven pairs.  Falsifier: any exact row lost or first over-bar earlier.
- R308-P2: FOLDUNIT alone refuses (non-positive live W thickness) on at
  least one of rung 0, OMT-4, rung 10 (round 282).  Falsifier: it completes
  every ladder.
- R308-P3: TOPV alone fails D122 (c) on rung 0 or OMT-4 (Decision 125's
  growth).  Falsifier: it passes (c)+(d) on all seven pairs.
- R308-P4: CARRY leaves every kt-1 row identical on the five independent
  ORCA2 ladders (from rest the slot holds the entry height).  Falsifier:
  any kt-1 row moves.
- R308-P5 (PLAUSIBLE): FOLD refuses or fails (c) on at least one pair,
  because its round-303 partner (the uncertified transport mask) is absent.
  Falsifier: FOLD passes (c)+(d) on all seven pairs.
- R308-P6: with every new selector off, GYRE's ten-step residual archive and
  day-30 snapshots are byte-identical base vs import, and the DINO and VORTEX
  step jaxprs are identical.  Falsifier: any byte differs.

## Landing rule

A statement (or a group measured together AND apart) lands on a card only if
it passes D122 (c)+(d) on all seven pairs; TOPV stays off regardless while
Decision 125 holds it.  Every other statement stays off with its numbers.
