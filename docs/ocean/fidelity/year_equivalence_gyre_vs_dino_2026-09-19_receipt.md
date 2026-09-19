# GYRE versus DINO: one-year equivalence to NEMO

Date: 2026-09-19

Status at creation: **PREREGISTERED; no measurement command in this receipt had
been run.**  The commit that first adds this file is the audit record for that
ordering.  Final results will label every number either **MEASURED** or
**REPRODUCED-FROM-ARTIFACT**.

## Question and fixed comparison

Compare legoESM with NEMO at days 30, 60, 90, 120, 180, 240, 300, and 360 for:

- GYRE from rest, using the campaign's GYRE year harness;
- DINO TWIN, using NEMO-bridged initial state and forcing;
- DINO STANDALONE from rest, using existing artifacts unless a rerun costs less
  than two GPU-hours.

For each case and field, the absolute metric is wet-cell RMS of the legoESM minus
NEMO field.  The normalized metric is that RMS divided by NEMO's own spatial
standard deviation for the same field, mask, and day.  T, S, u, and v are 3-D;
SSH is 2-D.  All scoring is fp64.

## Preregistered predictions

- **P1:** GYRE day-360 T3D RMS is in `[3e-4, 3e-3] K`.  **CONFIRM** inside that
  closed interval; **REFUTE** outside it.  Fit `RMS ~ t^a` and report whether the
  series saturates like DINO's.
- **P2:** DINO TWIN day-360 T3D RMS reproduces `3.862e-3 K` within 5%, i.e.
  `[3.6689e-3, 4.0551e-3] K`.  **CONFIRM** inside that closed interval;
  **REFUTE** otherwise and identify what changed.
- **P3:** At every listed day, GYRE's normalized T error is below DINO TWIN's.
  **CONFIRM** only if all eight strict inequalities hold; **REFUTE** at the first
  listed day where one does not.
- **P4:** GYRE's daily T3D RMS series is not monotone; the prior observation is a
  bump near day 23 of `5.28e-4 K`.  **CONFIRM** if at least one daily increment
  is negative; **REFUTE** if every daily increment is non-negative.  Report the
  largest consecutive-day ratio for both GYRE and DINO TWIN.

## Provenance fixed before measurement

- GYRE branch: `docs/year-equivalence-gyre-dino` at
  `c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd`.
- DINO branch: `fix/dino-faithful-grid-true-frame` at
  `f0a82e87ca545e55df548bfd5e0d55c50d1330f8`.
- GYRE NEMO reference:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/nemo_seed0`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence`.

## Comparison table

The final receipt will place the single full comparison table here.  Rows are
the eight preregistered days; each case has absolute and normalized columns for
T, S, u, v, and SSH.

## Commands, controls, results, and verdicts

To be completed after measurement without changing the definitions or verdict
rules above.
