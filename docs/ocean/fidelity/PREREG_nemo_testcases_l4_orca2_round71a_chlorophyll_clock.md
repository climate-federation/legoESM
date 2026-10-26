# ORCA2 round 71a preregistration addendum — chlorophyll clock retraction

Date frozen: 2026-09-28

Base preregistration: `d72c49822`

This addendum was frozen before any legoESM month execution.  It does not
rewrite round 71's original predictions.

## Source finding

Round 71's instrument review refuted the assumption behind prediction R71-P2.
The shared ORCA2 ladder helper uses `(248 + kt) / 496` for every step.  NEMO's
compiled statement instead forms
`(isecsbc - before_seconds) / (after_seconds - before_seconds)` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246`.
The admitted run log independently resolves the records:

- kt=1: midpoint 0.0625 day, December/January centres -15.5/15.5 days;
- kt=240: midpoint 29.9375 days, January/February centres 15.5/45.0 days.

Therefore steps 1--124 use `(2*kt + 247) / 496`, and steps 125--240 use
`(2*kt - 249) / 472`, with NEMO's before-multiply plus after-multiply
association.  The old expression happens to agree at kt=1, the only emitted
chlorophyll witness the ten-step ladder checked, and is already wrong at kt=2.

## Retraction and replacement frozen predicates

1. Original R71-P2 (exact reproduction of round 66's kt=10 metrics) is now
   predicted **REFUTED**.  Round 66's independent kt=2--10 trajectory numbers
   used the stale half-rate chlorophyll clock and are retracted as exact-input
   results.  The kt=1 entry result is unaffected.
2. The replacement calibration runs two ten-step legoESM arms from the same
   independent state with the same corrected chlorophyll clock.  One reads
   round 69's self-describing surface frames; the other reads the admitted
   round-5 surface frames.  They predict raw-bit identity for all terminal
   `T`, `S`, `u`, `v`, and `ssh` cells.  Any difference invalidates the month
   reader and blocks step-240 scoring.
3. The corrected new-schema arm also rescored against NEMO's kt=10 stage-3
   frame predicts at least one metric differs from round 66.  Exact equality
   **REFUTES** the source-clock diagnosis and blocks the month until reconciled.
4. Round 71's remaining predictions stand: independent card-own entry, 240
   finite fp64 steps, exact restart orientation/provenance, five terminal rows,
   temperature predicted largest by raw maximum, and HELD disposition.

This is an instrument correction only.  It changes no model file, card,
selector, forcing source, time step, sea-ice state, or landing predicate.
