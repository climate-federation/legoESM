# ORCA2 round 71 preregistration — independent month ranking

Date frozen: 2026-09-28

Base: `5311ab72959512901a105eec9621ee754c34f2d6`

Claim label: **independent**.  legoESM starts from the ORCA2 card's own ocean
state.  No NEMO entry field is substituted.  NEMO's own from-rest terminal
restart is the comparator.

Sea ice remains out of scope.  The card's six-entry `unmeasured_features`
tuple, all selectors, the 10,800 s time step, and the 240-step window remain
frozen.

## Known input, not a prediction

The operator ran round 69's committed launcher before this preregistration.
Its retained log ends with `STATUS PASS_MONTH_SURFACE_RECORD` and names 480
self-describing rank-step surface frames.  That observation is an external
input to this round, not preregistered evidence.  The scientific predicates
below are measured only after this file is committed.

## Compiled boundaries read before measurement

The admitted NEMO program linearly combines its before/after forcing records
at `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246`.
Its terminal ocean restart writes `sshn`, `un`, `vn`, `tn`, and `sn` from the
before slot at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184`,
called after the RK3 state rotation at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:255-271`.

## Frozen predictions and falsifiers

1. **Record admission.**  Re-running the committed round-69 admission against
   the existing target predicts `PASS_MONTH_SURFACE_RECORD`: 480 frames, 200
   ten-step raw-bit field comparisons, four bit-exact terminal restart shards,
   and the pinned producer/source digests.  Any failed predicate is a record
   refusal.  All six existing plants must still fire.
2. **Ten-step instrument calibration.**  The month runner predicts that its
   independent state after step 10 reproduces every field metric in round 66's
   independent kt=10 stage-3 row exactly.  A differing count, maximum, mean,
   RMS, bit flag, or first unequal index invalidates the month instrument; no
   step-240 number may then be quoted.
3. **Independent execution.**  Starting from the card's own state with no SSH
   bridge predicts 240 completed production-JIT CPU fp64/libm steps, consuming
   exactly one admitted two-rank surface frame per step.  Every terminal field
   must be finite fp64.  A missing frame, changed card registry, bridge, wrong
   precision, non-finite value, or incomplete step count refuses.
4. **Terminal restart frame.**  The two NEMO ocean restart shards predict one
   unambiguous 148 x 180 owned-domain frame after the two-cell halos are
   removed and the longitude slabs are concatenated.  The assembled restart
   latitude/longitude must match the card orientation, and planted reversal or
   one-ULP payload changes must refuse.
5. **Magnitude ranking.**  All five terminal fields (`T`, `S`, `u`, `v`,
   `ssh`) predict non-bit rows.  Temperature predicts the largest raw
   `max_abs` row, continuing the step-10 ordering.  A different largest row is
   **REFUTED**, not reinterpreted.  The report records both max-absolute and
   RMS rankings because the fields have different units; neither is called a
   dimensionless cross-field skill score.
6. **Disposition.**  This is a measurement round.  It predicts **HELD** with
   the terminal ranking and a named largest row.  No physics landing is
   authorized before that month-scale ranking exists.

Failed predictions remain **REFUTED** in the receipt.  The ten-step
calibration, orientation check, and terminal scoring are separate predicates;
none may waive another.

## Required controls and validation

- Reuse the existing round-1 ORCA2 card, production step, field scorer, and
  forcing mapper; add no second ocean implementation.
- Parse the round-69 self-describing surface records and the admitted NEMO
  restart schema; do not predict whole-file byte counts.
- Wrong initial mode, dropped surface frame, step-10 score mutation, restart
  orientation reversal, terminal one-ULP mutation, and non-finite terminal
  payload must each refuse.
- Run the focused tests, both citation gates with a firing receipt citation
  plant, the 170-test card battery, and the required separate read-only Codex
  review.  No `packages/` change is authorized.

ASKED: rank Decision 52's independent ORCA2 month-scale ocean errors by
magnitude under the admitted forcing protocol.

UNASKED: configuration, selector, threshold, forcing reconstruction,
stabiliser, carried state, model arithmetic, sea ice, and the held QCO/RK
change.
