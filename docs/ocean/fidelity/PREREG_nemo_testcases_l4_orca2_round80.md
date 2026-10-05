# ORCA2 round 80 preregistration — independent month and kt=10 vertical-mixing extremes

Date frozen: 2026-10-01

Base: `16e99b7eb694` (Decision 77's internal-wave-mixing arm and molecular
background reset are already landed).

This round has two separately labelled measurements.

* **Independent month:** legoESM starts from the ORCA2 card's own T, S, u, v,
  and sea surface.  NEMO's own from-rest 240-step restart is only the
  comparator.  The round-71 month protocol, records, 10,800 s step, surface
  operands, fp64/libm policy, and terminal five-field scorer are frozen.
* **Given NEMO's entry:** the ten-step diagnostic replaces only the initial
  sea surface with NEMO's recorded value, as authorized by Decision 52.  The
  baseline and each explicitly named avt/avm consumer arm use the same entry,
  surface operands, records, and ladder scorer.

Sea ice remains out of scope.  The ORCA2 card's exact seven-entry
`unmeasured_features` registry, all selectors, carried state, thresholds, and
model arithmetic are frozen.  No model or configuration landing is authorized
by this preregistration.

## Existing evidence, not predictions

Round 79a's post-fold-in independent month is the BEFORE row: T rms
`0.18871605115627163`, max `21.16204086038964`; S rms
`0.04399530164405456`, max `4.2942151451623225`; u rms
`0.019175453374117628`; v rms `0.012956413517199093`; ssh rms
`0.06850832563417603`.  Those numbers predate Decision 77's wave arm and
background reset.

Round 79b's three-arm ladder records the landed baseline at kt=10 stage 3 as T
max `1.23675` and rms `0.0102148`, while the background-only arm is T max
`1.24107`.  The operator note abbreviates the landed extreme as `1.2411`.
This disagreement is not explained away: the round-80 baseline must reproduce
the landed background-plus-wave row before any consumer substitution is
interpreted.  Reproducing the background-only value instead is an instrument
failure, most likely an unthreaded wave-forcing map.

The admitted round-79b record carries NEMO's final `avt` and `avm` for both
ranks and all ten steps.  In the record's compiled source the closure is copied
to the live arrays, then river-mouth, convection, double-diffusive, and
internal-wave contributions run in that order
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/zdfphy.f90:349-381`).
NEMO's tracer consumer selects avt for temperature and avs for salinity before
forming its implicit matrix
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/trazdf.f90:173-235`).
Its momentum consumer forms the implicit matrix from adjacent T-point avm
values
(`ORCA2_ORCA1ICE_OMIP_L4_R79BZDF/BLD/ppsrc/nemo/dynzdf.f90:183-206`).

Search-before-build found and will reuse the round-71 month runner, round-72
wet-volume and supplied-basin scorer, round-1 ten-step ladder/scorer, round-79b
self-describing record parser, and the existing private production-JIT
vertical-K consumer seam.  No second model path, record parser, or implicit
solver will be written.

## Frozen predictions and falsifiers

1. **Month instrument.**  The 240-step run completes under production CPU JIT
   fp64/libm; the first ten steps remain exactly calibrated to the admitted
   short record.  A calibration difference or changed input digest refuses the
   measurement.
2. **Month ranking.**  All five terminal fields remain non-bit, and temperature
   remains first by both maximum absolute error and unweighted RMS.  Any exact
   field or another leading field is retained as **REFUTED**.
3. **Decision-77 month effect.**  Based on the ten-step volume result, the
   wave-plus-background pair is predicted to reduce the independent-month T
   and S RMS relative to round 79a.  Either RMS staying equal or increasing is
   **REFUTED**.  No direction is predicted for maxima: both before/after values
   will be registered.
4. **Month region.**  The supplied Atlantic/Pacific/Indian masks plus exact wet
   complement remain binary, disjoint, and exhaustive.  Pacific is predicted
   to retain the largest wet-volume-weighted T squared-error contribution.
   A different owner is **REFUTED**.  T and S terminal argmax cells are also
   reported with their supplied-mask region.
5. **Landed ten-step baseline.**  The ordinary kt=1..10 run must reproduce
   round 79b's background-plus-wave kt=10 stage-3 row, not its background-only
   row.  Failure stops the consumer arms; it is not scored as a physics result.
6. **Extreme geography.**  The rank-0 kt=10 T and S argmax cells are predicted
   not to be on the northern-fold row and not to be river-mouth columns.  Each
   predicate is scored independently.  Land adjacency and convection-site
   membership are observations, not assumptions.
7. **Recorded-consumer substitutions.**  Replacing NEMO avt only at the live
   implicit consumer is predicted to move the kt=10 T and S extremes toward
   NEMO.  Replacing avm only is predicted to have a smaller effect on both
   tracer extrema.  The pair is predicted to move at least as far as avt alone.
   Each failed ordering is retained as **REFUTED**; no cancellation is inferred
   from global means.
8. **Process ranking at each extreme.**  At both interfaces bounding each
   argmax cell, rank the recorded closure, river-mouth, convection,
   double-diffusive, and internal-wave contributions by absolute coefficient.
   At least one argmax cell is predicted to touch an enhanced-convection
   interface.  If neither does, that prediction is **REFUTED**.  This is a
   given-entry kt=10 coefficient ranking, not an independent-month attribution.
9. **Disposition.**  The round predicts **HELD** unless one source-exact,
   one-variable consumer statement passes the full ORCA2 ladder and shared-card
   landing gates.  A missing second tracer diffusivity or river-mouth arm is
   named as an OPEN statement, never approximated or stabilized.

## Controls and validation

The round-80 gate must refuse a one-ULP terminal month mutation, an overlapping
region mask, a moved ten-step baseline, an inert avt substitution, an aliased
avm arm, and a false argmax location.  The receipt citation gate must pass for
the default and round receipt, and a real citation plant must fire.  Request the
separate `codex exec --sandbox read-only` review and retain its verdict or exact
failure.  Run focused round-71/72/79b/80 tests, the card battery, and the wide
ocean-fidelity battery once, one pytest process at a time.

ASKED: remeasure the independent month after Decision 77; report every field
BEFORE to AFTER and the wet-region ranking; locate the given-entry kt=10 T/S
extremes; rank the recorded vertical-mixing processes there; substitute NEMO's
avt and avm at their consumers one variable at a time.

UNASKED: selectors, thresholds, configuration, model arithmetic, stabilisers,
carried state, NEMO output, sea ice, double-diffusive implementation,
river-mouth implementation, and any landing not already authorized by the
standing rules.
