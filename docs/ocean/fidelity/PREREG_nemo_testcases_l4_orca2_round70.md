# ORCA2 round 70 preregistration — month-surface record handoff

Date frozen: 2026-09-28

Base: `e804de64a`

Claim label: **independent**.  The future legoESM month starts from its own
ORCA2 card state.  This round does not substitute a NEMO entry field and does
not produce a month-error number.

Sea ice remains out of scope.  The ORCA2 card, every selector, and its
six-entry `unmeasured_features` tuple remain unchanged.

## Known input, not a prediction

The initial orientation census found no round-69 target configuration and no
round-69 run directory.  That observation preceded this file and is therefore
not presented as preregistered evidence.  Round 69 already preregistered the
scientific record predicates before any acquisition:
`PREREG_nemo_testcases_l4_orca2_round69_month_surface_inputs.md`.

## Frozen predictions and falsifiers

1. **Record census.**  A post-commit census predicts that the requested run is
   still absent.  Finding its run directory refutes this and triggers
   `--admit-existing`; no rebuild is permitted.
2. **Acquisition handoff.**  With a clean committed tree, the existing
   round-69 launcher predicts `ORCA2_ROUND69_SURFACE_PREFLIGHT_READY` without
   creating a NEMO target or run directory.  Any pin, artifact, syntax, schema,
   or preflight failure refutes readiness and must be repaired before another
   acquisition request.
3. **Source boundary.**  The admitted compiled program completes `sbc` before
   the writer boundary at
   `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:151-152`.
   Source preflight predicts the committed additions-only patch still places
   the WRITE-only call at that boundary and changes no model field.
4. **Disposition.**  If the record remains absent and preflight passes, this
   round predicts `STOPPED_FOR_RECORD` and reissues the same launcher.  If the
   record exists, it predicts admission followed by the independent 240-step
   legoESM trajectory and terminal SSH/T/S/u/v magnitude ranking.

Failed predictions remain **REFUTED**.  The complete inventory, ten-step
calibration, and terminal-restart passivity predicates frozen in round 69
remain unmeasured until the operator record exists; none may waive another.

## Required controls

- Run the acquisition preflight only after this preregistration is committed.
- Preserve all six round-69 plant tests; do not relax a count, schema, or pin.
- Run the focused acquisition tests, both citation gates with a firing plant,
  and the required separate read-only Codex review.
- No `packages/` or configuration change is authorized in this record-handoff
  round.

ASKED: continue Decision 52's independent ORCA2 month ranking from the actual
branch tip.

UNASKED: configuration, selector, forcing reconstruction, thresholds,
stabiliser, carried state, model arithmetic, sea ice, and the held QCO/RK
change.
