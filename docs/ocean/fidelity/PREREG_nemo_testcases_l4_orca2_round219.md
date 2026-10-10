# Preregistration: ORCA2 round 219 — OMT-2 admission and vector-unit score

Date: 2026-10-10. Frozen base: `618a2eec6`. Scope: admit the existing
round-218 OMT-2 linear-drag record after repairing its exact compiled-stop
matcher, build the gate-local OMT-2 card, run both labelled kt=1..10 ladders,
and score the complete round-217 vector unit atomically.

No NEMO run, package physics, shipped ORCA2 card, sea-ice selector, carried
state, stabiliser, threshold, or `unmeasured_features` tuple changes in this
round. All trajectory numbers are labelled either **independent OMT-2** or
**given NEMO's entry OMT-2**. The existing record is read-only; a checker
mismatch is repaired and re-admitted without rebuilding or rerunning NEMO.

## Source and observed record boundary

The operator-run smoke, calibration, and two P3 twins reached `STOP 0` through
kt=10. The 96-step boundary run stopped at kt=11. Compiled
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stpctl.f90:243-250` selects the
extrema refusal, and `stpctl.f90:293-310` emits the full stop text and abort
state. The round-218 launcher checked only a shorter literal prefix and thus
misclassified the expected boundary as an acquisition failure. The repair
must match the compiled full line (or a mechanically anchored invariant
prefix) and must not weaken the step, abort-code, abort-state, or record
identity checks.

The OMT-2 card edge is the already-cited NEMO linear implicit composition:
`zdfdrg.f90:258-284,540-547`, `dynzdf.f90:158-169,303-312,470-474`, and
`dynspg_ts.f90:1404-1463` in the same compiled build. It changes OMT-1 only to
`bottom_drag_scheme="nemo_linear"`, the deck-resolved `rn_Cd0=1.e-3` and
`rn_Uc0=0.4`, `zdf_drag_in_matrix=True`,
`zdf_baroclinic_only=True`, and `barotropic_drag_substep=True`.

## Frozen predictions and falsifiers

1. **R219-P1 — existing record admits without rerun.** CONFIRM: both twins
   contain 80 self-described frames, all twin fields are bit-identical, their
   kt=10 terminal restarts are byte-identical to calibration, all content
   plants fire, and the month boundary admits as compiled `stp_ctl` at kt=11
   with a rank-0 abort state. REFUTE: any content/identity comparison fails or
   the boundary is not exactly kt=11. No failed comparison may be waived.
2. **R219-P2 — round-218 month prediction is refuted.** CONFIRM: the admitted
   record classifies R218-P3 as REFUTED at kt=11, with `|V|=10.54 m/s` as the
   triggering printed extremum. REFUTE: the gate finds a later/completed
   boundary or a different triggering field.
3. **R219-P3 — exact card edge and entry.** CONFIRM: the gate-local OMT-2 card
   differs from OMT-1 only in the five drag/composition fields above; both
   labels retain the admitted bit-exact kt=1 entry classification. REFUTE: any
   other resolved field changes or either entry leaves the admitted class.
4. **R219-P4 — baseline OMT-2 ladders.** CONFIRM: both baseline labels complete
   40 checkpoints / 200 rows through kt=10 and remain finite. REFUTE: either
   label refuses, becomes non-finite, or has an incomplete row set.
5. **R219-P5 — complete vector unit at OMT-2.** The linear-drag rung supplies
   the compensating partner missing on OMT-1/rung 0. CONFIRM: both candidate
   OMT-2 ladders complete kt=1..10, the first-over-bar row moves toward or is
   unchanged, no exact row is lost, a strict majority of RMS-moved rows move
   toward NEMO, and final stage-3 SSH maximum does not worsen. REFUTE: any
   runtime refusal, exact-row loss, majority-away census, earlier first debt,
   or worse final SSH maximum. A qualifying OMT-2 result does not authorise a
   rung-0 production landing.
6. **R219-P6 — controls.** The repaired stop-line mismatch plant, every
   round-218 record-content plant, OMT-2 card-module and entry-bit plants, and
   Decision-96 exact-loss/false-majority plants must all refuse. Any green
   plant invalidates its associated number.

## Stop conditions

If the existing record does not admit, status is `STOPPED_FOR_RECORD` with the
exact failed predicate; NEMO is not rerun. If OMT-2 baseline or candidate
refuses, status is `HELD` with the first boundary. OMT-3 waits. No
configuration decision is requested.
