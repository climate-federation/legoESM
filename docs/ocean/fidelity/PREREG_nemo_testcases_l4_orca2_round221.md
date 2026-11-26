# Preregistration: ORCA2 round 221 — OMT-3 admission and vector-unit score

Date: 2026-10-10. Frozen base: `adef2eefc`. Scope: diagnose the operator-run
round-220 OMT-3 acquisition without rerunning NEMO for a checker-only failure,
admit every sound existing record, build the gate-local OMT-3 card, run both
labelled ladders, and score the complete round-217 vector unit atomically.

No package physics, shipped ORCA2 card, sea-ice selector, carried state,
stabiliser, threshold, or `unmeasured_features` tuple changes are authorised.
All trajectory numbers will be labelled either **independent OMT-3** or
**given NEMO's entry OMT-3**. A malformed or incomplete record stops the round;
a checker mismatch is repaired against the self-described record and admitted
with `--admit-existing`.

## Frozen source edge

OMT-3 is the admitted OMT-2 deck plus the shipped lateral momentum-diffusion
module: `ln_dynldf_lap=.true.` and the explicit
`ln_dynldf_OFF=.true.` override removed. The compiled record build reads and
validates those selectors at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-185` and
`ldfdyn.f90:221-228`, dispatches the iso-level laplacian at
`dynldf.f90:81-90`, and evaluates the selected div-curl statement at
`dynldf_lev.f90:121-140`. No other module switch may move.

Search-before-build found the round-220 OMT-3 deck and frame-record gates and
the round-219 OMT-2 card/ladder gate. This round extends those existing tools;
it does not create a second ladder implementation.

## Frozen predictions and falsifiers

1. **R221-P1 — operator exit 123 is an expected compiled boundary or a
   check-only mismatch.** CONFIRM: the smoke, calibration, and both P3 twins
   completed the requested ten steps and the 96-step run stopped strictly
   after kt=10 at a compiled `stp_ctl` boundary, or all record payloads are
   complete and only the committed checker rejected them. REFUTE: either twin
   stops at or before kt=10, any required frame is absent/malformed/non-finite,
   terminal restart identity fails, or the NEMO run itself reports an
   instrument/configuration error.
2. **R221-P2 — existing OMT-3 record admits without a NEMO rerun.** CONFIRM:
   80 self-described frames per twin, all 400 twin field comparisons, and all
   four terminal-restart comparisons pass; every content plant fires; the
   month boundary is either kt=96 completion or the exact compiled boundary
   after kt=10. REFUTE: any content or identity predicate fails after a
   header-driven checker repair.
3. **R221-P3 — exact gate-local card edge.** CONFIRM: OMT-3 differs from
   OMT-2 only by the already-shared NEMO div-curl lateral-diffusion selection
   and its recorded coefficient/metric operands; both entry labels retain the
   admitted bit classification. REFUTE: any unrelated resolved field moves or
   either entry classification changes.
4. **R221-P4 — baseline ladders.** CONFIRM: both baseline labels complete the
   admitted OMT-3 record horizon with finite 40-checkpoint/200-row ladders.
   REFUTE: either baseline refuses, becomes non-finite, loses a checkpoint, or
   the record horizon is shorter than kt=10.
5. **R221-P5 — complete vector unit first refuses on OMT-3.** The lateral
   momentum-diffusion rung contains the first missing compensating partner.
   CONFIRM: both baseline ladders complete, while the atomic candidate first
   recreates the registered kt=8 live-W-thickness refusal. REFUTE: the
   candidate completes, refuses at another boundary, loses an exact row
   earlier, or baseline itself refuses. If confirmed, the module is walked in
   compiled source order before any production landing; if refuted, OMT-4 is
   next and no constituent lands.
6. **R221-P6 — controls.** Record cadence/header/field/truncation/non-finite/
   missing-frame/twin-ULP/terminal-byte/boundary plants, card/entry plants,
   Decision-96 exact-loss/false-majority plants, and citation shift plant must
   all refuse. Any green plant invalidates its associated number.

## Stop conditions

An incomplete or physics-invalid record yields `STOPPED_FOR_RECORD` and a new
operator launcher only if a rerun is genuinely required. A checker-only defect
is repaired and the existing output admitted. A baseline or atomic-candidate
runtime refusal yields `HELD` with the first boundary. No configuration
decision is requested.
