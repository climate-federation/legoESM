# Preregistration: ORCA2 round 218 — OMT-2 linear bottom drag

Date: 2026-10-10. Frozen base: `cc4226e1a`. Scope: Decision-109 OMT-2,
the admitted OMT-1 vector-form deck plus the shipped ORCA2 linear implicit
bottom-drag module. This file is committed before any OMT-2 trajectory or
record measurement.

No package physics, shipped ORCA2 card, sea-ice selector, carried state,
stabiliser, threshold, or `unmeasured_features` tuple changes in this round.
The OMT-2 card remains gate-local until its record admits. All trajectory
numbers will be labelled either **independent OMT-2** or **given NEMO's entry
OMT-2**.

## Source statement and controlled edge

The admitted OMT-1 deck resolves `ln_drg_OFF=.true.` and `ln_lin=.false.`.
OMT-2 restores the rung-0 `&namdrg` block: `ln_lin=.true.`, with
`ln_drg_OFF=.false.` inherited from the compiled reference namelist. The
resolved linear arm has `ln_drgimp=.true.`, `rn_Cd0=1.e-3`, and `rn_Uc0=0.4`.
The compiled record build forms and consumes that drag in
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/zdfdrg.f90:150-195`, removes and
restores the barotropic component at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynzdf.f90:153-181`, and inserts the
implicit partial-cell bottom diagonals at `dynzdf.f90:297-315` and
`dynzdf.f90:464-483`. The split-explicit consumer initialises the face drag
at `dynspg_ts.f90:1404-1493` in the same compiled tree.

The card will reuse the existing exact shared composition already selected by
the seamount SMT-2 card: `bottom_drag_scheme="nemo_linear"`, the resolved
linear coefficients, `zdf_drag_in_matrix=True`,
`zdf_baroclinic_only=True`, and `barotropic_drag_substep=True`. OMT-1's other
four disabled modules remain disabled.

## Frozen predictions and falsifiers

1. **R218-P1 — exact one-module deck edge.** The canonical OMT-2 deck differs
   from admitted OMT-1 only in `namdrg`: `ln_lin .false. -> .true.` and removal
   of the explicit `ln_drg_off=.true.` override so the resolved reference value
   is false. CONFIRM: the deck gate reports exactly those assignment changes,
   the resolved log reports linear implicit drag, and all OMT-1 selectors
   outside `namdrg` are unchanged. REFUTE: any other namelist assignment,
   resolved selector, input hash, or run-control physics line moves.
2. **R218-P2 — ten-step record exists.** Linear implicit drag carries NEMO
   past OMT-1's kt=9 `stp_ctl` boundary. CONFIRM: the identical two-step smoke
   and both rank-complete additions-only twins finish kt=10, their 80 frames
   are finite and bit-identical, and their terminal restarts match the
   uninstrumented calibration. REFUTE: NEMO stops at or before kt=10, a frame
   is missing/malformed, either twin differs, or the record build changes a
   terminal bit.
3. **R218-P3 — 96-step boundary.** Bottom drag stabilises the isolated deck
   through the requested 96 steps. CONFIRM: the uninstrumented boundary run
   reaches `STOP 0` at kt=96 with finite even-step restarts through 90 and the
   step-95 non-terminal restart. REFUTE: compiled `stp_ctl`, a non-finite
   field, or another runtime refusal stops the run before kt=96; the exact
   first boundary then replaces the prediction and is retained as REFUTED.
4. **R218-P4 — card edge and entry.** The gate-local OMT-2 card differs from
   OMT-1 only by the five resolved drag/composition fields named above. Both
   labels retain the admitted bit-exact kt=1 entry classification. REFUTE: any
   other resolved card field changes or either entry comparison leaves the
   admitted classification.
5. **R218-P5 — complete vector unit at OMT-2.** The round-217 indivisible
   vector unit no longer triggers the kt=8 live-W-thickness refusal once the
   shipped drag module is present. CONFIRM: both 200-row OMT-2 candidate
   ladders complete kt=1..10, the first-over-bar row moves toward or is
   unchanged, no exact row is lost, the majority of RMS-moved rows move toward
   NEMO, and final SSH maximum does not worsen. REFUTE: any refusal/non-finite,
   exact-row loss, majority-away census, earlier first debt, or worse final SSH
   maximum. A qualifying OMT-2 result does not authorise landing on rung 0.
6. **R218-P6 — controls.** Deck extra-delta/wrong-selector plants; record
   cadence/header/field/truncation/non-finite/missing-frame/twin-ULP/
   terminal-byte/binary plants; card-module and entry-bit plants; and the
   complete-unit Decision-96 plants must all refuse. Any green plant invalidates
   its associated number.

## Stop conditions

The operator runs NEMO acquisitions. If the OMT-2 record is absent after this
round's committed preflight, status is `STOPPED_FOR_RECORD` and the exact
launcher path is reported. No OMT-2 trajectory claim is made from OMT-1 data.
If the record admits, score both labels and the complete unit before OMT-3.

