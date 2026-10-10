# Preregistration: ORCA2 round 222 — OMT-4 tracer advection

Date: 2026-10-10. Frozen base: `9363ce511`. Scope: Decision-109 OMT-4,
the admitted OMT-3 deck plus the shipped ORCA2 tracer-advection module. This
file is committed before any OMT-4 trajectory or record measurement.

No package physics, shipped rung-0/rung-10 card, sea-ice selector, carried
state, stabiliser, threshold, or `unmeasured_features` tuple changes in this
round. The OMT-4 card remains gate-local until its record admits. All later
trajectory numbers will be labelled either **independent OMT-4** or **given
NEMO's entry OMT-4**.

## Source statement and controlled edge

The admitted OMT-3 deck resolves `ln_traadv_OFF=.true.` and
`ln_traadv_fct=.false.`. OMT-4 restores the rung-0 `&namtra_adv` choices:
`ln_traadv_fct=.true.`, removes the explicit `ln_traadv_OFF=.true.` override,
and retains `nn_fct_h=2` plus `nn_fct_v=2`.

The compiled record build declares and reads those selectors at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv.f90:592-602`, requires one
and only one tracer-advection choice at `traadv.f90:630-647`, and resolves FCT
at `traadv.f90:668-674`. The live stage program calls tracer advection at
`stprk3_stg.f90:633-643`. The dispatcher evaluates centred advection at
stages 1 and 2 and FCT at stage 3 at `traadv.f90:497-540`.

Search-before-build found the proven round-220 OMT-3 binary-reuse launcher,
deck gate, frame-record gate, and round-221 card/ladder gate. Round 222 extends
that protocol; it does not create a second ladder implementation. The NEMO
executable and additions-only P3 writer remain byte-identical.

## Frozen predictions and falsifiers

1. **R222-P1 — exact one-module deck edge.** CONFIRM: the canonical OMT-4
   deck differs from admitted OMT-3 only by
   `namtra_adv.ln_traadv_fct: .false. -> .true.` and removal of the explicit
   `namtra_adv.ln_traadv_off=.true.` override; vector momentum advection,
   linear implicit drag, lateral momentum diffusion, and tracer-diffusion OFF
   remain unchanged. REFUTE: any other namelist assignment, resolved selector,
   input hash, or run-control physics line moves.
2. **R222-P2 — ten-step record exists.** Tracer advection carries NEMO through
   kt=10. CONFIRM: the identical two-step smoke and both rank-complete
   additions-only twins finish kt=10, their 80 frames are finite and
   bit-identical, and terminal restarts match the uninstrumented calibration.
   REFUTE: NEMO stops at or before kt=10, any frame is missing or malformed,
   either twin differs, or the writer changes a terminal bit.
3. **R222-P3 — month boundary moves later.** OMT-4 survives beyond OMT-3's
   registered kt=15 speed stop. CONFIRM: the 96-step boundary run completes or
   reaches compiled `stp_ctl` strictly after kt=15. REFUTE: its exact compiled
   boundary is kt=15 or earlier; the measured boundary replaces this
   prediction and stays recorded as REFUTED.
4. **R222-P4 — tracer-advection attribution.** The complete round-217 vector
   unit first recreates the independent kt=8 live-W-thickness refusal on
   OMT-4, naming tracer advection as the first module carrying its missing
   cancelling partner. CONFIRM: both baseline OMT-4 ladders complete but the
   atomic candidate refuses first at kt=8 on the registered live-W guard.
   REFUTE: the candidate completes the available ladder, refuses at a
   different boundary, loses an exact row earlier, or the baseline itself
   refuses. This prediction is measured only after the record admits and does
   not authorise a production landing.
5. **R222-P5 — controls.** Deck extra-delta/wrong-selector plants; record
   cadence/header/field/truncation/non-finite/missing-frame/twin-ULP/
   terminal-byte/binary/early-boundary/stop-line plants; and future
   card/Decision-96 plants must all refuse. Any green plant invalidates its
   associated number.

## Stop conditions

The operator runs NEMO acquisitions. If the OMT-4 record is absent after this
round's committed preflight, status is `STOPPED_FOR_RECORD` and the exact
launcher path is reported. No OMT-4 trajectory claim is made from OMT-3 data.
If the record admits in-round, both labelled ladders and the complete unit are
scored before OMT-5. No configuration decision is requested.
