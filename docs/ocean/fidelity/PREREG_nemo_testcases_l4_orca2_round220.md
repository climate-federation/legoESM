# Preregistration: ORCA2 round 220 — OMT-3 lateral momentum diffusion

Date: 2026-10-10. Frozen base: `fee85863a`. Scope: Decision-109 OMT-3,
the admitted OMT-2 deck plus the shipped ORCA2 lateral momentum-diffusion
module. This file is committed before any OMT-3 trajectory or record
measurement.

No package physics, shipped ORCA2 card, sea-ice selector, carried state,
stabiliser, threshold, or `unmeasured_features` tuple changes in this round.
The OMT-3 card remains gate-local until its record admits. All later
trajectory numbers will be labelled either **independent OMT-3** or **given
NEMO's entry OMT-3**.

## Source statement and controlled edge

The admitted OMT-2 deck resolves `ln_dynldf_OFF=.true.` and
`ln_dynldf_lap=.false.`. OMT-3 restores the rung-0 `&namdyn_ldf` choices:
`ln_dynldf_lap=.true.`, removes the explicit `ln_dynldf_OFF=.true.` override
so the reference value remains false, and retains `ln_dynldf_lev=.true.` plus
`nn_ahm_ijk_t=-30`.

The compiled record build reads both namelists at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-185`, requires
exactly one OFF/laplacian/bilaplacian choice at `ldfdyn.f90:221-228`, resolves
the partial-step iso-level laplacian at `ldfdyn.f90:246-258`, and allocates
the live viscosity fields only when the OFF selector is false at
`ldfdyn.f90:297-313`. The stage program calls lateral momentum diffusion only
at stage 3 (`stprk3_stg.f90:488-494`); the selected dispatcher calls the
iso-level laplacian at `dynldf.f90:81-90`. Its live div-curl statement is in
`dynldf_lev.f90:121-140`.

Search-before-build found the proven round-218 OMT-2 binary-reuse launcher,
deck gate, and frame-record gate. Round 220 copies that protocol file by file;
only the deck edge, OMT-3 target names, and round-specific status labels may
change. The NEMO executable and additions-only P3 writer remain byte-identical.

## Frozen predictions and falsifiers

1. **R220-P1 — exact one-module deck edge.** CONFIRM: the canonical OMT-3
   deck differs from admitted OMT-2 only by
   `namdyn_ldf.ln_dynldf_lap: .false. -> .true.` and removal of the explicit
   `namdyn_ldf.ln_dynldf_off=.true.` override; vector momentum advection,
   linear implicit drag, tracer-advection OFF, and tracer-diffusion OFF remain
   unchanged. REFUTE: any other namelist assignment, resolved selector, input
   hash, or run-control physics line moves.
2. **R220-P2 — ten-step record exists.** Lateral momentum diffusion carries
   NEMO through kt=10. CONFIRM: the identical two-step smoke and both
   rank-complete additions-only twins finish kt=10, their 80 frames are finite
   and bit-identical, and terminal restarts match the uninstrumented
   calibration. REFUTE: NEMO stops at or before kt=10, any frame is missing or
   malformed, either twin differs, or the writer changes a terminal bit.
3. **R220-P3 — 96-step boundary.** The OMT-3 oracle survives beyond OMT-2's
   kt=11 speed stop and completes the requested 96 steps. CONFIRM: the
   uninstrumented boundary run reaches `STOP 0` at kt=96 with finite requested
   restarts. REFUTE: compiled `stp_ctl`, a non-finite field, or another runtime
   refusal stops it earlier; the exact first boundary replaces this prediction
   and remains recorded as REFUTED.
4. **R220-P4 — module attribution.** The complete round-217 vector unit first
   recreates the independent kt=8 live-W-thickness refusal on OMT-3, naming
   lateral momentum diffusion as the first module that contains its missing
   cancelling partner. CONFIRM: the baseline OMT-3 ladders complete but the
   atomic candidate refuses first at kt=8 on the registered live-W guard.
   REFUTE: the candidate completes the available OMT-3 ladder, refuses at a
   different boundary, loses an exact row earlier, or the baseline itself
   refuses. This prediction is measured only after the record admits; it does
   not authorise a production landing.
5. **R220-P5 — controls.** Deck extra-delta/wrong-selector plants; record
   cadence/header/field/truncation/non-finite/missing-frame/twin-ULP/
   terminal-byte/binary/early-boundary plants; and future card/Decision-96
   plants must all refuse. Any green plant invalidates its associated number.

## Stop conditions

The operator runs NEMO acquisitions. If the OMT-3 record is absent after this
round's committed preflight, status is `STOPPED_FOR_RECORD` and the exact
launcher path is reported. No OMT-3 trajectory claim is made from OMT-2 data.
If the record admits in-round, both labelled ladders and the complete unit are
scored before OMT-4. No configuration decision is requested.
