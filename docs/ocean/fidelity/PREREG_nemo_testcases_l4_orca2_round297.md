# ORCA2 round 297 preregistration — kt=7 stage-1 writer record

Date: 2026-10-16. Frozen base: `6ca3a8892`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. Round 296 leaves the
complete private fold unit finite through kt=7 and first grows by more than
10x at kt=7 stage 1 in the fold band. This round walks that finite stage in
compiled source order. It changes no production package, card, deck, selector,
carried state, stabiliser, sea-ice field, or `unmeasured_features` entry.

## Compiled order and record requirement

The compiled OMT-4 stage program executes the vector-form stage-1 momentum
RHS before `stp_RK3_stg`, then applies the raw velocity update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:467-481`, the
barotropic correction at `:521-550`, the active-tracer transport at `:568-592`,
tracer advection at `:617-662`, the surface tracer source at `:664-672`, the
stage-1 QCO tracer update at `:691-720`, and the post-stage U/V/T/S association
at `:798-820`.

The admitted round-222 stage frames contain only entry and completed-stage
states at kt=7. The admitted round-280 source record has the needed 23
self-describing intermediate fields, but only at kt=1. Therefore the first
source-ordered writer at kt=7 cannot be named without the same rank-complete
schema at kt=7.

The acquisition copies the round-280 launcher, writer, parser, and additions-
only proof. The only scientific record change is the selected step, 1 to 7;
the target has a new magic/name/build/run identity. Admission parses every
header and payload, requires exactly-once global rank coverage, and requires
all 80 inherited stage frames plus the two kt=10 restarts to remain byte-
identical to the admitted OMT-4 source run.

## Frozen predictions and falsifiers

1. **R297-P1 — record gap.** No admitted rank-complete record contains the
   kt=7 stage-1 `rhs/pre/cor/trp/adv/sbc/qco/final` sequence. Finding such a
   record refutes the acquisition need; it must be used instead.
2. **R297-P2 — additions-only acquisition.** Two rank records cover the
   148x180 domain exactly once, carry all 23 named fields in the inherited
   order, and reproduce the 80 inherited frames and both terminal restarts
   byte-for-byte. Any moved inherited byte or malformed field refuses the
   record.
3. **R297-P3 — first writer.** Replaying the private unit from the finite kt=7
   entry under both labels first exceeds the prior stage maximum after tracer
   advection (`adv_t`/`adv_s`). An earlier `rhs/pre/cor/trp` boundary or a later
   `sbc/qco/final` boundary is a retained **REFUTED** prediction and becomes
   the next walk boundary.
4. **R297-P4 — program-point control.** Candidate and oracle rows are compared
   at the same named boundary. Candidate self-replay and NEMO record replay
   must each reproduce their own completed stage below the campaign floor;
   otherwise no writer is citable.
5. **R297-P5 — controls.** Header, field-name, field-dimension, truncation,
   swapped-rank, coverage, inherited-frame byte, restart byte, writer layout,
   and one-bit plants must each refuse at the named invariant.

## Stop conditions

This sandbox must not run `mpirun`. A complete fail-closed launcher with a
clean preflight ends the round as `STOPPED_FOR_RECORD`; the operator runs it.
After admission, the next round scores both labels and walks the first writer
offline. The complete fold unit stays private and atomic. Decision 125's
top-V exchange and the resolved stage-mask statement remain off in production.
Decision 126 remains pending; shipped runoff and sea ice are unchanged.

## Citation correction appended 2026-10-16

The frozen compiled-order paragraph cites `stprk3_stg.f90:467-481` for the
raw vector update. Lines 467-476 are the stage-integration heading and case
dispatch; the executable vector update is exactly `:477-480`. The frozen
prediction and source order are unchanged; this correction governs the
receipt and citation gate.

## Citation correction appended 2026-10-16, second

The frozen compiled-order paragraph cites `stprk3_stg.f90:568-592` for the
active-tracer transport branch. The branch begins at line 565; lines 568-592
contain its calls and second dispatch. The mechanically anchored span is
`:565-592`. No prediction, program order, or record field changes.
