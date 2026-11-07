# NEMO testcase Lane 4 — ORCA2 card round 43 preregistration

Date: 2026-09-27

Parent: `0d1231eaf7bf346feaab77f56da9875f12963e32`

Status: **PREREGISTERED BEFORE ROUND-43 SCIENTIFIC SCORING.**

All scientific numbers are **given NEMO's entry**.  The six sea-ice
selectors and the card's `unmeasured_features` tuple remain frozen.

Round 42 returned the full walk to kt=1 stage-1 temperature.  The admitted
ORCA1ICE record already contains the stage-1 metric transports, the tracer
accumulator after centered advection and after surface/runoff sources, the
external-mode final sea surface and depth-integrated transports, and the
stage-1 output.  This round uses those existing fields to distinguish the
external-mode handoff from the downstream tracer statements.  It reuses the
production step and its existing WRITE-only hooks; no second tracer or
barotropic implementation is permitted.

## Executing compiled order

The executing HYB branch forms the barotropic correction and the metric
stage transports at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:265-284`.
The stage-1 FCT selector executes the CEN2 precursor whose horizontal and
vertical accumulator statements are at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/traadv_cen.f90:143-160,202-227`.
The following surface and runoff source statements are at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/trasbc.f90:275-328`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R43-P1 | The landed Round-42 ladder reproduces before any arm. | kt=1 stage-1 T remains 233,341 / 399,600 unequal with maximum `0.0014770192519700243 K`; S remains 233,341 / 399,600 unequal with maximum `0.0003277315524314872`. | Any count or maximum moves; stop for instrument drift. |
| R43-P2 | The admitted ORCA1ICE stage-1 tracer record is complete and the existing scorer binds. | Exact schema, finite payload, recorded digests, and a one-ULP active-cell plant fires. | A missing/non-finite field, schema drift, or plant that does not fire; stop for record/instrument repair. |
| R43-P3 | Given NEMO's recorded final sea surface and depth-integrated transports, the current production stage-1 metric transports, centered tracer accumulator, surface/runoff accumulator, and T/S stage output are bit-exact on the record-backed support-safe rank-0 domain. | Every ordered row is 0 unequal. | The first nonzero ordered row replaces this prediction; name it and do not attribute downstream rows across it. |
| R43-P4 | Without the endpoint substitution, the first over-bar tracer-side boundary is the metric transport, before CEN2 and surface/runoff arithmetic. | Production metric transport is non-bit; the endpoint-only arm closes it and all downstream tracer boundaries. | Production metric transport is exact, or the endpoint-only arm leaves it exact but a later boundary non-bit. |
| R43-P5 | No model statement lands unless one source-cited statement alone closes the full stage-1 T/S row and all ORCA2/GYRE gates pass. | One statement reaches 0 unequal and every required gate passes. | The owner is a multi-field handoff, any residual remains, or any gate worsens; hold and walk the first handoff operand next round. |

The score domain is the record-backed rank-0 wet interior whose one-cell
stencil does not require the absent MPI neighbor or northern-fold halo.  The
gate must print CPU, fp64, scalar-libm, exact record digests, shapes, support
census, and every ordered row.  It must refuse a missing field, a non-finite
field, an unexpected resolved selector, and an endpoint arm that changes any
input besides final sea surface and depth-integrated transports.

## Choices

ASKED: return to the first whole-card non-bit statement and walk the
barotropic/transport handoff before changing another operator.

UNASKED: none.  No configuration, carried state, selector, sea-ice field,
stabilizer, score, or scientific threshold changes.
