# ORCA2 round 288 preregistration — resolved-mask / top-V atomic pair

Date: 2026-10-16

Round 286 proved that NEMO's resolved three-dimensional stage mask closes the
independent OMT-4 stage-1 raw V update bit-for-bit, but the same statement alone
exposes a rung-0 live-thickness refusal during kt=4.  Decision 125 separately
holds NEMO's completed-stage V fold exchange because it amplifies rung-0 and
OMT-4 sea surface height when the compact mask remains in use.  These two
source-exact statements share the stage V field and fail in opposite
directions, so this round measures them as one cancelling pair before walking
another leaf.

The compiled ORCA2 record build constructs and associates `vmask` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232`, consumes it in
the vector-form update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`, and applies
the post-stage U/V/T/S association at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:812-820`.

Every number is **independent rung 0** unless explicitly labelled otherwise.
The first measurement is CPU, JIT, fp64/libm, uses the admitted rank-complete
rung-0 stage frames, and compares matched post-association stage boundaries.
No in-executable observer is added; the existing completed-stage hooks and
their passivity checks are reused.

## Frozen predictions and falsifiers

1. **R288-P1 — controls reproduce.**  The production control completes all
   200 rung-0 rows.  The resolved-mask-only arm reproduces round 286: kt=1..3
   complete and kt=4 refuses on non-positive/non-finite live W thickness.
   Either control moving refutes the instrument.
2. **R288-P2 — atomic-pair closure.**  With the resolved mask and completed-
   stage V fold exchange enabled together, the candidate completes kt=1..10.
   A refusal at or before kt=4 refutes the proposed compensating pair.
3. **R288-P3 — first boundary.**  At matched program points, the pair moves
   kt=1 stage-1 V toward NEMO relative to both single arms and loses no exact
   stage row.  If the pair still fails, the first completed stage where its
   error grows by more than 10x over the preceding boundary is the next owner;
   the later live-thickness guard is downstream damage.
4. **R288-P4 — landing predicate.**  Only if P1-P3 pass does the atomic pair
   proceed to the standing full-card gates.  It lands only with no exact-row
   loss, unchanged-or-toward first-over-bar rows, and Decision-96 eligibility
   on rung 0, OMT-1..4 and rung 10 under both labels, plus green GYRE, DINO,
   tank, citation, test, and operator-review gates.  Any red holds both
   statements private.
5. **R288-P5 — non-vacuity.**  Removing either member must recover its
   registered single-arm failure, and a one-bit stage-state plant must be
   rejected.  A control that stays green refutes the measurement.

Decision 126 remains pending and unchanged.  Sea ice and its six selectors
remain exactly as carried by the shipped card.  No stabiliser, floor, clip,
carried-state change, or NEMO acquisition is authorised.

Independent review: operator, as required for this lane.
