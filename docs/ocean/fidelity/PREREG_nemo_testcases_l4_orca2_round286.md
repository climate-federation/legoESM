# ORCA2 round 286 preregistration — resolved WS-RK3 stage masks

Date: 2026-10-15

This round resumes round 283's first non-bit statement.  NEMO constructs and
laterally associates three-dimensional U/V masks in
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dommsk.f90:206-232`, then the live
vector-form WS-RK3 update consumes those arrays at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`.
legoESM currently feeds that update masks reconstructed from compact T-cell
activity.  The candidate routes only the stage U/V mask operands through the
existing `nemo_qco_resolved_mesh_operands` source; it adds no builder, default,
stabiliser, carried state, runoff change, or sea-ice change.

All trajectory measurements are CPU, fp64, and keep the two start labels
separate.  Decision 96 is the landing predicate because this statement is not
yet certified on a geometry-ladder rung.

## Frozen predictions and falsifiers

1. **R286-P1 — operand closure.**  On independent OMT-4 kt=1 stage 1, the
   resolved U/V stage masks are bit-exact against NEMO and the raw V update
   closes from round 283's 1,319 unequal active cells to 0.  Any valued mask or
   raw-update difference refutes the attribution and stops the landing.
2. **R286-P2 — non-vacuity.**  Restoring the compact stage mask recreates all
   1,319 raw-update differences.  A planted one-bit resolved-mask error must be
   rejected.  Either control staying green refutes the instrument.
3. **R286-P3 — Decision-96 census.**  The single statement loses no exact row,
   leaves the first over-bar row unchanged or toward, and has a strict majority
   of RMS-moving rows toward NEMO on each of rung 0, OMT-1..4, and rung 10
   under both applicable labels.  Any card failing one predicate holds the
   candidate.
4. **R286-P4 — shared cards.**  GYRE's certified ten-step ladder and daily
   snapshots are byte-identical; DINO, tanks, and generic cards stay inside
   their existing gates.  Any new red attributable to the statement holds it.
5. **R286-P5 — held fold unit.**  Only after R286-P1..P4 pass, re-score the held
   vector-form fold unit.  If its kt=8 live-W-thickness refusal disappears,
   register the resolved mask as its missing partner; if unchanged, keep the
   fold unit held and name its first matched passive boundary.  This outcome is
   diagnostic and cannot alter R286-P3.

Decision 126 remains pending and unchanged.  Decision 125 keeps the completed
top-V fold exchange off.  Independent review is performed by the operator per
the current lane instruction.
