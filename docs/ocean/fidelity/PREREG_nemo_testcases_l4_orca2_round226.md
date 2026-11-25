# Preregistration: ORCA2 round 226 — OMT-4 final FCT RHS walk

Date: 2026-10-10. Frozen base: `35b5f6bff`. Scope: continue round 225's
source-ordered OMT-4 stage-3 FCT walk after the limiter, through the final
limited-flux divergence, the two additions to tracer `Krhs`, and the stage-3
consumer. This file is committed before any round-226 replay or trajectory.

No configuration, carried-state, stabiliser, threshold, sea-ice selector, or
`unmeasured_features` change is authorised. OMT-5 stays blocked. All replays
are CPU, production JIT, fp64/libm, and use admitted passive completed states;
no in-executable observer is permitted.

## Existing implementation reused

Repository search found the complete source-associated implementation in the
retained GYRE round-112 patch
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round112_fct_metric_upstream_held.patch`.
Its downstream portion preserves NEMO's already divided concentration RHS and
stage weight. Round 112 proved that portion bit-exact given NEMO's recorded
inputs but reverted the whole, larger candidate after its GYRE month veto.
Round 226 does not revive the upstream portion or reuse its trajectory claim;
it isolates and remeasures only the downstream statements on the current tree.

## Compiled source order

The active two-step upstream routine first adds the averaged-upstream flux
divergence to `Krhs` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:598-609`.
After `nonosc`, it differences the limited antidiffusive faces, divides by the
live `Kmm` T-cell thickness, masks, and adds that rate to the same `Krhs` at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:318-330`.
The compiled stage program zeroes `Krhs`, calls tracer advection, and passes
the completed rate to the stage-3 implicit tracer solve at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:600-649,700-760`.

## Frozen predictions and falsifiers

1. **R226-P1 — final RHS is the next non-bit boundary.** CONFIRM: on the
   admitted OMT-4 kt=1 stage-3 state, the source-associated upstream-RHS plus
   limited-flux-RHS add differs bitwise from the current combined content
   divergence, while every limiter coefficient named in round 225 is held
   fixed. The existing rank-0 `NEMO_L2_RKTR3_1` record must show the literal
   RHS at least as close to NEMO's `after_advection` field as the current form.
   REFUTE: the two forms are bit-identical, or the source-associated form is
   farther from NEMO on the record's owned wet cells.
2. **R226-P2 — stage association is live.** CONFIRM: carrying the already
   divided concentration RHS through NEMO's stage weight changes at least one
   active OMT-4 tracer cell; a planted conversion back to generic content
   divergence must restore the baseline bits. REFUTE: the candidate is inert
   or the plant does not restore the baseline.
3. **R226-P3 — statement sufficiency.** CONFIRM: the private complete atomic
   unit (rounds 215--225 plus this downstream statement) removes or delays the
   independent OMT-4 kt=8 live-W-thickness refusal. REFUTE: the boundary and
   refusal log remain byte-identical. A refuted candidate is reverted and no
   Decision-96 census is manufactured.
4. **R226-P4 — landing predicate.** If P3 confirms, land only when the full
   Decision-96 census passes on OMT-4, rung 0, rung 10, GYRE, DINO, and tanks,
   with every touched row registered. Any exact-row loss or gate red holds the
   whole unit.
5. **R226-P5 — controls.** Plants for record ownership, source association,
   stage-consumer liveness, and sufficiency must each refuse.

## Stop conditions

If the admitted rank-0 record cannot be aligned to its owned domain, report
that limitation and continue only with the source-associated internal replay;
do not infer missing rank-1 oracle values. If P3 refutes, status is `HELD`, the
candidate is retracted, and the OPEN boundary becomes the stage-3 implicit
tracer solve. If P3 confirms but any landing gate fails, retain the candidate
only as a committed held patch.
