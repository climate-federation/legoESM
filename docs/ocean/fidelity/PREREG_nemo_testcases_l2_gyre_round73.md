# Preregistration: NEMO-testcases L2 GYRE round 73 stage-1 transport primitives

Date: 2026-09-12. Frozen before parsing any numeric payload from the acquired
round-72 stage-1 transport record, exposing new live-model operands, or editing
a production numerical statement.

## Ranked boundary and immutable records

Round 72 found stage-1 `zFu` to be the first non-bit tracer-stage statement at
kt=2: all 17,400 wet U-face cells differ, with maximum absolute difference
`0.8916110997497526`; stage-1 `zFv` follows with all 17,100 wet faces unequal
and maximum `0.7485346468365606`. This is earlier than the stage-1 tracer RHS
and stage update, so it owns the source walk regardless of their smaller
absolute errors.

The newly acquired immutable record is
`round72/oracle_stage1_transport/oracle_rkstage1_transport_operands_kt00000002.bin`,
produced from legoESM commit
`7be44bb51402256ca8d77f734867a0c9d651bb9b`. Admission must reproduce the
operator-reported census: 47 of 67 inherited files exact, 20 changed, and 132
admitted consumed values. The final restart and mesh mask must remain
bit-identical to round 64. The acquired record gate must reproduce exact `zub`,
`zvb`, `zFu`, and `zFv` replays before the live comparison is trusted.

The recorded before trajectory remains Decision 36 under
`decision36_nemo_face_shear/`; no scratch toggle may replace it.

## Compiled source and ordered walk

The compiled R72 GYRE program takes the active `np_HYB` arm and evaluates
`zub = un_adv*(r1_hu_0/(1+r3u(Kmm))) - uu_b(Kmm)` before its V analogue at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:287-291`.
It then evaluates
`zFu = (e2u*(e3u_3d*(1+r3u(Kmm)*umask))) *
(uu(Kmm)+zub*umask)` before `zFv` at the same compiled source's lines
`:294-297`. The WRITE-only record captures, in that source context, metric,
live face thickness, Kmm velocity, correction, mask, final transport,
`un_adv`, live inverse depth, and Kmm `uu_b` for U followed by the V siblings
at `:299-321`.

legoESM's shared `_nemo_ws_stage_transport` calls the shared live-QCO face
builder, selects the separately carried stage-1 Kbb barotropic velocity, forms
the correction in `_nemo_stage_corrected_velocity`, and forms the metric-bearing
transport in `_nemo_metric_stage_transport`. Extend only its existing private
WRITE-only operand exposure. Do not add a second transport implementation or a
constructible selector.

## Frozen measurements and falsifiers

First rerun the acquired record gate with its exact commit stamp. Prediction:
its header and EOF checks pass, and the four oracle-only replay rows (`zub`,
`zvb`, `zFu`, `zFv`) are bit-exact. Its existing replay-ULP plant must exit
nonzero.

Then start from the same independently constructed, oracle-seeded kt=2 state as
round 72 and expose the stage-1 U path in compiled order:

1. `un_adv`;
2. `r1_hu_0/(1+r3u(Kmm))`;
3. their product;
4. `uu_b(Kmm)`;
5. `zub`;
6. `e2u`;
7. `e3u(Kmm)`;
8. `uu(Kmm)`;
9. `umask`;
10. `uu + zub*umask`;
11. `e2u*e3u(Kmm)`;
12. `zFu`.

Only after the U stop is known, score the corresponding V rows and retain them
as secondary evidence; do not let a V result supersede the earlier U source
order. Every exposed/record/grid/state array must print float64. Each exposure
must leave every non-exposed ordinary output bit-identical, and repeated
independently constructed ordinary steps must remain exact.

Prediction: `un_adv` is the first non-bit U statement. The inverse depth,
carried Kmm `uu_b`, metric, live thickness, Kmm velocity, and mask will each be
bit-exact. The multiplication `un_adv*inverse_depth`, `zub`, corrected velocity,
and `zFu` will remain non-bit. The V walk will analogously stop first at
`vn_adv`. A valid earlier mismatch, exact `un_adv`, or any failed identity or
dtype control records this prediction as REFUTED and stops at the measured
boundary.

The magnitude prediction is deliberately on the final affected field: the
clean walk must reproduce round 72's stage-1 `zFu` census exactly (17,400
unequal, maximum `0.8916110997497526`) and the `zFv` census exactly (17,100
unequal, maximum `0.7485346468365606`). Any discrepancy invalidates the live
instrument before attribution.

Plant 1 is the acquired gate's one-ULP replay perturbation and must exit
nonzero. Plant 2 replaces the recorded `un_adv` comparison target by the live
value after admission; the first-U-boundary predicate must no longer identify
`un_adv`, so the command must exit nonzero. The planted target must be nonzero
on at least one wet face. Both failures remain evidence.

## Landing and Rule 12

If an input field is the first non-bit statement, it is not an eligible fix:
stop for a record of its producer rather than infer upstream ownership. If all
inputs are exact and a shared arithmetic association is first non-bit, a
production edit is eligible only after oracle-input replay proves that exact
statement bit-for-bit and a planted old association fails.

Before any eligible production edit, freeze the complete GYRE kt1--10
cellwise prediction against Decision 36: register every moved row, forbid an
AT-BAR row from leaving the bar, and forbid an earlier first-over-bar boundary.
Then run the canonical GYRE ladder and the fixed member-0 days 1--30 harness and
score every day. An upstream stop leaves both trajectories UNREACHED.

LOCK_EXCHANGE and OVERFLOW share the WS stage transport and require exact
kt1--10 before/after artifacts for any production edit. DINO executes the
shared stage/transport code through a different program; its 96--98% per-row
cancellation risk requires an execution proof and its own row gate. ORCA2
remains UNMEASURED WITH SPEC: resolve its compiled card; record the stage-1/2
transport primitives, compound corrections, `zF`, tracer RHS and Kaa for
kt1--10; replay every statement in compiled order; register all moved rows,
preserve every AT-BAR row, and forbid an earlier first-over-bar boundary.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. Failed predictions remain in the receipt as REFUTED.
