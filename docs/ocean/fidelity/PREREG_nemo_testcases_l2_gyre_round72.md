# Preregistration: NEMO-testcases L2 GYRE round 72 stage-1/2 tracer walk

Date: 2026-09-12. Frozen before parsing any round-71 payload values, running
the round-72 live-model instrument, or editing a production numerical
statement.

## Ranked boundary and immutable records

Round 71 refuted the stage-3 Kmm tracer as the magnitude owner: replacing both
Kmm tracers moved the LDF-only kt3 T maximum from
`8.916073106490785e-7 K` only to `8.916073035436511e-7 K`, and removed only
3 of 1,375 retained T cells and 14 of 1,891 retained S cells. Kmm itself
remains non-bit, so its producing stage-2 update is the first unmeasured
upstream statement and transport/metric inputs later in the stage-3 FCT call
remain out of order.

The new immutable record is
`round71/oracle_fct_stage2/oracle_rktracer_operands_kt00000002_s{1,2}.bin`,
produced by `5f9df918ca4d6d9536d920ccd82888df329ec306`. Admission must reproduce
the record gate's exact headers and stage bridges, plus the twin admission's
45 of 65 inherited files exact, 20 changed, and 132 admitted consumed values.
The recorded restart and mesh must remain bit-identical to round 64. The
round-71 clean gate and all three record plants are immutable evidence, not
recomputed outputs to edit.

The before trajectory arm remains Decision 36's recorded trajectory under
`decision36_nemo_face_shear/`. No scratch toggle can become the before arm.

## Compiled statements and ordered walk

The compiled GYRE stage program sets `rDt=rn_Dt/3` at stage 1 and
`rDt=rn_Dt/2` at stage 2 in
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:140-148` and
`:197-202`. The resolved run prints `rn_Dt=14400 s`; instantiate and verify
that value before using 4800 s and 7200 s in a replay.

For both stages, the active program clears `Krhs`, calls `tra_adv`, then
`tra_sbc_RK3` at compiled `stprk3_stg.f90:824-872`. It updates every active
T/S cell in loop order as
`((1+r3t(Kbb))*ts(Kbb) + rDt*(1+r3t(Kmm))*ts(Krhs)*tmask) /
(1+r3t(Kaa))` at `:891-903`. The writer records zero RHS, `zFu/zFv/zFw`,
post-advection RHS, post-SBC RHS, Kbb/Kmm/Kaa tracers, and the three r3t
levels at `:843-853`, `:864-872`, and `:907-913`.

legoESM's one shared WS helper uses the corresponding QCO recurrence inside
`_stage`; its stages 1 and 2 use the centred-advection RHS and the same
stage-specific QCO weights. The existing diagnostic hooks expose stage-1/2
tracers, stage-1 post-advection/post-SBC RHS, and either stage transport. Use
those hooks and extend the round-71 record reader to expose its already-stored
r3t arrays. Do not add a second tracer update implementation to production.

## Frozen measurements and falsifiers

First prove the record can rebuild itself. For each stage and tracer, replay
the compiled update with the recorded Kbb, Kmm, post-SBC Krhs, r3t triplet,
the instantiated rDt, and the owned wet mask. Prediction: all four replays are
bit-exact to recorded Kaa over all 17,600 wet cells. Any unequal cell invalidates
the arithmetic instrument; no live-model attribution may follow.

Then run independently constructed, JIT-native legoESM steps from the same
oracle-seeded kt2 state and expose, in source order:

1. stage-1 zero RHS, Kbb, Kmm, zFu, zFv, zFw, post-advection RHS,
   post-SBC RHS, QCO weights, and Kaa;
2. stage-2 zero RHS, Kbb, Kmm, zFu, zFv, zFw, post-advection RHS,
   post-SBC RHS, QCO weights, and Kaa.

Every exposed array and geometry operand must print fp64. Repeated ordinary
steps and every independently exposed arm must leave all non-exposed outputs
bit-identical. The live stage bridge must be bit-exact, and live stage-2 Kaa
must reproduce round 71's recorded stage-3 Kmm census exactly: 17,994 unequal
T cells with maximum `8.369461070856232e-7 K`, and 16,769 unequal S cells with
maximum `6.794565621248694e-8`.

Prediction: all owned stage-1 rows through Kaa are bit-exact. The first
non-bit live statement will be the stage-2 transport triplet, with `zFu` or
`zFv` the first unequal array in compiled argument order; stage-2
post-advection RHS and Kaa will then be non-bit for both tracers. A valid run
with an earlier unequal row or an exact stage-2 transport records this
prediction as REFUTED and stops at the actual first non-bit statement.

If stage-2 transport is first non-bit, run one causal arm replacing exactly
the stage-2 `zFu/zFv/zFw` passed to the shared tracer helper by the admitted
record, leaving momentum, stages 1 and 3, sources, geometry, state, and public
configuration unchanged. Prediction: the arm reduces the live stage-2 Kaa
maximum error by at least fourfold for T and S and removes at least half of
each unequal-cell set. It need not become exact because the transport producer's
own inputs may remain unequal. Failure of either magnitude condition REFUTES
transport as the Kmm magnitude owner and leaves production unchanged.

Plant 1 moves one wet recorded stage-2 Kaa-T value by one `nextafter` ULP; the
oracle self-replay exact predicate must fail and the command must exit nonzero.
Plant 2 replaces the admitted stage-2 transport target with the live triplet;
the active-arm movement predicate must fail and the command must exit nonzero.
Both plant outputs remain evidence.

## Landing and Rule 12

An oracle-input substitution cannot land. A source edit is eligible only if
the walk names an independently computable shared legoESM statement and proves
that statement bit-exact on NEMO inputs. If the first non-bit row is an input
produced earlier than the tracer update and available records cannot walk its
producer, stop for a new record rather than infer it.

Any eligible edit must first freeze the complete GYRE kt1--10 cellwise
prediction against Decision 36: every moved row registered, no AT-BAR row
leaves the bar, and first-over-bar does not move earlier. Then run the canonical
GYRE ladder and the fixed member-0 days 1--30 harness and score every day. A
local failure or an upstream stopping boundary leaves those trajectories
UNREACHED.

LOCK_EXCHANGE and OVERFLOW share the WS stage/transport code; an eligible edit
there requires exact kt1--10 before/after artifacts for both tanks. DINO uses
the shared tracer/FCT internals through a different time program, so execution
must be shown and its known per-row cancellation risk gated. ORCA2 remains
UNMEASURED WITH SPEC: resolve its compiled card; record stage-1/2 Kbb, Kmm,
Krhs, Kaa, transports, r3t, and stage-3 complete FCT operands for kt1--10;
require exact statement replay, all moved rows registered, no AT-BAR loss, and
no earlier first-over-bar row.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest may change. Failed predictions remain in the receipt as REFUTED.
