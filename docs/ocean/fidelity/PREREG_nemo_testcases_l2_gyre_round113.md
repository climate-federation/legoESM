# Preregistration — NEMO testcase L2 GYRE round 113

Date: 2026-09-18

This document is frozen before either Round-113 scientific measurement.  The
inherited Round-112 citation-map repair and its four-file admission gate were
run first because the operator explicitly made that repair a precondition for
any Round-113 work; those are bookkeeping controls, not measurements of a
candidate or a model trajectory.

## P1 — close the Round-110 generic-card blast radius

The statement already landed in Round 110 routes the same-stage GM/Redi
increment into the stage-3 WS tracer source.  The executing-card census will
be derived from the resolved recipes, never from a handwritten list.  In
particular, `build_nemo_gyre_recipe()` is predicted to resolve
`tracer_time_integrator == "rk3_ws"` with non-null `gm_redi` and therefore to
execute the route.

The existing generic NEMO-GYRE certification trajectory (the deterministic
three-step forced trajectory used by `test_nemo_recipe.py`) will be run at the
landing's own base commit `51a4d088c` and at the current descendant.  T, S, u,
v, and eta will be registered after every step by exact unequal-cell count and
maximum absolute move.  Prediction: T and S move at step 1; momentum and eta
do not move before the changed tracers can feed a later step; every existing
generic-recipe certification assertion keeps the same pass/fail disposition.
Falsifier: the resolved card does not execute the route, no tracer moves, any
certification assertion changes from pass to fail, or a moved field is absent
from the registry.  A newly failing certified assertion requires
`DECISION_NEEDED`; it will not be silently re-baselined.

The Round-110 receipt will also record the operator's already measured
same-tip before arm under `round110/before_same_tip_51a4d088c`, its exact
identity to the older arm, and the registered day-23 T/S/u/v co-spike with ssh
remaining on trend.

## P2 — make the Decision-43 controls non-vacuous

`all_moved_rows_registered` will compare the complete set of nonzero
comparison rows with an explicit registry, including missing and unexpected
names; it will no longer mean merely that at least one row moved.  A plant
which removes one real registry entry is predicted to exit nonzero.

The real-card control will execute the production step with the resolved
generic NEMO-GYRE recipe and observe the stage-3 source route.  The DINO
fail-closed control will use the same executable-path witness rather than
accepting configuration flags alone.  Removing the production route guard is
predicted to make the execution witness test fail.  Falsifier: either mutant
still passes.

## P3 — live-input FCT producer walk

The admitted passive Round-111 record
`round111/oracle_fct_writers/oracle_fct_writers_kt00000002_s3.bin` remains the
oracle.  The existing Round-67/112 production instrument will be extended; no
second FCT harness will be written.  At kt=2 stage 3 it will score the live
production inputs against the record in compiled consumer order:

1. Kbb base T/S;
2. U, V, and W transports;
3. Kbb and Kmm tracer thicknesses;
4. tmask/wmask;
5. reciprocal cell area; and
6. p2dt.

Each row reports exact unequal cells and maximum absolute difference under the
production-jitted step.  Production eager is reported separately and never
substituted for the JIT result.  The Kbb base and static mask/metric/p2dt rows
are predicted BIT because the model is driven from NEMO's admitted step entry
and the geometries were previously machine-identical.  The first non-BIT live
input is predicted to be the U-transport family; V/W may also be non-BIT.
Falsifier: any earlier row is non-BIT, or U transport is BIT.

Beginning with the first non-BIT family, one family at a time will be replaced
by the recorded NEMO values while the complete production-jitted step runs.
The primary output is the directly recorded `adv_up1_T/S` boundary; kt3 T/S
and the complete state move are secondary.  Prediction: the transport-family
arm reduces the T `adv_up1` maximum error by at least 2x but does not make both
tracers BIT; therefore this round names the next upstream owner and does not
land physics.  A material kt3 improvement means at least 10% in T maximum
error.  Falsifiers are (a) no 2x `adv_up1_T` reduction, (b) both tracer rows
become BIT, or (c) kt3 T improves by at least 10%; all are retained as measured
outcomes rather than used to revise the prediction.

A one-ULP perturbation to the first finite, nonzero recorded member of the
first non-BIT family must change its production-JIT input row and a consumed
upstream result; the plant exits nonzero and prints `STATUS PLANT-FIRED`.

No candidate advances to the Decision-43 ladder/month gate unless the live
family substitution closes both directly recorded `adv_up1` tracer rows or
materially improves kt3 T.  If it does, the frozen landing criteria are:
day-30 T rms decreases against a same-base measured arm, first-over-bar is not
earlier, no kt=1 AT-BAR row leaves the bar, every moved row is registered, and
every other executing card is measured.  DINO is measured only if the
recipe-derived execution census says it shares the exact statement.

ORCA2 remains `UNMEASURED`: before any claim, construct the ORCA2 NEMO-identity
recipe, run the same production input/output census on an admitted ORCA2
stage-3 record, and then run its certified trajectory gate.
