# ORCA2 round 74 preregistration — independent metric-U transport walk

Date frozen: 2026-09-29

Base: `083cefe542b2794f313470e93e44f26471e9eefc`

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. No NEMO entry field, external-mode
endpoint, stage transport, or carried state is substituted into the production
arm.

Sea ice remains out of scope. The six-entry `unmeasured_features` tuple,
selectors, 10,800 s step, thresholds, and carried state are frozen.

## Existing evidence, not predictions

Round 73 found the first non-bit stage-1 tracer operand at metric `pU`:
221,640 / 251,670 active faces, maximum absolute difference
`321212.60790659266 m3 s-1`. The downstream CEN2 U-face flux is the first
non-bit tracer arithmetic statement. The admitted rank-0 transport record
contains NEMO's `e2u`, `e3u(Kmm)`, `uu(Kmm)`, `zub`, `umask`, `zFu`,
`un_adv`, `r1_hu(Kmm)`, and `uu_b(Kmm)` operands.

The search-before-build audit found and will reuse the existing phase-2
stage-1 transport record reader and scorer, the round-13 production exposure
for thickness/corrected velocity/transport average, the round-43 independent
step runner, and the production `_nemo_stage_corrected_velocity`,
`_nemo_ws_qco_stage_faces`, and `_nemo_metric_stage_transport` helpers. No
second transport implementation or new model hook is authorized.

## Compiled path and order

The active compiled ORCA2 statement first constructs

`zub = un_adv * (r1_hu_0 / (1 + r3u(Kmm))) - uu_b(Kmm)`

and then

`zFu = e2u * e3u(Kmm) * (uu(Kmm) + zub * umask)`

at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:265-280`.
The gate walks U before V and scores these source-ordered boundaries:

1. `un_adv`, live inverse depth, and `uu_b(Kmm)`;
2. `zub`;
3. `uu(Kmm)` and `umask`;
4. corrected velocity `uu + zub*umask`;
5. `e2u`, `e3u(Kmm)`, and their associated product;
6. final metric transport `zFu`.

All rows use raw-bit fp64 comparison on the same rank-0 active U support. A
separate recorded-operand replay must reproduce `zub`, corrected velocity,
the metric-thickness product, and `zFu` exactly before production attribution
is admitted.

## Frozen predictions and falsifiers

1. **Card execution census.** ORCA2, GYRE, OVERFLOW, and LOCK all predict the
   shared RK3-WS stage-transport construction. A different resolved time
   integrator or an unexecuted statement is **REFUTED** and refuses scope.
2. **Record calibration.** Replaying the two compiled U statements from the
   admitted NEMO operands predicts 0 unequal active cells at every derived
   boundary. Any mismatch invalidates the instrument.
3. **First primitive input.** Independent production `un_adv` predicts the
   first non-bit primitive in source order. If exact, this is **REFUTED** and
   the gate names the first observed non-bit row without relabeling.
4. **Entry velocity.** Independent `uu(Kmm)` predicts raw-bit equality to the
   NEMO record. Any mismatch is **REFUTED** and retained.
5. **Thickness consequence.** Independent entry SSH predicts non-bit
   `e3u(Kmm)`. Exact thickness is **REFUTED** and retained.
6. **Downstream reproduction.** The final production `zFu` tuple predicts
   round 73 exactly: 221,640 / 251,670 unequal and maximum
   `321212.60790659266 m3 s-1`. Any changed tuple refuses the walk.
7. **Disposition.** This diagnostic round predicts **HELD**. No model,
   configuration, selector, threshold, stabiliser, carried-state, or sea-ice
   change is authorized.

Failed predictions remain **REFUTED**. Primitive-input, derived-boundary,
calibration, and downstream-reproduction predicates are separate.

## Controls and validation

- A one-ULP mutation of an exact recorded replay boundary must fire.
- Replacing production `un_adv` with the exact recorded value must change the
  first-non-bit classification and fire the frozen prediction gate.
- A false card execution selector and a changed round-73 `zFu` tuple must fire.
- Run focused tests, the shared-card and tank batteries, both citation gates
  with a real firing citation plant, `tests/ocean/fidelity -n 12` once, and
  the required read-only Codex review.

ASKED: walk independent ORCA2's stage-1 metric-U transport construction
upstream of the first non-bit CEN2 flux and name the first source-ordered
non-bit operand or arithmetic statement.

UNASKED: model arithmetic, configuration, new NEMO output, selectors, sea ice,
and the held shared tracer QCO/RK candidate.

## Addendum after the first fail-closed run

The first committed run reached classification and refused the frozen round-73
reproduction row before emitting a report: scoring on the record's own
`umask` produced 226,236 cells, not round 73's 251,670-cell card support. The
support assumption is therefore **REFUTED**. A direct census gives 30,030
card-active/NEMO-dry cells and 4,596 NEMO-active/card-dry cells.

The rerun preserves round 73's card support for every scalar/transport row so
its frozen `zFu` tuple is comparable, reports the four-way support census, and
scores `umask` itself over the full owned rank-0 array. The existing
recorded-operand replay remains separately labelled on the record's own mask.
No prediction, model path, or production operand changed.
