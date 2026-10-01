# NEMO testcase fidelity: round 194 / VORTEX round 10 ZAD operand walk

**Status: HELD.** The first non-bit ZAD operand is the stage vertical velocity
`ww`. Selecting NEMO's literal two-solve vector program makes the stage-2 and
stage-3 ZAD accumulator exact to the compiled-rounding floor and improves the
VORTEX-vector kt=2 U/V rows by 270x/316x, but the certified 50-row trajectory
comparison is red: kt=5 S leaves AT-BAR and later debt cells worsen by more
than the unchanged two-ULP ratchet. Production is restored; the candidate is
preserved only in
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l1_vortex_round194_literal_wzv_held.patch`.

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round194/`.
The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_round194.md`, committed as
`c87489f602f5` before measurement.

## Compiled program

The record's vector branch calls continuity on raw stage velocity before
momentum advection, while the flux branch calls it on transports, at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:289-300`.
The velocity and transport forms build horizontal divergence in distinct
compiled arms at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/divhor.f90:123-140`; both feed the
bottom-up W recurrence, including the quasi-Eulerian thickness-change term,
at `VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/sshwzv.f90:271-299`.

`dyn_zad` consumes `ww(jk+1)`, the Kmm velocity difference, and the live Kmm
face thickness, with its separate bottom statement, at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynzad.f90:105-137`. After momentum
has consumed the velocity-form W, the vector tracer path re-solves on
transports at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/traadv.f90:268-273`. Thus one shared
generic W is not NEMO's vector stage program.

## Production-JIT operand walk

`zad_operands.json` is stamped to final clean commit
`4e916e8ecc4d65c45df85bd4227768836095a52c`, CPU fp64/libm, with JIT enabled.
The existing stage-entry harness drives `model.step` from NEMO's recorded
stage entry; the new hooks only observe/substitute the operands passed through
that same production step.

| stage | operand | unequal | max absolute difference |
|---:|---|---:|---:|
| 2 | Kmm U velocity | 0 | 0 |
| 2 | Kmm V velocity | 0 | 0 |
| 2 | W | 37,210 | `1.0128307690552597e-05 m s-1` |
| 2 | U-face thickness | 0 | 0 |
| 2 | V-face thickness | 0 | 0 |
| 3 | Kmm U velocity | 0 | 0 |
| 3 | Kmm V velocity | 0 | 0 |
| 3 | W | 37,210 | `1.0128307690552706e-05 m s-1` |
| 3 | U-face thickness | 0 | 0 |
| 3 | V-face thickness | 0 | 0 |

The W vertical-index control is exact at the bottom interface and unequal at
3,721 surface plus 33,489 interior words. Its maximum residual by interface is
`[1.0128307691e-5, 9.1154769215e-6, 8.1026461524e-6,
7.0898153834e-6, 6.0769846143e-6, 5.0641538453e-6,
4.0513230762e-6, 3.0384923072e-6, 2.0256615381e-6,
1.0128307691e-6, 0] m s-1`. This is not a shifted interface: candidate minus
oracle equals the negative of the compiled scale-factor contribution to
`1.80e-19 m s-1` at both stages. legoESM's generic arm omitted that term.

## One-variable ownership

| stage | arm | ZAD U max | ZAD V max | fraction removed |
|---:|---|---:|---:|---:|
| 2 | production | `1.8741000428408085e-09` | `1.8522221618642312e-09` | — |
| 2 | recorded velocity only | same | same | 0 |
| 2 | recorded W only | `1.3552527156068805e-20` | `2.7105054312137611e-20` | > 0.99999999998 |
| 2 | recorded thickness only | same | same | 0 |
| 2 | all recorded | `1.3552527156068805e-20` | `2.7105054312137611e-20` | > 0.99999999998 |
| 3 | production | `1.8905432975169711e-09` | `1.8685471232664545e-09` | — |
| 3 | recorded velocity only | same | same | 0 |
| 3 | recorded W only | `2.0328790734103208e-20` | `1.3552527156068805e-20` | > 0.99999999998 |
| 3 | recorded thickness only | same | same | 0 |
| 3 | all recorded | `2.0328790734103208e-20` | `1.3552527156068805e-20` | > 0.99999999998 |

The first non-bit producer is therefore the omitted literal WZV program, not
stage velocity, face thickness, or the vertical index convention. The held
candidate states the existing `wzv_call2_evaluation="nemo_literal"` and
`nemo_stage_momentum_wzv_split=True` options on `VORTEX_VEC-zco` only. The
flux card remains unchanged.

## Certified trajectory veto — complete row registry

Before is the same-card round-191 arm; candidate is clean commit
`858822b5f4c1c11d074d59b9d03280620e95d3c4`. Values are normalized maximum
absolute residuals from `vortex_vec_ladder.json`; the cellwise comparison and
every violation are in `vortex_vec_compare.json`.

| kt | T before → candidate | S before → candidate | U before → candidate | V before → candidate | SSH before → candidate |
|---:|---|---|---|---|---|
| 1 | `0` → `0` | `0` → `0` | `2.220446049e-16` → `2.220446049e-16` | `2.220446049e-16` → `2.220446049e-16` | `1.355252716e-20` → `1.355252716e-20` |
| 2 | `3.625489425e-09` → `1.581426548e-13` | `4.060244204e-16` → `4.060244204e-16` | `3.369329786e-06` → `1.246261511e-08` | `3.337023651e-06` → `1.056274559e-08` | `3.709009770e-08` → `3.709009770e-08` |
| 3 | `3.261102708e-08` → `2.655784623e-11` | `6.090366307e-16` → `8.120488409e-16` | `4.010084241e-06` → `2.576717467e-07` | `3.983891562e-06` → `2.092700415e-07` | `6.505721262e-06` → `6.505721262e-06` |
| 4 | `1.140348162e-07` → `1.384688044e-10` | `8.120488409e-16` → `8.120488409e-16` | `2.372738925e-06` → `1.079603365e-06` | `2.329281362e-06` → `9.982875433e-07` | `8.230603660e-06` → `8.231283697e-06` |
| 5 | `1.691849259e-07` → `1.724667923e-10` | `8.120488409e-16` → `1.015061051e-15` | `3.712188374e-06` → `1.070166769e-06` | `3.595140521e-06` → `1.056911427e-06` | `8.026674186e-06` → `8.028662124e-06` |
| 6 | `1.724755056e-07` → `2.044450818e-10` | `1.015061051e-15` → `1.015061051e-15` | `4.743071120e-06` → `1.302391311e-06` | `4.596828562e-06` → `1.223588816e-06` | `8.026496078e-06` → `8.028656643e-06` |
| 7 | `1.582145461e-07` → `2.847479094e-10` | `1.015061051e-15` → `8.120488409e-16` | `5.458853057e-06` → `1.815543503e-06` | `5.314326006e-06` → `1.612529930e-06` | `5.232805557e-06` → `5.227747438e-06` |
| 8 | `1.475225827e-07` → `4.029126797e-10` | `1.015061051e-15` → `1.015061051e-15` | `5.939296321e-06` → `2.146444209e-06` | `5.903662329e-06` → `1.879741671e-06` | `6.015781366e-06` → `6.015185508e-06` |
| 9 | `1.455210462e-07` → `5.040978550e-10` | `1.218073261e-15` → `1.015061051e-15` | `5.592987248e-06` → `2.317393920e-06` | `5.483716120e-06` → `2.058856910e-06` | `4.593345490e-06` → `4.599853164e-06` |
| 10 | `1.495331189e-07` → `5.586929816e-10` | `1.218073261e-15` → `1.218073261e-15` | `4.865475868e-06` → `2.426729208e-06` | `4.724706786e-06` → `2.164076606e-06` | `5.356564719e-06` → `5.360715993e-06` |

The candidate leaves all kt=1 rows unchanged and first-over-bar at kt=2. It
improves the headline kt=2 T/U/V rows enormously, but it does not bring them
to the bar; kt=5 S crosses AT-BAR to DEBT and the mechanical two-ULP cellwise
gate reports `FAIL`. This is a trajectory veto, so no GYRE, DINO month, tank,
generic-card, month, or year landing claim is made. Production at final commit
`4e916e8ecc4d65c45df85bd4227768836095a52c` reproduces the before arm.

## Predictions, plants, review, and tests

* First non-bit operand W: **CONFIRMED**. Velocity and thickness are BIT; W is
  not.
* Recorded W removes at least 50% of each ZAD maximum: **CONFIRMED**, removing
  all but `1.36e-20..2.71e-20`.
* Velocity/thickness substitutions remove less than 10%: **CONFIRMED**, zero.
* Literal WZV candidate passes the landing gate: **REFUTED** by the AT-BAR S
  loss and two-ULP violations; the failed prediction is retained.
* The W one-ULP production plant prints `STATUS PLANT-FIRED` and exits 1.
  The commit-stamp plant refuses the wrong SHA and exits 2. The preregistered
  one-ULP thickness-output prediction is **REFUTED**: even a field-wide
  one-ULP move is rounded away at the accumulated ZAD output and the gate
  refuses it. The direct operand-row unit plant still catches one ULP; this is
  not represented as a production-output plant.
* Separate Codex review was attempted exactly as required but is unavailable
  in-sandbox. Verbatim result: `Error: failed to initialize in-process
  app-server client: Read-only file system (os error 30)`.

Validation summaries appear below after the fail-closed commands complete.

## OPEN — round 195 / VORTEX round 11

Keep the literal-WZV patch HELD. In a scratch arm apply it and continue in
compiled stage order at the first post-ZAD boundary: the stage-2 velocity
update and barotropic correction, then the stage-3 update. Name the first
non-bit producer of the remaining kt=2 U/V `1.2462615e-08 /
1.0562746e-08`, and test the smallest same-stage bundle through the unchanged
two-ULP trajectory gate. Do not start the flux card or Decision-74 resolution
ladder until the vector kt=2 rows are at the bar or proven harness floor.
