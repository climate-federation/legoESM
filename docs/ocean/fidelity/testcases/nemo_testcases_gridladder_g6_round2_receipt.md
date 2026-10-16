# Grid ladder rung G6 — round 2 receipt (gridladder_r37)

Status: **LANDED; G6 NOT CERTIFIED.**  The acquisition is passive and admitted,
but G6 is DEBT under both labels.  The first non-bit statement is kt 1 stage 2
`vor_een`, after an all-level bit-identical HPG accumulator.  The landed fold
operands reduce the surface V boundary there from 2.079e-06 to 3.795e-09, but
do not close it.

Preregistration and deck: `PREREG_nemo_testcases_gridladder_g6_round1.md` and
round-1 receipt.  Record:
`gridladder_rounds/round36/oracle_g6_r36/`.  Round-2 evidence:
`gridladder_rounds/round37/final/`.  The final JSON SHA-256 prefixes are:
independent `503d86b1389249c8`, given-entry `b2c44fdba726d579`, stages
`14e6f663d4cfad56`, operators
`e468ef75d3865f8b7c57b5a4b3845e1d9a557af232bbaeb246a7b0a87f5d6f32`, without HPG fold geometry
`3a0a28453833eb5e`, and without raw V mask `8e2ddf586b83601c`.

## 1. Passivity first, then admission

At the last common step, the writer-free reference and twin A restart are
byte-identical in all 38 variables:
`{"ORCA2G1_00000006_restart.nc": 38}`.  Only after that comparison did the
existing gate report `PASS_G5_ADMISSION unequal_rows=None`.  Its manifest
(`admission.json`, SHA-256 `b148c65ec08f29b3`) also records restart/frame
consistency for steps 2, 3, 4, 5 and 6, stage-3 = next-entry for kt 1..5, and
36 twin-identical files.  Acquisition verdict:
`G6_ACQUISITION_PASS steps=1..6 restarts=2..6 reference_completed=7`.

The writer-free reference completed kt 7 and stopped while starting kt 8.
`ref_100step/ocean.output:2206,2209` says why: `stp_ctl` found `|U| = 14.84
m/s > 10 m/s` at `(i,j,k)=(159,88,14)` on rank 0.  With `nn_fsbc=2`, the
admitted/certifiable record is therefore kt 1..6; NEMO was not rerun.

## 2. First non-bit statement and the controlled pair

The source-ordered kt-1 stage-2 accumulator is:

| boundary | surface U max/n | surface V max/n | all-level U full/partial | all-level V full/partial |
|---|---:|---:|---:|---:|
| after HPG | 0/0 | 0/0 | 0/0; 0/0 | 0/0; 0/0 |
| after VOR | 6.676e-10/68 | 3.795e-09/136 | 7.358e-10/1232; 7.466e-10/60 | 3.795e-09/2513; 2.147e-09/117 |
| after ZAD | 9.961e-10/84 | 3.578e-09/197 | 1.335e-09/1535; 1.160e-09/72 | 9.696e-09/3092; 8.351e-09/140 |

All dry counts are zero.  This makes `vor_een`, not the external mode, HPG,
ZAD, or the velocity update, the first remaining non-bit statement.

The HPG pair is non-vacuous and both halves are required:

| option state at `after_hpg` | surface U | surface V | all-level V full | all-level V partial |
|---|---:|---:|---:|---:|
| both selected | 0/0 | 0/0 | 0/0 | 0/0 |
| fold geometry off, raw V mask on | 0/0 | 4.456e-08/68 | 4.456e-08/1257 | 3.392e-08/62 |
| fold geometry on, raw V mask off | 0/0 | 3.408e-07/68 | 2.813e-06/1257 | 3.163e-06/62 |

The AL81 top-V fold halo is also material: after the HPG pair but before that
halo, the post-hoc registered operator arm gave 2.079e-06/136 at surface V;
the landed arm gives 3.795e-09/136.  The focused fold-halo unit plants the
unfolded north operand and fails without the landing.  The remaining 136
surface V cells mean this is a reduction, not a closure.

NEMO's executed statements are explicit.  `hpg_sco` opens at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynhpg.f90:340`; its surface V geometry and
update are at :391-402 and its interior V recurrence/update at :417-427.
`vor_een` opens at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:692`; G6 executes np_CRV at
:762-770, transports at :783-786, and the AL81 triads and U/V update at
:789-803.  The vector velocity update is the live branch at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-470` (enclosing
`stp_RK3_stg` at :78).

The round-1 external-mode prediction is **refuted as owner of this first
boundary**.  In NEMO the vector external update uses raw `ssvmask` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:666-678`; `hv_0` occurs in the
flux-form `ELSE` at :681-701, which does not execute on G6.  Thus raw
`ssvmask` is confirmed as an operand owner here, while stored V depth `hv_0`
is not.  The stage-1 record is exact at every field and depth, independently
excluding the external/barotropic entry as the first boundary.

## 3. Both labels, every recorded surface field

Each entry is `max_abs / bitwise-unequal cells`; bar = 1e-15.  `d` is a
direct NEMO restart entry and `x` is the registered two-step result.  kt 1 is
the common initial entry.  The JSONs retain the additional overlapping
two-step controls as well as every per-field regional row.

### INDEPENDENT

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/0 | 0/0 | 0/0 | 7.089e-06/135 | 7.961e-06/203 | 0/0 | 0/0 |
| 2 | 4.461e-03/16433 | 4.032e-03/15853 | 3.326e-03/15943 | 8.087e-03/15853 | 4.912e-03/15943 | 1.044e-03/16433 | 3.117e-03/16433 |
| 3 | 9.152e-03/16433 | 1.080e-02/15853 | 5.253e-03/15943 | 1.311e-02/15853 | 6.716e-03/15943 | 4.009e-03/16433 | 7.847e-03/16433 |
| 4 | 1.855e-02/16433 | 1.229e-02/15853 | 9.786e-03/15943 | 1.682e-02/15853 | 9.214e-03/15943 | 7.130e-03/16433 | 1.096e-02/16433 |
| 5 | 2.883e-02/16433 | 1.427e-02/15853 | 7.508e-03/15943 | 1.150e-02/15853 | 1.078e-02/15943 | 9.387e-03/16433 | 1.736e-02/16433 |
| 6 | 5.074e-02/16433 | 2.714e-02/15853 | 2.234e-02/15943 | 2.860e-02/15853 | 2.236e-02/15943 | 1.172e-02/16433 | 1.771e-02/16433 |

Regional envelope below is the largest absolute error in each region as
`max field / unequal cells for that field`; the artifact contains the full
7-field matrix.  Coast/open are the two interior splits.

| kt | fold | seam | south | coast | open |
|---|---:|---:|---:|---:|---:|
| 1 | 7.961e-06 v/200 | 0 ssh/0 | 0 ssh/0 | 1.388e-17 v/2 | 1.735e-18 v/1 |
| 2 | 1.537e-04 v/200 | 1.250e-03 u/270 | 6.217e-05 v/46 | 4.461e-03 ssh/2129 | 8.087e-03 u/13772 |
| 3 | 1.165e-04 u/198 | 3.140e-03 u/270 | 1.055e-03 v/46 | 1.082e-02 u/1571 | 1.311e-02 u/13772 |
| 4 | 1.870e-04 v/200 | 4.436e-03 u/270 | 5.114e-04 ssh/46 | 1.855e-02 ssh/2129 | 1.120e-02 u/13772 |
| 5 | 2.689e-04 ssh/205 | 3.352e-03 v/274 | 1.340e-03 ssh/46 | 2.883e-02 ssh/2129 | 1.489e-02 ssh/13772 |
| 6 | 3.330e-04 ssh/205 | 4.757e-03 ssh/281 | 9.619e-04 v/46 | 5.074e-02 ssh/2129 | 9.524e-03 u/13772 |

### GIVEN-NEMO-ENTRY

| path | ssh | uu_b | vv_b | u | v | T | S |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1d | 0/0 | 0/0 | 0/0 | 7.089e-06/135 | 7.961e-06/203 | 0/0 | 0/0 |
| 2x | 4.461e-03/16433 | 4.032e-03/15853 | 3.326e-03/15943 | 8.087e-03/15853 | 4.912e-03/15943 | 1.044e-03/16433 | 3.117e-03/16433 |
| 3d | 9.078e-03/16433 | 8.415e-03/15853 | 5.584e-03/15943 | 1.273e-02/15853 | 6.931e-03/15943 | 2.609e-03/16433 | 7.744e-03/16433 |
| 4d | 1.703e-02/16433 | 1.043e-02/15853 | 7.449e-03/15943 | 1.806e-02/15853 | 7.102e-03/15943 | 3.229e-03/16433 | 4.528e-03/16433 |
| 5d | 2.671e-02/16433 | 1.580e-02/15853 | 7.635e-03/15943 | 1.642e-02/15853 | 8.693e-03/15943 | 4.980e-03/16433 | 6.989e-03/16433 |
| 6d | 8.026e-02/16433 | 3.793e-02/15853 | 1.828e-02/15943 | 4.433e-02/15853 | 2.087e-02/15943 | 1.478e-02/16433 | 2.076e-02/16433 |

| path | fold | seam | south | coast | open |
|---|---:|---:|---:|---:|---:|
| 1d | 7.961e-06 v/200 | 0 ssh/0 | 0 ssh/0 | 1.388e-17 v/2 | 1.735e-18 v/1 |
| 2x | 1.537e-04 v/200 | 1.250e-03 u/270 | 6.217e-05 v/46 | 4.461e-03 ssh/2129 | 8.087e-03 u/13772 |
| 3d | 7.123e-05 v/200 | 2.866e-03 u/270 | 1.074e-03 v/46 | 9.078e-03 ssh/2129 | 1.273e-02 u/13772 |
| 4d | 1.144e-04 v/200 | 3.140e-03 v/274 | 2.573e-04 u/42 | 1.806e-02 u/1571 | 8.692e-03 u/13772 |
| 5d | 1.155e-04 v/200 | 3.322e-03 ssh/281 | 5.084e-04 v/46 | 2.671e-02 ssh/2129 | 8.610e-03 ssh/13772 |
| 6d | 2.061e-04 u/198 | 2.773e-03 ssh/281 | 1.055e-03 v/46 | 8.026e-02 ssh/2129 | 8.048e-03 vv_b/13772 |

## 4. Depth classes

Each cell is `relative max / unequal cells`; every dry unequal count is zero.
There is deliberately no kt-1 restart.  The admitted stage-3 exposure supplies
that endpoint: U full 4.72e-06/2716, partial 5.51e-06/129; V full
5.89e-06/3888, partial 4.95e-06/187; T and S are 0/0 in both classes.

| label/path | field | kt2 full/partial | kt3 full/partial | kt4 full/partial | kt5 full/partial | kt6 full/partial |
|---|---|---:|---:|---:|---:|---:|
| independent | u | 1.728e-03/397940; 1.114e-03/15090 | 2.590e-03/397940; 1.007e-02/15090 | 5.998e-03/397940; 1.029e-01/15090 | 4.472e-02/397940; 1.657e-01/15090 | 5.318e-02/397940; 2.512e-01/15090 |
| independent | v | 2.425e-03/399994; 1.645e-03/15181 | 2.264e-03/399994; 1.613e-03/15181 | 2.020e-03/399994; 2.631e-03/15181 | 3.704e-03/399994; 3.566e-03/15181 | 4.806e-03/399994; 2.457e-03/15181 |
| independent | T | 3.470e-05/414998; 3.127e-05/15554 | 1.333e-04/414998; 5.925e-05/15554 | 2.375e-04/414998; 9.020e-05/15554 | 3.127e-04/414998; 2.408e-04/15554 | 3.955e-04/414998; 3.929e-04/15554 |
| independent | S | 7.782e-05/414998; 6.148e-05/15554 | 1.958e-04/414998; 1.962e-04/15554 | 2.732e-04/414998; 2.707e-04/15554 | 4.298e-04/414998; 4.302e-04/15554 | 4.424e-04/414998; 4.369e-04/15554 |
| given (2x, 3d..6d) | u | 1.728e-03/397940; 1.114e-03/15090 | 2.516e-03/397940; 1.015e-02/15090 | 4.502e-03/397940; 9.670e-02/15090 | 3.030e-02/397940; 1.480e-01/15090 | 4.741e-02/397940; 2.489e-01/15090 |
| given (2x, 3d..6d) | v | 2.425e-03/399994; 1.645e-03/15181 | 2.345e-03/399994; 1.888e-03/15181 | 1.724e-03/399994; 1.607e-03/15181 | 1.620e-03/399994; 1.384e-03/15181 | 3.633e-03/399994; 1.609e-03/15181 |
| given (2x, 3d..6d) | T | 3.470e-05/414998; 3.127e-05/15554 | 8.676e-05/414998; 5.848e-05/15554 | 1.076e-04/414998; 6.606e-05/15554 | 1.661e-04/414998; 1.117e-04/15554 | 4.919e-04/414998; 1.600e-04/15554 |
| given (2x, 3d..6d) | S | 7.782e-05/414998; 6.148e-05/15554 | 1.932e-04/414998; 1.936e-04/15554 | 1.130e-04/414998; 1.134e-04/15554 | 1.743e-04/414998; 1.750e-04/15554 | 5.178e-04/414998; 5.199e-04/15554 |

Every field is therefore DEBT under both labels by kt 2, and G6 is not
certified.  Restart resets materially change kt 3..6 amplitudes but do not
change the verdict.

## 5. Landed transcription and scope

Two G6-only, default-off selectors transcribe the measured HPG geometry/raw
V-mask pair.  The north AL81 V update receives NEMO's T-pivot F triads and
signed U-transport fold halo only when G6's `rk3_vector_raw_vmask` selector is
also on; G1, G1b, G2, G4 and G5 retain the compact-mask path.  The code follows NEMO's stored/native column
layout before restoring the cyclic column; the first trial folded a cyclic
slice and was two columns wrong, so it was rejected.  G1b and all non-G6
cards retain their prior configuration.  `ORCA2G6_UNMEASURED` no longer says
the admitted record is unadmitted; it retains
`nemo_post_si3_surface_records` and `vector_form_multilevel_step`.

The ladder gained source-ordered operator exposures, the two before-arms, full
depth scoring at every given entry, and a final-step restart fallback for T/S.
That last fallback is necessary because kt 6 has no kt-7 entry frame; it does
not manufacture a record.

## Tests and review

- Focused G6 card, AL81 fold-halo, HPG-fold and raw-V-consumer tests passed
  on CPU fp64.  The traced-JIT consumer test asserts exact binary64 words at
  both the completed RHS and raw stage update; disabling the RHS consumer
  fails its first word assertion, and independently disabling the stage-mask
  consumer returns negative zero and fails its stage-word assertion.
- Scope certification on the final tree: G1's 100-step test passed; G2's
  100-step all-field/depth/history test passed; the G4 and G5 focused files
  passed 42 tests with their two slow acquisition tests deselected.  A direct
  G1b 100-step INDEPENDENT rescore reproduced its recorded label exactly:
  598 rows, 575 AT-BAR and 23 DEBT, first over-bar at kt 81 SSH, with 170
  bit-exact rows, 178 exact-modulo-D127 rows, 3,216,400 normal-difference
  cells and 222 flushed-subnormal cells unchanged.
- Required receipt-citation + private-import files: 23 passed (CPU fp64),
  after mechanically re-anchoring 64 gate pins and 20 audited-receipt pins.
- independent review: operator (Claude)

ASKED: passivity before admission; both labels kt 1..6; every surface field;
fold/seam/south/coast/open and full/partial/dry splits; the first non-bit
statement; NEMO citations; the `hv_0`/raw-`ssvmask` ownership test; G6-card
selected transcription; synchronous focused tests and required gates.

UNASKED, offered for revert: the reusable `operators` arm and its two G6
before-flags; recording direct-entry all-level rows; the final-step tracer
restart fallback.  All are instrument-only except the two G6 selectors and
the AL81 fold-halo transcription.

UNVERIFIED: the remaining `vor_een` 3.795e-09 surface-V owner.  It is the next
registered G6 debt; no attribution beyond the measured boundary is claimed.
