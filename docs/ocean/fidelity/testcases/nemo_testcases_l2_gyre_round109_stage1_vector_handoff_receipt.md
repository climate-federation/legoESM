# NEMO-testcases L2 GYRE round 109 receipt: stage-1 vector handoff

Date: 2026-09-18

Incoming tip: `5ffef4983dda4ece3faca523def617e4cdf1d23c`

Round status: **HELD — diagnostic instrumentation landed; the numerical
candidate was restored after the 954-row trajectory gate refused it**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round109/`

## Outcome first

The first non-bit model statement after the already-BIT kt1 stage-1 operator
walk is the projection of the full momentum accumulator onto a baroclinic
source.  Given NEMO's recorded entry, the full accumulator is BIT in U and V,
but that projection is unequal in all 17,400 U and 17,100 V wet cells, with
maximum `2.0121494123449567e-8 m s-2`.  The same boundary is first under the
complete production JIT and the complete production closure with JIT disabled.

That projection has no statement in the compiled vector program.  NEMO
accumulates HPG, LDF, vorticity, KEG, and ZAD into `Krhs` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176`, diagnoses its
depth mean into separate `Ue_rhs` and `Ve_rhs` arrays without overwriting
`Krhs` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:202-213`, passes that
slot through the external solve into stage 1 at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3.f90:188-202`, and then
executes the vector assignment from that unchanged slot at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:668-674`.

The compiled stage constructs W/transport immediately before the stage RHS,
but explicitly does not use stage-1 W for vector momentum at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:326-347`; the sole
stage-1 momentum call there is guarded by the non-vector branch at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:367-373`.  The
barotropic correction follows the RK write at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:734-759`.

A frozen candidate therefore carried the full accumulator directly through
the vector stage-1 write and left the model-only post-external momentum
recomputation to the non-vector arm.  It made all fourteen handoff rows BIT in
production JIT and eager, made isolated-JIT raw and corrected U/V BIT, and
made the given-entry kt1 stage-1 U/V outputs BIT.  Nevertheless, Rule 12
failed: 88 of 954 rows moved, 64 rows contained a cell worsening by more than
two row-scale oracle ULP, and day-30 T RMS worsened by
`1.088950253e-9 K`.  The candidate is therefore **HELD**, not landed.

## Provenance and immutable arms

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round109.md` at
`607dfea38`.  The consolidated instrument landed at `2d57550fc`; the isolated
correction bridge was repaired at `50522d116`; the frozen candidate addendum
landed at `068e18a31`; the measured candidate is `2c9bcbf43`; and production
was restored at `eb8b062df`.  The exact held hunk is preserved in
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round109_held_vector_stage1_handoff.patch`.

The immutable short-trajectory before arm is
`phase3/merge_main_2026-09-17/after/ladder.json`.  The mechanically complete
954-row comparison uses the bit-identical full baseline
`phase3/round96/ladder_full.json`; the merge receipt proves that the later
main merge moved no trajectory row.  The immutable day arm is
`phase3/merge_main_2026-09-17/after/day_gap.json`.

The first isolated corrected-output artifact was invalid: it used the stage
writer's pre-external zero Kaa slot as the correction target and produced an
approximately `4e-4` error.  It is retained as
`handoff_corrected_output_isolated_jit.json`, not silently replaced.  The
admitted correction target is the recorded post-external target in
`handoff_corrected_output_isolated_jit_v2.json`.

## Baseline compiled-order handoff walk

Counts are unequal IEEE binary64 words.  PJ is the complete production step;
PE is the complete production closure with JIT disabled.  Both use NEMO's
recorded kt1 stage entry and external target.

| boundary | PJ U / V unequal, max | PE U / V unequal, max | verdict |
|---|---:|---:|---|
| full accumulator | `0 / 0`, `0.0` | `0 / 0`, `0.0` | BIT |
| projected accumulator | `17,400 / 17,100`, `2.0121494123449567e-8` | same | first non-bit statement |
| after post-external transport replacement | `17,400 / 17,100`, `2.0121494123449567e-8` | same | discrepancy retained |
| after W/ZAD replacement | `17,400 / 17,100`, `2.0121494123449567e-8` | same | discrepancy retained |
| final RK input | `17,400 / 17,100`, `2.0121494123449567e-8` | same | discrepancy retained |
| raw RK write | `17,400 / 17,100`, `9.658317179255792e-5` | same | discrepancy retained |
| corrected output | `7,620 / 8,460`, `5.421010862427522e-20` | `7,920 / 8,640`, same max | JIT/eager fusion changes counts after the same first boundary |

The isolated-JIT raw and corrected statements are BIT when supplied NEMO's
recorded full source, entry velocity, clocks, masks, admitted external target,
and reference-depth correction geometry.  Isolated JIT is a discriminator,
not a substitute for the production rows above.

## Candidate stage proof and plant

Under the frozen candidate, every one of the seven selected U/V boundaries is
BIT in both PJ and PE: fourteen zero-count rows per mode.  The isolated-JIT
raw write and corrected output are also BIT.  The consolidated candidate
stage twin changes the given-entry kt1 stage-1 U/V rows from
`7,620 / 8,460` unequal at `5.421010862427522e-20` to BIT and advances the
first owned non-bit output to `ww`, 10,548 unequal cells at
`1.6543612251060553e-23` (AT-BAR).  No unrelated given-entry kt1 stage-1
field moves.

The candidate addendum predicted that the nonzero ULP plant could target raw
RK U.  That sub-prediction is **REFUTED**: the exact raw target on this
from-rest entry has no finite nonzero U cell, and the gate refused to perturb
a zero.  The required production plant was then run at the next exact,
finite, nonzero boundary, corrected U.  It changed exactly one cell at
`[1,1,0]` by `6.776263578034403e-21`, from clean count 0 to 1, left V BIT,
printed `STATUS PLANT-FIRED`, and exited 1.  Artifact:
`candidate_handoff_corrected_output_production_jit_plant.json`.

## Decision-41 stage tables

The final restored stage twin passed at clean commit
`eb8b062df684619a01b89590edc1f8d5b4bee022`.  Its complete `stage_twin`
payload is exactly equal to the pre-candidate instrument baseline, not merely
equal in the counts below.  Cells are **BIT / AT-BAR / DEBT**.

Given NEMO's recorded entry:

| kt | stage 1 | external step | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | `13 / 4 / 0` | `11 / 0 / 0` | `14 / 2 / 1` | `11 / 1 / 5` |
| 2 | `10 / 4 / 3` | `0 / 0 / 11` | `11 / 3 / 3` | `10 / 2 / 5` |

Model-chained entry:

| kt | stage 1 | external step | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | `8 / 4 / 5` | `10 / 1 / 0` | `5 / 5 / 7` | `3 / 5 / 9` |
| 2 | `3 / 0 / 14` | `0 / 0 / 11` | `0 / 0 / 17` | `0 / 0 / 17` |

The candidate's given-entry kt1 stage-1 cell is `15 / 2 / 0`, and its chained
kt1 stage-1 cell is `10 / 2 / 5`; U/V are the two rows that become BIT.  After
the Rule-12 veto and restoration, the first owned stage output is again kt1
stage-1 U, 7,620 unequal cells at `5.421010862427522e-20` (AT-BAR).

## Rule 12 and required headline

The complete oracle-relative gate is
`candidate_rule12_full.json`.  It compares all 954 certified rows against the
unchanged full baseline and reports:

- status `FAIL`;
- 88 moved rows and 64 violating rows;
- 281,040 cells worse than the two-ULP bar;
- 401,094 improved cells and 316,443 worsened cells;
- zero row-status changes and no AT-BAR row leaving its class;
- first-over-bar unchanged at kt2 U/V;
- largest oracle-residual worsening
  `1,962,944,187,516` row-scale oracle ULP.

The initial comparison against the compact merge-main artifact intersected
only 70 rows and is retained as `candidate_rule12.json`; it is not the
certification result.  `candidate_rule12_full.json`, with 954 rows, is the
landing veto.

| required headline | before | candidate | movement |
|---|---:|---:|---:|
| kt2 T max | `1.4210854715202004e-14` | `1.4210854715202004e-14` | same max; cell counts move |
| kt2 S max | `2.1316282072803006e-14` | `2.1316282072803006e-14` | same max; cell counts move |
| kt2 U max | `2.7377110452773967e-12` | `2.7377110452773967e-12` | same max; cells move |
| kt2 V max | `3.284922138989399e-12` | `3.2849219221489645e-12` | improves `2.168404344971009e-19` |
| kt3 T max | `1.627497246303733e-4` K | `1.6274970116469945e-4` K | improves `2.346567385e-11` K |
| kt3 S max | `6.327735185607253e-6` | `6.327751449930474e-6` | worsens `1.6264323221e-11` |
| day-30 T RMS | `1.2397011296352737e-2` K | `1.239701238530299e-2` K | worsens `1.088950253e-9` K |

Thus P6's no-headline-movement prediction and its `<1e-10 K` day-30 bound are
both refuted.  The candidate's kt3 T influence is about seven million times
smaller than the `1.6275e-4 K` gap, and it worsens rather than owns the
`1.2397e-2 K` day-30 gap.

### Complete registered moved-row table

This is the full set of 88 nonzero `field_moves` rows; none is omitted.

| row | max model movement | improved | worsened | >2-ULP violations |
|---|---:|---:|---:|---:|
| `GYRE-zco.kt1.stage1.u` | `5.4210108624275222e-20` | 7,620 | 0 | 0 |
| `GYRE-zco.kt1.stage1.v` | `5.4210108624275222e-20` | 8,460 | 0 | 0 |
| `GYRE-zco.kt1.stage2.oracle_stage1_thermodynamic_bundle.u` | `1.0842021724855044e-19` | 2,594 | 854 | 0 |
| `GYRE-zco.kt1.stage2.oracle_stage1_thermodynamic_bundle.v` | `8.1315162936412833e-20` | 2,172 | 642 | 0 |
| `GYRE-zco.kt1.stage2.u` | `1.0842021724855044e-19` | 2,594 | 854 | 0 |
| `GYRE-zco.kt1.stage2.v` | `8.1315162936412833e-20` | 2,172 | 642 | 0 |
| `GYRE-zco.kt1.stage3.u` | `6.2450045135165055e-17` | 2,289 | 2,257 | 0 |
| `GYRE-zco.kt1.stage3.v` | `9.0205620750793969e-17` | 2,300 | 2,290 | 0 |
| `GYRE-zco.kt1.zdf_entry.avm` | `2.7755575615628914e-17` | 229 | 124 | 0 |
| `GYRE-zco.kt1.zdf_entry.avt` | `1.7347234759768071e-18` | 229 | 122 | 0 |
| `GYRE-zco.kt10.after.uu_b` | `5.9925788723630413e-11` | 357 | 223 | 223 |
| `GYRE-zco.kt10.after.vv_b` | `6.5608493789691413e-11` | 253 | 317 | 317 |
| `GYRE-zco.kt10.before.S` | `3.3321555292786798e-8` | 9,005 | 8,461 | 7,638 |
| `GYRE-zco.kt10.before.T` | `7.8128568503643692e-8` | 9,215 | 8,750 | 8,639 |
| `GYRE-zco.kt10.before.ssh` | `5.6702580547574044e-9` | 288 | 312 | 312 |
| `GYRE-zco.kt10.before.u` | `4.5044051972219223e-7` | 9,595 | 7,805 | 7,805 |
| `GYRE-zco.kt10.before.v` | `5.9356920638700683e-7` | 7,827 | 9,273 | 9,273 |
| `GYRE-zco.kt2.after.uu_b` | `1.8865117801247777e-17` | 309 | 270 | 0 |
| `GYRE-zco.kt2.after.vv_b` | `1.5856456772600502e-17` | 293 | 277 | 0 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.S` | `2.1316282072803006e-14` | 88 | 92 | 2 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.T` | `1.0658141036401503e-14` | 107 | 98 | 2 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.u` | `1.4779844015322396e-15` | 2,474 | 2,560 | 2 |
| `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.v` | `1.4918621893400541e-15` | 2,231 | 2,063 | 2 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.S` | `2.1316282072803006e-14` | 100 | 74 | 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.T` | `1.0658141036401503e-14` | 105 | 86 | 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.u` | `9.0205620750793969e-17` | 2,605 | 2,153 | 0 |
| `GYRE-zco.kt2.arm.omit_freshwater_forcing.v` | `9.0205620750793969e-17` | 2,144 | 2,285 | 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.S` | `1.4210854715202004e-14` | 98 | 65 | 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.T` | `1.0658141036401503e-14` | 100 | 65 | 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.u` | `6.2450045135165055e-17` | 2,289 | 2,257 | 0 |
| `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.v` | `9.0205620750793969e-17` | 2,300 | 2,290 | 0 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.S` | `0.095462621568991324` | 16,152 | 1,709 | 1,687 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.T` | `0.060248666044543597` | 16,259 | 1,740 | 1,728 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.u` | `0.00090193609216617193` | 14,559 | 2,841 | 2,841 |
| `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.v` | `0.00089825329559687576` | 15,087 | 2,013 | 2,013 |
| `GYRE-zco.kt2.before.S` | `1.4210854715202004e-14` | 98 | 65 | 0 |
| `GYRE-zco.kt2.before.T` | `1.0658141036401503e-14` | 100 | 65 | 0 |
| `GYRE-zco.kt2.before.u` | `6.2450045135165055e-17` | 2,289 | 2,257 | 0 |
| `GYRE-zco.kt2.before.v` | `9.0205620750793969e-17` | 2,300 | 2,290 | 0 |
| `GYRE-zco.kt3.after.uu_b` | `1.4963363827082826e-11` | 216 | 364 | 364 |
| `GYRE-zco.kt3.after.vv_b` | `1.7765925842548776e-11` | 270 | 300 | 300 |
| `GYRE-zco.kt3.before.S` | `2.2445902914114413e-9` | 9,084 | 6,248 | 5,952 |
| `GYRE-zco.kt3.before.T` | `2.7645974398637918e-8` | 10,343 | 7,402 | 7,268 |
| `GYRE-zco.kt3.before.ssh` | `1.6145721912219635e-14` | 368 | 232 | 14 |
| `GYRE-zco.kt3.before.u` | `3.8344441652925221e-7` | 10,141 | 7,259 | 7,259 |
| `GYRE-zco.kt3.before.v` | `3.8233879398608117e-7` | 8,307 | 8,793 | 8,793 |
| `GYRE-zco.kt4.after.uu_b` | `2.9046860619899584e-11` | 377 | 203 | 203 |
| `GYRE-zco.kt4.after.vv_b` | `2.7994663838047835e-11` | 243 | 327 | 327 |
| `GYRE-zco.kt4.before.S` | `1.7198274804286484e-8` | 9,232 | 7,510 | 6,549 |
| `GYRE-zco.kt4.before.T` | `1.2514311720224214e-7` | 9,793 | 8,095 | 7,943 |
| `GYRE-zco.kt4.before.ssh` | `1.9792870461100148e-9` | 303 | 297 | 297 |
| `GYRE-zco.kt4.before.u` | `3.5704398446134711e-7` | 8,154 | 9,246 | 9,246 |
| `GYRE-zco.kt4.before.v` | `7.4185730649604237e-7` | 10,517 | 6,583 | 6,583 |
| `GYRE-zco.kt5.after.uu_b` | `3.9913191241663926e-11` | 343 | 237 | 237 |
| `GYRE-zco.kt5.after.vv_b` | `5.4259319517339748e-11` | 281 | 289 | 289 |
| `GYRE-zco.kt5.before.S` | `6.022546017447894e-9` | 8,113 | 8,980 | 7,838 |
| `GYRE-zco.kt5.before.T` | `7.2995476330106612e-8` | 8,269 | 9,628 | 9,354 |
| `GYRE-zco.kt5.before.ssh` | `4.4092534658146665e-9` | 325 | 275 | 275 |
| `GYRE-zco.kt5.before.u` | `7.0780449652846854e-7` | 7,047 | 10,353 | 10,353 |
| `GYRE-zco.kt5.before.v` | `3.2434979906244277e-7` | 9,445 | 7,655 | 7,655 |
| `GYRE-zco.kt6.after.uu_b` | `6.2878321690010286e-11` | 277 | 303 | 303 |
| `GYRE-zco.kt6.after.vv_b` | `4.8247328875150597e-11` | 312 | 258 | 258 |
| `GYRE-zco.kt6.before.S` | `4.8299476418378617e-8` | 8,143 | 9,145 | 8,133 |
| `GYRE-zco.kt6.before.T` | `2.2953446432438795e-7` | 8,467 | 9,477 | 9,337 |
| `GYRE-zco.kt6.before.ssh` | `5.7712736751050486e-9` | 307 | 293 | 293 |
| `GYRE-zco.kt6.before.u` | `3.9031130348241017e-7` | 8,777 | 8,623 | 8,623 |
| `GYRE-zco.kt6.before.v` | `6.5041565465613627e-7` | 7,488 | 9,612 | 9,612 |
| `GYRE-zco.kt7.after.uu_b` | `5.7909502063427376e-11` | 267 | 313 | 313 |
| `GYRE-zco.kt7.after.vv_b` | `6.8227528227798873e-11` | 267 | 303 | 303 |
| `GYRE-zco.kt7.before.S` | `5.7900493288798316e-8` | 8,345 | 8,945 | 7,985 |
| `GYRE-zco.kt7.before.T` | `2.0975554804181229e-7` | 8,646 | 9,308 | 9,216 |
| `GYRE-zco.kt7.before.ssh` | `3.7572209492936492e-9` | 274 | 326 | 326 |
| `GYRE-zco.kt7.before.u` | `6.9685262363457234e-7` | 8,307 | 9,093 | 9,093 |
| `GYRE-zco.kt7.before.v` | `4.8797327607125185e-7` | 8,691 | 8,409 | 8,409 |
| `GYRE-zco.kt8.after.uu_b` | `8.0666807868862156e-11` | 216 | 364 | 364 |
| `GYRE-zco.kt8.after.vv_b` | `6.1000725738030726e-11` | 318 | 252 | 252 |
| `GYRE-zco.kt8.before.S` | `1.7139250019226893e-8` | 8,652 | 8,765 | 7,858 |
| `GYRE-zco.kt8.before.T` | `9.9975729028756177e-8` | 9,062 | 8,893 | 8,751 |
| `GYRE-zco.kt8.before.ssh` | `5.9497829354299148e-9` | 337 | 263 | 263 |
| `GYRE-zco.kt8.before.u` | `4.7865250465961573e-7` | 8,258 | 9,142 | 9,142 |
| `GYRE-zco.kt8.before.v` | `7.1129939459464719e-7` | 8,387 | 8,713 | 8,713 |
| `GYRE-zco.kt9.after.uu_b` | `6.9151983184828625e-11` | 317 | 263 | 263 |
| `GYRE-zco.kt9.after.vv_b` | `5.3434520805480112e-11` | 257 | 313 | 313 |
| `GYRE-zco.kt9.before.S` | `6.5029652773773705e-9` | 9,064 | 8,389 | 7,552 |
| `GYRE-zco.kt9.before.T` | `9.4455415933225595e-8` | 9,531 | 8,431 | 8,352 |
| `GYRE-zco.kt9.before.ssh` | `4.3056829074350489e-9` | 312 | 288 | 288 |
| `GYRE-zco.kt9.before.u` | `6.9751723085105521e-7` | 9,119 | 8,281 | 8,281 |
| `GYRE-zco.kt9.before.v` | `4.588293366193999e-7` | 8,239 | 8,861 | 8,861 |

## Other configurations

P7's statement that DINO executes this vector WS-RK3 handoff is **REFUTED**.
The resolved DINO identity recipe selects the leapfrog/MLF outer program and
does not select the `rk3_ws` stage program.  The candidate-only DINO unit tree
still reported `128 passed, 9 warnings in 106.88s`, but that is a
non-execution preservation check, not a fidelity pass and not evidence about
regional cancellation.  The restored tree is tested again below.

LOCK_EXCHANGE and OVERFLOW resolve `momentum_time_integrator=rk3_ws` but
`momentum_advection=flux_form`; the candidate predicate required
`vector_invariant`.  They therefore preserve the existing non-vector program
byte-for-byte and do not execute the changed statement.  The focused
LOCK_EXCHANGE transport regression passed under the candidate.  Neither tank
is falsely relabelled as an execution pass.

ORCA2 remains **UNMEASURED-WITH-SPEC**: instantiate its exact NEMO-identity
card in binary64, resolve integrator and advection branch, then compare from
one identical entry the production stage-1 full and projected RHS, the
post-transport/W boundary, raw and corrected U/V, transports, W, and every
next-stage consumed field before and after this held candidate.

## Frozen prediction ledger

| prediction | verdict | evidence |
|---|---|---|
| P1 reproduced baseline | **CONFIRMED** | exact full accumulator; projected and later RHS `17,400 / 17,100` at `2.0121494123449567e-8`; restored stage U/V `7,620 / 8,460` |
| P2 one-boundary walk | **CONFIRMED** | projection first under production JIT and eager; isolated raw/corrected statements BIT |
| P3 compiled vector discriminator | **CONFIRMED** | compiled branch uses stage W only for tracers and calls momentum advection only for non-vector stage 1 |
| P4 narrower candidate local proof | **CONFIRMED** | all 14 handoff rows BIT in each production mode; candidate kt1 stage-1 U/V BIT |
| P5 raw-RK plant target | **REFUTED**, control itself **CONFIRMED** | raw target has no nonzero cell; corrected-output plant flips exactly one nonzero U word and exits 1 |
| P6 Rule 12 and magnitude | **REFUTED** | 64 violating rows; headline movement; day-30 T worsens `1.088950253e-9 K` |
| P7 DINO executes vector WS-RK3 | **REFUTED** | DINO is leapfrog/MLF; tanks are RK3 flux-form; candidate is absent from all three branches |
| P8 instrumentation neutrality | **CONFIRMED** | final restored complete stage payload equals the pre-candidate baseline |

## Verification and independent review

The separate review was invoked with `codex exec --sandbox read-only` from
the clean writable clone.  It exited 1 before reading the diff, so there is
no scientific verdict to conceal or quote.  Its complete diagnostic was:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
>
> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Accordingly: **independent review unavailable in-sandbox**.  Operator note C
explicitly permits the round to continue in this condition; it does not turn
the unavailable review into an approval.

The receipt citation gate reported `STATUS PASS`: 7 citations found, no
failures, no unmapped citations, and no failing map entries.  Its deliberately
shifted first compiled-stp2d citation reported `STATUS FAIL` and exited 1 with
`SYMBOL-NOT-AT-LINE`.  The citation-gate pytest summary was `16 passed in
1.85s`; after the final rigid re-anchor it was rerun and reported `16 passed
in 1.87s`.

The focused candidate transport/stage tests reported `38 passed in 119.23s`.
After restoring the rejected candidate, the same focused suite reported `38
passed in 120.08s`; the clean round-29 replay reported `9 passed in 2.29s`;
and the private live-trace guard reported `3 passed in 0.56s`.  The
candidate-only DINO suite reported `128 passed, 9 warnings in 106.88s`, and
the final restored DINO suite reported `128 passed, 9 warnings in 105.29s`.
After the two focused report stamps were added, the final focused rerun
reported `38 passed in 122.39s`.
The formerly documented RK3 unit red was rechecked independently and now
reported `1 passed, 1 warning in 85.20s`.

The report-emitter ratchet remains the declared incoming red.  Its exact
focused summary was `1 failed in 1.60s`.  The offender dictionary contains
the same nine filenames and counts as the untouched incoming checkout,
including 12 pre-existing dictionaries in the round-46 gate.  During final
verification the two new focused report dictionaries initially raised that
count to 14; both were given fail-closed `worktree_stamp()` fields and the
dictionary returned exactly to the incoming value.  No pre-existing offender
was changed.

The requested complete-tree command was run on CPU.  The first combined
12-worker invocation collected 8,163 tests but repeatedly aborted workers in
JAX `backend_compile_and_load`; it was interrupted after losing forward
progress and emitted **no terminal summary**.  Splitting the trees, as the
repository instructions require for the compiler ceiling, produced these
additional bounded results:

* fidelity with four workers reached 97% with four failures, then stopped
  producing output; interruption emitted **no terminal summary**;
* sequential fidelity fail-fast passed every round-109 and neighboring stage
  test through 72%, then stopped inside the same long stage-sweep case for
  more than six minutes; interruption emitted **no terminal summary**;
* the four-worker unit tree reached 23% before a worker fatally aborted in
  JAX `backend_compile_and_load`; interruption emitted **no terminal
  summary**;
* a fresh-process first unit shard did terminate and reported `11 failed,
  818 passed, 1 skipped, 2 xfailed, 5 warnings in 599.53s (0:09:59)`; its
  terminal log includes unrelated assertion failures plus LLVM
  `Cannot allocate memory` failures.  It is a broad-tree diagnostic, not a
  green gate and not evidence for landing the held candidate.

An earlier sequential fidelity attempt, made before the trace-layout repair
was committed, correctly exercised the fail-closed dirty-tree refusal and
reported `1 failed, 596 passed, 5 skipped, 21 deselected in 319.54s
(0:05:19)`.  The refused round-29 file then passed 9/9 once the repair was
committed, as recorded above.  No broad run is represented as passing.

## OPEN — round 110

1. Stay at kt1 stage-1 U.  The source-exact handoff cannot land because its
   chained trajectory violates Rule 12; do not move downstream to `ww`, TKE,
   or `bn2` merely because the candidate's local table advances there.
2. Use the existing year-owner NEMO step-entry dumps for kt1..60 to run the
   held handoff from identical later step entries.  Find the earliest step at
   which carrying the full source ceases to improve the next recorded state,
   and separate an inherited entry error from a local stage error under the
   complete production JIT.  This is the discriminating measurement for why
   a kt1 source-exact handoff damages 64 ladder rows.
3. At that earliest step, score full accumulator, projected accumulator,
   post-transport/W RHS, raw output, corrected output, and next NEMO step
   entry together.  Plant the production row and preregister any new candidate
   before editing arithmetic.  Re-evaluate the held patch only within kt1
   stage 1; do not bundle across stages.
4. A landing still requires local production-JIT stage closure and the full
   954-row Rule-12 ladder plus days 1-30.  Re-audit execution on DINO and both
   tanks if the eventual candidate reaches a shared branch, and retain the
   ORCA2 one-step spec.

No NEMO acquisition, configuration decision, or carried-state decision is
requested.
