# NEMO testcase L2 GYRE round 99: stage-one ratio/clock pair held

Date: 2026-09-16  
Status: **HELD; LOCAL STAGE PROOF PASSED, RULE 12 FAILED**  
Writable clone: `/tmp/autopilot-work-124594901`  
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round99`

## Verdict

The Round-98 record was recovered without rebuilding NEMO. Reading the
compiled stage program changed the owner twice: the 201-cell `r3t(Kaa)`
residual was not a bad reciprocal but an omitted independent HYB ratio
interpolation, and the following 18,000-cell W residual was the model's
full-step clock paired with NEMO's stage-local ratio. The final same-stage
candidate paired NEMO's ratio association, stage-local clock, source-ordered W
recurrence, and unprojected stage-one momentum source. Given NEMO's kt=1
stage-1 entry, all 17 required outputs became bit-identical.

The candidate cannot land. The canonical 954-row trajectory comparison moved
85 rows; 56 rows and 131,713 cells violated the two-ULP cellwise movement bar.
No class changed and first-over-bar remained kt2 U/V, but Rule 12 is still
**FAIL**. Day-30 T RMS improved by only
`3.66062562207059e-12 K`. Production was restored at
`6f2b909f9b085c8ffc671c70b0982766f5d5ba53`; the complete candidate survives
only as
`scripts/validate/ocean_fidelity/testcases/manifests/
nemo_testcase_l2_gyre_round99_held_stage1_w_ratio_clock_full_rhs.patch`.

No configuration, coefficient, timestep, carried state, stabilizer, year
harness, reconciliation gate, freshwater pair, or #1484 guard changed.

## Frozen sequence and retractions

The initial direct-walk contract was committed at
`110bc3e4d67482c2b5870f3f50cb5bc8ab73abaf` before parsing the recovered
payload. The HYB association addendum was committed at
`bff4b8b08e4de8522ba7d2cfe90d67e2ecd0f942` before replaying it. The clock
addendum and paired-candidate distinction were committed at
`8819aeed66cffea119712fa5c2ac5c385a631414` and
`f719ed0e43825f0333c22135f8c0be8fda316dc6` before measuring the pair.

The frozen predictions resolved as follows.

1. **CONFIRMED:** the admitted direct `hdiv`, `e3div`, `r3t(Kbb)`, static
   thickness, and reciprocal clock inputs are BIT against the source-rounded
   walk.
2. **REFUTED:** rebuilding the stored static reciprocal and multiplying the
   already-interpolated SSH did not make `r3t(Kaa)` BIT. It left 201/600 cells
   unequal at `2.6469779601696886e-23`; final W then left 5,289 cells unequal
   at the same maximum.
3. **CONFIRMED, replacing prediction 2:** NEMO independently interpolates the
   QCO ratio after forming the full-step ratio. That association makes
   `r3t(Kaa)`, every direct W recurrence boundary, and final W BIT.
4. **REFUTED:** the source-ordered ratio/W candidate plus the full momentum
   source did not close the stage. With the model's 14,400-s clock, W stayed
   unequal in all 18,000 wet cells at `1.318522148478393e-7 m s-1`; T and S
   also remained DEBT.
5. **CONFIRMED locally:** pairing the stage-local ratio with the 4,800-s
   stage-one clock made all 17 kt=1 stage-1 outputs BIT. This does not revive
   Round 21's correctly refuted clock-only arm; neither half is eligible
   alone.
6. **REFUTED globally:** the frozen bitwise-trajectory prediction failed even
   though all six headline ladder maxima were unchanged. The cellwise Rule-12
   gate and day-30 member both moved.

The direct-walk evidence is `stage_w_direct_43732e.json` (SHA-256
`95ae3cbba6382d66ee74df119a8889d18b42a7893308a075c2dbdfd3874d055b`).
The corrected HYB replay is `stage_w_hyb_a5a0e797.json` (SHA-256
`5adda65e33254b3fa6214d12190ac2c9b4b3d677aff406125a3fd7d8d46ff1c6`).
The half-state stage veto is `candidate_stage_twin_cdddd17d.json` (SHA-256
`5fb59fa42ffb90d8a450fd0c0627b7f22cce4b314a9838f85425f47377ca6815`).

## Round-98 record recovery

The compiled record is 884,028 bytes. Its two divergence arrays use local
`34*24*31` extents; its two ratio arrays use full `36*26` extents; static
thickness and W use full `36*26*31` extents; and it carries one scalar. The
exact byte expression is
`16 + 9*4 + (2*34*24*31 + 2*36*26 + 2*36*26*31 + 1)*8`.

The recovered record has SHA-256
`ebdae2ad5135ad3ff6887f1d3e0ce440398bef832e0343c6b179fdbfc55351a4`,
the executed binary has SHA-256
`659c934535173b7124840c725b7e85ef615feabdf2305bee671e51c0863f367a`,
and the original run ended with `STOP 0`. Recovery verified those identities,
the source/toolchain manifests, restart and mesh byte identity, exact physical
EOF, and consumed-field admission before writing the missing stamp and
`RUN_DONE`. It did not call `makenemo` or `mpirun`.

The admission report passed with no violations; artifact SHA-256 is
`add99d49e83f968b4d3647c385ae8312b41fc76fd13521556d5e9e1a9ee1b20a`.
The wrong-layout and wrong-size controls both exited 69 with named `REFUSE`
messages. Their log SHA-256 values are
`e01f61653ba0e81d63b84e8342e6f826ccde415284f1963849bc5ef526cf2459`
and
`fc17cbf273b8aff8cae457f51b9cf432fe54f5aaabe8cbb432926b5303260469`.

## Candidate stage-twin tables

`B`, `A`, and `D` mean BIT, AT-BAR-not-BIT, and DEBT. Cells are
`class/unequal/max_abs`. The canonical row-per-field table is
`candidate_pair_stage_twin_b01e550a.json`, SHA-256
`d37ac54fbe9bcf587bfa3a8feccc4c013cb54545c3049a2f84ff8151ce28f74d`.

Given NEMO's recorded entry:

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | B/0/0 | Hu B/0/0 ; Hv B/0/0 | — | — | histories all six B/0/0 |
| 1 | 1 | **B/0/0 ; B/0/0** | **B/0/0 ; B/0/0** | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | **B/0/0** | all five B/0/0 | **B/0/0 ; B/0/0 ; B/0/0** |
| 1 | 2 | B/0/0 ; B/0/0 | A/4787/1.0842022e-19 ; A/4600/6.7762636e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 1 | 3 | D/8851/1.2830175e-2 ; D/8394/5.1963777e-4 | D/17400/1.0117023e-2 ; D/17100/1.0117027e-2 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 2 | external | — | uu D/580/5.0398350e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725601e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099464e-4 | — | — | all six histories D |
| 2 | 1 | A/4493/3.5527137e-15 ; A/4436/7.1054274e-15 | D/17400/9.1413673e-6 ; D/17100/9.3268149e-6 | A/273/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 2 | 2 | A/4551/3.5527137e-15 ; A/4497/7.1054274e-15 | D/17400/7.5775705e-6 ; D/17100/9.6194351e-6 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326762e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 2 | 3 | D/17988/7.9703416e-3 ; D/15488/1.0668959e-3 | D/17400/2.4760020e-3 ; D/17100/5.3033396e-3 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326760e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |

The candidate moves the first owned non-bit output to kt=1 stage 2 U/V
(AT-BAR-not-BIT). Because Rule 12 vetoed the candidate and production was
restored, the campaign itself remains at kt=1 stage-1 W.

Model-chained entry:

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | A/600/4.3368087e-19 | Hu B/0/0 ; Hv B/0/0 | — | — | histories all six B/0/0 |
| 1 | 1 | **B/0/0 ; B/0/0** | **B/0/0 ; B/0/0** | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | **B/0/0** | en D/914/5.5368e-3 ; avm D/5714/8.1762e-2 ; avt D/5714/8.1762e-3 ; dissl D/17400/6.3985e-3 ; surface D/571/8.9940e-3 | **B/0/0 ; B/0/0 ; B/0/0** |
| 1 | 2 | B/0/0 ; B/0/0 | A/4787/1.0842022e-19 ; A/4600/6.7762636e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | same five TKE rows as kt1 stage 1 | B/0/0 ; B/0/0 ; B/0/0 |
| 1 | 3 | A/6859/1.4210855e-14 ; A/6455/2.1316282e-14 | D/17398/2.7377110e-12 ; D/17100/3.2849221e-12 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | same five TKE rows as kt1 stage 1 | A/4750/3.1832315e-12 ; A/4528/2.1600499e-12 ; D/17184/5.2665428e-11 |
| 2 | external | — | uu D/580/5.0398351e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725590e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099463e-4 | — | — | all six histories D |
| 2 | 1 | D/15470/2.1881164e-11 ; D/11353/1.4921397e-12 | D/17400/9.1389188e-6 ; D/17100/9.3307505e-6 | D/600/2.3575197e-7 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | en D/979/1.6795e-3 ; avm D/5723/3.1057e-2 ; avt D/5723/5.7930e-2 ; dissl D/17400/4.8875e-3 ; surface D/571/1.0767e-6 | D/17400/8.9161e-1 ; D/17100/7.4854e-1 ; D/17997/1.4739e-4 |
| 2 | 2 | D/17993/8.3278006e-7 ; D/17079/6.7607530e-8 | D/17400/7.4535508e-6 ; D/17100/9.6175689e-6 | D/600/3.5362795e-7 | D/18000/1.6484023e-8 ; D/17400/1.5141097e-8 ; D/17100/1.5097498e-8 | D/18000/4.8326762e-7 | same five TKE rows as kt2 stage 1 | D/17400/10.322200 ; D/17100/10.534822 ; D/17993/45.868521 |
| 2 | 3 | D/18000/1.6274972e-4 ; D/17378/6.3277352e-6 | D/17400/1.2295528e-5 ; D/17100/2.3465017e-5 | D/600/7.0725590e-7 | D/18000/2.4725978e-8 ; D/17400/2.2711674e-8 ; D/17100/2.2646248e-8 | D/18000/4.8326760e-7 | same five TKE rows as kt2 stage 1 | D/17400/8.1093193 ; D/17100/10.850152 ; D/17986/23.680199 |

## Rule 12 and magnitude

| headline | immutable before | paired candidate | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | `1.4210854715202004e-14` | AT-BAR retained |
| kt2 S | `2.1316282072803006e-14` | `2.1316282072803006e-14` | AT-BAR retained |
| kt2 U | `2.7377110452773967e-12` | `2.7377110452773967e-12` | first-over-bar unchanged |
| kt2 V | `3.284922138989399e-12` | `3.284922138989399e-12` | first-over-bar unchanged |
| kt3 T | `1.627497246303733e-4` | `1.627497246303733e-4` | unchanged |
| kt3 S | `6.327735185607253e-6` | `6.327735185607253e-6` | unchanged |
| day-30 T RMS | `1.2397011295506804e-2 K` | `1.2397011291846179e-2 K` | improved `3.66062562207059e-12 K`; invariance refuted |

The full Rule-12 artifact is `ladder_rule12_b01e550a.json`, SHA-256
`78ef102e2d4f584225e238fef66cf64ada837f9c3cd0a206193a3254d259a1e0`.
It checks all 954 registered rows, retains every one of the 85 moved rows,
records 56 violating rows and 131,713 violating cells, records no class change,
and keeps first-over-bar at kt2 U/V. The fresh member completed days 1--30;
its score artifact is `day_gap_b01e550a.json`, SHA-256
`81896fb17b24e04b5a5dcb742ae770f1623837cfbc3cb6ad7f4da33678218f17`.

## Testcase dispositions

| lane | disposition |
|---|---|
| GYRE stage twin | Paired candidate makes all 17 kt1-stage1 outputs BIT given NEMO entry; restored production leaves stage-1 W owned and non-BIT |
| GYRE kt=1--10 | **FAIL / HELD:** 85/954 rows moved; 56 rows and 131,713 cells violate Rule 12; no class change; first-over-bar remains kt2 U/V |
| GYRE days 1--30 | Fresh candidate member complete; day-30 T RMS improves by `3.66062562207059e-12 K` |
| LOCK_EXCHANGE-zco | Shared-path ownership and constructibility tests pass; no tank-fidelity claim from GYRE operands |
| OVERFLOW-zps | Shared-path, partial-cell, and constructibility tests pass; no tank-fidelity claim from GYRE operands |
| DINO | **SHARED-STATEMENT RISK:** the shared QCO W helper and WS-RK3 clock execute; no trajectory-neutrality claim, and the 96--98% regional-cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** prove the resolved integrator, then record native stage-entry transports, direct W intermediates and carries, per-stage clocks/ratios, stage outputs, histories, and closure state with production-JIT red plants |

## Controls, restoration, and acquisition handoff

The direct `r3t(Kaa)` one-ULP plant, exact incoming-carry one-ULP plant,
record-producer stamp plant, and commit-stamp plant each exited 1. The first two
changed exactly one named cell. The restored-production walk is
`final_restore_w_walk_6f2b909f.json`, SHA-256
`a0ccc6e2f13052c79a44e5929729e4066b575cbbace335c61974cf468a6c9958`;
it proves the held candidate did not leak into production.

Before the read-forward found the HYB association, the round added a fail-closed
source card for the exact same-call SSH and stored reciprocal operands. Its
new target is `GYRE_OMIP_L2_P3_SM_R99R3OP`; it copies the source card
file-by-file, applies only an additive writer, proves the patched source with
`gfortran -fsyntax-only`, and admits restart/mesh identity plus a new exact-EOF
record. Preflight exits 0; removing the stored reciprocal from the write layout
exits 69. The operator-facing script remains
`/tmp/autopilot-work-124594901/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round99_stage1_r3_operands/run.sh`.
It was not run in this round. If the operator runs it, the next round must
consume its admission and binary stamp; if the wrapper stops after NEMO,
recover with `--admit-existing` rather than rebuild an unchanged target.

## Compiled-source citations

The record-size correction follows the compiled allocation of full W and local
`hdiv` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/oce.f90:99-104`, the local
`ze3div` declaration at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:258-260`, and the
resolved loop bounds at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/mppini.f90:1501-1516`.
The producer writes the two local divergence fields, two full ratio fields,
full static thickness and clock at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:283-291`, then executes
and writes the bottom-up W recurrence at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/sshwzv.f90:305-317`.

The transport branch divides its flux divergence by live thickness at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90:132-139` and
materializes the separate live-thickness product at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/divhor.f90:152-154`. The executing
tracer branch calls transport-form W and then materializes `pFw` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/traadv.f90:266-280`.

NEMO builds the stored static reciprocal in the compiled column loop at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domain.f90:198-212` and forms a
full-step ratio from SSH and that reciprocal at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/domqco.f90:256-258`. The stage-1
program saves full-step SSH, independently interpolates SSH, forms the
full-step ratio, and then independently interpolates `r3t(Kaa)` at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:150-192`.

The same compiled program assigns the stage-local reciprocal clocks at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:140-150`,
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:198-202`, and
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:242-246`.

Finally, the external program accumulates the full momentum source without
projecting its slot at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:141-176`, diagnoses the
barotropic mean into separate arrays at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stp2d.f90:202-213`, and the
selected vector stage consumes the unchanged full source at
`GYRE_OMIP_L2_P3_SM_R98WWALK/BLD/ppsrc/nemo/stprk3_stg.f90:668-674`.

## Review, validation, and choices

The required separate read-only Codex review was attempted on the paired
candidate and the 954-row Rule-12 table. Independent review was unavailable
in-sandbox. Its verbatim terminal result was:
**“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. Artifact SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
The unavailable review is not treated as approval; the trajectory gate already
requires HOLD.

The focused stage/citation unit suite passed **35 tests in 4.31 s**. The
separate LOCK_EXCHANGE, OVERFLOW, and card-constructibility suite passed
**31 tests in 226.36 s** after the operator-killed process was resumed; its
artifact SHA-256 is
`fc5333c88538b232c0c43d22673633443403a3ccd02033cdc32bcc10fd2257da`.
The receipt citation gate maps all 17 compiled-source citations and passes;
shifting the stage-1 clock citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`.

GitHub issue #1455 could not be read or updated: this clone's only remote is
the local read-only source checkout. No issue-update claim is made.

Choices made: none. All numerical changes measured here were source statements
selected by the user's Decision 41 stage order. The candidate was restored
after the immutable Rule-12 gate failed. No unasked choice was made.

UNVERIFIED: DINO trajectory neutrality and ORCA2 stage identity remain
unmeasured for this held candidate. The Round-99 operand acquisition has not
been run.

## OPEN for round 100

1. Apply operator note L/L-amend before any new landing attempt. Re-run the
   held Round-89 assignment-boundary member's local proof in three explicitly
   labelled modes: isolated-closure eager, isolated-closure JIT, and through
   the production step/stage twin. The production-step plant must fire. Report
   unequal-cell counts for all three and adopt production-JIT proof as binding
   only if the registered discriminator fires.
2. Stay at restored production's kt=1 stage-1 W boundary. Re-evaluate the held
   Round-99 ratio/clock/full-RHS candidate only through the production step,
   not an isolated statement closure. Determine which production-fused operand
   or time level explains why a locally BIT stage candidate moved 85 trajectory
   rows; do not advance to the external step or stage 2.
3. If the operator ran the Round-99 acquisition, admit the existing new target
   and use its exact same-call SSH/reciprocal record as an independent operand
   check. Do not rerun `makenemo` or `mpirun` for an unchanged binary.
4. A candidate may land only if its production-step kt=1 stage-1 outputs are
   BIT, all production-JIT plants fire, and the 954-row Rule-12 gate passes.
   Preserve every failed eager/JIT prediction and every moved row.
5. Keep the magnitude targets explicit: kt3 T
   `1.627497246303733e-4 K` and day-30 T RMS
   `1.2397011295506804e-2 K`. Do not infer W ownership of either from local
   exactness.

No configuration or carried-state decision is needed.
