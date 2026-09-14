# NEMO-testcases L2 GYRE round 94 receipt: stage closure remains held

Date: 2026-09-14. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**HELD; NO PRODUCTION PHYSICS LANDED.** The single Round-46/51 stage gate now
scores 78 stage-entry rows and 124 output rows in each of the
given-NEMO-entry and model-chained tables. It closes all 24 previously missing
absolute-history rows, all admitted direct transport rows, the QCO ratios, and
the admitted TKE/coefficient bundle. The final artifact is
`round94/stage_twin_final_stamped.json`, SHA-256
`0db0dc68aa3c1ead7a984aa966b0266981cdfcbbbef03f51ed05ce707ba1527d`,
at clean instrument commit `320cba4f79729ab51b56ed2b3518a0222be80ee3`.

The gate prints `STATUS UNMEASURED` and exits nonzero. Nine required rows are
still loud: kt=2 stage-3 `zFu/zFv/zFw` in both tables and the model-chained
kt=1 `tke_en` row at each stage. Consequently `first_owned_nonbit` is
mechanically `null`. The retained ordering diagnostic is kt=1 stage-1 U,
AT-BAR but non-bit in 7,620 cells at `5.421010862427522e-20`; the larger
measured field at that boundary is W, DEBT in all 18,000 cells at
`3.5937485546815465e-8`. **FIRST OWNED STAGE: NONE. FIRST NON-BIT STATEMENT:
NONE.** Decision 41 forbids an operator walk or statement claim before the
stage contract closes.

No production package, configuration, coefficient, timestep, stabilizer,
carried-state policy, year harness, reconciliation gate, freshwater pair,
#1484 guard, held manifest, NEMO source, or NEMO executable changed.

## Compiled program and record boundaries

The compiled driver calls `zdf_phy` before the external solve at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168-190`, then executes
stages 1, 2, and 3 and their pointer swaps at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:195-222`. The Round-46
stage writer records the state, TKE, coefficient, and QCO fields with their
actual Kbb/Kmm/Kaa header at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/l2_r46_stage.f90:120-147`.

The external solver writes its live histories before rotating
`bb <- b`, `b <- n`, `n <- a`; the executing rotations are at
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:834-846`.
Round 94 therefore scores the last record's pre-swap `n/b` as the completed
`b/bb` endpoint, rather than treating the record as post-swap.

The stage constructs horizontal transports at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:278-324` and calls
`tra_adv_trp` before the direct stage-3 record at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:792-819`.
The admitted stage-1/2 stream is likewise written after that call at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/stprk3_stg.f90:843-853`.
Inside `tra_adv_trp`, NEMO zeros the bottom transports at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90:247-250` and, where a
vertical transport exists, runs the stage-3 modifiers and assigns
`pFw=e1e2t*ww` at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/traadv.f90:253-280`.

## Stage-entry identity

Cell format below is `class / unequal cells / max abs`. All 30 T/S/U/V/SSH
rows and all 30 TKE `en/avm/avt/dissl/surface-avm` rows are BIT. The 18 QCO
ratio rows are:

| kt | stage | r3t / r3u / r3v |
|---:|---:|---|
| 1 | 1 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 |
| 1 | 2 | AT-BAR/600/1.1069635e-16 ; AT-BAR/580/1.1071552e-16 ; AT-BAR/570/1.1094453e-16 |
| 1 | 3 | AT-BAR/600/1.1017615e-16 ; AT-BAR/580/1.1084456e-16 ; AT-BAR/570/1.1097405e-16 |
| 2 | 1 | AT-BAR/600/1.1031574e-16 ; AT-BAR/580/1.1010195e-16 ; AT-BAR/570/1.1078900e-16 |
| 2 | 2 | AT-BAR/600/1.1055103e-16 ; AT-BAR/580/1.1074680e-16 ; AT-BAR/570/1.1064050e-16 |
| 2 | 3 | AT-BAR/600/1.1083024e-16 ; AT-BAR/580/1.1047771e-16 ; AT-BAR/570/1.1036056e-16 |

Thus the frozen prediction that every newly added context-entry row would be
BIT is **REFUTED** for 15 QCO rows. They remain AT-BAR, never relabeled BIT.
The TKE-entry portion of the prediction is confirmed.

The final-commit one-ULP artifact
`round94/stage_context_ulp_plant_final.json`, SHA-256
`5189d6e57eb16958cdba605123a434de751ec861cebe5c0dce1cd412cdf65068`,
changes exactly one finite kt=1 stage-1 `tke_en` cell: one unequal cell at
`3.3881317890172014e-21`. The plant command exits 1 as required. The separate
commit-stamp plant also exits 1 before record consumption; its log SHA-256 is
`c168fd047d23419d2530ac37cee7c1fd1f0b619be07e466c1f2dcd287f26f743`.

## Given-NEMO-entry output table

`B`, `A`, `D`, and `U` mean BIT, AT-BAR, DEBT, and
UNMEASURED_WITH_SPEC. Every cell is `class/unequal/max`; the literal JSON has
one row per field.

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | B/0/0 | Hu B/0/0 ; Hv B/0/0 | — | — | histories ub/ubb/vb/vbb/sshb/sshbb all B/0/0 |
| 1 | 1 | B/0/0 ; B/0/0 | A/7620/5.4210109e-20 ; A/8460/5.4210109e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/3.5937486e-8 | all five B/0/0 | B/0/0 ; B/0/0 ; A/10232/1.9895197e-13 |
| 1 | 2 | B/0/0 ; B/0/0 | A/4787/1.0842022e-19 ; A/4600/6.7762636e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 1 | 3 | D/8851/1.2830175e-2 ; D/8394/5.1963777e-4 | D/17400/1.0117023e-2 ; D/17100/1.0117027e-2 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 2 | external | — | uu D/580/5.0398350e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725601e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099464e-4 | — | — | ub D/580/4.9213669e-8 ; ubb D/580/4.8141663e-8 ; vb D/570/4.6857490e-8 ; vbb D/570/4.5674083e-8 ; sshb D/600/6.8902330e-7 ; sshbb D/600/6.7000577e-7 |
| 2 | 1 | A/4493/3.5527137e-15 ; A/4436/7.1054274e-15 | D/17400/9.1413673e-6 ; D/17100/9.3268149e-6 | A/273/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/14004/3.6379788e-12 |
| 2 | 2 | A/4551/3.5527137e-15 ; A/4497/7.1054274e-15 | D/17400/7.5775705e-6 ; D/17100/9.6194351e-6 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326762e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/812/3.6379788e-12 |
| 2 | 3 | D/17988/7.9703416e-3 ; D/15488/1.0668959e-3 | D/17400/2.4760020e-3 ; D/17100/5.3033396e-3 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326760e-7 | all five B/0/0 | U ; U ; U |

The large kt=1 stage-3 result after adding the recorded TKE context is a new
discriminator, not an ownership verdict: T/S/U/V reach `1.2830175e-2`,
`5.1963777e-4`, `1.0117023e-2`, and `1.0117027e-2`. It shows that the
currently injected context is not yet a sufficient, correctly timed stage
program boundary even though every injected entry row is BIT or AT-BAR.

## Model-chained output table

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | A/600/4.3368087e-19 | Hu B/0/0 ; Hv B/0/0 | — | — | all six histories B/0/0 |
| 1 | 1 | B/0/0 ; B/0/0 | A/7620/5.4210109e-20 ; A/8460/5.4210109e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/3.5937486e-8 | U ; D/5714/8.1761548e-2 ; D/5714/8.1761548e-3 ; D/17400/6.3984618e-3 ; D/571/8.9940458e-3 | B/0/0 ; B/0/0 ; A/10232/1.9895197e-13 |
| 1 | 2 | A/2/3.5527137e-15 ; B/0/0 | A/6337/1.0842022e-19 ; A/5944/8.1315163e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | same five rows as kt1 stage 1 | A/7540/1.8189894e-12 ; A/8315/1.8189894e-12 ; D/15050/4.1836756e-11 |
| 1 | 3 | A/6858/1.4210855e-14 ; A/6457/2.1316282e-14 | D/17398/2.7377110e-12 ; D/17100/3.2849221e-12 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | same five rows as kt1 stage 1 | A/6296/3.1832315e-12 ; A/5860/2.1600499e-12 ; D/17446/5.1116444e-11 |
| 2 | external | — | uu D/580/5.0398351e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725590e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099463e-4 | — | — | ub D/580/4.9213669e-8 ; ubb D/580/4.8141663e-8 ; vb D/570/4.6857490e-8 ; vbb D/570/4.5674082e-8 ; sshb D/600/6.8902319e-7 ; sshbb D/600/6.7000566e-7 |
| 2 | 1 | D/15470/2.1881164e-11 ; D/11354/1.4992452e-12 | D/17400/9.1389188e-6 ; D/17100/9.3307505e-6 | D/600/2.3575197e-7 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | en D/979/1.6794735e-3 ; avm D/5723/3.1057322e-2 ; avt D/5723/5.7930292e-2 ; dissl D/17400/4.8874792e-3 ; surface D/571/1.0766770e-6 | D/17400/8.9161073e-1 ; D/17100/7.4853596e-1 ; D/17994/1.4739146e-4 |
| 2 | 2 | D/17993/8.3278006e-7 ; D/17079/6.7607530e-8 | D/17400/7.4535508e-6 ; D/17100/9.6175689e-6 | D/600/3.5362795e-7 | D/18000/1.6484023e-8 ; D/17400/1.5141097e-8 ; D/17100/1.5097498e-8 | D/18000/4.8326762e-7 | same five rows as kt2 stage 1 | D/17400/1.0322200e1 ; D/17100/1.0534822e1 ; D/17996/4.5868521e1 |
| 2 | 3 | D/18000/1.6274972e-4 ; D/17378/6.3277352e-6 | D/17400/1.2295528e-5 ; D/17100/2.3465017e-5 | D/600/7.0725590e-7 | D/18000/2.4725978e-8 ; D/17400/2.2711674e-8 ; D/17100/2.2646248e-8 | D/18000/4.8326760e-7 | same five rows as kt2 stage 1 | U ; U ; U |

## Frozen predictions and falsifiers

1. **CONFIRMED:** all six kt=1 given-entry absolute histories are BIT. At kt=2
   all six are measured DEBT rather than absent.
2. **CONFIRMED:** direct `zFu/zFv` are BIT given NEMO input at kt=1 stages
   1--3 and kt=2 stages 1--2. The kt=1 stage-1 `zFw` row is non-bit as
   predicted: AT-BAR in 10,232 cells at `1.9895196601282805e-13`. This is a
   transport-scale value; the frozen `3.5937485546815465e-8` magnitude belongs
   to the W velocity row and was not reused as a transport magnitude.
3. **PARTLY REFUTED:** all TKE entry rows are BIT and the ULP plant fires, but
   15 QCO entry rows are only AT-BAR as listed above.
4. **CONFIRMED:** no admitted direct kt=2 stage-3 post-`tra_adv_trp` record
   exists. Six paired output rows remain `UNMEASURED_WITH_SPEC`, the gate stays
   UNMEASURED, and no owner is promoted.
5. **CONFIRMED:** WRITE-only instrumentation did not move the certified
   trajectory or day-30 statistic.

Four development failures are retained under `round94/`: attempt 1 rejected a
kt-specific tracer pointer mismatch, attempt 2 rejected QCO face shape, attempt
3 exposed a missing chained TKE field instead of producing a table, and the
first complete table exposed that the observer had scored the caller's
pre-shim state rather than the public step's seeded coefficient carry. None is
cited scientifically. The corrected observer was rerun clean, and a structural
JSON comparison proved its numerical rows identical after the later ownership
label retraction.

## Rule 12, magnitude, and testcases

The fresh clean ladder artifact `round94/ladder.json`, SHA-256
`92ab767797ae11d40037ca55f53f58eede6161ab63180075283dcf13fc050608`,
preserves the Round-85 arm:

| row | before | Round 94 | disposition |
|---|---:|---:|---|
| kt2 T | 1.4210854715202004e-14 | 1.4210854715202004e-14 | AT-BAR retained |
| kt2 S | 2.1316282072803006e-14 | 2.1316282072803006e-14 | AT-BAR retained |
| kt2 U | 2.7377110452773967e-12 | 2.7377110452773967e-12 | first-over-bar unchanged |
| kt2 V | 3.284922138989399e-12 | 3.284922138989399e-12 | first-over-bar unchanged |
| kt3 T | 1.627497246303733e-4 | 1.627497246303733e-4 | unchanged |
| kt3 S | 6.327735185607253e-6 | 6.327735185607253e-6 | unchanged |

The fresh 30-day member completed all days. `round94/day_gap.json`, SHA-256
`cd0f2b83ab3f56d8a11af224972b0d8d29602599d48e98fde71c891cd2a7d686`,
reports day-30 T RMS `1.2397011295506804e-2 K`, exactly unchanged.

| testcase | disposition |
|---|---|
| GYRE kt=1--10 | No candidate; every headline is unchanged and first-over-bar remains kt2 U/V |
| GYRE days 1--30 | No candidate; day-30 T RMS unchanged |
| LOCK_EXCHANGE-zco | Private hooks remain off and unconstructible by the card; no physics claim changed |
| OVERFLOW-zps | Private hooks remain off and unconstructible by the card; no physics claim changed |
| DINO | Shared WS-RK3-stage risk remains explicit; no numerical-neutrality claim, and the 96--98% regional-cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record native selected-integrator stage entries, closure carries, transports, histories, and next-stage outputs; require paired bitwise tables and red plants; reject missing rows, AT-BAR loss, or earlier first-over-bar |

## Acquisition, review, citations, and tests

No existing record tree contains
`oracle_tracer_transport_kt00000002_s3.bin`. The new user-run acquisition
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round94_stage_closure/run.sh`
clones `GYRE_PISCES` into the new `GYRE_OMIP_L2_P3_SM_R94STGCLS` target,
copies the Round-75 source card file by file, applies only an additive
post-`tra_adv_trp` writer, compiles it with `gfortran -fsyntax-only`, requires
the original restart and mesh to twin, admits only the new record, and ships
red layout/admission plants. Its local preflight passed; log SHA-256
`b6e6506bf2df198d20ed106181aac888f715e4fec01e108d0f749cd4cfe5a381`.
The layout plant exited 69; log SHA-256
`e18e9d206183793aad8ce2035d40279597a6f38af5c942a866ccdb8842e6bb0b`.
No `makenemo` or `mpirun` was run in this round.

The required separate command was attempted with `codex exec --sandbox
read-only`. Independent review is unavailable in-sandbox. Its terminal result,
quoted verbatim, is: **“Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)”**. The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No review verdict is treated as approval; no physics landed.

The GitHub #1455 read/post path is unavailable: the writable clone's only
remote is a local shared path, and `gh` reports “none of the git remotes
configured for this repository point to a known GitHub host.” No ledger-post
claim is made.

The normal citation gate passed all nine compiled-source citations with no
unmapped citation or global map failure; artifact SHA-256
`02eba7b5b32914afcaf4ceed4482fed434d20a0fc06268d988c3bdca30bcd412`.
Its shifted Round-81 history citation plant exited 1 with
`SYMBOL-NOT-AT-LINE`; artifact SHA-256
`af7406098d5343896e7cc83bca70d43948df2eb57973bc7acec175753a6ebe6d`.

The final focused stage, live-observer, phase-3, year-member, year-owner,
citation-gate, and citation-metadata suite passed **99 tests in 55.65 s**; log
SHA-256 `c2cd3ff9a84bb1eae42322e051fe5edc227eca3d543e986edac3c6bcf6ffc42e`.
The known unrelated `test_rk3_ws_differs_from_rk3_and_is_finite` red test was
not selected.

## OPEN for round 95

1. Operator runs the named Round-94 acquisition. Admit its kt=2 stage-3 direct
   transport record, add it to the existing gate, and rerun both tables. Do not
   write another stage harness.
2. Close the model-side kt=1 `tke_en` observation at the actual consumed
   boundary. The public cold-start state has no explicit energy carry while it
   seeds four coefficient carries; expose the stage program's actual TKE
   boundary WRITE-only. Do not silently fill it and do not equate an injected
   pre-step field with a traced consumed field.
3. Resolve the new kt=1 stage-3 discriminator: with recorded TKE context,
   isolated T/S/U/V worsen to order `1e-2`. Determine which still-unbridged
   input or time level produces that change. Until the complete contract
   closes, `first_owned_nonbit` must remain null and no operator or held patch
   is eligible.
4. After all required rows close, take the first owned stage in execution
   order and walk its compiled statements. A landing still requires both its
   stage row to become exact given NEMO entry and the immutable Rule-12 ladder
   to pass.

No configuration or carried-state decision is needed.
