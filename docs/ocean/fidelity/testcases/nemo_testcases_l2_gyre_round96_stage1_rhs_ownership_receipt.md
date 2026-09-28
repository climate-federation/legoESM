# NEMO-testcases L2 GYRE round 96 receipt: stage-one RHS ownership corrected

Date: 2026-09-14. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**HELD; THE STAGE CONTRACT WAS CORRECTED, NO PRODUCTION PHYSICS LANDED.**
Round 95's claim that kt=1 stage-1 U was stage-owned is **RETRACTED**. The
stage table omitted the momentum RHS consumed by the stage. Once scored, that
incoming RHS is already DEBT: U has 17,400 unequal wet cells, V has 17,100,
and both have maximum absolute error `2.0121494123449567e-8 m s-2`.

**FIRST NON-BIT STATEMENT:** legoESM hands the kt=1 stage-1 vector assignment
the depth-mean-removed perturbation RHS, whereas NEMO hands it the full
`Krhs`. NEMO forms the full source accumulator in HPG, LDF, VOR, KEG, ZAD
order at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-176`, computes its
vertical mean into the separate `Ue_rhs/Ve_rhs` arrays at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:206-207`, and passes the
unchanged `Nrhs` slot from `stp_2D` into all three stage calls at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215`. The vector
assignment reads that full `Krhs` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:664-672`.

This statement was named only after the frozen measurement. The Round-96
preregistration says that a non-bit incoming RHS revokes stage ownership and
makes a production candidate ineligible, so the NEMO-full-RHS candidate is
OPEN for Round 97. There is no configuration choice and no stabilizer.

## Instrument and local walk

The existing Round-46/51 stage twin was extended in place; no second harness
was created. It now publishes two consumed-RHS rows for every kt/stage and a
WRITE-only four-boundary stage-1 RHS trace. Its clean final artifact is
`round96/stage_twin_final.json` (SHA-256
`108ad29d08448284ba9a640e08d7e4c2402a65d45e1d642ce7e47ea2c71e31f9`) at fail-closed instrument
commit `5cc8d5c771face64e95ece4898939d178631cdf1`.

The isolated production-JIT operator walk, driven by NEMO inputs and rebuilt
in NEMO accumulator order, is BIT at every boundary:

| boundary | U unequal/max | V unequal/max |
|---|---:|---:|
| post HPG | 0 / 0 | 0 / 0 |
| post LDF | 0 / 0 | 0 / 0 |
| post VOR | 0 / 0 | 0 / 0 |
| post ADV (KEG+ZAD) | 0 / 0 | 0 / 0 |

The ordinary live stage path is already non-bit at its first captured RHS
boundary, and none of the later reassociations changes the residual:

| live boundary | U class/unequal/max | V class/unequal/max |
|---|---|---|
| operator accumulator | D/17400/2.0121494123449567e-8 | D/17100/2.0121494123449567e-8 |
| transport reconciliation | D/17400/2.0121494123449567e-8 | D/17100/2.0121494123449567e-8 |
| ZAD reassociation | D/17400/2.0121494123449567e-8 | D/17100/2.0121494123449567e-8 |
| RK-assignment input | D/17400/2.0121494123449567e-8 | D/17100/2.0121494123449567e-8 |

This comparison refutes the earlier hypothesis that one of the five compiled
operators owns the stage output. The owner is the model's projection of the
otherwise exact source accumulator before the stage assignment.

## Stage-entry identity

All T/S/U/V/SSH, TKE closure, and QCO rows from Round 95 reproduce. The newly
required consumed momentum-RHS rows are:

| kt | stage | U class/unequal/max | V class/unequal/max |
|---:|---:|---|---|
| 1 | 1 | D/17400/2.0121494123449567e-8 | D/17100/2.0121494123449567e-8 |
| 1 | 2 | A/4356/3.3087224502121107e-24 | A/4367/3.3087224502121107e-24 |
| 1 | 3 | A/17376/2.0538452096498295e-16 | A/17099/2.4672270330662615e-16 |
| 2 | 1 | D/17400/2.3760625721748291e-8 | D/17100/4.8052267589134306e-8 |
| 2 | 2 | D/17399/1.0575028310679399e-9 | D/17099/1.3433964297128585e-9 |
| 2 | 3 | D/17400/1.0182843857317243e-9 | D/17100/1.4467995550892100e-9 |

The preregistered claim that a one-ULP perturbation could *flip* the already
DEBT kt1-stage1 RHS row is **REFUTED**: a red row cannot flip from BIT. That
failed premise is retained here. The replacement non-vacuous control changes
one U cell at the preceding exact post-HPG accumulator boundary, is detected
as exactly one unequal cell, and exits nonzero. Artifact
`round96/stage_rhs_ulp_plant.json` has SHA-256
`66b8e0b42f75e12c3c3baedcb52c4e8046c1394bdfe7e7c580f02e457df36e0b`. The commit-stamp
plant exits nonzero before record consumption with `commit stamp mismatch`.

## Given-NEMO-entry output table

`B`, `A`, and `D` mean BIT, AT-BAR, and DEBT. Every cell is
`class/unequal/max`; the clean JSON is the canonical row-per-field table.
These 124 rows reproduce Round 95 exactly.

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | B/0/0 | Hu B/0/0 ; Hv B/0/0 | — | — | histories ub/ubb/vb/vbb/sshb/sshbb all B/0/0 |
| 1 | 1 | B/0/0 ; B/0/0 | A/7620/5.4210109e-20 ; A/8460/5.4210109e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/3.5937486e-8 | all five B/0/0 | B/0/0 ; B/0/0 ; A/10232/1.9895197e-13 |
| 1 | 2 | B/0/0 ; B/0/0 | A/4787/1.0842022e-19 ; A/4600/6.7762636e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 1 | 3 | D/8851/1.2830175e-2 ; D/8394/5.1963777e-4 | D/17400/1.0117023e-2 ; D/17100/1.0117027e-2 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 2 | external | — | uu D/580/5.0398350e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725601e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099464e-4 | — | — | ub D/580/4.9213669e-8 ; ubb D/580/4.8141663e-8 ; vb D/570/4.6857490e-8 ; vbb D/570/4.5674083e-8 ; sshb D/600/6.8902330e-7 ; sshbb D/600/6.7000577e-7 |
| 2 | 1 | A/4493/3.5527137e-15 ; A/4436/7.1054274e-15 | D/17400/9.1413673e-6 ; D/17100/9.3268149e-6 | A/273/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/14004/3.6379788e-12 |
| 2 | 2 | A/4551/3.5527137e-15 ; A/4497/7.1054274e-15 | D/17400/7.5775705e-6 ; D/17100/9.6194351e-6 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326762e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/812/3.6379788e-12 |
| 2 | 3 | D/17988/7.9703416e-3 ; D/15488/1.0668959e-3 | D/17400/2.4760020e-3 ; D/17100/5.3033396e-3 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326760e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/364/7.2759576e-12 |

## Model-chained output table

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | A/600/4.3368087e-19 | Hu B/0/0 ; Hv B/0/0 | — | — | all six histories B/0/0 |
| 1 | 1 | B/0/0 ; B/0/0 | A/7620/5.4210109e-20 ; A/8460/5.4210109e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/3.5937486e-8 | en D/914/5.5367746e-3 ; avm D/5714/8.1761548e-2 ; avt D/5714/8.1761548e-3 ; dissl D/17400/6.3984618e-3 ; surface D/571/8.9940458e-3 | B/0/0 ; B/0/0 ; A/10232/1.9895197e-13 |
| 1 | 2 | A/2/3.5527137e-15 ; B/0/0 | A/6337/1.0842022e-19 ; A/5944/8.1315163e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | same five rows as kt1 stage 1 | A/7540/1.8189894e-12 ; A/8315/1.8189894e-12 ; D/15050/4.1836756e-11 |
| 1 | 3 | A/6858/1.4210855e-14 ; A/6457/2.1316282e-14 | D/17398/2.7377110e-12 ; D/17100/3.2849221e-12 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | same five rows as kt1 stage 1 | A/6296/3.1832315e-12 ; A/5860/2.1600499e-12 ; D/17446/5.1116444e-11 |
| 2 | external | — | uu D/580/5.0398351e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725590e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099463e-4 | — | — | ub D/580/4.9213669e-8 ; ubb D/580/4.8141663e-8 ; vb D/570/4.6857490e-8 ; vbb D/570/4.5674082e-8 ; sshb D/600/6.8902319e-7 ; sshbb D/600/6.7000566e-7 |
| 2 | 1 | D/15470/2.1881164e-11 ; D/11354/1.4992452e-12 | D/17400/9.1389188e-6 ; D/17100/9.3307505e-6 | D/600/2.3575197e-7 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | en D/979/1.6794735e-3 ; avm D/5723/3.1057322e-2 ; avt D/5723/5.7930292e-2 ; dissl D/17400/4.8874792e-3 ; surface D/571/1.0766770e-6 | D/17400/8.9161073e-1 ; D/17100/7.4853596e-1 ; D/17994/1.4739146e-4 |
| 2 | 2 | D/17993/8.3278006e-7 ; D/17079/6.7607530e-8 | D/17400/7.4535508e-6 ; D/17100/9.6175689e-6 | D/600/3.5362795e-7 | D/18000/1.6484023e-8 ; D/17400/1.5141097e-8 ; D/17100/1.5097498e-8 | D/18000/4.8326762e-7 | same five rows as kt2 stage 1 | D/17400/1.0322200e1 ; D/17100/1.0534822e1 ; D/17996/4.5868521e1 |
| 2 | 3 | D/18000/1.6274972e-4 ; D/17378/6.3277352e-6 | D/17400/1.2295528e-5 ; D/17100/2.3465017e-5 | D/600/7.0725590e-7 | D/18000/2.4725978e-8 ; D/17400/2.2711674e-8 ; D/17100/2.2646248e-8 | D/18000/4.8326760e-7 | same five rows as kt2 stage 1 | D/17400/8.1093193 ; D/17100/10.850152 ; D/17994/23.680199 |

The output tables are unchanged because the observer does not change the
stage. Their interpretation changed: kt1 stage-1 U/V cannot be called owned
while their consumed RHS is already non-bit.

## Frozen predictions and falsifiers

1. **CONFIRMED:** the kt1 stage-1 incoming RHS is non-bit and inherited from
   the pre-assignment `stp_2D` program.
2. **NOT REACHED:** the assignment and correction predictions were conditional
   on a BIT incoming RHS; the frozen stop rule prevents using derived replay
   arithmetic as ownership evidence.
3. **CONFIRMED:** the private observer is WRITE-only; every Round-95 overlap
   and all trajectory headlines reproduce.
4. **CONFIRMED:** no production candidate is eligible in this round.
5. **REFUTED:** the one-ULP plant could not flip an already-red RHS row; the
   replacement exact-boundary plant is red and non-vacuous.

## Rule 12 and testcase dispositions

The full registered-row comparison against the Round-85 after arm is
`round96/ladder_full_rule12.json` (SHA-256
`b77063030740c63e61bee3fea250403929951a9b2ae7408b9dea931064adcd2a`), with verdict
**PASS** across all 954 registered rows: zero worsened cells, zero status
changes, zero violations, and maximum worsening zero ulps. No row moved.

| row | Round-85 before | Round 96 | disposition |
|---|---:|---:|---|
| kt2 T | 1.4210854715202004e-14 | 1.4210854715202004e-14 | AT-BAR retained |
| kt2 S | 2.1316282072803006e-14 | 2.1316282072803006e-14 | AT-BAR retained |
| kt2 U | 2.7377110452773967e-12 | 2.7377110452773967e-12 | first-over-bar unchanged |
| kt2 V | 3.284922138989399e-12 | 3.284922138989399e-12 | first-over-bar unchanged |
| kt3 T | 1.627497246303733e-4 | 1.627497246303733e-4 | unchanged |
| kt3 S | 6.327735185607253e-6 | 6.327735185607253e-6 | unchanged |

The fresh 30-day member completed every day. `round96/day_gap.json` has
SHA-256 `0c054d10ed75561c5a64dec734f24c9beaa10ecefd282b0d8a2e6f90b6295df7`
and day-30 T RMS `1.2397011295506804e-2 K`, exactly
unchanged.

| testcase | disposition |
|---|---|
| GYRE stage twin | Ownership corrected; the first non-bit consumed statement is full-RHS versus perturbation-RHS selection before kt1 stage 1 |
| GYRE kt=1--10 | No candidate; all headlines unchanged and first-over-bar remains kt2 U/V |
| GYRE days 1--30 | Fresh member; day-30 T RMS unchanged |
| LOCK_EXCHANGE-zco | Observer-only code is private and off; focused regular-path tests pass |
| OVERFLOW-zps | Same private-observer disposition; no tank-physics claim |
| DINO | The shared WS-RK3 stage-selection risk is explicit; no numerical-neutrality or regional-cancellation claim |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record the selected integrator's unprojected momentum RHS, separate vertical mean, native stage entries/outputs, histories, transports and closure carries; require paired bitwise tables and red plants |

## Review, citations, and tests

The required separate `codex exec --sandbox read-only` pass was attempted.
Independent review is unavailable in-sandbox. Its verdict is quoted verbatim:
**“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. No review result is treated as approval and no physics
landed. The log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The citation gate passes all four compiled-source citations with no unmapped
citation, failed anchor, or global map failure. Artifact SHA-256 is
`2c51fb23a1d9876b5845b97eabaf8130a982faa60b0cded88fe261730e004ff2`.
The shifted `stp2d` citation plant exits nonzero with
`SYMBOL-NOT-AT-LINE`; its artifact SHA-256 is
`91dac00e3ebe77092ce9f966f34e6094adb3ceb98653efdd9c670585f0742f95`.

The focused stage/live-observer/citation suite passes **33 tests in 4.19 s**;
log SHA-256 is
`f8b8cb6aed92df917fbd060fb492dcdb1155156934489783fda8792cdde9f666`.
The separate LOCK_EXCHANGE, OVERFLOW and card-constructibility suite passes
**31 tests in 239.30 s**. The observer is private and unrequested on those
paths, so they do not execute the added return payload. The known unrelated
`test_rk3_ws_differs_from_rk3_and_is_finite` red test was not selected.

## OPEN for round 97

1. Preregister the kt1-stage1 full-RHS candidate. Change the shared
   NEMO-identity stage-1 assignment input from the projected perturbation RHS
   to NEMO's full `Krhs`; do not change the separate slow forcing or the
   barotropic target.
2. Require the kt1-stage1 consumed-RHS row and U/V output rows to become BIT
   given NEMO entry. Then require the full Rule-12 ladder; hold the candidate
   if any AT-BAR row leaves the bar or first-over-bar moves earlier.
3. If the stage output does not become BIT, extend the existing stage gate
   with direct post-update and post-correction comparisons; do not infer a
   narrower owner from algebraic replay.
4. Only after kt1 stage 1 is exact may the walk advance to the external step
   and later stages. Keep kt3 T (`1.627497246303733e-4 K`) and day-30 T RMS
   (`1.2397011295506804e-2 K`) visible as the magnitude targets.

No acquisition or user configuration/carried-state decision is needed.
