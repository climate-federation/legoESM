# NEMO-testcases L2 GYRE round 95 receipt: stage-twin contract complete

Date: 2026-09-14. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**HELD; THE STAGE-TWIN INSTRUMENT LANDED, NO PRODUCTION PHYSICS LANDED.**
The one Round-46/51 stage gate now has no required `UNMEASURED_WITH_SPEC`
row. Its clean artifact is `round95/stage_twin.json`, SHA-256
`01d754256bfbd85e573a46ffb5eeb5b891ac9c85f43394c153c3f1054736e70a`,
at instrument commit `1e9901b4e4166f5f8662c7661b379cceff4ebb4d`. It contains 78 explicit
stage-entry rows and 124 output rows in each of the given-NEMO-entry and
model-chained tables. The 121 given-entry and 118 chained rows shared with
Round 94 reproduce exactly in name, unequal-cell count, maximum absolute
error, and classification.

The gate now names the first owned non-bit output: **kt=1, stage 1, U**, with
7,620 unequal cells and maximum absolute error
`5.421010862427522e-20` (AT-BAR). The larger W output at that same stage is
DEBT in 18,000 cells at `3.5937485546815465e-8`; stage ordering, not magnitude,
therefore makes kt=1 stage 1 the next operator walk. **FIRST OWNED STAGE:
kt=1 stage 1. STATEMENT NAMED THIS ROUND: NONE.** This was the Decision-41
instrument-completion round and its committed preregistration prohibited a
physics candidate; the statement walk is OPEN below.

No production configuration, coefficient, timestep, stabilizer, carried-state
policy, year harness, reconciliation gate, freshwater pair, #1484 guard, held
manifest, NEMO source, NEMO executable, or physical operator changed. The
only production-package change is a private WRITE-only trace of the exact TKE
operand already consumed when the existing live-stage observer is requested.

## Compiled program and admitted boundary

The acquired compiled stage calls `tra_adv_trp` with its live
`Kbb/Kmm/Kaa/Krhs` pointers and then writes the missing kt=2 stage-3
`zFu/zFv/zFw` payload at
`GYRE_OMIP_L2_P3_SM_R94STGCLS/BLD/ppsrc/nemo/stprk3_stg.f90:792-834`.
The header is `(kt,stage,Kbb,Kmm,Kaa,Krhs)=(2,3,3,2,1,1)`. That rotation is
consistent with the compiled stage calls and swaps at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:200-226`; the first
reader attempt's hard-coded kt=1 tuple is retained as **REFUTED** and was
replaced by validation against the corresponding stage-record header.

NEMO computes `zdf_phy` before the external step and stage program at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:168-190`. The model-side
observer now returns the closure path's already-selected TKE seed; it neither
reconstructs a field nor changes that computation. The regular path does not
request or return the extra trace value.

The acquired binary has SHA-256
`607583cfcbf2df2007cc8af5b619f40982b1d740ac84c970c28aacb4c3ba136f`.
The clean consumed-field admission at commit `1e9901b4e416` is PASS:
48/69 inherited records are byte-identical, 21 have only classified changes,
196 differences are admitted in declared halo/undefined regions, the restart
and mesh are byte-identical, and exactly the named record is new. Artifact
`round95/acquisition_admission_clean.json` has SHA-256
`edbc4479e340f2240739a9a9135f03e85f38847aa92d6b3c0d554fbf703240c6`.
Its consumed-cell plant changes one owned value, exits 1 with FAIL, and has
SHA-256 `58e03372ffbec3fb43749922605df135fc25e26527d4bedddf8bd498b7171259`.

## Stage-entry identity

All 30 T/S/U/V/SSH rows and all 30 TKE `en/avm/avt/dissl/surface-avm`
entry rows are BIT. The QCO rows remain within the immutable bar:

| kt | stage | r3t / r3u / r3v (`class/unequal/max`) |
|---:|---:|---|
| 1 | 1 | BIT/0/0 ; BIT/0/0 ; BIT/0/0 |
| 1 | 2 | AT-BAR/600/1.1069635e-16 ; AT-BAR/580/1.1071552e-16 ; AT-BAR/570/1.1094453e-16 |
| 1 | 3 | AT-BAR/600/1.1017615e-16 ; AT-BAR/580/1.1084456e-16 ; AT-BAR/570/1.1097405e-16 |
| 2 | 1 | AT-BAR/600/1.1031574e-16 ; AT-BAR/580/1.1010195e-16 ; AT-BAR/570/1.1078900e-16 |
| 2 | 2 | AT-BAR/600/1.1055103e-16 ; AT-BAR/580/1.1074680e-16 ; AT-BAR/570/1.1064050e-16 |
| 2 | 3 | AT-BAR/600/1.1083024e-16 ; AT-BAR/580/1.1047771e-16 ; AT-BAR/570/1.1036056e-16 |

The one-ULP stage-context plant exits 1 and flips exactly the kt=1 stage-1
`tke_en` entry row: one unequal cell at `3.3881317890172014e-21`. Artifact
`round95/stage_context_ulp_plant.json` has SHA-256
`d6301bfaf7f1f6045327eacc62cac59c7110ae3ae1d97736242a034d36208b1c`.
The separate commit-stamp plant exits 1 before record consumption with
`commit stamp mismatch`.

## Given-NEMO-entry output table

`B`, `A`, and `D` mean BIT, AT-BAR, and DEBT. Every cell is
`class/unequal/max`; the cited JSON is the canonical table with one row per
stage output field.

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

The given-entry TKE rows are BIT while the chained kt=1 TKE rows are DEBT.
That comparison identifies the TKE discrepancy as inherited from upstream
cold-start state, not owned by any RK stage. The stage-1 U discrepancy remains
when the stage is driven from NEMO's own entry, so it is stage-owned.

## Frozen predictions and falsifiers

1. **CONFIRMED:** the clean admission passes with identical restart and mesh,
   exactly one allowed new record, and a red consumed-field plant.
2. **CONFIRMED after one retained refutation:** the binary64 payload is finite
   and all six transport rows are measured. The first reader rejected its
   valid kt=2 rotated pointers because it assumed kt=1 slots; the compiled
   driver and record header falsified that assumption.
3. **CONFIRMED:** the actual traced kt=1 cold-start TKE operand is BIT against
   NEMO at each given-entry stage boundary. The chained values are measured
   DEBT, not hidden by the former `None` sentinel.
4. **CONFIRMED:** there are no missing required rows; the first owned non-bit
   row is kt=1 stage-1 U at the frozen value. No preceding external output is
   DEBT.
5. **CONFIRMED:** all overlapping table rows, the certified trajectory, and
   the day-30 statistic are unchanged.

## Rule 12 and testcase dispositions

The clean ladder artifact `round95/ladder.json`, SHA-256
`649fa40fbb02e307a9e970af63156330c81dd303b9c28b174d0613055c6ded54`,
retains first-over-bar at kt2 U/V:

| row | Round-85 before | Round 95 | disposition |
|---|---:|---:|---|
| kt2 T | 1.4210854715202004e-14 | 1.4210854715202004e-14 | AT-BAR retained |
| kt2 S | 2.1316282072803006e-14 | 2.1316282072803006e-14 | AT-BAR retained |
| kt2 U | 2.7377110452773967e-12 | 2.7377110452773967e-12 | first-over-bar unchanged |
| kt2 V | 3.284922138989399e-12 | 3.284922138989399e-12 | first-over-bar unchanged |
| kt3 T | 1.627497246303733e-4 | 1.627497246303733e-4 | unchanged |
| kt3 S | 6.327735185607253e-6 | 6.327735185607253e-6 | unchanged |

The fresh 30-day member completed all days. `round95/day_gap.json`, SHA-256
`b8e3e2999578b4abbf2bf9fd327481ce6f5610dd88f739c10ca6b35b5fa52380`,
reports day-30 T RMS `1.2397011295506804e-2 K`, exactly unchanged.

| testcase | disposition |
|---|---|
| GYRE kt=1--10 | No physics candidate; every headline is unchanged and first-over-bar remains kt2 U/V |
| GYRE days 1--30 | Fresh member; day-30 T RMS unchanged |
| LOCK_EXCHANGE-zco | Private trace remains off and unconstructible by the card; focused shared-path tests cover the unchanged regular return shape |
| OVERFLOW-zps | Same private-trace disposition; no tank-physics claim |
| DINO | Shared WS-RK3 trace-shape risk remains explicit; no numerical-neutrality claim, and the 96--98% regional-cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record native selected-integrator stage entries, closure carries, transports, histories, and next-stage outputs; require paired bitwise tables and red plants; reject missing rows, AT-BAR loss, or earlier first-over-bar |

## Review, citations, and tests

The required separate `codex exec --sandbox read-only` review was attempted.
Independent review is unavailable in-sandbox. Its result, quoted verbatim, is:
**“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. No review verdict is treated as approval; no physics
landed. The log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The GitHub #1455 read/post path remains unavailable because this clone's only
remote is a local shared path. No ledger-post claim is made; the diagnostic
log SHA-256 is
`c2f4fae9d835564474a18cf359bdaf697286a9fff69676cfd3a3fe457e045a42`.

The final citation gate passes all three compiled-source citations with no
unmapped citation, failed anchor, or global map failure. Its clean artifact
SHA-256 is
`762fb52947a382d700fc3cf298c13726b066ea5f69ec60ba6be29e5b86267ea2`.
The shifted Round-94 transport citation plant exits 1 with
`SYMBOL-NOT-AT-LINE`; its artifact SHA-256 is
`e80da7a270926c1eb12006e8445b6fa26a703aea389065fffa435de5b3f0d3b0`.

The final focused stage, live-observer, phase-3, year-member, year-owner,
time-level, TKE-carry, and citation suite passed **168 tests in 75.79 s**; log
SHA-256 is
`66c7ecb6f8e4382150bb11154a6f6d699d478a73493795615ce2cd2c0ec9513a`.
The known unrelated `test_rk3_ws_differs_from_rk3_and_is_finite` red test was
not selected.

## OPEN for round 96

1. Walk kt=1 stage 1 in the compiled execution order, using the existing
   operator tools and NEMO-recorded stage entry. Name the first statement that
   makes U non-bit; stage-1 W is also owned but occurs later in the output
   ordering and must not bypass U.
2. Preregister a candidate only after that statement is named from the
   compiled branch. A landing must make the given-entry stage row BIT and must
   also pass the immutable Rule-12 ladder; AT-BAR is not bit-exact.
3. Treat the chained cold-start TKE debt as upstream inherited state. Do not
   patch an RK stage to repair it and do not revisit held downstream manifests
   until stage order reaches them.
4. Keep magnitude visible: after the stage-order prerequisite, prioritize the
   stage-owned route that can explain kt3 T (`1.627497246303733e-4 K`) and the
   day-30 T gap (`1.2397011295506804e-2 K`).

No acquisition or configuration/carried-state decision is needed.
