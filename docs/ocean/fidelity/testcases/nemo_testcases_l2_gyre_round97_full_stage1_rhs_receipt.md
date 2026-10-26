# NEMO-testcases L2 GYRE round 97 receipt: full stage-one RHS held

Date: 2026-09-14. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict

**HELD; THE SOURCE-EXACT STAGE CANDIDATE FAILED RULE 12 AND PRODUCTION WAS
RESTORED.** The compiled-source statement named in Round 96 is confirmed:
legoESM projected the full momentum source to a depth-mean-free RHS before
stage 1, while NEMO preserves the full `Krhs`. The candidate makes the
kt=1 stage-1 consumed RHS and corrected U/V outputs BIT given NEMO's entry,
but the canonical trajectory comparison is FAIL: 85 of 954 registered fields
move and 56 rows contain a cell beyond the two-ULP oracle-relative movement
bar. No row changes class and first-over-bar remains kt2 U/V.

The measured candidate stamp is `d0dce11ba0729418039233d05fe0bc14624942b2`;
the identical-tree commit in the delivered lineage is
`3bca78d0270c2795908f273cb6ab8954889659c5`. Production was restored at
`12018185fb9e47fae2b13c6ef3f26996fd6780bc` and
the candidate is retained only in
`scripts/validate/ocean_fidelity/testcases/manifests/
nemo_testcase_l2_gyre_round97_held_full_stage1_rhs.patch`. No configuration,
carried state, coefficient, timestep, or stabilizer changed.

## Compiled statement and local proof

NEMO accumulates HPG, LDF, VOR, KEG and ZAD into the unchanged `Krhs` slot at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:141-176`. It diagnoses
the depth mean into the separate `Ue_rhs/Ve_rhs` fields at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stp2d.f90:206-207`. The compiled
driver passes that same `Nrhs` through all three stage calls at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:190-215`, and the
selected stage-1 vector assignment consumes the full `Krhs` at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:664-672`.

The existing Decision-41 stage twin, not a new harness, was run at the clean
candidate commit. Artifact `round97/stage_twin.json`, SHA-256
`761635cd990dc5f426f68a3e70a4e756a40ff12c918a5d51f73bbcce0122212c`,
reports both consumed momentum-RHS rows and all eight live RHS-walk rows BIT.
Stage-1 U/V move from 7,620/8,460 unequal cells at
`5.421010862427522e-20` to 0/0. The one-ULP accumulator plant flips exactly
one wet U cell and exits 1; artifact SHA-256 is
`fa5945763e0c6747e49b2d705869b5c4c678844d607f0e190d01779f340d6642`.
The separate commit-stamp plant exits 1 before record consumption with
`commit stamp mismatch`; log SHA-256 is
`a11f327c3479d5f585bdc74b8ff217534e31a0331eabaf6ea38a3bdd392c7398`.

The direct executing-symbol guard was also shown non-vacuous: restoring the
projected RHS makes the guard fail, one test, exit 1. Its log SHA-256 is
`661c21fdad39de5de45cdc2108ab3a319144ab719e7a00b1affd095c3a6ca813`.

## Candidate given-NEMO-entry stage table

`B`, `A`, and `D` mean BIT, AT-BAR, and DEBT. Every cell is
`class/unequal/max`. The cited JSON is the canonical row-per-field table.

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | B/0/0 | Hu B/0/0 ; Hv B/0/0 | — | — | histories all six B/0/0 |
| 1 | 1 | B/0/0 ; B/0/0 | **B/0/0 ; B/0/0** | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/3.5937486e-8 | all five B/0/0 | B/0/0 ; B/0/0 ; A/10232/1.9895197e-13 |
| 1 | 2 | B/0/0 ; B/0/0 | A/4787/1.0842022e-19 ; A/4600/6.7762636e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 1 | 3 | D/8851/1.2830175e-2 ; D/8394/5.1963777e-4 | D/17400/1.0117023e-2 ; D/17100/1.0117027e-2 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; B/0/0 |
| 2 | external | — | uu D/580/5.0398350e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725601e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099464e-4 | — | — | histories ub/ubb/vb/vbb/sshb/sshbb all D |
| 2 | 1 | A/4493/3.5527137e-15 ; A/4436/7.1054274e-15 | D/17400/9.1413673e-6 ; D/17100/9.3268149e-6 | A/273/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/14004/3.6379788e-12 |
| 2 | 2 | A/4551/3.5527137e-15 ; A/4497/7.1054274e-15 | D/17400/7.5775705e-6 ; D/17100/9.6194351e-6 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326762e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/812/3.6379788e-12 |
| 2 | 3 | D/17988/7.9703416e-3 ; D/15488/1.0668959e-3 | D/17400/2.4760020e-3 ; D/17100/5.3033396e-3 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/4.8326760e-7 | all five B/0/0 | B/0/0 ; B/0/0 ; A/364/7.2759576e-12 |

The first remaining owned non-bit output is kt=1 stage-1 W: all 18,000 wet
cells differ, maximum `3.5937485546815465e-8`. No statement inside its
compiled-order W walk is named this round.

## Candidate model-chained stage table

| kt | stage | T / S | U / V | SSH | e3t / e3u / e3v | W | TKE en / avm / avt / dissl / surface | zFu / zFv / zFw |
|---:|---|---|---|---|---|---|---|---|
| 1 | external | — | uu B/0/0 ; vv B/0/0 | A/600/4.3368087e-19 | Hu B/0/0 ; Hv B/0/0 | — | — | histories all six B/0/0 |
| 1 | 1 | B/0/0 ; B/0/0 | **B/0/0 ; B/0/0** | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/3.5937486e-8 | en D/914/5.5367746e-3 ; avm D/5714/8.1761548e-2 ; avt D/5714/8.1761548e-3 ; dissl D/17400/6.3984618e-3 ; surface D/571/8.9940458e-3 | B/0/0 ; B/0/0 ; A/10232/1.9895197e-13 |
| 1 | 2 | B/0/0 ; B/0/0 | A/4787/1.0842022e-19 ; A/4600/6.7762636e-20 | B/0/0 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083153e-7 | same five rows as kt1 stage 1 | **B/0/0 ; B/0/0 ; B/0/0** |
| 1 | 3 | A/6859/1.4210855e-14 ; A/6455/2.1316282e-14 | D/17398/2.7377110e-12 ; D/17100/3.2849221e-12 | A/600/4.3368087e-19 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083159e-7 | same five rows as kt1 stage 1 | A/4750/3.1832315e-12 ; A/4528/2.1600499e-12 ; D/17184/5.2665428e-11 |
| 2 | external | — | uu D/580/5.0398351e-8 ; vv D/570/4.8006350e-8 | D/600/7.0725590e-7 | Hu D/580/1.2029896e-4 ; Hv D/570/1.0099463e-4 | — | — | histories ub/ubb/vb/vbb/sshb/sshbb all D |
| 2 | 1 | D/15470/2.1881164e-11 ; D/11353/1.4921397e-12 | D/17400/9.1389188e-6 ; D/17100/9.3307505e-6 | D/600/2.3575197e-7 | B/0/0 ; B/0/0 ; B/0/0 | D/18000/7.9083201e-7 | en D/979/1.6794735e-3 ; avm D/5723/3.1057322e-2 ; avt D/5723/5.7930292e-2 ; dissl D/17400/4.8874792e-3 ; surface D/571/1.0766770e-6 | D/17400/8.9161073e-1 ; D/17100/7.4853596e-1 ; D/17997/1.4739146e-4 |
| 2 | 2 | D/17993/8.3278006e-7 ; D/17079/6.7607530e-8 | D/17400/7.4535508e-6 ; D/17100/9.6175689e-6 | D/600/3.5362795e-7 | D/18000/1.6484023e-8 ; D/17400/1.5141097e-8 ; D/17100/1.5097498e-8 | D/18000/4.8326762e-7 | same five rows as kt2 stage 1 | D/17400/10.322200 ; D/17100/10.534822 ; D/17993/45.868521 |
| 2 | 3 | D/18000/1.6274972e-4 ; D/17378/6.3277352e-6 | D/17400/1.2295528e-5 ; D/17100/2.3465017e-5 | D/600/7.0725590e-7 | D/18000/2.4725978e-8 ; D/17400/2.2711674e-8 ; D/17100/2.2646248e-8 | D/18000/4.8326760e-7 | same five rows as kt2 stage 1 | D/17400/8.1093193 ; D/17100/10.850152 ; D/17985/23.680199 |

The paired tables classify exactness as Decision 41 requires. The candidate
fixes its named stage output locally and in the chain. It does not clear the
stage because W remains owned and non-bit; it also cannot land because the
trajectory veto below is red.

## Frozen predictions and Rule 12

The local predictions are **CONFIRMED**: both RHS rows, the eight live-walk
rows and stage-1 U/V are BIT; both plants are red. The headline invariance
prediction is also **CONFIRMED** exactly:

| row | Round-85 before | Round-97 candidate | disposition |
|---|---:|---:|---|
| kt2 T | 1.4210854715202004e-14 | 1.4210854715202004e-14 | AT-BAR retained |
| kt2 S | 2.1316282072803006e-14 | 2.1316282072803006e-14 | AT-BAR retained |
| kt2 U | 2.7377110452773967e-12 | 2.7377110452773967e-12 | first-over-bar unchanged |
| kt2 V | 3.284922138989399e-12 | 3.284922138989399e-12 | first-over-bar unchanged |
| kt3 T | 1.627497246303733e-4 | 1.627497246303733e-4 | unchanged |
| kt3 S | 6.327735185607253e-6 | 6.327735185607253e-6 | unchanged |

The full cellwise table is `round97/ladder_rule12.json`, SHA-256
`1d2bdd640ffcdf5e12b056913ad8d9a972910376736dfc419de28ab8e66c45d8`.
It compares all 954 registered rows, lists every one of the 85 moved rows, and
records 56 violations. There are no classification changes and first-over-bar
is kt2 U/V on both sides. The largest worsening is in an existing causal arm;
the ordinary trajectory still develops cellwise movements over the two-ULP
bar. Rule 12 is therefore **FAIL**, regardless of unchanged maxima.

The fresh 30-day candidate member completed every day. `round97/day_gap.json`,
SHA-256
`2baac87ac5f2eddbd2f7fc490f63eb7325120faeb595296edbfc9e44f9a590ef`,
reports day-30 T RMS `1.239701129497169e-2 K`, an improvement of
`5.351136200815176e-13 K` from the Round-85 before arm
(`1.2397011295506804e-2 K`). The preregistered bitwise-invariance prediction is
therefore **REFUTED**; the receipt does not round this movement away.

| testcase | disposition |
|---|---|
| GYRE stage twin | Named full-RHS statement becomes BIT, but kt1-stage1 W is now the first remaining owned non-bit output |
| GYRE kt=1--10 | Rule-12 FAIL: 85/954 rows moved, 56 cellwise violations, no class change, first-over-bar unchanged |
| GYRE days 1--30 | Fresh candidate member; day-30 T RMS improves by 5.351136200815176e-13 K |
| LOCK_EXCHANGE-zco | Shared-path and constructibility tests pass; no tank-fidelity claim from GYRE operands |
| OVERFLOW-zps | Shared-path and partial-cell constructibility tests pass; no tank-fidelity claim from GYRE operands |
| DINO | Shared WS-RK3 source-selection risk remains explicit; no trajectory-neutrality claim and the 96--98% regional-cancellation warning remains in force |
| ORCA2 | **UNMEASURED-WITH-SPEC:** record full unprojected RHS, separate mean, native stage entries/outputs, histories, transports and closure carries, with red plants |

## Review, citations, tests, and choices

The required separate `codex exec --sandbox read-only` review was attempted on
the candidate diff and Rule-12 table. Independent review is unavailable
in-sandbox. Its exact result is: **“Error: failed to initialize in-process
app-server client: Read-only file system (os error 30)”**. The log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No unavailable review is treated as approval; the trajectory gate already
requires HOLD.

The candidate stage-gate suite passed **15 tests in 2.64 s**. The separate
LOCK_EXCHANGE, OVERFLOW and card-constructibility suite passed **31 tests in
223.60 s**; log SHA-256 is
`942c5da3f6418dc982c34c1a8cbe6c3cd0bf406d8d71ec1e63a18aeba66101ff`.
The final focused stage/live-operand/citation suite passed **33 tests in
4.03 s**; log SHA-256 is
`6368d0bfad500876a4aef013a62dacddeca6d7a32380619c68c0d6abf72cd328`.
The clean citation gate audited all four compiled-source citations and passed
(artifact SHA-256
`00f32a65310b05c8fa7ecacad4b794f2bff64ac9808051e48303455d948709e0`).
Shifting the first citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE` (plant artifact SHA-256
`723b33e006d2a55ac77b4d92acc0457aebe80e81e5af02db6538f2b391935096`).

GitHub issue #1455 could not be read or posted: this clone's only remote is the
local read-only source checkout. No issue-update claim is made. Diagnostic log
SHA-256 is
`24091f370d800a5da3e997148e9a5b4d934a0150c13d1c17104563884c1fae27`.

Choices made: none. The user already selected the shared NEMO-identity program,
the stage proof, the immutable Rule-12 comparator, and the carried-state
policies. No unasked choice was made.

UNVERIFIED: DINO trajectory neutrality and ORCA2 stage identity remain
unmeasured for this candidate; no claim is made for either.

## OPEN for round 98

1. Start from restored production and stay in kt=1 stage 1. Preregister the
   compiled-order W walk using the existing stage twin and recorded `ww`
   operands; do not advance to the external step or stage 2.
2. Name the first statement that makes W non-bit given NEMO's stage entry.
   The held full-RHS patch may be re-evaluated only with a same-stage W
   candidate whose local stage proof closes; it may not land alone and may not
   be bundled across stages.
3. Require the combined same-stage result to make every affected stage output
   BIT and pass the canonical 954-row Rule-12 gate. Preserve every failed
   prediction and every moved row.
4. Keep the magnitude targets visible: kt3 T
   `1.627497246303733e-4 K` and day-30 T RMS
   `1.2397011295506804e-2 K`. Do not infer that W owns either until a controlled
   stage candidate and ladder measurement show it.

No acquisition or configuration/carried-state decision is needed.
