# NEMO testcase L2 GYRE round 147 — developed RHS-family ranking receipt

Date: 2026-09-22  
Incoming lane tip: `8f3f3565d3f7c8f9ab819fd2fdfb79a3ca3210c6`  
Preregistration commit: `8e9d5a37f`  
Headline measurement commit: `f5288a6ec1102d8b05d3969c8df7ff0677293d24`  
Final plant commit: `5afe872375d62019697ce5874b52a897c918aa82`  
Status: **HELD — the developed-state magnitude owner is the LDF family
boundary; no internal LDF statement, year owner, or physics change is claimed**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round147/`.

## Result

The Round-146 record is admitted without rebuilding or rerunning NEMO.  Its
original exit 71 was a false refusal caused by whole-array comparison of
uninitialized halo storage.  Every trajectory-owned value is BIT.  The failed
whole-array result remains at `passive_discrimination.json`; the corrected
admission does not erase it.

From NEMO's exact completed day-180 state, the production-JIT directed ranking
is unambiguous: replacing only the LDF-family contribution reduces the
incoming slow-forcing maximum by **68.9461721587972% U** and
**75.4145976079978% V**.  HPG, VOR, KEG, and ZAD each reduce neither maximum.
The frozen prediction that HPG would own more than half of both gaps is
**REFUTED**.

The first non-bit compiled boundary is therefore LDF's in-place `Krhs`
accumulation, not HPG.  NEMO computes its curl/divergence intermediates and
adds them to U/V at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.
This measurement does **not** distinguish an LDF-operator difference from the
rounding of that in-place addition: each NEMO family contribution is recovered
by subtracting adjacent cumulative snapshots.  No narrower internal statement
is named, and no annual substitution is fabricated from the single day-180
record.  Day-240 sensitivity remains **UNMEASURED**.

## Round-146 admission repair

The complete field decoder first reproduced the operator's refusal.  Only four
full-domain arrays differed:

| field | full unequal; max abs | compiled-owned unequal / 704 | excluded halo unequal / 232 |
|---|---:|---:|---:|
| `CdU_u` | 189; `1.0e-1` | 0 | 189 |
| `CdU_v` | 150; `8.192533274826719e-4` | 0 | 150 |
| `utauU` | 4; `2.240824206237e-312` | 0 | 4 |
| `vtauV` | 4; `2.240824206237e-312` | 0 | 4 |

All geometry, masks, completed 3-D RHS, depth means, post-drag values,
post-wind values, inverse depths, and density reciprocal are BIT over their
full registered extents.  Both step-1080 and step-1081 restarts plus the
process-budget, external-step, QCO, and slow-forcing records are byte-identical.

This ownership is compiled-source fact, not a tolerance: `dyn_drg_init`
declares full-domain `pCdU_u`/`pCdU_v` output arrays, says the method is inner
domain only, and assigns only `ntsi:ntei,ntsj:ntej` at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/dynspg_ts.f90:1460-1493`.
The caller consumes drag and wind over the same inner loop at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:236-250`.

The corrected admission reports PASS.  The new family record's final ZAD pair
is BIT both to its same-run inherited completed RHS and to Round 140 on all
17,400 U / 17,100 V wet cells.  Record SHA-256 is
`d7d972ee26184698809ac5481383c81fe17a05708b9cdfffc2ddc713322e840a`.
Header, truncation, final-ULP, missing-field, parent-wet-ULP,
parent-dry-ULP, parent-interior-ULP, parent-halo-ULP, and restart-byte plants
all print `STATUS PLANT-FIRED` and exit nonzero.

## Production-JIT family ranking

The compiled producer calls and records HPG, LDF, VOR, KEG, and ZAD in order
at `GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192`.
The gate reconstructs each NEMO family from adjacent cumulative boundaries,
replaces that one model family in the completed 3-D RHS, and carries the result
through the complete production-JIT step.  The returned traced/untraced state
is BIT over 23 leaves and 198,956 cells.

All 24 registered rows are below; every arm retains 580 wet U and 570 wet V
unequal cells.  Values are maximum absolute error against NEMO's incoming or
final slow forcing.

| arm | incoming U | incoming V | final U | final V |
|---|---:|---:|---:|---:|
| ordinary | `4.2854247978022983e-13` | `4.433308633699682e-13` | `4.2854247978022983e-13` | `4.4333086294645174e-13` |
| HPG-directed | `4.2854247978022983e-13` | `4.433308633699682e-13` | `4.2854247978022983e-13` | `4.4333086294645174e-13` |
| LDF-directed | `1.3307884389737387e-13` | `1.089946766874442e-13` | `1.3307884389737387e-13` | `1.089946766874442e-13` |
| VOR-directed | `4.2854247978022983e-13` | `4.433308633699682e-13` | `4.2854247978022983e-13` | `4.4333086294645174e-13` |
| KEG-directed | `4.2854247978022983e-13` | `4.433308633699682e-13` | `4.2854247978022983e-13` | `4.4333086294645174e-13` |
| ZAD-directed | `4.2854247978022983e-13` | `4.433308633699682e-13` | `4.2854247978022983e-13` | `4.4333086294645174e-13` |

The local family rows explain that ordering:

| family | U unequal / 17,400; max | V unequal / 17,100; max |
|---|---:|---:|
| HPG | 0; `0` | 0; `0` |
| LDF | 17,400; `8.227285494452055e-12` | 17,100; `1.261292422179936e-11` |
| VOR | 10,591; `8.470329472543003e-22` | 9,501; `1.6940658945086007e-21` |
| KEG | 17,347; `8.470329472543003e-22` | 17,067; `8.420698635789822e-22` |
| ZAD | 17,377; `8.311097204626546e-22` | 17,094; `8.4203108948776875e-22` |

The LDF-directed rows exactly reproduce the already admitted completed-RHS
directed maxima, closing the family attribution while preserving the known
downstream residual.  The ordinary completed-RHS local gaps are
`8.227285494875571e-12` U and `1.261292422179936e-11` V.

## Frozen predictions and controls

| preregistered claim or falsifier | result | verdict |
|---|---|---|
| refusal is confined to unowned storage | all owned values and inherited trajectory artifacts BIT; four excluded halos move | CONFIRMED |
| any restart/owned RHS/geometry/mask movement refuses the record | none moved | falsifier did not fire |
| final ZAD pair closes to same-run and Round-140 owned RHS | both U/V comparisons BIT | CONFIRMED |
| HPG carries more than half of both downstream maxima | HPG changes neither; LDF reduces 68.946% / 75.415% | REFUTED |
| a narrower LDF statement and day-240 carry can be named from this record | adjacent-boundary subtraction cannot separate operator from accumulator association | UNMEASURED |

The closed 24-row registry omission plant prints
`ROUND147 RHS FAMILY FAMILY-MISSING-ROW STATUS PLANT-FIRED` and exits 1.
The first attempted one-ULP directed-RHS plant survived the isolated depth sum
but was rounded away by the complete fused step; its failed run is retained
and is not called a pass.  The repaired single-word plant changes the consumed
LDF-directed completed RHS by 65,536 ULP at native index `[2,10,0]`, moves
exactly one incoming U face by `6.45597924485387e-20`, prints
`ROUND147 RHS FAMILY FAMILY-INPUT-ULP STATUS PLANT-FIRED`, and exits 1.

The separate read-only Codex review was attempted.  Independent review was
unavailable in-sandbox; its verbatim terminal finding was:
`Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`.  This is not represented as a SHIP verdict.

## Trajectory and scope

No production code, physics, configuration, default, carried state, scheme,
stabilizer, canonical NEMO source, or immutable before arm changed.  Therefore
no Decision-43/45 trajectory is an after arm and the certified headline stays:

| row | unchanged value |
|---|---:|
| kt2 U RMS | `2.7377110452773967e-12` |
| kt2 V RMS | `3.2849219221489645e-12` |
| kt3 T RMS | `8.659373840202989e-7 K` |
| kt3 S RMS | `7.027291104577671e-8` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674023317539e-2 K` |
| day-360 T3D RMS | `1.122357124784366e-2 K` |

DINO, LOCK_EXCHANGE, OVERFLOW, and tanks execute no changed production path.
ORCA2 is `UNMEASURED-WITH-SPEC`; this GYRE diagnostic neither measures nor
changes its LDF route.  No configuration choice was made.

## Evidence and tests

| artifact | SHA-256 |
|---|---|
| `family_ranking.json` | `138c17d884915d75ddbe44be1e6c5f52f0a9e4aae6679191cf09df2de9d3303e` |
| `family_missing_row_plant.json` | `8e763275742c0c8182aef20049d1f78f713c9c3e041bd0df0fc1d00740164d7b` |
| `family_input_ulp_plant.json` | `55705b43b339b163cf85c0b6c1902b4995cfea585fee8cb9ece8c81d7b7bb42b` |
| failed `passive_discrimination.json` | `c08f58d94157b07d9c9b02aeacf308e944d6e894e54ac0847079d85106f2341d` |
| corrected Round-146 validation | `8970d82321f462536973164477a9d1c24b0b3903bafb184f06dad8c1b3583955` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `focused_tests_final.log` | `ee767b58dd0a76c46c49afcbfa88cfa5d29a92609c9c61033463c43bf57417f3` |

The final focused suite reports `57 passed in 2.49s`.  The preregistration,
receipt, and Round-146 amendment citation gates pass with 2, 4, and 1 mapped
citations respectively and no unmapped or failing entries.  Shifting the
active LDF citation by two lines reports FAIL and exits 1.

## OPEN — round 148

Stay on NEMO's day-180 entry and the LDF family only.  Extend the existing
Round-50 literal LDF walk—do not create a second harness—to score the compiled
curl/divergence intermediates and the in-place U/V additions under production
JIT.  Distinguish operator output from accumulator association using
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`;
the adjacent-snapshot subtraction is not an oracle
for the isolated addend.  Only a source-exact statement candidate may then be
run through the ladder, month, and 360-day Decision-43/45 gate.  If existing
restart/static operands cannot close that literal walk, preregister a direct
LDF-intermediate acquisition before writing its run script.

`ACQUISITION_NEEDED`: NONE.  
`DECISION_NEEDED`: NONE.  
`ROUND_STATUS`: HELD.
