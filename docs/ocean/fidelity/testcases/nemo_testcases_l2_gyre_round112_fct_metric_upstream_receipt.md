# NEMO-testcases L2 GYRE round 112 receipt: metric-FCT upstream program

Date: 2026-09-18

Incoming tip: `a6bb76339dec6a9395cdcad36cea37ac5c2a1b5a`

Preregistration commit: `ba19d6bb480c22b3b030a6a2d3785ea7a13a99ce`

Diagnostic commits: `1bc2523ee96cc8b2f554c0a285555eb99ee3fc85`,
`41bfb49a05fbe1c119c792f5d74bc724c9660d7c`,
`cfb132e0ba135116aa192862764d1c35af9e2451`, and
`b92e80944b9e83774c9a17084f3c231a327a974d`

Candidate commits: `e1964c6a32ffa8673b12f24387c89c4c2a322442`,
`56de4ccc54f8176fef0adf46d97967a95fcd08c7`, and
`f411ab255f6fabeb7bb67aa0647c42a24897d8b3`

Gate/manifest commits: `5a3d4c6d8afe39c944752cba239a04766dec8856`
and `8adb7f2a8bb87fc50e1667ca01df86434b601a60`

Production-revert commit: `6578436dad2c7ee1721c505921d982e7f8b5c35a`

Round status: **HELD — the candidate is source-exact given NEMO's recorded
inputs, but the independent model still enters it with non-bit upstream state
and day-30 T RMS worsens from `6.890484901489568e-5` to
`6.8904868748620001e-5 K`; no physics lands**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round112/`

## Outcome first

The admitted Round-111 record reproduces exactly and all four fail-closed
controls fire.  The untouched compiled-order walk names the first non-bit
boundary as the horizontal first-upwind faces.  Temperature u/v faces differ
in 5,958/5,964 cells, each by at most
`2.3283064365386963e-10` / `4.6566128730773926e-10`; salinity differs in
5,947/5,916 cells at the same maxima.  W is written later and therefore does
not own the first discrepancy.

Transcribing NEMO's metric-bearing transport program closes every recorded
first face, divergence, midpoint, averaged face, final divergence, and direct
RHS row for both tracers.  This holds through the complete production step
under JIT, production eager, and isolated-closure JIT.  The qualifying
source-input plant changes a recorded finite p_u word, changes the source
boundary, prints `STATUS PLANT-FIRED`, and exits 1.

That local result is conditional on NEMO's recorded inputs.  With legoESM's
own chained inputs, the candidate direct `adv_up1` boundary remains unequal in
18,000/18,000 wet cells: T/S maxima are
`6.545655547463206e-11` / `5.6805312195889e-12 s-1`.  Multiplying by the
`14400 s` step gives `9.425743988347016e-7 K` /
`8.179964956208016e-8 psu`, the scale of the kt3 residual.  This is a magnitude
bound, not an additive attribution.

The full trajectory then vetoes the candidate.  Its day-30 T RMS is larger by
`1.9733724325302938e-11 K` (improvement factor
`0.9999997136091443`).  First-over-bar remains kt2 U/V, every kt1 row retains
its classification, and 60 moved rows are registered below, but Decision
43(a) fails.  The three numerical commits were reverted; production packages
and their candidate test are byte-identical to the incoming tip.  The exact
candidate is preserved only as the held manifest
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round112_fct_metric_upstream_held.patch`.

The immutable before arm remains
`phase3/round110/candidate/{ladder.json,day_gap.json}`.  There is no new before
arm.

## Prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round112.md`.

| prediction | disposition | evidence |
|---|---|---|
| P1, admission/calibration | **CONFIRMED** | SHA-256, 34-field schema, 5,849,032-byte EOF, positive-zero entry RHS, `p2dt=14400`, and `exact=45/65 changed=20 admitted=132` reproduce.  Normal admission passes; stamp, truncation, RHS-entry-ULP, and inherited twin plants all exit nonzero. |
| P2, first non-bit boundary | **CONFIRMED** | Both horizontal first-face pairs are non-bit for T and S under production JIT and eager.  No earlier input row is non-bit; W follows the horizontal writes. |
| P3, source candidate | **CONFIRMED locally, conditional on NEMO inputs** | All 20 rows are BIT in production JIT and eager, and the isolated-JIT cross-check is also BIT.  The production plant fires. |
| P4, local magnitude | **REFUTED** | Under the model's own upstream state, direct T/S `adv_up1` remain non-bit in all 18,000 wet cells, and local kt3 T/S are `8.600420500215478e-7` / `6.979443156751586e-8`, exceeding the frozen `6.0e-8` / `7.0e-9` ceilings. |
| P5, Decision-43 trajectory | **REFUTED and landing rejected** | Day-30 T RMS is `6.8904868748620001e-5 K`, above the before value.  The predicted kt3 bounds also fail; kt2 U/V are unchanged. |
| P6, shared-card risk | **PARTLY REFUTED** | Recipe-derived execution says LOCK_EXCHANGE and OVERFLOW do not enter this route because adaptive implicit vertical advection is enabled; both DINO recipes use the Euler tracer lane and do not enter it.  The generic NEMO-GYRE recipe does execute.  It remains explicitly unmeasured because the primary GYRE month veto already makes the candidate unlandable and the numerical patch was reverted. |

## Record admission and controls

The operator-produced record is
`phase3/round111/oracle_fct_writers/oracle_fct_writers_kt00000002_s3.bin`,
SHA-256 `157a5a0ec7cd8607c5c5ee924f788926db5cff5c62e969b853a42f623310505f`,
from commit `a6bb76339dec6a9395cdcad36cea37ac5c2a1b5a`.  Round 112 independently
parsed all 34 self-describing fields to exact EOF and reproduced the twin
admission `exact=45/65 changed=20 admitted=132`.

| control | expected | observed |
|---|---|---|
| normal record gate | pass | `STATUS PASS`, exit 0 |
| producer-stamp plant | reject | `STATUS PLANT-FIRED`, exit 1 |
| truncation plant | reject | `STATUS PLANT-FIRED`, exit 1 |
| RHS-entry-ULP plant | reject | `STATUS PLANT-FIRED`, exit 1 |
| inherited twin admission | pass | `exact=45/65 changed=20 admitted=132`, exit 0 |
| twin mutation plant | reject | `STATUS PLANT-FIRED`, exit 1 |

Evidence is under `admission/`.  No record field was inferred by subtracting
two oracle outputs.

## Compiled source and first non-bit statement

The record's compiled active branch loops over tracers and calls the two-step
FCT upstream routine at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:167-176`.
The stage program constructs the horizontal metric-bearing transports at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:295-296`,
constructs the vertical transport at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:326-346`, and
passes all three to tracer advection at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stprk3_stg.f90:860`.

Within the called routine, the first horizontal statements multiply those
already rounded transports directly by the selected upstream tracer; the
vertical face follows.  These are the first non-bit statements at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:503-533`.
NEMO then differences the faces and forms the midpoint concentration at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:536-549`, forms
the second-step arithmetic-average faces at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:572-608`, and
differences/divides/adds them to the live RHS at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:611-623`.
The downstream limiter and anti-diffusive add are at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/traadv_fct.f90:322-333`; they
were not changed because the source-exact upstream candidate failed the
trajectory criterion first.

The prior implementation divided metric-bearing transports to metric-free
values, multiplied those by tracer, and restored horizontal metrics in a
generic divergence.  Those operations are algebraically equivalent but not
bitwise associated like the compiled statements.  The held candidate routes
the existing metric-bearing values directly and preserves the compiled
`MAX`/`MIN`, product, difference, reciprocal-area, thickness division, mask,
and RHS-add associations.  It adds no selector, coefficient, stabilizer,
state field, or restart payload.

## Source-order walk

Each cell below is `cells unequal / max abs`.  The source-literal column is
ordered production-step JIT / production eager / isolated-closure JIT.

| tracer / compiled boundary | current production JIT | current production eager | source-literal JIT / eager / isolated JIT |
|---|---:|---:|---:|
| T `first_u` | 5958 / `2.3283064365386963e-10` | 5958 / `2.3283064365386963e-10` | 0 / 0 / 0 |
| T `first_v` | 5964 / `4.6566128730773926e-10` | 5964 / `4.6566128730773926e-10` | 0 / 0 / 0 |
| T `first_w` | 5756 / `9.3132257461547852e-10` | 5756 / `9.3132257461547852e-10` | 0 / 0 / 0 |
| T `first_div` | 17475 / `9.994988777600744e-20` | 17476 / `9.994988777600744e-20` | 0 / 0 / 0 |
| T `midpoint` | 3896 / `3.5527136788005009e-15` | 6 / `3.5527136788005009e-15` | 0 / 0 / 0 |
| T `average_u` | 8374 / `2.3283064365386963e-10` | 7888 / `2.3283064365386963e-10` | 0 / 0 / 0 |
| T `average_v` | 8403 / `6.9849193096160889e-10` | 7871 / `4.6566128730773926e-10` | 0 / 0 / 0 |
| T `average_w` | 8283 / `9.3132257461547852e-10` | 7828 / `9.3132257461547852e-10` | 0 / 0 / 0 |
| T `final_div` | 17599 / `1.5924219408380846e-19` | 17579 / `1.4568966692773966e-19` | 0 / 0 / 0 |
| T `rhs_after` | 20749 / `1.4187801866509531e-20` | 20636 / `1.164670302474663e-20` | 0 / 0 / 0 |
| S `first_u` | 5947 / `2.3283064365386963e-10` | 5947 / `2.3283064365386963e-10` | 0 / 0 / 0 |
| S `first_v` | 5916 / `4.6566128730773926e-10` | 5916 / `4.6566128730773926e-10` | 0 / 0 / 0 |
| S `first_w` | 5729 / `1.862645149230957e-9` | 5729 / `1.862645149230957e-9` | 0 / 0 / 0 |
| S `first_div` | 17921 / `1.9439406139486193e-19` | 17923 / `1.9439406139486193e-19` | 0 / 0 / 0 |
| S `midpoint` | 3452 / `7.1054273576010019e-15` | 7 / `7.1054273576010019e-15` | 0 / 0 / 0 |
| S `average_u` | 8485 / `4.6566128730773926e-10` | 7939 / `4.6566128730773926e-10` | 0 / 0 / 0 |
| S `average_v` | 8379 / `9.3132257461547852e-10` | 7873 / `9.3132257461547852e-10` | 0 / 0 / 0 |
| S `average_w` | 8298 / `1.862645149230957e-9` | 7768 / `1.862645149230957e-9` | 0 / 0 / 0 |
| S `final_div` | 17937 / `3.4855405779514459e-19` | 17934 / `2.8857353721769945e-19` | 0 / 0 / 0 |
| S `rhs_after` | 21056 / `2.5693884187122146e-20` | 21047 / `2.5693884187122146e-20` | 0 / 0 / 0 |

The source-literal entries are all `0 / 0`; the compact notation does not
omit maxima.  Production JIT, not isolated JIT, is the qualifying proof.
`fct_walk_candidate_final.json` contains every full statistic.

## Own-state causal check and local forecast

The candidate is then run without replacing the model's upstream state.

| row | cells unequal / cells | max abs |
|---|---:|---:|
| T direct `adv_up1` | 18,000 / 18,000 | `6.545655547463206e-11 s-1` |
| S direct `adv_up1` | 18,000 / 18,000 | `5.6805312195889e-12 s-1` |
| T combined upstream+anti vs NEMO `after_adv` | 18,000 / 18,000 | `5.97774118361013e-11 s-1` |
| S combined upstream+anti vs NEMO `after_adv` | 18,000 / 18,000 | `4.957434813367168e-12 s-1` |
| local kt3 T | 17,999 / 18,000 | `8.600420500215478e-7 K` |
| local kt3 S | 17,229 / 18,000 | `6.979443156751586e-8 psu` |

The candidate split and combined implementations are mutually BIT for all
18,000 cells; reassociating the downstream two-write boundary is not the
missing cure.  The causal conclusion is bounded: at least one recorded live
input to the source-exact upstream writer differs in the independently chained
model.  This receipt does not yet assign that discrepancy to transport,
tracer, thickness/free surface, mask, or reciprocal area.

## Decision-43 headline

| required row | immutable before | candidate | disposition |
|---|---:|---:|---|
| kt2 T max | `1.4210854715202004e-14` | `1.4210854715202004e-14` | AT-BAR retained |
| kt2 S max | `2.1316282072803006e-14` | `2.1316282072803006e-14` | AT-BAR retained |
| kt2 U max | `2.7377110452773967e-12` | `2.7377110452773967e-12` | first-over-bar DEBT unchanged |
| kt2 V max | `3.2849219221489645e-12` | `3.2849219221489645e-12` | first-over-bar DEBT unchanged |
| kt3 T max | `8.600419718618468e-7` | `8.600419683091332e-7` | improves by `3.552713678800501e-15` at the max cell |
| kt3 S max | `6.979441735666114e-8` | `6.979441735666114e-8` | unchanged maximum; field moves |
| day-30 T RMS | `6.890484901489568e-5` | `6.8904868748620001e-5` | **worsens by `1.9733724325302938e-11`; veto** |

The comparison covers all 70 certified ladder rows.  Sixty fields move, no
classification changes, no kt1 field moves, and first-over-bar remains kt2
U/V.  The old no-worsening ULP comparator is intentionally not Decision 43;
its largest residual worsening is `156104.359375` row-scale ULPs.

## Complete moved-row registry

Every nonzero candidate field move is listed; equal displayed maxima can
still contain changed cells.  “Improved/worsened” counts compare each cell's
absolute oracle residual.

| registered row | before max / status | candidate max / status | max candidate field move | improved/worsened cells |
|---|---:|---:|---:|---:|
| `GYRE-zco.kt2.after.uu_b` | `5.0398350609254412e-08` / DEBT | `5.0398350349587992e-08` / DEBT | `5.1380340954088055e-16` | 307/273 |
| `GYRE-zco.kt2.after.vv_b` | `4.8006349783109338e-08` / DEBT | `4.8006350396225667e-08` / DEBT | `6.2189836613768534e-16` | 310/260 |
| `GYRE-zco.kt2.before.S` | `2.1316282072803006e-14` / AT-BAR | `2.1316282072803006e-14` / AT-BAR | `1.4210854715202004e-14` | 3428/681 |
| `GYRE-zco.kt2.before.T` | `1.4210854715202004e-14` / AT-BAR | `1.4210854715202004e-14` / AT-BAR | `1.0658141036401503e-14` | 2500/1065 |
| `GYRE-zco.kt3.after.uu_b` | `1.1804872147933975e-07` / DEBT | `1.1804872188960185e-07` / DEBT | `1.2710102068047568e-15` | 348/232 |
| `GYRE-zco.kt3.after.vv_b` | `7.0736399770065204e-08` / DEBT | `7.0736399105774533e-08` / DEBT | `1.5426028510123757e-15` | 274/296 |
| `GYRE-zco.kt3.before.S` | `6.9794417356661143e-08` / DEBT | `6.9794417356661143e-08` / DEBT | `3.4816594052244909e-13` | 4636/4429 |
| `GYRE-zco.kt3.before.T` | `8.6004197186184683e-07` / DEBT | `8.6004196830913315e-07` / DEBT | `6.7501559897209518e-13` | 8029/7421 |
| `GYRE-zco.kt3.before.ssh` | `7.072558983222451e-07` / DEBT | `7.0725590513526313e-07` / DEBT | `2.2950153062695211e-13` | 348/252 |
| `GYRE-zco.kt3.before.u` | `1.2295527576644538e-05` / DEBT | `1.229552755561969e-05` / DEBT | `4.1699510597986711e-13` | 8578/8822 |
| `GYRE-zco.kt3.before.v` | `2.3465017237225827e-05` / DEBT | `2.3465017254177545e-05` / DEBT | `5.3258786270049541e-13` | 8611/8489 |
| `GYRE-zco.kt4.after.uu_b` | `1.4078950131722265e-07` / DEBT | `1.4078950028592954e-07` / DEBT | `2.8323697554011318e-15` | 263/317 |
| `GYRE-zco.kt4.after.vv_b` | `9.3848314050896074e-08` / DEBT | `9.3848314080061113e-08` / DEBT | `2.7618695091352619e-15` | 260/310 |
| `GYRE-zco.kt4.before.S` | `4.7252278534415382e-07` / DEBT | `4.7251069190679118e-07` / DEBT | `5.2850168685836252e-11` | 6321/6168 |
| `GYRE-zco.kt4.before.T` | `5.8269268166100119e-06` / DEBT | `5.8269267881883025e-06` / DEBT | `3.3313796166112297e-11` | 8300/8550 |
| `GYRE-zco.kt4.before.ssh` | `4.6933513034698796e-07` / DEBT | `4.6933526012295224e-07` / DEBT | `3.2622428386029287e-13` | 309/291 |
| `GYRE-zco.kt4.before.u` | `2.0552956758734808e-05` / DEBT | `2.0552956657704513e-05` / DEBT | `7.0230095610551713e-13` | 8647/8753 |
| `GYRE-zco.kt4.before.v` | `1.7604221222392025e-05` / DEBT | `1.7604221245789975e-05` / DEBT | `7.7110588964438009e-13` | 8372/8728 |
| `GYRE-zco.kt5.after.uu_b` | `1.492776845594436e-07` / DEBT | `1.4927768484133616e-07` / DEBT | `4.4582393332603942e-15` | 316/264 |
| `GYRE-zco.kt5.after.vv_b` | `7.9384012524521402e-08` / DEBT | `7.9384014124586968e-08` / DEBT | `3.7077816945202402e-15` | 278/292 |
| `GYRE-zco.kt5.before.S` | `4.2859174698151037e-07` / DEBT | `4.285917754032198e-07` / DEBT | `9.2512664195965044e-11` | 6971/6743 |
| `GYRE-zco.kt5.before.T` | `7.5037487796691948e-06` / DEBT | `7.5037488080909043e-06` / DEBT | `1.1697309787450649e-10` | 8565/8601 |
| `GYRE-zco.kt5.before.ssh` | `3.688991887561624e-07` / DEBT | `3.6889903967446055e-07` / DEBT | `7.8741700659801239e-13` | 344/256 |
| `GYRE-zco.kt5.before.u` | `3.7099124175329163e-05` / DEBT | `3.7099124223649669e-05` / DEBT | `8.0609130481690272e-13` | 8723/8677 |
| `GYRE-zco.kt5.before.v` | `1.8740728250601912e-05` / DEBT | `1.8740728045057997e-05` / DEBT | `2.3670093662886416e-12` | 8409/8691 |
| `GYRE-zco.kt6.after.uu_b` | `1.2779222405279178e-07` / DEBT | `1.2779222553381195e-07` / DEBT | `7.3207092514751548e-15` | 312/268 |
| `GYRE-zco.kt6.after.vv_b` | `1.0840287174656259e-07` / DEBT | `1.0840287419447425e-07` / DEBT | `4.7457155392949257e-15` | 270/300 |
| `GYRE-zco.kt6.before.S` | `9.2775731985739185e-07` / DEBT | `9.2775843540948699e-07` / DEBT | `1.3211831628723303e-10` | 7510/7104 |
| `GYRE-zco.kt6.before.T` | `1.1337056005089607e-05` / DEBT | `1.1337055934035334e-05` / DEBT | `8.2106765830758377e-11` | 9068/8276 |
| `GYRE-zco.kt6.before.ssh` | `5.5872756314677766e-07` / DEBT | `5.5872744025723189e-07` / DEBT | `1.7070633101523569e-12` | 311/289 |
| `GYRE-zco.kt6.before.u` | `4.1657379143527403e-05` / DEBT | `4.1657378900047254e-05` / DEBT | `4.9127472923071736e-12` | 8660/8740 |
| `GYRE-zco.kt6.before.v` | `3.5678782491713711e-05` / DEBT | `3.5678782260008431e-05` / DEBT | `4.8400623786637453e-12` | 8594/8506 |
| `GYRE-zco.kt7.after.uu_b` | `1.069594848758168e-07` / DEBT | `1.0695948426736254e-07` / DEBT | `9.4867690092481638e-15` | 298/282 |
| `GYRE-zco.kt7.after.vv_b` | `1.4349374571551166e-07` / DEBT | `1.434937451762295e-07` / DEBT | `6.5489876976271288e-15` | 287/283 |
| `GYRE-zco.kt7.before.S` | `7.537604886920235e-07` / DEBT | `7.5376050290287822e-07` / DEBT | `1.7979573385673575e-10` | 7622/7493 |
| `GYRE-zco.kt7.before.T` | `1.1442120872118267e-05` / DEBT | `1.1442120641191877e-05` / DEBT | `1.2172307606306276e-10` | 8893/8570 |
| `GYRE-zco.kt7.before.ssh` | `8.827015061667505e-07` / DEBT | `8.8270251927258236e-07` / DEBT | `2.2304380564719395e-12` | 293/307 |
| `GYRE-zco.kt7.before.u` | `3.0582514300339647e-05` / DEBT | `3.0582514254390292e-05` / DEBT | `1.0218101104825239e-11` | 8772/8628 |
| `GYRE-zco.kt7.before.v` | `2.1825902969774529e-05` / DEBT | `2.1825902335847807e-05` / DEBT | `3.1836101463400546e-11` | 8423/8677 |
| `GYRE-zco.kt8.after.uu_b` | `1.336788745710674e-07` / DEBT | `1.3367887335155679e-07` / DEBT | `8.1913642535624831e-15` | 292/288 |
| `GYRE-zco.kt8.after.vv_b` | `1.4235700907189522e-07` / DEBT | `1.423569963776225e-07` / DEBT | `1.2854734637857135e-14` | 286/284 |
| `GYRE-zco.kt8.before.S` | `1.0231654883341434e-06` / DEBT | `1.0231700642293617e-06` / DEBT | `2.5472246534263832e-10` | 7996/7465 |
| `GYRE-zco.kt8.before.T` | `9.4578708313974857e-06` / DEBT | `9.4578708420556268e-06` / DEBT | `1.5618795146110642e-10` | 9040/8477 |
| `GYRE-zco.kt8.before.ssh` | `6.3984443173653593e-07` / DEBT | `6.3984587805095833e-07` / DEBT | `3.020541282372502e-12` | 272/328 |
| `GYRE-zco.kt8.before.u` | `4.1569950919390086e-05` / DEBT | `4.1569951011691253e-05` / DEBT | `5.4363753082542488e-12` | 8800/8600 |
| `GYRE-zco.kt8.before.v` | `3.5962302026254327e-05` / DEBT | `3.5962301952776644e-05` / DEBT | `2.017697120493267e-11` | 8389/8711 |
| `GYRE-zco.kt9.after.uu_b` | `9.8408898053028993e-08` / DEBT | `9.8408904311477613e-08` / DEBT | `1.0838985958772085e-14` | 308/272 |
| `GYRE-zco.kt9.after.vv_b` | `2.252500010148633e-07` / DEBT | `2.2524999364792637e-07` / DEBT | `1.5831520122633336e-14` | 292/278 |
| `GYRE-zco.kt9.before.S` | `8.6036124713473328e-07` / DEBT | `8.6029974255552588e-07` / DEBT | `3.3950442457353347e-10` | 8206/7613 |
| `GYRE-zco.kt9.before.T` | `7.0207171880554142e-06` / DEBT | `7.0207175433267821e-06` / DEBT | `1.8685497593651235e-10` | 8901/8670 |
| `GYRE-zco.kt9.before.ssh` | `9.0001235190267145e-07` / DEBT | `9.0001226886015761e-07` / DEBT | `4.2887869904778553e-12` | 281/319 |
| `GYRE-zco.kt9.before.u` | `2.9207515902711454e-05` / DEBT | `2.9207515554648296e-05` / DEBT | `5.104337091887956e-12` | 8568/8832 |
| `GYRE-zco.kt9.before.v` | `3.3940538973480962e-05` / DEBT | `3.3940539409310286e-05` / DEBT | `2.220392619767253e-11` | 8651/8449 |
| `GYRE-zco.kt10.after.uu_b` | `8.9948433665797134e-08` / DEBT | `8.9948433386940335e-08` / DEBT | `1.5679772476066833e-14` | 295/285 |
| `GYRE-zco.kt10.after.vv_b` | `1.8475276181931927e-07` / DEBT | `1.847527571351322e-07` / DEBT | `1.2067658070741283e-14` | 296/274 |
| `GYRE-zco.kt10.before.S` | `1.2955953181403856e-06` / DEBT | `1.2956027859445385e-06` / DEBT | `3.7735503610747401e-10` | 7994/8011 |
| `GYRE-zco.kt10.before.T` | `8.9813185475406954e-06` / DEBT | `8.981318650569392e-06` / DEBT | `2.4934365683293436e-10` | 8838/8817 |
| `GYRE-zco.kt10.before.ssh` | `6.6429194487890864e-07` / DEBT | `6.642925442822481e-07` / DEBT | `5.6005963755545451e-12` | 273/327 |
| `GYRE-zco.kt10.before.u` | `2.8633490999089607e-05` / DEBT | `2.8633491165264841e-05` / DEBT | `3.3290342704717091e-11` | 8540/8860 |
| `GYRE-zco.kt10.before.v` | `3.2166251382306603e-05` / DEBT | `3.2166251577959992e-05` / DEBT | `5.5444551727568125e-11` | 8384/8716 |

## Full days 1--30 T-RMS registry

| day | before T RMS (K) | candidate T RMS (K) | candidate minus before (K) |
|---:|---:|---:|---:|
| 1 | `7.5655328496746667e-07` | `7.5655322370345814e-07` | `-6.1264008532475021e-14` |
| 2 | `1.0564417312582731e-06` | `1.0564415346550699e-06` | `-1.9660320315415642e-13` |
| 3 | `2.1881655471547086e-06` | `2.1881655332649543e-06` | `-1.3889754265776792e-14` |
| 4 | `3.5150114727082923e-06` | `3.5150114473683487e-06` | `-2.5339943529178056e-14` |
| 5 | `4.6496077950060062e-06` | `4.6496076181817917e-06` | `-1.7682421451809417e-13` |
| 6 | `5.8531968488219884e-06` | `5.8531986886940516e-06` | `+1.8398720631949386e-12` |
| 7 | `7.2057310251594621e-06` | `7.205732611300762e-06` | `+1.5861412999544112e-12` |
| 8 | `8.7970985282601871e-06` | `8.7970982267504646e-06` | `-3.0150972254376456e-13` |
| 9 | `1.0427834442859521e-05` | `1.0427834510038141e-05` | `+6.7178620115739343e-14` |
| 10 | `1.2161409371181518e-05` | `1.2161409340797906e-05` | `-3.0383612225032101e-14` |
| 11 | `1.4057891994939215e-05` | `1.4057892955290207e-05` | `+9.6035099191287953e-13` |
| 12 | `1.6554351296201494e-05` | `1.6554308576930136e-05` | `-4.2719271357654319e-11` |
| 13 | `1.8127851574058202e-05` | `1.8127852059665934e-05` | `+4.8560773254651725e-13` |
| 14 | `2.0505725616402102e-05` | `2.0505727810578994e-05` | `+2.1941768910962393e-12` |
| 15 | `2.2576092158020803e-05` | `2.2576091674902114e-05` | `-4.8311868896156081e-13` |
| 16 | `2.714004941109638e-05` | `2.7140049290840301e-05` | `-1.202560796710779e-13` |
| 17 | `2.7465358397286287e-05` | `2.746535781678668e-05` | `-5.8049960712370889e-13` |
| 18 | `3.0267641264895023e-05` | `3.0267621906670056e-05` | `-1.935822496699764e-11` |
| 19 | `3.2524792340745243e-05` | `3.2524791280226911e-05` | `-1.060518331681172e-12` |
| 20 | `3.5311048345668514e-05` | `3.5311048021163998e-05` | `-3.2450451559803278e-13` |
| 21 | `3.8092932346151557e-05` | `3.8092931227154525e-05` | `-1.1189970323499492e-12` |
| 22 | `5.1454972869423373e-05` | `5.1454879754713217e-05` | `-9.3114710156250689e-11` |
| 23 | `5.2838673895382613e-04` | `5.2838673867414436e-04` | `-2.7968176819631596e-13` |
| 24 | `4.6474097324106723e-05` | `4.6474095068639168e-05` | `-2.2554675548026523e-12` |
| 25 | `7.1008874223996358e-05` | `7.1008766825218119e-05` | `-1.0739877823878752e-10` |
| 26 | `5.1687887690875987e-05` | `5.1687885671678034e-05` | `-2.0191979531003316e-12` |
| 27 | `5.472033708254828e-05` | `5.4720345752014921e-05` | `+8.6694666406156827e-12` |
| 28 | `5.7241643675798917e-05` | `5.7241641972202458e-05` | `-1.7035964585093537e-12` |
| 29 | `1.0514260873961176e-04` | `1.0514230071883192e-04` | `-3.0802077983496309e-10` |
| 30 | `6.8904849014895676e-05` | `6.8904868748620001e-05` | `+1.9733724325302938e-11` |

## Shared-card execution and blast radius

The execution set is derived from resolved recipes, not a hard-coded card
list.

| resolved card | tracer lane | adaptive implicit | GM/Redi | executes candidate route | measurement disposition |
|---|---|---:|---:|---:|---|
| GYRE-zco | rk3_ws / fct2 | false | configured | yes | full ladder and month measured |
| NEMO-GYRE-recipe | rk3_ws / fct2 | false | configured | yes | **UNMEASURED**; candidate already vetoed and reverted |
| LOCK_EXCHANGE-zco | rk3_ws / fct2 | true | absent | no | shown not to execute |
| OVERFLOW-zps | rk3_ws / fct2 | true | absent | no | shown not to execute |
| DINO:nemo_dino_kamm | Euler / fct2 | false | configured | no | shown not to execute |
| DINO:nemo_dino_kamm_mlf | Euler tracer, leapfrog outer / fct2 | false | configured | no | shown not to execute |

The fail-closed route gate reports `FAIL`, not `PASS`: day-30 does not
decrease and the generic executing recipe lacks a certified numerical
before/after measurement.  No waiver is requested because no numerical code
lands.  If this held route is reconsidered, that generic card must first get
its own certified before/after gate.

ORCA2 remains **UNMEASURED-WITH-SPEC**: resolve its tracer integrator and FCT
dispatch, record both FCT writes and every base tracer, u/v/w transport,
Kbb/Kmm free-surface/thickness, mask, and reciprocal-area operand at kt1--10,
replay the production fp64 JIT closure, and score the next consumed T/S state.

## Independent review

PENDING.

## Citation control and tests

PENDING.

## ASKED / UNASKED

ASKED and completed: independent admission and all plants; compiled-order
production JIT/eager/isolated walk; first non-bit statement; literal
compiled-source candidate; source-input production plant; own-state causal
check; full kt1--10 ladder; full days 1--30 score; complete moved-row
registry; recipe-derived blast-radius gate; held manifest; and production
revert.

UNASKED and not done: NEMO source was not modified; `makenemo` and `mpirun`
were not run; no oracle output entered the independent implementation; no
physics, configuration, default, coefficient, stabilizer, carried state, or
restart schema landed; and the year harness, reconciliation gate, freshwater
pair, and #1484 guard were not touched.

## OPEN for round 113

1. Stay on the magnitude-owning stage-3 FCT family, but move upstream of the
   source-exact writer.  Extend the existing production capture to score the
   model's actual live `base_T/S`, `p_u`, `p_v`, `p_w`, Kbb/Kmm free-surface
   thickness ratios, masks, and reciprocal area against the already admitted
   Round-111 record.  No new NEMO acquisition is needed: every field is in
   that record.
2. Name the first non-bit live input in producer/execution order and perform a
   one-family-at-a-time substitution through the complete production JIT
   step.  The discriminating measurement is whether that substitution closes
   the direct `adv_up1` boundary and materially reduces kt3 T.  Do not infer
   ownership from the `14400 s` magnitude bound alone.
3. Do not jump downstream to the anti-diffusive limiter.  The low-order writer
   is already bit-exact for NEMO inputs and remains non-bit only with the
   model's independently computed inputs.
4. Preserve `phase3/round110/candidate/{ladder.json,day_gap.json}` as the
   immutable before arm.  A future candidate must pass Decision 43 and, if it
   reaches a landing attempt, measure every recipe-derived executing card,
   including `build_nemo_gyre_recipe()`.
5. Keep ORCA2 UNMEASURED until the explicit operand-and-consumer spec above is
   executed.  The GYRE residual is not zero and no identity claim is made.
