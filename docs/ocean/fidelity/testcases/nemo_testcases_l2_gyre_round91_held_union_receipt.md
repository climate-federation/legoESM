# NEMO testcase L2 GYRE Round 91 receipt — held-patch union discriminator

**Date:** 2026-09-14  
**Incoming tip:** `ebcee36f321fcd3296e6fb0968adde6bddc38a7d`  
**Frozen preregistration:** `b21bab8f10017df5bd3951d0cff1cd9883ec99d1`  
**Measured candidate:** `39d346fca638c2f8861e3b8952d8266af24bf333`  
**Production-restoration commit:** `28dbd045da1a6dc58b8a930be241d493658fe768`  
**Evidence:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round91/`

## Outcome

**HELD.  Nothing in the shared implementation lands.**  The operator-requested
union of every held candidate that retained local source exactness fails both
forms of Rule 12.  Relative to the immutable Round-85 arm, its first-over-bar
set moves backward from kt2 U/V to kt2 T/S/U/V, its kt2 errors become
T `3.0652394795183113e-3`, S `4.835710022078388e-3`, U
`6.733005735178965e-7`, V `1.3183569256688065e-6`, and its day-30 T RMS grows
from `1.2397011295506804e-2 K` to `1.2589964172802363e-2 K` (+`1.929528772954e-4
K`, +1.556%).  The candidate's production and test paths were then restored
byte-for-byte to the preregistration parent; only this preregistration, this
receipt, and citation-gate maintenance remain.

The numerical part of the frozen prediction is confirmed: the six headline
ladder values equal Round 89's preregistered bundle values.  The admission
premise is **REFUTED**, however: Round 62's tracer coefficient/content member
is not locally exact at this tip and was excluded before the union trajectory.
There is no post-hoc repair.

## Frozen arm and union inventory

The immutable comparator is the Round-85 after arm.  It already contains the
Round-47 stage-ZAD, Round-79 history pair, and Round-82 drag-boundary changes;
applying those patches again would double-apply landed code.  I re-ran their
local proofs on the assembled candidate instead.  The remaining manifest
inventory was exhaustive:

| member | current-tip local result | union action |
|---|---|---|
| Round 47 stage ZAD | bit exact, U 0/17,400 and V 0/17,100 unequal | already in Round 85; re-proved |
| Round 79 histories pair | `CONFIRMED`; history plant fired with exit 1 | already in Round 85; re-proved |
| Round 82 drag boundary | `CONFIRMED`; coefficient/depth/transport rows exact; plant exit 1 | already in Round 85; re-proved |
| Round 60 TKE association | all six promoted operator rows exact; plant exit 1 | applied |
| Round 62 tracer coefficient | **DEBT**: assembly `zwd` 3,120/21,120 unequal, max `2.997100176778e2` | excluded as preregistered falsifier |
| Round 88 Kaa/W/WZV/ZAD | five rows bit exact, 0 unequal; four independent plants exit 1 | supplied by Round-89 superset |
| Round 89 assignment boundary | all six source-assignment rows bit exact, 0 unequal; plant exit 1 | applied as strict Round-88 superset |
| Round 51 deviation history | explicitly NOT_APPLIED and obsolete after Decision 37 | excluded |

Candidate assembly is reproducible from commit `530f32c51ad0ddd3023298b9de14937e3ec5b130`;
commit `39d346fca638c2f8861e3b8952d8266af24bf333` removes the refuted Round-62
member before every reported union trajectory.  Every local-proof and
trajectory artifact carries the clean `39d346f...` worktree stamp except the
deliberately failed Round-62 artifact, which carries the clean `530f32c...`
stamp at which it was tested.

## Source audit and first non-bit statement

The history member implements the compiled AB3-AM4 midpoint construction and
the six absolute history swaps at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-505` and
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:783-795`.
The drag member implements the compiled bottom-speed coefficient
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/zdfdrg.f90:229-235`, the Kmm/Kbb
face depths
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:363-375`, and the
explicit non-WAD drag update
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:703-706`.

The TKE member is the source-ordered compiled matrix/RHS and Prandtl block at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:394-433`, followed by the
compiled mixing-length, viscosity, diffusivity, and Prandtl application at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:589-701`.  With NEMO's
own recorded inputs, `matrix_rhs_sweep_en_floor`, surface handling, raw mixing
length, Prandtl factor, `avt` derivation, and pre-EVD copy are each 0/17,400
unequal.  Its first carried input discrepancy remains buoyancy/dissipation,
744/17,400 unequal at `1.3877787807814457e-17`; the source operator itself is
not claimed to own that upstream discrepancy.

The W/ZAD member implements compiled quasi-Eulerian W at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/sshwzv.f90:293-300` and compiled
ZAD at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/dynzad.f90:102-138`; both are bit
exact on NEMO inputs.  The carried Kaa scratch is later extrapolated by the
compiled stage program at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3.f90:220-226`.

The assignment member reproduces the compiled stage-1/2 vector assignment at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:671-674` bit for bit.
The **first live non-bit statement in this walk** is immediately after the
compiled barotropic correction at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:728-759`: stage-1
post-baro U is 17,400/17,400 unequal with max `1.2184073888699132e-10`, and V
is 17,100/17,100 unequal with max `1.6794210466741788e-10`.  That is the next
magnitude-ranked source boundary; no downstream statement is promoted over it.

Round 62 was re-read against the compiled implicit tracer assembly and three
recurrences at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:461-477`,
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:523-528`,
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:545-564`, and
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/trazdf.f90:573-578`.  The current-tip
replay is non-bit at assembly, so this member fails the user's calibration-grade
local-proof prerequisite even before Rule 12.

## Certified GYRE ladder

| row | Round-85 before | union | result |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | `3.0652394795183113e-3` | AT-BAR -> DEBT |
| kt2 S | `2.1316282072803006e-14` | `4.835710022078388e-3` | AT-BAR -> DEBT |
| kt2 U | `2.7377110452773967e-12` | `6.733005735178965e-7` | worsened |
| kt2 V | `3.284922138989399e-12` | `1.3183569256688065e-6` | worsened |
| kt2 SSH | `0.0` | `0.0` | remains AT-BAR |
| kt3 T | `1.627497246303733e-4` | `1.566544749833554e-2` | worsened 96.25x |
| kt3 S | `6.327735185607253e-6` | `4.324733116938262e-3` | worsened 683.44x |
| kt3 U | `1.2295527572661613e-5` | `9.721854630545028e-3` | worsened |
| kt3 V | `2.3465017240615477e-5` | `1.5778692679240987e-2` | worsened |
| kt3 SSH | `7.072558982001639e-7` | `2.211781713743041e-5` | worsened |

The canonical comparator fails with 954 rows examined, 88 per-cell violations,
largest worsening `7.1009755143897438e13` row-scale oracle ULP, and
first-over-bar `{U,V}@kt2 -> {T,S,U,V}@kt2`.  The Decision-38 registered-row
gate also fails: 86 rows moved, four AT-BAR-to-DEBT status crossings, and none
of the kt2 U/V target rows moved toward the bar.  Artifacts are
`union_kt1_10.json` (SHA-256 `b1b97702...`), `union_rule12_complete.json`
(`e560fbb4...`), and `union_rule12.json` (`8137f52f...`).

## Days 1-30

The required run continued after the ladder falsifier.  Day-30 values are:

| field | union RMS vs NEMO |
|---|---:|
| T | `1.2589964172802363e-2 K` |
| S | `4.618881966334352e-3` |
| U | `5.808072959863522e-4 m s-1` |
| V | `7.911512205555596e-4 m s-1` |
| SSH | `5.19481183388901e-4 m` |

The T headline worsens from Round 85 by `1.92952877295559e-4 K` (1.556%).
The 30-row scorer artifact is `union_day_gap.json` (SHA-256 `04698590...`) and
is stamped clean at the measured candidate.

## Tanks and cross-card risk

LOCK_EXCHANGE-zco and OVERFLOW-zps each compare 50/50 certified rows
bit-identically to their Round-85 arms: zero moved rows, zero maximum worsening,
and unchanged first-over-bar (`kt4 U` for LOCK, `kt2 T/U` for OVERFLOW).
Their absolute gates retain pre-existing DEBT; the union introduces none.

DINO is **not claimed neutral**.  The TKE and W/ZAD helpers are shared
implementation, so a source-proof landing would carry shared-statement risk;
this discriminator did not run a DINO numerical arm.  The campaign synthesis
also warns that operator-exact changes can be climate-inert or can expose
cancelling errors, so GYRE's failed union cannot be promoted on source proof
alone.

ORCA2 remains **UNMEASURED** for this union.  Before any landing it requires a
clean fp64 ORCA2 identity member using the same restart/card and compiled NEMO
5.0.2 branch, scored at kt=1..10 with the same five-field row schema, an
explicit first-over-bar comparison against its current baseline, and a plant
that perturbs a nonzero active row.  No ORCA2 verdict is inferred from tanks.

## Round-90 record status

The operator's acquisition reached `STOP 0`; NEMO compiled and ran.  Admission
then failed in the legoESM reader, not in NEMO: the record declares global
`36x26x31` dimensions and owned bounds `3:34,3:24`, while the writer correctly
emitted the owned rank-2 `baro_zub` as `32x22`.  The Round-90 reader demands
`36x26` for every rank-2 payload and rejected `(2,32,22,1)`.  The existing
record can therefore be admitted by a rank-aware, owned-extent reader fix next
round; no new NEMO acquisition is needed.

## Controls and tests

- Every local-proof plant exited nonzero: histories, drag, TKE, Kaa scratch,
  direct W, ZAD U, and assignment boundary.
- The assembled candidate passed 218 focused tests in 94.09 s before
  trajectory measurement.
- The citation gate found 16/16 mapped citations, no failures, no map-audit
  failures, and passed all nine self-controls.  Shifting the first midpoint
  citation by two lines produced `SYMBOL-NOT-AT-LINE` and exit 1 as required.
- The restored final tree passed all 240 focused tests in 95.51 s.  This suite
  covers TKE operands and identity, histories/drag bundle gates, receipt
  citations, testcase configuration, and restart loading.
- A separate read-only Codex review was requested; its exact terminal verdict
  is recorded below after the review attempt.

## Independent review

**Independent review unavailable in-sandbox.**  The required command exited 1
before producing `SHIP`/`DO NOT SHIP`; this is not treated as approval.  Its
terminal verdict is quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)  
> Reading additional input from stdin...  
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

The complete capture is `round91/codex_review.txt`.  Per the operator's standing
instruction, work continued through mechanical citation and test gates; no
candidate physics was landed.

## Complete registered moved-row table

The table below is the gate's complete 86-row register.  “Move” is maximum
absolute candidate-minus-Round-85 field movement; “worsening” is the maximum
increase in absolute NEMO residual.  Improved/worsened are per-cell counts.

| # | registered row | move | worsening | improved/worsened |
|---:|---|---:|---:|---:|
| 1 | `GYRE-zco.kt1.stage3.u` | `6.73300663454879083e-07` | `6.73300483580913968e-07` | 0/17400 |
| 2 | `GYRE-zco.kt1.stage3.v` | `1.31835709343109493e-06` | `1.31835675790651806e-06` | 0/17100 |
| 3 | `GYRE-zco.kt1.zdf_entry.avm` | `1.38777878078144568e-17` | `4.33680868994201774e-19` | 1883/1874 |
| 4 | `GYRE-zco.kt1.zdf_entry.avt` | `1.73472347597680709e-18` | `5.42101086242752217e-20` | 1850/1857 |
| 5 | `GYRE-zco.kt10.after.uu_b` | `1.78525859740328099e-06` | `1.78525859740328099e-06` | 14/566 |
| 6 | `GYRE-zco.kt10.after.vv_b` | `1.08098340849665381e-06` | `1.08053237406809804e-06` | 34/536 |
| 7 | `GYRE-zco.kt10.before.S` | `8.69032896094523721e-03` | `8.66438514723455455e-03` | 744/17256 |
| 8 | `GYRE-zco.kt10.before.T` | `1.35382771165915017e-02` | `1.35382771165915017e-02` | 1804/16196 |
| 9 | `GYRE-zco.kt10.before.ssh` | `4.07903557921376891e-05` | `4.06137463056822912e-05` | 51/549 |
| 10 | `GYRE-zco.kt10.before.u` | `3.96403137872523501e-03` | `3.92482551457900772e-03` | 1475/15925 |
| 11 | `GYRE-zco.kt10.before.v` | `1.00333530793185154e-02` | `1.00155394903822491e-02` | 1726/15374 |
| 12 | `GYRE-zco.kt2.after.uu_b` | `1.91669062486930449e-07` | `1.62292111819287523e-07` | 43/537 |
| 13 | `GYRE-zco.kt2.after.vv_b` | `3.30848666176388265e-07` | `3.22212590857977668e-07` | 36/534 |
| 14 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.S` | `4.83597368894805868e-03` | `4.83597368894805868e-03` | 0/18000 |
| 15 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.T` | `3.06540661159715455e-03` | `3.06540661159715455e-03` | 0/18000 |
| 16 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.u` | `6.73321822438840718e-07` | `6.65909311567565703e-07` | 742/16658 |
| 17 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.v` | `1.31842973282261279e-06` | `1.31842973282261279e-06` | 486/16614 |
| 18 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.S` | `4.88313031022613586e-03` | `4.88313031022613586e-03` | 1047/16953 |
| 19 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.T` | `3.09529822046883396e-03` | `3.09529822046883396e-03` | 3424/14576 |
| 20 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.u` | `6.85025668236387145e-07` | `6.85025668236387145e-07` | 10892/6508 |
| 21 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.v` | `1.33587905175140687e-06` | `1.18631029326604197e-06` | 5995/11105 |
| 22 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.S` | `4.83571002207838774e-03` | `4.83571002207838774e-03` | 0/18000 |
| 23 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.T` | `3.06523947952186404e-03` | `3.06523947951475861e-03` | 0/18000 |
| 24 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.u` | `6.73300663454879083e-07` | `6.73300483580913968e-07` | 0/17400 |
| 25 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.v` | `1.31835709343109493e-06` | `1.31835675790651806e-06` | 0/17100 |
| 26 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.S` | `4.84148881399448783e-03` | `4.17499411012300925e-03` | 7202/10798 |
| 27 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.T` | `3.06886819665663779e-03` | `2.64642922280700077e-03` | 10185/7815 |
| 28 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.u` | `1.17815296330693229e-06` | `1.01237431122349253e-06` | 8703/8697 |
| 29 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.v` | `2.29757600489716250e-06` | `8.32012958834060247e-07` | 14850/2250 |
| 30 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.S` | `1.07083697024279445e-04` | `1.07083697024279445e-04` | 1334/16666 |
| 31 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.T` | `6.78883630023108253e-05` | `6.78883630023108253e-05` | 5448/12552 |
| 32 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.u` | `1.87571005454747714e-08` | `1.66572027391774426e-08` | 8264/9136 |
| 33 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.v` | `2.15532654506368966e-08` | `1.91405141726995985e-08` | 7885/9215 |
| 34 | `GYRE-zco.kt2.before.S` | `4.83571002207838774e-03` | `4.83571002207838774e-03` | 0/18000 |
| 35 | `GYRE-zco.kt2.before.T` | `3.06523947952186404e-03` | `3.06523947951475861e-03` | 0/18000 |
| 36 | `GYRE-zco.kt2.before.u` | `6.73300663454879083e-07` | `6.73300483580913968e-07` | 0/17400 |
| 37 | `GYRE-zco.kt2.before.v` | `1.31835709343109493e-06` | `1.31835675790651806e-06` | 0/17100 |
| 38 | `GYRE-zco.kt3.after.uu_b` | `4.25870353230103221e-07` | `3.91619774484097682e-07` | 11/569 |
| 39 | `GYRE-zco.kt3.after.vv_b` | `4.62441480258556515e-07` | `4.62441480258556515e-07` | 57/513 |
| 40 | `GYRE-zco.kt3.before.S` | `4.32463813389460938e-03` | `4.32463813389460938e-03` | 5/17995 |
| 41 | `GYRE-zco.kt3.before.T` | `1.56707234217954294e-02` | `1.56601715748756476e-02` | 53/17947 |
| 42 | `GYRE-zco.kt3.before.ssh` | `2.21488927941947394e-05` | `2.20911488499877986e-05` | 8/592 |
| 43 | `GYRE-zco.kt3.before.u` | `9.72563482253791661e-03` | `9.71807443855213920e-03` | 580/16820 |
| 44 | `GYRE-zco.kt3.before.v` | `1.57900523317320574e-02` | `1.57673330267499162e-02` | 613/16487 |
| 45 | `GYRE-zco.kt4.after.uu_b` | `8.27149552054995581e-07` | `7.97188448187040322e-07` | 44/536 |
| 46 | `GYRE-zco.kt4.after.vv_b` | `3.75848022715887463e-07` | `2.95442557377866336e-07` | 23/547 |
| 47 | `GYRE-zco.kt4.before.S` | `7.74239099202134184e-03` | `7.74209654046842388e-03` | 23/17977 |
| 48 | `GYRE-zco.kt4.before.T` | `3.29161883191986249e-02` | `3.29161883191986249e-02` | 288/17712 |
| 49 | `GYRE-zco.kt4.before.ssh` | `2.63397214114854603e-05` | `2.60785280171097616e-05` | 1/599 |
| 50 | `GYRE-zco.kt4.before.u` | `3.68187103360587088e-03` | `3.68135181537790953e-03` | 708/16692 |
| 51 | `GYRE-zco.kt4.before.v` | `1.37269523358991563e-02` | `1.36988879020776000e-02` | 382/16718 |
| 52 | `GYRE-zco.kt5.after.uu_b` | `1.09034872389809738e-06` | `1.09034872389809738e-06` | 14/566 |
| 53 | `GYRE-zco.kt5.after.vv_b` | `5.78525249163817923e-07` | `5.78525249163817923e-07` | 62/508 |
| 54 | `GYRE-zco.kt5.before.S` | `4.51290029523931935e-03` | `4.51290029523931935e-03` | 203/17797 |
| 55 | `GYRE-zco.kt5.before.T` | `7.43687899134570785e-03` | `7.39413765792917843e-03` | 818/17182 |
| 56 | `GYRE-zco.kt5.before.ssh` | `4.23669263257466389e-05` | `4.08185292284284160e-05` | 6/594 |
| 57 | `GYRE-zco.kt5.before.u` | `2.97793574202108793e-03` | `2.97793574202108793e-03` | 585/16815 |
| 58 | `GYRE-zco.kt5.before.v` | `4.05387891038428566e-03` | `4.05311708619484484e-03` | 790/16310 |
| 59 | `GYRE-zco.kt6.after.uu_b` | `1.37137769897236958e-06` | `1.37137769897236958e-06` | 8/572 |
| 60 | `GYRE-zco.kt6.after.vv_b` | `6.75438165050144118e-07` | `6.33724747621392318e-07` | 1/569 |
| 61 | `GYRE-zco.kt6.before.S` | `5.33669007526071937e-03` | `5.33669007526071937e-03` | 209/17791 |
| 62 | `GYRE-zco.kt6.before.T` | `2.91514792509488530e-02` | `2.91514792509488530e-02` | 1035/16965 |
| 63 | `GYRE-zco.kt6.before.ssh` | `4.39436472821292182e-05` | `4.39436472821292182e-05` | 6/594 |
| 64 | `GYRE-zco.kt6.before.u` | `3.42308930717802155e-03` | `3.41491786423749055e-03` | 717/16683 |
| 65 | `GYRE-zco.kt6.before.v` | `1.40221847719767187e-02` | `1.40024233592780586e-02` | 1153/15947 |
| 66 | `GYRE-zco.kt7.after.uu_b` | `1.66717155779131168e-06` | `1.66717155779131168e-06` | 7/573 |
| 67 | `GYRE-zco.kt7.after.vv_b` | `7.47483267913911717e-07` | `7.47483267913911717e-07` | 17/553 |
| 68 | `GYRE-zco.kt7.before.S` | `6.44103823034924972e-03` | `6.44103823034924972e-03` | 411/17589 |
| 69 | `GYRE-zco.kt7.before.T` | `2.32078594633335911e-02` | `2.32078594633335911e-02` | 1383/16617 |
| 70 | `GYRE-zco.kt7.before.ssh` | `5.05165695715303637e-05` | `4.33269330353308526e-05` | 11/589 |
| 71 | `GYRE-zco.kt7.before.u` | `6.11033070752081166e-03` | `6.11033070752081253e-03` | 782/16618 |
| 72 | `GYRE-zco.kt7.before.v` | `1.28173745532011336e-02` | `1.28173745532011336e-02` | 1073/16027 |
| 73 | `GYRE-zco.kt8.after.uu_b` | `1.81977486334375382e-06` | `1.81977486334375382e-06` | 0/580 |
| 74 | `GYRE-zco.kt8.after.vv_b` | `6.23172812582763191e-07` | `5.80160865561347994e-07` | 61/509 |
| 75 | `GYRE-zco.kt8.before.S` | `7.81285223258265660e-03` | `7.81285223258265660e-03` | 558/17442 |
| 76 | `GYRE-zco.kt8.before.T` | `2.80011793130512388e-02` | `2.80011793130512388e-02` | 1722/16278 |
| 77 | `GYRE-zco.kt8.before.ssh` | `4.48928697614728485e-05` | `4.07634136051528151e-05` | 21/579 |
| 78 | `GYRE-zco.kt8.before.u` | `8.16325298615196149e-03` | `8.14625966298525295e-03` | 1073/16327 |
| 79 | `GYRE-zco.kt8.before.v` | `1.55674193203936863e-02` | `1.55642108219264420e-02` | 1224/15876 |
| 80 | `GYRE-zco.kt9.after.uu_b` | `1.38686486357694413e-06` | `1.27054512657565413e-06` | 8/572 |
| 81 | `GYRE-zco.kt9.after.vv_b` | `8.38922311419154111e-07` | `7.62517593277554570e-07` | 57/513 |
| 82 | `GYRE-zco.kt9.before.S` | `7.75385227562708224e-03` | `7.75385227562708224e-03` | 650/17350 |
| 83 | `GYRE-zco.kt9.before.T` | `2.05843814062980357e-02` | `2.05843814062980357e-02` | 1818/16182 |
| 84 | `GYRE-zco.kt9.before.ssh` | `5.17340718173275871e-05` | `5.11337853882527782e-05` | 30/570 |
| 85 | `GYRE-zco.kt9.before.u` | `5.42780713741297660e-03` | `5.42780713741297660e-03` | 1119/16281 |
| 86 | `GYRE-zco.kt9.before.v` | `6.78180880260359845e-03` | `6.75101043982071283e-03` | 1600/15500 |

## OPEN

1. **Decision 40 remains the policy question:** should locally source-exact
   statements be allowed to land when the trajectory gate refuses them?  This
   round's evidence strongly supports **NO**: the full union exposes
   compensation, advances the first-over-bar to two additional tracer rows,
   and worsens the magnitude-leading day-30 T metric.
2. Repair the Round-90 reader to accept owned `32x22` rank-2 and
   `32x22x31` rank-3 payloads while retaining global-header, field-contract,
   finite-value, and planted-shift controls.  Admit the already acquired
   record; do not request or run NEMO again.
3. Use that admitted record to split the first live post-baro boundary into
   diagnosed correction versus application order.  Pre-register the first
   non-bit source statement and measure magnitude before any landing.
4. Keep Round 60 and Round 89 held; keep Round 62 excluded unless a new source
   walk re-establishes local exactness.  The candidate is recoverable at
   `39d346f...`; the shared implementation at the final tip remains Round 85.
5. If any shared TKE or W/ZAD statement is reconsidered, add a DINO numerical
   arm.  ORCA2 remains UNMEASURED-with-spec as stated above.
