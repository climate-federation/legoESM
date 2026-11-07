# NEMO testcase L2 GYRE Round 92 receipt — Round-89 damage bisection

**Date:** 2026-09-14  
**Incoming tip:** `dc62e1838499`  
**Frozen preregistration:** `b36451ef6bf7`  
**Measured candidate:** `bb963136d34139631682f44b33f069542db98128`  
**Production-restoration commit:** `f4a96b727423`  
**Evidence:** `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round92/`

## Outcome

**HELD. No candidate production physics lands.** Applied alone to the restored
Round-85 implementation, the held Round-89 Kaa/W/WZV/ZAD/source-assignment
member reproduces the Round-91 union's ladder damage exactly. It moves
first-over-bar backward from kt2 U/V to kt2 T/S/U/V, changes kt2 T/S/U/V to
`3.0652394795183113e-3`, `4.835710022078388e-3`,
`6.733005735178965e-7`, and `1.3183569256688065e-6`, and changes kt3 T/S to
`1.566544749833554e-2` and `4.324733116938262e-3`. The complete Rule-12 gate
fails with 954 certified rows, 88 violations, and 84 moved rows.

The damage is now discriminated from the member's local proof. Its assignment
is exact only when supplied with NEMO's recorded Kbb and Krhs. In the
independent live candidate, Kbb is bit exact but the Krhs input already differs
at every active cell: U 17,400/17,400 with maximum
`2.3323740139467519e-8`, and V 17,100/17,100 with maximum
`4.8049022149307440e-8`. The local proof therefore substituted the upstream
operand which legoESM does not reproduce. This diagnosis is recorded in the
held manifest header; the candidate production and candidate-only tests were
then restored byte-for-byte.

Two frozen predictions are explicitly **REFUTED** and retained. The composed
given-NEMO-input correction helper is not bit exact even though its diagnosed
zub/zvb are; and the member-alone day-30 T RMS is
`1.258996412021899e-2 K`, not the Round-91 union value
`1.2589964172802363e-2 K` (difference `-5.258337341940145e-11 K`). Neither
refutation rescues the failed ladder.

## Candidate and source proof

The whole Round-89 manifest was the one preregistered member and was applied
alone. Its five Kaa/W/WZV/ZAD rows all remain bit exact on NEMO inputs:

| local row | unequal / wet | maximum |
|---|---:|---:|
| carried Kaa SSH | 0 / 600 | `0.0` |
| direct shared W | 0 / 18,000 | `0.0` |
| captured stage-1 W | 0 / 18,000 | `0.0` |
| ZAD U | 0 / 17,400 | `0.0` |
| ZAD V | 0 / 17,100 | `0.0` |

All six stage assignment rows also remain bit exact on recorded NEMO inputs:
stage 1, stage 2, and the stage-3 transcription each have 0/17,400 U and
0/17,100 V unequal. The three independent Kaa/W/ZAD plants and the assignment
plant each exited 1.

## Compiled-source walk and first non-bit statement

The acquired GYRE card's executing vector branch assigns stage-1/2 Kaa from
Kbb plus `rDt*Krhs`, then masks, at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:668-687`. The direct
operand split follows that compiled expression in source order. Kbb U/V are
both 0 unequal; Krhs is the **first live non-bit input** at 17,400/17,400 U and
17,100/17,100 V unequal. Raw Kaa then differs by
`1.1195395266944408e-4` U and `2.3063530631667572e-4` V. Thus this candidate's
local assignment proof and ladder disagree because its calibration uses NEMO
Krhs in place of legoESM's already-different live Krhs. No downstream statement
is promoted over that measured boundary.

The compiled program next forms zub/zvb from the external target, current Kaa
velocity, executing thickness, and reference reciprocal, then adds each
correction to the active levels at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/stprk3_stg.f90:728-765`. The target
comes from the external-mode weighted accumulation and normalization at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/dynspg_ts.f90:763-804`; reference
depths and reciprocals are constructed at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/domain.f90:195-215`.

At that split, reference thickness and reciprocal are bit exact. The live
target is non-bit by `1.041557725484088e-11` U and
`7.05077619553296e-12` V; live final Kaa is non-bit by
`1.2184073888699132e-10` U and `1.6794210466741788e-10` V. These downstream
rows do not supersede Krhs as the first direct mismatch.

## Round-90 record admission and calibration retraction

The compiled write-only recorder serializes every field's actual shape at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90:35-51`. Its
correction arguments are assumed-shape owned arrays, while raw/final state,
targets, geometry, and masks are emitted from their declared arrays at
`GYRE_OMIP_L2_P3_SM_R90BARO/BLD/ppsrc/nemo/l2_r90_baro.f90:69-103`. The reader
now admits that contract: global `36x26x31`, owned bounds `3:34,3:24`, owned
zub/zvb `32x22`, and global shapes for the other fields. It still fails closed
on magic, version, field inventory, per-field rank/extent, finite values,
producer commit, digest, and final-add replay. The existing record is `READY`;
direct `raw + z*mask` is exact for all 17,400 U and 17,100 V wet values. Its
final-add ULP plant exits 1, and a synthetic wrong owned extent is rejected.
No new oracle acquisition is needed.

The preregistered composed-helper calibration is **REFUTED**. On NEMO inputs,
the computed zub/zvb themselves are exact (0/580 U and 0/570 V unequal), and
the record's static thickness is also bit-identical to the executing companion
Kaa thickness. But the composed JAX helper's final is unequal in 1,815/17,400
U cells, maximum `3.4694469519536142e-18`, and 3,132/17,100 V cells, maximum
`8.6736173798840355e-19`. Its live self-replay is likewise non-bit in 1,450 U
and 1,667 V cells. This is an AT-BAR association/fusion calibration debt, not
the trajectory damage owner. The split now reports `REFUTED` instead of
aborting or printing a false bit-exact claim, and its plant proves the clean
comparison changes.

Two superseded instrument attempts are retained as failed evidence. The first
used the 31-level record mask against NEMO's 30 active levels; the second
hypothesized a static-versus-Kaa thickness mismatch. The corrected instrument
shows the two thickness fields are bit-identical, so that hypothesis is
withdrawn. Logs `baro_split.log`, `baro_split_final.log`, and
`baro_split_final2.log` remain in the evidence root; only
`baro_split_final3.json` is the citable result.

## Certified GYRE ladder

| row | Round-85 before | Round-89 alone | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | `3.0652394795183113e-3` | AT-BAR -> DEBT |
| kt2 S | `2.1316282072803006e-14` | `4.835710022078388e-3` | AT-BAR -> DEBT |
| kt2 U | `2.7377110452773967e-12` | `6.733005735178965e-7` | worsened |
| kt2 V | `3.284922138989399e-12` | `1.3183569256688065e-6` | worsened |
| kt2 SSH | `0.0` | `0.0` | remains AT-BAR |
| kt3 T | `1.627497246303733e-4` | `1.566544749833554e-2` | worsened 96.25x |
| kt3 S | `6.327735185607253e-6` | `4.324733116938262e-3` | worsened 683.44x |

The numerical ladder prediction is **CONFIRMED** exactly at all six frozen
T/S/U/V headline values. First-over-bar regresses from `{U,V}@kt2` to
`{T,S,U,V}@kt2`. The complete comparator reports 954 certified rows, 88
violations, 84 moved rows, four AT-BAR-to-DEBT crossings, and largest residual
worsening `7.100975514389744e13` row-scale oracle ULP. The ordinary candidate
artifact is `round89_alone_kt1_10_complete.json` (SHA-256
`ecc3c04b4534da3b...`); the full gate is
`round89_alone_rule12_complete.json` (`016ebb6ee5ecd24a...`). Its planted
violation exits 1 (`27213bed7bb74553...`).

## Days 1-30

The required independent 30-day member completed despite the ladder
falsifier. Day 30 is:

| field | RMS against NEMO |
|---|---:|
| T | `1.258996412021899e-2 K` |
| S | `4.618881948599579e-3` |
| U | `5.808072952705772e-4 m s-1` |
| V | `7.911512182415908e-4 m s-1` |
| SSH | `5.194811827472877e-4 m` |

T worsens from the Round-85 before arm by `1.929528247121854e-4 K`
(1.556446%). The frozen equality-to-union prediction is **REFUTED** by the
`-5.258337341940145e-11 K` difference stated above; this is consistent with
Round 60 being numerically silent through kt3 headline maxima but not exactly
silent by day 30. The 30-row scorer is `round89_alone_day_gap.json` (SHA-256
`60810075c0bcd3e0...`), clean-stamped at the measured candidate.

## Tanks and cross-card disposition

No landing was possible, so no new tank claim is made. The Round-91 union
superset moved zero of 50 LOCK_EXCHANGE-zco certified rows and zero of 50
OVERFLOW-zps certified rows against their Round-85 arms; this retained evidence
is sufficient only to show that this held experiment did not require a tank
discriminator before refusal.

DINO is **not claimed neutral**. Kaa carry, W/WZV, ZAD, stage assignment, and
barotropic correction are shared statements, and DINO's 96--98% regional
cancellation warning makes a trajectory-rejected source change particularly
unsafe. No DINO numerical arm was run for a candidate that already failed
GYRE at kt2.

ORCA2 remains **UNMEASURED-WITH-SPEC**: select its compiled integrator; align
Kaa/Kbb SSH, WZV operands/carries, ZAD, source assignment, correction operands,
and restart state independently on native masks; score kt=1..10 T/S/U/V/SSH
against its current clean baseline; and reject any missing state, wet-input
mismatch, AT-BAR loss, or earlier first-over-bar. No ORCA2 inference is made
from GYRE or the tanks.

## Controls, tests, and provenance

- The acquired record is `READY`; its direct final-add plant exits 1, and the
  mixed-extent reader test includes a wrong-owned-extent rejection.
- All Kaa/W/ZAD and source-assignment local plants exit 1.
- The correction split's plant exits 1 and differs from its clean `REFUTED`
  output, so a permanently-refuted status cannot satisfy the control.
- The full Rule-12 plant exits 1. Its failure is retained, not treated as a
  successful scientific gate.
- Every scientific artifact is clean-stamped at measured candidate
  `bb963136d341`; the production restoration was verified byte-for-byte against
  the preregistration parent for every path touched by the held patch.
- The final focused suite passes 23/23 tests in 1.85 s. It covers the admitted
  record and mixed extents, live trace observer/privacy, the citation parser,
  full-map audit, and citation-gate controls. The known unrelated RK3-WS test
  was not in this focused selection and was neither encountered nor waived.
- The receipt citation gate finds 6/6 mapped compiled-source citations, zero
  failures, zero map-audit failures, and all nine self-controls firing. Its
  shifted assignment-citation plant produces `SYMBOL-NOT-AT-LINE` and exits 1.

## Independent review

**Independent review unavailable in-sandbox.** The required command exited 1
before producing a `SHIP` or `DO NOT SHIP` verdict; this is not treated as
approval. Its terminal verdict is quoted verbatim:

> WARNING: proceeding, even though we could not create PATH aliases: Read-only
> file system (os error 30)
>
> Reading additional input from stdin...
>
> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

The complete capture is `round92/codex_review.txt`. Per the operator's standing
instruction, the round continued through the mechanical gates; no candidate
physics landed.

## Complete registered moved-row table

The table below is the current candidate gate's complete 84-row register.
“Move” is maximum absolute candidate-minus-Round-85 field movement;
“worsening” is the maximum increase in absolute NEMO residual. Improved and
worsened counts are per cell.

+| # | registered row | move | worsening | improved/worsened |
|---:|---|---:|---:|---:|
| 1 | `GYRE-zco.kt1.stage3.u` | `6.73300663454879083e-07` | `6.73300483580913968e-07` | 0/17400 |
| 2 | `GYRE-zco.kt1.stage3.v` | `1.31835709343109493e-06` | `1.31835675790651806e-06` | 0/17100 |
| 3 | `GYRE-zco.kt10.after.uu_b` | `1.78525860042039880e-06` | `1.78525860042039880e-06` | 14/566 |
| 4 | `GYRE-zco.kt10.after.vv_b` | `1.08098341081424437e-06` | `1.08053237541858027e-06` | 34/536 |
| 5 | `GYRE-zco.kt10.before.S` | `8.69032868926922220e-03` | `8.66438487555853953e-03` | 744/17256 |
| 6 | `GYRE-zco.kt10.before.T` | `1.35382771163712334e-02` | `1.35382771163712334e-02` | 1804/16196 |
| 7 | `GYRE-zco.kt10.before.ssh` | `4.07903562433514083e-05` | `4.06137467568960103e-05` | 51/549 |
| 8 | `GYRE-zco.kt10.before.u` | `3.96403137823650789e-03` | `3.92482551409028060e-03` | 1475/15925 |
| 9 | `GYRE-zco.kt10.before.v` | `1.00333530791481759e-02` | `1.00155394902119096e-02` | 1726/15374 |
| 10 | `GYRE-zco.kt2.after.uu_b` | `1.91669062487350578e-07` | `1.62292111821374613e-07` | 43/537 |
| 11 | `GYRE-zco.kt2.after.vv_b` | `3.30848666176063004e-07` | `3.22212590857815038e-07` | 36/534 |
| 12 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.S` | `4.83597368894805868e-03` | `4.83597368894805868e-03` | 0/18000 |
| 13 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.T` | `3.06540661159715455e-03` | `3.06540661159715455e-03` | 0/18000 |
| 14 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.u` | `6.73321822438840718e-07` | `6.65909311567565703e-07` | 742/16658 |
| 15 | `GYRE-zco.kt2.arm.omit_barotropic_substep_drag.v` | `1.31842973282261279e-06` | `1.31842973282261279e-06` | 486/16614 |
| 16 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.S` | `4.88313031022613586e-03` | `4.88313031022613586e-03` | 1047/16953 |
| 17 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.T` | `3.09529822046883396e-03` | `3.09529822046883396e-03` | 3424/14576 |
| 18 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.u` | `6.85025668236387145e-07` | `6.85025668236387145e-07` | 10892/6508 |
| 19 | `GYRE-zco.kt2.arm.omit_freshwater_forcing.v` | `1.33587905175140687e-06` | `1.18631029326604197e-06` | 5995/11105 |
| 20 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.S` | `4.83571002207838774e-03` | `4.83571002207838774e-03` | 0/18000 |
| 21 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.T` | `3.06523947952186404e-03` | `3.06523947951475861e-03` | 0/18000 |
| 22 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.u` | `6.73300663454879083e-07` | `6.73300483580913968e-07` | 0/17400 |
| 23 | `GYRE-zco.kt2.arm.omit_momentum_transport_reconcile.v` | `1.31835709343109493e-06` | `1.31835675790651806e-06` | 0/17100 |
| 24 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.S` | `4.84148881399448783e-03` | `4.17499411012300925e-03` | 7202/10798 |
| 25 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.T` | `3.06886819665663779e-03` | `2.64642922280700077e-03` | 10185/7815 |
| 26 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.u` | `1.17815296330693229e-06` | `1.01237431122349253e-06` | 8703/8697 |
| 27 | `GYRE-zco.kt2.arm.omit_stage_barotropic_correction.v` | `2.29757600489716250e-06` | `8.32012958834060247e-07` | 14850/2250 |
| 28 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.S` | `1.07083697024279445e-04` | `1.07083697024279445e-04` | 1334/16666 |
| 29 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.T` | `6.78883630023108253e-05` | `6.78883630023108253e-05` | 5448/12552 |
| 30 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.u` | `1.87571005454747714e-08` | `1.66572027391774426e-08` | 8264/9136 |
| 31 | `GYRE-zco.kt2.arm.omit_surface_boundary_forcing.v` | `2.15532654506368966e-08` | `1.91405141726995985e-08` | 7885/9215 |
| 32 | `GYRE-zco.kt2.before.S` | `4.83571002207838774e-03` | `4.83571002207838774e-03` | 0/18000 |
| 33 | `GYRE-zco.kt2.before.T` | `3.06523947952186404e-03` | `3.06523947951475861e-03` | 0/18000 |
| 34 | `GYRE-zco.kt2.before.u` | `6.73300663454879083e-07` | `6.73300483580913968e-07` | 0/17400 |
| 35 | `GYRE-zco.kt2.before.v` | `1.31835709343109493e-06` | `1.31835675790651806e-06` | 0/17100 |
| 36 | `GYRE-zco.kt3.after.uu_b` | `4.25870353229561120e-07` | `3.91619774482579799e-07` | 11/569 |
| 37 | `GYRE-zco.kt3.after.vv_b` | `4.62441480254653387e-07` | `4.62441480254653387e-07` | 57/513 |
| 38 | `GYRE-zco.kt3.before.S` | `4.32463813389460938e-03` | `4.32463813389460938e-03` | 5/17995 |
| 39 | `GYRE-zco.kt3.before.T` | `1.56707234217954294e-02` | `1.56601715748756476e-02` | 53/17947 |
| 40 | `GYRE-zco.kt3.before.ssh` | `2.21488927942235792e-05` | `2.20911488500047121e-05` | 8/592 |
| 41 | `GYRE-zco.kt3.before.u` | `9.72563482253791661e-03` | `9.71807443855213920e-03` | 580/16820 |
| 42 | `GYRE-zco.kt3.before.v` | `1.57900523317320574e-02` | `1.57673330267499162e-02` | 613/16487 |
| 43 | `GYRE-zco.kt4.after.uu_b` | `8.27149552173769929e-07` | `7.97188448261416591e-07` | 44/536 |
| 44 | `GYRE-zco.kt4.after.vv_b` | `3.75848023007605610e-07` | `2.95442557669584483e-07` | 23/547 |
| 45 | `GYRE-zco.kt4.before.S` | `7.74239099202134184e-03` | `7.74209654046842388e-03` | 23/17977 |
| 46 | `GYRE-zco.kt4.before.T` | `3.29161883191986249e-02` | `3.29161883191986249e-02` | 288/17712 |
| 47 | `GYRE-zco.kt4.before.ssh` | `2.63397214114759193e-05` | `2.60785280171908600e-05` | 1/599 |
| 48 | `GYRE-zco.kt4.before.u` | `3.68187103360588909e-03` | `3.68135181537791647e-03` | 708/16692 |
| 49 | `GYRE-zco.kt4.before.v` | `1.37269523358991563e-02` | `1.36988879020776000e-02` | 382/16718 |
| 50 | `GYRE-zco.kt5.after.uu_b` | `1.09034872365762134e-06` | `1.09034872365762134e-06` | 14/566 |
| 51 | `GYRE-zco.kt5.after.vv_b` | `5.78525249261396118e-07` | `5.78525249261396118e-07` | 62/508 |
| 52 | `GYRE-zco.kt5.before.S` | `4.51290029523221392e-03` | `4.51290029523221392e-03` | 203/17797 |
| 53 | `GYRE-zco.kt5.before.T` | `7.43687899139544584e-03` | `7.39413765797891642e-03` | 818/17182 |
| 54 | `GYRE-zco.kt5.before.ssh` | `4.23669262973778386e-05` | `4.08185292000596157e-05` | 6/594 |
| 55 | `GYRE-zco.kt5.before.u` | `2.97793574201127981e-03` | `2.97793574201127981e-03` | 585/16815 |
| 56 | `GYRE-zco.kt5.before.v` | `4.05387891038433423e-03` | `4.05311708619485699e-03` | 790/16310 |
| 57 | `GYRE-zco.kt6.after.uu_b` | `1.37137770024609029e-06` | `1.37137770024609029e-06` | 8/572 |
| 58 | `GYRE-zco.kt6.after.vv_b` | `6.75438164291853119e-07` | `6.33724746023928837e-07` | 1/569 |
| 59 | `GYRE-zco.kt6.before.S` | `5.33669007526071937e-03` | `5.33669007526071937e-03` | 209/17791 |
| 60 | `GYRE-zco.kt6.before.T` | `2.91514792537910239e-02` | `2.91514792537910239e-02` | 1035/16965 |
| 61 | `GYRE-zco.kt6.before.ssh` | `4.39436472833149017e-05` | `4.39436472833149017e-05` | 6/594 |
| 62 | `GYRE-zco.kt6.before.u` | `3.42308930717720537e-03` | `3.41491786423667436e-03` | 717/16683 |
| 63 | `GYRE-zco.kt6.before.v` | `1.40221847720065698e-02` | `1.40024233591834815e-02` | 1153/15947 |
| 64 | `GYRE-zco.kt7.after.uu_b` | `1.66717155655445384e-06` | `1.66717155655445384e-06` | 7/573 |
| 65 | `GYRE-zco.kt7.after.vv_b` | `7.47483271676526936e-07` | `7.47483271676526936e-07` | 17/553 |
| 66 | `GYRE-zco.kt7.before.S` | `6.44103823030661715e-03` | `6.44103823030661715e-03` | 411/17589 |
| 67 | `GYRE-zco.kt7.before.T` | `2.32078594621043521e-02` | `2.32078594621043521e-02` | 1383/16617 |
| 68 | `GYRE-zco.kt7.before.ssh` | `5.05165707932555587e-05` | `4.33269332454824102e-05` | 11/589 |
| 69 | `GYRE-zco.kt7.before.u` | `6.11033070747685984e-03` | `6.11033070747685984e-03` | 782/16618 |
| 70 | `GYRE-zco.kt7.before.v` | `1.28173745532076527e-02` | `1.28173745532076527e-02` | 1073/16027 |
| 71 | `GYRE-zco.kt8.after.uu_b` | `1.81977486949942008e-06` | `1.81977486949942008e-06` | 0/580 |
| 72 | `GYRE-zco.kt8.after.vv_b` | `6.23172808976706766e-07` | `5.80160861955291568e-07` | 61/509 |
| 73 | `GYRE-zco.kt8.before.S` | `7.81285223248318061e-03` | `7.81285223248318061e-03` | 558/17442 |
| 74 | `GYRE-zco.kt8.before.T` | `2.80011793130583442e-02` | `2.80011793130583442e-02` | 1722/16278 |
| 75 | `GYRE-zco.kt8.before.ssh` | `4.48928699805020703e-05` | `4.07634134622619076e-05` | 21/579 |
| 76 | `GYRE-zco.kt8.before.u` | `8.16325298623397227e-03` | `8.14625966306726373e-03` | 1073/16327 |
| 77 | `GYRE-zco.kt8.before.v` | `1.55674193183924607e-02` | `1.55642108199252165e-02` | 1224/15876 |
| 78 | `GYRE-zco.kt9.after.uu_b` | `1.38686485936330081e-06` | `1.27054512268770514e-06` | 8/572 |
| 79 | `GYRE-zco.kt9.after.vv_b` | `8.38922312487201671e-07` | `7.62517592704228461e-07` | 57/513 |
| 80 | `GYRE-zco.kt9.before.S` | `7.75385225722402538e-03` | `7.75385225722402538e-03` | 650/17350 |
| 81 | `GYRE-zco.kt9.before.T` | `2.05843814061275054e-02` | `2.05843814061275054e-02` | 1818/16182 |
| 82 | `GYRE-zco.kt9.before.ssh` | `5.17340716891818292e-05` | `5.11337853671776227e-05` | 30/570 |
| 83 | `GYRE-zco.kt9.before.u` | `5.42780713759715566e-03` | `5.42780713759715566e-03` | 1119/16281 |
| 84 | `GYRE-zco.kt9.before.v` | `6.78180880270320974e-03` | `6.75101043992032412e-03` | 1600/15500 |

## OPEN for Round 93

1. **Continue Decision-40 member bisection by magnitude.** Round 89 is now the
   proven dominant damaging member and stays held. Apply Round 60 TKE alone to
   the restored tip, re-prove its local rows and plants, score kt2 T/S/U/V and
   kt3 T/S, and annotate its manifest with its member-alone damage or
   harmlessness. Do not infer exact silence from the Round-91 union's headline
   maxima; the day-30 refutation shows a residual interaction.
2. **Do not retry Round 89 downstream.** Its local source-assignment proof is
   conditional on substituted NEMO Krhs, whereas live legoESM Krhs is non-bit
   at every active U/V cell. Any future reconsideration must first walk that
   upstream RHS in compiled source order and prove the live operand exact.
3. **Keep correction association explicit.** The direct NEMO zub/zvb and
   final-add record are exact, but the composed JAX helper is AT-BAR-not-exact.
   This is a separate calibration debt and not permission to change production
   while the larger Krhs boundary remains.
4. **No acquisition or configuration decision is pending.** The Round-90
   record is admitted and sufficient for this boundary. All held patches remain
   held; Rule 12 remains the only landing criterion.
5. **Issue #1455 was not updated.** `gh auth status` reports that the configured
   `dhruvbalwada` token is invalid. The next authenticated round should post
   this receipt's held verdict, the 84-row register, and the Krhs diagnosis;
   `round92/github_auth_status.txt` preserves the failure.
