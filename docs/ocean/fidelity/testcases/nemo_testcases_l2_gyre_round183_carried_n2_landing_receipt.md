# Round 183 receipt — carried step-entry `rn2b` landing

**Status: LANDED.**  Decision 61 authorizes the Round-181 candidate to land
with DINO developed-state registered **UNMEASURED** because the unchanged
Round-182 control destroys every prognostic family in its first bridged step.
The one-variable GYRE card route makes the developed-state mixed-layer outputs
bit-exact and reduces T3D RMS by 28.24x at day 30, 249.74x at day 240, and
4.20x at day 360.  The certified first-over-bar remains kt=3 and every kt=1
and kt=2 row is unchanged.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round183.md`,
commit `ac8589423`.  Candidate measurement commit: `4ff2a004f`.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round183/`.

## Compiled statement and implementation

The record-producing GYRE program computes `rn2b` once from `Nbb`, copies it
to `rn2`, and calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:159-168`; it later
passes the same stored `rn2b` to `ldf_slp` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-180`.
The compiled producer evaluates the active W levels at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90:1609-1619`.
The mixed-layer recurrence consumes that `rn2b`, multiplies it by the live W
thickness, applies the strict below-threshold branch, and gathers `hmlp` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124`.

The GYRE card now selects its existing `carried_step_entry` N2 route.  The
shared step builds the already existing step-entry N2/thickness bundle when
either implicit mixing or that selector needs it, and the shared MLD helper
consumes the bundle rather than evaluating and discarding a second local
`eosbn2`.  The selector, bundle type, formulas, threshold, timestep, restart
schema, library default, and every non-GYRE card default are unchanged.  The
production diff remains JAX-array-only and introduces no mutation or pytree
field.

## Local proof and plants

The admitted day-180 entry is NEMO restart step 1080; all T/S/u/v/ssh, TKE,
and six barotropic-history source checks report zero unequal cells.  Through
the complete production closure, not an isolated expression, the table is:

| execution | `nmln` unequal / max | `hmlp` unequal / max (m) |
|---|---:|---:|
| baseline production JIT | 14 / 1 | 14 / 1.3310140874585437e2 |
| baseline production eager | 14 / 1 | 14 / 1.3310140874585437e2 |
| carried production JIT | 0 / 0 | 0 / 0 |
| carried production eager | 0 / 0 | 0 / 0 |

The literal recurrence rebuilt directly from the recorded NEMO operands is
also bit-exact in all 600 columns.  The one-level upstream plant preserves
the recorded `prd`, `pn2`, and `gdept` operands, moves the downstream
`nmln1`, `zhmlpt1`, `r1_hmlu2`, and `uslp4` rows, prints
`STATUS PLANT-FIRED`, and exits 1.  The focused dead-recompute regression
fails if the carried input is ignored.

## Certified ladder and moved-row registry

The same-tip baseline and candidate both have first-over-bar kt=3.  All five
kt=1 fields are bit-identical; kt2 T/S/U/V remain respectively
`1.4210854715202004e-14`, `2.1316282072803006e-14`,
`8.326672684688674e-17`, and `9.71445146547012e-17`, all AT-BAR.  Candidate
kt3 T/S are `4.9403105251144552e-07` and
`4.0085410546453204e-08`.  Decision 43 permits registered kt>=2 movement;
there are exactly 53 such rows and no status transition:

| row | before max abs | after max abs | improved cells | worsened cells |
|---|---:|---:|---:|---:|
| `GYRE-zco.kt10.after.uu_b` | 8.83599402168224607e-08 | 8.82884054051111256e-08 | 287 | 293 |
| `GYRE-zco.kt10.after.vv_b` | 1.86806483085284747e-07 | 1.86792462924384450e-07 | 258 | 312 |
| `GYRE-zco.kt10.before.S` | 3.89135848877231183e-07 | 2.99949292070778029e-07 | 13183 | 4393 |
| `GYRE-zco.kt10.before.T` | 7.14460109207948335e-06 | 7.90666589978172851e-07 | 13424 | 4569 |
| `GYRE-zco.kt10.before.ssh` | 4.20843504361084753e-07 | 3.96187362813361921e-07 | 301 | 299 |
| `GYRE-zco.kt10.before.u` | 3.83915270255053009e-06 | 3.80045692572467120e-06 | 9143 | 8257 |
| `GYRE-zco.kt10.before.v` | 5.30468073692974551e-06 | 5.27587317372332776e-06 | 8901 | 8199 |
| `GYRE-zco.kt3.after.uu_b` | 1.19050781525933264e-07 | 1.19050660090844716e-07 | 267 | 313 |
| `GYRE-zco.kt3.after.vv_b` | 7.09070243058327818e-08 | 7.09072666178625979e-08 | 293 | 277 |
| `GYRE-zco.kt3.before.S` | 4.00862987248729041e-08 | 4.00854105464532040e-08 | 5448 | 6741 |
| `GYRE-zco.kt3.before.T` | 4.94007107221250408e-07 | 4.94031052511445523e-07 | 7648 | 9771 |
| `GYRE-zco.kt4.after.uu_b` | 1.41522353173028845e-07 | 1.41527190505603279e-07 | 227 | 353 |
| `GYRE-zco.kt4.after.vv_b` | 9.48910905111286992e-08 | 9.48979453238530876e-08 | 248 | 322 |
| `GYRE-zco.kt4.before.S` | 9.05886565760738449e-08 | 9.04064236806334520e-08 | 9372 | 7323 |
| `GYRE-zco.kt4.before.T` | 6.83756400832180589e-07 | 6.84401818773494597e-07 | 9936 | 7971 |
| `GYRE-zco.kt4.before.ssh` | 4.57991816644152949e-07 | 4.57991058076465118e-07 | 330 | 270 |
| `GYRE-zco.kt4.before.u` | 2.50412361080948331e-06 | 2.50412715784020801e-06 | 8810 | 8590 |
| `GYRE-zco.kt4.before.v` | 4.82784276371883703e-06 | 4.82782969667139295e-06 | 8454 | 8646 |
| `GYRE-zco.kt5.after.uu_b` | 1.47914054571436637e-07 | 1.47913914493816995e-07 | 272 | 308 |
| `GYRE-zco.kt5.after.vv_b` | 8.12318890928628856e-08 | 8.12586542536970768e-08 | 281 | 289 |
| `GYRE-zco.kt5.before.S` | 8.76373675851027656e-08 | 8.74163603725719440e-08 | 10013 | 7099 |
| `GYRE-zco.kt5.before.T` | 7.83308351515188406e-07 | 7.10542632731403501e-07 | 10533 | 7423 |
| `GYRE-zco.kt5.before.ssh` | 3.13194187884571762e-07 | 3.13208255014844894e-07 | 279 | 321 |
| `GYRE-zco.kt5.before.u` | 5.35330081178481132e-06 | 5.35282723877761324e-06 | 8028 | 9372 |
| `GYRE-zco.kt5.before.v` | 3.12724047083076055e-06 | 3.12776947753768981e-06 | 8103 | 8997 |
| `GYRE-zco.kt6.after.uu_b` | 1.30099703171215973e-07 | 1.30068838357201913e-07 | 283 | 297 |
| `GYRE-zco.kt6.after.vv_b` | 1.07389680820656197e-07 | 1.07367267802383273e-07 | 272 | 298 |
| `GYRE-zco.kt6.before.S` | 1.01893071757785947e-07 | 1.02873450202878303e-07 | 9830 | 7508 |
| `GYRE-zco.kt6.before.T` | 9.46511249821924139e-07 | 8.46628761763668081e-07 | 10220 | 7744 |
| `GYRE-zco.kt6.before.ssh` | 2.99562232945575768e-07 | 2.99414141695375502e-07 | 332 | 268 |
| `GYRE-zco.kt6.before.u` | 5.22023153219462873e-06 | 5.21637252011360490e-06 | 8698 | 8702 |
| `GYRE-zco.kt6.before.v` | 3.65894237981884488e-06 | 3.66857912719028775e-06 | 8546 | 8554 |
| `GYRE-zco.kt7.after.uu_b` | 1.04663674036203219e-07 | 1.04706633931393039e-07 | 271 | 309 |
| `GYRE-zco.kt7.after.vv_b` | 1.44032942916722667e-07 | 1.43987993677947151e-07 | 279 | 291 |
| `GYRE-zco.kt7.before.S` | 3.24584064514965576e-07 | 2.33569465990512981e-07 | 11065 | 6352 |
| `GYRE-zco.kt7.before.T` | 8.85916390558350031e-06 | 1.09287084626430442e-06 | 11349 | 6630 |
| `GYRE-zco.kt7.before.ssh` | 4.63273312332969540e-07 | 4.71311204701266653e-07 | 348 | 252 |
| `GYRE-zco.kt7.before.u` | 2.78917228440332676e-06 | 2.78837193087161372e-06 | 8945 | 8455 |
| `GYRE-zco.kt7.before.v` | 4.92839697968303070e-06 | 4.93745126814350982e-06 | 8649 | 8451 |
| `GYRE-zco.kt8.after.uu_b` | 1.31561377124693110e-07 | 1.31532721930481718e-07 | 268 | 312 |
| `GYRE-zco.kt8.after.vv_b` | 1.44922250460560731e-07 | 1.44894541474413496e-07 | 251 | 319 |
| `GYRE-zco.kt8.before.S` | 3.38478365335959097e-07 | 1.61470595116952609e-07 | 12585 | 4917 |
| `GYRE-zco.kt8.before.T` | 9.20037180662802712e-06 | 1.00794007451554535e-06 | 12976 | 5016 |
| `GYRE-zco.kt8.before.ssh` | 4.09196899097546973e-07 | 3.90642211957076962e-07 | 311 | 289 |
| `GYRE-zco.kt8.before.u` | 4.81588507090027762e-06 | 4.85151870532364582e-06 | 9161 | 8239 |
| `GYRE-zco.kt8.before.v` | 2.77971665997578388e-06 | 2.78700927702074663e-06 | 8861 | 8239 |
| `GYRE-zco.kt9.after.uu_b` | 9.85576323457061643e-08 | 9.85904158685643739e-08 | 282 | 298 |
| `GYRE-zco.kt9.after.vv_b` | 2.27677215575063790e-07 | 2.27632822164675064e-07 | 253 | 317 |
| `GYRE-zco.kt9.before.S` | 2.88597810538249178e-07 | 2.62386315341700538e-07 | 12672 | 4861 |
| `GYRE-zco.kt9.before.T` | 7.80175495052048973e-06 | 9.79886650043226837e-07 | 12957 | 5036 |
| `GYRE-zco.kt9.before.ssh` | 5.68624840843398593e-07 | 5.62192586994487292e-07 | 362 | 238 |
| `GYRE-zco.kt9.before.u` | 2.51116152653711733e-06 | 2.50669478638228249e-06 | 8975 | 8425 |
| `GYRE-zco.kt9.before.v` | 3.93806783602346222e-06 | 3.94128898817402984e-06 | 8490 | 8610 |

The explicit TSV registry and the JSON moved-row set match exactly.  Removing
one registry row prints `STATUS PLANT-FIRED` and exits 1.

## Month and year

Both arms were measured from rest with the same CPU/fp64 harness and commit
stamps.  Every registered checkpoint improves:

| day | before T3D RMS (K) | after T3D RMS (K) | delta (K) |
|---:|---:|---:|---:|
| 30 | 6.57257437477060260e-05 | 2.32767720506839873e-06 | -6.33980665426376298e-05 |
| 60 | 2.08080200035203869e-04 | 1.47857643947091489e-05 | -1.93294435640494730e-04 |
| 90 | 1.86430187737560626e-03 | 1.62885181062949244e-05 | -1.84801335926931126e-03 |
| 120 | 1.04847381054101003e-03 | 1.09638744977205423e-04 | -9.38835065563804663e-04 |
| 180 | 3.58388546864078263e-03 | 6.11330337949491244e-05 | -3.52275243484583347e-03 |
| 240 | 1.64483607011786798e-02 | 6.58617188147951740e-05 | -1.63824989823638835e-02 |
| 300 | 1.36029900410691660e-02 | 5.47485769059515161e-05 | -1.35482414641632139e-02 |
| 360 | 1.12256600185513065e-02 | 2.67099238532946892e-03 | -8.55466763322183799e-03 |

The year-day240-worse plant prints `STATUS PLANT-FIRED` and exits 1.  The
candidate arm is the new immutable before arm for subsequent GYRE work:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round183/after_year/` plus
`after_ladder.json` and `after_day_gap.json`.

## Card census and blast radius

| card | executes route | result |
|---|---|---|
| GYRE-zco | yes | local proof, ladder, month, and year PASS |
| DINO `nemo_dino_kamm` | yes | from-rest DINO suite PASS; developed-state UNMEASURED under Decision 61 |
| DINO `nemo_dino_kamm_mlf` | yes | from-rest DINO suite PASS; developed-state UNMEASURED under Decision 61 |
| generic NEMO-GYRE recipe | no (`rho_c`) | independent three-step certified gate PASS |
| ORCA2-zps | no (`recompute`) | UNMEASURED-with-spec; route does not execute |
| LOCK_EXCHANGE-zco | no GM/Redi | tank/focused gate PASS |
| OVERFLOW-zps | no GM/Redi | tank/focused gate PASS |

Round 182 proves why “developed-state UNMEASURED” is not an inferred waiver:
the unchanged DINO control's first bridged step returned 10,348 non-finite
`eta`, 342,134 each in T and S, 379,692 in u, and 372,528 in v before the
next-step geometry guard observed the damage.  Round 184 owns that bridge
defect.  No DINO trajectory claim is made here.

The recipe-derived admission census names exactly GYRE and the two DINO Kamm
cards as executing.  A gate defect discovered this round made shared DINO
execution an unconditional veto even after every executing card was measured;
the gate now admits a shared statement only through
`all_executing_cards_measured`, with a regression proving an omitted DINO row
still fails.

## Review, citations, and tests

The required read-only Codex review was attempted after the complete
implementation and gate diff.  It produced no scientific verdict; its final
line is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**.  A second read-only
Claude review was attempted and also produced no scientific verdict; its
terminal line is quoted verbatim:

> API Error: Can't reach the API server — check your internet or DNS (ENOTFOUND)

The cumulative citation gate passes after rigidly shifting all 36 map entries
and all 10 cited occurrences displaced by this landing.  Its planted shift of
`ocean_model_latlon_cgrid.py:8956-8958` exits 1.  The Round-183 direct receipt
citation gate and its compiled-source shift plant also pass/fire respectively.

Focused results: DINO **128 passed, 9 warnings**; generic recipe **25 passed**;
MLF production-step transcription **22 passed**; Decision-43/45 gate
**14 passed**.  The MLD transcription file reports **1 failed, 20 passed**;
the sole failure is the starting-tip
`test_treguier_kappa_uses_the_same_n2_variant_as_the_slopes`, which searches
the dispatcher for a call already moved to `native_treguier_kappa_for_state`
before this round.  The initial combined run reports **22 failed, 233 passed**
after LLVM exhausted memory; fresh per-file processes turn 21 of those 22
into passes and leave only that base-identical source-inspection failure.

Final full-tree result: **PENDING**.

All frozen numeric predictions are confirmed.  No acquisition or
configuration decision is needed.

## OPEN

Round 184 repairs the DINO developed-state bridge on the unchanged tree: start
from the first bridged production step, name and cite the first statement that
turns a previously finite prognostic family non-finite, land the minimal fix
with a non-vacuous production plant, then measure both DINO cards with the
Round-183 carried route and register the result here.  Do not attribute the
downstream step-2 geometry guard as the first statement.
