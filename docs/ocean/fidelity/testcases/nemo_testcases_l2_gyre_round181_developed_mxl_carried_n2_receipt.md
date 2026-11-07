# Round 181 receipt — developed `zdf_mxl` carried-N2 walk

**Status: HELD.**  The one-variable GYRE candidate is locally source-exact and
strongly beneficial, but it is not certified on its complete blast radius.
Both DINO Kamm cards execute the same carried-N2 route, and both the candidate
and recompute control stop before the first step at the same pre-existing raw
mesh `e3w_int` positivity guard.  Production physics and card defaults were
therefore restored.  No immutable before arm moved.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round181.md`, commit
`eb33cf840`.  Candidate science commit: `077d7ee97`.  Production-restoration
commit: `05d014b48`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round181/`.

## Compiled statement and first non-bit boundary

The compiled stage program computes `rab_b` and `rn2b` once from `Nbb`, copies
them to `rab_n` and `rn2`, and calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:159-168`.
It then passes that same stored `rn2b` to `ldf_slp` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/stprk3.f90:173-180`.
The active `bn2` formula is the level loop, interpolation, live-depth divisor,
and W mask at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/eosbn2.f90:1609-1619`.

The mixed-layer recurrence initializes at `nlb10`, accumulates
`MAX(rn2b,0)*e3w_1d*(1+r3t)`, applies the strict below-threshold test, and
converts the selected W index to live depth at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124`.
The build defines the ten-metre reference and `nlb10` at
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/domzgr.f90:382-386`.

The first non-bit statement is therefore the **routing of the mixed-layer
recurrence's N2 operand**: NEMO consumes the one stored step-entry `rn2b`,
while the legoESM GYRE card recomputes N2 inside its mixed-layer helper.  This
is not a change to the `bn2` arithmetic or to the recurrence threshold.

## Calibrated production-step proof

The existing Round-179 admitted record supplies every discriminating operand;
no acquisition was needed.  A literal scalar reconstruction first reproduced
the compiled source recurrence exactly: Fortran `nlb10=2`,
`zrefdep=9.00494694181948 m`, and
`zN2_c=9.558138401559454e-5 s-2 m`; NEMO `nmln` and `hmlp` each have zero
unequal wet columns.  The observer is passive in every arm (zero returned-state
bytes moved versus a separately compiled ordinary production step).

| row | baseline JIT unequal / max abs | carried JIT unequal / max abs | baseline eager unequal / max abs | carried eager unequal / max abs |
|---|---:|---:|---:|---:|
| `nmln` | 14 / 1 | 0 / 0 | 14 / 1 | 0 / 0 |
| `hmlp` | 14 / 1.3310140874585437e2 m | 0 / 0 | 14 / 1.3310140874585437e2 m | 0 / 0 |
| `zhmlpt` | 14 / 1.1912680066521602e2 m | 0 / 0 | 14 / 1.1912680066521602e2 m | 0 / 0 |
| `pn2` | 17,400 / 1.8906742561290626e-6 | 16,394 / 1.8431436932253575e-18 | 17,400 / 1.8906742561290626e-6 | 0 / 0 |
| `r3t_Kmm` | 600 / 1.1071059433792607e-16 | 600 / 1.1071059433792607e-16 | 600 / 1.1071059433792607e-16 | 600 / 1.1071059433792607e-16 |
| `uslp` | 16,820 / 1.8435605560488408e-4 | 16,819 / 9.541340800940917e-13 | 16,820 / 1.8435605560391394e-4 | 16,208 / 1.3324482965956395e-12 |

Thus the carried route closes the source-owned integer index and depth under
the authoritative full production JIT step and eager execution.  The JIT-only
last-bit `pn2` remainder is the already documented compiled/eager association
floor; it does not change either branch decision.  The first remaining
registered row is inherited `r3t_Kmm`, not an owned recurrence write.

The final one-level plant changes 308 returned-state bytes and moves one
`nmln`, one `zhmlpt`, two `r1_hmlu`, and four `uslp` cells, while `prd`, `pn2`,
and `gdept_1d` remain unchanged.  It prints `STATUS PLANT-FIRED` and exits 1.
An earlier plant compared the carried mutation with the recompute baseline and
was invalid; it was corrected before this final control and is retained in the
evidence directory as an instrument defect, not evidence.

## GYRE trajectory result

The candidate left every kt=1 and kt=2 row unchanged.  kt2 T/S/U/V remain
`1.4210854715202004e-14 / 2.1316282072803006e-14 /
8.326672684688674e-17 / 9.71445146547012e-17`, all AT-BAR, and the first
over-bar row remains kt=3.  kt3 T moved from
`4.940071072212504e-7` to `4.940310525114455e-7`; kt3 S moved from
`4.0086298724872904e-8` to `4.0085410546453204e-8`.  The complete metric-row
movement table is below; cellwise improved/worsened counts for all 70 rows,
including unchanged row metrics whose cells moved, are preserved in
`ladder_comparison.json`.

| row | before max abs | candidate max abs | direction |
|---|---:|---:|---|
| `kt3.S` | 4.00862987248729041e-08 | 4.00854105464532040e-08 | improved |
| `kt3.T` | 4.94007107221250408e-07 | 4.94031052511445523e-07 | worsened |
| `kt4.S` | 9.05886565760738449e-08 | 9.04064236806334520e-08 | improved |
| `kt4.T` | 6.83756400832180589e-07 | 6.84401818773494597e-07 | worsened |
| `kt4.ssh` | 4.57991816644152949e-07 | 4.57991058076465118e-07 | improved |
| `kt4.u` | 2.50412361080948331e-06 | 2.50412715784020801e-06 | worsened |
| `kt4.v` | 4.82784276371883703e-06 | 4.82782969667139295e-06 | improved |
| `kt5.S` | 8.76373675851027656e-08 | 8.74163603725719440e-08 | improved |
| `kt5.T` | 7.83308351515188406e-07 | 7.10542632731403501e-07 | improved |
| `kt5.ssh` | 3.13194187884571762e-07 | 3.13208255014844894e-07 | worsened |
| `kt5.u` | 5.35330081178481132e-06 | 5.35282723877761324e-06 | improved |
| `kt5.v` | 3.12724047083076055e-06 | 3.12776947753768981e-06 | worsened |
| `kt6.S` | 1.01893071757785947e-07 | 1.02873450202878303e-07 | worsened |
| `kt6.T` | 9.46511249821924139e-07 | 8.46628761763668081e-07 | improved |
| `kt6.ssh` | 2.99562232945575768e-07 | 2.99414141695375502e-07 | improved |
| `kt6.u` | 5.22023153219462873e-06 | 5.21637252011360490e-06 | improved |
| `kt6.v` | 3.65894237981884488e-06 | 3.66857912719028775e-06 | worsened |
| `kt7.S` | 3.24584064514965576e-07 | 2.33569465990512981e-07 | improved |
| `kt7.T` | 8.85916390558350031e-06 | 1.09287084626430442e-06 | improved |
| `kt7.ssh` | 4.63273312332969540e-07 | 4.71311204701266653e-07 | worsened |
| `kt7.u` | 2.78917228440332676e-06 | 2.78837193087161372e-06 | improved |
| `kt7.v` | 4.92839697968303070e-06 | 4.93745126814350982e-06 | worsened |
| `kt8.S` | 3.38478365335959097e-07 | 1.61470595116952609e-07 | improved |
| `kt8.T` | 9.20037180662802712e-06 | 1.00794007451554535e-06 | improved |
| `kt8.ssh` | 4.09196899097546973e-07 | 3.90642211957076962e-07 | improved |
| `kt8.u` | 4.81588507090027762e-06 | 4.85151870532364582e-06 | worsened |
| `kt8.v` | 2.77971665997578388e-06 | 2.78700927702074663e-06 | worsened |
| `kt9.S` | 2.88597810538249178e-07 | 2.62386315341700538e-07 | improved |
| `kt9.T` | 7.80175495052048973e-06 | 9.79886650043226837e-07 | improved |
| `kt9.ssh` | 5.68624840843398593e-07 | 5.62192586994487292e-07 | improved |
| `kt9.u` | 2.51116152653711733e-06 | 2.50669478638228249e-06 | improved |
| `kt9.v` | 3.93806783602346222e-06 | 3.94128898817402984e-06 | worsened |
| `kt10.S` | 3.89135848877231183e-07 | 2.99949292070778029e-07 | improved |
| `kt10.T` | 7.14460109207948335e-06 | 7.90666589978172851e-07 | improved |
| `kt10.ssh` | 4.20843504361084753e-07 | 3.96187362813361921e-07 | improved |
| `kt10.u` | 3.83915270255053009e-06 | 3.80045692572467120e-06 | improved |
| `kt10.v` | 5.30468073692974551e-06 | 5.27587317372332776e-06 | improved |

The old cellwise no-worsening comparison reports FAIL, as expected: Decision
43 expressly permits kt>=2 rows to worsen when the first-over-bar row does not
move earlier and all movement is registered.  It is not the landing veto here.

The month and year measurements are much better than the immutable Round-163
arm:

| day | before T RMS (K) | candidate T RMS (K) |
|---:|---:|---:|
| 30 | 6.5725743747706026e-05 | 2.3276772050683987e-06 |
| 60 | 2.0808020003520387e-04 | 1.4785764394709149e-05 |
| 90 | 1.8643018773756063e-03 | 1.6288518106294924e-05 |
| 120 | 1.0484738105410100e-03 | 1.0963874497720542e-04 |
| 180 | 3.5838854686407826e-03 | 6.1133033794949124e-05 |
| 240 | 1.6448360701178680e-02 | 6.5861718814795174e-05 |
| 300 | 1.3602990041069166e-02 | 5.4748576905951516e-05 |
| 360 | 1.1225660018551306e-02 | 2.6709923853294689e-03 |

All eight year rows improve.  Day 30 improves by 28.24x, day 240 by 249.74x,
and day 360 by 4.20x.  These are candidate measurements only; because the
blast radius is incomplete, they do not establish a new baseline.

## Recipe-derived blast radius and HOLD verdict

The census is derived from each resolved recipe's GM/Redi presence,
`mld_criterion`, and `slope_n2_evaluation`:

| card | executes carried-N2 route in restored production? | candidate implication |
|---|---|---|
| GYRE-zco | no (`recompute`) | candidate changes this card |
| DINO `nemo_dino_kamm` | yes | must be measured |
| DINO `nemo_dino_kamm_mlf` | yes | must be measured |
| LOCK_EXCHANGE-zco | no | not executed |
| OVERFLOW-zps | no | not executed |
| ORCA2-zps | no (`recompute`) | UNMEASURED-with-spec; not executed |
| generic NEMO-GYRE recipe | no (`rho_c`) | not executed |

The cheapest committed one-day DINO Kamm twin was run for both the carried and
recompute control arms.  Each verifies day-0 T/SSH/U/V exactly, verifies the
before-level bridge exactly, and verifies the TKE state and coefficients
exactly.  Each then refuses before its first step with the same named error:

> raw-mesh e3w_int must contain only finite values > 0

This upstream failure means the changed MLD statement did not execute in
either arm.  It is not evidence for or against the candidate's DINO effect.
Under the fail-closed blast-radius rule the candidate is **HELD**, and all
production physics/configuration changes were reverted.  The full private
GYRE gate was also not represented as complete: two early attempts exposed
and fixed an explicit-path bundle omission and an inconsistent synthetic
TKE-disabled control; a final attempt was deliberately aborted when its
stamped tree changed.  The independently complete ladder, month, and year
artifacts above remain valid at their stamped candidate commit.

## Predictions, review, and verification

Predictions 1, 2, 4, 5, and 6 are **CONFIRMED**.  Prediction 3 is **CONFIRMED
with a routing qualification**: the direct N2 operand is non-bit because the
model recomputes it, while the first source difference is NEMO's reuse of the
stored `rn2b`, not different `bn2` arithmetic.  Prediction 7 reaches the HOLD
branch because DINO could not certify the shared statement.

The required read-only `codex exec` pass was attempted.  It produced no
scientific verdict because the in-process app-server could not initialize in
the read-only sandbox.  Its terminal line is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore: **independent review unavailable in-sandbox**.  This is not a SHIP,
HOLD, or DO NOT SHIP verdict.

The citation gate passes all compiled-source mappings.  Its shifted
`GYRE_OMIP_L2_P3_SM_R179SLPWALK/BLD/ppsrc/nemo/zdfmxl.f90:109-124` plant exits
1 with `SYMBOL-NOT-AT-LINE`, as required.

The first focused suite reports **1 failed, 108 passed in 99.81s**.  The sole
failure is
`test_nemo_zdfmxl_transcription.py::test_treguier_kappa_uses_the_same_n2_variant_as_the_slopes`:
its source-inspection assertion still searches `gm_redi_tracer_tendency_latlon`
for a call that the starting tip already placed in
`native_treguier_kappa_for_state`.  Neither the test nor the implementation
differs from `1ffb26a044`; it is base-identical and unrelated to this round.
The same battery with only that test deselected reports **108 passed, 1
deselected in 99.23s**.  This second run includes the year-owner harness,
Decision-43 census, native-slope and mixed-layer transcription tests, literal
slope tests, and the citation regression suite.

| artifact | SHA-256 |
|---|---|
| `daily_record_audit_final2.json` | `f5a80298b103b65e981a103319200549a0a7276c2afe33d7decab20ba1ab93cc` |
| `developed_mxl_walk_initial.json` | `1f9b14c2399e3ce5c3e1a0934660861c07c7e0a554741782adfe21b8535f5504` |
| `developed_mxl_plant_final.log` | `51132447758c49a9d3087e75f0b6ee39a22073d21051c3588a889794a6e15157` |
| `ladder_trajectory.json` | `be326a539f0e30784010f7625e1e5843035fce2b89e5e78a84cf66f2acb92c55` |
| `ladder_comparison.json` | `c3f05bdab4fe94cd871451255f7a324232078e99d1507a8724f9783794829180` |
| `day_gap_candidate.json` | `1793270d5313ba430c9b1ff6ef708faf12944a8feff82d73ec57a7102ce17219` |
| `year_gap_candidate.json` | `5eb7c741dddd96aef1012865cf1634e7b8d3959b82a19f129a977b31bb03f5a5` |
| `dino/carried_day1_final.log` | `cb664771ef9b1d53fd48e397211e21898e16afd6461e3478245d487b38901b93` |
| `dino/recompute_day1.log` | `9345a132e8f1c1fc02963068d17b447a6ad0dcc82bd60767b72572f8b11b9f19` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |
| `citation_gate_final.json` | `2419ee766b248c76ec453af7bf54274f81e87dd014921a65ec2d7e1672bedba4` |
| `citation_gate_shift_plant.json` | `a3bff00d3a20b07fb23738e75574b1ddea88d5207ac48a5b62e1e1f40d165f7c` |
| `focused_tests.log` | `3b3696710f6b07ac4490306c06b402d8c34929ba7a7596eed9ef4a968afea6e7` |
| `focused_tests_round_owned.log` | `787d414255b548bed366ec41ea75ee3f9640586468d9768f03f133adb047f91c` |

Choices made: none.  No configuration choice or carried-state schema changed.

## OPEN — Round 182

Repair the pre-existing DINO bridge blocker before reconsidering this
candidate.  Starting from the compiled DINO coordinate source, identify why
the raw-mesh `e3w_int` presented to the production step contains non-finite or
non-positive land/halo values, and make the bridge retain NEMO's positive
reference geometry there while preserving wet values.  Add a wet/dry/halo
fail-closed gate and a planted invalid cell.  Then rerun the one-day carried
and recompute DINO arms.  Only if both reach the mixed-layer statement may the
Round-181 carried-N2 candidate be reapplied and rerun through the local proof,
complete GYRE ladder/month/year gate, and recipe-derived blast-radius gate.
No NEMO acquisition is presently required; request one only if the existing
compiled DINO mesh and restart cannot discriminate the bridge producer.
