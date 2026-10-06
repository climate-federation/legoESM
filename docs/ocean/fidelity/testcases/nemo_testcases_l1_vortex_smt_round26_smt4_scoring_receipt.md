# Receipt — VORTEX_SMT round 26 (lane round 238): SMT-4 scoring and first boundary

**Status: HELD.** The SMT-4 record is admitted, the explicit card and its
partial-cell geometry are complete, and the 10-step and 100-day trajectories
are registered. No physics statement landed. The preregistered prediction
that the new debt first appears after `dyn_ldf` is **REFUTED**: the first
production non-bit boundary is the already-known HPG accumulator association.
NEMO source order makes HPG bit-exact, then leaves only last-bit VOR and a
`4.73e-13 / 1.62e-13 m s-2` U/V residual before lateral diffusion; adding
`dyn_ldf` does not increase either maximum.

Base: `96204ad3d` (round 237). Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round238/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round26_smt4_scoring.md`.

## 1. Driver repair and record admission

The shared VORTEX driver now creates each target directory before asking `df`
for that target's filesystem. The regression test pins the order
`loop -> mkdir -> df -> done`; removing the new line makes its source lookup
fail. No deck, writer, build target, or record changed.

The operator-completed arms required no rebuild or rerun. Their existing
admission reports say:

| arm | status | parsed records | step-10 restart vs plain build |
|---|---|---:|---|
| kt=1..10 + stage terms | ADMITTED | 27 | byte-identical |
| 100-day daily restarts | ADMITTED | 3,067 | byte-identical |

Both `run.user.log` files end in `STOP 0` and `RUN_DONE`. Re-running the
wrapper produced `ROUND237_SMT4_KT1_10_ALREADY_ADMITTED`,
`ROUND237_SMT4_DAY100_ALREADY_ADMITTED`, then
`ROUND237_SMT4_RECORD_READY`. The header, field-name, and truncation plants
remain nonzero controls in the admitted reports; the checker parses every
group's declared rank and extents to EOF and never predicts a byte count.

## 2. Resolved card and compiled program

The new explicit card is `VORTEX_SMT4_VEC-zps`. It inherits SMT-3 and changes
only lateral momentum diffusion: div-rot type 0, level Laplacian, coefficient
mode 20, `rn_Uv=0.1 m/s`, `rn_Lv=10 km`, and `rn_ahm_b=0`. The compiled
target declares the complete `namdyn_ldf` tuple and reads reference then card
namelists at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:177-185`.
For z/partial-step geometry, Laplacian plus level direction resolves to
`np_lap` at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:221-278`.
Mode 20 forms `zUfac=0.5*rn_Uv` and calls `ldf_c2d` at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/ldfdyn.f90:311-346`.

The compiled dispatcher sends `np_lap` to `dynldf_lev_lap` at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf.f90:81-90`.
That routine forms the `zwf` curl and `zwt` divergence with live partial-cell
thicknesses, then adds the U/V tendencies with live `Kmm` face divisors at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.
The stage program's actual execution order is HPG, VOR, ADV at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344`,
then stage-3 `dyn_ldf` at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:398-404`.

No hidden choice was introduced. Decision 93 authorised this exact mode-20
stand-in for ORCA2 rung 0's unavailable file-backed coefficient; every other
scientific option remains SMT-3's explicit value.

## 3. Geometry identity

The common geometry gate reports **GEOMETRY IDENTICAL** on all 18 rows:
`k_bot`, reference `e3t/e3u/e3v/e3f`, card operands, resolved QCO U/V faces,
northern/eastern boundary controls, `e3w`, all three masks, and the card's
initial geometry. The face-thickness check is non-vacuous: 1,164 cells have
resolved `e3u_0 != e3t_0`, yet every face array equals NEMO bit-for-bit.

## 4. Certified 10-step ladder

All five kt=1 rows remain AT-BAR. First-over-bar is kt=2 on T/u/v/ssh; S stays
AT-BAR through kt=3 and first exceeds the bar at kt=4. The complete SMT-4
registry is:

| kt | T | S | U | V | SSH |
|---:|---:|---:|---:|---:|---:|
| 1 | `0` | `0` | `2.220446049250313e-16` | `2.220446049250313e-16` | `1.355252715606881e-20` |
| 2 | `6.957539680188801e-10` | `6.090366306515142e-16` | `3.215977248394175e-09` | `6.826891959729742e-10` | `3.938387393986886e-10` |
| 3 | `2.967710564385140e-09` | `8.120488408686854e-16` | `1.020709123819574e-08` | `1.655891325967607e-09` | `6.240860606077092e-10` |
| 4 | `2.415312741199901e-09` | `1.015061051085857e-15` | `2.308118721844332e-08` | `3.490460296663722e-09` | `8.106746940406140e-10` |
| 5 | `4.951120894428506e-09` | `1.015061051085856e-15` | `3.639640225316931e-08` | `6.094295553881607e-09` | `1.105926239475252e-09` |
| 6 | `8.197556015446484e-09` | `1.421085471520199e-15` | `5.428826766948336e-08` | `1.141989562948044e-08` | `1.355555334647818e-09` |
| 7 | `1.224529682418209e-08` | `1.218073261303027e-15` | `1.435806865693334e-07` | `2.184259867200639e-08` | `3.376272808205960e-09` |
| 8 | `1.724596761302958e-08` | `1.421085471520198e-15` | `3.747712989665697e-07` | `4.044365841488970e-08` | `7.740548139956172e-09` |
| 9 | `2.435821570460983e-08` | `1.624097681737369e-15` | `6.701961815409885e-07` | `1.093947741229867e-07` | `1.053818847853230e-08` |
| 10 | `3.463009962357514e-08` | `1.624097681737369e-15` | `8.309995438229856e-07` | `2.098501566084529e-07` | `1.297493420343576e-08` |

The registry comparator keys all 50 rows and deliberately refuses a missing
row. Relative to the SMT-3 registry, 42 rows move: 23 have a smaller and 19 a
larger residual against their **own card's** NEMO oracle; there are zero
AT-BAR/DEBT status changes. These are two physical decks, so direction is a
registry description, not a candidate/control claim. Every movement is:

| row | SMT-3 | SMT-4 | direction vs own oracle |
|---|---:|---:|---|
| kt2 T | `6.957537701781045e-10` | `6.957539680188801e-10` | away |
| kt2 ssh | `3.938355197519172e-10` | `3.938387393986886e-10` | away |
| kt2 u | `3.178975381545879e-09` | `3.215977248394175e-09` | away |
| kt2 v | `6.728428985844359e-10` | `6.826891959729742e-10` | away |
| kt3 T | `2.839158383429883e-09` | `2.967710564385140e-09` | away |
| kt3 ssh | `6.231736793260723e-10` | `6.240860606077092e-10` | away |
| kt3 u | `1.013289665369177e-08` | `1.020709123819574e-08` | away |
| kt3 v | `1.654683991821754e-09` | `1.655891325967607e-09` | away |
| kt4 S | `1.015061051085856e-15` | `1.015061051085857e-15` | away |
| kt4 T | `2.417274950323488e-09` | `2.415312741199901e-09` | toward |
| kt4 ssh | `8.099967918617779e-10` | `8.106746940406140e-10` | away |
| kt4 u | `2.285250661274163e-08` | `2.308118721844332e-08` | away |
| kt4 v | `3.529006063502510e-09` | `3.490460296663722e-09` | toward |
| kt5 T | `4.965180625967719e-09` | `4.951120894428506e-09` | toward |
| kt5 ssh | `1.104131674978248e-09` | `1.105926239475252e-09` | away |
| kt5 u | `3.626384267874094e-08` | `3.639640225316931e-08` | away |
| kt5 v | `6.223943748405458e-09` | `6.094295553881607e-09` | toward |
| kt6 S | `1.218073261303028e-15` | `1.421085471520199e-15` | away |
| kt6 T | `8.243512792624066e-09` | `8.197556015446484e-09` | toward |
| kt6 ssh | `1.297312923753680e-09` | `1.355555334647818e-09` | away |
| kt6 u | `5.425727961361204e-08` | `5.428826766948336e-08` | away |
| kt6 v | `1.154071884172832e-08` | `1.141989562948044e-08` | toward |
| kt7 S | `1.421085471520199e-15` | `1.218073261303027e-15` | toward |
| kt7 T | `1.235688425850942e-08` | `1.224529682418209e-08` | toward |
| kt7 ssh | `3.422397134786315e-09` | `3.376272808205960e-09` | toward |
| kt7 u | `1.512125080566085e-07` | `1.435806865693334e-07` | toward |
| kt7 v | `2.217514574689875e-08` | `2.184259867200639e-08` | toward |
| kt8 S | `1.624097681737370e-15` | `1.421085471520198e-15` | toward |
| kt8 T | `1.747103006144312e-08` | `1.724596761302958e-08` | toward |
| kt8 ssh | `7.703695974539682e-09` | `7.740548139956172e-09` | away |
| kt8 u | `3.833577468595473e-07` | `3.747712989665697e-07` | toward |
| kt8 v | `4.287059453614295e-08` | `4.044365841488970e-08` | toward |
| kt9 S | `1.827109891954541e-15` | `1.624097681737369e-15` | toward |
| kt9 T | `2.455230544376252e-08` | `2.435821570460983e-08` | toward |
| kt9 ssh | `1.050440961948215e-08` | `1.053818847853230e-08` | away |
| kt9 u | `6.684648320137095e-07` | `6.701961815409885e-07` | away |
| kt9 v | `1.127359968592872e-07` | `1.093947741229867e-07` | toward |
| kt10 S | `1.827109891954540e-15` | `1.624097681737369e-15` | toward |
| kt10 T | `3.498342973430311e-08` | `3.463009962357514e-08` | toward |
| kt10 ssh | `1.316862729278112e-08` | `1.297493420343576e-08` | toward |
| kt10 u | `8.402824479748999e-07` | `8.309995438229856e-07` | toward |
| kt10 v | `2.181972887413521e-07` | `2.098501566084529e-07` | toward |

The eight unchanged rows are the five kt1 rows plus kt2 S, kt3 S, and kt5 S.
The missing-row registry plant exits 1 with `STATUS PLANT-FIRED`.

## 5. Production-JIT first-boundary walk

The walk starts stage 3 from NEMO's recorded stage-2 output and substitutes
NEMO's external-mode handoff. It then scores cumulative HPG, VOR, ADV,
pre-LDF, and post-LDF frames from the production-jitted step.

| boundary | U unequal / max abs | V unequal / max abs |
|---|---:|---:|
| HPG | `36,522 / 6.728440153286030e-06` | `36,522 / 6.716707271979775e-06` |
| VOR | `36,492 / 6.728440153286036e-06` | `36,522 / 6.716707271979778e-06` |
| ADV / pre-LDF | `22,611 / 4.732338229374326e-13` | `22,335 / 1.620657914366122e-13` |
| post-LDF | `22,582 / 4.732338229374326e-13` | `22,454 / 1.620657914366122e-13` |

Thus R26-P4 is refuted at its explicit falsifier: pre-LDF is not bit-exact.
The first non-bit statement is the accumulator association at HPG. NEMO
overwrites with HPG, then accumulates VOR and ADV in the compiled order cited
above; legoESM's ordinary combined evaluation has already associated the KEG
part when that HPG frame is exposed. The one-variable source-order arm makes
both HPG rows BIT. Under that arm, VOR is the next non-bit boundary at only
`2.710505431213761e-20 / 1.355252715606881e-20 m s-2`, and pre-LDF remains
`4.732338229374326e-13 / 1.620657914366122e-13 m s-2`. This reproduces the
round-193 mechanism on the partial-cell rung rather than assuming it.

The HPG plant moves exactly one scored production cell by one ULP, reports
`plant_movement_cells=1`, prints `STATUS PLANT-FIRED`, and exits 1. It cannot
pass merely because the clean HPG row is already non-bit.

No source-order candidate is landed. Round 193 already measured this cited
association as locally exact but trajectory-inert on the flat vector card;
this round does not promote it without an SMT-4 trajectory measurement. The
LDF internal operand walk is not opened because the admitted boundary record
shows no new maximum at `dyn_ldf`; naming `zwf`, `zwt`, or a thickness factor
would be post-hoc inference.

## 6. Shipped 100-day comparison

The scorer reproduces the certified kt=1..10 values before accepting the daily
curve. RMS and maxima against NEMO are:

| day | T rms / max K | U rms / max m/s | V rms / max m/s | SSH rms / max m |
|---:|---:|---:|---:|---:|
| 1 | `1.405693975e-07 / 1.656655910e-05` | `6.286792087e-08 / 6.134416567e-06` | `7.431603295e-08 / 6.945482528e-06` | `1.104148579e-08 / 1.825341884e-07` |
| 2 | `5.100120075e-07 / 2.762692694e-05` | `1.065039598e-07 / 5.349402916e-06` | `1.162550943e-07 / 1.398670522e-05` | `2.408124881e-08 / 2.041301283e-07` |
| 5 | `4.424519998e-06 / 4.717998696e-04` | `1.775484908e-06 / 2.066524131e-04` | `1.557389852e-06 / 1.884112806e-04` | `1.787936677e-07 / 1.112697741e-06` |
| 10 | `2.454180680e-05 / 2.356847707e-03` | `7.312516990e-06 / 8.627041680e-04` | `7.914167801e-06 / 6.414736349e-04` | `7.704382891e-07 / 4.035443991e-06` |
| 20 | `7.226803498e-05 / 6.097387029e-03` | `1.608965554e-05 / 1.382646873e-03` | `1.741920380e-05 / 1.154321811e-03` | `3.463151482e-06 / 1.438721865e-05` |
| 30 | `1.450215284e-04 / 1.185073149e-02` | `2.673795366e-05 / 1.377274492e-03` | `2.985170460e-05 / 1.903269130e-03` | `7.380713703e-06 / 3.743847510e-05` |
| 60 | `2.003482609e-04 / 9.383780316e-03` | `2.947599702e-05 / 1.120145107e-03` | `2.982349828e-05 / 2.086656129e-03` | `1.247144123e-05 / 8.324561119e-05` |
| 100 | `2.552708052e-04 / 9.830831628e-03` | `2.834815879e-05 / 9.356145245e-04` | `2.715247914e-05 / 9.952453748e-04` | `1.171711158e-05 / 7.429504203e-05` |

This is a measurement baseline, not an acceptance-bar claim.

## 7. Blast radius, review, citations, and tests

The new card is selected only by its new case name; no existing card's
resolved configuration changes. The required private-workdir DINO month gate
reports:

> `DINO from-rest month day-30 wet 3-D T rms vs NEMO kt=960: 2.056821682e-03 K against bar 2.244317642e-03 K (certified 2.040288765e-03 K) -- PASS`

This equals the inherited post-SMT-3 lane value, so the card addition does not
move DINO. Tanks, flat VORTEX, existing SMT cards, GYRE, and ORCA2 cannot
dispatch the new card name; no shared physics changed. The geometry builder,
operator, and existing card branches are reused rather than duplicated.

The required separate read-only Codex review was attempted on the complete
committed diff. Verbatim result:

> `WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)`
> `Reading additional input from stdin...`
> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

**Independent review unavailable in-sandbox.** It produced no `SHIP`, `HOLD`,
or `DO NOT SHIP` verdict. No physics is landed, and the round's HELD outcome
does not depend on treating review silence as approval.

The focused suite covering the acquisition driver, SMT cards, shared
prognostic/barotropic path, and generic NEMO recipe reports:

> `65 passed in 474.24s (0:07:54)`

Citation-gate and shifted-plant results are added after the committed receipt
is reviewed.

## 8. OPEN — next round

1. Measure the one-variable NEMO source-order association on the **SMT-4
   50-row ladder and 100-day curve**. It is locally BIT at HPG here, but the
   flat-card trajectory was inert in round 193; magnitude, not local bits,
   decides whether it is relevant to this partial-cell rung.
2. If source order is again trajectory-inert, do not walk its last-bit VOR
   remainder. Rank the SMT-4 100-day growth by the existing process-family
   substitution method; only open a `dynldf_lev` operand acquisition if that
   ranking actually points back to lateral momentum diffusion.
3. ORCA2 pointer: its rung-0 level-Laplacian operator executes the same
   compiled `dynldf_lev` equations, but with a file-backed 3-D coefficient
   rather than SMT-4's Decision-93 mode-20 stand-in. This receipt licenses the
   geometry/card machinery and identifies the upstream HPG association; it
   licenses no ORCA2 physics fold-in.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
