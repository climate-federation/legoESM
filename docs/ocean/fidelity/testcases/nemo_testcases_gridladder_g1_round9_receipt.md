# Grid ladder rung G1 — round 9 receipt and CERTIFICATION (gridladder_r9)

Status: **LANDED** (three cited statements, G1-card-selected). The rung is
**CERTIFIED at the 1e-15 bar**: the 100-step record has no row over the bar
in either label, and outside one registered environment row (steps 8 to 11)
every scored field is NEMO's bit for bit.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round9.md
(commit 0f657b167, before any measurement).
Code: b19b8dd72 (the three options), 9765538e7 (ladder census and
environment arm), 1594b2379 (tests), and this receipt's commit (review
note, one ladder key).
Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round9/final/.
Record: the operator's round-4 acquisition
(.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step). NEMO was not
re-run; **no acquisition is needed**.

## Take-aways

1. **Headline, first step over the 1e-15 bar: none, on any field, in 100
   steps (INDEPENDENT); none in steps 1 to 50 (GIVEN-NEMO-ENTRY).** Round 8:
   sea level 57, V 85, U 97.
2. **Steps 12 to 100: 0 unequal cells on sea level, both velocities, both
   barotropic velocities, T and S.** Steps 1 to 7: 0. Steps 8 to 11: the
   tiny-number class only, largest difference 4.7e-289.
3. **Items 1 and 3 were one statement family.** NEMO interpolates the
   thickness ratios it stores; the card took the ratio of the interpolated
   sea level. T point: step 40. U and V faces: steps 5 and 21.
4. **The second cell at step 40 was a different statement:** the depth the
   density reads. Compiled as one graph, two constant factors were folded
   and the depth was never rounded. 3,474 cells differ in the scaled depth;
   one changes the density.
5. **Item 2 (step 57) was not made at step 57.** It disappears with the
   three statements; each of two of them alone brings an over-bar sea level
   back (steps 63 and 76).
6. **The tiny-number class is measured and registered:** NEMO's build keeps
   subnormal numbers, the card's executable returns zero for them. First at
   step 8, stage 3, in the vorticity trend, from bit-equal operands.
7. **ORCA2 needs no extra domain array for statement F.** Its domain file
   holds no U or V face areas; NEMO computes e1 times e2, as the card does.
   Round 8's caveat was wrong.

## 0. Admission and controls

- No new record. Round 8's card is this round's before picture (its
  receipt: 57 / 85 / 97).
- Score plant exits 1 (plant_score.log).
- Each instrument was scored against NEMO before use: the stage replay
  against NEMO's stage frames, 0 unequal at steps 5, 8, 21, 40; the term
  split's sum against the card's exposed right-hand side, 0 unequal; the
  stage statements replayed in NEMO's order, 0 unequal at step 40
  (stage_replay_kt40.log).
- Offline, the production pressure-gradient function on NEMO's own stage-3
  operands at step 40 equals the replay on every face, compiled and not
  (probe, uncommitted). So the difference was in an operand.

## 1. Item 1: the T-point ratio (step 40, cell (55, 83))

NEMO: `r3t(Kaa) = r1_2 * ( r3t(Kbb) + r3ta )`
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217`), read by stage
2's tracer update and, as Kmm, by stage 3's pressure gradient
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:380-394`) and density.

Option: rk3_stage_r3t_nemo_literal (model config, default False). One plus
the mean of the two stored ratios is handed to stage 2's tracer weight, and
to the pressure-gradient stretch and the density depth of stage 3. Stage 2
receives the N+1/3 twin
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:171-173`) the same way;
that half was NOT in the preregistered candidate (section 11, UNASKED 1).

GIVEN-NEMO-ENTRY, step 40, unequal cells (stages_kt40_*_dirty.json):

| row | option off | option on |
|---|---|---|
| stage 2, T and S | 1 and 1, cell (55, 83) | 0 and 0 |
| stage 3, right-hand side, U / V faces | 4 / 4 | 2 / 2, the faces of cell (101, 65) |
| stage 3, final U / V | 2 / 1 | 1 / 1 |

## 2. The second cell (101, 65): the density's depth

Term split at stage 3: pressure gradient 2 U and 2 V faces, vorticity 0.
The four faces are the faces of one T cell.

| question | instrument | reading |
|---|---|---|
| is the handed stretch NEMO's? | helper against NEMO's recorded ratio | 0 unequal |
| is it the depth's form (quotient, ratio of the mean sea level)? | stage_operand_replay, steps 1 to 50 | depth differs on 4 cells in 50 steps; density 0 unequal. **No** |
| is it the density path? | the model's own function on NEMO's stage-2 frame | uncompiled 0 unequal; compiled 1, cell (101, 65) |
| which intermediate? | depth and density compiled apart, then together | apart: 0. Together: scaled depth 3,474 cells, density 1 cell |

The last two rows are uncommitted probes; the committed test
(test_density_depth_is_the_rounded_product_only_with_the_option) pins the
same thing: more than 1,000 cells with the option off, 0 with it on.

NEMO: `zh = ((gdept_1d(jk))*(1._wp+r3t(ji,jj,Knn))) * r1_Z0`
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/eosbn2.f90:324` for the step's first
call, `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/eosbn2.f90:559` for the stage call
at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:323`). Two rounded
products. The compiled card multiplied the stretch by the two constants'
product.

Option: nemo_eos_gdept_rounded_product (model config, default False).
Result: step 40 is bit-identical at every stage (stages_kt40_card2_dirty).

## 3. Item 3: the stage-3 faces at steps 5 and 21

- V-P2 **REFUTED**: the unequal term is the vorticity term, not the pressure
  gradient. Step 5: V faces (77, 46) and (77, 47). Step 21: U faces (76, 60)
  and (77, 61).
- Each pair shares one transport. Offline (stage_operand_replay): at step 5
  stage 3, one plus the U ratio differs on exactly one face, (77, 46); at
  step 21 stage 3, one plus the V ratio on exactly one face, (76, 61).

NEMO: `r3u(Kaa)` and `r3v(Kaa)` are the same interpolation of stored ratios
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:171-173`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217`); the transports
read them (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:766-767`).

Option: rk3_stage_r3uv_nemo_literal (model config, default False), handed
to the stage velocity weights and the stage face thickness of stages 2 and 3.
Result: steps 5 and 21 are bit-identical at every stage.

One plus the ratio, faces unequal between the two forms, steps 1 to 50:
stage 2 at steps 7, 12, 15, 24, 43 (two), 45; stage 3 at steps 5, 6, 21, 22.
One face each otherwise. The bare ratios differ on 3,500 to 10,000 faces.

## 4. The 100-step record

First step leaving the 1e-15 bar; largest normalised error. INDEPENDENT.

| card | ssh | u | v | T | worst ssh | worst u | unequal cells after step 11 |
|---|---|---|---|---|---|---|---|
| round 8's card (its receipt) | 57 | 97 | 85 | none | 1.282e-14 | 1.351e-15 | yes |
| **the card (three options)** | **none** | **none** | **none** | **none** | 9.1e-305 | 1.4e-305 | **0** |
| without the T ratio | 76 | 98 | 91 | none | 1.172e-14 | 1.331e-15 | yes |
| without the density depth | 63 | none | none | none | 1.076e-14 | 9.437e-16 | yes |
| without the U and V ratios | none | none | none | none | 5.334e-17 | 1.041e-17 | yes |

The card was run three times (b19b8dd72, 9765538e7, final tree): 0 of 598
rows differ between runs.

GIVEN-NEMO-ENTRY, whole step, steps 1 to 50 (given_entry_50.json; the
record's entry frames end at step 50): no row over the bar except T and S at
step 1, which is NEMO's land values (10,207 cells at 20, the land count), as
in every round. Unequal cells only at steps 8 to 11.

## 5. Item 2: the step-57 event

- S-P1 **NOT TESTED as worded**: the given-entry arm stops at step 50.
- What was measured instead: with the three options the cell (67, 76) and
  every other cell is bit-equal at step 57. Removing the density-depth
  option alone puts sea level over the bar at step 63; removing the T ratio
  alone, at step 76. So the event is the accumulated last-place differences
  of these statements, not a statement at step 57.

## 6. Item 4: the tiny-number class (registered)

**Registered environment row.** Steps 8 to 11; sea level, both velocities,
both barotropic velocities; largest difference 4.7e-289, normalised
9.1e-305; absent before step 8 and after step 11.

| question | reading |
|---|---|
| which numbers | NEMO holds nonzero values from 2e-323 to 1.1e-308, all below the smallest normal double (2.2251e-308). The card holds exactly zero there, and holds no subnormal number anywhere in 100 steps |
| how many | step 8: 2 cells (U). Steps 9 to 11: 126 flushed cells in all, plus 159 descendants: normal numbers below 4.7e-289 that differ because an input was flushed |
| which statement first | step 8, stage 3, the vorticity trend (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:779-784`): 9 U and 7 V faces, largest 1.8e-308. Then the stage correction (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`): 2 faces |
| are the operands equal | yes: step 8's entry and its stage-1 and stage-2 frames are 0 unequal. Normal operands, subnormal exact result |
| NEMO's side | arch file line 27 (cfgs/ORCA2G1_OMIP_L1_P3/BLD/arch_nemo.fcm): -O3, no fast-math, no flush flag, no -march. Read off the flags, not measured in the binary. Its record holds subnormals, which is the measurement |
| the card's side | measured (subnormal_environment.json): a compiled product with a subnormal result returns 0; a subnormal operand times 1e10 returns 0; the same in uncompiled jax.numpy; numpy in the same process keeps both, before and after |

- N-P1 HOLDS at step 8; **REFUTED** for steps 9 to 11 (descendants are
  normal numbers). N-P2, N-P3, N-P4 HOLD.
- The pressure gradient at step 8 stage 3 is 0 unequal.

## 7. Item 5: statement F on ORCA2

- NEMO takes the U and V face areas from the domain file only if it holds
  a variable `e1e2u` (and `e1e2v`); otherwise it computes e1u times e2u and
  e1v times e2v
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:157-160`), then the
  reciprocals (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:170-171`).
- ORCA2's domain file (ORCA_R2_zps_domcfg.nc, the copy in
  orca2_rounds/round1/acquisition/orca1ice_surface_every_step_a_np2) holds
  **no** `e1e2u` or `e1e2v`. That run's ocean.output prints "u- & v-surfaces
  calculated as e1 e2 product". G1's own domain file and log: the same.
- So statement F on ORCA2 needs `e1u`, `e2u`, `e1v`, `e2v` (domain_cfg
  names; mesh_mask carries the same names) and nothing else.
- The ORCA2 card's raw-mesh builder already forms the two products from
  those four arrays (recipe, the three nemo_e1e2 lines of
  build_orca2_zps_card), and the resolved builder in vertical.py reads them.
  Statement F's helper uses e1 times e2: correct for ORCA2 as it stands.
- One file and one log were checked, not every ORCA2 build.

## 8. Gates for the landing

- No AT-BAR row leaves the bar in either label; no first step over the bar
  moves earlier; nothing to register under Decision 96.
- Three options, default False, appended last, selected only by the G1 card
  (test). Caller grep: each name appears in its field, its reader, the G1
  card (selection and validator), the ladder's before arm and the tests.
- Searched before building (grep of the model, the tendency module, the
  vertical module and the density module): the stage-1 T stretch, the F
  ratio interpolation, the face-geometry builder, the rounding guard and the
  operator-exposure hook existed and are reused. New code: five small
  helpers at the ends of two files; three in-place parameters.
- Line counts unchanged in the vertical, density and recipe files.
- Closed cards (GYRE, VORTEX, TSUNAMI, ORCA2, DINO) NOT re-scored. Default
  paths read as unchanged, not measured.

## 9. ORCA2 pointer

| statement | G1 | does ORCA2's deck reach it? |
|---|---|---|
| T ratio interpolated, handed to its readers | landed, selected | yes, same stage statements. UNMEASURED |
| U and V ratios interpolated | landed, selected | yes. **Two readers are not covered** (review, section 12): the stage transport geometry and the implicit vertical solve's interface thickness. Both matter with more than one level or advected tracers |
| density depth rounded before the scale | landed, selected | yes, every level. UNMEASURED |
| statement F (round 8) | landed | no extra arrays needed (section 7) |
| subnormal treatment | registered | yes wherever a signal front passes 1e-308 |

## 10. Predictions

| id | result |
|---|---|
| R-P1 stage statements in NEMO's order | HOLDS, 0 unequal at step 40 |
| R-P2 stage-2 T and S and the faces of (55, 83) | HOLDS |
| R-P3 the second cell stays | HOLDS; owner: the density depth |
| R-P4 T and S through 100; sea level through 40 | HOLDS (through 100) |
| R-P5 stage-1 twin differs on fewer than ten cells a step | HOLDS: at most one (steps 15, 23, 35). Measured on the depth product, not on the bare sum |
| S-P1 one step from NEMO's step-57 entry | NOT TESTED (frames end at step 50) |
| S-P2 the chain's first difference is the owner | HOLDS in the form of section 5 |
| V-P1 replay control | HOLDS |
| V-P2 the term is the pressure gradient | **REFUTED**: the vorticity term |
| V-P3 item 1's option does not remove them | HOLDS |
| N-P1 every member subnormal against zero | HOLDS at step 8; **REFUTED** at 9 to 11 |
| N-P2 no flush flag in NEMO's build | HOLDS (read) |
| N-P3 the compiled product returns zero | HOLDS |
| N-P4 operands bit-equal at the first step | HOLDS |

## 11. Corrections to my own work, and choices

Corrections:
1. V-P2 was wrong, and so was round 8's PLAUSIBLE reading behind it.
2. My first reading of cell (101, 65) was the depth's form. The offline arm
   refuted it before any code was written for it.
3. Round 8's receipt said ORCA2 reads its face areas from the domain file.
   It does not (section 7).
4. A new row key broke an existing test's reading of the stage table, and a
   loop variable shadowed a helper. Both fixed; the affected tests re-run.
   The step-5, 21 and 40 tables were taken before the key was renamed; only
   the key's name changed.
5. The first review launch waited on standard input and ended on its own
   timeout; it was re-run with input closed.

ASKED (round brief): the five items in order; cite; land card-selected;
re-score after landing; certify if at the bar; tests; one review.

UNASKED, each offered for revert:
1. Stage 2's readers also receive NEMO's N+1/3 T ratio. The preregistration
   kept it out. Reason: the same cited statement, and the offline count
   shows the operand differs (steps 15, 23, 35).
2. The density-depth option: a statement not on the brief's list; it owned
   the second cell.
3. The re-score was done once, after all three options, not after each.
   The three before arms supply the per-statement attribution.
4. Three option-position pins re-pinned (three fields appended).
5. The F-ratio helper now calls the shared interpolation helper (same
   arithmetic; the step-2 stage tests pass).
6. Review findings 1 and 2 are registered, not coded (section 12).
7. The certification is declared with GIVEN-NEMO-ENTRY covering steps 1 to
   50 and INDEPENDENT covering 1 to 100.

## 12. Tests and review

Single review (codex): **findings**. Log:
.../gridladder_rounds/round9/codex_review_r9.log (its last lines).
- "CONFIRMED — Stage-2/3 transport geometry still derives `e3u/e3v(Kmm)`
  from interpolated SSH: `_nemo_ws_stage_transport` never receives
  `nemo_stage_pair`". True. Not coded: on this one-level deck with no tracer
  advection it reaches no scored field, so no walk names it. Recorded in the
  option's comment and in OPEN.
- "CONFIRMED — Stage-3 implicit ZDF independently reconstructs `r3u/r3v`
  from `_eta_now`". True. Not coded: with one level that array is empty.
  Recorded the same way.
- "CONFIRMED — Round-9 tests are vacuous for both omissions". True: there
  is nothing to test until they are coded.
- "VERDICT: findings". Nothing was reported against stage levels, the pair
  order, the default paths, the ladder's new arms or the other tests.
- A single review, not a dual one (the lane's rule 6). Claim review before
  code: NOT done.

Focused tests, synchronous:
- 13 new round-9 tests: 13 passed (tests_g1_r9_new.log). Each option off is
  a plant in the 100-step test; the step-5 term test has its own plant.
- The other 68 G1 ladder tests and the tools file: 88 passed, 1 failed
  (tests_g1_r9_rest.log); the failure was my new row key. After the fix the
  5 affected tests pass (tests_g1_r9_fix.log). The whole file was NOT re-run
  after the fix.
- Seven ratchet files: 11,767 passed, 4 skipped, 4 failed
  (tests_ratchets.log): the same four as rounds 1 to 8.
- The coefficient-builder callers: 20 passed, 1 failed
  (tests_builder_callers.log): the same test as round 8.
- NOT RUN: any closed-card ladder; the wider model, vertical and density
  unit selections; the inline-import budget.

Compliance, stated because no gate enforces it: search-before-build
(section 8); one variable per before arm; instruments checked before
quoting (section 0); non-vacuity (plants above).

Stamps: record100_1_card and its three before arms, given_entry_50 and
plant_score are b19b8dd72; record100_2_card, stages_kt8 and
subnormal_environment are 9765538e7; record100_3_card and stage_replay_kt40
are the final tree. The step-5, 21 and 40 stage tables and the 50-step
operand table are dirty-tree runs, one level up from final/.

UNVERIFIED:
- that no closed card moves (default paths read, not run);
- the ORCA2 column of section 9;
- NEMO's subnormal handling in the binary itself (flags and record only);
- that every ORCA2 build's domain file lacks the face-area variables.

## CERTIFICATION

Rung G1 (the TSUNAMI program on ORCA2's horizontal grid) is certified on
the operator's 100-step record at the campaign's 1e-15 bar:

- INDEPENDENT, steps 1 to 100: sea level, U, V, barotropic U and V, T, S:
  no row over the bar; largest normalised error 9.1e-305.
- GIVEN-NEMO-ENTRY, steps 1 to 50: the same, apart from the step-1 T and S
  row that is NEMO's land values.
- Bit-identical on every scored field at every step except the registered
  environment row of section 6.

Limits of the claim: one record, one machine; closed cards not re-scored;
single review.

## OPEN (for the lane's next rung, not for G1)

1. Hand the interpolated U and V ratios to the stage transport geometry and
   to the implicit vertical solve's interface thickness; cite the level
   each reads. Needs a deck with more than one level (G2).
2. The 50-step limit of the given-entry arm (the record's entry frames).
3. ORCA2 import of this round's three options and of statement F.

## Citation gate and final tree (run after the receipt's first commit)

PASS: 11 citations, 0 unmapped, whole-map audit clean
(final/citation_receipt.json); the shifted-citation plant exits 1
(final/citation_plant.json). Four map entries are new this round (the two
density depth statements, the stage's density call, the face-area branch);
the other seven were mapped by earlier rounds.

Final tree d4ba35fa0 (everything after it is this receipt's text):
record100_3_card.json has no row over the bar and differs from the first
run in 0 of 598 rows; the stage statements replayed in NEMO's order at step
40 are 0 unequal on all 15 rows.

SHA-256 prefixes, final/: citation_plant.json=7a701ee4c1c5ca84, citation_receipt.json=2d88ee871c4e27bd, given_entry_50.json=b4866f0c55570f37, plant_score.json=c73ac0d39d3ee629, record100_1_card.json=6fc8622b49ad4c51, record100_1_without_eos_gdept.json=ad9dfeaaa58ba257, record100_1_without_stage_r3t.json=2112d844dc7d6d53, record100_1_without_stage_r3uv.json=60c7e1e4a3786666, record100_2_card.json=d7a6586f82eb2f8c, record100_3_card.json=c5c17bd56bef4fc7, stage_replay_kt40.json=905ed006fb917141, stages_kt8.json=b3c468a0f984b133, subnormal_environment.json=e3e3eea45cbed332.
One level up: codex_review_r9.log=c9a27ddb1f8a2380, tests_g1_r9_new.log=962d6ff4032c162c, tests_g1_r9_rest.log=26ab5516fba877a7, tests_g1_r9_fix.log=3e8ab0f6a7a71ac3, tests_ratchets.log=5479270827244b00, tests_builder_callers.log=99a08c462722fa0c, stages_kt40_before_dirty.json=3fd2ce324d52e8a5, stages_kt40_card_dirty.json=404ef94f27ffeeb1, stages_kt40_card2_dirty.json=3c44af389decd334, stages_kt5_card2_dirty.json=b4d7cea7d8fafcc4, stages_kt21_card2_dirty.json=79b695f190e2c1e7, stages_kt5_card3_dirty.json=20a0bfdc1cbb0d5e, stages_kt21_card3_dirty.json=b3adb66709ef4eb7, stage_operand_1_50_dirty.json=4bd49c709fcb5736.
