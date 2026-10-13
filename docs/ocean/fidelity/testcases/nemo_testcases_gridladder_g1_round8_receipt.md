# Grid ladder rung G1 — round 8 receipt (gridladder_r8)

Status: **LANDED** (two cited statements, G1-card-selected). The rung is
**NOT certified**: the 100-step record is not at the bar.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round8.md
(commit fd083cd6b; it lists the locating walk of item 1 that preceded it).
Code: dd64f53e5 (solver ratio order), ff5b917f1 and a41d336e7 (stored
T-point ratio, two sites), 2c02b4257 (tests, ladder), 2efcb911f (review
fix). Evidence:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round8/final/.
Record: the operator's round-4 acquisition
(.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step). NEMO was not
re-run and **no acquisition is needed**: every question of this round was
separated offline.

The operator's ruling of 2026-10-13 (Decision 96) is applied: round 7's
statements and the entry-depth statement stay selected and are not re-asked.

## Take-aways

1. **Headline, first step over the 1e-15 bar, INDEPENDENT: sea level
   21 -> 57, V 54 -> 85, U 54 -> 97; T and S never** (round 7: T at 43).
   Steps with a row over the bar: 77 -> 44.
2. **T and S are NEMO's bit for bit for all 100 steps.** Round 7: first
   unequal at step 19.
3. **The chain is bit-identical on every scored field through step 39**,
   with two exceptions that do not propagate: cells below 5e-289 at steps 8
   to 11 (take-away 6) and one 3-D U face at step 21. Round 7: through
   step 3.
4. **Item 1 was not in the substep loop.** The solver's Coriolis
   coefficients are stretched with thickness ratios the card built by
   dividing by the face area; NEMO multiplies by a stored reciprocal. One U
   point at step 3 survives the `1 + ratio` that hides all the others.
5. **The tracer carrier was a fused multiply-add.** The card's compiled
   `1 + ssh * (1 / depth)` is contracted into one operation; NEMO stores the
   product first. One T cell at step 19.
6. **A new class, not a statement:** the card's executable flushes numbers
   below the smallest normal double to zero; NEMO keeps them. At the bar by
   300 orders of magnitude; not bit-identical at steps 8 to 11.
7. **What is first now:** step 40, one T cell: NEMO interpolates the
   T-point ratios at stage 2, the card takes the ratio of the interpolated
   sea level. Named and measured, **not landed** (section 5).

## 0. Admission and controls

- No new record. Round 7's card is reproduced by this round's before arm
  (ratio order off, stored ratio on reads 21 / 54 / 54; section 4).
- Score plant exits 1 (plant_score.log). The replay's own plant exits 1
  (ratio_order_plant.log).
- Each offline replay is scored against NEMO's record before use:
  thickness ratios 0 unequal at steps 1 to 10; V-face coefficients 0 unequal
  at steps 1 and 2; stage tracer update 0 unequal at steps 19 and 40.

## 1. Item 1: the barotropic V face after step 3

### 1.1 The walk (INDEPENDENT chain; offline)

| question | instrument | reading |
|---|---|---|
| is step 2's loop NEMO's? | substep table, card's own chain | every operand, every substep: 0 unequal |
| are the six carried histories NEMO's at step-3 entry? | rebuilt from the step-2 substep record | 0 unequal (probe, uncommitted) |
| where does step 3 differ? | NEMO's solver-exit frame | sea level 0, U 0, V 1 face, (77, 52), one last place |
| is the velocity update the card's own? | numpy replay from the card's step-3 operands | equals the card on every face of all 13 substeps (probe) |
| which operand? | NEMO's V-face coefficients replayed from NEMO's recorded ratios | 3 entries unequal against the builder at step 3; 0 at steps 1, 2, 4 to 10 |

NEMO's solver does not average here (the deck's filter): the exit velocity
is the last substep's (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:667-669`),
so one face of the last update is the whole event.

### 1.2 Ratio table (ratio_order_kt1_10.json; NEMO's record only)

Points not bit-equal to NEMO's recorded entry ratio, of 26,460 (top row not
scored). "1 + r" is the same count after one is added.

| step | written order, U / V / F | divided by the area, U / V / F | 1 + r, divided |
|---|---|---|---|
| 1 | 0 / 0 / 0 | 55 / 38 / 43 | 0 |
| 2 | 0 / 0 / 0 | 353 / 210 / 237 | 0 |
| 3 | 0 / 0 / 0 | 784 / 613 / 576 | **1 (U)** |
| 5 | 0 / 0 / 0 | 1,775 / 1,609 / 1,565 | 0 |
| 10 | 0 / 0 / 0 | 3,464 / 3,482 / 3,361 | 0 |

V-face coefficients at step 3, production builder against the replay, open
faces (12,332): option off 3 unequal ((77, 52), (78, 52), (78, 51), all
reading U point (78, 51)); option on 0.

### 1.3 Statement F: the solver's ratios in the written order

NEMO: the area-weighted sea-level mean times the inverse rest depth times
the stored reciprocal face area
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:266-269`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:281-285`; the reciprocals are
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:156` and
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domhgr.f90:170-171`). The coefficient
statement that reads them is
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:987-1008`. The card divided
by the area.

Option: barotropic_een_ratio_written_order (BarotropicConfig, default
False). One helper at the end of the solver file; the builder's line count
is unchanged.

Result: V face after step 3 -> 0. The chain is bit-identical through step 7.

## 2. Item 3: the tracer carrier (moved up: it was first once item 1 landed)

After statement F the first real difference is T and S at one cell,
inner-domain (58, 60), after step 19; everything before is bit-equal.

| row at step 19 | reading |
|---|---|
| stage tracer update replayed in NEMO's order (control) | 0 unequal, stages 1 to 3 |
| the card's stage 3 from NEMO's stage entry | T 1, S 1, cell (58, 60) |
| replay with the AFTER factor as a fused multiply-add | equals the card on every wet cell; 1 unequal against NEMO |
| replay with every other form tried (quotient; fused before factor) | equals NEMO, 1 unequal against the card |

The last two rows are an uncommitted probe (probe5). A second uncommitted
check: with the flag off the compiled Jacobian returns the fused value on 8
of 8 operands chosen to discriminate; with it on, the written value (that
half is a committed test).

### Statement G: the stored T-point ratio

NEMO writes the ratio into an array
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:257`) and the thickness reads
one plus it (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:233`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:284-285`). Two rounded
operations. The card's two helpers formed the sum bare.

Option: nemo_r3t_stored_product (a static flag on the partial-cell
coordinate, default False), read by the shared Jacobian and by the
reciprocal stretch helper. Second site found by re-scoring: with the first
alone, step 20 still had 11 sea-level cells, from stage-1 and stage-2 T and
S at (58, 60) and two U and two V faces of the pressure gradient.

- T-P2 **REFUTED**: the card's own update was unequal from NEMO's frames.

## 3. The 100-step record

First step leaving the 1e-15 bar; largest normalised error. INDEPENDENT.

| card | ssh | u | uu_b | vv_b | v | T | worst ssh | worst u |
|---|---|---|---|---|---|---|---|---|
| round 7's card (its receipt) | 21 | 54 | 54 | 54 | 54 | 43 | 1.871e-14 | 2.269e-15 |
| + F only | 44 | 69 | 69 | 58 | 58 | none | 1.361e-14 | 1.564e-15 |
| + G only | 21 | 54 | 54 | 54 | 54 | 43 | 1.411e-14 | 2.077e-15 |
| **the card (F, G)** | **57** | **97** | 97 | 85 | **85** | **none** | 1.282e-14 | 1.351e-15 |
| the card, entry depth deselected | 21 | 54 | 54 | 47 | 47 | 42 | 1.438e-14 | 1.747e-15 |

No row moved earlier. The final-tree run (record100_5_card.json, 2efcb911f)
differs from the pre-review run in 0 of 598 rows.

Cells not bit-equal / largest normalised, the card:

| step | sea level | U (3-D) | T |
|---|---|---|---|
| 1 to 7 | 0 | 0 | 0 |
| 8 | 0 | 2 / 1e-308 | 0 |
| 9 to 11 | 21, 31, 21 / below 1e-304 | 18, 14, 6 | 0 |
| 12 to 20 | 0 | 0 | 0 |
| 21 | 0 | 1 / 2.7e-20 | 0 |
| 22 to 39 | 0 | 0 | 0 |
| 40 | 0 | 2 / 4.3e-19 | 0 |
| 41 | 10 / 3.5e-18 | 8 | 0 |
| 56 | 4,936 / 2.8e-17 | 4,726 | 0 |
| 57 | 5,574 / **6.38e-15** | 5,253 | 0 |
| 100 | 16,230 / 9.95e-15 | 15,682 | 0 |

The step-57 event: sea level, cell (67, 76), 6.38e-15 (NEMO's largest sea
level there 4.7e-02); one step earlier the largest is 2.8e-17. T and S are
bit-equal, so this is not the tracer carrier of rounds 4 to 7. Not
attributed.

GIVEN-NEMO-ENTRY, whole step, steps 1 to 10 (given_entry_10.json): 0
unequal at steps 2 to 7; the tiny-number cells at 8 to 10; step 1's T and S
row is NEMO's land values, as in every round.

## 4. Item 2: the stage-3 right-hand side

- V-P1 HOLDS: the two V faces at step 5 are still there (5.0e-24, at
  (77, 47)); final velocity 0 unequal.
- The same shape at step 21: two U faces of the stage-3 right-hand side
  (8.3e-25, at (77, 61)), one final U face (2.7e-20) that is gone a step
  later.
- V-P2 **NOT TESTED**: the term was not split. PLAUSIBLE only: the size is
  the last place of a pressure-gradient-sized term, not of the vorticity
  term.

## 5. First non-bit now: step 40 (named, not landed)

GIVEN-NEMO-ENTRY at step 40: stage-2 T and S differ on one cell, (55, 83);
stage 3 then has 2 U and 1 V faces. NEMO's stage-2 T-point ratio is the mean
of the two ratios it holds
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217`); the card takes
the ratio of the mean sea level. Committed arm (stage_replay_kt40.json): the
two forms of one plus the ratio differ on 1 cell at step 40, 0 at step 19.
This is the T-point twin of round 7's statement D.

Not landed: the card's stage-3 consumers (the tracer weight, the pressure
gradient's stretched depth) take a sea level, not a ratio, so the statement
needs a ratio handed to each. A second cell at step 40, (101, 65), has four
unequal right-hand-side faces at 3.1e-19 and is not attributed.

## 6. Item 4: the rounding-row question

Q-P1 HOLDS: no rounding row is registered.

The tiny-number class is recorded, not registered: at steps 8 to 11 NEMO
holds values between 1e-320 and 1e-308 at the front of the signal where the
card holds exactly zero (the card has 0 such cells at every step; probe7,
uncommitted). The largest difference this causes is 4.7e-289, and it is gone
at step 12. It is an arithmetic-environment difference, not a NEMO
statement, so it has no citation and cannot be a registered rounding row
under the rule. It does not touch the bar.

## 7. Gates for the landing

- No AT-BAR row leaves the bar in either label; no first step over the bar
  moves earlier.
- Both options default False, appended last, selected only by the G1 card
  (test). Caller grep: each name appears in its field, its use (one site for
  F, two for G plus the two coordinate constructors), the G1 card (selection
  and validator), the ladder's before arms and the tests.
- Searched before building (grep of the solver, the vertical module and the
  equation-of-state module): the ratio stencil, the rounding guard, the
  fold exchange and the static coordinate flag pattern existed and are
  reused. New code: two small helpers at the ends of two files; one ladder
  arm.
- No line count changes inside the solver, vertical, equation-of-state or
  recipe files; the config file gains five lines at the end of one class.
- Closed cards (GYRE, VORTEX, TSUNAMI, ORCA2, DINO) NOT re-scored. Default
  paths read as unchanged, not measured.

## 8. ORCA2 pointer

| statement | G1 | does ORCA2's deck reach it? |
|---|---|---|
| F, solver ratios times the reciprocal area | landed, selected | yes, same builder. **Caveat:** ORCA2 reads its U and V face areas from the domain file (reduced straits); the helper uses e1 times e2 and says so. Import needs the file's areas. UNMEASURED |
| G, stored T-point ratio | landed, selected | yes: every thickness of every level. With 31 levels each cell reads the same factor, so one event moves a whole column. UNMEASURED |
| stage-2 T ratio interpolated (section 5) | named only | yes, same stage statement |
| numbers below the smallest normal double | recorded | yes wherever a signal front crosses 1e-308; harmless at the bar |

## 9. Predictions

| id | result |
|---|---|
| C-P1 written-order ratios equal NEMO's | HOLDS, 0 unequal, steps 1 to 10 |
| C-P2 divided order differs on more than a tenth; 1 + r on fewer than ten | first half **REFUTED** (0.2% at step 1, 3% at step 3, 13% at step 10); second half HOLDS (1 point in 10 steps) |
| C-P3 coefficients 0 with the option; V face 0 | HOLDS |
| C-P4 chain bit-identical through step 4 | HOLDS (through step 7) |
| C-P5 record not at the bar | HOLDS |
| V-P1 the step-5 faces remain | HOLDS |
| V-P2 the term is the vorticity term's operand | NOT TESTED |
| T-P1 tracer replay control | HOLDS at steps 19 and 40 (not run at 20, 21) |
| T-P2 the card's update is bit-equal from NEMO's frames | **REFUTED**: statement G |
| T-P3 first T difference later than step 20 | HOLDS: none in 100 steps |
| Q-P1 no rounding row registered | HOLDS |

## 10. Corrections to my own work, and choices

Corrections:
1. The preregistration's "already measured" list says the three unequal
   coefficients read F point (78, 51). They read U point (78, 51): the one
   point where 1 + ratio differs is a U ratio.
2. C-P2's first half was wrong (section 9).
3. T-P2 was wrong: I expected the sea level to be the carrier.
4. One new test asserted that a traced flag gives the same bits as the flag
   off. The test was wrong (the two compile differently), not the code;
   it now checks the value to one part in 1e15.
5. My first replay of the velocity update indexed V faces one row off and
   failed its own control; fixed before any reading was taken from it.

ASKED (round brief): the four items in order; first non-bit with citation;
land cited statements card-selected; re-score after each; tests; one review.

UNASKED, each offered for revert:
1. Item 3 was done before item 2: after statement F it was the first
   difference of the chain.
2. Statement G is a flag on the coordinate, not on the model config: the
   shared Jacobian has 37 call sites and takes the coordinate, and the
   coordinate already carries one static NEMO flag. A traced flag reads as
   off.
3. Three option-position pins and two event tests were re-pinned to the
   re-measured record (no event through step 30; step 21 at the bar with
   the chain's own tracers).
4. The stage-2 T-ratio statement was not landed (section 5).

## 11. Tests and review

Single review (codex): **findings**, two CONFIRMED and acted on. Log:
.../gridladder_rounds/round8/codex_review_r8.log (its last 60 lines).
- "CONFIRMED — `nemo_r3t_stored_product` is a normal `NamedTuple` pytree
  leaf, then used in Python control flow ... Passing the coordinate through
  `jax.jit` traces the bool and can trigger tracer-bool conversion." Acted
  on in 2efcb911f: both sites test identity with True; a test passes the
  coordinate through jit.
- "CONFIRMED — coordinate reconstruction can silently reset the flag. Both
  constructors omit `nemo_r3t_stored_product`". Acted on: both forward it.
  Both take a z-star reference, which cannot carry the flag today.
- "VERDICT: findings". No finding was reported against the ratio order, the
  default paths, the replay's index shifts or the tests. No second reviewer:
  a single review, not a dual one. Claim review before code: NOT done (the
  lane's rule is one review of the diff).

Focused tests, synchronous:
- The G1 file, 68 tests: 16 passed (tests_g1_r8_a.log) and 52 passed
  (tests_g1_r8_b.log). 8 are new. After the review fix only the round-8
  tests were re-run (tests_g1_r8_c.log: 1 failed, my wrong expectation, then
  4 passed); the whole file was NOT re-run.
- Seven ratchet files: 6,287 passed, 2 skipped, 4 failed
  (tests_ratchets.log): the same four as rounds 1 to 7, none in a file this
  round touched.
- Other callers of the coefficient builder: 32 passed, 1 failed
  (tests_builder_callers.log); the failing test fails identically on round
  7's tree (1ce119a43).
- NOT RUN: any closed-card ladder; the wider solver, vertical and
  equation-of-state unit selections; the inline-import budget.

Compliance, stated because no gate enforces it: search-before-build
(section 7); one variable per arm (F alone, G alone, each off); instruments
checked before quoting (controls in section 0; one replay failed its
control and was fixed first); non-vacuity (each option off turns its record
test red; both plants exit 1).

Stamps: record100_1_card is dd64f53e5; record100_3_card is a41d336e7; its
three before arms are a41d336e7 with the ladder's new flag uncommitted
(dirty); record100_5_card, given_entry_10, ratio_order_kt1_10 and
plant_score are 2efcb911f. The step-5, 19, 20, 21 and 40 tables are
ff5b917f1 or a41d336e7.

UNVERIFIED:
- that no closed card moves (default paths read, not run);
- the ORCA2 column of section 8;
- the fused-operation reading beyond the two checks of section 2 (both
  uncommitted on the card's side);
- the cause of the step-57 event, of the second cell at step 40, and of the
  stage-3 faces at steps 5 and 21;
- which component of the card's executable flushes tiny numbers.

SHA-256 prefixes, final/: given_entry_10.json=da7951f724ac144b, handoff_kt20.json=54c79c0049026965, plant_score.json=5c9702ad28cc6c05, ratio_order_kt1_10.json=4a9b088c41142134, ratio_order_plant.json=da6c7570f67e6b91, record100_1_card.json=b54d8ca894e09b5a, record100_3_card.json=48a545737ed2af10, record100_3_without_entry_depth.json=0a0b620ce1c13fb0, record100_3_without_r3t_product.json=fdb7811639eb7500, record100_3_without_ratio_order.json=76f41d465616042a, record100_5_card.json=18060c3e6af00185, rhs_terms_kt20_before.json=5fcf41c0a4000408, stage_replay_kt19.json=83d7b2222619c624, stage_replay_kt40.json=bcf915aa149e8a6a, stages_kt19.json=9d63bd76a8dda48c, stages_kt20.json=cf8df24a5950aa05, stages_kt21.json=1e0f2396ca6eb726, stages_kt40.json=f13d32a467defef9, stages_kt5.json=b7abd007a6c8e421, substeps_independent_kt2.json=abbdcc71090936fa.
One level up: codex_review_r8.log=2bfaa259b273634e, tests_g1_r8_a.log=b8eaf3714f74a48e, tests_g1_r8_b.log=990b477100dbc90d, tests_ratchets.log=73c6143825a286e5, tests_builder_callers.log=1f364aad872d707f. The citation gate's output and tests_g1_r8_c.log were written after this list.

## OPEN for round 9

1. Step 40, T cell (55, 83): hand NEMO's interpolated stage-2 T-point ratio
   to the stage-2 tracer weight and to stage 3's stretched depth (section
   5). Then the second cell, (101, 65).
2. The step-57 sea-level event at (67, 76): one-step replay from NEMO's
   step-57 entry; T and S are no longer the carrier.
3. The stage-3 right-hand-side faces at steps 5 and 21: split the term.
4. The tiny-number class: say which side is the environment to match
   (a user question only if a bit-identical record is wanted; the bar does
   not need it).
5. ORCA2 import of statement F needs the domain file's U and V face areas.
