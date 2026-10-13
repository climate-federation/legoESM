# Grid ladder rung G1 — round 7 receipt (gridladder_r7)

Status: **LANDED** (four cited statements, G1-card-selected), with **one
registered gate red**: sea level's first step over the bar moved EARLIER,
23 -> 21. The rung is **NOT certified**: the 100-step record is not at the bar.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g1_round7.md
(commit c183dd28f, before any measurement). Code: c1b4d0d69 (vorticity
order), f6f7e3bf6 (stage-3 residual, top V row), 9852cfd5d (stage F ratio,
tests), 1c1e1683d (re-pins), fa218f555 (review fix). Evidence:
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round7/final/.
Record: the operator's round-4 acquisition
(.../gridladder_rounds/round4/oracle_g1_r4/p3_a_100step). NEMO was not re-run.

Decision 119 is pending. The card's entry-depth selection was not touched;
every headline below carries the deselected arm's reading next to it.

## Take-aways

1. **The INDEPENDENT chain is NEMO's bit for bit through step 3.** Every
   scored field, every cell, steps 1 to 3, except one barotropic V face
   after step 3. Round 6's card had 547 unequal U faces after step 1 and 742
   unequal sea-level cells after step 2.
2. **One statement owned items 1, 2 and 3 of round 6's list.** The card's
   vorticity operator ran NEMO's triad statement in a different operation
   order. Every operand was already bit-equal.
3. **Stage 3 had two more.** The card added the barotropic velocity back
   after the vertical solve, where NEMO hands the residual on. And NEMO's F
   ratio at stages 2 and 3 is an interpolation of ratios, not the ratio of an
   interpolated sea level (one F point of 26,640 at step 2).
4. **The stored top V row is closed.** V's first step over the bar: 19 -> 54;
   its worst error: 1.4e-02 (normalised) -> 2.6e-15. No other row moved.
5. **Headline: sea level 23 -> 21, velocities 54 -> 54, V 19 -> 54.**
   Deselected arm (Decision 119): 21, 54, V 47. The sea-level step is
   EARLIER. That breaks the gate "no first step over the bar earlier" and is
   registered in section 5. The event is still carried by the chain's T and S.
6. **Decision 119 now has a discriminating row.** With the entry depth
   selected the chain is bit-identical through step 3; deselected it has 2
   unequal sea-level cells after step 1 and 528 after step 2.
7. **What is first now:** one barotropic V face after step 3, then 7
   sea-level cells after step 4. At step 4 the right-hand side and all three
   stages are bit-equal given NEMO's entry, so it enters through the 2-D
   solver or an operand the score does not read. Not attributed.

## 0. Admission and controls

- No new record. The before arm (the four new options off) reproduces round
  6's card to every printed digit: 23 / 54 / 54 / 54 / 19, worst sea level
  1.479e-14 (record100_0_before.json).
- Score plant exits 1 (plant_score.log).
- Each offline replay is scored against NEMO's record before it is used:
  the right-hand side 0 unequal at steps 1 to 5; stage 2 and stage 3
  velocities 0 unequal at steps 1, 2, 4 and 5.

## 1. Item 1: the 3-D right-hand side at step 2

### 1.1 The replay and its control

Offline numpy, from NEMO's step-entry frame with its halos and the mesh
file: the pressure gradient on the one level
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:380-394`) and the planetary
vorticity term (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:720`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:727`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:766-767`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:772-775`,
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:779-784`). The density anomaly
is the shared library function on NEMO's recorded T and S at NEMO's
stretched depth. The top V row is not replayed.

Control: replayed pressure gradient + vorticity term against NEMO's
recorded right-hand side: **0 unequal** of 15,853 U and 15,875 V wet faces,
steps 1, 2, 3 (and 4, 5). R-P1 HOLDS.

### 1.2 Term table (GIVEN-NEMO-ENTRY; the model's write-only term observer)

Unequal wet faces, U / V; largest error (NEMO about 2.1e-07).

| row | step 1 | step 2 | step 3 |
|---|---|---|---|
| round 6's card: pressure gradient vs replay | 0 / 0 | 0 / 0 | 0 / 0 |
| round 6's card: vorticity term vs replay | 0 / 0 | 763 / 752, 7.9e-23 | 1,699 / 1,746, 7.9e-23 |
| round 6's card: whole right-hand side vs NEMO | 0 / 0 | 620 / 766 | 1,561 / 1,765 |
| the card: all three rows | 0 / 0 | **0 / 0** | **0 / 0** |

All unequal faces are interior; fold, seam and south are 0. Round 6 read
618 / 766 and 1,578 / 1,765 for the third row through a different observer.

- R-P2 HOLDS: the pressure gradient is bit-equal; the vorticity term is the
  first non-bit term.
- R-P3 **REFUTED**: no operand differs. The landing itself shows it: with
  only the operation order changed, the term is 0 unequal. (An uncommitted
  probe also read both face thicknesses, the F thickness and the Coriolis
  parameter as 0 unequal at steps 2 and 3.)

### 1.3 Statement A

NEMO's vor_een: the vorticity times the RECIPROCAL of the F thickness; the
transport as (width times thickness) times velocity; four unweighted triads
in a fixed summation order; the trend as (1/12 times inverse width) times
((a+b)+(c+d)). The card's operator divided by the thickness, put 1/12 inside
each triad, multiplied by the width last and summed the four products left
to right. Equal as real numbers; last place different.

Option: een_vorticity_written_order (LatLonCGridOceanConfig, default False).
One shared triad operator, one new argument; every product and sum rounded
once with the shared rounding guard.

## 2. Items 2 and 3: stage 2 of step 1; the stage-1 U raw velocity

Stage table, GIVEN-NEMO-ENTRY, unequal interior faces U / V.

| arm | stage 1, steps 1 / 2 / 3 | stage 2, steps 1 / 2 / 3 | stage 3, steps 1 / 2 / 3 |
|---|---|---|---|
| round 6's card | 0/0, 6/4, 13/9 | 123/238 at step 1 | 542/462 at step 1 |
| + A | 0/0, 0/0, 0/0 | 0/0, 0/0, 0/0 | 536/441, 1,063/897, 1,727/1,534 |
| + A + B (section 3) | 0/0, 0/0 | 0/0, 0/0 | 0/0, 1/0 (steps 1, 2) |
| the card (A, B, D) | 0/0 | 0/0 | 0/0 (steps 2, 4, 5) |

- Stage 2 offline: right-hand side replayed from NEMO's stage-1 frame, then
  the written update (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:373-378`)
  and correction (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:413-414`,
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:417-418`): **0 unequal**
  against NEMO's stage-2 frame. S-P1 HOLDS at stage 2; no acquisition is
  needed. S-P2 HOLDS: the owner was statement A.
- Item 3 (17 and 63 faces at steps 2, 3): **0** with A. U-P1 NOT TESTED as
  written: the row vanished before the U ratio was substituted. The U ratio
  is bit-equal to NEMO's (section 1.2), so it was never the operand.
- The stage-3 T and S rows of step 1 read 10,207 cells in every arm,
  round 6's file included: NEMO's step-1 frames hold T and S on land. Not a
  card row.

## 3. Stage 3

### 3.1 Statement B: the barotropic residual

My first stage-3 replay failed its own control (547 / 480 faces). Two
things were missing from it, both read from the source afterwards:

- the F ratio stage 3 reads is the mean of the entry and after ratios
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:239`); the stage-3 frame
  holds the after ratio, written back at
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:546`;
- NEMO's stage-3 time step
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:145-150`) is followed by
  the removal of the barotropic velocity
  (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:160-161`). The deck resolves
  implicit drag on (the reference default), which switches that branch on
  although drag itself is off. The one-level solve after it is the identity.

With both, the replay is **0 unequal** against NEMO's stage-3 frame.

The card subtracted the same barotropic velocity before its solve and ADDED
IT BACK after it; NEMO hands the residual to the stage correction. Card
operand rows at stage 3, steps 1 / 2, U (V alike):

| row | round 6's card + A | + B |
|---|---|---|
| card right-hand side vs replay | 0 / 2 | 0 / 2 |
| card raw velocity vs written update of its own right-hand side | 666 / 1,020 | 0 / 0 |
| card stage-3 velocity vs NEMO | 536 / 1,063 | 0 / 1 |

Option: zdf_barotropic_removal_nemo_literal (default False). Refused unless
the WS-RK3 stage correction that restores the depth mean is active.

### 3.2 Statement D: the stage F ratio

The two faces left at step 2 sit next to one F point, (84, 49). NEMO builds
the stage F ratio from the two ratios it holds:
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:218` at stage 2 and
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:239` at stage 3. The card
took the ratio of the stage's interpolated sea level.

| F thickness at step 2, stage 3, against NEMO's | unequal of 26,640 |
|---|---|
| from the ratio of the interpolated sea level (the card before) | 1 |
| from the interpolated ratios (statement D) | 0 |

The helper's entry and after ratios are bit-equal to NEMO's recorded ones.
At stage 2 the two forms agreed at steps 1 to 3. Option:
rk3_stage_r3f_nemo_literal (default False). With it the one U face of the
INDEPENDENT chain at step 2 is 0.

## 4. Item 4: the stored top V row (statement C)

NEMO ends every stage with the fold exchange of the 3-D velocity
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:572-573`). On the T-pivot
fold it leaves the top V row as the image of the row below: mirrored, sign
flipped, on the faces wet there. The card's compact mask kept the stored row
at zero. Same permutation, sign and wet rule as the image the vorticity
term already reads.

- NEMO's own after frame at step 100, top row zeroed and handed to the
  helper: NEMO's row comes back bit for bit (test).
- Applied after each stage's correction. At step 5 the stage-1, stage-2
  and stage-3 frames' top V row: 5, 5, 6 unequal faces -> 0.
- 100-step record: V first over the bar 19 -> 54; V worst 1.446e-02 ->
  2.099e-15 (alone) and 2.618e-15 (on the card). With C alone every other
  number of the headline table is unchanged. V-P1 HOLDS.

Option: rk3_fold_exchange_top_v_row (default False).

## 5. The 100-step record

First step leaving the 1e-15 bar; largest normalised error over 100 steps.
INDEPENDENT.

| card | ssh | u | uu_b | vv_b | v | worst ssh | worst u |
|---|---|---|---|---|---|---|---|
| before (round 6's card) | 23 | 54 | 54 | 54 | 19 | 1.479e-14 | 2.181e-15 |
| + C only | 23 | 54 | 54 | 54 | 54 | 1.479e-14 | 2.181e-15 |
| + B only | 23 | 54 | 54 | 54 | 19 | 1.454e-14 | 1.812e-15 |
| + A only | **17** | **27** | 27 | 27 | 19 | 1.578e-14 | 1.982e-15 |
| + A + B | **21** | 54 | 54 | 54 | 19 | 1.871e-14 | 2.269e-15 |
| **the card (A, B, C, D)** | **21** | 54 | 54 | 54 | 54 | 1.871e-14 | 2.269e-15 |
| the card without A | 23 | 54 | 54 | 54 | 54 | 1.454e-14 | 1.812e-15 |
| the card without B | 17 | 27 | 27 | 27 | 27 | 1.578e-14 | 1.982e-15 |
| the card without D | 21 | 54 | 54 | 54 | 54 | 1.871e-14 | 2.269e-15 |
| the card with the entry depth deselected (Decision 119) | 21 | 54 | 54 | 47 | 47 | 1.510e-14 | 2.066e-15 |

**REGISTERED RED.** Sea level first leaves the bar at step 21, not 23. A
alone reads 17 and 27. H-P1 REFUTED. The preregistered gate "no first step
over the bar earlier" FAILS for A, and for the set. They are selected on
the card because the round's brief asks for cited statements to be
card-selected and each is bit-proven at its own row. Reverting is one line
of the card; see DECISION_NEEDED.

What the same record shows cell by cell (cells not bit-equal / normalised):

| step | sea level, before | the card | deselected arm | U, before | the card |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 2 / 6.9e-18 | 547 | 0 |
| 2 | 742 / 2.1e-17 | 0 | 528 / 2.8e-17 | 1,226 | 0 |
| 3 | 2,185 / 2.4e-17 | 0 | 1,301 / 2.6e-17 | 2,449 | 0 |
| 4 | 3,694 / 2.1e-17 | 7 / 6.9e-18 | 2,386 / 2.8e-17 | 3,813 | 4 |
| 10 | 10,687 / 5.0e-17 | 560 / 1.0e-17 | 8,717 / 3.0e-17 | 9,620 | 503 |
| 20 | 12,952 / 5.9e-17 | 3,117 / 5.9e-17 | 12,741 / 5.7e-17 | 12,334 | 2,930 |
| 21 | 13,217 / 3.6e-17 | 3,551 / 5.89e-15 | 13,000 / 5.89e-15 | 12,391 | 3,353 |
| 23 | 13,619 / 5.73e-15 | 4,453 / 5.45e-15 | 13,250 / 8.06e-15 | 12,805 | 4,255 |
| 40 | 15,580 / 5.0e-15 | 12,318 / 1.7e-15 | 15,513 / 5.0e-15 | 14,969 | 11,644 |
| 100 | 16,409 / 1.32e-14 | 16,379 / 1.18e-14 | 16,419 / 1.37e-14 | 15,822 | 15,803 |

- After step 3 the card's only unequal scored cell is one barotropic V
  face. After step 4: sea level 7, uu_b 8, vv_b 5, u 4, v 3, T and S 0.
- 77 of 100 steps have a row over the bar (82 before; 80 deselected).
- T / S cells that differ, before -> card: step 20: 3/2 -> 1/0; 23: 4/2 ->
  4/4; 40: 49/17 -> 48/17. The tracer flip is still there.

The step-21 event (interior_events_24.json): 1 cell over the bar,
inner-domain (76, 34), -5.89e-15 (NEMO -1.53e-03 there), 7 cells from land,
71 rows from the fold; its entry error is 1.7e-17. Same carrier as rounds 4
to 6, re-measured at the new cell: one step from NEMO's step-21 entry with
NEMO's T and S is at the bar; with the card's own T and S it is over the
bar at that cell (both arms are tests). Steps 22, 23, 24: 2, 2, 1 cells.

GIVEN-NEMO-ENTRY, whole step, steps 1 to 10 (sea-level cells not
bit-equal): 0, 0, 0, 0, 21, 86, 151, 228, 318, 428. Every row AT-BAR;
largest normalised 8.7e-18. Round 6: 0, 564, 1,451, 2,328, ... 6,534. The
barotropic histories past step 1 come from the card's own chain, which is
unequal from step 4. Not run past step 10.

INDEPENDENT ladder, steps 1 to 10 (sea-level cells): 0, 0, 0, 7, 41, 98,
-, 288, -, 560; every row AT-BAR, no field leaves the bar in either label.

## 6. Gates for the landing

- No AT-BAR row of steps 1 to 10 leaves the bar in either label.
- First step over the bar: V 19 -> 54; u, uu_b, vv_b 54 -> 54; **sea level
  23 -> 21: RED, registered above.**
- Each option is default False, appended last, and selected only by the G1
  card (test). Caller grep after the change: each name appears in its
  field, its one use, the G1 card (selection and validator), the ladder's
  before arms and the tests.
- Searched before building (grep of the operators, the model and the
  vertical module): the triad operator, the rounding guard, the fold image
  rule, the F-thickness helper and the stage correction existed and are
  reused. New code: one argument on the triad operator, two arguments on the
  F-thickness helper, three small helpers at the end of the model file.
- No cited line of the model, the tendency module, the vertical module or
  the recipe moves: in-place lines keep their count; helpers sit at the ends
  of files.
- Closed cards (GYRE, VORTEX, TSUNAMI, ORCA2) were NOT re-scored. The
  default paths are unchanged by construction (review: "No default-False
  branch was found to alter the legacy arithmetic when unselected"); that is
  read, not measured.

## 7. ORCA2 pointer

| statement | G1 | ORCA2 lane has it? | does ORCA2's deck reach it? |
|---|---|---|---|
| A, vor_een operation order | landed, selected | no | yes: same operator, with relative vorticity added. The option orders the shared part; the relative-vorticity sum is not touched by it. UNMEASURED |
| B, dyn_zdf residual not re-added | landed, selected | no | yes: implicit drag with the split-explicit solver. With real drag and many levels the solve is not the identity; the statement is the same. UNMEASURED |
| C, top V row after each stage | landed, selected | no | yes: same fold, same exchange. Stored row only on G1; UNMEASURED there |
| D, stage F ratio interpolated | landed, selected | no | yes: any step with sea level; a rare last-place event. UNMEASURED |

## 8. Predictions

| id | result |
|---|---|
| R-P1 replay equals NEMO's right-hand side | HOLDS, 0 unequal |
| R-P2 pressure gradient bit-equal, vorticity term first | HOLDS |
| R-P3 the cause is an operand (a thickness ratio) | **REFUTED**: every operand bit-equal; the operation order |
| S-P1 stage-2 replay equals NEMO's stage-2 frame | HOLDS; stage 3 only after two statements I had missed |
| S-P2 the stage-2 owner is item 1's | HOLDS |
| U-P1 the 17 / 63 faces are the U ratio | NOT TESTED: the row vanished with A; the U ratio is bit-equal |
| V-P1 V 19 -> 54, worst about 2.1e-15, nothing else moves | HOLDS |
| H-P1 not at the bar; sea level 23 or later | first half HOLDS; second half **REFUTED**: 21 |
| H-P2 no rounding row registered | HOLDS: none registered |

## 9. Corrections to my own work, and choices

Corrections:
1. R-P3 was wrong: I preregistered a ratio operand; it is the triad's order.
2. My stage-3 replay failed its control twice over (the F ratio it read and
   the barotropic removal). Found by the control, fixed from the source.
3. I landed the V exchange after stage 3 only; NEMO runs it after every
   stage. Found by the review; fixed in fa218f555 (0 of 598 record rows
   move).
4. Line ranges: the preregistration and commits c1b4d0d69, f6f7e3bf6 cite
   the pressure gradient at 379-394 (it is 380-394), the transports at
   767-768 (766-767), the trend at 779-783 and 779-785 (779-784), the
   stage-3 update at 144-149 (145-150). Comments are fixed in the tree; the
   messages stand as written.
5. H-P1's second half was wrong (section 5).

ASKED (round brief): the five items in order; first non-bit with citation;
land cited statements card-selected; re-score after each; tests; one review.

UNASKED, each offered for revert:
1. A, B and D stay selected although sea level's first step over the bar
   moved 23 -> 21. My pick: keep. See DECISION_NEEDED.
2. D was landed although round 6's list did not name it: it was the first
   unequal row of the chain once A and B were in.
3. Three older tests pinned option positions or the first event (step 23,
   cell (74, 84)); re-pinned to the re-measured facts (step 21, (76, 34)).
4. The older vorticity replay now delegates to the new whole-domain one.
5. Item 5 of the list (the rounding-row question) was not started.

Left alone on purpose: the 2-D solver at step 4; the one barotropic V face
after step 3; two V faces of the stage-3 right-hand side at step 5 (5.0e-24,
final velocity 0 unequal); the U pivot row's exchange (no unequal face).

## 10. Tests and review

Single review (codex): **findings**, two CONFIRMED and acted on, one not.
Log: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round7/codex_review_r7.log.
- "HIGH — CONFIRMED: claim (3) applies the fold exchange at the wrong
  frequency ... the V-fold exchange occurs after every RK3 stage." Fixed in
  fa218f555; measured before and after.
- "MEDIUM — CONFIRMED: the fold-exchange test can pass with the production
  feature removed." Fixed: a record test drives the production call through
  the stage table at step 5, with the option off as its plant.
- "LOW — CONFIRMED: the oracle-backed round-7 tests silently skip outside
  one absolute local dataset." Not acted on: that is this file's stated
  design for record tests.
- Surviving: "No static refutation was found for: (1) ... (2) ... (4) ...
  (5)" and "No default-False branch was found to alter the legacy
  arithmetic when unselected." "VERDICT: findings". No second reviewer: a
  single review, not a dual one. Claim review before code: NOT done (the
  lane's rule is one review of the diff).

Focused tests, synchronous, in chunks:
- The G1 file, 59 tests: every one passed in some chunk (tests_g1_r7_a to
  tests_g1_r7_h.log). 15 are new this round. Two chunks went red first and
  were fixed: an older position pin, and my own F-ratio test, which built
  the card without setting fp64 (the test was wrong, not the code). Chunk d
  hit the ten-minute limit after 35 tests; its tail ran in chunks e and g.
  NOT re-run whole after the review fix: only the two fold tests were.
- Seven ratchet files: 6,287 passed, 2 skipped, 4 failed (tests_ratchets.log).
  The same four as rounds 1 to 6, none in a file this round touched: the
  slab-land dispatch guard, the jra55 literal, the land restart
  coefficients, the shortwave parameter spec.
- NOT RUN: any closed-card ladder (GYRE, VORTEX, TSUNAMI, ORCA2); the wider
  operator, barotropic and stage unit selections; the inline-import budget.

Compliance, stated because no gate enforces it: search-before-build (section
6); one variable per arm (each statement alone and each one off); instruments
checked before quoting (every replay has a 0-unequal control against NEMO;
the stage-3 replay failed its control and was corrected before use);
non-vacuity (each option off turns its record test red; score plant exits 1;
the right-hand-side control fires on a planted cell).

Stamps: record arms record100_0 to record100_5 are 9852cfd5d or earlier
(their json names the commit); record100_6_card and its deselected arm and
the two term tables are fa218f555. The stage tables with A only are
c1b4d0d69; the later stage tables (A + B; steps 4, 5) are uncommitted-tree
runs kept one level above final/ with a dirty stamp, and the card's stage
table at the final tree is appended below.

UNVERIFIED:
- that no closed card moves (default paths read, not run);
- the ORCA2 column of section 7 (read, not run);
- that the step-21 event is the same physical carrier as step 23's beyond
  the two tracer arms (same test, new cell);
- operand equality in section 1.2 beyond what the landing implies (probe
  uncommitted);
- what makes the barotropic V face after step 3 unequal.

## 11. Final tree, citation gate and evidence hashes

- Final code commit: c3619b1a0 (two earlier-cited lines restored after the
  citation gate's whole-map audit went red on them; no arithmetic change).
  The 100-step record re-run there differs from fa218f555's in 0 of 598
  rows (record100_7_card.json): 21 / 54 / 54 / 54 / 54.
- The card's stage table at a2254e960, steps 1 to 3, sea level / U / V:
  0 unequal at every stage of every step (stages_kt1_3_card.json).
- Citation gate on this receipt: PASS, 15 citations, 0 unmapped, whole-map
  audit clean; the shifted-citation plant exits 1 (citation_receipt.json,
  citation_plant.json). It first FAILED: four new entries named the wrong
  occurrence of a repeated line, and two earlier anchors were the lines my
  edits had rewritten. Fixed in c3619b1a0.
- After c3619b1a0 the fold, F-ratio and selection tests were re-run (10
  passed, tests_g1_r7_i.log). The stage-table fold test was NOT re-run; the
  record above covers the stage-3 site it exercises.
- The given-entry ladder and the event census are stamped 9852cfd5d, before
  the review fix, which moved no record row.

SHA-256 prefixes. final/: citation_plant.json=3a087a6793ee7b85, citation_receipt.json=6bf1c8ae02bd4407, given_entry_10.json=bf0248d29fc2b25e, interior_events_24.json=de15bebef27d2677, plant_score.json=aadbe3f2a8645597, record100_0_before.json=100386c702051c6d, record100_1b_zdf_only.json=0074890fc2992499, record100_1c_foldv_only.json=fa3df1e2de8e1e06, record100_1_vor.json=2da86fdd485230e0, record100_2_vor_zdf.json=445c79299a928542, record100_4_card.json=2813efac5522e1f7, record100_4_card_without_entry_depth.json=19f87ca42b867b79, record100_5_card.json=f10d9f62f1ac3e90, record100_5_card_without_entry_depth.json=21df235edc6f36cf, record100_5_without_fold_exchange_v.json=acd1fd22b0ea2a45, record100_5_without_stage_r3f.json=f364fdff0526a199, record100_5_without_vor_order.json=5b041e26b7f59e6c, record100_5_without_zdf_residual.json=ca8743bff7e40d30, record100_6_card.json=dcca2ae7b90a1f8b, record100_6_card_without_entry_depth.json=9bde8c7752c3882c, record100_7_card.json=7596e0e601838e5a, rhs_terms_kt1_3_card.json=5f4e72156d7640b3, rhs_terms_kt1_3_without_vor_order.json=3e9af015a0e288e0, stages_kt1_3_card.json=c3e0acd91b0627eb.
One level up: codex_review_r7.log=c60677ba92f3120a, tests_g1_r7_a.log=958d01bfff86cf38, tests_g1_r7_b.log=7226de3438542715, tests_g1_r7_c.log=46ecb7f460b4d3a5, tests_g1_r7_d.log=80f41ff7c56824be, tests_g1_r7_e.log=f6cdfdbda954e873, tests_g1_r7_f.log=3a153a7f7862c48a, tests_g1_r7_g.log=2744e59ae903435a, tests_g1_r7_h.log=1371f8477838d526, tests_g1_r7_i.log=a5ca6718d82f7381, tests_ratchets.log=aab7ae637ce849a1, stages_kt1_2_zdf_dirty.json=292120329bfdefd9, stages_kt4_5_dirty.json=850bd24a2842526c, stages_kt5_dirty.json=340fd17f6f4f6c2e, rhs_terms_kt4_5_dirty.json=399689e754af7157.
This list was written before the receipt's last commit; citation_receipt.json
is regenerated on the final receipt and its hash above is the earlier run's.

## OPEN for round 8

1. The one barotropic V face after step 3 and the 7 sea-level cells after
   step 4 of the INDEPENDENT chain: the stages and the right-hand side are
   bit-equal given NEMO's entry there, so walk the 2-D solver at step 3
   from the card's own chain (the substep table needs a substep record at
   step 3 or 4: the record holds them at steps 1, 2, 17, 31, 41 only). If
   the step-2 substep record cannot separate it, write the acquisition for
   substep records at steps 3 and 4 and return ACQUISITION_NEEDED.
2. The two V faces of the stage-3 right-hand side at step 5 (5.0e-24).
3. The step-21 event and the tracer carrier (T and S first differ at step
   20): replay the stage tracer update from NEMO's frames at the first
   differing cell.
4. Only then the rounding-row question for whatever is left.
