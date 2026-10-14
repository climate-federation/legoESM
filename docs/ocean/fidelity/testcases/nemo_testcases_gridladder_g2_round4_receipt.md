# Grid ladder rung G2 — round 4 receipt and CERTIFICATION (gridladder_r13)

Status: **LANDED and CERTIFIED.** One cited NEMO statement, selected only by
the G2 card. After it the 100-step record is **bit-identical** to NEMO on
every scored field, at every step, both labels, both depth classes.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g2_round4.md
(commit 03ba6f4bd, before the candidate was measured).
Code: 3e37d484b (the landing), 6e37798cd (right-hand-side arm, certification
test), 30450a1f3 (review fix). Every number below was regenerated from the
clean commit 6e37798cd (the right-hand-side arm from 30450a1f3; stamp in each
json). Record: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/oracle_g2_r10
(twins a, b). Evidence: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round13/
(final numbers under final/, the exploratory walk under explore/).
NEMO citations are to the compiled sources under ORCA2G1_OMIP_L1_P3; both
files cited are byte-identical in the record binary's build
ORCA2G1_OMIP_L1_P5 (cmp, this round). **Acquisition needed:** none.

## Take-aways

1. **Round 3's open seed is named.** NEMO forms the barotropic forcing as a
   Fortran SUM over the levels, added in ascending order
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179`). The card used a
   library sum whose order the compiler chooses. Inside the full step the
   compiler vectorised it: about a hundred faces differed in the last place,
   and the 65 substeps spread that to ~600 cells.
2. **It was not fusion.** Turning off fused instructions, the algebraic
   simplifier, constant folding or the new fusion emitters changed nothing.
   Forbidding vector code alone restored NEMO's result.
3. **Landed, G2-card-selected.** The sum now runs in NEMO's order through
   the helper the model already had for the stage correction.
4. **G2 is certified.** 100 steps: 0 cells differ on sea level, both
   barotropic velocities, level-1 U, V, T, S at every step; all 30 levels at
   the 14 restart steps, full and partial cells; the six barotropic
   histories every tenth step. Given-NEMO-entry: 110 of 110 rows equal.
5. **Round 2's "5e-6" right-hand-side gap was dry faces.** The card's
   stage-1 right-hand side equals NEMO's frame on every wet face. NEMO's
   frame alone holds an unmasked pressure gradient on land faces
   (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:393-394`), which the
   depth mean masks out.
6. **Fusion census: nothing left to classify.** With and without fused
   instructions the 100-step record is identical and bit-equal to NEMO.
7. **G1 is unchanged.** Its 100-step certification test passes.

## 1. The walk (exploratory, before the preregistration)

Step 1 from the exact initial state, the card at 8b17016ed. Cells not
bit-equal against NEMO's solver exit (sea level, barotropic U, V) and step
frames (level-1 U, V, T, S). Probe copied to explore/_probe_g2r4_handback.py.

| arm | ssh | U_b | V_b | u1 | v1 | T1 | S1 |
|---|---|---|---|---|---|---|---|
| production | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| hand back the loop's entry forcing | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hand back the forcing before the Coriolis subtraction | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hand back the depth-mean forcing | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hand back the 3-D right-hand side | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| production, fused instructions off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, algebraic simplifier off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, constant folding off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, new fusion emitters off | 612 | 591 | 642 | 594 | 633 | 0 | 0 |
| production, scalar code only | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The cut that the brief asked for (entry forcing, then exit, then the
after-loop values one at a time) stopped at its first step: the entry
forcing alone restores everything. Cutting further upstream, every hand-back
restores, even with constant folding off, because each one also changes what
the compiler sees. So the hand-back cannot name a statement below the 3-D
right-hand side; the scalar-code control names the class instead (a vector
reduction order), and the first reduction on the path is the one cited.
For this deck sea level and the barotropic velocities leave the loop as
its last substep, unchanged (AB3/AM4 filter); only the two transport
averages take one division after the loop, and the entry-forcing hand-back
had already restored them.

## 2. Predictions

| prediction | outcome |
|---|---|
| R4-P1 with the option, step 1 is NEMO's bit for bit on all seven fields, no flag | HOLDS: 0 cells on each |
| R4-P2 ascending replay equals the NEMO-equal depth mean; a 4-lane order differs somewhere | HOLDS: 0 of 15,853 u and 0 of 15,943 v faces; 4-lane differs on 104 and 77 |
| R4-P3 100 steps, first over the bar later than 16 on every field | HOLDS: never over the bar; never not bit-equal |
| R4-P4 given-entry rows stay at the bar | HOLDS: 110 of 110 bit-equal |
| R4-P5 G1 unchanged | HOLDS: 4 passed |
| R4-P6 certification (no direction predicted) | CERTIFIED, section 5 |

## 3. The statement and the landing

NEMO, vector-invariant momentum (ORCA2's `ln_dynadv_vec`):
`Ue_rhs = SUM( e3u_3d(1:jpkm1)*uu(1:jpkm1,Krhs)*umask(1:jpkm1) ) * r1_hu_0`
and the same for V (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179`).
A SUM over a rank-1 section, added in ascending k by gfortran.

**Offline replay** (rule 5; final/rhs_frame.json, arm `rhs_frame`). The
statement in numpy, ascending k, with NEMO's mesh file (`hu_0` summed level
by level, `r1_hu_0 = ss/(hu_0 + 1 - ss)`), on the traced solver's 3-D
right-hand side: equal to the card's depth mean on every wet face,
including the 68 of the fold row. The same values added in four lanes
differ on 104 u and 77 v faces. The traced card's loop-entry forcing equals
NEMO's on every cell (final/substeps_kt1.json, row `entry.zu_frc`), which
ties the replay to NEMO.

**Landed.** Option `barotropic_rhs_ascending_level_sum`, default False,
appended as the last configuration field, selected by the G2 card only
through its own two-entry rung table. The three sums of the card's depth
mean (u, v, the fold-unit v) use the existing ascending-k helper, promoted
from private to public for this second caller (three call sites renamed).
Searched before building: the model for an ordered level sum (found that
helper in the barotropic module, used only by the stage correction) and
the card for an existing option on this statement (none).

- G1: not selected. One level: a sum of one term has no order.
- Caller grep (committed test): one assignment in the package, the shared
  G1/G2 builder's. Library default and every other card unchanged.
- ORCA2 pointer: ORCA2 runs the same compiled statement on 31 levels and
  does NOT have this option. Not selected for ORCA2 here (another lane's
  card).
- Effect on the stage-1 handoff: the solver walk's "ordinary handoff"
  barotropic U and V now equal NEMO's exit (591 and 642 cells before, 0
  after).

## 4. The stage-1 right-hand side (round-2 R2-P3, round-1 G-P6)

NEMO's frame is the 3-D right-hand side after the barotropic solve returns;
the pressure gradient assigns it first, then viscosity and Coriolis add.

| step 1, level 1 | u | v |
|---|---|---|
| wet faces | 15,853 | 15,943 |
| card against NEMO, wet faces not bit-equal | 0 | 0 |
| card against NEMO, dry faces not bit-equal | 1,160 | 1,049 |
| card nonzero on dry faces | 0 | 0 |
| NEMO's largest dry-face value | 5.12e-6 | 5.21e-6 |

NEMO's sco pressure gradient writes level 1 without a mask
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynhpg.f90:393-394`). The depth mean
multiplies by the mask, and no dry cell differs in any restart, so those
values reach nothing scored. R2-P3 (the solver's entry forcing) is the
substep walk's entry row: 0 cells differ. Both are now VALIDATED
exposures, both gated by tests.

## 5. CERTIFICATION

Rung G2 (G1 plus ORCA2's real bathymetry and partial cells, 30 wet levels)
is certified on the operator's 100-step record at the campaign's 1e-15 bar:

- INDEPENDENT, steps 1 to 100: sea level, barotropic U and V, level-1 U
  and V every step; level-1 T and S where NEMO wrote them (steps 1 to 49);
  all levels at steps 2, 4, 6, 8, 10, 20, ..., 100, split into full and
  partial cells; the six barotropic histories at steps 10 to 100. Every one
  bit-identical: largest difference 0.0, no dry cell differs.
- GIVEN-NEMO-ENTRY (entries 1, 3, 5, 7, 9; one and two steps): 110 of 110
  rows bit-identical.
- The before arm (same commit, option off) reproduces round 3 digit for
  digit (step 100: sea level 6.41e-14, level-1 V 1.01e-12; first over the
  bar step 16), so the two arms differ in the one option.

| step | ssh before | ssh after | level-1 U before | level-1 U after | level-1 V before | level-1 V after |
|---|---|---|---|---|---|---|
| 1 | 1.13e-17 (612) | 0 | 2.09e-18 (594) | 0 | 4.46e-18 (633) | 0 |
| 2 | 1.39e-17 | 0 | 2.49e-18 | 0 | 4.69e-18 | 0 |
| 16 | 2.04e-15 | 0 | 1.70e-15 | 0 | 7.75e-16 | 0 |
| 50 | 5.45e-14 | 0 | 6.84e-14 | 0 | 1.29e-13 | 0 |
| 100 | 6.41e-14 | 0 | 2.65e-13 | 0 | 1.01e-12 | 0 |

Limits of the claim: one record, one machine, one compiler build; levels 2
to 30 at steps 2, 4, 6, 8 have only twin b as a witness (round 3); closed
cards not re-scored; single review. The card still names
`multi_level_partial_cell_step` as unmeasured; G1 kept its labels after
certification too, so I left it (offered below).

## 6. Fusion-off census (item 4)

The flag stays a diagnostic; no card selects it.

| 100-step record | rows not bit-equal |
|---|---|
| card as landed | 0 |
| card as landed, fused instructions off | 0 |
| fusion-class remainder | 0 |
| other remainder | 0 |

## 7. Gates

| claim | gate | result |
|---|---|---|
| the landing is needed | step-1 test with its before arm | pass (before: over 500 cells on sea level and U_b; after: 0 on all seven) |
| the replay is NEMO's order | unit test against a Python loop; record test (0 unequal, fold row included, 4-lane > 50) | pass |
| the option is G2's only | package scan test; resolved-pair test; last-field test | pass |
| certification | 100-step test, every field and class, 60 history rows | pass |
| scorers are not vacuous | score plant and level plant | both exit 1 |
| right-hand-side frame | record test, wet faces 0, dry faces > 1000 | pass |
| G1 certification | G1 100-step test | 4 passed |
| citations | citation gate on this receipt | see section 10 |
| dual review, claim review before code | NO GATE | single review only (lane rule 6); claim review NOT done |

Focused tests, synchronous, final tree: the two G2 files, the tanks file and
the vertical-solver unit file, 46 passed (tests_final_fast.log); the seven
slow G2 record tests, 7 passed (tests_final_slow.log, at 6e37798cd) and
the right-hand-side test again after the review fix, 1 passed; the G1
100-step certification, 4 passed. Ratchets: 6084 passed, 3 failed, the
same three as round 3, in files this diff does not touch (a forcing test
file, the land restart module, the slab land dispatcher). NOT RUN: the rest
of the G1 ladder file; any closed-card ladder.

## 8. Review

Single review (codex): **"VERDICT: findings"** (codex_review_r13.log,
scope 8b17016ed..6e37798cd). Its other checks: "Defaults, G2-only
selection, option plumbing, face slicing, and the 100-step scored-field/
depth-class coverage otherwise check out."

- "CONFIRMED — The offline replay is not compared with NEMO's
  `Ue_rhs/Ve_rhs`." True: NEMO does not record that array. The replay is
  tied to NEMO through the substep walk's entry-forcing row (0 cells), and
  the landing is judged on NEMO's own fields, not on the replay. Stated in
  section 3; not fixable without a new record.
- "CONFIRMED — The replay explicitly excludes the northern fold-unit V
  row". Fixed in 30450a1f3: the row is scored (its fold unit reads NEMO's
  raw mesh, which is the mesh file's); 0 of its 68 wet faces differ; the
  test pins that every wet face is scored. Not re-reviewed.

## 9. Corrections and choices

Corrections to my own work:
1. My first unit test asserted that a lone compiled sum differs from
   Fortran's order. It does not: XLA adds in order when the sum stands
   alone and vectorises only inside the full step. The test now checks the
   helper's order; the option's effect is pinned by the record test's
   before arm.

ASKED (round brief): name section 6's statement by the finer hand-back,
land it G2-card-selected, re-score 100 steps, certify, the right-hand-side
exposure, the fusion census, one review.

NOT FOLLOWED, with reason: "the after-loop values one at a time" (the
entry-forcing hand-back already restored everything, so the seed lies
before the loop).

UNASKED, each offered for revert:
1. The XLA compile flags of section 1 as diagnostics (none is selected by
   anything).
2. The option's name, and promoting the private ascending-sum helper to
   public (three call sites renamed; a ratchet bans cross-module private
   imports).
3. The new ladder arm `rhs_frame` and its replay.
4. The G2 100-step certification test (about four minutes).
5. The re-anchor tool rewrote 59 line pins in the citation map and 18 in
   one older receipt (lines moved by this round's edits).
6. Leaving the card's "unmeasured" label in place after certification.

Compliance, stated because no gate enforces it: search-before-build
(section 3); controlled comparison (before and after differ in one option,
same record, same script, same commit; the before arm reproduces round 3);
instrument validated before quoted (the replay by its NEMO tie and the
4-lane contrast; the scorers by their plants); non-vacuity (section 7).

UNVERIFIED:
- that the ORCA2 card would accept the option as built;
- the certification on another machine or compiler build (the defect class
  is compiler-dependent; the option removes this one site's dependence,
  not others the compiler might find elsewhere);
- the three ratchet failures on the base commit.

## 10. OPEN (for the lane's next rung, not for G2)

1. G3 = G2 plus ORCA2's real initial T and S (density, pressure gradient).
2. ORCA2 import of this round's option and of rounds 2 and 3's, by the
   ORCA2 lane.
3. A record of NEMO's `Ue_rhs` itself, if a direct witness of the depth
   mean is wanted (not needed for the certification).

## 11. Citation gate and evidence hashes

Citation gate on this receipt at commit 627216616 (citation_receipt.json):
PASS, 2 citations, 0 unmapped, map audit clean; the shifted-citation plant
on `stp2d.f90:178-179` exits 1; the gate's own test file: 17 passed. This
section was added after that run and holds no citation.

SHA-256 prefixes, round13/: final/independent_after.json=bb18f8cdac6ccb89,
final/independent_before.json=09e12ab9ffad668b,
final/independent_after_nofma.json=f3a74707186b43f6,
final/given_entry_after.json=5645e9f3e926e68e,
final/rhs_frame.json=5230df4355f0a00f,
final/substeps_kt1.json=24f33beedbc42f76,
final/plant_score.json=9eede49a55ebf2d2,
final/plant_level.json=dd6e2ac6305b5ad8,
explore/replay_ue_rhs.json=f29fcd2665a9b7de,
codex_review_r13.log=4b4b2d6bbb662a76,
tests_final_fast.log=11a1e87cde0728fa,
tests_final_slow.log=7ea822315b728796,
tests_final_slow_rhs.log=f03f54c7bf4d91b3,
tests_g1_100step.log=4de921eeaf62ec19,
tests_final_ratchets.log=b4b6223eac638e42,
citation_receipt.json=ba0fda64811b3402.
