# Grid ladder rung G3 — round 3 receipt (gridladder_r16)

Status: **LANDED — G3 CERTIFIED.** kt = 1..10 is bit-identical to NEMO on
every field, every level, both depth classes, every region, under both
labels (INDEPENDENT and GIVEN-NEMO-ENTRY). kt = 11 is NEMO's registered
stop.
Preregistration: docs/ocean/fidelity/PREREG_nemo_testcases_gridladder_g3_round3.md
(fe46ccb45, before any comparison).
Code: 07a184943 (the statement, card selection, ladder before-arm, tests),
07e18b56a (review fixes), 7c1b5bd4e, 802527439, ff657b977 (citation pins moved by
the edit), this receipt's commit.
Record: gridladder_rounds/round15/oracle_g3_r15 (operator's round-2
acquisition, ACQUISITION_DONE). Evidence: gridladder_rounds/round16/.
Decision 120 pending: as-built deck; the hand-edit cells play no part
(0 unequal among them, kt 1..10, before and after the landing).

## Take-aways

1. **The first unequal bit was one u cell at step 3**: level 13, full
   class, (y, x) = (21, 32) in NEMO's index order, 3.5e-18 (one last
   place). Level 1 and every 2-D field stayed equal until step 6; the bar
   was first crossed at step 9 (u, partial class 2.6e-15).
2. **The statement is the vertical momentum solve's interface thickness.**
   NEMO divides by `e3w_1d(jk)*(1+r3u(Kmm))` (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:192-198`,
   V twin `:361-367`), and at stage 3 its stored r3u(Kmm) is the MEAN OF THE
   TWO RATIOS, `r1_2*(r3u(Kbb)+r3ua)` (`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217`,
   under `:48`). The card took the ratio of the mean sea level.
3. **Landed** as a new option, selected by the G3 card only, that feeds the
   existing stage-ratio helper (the one round 7 of G1 landed for vorticity)
   into the solve. After it: 0 unequal cells anywhere, kt 1..10.
4. **Past the record** (kt = 11, NEMO's abort state): sea level, u, v and
   all three stop-check maxima now identical (|ssh| was one digit off);
   T and S differ in ONE cell each, at the bar (1.2e-16, 1.8e-16). Open,
   not part of G3's record.

## 1. Before the landing (INDEPENDENT, tree f797baf65 + prereg)

| kt | first unequal | u all levels (full / partial) | v | T / S |
|---|---|---|---|---|
| 1-2 | none | 0 / 0 | 0 / 0 | 0 |
| 3 | u, level 13 | 1 cell 2.3e-18 / 0 | 0 | 0 |
| 4 | u, level 1 | 3 cells 2.9e-17 / 0 | 0 | 0 |
| 5 | u, v | 27 / 1 | 39 / 0 | 0 |
| 6 | sea level, uu_b, vv_b | 1,692 / 45 | 1,431 / 49 | 0 |
| 8 | T, S | 61,754 / 2,290 at bar | 62,632 / 2,239 | 13 / 11 |
| 9 | **u over** | partial 2.6e-15 | at bar | at bar |
| 10 | u, v over | 1.6e-14 / 9.9e-15 | 5.3e-15 / 2.9e-15 | at bar |

Spread, not a new statement each step: one cell at kt 3, three at kt 4,
the 2-D fields only from kt 6, T/S only from kt 8. Region and depth-class
splits at kt 10 as in round 2 (open water dominant); the origin cell is
open water, not fold, seam, south wall or coast.

## 2. Attribution (offline replay, rule 5)

| check at kt 3 (entry = NEMO's restart 2, bit-equal to the card's) | result |
|---|---|
| NEMO's dyn_zdf replayed on the card's own stage-3 inputs vs NEMO's restart 3, all wet u/v | 0 unequal |
| card's solve vs that replay | 1 unequal: the same cell, level 13 |
| card's solve input vs the replay's input | 0 unequal |
| card's solver arithmetic fed the replay's coefficients vs the replay | 0 unequal |
| NEMO's stage-2 r3u vs r1_2*(r3u(Kbb)+r3ua) from NEMO's own frames, kt 1..10 | 0 unequal at every step |

So the solver is NEMO's; one coefficient operand (the "now" ratio) was
not. The observer is passive (checked by the arm). CONFIRMED by the
landing: the cell and every later difference vanish.

## 3. After the landing (07a184943 model code)

| label | steps | rows | unequal |
|---|---|---|---|
| INDEPENDENT | kt 1..10 | sea level, uu_b, vv_b, level-1 u v T S every step; u v T S all levels kt 2..10; six barotropic histories kt 10 | 0 |
| GIVEN-NEMO-ENTRY | entries 1, 3..9, one step and two steps | 2-D and level 1 at the entry step; all levels at the next step (kt 2..10) | 0 |
| chain on round 1's record | kt 2, 10, 11 | as above; kt 11 abort state | 0 at 2 and 10; kt 11: T 1, S 1 (at bar) |

Not covered: GIVEN-NEMO-ENTRY one-step at kt 2 (no NEMO restart of step 1;
kt 2 is covered by entry 1's two-step rows). The review fixes add only
refusals, so the post-review tree computes the same numbers (step-3 test
re-run on it: passed).

## 4. Predictions

| prediction | outcome |
|---|---|
| P1 kt 1 bit-identical | HELD |
| P2 first unequal at the bar, crossing later by growth | HELD (kt 3 at 2.3e-18; over at kt 9) |
| P3 given-entry reproduces it in one step | HELD (entry restart 2 equals the card's state; the step-3 cell is the card's alone) |
| P4 first unequal stage in the density path | **REFUTED**: the vertical momentum solve's thickness ratio |

## 5. Shared code and selection

| change | where | who selects |
|---|---|---|
| option: dyn_zdf reads NEMO's stored stage ratio | model config, implicit-mixing step | G3 card only (`_ORCA2G_ZDF_STAGE_R3U`) |
| refusal if set where it would do nothing (no NEMO face thicknesses, not RK3-WS, no 1-D interface ladder) | same | — |
| before-arm switch | G2 ladder script | — |

Library default False; G1, G2, ORCA2 and every other card unchanged
(the only package assignment is the G3 rung table; test pins it).
Searched before building: the stage-ratio helper and its pair (reused),
the ladder's replay arms (reused), the existing before-arm pattern
(copied). ORCA2 compiles the same dyn_zdf statement; its card does not
select the option: UNMEASURED there.

## 6. Tests (synchronous)

- Fast focused (G3, G2 ladder, G2 card, literal-solver unit): 38 passed.
- G3 slow: chain kt 2/10/11 and the new step-3 non-vacuity test, 2 passed
  (469 s); step-3 test re-run after review fixes, 1 passed.
- Non-vacuity: deselected, the card's step 3 has exactly one unequal u
  cell (level 13, full); selected, 0 everywhere. Refusal plant: setting the
  option without NEMO's face thicknesses raises.
- G2 slow (step 1 ascending sum, step 2 all levels): 2 passed — G2 untouched.

## 7. Review

Single review (codex): **"VERDICT: findings"** — "static tracing found
operand order, Kbb/ssha pairing, Kaa isolation, signs, and U/V staggering
correct" (codex_review_r16.log).

- "Setting the flag can silently do nothing": fixed, three refusals; plant
  test added.
- "test checks only full cells for non-U fields": fixed, every class.
- "the rule also applies to multi-level G2, but its card leaves it off":
  not changed — the round brief scopes the landing to the G3 card. G2's
  100-step record is bit-identical without it. Open item 2.

## 8. Choices

ASKED (round brief): G3-card selection; as-built deck; kt 1..10 record.

UNASKED, each offered for revert:
1. The option's name and the refusal conditions.
2. GIVEN-NEMO-ENTRY entries 3..9 plus 1 (all the record allows).

No gate exists for: dual review (single, lane rule 6), claim review before
code (not done), non-vacuity (shown above).

## 9. Open

1. kt 11 (past the record): one T and one S cell differ at the bar.
   PLAUSIBLE only: the tracer solve's e3w(Kmm) also builds its ratio from
   the mean sea level rather than NEMO's stored r3t(Kmm). Unmeasured.
2. Whether G2 (and the ORCA2 pointer) should also select the option: same
   NEMO statement; G2 certified without it. A user question, not asked here.
3. Decision 120 (hand-edit cells): still pending; nothing here depends on it.
4. Next rung (forcing) per D117.

UNVERIFIED:
- that the restart of a 100-step deck at step 10 equals a 10-step deck's
  (now measured indirectly: both chains score 0 against each).

## 10. Citation gate and evidence hashes

Citation gate on this receipt at commit ff657b977 (citation_receipt.json):
PASS, 4 citations, 0 unmapped; the shifted-citation plant on the stage
ratio exits 1. On the preregistration: PASS, 0 citations. The edit moved
43 legoESM line pins in the gate's map and 12 spans in its default
receipt (re-keyed by an exact old-to-new line map; one range's extent
120 -> 122); gate tests 17 passed, all receipt-citation tests 196 passed.

citation_receipt.json=9cdfa80cffa7c672, citation_receipt_plant.json=65bcea07ec3df3fc,
citation_prereg.json=6d02687d8d2c7899, tests_citation_gate.log=305b114452eccd02,
tests_citation_all.log=5aaabb9eb3e53597.
SHA-256 prefixes, round16/: indep_kt1_5.json=abecbe6486282780,
indep_kt6_10.json=efbb8e3b37cfe7bd, zdf_replay_kt3.json=bae7f10b1a4ff5a4,
fix1_indep_kt1_5.json=589e1d6f8d4173e6, fix1_indep_kt6_10.json=d8956e9dbddd5c1d,
fix1_given_odd.json=6a42d8feab032ff6, fix1_given_even.json=317e4bcd0f219c38,
fix1_chain_kt2_10_11.json=8f1a5ce7279f47bb,
tests_focused_fast_r2.log=788a53f428b38da9, tests_g3_slow.log=5472f8a09c4737c1,
tests_g3_step3_slow_r2.log=493772d80f7a6873, tests_g2_slow.log=c61914c87f86896e,
codex_review_r16.log=30a86a493faea0f1, _probe_g3r3_zdf_cell.py=d3ed90e854ab6d57,
_probe_g3r3_r3u.py=fa5415fe433e48cc.
NEMO, round15/oracle_g3_r15/: p3_a step frame kt 3=669e2d85cde7994f,
p3_c restart 3=21dcba99d7b868c3, p3_b restart 2=5dfb6cc9f57866ba,
p3_a restart 10=36882e0bd78abdb9.
