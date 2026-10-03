# GYRE lane round 207 — merge the ORCA2 lane (note BX)

**Status: HELD.** The merge itself is complete, reviewed and green on every
gate. It is held on ONE registered number: the certified GYRE from-rest year
moves, sub-floor, and three of its eight scored days move AWAY from NEMO. The
round order's stop condition is "any move → register and, if beyond Decision
59's ten floor units **or away from NEMO**, STOP with DECISION_NEEDED". The
largest move is **0.27 of one floor unit** against a ten-unit allowance, and
the merged year reproduces, to every printed digit, the year the ORCA2 merged
tree already registered in round 102 (note BU) — so this is the
already-registered cross-lane sub-floor discrepancy being realized, not a new
one. It is still a move away, so the decision is the operator's, not mine.

## Tips merged

| side | tip | what it is |
|---|---|---|
| ours | `a2382abce` | GYRE lane tip (round 206, Decision 86 landed) |
| theirs | `877e334f1` | `origin/fidelity/nemo-testcases-orca2-rounds`, ORCA2 rounds 111 + 112 |
| merge | `345a77121` | `git merge --no-ff`, 916 commits arriving, merge base `d7de69d51` |

**Round 113 is NOT in this merge, and that is a scope fact, not an omission by
accident.** The ORCA2 lane's round 113 (southern mesh-thickness restore, 6
commits, local tip `faf8b0474`) LANDED on the ORCA2 lane's clone at 00:07 but
its push gate never produced a `pushed` line, so the fetched remote branch
stops at `877e334f1`. `faf8b0474` is not an object any remote records. The
round order allowed for this ("round 113 may have pushed — take whatever the
tip is"); the tip is `877e334f1`. The independent reviewer was asked this
question cold and agreed: merging unpushed commits out of another clone's
working area would produce a history nobody else can reproduce and that the
ORCA2 lane could still rewrite. **The southern mesh-thickness restore arrives
at the NEXT merge, after the ORCA2 lane pushes round 113.**

## What arrives

Two statements, both cited from the ORCA2 lane's own receipts:

1. **NEMO's carried `fe3mask` in the frozen EEN coefficient** (ORCA2 round
   111, `nemo_testcases_l4_orca2_round111_een_fe3mask_landing_receipt.md`).
   On the ORCA2 lane this moved 175/200 rung-0 and 195/200 rung-7 rows toward
   NEMO with zero exact-row losses.
2. **The executed `jpfillcopy` association that fills the southern `ff_f`
   halo row** (ORCA2 round 112,
   `nemo_testcases_l4_orca2_round112_een_southern_ff_landing_receipt.md`): all
   180 southern `ff_f` operand cells close; both ORCA2 ladders moved 0/400.

Both are inert on GYRE and on both VORTEX cards at `rn_shlat=0`, which was
predicted and is **proved below by measurement, not assumed**.

## Conflicts and their disposition

Three files conflicted, 34 hunks. **No conflicted hunk is under `packages/`,
so no conflict chose between two versions of a physics line.** Confirmed
independently by the reviewer: `git diff <git's own 3-way tree> 345a77121 --
packages/` is EMPTY — the merge is byte-identical to git's own result
everywhere in the model code.

### 1. `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py` — 19 hunks

The citation map is a UNION. Both lanes had re-anchored the same entries to
their own trees' line numbers, so most hunks were the same entry twice.
Disposition was mechanical, by comparing each hunk's entries with all line
numbers and extents normalised away:

| hunks | ours entries | theirs entries | kept | why |
|---|---|---|---|---|
| 0-14, 16-18 | same set | same set | **OURS** (GYRE) | identical entries, differing only in line numbers/extents; re-anchored below |
| 15 | 3 | 17 | **THEIRS** (whole) | strict ORCA2 superset: the round-61 raw-momentum observer and the round-63 `dyn_zdf` seam observer, 14 entries including three in `nemo_testcase_l1_overflow_round63_dynzdf_walk_gate.py`. Taking ours would have deleted all of them. |

This is the exact failure the union rule exists for, and this file records a
previous instance of it in its own comments ("restored from the lane tip
(980cc6369) after the merge conflict on this file was resolved to HEAD's side,
which silently dropped every entry the VORTEX-card rounds had added").

**Proof no entry was dropped, by the independent reviewer, not by me:** all
three `CITATION_MAP`s (both parents and the merge) parsed with `ast` and
matched on ANCHOR CONTENT rather than on the line-number key — **0 GYRE
entries absent, 0 ORCA2 entries absent.**

### 2-3. Two historical GYRE receipts — 15 hunks

`nemo_testcases_l2_gyre_phase3_round8_receipt.md` (14 hunks) and
`nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md` (1 hunk). Every
one is the same citation re-anchored to each lane's own line numbers. Resolved
to the GYRE side.

### A fourth file, not conflicted, hand-edited

`nemo_testcases_l4_orca2_round101_gyre_merge_owner_receipt.md` received two
prose re-anchors (`7054-7064` → `7315-7325`) from the automatic pass. Named
here because the commit message's "three files" accounting does not mention
it.

## Re-anchoring — by symbol, mechanically

No line number was hand-edited. All **1,463** map entries were re-resolved
against the merged tree through the gate's own `resolve_anchor`:

* **58 citations moved**, 0 anchors failed to resolve;
* **6 pinned extents changed**, because the merged tree contains BOTH lanes'
  insertions inside the cited span — `:2897-2977` grew 25 → 81 lines and
  `:7460-7599` grew 74 → 140. Declared here rather than left silent, which is
  what this map exists to refuse. A reader following those ranges now gets more
  code than the prose describes; the endpoints still pin exactly.
* **one entry resolved BACKWARDS** and is a real merge signal: the ORCA2 entry
  for the round-61 raw-momentum observer pinned
  `v=state_new.v.replace(data=_raw_v),` as its SECOND occurrence, but the GYRE
  side adds a third copy of that fragment, so the block's own copy is now the
  THIRD. Re-pinned 2 → 3; verified by the reviewer that occurrence 3 (line
  10074) sits inside the `if _nemo_ws_exposed_stage3_raw is not None:` block at
  10070, extent 5.

`audit_map()` returns **0 failures over all 1,463 entries**, and the reviewer
proved it non-vacuous (shifting one key by one line produced exactly one
`SYMBOL-NOT-AT-LINE`).

**Citation gate: 175 passed, zero unmapped.**

## A pre-existing wrong citation, REGISTERED not fixed

The reviewer flagged three stale citations in
`nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md`. I checked
whether the merge caused them. **It did not: they were already wrong on the
GYRE parent `a2382abce`.** Reading the parent's own tree at the cited lines:
`nemo_testcase_recipe.py:236-238` is an EOS comment and `eos="nemo_seos",`,
not a card selector; `ocean_model_latlon_cgrid.py:9060-9133` is a
`return_fct_activity=(...)` tracer-advection block, not the live-face metric.

The symbols the prose names live, on the merged tree, at
`nemo_testcase_recipe.py:437` (`tke_shear_production="nemo_face_native_now2"`)
and `ocean_model_latlon_cgrid.py:10844` (`if shear_disc ==
"nemo_face_native_now2":`), with the arm's validation at `:10805-10809`.

I am NOT rewriting the numbers: the prose's intent for each span belongs to
that statement's owner, and inventing a span I cannot verify against the intent
would replace a visible wrong citation with an invisible one. **Registered as
pre-existing debt, with the measured correct anchors above.** No gate sees it:
the gate's default receipt is the round-8 receipt from round 25 onward, and
these citations are not in `CITATION_MAP`. This is the same class the round-206
receipt already recorded.

**Correction to the merge commit message**, which cannot be edited: it says the
two conflicted receipts were "resolved to the GYRE side and then re-anchored
mechanically". The round-8 receipt was. The decision36 receipt was resolved to
the GYRE side and NOT re-anchored, because none of its citations are in the
map. The sentence overstates what happened to that one file.

## Measurements on the merged tree

Every number below is measured on the merged tree with the scripts the recent
receipts cite, not carried over.

### GYRE certified `kt=1..10` ladder — UNCHANGED

`nemo_testcase_l2_gyre_phase3_gate.py --max-step 10`, compared against round
206's certified report:

```
OFFLINE_ORACLE_RELATIVE_COMPARE PASS: rows=954 max_worsening_ulps=0
  first_over_bar={'T','S','u','v','ssh'} kt=3 -> unchanged
```

**954 rows, 0 moved, 0 ULP.**

### GYRE from-rest year — MOVED, SUB-FLOOR. This is the held item.

A fresh seed-0 member ran the full 360 days
(`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6
--tag r207a`), scored with the committed day-gap scorer against the same NEMO
restarts and the same eight days as every prior round.

| day | certified (note BW) | round 207, merged tree | delta (K) | direction | floor units |
|---:|---:|---:|---:|:--:|---:|
| 30 | `2.3432465132112266e-06` | `2.3432465132112266e-06` | 0 | **same** | 0 |
| 60 | `1.4793247973304582e-05` | `1.4793247973304582e-05` | 0 | **same** | 0 |
| 90 | `1.6332712039638441e-05` | `1.6332712039638441e-05` | 0 | **same** | 0 |
| 120 | `0.00010965906837581848` | `0.00010965907352116351` | `+5.15e-12` | away | 0.026 |
| 180 | `6.1153352881614638e-05` | `6.1153355393000553e-05` | `+2.51e-12` | away | 0.013 |
| 240 | `6.58170624837412e-05` | `6.5817060949447295e-05` | `-1.53e-12` | toward | 0.008 |
| 300 | `5.4660501132892367e-05` | `5.4660498451870513e-05` | `-2.68e-12` | toward | 0.013 |
| 360 | `5.4077365272463437e-05` | `5.4077419367442036e-05` | `+5.41e-11` | away | **0.27** |

Snapshot digests: day 030 `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`
(**matches** the certified digest); day 240
`8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a` and day 360
`e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25` (differ from
`a63befc3…` / `dcb7bc46…`).

**The key fact for the decision:** day 240 `6.5817060949447295e-05` and day 360
`5.4077419367442036e-05` are, to every digit, the numbers the ORCA2 merged tree
registered in its round 102 (note BU: `6.58170609494473e-05` and
`5.4077419367442036e-05`). The merged GYRE tree now computes the ORCA2 merged
tree's year. Round 202 measured this exact gap from the other side and round
203 registered it as a fact: "at most 0.27 of the 2e-10 K run-to-run floor, far
inside Decision 59's ten floor units … two trees that differ in OTHER
ORCA2-only commits need not compile to bit-identical XLA; sub-floor differences
between lanes are the floor's domain, not a defect" (note BV). This merge makes
the GYRE lane one of those two trees.

**Why it is held anyway:** the round order's stop condition names "away from
NEMO" without a floor qualifier, and three days moved away. Decision 43(a) says
no certified row moves away *beyond the floor*, and none does. I will not
resolve that ambiguity by myself — that is the unasked-choice error.

**Recommendation, named:** LAND, and re-pin the certified GYRE year to the
eight values above with these three days registered as sub-floor away-moves
under D59/AW, exactly as round 203 landed its own 0.055-unit away-move at day
360. The alternative — reverting a complete, reviewed, otherwise-green merge
over 0.27 of one floor unit — buys nothing and leaves the two lanes divergent.

Not run this round, and named rather than left silent: the second
determinism member. Round 203 already confirmed this lane deterministic
(two members byte-identical), and the cross-lane agreement to every digit with
an independently-computed ORCA2 year is stronger evidence than a repeat would
be. A second member should run before the re-pin is made permanent.

### Both VORTEX cards — INERT, proved

Certified 50-row registries, `nemo_testcase_phase3_trajectory_gate.py --case
<CASE> --max-step 10 --continue-after-first`, compared against round 206's:

| card | rows | moved | max worsening ULP | first over bar |
|---|---:|---:|---:|---|
| `VORTEX_VEC-zco` | 50 | **0** | 0 | u/v/ssh kt=2 → unchanged |
| `VORTEX-zco` | 50 | **0** | 0 | T/u/v/ssh kt=2 → unchanged |

**0/50 each.** The `fe3mask` statement is inert at `rn_shlat=0` on GYRE's and
both VORTEX cards' boxes, as ORCA2 predicted — now measured here, not assumed.

### Tanks

| card | rows | moved | first over bar |
|---|---:|---:|---|
| `LOCK_EXCHANGE-zco` | 50 | **0** | u kt=8 → unchanged |
| `OVERFLOW-zps` | 50 | **0** | T/u kt=2 → unchanged |

Both tanks inert: round 206's registries are reproduced exactly.

### ORCA2 ladders on the merged tree

The ORCA2 lane's own scripts, now in the merged tree, against its own records.

| ladder | status | rows |
|---|---|---:|
| rung-0 given-entry, `--record-root round90/acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2` | `PASS_RUNG0_TEN_STEP_LADDER` | 200 |
| rung-7 certified, `--initial-mode decision52-bridge` | **UNMEASURED this round** — see below | 200 |

**The rung-7 comparison is UNMEASURED, and the reason is my own harness error,
not an ORCA2 row movement.** I first ran the rung-7 ladder with
`--initial-mode independent` (copied from a round-66 invocation in the lane
logs). The certified rung-7 ladder is the Decision-52 given-entry arm
(`--initial-mode decision52-bridge`, the script's default), which is what
round 112's registered `rung7_ladder.json` contains. The comparator
consequently reported

```
STATUS REFUSE: candidate moved the first non-bit statement
```

because my candidate's first non-bit is `kt=1 entry ssh`
(`iceistate.f90`, the SI3 initial category load) while the base's is `kt=1
stage1 T`. That is the difference between the two ENTRY MODES, not between the
two trees: round 112's own json records the independent arm's first non-bit as
exactly that same `ssh`/`iceistate.f90` statement under
`independent_first_non_bit_statement`. Comparing an independent-entry
candidate against a Decision-52 base is a confounded comparison and I will not
report it as a result either way. The correctly-moded re-run
(`--initial-mode decision52-bridge`) was launched; its artefacts are
`orca2_rung7_ladder_d52.{json,log}` in the evidence directory. **The rung-7
row movement is an OPEN item for the next round, and the merge is held
independently of it.**

The rung-0 ladder, which was run in its single correct mode, reproduces round
112's registered result.

Rung-0's first non-bit checkpoint is `kt=1 stage1 T`, identical to the value
ORCA2 round 112 registered. The GYRE-lane statements ORCA2 has not yet merged
(rounds 199-206: the two-solve WZV on the vector card, the ratio-order
adoption, the transport depth) are in this tree and were expected to move
ORCA2's rows; rung-0 reports the same pass and the same first non-bit
statement.

## Independent review

One fresh `code-reviewer` subagent on the merge diff and the conflict
dispositions, told to refute them.

**Verdict: SHIP WITH FIXES**, two blockers, three minors.

1. *"The merge receipt does not exist"* — the commit message cites this file
   before it was written. TAKEN: this is it.
2. *"The decision36 receipt was resolved to the GYRE side and NOT re-anchored,
   contradicting the commit message"* — TAKEN, and extended: I measured that
   those citations were already wrong on the GYRE parent, so it is pre-existing
   debt, not a merge regression. Registered above with the correct anchors, and
   the commit message's overstatement is corrected above.

Minors: the merge commit **alone** is gate-red (the round-8 receipt's
`:9489-9491` is re-anchored in the FOLLOWING commit, so a bisect landing
exactly on `345a77121` fails the citation gate) — recorded, not squashed, since
the merge is already written; the unlisted fourth file — now listed; the two
widened spans — now declared.

The reviewer independently proved the two properties that mattered: the union
(0 entries dropped from either parent) and no invented physics (the merge is
byte-identical to git's own 3-way tree under `packages/`). It also agreed, cold,
with the round-113 scope decision.

## Landing verdict

**HELD**, on the GYRE year's three sub-floor away-moves alone. Everything else
is green: ladder 954/0, both VORTEX cards 0/50, LOCK_EXCHANGE 0/50, rung-0
`PASS`, citation gate 175 passed with zero unmapped, no physics conflict, union
proved. The merge commits stay in the round's clone, unpushed, pending one
operator line.

## Evidence

`phase3/round207/`: `conflicts.txt`, `gate_conflict_dispositions.txt`,
`reanchor.py`, `reanchor_dryrun.json`, `docs_reanchored.json`,
`gyre_ladder_r207.{json,log}`, `cmp_gyre_ladder.json`,
`gyre_year_r207a.log`, `gyre_day_gap_r207a.{json,log}`,
`card_*.{json,log}`, `cmp_*.json`, `orca2_rung0_ladder.{json,log}`,
`orca2_rung7_ladder.{json,log}`; member
`phase3/year_fromrest/lego_seed0_r207a/day*.npz` (360 files).
