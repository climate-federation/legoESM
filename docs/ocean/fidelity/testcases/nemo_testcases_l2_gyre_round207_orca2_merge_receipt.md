# GYRE lane round 207 — merge the ORCA2 lane (note BX)

**Status: LANDED.** The merge adopts the ORCA2 lane's `fe3mask` and southern
`ff_f` statements, is green on every gate, and **re-pins the certified GYRE
from-rest year** to this tree's measured values.

The year moves, sub-floor. The operator took the decision (2026-10-03,
superseding this round's own stop condition, which was too strict): the move
is inside the lane's gate, because **Decision 43(a) forbids a certified row
moving away BEYOND the floor** and none does, and **Decision 59 / note AW
admits a registered year move under ten floor units** — the largest here is
**0.27 of one floor unit**. Note B35 on the ORCA2 lane already applied this
same reading to this same cross-lane gap. Note BW's year values are SUPERSEDED
by the table below.

## Tips merged

| side | tip | what it is |
|---|---|---|
| ours | `a2382abce` | GYRE lane tip (round 206, Decision 86 landed) |
| theirs | `877e334f1` | `origin/fidelity/nemo-testcases-orca2-rounds`, ORCA2 rounds 111 + 112 |
| merge | `345a77121` | `git merge --no-ff`, 916 commits arriving, merge base `d7de69d51` |

**Rounds 113 AND 114 are NOT in this merge, and that is a scope fact, not an
omission by accident.** The ORCA2 lane's round 113 (southern mesh-thickness restore, 6
commits, local tip `faf8b0474`) LANDED on the ORCA2 lane's clone at 00:07 but
its push gate never produced a `pushed` line, so the fetched remote branch
stops at `877e334f1`. `faf8b0474` is not an object any remote records. The
round order allowed for this ("round 113 may have pushed — take whatever the
tip is"); the tip is `877e334f1`. The independent reviewer was asked this
question cold and agreed: merging unpushed commits out of another clone's
working area would produce a history nobody else can reproduce and that the
ORCA2 lane could still rewrite. **The southern mesh-thickness restore arrives at the NEXT merge.** The ORCA2
lane has since pushed `239ee7007`, which carries BOTH round 113's southern
mesh-thickness restore and round 114's southern frozen-mask zero-fill. Neither
statement is in this tree; both are the next merge round's adoption, and until
then this lane's ORCA2 rows are measured without them.

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

### GYRE from-rest year — MOVED SUB-FLOOR, AND RE-PINNED

A fresh seed-0 member ran the full 360 days
(`nemo_testcase_l2_gyre_year_fromrest.py --member 0 --days 360 --snap-steps 6
--tag r207a`), scored with the committed day-gap scorer against the same NEMO
restarts and the same eight days as every prior round.

**THE CERTIFIED GYRE YEAR, AS OF THIS ROUND.** These eight values and the three
digests below replace note BW's; every later "GYRE byte-identical" check on
this lane uses them.

| day | T rms vs NEMO (K), certified from here | previous pin (note BW) | delta (K) | floor units | direction |
|---:|---:|---:|---:|---:|:--|
| 30 | `2.3432465132112266e-06` | same | 0 | 0 | unchanged |
| 60 | `1.4793247973304582e-05` | same | 0 | 0 | unchanged |
| 90 | `1.6332712039638441e-05` | same | 0 | 0 | unchanged |
| 120 | `0.00010965907352116351` | `0.00010965906837581848` | `+5.15e-12` | 0.026 | **away** |
| 180 | `6.1153355393000553e-05` | `6.1153352881614638e-05` | `+2.51e-12` | 0.013 | **away** |
| 240 | `6.5817060949447295e-05` | `6.58170624837412e-05` | `-1.53e-12` | 0.008 | toward |
| 300 | `5.4660498451870513e-05` | `5.4660501132892367e-05` | `-2.68e-12` | 0.013 | toward |
| 360 | `5.4077419367442036e-05` | `5.4077365272463437e-05` | `+5.41e-11` | **0.270** | **away** |

Five days moved, three away and two toward, every one of them under
**0.27 of the 2e-10 K run-to-run floor** against Decision 59's ten-unit
allowance. **Admitted and registered under Decision 59 / note AW.** Decision
43(a) is satisfied: no certified row moves away *beyond the floor*.

**Certified snapshot digests, as of this round** (file-level SHA-256 of the
member's daily `.npz`):

| day | certified from here | previous (note BW) |
|---|---|---|
| 030 | `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba` | unchanged |
| 240 | `8b9cd60475626373d9a2fec0baa508f06c1f91aed4c3d7a24fc5d8b3f5878c8a` | `a63befc3…` superseded |
| 360 | `e3e0a068346c7866f0a318bb32141f2fb326cc05c158a1c95bc39b091f585e25` | `dcb7bc46…` superseded |

**Why the move is the floor's domain and not a defect.** Day 240
`6.5817060949447295e-05` and day 360 `5.4077419367442036e-05` are, to every
digit, the values the ORCA2 merged tree registered in its round 102 (note BU:
`6.58170609494473e-05`, `5.4077419367442036e-05`). This merge brings 916 ORCA2
commits into the GYRE lane, so the two trees now compile to the same XLA and
compute the same year. Round 202 measured this gap from the other side and
round 203 registered it: "two trees that differ in OTHER ORCA2-only commits
need not compile to bit-identical XLA; sub-floor differences between lanes are
the floor's domain, not a defect" (note BV). Agreement to every digit with an
independently-computed year on another lane is stronger evidence than a
repeated member would be.

**Non-vacuity — the old pin fails against this year.** A re-pin nobody can
fail is not a re-pin, so both pins were run against the measured member
(`phase3/round207/year_repin_nonvacuity.txt`):

```
old pin (note BW / round 203): FAIL (5 of 8 days disagree)
new pin (round 207, merged tree): PASS (0 of 8 days disagree)
max |move| = 0.270 floor units (day 360), allowance 10
```

**Where the year is pinned.** A grep of `docs/`, `tests/`, `scripts/` and
`packages/` for the superseded day-240/360 values and digests returns ONLY
markdown receipts — **no test, gate or script hardcodes the year**; the lane's
gates always recompute it. The receipts for rounds 202-206 are historical
commitments recording what was certified when they ran and are deliberately
NOT rewritten; this receipt is the live pin.

**Not run this round, and named rather than left silent:** the second
determinism member. Round 203 confirmed this lane deterministic (two members
byte-identical) and the cross-lane digit-for-digit agreement above is the
stronger check, but a second member should run on this tip before the next
year-sensitive landing.

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
| rung-7 certified, `--initial-mode decision52-bridge` | `LADDER_MEASURED` | 200 |

Compared against round 112's registered ladders with the ORCA2 lane's own
comparator (`nemo_testcase_l4_orca2_round111_ladder_compare.py`):

```
STATUS PASS_R111_ORCA2_LADDER_COMPARE
rung0: row_count 200, moved_row_count 0, bit_identical_losses 0, first non-bit unchanged
rung7: row_count 200, moved_row_count 0, bit_identical_losses 0, first non-bit unchanged
```

**0/400 rows moved, zero exact-row losses, both first non-bit statements
unchanged** (rung-0 `kt=1 stage1 T`; rung-7 `kt=1 stage1 T`, `UNATTRIBUTED`,
with the same ruled-out `trasbc.f90:314-328` runoff statement and the same
233,341 / 231,291 cell counts).

A real finding, stated because the round order expected the opposite: the
GYRE-lane statements ORCA2 has NOT yet merged — the two-solve WZV on the
vector card (round 199), the shared stage-one ratio order (round 203) and the
transport depth (round 206) — were predicted to move ORCA2's rows. **They move
none of the 400.** All three are card-scoped or shared-path statements whose
ORCA2 branch resolves the same way, so ORCA2's trajectory is unaffected.

### One cost owned

My first rung-7 run was mode-confounded and I nearly reported its refusal as a
merge regression. I ran the rung-7 ladder with
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
(`--initial-mode decision52-bridge`) is the measurement reported above, and it
PASSES. The refusal was entirely my confound; artefacts
`orca2_rung7_ladder{,_d52}.{json,log}` keep both arms so the claim is
checkable. The lesson is the campaign's own controlled-comparison rule: an arm
copied from an old log is not the certified arm until its mode is diffed
against the base's.

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

**LANDED.** GYRE ladder 954 rows 0 moved; the certified year re-pinned to the
eight values above with five sub-floor moves registered under Decision 59 /
note AW; both VORTEX cards 0/50; both tanks 0/50; both ORCA2 ladders 0/400
with unchanged first non-bit statements; citation gate 175 passed with zero
unmapped; the lane's push-gate battery 136 passed in 1139.11 s; DINO month
gate run by `land.sh`. No conflicted hunk touched a physics line and the union
of both lanes' citation maps is proved complete.

Carried OPEN to the next round: adopt ORCA2 `239ee7007` (rounds 113 + 114);
the decision36 receipt's three pre-existing stale citations; the two widened
citation spans.

## Evidence

`phase3/round207/`: `conflicts.txt`, `gate_conflict_dispositions.txt`,
`reanchor.py`, `reanchor_dryrun.json`, `docs_reanchored.json`,
`gyre_ladder_r207.{json,log}`, `cmp_gyre_ladder.json`,
`gyre_year_r207a.log`, `gyre_day_gap_r207a.{json,log}`,
`card_*.{json,log}`, `cmp_*.json`, `orca2_rung0_ladder.{json,log}`,
`orca2_rung7_ladder.{json,log}`; member
`phase3/year_fromrest/lego_seed0_r207a/day*.npz` (360 files).
