# ORCA2 lane — receipt for merging the GYRE lane, 2026-09-25

Status: **SHIP the merge, HOLD the ORCA2 ladder certification.** The five
conflicts resolve from NEMO's compiled source and none of them moves a number.
GYRE is byte-identical to the GYRE lane's own post-merge trajectory. ORCA2's
ten-step ladder DOES move, and the two commits that move it arrive from the
GYRE lane and from GitHub main — not from any resolution taken here. That
movement is registered, one contributor is proven by a one-variable control,
and certifying or rejecting the new ladder is carried to DECISION_NEEDED.

## 1. Scope and provenance

Merge commit `20250df348cb6968741b133c68d57bd04318d3c8`, parents ORCA2 lane
`4fd7b3306` (tip `b03f78bb5` plus the preregistration, docs only) and GYRE lane
`d3631f884`, which already contains GitHub `main` as of 2026-09-25 through the
GYRE lane's own merge commit `839e988c8`. Merge base `dda3f3257`. ORCA2 was
186 commits ahead of that base, GYRE 701. Ordinary merge: no squash, no
rebase, two parents.

Commit range `4fd7b3306..65fb85435` (six commits): the preregistration, the
merge, the fold-refusal correction, the one-variable control, its entry-point
fix, and the review-finding closures.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_merge_gyre_2026-09-25.md`,
committed before the merge ran. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/merge_gyre_2026-09-25/`.

## 2. Textual conflicts

Exactly the five the preregistration named, no more and no fewer (M-P1
CONFIRMED).

| # | file | what each side held | resolution and the statement that decides it |
|---|---|---|---|
| 1 | `nemo_testcases_l2_gyre_decision36_nemo_face_shear_receipt.md` | the same sentence with different line numbers for the turbulence shear routing arm | neither parent's numbers survive: both are stale against the merged tree. The two spans were re-read and the sentence now names the ones that hold the statement. Raised by the independent review, closed in `65fb85435`. |
| 2 | `nemo_testcases_l2_gyre_phase3_round8_receipt.md` | fourteen citation-line shifts | each rewritten to the merged tree's own line, driven by the citation map rather than by choosing a side. |
| 3 | `ocean_model_latlon_cgrid.py` | ORCA2 appended one private measurement hook to a named tuple; GYRE appended fourteen | UNION, both blocks kept. Decided by inspection, not preference: every one of the tuple's construction sites in the tree is keyword-only, so the field order carries no claim. Reviewed independently by AST over all sites. |
| 4 | `ocean_pe_latlon_cgrid.py` (two hunks) | see section 3 | the fold hunk is the only conflict on a NEMO path. |
| 5 | `nemo_testcase_receipt_citation_gate.py` (seventeen hunks) | both lanes' pinned citations, with the shared ones drifted apart | UNION of the entry sets — 1010 entries, zero key collisions — then a mechanical re-anchor of every entry against the merged line numbers. Section 5. |

### 3. Conflict 4, the fold vertex thickness — the deciding statement

The helper builds the F-point thickness the energy/enstrophy vorticity scheme
consumes. Three positions met here.

- The **GYRE lane** raised on `een_e3f_scheme == "nemo_avg4"` for EVERY
  tripolar fold. Its certified use of that scheme is a closed beta-plane box,
  the fold row was not built there, and main had just added a second fold
  branch that would have bypassed a refusal placed lower down.
- The **ORCA2 lane** had transcribed that fold row from NEMO and runs it:
  `dyn_vor_init` evaluates the masked four-cell average over the owned domain
  (`dynvor.f90:913-919`), completes the field with the ordinary F-point
  north-fold exchange, sign +1 (`:935`), and only then substitutes the
  reference thickness for any remaining zero (`:937`); under a T pivot that
  exchange rewrites the last owned row from the row immediately below it at
  the mirrored longitude (`lbcnfd.f90:722-746`).
- **main** added a branch for a stored-pivot mesh.

The first resolution kept the refusal and narrowed it to a stored-pivot mesh.
MEASUREMENT REFUTED THAT: the ORCA2 mesh classifies as stored-pivot under the
exact classifier main brought with it, so the refusal fired on the one card
the transcription was written for and the ORCA2 ladder stopped at the first
step. The classifier changes nothing else about that mesh — every field the
fold descriptor carried before the merge is identical afterwards, and the
F-point permutation it adds is exactly the reversed index the transcription
already uses. So the refusal is RETIRED for that scheme (`6167d74eb`), on the
ground that the transcription reads NEMO's F origin directly rather than the
storage convention detected for generic scalar exchange, which a committed
test now pins by running it with the layout flag both ways and requiring the
same bits. main's pivot branch keeps its precedence for the other two schemes,
which is what the GYRE lane's own merged tree does.

GYRE, the lock-exchange and the overflow cards never reach any of this: all
three resolve with no active fold, checked directly rather than argued.

## 4. Invariant (i) — ORCA2's ten-step ladder: **REFUTED**

The ladder runs and still reports `LADDER_MEASURED`, labelled
`MEASURED_INDEPENDENT_WITH_DECISION52_SSH`, from a clean committed detached
worktree. But it is not round 20's ladder.

| checkpoint | round 20 | merged | reading |
|---|---|---|---|
| kt 1 entry, all five fields | bit-identical | bit-identical | the Decision-52 entry bridge is untouched |
| kt 1 stage 1, all five fields | unchanged | unchanged | the first non-bit statement is the SAME: stage-1 `T`, 233,341 cells, `trasbc.f90:314-328` still ruled out |
| kt 1 stage 2 `u` max | `0.064633488618316082` | `0.064633499882926077` | first moved row, +1.1e-08 (1.7e-07 relative) |
| kt 1 stage 2 `v` max | `0.034014784775202977` | `0.034014715778048182` | -6.9e-08 |
| kt 10 entry `T` max | `3.9430791763114783` | `3.947126188631776` | +0.10 per cent |

185 of the 200 ladder rows moved. The unequal-cell counts are almost all
unchanged; the first two that move are kt 2 stage 1 `u` and `v`, by 2 and 26
cells.

### Attribution — one contributor proven, one named

The movement is NOT caused by any conflict resolution: it appears with the
fold branch unreachable for the schemes it touches, and the first mover is
stage-2 momentum.

**Contributor A, CONFIRMED by a one-variable control.** GitHub main's exact
tripolar storage-layout classifier (`92dedb6ab8`, PR #1749, 2026-09-18)
reclassifies the ORCA2 mesh as stored-pivot, which switches the north-fold
ghost-row source, the meridional derivative's beyond-fold partner and the
density-Jacobian pressure gradient's partner column from the stored top row to
the row below it. The committed control
`nemo_testcase_l4_orca2_merge_gyre_fold_layout_control.py` puts the descriptor
back into the lane parent's shape and changes nothing else. It moves the
ladder: kt 1 stage 2 `u` max goes to `0.064633491838392`, between round 20 and
the merged arm. So the reclassification owns part of the movement — and only
part, because the control does not restore round 20.

**Contributor B, PLAUSIBLE, read off the code and corroborated by one
measurement, not yet substituted.** `ddb70da1a4` ("Derive live momentum
diffusion geometry from card state", GYRE lane, 2026-09-22) rebuilt the shared
literal vorticity thickness producer from each card's own reference thickness,
masks, areas and sea-surface height instead of from the bridge-carried NEMO
record. The ORCA2 card DOES carry that record, and its recorded zero-vertex
reference thickness differs from the derived one on all 799,200 cells
(max 651 m). The producer consumes the derived value only where the masked
four-cell average is exactly zero, plus in the F-depth sum, which is why the
ladder moves at 1e-08 rather than at the raw field's scale. This lands on
momentum at stage 2, which is where the first movement appears, and leaves the
stage-2 tracer untouched, which is what is measured. It is NOT proven: no
substitution arm was run.

Both commits arrive through the merge rather than from it. Neither was
measured on ORCA2. This is carried to DECISION_NEEDED.

## 5. Invariant (ii) — GYRE's certified trajectory: **CONFIRMED, to the byte**

Run from a clean committed detached worktree at the merge commit, with no
`LEGOESM_GATE_ALLOW_DIRTY` escape; the stamp records `clean: true`.

| comparison against the GYRE lane's own post-merge arm | result |
|---|---|
| ten-step ladder document, every content key | **0 differing leaves**; only the worktree stamp differs |
| `ladder.residuals.npz` sha256 | `43f37831256832c3` on both |
| the eight certified year rows, every field | **0 differing row fields** |
| 360 daily snapshots | **360 of 360 byte-identical** |
| day 30 / 240 / 360 snapshot sha256 | `a66143733bcc9e4e` / `0d4f16d0c51da705` / `c6b7e1523b9a6dbb`, equal on both |

The three scored numbers, reproducing the preregistered targets to every
printed digit: day 30 `6.572574374770603e-05` K, day 240
`1.644836070117868e-02` K, day 360 `1.1225660018551306e-02` K.

**The "GYRE unchanged" gate on this lane is RE-BASED here, and this says so.**
Every ORCA2 receipt up to round 20 stated GYRE's identity as day-30 snapshot
digest `14a7e64b4512860e`. That is the PRE-round-163 value. Round 163 landed
GYRE's second per-stage continuity solve and this merge brings it across, so
from this round on the ORCA2 lane's GYRE gate is the GYRE lane's own current
numbers, listed above. The older digest is superseded, not contradicted.

The second continuity solve stays OFF for ORCA2: the card sets
`nemo_stage_momentum_wzv_split` explicitly `False` and GYRE's explicitly
`True`, the execution predicate returns False for ORCA2 and True for GYRE, and
a card leaving the field unset raises rather than defaulting. Decision 58
turns it on later, in its own measured round, not here.

Not one leaf of either card's resolved model configuration changed. Seven
leaves are NEW on each card: the explicit continuity-solve field, and three
lateral-mixing fields main added whose defaults reproduce existing behaviour.
No default moved.

## 6. Citation gate

The union is 1010 map entries with no key collision between the two lanes'
new blocks. Fifty-eight entries then failed the audit against the merged line
numbers and were re-anchored mechanically, by resolving each pinned anchor in
the merged file rather than by shifting by hand:

- **55 rigid shifts** — the same delta on every endpoint, extent unchanged.
  Deltas run +4 to +544 in the step module, +7 and +36 in the stage module,
  +7 and +13 in the recipe, +8 in the vertical module.
- **3 that are NOT rigid, each recorded with its reason.**
  `ocean_model_latlon_cgrid.py:7540-7972` to `:7988-8421`, extent 433 to 434,
  because main inserted one line inside the cited span — the same exception
  the GYRE lane recorded, reproduced here.
  `nemo_testcase_l2_gyre_round54_tke_operands.py:225-239`, no line moved but
  extent 12 to 15, because the GYRE lane widened that gate file and the ORCA2
  lane's copy of the extent had not followed.
  `nemo_testcase_recipe.py:175,405,1567` to `:359,604,2202`, three anchors
  with three different deltas because two insertions sit between them; extent
  unchanged.

Read against the GYRE parent rather than the ORCA2 one, a third extent also
changes — `ocean_model_latlon_cgrid.py` 70 to 127 lines — because the ORCA2
lane had widened that span for its river-runoff deposit and the merged tree
carries that insertion. The merge commit message says "two extents"; counted
from the GYRE parent it is three. Named here rather than left to the commit.

Thirty-six citations in five documents' prose were rewritten with the same
map, so no receipt sentence points at a line the merge moved.

On the clean merged tree the gate reports `status PASS`, **274 citations, 0
failures, 0 map entries failing audit, 0 unmapped**. Planted, it fails as it
must; the fired line is

    {'citation': 'ocean_model_latlon_cgrid.py:6806', 'status':
     'SYMBOL-NOT-AT-LINE', 'endpoint': 'first', 'line': 6808,
     'detail': 'that symbol identifies line 6806'}

and all nine self-test plants fired (seven defects flagged, two unplanted
baselines passing).

## 7. Tests

Every summary line below is pytest's own last line, one battery at a time.

**ORCA2 push gate** (the five files in `autopilot_orca2/autopilot_max.sh`):

> `127 passed in 378.96s (0:06:18)`

**GYRE push gate** (the six files in `autopilot/autopilot_max.sh`):

> `135 passed in 952.19s (0:15:52)`

the same count the GYRE lane reported at its own merge.

**Card gates** — DINO, the L1 tanks Rule-12 gate, the lock-exchange
slow-forcing owner, the overflow barotropic gate:

> `160 passed, 9 warnings in 338.99s (0:05:38)`

plus the round-34 tank zdf-removal gate, run separately:

> `10 passed in 7.54s`

170 in total, the same count round 163 and the GYRE lane's merge reported, so
DINO, both tanks, the lock-exchange and the overflow cards are unchanged
(M-P6 CONFIRMED).

**Focused tests at the final tip** — the two new fold-routing tests and the
citation-gate module:

> `25 passed in 6.07s`

**Broad ocean battery** — `tests/ocean/fidelity tests/ocean/unit
tests/unit/test_run_omip_cli.py`, `-n 12`. THIS BATTERY IS NOT REPRESENTED AS
GREEN and it is not used as evidence here. Three attempts were made on this
host, each under twelve parallel workers, and each was destroyed by worker
death rather than by the tree: the first ended in an xdist INTERNALERROR, the
second replaced a crashed worker twice and reported `238 failed, 8626 passed,
179 skipped, 2 xfailed, 13 errors in 3282.63s`, and the classification re-run
of its 251 flagged identifiers died inside xdist with a `MemoryError` while
other agents' batteries shared the host. Two facts bound the reading: the
flagged set spans fifty-four files and behaves like the documented
worker-pressure artefact on this machine rather than like a merge defect, and
every gate that IS quoted above passed at the same tip with the same counts
both lanes report. All three logs are in the evidence directory; classifying their residue
against both lanes' known-red lists is carried to OPEN rather than claimed
here.

**Non-vacuity of the two new tests.** With the retired refusal put back by
hand, both fail and nothing else does — `2 failed, 7 passed in 3.83s` — and
the revert was verified with `git status --porcelain`, which is empty. So the
tests bind on the resolution this merge took, not on their own fixtures.

## 8. Independent reviews

Two, both on the merge commit and its follow-up.

**Claude code-reviewer subagent — SHIP.** It checked every construction site
of the private hook tuple by AST (194 sites, none positional), enumerated all
six combinations of scheme and layout in the fold branch and named which
parent each reproduces, and diffed the citation map's key sets against both
parents (730 common keys, no value conflict, no dropped anchor, and every
range key's extent equal to its own span). Two suggestions: the merge commit
undercounts the changed extents, and the map carries seven duplicate literal
keys inherited identically from both parents.

**codex `exec --sandbox read-only` — SHIP WITH CHANGES.** Per file: the
round-8 receipt clean, the step module a clean union, the stage module correct
after the follow-up, the citation map an exact parent union by citation
identity. Three findings: a materially wrong source citation in the
Decision-36 receipt, the seven duplicate literal keys, and no committed direct
regression for the stored-pivot case the follow-up turns on.

| finding | disposition |
|---|---|
| Decision-36 receipt cites two spans that now hold salt-flux and runoff code | **FIXED** in `65fb85435`. Stale on both parents and outside the gate's declared coverage, which is why nothing caught it. Both spans re-read and renamed. |
| no direct regression on a stored-pivot mesh selecting the four-cell scheme | **FIXED** in `65fb85435`: two tests, one pinning that each scheme takes its own branch on that mesh with the two index maps asserted different, one requiring the transcription to return the same bits with the layout flag either way. |
| seven duplicate literal keys in the citation map | **REGISTERED, not fixed.** Present identically in both parents; Python keeps the last, so the map is smaller than it reads. Not introduced here and not a merge defect; carried to OPEN. |
| the merge commit undercounts the changed extents | **ANSWERED in section 6** rather than in the commit, which cannot be edited. |

Both reviewers independently reached the same reading of the fold resolution,
and neither found a dropped hunk, a duplicated definition or a moved default.

## 9. Choices

ASKED: perform the merge the operator ordered; resolve every conflict from
NEMO's compiled source; keep both lanes' NEMO identities; keep ORCA2's second
continuity solve off.

UNASKED: none. Decisions 54, 57 and 58 remain pending and untouched. No
configuration value, carried-state field, stabiliser or sea-ice selector was
changed, and no production statement was landed.

## OPEN

1. **ORCA2's ten-step ladder has moved and is NOT re-certified here.** One
   contributor is proven by control, one is named but not substituted. The
   next ORCA2 round should run the substitution arm for the second before the
   new ladder is adopted as the lane's reference.
2. The seven duplicate literal keys in the citation map, inherited from both
   parents.
3. A pivot-layout mesh selecting the three-cell average is still unmeasured
   against NEMO; the GYRE lane registered this and it stays open.
4. Everything round 20 left open stands: the ranked slow-forcing acquisition,
   the northern-fold mask and wind-stress operands, and Decisions 54, 57, 58.
