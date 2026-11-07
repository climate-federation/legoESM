# Preregistration — ORCA2 round 78, bisecting the GYRE fold-in

Written and committed before any ladder of this round has been run.  Nothing
below is measured yet; the only numbers quoted are round 77's, taken from its
committed records.

## 0. The metric, and one correction to the order

The order names "the given-entry ladder's step-10 end-of-step T rms and S rms —
the rows round 77 saw worsen 13%/15%".  Those are two different rows.  Read out
of round 77's committed ladder JSONs
(`phase3/orca2_rounds/round77_foldin/ladder_{before,after}_decision52bridge.json`),
at kt 10, checkpoint `stage3` (the end of the step), the "given NEMO's entry"
ladder:

| row | pre-merge `1142d4182` | merged tip | move |
|---|---|---|---|
| `T` **max** | `0.9838385161101275` | `1.114220520669864` | **AWAY +13.25%** |
| `S` **max** | `0.22099092586135072` | `0.25508044621046366` | **AWAY +15.43%** |
| `T` **rms** | `0.010433315642886491` | `0.010613818039912676` | AWAY +1.73% |
| `S` **rms** | `0.003029909179584292` | `0.002919141602961423` | TOWARD −3.66% |

So the 13%/15% rows are the **maxima**, not the rms.  The S rms actually moves
TOWARD NEMO.  This round therefore bisects on the row that carries the
13% move — **kt 10, stage 3, `T` max_abs** — and records `S` max, `T` rms and
`S` rms beside it at every point, so no reader has to take the choice on trust.

**Primary verdict function.** A bisect point is `bad` when its kt 10 stage-3
`T` max_abs is at or above the midpoint `1.049` of the two endpoints, `good`
when it is below.  The endpoints are 13% apart, so any threshold inside the gap
gives the same answer unless the move is gradual — which is itself a result and
is reported if the recorded values show it.

**Census beside it.** At every point, the whole 200-row census against the
pre-merge ladder (rows unchanged / toward NEMO / away from NEMO on `max_abs`),
so a point that moves many rows a little is distinguishable from the one that
moves the headline rows a lot.

## 1. The endpoints

* `good` = the pre-merge ORCA2 tree `1142d4182`, given-entry ladder.
* `bad` = this lane's tip `629d5f4c5`, given-entry ladder.

Both are RE-RUN in this round rather than quoted, because round 77's `after`
arm was measured on a different clone.

## 2. The bisect space

The GYRE lane's first-parent commits in `d3631f884..ea12107cb` that touch
`packages/` or `src/`: **43** commits (`d3631f884` is the merge base round 77
recorded).  At each point the tree is built as
`merge(1142d4182, X)`, i.e. the pre-merge ORCA2 tree with the GYRE chain
truncated at `X`.  Conflicts outside `packages/` and `src/` (the citation map
and receipts, which round 77 resolved by re-anchoring) are taken from the ORCA2
side, because they are documentation the ladder never reads; any conflict
INSIDE `packages/` or `src/` stops the bisect and is reported instead of being
resolved by a rule.  Where the ORCA2 card refuses to construct on an unset
field, the round-77 explicit-field statement (`0b5a4fa35`, which adds
`nemo_first_wzv_after_ssh="rk3_extrapolated"` to the ORCA2 arm of the recipe) is
applied on top and SAID for that point.

## 3. Frozen predictions

* **Q1 — one commit owns it.** The kt 10 stage-3 `T` max steps from
  ≈`0.984` to ≈`1.114` across a SINGLE bisect point.  REFUTED if the recorded
  values climb across several points, in which case the round reports a
  gradient and the largest single step rather than a culprit.
* **Q2 — the culprit is not either refuted candidate.** It is neither the
  mixing-length group (`0ebdb6d83`) nor the after-SSH statement (`34da4f2af`,
  `cb4076e1e`), both of which round 77 refuted with committed one-variable arms.
  REFUTED if the bisect lands on one of those.
* **Q3 — the endpoints reproduce.** The two endpoint runs reproduce round 77's
  committed given-entry numbers to the last digit.  REFUTED by any differing
  digit, which would mean the ladder is not deterministic across clones and
  every magnitude claim on this lane needs a run-to-run floor first.

## 4. Not in scope

No physics is landed on a prediction.  If the culprit statement is NEMO's for
ORCA2's own compiled program, nothing is reverted and the round reports what the
statement uncovered; a scoping change is landed only under the standing gates
(note B8, as read by note B12).
