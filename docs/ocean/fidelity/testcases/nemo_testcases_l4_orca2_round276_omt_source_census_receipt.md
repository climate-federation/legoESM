# ORCA2 round 276 — OMT early/late source-unit census

**Status: HELD.**  Seven NEMO-cited singleton arms were measured independently
on rung 0 and OMT-1, with survivors carried through OMT-2/3 and the final
survivor through OMT-4.  NEMO's ascending-level depth sum is net-toward on
rung 0 and every OMT rung, but is majority-away on the shipped rung-10 card
(78 toward / 92 away / 10 score-equal).  Decision 96 therefore forbids its
production selection.  Every imported selector remains false on production
ORCA2/OMT cards; no physics statement lands in this round.

## Frozen protocol and compiled statements

The predictions and falsifiers are frozen in
`PREREG_nemo_testcases_l4_orca2_round276.md`, committed at lane base
`2eaeaf515d818df52b1fd88413a299c5db7b8b6f`.  Counts below are RMS directions
Amendment (2026-10-29 provenance audit): the full SHA is `2eaeaf515d2fa69ccdcaba8e9fe6e17cd8a64959`; the tail of the value above was mistyped.
over bit-touched rows, written `toward / away / score-equal`.  Early means
kt=1..3 and late means kt>=4.  Rung 0 and OMT measurements are
**independent** (legoESM's corrected from-rest entry); the shipped rung-10
landing gate is explicitly **given NEMO's entry**.  Labels are never mixed.

The imported G2 statements were not re-derived.  The G2 receipts cite the
record build that compiled the same keys and statements:

- the U and V momentum matrices use distinct live face and 1-D interface
  thickness divisors at
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:192-198` and
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:361-367`;
- the implicit tracer matrix rounds the live T thickness before its diagonal
  subtraction at
  `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/trazdf.f90:231-233`;
- vector-form momentum vertically sums the 3-D U/V forcing in Fortran level
  order at `ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:178-179`.

The first five units and their compiled ORCA2 citations remain those gated in
round 275.  The two new implementations are default-false selector additions.
The card's six sea-ice selectors and `unmeasured_features` tuple are unchanged.

## Rung-0 census

Round 275 supplied the first five singleton measurements; this round added
the two G2 statements using the same admitted control and row classifier.
Every arm lost zero exact rows.  The first non-bit row remains kt=1 stage-1 T.

| source unit | all toward / away / equal | early | late | first away | D96 |
|---|---:|---:|---:|---:|---|
| EEN ratio-fold + F-mask-pivot pair | 114 / 61 / 0 | round-275 record | round-275 record | registered in round 275 | pass |
| stored T-ratio product | 73 / 67 / 29 | round-275 record | round-275 record | registered in round 275 | pass |
| stage T-ratio literal order | 68 / 66 / 34 | round-275 record | round-275 record | registered in round 275 | pass |
| stage velocity written order | 89 / 82 / 18 | round-275 record | round-275 record | registered in round 275 | pass |
| G2 momentum thickness divisors | 132 / 47 / 0 | round-275 record | round-275 record | registered in round 275 | pass |
| G2 tracer diagonal rounding | 76 / 103 / 0 | 12 / 27 / 0 | 64 / 76 / 0 | kt=1 | **fail** |
| G2 ascending RHS sum | 91 / 86 / 18 | 20 / 20 / 15 | 71 / 66 / 3 | kt=1 | pass; first debt toward |

The tracer-diagonal prediction is **REFUTED** at the first production rung:
its early and total rows are majority-away.  It remains off.  The ascending
sum is the only new rung-0 qualifier.

## OMT-1 singleton census

OMT-1's NEMO record ends at kt=8, so its complete row set is 160 rows; all
other ten-step cards have 200.  Every arm below has zero exact-row losses and
the first-over-bar row is unchanged unless noted.

| source unit | all toward / away / equal | early | late | first away | D96 |
|---|---:|---:|---:|---:|---|
| EEN ratio-fold + F-mask-pivot pair | 57 / 78 / 0 | 4 / 31 / 0 | 53 / 47 / 0 | kt=2 | fail |
| stored T-ratio product | 30 / 37 / 68 | 0 / 2 / 36 | 30 / 35 / 32 | kt=2 | fail |
| stage T-ratio literal order | 12 / 37 / 64 | 0 / 2 / 16 | 12 / 35 / 48 | kt=2 | fail |
| stage velocity written order | 64 / 27 / 48 | 10 / 3 / 26 | 54 / 24 / 22 | kt=2 | pass |
| G2 momentum thickness divisors | 61 / 78 / 0 | 17 / 22 / 0 | 44 / 56 / 0 | kt=1 | fail |
| G2 tracer diagonal rounding | 36 / 103 / 0 | 22 / 17 / 0 | 14 / 86 / 0 | kt=1 | fail |
| G2 ascending RHS sum | 80 / 68 / 5 | 37 / 11 / 5 | 43 / 57 / 0 | kt=1 | pass |

The G2 tracer row **CONFIRMS** the late-amplification discriminator: OMT-1
is early net-toward but late 14/86.  That does not rescue it, because rung 0
is already a hard Decision-96 red.  Conversely, the EEN pair, both T-ratio
units, and the momentum divisors have early away majorities and are genuine
OMT-1 program interactions, not merely late chaotic votes.

Preregistration P2 is **REFUTED**.  Stage velocity is early net-toward on
OMT-1; it first becomes majority-away only after bottom drag enters at OMT-2.

## Survivor walk through OMT-2, OMT-3, and OMT-4

| card/unit | all toward / away / equal | early | late | first away | D96 |
|---|---:|---:|---:|---:|---|
| OMT-2 stage velocity | 48 / 101 / 33 | 4 / 7 / 31 | 44 / 94 / 2 | kt=2 | fail |
| OMT-2 ascending RHS sum | 89 / 76 / 30 | 13 / 16 / 26 | 76 / 60 / 4 | kt=1 | pass overall |
| OMT-3 ascending RHS sum | 80 / 66 / 48 | 14 / 12 / 28 | 66 / 54 / 20 | kt=1 | pass |
| OMT-4 ascending RHS sum | 103 / 80 / 12 | 26 / 19 / 10 | 77 / 61 / 2 | kt=1 | pass; first debt toward |

Stage velocity's first cross-rung red is therefore the interaction with
OMT-2's linear implicit bottom drag.  It remains off.  The ascending sum is
the sole survivor.  Its OMT-2 early subset is 13/16, but Decision 96 is
defined on all RMS-moving rows and the full OMT-2 census is 89/76; the early
split is retained as evidence rather than silently promoted to a veto.

## Shipped-card veto and disposition

On the shipped rung-10 card, **given NEMO's entry**, the ascending sum is
78 / 92 / 10 overall, 12 / 19 / 9 early, and 66 / 73 / 1 late.  The first
away vote is at kt=2; the first-over-bar row is unchanged; no exact row is
lost.  The terminal SSH maximum moves
0.28720734213774596 -> 0.2872073421383425 m.  Unlike OMT-1's tracer row,
this is not a late-only classification: rung 10 is majority-away both early
and late.  The selector was removed from the production card before the
final gates.  The support code and fail-closed measurement gate remain with
both selectors false by default.

Predictions dispose as follows:

- R276-P1 **CONFIRMED**: all seven singleton arms have complete rung-0 and
  OMT-1 censuses with required controls.
- R276-P2 **REFUTED**: stage velocity passes OMT-1 and first fails OMT-2.
- R276-P3 **CONFIRMED** by the OMT-1 tracer-diagonal 22/17 early versus
  14/86 late split.
- R276-P4 **REFUTED**: the tracer diagonal is 76/103 already on rung 0;
  only the ascending sum qualifies there.
- R276-P5 **CONFIRMED as a gate**: rung 10 vetoed the only full OMT survivor,
  and the production selector was removed.

The first version of the imported ascending-sum unit test incorrectly
predicted that one isolated compiled reduction must differ.  The G2 receipt
had established a difference only in the complete four-lane stage path.
That plant was retracted before any model claim: the final control uses four
lanes, changes the summation order mechanically, and fires.  The exact-row
loss plant (kt=1 entry T) and inherited-extra-selector plant both fire.

## Gates and tests

- Final GYRE trajectory comparison is 70/70 rows byte-identical to the
  round-270 baseline; first-over-bar remains kt=3.  Residual archive SHA256
  is `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5`.
  The CPU 30-day member is byte-identical at every daily snapshot; day-30
  SHA256 remains
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- Focused import tests pass 21/21; shared generic/DINO/tank tests pass 61/61;
  the push battery passes 128/128.
- The one allowed `tests/ocean/fidelity -n 12` battery collected 3,582 tests,
  reached 99%, printed 3,540 passes and 22 failures, then timed out after a
  worker hang.  The 22 IDs were rerun serially: 13 passed.  One round-caused
  append-order assertion was repaired and passes 1/1.  The remaining eight
  reds are exactly the round-275 base inventory: SI3 provenance; stale GYRE
  round-129 stamp; two stale GYRE round-33 mutation anchors; stale OMT-1
  module assertion; round-228's stale citation span; allow-dirty scope; and
  worktree stamp.  The full parallel battery was not rerun.
- Default and round-receipt citation gates pass with no unmapped citations;
  the shifted compiled-source plant fires.
- **independent review unavailable in-sandbox**: `codex exec --sandbox
  read-only` failed before review because its app-server client could not
  initialize on the read-only filesystem.

## OPEN

Before the ascending RHS statement can be selected, locate the first kt=2
shipped-rung-10 interaction that turns its 91/86 rung-0 and 103/80 OMT-4
votes into 78/92.  Compare only the modules and carried fields added between
OMT-4 and shipped rung 10, in compiled source order, one variable at a time;
do not infer a configuration choice.  The stage-velocity/bottom-drag
interaction and the four OMT-1 source-unit reds remain registered debt.  No
configuration, stabiliser, carried-state, or sea-ice change is authorised.
