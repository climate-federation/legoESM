# PREREG — grid ladder G9, round 3 (gridladder_r48)

Frozen before anything this round is measured.  Record: round 46's
(`gridladder_rounds/round46/oracle_g9_r46/twin_a`), no new acquisition.
Card at `9146dbae1` (round 47 landed).

## Item 0 — NEMO's end-of-stage fold (stprk3_stg.f90:795-796, branch
`.NOT. ln_shuman`, `.NOT. l_zdfsh2`)

`lbc_lnk(uu 'U' -1, vv 'V' -1, ts(T) 'T' +1, ts(S) 'T' +1, ldfull)` after
every stage.  The card already folds `vv` (top V row = image).  It does not
fold `uu`, T or S: their stored pivot row keeps whatever the stage wrote.
Landing: the pivot-row half overwrite (existing shared helper) for U, T, S
after each of the three stages, G9-selected only.

- P0a: card pivot-row self-mirror breaks (T, S, U; wet pairs, all levels,
  independent label) = 0 at every kt 1..10.  NEMO's are 0.
  Refuted by any nonzero card count.
- P0b: given-NEMO-entry rows: no status worsens.  Independent kt 1..10:
  no row status worsens; pivot-row T/S errors at kt 7..10 do not grow.
  Refuted by any status going AT-BAR -> DEBT or exact -> non-exact.
- P0c: the kt-1 v DEBT row (59 interior cells, 1.46e-15) is unchanged:
  it is interior, the fold acts only on the pivot row.  Refuted if it moves.

## Item 1 — the kt-1 v debt

Predicted first statement: the stage-1 tracer update, specifically the
stage-1 vertical transport `zFw` the card feeds to the tracer advection
(round 47: 355,104 cells off, ≤2.6e-16 normalised), carried into stage-1
T/S last place -> stage-2 density -> stage-2 HPG -> stage-2 v.

- Link test (item 2): kt 1, card stage 2 started from the card's own stage-1
  u, v, ssh but NEMO's stage-1 T and S (record `1_tn_all_aa`, `1_sn_all_aa`).
  PREDICT: stage-2 v = NEMO bit-exact on every cell (0 unequal), and the
  kt-1 step-end v row exact or AT-BAR.  REFUTED if stage-2 v keeps ≥1
  unequal cell: then the link is wrong and v's own path is walked.
- Walk (item 3): PREDICT the first differing stage-1 tracer statement is the
  card's `zFw` (not `zFu`/`zFv`, not the CEN2 RHS association, not the
  stage update).  REFUTED if the card's stage-1 `zFw` equals NEMO's
  recorded `zFw` bit-for-bit on wet cells, or if `zFu`/`zFv` already differ.
- PREDICT the mechanism is an XLA association/contraction (reciprocal
  multiply or FMA) on a NEMO stored-reciprocal form, fixed by a
  materialisation at a cited NEMO line.  PLAUSIBLE only; refuted if the
  card's eager evaluation also differs.

## Certification criterion (item 4)

G9 CERTIFIED iff every row of both labels kt 1..10 is exact or AT-BAR
(normalised ≤1e-15; D127 subnormals counted separately).  Anything else:
not certified, and the remaining first statement is named.
