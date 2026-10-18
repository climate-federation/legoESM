# PREREG — grid ladder G9, round 4 (gridladder_r49)

Frozen before anything in item 2 is measured.  Record: round 46's
(`gridladder_rounds/round46/oracle_g9_r46/twin_a`), no new acquisition.
Card at `2ae9b6774` (round 48 landed) + the item-1 test commit.

## The debt

kt-1 step-end T 60,401 / S 1,414 cells off (≤4.7e-16 normalised, AT-BAR);
stage-1 and stage-2 T/S exact.  Round 48: substituting NEMO's stage-3 zF
triplet leaves 60,162 / 1,342, so the transports do not own it.

NEMO's stage 3 (record build `stprk3_stg.f90:603-604, 637, 645, 717, 734,
753`; `trazdf.f90:284-298`): Krhs = 0; tra_adv (FCT, nn_fct_h = nn_fct_v = 2)
adds the advective rate; tra_sbc; tra_qsr; tra_ldf (OFF); then tra_zdf_imp
forms `zrhs = e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` and solves.

The card at stage 3 does NOT form Krhs: it builds the content
`e3t(Kbb)*T - dt*fd2` (flux divergence content) and adds
`(dt*e3t(Kmm))*sources` (its own `ponytail:` note: "exact only when
advection is OFF").

## Predictions

- P1 (first statement): the first differing stage-3 statement is the
  `trazdf.f90:288-289` content `zrhs` — the card's advective term
  `dt*fd2` is not NEMO's `(p2dt*e3t(Kmm))*Krhs` with Krhs accumulated
  adv → sbc → qsr.  Test: the card's pre-implicit content vs NEMO's literal
  zrhs evaluated offline from the record (Kbb, after-qsr Krhs, r3t(Kbb),
  r3t(Kmm), e3t_0).  REFUTED if they are bit-equal on every wet cell.
- P2 (sufficiency): NEMO's literal zrhs substituted as the card's
  pre-implicit content makes kt-1 step-end T and S exact on every wet cell
  (60,401 / 1,414 → 0 / 0).  REFUTED by any remaining cell: then the
  implicit solve (`trazdf.f90:269-298`) or the salinity guard
  (`trazdf.f90:106`) owns the rest and is walked next.
- P3 (Krhs itself): the card's FCT final stage-3 advective rate (the shared
  FCT's NEMO-literal trace) equals NEMO's recorded after-advection Krhs on
  every wet cell given NEMO's triplet.  REFUTED by ≥1 unequal cell: then
  the FCT stage-3 form is first, ahead of P1.
- P4 (mechanism): an association difference in the card's own algebra
  (content built from a flux divergence instead of NEMO's rate), NOT an XLA
  reciprocal-multiply/FMA contraction.  REFUTED if the card's jitted
  evaluation of NEMO's literal statement on NEMO's operands differs from
  the numpy evaluation of the same statement.

## Certification criterion

G9 CERTIFIED iff every row of both labels kt 1..10 is exact or AT-BAR
(normalised ≤1e-15; D127 subnormals counted separately).  Otherwise: not
certified, and the remaining first statement is named.
