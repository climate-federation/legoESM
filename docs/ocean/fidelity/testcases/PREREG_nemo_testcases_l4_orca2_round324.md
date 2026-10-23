# PREREG — ORCA2 round 324: DECISION 150, rung-10 comparison runs NEMO's implicit ice-ocean top drag

Frozen before any measurement of this round (Claude round, 2026-10-22).

## What changes

- The rung-10 comparison harness only (the ORCA2 ladder gate's
  `comparison_card` and this round's D122 drag arm) selects the existing G5
  path `implicit_ice_ocean_top_drag`. The step reads NEMO's recorded
  rCdU_ice, which the ladder already passes every step
  (rCdU_ice = -r1_rho0*tmod_io*at_i, iceupdate.F90:391, with tmod_io from
  iceupdate.F90:369; rCdU_top = rCdU_ice, zdfdrg.F90:126; the record states
  ln_drgice_imp = T). No model code changes.
- The shipped rung-10 card keeps the switch off (DECISION 133).

## Arms

- control: the shipped card (round 323's), all seven D96 pairs.
- drag: rung 10 (both labels) = `comparison_card(shipped)`; rung 0 and
  OMT-1..4 = the shipped card (no replayed ice record there), stated.

## Predictions (falsifiers in brackets)

- P1 drag arm completes, finite, both rung-10 labels [any refusal or NaN].
- P2 rung 10 given entry, kt-10 stage 3 (rank 0): max sea-level error
  0.1686 -> about 0.0021 m [> 0.02 m]; RMS T 3.53e-3 -> about 1.58e-3 K
  [> 3.0e-3 K]; worst velocity 0.154 -> about 0.05 m/s [> 0.10 m/s].
  Source: the operator's reviewer probe (uncommitted), a hypothesis.
- P3 rung 10 independent, kt-10 stage 3: max sea-level error falls
  [rises above 0.1835 m].
- P4 full-domain kt-10 entry (both ranks), both labels: max sea-level error
  falls [rises above round 323's 0.3089 / 0.3248 m].
- P5 Decision 122 on all seven pairs with DECISION 139's recomputed
  classifier: PASS, 0 exact rows lost; rung 0 and OMT-1..4 rows identical
  between arms [FAIL, or any moved row on those five pairs].
- P6 rung 10 given entry: the first over-bar row (round 323: kt 1 stage-1 T)
  is equal or moves later under D139 [moves earlier or gets worse].
- P7 gradients, drag on, N = 1: T loss w.r.t. initial T and w.r.t. rn_Ud,
  0 non-finite, reverse = forward to <= 1e-12 relative [NaN, or gap > 1e-12].

## Landing rule

D122 PASS on all seven pairs, both labels -> the comparison harness keeps
the drag. Any fail or lean away -> keep it off, report the numbers, ask.
