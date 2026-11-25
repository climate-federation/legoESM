# PREREG — ORCA2 round 325: L9 import, rung 10 switches on L3's eddy-induced velocity (DECISION 151)

Frozen before any measurement of this round (Claude round, 2026-10-23).

## What changes

- Import (default-off, `10f26fc87`): lane A's L9 selector
  `rk3_stage3_l9_eiv_nemo_order` (gridladder `e4f204272d`) and its guard
  hunk (tip `e3aed2ffc6`): L8's composition admits L2's block or L9's exact
  L3 selection. The eddy-velocity statements themselves are already on this
  branch (lane B L3, round 321).
- Selection: rung 10 only, `gm_redi = orca2_l3_eiv_gm_redi()` plus the L9
  selector, as ORCA2's `&namtra_eiv` sets it: ln_ldfeiv = T,
  nn_aei_ijk_t = 21, rn_Ue = 0.03, rn_Le = 200 km (EXP00 namelist_cfg
  :317-323), ln_eke_equ = F (namelist_ref :980). Rung 0 and OMT-1..4 state
  ln_ldfeiv = F and do not change.

## Arms (all from one final tree)

- control: rung 10 with L3 reverted (round 324's card); other cards shipped.
- l3: the shipped card after selection.
- control_drag / l3_drag: the same, with rung 10 through the ladder's
  `comparison_card` (DECISION 150's top drag); other cards unchanged.

## Predictions (falsifiers in brackets)

- P0 flag-off: 11 normalized step programs identical, import vs base; rung
  10, 3 steps, every leaf bit-identical [any difference].
- P1 both L3 arms complete, finite, both labels [refusal or NaN].
- P2 Decision 122 (DECISION 139 classifier), seven pairs: PASS in both drag
  arms, 0 exact rows lost, five non-rung-10 pairs 0 rows moved [any FAIL].
- P3 the first over-bar row of each rung-10 pair is equal: EIV acts at stage
  3 only [moves earlier].
- P4 kt-10 stage-3 (rank 0) RMS T falls with L3 in the drag-on arm, both
  labels [rises]. Drag-off arm: no direction predicted (its error is
  dominated by the missing top drag).
- P5 kt-10 max sea-level error moves < 10 % in each arm (the bolus is a
  vertical-streamfunction transport, zero column sum) [>= 10 %].
- P6 gradients, rung 10, N = 1, L3 on: T loss w.r.t. initial T and rn_Ud,
  0 non-finite, reverse = forward <= 1e-12 relative [NaN or larger gap];
  dL/dln(rn_Ud) within 1 % of round 324's -4.5136e-5 [outside].

## Landing rule

Both drag arms PASS -> land the selection. Both FAIL -> keep L3 off, report.
They disagree -> stop and ask (numbered options, my pick); no baseline
chosen here.
