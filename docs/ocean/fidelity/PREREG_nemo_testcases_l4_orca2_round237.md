# ORCA2 round 237 preregistration — required barotropic-filter alpha

Date: 2026-10-10. Frozen base: `792f4598ccab34ad7a2d784cd18be7d48009b51a`.
Scope: Decision 115 only. Decision 114's raw-face geometry unit is deferred to
round 238. No deck, carried state, sea-ice selector, stabiliser, or
`unmeasured_features` tuple changes in this round.

## Frozen source statement and card registry

The compiled OMT-4 source calls `ts_bck_interp` before the surface-pressure
gradient at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:604-612`.
For substep three onward the executing routine derives the four SSH-history
weights from `rn_bt_alpha` at `:1522-1557`. The admitted OMT-4 deck states
`nn_bt_flt=3` and `rn_bt_alpha=0.09` at
`orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg:386-387`.
Round 235 measured that NEMO's recorded operands replay with 8,794/13,320
unequal cells at the shared-library value 0.07 and 0/13,320 at 0.09.

Decision 115 freezes the card registry; values are read from the named decks,
not selected in this round:

| card family | resolved filter | required `rn_bt_alpha` | effect |
|---|---:|---:|---|
| ORCA2, OMT-0..4, hierarchy rungs 0/10 | 3 | 0.09 | live |
| GYRE, VORTEX, LOCK_EXCHANGE | 3 | 0.07 | live; current effective value |
| OVERFLOW | 1 | 0.0 | stated but inert in filter 1 |
| DINO NEMO card | 2 | 0.0 | stated but inert in filter 2 |

The field has no physical library default: the general configuration sentinel
is `None`; every NEMO testcase card states a finite deck value and the card
validator refuses the sentinel. The existing coefficient builder is reused and
its `alpha` argument becomes required. No second interpolation implementation
is authorised.

## Frozen predictions and falsifiers

- **R237-P1 — registry and refusal.** CONFIRM if every NEMO testcase card
  constructs with the table value, removing any one value makes its validator
  refuse, and the instantiated production configs print those values. REFUTE
  on a silent fallback, an unstated card, or a deck/card disagreement.
- **R237-P2 — source-exact coefficient boundary.** CONFIRM if OMT-4's substep-3
  interpolation boundary moves from 8,794/13,320 unequal to 0/13,320 using
  the production card field, with the exact round-235 0.09 weight vector
  `[0.7125337174, 0.2525882038999999, 0.03722244,
  -0.0023443612999999985]`. REFUTE if one cell or one weight differs. A
  one-ULP-alpha plant must refuse.
- **R237-P3 — preserved cards.** CONFIRM if GYRE's certified ladder/year,
  DINO's month gate, LOCK_EXCHANGE, OVERFLOW and VORTEX/tank gates are
  byte-identical to their frozen baselines: they now state the same value the
  old implementation effectively used, or an inert deck value. REFUTE and
  retain any moved row under its label.
- **R237-P4 — ORCA2 landing predicate.** Prediction: OMT-1..4, independent
  rung 0, and given-NEMO-entry rung 10 execute the changed statement. No exact
  certified row leaves the bar; the first-over-bar row moves toward NEMO or is
  unchanged; a majority of RMS-moved rows move toward NEMO. REFUTE by naming
  the first failing row and HOLD the landing.
- **R237-P5 — month.** Run OMT-4's independent 96-step protocol and rung-0's
  independent month protocol before and after. CONFIRM if neither acquires an
  earlier refusal/non-finite boundary; retain per-field RMS/max or the exact
  first refusal step and cell under each label. REFUTE by an earlier boundary.
- **R237-P6 — implementation integrity.** The direct test must fail when the
  production field is replaced by 0.07 or removed. Citation gates must have no
  unmapped anchors and the rigid-shift plant must fire. The shared
  implementation remains JIT/autodiff-pure and introduces no stabiliser.

## Landing rule

`LANDED` requires R237-P1/P2/P3/P4/P6 and no earlier month boundary in P5,
plus the focused tests, the once-only `tests/ocean/fidelity -n 12` battery, the
citation gate, and an independent read-only diff review. Otherwise `HELD` with
the exact red row. No acquisition is expected: all scoring records already
exist.

ASKED choices: Decision 115 and standing Decision 96. Pending and untouched:
Decision 114. UNASKED choices: empty.
