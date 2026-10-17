# PREREG — ORCA2 round 306: drop the "linear implicit drag unmeasured" label (Decision 133, rest)

Frozen before measurement (2026-10-17).

## Change (label only)

- Rung 10 (shipped card): remove `linear_implicit_bottom_drag` from its seven
  unmeasured labels; six remain.
- Rung 0: its only label is removed (empty tuple). Its validator's execution
  check, which refused with that label's name unconditionally, refuses on
  whatever labels the card still carries, and on none passes only after every
  rung-0 selector check (the drag selectors included) has passed.
- Every gate, test and probe that asserts the old tuple is updated.
- No card selection, no other label, no model arithmetic changes.

Why the label is false now: round 305 (Decision 133) made rung 0 and rung 10
run NEMO's linear implicit drag (`ORCA2_OMIP_L4/EXP00/namelist_cfg:251`,
`ln_lin`, `ln_drgimp`) with G7's ladder-certified bottom statements, and
measured both cards under Decision 122 (seven pairs, 0 exact rows lost).
Caveat kept: G7 certifies drag WITH sea ice; drag without ice (rung 0) has no
bit-exact certification on any rung — it is measured (D122), not certified.

## Predictions

- R306-P1: every card's resolved model config (544/607 leaves, round-305
  dumper) is identical before and after; only the unmeasured tuple differs
  (rung 0: one label -> none; rung 10: seven -> six).
  Falsifier: any config leaf differs, or any other card's tuple moves.
- R306-P2: the seven round-305 D133 arms rerun on the new tree reproduce
  round 305's rows exactly (every field row equal).
  Falsifier: any row differs.
- R306-P3: the rung-0 execution check passes on the shipped rung-0 card and
  refuses a rung-0 card carrying any remaining label, naming that label.
  Falsifier: either outcome inverts.
- R306-P4: the shipped card's execution check still refuses, naming exactly
  the six remaining labels.
  Falsifier: it passes, or names a different set.

## Landing rule

Land only if P1-P4 all hold. No other edit.
