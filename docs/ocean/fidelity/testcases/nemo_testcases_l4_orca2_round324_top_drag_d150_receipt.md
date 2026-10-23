# ORCA2 round 324 — the rung-10 comparison runs NEMO's implicit ice-ocean top drag (DECISION 150)

Date: 2026-10-22 (Claude round; codex review: operator).

Status: **LANDED-pending** (operator review + push gate).

## Verdict

- The rung-10 NEMO comparison now runs the existing G5 implicit ice-ocean
  top drag, fed NEMO's recorded coefficient each step. Decision 122 passes
  on all seven pairs, both labels; 0 exact rows lost.
- Given entry: exact rows 5 -> 15. Kt-1 stage-1 T, S, u and sea level are
  now bit-exact; kt-10 max sea-level error 0.169 -> 0.0021 m.
- The shipped rung-10 card keeps the switch off; selecting it without the
  record refuses on every run path (plant shown).
- No model code changed: every card's step program and switch-off value is
  untouched.

## What changed (base `5220fcc38`)

- `9b300aa95` the ladder's `comparison_card` (rung 10 with
  `implicit_ice_ocean_top_drag` on; refuses a card that already selects
  it) used by the ladder gate's trajectory; round-324 D122 gate
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round324_top_drag_d122_gate.py`:
  control = shipped card, drag = comparison card on rung 10, shipped card
  on rung 0 / OMT-1..4; D139 classifier; kt-10 row summary); refusal test;
  preregistration; citation-map entries.
- `b64385d37` committed gradient probe
  (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round324_grad_probe.py`,
  round 323's probe plus `comparison_card`) and an arm-selection test.
- `ba0df18d2` round-314 audit reason for ln_drgice_imp; table row.
- The coefficient is the record the ladder already passed every step
  (SI3's rCdU_ice): `ICE/iceupdate.F90:391` builds it from
  `ICE/iceupdate.F90:369`'s tmod_io and the ice fraction; with no ice-shelf
  cavity `ZDF/zdfdrg.F90:126` makes it the top-drag coefficient. The
  record states ln_drgice_imp = T (its ocean.output). The model's certified
  G5 statements consume it; nothing re-implemented.

## Decision 122, seven pairs (arms at `b64385d37`, `orca2_rounds/round324/arms/`)

| pair | exact rows lost | first over-bar (D139 recomputed) | census toward/away (early, late) |
|---|---:|---|---|
| rung 0, OMT-1..4 | 0 | equal | 0/0 — drag arm is the shipped card; 0 of 960 rows differ |
| rung 10 independent | 0 | equal (kt 1 entry sea level) | 192/3 (52/3, 140/0) |
| rung 10 given entry | 0 | toward (kt 1 stage-1 T -> kt 1 stage-1 v) | 195/0 (55/0, 140/0) |

`decide7_drag.json`: `PASS_R278_D122_DECISION`.

- The three away rows (independent): kt-1 stage-1 and stage-2 and kt-2
  entry sea-level RMS, each worse by 5e-5 relative (0.0125766 ->
  0.0125773 m); the same rows' max error falls 0.0198 -> 0.0155 m. That
  label's sea level is dominated by the card's own kt-1 entry sea level
  (0.0155 m max, unchanged). Not a lean away.
- Given entry's new first non-bit row, kt-1 stage-1 v: 2,032 unequal
  points, max |difference| exactly 0, so signed zeros; the first is on the
  last row (north fold). Where the rest sit: not located.

## Rung 10, kt-10 errors (|card - NEMO|; `arms/kt10_rung10.json`)

Full domain, both ranks (kt-10 entry = after nine steps; 799,200 T points):

| label | arm | max sea level (m) | max T (K) | RMS T (K) | max u / v (m/s) |
|---|---|---:|---:|---:|---:|
| independent | round 323 | 0.30894873 | 0.26943744 | 4.33878e-3 | 0.2215 / 0.1711 |
| independent | drag | 0.02255388 | 0.19457645 | 1.92985e-3 | 0.0472 / 0.0516 |
| given entry | round 323 | 0.32475315 | 0.26947486 | 4.33901e-3 | 0.2217 / 0.1711 |
| given entry | drag | 0.00705928 | 0.19458572 | 1.93002e-3 | 0.0472 / 0.0516 |

Kt-10 stage 3 (rank 0 only, 399,600 points; NEMO wrote stage frames for
rank 0 alone):

| label | arm | max sea level (m) | max T (K) | RMS T (K) | max u / v (m/s) |
|---|---|---:|---:|---:|---:|
| independent | round 323 | 0.18349390 | 0.20180779 | 3.53189e-3 | 0.1536 / 0.1409 |
| independent | drag | 0.01751246 | 0.18333344 | 1.57797e-3 | 0.0276 / 0.0525 |
| given entry | round 323 | 0.16859636 | 0.20181106 | 3.53214e-3 | 0.1535 / 0.1409 |
| given entry | drag | 0.00208004 | 0.18333441 | 1.57795e-3 | 0.0275 / 0.0525 |

The reviewer's numbers (0.0021 m, 1.58e-3 K, 0.05 m/s; given entry, stage
3) reproduce with the committed gate. Max S (full domain) 0.2012 -> 0.0478.

## Refusal plant (`tests/ocean/fidelity/test_orca2_round324_top_drag_replay.py`)

- VORTEX with the switch on and no recorded coefficient: step (with and
  without surface forcing), step_checked, integrate (with and without
  per-step inputs) and integrate_scan (with and without) all refuse:
  9 passed (`gates/new_tests.log`).
- All three refusal statements disabled: 7 failed (`ValueError: None is
  not a valid value for jnp.array` instead of the refusal), 1 passed
  (`gates/plant_test_unguarded.log`); restored, `git status` clean.
- The shipped card keeps the switch off (round 305's selection test,
  unchanged); `comparison_card` refuses a card already selecting it.

## Flag-off identity

- `git diff 5220fcc38 HEAD -- packages src` is empty: no step program
  changes on any card (rung 0, OMT-0..4, rung 10, GYRE, VORTEX, DINO).
- Control arms equal round 323's D122 rows on all seven pairs: 0 of 1,360
  rows differ.

## Gradients (drag on, N = 1; `grad/grad_rung10_n1_drag_on.json`)

- Comparison card, top drag on, the recorded coefficient passed (probe
  stamps both); probe at clean `b64385d37`, 2,621 s, peak 70.2 GiB.
- T loss w.r.t. initial T: 0 non-finite entries (sea-level loss also 0);
  one T direction reverse vs forward 1.6e-14 (sea level 2.3e-15).
- rn_Ud: reverse -4.513638e-5, forward equal to 1.7e-15, best FD 2.8e-6
  (round 323, drag off: -4.51364e-5). Sea-level loss: exactly 0 at one
  step, as in round 323.
- Traced and production steps bit-identical (0 leaves); rebuilt-at-1e-4
  model equals the traced step (0 leaves).
- No model function touched, so every round-316 guard is unchanged.

## Predictions (`PREREG_nemo_testcases_l4_orca2_round324.md`)

- P1 completes, finite: CONFIRMED.
- P2 given entry stage 3: CONFIRMED (0.00208 m, 1.578e-3 K, 0.0525 m/s).
- P3 independent stage-3 sea level falls: CONFIRMED (0.1835 -> 0.0175 m).
- P4 full-domain kt-10 entry sea level falls: CONFIRMED (0.3089 -> 0.0226,
  0.3248 -> 0.0071 m).
- P5 D122 PASS, five non-ice pairs identical: CONFIRMED.
- P6 given-entry first over-bar equal or later: CONFIRMED (later, toward).
- P7 gradients: CONFIRMED (0 non-finite; gaps 1.6e-14 and 1.7e-15).

## Round-314 audit row

- ln_drgice_imp stays DIFFERS (b:D133) against the shipped card; reason now
  names DECISION 150's comparison. Counts unchanged (176/4 vs record,
  175/5 vs EXP00). The regenerated table also rewrote the run-length row's
  hand annotation; that row was left as committed.

## Choices made

- Drag on in the rung-10 comparison harness only — ASKED (DECISION 150,
  round prompt).
- The ladder gate's own trajectory uses the comparison card too (the
  "ladder gate" named in the prompt) — ASKED.
- Rung 0 / OMT-1..4 drag arm = shipped card (no replayed ice record) —
  ASKED (prompt: rung-10 comparison only).
- Refusal test on VORTEX, not the ORCA2 card (same guards, seconds not
  minutes) — UNASKED, instrument choice; offered for revert.

## OPEN

- Future rung-10 D122 gates must build both arms on `comparison_card`
  (round 285's shared rung-10 ladder still takes the card it is given).
- New given-entry first non-bit row: 2,032 signed zeros in kt-1 stage-1 v
  (first on the north-fold row; the rest not located).

## Gates (x64, CPU; logs `orca2_rounds/round324/gates/`)

- New and adjacent drag tests (round 324 refusal + arms, rounds 304/305
  drag imports): 29 passed (`new_tests.log`).
- Default gate (land.sh list), split for the 10-minute cap: citation-gate
  test, private-import ratchet, TKE terms, recipe 119 passed; prognostic
  barotropic 8 passed; freshwater, stage face mask, ORCA2 parallel receipt
  20 passed = 147 passed, 0 failed (`default_gate_1.log`, `_3`, `_4`; the
  first `_2` attempt hit the cap mid-file under load and was re-run split).
- Receipt citation gate on this receipt: PASS, 3 citations, 0 failures
  (`citation_gate_receipt.log`).

codex review: operator
