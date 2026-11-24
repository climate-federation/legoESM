# ORCA2 round 306 — drop the "linear implicit drag unmeasured" label (Decision 133, rest)

Date: 2026-10-17

Status: **LANDED-pending.** Rung 0 and rung 10 no longer carry the
`linear_implicit_bottom_drag` label. Label-only: every card's resolved model
config and every ladder row are unchanged.

Preregistration `PREREG_nemo_testcases_l4_orca2_round306.md` (commit
`4c31bda8a`, frozen before measurement). Label commit `e850559f7` (measured
clean); citation re-anchor `test(ocean)` commit after it. Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round306/`.
codex review: paused (quota). Independent review: operator (Claude).

## Why the label is false now

Round 305 (Decision 133) made rung 0 and rung 10 run NEMO's linear implicit
drag (`ORCA2_OMIP_L4/EXP00/namelist_cfg:251`, `ln_lin`, with `ln_drgimp`)
through G7's ladder-certified bottom statements, and measured both cards on
all seven Decision-122 pairs: 0 exact rows lost, first over-bar row toward
(rung 0; rung 10 given NEMO's entry) or equal (elsewhere). That measurement
replaces the label.

Caveat, kept on record: G7 certifies the drag WITH sea ice. Drag WITHOUT ice
(rung 0) has no bit-exact certification on any rung; it is measured under
Decision 122, not certified.

## What changed

- Shipped card (rung 10): seven labels -> six. Still refused for execution.
- Rung 0: one label -> none. Its execution check used to refuse with the drag
  label's name unconditionally; it now refuses on whatever labels the card
  carries, and with none it still checks every rung-0 selector (the drag
  selectors included) before passing. Rung 0 is now execution-ready.
- 14 gates/probes and 3 tests that pinned the old tuple now pin the new one
  (recipe; rung-0 card, ladder, month, step-16, step-36, growth-walk,
  early/initial-growth, final-mean, month-boundary gates; phase-2, rung-10
  ladder and independent-start registries).
- Not changed: the parallel-inventory probe pins the tuple of a fixed
  historical revision, which is still true for that revision.

## Measurements

| prediction | result |
|---|---|
| P1 resolved configs (round-305 dumper, 544 leaves; rung 10 607) | CONFIRMED: 0 leaves differ on all seven cards; only the label tuple moves (rung 0 one -> none, rung 10 seven -> six) |
| P2 seven D133 arms rerun on `e850559f7` | CONFIRMED: every ladder identical to round 305 (1360 rows; only the commit stamp differs) |
| P3 rung-0 execution check | CONFIRMED: passes on the rung-0 card; refuses a rung-0 card carrying `staged_gm_eiv` or `freshwater_budget_carry`, naming it |
| P4 shipped card execution check | CONFIRMED: refuses, naming exactly the six remaining labels |

- Non-vacuity: with the recipe and rung-0 gate restored to the base, 5 of
  the 10 rung-0 card tests fail (tree restored, verified clean).
- Gates: this receipt's citation gate PASS (1 citation, 0 failures); default citation gate +
  private-import ratchet 23 passed. Battery: landing set 150 passed + stage
  face mask 2 passed (run in two parts); edited-gate tests 84 passed; rung-0
  card, month, boundary, independent-start, D133 selection, parallel
  inventory, internal-wave forcing tests 81 passed.
- GYRE: no GYRE code or config touched (label tuple is ORCA2 card data only).

Choices: ASKED (round prompt) the label drop and the rung-0 check change.
UNASKED: none.

## OPEN

1. V1 + V2 vertical-mixing import under D122/D130 for the ORCA2 card that
   runs TKE (addenda 101, 103).
2. Unchanged: fold unit + transport mask + top-V atomic promotion (round
   303); D126 and D132 pending; top drag stays off until the card computes
   the ice-ocean drag coefficient itself.
