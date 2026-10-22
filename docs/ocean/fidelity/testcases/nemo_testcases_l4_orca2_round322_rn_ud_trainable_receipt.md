# ORCA2 round 322 — rn_Ud trainable only on explicit request; kappa_Redi frozen on rung 10

Date: 2026-10-22 (Claude round; codex review: operator).

Status: **LANDED-pending** (operator review + push gate + DINO month gate).

## Verdict

- NEMO's isoneutral diffusive velocity rn_Ud now has a trainable spec
  entry, bounds 0.0033-0.03 m/s (DECISION 149).
- No tier ever collects it; a run gets it only by naming it with a start
  value; naming it without one is refused.
- kappa_Redi is listed as frozen for rung-10 trainers (inert there since
  round 320).
- No card's forward value changes: nothing the step reads was edited.

## Bounds (cited)

NEMO mode 20 sets the laplacian prefactor to 1/2 rn_Ud and prints
aht0 = 1/2 rn_Ud rn_Ld (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:354-357`).
At ORCA2's rn_Ld = 200 km, kappa_Redi's spec range 330-3000 m2/s maps to
rn_Ud = 330 / 1e5 = 0.0033 and 3000 / 1e5 = 0.03 m/s. It contains NEMO's
reference 0.01 and ORCA2's 0.018 (`ORCA2_OMIP_L4/EXP00/namelist_cfg:306`),
which sits at 0.55 of the range.

## Provenance (commits, this branch)

- `b4014b024` spec entry (GMRediConfig, `explicit_only`), collector rule,
  spec gate admits `float | None` fields as eligible (never required),
  rung-10 frozen list, tests, preregistration.
- `70134d859` frozen list moved to the end of the recipe file, so no
  citation anchor moves (mid-file it shifted 11).
- Tests: `tests/ocean/fidelity/test_orca2_round322_rn_ud_trainable.py`.
  Evidence `orca2_rounds/round322/`.

## How it is enforced

- The spec marks the entry `explicit_only`. The collector skips it at
  every tier; with `include` it requires a start value, else raises.
- The collector refuses any spec'd field without a library default unless
  it is `explicit_only` (rn_Ud's config field is unset by default; each
  card sets it).
- The rung-10 freeze is a declared exclude list next to the card; no
  rung-10 trainer exists yet, so it binds only a trainer that passes it.
  NO GATE beyond the test that the list drops kappa_Redi and keeps rn_Ud.

## Predictions (PREREG_nemo_testcases_l4_orca2_round322.md)

- P1 no tier returns rn_Ud: CONFIRMED (core, extended, aggressive).
- P2 explicit request with 0.018 returns exactly rn_Ud (seed round-trips
  to 1e-12); without a start value it raises: CONFIRMED.
- P3, P4 rung-10 gradient: PENDING (below).

## Rung-10 gradient (N = 1, T loss)

PENDING: measured after this receipt's commit (compile ~25 min).

## Gates (x64, CPU; logs `orca2_rounds/round322/gates/`)

- Parameter gates: params-reachability audit and bounds-agreement pass.
  Of the parameter battery (reachability, bounds, spec gate, collector,
  config loader, OMIP CLI, round-320 selection) 199 passed, 5 failed; the
  same 5 fail at base `721d42140` (spec-module drift, a shortwave spec,
  a seeded-default tolerance, two atmosphere loader scans). One of them
  shrank: the shortwave spec's orphan list loses its `float | None` field.
- Round-322 tests (without the gradient): 7 passed.

## Review

- codex review: operator.

## Choices

- ASKED (DECISION 149): rn_Ud entry, bounds, explicit-only, kappa_Redi
  frozen on rung 10.
- UNASKED: rn_Ud's spec tier is 2 (as kappa_Redi); with `explicit_only`
  the tier selects nothing.
- UNASKED: the spec gate now treats `float | None` fields as eligible
  (needed to spec rn_Ud at all; never forces one into a spec).
- UNASKED: the freeze is a declared list consumed by tests, not wired into
  a trainer (none exists for rung 10).
