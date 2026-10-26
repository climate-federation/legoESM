# ORCA2 round 77 receipt — substep-2 external U-transport pair

Date: 2026-09-29

Base: `a84ac5ec2a416aefdec450d746c83b2a32aa8888`

Disposition: **HELD; `ua_e` is the dominant non-cancelling operand of the
first non-bit external U-transport statement**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. NEMO arrays enter only the three
labelled diagnostic substitution arms.

Sea ice remains out of scope. The ORCA2 card's six-entry
`unmeasured_features` tuple, selectors, 10,800 s step, thresholds, and carried
state are unchanged.

## Answer

All four built cards execute the split-explicit external transport statement.
Their resolved `(solver, substeps, filter)` values remain ORCA2
`(explicit_substep, 65, nemo_ab3am4)`, GYRE `(explicit_substep, 50,
nemo_ab3am4)`, OVERFLOW `(explicit_substep, 3, nemo_boxcar1_ab3)`, and LOCK
`(explicit_substep, 1, nemo_ab3am4)`.

The compiled ORCA2 source selects the predictor coefficients at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:460-469`,
forms `ua_e` from the three velocity history levels at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:476-485`,
forms the midpoint sea surface and `zhup2_e` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:487-522`,
then evaluates `zhU = e2u * ua_e * zhup2_e` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:530-536`.

On NEMO's recorded rank-0 2-D U activity mask, the four frozen arms give:

| arm | `zhU` unequal / 8,568 | maximum (m3 s-1) | RMS (m3 s-1) |
|---|---:|---:|---:|
| independent live/live baseline | 8,568 | 23.712296310346574 | 2.6045999142426335 |
| NEMO `ua_e` only | 8,568 | 10.969580354169011 | 0.4443888899978583 |
| NEMO `zhup2_e` only | 8,568 | 17.93181318767506 | 2.5244558385066944 |
| both NEMO operands | 0 | 0.0 | 0.0 |

Replacing only `ua_e` reduces RMS by a factor of 5.86 and leaves 17.6% of the
baseline RMS. Replacing only `zhup2_e` leaves 96.9% of the baseline RMS. The
recorded pair closes every active column raw-bit exactly, validating the
statement transcription, record alignment, and source ordering.

The frozen prediction is **CONFIRMED**: `ua_e` is the dominant non-cancelling
operand at this boundary. Neither single arm is bit-exact, so no isolated
production statement is eligible to land. The next walk is the compiled
substep-1 velocity update that produces substep-2 `ua_e`; the independent SSH
selector gap and the `zhup2_e` path remain frozen.

## Frozen prediction ledger

| prediction | verdict | deciding evidence |
|---|---|---|
| round-76 baseline reproduces | **CONFIRMED** | count, maximum, RMS, boundary, and substep reproduce exactly |
| four arms differ only in declared operand source | **CONFIRMED** | source ledgers and fixed metric/mask are gate predicates |
| recorded operand pair closes `zhU` | **CONFIRMED** | 0 / 8,568 unequal |
| `ua_e` is the dominant owner | **CONFIRMED** | `0.4443888899978583 < 2.5244558385066944 < 2.6045999142426335 m3 s-1` |
| depth-only remains non-bit | **CONFIRMED** | 8,568 / 8,568 unequal |
| disposition is HELD | **CONFIRMED** | no model or configuration statement changed |

No prediction was rewritten after measurement. A citation-only correction to
the preregistration separates the coefficient, velocity, and depth source
ranges; it changes no prediction or measured value.

## Instrument and controls

The round-77 gate reuses round 76's production substep trace, admitted
self-describing record parser, exact activity mask, card census, raw-bit
scorer, and the shared production metric-transport helper. No new numerical
kernel or record parser was written. The metric and mask remain fixed across
all four arms.

All five plants fire:

- false card selector: `resolved external-transport scope changed`;
- changed round-76 RMS: `round-76 baseline rms_abs changed`;
- one-cell pair debt: `recorded ua_e/zhup2_e pair did not close zhU`;
- aliased operand ledger: `oracle_velocity_only did not use its declared operands`;
- swapped single-arm ordering: `declared dominant owner disagrees with measured arms`.

The record is
`round5/acquisition/orca1ice_surface_entry_every_step_a_np2/oracle_bt_advmean_operands_kt00000001.bin`,
SHA-256 `b9fb8ed3a2f8edd171b300650d21cf737bc839a0a3c7de1f66ecc94fe274e388`.
The activity-mask record SHA-256 is
`a3b009d15904df0052cdc435ce3c19e4e6700cfe26a57645449a35db1a00fda3`.
Execution is CPU, production JIT, fp64, scalar-libm.

## Validation

- pair gate: `PASS_EXTERNAL_TRANSPORT_PAIR`;
- all five plants: `PLANT-FIRED`;
- citation gate: this receipt PASS at 4 / 4 mapped citations; default receipt
  PASS at 274 / 274, with zero failures, unmapped citations, or map-audit
  failures; the shifted metric-transport citation plant FAILS as required;
- focused round-75 through round-77 and citation battery: 37 passed;
- the one permitted `tests/ocean/fidelity -n 12` battery collected 2,086
  tests, reached 99%, and passed all seven round-77 tests before the standing
  no-summary wrapper stall after all pytest workers exited. Its five observed
  failures reproduce in one serial isolation battery: the round-129 certified
  phase-3 stepping stamp moved, the round-51 private trace tuple accumulated
  newer fields, SI3 `MY_SRC` A is not verbatim, `hires_lane_surface` is absent
  from the case board, and four older report emitters lack worktree stamps;
- required separate read-only Codex review: **independent review unavailable
  in-sandbox**; the app-server client could not initialize on the read-only
  filesystem. Exact output is retained in `codex_review.log`.

No `packages/` file changed. ORCA2, GYRE, DINO, OVERFLOW, and LOCK trajectories
cannot move, so shared trajectory landing gates do not apply to this diagnostic
round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round77/`.

## Scope ledger

ASKED: measure the one-variable `ua_e` and `zhup2_e` substitutions and their
pair at independent ORCA2's first non-bit substep-2 `zhU` statement.

UNASKED and unchanged: model arithmetic, configuration, selectors, thresholds,
stabilisers, carried state, sea ice, the registered downstream geometry pair,
and the held tracer QCO/RK statement.

No unasked choice was made.

## OPEN

1. Walk substep-2 `ua_e` upstream in compiled order: predictor coefficients,
   `un_e`/`ub_e`/`ubb_e`, then the substep-1 pressure-gradient, Coriolis,
   bottom-drag, slow-forcing, boundary-exchange, and history-rotation updates.
2. Keep `zhup2_e` frozen. Its independent SSH owner remains
   `STOP_SELECTOR_GAP`; do not change the six sea-ice selectors.
3. The live inverse-depth/live-thickness errors remain a registered downstream
   compensating pair; neither lands alone.
4. The held tracer QCO/RK statement remains downstream and unchanged.
