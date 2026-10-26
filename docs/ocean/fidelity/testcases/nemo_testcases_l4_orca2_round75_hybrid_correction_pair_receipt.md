# ORCA2 round 75 receipt — hybrid-correction cancelling pair

Date: 2026-09-29

Base: `ef8ea59eb8f2e7004329843b875a6a1706ffe1bc`

Disposition: **HELD; `un_adv` owns the next upstream walk, and the predicted
two-sided cancellation at `zub` is refuted**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. NEMO operands appear only in the three
explicit diagnostic substitution arms. The production baseline remains
independent.

Sea ice remains out of scope. The ORCA2 card's six-entry
`unmeasured_features` tuple, selectors, 10,800 s step, thresholds, and carried
state are unchanged.

## Answer

All four built cards resolve the shared RK3-WS momentum and tracer stage
program, so ORCA2, GYRE, OVERFLOW, and LOCK execute the measured statement.
The compiled ORCA2 build fixes `n_baro_upd=np_HYB` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:45-49`.
Its active statement is

`zub = un_adv * inverse_depth - uu_b(Kmm)`

followed by the metric transport at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:274-284`.

On NEMO's recorded rank-0 3-D U activity mask, the four frozen arms give:

| arm | `zub` unequal / 8,568 | `zub` RMS (m s-1) | `zFu` unequal / 226,236 | `zFu` RMS (m3 s-1) |
|---|---:|---:|---:|---:|
| independent baseline | 8,568 | 6.613384103192950e-04 | 226,236 | 1.0157934200517118e+04 |
| NEMO `un_adv` only | 8,568 | 1.1058764872516897e-06 | 129,395 | 1.971885366122676e-11 |
| NEMO inverse depth only | 8,568 | 6.617105505498293e-04 | 226,236 | 1.0158305247651699e+04 |
| both NEMO operands | 0 | 0.0 | 226,236 | 7.909278196464360e-01 |

Replacing only `un_adv` reduces `zub` RMS by a factor of about 598 and reduces
the downstream `zFu` maximum from `321212.60790659266` to
`9.313225746154785e-10 m3 s-1`. Replacing only inverse depth moves `zub` RMS
away from NEMO, from `6.613384103192950e-04` to
`6.617105505498293e-04 m s-1`. The exact pair closes both `zub` and corrected
velocity at 0 unequal cells, validating the statement transcription, but it
leaves `zFu` non-bit because the arm deliberately retains independent live
face thickness.

The predicted *two-sided* cancelling pair is therefore **REFUTED**. The
dominant non-cancelling owner at this statement is `un_adv`. The downstream
readout also exposes a separate geometric consistency: live inverse depth and
live thickness compensate almost completely when NEMO `un_adv` is supplied;
substituting NEMO inverse depth without NEMO thickness removes that
compensation. This is a measured operand relation, not authorization to change
either geometry field.

## Frozen prediction ledger

| prediction | verdict | deciding evidence |
|---|---|---|
| round-74 baseline reproduces | **CONFIRMED** | exact frozen `zub` and `zFu` count/max tuples |
| both recorded operands close `zub` and corrected velocity | **CONFIRMED** | 0 / 8,568 and 0 / 226,236 unequal |
| recorded `un_adv` alone worsens `zub` RMS | **REFUTED** | RMS falls from `6.613e-4` to `1.106e-6 m s-1` |
| recorded inverse depth alone worsens `zub` RMS | **CONFIRMED** | RMS rises to `6.617e-4 m s-1` |
| cancellation is two-sided | **REFUTED** | only the inverse-depth arm worsens; `un_adv` improves by about 598x |
| paired `zFu` remains DEBT | **CONFIRMED** | 226,236 / 226,236 unequal, RMS `0.790927819646436` |
| disposition is HELD | **CONFIRMED** | no model or configuration statement changed |

Failed predictions remain **REFUTED**. Exact pair closure, individual-arm
direction, downstream thickness debt, and execution scope are separate.

## Instrument and controls

The round-75 gate reuses round 74's production runner, admitted
self-describing record reader, raw-bit scorer, activity mask, card census, and
shared production source-rounding helpers. It first requires round 74's gate
to pass, then holds velocity, mask, metric, live face thickness, and
barotropic velocity fixed across the four arms.

All four runtime plants fire:

- false card selector: `resolved stage-transport scope changed`;
- changed baseline maximum: `baseline zub max_abs changed`;
- one-ULP-equivalent pair-closure violation: `paired operands did not close zub`;
- aliased arm operands: `oracle_un_adv did not use its declared operands`.

The record is
`round5/acquisition/orca1ice_surface_entry_every_step_a_np2/oracle_rkstage1_transport_operands_kt00000001.bin`,
SHA-256 `a3b009d15904df0052cdc435ce3c19e4e6700cfe26a57645449a35db1a00fda3`.
Execution is CPU, production JIT, fp64, scalar-libm.

## Validation

- pair gate: `PASS_HYBRID_CORRECTION_PAIR`;
- default citation gate: PASS, 274 citations, zero failures, zero unmapped,
  and zero map-audit failures; this receipt's gate: PASS, two citations, zero
  failures, zero unmapped, and zero map-audit failures; the shifted hybrid
  statement plant FIRES with `SYMBOL-NOT-AT-LINE`;
- focused round-71 through round-75 plus citation tests: `57 passed`;
- shared-card battery: `160 passed` with nine existing dtype warnings;
- tank battery: `10 passed`;
- wide `tests/ocean/fidelity -n 12`: 2,072 collected, reached 99%, and all
  seven round-75 tests passed before the standing no-summary wrapper stall.
  The five observed failures reproduce in a serial failing-ID rerun: stale
  round-129 phase-3 certification, round-51 private trace registry, SI3
  `MY_SRC` provenance, four worktree-stamp emitters, and missing
  `hires_lane_surface` case-board row;
- separate read-only Codex review: **independent review unavailable
  in-sandbox**; `codex exec --sandbox read-only` exited with
  `failed to initialize in-process app-server client: Read-only file system`.

No `packages/` file changed. GYRE, DINO, tank, and ORCA2 trajectories cannot
move, so trajectory landing gates do not apply to this diagnostic round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round75/`.

## Scope ledger

ASKED: measure the two-input substitution pair at independent ORCA2's first
non-bit hybrid correction and name the operand that owns the next walk.

UNASKED and unchanged: model arithmetic, configuration, selectors, thresholds,
stabilisers, carried state, sea ice, and the held tracer QCO/RK statement.

No unasked choice was made. No new data source or score support was selected.

## OPEN

1. Walk independent `un_adv` upstream through its compiled external-mode
   producer, preserving the live inverse-depth/live-thickness pair and using
   one recorded operand per arm.
2. Before any `un_adv` landing, measure the newly exposed consistency pair at
   the final transport: live inverse depth plus live thickness nearly closes
   `zFu`, whereas recorded inverse depth plus live thickness leaves RMS
   `0.790927819646436 m3 s-1`.
3. The independent kt=1 SSH selector gap remains `STOP_SELECTOR_GAP`; do not
   change the six frozen sea-ice selectors.
4. The held tracer QCO/RK statement remains downstream and unchanged.
