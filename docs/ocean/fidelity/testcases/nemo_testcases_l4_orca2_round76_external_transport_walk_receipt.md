# ORCA2 round 76 receipt — independent external U-transport walk

Date: 2026-09-29

Base: `dbfc411b94d7293020792d4bbc887f796b5ff5de`

Disposition: **HELD; substep 2's metric U transport is the first non-bit
external-mode producer statement, and the downstream geometry errors form a
two-sided compensating pair**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and exact kt=1 surface operands. NEMO arrays enter only the labelled
diagnostic replay and substitution arms.

Sea ice remains out of scope. The ORCA2 card's six-entry
`unmeasured_features` tuple, selectors, 10,800 s step, thresholds, and carried
state are unchanged.

## Answer

All four built cards execute the split-explicit external transport
accumulator. Their resolved `(solver, substeps, filter)` values are ORCA2
`(explicit_substep, 65, nemo_ab3am4)`, GYRE `(explicit_substep, 50,
nemo_ab3am4)`, OVERFLOW `(explicit_substep, 3, nemo_boxcar1_ab3)`, and LOCK
`(explicit_substep, 1, nemo_ab3am4)`.

The compiled ORCA2 source forms the mid-step U transport as
`zhU = e2u * ua_e * zhup2_e` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:530-536`,
accumulates `wgtbtp2 * zhU * r1_e2u` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:563-580`,
then normalizes and exchanges the endpoint at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/dynspg_ts.f90:829-847`.

The zero accumulator seed, all 65 weights, substep-1 metric transport, and
substep-1 accumulator exit are raw-bit exact. At substep 2, `zhU` is the first
non-bit U producer statement: 8,568 / 8,568 active rank-0 columns differ,
maximum `23.712296310346574 m3 s-1`, RMS
`2.6045999142426335 m3 s-1`. Both statement operands are already non-bit:
`ua_e` differs on 8,568 / 8,568, maximum
`1.3951550259711822e-06 m s-1`, RMS
`3.78956315788974e-08 m s-1`; `zhup2_e` differs on 8,568 / 8,568, maximum
`0.015548358737760282 m`, RMS `0.015479122805740327 m`.

The trace is calibrated: exposing all substeps leaves the ordinary production
transport endpoint exact on 8,568 / 8,568 cells. Replaying all 65 metric
transport and accumulator-add statements through the shared production
helpers on NEMO's recorded operands closes every row raw-bit exactly; the
pre-exchange normalized endpoint is also exact.

The downstream geometry relation is a real compensating pair. With NEMO
`un_adv` fixed, live inverse depth plus live thickness leaves `zFu` RMS
`1.971885366122676e-11 m3 s-1` (129,395 / 226,236 cells unequal, maximum five
ULP). Replacing only inverse depth worsens RMS to
`0.790927819646436 m3 s-1`; replacing only thickness worsens it to
`0.7908147706107803 m3 s-1`. Replacing both closes `zub`, corrected velocity,
and `zFu` at 0 unequal cells. Neither geometry operand is eligible for an
isolated landing.

## Frozen prediction ledger

| prediction | verdict | deciding evidence |
|---|---|---|
| record replay closes all 65 rows | **CONFIRMED** | every metric and accumulator exit is raw-bit exact |
| exposed endpoint equals ordinary production | **CONFIRMED** | 0 / 8,568 unequal |
| zero seed and all weights are exact | **CONFIRMED** | seed and 65 weight rows are AT-BAR |
| first U statement is substep-2 `zhU` | **CONFIRMED** | substep 1 closes; substep 2 differs on 8,568 / 8,568 |
| recorded operands close the shared statements | **CONFIRMED** | 65 / 65 metric and exit rows exact |
| round-75 geometry rows reproduce | **CONFIRMED** | both RMS values reproduce exactly |
| geometry consistency is two-sided | **CONFIRMED** | either single substitution worsens; the pair is exact |
| disposition is HELD | **CONFIRMED** | no model or configuration statement changed |

The first measurement attempt ended without a gate status and an empty log,
matching the known per-process compiler-limit failure. The first completed
attempt then refused its own endpoint calibration because it compared the
substep result to the round-74 `transport_average` exposure rather than the
ordinary step's established `barotropic_targets[2]` endpoint. That instrument
wiring is retracted and corrected; the failed log remains in the evidence.
No scientific prediction was scored by either failed attempt.

## Controls and validation

The five plants all fire: false card scope, one-row accumulator debt, moved
first boundary, aliased geometry sources, and changed round-75 baseline. The
focused round-74 through round-76 tests pass. The citation gate, shared-card
and tank batteries, wide ocean-fidelity battery, and independent review are
recorded in the final validation ledger below.

No `packages/` file changed. ORCA2, GYRE, DINO, OVERFLOW, and LOCK trajectories
cannot move, so trajectory landing gates do not apply to this diagnostic
round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round76/`.

## Scope ledger

ASKED: walk independent `un_adv` through its compiled external-mode producer
and measure the live inverse-depth/live-thickness consistency at `zFu`.

UNASKED and unchanged: model arithmetic, configuration, selectors, thresholds,
stabilisers, carried state, sea ice, and the held tracer QCO/RK statement.

No unasked choice was made.

## OPEN

1. At substep 2's first non-bit `zhU`, measure the one-variable `ua_e` and
   `zhup2_e` substitutions and their pair before walking either operand.
2. If `ua_e` owns the non-cancelling residual, walk its substep-1 compiled
   pressure-gradient/Coriolis/drag/slow-forcing update in source order. If
   `zhup2_e` owns it, stop at the already frozen independent SSH selector gap;
   do not change the six sea-ice selectors.
3. The live inverse-depth/live-thickness errors remain a registered
   compensating pair; neither lands alone.
4. The held tracer QCO/RK statement remains downstream and unchanged.
