# ORCA2 round 66 receipt — Decision 52 independent start

Date: 2026-09-28

Base: `7ab42435b2f56f8c2a97cedee5bb2ea3fd1737d4`

Disposition: **HELD; the independent ten-step headline is measured and no
model statement lands**

Every numerical result in this receipt is labelled **independent**: legoESM
starts from its own card state and is compared only with NEMO's own from-rest
trajectory.  No NEMO entry state is substituted.  The exact external surface
operands are shared by both trajectories, as in the admitted matched protocol.
No result below is mixed with a "given NEMO's recorded entry" twin.

Sea ice, its selectors, and the ORCA2 card's six-entry `unmeasured_features`
tuple remain frozen and out of scope.

## Answer

The independent ORCA2 ladder completes all ten steps: 40 ordered checkpoints
and 200 field rows, all finite fp64 under production JIT and scalar libm.  The
classifier confirms that the executed kt=1 state is the card's own state and
that the Decision-52 SSH bridge is absent.

The first non-bit independent statement is kt=1 entry SSH.  Independent T, S,
u, and v are bit-identical on all `799,200` cells each.  SSH differs on
`16,433 / 26,640` cells, maximum `0.015479333813968585 m`, RMS
`0.012157480547162937 m`.  NEMO's compiled owner is its five-category
snow/ice-mass adjustment at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465`.
That owner is explicitly out of scope, so the mismatch is measured and kept;
it is not transcribed, bridged, or called ocean debt.

At kt=10 the largest absolute independent row is full-domain entry T:
`430,552 / 799,200` unequal, maximum `1.2654114119557178 degC`, RMS
`0.01014923364928614 degC`.  The largest stage-3 independent row is also T:
`233,341 / 399,600` unequal, maximum `0.9841317699618912 degC`, RMS
`0.010433776400167207 degC`.  The full 20-row kt=10 table is mechanically
stored in `independent_summary.json`; stage checkpoints cover the admitted
rank-0 half while step-entry checkpoints cover both MPI ranks.

This is now the lane's headline short-horizon metric.  It is not an
attribution: the trajectory starts with the known out-of-scope SSH difference
and accumulates every remaining ocean difference.  The next experiment must
rank month-scale ORCA2 field errors by magnitude before another bit-row walk.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R66-P1 entry | **CONFIRMED** | Independent T/S/u/v are `0 / 799,200` unequal; SSH is exactly `16,433 / 26,640`, max `0.015479333813968585 m`. |
| R66-P2 no bridge | **CONFIRMED** | Executed mode is `card_own_state`; bridge payload is null; the bridge plant refuses. |
| R66-P3 first non-bit | **CONFIRMED** | First row is kt=1 entry SSH with the compiled ice-mass citation. |
| R66-P4 complete ladder | **CONFIRMED** | 40 checkpoints, 200 field rows, all metrics finite and fp64; no execution blocker. |
| R66-P5 disposition | **CONFIRMED** | Measurement only; `packages/` is byte-identical to the base. |

The first classifier attempt refused on JSON dictionary order: the ladder
serializes field keys alphabetically, while the classifier incorrectly
required insertion order.  This instrument defect is fixed and directly
tested; it changed no measurement and is not a failed scientific prediction.

## Compiled initial-state program

NEMO initializes T/S through its data reader, initializes u/v at rest, and
copies the before level at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140`.
The executing ORCA_R2 hand alterations and mask are at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:217-254` and
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dtatsd.f90:307-310`.
Round 7's transcription remains exact for those four ocean fields.

NEMO subsequently applies the snow/ice-mass SSH adjustment cited above.
Because the lane's one-category SI3 cannot represent that five-category
identity and sea ice is frozen at `STOP_SELECTOR_GAP`, independent legoESM
retains its own SSH rather than importing the adjustment.

## Controls, scope, and validation

All controls fired:

- applying the historical SSH bridge refused `card_own_state`;
- changing the SSH unequal count from `16,433` to `16,434` refused;
- deleting one checkpoint refused the 40-checkpoint inventory;
- a one-ULP mutation of an exact kt=1 T cell refused the entry identity.

No file under `packages/` changed.  ORCA2, GYRE, DINO, OVERFLOW,
LOCK_EXCHANGE, and generic-card model trajectories therefore cannot move;
their trajectory gates are not applicable to this measurement-only round.

Validation results:

- Independent ladder: `LADDER_MEASURED`; round classifier
  `PASS_INDEPENDENT_LADDER`.
- Focused ladder/classifier controls: `24 passed`, followed by the JSON-order
  regression test (`7 passed`).
- Shared card battery: `160 passed` with nine dtype warnings; separate tank
  set: `10 passed`.
- Full `tests/ocean/fidelity -n 12` battery: `2,002 passed`, `7 skipped`, and
  exactly five standing failures: SI3 `MY_SRC` provenance, round-129 stale
  phase-3 certification, round-51 private trace registry, four worktree-stamp
  emitters, and the missing `hires_lane_surface` case-board row.  No round-66
  test failed.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round66/`.
The decisive files are `independent_ladder.json`, `independent_summary.json`,
the four plant logs, both card-battery logs, and `ocean_fidelity.log`.

## Scope ledger

ASKED: Decision 52's independent-start ORCA2 ten-step ladder.

UNASKED and unchanged: configuration, selector, threshold, forcing,
stabiliser, carried state, NEMO arithmetic, sea ice, and the held shared tracer
QCO/RK candidate.

## OPEN

1. Rank month-scale **independent** ORCA2 errors by magnitude, using NEMO's
   own from-rest month as the comparator and the same matched forcing protocol.
   Do not walk more bit rows before this ranking.
2. The month-scale lane needs a NEMO from-rest record if no admitted record
   already spans the selected month; write a fail-closed acquisition rather
   than extrapolating the ten-step ladder.
3. The independent kt=1 SSH difference remains explicitly out of scope at the
   sea-ice `STOP_SELECTOR_GAP`; do not change the six selectors.
4. The source-ordered tracer QCO/RK statement remains locally exact but held
   by round 48's five OVERFLOW rows.
5. Round 20's slow-forcing/barotropic walk remains open.
