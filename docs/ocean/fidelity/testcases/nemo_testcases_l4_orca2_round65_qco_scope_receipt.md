# ORCA2 round 65 receipt — QCO boundary scope

Date: 2026-09-28

Base: `a18927ae417f4606a1b1a951529388bff79d7ae0`

Disposition: **HELD; no model statement lands**

Claim labels: compiled routing is **independent**; tracer replay numbers are
**given NEMO's recorded entry**.  No independent-start number appears in the
record-backed table.  Sea ice, its six selectors, and the ORCA2 card's
`unmeasured_features` tuple remain frozen.

## Answer

Round 60's thickness-weighted momentum boundary is not an ORCA2 statement.
The instantiated card and compiled predicate agree that ORCA2 executes the
vector velocity update; only OVERFLOW and LOCK_EXCHANGE execute the alternate
thickness-weighted momentum update.  The imported reading of round 60 as an
ORCA2 boundary is **RETRACTED**.  Round 60's own OVERFLOW measurement remains
valid and independent.

The first actual ORCA2 non-bit statement at the returned boundary is the
shared tracer QCO/RK assignment.  On the admitted record, current production
differs from NEMO in `57,141 / 228,641` active T cells and
`57,169 / 228,641` active S cells.  Both the literal replay and the JAX
source-ordered replay are bit-exact (`0 / 228,641` for each tracer); the fused
control retains exactly `57,141` and `57,169` unequal cells.  Every numerical
array is float64, JIT is enabled, and the one-ULP literal-replay plant refuses.

This is the statement round 48 already proved locally correct.  It does not
land here: round 48's independent OVERFLOW gate still controls the shared
candidate and still carries five later U rows beyond its frozen 2-ULP
non-regression bar.  This scope round does not retry that candidate, infer a
cancelling pair, or weaken a threshold.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R65-P1 resolved execution | **CONFIRMED** | Four-card census has zero disagreements: ORCA2/GYRE vector-invariant; OVERFLOW/LOCK flux-form UP3. |
| R65-P2 round-60 scope | **CONFIRMED** | ORCA2 executes the vector velocity arm and cannot execute the alternate thickness-weighted velocity arm. |
| R65-P3 first actual statement | **CONFIRMED** | Production T/S `57,141/57,169` unequal; literal and source-ordered replays both zero; fused control retains both debts. |
| R65-P4 plants | **CONFIRMED** | Scope, count, compiled-predicate, and one-ULP plants all refuse. |
| R65-P5 disposition | **CONFIRMED** | First actual ORCA2 statement is tracer QCO/RK; no package or physics change lands. |

## Compiled source and result

The compiled stage-1/2 selector takes the velocity-form arm when
`ln_dynadv_vec` is true, and places the thickness-weighted momentum expression
only in the `ELSE`, at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:467-480`.
The instantiated census resolves ORCA2 and GYRE to the first arm and both tanks
to the second.  The gate's predicate mutation and card-scope mutation each
refuse, so this classification does not rest on prose.

The tracer assignment is outside that momentum selector.  It separately
materializes the before-level product, stage-RHS product, their sum, and the
after-level division at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.
The composition gate consumes the existing admitted scorer rather than
implementing another spelling of this arithmetic.

The routing result is **independent** because it follows only the instantiated
cards and compiled branch predicate.  The T/S counts are **given NEMO's
recorded entry** under Decision 52, from the round-5 admitted operand stream.

## Shared-card disposition

The final `packages/` tree is byte-identical to the base.  ORCA2, GYRE, DINO,
OVERFLOW, LOCK_EXCHANGE, and generic-card trajectories therefore cannot move;
the trajectory gates are not applicable to this measurement-only round.  The
full card battery nevertheless passed `160 + 10` tests.

## Review and validation

The required separate review was attempted with `codex exec --sandbox
read-only`.  Exact terminal verdict:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore: **independent review unavailable in-sandbox**.

Validation results:

- Four-card execution census: `PASS`, zero disagreements.
- Record-backed QCO/RK scorer: `PASS_MEASUREMENT_COMPLETE`; frozen counts
  reproduced exactly; one-ULP plant exited nonzero.
- Round-65 composition gate: `PASS_QCO_SCOPE`; scope and count plants exited
  nonzero; the compiled-predicate unit plant refused.
- Focused new/scope tests: `9 passed`.  A broader 43-test focused invocation
  reached 26/43 before reproducing a compiler zero-progress stall in the
  pre-existing tracer-RK3 tests; it is not called green.
- Full card battery: `160 passed` with nine dtype warnings, then `10 passed`.
- Receipt citation audit: both citations `PASS`; shifting the momentum-selector
  citation exits nonzero.  The default cumulative audit also `PASS`es with
  zero unmapped citations, failures, or map-audit failures.
- The single permitted `tests/ocean/fidelity -n 12` invocation collected
  `2,006` tests, reached 99%, reproduced exactly the five standing failures,
  then produced no output for more than five minutes and was interrupted.
  The five IDs were rerun serially and retained their registered signatures:
  SI3 `MY_SRC` provenance, round-51 private trace registry, round-129 stale
  phase-3 certification, missing `hires_lane_surface` case-board row, and four
  worktree-stamp emitters.  No round-65 test failed.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round65/`.
The decisive files are `card_scope.json`, `qco_current.json`,
`qco_plant.log`, `qco_scope_final.json`, both composition-plant logs,
`card_battery_160.log`, `card_battery_10.log`, `ocean_fidelity.log`, and
`known_reds_isolated.log`.

## Scope ledger

ASKED: re-scope the QCO boundary to ORCA2's executing vector program and name
its first actual non-bit statement.

UNASKED and unchanged: configuration, default, forcing, threshold,
stabiliser, carried state, initial state, NEMO arithmetic, and sea ice.

## OPEN

1. Execute Decision 52's independent-start ORCA2 ladder next.  Label every
   resulting number **independent** and compare it only with NEMO's own
   from-rest trajectory.
2. Once the independent ladder exists, rank month-scale ORCA2 errors by
   magnitude before walking more bit rows.
3. The source-ordered tracer QCO/RK statement remains locally exact but held
   by round 48's five OVERFLOW rows; no cancelling owner has been named.
4. Round 20's slow-forcing/barotropic walk remains open.
5. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.
