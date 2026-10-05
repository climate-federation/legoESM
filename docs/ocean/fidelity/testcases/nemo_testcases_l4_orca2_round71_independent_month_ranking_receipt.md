# ORCA2 round 71 receipt — independent month ranking

Date: 2026-09-28

Base: `5311ab729`

Disposition: **HELD; the independent month is measured and temperature owns
the next magnitude walk**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state. No NEMO entry field is substituted. The comparison target is NEMO's
own from-rest 240-step restart.

Sea ice remains out of scope. The ORCA2 card, all selectors, and the exact
six-entry `unmeasured_features` tuple are unchanged.

## Answer

The operator's round-69 surface record is admissible: 480 self-describing
rank-step frames, 200 raw-bit ten-step calibration comparisons, and four
bit-exact terminal restart shards. The direct admission and all six admission
plants pass/fail as intended.

The independent legoESM ocean then completed 240 production CPU/JIT/fp64
steps. Every terminal field is non-bit. Temperature ranks first by both
maximum absolute error and RMS, so **temperature owns the next month-scale
walk**.

| rank | field | units | max absolute | RMS | unequal / total |
|---:|---|---|---:|---:|---:|
| 1 | T | degC | 21.637832697714707 | 0.17564842520962015 | 438484 / 799200 |
| 2 | ssh | m | 5.563319462718695 | 0.06822817352994356 | 16433 / 26640 |
| 3 | S | g/kg | 2.3984272944912632 | 0.03606444372794216 | 430552 / 799200 |
| 4 | u | m/s | 2.054457499249187 | 0.022406667645546077 | 444097 / 799200 |
| 5 | v | m/s | 1.1310611325760693 | 0.013915785410822645 | 449483 / 799200 |

This table is exclusively independent. It is not mixed with the earlier
"given NEMO's entry" twin measurements.

## Instrument correction and retained retraction

The first attempt refused before integration because the new surface record
contains the ten production operands while the legacy frame also contains 25
passive/debug streams. Calibration now requires the exact ten-field admitted
registry, compares every consumed operand raw-bit exactly, and records the
legacy-only names. The corrected predicate completed 100 comparisons and the
new/old forcing paths produced a bit-identical ten-step state.

Round 66's reconstructed chlorophyll clock was valid only at step 1. NEMO
interpolates from its resolved record centres using the live seconds counter
at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/fldread.f90:235-246`.
The corrected clock switches from December/January to January/February at
step 125. Therefore round 66's kt=2..10 exact-input trajectory metrics are
**RETRACTED** as stale-instrument numbers. The corrected kt=10 metrics do not
reproduce them, as preregistered. The round-66 independent initial-state
census remains valid.

NEMO writes the terminal before fields `sshn`, `un`, `vn`, `tn`, and `sn` at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/restart.f90:170-184`.
The stage-3 swap makes `Nbb` the completed state before that restart call at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:223-272`.
Both restart shards' float32 longitude/latitude arrays are bit-identical to
the card orientation, and the terminal shard digests match the admitted
ledger.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R71-P1 direct admission | **CONFIRMED** | The existing record passed inventory, ten-step calibration, restart passivity, and all six plants. |
| R71-P2 reproduce round-66 kt=1..10 metrics | **REFUTED** | Source inspection found the stale chlorophyll clock; round 71a froze the correction before rerunning, and corrected kt=10 differs. |
| R71-P3 complete independent month | **CONFIRMED** | 240 steps and 480 rank-step surface frames consumed; status `PASS_INDEPENDENT_MONTH_RANKING`. |
| R71-P4 terminal restart orientation | **CONFIRMED** | Both coordinate arrays are raw-bit identical to the card and both restart digests match the ledger. |
| R71-P5 all five fields non-bit | **CONFIRMED** | No terminal row is bit-identical. |
| R71-P6 temperature largest by max absolute error | **CONFIRMED** | 21.637832697714707 K, ahead of SSH's 5.563319462718695 m. |
| R71-P7 disposition HELD | **CONFIRMED** | This is a ranking round; no model statement was changed. |
| R71a corrected clock changes old kt=10 metrics | **CONFIRMED** | The classifier records `round66_metrics_reproduced: false`. |

Failed predictions are retained as **REFUTED**. No later measurement silently
rewrites the original preregistration.

## Validation

- independent month gate: `PASS_INDEPENDENT_MONTH_RANKING`; wall time
  9296.985275 seconds; JSON SHA-256
  `b0aef68e2072b821018ad84df82c517183716c3aa9fd7570f4a88507dd2dd2b6`;
- six month classifier plants: all FIRED (initial mode, frame count, ten-step
  calibration, restart orientation, terminal digest, and non-finite field);
- default citation gate: PASS, 274 citations, zero failures and zero unmapped;
  this receipt's gate: PASS, three citations, zero failures and zero unmapped;
  the shifted `fldread` citation plant FIRES with `SYMBOL-NOT-AT-LINE`;
- focused round-71 tests: `12 passed`;
- shared-card battery: `160 passed` with nine dtype warnings;
- tank battery: `10 passed`;
- wide `tests/ocean/fidelity -n 12`: 2,043 collected; reached 99% with 2,020
  passes, five registered failures, and seven skips, then reproduced round
  70's no-summary wrapper stall after the workers disappeared. The five IDs
  were rerun serially and reproduced the standing signatures: round-51
  private trace registry, SI3 `MY_SRC` provenance, round-129 stale phase-3
  certification, missing `hires_lane_surface` case-board row, and four
  worktree-stamp emitters. All 12 round-71 tests passed in the wide run.

No `packages/` file changed. GYRE, DINO, and tank trajectories therefore
cannot move; trajectory landing gates are not applicable to this
measurement/instrument round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round71/`.

The required separate `codex exec --sandbox read-only` review was attempted
three times, including once on the final committed diff. Its terminal verdict
was `Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`; therefore **independent review unavailable in-sandbox**.
All attempts are retained in the evidence directory.

## Scope ledger

ASKED: admit the operator's month surface record, run the Decision 52
independent ORCA2 ocean month, and rank the terminal errors by magnitude.

UNASKED and unchanged: model arithmetic, configuration, selectors,
thresholds, stabilisers, carried state, sea ice, and the held QCO/RK change.

## OPEN

1. Temperature owns the next month-scale walk. Acquire or use admitted
   intermediate from-rest checkpoints to rank the first growth interval by
   depth and region, then name the first source-ordered non-bit temperature
   statement; do not return to aggregate bit rows first.
2. Preserve the corrected piecewise chlorophyll clock and the independent
   label in every downstream table.
3. The independent kt=1 SSH selector gap remains at `STOP_SELECTOR_GAP`; all
   six sea-ice selectors stay frozen.
4. The source-ordered tracer QCO/RK statement remains locally exact but held
   by round 48's five OVERFLOW rows.
