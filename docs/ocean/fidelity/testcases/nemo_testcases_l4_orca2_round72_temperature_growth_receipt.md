# ORCA2 round 72 receipt — independent temperature growth

Date: 2026-09-29

Base: `5fcad2d985fe5dbc1b408ac532d435b7a6886159`

Disposition: **HELD; tracer advection owns the next source-ordered walk**

Claim label: **independent**. legoESM starts from the ORCA2 card's own ocean
state and consumes the admitted surface operands. No NEMO entry field is
substituted. NEMO's own from-rest restarts are the comparators.

Sea ice remains out of scope. The ORCA2 card, selectors, 10,800 s time step,
and exact six-entry `unmeasured_features` tuple are unchanged.

## Answer

The independent temperature error grows primarily after the ten-step ladder.
On the active T cells, the fp64 volume-weighted RMS is exactly zero at entry,
`0.0048328820175236025 K` at kt=10, and `0.06100404684984332 K` at kt=240.

| independent interval | increase in volume-weighted squared-error numerator (K2 m3) | end weighted RMS (K) |
|---|---:|---:|
| kt=10 -> 240 | 5.035212621412946e15 | 0.06100404684984332 |
| kt=0 -> 10 | 3.1801470807639953e13 | 0.0048328820175236025 |

The kt=10 restart is calibrated against the admitted rank-0 completed-stage T
frame on 233,341 owned wet cells: 0 unequal, raw-bit exact. Both depth and
region decompositions close to the global kt=240 squared-error numerator with
zero observed reduction error.

For the preregistered total kt=0 -> 240 depth ranking, the leading native
levels are:

| rank | level | squared-error contribution (K2 m3) | kt=240 weighted RMS (K) | kt=240 max absolute (K) |
|---:|---:|---:|---:|---:|
| 1 | k=0 | 7.009723723798712e14 | 0.4429487793183157 | 3.9708633394988038 |
| 2 | k=1 | 5.899705219702601e14 | 0.4063504849214416 | 5.809272936998733 |
| 3 | k=3 | 5.393442801467135e14 | 0.389331196475606 | 12.077454806354094 |
| 4 | k=4 | 4.543964084256562e14 | 0.3580763197339674 | 10.543666301892609 |
| 5 | k=2 | 4.3686684039440894e14 | 0.34964411047548916 | 7.214622159012196 |

The short kt=0 -> 10 interval has a different owner, k=26
(`4.0024055640309385e12 K2 m3`), followed by k=0. This does not falsify the
frozen depth prediction, which names the total kt=0 -> 240 contribution.

The supplied, exact wet-column basin partition ranks:

| rank | supplied-mask region | squared-error contribution, kt=0 -> 240 (K2 m3) | kt=240 weighted RMS (K) | kt=240 max absolute (K) |
|---:|---|---:|---:|---:|
| 1 | Pacific | 1.977602702586194e15 | 0.05154904346037755 | 4.1902804724677996 |
| 2 | Atlantic | 1.7240769901003658e15 | 0.07033305566229126 | 4.728692741274198 |
| 3 | Indian | 1.360873802155285e15 | 0.07115605330175129 | 21.637832697714707 |
| 4 | other wet complement | 4.460597378741007e12 | 0.3644326454425571 | 1.7347251315258094 |

The partition uses only `atlmsk`, `pacmsk`, and `indmsk` from the supplied
`subbasins.nc`, plus their exact wet complement: binary, zero overlapping wet
cells, and zero uncovered wet cells after the complement. Pacific owns the
integrated squared error, while the terminal maximum is independently located
in the Indian mask at grid index `[87, 158, 13]`, longitude
`42.75044945987611`, latitude `12.370966298224346`, and reference depth
`-142.59327683319088 m`.

Round 71's all-cell temperature row is reproduced exactly. Its 438,484 unequal
cells comprise 430,552 active wet cells and 7,932 dry signed-zero mismatches;
the dry mismatches have numerical absolute error zero. The round-71 all-cell
maximum and RMS remain correct, but the physically ranked headline here is
wet-only and volume weighted.

## First non-bit statement

The independent kt=1 rank-0 tracer entry is raw-bit exact on all 228,641
recorded wet cells. Immediately after the compiled tracer-advection call, all
228,641 cells are non-bit and the maximum temperature error is
`3.0869700763080185e-07 K`:

`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:633-643`

The owning statement is `CALL tra_adv` at line 637. The next surface-source
boundary at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:645-651`
has the same maximum, so it is not first. The subsequent stage-1 QCO/RK update
at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:670-680`
raises the maximum to `0.0011816070735529705 K`, but is downstream of the
already non-bit advection accumulator.

Therefore **tracer advection owns the next source-ordered statement walk**.
This round does not attribute the difference to a statement inside
`tra_adv`; that narrower attribution is OPEN.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R72-P1 restart/time-level calibration | **CONFIRMED** | kt=10 restart versus completed-stage T: 233,341 wet cells, 0 unequal. |
| R72-P2 execution replay | **CONFIRMED** | The full round-71 all-cell terminal T tuple reproduced exactly. |
| R72-P3 interval owner | **CONFIRMED** | kt=10 -> 240 contributes 5.035212621412946e15 K2 m3, versus 3.1801470807639953e13 for kt=0 -> 10. |
| R72-P4 depth owner | **CONFIRMED** | k=0 has the largest total kt=0 -> 240 contribution. |
| R72-P5 region owner | **CONFIRMED** | Pacific has the largest total kt=0 -> 240 contribution. |
| R72-P6 source-order owner | **CONFIRMED** | Entry is exact; after-advection is the first non-bit boundary. |
| R72-P7 disposition HELD | **CONFIRMED** | No model statement or configuration was changed. |

No prediction was refuted. The interval, depth, region, dry-cell,
restart-calibration, and source-boundary predicates remain separate.

## Validation

- temperature-growth gate: `PASS_TEMPERATURE_GROWTH_RANKING`; 240 production
  CPU/JIT/fp64/libm steps; JSON SHA-256
  `e8e73ecac6ed8619affe880f148e407181de4c9d642f6334069f4578b95b1aff`;
- restart-time-level, dry-inclusion, region-overlap, interval-order,
  temperature-boundary, and terminal-ULP plants: all six FIRED;
- default citation gate: PASS, 274 citations, zero failures, zero unmapped,
  and zero map-audit failures; this receipt's gate: PASS, three citations,
  zero failures, zero unmapped, and zero map-audit failures; the shifted
  `stprk3_stg` citation plant FIRES with `SYMBOL-NOT-AT-LINE`;
- focused round-71/72 and citation tests: `36 passed`;
- shared-card battery: `160 passed` with nine dtype warnings;
- tank battery: `10 passed`;
- wide `tests/ocean/fidelity -n 12`: 2,051 collected; reached 99% with 2,026
  passes, five registered failures, and seven skips, then reproduced round
  71's no-summary wrapper stall. All eight round-72 tests passed in the wide
  run. The five failing IDs were rerun serially and reproduced the standing
  signatures: round-51 private trace registry, SI3 `MY_SRC` provenance, four
  worktree-stamp emitters, missing `hires_lane_surface` case-board row, and
  round-129 stale phase-3 certification.

No `packages/` file changed. GYRE, DINO, and tank trajectories therefore
cannot move; trajectory landing gates are not applicable to this diagnostic
and instrument round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round72/`.

The required separate `codex exec --sandbox read-only` review was attempted
before measurement and again on the final committed diff. Both returned
`Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`; therefore **independent review unavailable
in-sandbox**. Both logs are retained in the evidence directory.

## Scope ledger

ASKED: rank independent ORCA2 temperature-error growth by the available exact
intervals, native depth, and supplied basin regions, and name the first
source-ordered non-bit temperature statement.

UNASKED and unchanged: model arithmetic, configuration, forcing
reconstruction, selectors, thresholds, stabilisers, carried state, new NEMO
output, sea ice, and the held shared tracer QCO/RK candidate.

## OPEN

1. Walk inside ORCA2's selected `tra_adv` path from the admitted exact kt=1
   entry and transport operands. Name the first non-bit arithmetic statement
   in compiled source order; first print the card-execution census required by
   round 61.
2. Preserve the independent label, wet-only volume metric, native-depth bins,
   and supplied-mask basin partition in downstream rankings.
3. The independent kt=1 SSH selector gap remains at `STOP_SELECTOR_GAP`; all
   six sea-ice selectors stay frozen.
4. The source-ordered tracer QCO/RK statement remains downstream and locally
   non-bit; its previously held shared change is not authorized by this round.
