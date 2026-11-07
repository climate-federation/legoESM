# Round 185 receipt — carried-N2 day-240 owner re-ranking

**Status: STOPPED_FOR_RECORD.**  On the Round-183 carried-step-entry-N2
baseline, the largest surviving day-240 process-budget row is the temperature
state already present at day 180: `+3.879177135308106e-05 K` signed carry.
Shortwave is second at `+3.738270862575350e-05 K`.  The old vertical-
diffusion owner collapses from `+2.4168271578053416e-02 K` to
`+1.4456380859143594e-05 K`.  The admitted oracle stream begins at day 180,
so it cannot name the statement that created the winning incoming row.  No
physics, configuration, card, carried state, default, or stabilizer changed.

Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round185.md`, commit
`ccfba56b3`.  Trace/scorer commit: `aeeaed0c1`.  Acquisition-preparation
commit: `d24db976a`.  Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round185/`.

## Compiled process order

The admitted oracle writer opens a stage-3 frame and writes the before tracer
and three free-surface ratios at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:818-830`.
The compiled stage calls advection and then the RK3 surface boundary and writes
both accumulated boundaries at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:861-869`.
It calls shortwave and lateral diffusion, writing both subsequent boundaries,
at `GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:930-952`,
then calls the implicit vertical solve and writes the after tracer at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/stprk3_stg.f90:964-970`.

Inside the solve, the compiled program forms the live-thickness weighted
temperature content and forward recurrence at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:545-560`, then performs
the backward recurrence at
`GYRE_OMIP_L2_P3_SM_R123PROC/BLD/ppsrc/nemo/trazdf.f90:563-578`.
These direct boundaries define the rows below.  A projection row is not
relabeled as a source statement.

## Trace admission and endpoint closure

The owner self-check passes.  The original Round-123 oracle stream re-admits
as 360 ordered records and `509,508,000` bytes.  The new legoESM trace runs
the independent production JIT trajectory from rest, records steps
1081--1440, and carries only the separately evaluated ordinary production
state.  It reports zero unequal carried-state bytes.

The generated day-180 and day-240 T/S/u/v/SSH snapshots are bit-identical to
the immutable Round-183 after arm in all ten comparisons.  The exact endpoint
is `6.5861718814795174e-05 K`; the explicit expected-value control rejects a
one-ULP-different expected scalar.  The component arrays reconstruct the
independent endpoint with maximum wet-cell residual
`3.552713678800501e-15 K`.  The signed carries sum to
`6.5861718814793277e-05 K`, differing from the RMS by
`-1.8973538018496328e-18 K`.  The cancellation ratio is
`1.889274213163219`.

## Day-240 magnitude ranking

Ranking is by absolute signed projection onto the independent day-240
temperature-error array.  Component RMS is descriptive; it is not the rank.
Day-30 process carry is unavailable because the oracle stream starts at day
180.

| rank | owner | day-240 signed carry (K) | component RMS (K) | strongest birth block | largest depth / longitude / latitude partition |
|---:|---|---:|---:|---|---|
| 1 | incoming day-180 state | `+3.879177135308106e-05` | `6.113303379494912e-05` | before day 180 | 0--100 m / east third / south <=37.2 N |
| 2 | shortwave | `+3.738270862575350e-05` | `9.056733921955400e-05` | days 200--210, `+6.284854353775624e-06 K` | 0--100 m / west third / south <=37.2 N |
| 3 | surface boundary | `-2.630326036951712e-05` | `9.036622145680882e-05` | days 230--240, `-4.772341746661814e-06 K` | 0--100 m / west third / south <=37.2 N |
| 4 | vertical diffusion | `+1.445638085914359e-05` | `6.308124619704875e-05` | days 230--240, `+1.550545081166303e-05 K` | 0--100 m / interior third / south <=37.2 N |
| 5 | advection | `+4.515422064864913e-06` | `3.663589294397633e-05` | days 230--240, `+2.576423470499433e-06 K` | 0--100 m / interior third / south <=37.2 N |
| 6 | lateral diffusion | `-2.979510240852815e-06` | `4.416628617197218e-05` | days 210--220, `-7.339349324572241e-07 K` | 0--100 m / east third / north >37.2 N |
| 7 | free-surface geometry | `-1.793477933083419e-09` | `3.400737670850542e-09` | days 200--210 | 0--100 m / west third / south <=37.2 N |
| 8 | rounding closure | `+2.532218223705621e-16` | `2.951755411727669e-14` | days 210--220 | 100--1000 m / interior third / north >37.2 N |

The preregistered incoming-owner prediction is confirmed.  The vertical row
does not retain its pre-landing carry: its signed magnitude falls by a factor
of `1671.79`.  This supersedes the Round-124 ranking for the carried-N2
baseline; it does not retract Round 124's measurement on its own earlier arm.

## Why no statement lands

The winning row is a boundary condition for this interval, not an operator:
`legoESM(day180) - NEMO(day180)`.  Its true birth lies somewhere in steps
1--1080.  The existing developed-state records start at step 1081, and the
from-rest ladder covers only steps 1--10.  Therefore no record in the campaign
enumerates the compiled process boundary that first creates the majority
owner.  Naming shortwave, vertical diffusion, or any day-180-to-240 statement
would skip the larger incoming row and violate magnitude ordering.

No first non-bit statement is claimed, no held patch is re-evaluated, and the
Decision-43/45/55/59 landing gate is not invoked.  The immutable before arm
remains Round 183.

## Frozen pre-day-180 acquisition

The operator-run script is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round185_pre180_process/run.sh`.
It clones the exact admitted `GYRE_OMIP_L2_P3_SM_YRPERT` source card into the
new target `GYRE_OMIP_L2_P3_SM_R185PREPROC`, copies `EXP00`, `MY_SRC`, and the
preprocessor card file by file, applies only an additive stage-3 writer plus
`nn_itend=1080`, and emits the identical nine-write/1,415,300-byte layout for
every step 1--1080.  Frozen totals are 1,080 records and
`1,528,524,000` bytes.

The script never calls `/usr/bin/time`; it uses the shell timer.  Every refusal
prints `REFUSE`.  Its source card passes `cpp` plus `gfortran -fsyntax-only`.
Admission requires the instrumented day-30 and day-180 restart files to remain
byte-identical to the uninstrumented year run with SHA-256 values
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`
and `6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`.
The generalized existing record gate keeps its Round-123 defaults but accepts
an explicit contiguous interval and explicit passive-restart hash set; the
stamp, truncation, SBC-ULP, SBC-effect, and trajectory-ULP plants must all
fire before `RUN_DONE` is written.

## Plants, tests, citations, and review

The lego trace stamp, one-ULP row, and stored production-effect plants each
print `STATUS PLANT-FIRED` and exit 1.  The endpoint-value control exits 1 and
states that the measured endpoint differs from the one-ULP-shifted explicit
expectation.  The focused owner suite reports:

```text
49 passed in 32.91s
```

The final citation/worktree-stamp suite reports **1 failed, 25 passed in
5.32s**.  The sole failure is the starting-tip worktree-stamp ratchet: its
four offenders are the Round-50, Round-146, Round-156, and Round-184 gates;
none is in the `0b6455560..HEAD` diff.  All 16 citation-gate tests pass.  No
model implementation changed, so a broad ocean physics battery would not
exercise any additional Round-185 path.

The separate read-only Codex review was attempted against the ranking,
generalized record gate, and acquisition script.  It produced no scientific
verdict.  Its verbatim terminal finding is:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Thus independent review is unavailable in-sandbox; it did not issue `DO NOT
SHIP`.  At committed receipt `bad0c8c55`, the direct citation gate passes all
6 citations with zero failures and zero unmapped citations; the default
cumulative gate passes all 274 citations with the same zero counts.  Shifting
the first compiled writer range by two lines produces `SYMBOL-NOT-AT-LINE`,
reports `status: FAIL`, and exits 1.

Key artifact SHA-256 values:

| artifact | SHA-256 |
|---|---|
| `day240_process_budget_final.json` | `e74ef5c69c17f6deba5cd536fc23203654e78d1ea7b9cf7491f41640bb4f6919` |
| `lego_process_trace_report.json` | `ff7adf2f5685cbe9b4b40f461c0a328028f9ee31f4b2cc20f96d9be71e78b45c` |
| `focused_year_owners_tests.log` | `2c8d314ce9b86d5bb633d9a5595f3efeabf15742327c40daf06cbb396c8c2bb9` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |

## Required unchanged headlines and blast radius

Because executable model code did not change, the certified rows remain:

- kt2 T/S/U/V: `1.4210854715202004e-14`,
  `2.1316282072803006e-14`, `8.326672684688674e-17`, and
  `9.714451465470120e-17`;
- kt3 T/S: `4.9403105251144552e-07` and
  `4.0085410546453204e-08`;
- day-30/day-240/day-360 T3D RMS: `2.3276772050683987e-06`,
  `6.5861718814795174e-05`, and `2.6709923853294689e-03 K`.

GYRE, generic NEMO-GYRE, both DINO cards, LOCK_EXCHANGE, and OVERFLOW execute
no changed model statement and have zero numerical blast radius.  ORCA2 is
**UNMEASURED-WITH-SPEC**: its native process record, independent production
trace, masks, and endpoint projection must be acquired before transferring
this ranking.

## OPEN — round 186

1. The operator runs the Round-185 acquisition script.  Admit its 1,080
   process frames only if both passive restarts are byte-identical and all five
   plants fire.
2. Extend the existing production process trace over steps 1--1080 on the
   Round-183 carried-N2 baseline; do not create a parallel harness.  Partition
   the interval into the existing daily or ten-day blocks and identify the
   earliest block that materially contributes to the measured day-180
   incoming carry.
3. Rank the same compiled rows over that earlier interval against the fixed
   day-240 endpoint.  The largest physical row becomes the statement walk;
   preserve the incoming initial-state row separately and require exact array
   closure.
4. Only a source-cited, one-variable statement can become a candidate, and it
   still requires the complete Decision-43/45/55/59 gate and executing-card
   census.  No downstream day-180-to-240 row may be promoted ahead of the
   pre-day-180 owner.

No configuration decision is requested.
