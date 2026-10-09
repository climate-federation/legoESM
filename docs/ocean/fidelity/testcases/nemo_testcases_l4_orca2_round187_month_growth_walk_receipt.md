# ORCA2 round 187: independent month growth walk and terminal-record correction

Date: 2026-10-08

Status: **STOPPED_FOR_RECORD**
Claim labels: every rung-0 number below is **independent**.  No given-entry
rung-7 number is mixed into the tables.

## LOUD CORRECTION FROM ROUND 188: the 95-step replacement is invalid

The round-187 claim that ending the experiment at step 95 was a safe way to
preserve the step-95 restart is **RETRACTED**.  The operator ran that launcher;
NEMO stopped during initialisation with exit 123 because experiment length 95
is not divisible by the resolved `nn_fsbc=2`.  No replacement restart was
written.  The compiled check is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-346`.

Round 188 replaces that protocol with an even 96-step run whose explicit
restart list ends `...,95,96`.  The step-96 file is a terminal sentinel: after
step 95 is written, NEMO advances `nitrst` to 96 at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202`, so the terminal open cannot
truncate the step-95 file.  The round-187 launcher now refuses immediately and
points to the round-188 launcher; it can no longer print the invalid READY
claim.

## Frozen question

Round 187 was preregistered before inspecting or scoring the round-186
restart payloads.  It asked whether the apparent step-95 `kt=0` was merely a
NEMO restart-time convention, where the independent trajectory first grew by
more than 10x, and which source-ordered stage-1 statement owned that growth.
The frozen predictions and falsifiers are in
`docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round187.md`.

No model file, configuration value, physical deck line, forcing, carried
state, stabiliser, sea-ice selector, or `unmeasured_features` entry changed.
The six sea-ice selectors remain exactly as carried by the ORCA2 card.

## Correction: step 95 is truncated, not conventionally stamped

The operator's proposed `kt`-convention explanation is **REFUTED**.  All four
step-95 files (two ranks in each of two twins) have `kt=0`, and each of T, S,
u, v, and SSH has a zero-length time dimension.  Each file is about 108 KiB,
whereas a valid checkpoint is about 15 MiB.  There is no step-95 ocean state
to admit.

The compiled program explains the exact failure.  With `ln_rst_list`, NEMO
initialises the list cursor and `nitrst` from the first list entry and selects
both the one-step-early and terminal-open conditions at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119`.  It then derives the
filename from the still-current `nitrst` and opens it for write at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:120-146`.  After a listed write it
closes the file, increments the list index with a capped `MIN`, and reloads
`nitrst` at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202`.  Therefore,
after writing the final listed checkpoint 95, the cursor stays pinned to 95;
terminal step 96 reopens the same filename for write but never reaches
`kt == nitrst`.  The existing payload is replaced by an empty NetCDF file.

The repaired admission gate does not call this record complete.  It admits
only steps 10 through 90, reports
`STOP_R187_TERMINAL_RESTART_OVERWRITTEN`, and requires a new target for step
95.  Across the admitted subset, all 18 rank/step twin comparisons have zero
unequal T/S/u/v/SSH values, and step 10 calibrates bit-for-bit to the admitted
round-83 record.  Missing-rank, twin-ULP, calibration, hidden-deck, and
terminal-payload plants all refuse.

Artifacts:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round187/existing_record_partial_admission.json`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round187/round186_admit_existing.log`

## Independent coarse growth table

The candidate entry T, S, u, v, and SSH are each bit-identical to NEMO.  The
fixed step-10 through step-90 comparison is record-backed and rank-complete.
The maximum column is the largest absolute error among the five fields; RMS
columns are whole-field RMS errors against the matching NEMO restart.

| step | maximum error (field, index) | ratio to prior maximum | T RMS (K) | S RMS (PSU) | u RMS (m/s) | v RMS (m/s) | SSH RMS (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 0.8633902030298275 T `[85,21,26]` | 4.316951015e9 vs 2e-10 floor | 0.005963034072069885 | 0.0016088940282822529 | 0.004343087142442177 | 0.004356804380110146 | 0.02802652392771078 |
| 20 | 1.5455573056631327 SSH `[11,133]` | 1.790102899 | 0.00987585481794429 | 0.0029283134905970253 | 0.005481394599912885 | 0.00580693369293246 | 0.044451336914902946 |
| 30 | 1.3782141824426408 SSH `[11,92]` | 0.891726355 | 0.013513765463289761 | 0.004186806067397323 | 0.006027504898305049 | 0.00637059118308224 | 0.05740879136897435 |
| 40 | 1.99351376712747 SSH `[11,92]` | 1.446446998 | 0.015650103265779768 | 0.005423826512969751 | 0.0066818732731821285 | 0.00797981568868069 | 0.06731136578573246 |
| 50 | 2.1358588853710776 v `[115,59,4]` | 1.071404131 | 0.01807435502408221 | 0.0065456633987958455 | 0.008297504804495875 | 0.009288706645736704 | 0.0787014085676689 |
| 60 | 2.8278310251530248 SSH `[117,57]` | 1.323978398 | 0.020253157263878142 | 0.007550161882639453 | 0.011594761707461185 | 0.010435160609799737 | 0.09765707518140823 |
| 70 | 6.1305767566484874 SSH `[5,128]` | 2.167943099 | 0.02243915306540838 | 0.008474220517215752 | 0.016283819839465464 | 0.015780358248754678 | 0.13224652473375695 |
| 80 | 5.643745271486383 SSH `[6,130]` | 0.920589611 | 0.02457378148214106 | 0.009341285768616989 | 0.024581482308815704 | 0.027847025122317678 | 0.17137830811408694 |
| 90 | 13.662396746636144 SSH `[96,175]` | 2.420803224 | 0.027596103254340372 | 0.010141647298123831 | 0.0405445834652363 | 0.04393552856835616 | 0.24606477171190166 |
| 95 | **UNMEASURED-with-spec** | truncated oracle payload | — | — | — | — | — |

The first coarse >10x boundary is therefore step 10.  Step 95 cannot change
that first-boundary verdict, but it remains required to complete the frozen
table and to compare the candidate's explosive step-90-to-95 growth.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round187/month_growth_partial.json`.

## Stage and statement refinement

The current certified independent ten-step ladder refines the first coarse
boundary without an executable observer.  Entry is bit-identical.  The first
stage >10x boundary is kt=1 stage 1: SSH maximum error
`0.1314585958201272 m` at `[j=1,i=49]`, a ratio of
`657292979.100636` to the fixed `2e-10` floor.  The ladder's first non-bit
field remains kt=1 stage-1 T; these are different statistics and are not
conflated.

The preregistered HPG-owner prediction is **REFUTED**.  In the executed
vector-invariant arm, NEMO forms the completed three-dimensional momentum
RHS and then vertically averages it with live face thickness, mask, and
inverse reference depth at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/stp2d.f90:206-219`.  The already-gated
source-associated replay continues to name this vertical average as the
first source-ordered non-bit statement.  Its existing status is
`HELD_BY_GYRE_2ULP_GATE`; round 187 neither relands nor bypasses that gate.

Artifacts:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round187/rung0_ladder.json`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round187/growth_walk_final.json`

## Retracted replacement acquisition

**RETRACTED by the operator-run result described above.**  The committed
round-187 replacement launcher used new twin target names and terminated at
step 95, but NEMO's compiled surface-boundary consistency check rejects that
odd experiment length before stepping.  Its syntactic preflight did not cover
this compiled runtime invariant.  The launcher now refuses and directs the
operator to round 188's terminal-sentinel acquisition.

Launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round187_growth_acquisition/run.sh`.

Preflight evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round187/acquisition_preflight.log`.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R187-P1 | **REFUTED** | All four step-95 files contain `kt=0` and zero-length T/S/u/v/SSH payloads; this is terminal overwrite, not a usable time convention. |
| R187-P2 | **CONFIRMED** | Steps 10–90 pass 18 rank-complete twin comparisons and step-10 calibration with zero unequal values. |
| R187-P3 | **CONFIRMED** | The first coarse >10x error-growth boundary is step 10. |
| R187-P4 | **PARTLY CONFIRMED / REFUTED** | kt=1 stage 1 is the first stage boundary; HPG is not the first source-ordered owner.  The completed-RHS depth average remains first. |
| R187-P5 | **CONFIRMED** | No scientific landing was attempted; model and card files are unchanged. |
| R187-P6 | **CONFIRMED** | All ten record/growth controls fire, and the replacement acquisition preflight passes. |

## Validation and review

Focused round-187 tests pass 7/7.  The required single
`tests/ocean/fidelity -n 12` invocation collected 2,886 tests and reached 99%
before its runner stopped producing output and no Python process remained;
the durable log contains 2,871 passes, 7 skips, and 4 failures.  The four are
registered pre-existing reds: SI3 scalar-math `MY_SRC` provenance, the
allow-dirty scope ratchet, the worktree-stamp ratchet, and the stale GYRE
spread-floor record.  Isolated reruns reproduced each failure.  The sole
scheduled test left without a terminal result was the pre-existing LOCK
stage-sweep planted-control end-to-end test; its isolated rerun likewise lost
its process without a result after collection.  Every round-187 test passed.

The default citation gate passes 274 citations with zero unmapped or failing
entries.  This receipt passes all four compiled citations; shifting the new
restart-open range by two lines makes the gate fail on its first endpoint.
The citation-gate unit suite passes 17/17.

The separate `codex exec --sandbox read-only` review was attempted and
returned `failed to initialize in-process app-server client: Read-only file
system`.  Verdict: **independent review unavailable in-sandbox**.

## OPEN

1. The operator must run the round-188 terminal-sentinel launcher.  Admission
   requires both ranks in two array-identical twins, the complete step
   10..95 list, and step-10 calibration to round 83.
2. Admit step 95, complete the frozen table, and confirm the already-first
   step-10 boundary.  Then resume the source-ordered walk from the named
   completed-RHS vertical-average statement under its standing GYRE gate.
3. The halo/V-transport unit remains private and HELD.  Re-test it only after
   the upstream independent-month statement is resolved; no partial operand
   may land.
4. Rung 7 and sea ice are untouched.  Sea ice remains exactly the ORCA2
   card's existing `unmeasured_features` declaration.
