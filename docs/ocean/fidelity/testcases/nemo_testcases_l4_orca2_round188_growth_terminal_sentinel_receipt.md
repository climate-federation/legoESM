# ORCA2 round 188: growth-restart terminal sentinel

Date: 2026-10-08

Status: **STOPPED_FOR_RECORD**

Claim labels: every rung-0 scientific number referenced here is
**independent**. No given-entry rung-7 number is mixed into the table or
verdict. Sea ice and the shipped rung-10 card are untouched.

## LOUD CORRECTION FROM ROUND 189: the eleven-entry list is invalid

The round-188 claim that the `10,...,95,96` restart list was preflight-ready is
**RETRACTED**. The operator ran it; NEMO stopped during namelist
initialisation with exit 123 and `iostat=5010`, before the first time step.
The executed binary declares `nn_stocklist` with only ten elements at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/in_out_manager.f90:51-51`, and reads the config
namelist at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/domain.f90:323-325`. The eleven-value
assignment is therefore not a valid NEMO deck. No restart payload exists.

The gate now refuses that actual deck with
`compiled nn_stocklist capacity 10 exceeded`, and the round-188 launcher exits
78 before staging or running. Round 189 replaces it with the bounded list
`95,96`, reusing the already-admitted step-10..90 prefix rather than requesting
eleven checkpoints in one run.

## Outcome

Round 187's proposed 95-step replacement protocol is retracted and disabled.
The operator ran it, and NEMO refused before the first time step with exit 123:

```text
sbc_init : experiment length (95) is NOT a multiple of nn_fsbc (2)
Impossible to properly do model restart
```

The repaired acquisition is preflight-ready under a new round-188 evidence
root and new twin target names. It runs 96 steps and appends a step-96 terminal
sentinel after the scientific step-95 checkpoint. It has not been run by the
operator yet, so step 95 remains **UNMEASURED-with-spec** and no scientific
landing is claimed.

## Compiled-source statement

The record binary is the admitted rung-0 `ORCA2_OMIP_L4` binary. Its compiled
source fixes the protocol:

| order | compiled statement | consequence |
|---:|---|---|
| 1 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-346` | With restart output enabled, NEMO requires the experiment length to be divisible by `nn_fsbc`; this deck resolves `nn_fsbc=2`, so 95 steps is invalid and 96 is valid. |
| 2 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:94-119` | The explicit restart-list cursor sets `nitrst`; a restart is opened one step early, and the terminal fallback opens only when no restart is already open. |
| 3 | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/restart.f90:188-202` | Writing step 95 closes that file and advances `nitrst` to the next list member. Making that member 96 ensures terminal step 96 opens a different file instead of truncating step 95. |

This is a run-protocol correction, not a configuration choice. The rendered
deck gate reports `physical_delta: []` and exactly:

```text
nn_itend    = 96
nn_stock    = 96
ln_rst_list = .true.
nn_stocklist = 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 96
nn_fsbc     = 2
```

Step 96 is checked for a complete, finite, correctly stamped, twin-identical
payload but is excluded from the frozen scientific growth table, whose steps
remain 10 through 95.

## Fail-closed instrument

The shared growth-record gate now renders only the valid 96-step sentinel
protocol. It retains support for classifying round 186's historical truncated
record, but it no longer emits the invalid 95-step deck. The round-187 launcher
itself refuses immediately, so the retracted READY claim cannot recur.

The round-188 launcher pins the preregistration and gate by SHA256, verifies
the admitted base deck/input manifests, uses new target names, demands two
ranks in two independent twins, and has no `/usr/bin/time` dependency. Its
admission plants cover missing rank, one-ULP twin movement, step-10 calibration,
hidden deck delta, missing terminal sentinel, truncated sentinel payload, and
a sentinel stamped as step 95. Unit controls for the three new sentinel cases
all fire.

Artifacts:

- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round188/acquisition/namelist_growth_sentinel_preflight`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round188/acquisition/deck_preflight.log`
- `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round188/acquisition_preflight.log`

Launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round188_growth_acquisition/run.sh`.

## Frozen prediction ledger

| prediction | outcome | evidence |
|---|---|---|
| R188-P1 | **CONFIRMED** | The rendered assignment inventory has zero physical deltas and adds only the 96-step restart protocol plus the explicit list ending in 96. |
| R188-P2 | **UNMEASURED-with-spec** | The operator-run twins do not yet exist. Admission requires complete finite `kt=95` and `kt=96` files on both ranks and exact twin equality. |
| R188-P3 | **UNMEASURED-with-spec** | Steps 10-90 remain admitted historical evidence, but the completed 10-95 table is withheld until the new step-95 payload admits. |
| R188-P4 | **UNMEASURED-with-spec / standing prior result** | No new passive ladder was run; round 187's completed-RHS vertical-average boundary is neither promoted nor withdrawn by a protocol preflight. |
| R188-P5 | **PARTLY CONFIRMED** | Unit controls for missing/truncated/misstamped sentinel fire; the launcher will run all seven record plants against the real twins before admission. |

## Validation and review

The focused growth-record gate suite passes 6/6, including the three new
terminal-sentinel controls. The historical round-186 partial record still
classifies `STOP_R187_TERMINAL_RESTART_OVERWRITTEN`; compatibility was checked
directly against its existing twin directories.

The required single `tests/ocean/fidelity -n 12` battery collected 2,887
tests. Its durable log records 2,858 passes, 7 skips, and 4 failures before the
final worker stopped producing output at 99%; 18 scheduled tests have no
terminal result because that worker hung on the pre-existing
`test_prediction_plant_is_fail_closed` end-to-end control. The four failures
are the registered pre-existing SI3 scalar-math provenance, allow-dirty scope,
worktree-stamp, and stale GYRE spread-floor reds. No round-188 test failed.
The stuck parent was interrupted after the log had been unchanged for more
than seven minutes; the full battery was not rerun.

The default citation gate passes 274 citations with zero unmapped, failing, or
unaudited map entries. This receipt passes 3/3 compiled citations, and the
round-187 correction passes 5/5. Shifting
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcmod.f90:341-346` by two lines makes the gate
fail at the first endpoint, as required.

The separate `codex exec --sandbox read-only` review was attempted and
returned `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**.

No package/model file changed, so no ORCA2, GYRE, DINO, tank, or month
trajectory could move in this round. This is an instrument/run-protocol round;
the usual scientific landing gates remain mandatory after the record admits.

## OPEN

1. The operator runs the round-188 launcher. The gate must admit steps 10..95
   scientifically and step 96 as the non-scientific terminal sentinel across
   both ranks and twins.
2. Complete the frozen step-95 row and confirm or refute step 10 as the first
   coarse >10x boundary.
3. Resume the already-named completed-RHS vertical-average statement only
   under its standing shared GYRE gate.
4. Re-test the private halo/V-transport unit only after that upstream statement
   is resolved. No partial operand may land.
