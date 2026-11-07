# Preregistration — ORCA2 round 90 rung-0 debug crash resolution

Date: 2026-10-01. Base: `e454c8255`. All labels are **independent**: this
round concerns the rung-0 from-rest NEMO instrument, not the given-entry
shipped-card twin.

## Frozen measurement

The only operator facts inspected before this file was committed are that the
round-89 launcher returned exit 139 and printed its ERR trap at launcher lines
166 and 172. The full acquisition log, target run directory, run stdout, and
compiled debug sources have not been inspected.

The round will produce:

1. A census of the exact round-89 target and whether `run.user.stdout.log`,
   `run.exit_code.txt`, and source-resolved compiled files exist.
2. The first concrete compiled-source frame from the debug run, if present,
   quoted as `file:line`, plus the exact write-only instrumentation statement
   at that line and its caller order in `stp_rk3`.
3. A control showing whether the double ERR-trap report is launcher
   bookkeeping around one MPI exit or evidence of two separate model runs.
4. If and only if a compiled line identifies an instrumentation defect, an
   additions-only repair and a fail-closed fresh optimized acquisition for 80
   entry/stage frames, with 20/20 terminal restarts required byte-identical to
   the admitted round-83 record.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R90-P1 | The debug executable ran exactly once and its nonzero MPI status propagated through nested ERR traps. | One run start, one MPI invocation/output, exit 139; no second target/run evidence. | More than one run or a pre-MPI failure: record the launcher defect and do not attribute NEMO. |
| R90-P2 | The target-local stdout survives despite the launcher stopping before its post-run copier and contains a source-resolved backtrace. | At least one `*.f90:N` frame in the target-local stdout, pinned to the compiled debug source. | No concrete line: **REFUTED**; prepare a new diagnostic acquisition, make no source repair. |
| R90-P3 | The first source-resolved frame inside the additions-only recorder names a dereference/canonicalisation statement, not ocean physics. | The line lies in the committed recorder additions and its removal/absence handling leaves the model call graph unchanged. | The line lies in unmodified NEMO physics: **REFUTED**; do not alter physics, report the instrument as only the exposing caller. |
| R90-P4 | One minimal recorder repair is enough to request the optimized frame record. | Source reading proves one invalid access and the repaired patch passes header/schema controls and additions-only checks. | Multiple ambiguous accesses or a scientific/configuration choice: stop with the exact unresolved operands or `DECISION_NEEDED`. |

## Landing and refusal bar

No legoESM model or card change is authorized. No scientific number from the
debug build is admissible. A new record launcher must use a new target, parse
self-describing headers, encode owner-off fields as ABSENT, refuse unexpected
ABSENT fields, and require rank-complete kt=1..10 entry/stage frames plus
round-83 terminal-restart byte identity. The existing sea-ice
`unmeasured_features` tuple is immutable. Any absent concrete line, failed
plant, non-additions-only patch, restart movement, configuration ambiguity, or
unresolved review finding leaves the round `STOPPED_FOR_RECORD`.

