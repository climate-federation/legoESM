# Preregistration — ORCA2 round 142 growth-record re-request

Date: 2026-10-04. Base: `25d7f732e`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round142`.
Every trajectory value remains **independent**: hierarchy rung 0 starts from
its own climatological T/S, zero velocity, and zero sea surface. Given-entry
ladder populations remain separate.

Round 141 committed the additions-only step-30..36 recorder and launcher. The
operator's attempt stopped before source staging, compilation, or NEMO because
`mktemp` could not allocate the launcher's syntax-proof directory on the full
root filesystem. Round 142 changes no acquisition artifact. It first reruns
the committed launcher's preflight after the disk cleanup, then requests that
same launcher for operator execution.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R142-P1 | The unchanged round-141 launcher now passes preflight because `/tmp` has more than 4 GiB free. | `--preflight-only` exits 0 and prints `ORCA2_ROUND141_GROWTH_PREFLIGHT_READY`. | Keep the failure verbatim; fix only a real content/toolchain defect, or stop if space is still insufficient. |
| R142-P2 | No round-141 target build or run directory exists because the failed attempt stopped at the first `mktemp`, before the creation checks and build. | The target configuration, calibration run, and 36-step run are all absent. | If any exists, do not rebuild; inspect whether it is complete and use `--admit-existing` only if the launcher's own predicates permit it. |
| R142-P3 | The source-layout, hidden-deck-delta, and producer-content controls still fire without editing the launcher. | Each plant exits nonzero and prints `STATUS PLANT-FIRED`. | Reject the re-request and repair the broken control in a new committed acquisition artifact. |
| R142-P4 | The science boundary remains unmeasured until the operator produces and the checker admits both rank streams for steps 30..36. | No NEMO-versus-legoESM growth number is reported in this round. | Any science number before `PASS_R141_GROWTH_RECORD` is rejected. |
| R142-P5 | The round changes documentation only; model, card, deck, acquisition content, carried state, stabilizers, sea ice, and `unmeasured_features` remain unchanged. | `packages/` and the round-141 acquisition directory have empty diffs. | Stop at the first non-documentation change. |

## Round bar

The unchanged launcher must pass preflight and all three controls. The operator
then runs it under the existing round-141 acquisition contract. No NEMO run is
attempted in the sandbox, no existing target is overwritten, and no physical
or configuration choice is made.
