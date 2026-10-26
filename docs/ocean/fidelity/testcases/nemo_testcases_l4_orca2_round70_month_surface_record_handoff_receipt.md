# ORCA2 round 70 receipt — month-surface record handoff

Date: 2026-09-28

Base: `e804de64a`

Disposition: **STOPPED_FOR_RECORD; the round-69 surface record is absent and
its existing acquisition remains preflight-ready**

Claim label: **independent**.  This round produces no month error.  The future
legoESM month starts from its own ORCA2 card state and compares with NEMO's own
from-rest month.  No NEMO entry field is substituted.

Sea ice remains out of scope.  The ORCA2 card, every selector, and the
six-entry `unmeasured_features` tuple are unchanged.

## Answer

The requested target config and run directory are both absent.  Consequently
the complete inventory, ten-step calibration, and terminal-restart passivity
predictions frozen in round 69 remain **UNMEASURED**.  No scientific number
can be ranked from a missing forcing record.

The committed round-69 launcher passes from this round's clean preregistration
commit and prints `ORCA2_ROUND69_SURFACE_PREFLIGHT_READY`.  It validates every
pin and committed artifact, compiles the schema gate, and creates neither a
NEMO config nor a run in preflight mode.  Repeating step 10 or reconstructing
steps 11--240 would choose a different forcing protocol, so neither was done.

The initial orientation census also saw the record was absent before the
round-70 preregistration.  The preregistration says so explicitly and does not
mislabel that observation as a frozen prediction.  The post-commit census
reproduced it.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R70-P1 record census | **CONFIRMED** | Post-commit census: target config absent; run directory absent. |
| R70-P2 acquisition handoff | **CONFIRMED** | Clean-tree preflight printed the exact ready marker and named the intended run. |
| R70-P3 source boundary | **CONFIRMED by source/preflight** | The pinned compiled call site and additions-only source checks remain green; real output is unmeasured. |
| R70-P4 disposition | **CONFIRMED** | No record exists to admit; this round stops for the already-committed acquisition. |

Round 69's R69-P1 through R69-P3 remain **UNMEASURED**, not inferred.  Failed
predictions remain **REFUTED**, and no admission predicate may waive another.

## Compiled source boundary

The compiled admitted program completes `sbc` and then calls the WRITE-only
surface observer at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:151-152`.
The acquisition preserves that order.  This round changes no NEMO source and
does not run `makenemo` or MPI.

## Validation

- record census: both requested paths absent;
- acquisition preflight: `ORCA2_ROUND69_SURFACE_PREFLIGHT_READY`;
- focused round-69 gate tests: `9 passed`;
- default citation gate: PASS, 274 citations, zero failures and zero unmapped;
  this receipt's gate: PASS, one citation, zero failures and zero unmapped;
  the rigid-shift plant FIRES with `SYMBOL-NOT-AT-LINE`;
- wide `tests/ocean/fidelity -n 12`: collected 2,031, reached 99%, logged
  2,006 passes, five registered failures and seven skips, then emitted no
  terminal summary after all pytest workers disappeared; the stalled wrapper
  was interrupted.  The five failures were rerun serially and reproduced the
  standing signatures: SI3 `MY_SRC` provenance, round-129 stale phase-3
  certification, round-51 private trace registry, four worktree-stamp
  emitters, and missing `hires_lane_surface` case-board row.  The round-69
  focused tests were green in both runs.

No `packages/` file changed.  Model trajectories cannot move, so GYRE/DINO/
tank trajectory comparisons are not applicable to this handoff-only round.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round70/`.

The required separate `codex exec --sandbox read-only` review was attempted.
Its terminal verdict was `Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)`; therefore **independent review
unavailable in-sandbox**.  The complete attempt is retained as
`codex_review.log` in the evidence directory.

## Scope ledger

ASKED: continue the independent ORCA2 month ranking from the actual branch
tip and preserve the exact forcing protocol.

UNASKED and unchanged: configuration, selector, threshold, forcing
reconstruction, stabiliser, carried state, model arithmetic, sea ice, and the
held QCO/RK change.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round69_month_surface_acquisition/run.sh --run`.
2. The next round uses `--admit-existing`, then runs the independent 240-step
   legoESM trajectory and ranks terminal SSH/T/S/u/v errors by magnitude.
3. The independent kt=1 SSH selector gap and held QCO/RK statement remain
   unchanged.
