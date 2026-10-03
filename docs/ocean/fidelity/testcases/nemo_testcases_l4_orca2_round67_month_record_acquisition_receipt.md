# ORCA2 round 67 receipt — independent month record acquisition

Date: 2026-09-28

Base: `9427601555a2b903a2246f744479497aad15c091`

Disposition: **STOPPED_FOR_RECORD; the pinned NEMO month is absent and its
fail-closed acquisition is preflight-ready**

Claim label: **independent**.  This round produces no month error number.
legoESM's future month starts from its own card state and will be compared only
with NEMO's own from-rest month.  No NEMO entry field is substituted.

Sea ice remains out of scope.  The ORCA2 card, its six-entry
`unmeasured_features` tuple, and every selector are unchanged.

## Answer

No admitted 30-day record matches the pinned `VARIANT_ORACLE_ORCA1ICE` deck.
The two existing 240-step records use the earlier ice namelist; their deck
manifest differs from the pin at `namelist_ice_cfg`.  Reusing either would mix
two external ocean-forcing protocols and violate Decision 52.

The committed acquisition stages two fresh runs with the admitted scalar-math
executable: a ten-step calibration and a 240-step month.  Calibration must be
bit-identical to every variable payload in all four pinned ocean/ice restart
shards before the month can run.  The month deck is copied from the pin and may
change only `nn_itend` and `nn_stock`, both from 10 to 240.  Its admission then
requires NEMO `STOP 0`, both ocean and both ice terminal shards, `kt=240`, and
finite float64 SSH/u/v/T/S payloads on both ranks.

The launcher is preflight-clean and names a new target.  It was not executed:
the standing PMIx sandbox refusal applies, so the operator must run it.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R67-P1 no admitted month | **CONFIRMED** | The record census found no 240-step root with the pinned deck-manifest hash; both old months carry the pre-pin ice namelist. |
| R67-P2 executable calibration | **UNMEASURED** | The operator must produce the fresh ten-step calibration; any unequal restart payload refuses before the month. |
| R67-P3 controlled run-length change | **CONFIRMED for preflight; UNMEASURED for output** | The source deck rewrite changes only `nn_itend` and `nn_stock`; admission rechecks the staged manifests and parsed assignments. |
| R67-P4 complete month | **UNMEASURED** | No fresh 240-step target exists yet. |
| R67-P5 disposition | **CONFIRMED** | No model statement landed; the round stops for the named record. |

The first preflight attempt refused before target creation because the newly
written gate mistyped two characters in the pinned deck-manifest digest.  The
digest was corrected to the record's measured SHA-256 and preflight then
passed.  This was an instrument defect, not a scientific prediction or NEMO
measurement.

## Compiled source and record boundary

The pinned compiled program reads T/S, initializes u/v at rest, and copies the
before level at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140`.
It then applies the existing snow/ice-mass SSH adjustment at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465`.
The acquisition copies the pinned deck, so neither statement nor the
out-of-scope ice identity is changed.

NetCDF supplies the self-describing variable names, dtypes, dimensions, and
payloads.  The admission gate does not predict file byte counts or header
tuples.  Its calibration comparison reuses the phase-1 raw-payload comparator,
including fill values and signed zero.

## Controls and validation

All available controls passed:

- shell syntax and Python byte-compilation passed;
- acquisition preflight printed `NAMELIST_PREFLIGHT_PASS` and
  `ORCA2_ROUND67_MONTH_PREFLIGHT_READY` for the fresh target;
- focused admission tests: `6 passed`; the one-ULP calibration,
  missing-terminal-shard, hidden-deck-delta, and non-finite-payload controls all
  refuse;
- shared-card battery: `160 passed` with nine dtype warnings; tank battery:
  `10 passed`;
- the single permitted `tests/ocean/fidelity -n 12` invocation collected
  `2,020` tests, reached 99%, reproduced the same five standing failures as
  rounds 65-66, then produced no output for five minutes and was interrupted.
  The five IDs were rerun serially and retained their registered signatures:
  round-51 private trace registry, SI3 `MY_SRC` provenance, four worktree-stamp
  emitters, missing `hires_lane_surface` case-board row, and round-129 stale
  phase-3 certification.  All six round-67 tests passed in the wide run.

The required separate review was attempted with `codex exec --sandbox
read-only`.  Exact terminal verdict:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore: **independent review unavailable in-sandbox**.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round67/`.
The decisive files are `preflight.log`, `focused.log`, both card-battery logs,
`ocean_fidelity.log`, `known_reds_isolated.log`, and `codex_review.log`.

## Scope ledger

ASKED: obtain the missing pinned NEMO from-rest month needed for the independent
ORCA2 magnitude ranking.

UNASKED and unchanged: configuration, selector, threshold, forcing,
stabiliser, carried state, model arithmetic, sea-ice implementation, and the
held shared tracer QCO/RK candidate.

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round67_month_acquisition/run.sh --run`.
2. The next round admits the resulting calibration and month, advances legoESM
   from its own initial state under the matched forcing protocol, and ranks
   SSH/T/S/u/v month errors by magnitude before any further bit-row walk.
3. The independent kt=1 SSH difference remains explicitly out of scope at the
   sea-ice `STOP_SELECTOR_GAP`; do not change the six selectors.
4. The source-ordered tracer QCO/RK statement remains locally exact but held
   by round 48's five OVERFLOW rows.
5. Round 20's slow-forcing/barotropic walk remains open.
