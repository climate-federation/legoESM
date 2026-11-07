# ORCA2 round 68 receipt — independent month record admission

Date: 2026-09-28

Base: `950b1e4c281ff3677b4613bd5987cf932e1cd6a9`

Disposition: **LANDED; the pinned NEMO month record is admitted and the
acquisition controls bind**

Claim label: **independent**.  This round admits NEMO's own from-rest record;
it does not run or score a legoESM month.  No NEMO entry field is loaded into
legoESM.

Sea ice remains out of scope.  The ORCA2 card, its six-entry
`unmeasured_features` tuple, and all selectors are unchanged.

## Answer

The operator's first attempt completed the ten-step NEMO calibration but its
post-run Python process could not import the repository-local `scripts`
package.  Adding the repository root to the launcher's `PYTHONPATH` repaired
that invocation defect.  A concurrent operator retry then completed both NEMO
arms with `STOP 0`: 10 steps in 17 seconds and 240 steps in 154 seconds.

The final gate reports `PASS_MONTH_RECORD`.  Every variable payload in all
four ten-step ocean/ice restart shards is bit-exact against the pinned record.
Both terminal ocean shards contain finite fp64 `sshn`, `un`, `vn`, `tn`, and
`sn` at `kt=240`; both ice shards are present.  The calibration and month
decks differ only in `namrun.nn_itend` and `namrun.nn_stock`, each 10 to 240.
The committed checksum ledger rechecks the admission JSON, all three plant
logs, and all eight restart shards.

The retry also exposed a vacuous round-67 control: the hidden-deck plant
mutated `namrun.rn_dt`, but this deck's executed time-step assignment is
`namdom.rn_dt`.  The plant therefore added an unscored dictionary key and
stayed green.  The repaired plant targets the real assignment and now refuses
an unauthorized deck delta.  Its fixture now carries the same namelist group
as the record.

## Frozen prediction ledger

| ID | verdict | deciding evidence |
|---|---|---|
| R68-P1 import diagnosis | **CONFIRMED** | Launcher-equivalent direct execution imports the gate after the repository root is added; the concurrent retry passed calibration admission. |
| R68-P2 retained calibration stamp | **REFUTED** | The operator's retry rebuilt both targets at `61314622ebff810e988159566e3ec1990cc6a00c`, not the preregistered branch tip `950b1e4c281ff3677b4613bd5987cf932e1cd6a9`.  Both explicit stamps are pinned and validated; no result is attributed to the wrong commit. |
| R68-P3 month absent / resume only | **REFUTED by concurrent external state** | Before this round's admission, the operator retry created and completed the month target.  This round used `--admit-existing`; it did not rerun NEMO or overwrite either target. |
| R68-P4 split producer provenance | **NOT EXERCISED** | The completed calibration and month share the same explicit producer commit.  The gate nevertheless checks the two producer operands independently, and two unit controls prove each mismatch refuses. |
| R68-P5 stop for record | **REFUTED** | The concurrent retry supplied the missing record, so this round admits it and lands the instrument repair. |

The prediction failures are retained here rather than rewritten after the
record appeared.

## Compiled source boundary

The admitted deck retains NEMO's from-rest ocean initialization at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/istate.f90:93-140` and its
out-of-scope snow/ice-mass SSH adjustment at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:440-465`.
This round changes neither statement; it verifies the copied deck and binary
by SHA-256 before admitting their output.

## Controls and validation

- acquisition admission: `PASS_MONTH_RECORD`; checksum ledger: all 12 rows
  `OK`;
- plants: one-ULP calibration payload, missing terminal shard, and executed
  hidden deck assignment all print `STATUS PLANT-FIRED`;
- producer provenance: wrong calibration and wrong month commits each refuse;
- focused gate tests: `8 passed`;
- shared-card battery: `160 passed` with nine dtype warnings; tank battery:
  `10 passed`;
- shell syntax and launcher-equivalent gate import pass.

The required separate review was attempted with `codex exec --sandbox
read-only`.  Exact terminal verdict:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore: **independent review unavailable in-sandbox**.

No `packages/` file changed.  ORCA2, GYRE, DINO, OVERFLOW, LOCK_EXCHANGE, and
generic-card trajectories therefore cannot move; their trajectory gates and
the wide ocean battery are not applicable to this instrument-only landing.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round68/` and the
admitted record remains under the round-67 `acquisition/` directory.  The
decisive files are `admit_existing.log`, `calibration_admission.json`, both
card-battery logs, and the record's `month_record_admission.json` plus
`round67_outputs.sha256`.

## Scope ledger

ASKED: repair and admit the pinned NEMO from-rest month required by Decision
52's independent ORCA2 ranking.

UNASKED and unchanged: configuration, selector, threshold, forcing,
stabiliser, carried state, model arithmetic, sea-ice implementation, and the
held QCO/RK change.

## OPEN

1. The independent legoESM month still cannot be run under the matched forcing
   protocol: its production step consumes exact `qns`, `qsr`, `emp`, `rnf`,
   `sfx`, native stresses, stress magnitude, ice concentration, and runoff
   tracer content at every step.  The admitted instrumented record supplies
   both MPI slabs only for `kt=1..10`; the new uninstrumented month supplies
   none of those streams.  Thus `460` of the `480` rank-step surface frames
   needed through `kt=240` are absent.  This is a post-hoc record census, not a
   month error result.
2. The next round must acquire a calibrated surface-input-only 240-step NEMO
   record, without the other multi-gigabyte debug streams, before ranking the
   independent month.  It must not substitute a new forcing reconstruction or
   silently reuse step 10.
3. The independent kt=1 SSH mismatch remains out of scope at the sea-ice
   `STOP_SELECTOR_GAP`; all six selectors stay frozen.
4. The source-ordered tracer QCO/RK statement remains locally exact but held
   by round 48's five OVERFLOW rows.
