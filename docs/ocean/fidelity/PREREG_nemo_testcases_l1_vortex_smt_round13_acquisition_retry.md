# Preregistration — VORTEX_SMT round 13 (lane round 225): execute the SMT-3 acquisition

Frozen before changing or rerunning the round-224 acquisition wrapper.  Base:
lane tip `0dee7cd1c06abcb525fea6a036c297d2eb1cf26f`.  Evidence lives under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round225/`; the NEMO record,
when the operator runs it, remains under round 224 as registered there.

## Input finding

The operator reported exit 0, but the supplied acquisition log contains only
the two dry preflights: lines 42 and 84 say `DRY RUN. Re-run with --run`, and
line 85 says `ROUND224_SMT3_PREFLIGHT_PASS`.  It contains no `makenemo`, NEMO
`STOP 0`, admission, or `ROUND224_SMT3_RECORD_READY` line.  The round-224
evidence root contains only its earlier preflight and citation artifacts; no
`oracle_vortex_smt3` record directory exists.  Therefore the acquisition was
not attempted and round 224's P2--P6 remain UNMEASURED.

The direct cause is read from the committed wrapper: an empty argument leaves
`do_run=0`, while only `--run` selects acquisition.  The operator executed the
path without that argument.  This is an interface defect in the wrapper, not a
NEMO or model result.

## Predictions and falsifiers

* **R13-P1 — one-variable wrapper repair.**  Empty arguments will execute the
  acquisition; `--preflight` will retain the old non-mutating dry mode, and
  `--run` will remain an accepted explicit spelling.  REFUTED if any target,
  source patch, deck value, admission criterion, record format, or evidence
  destination changes.
* **R13-P2 — explicit preflight.**  Running the repaired wrapper with
  `--preflight` reproduces both `PREFLIGHT_OK` lines and ends in a named
  preflight-pass line without invoking `makenemo` or `mpirun`.  REFUTED if the
  dry path mutates a build/evidence directory or lacks the named pass line.
* **R13-P3 — acquisition still needed.**  This round does not claim a NEMO
  record.  The operator-facing script is returned as `ACQUISITION_NEEDED` and
  round 225 stops for that record.  REFUTED if an admitted record already
  exists or appears during the dry preflight.
* **R13-P4 — controls.**  Shell syntax, the seven self-describing-record tests,
  the receipt citation gate, and its shifted-citation plant pass/fail as
  designed.  REFUTED by any unexpected failure or a plant that exits zero.

## No hidden choices

No physical, numerical, configuration, carried-state, target-name, record, or
gate choice changes.  This round only aligns the wrapper's default invocation
with the operator's acquisition protocol.  The existing target names may be
reused because neither target was built or run by the dry invocation.
