# NEMO testcase L2 GYRE phase 3 — round 133 daily-restart recovery receipt

Date: 2026-09-20

Incoming tip: `e9f3bb6a557581f0cc5b18a60a27757b60f49ef7`

Status: **STOPPED_FOR_RECORD — the Round-132 build is intact and its run never
started; the amended acquisition now defaults to a fail-closed, no-rebuild
resume, but the operator must run it before the daily record or four-family
attribution exists.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round133/`

## Outcome first

The Round-132 acquisition failed before NEMO executed. The operator log's
first error is:

```text
/usr/bin/time: No such file or directory
REFUSE: round132 acquisition failed at line 358 (status 127)
REFUSE: round132 acquisition failed at line 369 (status 127)
```

Line 358 was the pipeline whose first command was the absent timing wrapper;
line 369 was the enclosing subshell. The inherited ERR trap reported that one
failure twice. The formal clean-commit preflight at
`a41f1c3e745672de719095c08dd1398cc9ccd5e9` confirmed the consequence rather
than inferring it from status 127: zero restart files, zero `ocean.output`
files, and zero `STOP 0`, `NEMO_DONE`, or `RUN_DONE` markers. The original
failed stdout and start-time logs remain unchanged.

The expensive work is reusable. The target binary was built and staged with
SHA-256
`24aefbfb9f4b596c4c0c8002c17811577f3622b4c4bf34add13d770caf988eff`.
The preflight matched that independently registered digest against both the
build product and executed copy, matched all four compiled record-defining
programs byte-for-byte against the source build, matched all thirteen staged
inputs, reproduced the three authorized namelist changes, and bound the
preparation to Round-132 commit
`e9f3bb6a557581f0cc5b18a60a27757b60f49ef7`.

The existing acquisition script is therefore amended rather than replaced:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round132_daily_restarts/run.sh`

Its default mode is now `--resume`. That branch verifies the retained state,
writes distinct `run.resume.*` logs, invokes the pinned MPI launcher directly,
and cannot call `makenemo`. The old fresh-build path remains available only as
the explicit `--acquire` mode and already refuses because the registered target
exists. `/usr/bin/time` is no longer in either run path.

No NEMO run was performed by this agent. The 360-file record, monthly twins,
schema admission, and final ready marker remain **UNMEASURED**.

## Compiled-source record contract

The retained target's compiled `restart`, `stprk3`, `dynspg_ts`, and `zdftke`
files are byte-identical to the frozen
`GYRE_OMIP_L2_P3_SM_R41ADVSP` build. At the completed-step boundary NEMO swaps
the accepted state into `Nbb` and constructs the next extrapolated SSH at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:222-226`; after the
diagnostics it passes the resolved time levels to the restart writer at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/stprk3.f90:249-260`.

That compiled writer stores `sshn`, `un`, `vn`, `tn`, `sn`, both depth-mean
vectors, and `ssha` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/restart.f90:176-184`. The active
time-split branch stores all six barotropic histories at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/dynspg_ts.f90:974-980`; the active
TKE branch stores `en`, `avt_k`, `avm_k`, and `dissl` at
`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:895-901`.

Those statements establish the 16-field record contract. The recovery changes
only shell orchestration; it adds no writer, NEMO source, configuration choice,
or carried field.

## Registered failure and retained-state table

| row | Round-132 operator attempt | Round-133 committed preflight | verdict |
|---|---|---|---|
| target build | completed | binary digest rechecked | reusable |
| NEMO process | not launched; wrapper status 127 | no NEMO output found | **P0 CONFIRMED** |
| restart inventory | 0 | 0 | clean prepared run |
| completion markers | 0 | 0 | no false admission |
| binary identity | recorded but not post-failure rechecked | build and staged copy both `24aefb...eff` | exact |
| compiled record programs | build completed | 4/4 byte-identical to source build | exact |
| staged inputs | prepared | 13/13 byte-identical to target card | exact |
| namelist delta | intended three rows | exactly `nn_itend`, `nn_stock`, `nn_write` | exact |
| producer | Round-132 acquisition | `e9f3bb6a...` | exact |
| acquisition result | refused | operator run pending | **STOPPED_FOR_RECORD** |

The comparison is controlled: both columns inspect the same retained target
and run directory. The preflight writes only temporary provenance/syntax
directories and the external evidence log; it does not alter the target.

## Frozen prediction ledger

* **P0 CONFIRMED.** The missing `/usr/bin/time` wrapper prevented `mpirun` from
  starting. The predicted zero state files, output logs, and completion markers
  were all observed. The duplicate-refusal explanation is confirmed by the two
  reported script lines: the pipeline and its enclosing subshell.
* **P1 CONFIRMED.** The retained binary digest, producer commit, source-card
  manifests, cpp keys, scalar architecture, compiled record programs, staged
  inputs, namelist, and clean recovery commit all passed. No mismatch requests
  a new target.
* **P2 UNMEASURED beyond preflight.** The default branch contains no build and
  no timing wrapper, and its admission gates are committed, but the operator
  has not run NEMO. The prediction of 360 files and all bytewise twins remains
  frozen.
* **P3 CONFIRMED.** No legoESM physics, card, state, or trajectory ran or
  changed. The round stops for the missing record and names the resume script.

The Round-131 conditional prediction that daily vector/history resets remove
the largest day-240 temperature growth remains **UNMEASURED**. It is neither
confirmed nor refuted by acquisition plumbing.

## Fail-closed recovery and controls

The default path first reruns every retained-build check. It refuses any
partial NEMO output, any prior recovery log or stamp, a dirty worktree, an
unexpected producer, an altered binary, compiled branch, source/card manifest,
staged input, cpp key, architecture, or namelist. The registered MPI launcher
and symbol inspector are required by absolute path before the script reaches
the process call.

The run pipeline is now the condition of an explicit shell `if`. A nonzero MPI
or logging status therefore reaches one named `REFUSE` rather than being
intercepted twice by ERR inheritance. Original `run.user.*` failure logs are
read-only inputs to recovery; only new `run.resume.*` logs are permitted.

The binary-identity plant substituted a false expected digest without changing
either binary. It exited exactly 1 and printed:

```text
REFUSE: STATUS PLANT-FIRED: retained-binary-identity
```

The unknown-mode control exited 64 with a named refusal. After the operator
run, the existing complete-inventory control and missing-boundary and
required-variable plants run again before audit admission. `RUN_DONE` and
`ROUND133_DAILY_RESTART_RECORD_READY` are the final writes, after the exact
file census, twelve monthly byte twins, independent old-daily day-30 twin,
16-field gate, plants, and recovery-commit stamp.

Evidence SHA-256 values:

| artifact | SHA-256 |
|---|---|
| `resume_preflight_final.log` | `9ac866005c36937ab0dccd589b52df5747e3e6eeb7548b66c615e47208edcfbe` |
| `binary_identity_plant_final.log` | `793fc7a597639fa66ab9275258a2a5f649e8f313dd14dfbf9f1cfb45bba2c0ee` |
| `named_refuse_control.log` | `326ff8df9cbf01fb11cf58c162dbf889a0717503f861f1c307bd187f62cf4d9c` |

## Trajectory and cross-card disposition

No candidate implementation and no model trajectory were produced, so the
moved-row registry is empty. The certified GYRE ladder, month, day-240, and
day-360 before arms remain unchanged and were not rerun. The four requested
daily-nudging arms are all UNMEASURED. DINO, LOCK_EXCHANGE, OVERFLOW, ORCA2,
and the tanks cannot move because this round changes only an unexecuted shell
recovery path; ORCA2 remains UNMEASURED-WITH-SPEC for this GYRE experiment.

| required campaign row | before | after | disposition |
|---|---|---|---|
| GYRE `kt=1..10` ladder | immutable current arm | not rerun | no model diff |
| GYRE day 30 | immutable Round-130 free arm | not rerun | no model diff |
| GYRE day 240 T3D RMS | `1.64467402331753935e-2 K` | not rerun | attribution still blocked |
| GYRE day 360 | immutable Round-130 free arm | not rerun | attribution still blocked |
| DINO / tanks / LOCK / OVERFLOW | current certified arms | not rerun | statement does not execute |
| ORCA2 | UNMEASURED-WITH-SPEC | UNMEASURED-WITH-SPEC | untouched |

No Rule-12 or Decision-43/45 landing verdict is available or needed: zero
physics rows moved and nothing landed.

## Independent adversarial review

The required separate read-only Codex pass was invoked against the Round-133
implementation commits, the retained target, the preflight and plant evidence, and the
compiled source. Its prompt explicitly tried to refute the no-run diagnosis,
no-rebuild default, retained identity checks, non-vacuous plant, single-refusal
control flow, preservation of failed evidence, final-marker ordering, and
clean-commit stamp. It returned status 1 before reading the diff. Its complete
verbatim output was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

Thus, **independent review unavailable in-sandbox**. The retained log is
`phase3/round133/codex_review_initial.log`; it issued no `DO NOT SHIP` verdict.
No GLM reviewer interface is available in this environment, so the repository's
dual-review rule has NO GATE in this round.

## Verification

All Python commands used the required CPU/fp64 environment.

* `bash -n` accepted the amended acquisition script.
* The clean-commit recovery preflight returned zero with all four syntax
  proofs, the empty-failed-run row, the retained-build row, and the exact ready
  marker quoted below.
* The retained-binary plant returned 1 with `STATUS PLANT-FIRED`; the
  unknown-mode control and explicit fresh-build control each returned 64 with
  one named refusal. The fresh-build control stopped at the existing-target
  guard before any `makenemo` call.
* The focused Round-131 admission-gate and receipt-citation suites reported
  exactly `23 passed in 2.52s`.
* The unplanted citation run found five citations, zero unmapped citations,
  zero failures, an empty whole-map audit, all nine internal controls firing,
  and `status: PASS`.
* Shifting the completed-step citation by two lines returned exit 1 with
  `status: FAIL` and `SYMBOL-NOT-AT-LINE`. The retained artifacts are
  `phase3/round133/citation_gate.json` and
  `phase3/round133/citation_gate_shifted_plant.json`.

No full model or all-tree pytest battery ran: this round changes one operator
acquisition shell path and documentation, not model code, a card, a Python
gate, or NEMO source. The real-input preflight, planted identity failure,
fresh-build refusal, focused gate suite, and citation controls cover the
committed scope.

The committed acquisition preflight already reports:

```text
SYNTAX_PROOF_PASS restart.f90
SYNTAX_PROOF_PASS stprk3.f90
SYNTAX_PROOF_PASS dynspg_ts.f90
SYNTAX_PROOF_PASS zdftke.f90
FAILED_RUN_PASS restarts=0 ocean_output=0 completion_markers=0
RETAINED_BUILD_PASS binary_sha256=24aefbfb9f4b596c4c0c8002c17811577f3622b4c4bf34add13d770caf988eff compiled_files=4 staged_inputs=13 producer=e9f3bb6a557581f0cc5b18a60a27757b60f49ef7
ROUND133_DAILY_RESTART_RESUME_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round132/oracle_daily_restarts
```

Pre-implementation search found the existing retained-record recovery patterns
in the Round-98 and Round-117 acquisition scripts. This round extends the
Round-132 script rather than creating a second acquisition or re-deriving its
record gates.

No configuration or carried-state choice was made. The three namelist changes
are exactly the values already ordered and preregistered in Round 132. The
unasked-choice list is empty.

## OPEN — round 134

1. The operator runs the amended acquisition script with no arguments. It must
   print `ROUND133_DAILY_RESTART_RECORD_READY`; any `REFUSE` leaves the record
   unadmitted. Do not rerun `makenemo` or delete/replace the retained target.
2. Round 134 reruns the Round-131 daily-record gate against
   `phase3/round132/oracle_daily_restarts` from its own clean stamped commit.
   No legoESM reset arm starts unless it prints `STATUS ADMITTED`.
3. Run all four independent 360-day daily-reset families: T/S; u/v plus all six
   carried barotropic histories; SSH plus `ssha` and both SSH histories; and
   TKE state. Score every field at days 30/60/90/120/180/240/300/360 against
   the immutable Round-130 free arm and rank measured day-240 T3D removal.
4. Halve the winning family's reset cadence only if a separately certified
   NEMO record exists at every three-step boundary. Daily files do not supply
   those states.
5. Re-score `r99_wclock` alone only after all four owner-family arms and their
   attribution table are complete.

Production remains at the incoming model tip. `DECISION_NEEDED` is `NONE`.
