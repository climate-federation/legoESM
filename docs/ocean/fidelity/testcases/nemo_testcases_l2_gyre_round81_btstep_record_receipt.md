# NEMO testcases L2 GYRE round 81 — kt=2 external-step record receipt

Date: 2026-09-13

Starting commit: `5e0674fd31c7`

Writable clone: `/tmp/autopilot-work-Vw6v7EFu/repo`

Evidence root: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round81`

## Verdict

**STOPPED_FOR_RECORD.** No production numerical statement, configuration, or
carried state changed in this round.  The operator-completed Round 80
comparison refutes Round 79's required kt=2 U/V movement: all 954 registered
parent/candidate rows are unchanged.  The absolute-history candidate therefore
remains HOLD under Decision 37.  A clean, preregistered direct U recheck still
places the first observed non-bit row at external substep 2 `un_e`, but the
admitted record does not expose the intervening V, SSH, six histories, or
substep-1 pressure/forcing/update/swap statements.  Round 81 supplies a new,
fail-closed acquisition for those missing source-ordered operands and stops
before inferring an unseen owner.

No configuration choice was made.  Decision 37 remains the user's explicit
YES to the paired six-absolute-history representation and upstream
window-boundary reconciliation owner; the refuted raw-history-only arm stays
held.  NEMO source/build/run, the year harness, reconciliation gate,
freshwater pair, #1484 guard, held manifests, defaults, selectors, thresholds,
masks, and scientific configuration were not changed by this agent.

## Preregistration and inherited review disposition

Round 81 was frozen before its direct measurement at commit
`c4cf1a0d0624a247750025b4fc5215ea4a0d72be`.  It registered the exact Round 80
retraction, the R77 compiled statement order, the direct U prediction and
falsifiers, the complete 37-array acquisition layout, every control, and the
source-order prediction for the future record.  That prediction has not been
measured and remains frozen.

The operator-acquired Round 80 review said verbatim:

> - Prediction 2 is REFUTED: all 954 parent/candidate rows are unchanged, including kt=2 U/V. The receipt improperly relabels expected movement as kt3.
> - Correct the receipt to preserve REFUTED. Also certify V/SSH and all six persisted histories; the current gate covers U only (`v_live_walk` is withheld).
> - Pair enforcement and format-3 migration showed no separate code defect.
>
> HOLD

Every finding is dispositioned literally.  A dated correction in the Round 79
receipt withdraws the kt3 relabel; REFUTED is retained here and in the
preregistration; the candidate remains HOLD; and the new record covers V, SSH,
all six histories, and the compiled continuation sequence.  No separate defect
was invented for pair enforcement or format-3 migration.

## Round 80 exact comparison

The exact parent commit was `0078f9cc851c921176afd8e30660373129321712`
and the candidate was `add5cbd555b99ebe32ce4e46ae69538f67951883`.
The shared cellwise comparison passed with all 954 rows unchanged.  Both
preregistered movement rows changed by exactly zero cells: kt=2 U remained
`2.7478404751243857e-12` from NEMO and kt=2 V remained
`3.305560306813421e-12`.  This makes the Round 79 movement prediction
**REFUTED**, not deferred or relabeled.

Evidence:

- `round80_parent_after_comparison.json`, SHA-256
  `d986a66915fa20e885160dbb2ff000ef1b96042ec3b788d2397c91eb4414058a`.
- `round80_kt2_prediction_disposition.json`, SHA-256
  `9701f305a837ea816079fd9ce435a4d94b4ffa29b04617998008611770c43148`.

## Direct measurement after preregistration

At the clean preregistration commit, the existing Round 79 gate again admitted
the R77 U record and the live production-JIT walk.  Substep-1 coefficients,
`un_e`, `ub_e`, `ubb_e`, and `ua_e` are bit-exact.  The first U non-bit row is
substep-2 `un_e`: 580/580 wet faces differ, maximum absolute difference
`3.032539284029834e-09`, reference maximum magnitude
`4.352625078750904e-04`.  This exactly confirms the registered recheck; it does
not identify which preceding statement first diverges.

The record-dependent one-ULP `ubb_e` plant exited 1 and printed exactly
`ROUND79 HISTORY PLANT FIRED`.  The warnings from divisions at masked points
come from the inherited content-operand gate and did not change its finite wet
rows or verdict.

Evidence:

- `round81_u_history_recheck.json`, SHA-256
  `b857e07c903ab44c0a4dd6f05a3cb81bb2aa0903e62ccf333bd94e681f1375fc`.
- `round81_u_history_ulp_plant.json`, SHA-256
  `4bf35584fe18e9c1a603a9ca1308f2970b36bd6ccadb20d1a2d9878dc2b3e286`.

## Missing record and acquisition contract

The admitted R77 stream directly certifies only U midpoint coefficients and
the U current/b/bb/mid arrays.  It cannot certify V, SSH, their four remaining
absolute histories, or any later substep-1 operand.  Guessing the next owner
would violate the source-order walk, so this round adds a WRITE-only source
card for a new target, `GYRE_OMIP_L2_P3_SM_R81BTSTEP`, and a reader that admits
one new file, `oracle_bt_step_operands_kt00000002.bin`.

For every substep 1 through 50 over the owned 32-by-22 extent, the stream
contains 37 fp64 arrays in the frozen order: entry/b/bb/mid U,V,SSH; midpoint
U/V depths; metric transports; SSH forcing, divergence, and continuity
result; backward SSH midpoint; U/V pressure gradient; U/V Coriolis result;
drag coefficients and inverse depths; combined U/V trend; slow U/V forcing;
post-exchange U/V result; and post-swap U/V/SSH current.  Three staggered masks,
the external timestep, full/owned dimensions, field count, fp64 width, kt, and
cycle count precede the rows.  Writer and reader independently resolve the
exact size as 10,439,164 bytes.

The reader requires the exact header, size, finite values, consecutive 1--50
substeps, EOF, NOW time-level registration, producer commit and SHA stamp.  It
bit-replays all three forward midpoints, continuity, backward SSH interpolation,
explicit drag accumulation, vector velocity update, and the three swap
identities.  It also requires every U midpoint input and result to be
bit-identical to the admitted R77 U stream.  Stamp, header, truncation,
replay-ULP, swap-ULP, and R77-identity-ULP plants are record dependent and run
after acquisition.  The shared twin admission additionally runs its consumed
inherited-field and unexpected-inventory plants.

The operator script clones the source card file by file into the new target,
applies only the additive patch, compiles a new binary, runs the existing
10-step GYRE configuration, requires final restart and mesh twins, admits all
inherited artifacts, and permits only the new named record.  It never replaces
canonical NEMO source and refuses an existing target or run directory.
The no-argument wrapper handed to the operator has SHA-256
`f4dcef7a66ccfd4210e28691af6cf9c34826253a3ab901eadc89e99ef0d911c6`;
its unexpected-argument plant exited 64 and printed exactly
`REFUSE: run.sh takes no arguments` without invoking the acquisition.

Local preprocessing plus `gfortran -fsyntax-only` passed.  The committed clean
preflight printed exactly
`ROUND81_BTSTEP_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/round77/oracle_uamid_kt2`.
The first preflight failure was an instrument-only checker error: its grep
count matched both entry and post-swap SSH occurrences.  The correction pins
the terminal post-swap write; no scientific prediction or source-card field
changed.  The corrected layout plant exited 69 and printed
`REFUSE: layout plant removed the post-swap SSH field`.  The resolved-card
plant exited 65 and printed
`REFUSE: source run lacks resolved row: in iterations nn_e *= *50`.

Evidence SHA-256 values:

- corrected preflight:
  `ee7fcc74ff9e57a2873348b83aff025dd060f47f1bf8373a65c0b78df06ab6df`;
- layout plant:
  `6637881bb3a45ea9ac4135edafad97292f19d54d20a9f1f046bb6f6632ecf066`;
- resolved-card plant:
  `7237742ae29f0bfddd2168a57824d5a6a727f25630fdbf75370f18df6602ef92`.

## Separate Round 81 Codex review

After the acquisition, parser, direct tests, historical correction, and
citation mappings were committed cleanly, the mandated adversarial command was
run with `codex exec --sandbox read-only -C /tmp/autopilot-work-Vw6v7EFu/repo`.
It was asked to refute the 37-field layout, writer/reader order and byte count,
compiled arithmetic replays, plants, admission, no-choice claim, and Rule-12
table.  It exited 1 before model startup, so it produced no `SHIP`, `HOLD`, or
`DO NOT SHIP` verdict.  Its terminal line is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

This is retained as unavailable review, not treated as approval.  It did not
mark the acquisition DO NOT SHIP.  The earlier operator-acquired HOLD verdict
still governs the numerical candidate.  Review evidence SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## Rule-12 table

No production numerical statement changes in Round 81.  The candidate already
present in the shared implementation remains HOLD; the table therefore
preserves measured prior arms and specifies the new record without claiming a
new model result.

| lane | Round 81 disposition |
|---|---|
| GYRE source-order kt=2 | **MEASURED U / STOPPED FOR RECORD.** Substep-1 U current/b/bb/mid stays bit-exact; first observed U non-bit is substep-2 current. V, SSH, all six histories together, and pressure/forcing/update/swap await the new record. |
| GYRE kt=1..10 | **PRESERVED EXACT PARENT COMPARISON.** All 954 parent/candidate rows are unchanged; kt=2 U/V movement is REFUTED; no AT-BAR row left the bar and first-over-bar did not move earlier. |
| GYRE days 1..30 | **PRESERVED RECORDED-BEFORE RESULT ONLY.** Round 79's candidate artifacts remain prior controlled evidence; no new candidate, toggle, or rerun was introduced. |
| LOCK_EXCHANGE-zco | **PRESERVED.** The same split-explicit shared program retains Round 79's 50-row result; no Round 81 model statement changes. |
| OVERFLOW-zps | **STATEMENT NOT EXECUTED.** Its boxcar card does not enter the cross-window AB3/AM4 continuation branch; the new acquisition instrument is a GYRE-only source card. |
| DINO | **SHARED-STATEMENT RISK.** DINO's leapfrog card carries its own histories, but the shared restart surface and severe per-row cancellation forbid a neutrality inference. No DINO claim is made. |
| ORCA2 | **UNMEASURED-WITH-SPEC.** Independently align NEMO and legoESM T/S/U/V/SSH and all six absolute histories on native staggered masks; require elementwise fp64 equality and normalized L-infinity through kt=1..10; reject any AT-BAR loss, earlier first-over-bar, or wet-face history mismatch. |

## Gates and focused checks

- Exact CPU/fp64 direct U gate: PASS with the source-order result above.
- Direct U history-ULP plant: exited 1.
- Source-card preprocessing and `gfortran -fsyntax-only`: PASS.
- Clean acquisition preflight: PASS.
- Static layout and resolved-card plants: exited 69 and 65 respectively.
- New record parser, admitted U parser, and receipt citation-gate tests:
  **23 passed in 1.72 s**.  Evidence SHA-256:
  `debdb9258ed95512c3af2539df84473ca7fd413cfc50423547b1dd31ec5de4d4`.
- Receipt citation gate: **PASS**, 11/11 compiled citations mapped, zero
  citation or full-map failures.  Shifting the forcing-import citation by two
  lines exited 1 with `SYMBOL-NOT-AT-LINE`.
- Python compilation, `bash -n`, and `git diff --check`: PASS.
- The record-dependent admission and ULP plants are deliberately UNMEASURED
  until the operator produces the record; the acquisition refuses success
  unless every one exits nonzero.
- The known unrelated RK3_WS/MXL3 red test was not in these focused suites.

## Choices, uncertainty, and retained falsifier

Choices made: none.  The new target and record names are provenance, not a
scientific selector.  The source configuration, 50 external substeps, 288 s
external timestep, vector-form branch, masks, bounds, fields, order, fp64
policy, and comparison bars are resolved or inherited.  There is no mechanical
gate for the no-unasked-choice claim; this paragraph is the required
honour-system disclosure.

UNVERIFIED: the new record itself; V/SSH/six-history identity; the first
source-order mismatch between substep-1 midpoint and substep-2 U current; and
the frozen prediction that `slow_u` or `slow_v` is first.  That prediction is
falsified by any earlier history, continuity, pressure, Coriolis/drag, or
update mismatch, by both slow-forcing rows being bit-exact, or by a different
first row.  A failure must remain REFUTED in the next receipt.

## OPEN — exact handoff to round 82

1. The operator runs
   `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round81/run.sh` with no
   arguments.  Success must end with `ROUND81_BTSTEP_READY` and name the new
   run directory.  Do not run `makenemo` or `mpirun` inside the autonomous
   round.
2. Round 82 first reads and admits the record, all six record-dependent plants,
   the twin admission and both admission plants.  Verify the producer commit,
   byte inventory, final restart, mesh, and inherited records before reading a
   scientific value.
3. Compare the production-JIT V/SSH/six histories and then every substep-1 row
   in the frozen compiled order.  Preserve the `slow_u`/`slow_v` prediction as
   CONFIRMED or REFUTED without reordering post hoc.
4. The first non-bit input owns the next compiled-producer walk.  Only when all
   inputs are exact and the first shared result is not may a separately
   preregistered production change be considered.  Do not use a scratch toggle
   or resurrect the raw-history-only arm.
5. Keep the Decision-37 candidate HOLD until its paired owner produces the
   required registered movement, and keep DINO risk and ORCA2
   UNMEASURED-with-spec explicit.

## Compiled source citations

The running GYRE branch imports the slow two-dimensional forcing and removes
the barotropic Coriolis contribution before the external loop:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:288-324`.

It initializes continuation current values separately from the six absolute
histories:
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:339-378`.

The executing loop forms the U/V absolute-history midpoints at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:481-505`, the SSH
midpoint at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:512-518`, and the
metric transports and continuity result at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:549-583`.

The backward SSH interpolation and surface pressure gradient are at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:627-642`; the
Coriolis and explicit-drag accumulation is at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:644-666`; and the
executing vector-form update is at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:678-702`.

Boundary exchange precedes the absolute U/V/SSH history swap at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/dynspg_ts.f90:744-795`.

The compiled two-dimensional caller constructs the dynamics RHS through the
vertical average at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:127-214`, adds drag and
wind, forms SSH forcing, calls the split-explicit program, and deallocates its
operands at
`GYRE_OMIP_L2_P3_SM_R77UAMID5/BLD/ppsrc/nemo/stp2d.f90:220-311`.
