# Receipt — VORTEX_SMT round 31 (lane round 243): SMT-2 100-day acquisition

**Status: STOPPED_FOR_RECORD.** The deferred SMT-2 comparison cannot be scored
because its daily NEMO trajectory does not exist. The certified ten-step
record contains only the final restart and cannot supply days 1 through 100.
A committed fail-closed operator wrapper requests exactly that record;
production physics and every certified trajectory remain unchanged.

Base: `cfe1ebf81` (round 242). Preregistration commit: `865146003`.
Acquisition-tool commit: `1000baa4c`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round31_smt2_100day.md`.

## 1. Record inventory and exact request

The admitted SMT-2 record under `vortex_smt/round10` contains the ten
step-entry frames and only the step-10 restart. It contains no day-1..100
restart series. No scientific score is therefore reported this round.

The wrapper delegates to the existing common VORTEX driver's registered
`smt2vec100d` arm. That arm reuses the exact plain and instrumented binaries
named by the admitted SMT-2 binary manifest and refuses a hash mismatch. This
is not a new NEMO compilation: changing compiler output for an unchanged
source card would introduce a rounding confound. The only run-deck changes
are the already-committed 100-day continuation: `nn_itend=3000`,
`rn_Dt=2880 s`, and `nn_stock=30`.

The compiled executing restart program schedules frequency-based output at
`VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119`. Each
restart writes exactly the state fields consumed by the shared scorer — sea
surface height, U, V, temperature, and salinity — at
`VORTEX_SMT2_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`.

The new round-local target is:

`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round243/oracle_vortex_smt2/day100`

The wrapper refuses a dirty legoESM tree, an existing target, a missing common
driver, a certified-binary hash mismatch, malformed output, missing required
arrays, missing daily restarts, or any nonzero child command. Every unexpected
failure passes through a named `REFUSE` line.

## 2. Preflight and scorer preparation

The committed common driver was run without `--run`; NEMO and MPI were not
started. It reported:

> `GFORTRAN_SYNTAX_PASS .../vortex_r8_stage_terms.F90`
>
> `PREFLIGHT_OK variant smt2vec100d: instrument and deck patches apply to the shipped sources`
>
> `DRY RUN. Re-run with --run to build and acquire.`

The shared round-210 scorer now registers SMT-2 against the round-243 oracle
target and the latest current-tree SMT-2 registry from round 237. Rounds 238
through 242 contain no production model change, so this is also the current
certified short-run reference. The movie renderer imports the scorer's card
registry directly; it therefore gains SMT-2 without a second dispatch table.

The initial focused suite reports:

> `10 passed in 0.86s`

It covers the prior SMT-1 acquisition and score controls plus the new exact
card, oracle path, ladder path, acquisition variant, round-local target,
dirty-tree guard, named refusal, absence of `/usr/bin/time`, and shared movie
dispatch. Removing the SMT-2 registration makes the new direct test fail with
`KeyError: 'smt2'`; its logged exit is 1.

## 3. Frozen predictions and disposition

R31-P1 through R31-P5 are **OPEN** pending the operator acquisition and next
round's score. None is inferred from the ten-step record. R31-P6 is
**CONFIRMED**: this round changes only the shared scorer registry, a test,
acquisition tooling, preregistration, citation mappings, and this receipt. No
file under `packages/` or `src/` changes; no model, card, option, coefficient,
carried state, or certified row moves.

The existing SMT-2 long-run prediction is preserved without revision:
day-100 T RMS remains predicted within 2x of SMT-1's
`4.3321114781972461e-05 K`, and NEMO's day-100 maximum absolute U remains
predicted below SMT-1's. Both are still unmeasured for SMT-2.

## 4. Review and gates

The required separate read-only Codex review was attempted against the
complete committed round diff. It exited before reading the diff, so no SHIP
verdict is inferred. Its output is quoted verbatim:

> `WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)`
> `Reading additional input from stdin...`
> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

**Independent review unavailable in-sandbox.** The stopped-for-record verdict
does not depend on an inferred review: the required NEMO daily files are absent
and no scientific score is emitted.

The round citation gate reports **PASS: 2 citations, 0 failures, 0 unmapped**.
The cumulative default receipt reports **PASS: 274 citations, 0 failures, 0
unmapped**. Shifting the five-line restart-field citation by two lines reports
`SYMBOL-NOT-AT-LINE` and exits 1.

The final CPU-only focused suite covers the three round-241 acquisition
controls, four round-242 score controls, three round-243 controls, and the
complete citation-gate unit module:

> `27 passed in 3.63s`

## 5. OPEN — next round

Run the committed acquisition wrapper. Round 244 then verifies the common
driver's admission report and exact count of 100 daily restarts, requires the
SMT-2 kt=1..10 sanity to reproduce the round-237 registry, runs the shared
100-day scorer with `--cards smt2`, records T/u/v/ssh RMS and maxima for all
eight checkpoint days, tests the inherited day-100 T and NEMO-U predictions,
renders the 100-frame movie and four-day montage, and scores every frozen
prediction. If complete, the following round advances to SMT-3; no causal
owner is inferred from the SMT-2 curve.

**DECISION_NEEDED: NONE.**

**ACQUISITION_NEEDED:**
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-work-293491792/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smt_round31_smt2_100day/run.sh`
