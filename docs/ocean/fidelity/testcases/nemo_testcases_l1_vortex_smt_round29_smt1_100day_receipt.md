# Receipt — VORTEX_SMT round 29 (lane round 241): SMT-1 100-day acquisition

**Status: STOPPED_FOR_RECORD.** The deferred SMT-1 comparison cannot be scored
because its daily NEMO trajectory does not exist. The certified ten-step
record contains only one final restart and cannot supply days 1 through 100.
A committed fail-closed operator wrapper now requests exactly that record;
production physics and every certified trajectory remain unchanged.

Base: `9cfc4bd0b` (round 240). Preregistration commit: `321d056fc`.
Acquisition-tool commit: `a87d67d70`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round29_smt1_100day.md`.

## 1. Record inventory and exact request

The admitted SMT-1 record under `vortex_smt/round9` contains the ten step-entry
frames and only the step-10 restart. It contains no day-1..100 restart series.
No scientific score is therefore reported this round.

The wrapper delegates to the existing common VORTEX driver's registered
`smt1vec100d` arm. That arm reuses the exact plain and instrumented binaries
named by the admitted SMT-1 binary manifest and refuses a hash mismatch. This
is intentionally not a new NEMO compilation: changing compiler output for an
unchanged source card would add a rounding confound to the deferred comparison.
The only run-deck changes are the already-committed 100-day continuation:
`nn_itend=3000`, `rn_Dt=2880 s`, and `nn_stock=30`.

The compiled record producer schedules frequency-based restarts at
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:104-119`. Each
restart writes exactly the fields the common scorer consumes — sea-surface
height, U, V, temperature, and salinity — at
`VORTEX_SMT1_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/restart.f90:176-180`.

The requested target is new and round-local:

`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round241/oracle_vortex_smt1/day100`

The wrapper refuses a dirty legoESM tree, an existing target, a missing common
driver, a certified-binary hash mismatch, malformed output, missing required
arrays, missing daily restarts, or any nonzero child command. Every failure
passes through a named `REFUSE` line.

## 2. Preflight and scorer preparation

The committed common driver was run without `--run`; NEMO and MPI were not
started. It reported:

> `GFORTRAN_SYNTAX_PASS .../vortex_r8_stage_terms.F90`
>
> `PREFLIGHT_OK variant smt1vec100d: instrument and deck patches apply to the shipped sources`
>
> `DRY RUN. Re-run with --run to build and acquire.`

The existing round-210 scorer now registers `smt1` against the round-241
oracle target and the latest current-tree SMT-1 registry from round 237.
Rounds 238 through 240 contain no production model change, so this is the
current short-run reference. The movie renderer imports the scorer's card
registry directly; it therefore gains SMT-1 without a second dispatch table.

The initial focused suite reports:

> `3 passed in 0.64s`

It checks the exact card, oracle path, ladder path, acquisition variant,
round-local target, dirty-tree guard, named refusal, absence of `/usr/bin/time`,
and shared movie dispatch.

## 3. Frozen predictions and disposition

R29-P1 is **OPEN** pending the operator acquisition. R29-P2 through R29-P5 are
also **OPEN** because no daily oracle record exists; none is silently inferred
from the ten-step record. R29-P6 is **CONFIRMED**: this round changes only the
shared scorer's registry, a test, acquisition tooling, preregistration,
citation mappings, and this receipt. No file under `packages/` or `src/`
changes; no model, card, option, coefficient, carried state, or certified row
moves.

The required separate read-only Codex review was attempted against the
committed round diff. It exited before reading the diff, so no SHIP verdict is
inferred. Its output is quoted verbatim:

> `WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)`
> `Reading additional input from stdin...`
> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

**Independent review unavailable in-sandbox.** The stopped-for-record verdict
does not depend on an inferred review: the required NEMO daily files are
absent and no scientific score is emitted.

The round citation gate reports **PASS: 2 citations, 0 failures, 0 unmapped**.
The cumulative default receipt reports **PASS: 274 citations, 0 failures, 0
unmapped**. Shifting the five-line restart-field citation by two lines reports
`SYMBOL-NOT-AT-LINE` and exits 1.

The final CPU-only focused suite covers the three round-241 controls and the
complete citation-gate unit module:

> `20 passed in 7.13s`

### OPEN — next round

Run the committed acquisition wrapper. Round 242 then verifies the common
driver's admission report and exact count of 100 daily restarts, requires the
SMT-1 kt=1..10 sanity to reproduce the round-237 registry, runs the shared
100-day scorer with `--cards smt1`, records T/u/v/ssh RMS and maxima for all
eight checkpoint days, renders the 100-frame movie and four-day montage, and
scores every frozen prediction. If complete, the following round advances to
SMT-2; no causal owner is inferred from the SMT-1 curve.

**DECISION_NEEDED: NONE.**
