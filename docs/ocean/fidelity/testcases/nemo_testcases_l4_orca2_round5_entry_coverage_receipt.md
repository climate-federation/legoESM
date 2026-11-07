# NEMO testcase Lane 4 — ORCA2 card round 5 entry-coverage receipt

Date: 2026-09-22

Starting tip: `e96931f39599dbf12751a6e6ee06e221c5db2dfa`

Preregistration: `5c2b87dae`

Status: **STOPPED_FOR_RECORD.**  The returned acquisition is admitted and
complete for its stated surface-input purpose: both twins have 20 schema-valid
surface frames, raw A/B equality, 126 streams, and passive ordinary output.
The ocean trajectory still cannot run because NEMO's step-entry state writer
recorded only rank 0.  The gate now requires and names the ten absent rank-1
step-entry frames.  No candidate statement or kt=10 carried magnitude is
claimed.

No model package, card selector, scientific configuration, stabilizer, NEMO
source, or sea-ice registry entry changed.

## 1. Decision-52 labels and result

Twin and independent claims remain separate as required by Decision 52.

| claim or record | label | round-5 result | remaining boundary |
|---|---|---|---|
| Card initial T/S/SSH | independent | Not measured or changed. | NEMO's compiled independent initial-state construction remains its own later round. |
| Returned surface twins | given NEMO's entry | PASS: 20/20 frames in each twin, raw A/B exact, 126 streams, ordinary output passive. | None for surface coverage. |
| Decision-52 kt=1 entry | given NEMO's entry | Rank-0 T/S/u/v remain exact and SSH is the authorized differing operand.  Full-domain five-field identity is UNMEASURED. | Rank 1's NEMO step-entry state is absent. |
| ORCA2 kt=1..10 ladder | given NEMO's entry | `STOP_ENTRY_RECORD_GAP`; trajectory claim `UNMEASURED_ENTRY_RECORD_GAP`. | Ten rank-1 step-entry frames are absent. |
| First post-entry non-bit statement | given NEMO's entry | UNMEASURED. | Candidate did not run. |
| kt=10 carried magnitude | given NEMO's entry | UNMEASURED. | Candidate did not run. |
| Sea-ice selectors and registry | out of scope | Six-entry `unmeasured_features` tuple unchanged; exact-input gate remains `STOP_SELECTOR_GAP`. | No sea-ice work is authorized in this lane. |

The exact missing records are:

```text
oracle_step_entry_rank0001_kt00000001.bin
oracle_step_entry_rank0001_kt00000002.bin
oracle_step_entry_rank0001_kt00000003.bin
oracle_step_entry_rank0001_kt00000004.bin
oracle_step_entry_rank0001_kt00000005.bin
oracle_step_entry_rank0001_kt00000006.bin
oracle_step_entry_rank0001_kt00000007.bin
oracle_step_entry_rank0001_kt00000008.bin
oracle_step_entry_rank0001_kt00000009.bin
oracle_step_entry_rank0001_kt00000010.bin
```

## 2. Measurement and retraction in the tool

The preregistered unchanged gate first returned
`READY_FOR_CANDIDATE_TRAJECTORY`: surface required/present was 20/20 with no
missing name.  This confirms R5-P2 but retracts that status as a trajectory
readiness claim.  The gate had no full-domain entry coverage check.
`readiness.json` has SHA-256
`4585e3ade7abdc4e0d68a66c1ab938dc8a59f06316886df03b6cf1d183a3d43b`.

Commit `fa198d07b` puts the retraction in the tool.  It validates both MPI
slabs for all ten step entries before declaring candidate readiness.  Against
the returned root it exits 3 at
`REFUSE: STOP_ENTRY_RECORD_GAP: full-domain step-entry states are missing`,
reports entry required/present as 20/10, and gives exactly the names above.
The JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/round5_ladder.json`
(SHA-256
`23b0f8ce448ffbb6f0e701f5d762ad581bdaa019c05df01bd84ceaab8f80519a`).

The two `output.init` shards are not silently substituted.  On rank 0, the
shard differs from the recorded step entry in every field: T 233,341 cells
(max 1.1181856107919388 K), S 233,341 (0.38093247105651784 PSU), u 226,236
(0.53961070478826012 m/s), v 226,637 (0.7972046813658682 m/s), and SSH 8,794
(1.1881361175460943 m).  The gate records
`NOT_A_STEP_ENTRY_OPERAND`; therefore using rank 1's `output.init` would mix
different time/state boundaries rather than implement Decision 52.

## 3. Compiled ownership

The returned executable's step-entry writer is guarded by `lwp`, then creates
only the conventional kt-qualified filename at
`ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE/BLD/ppsrc/nemo/stprk3.f90:92-94`.
That branch executes at the top of every ORCA2 RK3 step.  In this two-rank
deck, `lwp` is true only on rank 0, so the ten conventional entry records
cover one 148x90 owned slab.  The round-4 patch made only the post-`sbc`
surface writer rank-aware; it did not change this earlier writer.

## 4. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R5-P1 | **CONFIRMED** | Frozen admission artifact is PASS with 20 surface frames, raw twin equality, 126 streams, and passive ordinary output (SHA-256 `ee11562d4309ae535d0359ba74a7dce22eb980b18e7bd196cbacc53bf10239f6`). |
| R5-P2 | **CONFIRMED, THEN READINESS CLAIM RETRACTED** | The unchanged gate returned exactly 20/20 and READY; the new entry-coverage row exposed that READY was incomplete. |
| R5-P3 | **REFUTED — STOPPED_FOR_RECORD** | Full-domain entry cannot be proved: rank 1 has no step-entry record, and its output-init shard is a different state boundary. |
| R5-P4 | **UNMEASURED — STOPPED_FOR_RECORD** | No candidate stage ran. |
| R5-P5 | **UNMEASURED — STOPPED_FOR_RECORD** | No candidate reached kt=10. |

## 5. Acquisition handoff

The existing acquisition path was extended rather than duplicated.  It now
targets the fresh fail-closed configuration
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY` and round-5 acquisition directory.  Its
WRITE-only patch preserves the conventional root filenames and writes
`oracle_step_entry_rank0001_kt########.bin` on rank 1.  The finalizer requires
20 surface frames, 20 entry frames, raw A/B equality, 136 total streams,
ordinary-output passivity, the compiled rank-aware writers, and pinned
producer hashes.

Preflight passed at
`ORCA2_ROUND1_SURFACE_PREFLIGHT_READY`.  The invalid-mode plant exited 64 at
the named usage refusal.  No `makenemo` or `mpirun` was attempted in the
sandbox.  The operator must run the acquisition path reported in the final
message with `--run`; success ends at
`ORCA2_ROUND5_FULL_ENTRY_ACQUISITION_PASS`.

## 6. Model, GYRE, choices, search, and review

No file under `packages/` changed, so the shared-model GYRE base/tip trajectory
requirement was not triggered.  No scientific or configuration choice was
made.  The fresh target and rank-qualified record names are operational
consequences of the measured compiled-writer gap and no-overwrite rule.

Pre-implementation search found the existing round-1 ladder's
`read_state_frame`, `surface_support`, and the existing acquisition/finalizer;
they were extended in place.  No parallel parser or launcher was created.

The required separate `codex exec --sandbox read-only` review was attempted on
the executable diff.  It stopped before reading the diff because its
in-process app-server could not initialize on the read-only filesystem
(`Read-only file system (os error 30)`).  Verdict: **independent review
unavailable in-sandbox**.  A second independent reviewer is not available in
this environment.

## 7. Verification

The focused ladder/citation tests pass 24/24 in 2.30 seconds (JUnit SHA-256
`50738035d258046b46bd6a69f406932ed512ba98d810c48503af5defcc0a85d5`).
Acquisition Bash syntax, patch dry-run, pinned-source preflight, and
invalid-mode control pass.

The citation gate passes from clean committed receipt `4a2ffcbb6`: one mapped
compiled-source citation, no failure, no unmapped citation, no map-audit
failure, and all 9/9 self-controls fired.  Its rigid two-line shift changes
the verdict to FAIL and exits 1.  Baseline and plant JSON SHA-256 values are
respectively
`6e3371a19638231547b6d0f1475413f118c2fe004d912259b33019ab1a1b0b01`
and
`f9c6313835823d91afe4ffdf40ec991dee7d835035c0c479c088960409512b49`.

The required single `tests/ocean/fidelity -n 12` invocation collected 1,517
tests, displayed one failure, reached 99%, then produced no output for five
minutes and was interrupted without a pytest summary.  It is **INCOMPLETE**,
not a pass.  The displayed failure is the listed pre-existing SI3 scalar-math
provenance red; its isolated rerun reproduces
`nemo_si3_scalarmath_v2_gate.GateError: A MY_SRC is not verbatim` (1 failed in
0.62 seconds).  The new ORCA2 entry-coverage tests all passed inside the full
run before the stall.

## 8. OPEN

1. The operator must run the reported acquisition with `--run`; otherwise the
   lane remains at `STOP_ENTRY_RECORD_GAP`.
2. After admission, the next round assembles both entry and surface slabs,
   applies only Decision 52's SSH bridge, proves five-field kt=1 identity, and
   executes the production ocean ladder through kt=10.
3. Once the step-1 walk names the first non-bit statement, schedule Decision
   52's separate independent initial T/S/SSH transcription round.
4. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Choices

ASKED: Decision 52's NEMO-entry SSH bridge remains the only authorized entry
replacement.  UNASKED: none.
