# NEMO testcase Lane 4 — ORCA2 card round 4 surface-coverage receipt

Date: 2026-09-22

Starting tip: `ca36b12db0986da145a4dd654e66d6cf2a3b9247`

Preregistration: `047263966`

Status: **STOPPED_FOR_RECORD.**  The operator's returned twins are complete and
admitted for what their patch wrote: ten raw-identical rank-0 surface frames
and 116 streams per twin.  They cannot support the ORCA2 card's kt=1..10
ocean ladder because each frame covers only the rank-0 148×90 owned slab of
the card's 148×180 ocean domain.  The gate now requires both MPI slabs and
names the ten rank-1 frames as the exact record gap.  No candidate statement
or kt=10 magnitude is claimed.

No model package, card selector, scientific configuration, stabilizer, NEMO
source, or sea-ice registry entry changed.

## 1. Decision-52 labels and gate inventory

Twin and independent claims remain separate as required by Decision 52.

| claim or record | label | round-4 result | remaining boundary |
|---|---|---|---|
| Card initial T/S/SSH | independent | Not measured or changed in this round. | Transcribe and bit-gate NEMO's compiled ORCA2 initial-state construction in its own round after the step-1 walk names its first statement. |
| Decision-52 kt=1 entry | given NEMO's entry | T/S/u/v remain historically exact.  The authorized SSH bridge was not executed because complete surface operands fail first. | Five-field entry identity remains UNMEASURED. |
| Returned ten-step twins | given NEMO's entry | A and B each completed kt=10, emitted ten notices and ten schema-valid conventional surface frames, and have identical 116-name inventories and raw-identical surface bytes. | Each conventional frame is rank 0 only. |
| ORCA2 kt=1..10 ladder | given NEMO's entry | `STOP_RECORD_GAP`; trajectory claim `UNMEASURED_RECORD_GAP`. | Ten rank-1 surface frames are absent. |
| First post-entry non-bit statement | given NEMO's entry | UNMEASURED. | The card cannot consume a half-domain surface operand. |
| kt=10 carried magnitude | given NEMO's entry | UNMEASURED. | No candidate trajectory ran. |
| Sea-ice selectors and registry | out of scope | Six-entry `unmeasured_features` tuple unchanged; exact-input gate remains `STOP_SELECTOR_GAP`. | No sea-ice work is authorized in this lane. |

The exact missing records are:

```text
oracle_ocean_surface_input_rank0001_kt00000001.bin
oracle_ocean_surface_input_rank0001_kt00000002.bin
oracle_ocean_surface_input_rank0001_kt00000003.bin
oracle_ocean_surface_input_rank0001_kt00000004.bin
oracle_ocean_surface_input_rank0001_kt00000005.bin
oracle_ocean_surface_input_rank0001_kt00000006.bin
oracle_ocean_surface_input_rank0001_kt00000007.bin
oracle_ocean_surface_input_rank0001_kt00000008.bin
oracle_ocean_surface_input_rank0001_kt00000009.bin
oracle_ocean_surface_input_rank0001_kt00000010.bin
```

## 2. Note-K stream reconciliation

The exact filename inventory (`stream_inventory.txt`, SHA-256
`0b92c7f37ae4bcbeadcd069d46aaa1280e292c2e46e4c5db2f5b1bc8c25b5533`)
found 116 baseline streams and 116 streams in each returned target.  A/B had
no filename difference.  Relative to the baseline, each target omitted nine
files and added the nine conventional surface frames kt=2..10.  Seven
omissions were already registered.  The other two were
`oracle_si3_bulk_operands.bin` and `oracle_tke_walk_kt00000002.bin`.

Those two streams do not belong to the surface patch.  Their writers occur in
older phase-specific instrumentation, and neither writer is compiled into the
returned R3SURFACE target.  This refutes the preregistered conditional-branch
explanation.  Commit `98ef44138` corrected the inherited count to 107, the
total to 116, and added both exact absences to the fail-closed registry.
Finalization then admitted the existing twins with
`twin_surface_frames_raw_exact: true` and the PASS marker.  The finalizer log
is `finalize.log`, SHA-256
`397dc0d2f47aa9096f44a503cadb28bf66ce1d04391be9e08704ec5f0da47b8e`.

This admits only what was recorded.  It is not a trajectory admission.

## 3. Coverage finding and retraction

The first post-admission ladder run reported ten required and ten present
frames.  That readiness result is **RETRACTED**: the gate counted filenames
but did not check global-domain coverage.  The record payload header is
`nx=94, ny=152, halo=2`, hence an owned shape of 148×90.  The card's SSH and
T shapes are respectively 148×180 and 148×180×30.  The frozen shape evidence
is `coverage_shapes.txt`, SHA-256
`a3b420203a353bb348eae155e9f25b9a6e485b1820e506b0b9155d748859af1e`.

Commit `84e23f785` puts the retraction in the tool.  The ladder now requires 20
frames: the ten existing rank-0 conventional names and ten explicit rank-1
names.  Against the returned record it exits 2 at
`REFUSE: STOP_RECORD_GAP: exact per-step ocean surface inputs are missing`,
reports required 20, present 10, and gives precisely the list in section 1.
The JSON is `round4_ladder.json`, SHA-256
`f74ea936858c471e14f1688269c2375252da6a1248cf2c87c77507035430a3d1`.
The one-representable-value kt=1 T plant exits 1 at
`REFUSE: planted kt1 T identity control fired`.

## 4. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R4-P1 | **CONFIRMED** | Both targets completed kt=10 with ten schema-valid frames, ten notices, identical inventories, and raw-identical frame bytes. |
| R4-P2 | **REFUTED** | Exactly two additional baseline streams were absent, but they were absent phase-specific instrumentation, not writers disabled by a live branch of this deck. |
| R4-P3 | **REFUTED** | Count 116 was correct and finalization passed, but a count-only edit was insufficient: the explicit absent registry and inherited-stream count also required correction. |
| R4-P4 | **REFUTED — STOPPED_FOR_RECORD** | Admission exposed rank-0-only frames.  The gate cannot score an ordered statement until the ten rank-1 operands exist. |

## 5. Acquisition handoff

The acquisition remains at the mandated round-1 path but now targets the new
fail-closed name `ORCA2_ORCA1ICE_OMIP_L4_R4FULLSURFACE` and round-4 evidence
directory.  Its source patch writes the conventional root filename on rank 0
and a rank-qualified filename on rank 1 for every kt=1..10.  The finalizer
requires 20 schema-valid frames, raw A/B equality for all 20, 126 total
streams, the compiled rank-aware writer, and all pinned producer hashes.

Explicit preflight passed.  The invalid-mode plant exited 64 at the named
usage refusal.  No `mpirun` was attempted in the sandbox.  The operator must
run the acquisition path reported in the final message with `--run`.

## 6. Model, GYRE, choices, and review

No file under `packages/` changed, so the shared-model GYRE base/tip trajectory
requirement was not triggered.  No scientific or configuration choice was
made.  The fresh target and rank-qualified record names are operational
consequences of the measured patch shortfall and the no-overwrite rule.

The required separate `codex exec --sandbox read-only` review was attempted on
the complete final executable diff.  It stopped before reading the diff
because the in-process app-server could not initialize on the read-only
filesystem (`Read-only file system (os error 30)`).  Verdict:
**independent review unavailable in-sandbox**.  The review log SHA-256 is
`4154359fb80a5b516b50cb7861ca5a5321eac33a58608714b68c46acba6d0603`.

## 7. Verification

The citation gate passed from clean committed receipt `6c72b3468`: one mapped
compiled-source citation, no failure, no unmapped citation, no map-audit
failure, and all 9/9 self-controls fired.  Its rigid two-line shift changed the
verdict to FAIL and exited 1.  The baseline and plant JSON SHA-256 values are
respectively
`a5a69d663568bae98033e0fe8bbfaed415075dab9663d4eba8ea80ea21973de1`
and
`a3721566aef5767c0c9472593e76117eed0a69b37c3f052d64be4e345d49be9d`.

The focused ladder/citation tests passed 23/23 in 2.25 seconds (log SHA-256
`6603f2bdb0d16ceeb5c954fa9179c623a67dd2cd19b6e236ffd2c75a150fe47a`).
The required single `tests/ocean/fidelity -n 12` invocation collected 1,516
tests, displayed one failure, reached 99%, then made no progress for a bounded
five-minute interval and was interrupted without a pytest summary.  It is
**INCOMPLETE**, not a pass (log SHA-256
`0996b127da3946c4dd8d99c5c6a2af1d80bdc8102da0b44f3f88ac21239a6f52`).
The displayed failure is the listed pre-existing SI3 scalar-math provenance
red.  Its isolated rerun reproduced `A MY_SRC is not verbatim` (log SHA-256
`dcb5f5afbeb0233320c9f60cb63012e73dcc09a14f12fecd62e7d3aeb4751931`).

## 8. OPEN

1. The operator must run the reported acquisition with `--run`.  It must end
   at `ORCA2_ROUND1_SURFACE_ACQUISITION_PASS`; otherwise this lane remains at
   the named refusal.
2. After admission, assemble both owned longitude slabs, perform only Decision
   52's authorized kt=1 SSH bridge, prove all five entry fields bit-identical,
   and execute the ocean kt=1..10 ladder to its first non-bit statement.
3. Once that statement is named, schedule the separate independent initial
   T/S/SSH construction and bit-exact gate.
4. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Mechanically cited finding

The returned writer executes only on NEMO's root process: it returns when
`lwp` is false, then opens one conventional kt-qualified file with
`STATUS='NEW'` at
`ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE/BLD/ppsrc/nemo/stprk3.f90:396-400`.
Therefore each returned frame is rank 0's local slab; it cannot be treated as
the full two-rank ORCA2 surface operand.
