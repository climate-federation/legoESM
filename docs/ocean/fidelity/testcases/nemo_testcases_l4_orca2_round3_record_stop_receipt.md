# NEMO testcase Lane 4 — ORCA2 card round 3 record-stop receipt

Date: 2026-09-22

Starting tip: `ae049c43d58273374273d6304eccf5f9e65d0401`

Preregistration: `b62355d4c`

Status: **STOPPED_FOR_RECORD.**  The returned operator action was again the
acquisition's preflight, not its run mode.  The pinned one-category ORCA1ICE
root contains only the kt=1 post-`sbc` ocean-surface-input frame.  Therefore
the Decision-52 twin trajectory, first post-entry non-bit statement, and kt=10
carried magnitude remain unmeasured.  No model package, NEMO source, card
selector, or sea-ice registry entry changed.

## 1. Decision-52 labels and record inventory

Twin and independent claims remain separate.

| claim or record | label | result | remaining boundary |
|---|---|---|---|
| Card initial T/S/u/v | independent | Historical kt=1 identity remains recorded; not advanced in this round. | Independent compiled ORCA2 T/S/SSH construction remains a later dedicated round. |
| Card initial SSH | independent | Historical difference to the pinned root is retained only as an independent initialization result. | It is not mixed into the Decision-52 twin. |
| Decision-52 kt=1 entry | given NEMO's entry | T/S/u/v are historically exact; SSH replacement is authorized but was not constructed because the ten-frame record gate stops first. | Five-field bridged identity remains UNMEASURED. |
| Returned operator log | record availability | Contains only `ORCA2_ROUND1_SURFACE_PREFLIGHT_READY`; no run-complete or admission marker. | A real NEMO acquisition is still required. |
| Pinned root surface inputs | given NEMO's entry | 1/10 present and schema-valid. | kt=2 through kt=10 are absent. |
| ORCA2 kt=1..10 ladder | given NEMO's entry | `STOP_RECORD_GAP`; trajectory claim is `UNMEASURED_RECORD_GAP`. | No candidate step or statement score is admissible. |
| First post-entry non-bit statement | given NEMO's entry | UNMEASURED. | It cannot be named from the V2-versus-ORCA1ICE root differential. |
| kt=10 carried magnitude | given NEMO's entry | UNMEASURED. | Requires matching per-step operands and a real legoESM trajectory. |
| Sea-ice selectors and registry | out of scope | Six-entry `unmeasured_features` tuple unchanged; exact-input verdict remains `STOP_SELECTOR_GAP`. | No sea-ice work is authorized in this lane. |

The exact missing records are:

```text
oracle_ocean_surface_input_kt00000002.bin
oracle_ocean_surface_input_kt00000003.bin
oracle_ocean_surface_input_kt00000004.bin
oracle_ocean_surface_input_kt00000005.bin
oracle_ocean_surface_input_kt00000006.bin
oracle_ocean_surface_input_kt00000007.bin
oracle_ocean_surface_input_kt00000008.bin
oracle_ocean_surface_input_kt00000009.bin
oracle_ocean_surface_input_kt00000010.bin
```

## 2. Measurement and controls

The unchanged ladder gate ran from clean committed producer `b62355d4c`.  It
exited 2 with `REFUSE: STOP_RECORD_GAP: exact per-step ocean surface inputs
are missing`.  The JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round3/round3_ladder.json`
(SHA-256 `6252f5f4411bec522b673f9f62b8e9670f1faf347d3bdd44e7edb40320f0a639`).
It records exactly one present frame and the nine names above.

The one-representable-step kt=1 T plant exited 1 at
`REFUSE: planted kt1 T identity control fired`; stderr SHA-256 is
`3995104ee2be58f124d3b6422bc1c94b1d79c014f9d4f224295d0f484c33bb79`.
No candidate step ran, no forcing frame was inferred or carried forward, and
no root differential is reported as a candidate trajectory.

## 3. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R3-P1 | CONFIRMED | Returned log is preflight-only; pinned root has only kt=1. |
| R3-P2 | CONFIRMED | Gate exited 2 with kt=2..10 as the exact missing set. |
| R3-P3 | CONFIRMED | No candidate step, post-entry statement, or kt=10 candidate magnitude was claimed. |
| R3-P4 | **REFUTED** | The default-mode repair was necessary but not sufficient: the prior target configuration name was already occupied, so the unchanged fail-closed run path would refuse it.  The final handoff also uses a fresh round-3 target and evidence directory. |

## 4. Acquisition handoff

The existing acquisition was reused rather than reimplemented.  Commits
`81690d81a` and `09d18d5ff` make its no-argument mode select the already
existing `--run` path and move that path to the fresh target
`ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE` with the empty round-3 acquisition
directory.  Explicit `--preflight-only`, `--run`, and `--finalize` modes remain.

The final script passes Bash syntax, explicit preflight, all pinned hashes and
patch dry-run checks.  Its invalid-mode control exits 64 with the named usage
refusal.  The new target is absent and the evidence directory is empty.  The
preflight log SHA-256 is
`c3d0d565fbded44b21c84a7d1b2c34693003c3856aa6cc1a35876dfb07b5572a`.
Per the operator note, no `mpirun` was attempted in the sandbox.

## 5. Model, GYRE, choices, and review

No file under `packages/` changed, so the shared-model GYRE base/tip
trajectory requirement was not triggered.  No scientific or configuration
choice was made.  The only operational choices were explicitly preregistered
default execution and, after R3-P4 was refuted, the fresh target/evidence names
required by the fail-closed no-overwrite rule.

The required separate `codex exec --sandbox read-only` review was attempted on
the final executable diff, including ephemeral mode.  It stopped before
reading the diff because its in-process app-server could not initialize on the
read-only filesystem.  Verdict: **independent review unavailable in-sandbox**.

## 6. Verification

The receipt citation gate passed from clean committed receipt `bab584a48`:
one mapped compiled-source citation, no failures or unmapped citations, no
map-audit failures, and all 9/9 self-controls fired.  Shifting that citation by
two lines changed the verdict to FAIL and exited 1.  The baseline and plant
JSON SHA-256 values are respectively
`423f32e7f4f7a3fde8e89d897266bdd51494300ab6f675cbecef5ee8fdc4ac83`
and `31db33537c5bbc725453e6ad763c8eb8d8f78916e72832df5edc42bec1859c58`.

The focused ladder/citation tests passed 23/23 in 2.27 seconds (log SHA-256
`5494a6880eb477c37f81ff18d1fabafbbfb603097edc3795fa96a941fc192426`).
The required single `tests/ocean/fidelity -n 12` invocation collected 1,516
tests, displayed one failure, reached 99%, then produced no progress for five
minutes and was interrupted without a pytest summary.  It is **INCOMPLETE**,
not a pass (log SHA-256
`0fbdb255e3cc676eaab835a3f54ecb2449ab9e450c92244821a446898855b3de`).
The displayed failure is the listed pre-existing SI3 scalar-math provenance
red; isolated rerun reproduced `A MY_SRC is not verbatim` (log SHA-256
`e38c53d46e9ec3673bbc2490caaec7ac1f4a920649c13286b029358da9dca9b9`).
The worktree-stamp ratchet passed 10/10 in isolation (log SHA-256
`d5e2bc126787b57240603e8afcfc39de23c1626759bde90425ccb40a3811e44d`).

## 7. OPEN

1. The operator must run the acquisition path reported in the final message
   with no argument.  Its default now performs the run; preflight requires an
   explicit flag.
2. After the acquisition reports its PASS marker, rerun the frozen order:
   bridge only kt=1 SSH, prove all five entry fields bit-identical, execute
   kt=1..10 with the matching surface frames, and stop at the first non-bit
   statement.
3. Once the step-1 walk names that statement, schedule Decision 52's separate
   independent initial T/S/SSH transcription round.
4. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Mechanically cited finding

Decision 52 changes only the twin's entry operand; it does not retract the
compiled-source ownership of the independent kt=1 SSH difference.  The active
ORCA2 branch sums category snow, ice, and pond mass and subtracts the global
sea-level adjustment from both ocean time levels at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:442-459`.  This independent result
is not used as a twin trajectory score.
