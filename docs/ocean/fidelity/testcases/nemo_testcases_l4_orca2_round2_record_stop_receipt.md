# NEMO testcase Lane 4 — ORCA2 card round 2 record-stop receipt

Date: 2026-09-22

Starting tip: `81dfaa3466f8ef5d5013450ae2f3f7bc760a8b43`

Preregistration: `e8197960c`

Status: **STOPPED_FOR_RECORD.**  The operator action reported in the handoff
ran the acquisition's default preflight mode, not its `--run` mode.  The pinned
root still has only the kt=1 post-`sbc` ocean-surface-input frame.  Therefore
the requested kt=1..10 legoESM trajectory, its first post-entry non-bit
statement, and its kt=10 carried magnitude remain unmeasured.  No model or
configuration file changed.

## 1. Decision-52 labels and round-2 gate inventory

Twin and independent claims are kept separate.

| claim or record | label | result | remaining boundary |
|---|---|---|---|
| Current card initial T/S/u/v | independent | Previously exact to both admitted roots at kt=1; not rerun as a trajectory in this round. | Independent T/S/SSH construction from compiled ORCA2 initialization remains owed as its own later round. |
| Current card initial SSH | independent | Differs from the pinned one-category root by 8,794/13,320 cells, maximum 0.015479333813968585 m, as recorded in round 1. | This historical independent result is not mixed into the Decision-52 twin. |
| Decision-52 kt=1 entry | given NEMO's entry | Authorized bridge: T/S/u/v are already exact and SSH must be replaced by the pinned entry operand.  No candidate state was advanced because the forcing record gate stopped first. | Five-field bridged identity and trajectory remain UNMEASURED until all ten surface frames are admitted. |
| Operator handoff log | record availability | Contains `ORCA2_ROUND1_SURFACE_PREFLIGHT_READY` and `ACQ_EXIT 0`; it contains no run-complete or admission marker. | Run the committed acquisition with its explicit `--run` argument outside the sandbox. |
| Pinned root surface inputs | given NEMO's entry | 1/10 present and schema-valid. | kt=2 through kt=10 are absent. |
| ORCA2 kt=1..10 ladder | given NEMO's entry | `STOP_RECORD_GAP`; trajectory claim is `UNMEASURED_RECORD_GAP`. | No candidate step or statement score is admissible. |
| First post-entry non-bit statement | given NEMO's entry | UNMEASURED. | Cannot be named from the V2-versus-ORCA1ICE root differential. |
| kt=10 carried magnitude | given NEMO's entry | UNMEASURED. | Requires the same exact per-step operands and a real legoESM trajectory. |
| Sea-ice selectors and registry | out of scope | Six-entry `unmeasured_features` tuple unchanged; round-20 exact-input verdict remains `STOP_SELECTOR_GAP`. | No sea-ice work is authorized in this lane. |

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

The unchanged round-1 ladder gate ran from clean committed producer
`e8197960cc29a77d1ea947ff4ba6714b7d103e67`.  It exited 2 with
`REFUSE: STOP_RECORD_GAP: exact per-step ocean surface inputs are missing`.
The JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round2/round2_ladder.json`
(SHA-256 `09be7114fb68b0f75ade6afe667923bd3283e4834415aa4252ca4e25c2f45430`).

The one-representable-step kt=1 T plant exited 1 at
`REFUSE: planted kt1 T identity control fired`; its stderr SHA-256 is
`3995104ee2be58f124d3b6422bc1c94b1d79c014f9d4f224295d0f484c33bb79`.
The acquisition preflight was rerun without launching NEMO and reproduced
`ORCA2_ROUND1_SURFACE_PREFLIGHT_READY`; log SHA-256 is
`c3d0d565fbded44b21c84a7d1b2c34693003c3856aa6cc1a35876dfb07b5572a`.

No NEMO process was launched, no forcing frame was inferred or carried
forward, and no V2-versus-ORCA1ICE number is reported as a candidate score.

## 3. Prediction ledger

| ID | verdict | evidence |
|---|---|---|
| R2-P1 | CONFIRMED | Operator marker is preflight-only; pinned root has only kt=1. |
| R2-P2 | CONFIRMED | Gate exited 2 with kt=2..10 as the exact missing set. |
| R2-P3 | UNMEASURED | Five-field Decision-52 bridged entry was not constructed because the record gate precedes candidate execution. |
| R2-P4 | CONFIRMED | No post-entry statement or kt=10 candidate magnitude was claimed. |

## 4. Model, GYRE, and choices

No file under `packages/` changed.  The shared-model GYRE base/tip trajectory
requirement was therefore not triggered.  No configuration choice was made:
Decision 52 was recorded as given, and the card's six sea-ice selectors and
registry entries were left unchanged.

## 5. Citation, tests, and independent review

The receipt citation gate passed with one mapped compiled-source citation, no
failures or unmapped citations, and all 9/9 self-controls firing.  Shifting
that citation by two lines changed the verdict to FAIL and exited 1, as
required.  The final artifacts are `citation_gate.json` and
`citation_gate_plant.json` under the round-2 evidence directory.

The focused ladder and citation tests passed 23/23 in 2.32 seconds (log
SHA-256 `ee2b2c8b3b8b662aa523ebe39fd41f411c7c5356e6d105406e4ca5468ad78017`).
The required single `tests/ocean/fidelity -n 12` invocation collected 1,516
tests, displayed one failure by 56%, reached 66%, and ended without a pytest
summary.  It is **INCOMPLETE**, not a pass; the truncated log SHA-256 is
`7ce2c79915969a2e8a63a74811246a7b628ea7e5d9f871ec372a267dca479889`.
The worker did not flush the failing node ID.  The listed pre-existing SI3
scalar-math provenance red was rerun in isolation and reproduced at `A MY_SRC
is not verbatim`; the worktree-stamp suite passed 10/10.

The required separate `codex exec --sandbox read-only` review stopped before
reading the diff because its in-process app-server could not initialize on the
read-only filesystem.  Verdict: **independent review unavailable in-sandbox**.

## 6. OPEN

1. The operator must run
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round1_surface_acquisition/run.sh --run`
   outside the sandbox.  The script creates new A/B roots, admits all ten
   frames, and refuses inherited-stream or ordinary-output movement.
2. After admission, rerun this round's frozen order: bridge only kt=1 SSH,
   prove the five entry fields exact, execute kt=1..10 with each matching
   surface frame, and stop at the first non-bit statement.
3. Once the step-1 walk names that statement, schedule Decision 52's separate
   independent initial T/S/SSH transcription round.
4. GitHub issue 1455 remains an operator-post action because no GitHub
   connector is installed in this environment.

## Mechanically cited finding

Decision 52 changes only the comparison entry operand; it does not retract the
compiled source ownership of the independent kt=1 SSH difference.  The active
ORCA2 branch sums category snow, ice, and pond mass and subtracts the resulting
global sea-level adjustment from both ocean time levels at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:442-459`.  This round does not use
that independent difference as a twin trajectory score.
