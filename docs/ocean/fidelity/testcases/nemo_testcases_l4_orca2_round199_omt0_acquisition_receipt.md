# ORCA2 round 199 — OMT-0 acquisition handoff

Date: 2026-10-09. Incoming tip: `d9f8c6950`. Preregistration commit:
`53344752a`. Acquisition implementation: `ab29262dd` plus executable-mode fix
`495d257b2`.

## Result

**STOPPED_FOR_RECORD.** Decision 103's OMT-0 NEMO deck, fail-closed launcher
and admission gate are committed and CPU-preflight clean. No NEMO process was
started: operator note AM requires the operator to run the launcher because
PMIx is blocked in this sandbox. Therefore no trajectory, first-non-bit
statement, ladder row or month error is claimed in this round.

The future record's claim label is **independent OMT-0**. It starts from the
hierarchy rung-0 corrected climatological T/S and from-rest velocity/SSH. It is
not a given-NEMO-entry score. The shipped rung-10 ORCA2 card, sea ice, all six
ice selectors and its `unmeasured_features` tuple are unchanged.
`git diff d9f8c6950..HEAD -- packages` is empty.

## Exact deck

The admitted rung-0 source deck has SHA-256
`b627f4e2d94e4619dbfa27f39b811732497f032b73be86adcc57ddca2dee0e91`.
The rendered OMT-0 deck has SHA-256
`9279c638a9fe50b551fcca8dc62dc1c19c0fc2c55bdcae7f3501a8afeb5ca8ac`.
The gate reports exactly five changed assignments and five additions:

| module | hierarchy rung 0 | OMT-0 |
|---|---|---|
| momentum advection | vector form on | advection OFF |
| lateral momentum diffusion | Laplacian on | diffusion OFF |
| tracer advection | FCT on | advection OFF |
| lateral tracer diffusion | isoneutral Laplacian on | diffusion OFF |
| bottom drag | linear drag on | free slip / drag OFF |

NEMO declares and enforces momentum advection as a one-of selection at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:162-190`; lateral momentum diffusion
at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:177-228`; tracer advection at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:586-633`; lateral tracer diffusion at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:214-268`; and bottom drag at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:371-401`. Thus each module requires
both disabling the old selected arm and enabling its compiled OFF arm. No
configuration choice was inferred.

The gate separately requires the retained OMT-0 core to stay selected: EEN,
split-explicit free surface, partial-cell HPG, constant vertical mixing,
NEMO-replacement EVD and no ice. CPP SHA-256 remains
`2e0d729f348b2377e52a6421afbb56e9dabbbc5ae57e39fbcfa1a3f5edbd8f67`;
binary SHA-256 remains
`c4907e476cf3969052b44c5c7fa966f3dac493e8cfb563f6554c8f3a27186343`.

## Acquisition contract

The launcher stages four fresh targets in this order:

1. two-step smoke, explicit restart step 1, terminal step 2;
2. twelve-step twin A, explicit restart steps 1 through 10, terminal step 12;
3. byte-identical-protocol twin B;
4. 96-step month, restart steps 10 through 90 by tens and 95, terminal step 96.

Every experiment length is divisible by the unchanged surface cadence of two.
Each explicit list is within NEMO's compiled ten-entry bound. The two-step
smoke runs before the longer targets. The gate reads each NetCDF file's own
`kt`, variable names, dtype, dimensions, shape and payload length; it does not
predict byte counts. It requires fp64 finite `sshn/un/vn/tn/sn`, both ranks,
array-identical twins, and month step 10 array-identical to both twins. The
launcher pins the deck, binary, CPP, preregistration and gate by content hash,
not by a moving producer commit.

The preflight emitted `PASS_OMT0_DECK`, `PASS_R199_OMT0_PREFLIGHT` and
`ORCA2_ROUND199_OMT0_PREFLIGHT_READY`. The extra-deck-delta, live-module and
over-capacity-list plants each refused. Record-dependent plants cannot run
before the operator record exists and remain explicitly unmeasured.

## Prediction ledger

| prediction | verdict |
|---|---|
| R199-P1 exact deck delta | **CONFIRMED** by exact text, assignment inventory and resolved retained selections. |
| R199-P2 executable smoke | **UNMEASURED** pending operator execution. |
| R199-P3 rank-complete twin identity | **REFUTED IN PART before execution**: the preregistered `200` field comparisons was an arithmetic error; 40 shards form 20 rank/step pairs × 5 fields = **100** twin comparisons. Identity itself remains UNMEASURED. The frozen preregistration is not edited. |
| R199-P4 month calibration and growth record | **UNMEASURED** pending operator execution. |
| R199-P5 record-only round | **CONFIRMED**: no package or recipe/card diff and no trajectory claim. |
| R199-P6 controls bind | **PARTIAL**: all three preflight controls fire; nine record controls are UNMEASURED pending output. |

## Validation and review

The direct round-199 suite passes 8/8; the combined focused round-199 and
citation-gate suites pass 25/25. Python compilation, shell syntax and
`git diff --check` pass. The required single `tests/ocean/fidelity -n 12`
battery collected 2,960 tests and advanced through 98%. It emitted only the
registered pre-existing worktree-stamp ratchet
`test_every_driver_that_arms_the_escape_scopes_it`, then stopped producing
output for more than 15 minutes and was bounded by one interrupt; it therefore
has no complete-suite PASS claim. The red was reproduced alone (1 failed) and
names the same 13 VORTEX drivers, none touched by this round. The launcher
first failed with `Permission denied` because its executable bit was absent;
commit `495d257b2` fixes the packaging defect, and the unchanged preflight then
passed. No NEMO run was attempted.

Independent read-only Codex review is unavailable in-sandbox. The required
separate invocation found the binary, then failed before reading the diff with
`failed to initialize in-process app-server client: Read-only file system (os
error 30)` and exit 1. No independent verdict is claimed. A GLM review tool is
not exposed in this environment, so no second review verdict is claimed.

No stabiliser, tolerance change, NEMO source modification, NEMO rebuild,
carried-state change or sea-ice change was introduced. ASKED choices: Decision
103's exact OMT-0 definition and run protocol. UNASKED choices: empty.

## OPEN

The operator runs
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round199_omt0_acquisition/run.sh --run`.
The next round admits the existing output, runs all record-dependent plants,
then builds the explicit OMT-0 legoESM card using the already-landed TSUNAMI
OFF arms, verifies geometry identity, and measures the independent and
given-entry ten-step ladders plus the independent month growth table. It names
OMT-0's first non-bit statement in compiled source order. No OMT-1 work begins
before OMT-0 reaches the bar or a measured cancelling unit.
