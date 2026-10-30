# SI3 Ladder Round 25 (`si3_r25`) — O4 acquisition-tool receipt

Date: 2026-10-29. Base: `d09c737c5a`. Scope: acquisition tooling only; no
real NEMO build or integration run was performed. The measurement contract was
frozen before implementation in
`docs/ice/fidelity/si3_r25_o4_acquisition_preregister.md:1-132` at commit
`700a161af7`.

Verdict: **PREFLIGHT PASS; ACQUISITION NEEDED.** The default route is read-only.
The complete preflight compiled and ran the patched Prather source with bounds
checking, proved production arithmetic unchanged for every emitted function,
scored every registered boundary at strict signed-zero 0 ULP, fired the
fail-closed plants, and exercised the generated acquisition driver end to end
with a stub executable. No `makenemo` or real `mpirun` was invoked.

## Same program and observer placement

The patch no longer splits `ice_dyn_adv_pra`. Public moment visibility is
declarative; entry and advection-exit copies occur at the call boundary in
`icedyn_adv`; zapsmall and final-link copies occur after their completed
operations in `icedyn`; the step copy occurs after `ice_update_flx` in
`icestp`; and the delayed CFL scalar is copied only after the Prather loop
nests and deallocation block. The scalar tap breaks module dependency cycles.

The same-program gate strips marked blocks and compares the complete
preprocessed `icedyn_adv_pra`, `icedyn_adv`, `icedyn`, and `icestp` units with
their pristine production units. It derives the allowed nonblank insertions
from the fuzz-zero patches themselves. Unmarked-arithmetic and changed-marked-
content plants fail. Six preprocessed-source placement checks each have a
dedicated misplacement plant. The complete implementation and lifecycle are
in `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh:180-510`.

The production-flag census preserves exact symbol names instead of merging
`.constprop`, `.isra`, `.part`, or `.cold` clones. It compares the emitted
function set and every FP mnemonic multiset, including zero-FP functions, and
maps each exact symbol back to a source routine only for reporting. The four
whole-unit totals are identical: Prather 3851 = 3851, advection caller 279 =
279, dynamics 310 = 310, and step 455 = 455. An unmarked in-function FP-add
plant changes the census. The census and actual-object admission path are in
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_passivity.py:20-173`.

## Record and score contract

Each of two steps emits nine records: grid; state entry; moments entry; CFL;
advection state exit; moments exit; post-zapsmall state; post-final-link state;
and end-of-step state. State records include the full prognostic/equivalent/
aggregated state, signed residual and advection accumulators, all five
transport diagnostics, surface temperature/salinity inputs, and snow/ice mass.
The source-derived extent gate validates 104 writer calls, including the
interior-only diagnostics.

The scorer replays both sweep orders and demands strict signed-zero 0 ULP at
the advection, zapsmall, final-link, and end-of-step boundaries. Every state,
residual, advection diagnostic, transport diagnostic, and moment participates;
full-domain and nonempty wet seam/north-fold regional maxima are required.
Separate wet one-bit plants cover all four state exits, and the sweep-order,
fold-sign, and subcycle-link plants use a nontrivial deck. The schema and score
implementation are in
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_gate.py:37-500`.

## Driver, admission, and tests

The O4 overlay is exactly two steps with `nn_stock=2`, thermodynamics off,
all dynamics off except prescribed-velocity ADV2D, Prather on, UMx off, and
both prescribed velocities equal to 0.5. Every variable is checked against
its owning reference-namelist group, with misspelling and misplacement plants.

The bounds fixture compiles and runs the patched NEMO source with the exact
writer under `-fcheck=bounds`, emits 18 records, distinguishes two ranks, and
checks signed negative zero. The stub uses an admitted SAS output, emits only
the real observer list and real restart names, reports `STOP 0`, and exercises
success, retained failure, in-place admission, rerun refusal, and move-aside
recovery. Admission pins and rehashes relative source, deck, input, binary,
record, artifact, and restart paths against the producer commit.

Seven focused tests cover schema, all score plants, independent and nontrivial
decks, extent/read-set controls, fuzz-zero patching, shell syntax, and the
receipt digest ratchet
(`tests/ocean/fidelity/test_nemo_si3_o4_prather_acquisition.py:1-126`). The
complete default preflight passes and explicitly reports that no real NEMO
integration was run.

## Computed SHA-256 evidence

Every digest below is produced with `sha256sum` and is rehashed by the focused
test; placeholders are replaced only from command output before commit.

SHA256 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh` `b4ec6a36027786ac64d39d9e98f3a6b7e2cf962ad9608fa32d3320999bead144`
SHA256 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/o4_prather_writer.F90` `5feebd59538f8179db6e76d82a09f64380faa8fd2adcb7eb6323c6ab7dc1c0fa`
SHA256 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/o4_prather_tap.F90` `ff80219ebea345f57c11bdebe58b85a4ec6e572ded576919819b38dba6def33d`
SHA256 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/write_stub_records.py` `4717c99e3e8952d3d99d80eeeed8db4268d54d1a24f6ee5c48f6a461fdb32806`
SHA256 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_gate.py` `69bbdcfc7ab8183240c03309b7fb9d48e1b875104a0176a58f2066ef56ee5d8f`
SHA256 `scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_passivity.py` `f393f8c226ef87d4a3502b2cae6f2dac3784757abe0df9d3463aeeab39551815`
SHA256 `tests/ocean/fidelity/test_nemo_si3_o4_prather_acquisition.py` `7cd8aec126546b039e12f39c99216d9c2af5a695cb7c5552f2180e5686f4004b`

ACQUISITION_NEEDED: scripts/validate/ocean_fidelity/testcases/nemo_si3_o4_orca2_prather_acquisition/run.sh
