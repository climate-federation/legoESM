# ORCA2 round 273 — source-complete EEN leaf record request

Date: 2026-10-13

Lane base: `26ec7877f`

Labels: **independent** and **given NEMO's entry**, always kept separate

Verdict: **STOPPED_FOR_RECORD**. The candidate-side fused replay produced no
admissible output, so no EEN leaf is named. A syntax-proved, additions-only,
rank-complete NEMO recorder is ready for the operator. No model, card, deck,
carried-state, initial-state, or sea-ice change lands.

## 1. Frozen question

The committed preregistration is
`PREREG_nemo_testcases_l4_orca2_round273.md` at `3ef3908fa`. Round 272's
passive boundary remains the input fact under each label: the live stage-2
after-VOR residual has maximum `1.2863307536456779e-8 m/s2` at `[19,33,3]`,
while its offline replay missed the candidate live value by
`2.5331686726199837e-9 m/s2`, above the `2e-10 m/s2` floor. Those are prior
measurements, not round-273 remeasurements.

The compiled OMT-4 build executes the combined planetary-plus-relative
`np_CRV` statement at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770`, constructs the
live transports at `:784-785`, and evaluates the four triads plus accumulator
at `:790-802`. The G1 result certifies only the planetary/order half; this
round therefore froze the relative-circulation leaf order without presuming
which leaf owns the residual.

## 2. Candidate replay disposition

The round-272 extraction materialised the relative circulation and its
consumer in separate JITs. Round 273 wrote a single-JIT fused replay and
temporarily restored the same private, non-constructible stage instrument
needed to obtain the passive completed operands. The first incomplete restore
failed loudly before producing a number. The complete attempt entered XLA
constant folding, emitted only the slow-operation diagnostic, and terminated
without JSON or NPZ output. It therefore supplies **no citable measurement**.
Its log SHA-256 is
`ed29853957f51351115d948aaf63b265e17f5ffa9ac2423b8a967e54d99ed4f6`.

All three temporary instrument commits were reverted. Mechanically,
`git diff --exit-code 26ec7877f..HEAD -- packages` returns zero. P1 is
**NOT MEASURED** because no replay output exists; P2 is
**UNMEASURED_COMPILER_STALL** and activates its preregistered acquisition
fallback; P3 is **NOT REACHED**. This is not a numerical refutation of any
relative-circulation leaf.

## 3. Rank-complete fallback record

The committed acquisition is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round273_een_leaf_acquisition/run.sh`.
It copies the admitted OMT-4 deck, applies an additions-only patch to the
record build, performs a two-step smoke run, then records both MPI ranks at
`kt=1`, stage 2, from the same `np_CRV` program point. Its self-describing
header records dimensions, rank, step, stage, and named payload lengths. The
payload follows the compiled statement order: entry accumulators; live F
thickness reciprocal; relative circulation; live U/V thicknesses and
transports; the four triads; final U/V trend; completed accumulators; metrics;
and masks.

The launcher is fail-closed and content-pinned. It verifies the source build,
deck, patch, checker, source hashes, resolved EEN branch, dead mask switch,
patch application with zero fuzz, Fortran syntax, both rank files, exact
header names and payload lengths, finiteness on owned support, twin record
identity, and kt=10 restart identity against the admitted round-230 source
run. It contains five plants covering layout, duplicate names, payload size,
missing rank, and additions-only restart drift.

CPU preflight passes with `SYNTAX_PROOF_PASS dynvor.f90` and
`ORCA2_ROUND273_EEN_PREFLIGHT_READY`; SHA-256
`4be5c4b11a7462c67fa98bfa82507f6402c0f8adb4f8b7dbde15dbe682cdeb5c`.
The layout plant refuses with `STATUS PLANT-FIRED layout`; SHA-256
`e53ef406dceeeb31b98b62f2a669b3bf16aa4e67741931f91ce265a158cfd163`.
P4 is therefore partly confirmed for the acquisition path; record-dependent
plants remain unmeasured until the operator run exists. P5 is confirmed: the
package diff is empty and no card, deck, selector, or ice field changed.

## 4. Mechanical validation

The focused acquisition and replay controls pass **9/9** in `0.74 s`;
evidence SHA-256
`325685250b65ec5120e77742341f5240ae5c3f43eced19bf19a2693908ccccd0`.
The required single `tests/ocean/fidelity -n 12` battery reached 91% with five
failure markers, then all pytest processes disappeared without printing node
IDs or a summary. The run was not repeated in parallel. Its incomplete log is
preserved with SHA-256
`261d2072c587f95014b7f1e0737ccbf6825b937808931966e6559859a87cdbcb`;
because this round changes no package code and its nine new focused controls
are green, the incomplete battery is registered but is not claimed green.

The receipt citation gate passes the compiled anchors above with no failure or
unmapped citation; its shifted-line plant fires. The default cumulative
citation audit also passes. A separate `codex exec --sandbox read-only`
review was attempted on the complete diff, but its in-process app-server
could not initialise on the read-only filesystem. Verdict: **independent
review unavailable in-sandbox**. The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

The operator runs the acquisition script above. The next round first uses
`--admit-existing`, then compares the production candidate operands with the
rank-complete NEMO stream under **independent** and **given NEMO's entry**
labels separately. It must reproduce the candidate completed VOR row below
the floor before reading any leaf score. Once calibrated, it walks the
relative-circulation stencil, metric, live F thickness, transports, four
triads, and final accumulator in the compiled order. If the run or admission
refuses, the refusal is fixed in the instrument; no physics landing follows
from this round.

No user decision is requested.
