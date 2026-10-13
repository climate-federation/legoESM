# ORCA2 round 271 — written-order EEN transfer refutation

Date: 2026-10-13

Lane base: `5ab4e9eebce9aabe88787b00d9a361d6a6bb158e`

Labels: **independent** and **given NEMO's entry**, kept separate below

Verdict: **HELD**. No model, card, deck, carried-state, initial-state, or
sea-ice change lands.

## 1. Frozen question and provenance

The frozen preregistration is
`PREREG_nemo_testcases_l4_orca2_round271.md` at `8d51998d6`. It names the two
grid-ladder round-7 statements to test before any ORCA2 measurement:

1. `c1b4d0d69`, NEMO's written arithmetic order for the EEN vorticity trend;
2. `9852cfd5d`, NEMO's interpolated F-point ratio at RK3 stages 2 and 3.

They were imported as a private arm at `fe0ccc474` and `616149e15`, with the
lane's existing shared call wiring retained and the G1-only script/test
changes omitted. The two ORCA2/OMT selectors and a direct known-answer
control were added at `e587ebf74`. The previously certified passive matched
program-point observer was restored at `cde7953ca`. After measurement, every
private package/card/observer change was retired by `5d8799a1f5` through
`ae6402d632`. The final command
`git diff 5ab4e9eeb -- packages tests/ocean/unit/test_orca2_g1_transfer.py`
is empty.

The statement implementations and selector semantics match the two pinned
commits; the conflict resolutions only unioned this lane's newer call wiring,
and no arithmetic was reimplemented. R271-P1 is therefore **CONFIRMED for the
authorized two-statement unit**. The omitted G1-only harness files are not
part of ORCA2's executable.

During validation the grid-ladder lane advanced to landed tip `1ce119a43a`.
The measured statement commits are ancestors of that tip, so R271-P6's branch
provenance condition is **CONFIRMED after measurement**. That provenance does
not override the frozen ORCA2 closure falsifier below. The G1 receipt also
explicitly calls the ORCA2 relative-vorticity addition unmeasured; G1's
one-level planetary-only proof is not an ORCA2 closure proof.

The six existing ORCA2 G1 selectors from round 270, every deck and entry
operand, the six sea-ice selectors, and the ORCA2 `unmeasured_features` tuple
remained unchanged. No configuration choice was made.

## 2. Compiled source order

The compiled ORCA2 record build executes `np_CRV`, not G1's planetary-only
`np_COR`. It builds the Coriolis-plus-relative-circulation quantity at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:762-770`, constructs the
live U/V transports at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:784-785`, forms the four
EEN triads and trends at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:790-802`, and adds those
trends to the stage accumulators at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:797-802`.

Accordingly, a failure of G1's written-order selector to close this row leaves
the combined `np_CRV` statement as the first source-ordered non-bit boundary;
it does **not** identify any one leaf operand as owner.

## 3. Matched stage-2 measurement

The measurement used the admitted OMT-4 frames and fold-transport operand
record. It ran on CPU in fp64 at clean private-arm commit
`cde7953ca1dd5456c709072d0f3d185941680915`. The restored observer is the
round-268/266 passive program point: candidate self-replay is checked before
the recorded after-VOR row is read.

| label | before max | candidate max | delta | candidate RMS | unequal | argmax |
|---|---:|---:|---:|---:|---:|---|
| independent | 1.2863307536456779e-8 | 1.2863307536453391e-8 | -3.3881317890172014e-21 | 1.2191816847794473e-10 | 249,102 | `[19,33,3]` |
| given NEMO's entry | 1.2863307536456779e-8 | 1.2863307536453391e-8 | -3.3881317890172014e-21 | 1.2191816847794473e-10 | 249,102 | `[19,33,3]` |

The campaign floor is `2e-10 m s-2`; the candidate maximum remains 64.3
floor units above it. Both labels agree exactly on the candidate boundary.
The accumulator self-replay has maximum and RMS `0.0`; its 11,676 bit
differences are signed-zero-only and are registered rather than called bit
identity. The matched stage-1 entry remains below the floor under both labels:
eta `0`, T `3.552713678800501e-15`, S `7.105427357601002e-15`, and u/v
`1.3877787807814457e-17` maximum.

Therefore R271-P2 is **REFUTED**: the two imported statements produce only a
`3.39e-21 m s-2` reduction and do not close the ORCA2 interior residual.
R271-P3 and R271-P4 are **NOT REACHED**: no Decision-96 trajectory census or
shared-card landing claim is made after the immediate source boundary fails.
The complete held fold unit is not re-scored because its required upstream
EEN closure did not occur.

Evidence hashes:

- classification JSON: `42d83b8d74549b5101fea29b260a33dee137fc862b8cc6f01ea6dc7ed847e49a`;
- independent measurement: `1d991907f37d444542b814f7615952562f65870da1741b3705f77154cc647b6d`;
- given-entry measurement: `c29532ee78afa28233a067e6a9a5d9623ece8a82e796dfd2d97b296d9dde6317`.

## 4. Mechanical gate and controls

`nemo_testcase_l4_orca2_round271_een_transfer_gate.py` requires both labels,
the admitted passive-measurement status, a matched entry below the floor,
candidate self-replay below the floor, no movement away, the still-open
residual, and the stable interior argmax. It returns
`PASS_R271_HELD_WRITTEN_ORDER_NOT_CLOSED`.

Its planted false closure sets the independent maximum to zero and exits 2:
`REFUSE: independent: residual unexpectedly closed; this HELD gate is stale`.
The gate's two focused tests pass. The private selector's direct
known-answer control passed 5/5 before retirement. Thus R271-P5 is
**CONFIRMED**.

## 5. Test battery, review, and citations

The required `tests/ocean/fidelity -n 12` battery was run once. It collected
3,468 tests, reached 98%, and was terminated after a late worker stopped
making progress at
`test_nemo_testcase_l1_vortex_round204_stage1_split.py::test_the_split_hook_at_its_default_changes_nothing`.
That exact test passes serially in isolation: **1 passed in 149.79 s**. The
only emitted failure is the registered pre-existing SI3 scalar-math
provenance red; it reproduces alone as
`GateError: A MY_SRC is not verbatim`. No round-271 test failed.

The separate `codex exec --sandbox read-only` review was attempted against
the complete round diff. It failed before reading the diff because the
in-process app-server could not initialize on the read-only filesystem.
Verdict: **independent review unavailable in-sandbox**. Retained log SHA-256:
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The receipt citation gate passes all four compiled-source citations with zero
unmapped or failed anchors. Shifting the `np_CRV` span by two lines makes the
gate fail as required. The default cumulative audit also passes: 274
citations, zero unmapped citations, zero failed anchors, and zero map-audit
failures. No cumulative citation needed re-anchoring because the final round
contains no model-file edit.

## OPEN

At the same passive, matched stage-2 entry, replay the executing combined
Coriolis-plus-relative-vorticity statement before substituting a leaf. Split
in compiled order: `ff_f`; relative circulation; `r1_e1e2f`; live
`e3f_0vor*(1+r3f)`; `zwx/zwy`; the four triad sums; final metric and
accumulator addition. The replay must reproduce legoESM's own completed VOR
row first, including the registered signed-zero census. The first leaf whose
source-exact substitution closes the interior `[19,33,3]` residual is then
re-tested under Decision 96 with the landed grid-ladder statements. The old
live-F-thickness owner prediction is superseded by this source-ordered split;
it is not a finding.

No acquisition or user decision is needed.
