# Preregistration: GYRE kt=2 adjudication and shared stage-ZAD transcription, round 47

Date: 2026-09-11. Frozen at legoESM `715c9865008e` before any round-47
admission, kt=2 score, candidate execution, or production edit. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round47/`. CPU/fp64 only.

## 1. Returned-record admission

Compare every inherited `oracle_*.bin` from round 41 with round 46 through the
round-21 schema-aware consumed-field admission. Extend that existing gate only
for the two inherited self-describing magics: `NEMO_L2_RKTS3_1` (16 header
integers) and `NEMO_L2_ADVSP_1` (17). Every differing element must be in the
two-cell halo or in the already registered, run-resolved undefined `zFw` slot.
The prediction is **ADMITTED**. CONFIRM requires zero owned differences in a
defined field, byte-identical restart and mesh, and a planted owned-field bit
change exiting nonzero. REFUTE on one owned/defined difference; name its
writer statement and stop before scoring or production changes.

The four doubles behind the reported 32 differing bytes in each widened kt=1
record will be decoded by field and Fortran `(i,j,k)` cell with both values.
The compiled round-46 writer opens `rkstage3_terms` before HPG and snapshots
after HPG/VOR/ADV (`R46KT2/BLD/ppsrc/nemo/stprk3_stg.f90:413-430,451-506`);
the split writer brackets `dyn_adv` (`stprk3_stg.f90:473-480`). Therefore an
owned difference in an accumulator field refutes WRITE-only status. If the
admission passes, the acquisition script will replace its raw kt=1 refusal by
this schema-aware gate; records with no undefined slots retain exact consumed
projections, and a separate raw-identity list is printed rather than inferred.

## 2. kt=2 score with the round-44 candidate

Reconstruct the rejected round-44 change from `5976cfba94d3` in the current
tree: the shared WS-RK3 stage program passes its already-built velocity-form
`ww` to the common momentum RHS. The compiled oracle constructs `ww` from
`uu/vv(Kmm)` (`stprk3_stg.f90:329-335`) and calls vector-form `dyn_adv` with
`Kmm` operands (`:473-476`); `dyn_zad` consumes `ww` and `uu/vv(Kmm)`
(`dynzad.f90:105-137`).

First require source replays of WZV, KEG, and ZAD to have zero unequal cells at
all six recorded stages. Then run the model's own given-input routes and the
kt=1-end to kt=2 trajectory stage by stage. Frozen prediction from round 46:
with the explicit candidate present, the first kt=2 difference is **stage 1,
after ZAD**. CONFIRM requires every preceding stage-1 boundary exact and at
least one wet U/V cell unequal after ZAD. REFUTE on an earlier difference or
an exact after-ZAD boundary. Any REFUTED result is recorded and walked in
compiled statement order; no post-hoc owner is promoted.

## 3. Landing rule

Only the one shared WS-RK3 stage program may change, with no new knob or
per-card arm. Eligibility requires zero unequal given-input cells on every
executing measured card: GYRE kt=1 and kt=2; LOCK_EXCHANGE/OVERFLOW round-33
ZDF and round-25 external-mode obligations. DINO's leap-frog path is a
separate branch and is reported as shared-statement risk. ORCA2 remains
UNMEASURED unless an equivalent per-stage record exists; the missing spec is
Kbb/Kmm/Kaa velocities, velocity-form `ww`, resolved `wsd`, live/reference
face thicknesses, masks/metrics, and named pre/post-ZAD accumulators at every
executed stage.

The kt=1..10 Rule-12 matrix is scored before and after on every registered
card. Every worsened row is registered; no first-over-bar may move earlier.
If GYRE kt=2 clears, kt=3 ownership requires a new source-order given-input
discriminator before attribution.

## Controls and decisions

Header, truncation, calibration, given-input, trajectory, legacy-twin,
consumed-field, and commit-stamp plants must each move their named row and exit
nonzero. The receipt citation gate and stamp ratchet must pass.

| status | item | disposition |
|---|---|---|
| ASKED | adjudicate the returned twin, score kt=2, and land only a Rule-12-eligible shared transcription | this round |
| UNASKED | configuration, physics threshold, scheme selection, or public API choice | none |
