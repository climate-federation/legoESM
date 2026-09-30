# NEMO testcase L4 ORCA2 round 61 — stage-3 raw-Kaa boundary

Date: 2026-09-28  
Incoming tip: `ef132df0e5f99313974c723aa78ef0f155036746`  
Frozen preregistration: `6059909bfd` plus base-stamp correction `b463975800`  
Final production disposition: **HELD; private observer only, no physics lands**  
Evidence: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round61/`

## Outcome

Round 61 closes round 60's sole missing production boundary.  These OVERFLOW
trajectory numbers are **independent**: both arms start from the same hash-bound
legoESM kt=3 state; NEMO frames are the oracle and are not substituted.

The held source-ordered UP3 arm enters stage-3 `dyn_zdf` **TOWARD** NEMO, then
leaves its `raw_kaa` boundary **MIXED**: 145 / 16,900 active U values move,
116 toward, 22 away, and 7 neutral.  L2 error moves from
`0.37929962498661096` to `0.37929962499334946`; L-infinity remains
`0.04480131363770535`; the maximum arm move is
`1.275775971193785e-09` m/s.

Frozen R61-P3 is therefore **REFUTED**: the barotropic replacement is not the
first stage-3 direction change.  Frozen R61-P4 is **CONFIRMED**: MIXED is not
the registered strict TOWARD-to-AWAY reversal, so no compensating owner is
named.  The earlier stage-2 QCO assignment remains the first direction change
in the whole walk.  The source-ordered UP3 and QCO/RK candidates remain held
and absent from the final tree.

The final package change is only a private, false-by-default WRITE-only
observer.  GYRE's base and final ten-step trajectories are byte-identical over
70 / 70 rows and all 210 residual arrays; all 30 daily snapshots through day
30 are byte-identical.  Configuration, carried state, and sea ice are
unchanged.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R61-P1 | **CONFIRMED** | The capture is after the ordinary implicit solve and before the deferred stage-3 depth-mean replacement; its late substitution changes only returned U/V. T, S, and ssh are 0 unequal, and conflicting momentum exposures refuse at construction. |
| R61-P2 | **CONFIRMED** | Round-50 admission is `AT_BAR`; the controlled entry SHA-256 remains `1d782be6d09446d40b7ef0526464144aa041645279706c73019bc19c6c706719`; the sidecar plant changes the entry unequal count 520 -> 521 and exits 2. |
| R61-P3 | **REFUTED** | Stage-3 raw Kaa is MIXED, not TOWARD: 145 moved, 116 / 22 / 7 toward / away / neutral. |
| R61-P4 | **CONFIRMED** | No row is strictly AWAY after the TOWARD stage-2 ADV boundary; `compensating_owner` remains null. |
| R61-P5 | **CONFIRMED** | GYRE: 70 rows unmoved, 210 / 210 residual arrays array-equal, first-over-bar unchanged, and 30 / 30 daily snapshots byte-identical. |
| R61-P6 | **CONFIRMED** | No physics/configuration/state/ice change remains; both held candidates are absent. |

## Compiled source and observer placement

The executing NEMO program calls `dyn_zdf` and writes `raw_kaa` at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:412-433`.
Only afterwards does it build and apply the barotropic correction at
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90:436-448`.

The private field and conflict guard are
`ocean_model_latlon_cgrid.py:1304` and
`ocean_model_latlon_cgrid.py:2763-2787`.  The production capture is immediately
before the existing deferred correction at
`ocean_model_latlon_cgrid.py:9336-9343`; the captured U/V replace returned
diagnostic slots only after the ordinary step at
`ocean_model_latlon_cgrid.py:9495-9499`.

The direct raw-Kaa row remains 707 / 16,900 unequal in both arms, with maximum
absolute error `0.04480131363770535` m/s.  That large common error is not an
UP3 attribution; this round measures only how the controlled arm moves it.

## Controlled walk completion

| boundary | arm direction | moved | toward / away / neutral | base L-inf -> arm L-inf |
|---|---|---:|---:|---:|
| stage-3 after ADV / after LDF / pre-ZDF | TOWARD | 262 | 124 / 138 / 0 | `2.6472396755999677e-09` -> `2.6472396752679308e-09` |
| stage-3 raw Kaa | **MIXED** | 145 | 116 / 22 / 7 | `0.04480131363770535` -> same |
| stage-3 postbar / kt4 entry | MIXED | 204 | 89 / 115 / 0 | `2.3323169326405768e-08` -> `2.3323170436628793e-08` |

The base and candidate reports have SHA-256
`5e526bdb2199fa2cc6a8a72624f0c3289b8b807cae3460d11547cdf043dc9ca8`
and `13ab134962762e0f2ff75ea4cf9ef613ab391b35921109b4dd3cb9f7c016689a`.
Their sidecars have SHA-256
`3ba3861e2fd76c0d2ae8b64d934b86d668d99cdc8410152bf61ccfc674ed7ccf`
and `e62a01830b4567a7548027bdc79d4ebf3097d5f16572438578ebc6832ffdddb8`.

The first full replay attempt was interrupted before producing evidence after
the first redundant observer entered the known per-process compiler-map risk.
The committed gate instead imports round 60's hash-bound sidecars, freshly
recomputes and requires exact equality of the controlled entry and ordinary
postbar endpoint, and compiles only the new observer.  Two later attempts
refused before scoring: the historical base report lacked the newer explicit
entry-hash field, and the first reconciliation double-converted an already
converted diagnostic layout.  Both refusals remain in `base.log` history; the
successful report is clean-stamped and the receipt cites only it.

## Shared-code gate

The final observer-only package tip and its clean base both produce the same
GYRE residual SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
The offline comparison is `PASS`: 70 certified rows, zero status changes,
zero violations, zero field movement, and unchanged first-over-bar at kt=3.
All 210 stored arrays are `np.array_equal`.

The two 30-day members each contain daily snapshots 1..30.  All 30 pairs are
byte-identical; day 30 has SHA-256
`e2daff3f91ec3d806c109d1f7dff592ed705d236a6bfdb6a8c82614596fb3ec3`
in both arms.

## Review and verification

The separate read-only Codex review did not start a reviewer model.  Its exact
terminal verdict was:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Therefore **independent review unavailable in-sandbox**.  The artifact
SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

Focused observer, round-50/51/60/61, and citation tests passed **38 / 38**.
The default citation audit passed 274 citations with zero unmapped entries; a
real shifted-line plant on the pre-implicit-state citation exited 1 with
`SYMBOL-NOT-AT-LINE`.
The round-61 receipt audit separately passed all 6 citations with zero unmapped
entries; its shifted raw-Kaa capture plant exited 1 with
`SYMBOL-NOT-AT-LINE`.

The one permitted `tests/ocean/fidelity -n 12` battery collected 1,976 tests,
reached 99%, and reproduced the registered terminal zero-progress stall before
a summary.  It emitted the same five pre-existing failures as round 60.  Their
isolated rerun retained the exact signatures: SI3 `MY_SRC` provenance,
round-129 stale phase-3 certification, round-51 private trace registry, the
three worktree-stamp offenders, and the missing `hires_lane_surface` case-board
row.  No round-61 test failed.

## Scope ledger

**ASKED.** Close the missing stage-3 post-vertical-mixing/pre-barotropic
boundary under the controlled held-UP3 arm and apply the strict owner rule.

**UNASKED and unchanged.** No configuration, default, forcing, threshold,
stabiliser, public state, physics statement, ORCA2 entry, or sea-ice field
changed.  The six ORCA2 ice selectors and `unmeasured_features` remain at
`STOP_SELECTOR_GAP`.

## OPEN

1. The first stage-3 TOWARD-to-MIXED change is inside compiled `dyn_zdf`.
   Before another pair retry, preregister its internal source-order walk:
   explicit Kaa update, barotropic-velocity subtraction, explicit bottom/top
   drag, and implicit vertical solve.  The admitted record has only pre-ZDF and
   raw-Kaa endpoints, so that round must first inventory whether an existing
   self-describing record carries those internals; otherwise write a fail-closed
   additions-only acquisition under a new target.
2. Keep source-ordered UP3 and QCO/RK held until a strict compensating statement
   is named.  MIXED remains evidence, not an owner.
3. After the step walk closes, run Decision 52's owed independent-start ORCA2
   ladder, then rank month-scale ORCA2 magnitudes.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: close the stage-3 raw-Kaa boundary.  
UNASKED: configuration, state, stabiliser, physics, and sea-ice changes.
