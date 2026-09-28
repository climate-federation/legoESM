# ORCA2 round 57 receipt — OVERFLOW stage-2 UP3 face flux

Date: 2026-09-27  
Base: `561483aeed99f3a678a879d31856929f41789afa`  
Preregistration: `bd240f1b56573742fc0ed22c7ef73ad0d12820f7`  
Measurement commit: `05e0a2cb837eff4ee5eeb433533f911434bc631e`  
Disposition: **HELD**  
ORCA2 claim label: **given NEMO's entry** (Decision 52; no ORCA2 score is
reported here)  
OVERFLOW claim label: **given NEMO's recorded operands**

## Answer

The first non-bit statement in the acquired kt=3 stage-2 UP3 walk is the U
T-face flux expression.  Its current legoESM evaluation differs on **282 / 17,000**
contributing faces, with maximum absolute difference
`0.005124451203774175`; the first mismatch is record index `[12,3,0]`, or
Fortran `(ji,jj,jk)=(13,4,1)`.

Every earlier recorded statement is bit-exact: the masked horizontal curvature
is 0 / 17,100 unequal, the advected-velocity pair is 0 / 17,000, the selected
curvature is 0 / 17,000, and a direct replay of NEMO's face-flux expression is
0 / 17,000.  NEMO forms the masked curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, selects
that curvature at `OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:182-192`,
and evaluates the first non-bit face flux at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:194-195`.

No model statement lands in this measurement round.  The result owns the next
one-statement landing attempt under the full shared-card gate.

## Record and admission

The committed instrument reads the round-56 self-describing record
`oracle_r56_up3_kt00000003_s2.bin` from the operator-produced
`OVERFLOW_OMIP_L1_P3_R56UP3` executable.  Its payload SHA-256 is
`79f409d8f0198e87bb46b016c6b3f0080ea8d5ec4ac6e2ae0b02e432f3d8e457`.
Header-derived parsing admits all 47 named fp64 arrays, the physical EOF,
origin `(3,3)`, shape `206 x 7 x 100`, producer commit, parent endpoints, and
inherited-state checks as `AT_BAR`.  The score uses the fixed 16,900-cell
active-U parent mask.  V remains `UNMEASURED_NO_ACTIVE_FACE` on this one-row
card.

## Source-order census

| statement | unequal / scored | maximum absolute difference | verdict |
|---|---:|---:|---|
| masked U horizontal curvature | 0 / 17,100 | 0 | `BIT_EXACT` |
| U advected-velocity pair | 0 / 17,000 | 0 | `BIT_EXACT` |
| sign-selected U curvature | 0 / 17,000 | 0 | `BIT_EXACT` |
| direct NEMO face-flux replay | 0 / 17,000 | 0 | `BIT_EXACT` |
| current legoESM U T-face flux | 282 / 17,000 | `0.005124451203774175` | first `NON_BIT` |

The 282 mismatches have two mechanically closed components:

- **4 wet/dry-boundary faces** carry the structural error.  NEMO's selected
  curvature is masked; the expanded legoESM reconstruction reads the raw
  neighbouring velocity.  Their maximum absolute difference is
  `0.005124451203774175` and maximum relative difference is `1/3`.
- **278 wet-interior faces** differ only through arithmetic association.
  Their maximum absolute difference is `1.1368683772161603e-13`, maximum
  relative difference `4.947835730959339e-16`, and maximum row-scale distance
  4 ULP.

Thus preregistered R57-P3 correctly named the face expression as the first
non-bit row, but its proposed algebraic-association mechanism was incomplete:
the missing mask is the dominant structural component, while association
explains the remaining wet-interior cells.

## Required retraction

An initial post-measurement instrument revision compared the writer's
`flux_u_t` field at record index `i+1` and reported 676 / 17,000 unequal with
maximum `1074`.  That result is **RETRACTED**.  The writer is called with
`zFu_t(ji+1,jj)` but stores the scalar under loop coordinate `(ji,jj)`; the
self-describing field is loop-aligned, not native-array-aligned.  The corrected
committed gate compares index `i`, adds an exact source-order NEMO replay, and
produces the 282-cell result above.  No retracted number is used as evidence.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R57-P1 | **CONFIRMED** | The record is `AT_BAR`; its header, payload, producer, parent and inherited-state checks pass unchanged. |
| R57-P2 | **CONFIRMED** | Recorded curvature, pair and selector statements replay with zero unequal cells. |
| R57-P3 | **CONFIRMED, mechanism refined** | The face expression is the first non-bit row at 282 / 17,000; four boundary cells expose the omitted mask and 278 wet-interior cells expose association. |
| R57-P4 | **CONFIRMED** | Advancing one baseline-equal face flux by one fp64 representable value changes 282 refusals to 283 and exits 2. |
| R57-P5 | **CONFIRMED** | The final `packages/` diff is empty; disposition is `HELD`. |

## Controls and verification

- Baseline statement gate: `FIRST_NON_BIT_NAMED`, exit 0.  The face-flux
  plant adds exactly one refusal and exits 2.
- Focused round-56/57 tests: **11 passed** in 0.74 s.  Ruff, Python compilation,
  and `git diff --check` pass.
- Push-gate-equivalent battery: **127 passed** in 364.73 s.
- Shared-card battery: **170 passed**, 9 warnings, in 355.50 s.
- `tests/ocean/fidelity -n 12` collected 1,968 items, reached 99%, and
  reproduced the registered xdist terminal stall.  Before interruption it
  emitted exactly the same five registered failures.  Each was rerun alone
  and retained its existing signature: SI3 `MY_SRC` scalar-math provenance,
  round-129 stale certification, round-51 private trace registry, the same
  three worktree-stamp offenders, and the `hires_lane_surface` case-board row.
- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- The default and this-receipt citation gates, including a shifted-line plant,
  are recorded in the final round commit.
- No `packages/` file changed.  Therefore no ORCA2, GYRE, DINO, tank or generic
  trajectory result is claimed for this measurement-only round.

## Scope ledger

**ASKED.** Admit the acquired record, walk the compiled UP3 statements in
source order, and name the first non-bit statement.

**UNASKED and unchanged.** No model state, recipe, selector, threshold, mask,
stabiliser, carried ORCA2 entry, or sea-ice field changed.  The ORCA2 card's
six ice selectors and `unmeasured_features` tuple remain exactly at
`STOP_SELECTOR_GAP`.

## OPEN

1. Preregister a one-statement implementation of NEMO's source-ordered masked
   curvature, sign selector and face-flux evaluation; land it only if the full
   GYRE Decision 43/45/55/59 gate plus ORCA2, OVERFLOW, DINO, tank and generic
   card gates pass.
2. Keep the held QCO pair behind that first-statement landing.
3. When the step-level walk closes, perform Decision 52's owed ORCA2
   **independent-start** ladder, then rank month-scale magnitudes.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: admit the round-56 record and name the first non-bit compiled statement.  
UNASKED: none.
