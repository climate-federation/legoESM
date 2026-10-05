# ORCA2 round 59 receipt — corrected GYRE predicate, OVERFLOW hold

Date: 2026-09-28  
Base: `360ba71cef9f6d8b7cda2e1eed6df1e401a37256`  
Preregistration: `42b2410761c206a149754892c33c7974983f8831`  
Measured candidate: `201fe69ba6e386552c4619b18d087cca4cf9881e`  
Candidate reversion: `dda89cdc1`  
Disposition: **HELD; candidate reverted**  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
GYRE and OVERFLOW claim label: **independent**  
Direct statement claim label: **given NEMO's recorded operands**

## Answer

Operator note B12 correctly admits a shared statement when GYRE is
byte-identical.  This candidate satisfies that branch: all 70 ten-step rows,
the residual sidecar, and all 360 daily snapshots are byte-identical to the
certified reference.  The direct source-ordered UP3 U T-face flux also closes
from 282 / 17,000 unequal to **0 / 17,000**.

The candidate nevertheless does **not** land.  The statement executes on
OVERFLOW, where 31 trajectory rows move and 20 rows violate the frozen strict
2-ULP oracle-relative bar.  The earliest failing row is
`OVERFLOW-zps.kt4.before.ssh`: 4 cells worsen beyond the bar, with maximum
row-scale worsening `4094.6171875` ULP.  The overall maximum is
`875018892.0` ULP on `OVERFLOW-zps.kt10.before.ssh`.  Frozen R59-P6 and R59-P7
are therefore **REFUTED**, and the candidate was reverted.

This refutes two premises in note B12 for this exact branch tip: ORCA2 is
unchanged rather than improved, and the complete shared-card gate is not
green.  The corrected GYRE predicate is not the blocker; the newly measured
OVERFLOW trajectory is.

## Compiled statement and controlled candidate

The executing compiled branch forms masked horizontal curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, selects
curvature from the sign of the advected-velocity pair at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:182-192`, and forms
the T-face flux at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:194-195`.

The candidate restored the byte-identical round-58 implementation: only the
NEMO-UP3 same-direction T-point flux used the compiled masked-curvature,
selector, and product order.  The Oceananigans arm, cross-direction fluxes,
configuration, state, forcing, thresholds, and sea ice were unchanged.  The
seven candidate files matched commit `48fbcfba2` before measurement.

The final `packages/` tree is byte-identical to base commit `360ba71cef`; the
two candidate-only files are absent.

## Mechanically gated results

| gate | controlled result | disposition |
|---|---|---|
| Base direct statement | 282 / 17,000 unequal; 4 boundary + 278 interior; maximum `0.005124451203774175` | reproduced |
| Base one-ULP plant | 283 / 17,000 unequal; exit 2 | control fired |
| Candidate direct statement | 0 / 17,000 unequal; every earlier source row remains exact | pass |
| Candidate one-ULP plant | 1 / 17,000 unequal; exit 2 | control fired |
| ORCA2 kt=1..10 | `LADDER_MEASURED`; 40 checkpoints; 0 moved rows; no AT-BAR loss | pass, given NEMO's entry |
| GYRE kt=1..10 | 70 rows; 0 moved; first-over-bar unchanged; residual SHA-256 `43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c` | pass, independent |
| GYRE comparison plant | synthetic 3-ULP worsening exits non-zero | control fired |
| GYRE 360-day | 360 / 360 registered `year` snapshots byte-identical | corrected B12 identity branch passes |
| GYRE day 30 / 240 / 360 T RMS | `6.572572612618985e-05` / `0.01644836113029585` / `0.01122565978973451` K, unchanged | pass by identity branch |
| OVERFLOW kt=1..10 | 31 / 50 rows move; **20 violating rows**; first-over-bar kt 2 T/U unchanged | **R59-P6 failure** |

The round-47 reference and round-59 candidate used the same committed
trajectory and offline comparison gates.  The candidate report and residual
sidecar are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round59/iterm_1/`.

An initial post-trajectory comparison used the stage-sweep report against the
whole-step reference.  That comparison is **RETRACTED** because the row schemas
are disjoint; it produced zero comparable rows and no scientific verdict.  The
committed whole-step trajectory gate was then run and produced the 50-row
comparison above.

## OVERFLOW violation register

| row | cells worse than 2 ULP | maximum row-scale worsening (ULP) |
|---|---:|---:|
| kt4 before ssh | 4 | `4094.6171875` |
| kt4 before u | 77 | `182.4154052734375` |
| kt5 before T | 2 | `3.0` |
| kt5 before ssh | 6 | `571525.15625` |
| kt5 before u | 126 | `26239.0732421875` |
| kt6 before T | 7 | `8.0` |
| kt6 before ssh | 9 | `12765624.125` |
| kt6 before u | 161 | `664096.8515625` |
| kt7 before T | 19 | `64.0` |
| kt7 before ssh | 12 | `98623726.75` |
| kt7 before u | 271 | `6270035.546875` |
| kt8 before T | 34 | `966.0` |
| kt8 before ssh | 15 | `363575500.5` |
| kt8 before u | 337 | `29458656.96875` |
| kt9 before T | 32 | `8487.0` |
| kt9 before ssh | 17 | `738802378.5` |
| kt9 before u | 400 | `79001111.0` |
| kt10 before T | 44 | `46915.0` |
| kt10 before ssh | 19 | `875018892.0` |
| kt10 before u | 434 | `130697409.125` |

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R59-P1 | **CONFIRMED** | Base count, four/278 partition, maximum, record admission, and plant reproduce. |
| R59-P2 | **CONFIRMED** | Candidate direct statement is 0 / 17,000 unequal without moving an earlier row. |
| R59-P3 | **PARTIAL, then not required for landing** | The direct production binding and plants pass and the candidate is content-identical to round 58; the gradient-focused candidate pytest was not repeated after R59-P6 fixed the held disposition. |
| R59-P4 | **CONFIRMED** | ORCA2 completes 40 checkpoints with zero moved rows. |
| R59-P5 | **CONFIRMED** | GYRE takes B12's identity branch: 0 ladder moves, identical residual bytes, and 360 / 360 identical daily snapshots. |
| R59-P6 | **REFUTED** | OVERFLOW has 20 violating trajectory rows; later shared-card gates were not reached. |
| R59-P7 | **REFUTED** | The one statement does not land; the model candidate is reverted. |

## Verification and review

- Separate read-only `codex exec` review: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`).
- Restored final-tree focused tests: **39 passed** in 21.40 s.
- `tests/ocean/fidelity -n 12` collected 1,968 items, reached 99%, reproduced
  the registered terminal stall, and was interrupted only after sustained zero
  progress.  It emitted the same five registered failures.  Their isolated
  rerun retained the existing signatures: SI3 `MY_SRC` provenance, round-129
  stale certification, round-51 private trace registry, the three worktree
  stamp offenders, and the `hires_lane_surface` case-board row.
- The default and this-receipt citation gates and their shifted-line plant are
  recorded after this receipt commit.

## Scope ledger

**ASKED.** Re-evaluate and attempt to land the measured source-ordered UP3
statement under operator note B12's corrected byte-identical-GYRE predicate.

**UNASKED and unchanged.** No configuration field, default, scheme, forcing,
threshold, stabiliser, carried state, ORCA2 entry, or sea-ice field changed.
ORCA2's six sea-ice selectors and `unmeasured_features` tuple remain at
`STOP_SELECTOR_GAP`.

## OPEN

1. Before retrying the source-ordered UP3 statement, walk the first downstream
   OVERFLOW movement from the admitted kt=3 stage-2 after-ADV endpoint through
   stage completion and the kt=4 entry, in compiled source order, to name the
   compensating statement responsible for the earliest ssh/u violations.
2. Keep the separately held QCO/RK candidate behind this UP3 disposition.
3. After the step-level walk closes, perform Decision 52's owed ORCA2
   **independent-start** ladder, then rank month-scale ORCA2 magnitudes.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: retry the source-ordered UP3 landing under corrected note B12.  
UNASKED: none.
