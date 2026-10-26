# ORCA2 round 58 receipt — source-ordered UP3 landing attempt

Date: 2026-09-27  
Base: `8c4f6a70027bebbfedc85482db8c0290d50c1296`  
Preregistration: `e6177e5bad44faac8158fc685f7bf06782996cd9`  
Candidate measurement commit: `48fbcfba2ce583be2685bee9d73c71cdae6a8129`  
Disposition: **HELD; candidate reverted**  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
GYRE claim label: **independent**  
Direct statement claim label: **given NEMO's recorded operands**

## Answer

The one-statement source-order candidate closed the measured UP3 U T-face
flux from 282 / 17,000 unequal to **0 / 17,000 unequal**.  The candidate also
left all 40 ORCA2 checkpoints unchanged, all 70 GYRE short-ladder rows
unchanged, and all 360 GYRE daily snapshots byte-identical to the controlled
base.

It nevertheless does **not** land.  Frozen prediction R58-P5 required the
standing GYRE Decision 43/45/55/59 admission to report a *strict decrease* in
day-30 temperature RMS.  The controlled values are exactly equal:
`6.572572612618985e-05 K` before and after.  R58-P5 is therefore **REFUTED**
and R58-P7's binding landing condition is false.  The model, test, instrument,
and citation changes were reverted in commit `de0fe8c56dcc5c93ffe4b21d5e7e430d16d85dbc`;
the final `packages/` tree is identical to the preregistration commit.

## Compiled statement and candidate

The compiled executing branch forms the masked horizontal curvature at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:157-166`, chooses
the curvature from the sign of the advected-velocity pair at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:182-192`, and forms
the T-face flux at
`OVERFLOW_OMIP_L1_P3_R56UP3/BLD/ppsrc/nemo/dynadv_up3.f90:194-195`.

The candidate routed only the `nemo_up3` same-direction T-face flux through
that statement order: masked curvature, velocity-pair selector, selected
curvature, then transport-sum multiplication.  It preserved the private
selector ablation, kept cross-direction fluxes and the Oceananigans arm on
their existing paths, and added eager/JIT, finite non-zero gradient, mask,
association, and production-binding controls.  Those focused controls passed
at the candidate commit, but none remains as dead production code after the
held disposition.

## Mechanically gated results

| gate | controlled result | disposition |
|---|---|---|
| Base direct statement | 282 / 17,000 unequal; maximum `0.005124451203774175` | reproduced |
| Direct candidate statement | 0 / 17,000 unequal; all four earlier rows also exact | candidate statement closed |
| Direct one-ULP plant | refusal count changed by one; non-zero exit | control fired |
| ORCA2 kt=1..10 | 40 checkpoints; 0 moved rows; no AT-BAR loss; first owner unchanged | pass, given NEMO's entry |
| GYRE kt=1..10 | 70 certified rows; 0 moved rows; first-over-bar kt 3 -> kt 3; residual sidecars byte-identical | pass, independent |
| GYRE 360-day | 360 / 360 daily snapshots byte-identical | no model movement |
| GYRE day 30 | `6.572572612618985e-05 -> 6.572572612618985e-05 K` | **R58-P5 strict-decrease failure** |
| GYRE day 240 | `0.01644836113029585 -> 0.01644836113029585 K` | unchanged |
| GYRE day 360 | `0.01122565978973451 -> 0.01122565978973451 K` | unchanged |

The GYRE ten-step residual sidecars share SHA-256
`43f37831256832c31f9a983352d8949e9947040faf01d5f4f39751dc1a70935c`.
The offline comparison reports zero maximum worsening ULP, no row-status
changes, and no violations.  The ORCA2 comparison reports no moved row at all,
so no moved-row registry entry exists to omit.

## Frozen predictions

| ID | verdict | evidence |
|---|---|---|
| R58-P1 | **CONFIRMED** | The clean base reproduced 282 unequal faces and maximum `0.005124451203774175`; its plant fired. |
| R58-P2 | **CONFIRMED** | The candidate direct gate reported 0 / 17,000 unequal without moving an earlier row. |
| R58-P3 | **CONFIRMED** | Eager/JIT, gradient, mask, association and production-binding focused controls passed. |
| R58-P4 | **CONFIRMED** | The controlled ORCA2 ladder completed 40 checkpoints with zero movement and no AT-BAR loss. |
| R58-P5 | **REFUTED** | GYRE was byte-identical rather than strictly improved at day 30.  The frozen confirmation predicate did not pass. |
| R58-P6 | **NOT REACHED** | The binding R58-P5 failure fixed the disposition before tank/card admission; no unrun card result is inferred. |
| R58-P7 | **REFUTED** | The statement did not land; the candidate was reverted. |

## Verification and review

- The candidate-focused unit/fidelity selection passed 29 tests before the
  binding trajectory measurement.
- The default citation gate passed with zero failures and zero unmapped
  citations at the candidate commit.  The final receipt citation gate and its
  shifted-line plant are recorded with the final documentation commit.
- Separate read-only review: **independent review unavailable in-sandbox**
  (`failed to initialize in-process app-server client: Read-only file system`).
- Final-tree focused tests: **39 passed** in 22.03 s.  The required
  `tests/ocean/fidelity -n 12` battery collected 1,968 items, reached 99% with
  no reported failure, and reproduced the registered xdist terminal stall;
  it was interrupted after the workers stopped making progress.  These runs
  verify the restored admitted implementation, not the held candidate.

## Scope ledger

**ASKED.** Preregister, implement, and judge one NEMO-cited source-ordered UP3
same-direction T-face statement under the shared landing gates.

**UNASKED and unchanged.** No configuration field, scheme, threshold,
stabiliser, carried state, ORCA2 entry, or sea-ice field changed.  The final
model tree is the admitted base.  ORCA2's six ice selectors and
`unmeasured_features` tuple remain at `STOP_SELECTOR_GAP`.

## OPEN

1. In a new preregistered round, resolve the admission-policy distinction
   mechanically: the shared-code rule's byte-identical GYRE branch versus the
   Decision-43 strict-improvement branch that applies when a statement moves
   GYRE.  If the byte-identical branch is admitted, retry this same single
   compiled statement without changing its measured implementation.
2. Keep the separately held QCO/RK candidate behind the UP3 disposition.
3. After the step-level walk closes, perform Decision 52's owed ORCA2
   **independent-start** ladder, then rank month-scale magnitudes.
4. Sea ice remains out of scope at `STOP_SELECTOR_GAP`.

ASKED: judge the source-ordered UP3 statement under the frozen gates.  
UNASKED: none.
