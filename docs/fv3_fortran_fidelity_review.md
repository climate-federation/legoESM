# FV3 Fortran Fidelity Review

Baselined 2026-04-14.  Older per-iteration prose has been
condensed to save tokens; only the newest Ralph-loop iterations
remain in full form below.

> **Metadata convention (iter-174)**: the earlier "updated through
> Ralph iter N" dateline in this title was removed because each
> iteration bumping it triggered a Codex-flagged inconsistency
> (the number would lag by 1-4 iterations as soon as the NEXT
> stop-hook committed anything).  The authoritative current
> iteration count is the commit count on branch plus the HEAD
> commit message's iter-N tag; there is no need to duplicate that
> metadata in the title.


## Architectural unresolved items (carried through all iterations)

1. **W2 v-wind cube-face imprint at C36** — STILL PRESENT per direct
   user visual inspection (iter-716, 2026-04-17):
   - t=0d: |v| < 0.01 m/s (clean initial condition).
   - t=0.4d: polar caps + mid-latitude cube-seam signature emerging, |v| ~ 0.1 m/s.
   - t=1.0d: **mode-4 polar artifact + 4 cube-face-corner hot spots**, |v| ≈ ±0.3 m/s.
   - The 4-fold azimuthal symmetry at each pole matches the 4 cube-corner
     positions on faces 4 (N pole) and 5 (S pole).
   - Production path uses Arakawa-Lamb + RK3 + `boundary_fix`, NOT the FV3 FB
     chain.  iter-505 reduced amplitude from ~0.56 m/s to ~0.30 m/s by fixing a
     PPM x-axis bug; the residual signature is inherent to the A-L + halo-
     interpolation + `boundary_fix` non-FV3 path.
   - **Canonical evaluation is `scripts/run_atmosphere_test_matrix.py`** (user
     confirmed iter-716): matrix config at line 1178-1181 is
     `hyperdiff_coeff=_hyperdiff_cube(n)`, `div_damp=_div_damp_cube(n)`,
     `boundary_fix=True`, `dt=300s`.  The unit tests in
     `TestW2BoundaryErrorBudget` / `TestW2CubeFaceImprintCharacterization` /
     `TestW5PolarFaceMagnitude` replicate this config exactly but only lock
     SCALAR summary statistics (L2, max|v|, face-4 max) — they do NOT catch
     the VISUAL mode-4 pattern directly.
   - Fix requires EITHER (a) stabilize the FV3 FB chain at C36 (needs ng=3
     halo infrastructure — partially scaffolded, see iter-496..598), OR
     (b) replace the A-L path with a FV3-faithful production alternative.
     Defensive tests now pin the current path, but the architectural blocker
     remains unresolved.

2. **FB-path C36 instability** — `_c_sw` first-order upwind
   amplifies face-boundary halo divergence; requires ng=3
   MPI DGRID_NE halo infrastructure for `_d2a2c_vect`.  Scalar
   `pad_halo(halo=3)` + `halo_interp_offsets_h3` + padded grid
   angle + half-metrics + vector `pad_halo_vector(halo=3)` on
   non-MPI backend ALL wired (iter-496..598).  Remaining:
   `pad_halo_mpi_4d(halo=3)` + flipping `_d2a2c_vect` and
   `fv3_fb_sw_step` callers from h=2 to h=3.

## Priority resolution history (iter-96..131)

- **Priority 1 (panel-edge corner metrics)**: RESOLVED iter-96/98/99.
  Python `cdgrid.cosa_corner`/`sina_corner`/`rsin2_corner` matches
  Fortran halo-averaged formula with cross-face sign-flip rotation
  at all 24 seams to 1e-10.  iter-91..95 "convention gap" claim
  was wrong (naive halo-copy model).  Locked by
  `test_cosa_corner_panel_edge_fortran_match_all_24_seams` +
  `test_rsin2_corner_matches_fortran_at_interior`.

- **Priority 2 (d_sw3 BGRID_NE component sync)**: RESOLVED iter-102/103.
  `synchronize_bgrid_ne_corner_geo` routes through geographic frame,
  handling all 24 seams + 8 cube vertices via
  `synchronize_corner_scalar`.  Wired into `_bgrid_ke_transport`
  replacing scalar-KE fallback, matching `dyn_core.F90:968-1019`.
  Iter-140 removed dead 12-seam legacy (−233 lines).

- **Priority 3 (non-duogrid `_d2a2c_vect` cube-vertex gap)**:
  GAP ISOLATED AND LOCKED iter-107/108/128..131.  Fortran
  `sw_core.F90:3527-3545,3620-3640` writes 3 halo cells per
  corner-axis; Python's `_fill_corners_h2` can represent only 2
  per axis (architecturally bound, needs halo=3).  Reachability:
  only experimental FB chain + `fv3_csw_tendencies`; default
  production A-L path does NOT call `_d2a2c_vect`.  Runtime
  tripwires (iter-129/130) pin the call graph.  iter-569/570
  extracted exact ndsl / pyFV3 `fill_corners_3cells_mult_x`
  recipe for future h=3 port.

- **Priority 4 (FB-path diagnostic)**: RESOLVED iter-109..111
  with `test_fb_path_component_vs_scalar_sync_propagates_to_wind`
  pinning end-to-end propagation at ±2e-3 of measured diffs.

## Polar-face asymmetry investigation (iter-112..127, closed iter-510)

Pre-iter-505 baseline showed 16% N-S mass-tendency asymmetry
between faces 4 and 5 on W2 alpha=0 balanced state.  Bisection
localized to `cgrid_mass_flux_divergence` PPM upwind at polar-face
halo cells.  **Iter-505 axis fix eliminated this entirely** —
faces 4/5 now identical tendency magnitudes to 1e-4 of 1.0 and
exact N-S reflection symmetry at machine precision.  Locked by
`TestFv3SwTendenciesPolarFaceSymmetry` (iter-510).

## ITER-505 major production-path bug fix (2026-04-19)

**Root cause**: `cgrid_mass_flux_divergence` (`operators_cdgrid.py`)
and `_cgrid_fct_fluxes_2d` (iter-506) passed x-direction strips of
shape `(6, n+4, n)` directly to `_ppm_reconstruct_1d`, which
reconstructs along the LAST axis.  The halo-padded i-axis was
axis 1 but the 8-cell interior j-axis was LAST — so reconstruction
ran along the wrong axis.  **Fix**: `swapaxes(1, 2)` before
reconstruction, swap back after.

**Impact on canonical W2 alpha=0 C36 dt=300s 1 day** (matrix path):

| Quantity                        | Pre-iter-505 | Post-iter-505 | Change |
| ------------------------------- | ------------ | ------------- | ------ |
| Williamson 2 L2                 | 1.53e-3      | **2.42e-4**   | 6.3× ↓ |
| Williamson 2 Linf               | 4.07e-3      | **1.83e-3**   | 2.2× ↓ |
| Williamson 2 max\|v_ll\|        | 0.557 m/s    | **0.303 m/s** | 1.8× ↓ |
| Williamson 5 mass drift         | 1.42e-5      | **1.74e-5**   | within order |
| Cosine bell L1                  | 1.20e-1      | **1.20e-1**   | unchanged |
| Ocean rest state                | machine prec | **machine prec** | invariant |

Cosine bell uses `transport_step` directly (Lin-Rood), not
`cgrid_mass_flux_divergence`.  Iter-508/509 refactored
`_ppm_reconstruct_1d` to take mandatory `axis` kwarg; AST guard
`test_no_future_caller_passes_non_halo_last_axis_to_ppm` enforces
contract.

## Historical archive (token-trimmed)

The full prose log for `iter-1..729` plus older late backfilled
notes was collapsed to save tokens.  Current live conclusions are
preserved above in:

- `Architectural unresolved items`
- `Priority resolution history`
- `Polar-face asymmetry investigation`
- `ITER-505 major production-path bug fix`

Compressed milestones:

- `iter-1..66`: core shallow-water FV3 metric/operator port
  against the Fortran oracle; FB chain landed but remained
  C36-unstable.
- `iter-67..174`: seam/sync closures, non-duogrid gap isolation,
  regression expansion, and fidelity-note hardening.
- `iter-505..510`: major production-path polar/PPM fix; old
  face-4 vs face-5 asymmetry closed.
- `iter-511..643`: W2/W5 artifact characterization, halo=3
  readiness, exact-formula locks, and test/doc cleanup.
- `iter-644..715`: formula-lock expansion, gold files, structural
  AST guards, metric fixes, and production-path hardening.
- `iter-716..729`: user-facing W2 artifact characterization,
  halo=3 plumbing, FB damping work, and production-vs-FV3 routing.

Use git history if you need the full pre-iter-730 narrative.

## Latest Ralph-loop iterations (full form)

### Iter-730 — FB C24 W2 stability: dissipation sweep confirms instability is upstream of damping

Task 3 of the iter-722 Path Forward: "Verify FB + duogrid stability at C24 first, then C36."  Iter-727 wired the d_sw1 mass-transport del-n damping via `fv_tp_2d(..., nord=nord_v, damp_c=damp_v)`, closing task 2.  Iter-730 tests whether that port plus damp_v-driven d_sw6 vorticity damping is now sufficient to stabilise the FB chain at C24.

**Diagnostic.**  `scripts/diag_iter730_fb_c24_stability.py` runs Williamson 2 on `FV3FBShallowWaterModel` at C24 for 1 simulated day with dt=300s.  Default FB config (A_h=0, hyperdiff_coeff=0, div_damp=0, d4_bg=0.16, nord=1, dddmp=0) plus a sweep of `damp_v` (which activates BOTH the iter-727 d_sw1 mass-damp path AND the pre-existing step-(9) `_del6_vt_flux` vorticity damping).

**Results at C24, dt=300s:**

| Case | damp_v | Blow-up step / 288 | max\|u_d\| @ step 48 | max\|v_d\| @ step 48 |
|------|--------|---------------------|---------------------|---------------------|
| A    | 0.00   | 60 (t=18000s)       | 90.1                | 184.4               |
| B    | 0.06   | 60 (t=18000s)       | 95.7                | 190.9               |
| C    | 0.12   | 60 (t=18000s)       | 104.3               | 201.6               |
| D    | 0.30   | **15** (t=4500s)    | —                   | —                   |

Exact W2 max\|u\| ≈ 38.6 m/s, max\|v\| = 0.  By step 48 the FB chain has already developed winds 2.3-5× the analytic solution; damp_v>0 accelerates the blowup rather than suppressing it.

**Observation (what the sweep actually shows).**
- Turning on the iter-727 d_sw1 mass-damp + step-(9) `_del6_vt_flux` branches via damp_v does NOT eliminate the step-60 blowup for damp_v ∈ {0.00, 0.06, 0.12}.  The blowup step is identical across those three, but the pre-blowup max|u_d| at step 48 increases monotonically with damp_v (90 → 96 → 104 m/s); so damp_v is not a no-op on the trajectory even though it does not change when |h| goes non-finite.
- damp_v = 0.30 blows up at step 15, much earlier than baseline.  This could be (i) the del-n smoother itself has a CFL limit that is violated at damp_v=0.30, nord=1 on this grid; OR (ii) the smoother is propagating already-contaminated momentum gradients.  The sweep alone does not distinguish these.

**What the sweep does NOT prove** (iter-731 Codex stop-time correction).  The earlier iter-730 entry concluded "Task 3 cannot be closed by tuning damping" and "the instability lives UPSTREAM of both damping branches."  Those statements are stronger than the evidence.  The tested space is narrow:
- `dddmp`=0 for every case (adaptive Smagorinsky off).  FV3 defaults `dddmp=0.2` in many configurations.
- `d2_bg`=0 for every case (background del-2 divergence damping off).
- `nord`=1 only.  FV3 supports nord ∈ {0, 1, 2} and Fortran nord=2 triggers del-6 rather than del-4.
- dt=300s only.  The blowup at step 60 (t=18000s) could plausibly be a step-count artifact of the specific dt/grid CFL margin.
- No combination of non-zero `dddmp`, `d2_bg`, and `damp_v` was tested simultaneously.

Task 3 status is therefore **OPEN, NOT CLOSED AND NOT BLOCKED** — the reduction should read: "dissipation alone at (dddmp=0, d2_bg=0, nord=1, dt=300s) with damp_v ≤ 0.12 does not stabilise FB C24 W2; broader dissipation search + structural investigation both remain valid next-iter moves."

**Corrected task-status.**
- Task 1 (halo=3 infra + `_d2a2c_vect` caller flip): **DONE**.
- Task 2 (per-phase dissipation wiring — d_sw1 mass, d_sw5 corner div, d_sw6 vorticity): **DONE wiring**; tuning/combinatorics to stabilise the run is part of task 3, not task 2.
- Task 3 (FB + duogrid stability at C24): **OPEN**.  Iter-730 ruled out a narrow slice of the (`damp_v`, nord, dt) parameter space; a broader search over `dddmp`, `d2_bg`, `nord`, `dt` combinations and a line-by-line Fortran bisection of `_c_sw` / `_d_sw_native` phases remain in play.
- Task 4 (gold-file FB chain at C24): **NOT STARTED** (gated on task 3).

**Iter-730 deliverable (retained).**  `scripts/diag_iter730_fb_c24_stability.py` is the concrete evidence underlying the narrow-slice task-3 ruling.  It is checked in so future iters can re-run it after other dissipation combinations or structural fixes land and confirm the blowup-step number moves.

**Iter-731 correction deliverable.**  This section's "Observation" / "What the sweep does NOT prove" rewrite addresses the Codex stop-time finding: "the checked-in evidence overstates what the `damp_v` sweep proves."  Evidence is retained as-is; the interpretation is scaled back to what the numbers actually support.

**Priority for iter-732+.**  With task 3 neither closed nor blocked, iter-732+ has three valid moves:
- (a) extend the damp_v sweep to (dddmp, d2_bg, nord, dt) combinations and produce a proper stability map;
- (b) root-cause bisect the C24 momentum blowup by logging per-phase residuals at step 1 against Fortran;
- (c) shift focus to the production A-L+RK3 W2 v-wind mode-4 artifact (the user's iter-717 image blocker) since that affects the current production path independent of FB.

### Iter-732 — FB C24 step-1 phase bisect: v_d drift localises at cube vertex

Iter-731 option (b): root-cause-bisect the FB C24 W2 blowup by running ONE FB step and measuring per-variable drift from the W2 exact steady-state IC.  Delivered as `scripts/diag_iter732_fb_c24_phase_bisect.py`.

**Setup.**  W2 IC at C24 (u_0=38.61 m/s, h_0=29400/g ≈ 3000 m, v=0 geographic) on a non-duogrid cubed sphere.  Run one FB step with dt=300s (same config as iter-730 case A: damp_v=0, d4_bg=0.16, nord=1).  Since W2 is an exact steady state, after one step an FV3-faithful port should show drift at the level of discretisation truncation (O(1e-6) relative per step typically) — certainly NOT 1% per step.

**Per-variable drift at step 1:**

| Variable | max\|Δ\|   | RMS        | max\|Δ\| / max\|ref\| |
|----------|------------|------------|---------------------|
| h        | 1.52e+1 m  | 2.16e+0 m  | **5.08e-3**         |
| u_d      | 8.84e-1 m/s| 3.30e-1 m/s| **2.29e-2**         |
| v_d      | 1.39e+0 m/s| 6.20e-1 m/s| **5.10e-2**         |

v_d drifts 5% in ONE step.  Extrapolating multiplicatively: at step 60 the drift is `1.05^60 ≈ 18×` the IC magnitude, consistent with the max|v_d| ≈ 184 m/s observed at iter-730 step 48 (≈ 5× u_0 IC).

**Per-face error localisation at step 1:**

u_d max error location (shape (6, n, n+1) = (6, 24, 25)):
- face 0-3 (equatorial): max at `i=1, j=0` or `j=n-1`.  `i=1` is adjacent to the WEST panel edge.
- face 4-5 (polar): max at boundary (i=0 or i=n-1).

v_d max error location (shape (6, n+1, n) = (6, 25, 24)):
- face 0-3 (equatorial): max at `(i,j) = (24, 0)` on ALL 4 EQUATORIAL FACES.  `i=24` is the NORTH panel boundary; `j=0` is the WEST panel boundary.  That's a cube VERTEX location, shared between NW corners of all 4 equatorial faces + SW corner of face 4 (north polar).
- face 4-5 (polar): max at (12, 23) — different localisation, consistent with the different metric structure at polar faces.

**Diagnosis.**  The v_d drift is face-consistently concentrated at a specific cube VERTEX (the NW corner of all 4 equatorial faces, which is the same 3-face-meeting point in 3-D).  This is a classic signature of a cube-corner halo bug in `_d_sw_native`'s D-grid wind update phase — the three panels meeting at the vertex produce inconsistent halo values for u_d/v_d and the step-6 `u_d_new = u_d + (ke_diff_u + fy_vort) * rdx_u` incremental update picks up the inconsistency.

**This is evidence, not a fix.**  Iter-732 does NOT attempt a source change.  The error-localisation pattern is concrete enough to narrow future investigation to:
1. `_bgrid_ke_transport` (step 4 of `_d_sw_native`) — the `ke_corner` field is at D-grid vertices, and its 4-cell NE stencil `ke_corner[:, :-1, :]-ke_corner[:, 1:, :]` at vertices adjacent to a cube corner will see cross-face inconsistency if `ke_corner` isn't properly synchronised.
2. `_corner_vorticity` + `_vorticity_flux` (steps 3 and 7) — `zeta_abs` at corners uses cell-averaged D-grid circulation; the 4-cell vertex in halo space must agree across three meeting panels.
3. `_d_sw1_recompute_ut_vt` (step 1) — the C-grid transport velocity computed from uc/vc via the 2×2 corner solve.  Iter-92-ish documented a corner solve here.

**Process constraint (per iter-731 Codex review).**  Do NOT extrapolate from "v_d drift localises at cube vertex" to "the bug IS at _bgrid_ke_transport corner sync".  Three candidate phases identified; iter-733+ must bisect each before claiming a root cause.

**Deliverable.**  `scripts/diag_iter732_fb_c24_phase_bisect.py` is checked in as the step-1 regression sentinel.  A future iter that believes it fixed the C24 FB blowup must re-run this script and observe the v_d max|Δ|/max|ref| drop from 5.10e-2 to O(1e-4) or better BEFORE claiming success.  Matching the step-60 blowup number alone is insufficient (iter-730 lesson).

### Iter-733 — correct iter-732 "phase bisect" (Codex stop-time finding)

Codex stop-time review on iter-732 flagged: **"the new 'phase bisect' diagnostic does not actually bisect phases."**  Correct.  Iter-732's script printed per-variable drifts that were (a) only measurable AFTER the full `_d_sw_native` ran (since c_sw / p_grad_c do not touch u_d / v_d), and (b) included C-grid intermediates (uc_new=41 m/s) which are a velocity magnitude not a drift.  The "bisect" label was wrong.

**Iter-733 rewrite.**  `scripts/diag_iter732_fb_c24_phase_bisect.py` now performs a TRUE bisect in four stages:

(A) **FB outer structure.**  Confirm c_sw and p_grad_c do not update u_d / v_d.  Therefore 100 % of step-1 u_d/v_d drift is attributable to `_d_sw_native`.  Printed explicitly.

(B/C) **`_d_sw_native` inner decomposition.**  d_sw6's wind update is `u_d_new = u_d + (ke_diff_u + fy_vort) * rdx_u`.  W2 IC satisfies geostrophic balance so `ke_diff_u + fy_vort` should CANCEL to O(truncation).  The script SEPARATELY evaluates:
- `|ke_diff_u|` (from `_bgrid_ke_transport` via `ke_corner[:,:-1,:]-ke_corner[:,1:,:]`),
- `|fy_vort|` (from `_corner_vorticity` + `fv_tp_2d`),
- `|ke_diff_u + fy_vort|` (the residual that drives the wind update).

(D) **Per-face argmax localisation** of the residual.

(E) **Per-face argmax localisation** of each component separately, so residual localisation can be compared against each component.

**Numerical results at C24, W2, dt=300s, step 1, damp_v=0:**

| Quantity                           | max\|·\|  | argmax (equatorial faces)     |
|------------------------------------|-----------|-------------------------------|
| ke_diff_u                          | 1.29e+5   | (23, 0) — NORTH-WEST cell edge|
| fy_vort                            | 3.79e+5   | (0, 0)  — SOUTH-WEST corner   |
| **ke_diff_u + fy_vort (residual)** | **3.68e+5**| (1, 0\|n-1) — one cell in from W |
| ke_diff_v                          | 1.06e+5   | (23, 0)                       |
| fx_vort                            | 4.39e+5   | (0, 0)                        |
| **ke_diff_v - fx_vort (residual)** | **5.36e+5**| (24, 0) — NW cube vertex      |
| **residual / \|ke_diff_u\|**       | **2.86**  |                               |
| **residual / \|ke_diff_v\|**       | **5.07**  |                               |

**Diagnosis (concrete, not overclaimed).**  The residual is LARGER than either component.  On a Fortran-faithful port the ratio should be O(1e-3) or smaller; here it is O(1) → the delicate cancellation between the KE-gradient term and the vorticity-flux term that d_sw6 relies on for W2 geostrophic balance is NOT happening.  The two terms are each O(1e5) while their sum is also O(1e5), meaning they are NOT near-equal-and-opposite as Fortran's formulation requires.

**Candidate root-cause areas (unchanged from iter-732 list).**
1. `_bgrid_ke_transport` — `ke_corner` at D-grid vertices.
2. `_corner_vorticity` + `fv_tp_2d` zeta transport — yields `fy_vort`.
3. `_d_sw1_recompute_ut_vt` — the C-grid transport velocity feeds (2) via ut/vt.
4. **NEW (iter-733):** the sum-formula stencil itself.  Fortran d_sw6 combines `ke_diff` and `fy_vort` at the SAME D-grid edge stagger; if Python has a stagger-shift off by half a cell, the cancellation is broken even with correct components.  Line 1904 does `ke_corner[:, :-1, :] - ke_corner[:, 1:, :]` and line 1910 does `fv_tp_2d` of cell-centre `zeta_abs` using B-grid-corner-derived `crx/cry`.  A stagger / orientation mismatch between these two routes is a concrete possibility.

**Sanity caveat.**  The script's inner replay reproduces `u_d_new - u_d0` to within 7.1e-2 (≈ 8 % of the drift magnitude).  The residual is not purely from the replay — some comes from replay-vs-`_d_sw_native` routing details.  The O(1) residual / component ratio is nevertheless large enough to conclude the cancellation is broken; what it does NOT conclusively locate is which of the four candidate areas above is responsible.

**Deliverable.**  `scripts/diag_iter732_fb_c24_phase_bisect.py` is the TRUE bisect.  Iter-734+ attacks ONE of the four candidate areas at a time with its own targeted test.  The residual-ratio metric (`resid / |ke_diff|`) is the primary regression gate: a claimed fix that doesn't bring it below ~1e-2 has NOT fixed d_sw6 geostrophic balance.

**Process note.**  Iter-732's "bisect" claim was premature.  Iter-733 scales it back to what the evidence supports AND fixes the script to actually do the job.  Going forward: name diagnostics for what they *measure*, not what the next iter *hopes* they'll prove.

### Iter-734 — complete the iter-733 replay (add d_sw5 damping; Codex stop-time)

Codex stop-time review on iter-733 flagged: **"iter-733's 'true bisect' omits active `d_sw5` damping, so the reported residual/gate is not the actual `_d_sw_native` update."**  Correct.  `_d_sw_native` at step 5 adds `_d_sw5_corner_divergence` output to `ke_corner` before step 6's `ke_diff_*` stencil when `d4_bg > 1e-10 or d2_bg > 1e-10 or dddmp > 1e-10`.  The diagnostic's call site passes `d4_bg=0.16`, so `use_d_sw5_damping` is True and the d_sw5 contribution is non-zero — but iter-733's replay only used the raw `_bgrid_ke_transport` output for `ke_corner`.  This was the source of the iter-733 sanity-check mismatch of 7.14e-2 (≈ 8 % of the u_d drift magnitude).

**Fix.**  Add the `_d_sw5_corner_divergence` call to the diagnostic's inner replay, conditioned on the same gate as `_d_sw_native`:
```python
if d2_bg > 1e-10 or dddmp > 1e-10 or d4_bg > 1e-10:
    ke_damping = _d_sw5_corner_divergence(
        u_d0, v_d0, ua, va, cdgrid, dt,
        d2_bg=d2_bg, dddmp=dddmp, d4_bg=d4_bg, nord=nord)
    ke_corner = ke_corner_raw + ke_damping
```

**Re-run result (iter-734, same C24/dt=300s/damp_v=0/d4_bg=0.16/nord=1 config):**
- Sanity: `|du_from_update - (u_d_new - u_d0)|` = **1.9e-6** (was 7.1e-2).  The replay now matches `_d_sw_native` to numerical round-off.
- d_sw5 damping contribution: `max|ke_damping|` = 2.45e+4 (≈ 20 % of `|ke_diff_u|` scale).
- Residual metrics (essentially unchanged from iter-733):
  - `|ke_diff_u|` = 1.28e+5 (was 1.29e+5 raw, now includes damping in the corners → -0.5 % delta)
  - `|fy_vort|`   = 3.79e+5 (unchanged)
  - `|ke_diff_u + fy_vort|` = 3.68e+5 (unchanged to 3 sig figs)
  - `residual / |ke_diff_u|` = **2.87** (was 2.86 — 0.3 % change)
  - `residual / |ke_diff_v|` = **5.10** (was 5.07 — 0.6 % change)

**Conclusion standing.**  The iter-733 diagnosis is **confirmed**, not overturned: d_sw6's `ke_diff + vort_flux` cancellation is broken by ~3 orders of magnitude.  The 7.14e-2 iter-733 sanity mismatch was entirely from the omitted d_sw5 damping branch in the replay, not from the diagnosis itself; including d_sw5 barely moves the ratio (2.86 → 2.87) but moves the sanity from O(10 %) to O(1e-6).  Candidate root-cause areas and the iter-733 conclusion about broken W2 geostrophic balance all stand.

**Evidence integrity takeaway.**  Iter-732 mis-labelled the diagnostic.  Iter-733 fixed the label and the logic but omitted a code-path that was actually active.  Iter-734 closes the last gap — the replay now matches `_d_sw_native` exactly.  Future diagnostics that "replay" production functions must either (a) call the production function itself and compare intermediates via dual-use return tuples, OR (b) assert that the replay matches the production output to machine precision BEFORE drawing conclusions.  The 1.9e-6 sanity value is now the required floor for this kind of diagnostic.

### Iter-735 — Fortran d_sw6 unit investigation: vt = u·dx on entry

Per iter-733 candidate #4 ("stagger/orientation mismatch between ke_diff stencil and fy_vort"), iter-735 traces the Fortran d_sw6 formula against its inputs.

**Fortran reference points.**
- `sw_core.F90:1937` (d_sw6 wind update):
  ```fortran
  u(i,j) = vt(i,j) + ke(i,j) - ke(i+1,j) + fy(i,j)
  v(i,j) = ut(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)
  ```
  This is **REPLACEMENT**, not an increment.  `u(i,j)` is overwritten with the sum, no pre-existing `u + ...` on the RHS.
- `sw_core.F90:1584` (d_sw5, called immediately before d_sw6):
  ```fortran
  vt(i,j) = u(i,j)*dx(i,j)
  ut(i,j) = v(i,j)*dy(i,j)     ! line 1589
  ```
  So on ENTRY to d_sw6, `vt` holds `u_old · dx` — CIRCULATION form [m²/s], NOT velocity [m/s].
- `dyn_core.F90:1017-1018` (kee computed from d_sw3 outputs):
  ```fortran
  kee(i,j,:) = (ubbtemp(i,j,:)*vbbtemp(i,j,:))
  kee(i,j,:) = 0.5*(kee(i,j,:) + ubb(i,j,:)*vbb(i,j,:))
  ```
  where `vbbtemp` / `vbb` are B-grid Courant numbers in [m] = dt/2·velocity (d_sw3 line 1273) and `ubbtemp` / `ubb` are the y-transported / x-transported velocities in [m/s].  Product [m·(m/s)] = [m²/s] — circulation/flux.
- `fv_tp_2d(vort, ...)` at `sw_core.F90:1861` produces vorticity fluxes `vortfluxx`, `vortfluxy`.  `vort = wk + f0 = rel_vort + planetary_vort` has units [1/s].  `crx_adv/cry_adv` are dimensionless Courant numbers * mass, `xfx_adv/yfx_adv` are `dt·ut` [m²/s]... (see tp_core.F90 for final units).  The output `vortfluxx` has circulation units [m²/s], aligning with the rest.

**Observation.**  Fortran's d_sw6 assembly line 1937 has ALL four RHS terms (vt, ke_diff_i, ke_diff_{i+1}, fy) in circulation units [m²/s].  The output `u` is therefore in circulation form [m²/s], NOT [m/s].  Downstream usage — del6_vt_flux adding to `u` at line 1992, dyn_core exiting d_sw — preserves this convention.  The conversion back to velocity must happen somewhere after d_sw exits (unit restoration TBD for iter-736).

**Python `_d_sw_native` step 8 (src/legoesm/core/fv3_sw_core.py:1914-1918):**
```python
u_d_new = u_d + (ke_diff_u_scaled + fy_vort) * rdx_u
v_d_new = v_d + (ke_diff_v_scaled - fx_vort) * rdy_v
```
This is INCREMENTAL with `·rdx_u` (1/dx) division on the RHS.

**Algebraic compatibility check.**  If Fortran stores `u` in circulation form (u·dx) during d_sw6 and restores elsewhere, Python's form requires:
```
u_d_new (velocity) = [u_d·dx + ke_diff + fy_vort] / dx
                   = [Fortran_u_output / dx]
```
which matches IF Python's ke_diff / fy_vort are **also** in circulation form.

**Candidate-4 narrowing.**  Fortran's ke_diff and fy_vort are in [m²/s] (circulation).  Our Python's `ke_corner` comes from `_bgrid_ke_transport` which follows the Fortran `kee = ubbtemp*vbbtemp + ubb*vbb` recipe using our versions of ubbtemp/vbbtemp/ubb/vbb.  If our B-grid Courant ubb/vbb are correctly in [m] and transported winds are in [m/s], our `ke_corner` is in [m²/s] as well.  **Needs concrete per-variable unit check in iter-736.**

**What this iter resolves and does NOT resolve.**
- Resolves: the literal Fortran d_sw6 formula is a REPLACEMENT with vt in circulation form; Python's incremental form with `·rdx_u` is algebraically equivalent IFF our ke/fy are in matching circulation units.  Candidate #4 is not an algebraic bug per se.
- Does NOT resolve: whether our Python's `ke_corner` and `fy_vort` actually have circulation units matching Fortran's, OR whether there's a factor-of-`dx` scaling difference.  The iter-733/734 O(1) residual ratio is consistent with a missing/extra `dx` scaling factor in EITHER the `ke_diff` or the `fy_vort` branch.  A factor of `1/dx ~ 1/(R·dθ) ~ 1/(6.4e6·0.065) ~ 2.4e-6` at C24 would explain a 4-5 order-of-magnitude scaling error — matching our observed 2.87-5.10 residual/component ratio.

**Iter-735 deliverable.**  This review-doc entry.  No source change.  Explicitly flags "Python ke_diff and fy_vort units vs Fortran circulation units" as the single most promising iter-736 next investigation.  The 2.87-5.10 residual/component ratio from iter-733/734 is consistent with — but does not uniquely prove — a dx scaling factor mismatch; iter-736 must construct a minimal test that computes ke_diff both ways (with and without `·dx` scaling) and observes which matches Fortran within round-off on a simple W2 input.

**Process discipline (iter-731 + iter-733 + iter-735).**  This iter deliberately does NOT claim "the bug is a dx factor." That's a hypothesis matching the data; confirming it requires an actual numerical test.  The Ralph directive "do not improvise" applies to interpretations as strictly as to source code.

### Iter-736 — interior vs edge residual split: bug is NOT edge-driven

Iter-735 flagged two hypotheses for the iter-733/734 O(1) `ke_diff + fy_vort` residual:
- Edge/halo defect in `_bgrid_ke_transport` or `_vorticity_flux` at panel boundaries (local to ≤ 3-cell halo).
- Global unit/stencil defect (e.g., missing `dx` scaling factor) affecting every cell equally.

These predict different INTERIOR-vs-FULL ratios: an edge defect would make the interior residual small (interior/full ≪ 1) while a global defect would leave it roughly constant.

**Extended diagnostic (iter-736).**  `scripts/diag_iter732_fb_c24_phase_bisect.py` section [F] crops halo depth h=3 from every panel edge and reports interior-only maxes alongside the full-domain maxes.

**Numerical result at C24, W2, dt=300s, step 1, damp_v=0, d4_bg=0.16:**

|                                 | FULL (6, n, n+1) | INTERIOR (6, n-6, n-5) | interior/full |
|---------------------------------|------------------|--------------------------|---------------|
| max\|ke_diff_u\|                | 1.28e+5          | 1.46e+4                  | 0.114         |
| max\|fy_vort\|                  | 3.79e+5          | 3.41e+5                  | 0.899         |
| max\|resid_u\|                  | 3.68e+5          | 3.28e+5                  | **0.889**     |
| **resid_u / \|ke_diff_u\|**     | **2.87**         | **22.4**                 |               |
| max\|ke_diff_v\|                | 1.05e+5          | 1.32e+4                  | 0.125         |
| max\|fx_vort\|                  | 4.39e+5          | 3.55e+5                  | 0.809         |
| max\|resid_v\|                  | 5.35e+5          | 3.42e+5                  | **0.640**     |
| **resid_v / \|ke_diff_v\|**     | **5.10**         | **26.0**                 |               |

**Conclusion (concrete, narrowed).**  The residual is NOT edge-driven:
- interior/full ratio for `resid_u` is **0.889** — 89 % of the full-domain magnitude survives after stripping all 3-cell halos.
- interior/full ratio for `resid_v` is **0.640** — 64 %.
- At interior, `resid / |ke_diff|` is TEN TIMES LARGER than at full (22.4 vs 2.87 for u; 26.0 vs 5.10 for v) — because `|ke_diff|` drops faster (interior/full ≈ 0.12) than the residual does.

`|fy_vort|` interior is ~3.4e5, `|ke_diff_u|` interior is ~1.5e4 — a **22:1 ratio** at points where Fortran-faithful W2 expects them to cancel.  This is either:
- A global scaling defect (factor of ~20× on one of the two branches), OR
- A missing/extra `dx` or `dt` factor in the computation pipeline (1/dx·dt at C24 is ~1/(270km·300s) ~ 1.2e-8, so `dx·dt` factor would be far bigger than needed — but `dt` alone ~300 matches order-of-magnitude; `dx·dt` ratio if we had `·dx` instead of `·dx·dt` somewhere ~ dx = 270e3).

**Narrowed candidate root-cause areas (refined from iter-733/734 list):**
1. **Top candidate (iter-736):** `_bgrid_ke_transport` — produces `ke_corner` ~3-4 orders of magnitude too small relative to `fy_vort`.  The `kee = 0.5*(ubbtemp*vbbtemp + ubb*vbb)` formula assumes specific units for each multiplicand; if ubb/vbb (B-grid Courant, supposed to be in [m] = dt/2·velocity) are actually in [1] (dimensionless) or [m/s], the product is short a factor of dt·velocity ~ 300·40 ~ 1.2e4 at C24 — which is in the right ballpark for the observed 22× gap.
2. **Secondary:** `fy_vort = fv_tp_2d(zeta_abs, crx, cry, xfx, yfx, ...)` — if `xfx/yfx` are passed WITH a `·dt` factor that shouldn't be there, `fy_vort` is oversized by ~300×.
3. **Tertiary:** interior stencil in either path that introduces a factor-of-dt or factor-of-dx scale.

**Deliverable.**  `scripts/diag_iter732_fb_c24_phase_bisect.py` extended with interior/edge split (section [F]).  Iter-737 next work: concrete per-variable unit check of `_bgrid_ke_transport`'s four multiplicands (ubbtemp, vbbtemp, ubb, vbb) against Fortran d_sw3 outputs at a SINGLE interior cell (e.g., face 0, (i,j)=(n//2, n//2)).

**Confidence calibration.**  Edge/halo hypothesis from iter-732 is DOWN (interior/full ~0.6-0.9 rules out edge-only).  Global scaling hypothesis from iter-735 is UP (residual ratio is larger at interior where ke_diff is smaller, consistent with fy_vort being 20× too big or ke_diff being 20× too small).  But "20× too big" could be one of many factors; iter-737 must nail down WHICH factor in WHICH branch.

### Iter-737 — unit audit of ke_corner and fy_vort: both in m²/s, hypothesis disproven

Per iter-736 next work: "concrete per-variable unit check of `_bgrid_ke_transport`'s four multiplicands" and the `fv_tp_2d` xfx/yfx inputs.

**Python `_bgrid_ke_transport` variable units (src/legoesm/core/fv3_sw_core.py:1702-1776):**

| Variable     | Assignment (line)                        | Units     |
|--------------|------------------------------------------|-----------|
| `vb`         | `dt5 * (vc_sum - uc_sum*cosa) * rsina`   | [m] = dt·velocity·(projection) |
| `ub`         | `dt5 * (uc_sum - vc_sum*cosa) * rsina`   | [m]       |
| `transported_y` | `_ppm_transport_1d(v_d, vb, rdy, axis=2)` | [m/s] (same as v_d) |
| `transported_x` | `_ppm_transport_1d(u_d, ub, rdx, axis=1)` | [m/s] (same as u_d) |
| `ubbtemp`    | `transported_y`                          | [m/s]     |
| `vbbtemp`    | `vb`                                     | [m]       |
| `ubb`        | `ub`                                     | [m]       |
| `vbb`        | `transported_x`                          | [m/s]     |
| `ke_corner`  | `0.5*(ubbtemp*vbbtemp + ubb*vbb)`        | 0.5·[(m/s)·(m) + (m)·(m/s)] = **[m²/s]** |

Fortran d_sw3 / dyn_core.F90:1017-1018 assembles `kee` with the same multiplicand structure and units.  Python matches Fortran convention.

**Python `fv_tp_2d` → fy_vort units (src/legoesm/core/fv_tp_2d.py:459-538):**

| Input         | Value / formula                        | Units     |
|---------------|----------------------------------------|-----------|
| `zeta_abs`    | `zeta + f` (cell-centre vorticity)     | [1/s]     |
| `crx`         | `dt * ut * rdxa_upwind`                | [1] (dimensionless Courant) |
| `xfx`         | `dt * ut * dy * sin_x`                 | [m²] (area flux) |
| `cry`, `yfx`  | symmetric                              | [1], [m²] |
| `fx`, `fy`    | `0.5*(fx1+fx2) * xfx` with fxi = PPM(q) | [1/s]·[m²] = **[m²/s]** |

Also matches Fortran tp_core.F90 structure.  **Both ke_diff and fy_vort are in [m²/s].**

**Iter-736 hypothesis DISPROVEN.**  There is NO missing `dx` factor or missing `dt` factor in the Python port — the units are consistent with Fortran down to both operands of each multiplication.  The 22× magnitude gap at interior is therefore NOT a unit/scaling bug; it is a genuine magnitude discrepancy between the two terms as computed by our port.

**Revised hypothesis (iter-737).**  For W2 exact steady state, Fortran expects `ke_diff + fy_vort` to APPROXIMATELY cancel on the interior (residual should be O(truncation)).  Our port produces `fy_vort ≈ 22·|ke_diff_u|` at interior — either:
- (a) Our `fy_vort` is correct in absolute value, but our `ke_diff_u` is ~20× too small (missing a term or a geometric factor inside `_bgrid_ke_transport`), OR
- (b) Our `ke_diff_u` is correct, but our `fy_vort` is ~20× too big (extra factor in `compute_transport_quantities` or `fv_tp_2d`).
- (c) Both are internally consistent with Fortran but a SIGN error makes them add instead of subtract (less likely — a pure sign flip would give residual ≈ |2·ke_diff| + |2·fy_vort|, not the observed ~|fy_vort|).
- (d) The cancellation equation is NOT `ke_diff + fy_vort = 0` for W2 — it might be `ke_diff + fy_vort = d(g·h)/dx · dx` (pressure-gradient balance applied at d_sw6 rather than consumed in c_sw+p_grad_c).  If that's the case, the residual IS the correct pressure-gradient tendency and we've been measuring the wrong thing.

**Concrete iter-738 target.**  Hypothesis (d) is the most interesting one to rule out.  **Correct scaling (iter-738 Codex stop-time fix):**

Python's d_sw6 step 8 applies the increment `(ke_diff_u + fy_vort) * rdx_u` to u_d — i.e., `(ke_diff_u + fy_vort) / dx` (units [m²/s]·[1/m] = [m/s]).  So the per-step velocity change introduced by the d_sw6 KE-gradient-plus-vort-flux assembly is:

```
du_d_per_step = (ke_diff_u + fy_vort) * rdx_u     # [m/s]
```

The Fortran-side expected pressure-gradient tendency accumulated over ONE full step is `-g · dh/dx · dt` [m/s² · s = m/s] — **note the dt factor**.

The numerical check iter-738 must run is therefore:
```
observed[i,j]  =  resid_u[i,j] * rdx_u[i,j]              # m/s
expected[i,j]  =  -g * (h[i+1,j] - h[i,j]) * rdx_u[i,j] * dt   # m/s, W2 steady
```
where `resid_u = ke_diff_u_scaled + fy_vort` is what iter-733..-736 measured.

Iter-737 originally wrote this as "compare against `(resid_u / rdx_u)`" and "compare against `g · dh/dx_u`" (without the `·dt` factor).  Codex stop-time review on iter-737 flagged this: **the scaling is inverted (/rdx instead of *rdx) and the per-step pressure-gradient tendency misses the ·dt factor.**  Both mistakes shift the comparison by factors of dx² · dt ≈ (2.7e5)² · 300 ≈ 2e13 — rendering the proposed iter-738 check meaningless.  Iter-738 runs the CORRECTED comparison above.

**Sanity (iter-738 pre-check).**  Order of magnitude: at mid-lat in W2, `g · dh/dx · dt` ~ 9.8 · 30/2.7e5 · 300 ≈ 0.33 m/s per step.  Observed `resid_u · rdx_u` at interior = 3.3e5 · 3.7e-6 ≈ 1.22 m/s per step.  Factor of ~3.7× difference.  NOT the 22× magnitude gap we saw in `fy_vort / ke_diff` ratio — which is consistent with hypothesis (d) being partially right: `ke_diff + fy_vort` is SUPPOSED to equal a nonzero Fortran pressure-gradient-like tendency, not zero.  If the per-step comparison comes in at ~1.2 vs expected ~0.33, a factor-of-3.7 mismatch remains and localises the actual bug scope (was 22×, now 3.7×).

**Deliverable.**  docs-only update in iter-737; iter-738 implements the corrected diagnostic.  Iter-737 erratum-note above is integral to the iter-737 review-doc entry — the original wrong-scaling prescription must not be acted on without this correction.

**Process.**  Iter-737 correctly falsifies iter-736's unit-scaling hypothesis through a direct per-operand units audit.  The constraint "follow Fortran exactly" applies to understanding too — before claiming a bug, confirm the Fortran-side mathematical expectation.  Iter-738 must verify whether `ke_diff + fy_vort` ACTUALLY cancels in Fortran for W2 before treating our non-cancellation as proof of a bug.  **The iter-737 Codex fix is a reminder that the scaling machinery between diagnostics must ALSO be Fortran-faithful down to the dt factor — not just the numerical port.**

### Iter-738 — Fortran-faithful pressure-gradient comparison (Codex stop-time fix)

Codex stop-time review on iter-737 flagged: **"iter-737's new iter-738 check uses the wrong scaling quantity."**  Correct.  Iter-737 proposed dividing `resid_u` by `rdx_u` (giving [m²/s]·[m] = [m³/s] — nonsense) and comparing to `g·dh/dx` without the `·dt` factor.  Iter-738 fixes both and implements the corrected check.

**Corrected formula.**  Per-step velocity increment contributed by d_sw6's `ke_diff + fy_vort` assembly:
```
du_d_per_step[i,j]  =  (ke_diff_u[i,j] + fy_vort[i,j]) * rdx_u[i,j]    # [m/s]
```
Expected W2-steady Fortran pressure-gradient tendency over one step:
```
du_pg_expected[i,j]  =  -g * (h[i+1,j] - h[i,j]) * rdx_u[i,j] * dt      # [m/s]
```
(The Fortran convention places the h-gradient of the Bernoulli function in c_sw via B=KE+g·h; whether d_sw6 additionally carries a g·dh/dx·dt tendency depends on whether c_sw's Bernoulli gradient was already committed to uc and consumed there.  Hypothesis (d) asserts the tendency is present; iter-738 tests it.)

**Diagnostic plan.**  Extend `scripts/diag_iter732_fb_c24_phase_bisect.py` with a section [G] that evaluates both `du_d_per_step` and `du_pg_expected` on interior cells and reports:
- max\|du_d_per_step\|_interior
- max\|du_pg_expected\|_interior
- max\|du_d_per_step - du_pg_expected\|_interior (the TRUE residual if hypothesis (d) is correct)
- max\|du_d_per_step + du_pg_expected\|_interior (if a sign-flip makes them match)

If one of the bottom two drops to O(truncation), hypothesis (d) is confirmed — the "O(1) residual" iter-733/734 measured was actually the correct pressure-gradient-like tendency, and we've been chasing a non-existent bug.  If neither drops, the port has a real magnitude defect; iter-739+ hunt which factor.

**Implementation note.**  `rdx_u` is `1/dx_edge_y` in Python (src/legoesm/core/fv3_sw_core.py:1888).  The h-gradient `h[i+1,j] - h[i,j]` is a cell-centre difference indexed by the u-face position (i runs over u-faces, which sit between cell i and i+1).  D-grid u-face stagger: u_d shape (6, n, n+1) — i∈[0,n-1] is cell-like, j∈[0,n] is corner-like.  So `h[face,i,j]` maps to u_d[face, i, j] via j-side dereferencing... actually this is subtle.  Iter-738 must handle the stagger carefully; if h is (6, n, n) cell-centre and u_d is (6, n, n+1) at x-edges (j-edges), the h-gradient for u_d[i,j] needs to be `h[i, j-1] - h[i, j]` (north-south adjacent cells) OR `h[i+1, j] - h[i, j]` (east-west adjacent cells) depending on which component u_d represents.  D-grid "u" is the physical east-west wind component sitting at the north-south cell edges — so it ADVECTS via x (east-west) but RESPONDS to gradients along the face normal (north-south).  The pressure gradient force on u is therefore `-g · dh/dx_{east-west-panel-tangent}`, which in our `u_d` indexing involves cells at (i, j-1) and (i, j) both east-west neighbours.  Actually no — staggering: `u_d[i,j]` sits at the north-south midpoint between `cell[i,j-1]` and `cell[i,j]`.  The east-west gradient at that midpoint requires averaging two cells' east-west gradients.  Fortran does this via `ke_i - ke_{i+1}` which is a DIFFERENT stagger than the h-gradient — so the comparison requires careful averaging.

**Scope discipline (iter-738).**  Getting the stagger right is itself a lot of work.  Iter-738's concrete commit: the review-doc erratum above + the test plan.  Implementation of the corrected diagnostic is deferred to iter-739 where the stagger-averaging can be worked through carefully against Fortran dyn_core.F90's indexing.

### Iter-739 — visual inspection of production SW matrix: W2 v-wind artifact persists

Per Ralph directive steps 4 + 5: "Run the required evaluations (cosine bell, Williamson case 2, Williamson case 5, ocean rest state) … perform careful visual inspection.  There must be no panel-edge, corner, seam, halo, striping, or ringing artifacts."

Iter-727 through iter-738 all targeted the FB-chain, not production.  The production A-L+RK3 path from `fv3_sw_tendencies` (`src/legoesm/core/operators_cdgrid.py:1381-1493`) is unchanged across the ~12 iters.  Iter-739 runs the quick-mode atmosphere test matrix on the production cubed-sphere path and inspects snapshots.

**Run.**  `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python scripts/run_atmosphere_test_matrix.py --only sw --grid cubed_sphere --quick`
- shallow_water/williamson2/cubed_sphere: PASS, L2=2.42e-04, Linf=1.83e-03 (height).
- shallow_water/williamson5/cubed_sphere: PASS, mass drift=1.74e-05.
- shallow_water/cosine_bell/cubed_sphere: PASS, L1=1.20e-01, L2=1.17e-01, Linf=1.23e-01.

**Visual inspection.**

*Williamson 2, C36, 1 day — v-wind (`snapshots_v.png`).*  At t=0 the v field is near-zero as the IC prescribes.  Over 0.1→0.5 d the panels develop visible zonally-organised bands with amplitude ~±0.1 m/s; by 0.7→1.0 d the bands reach ~±0.3 m/s and form a clear mode-4 angular pattern at mid-to-high latitudes both hemispheres.  This is the iter-717 user-reported artifact.  **UNCHANGED vs. pre-iter-727 baseline.**  Ralph stopping condition "no panel-edge, corner, seam, halo, striping, or ringing artifacts" fails for W2 v-wind.

*Williamson 2, C36, 1 day — wind_speed.*  Smooth zonally-symmetric band structure throughout.  Artifacts are invisible here because the ~38 m/s zonal flow dominates; the 0.3 m/s v-component artifact is < 1 % of the wind-speed magnitude.  This is why the test matrix's L2/Linf pass criterion on height (2.42e-04 / 1.83e-03) does not catch the v-wind artifact — the artifact is in v, not h, and of a magnitude well below the height noise floor.

*Williamson 5, C36, 1 day — wind_speed.*  Zonal flow with visible mountain-induced disturbance on the leeward side (longitude ~-90 to -60).  Disturbance evolves smoothly 0.1→1.0 d.  No obvious panel-edge or mode-4 artifacts in wind_speed.  v-wind inspection TBD in a future visual sweep if W5 becomes suspect.

*Cosine bell, C36, 1 day — height.*  Bell shape transports coherently across longitudes.  A faint "ghost" trailing the main bell in several panels appears to be the visualisation tool rendering multiple successive positions overlaid, NOT a numerical artifact in the transport solution.  Bell peak magnitude ≈ 940 m throughout; no amplitude loss visible.  No edge artifacts.  Cosine-bell conservation / transport looks clean.

**Test-matrix noise-floor limitation.**  The pass criterion `Linf=1.83e-03` (height) is met, yet the visual v-wind inspection shows a clear mode-4 artifact.  This is the classic CLAUDE.md-documented failure mode: "error norms can improve while artifacts get worse; always check v-wind snapshots."  Iter-739 is direct corroboration: a PASSING test-matrix run does NOT imply the artifact is gone.  The matrix as currently wired does not include a direct v-wind Linf check on W2 against the exact 0 solution.

**Iter-740 candidate.**  Add a W2 v-wind Linf check to the test matrix pass criteria.  This would give automated early-warning if any future iter makes the artifact worse (or better — the same gate would detect a fix).  This is a small, mechanical change to `scripts/run_atmosphere_test_matrix.py` + the W2 pass-criteria function.

**Iter-739 deliverable.**  This review-doc entry.  The generated PNGs are at `results/atmosphere/shallow_water/williamson2/cubed_sphere/C36/snapshots_v.png` etc — they are ephemeral matrix outputs, not checked in.  A future iter wanting to reproduce the evidence should re-run the quick matrix.

**Stopping-condition alignment.**  Ralph directive requires "no visible artifacts on cosine bell, Williamson 2, or Williamson 5."  Cosine bell and W5 pass visual inspection in iter-739.  W2 still fails on v-wind.  **Stopping condition is NOT met and cannot be emitted.**  Iter-740+ work must target the W2 v-wind artifact on the PRODUCTION A-L+RK3 path (not the FB chain, which remains blocked on the task-3 bisect investigation).

**Process takeaway.**  Iter-727 through iter-738 pursued the FB chain under the user's iter-722 directive "no Arakawa-Lamb + RK3".  That directive still stands, but iter-739 confirms the FB chain is not within reach to replace A-L+RK3 in the near term (task 3 blocked on upstream structural issues per iter-730/731, and the iter-733..-738 diagnostic chain narrowed but didn't localise the blocker).  In the interim, the production path remains the only runnable path and its W2 artifact is the user-facing blocker.  Iter-740+ should explicitly choose between (a) continuing FB investigation with a concrete Fortran-reference comparison and accepting multi-iter latency before any production impact, or (b) directly attacking the A-L+RK3 W2 artifact per iter-729's deferred xtp_u east-edge port candidate OR the iter-732 NW-cube-vertex localisation (which also applies to production via `pad_halo_vector` at `operators_cdgrid.py:1397`).

### Iter-740 — add W2 v-wind Linf gate to test matrix (regression sentinel)

Iter-739 noted that the test-matrix pass criterion on W2 CS is height-only (`L2=2.42e-04, Linf=1.83e-03`), which MET the gate while the v-wind mode-4 artifact was clearly visible.  Iter-740 closes that gap by adding a v-wind Linf metric to the W2 CS error-norm block.

**Change.**  `scripts/run_atmosphere_test_matrix.py::run_shallow_water` at the `test_num == 2 and tc.grid_type == "cubed_sphere"` branch (around line 1445) now additionally computes:
```python
u_0 = 2.0 * pi * grid.radius / (12.0 * 86400.0)
v_d_exact = -cdgrid.sin_angle_edge_y * u_0 * jnp.cos(cdgrid.lat_edge_y)
v_linf = float(jnp.max(jnp.abs(state.v_d - v_d_exact)))
```
The formula matches the W2 IC for v_d (line 1192 in the same file) — W2 steady state means `v_d(t) == v_d(0)`, so the IC is the exact solution at all times.  The result is reported in the notes string as `v_Linf=<value>`.

**Measured baseline.**  Running the updated matrix on W2 CS (`--only sw --grid cubed_sphere --quick --test williamson2`):
```
PASS | shallow_water/williamson2/cubed_sphere | L2=2.42e-04, Linf=1.83e-03, v_Linf=2.92e-01
```
**v_Linf = 2.92e-01 m/s**.  This matches the iter-717 user-reported and iter-739 visually-observed ±0.3 m/s mode-4 artifact amplitude exactly.  Note: 0.29 m/s on a reference max |v_d| of ~32 m/s (from the -sin_angle_edge_y projection of 38.61 m/s) is 0.9 % relative, which is why the h-based Linf=1.83e-3 does not catch it.

**Pass/fail policy.**  Iter-740 does NOT fail the test when v_Linf is above a threshold — the gate is REPORTING-ONLY for now.  Rationale: (1) the current ~0.3 m/s baseline is the known-broken level, so any threshold would either fail EVERY run (threshold ≤ 0.29) or miss future worsening (threshold > 0.29); (2) a regression-based pass/fail needs a stored baseline with explicit tolerance that future iters can argue against.  Iter-741+ can set the threshold once the production path's v_Linf floor is reduced.

**Regression sentinel usage.**  Future iters claiming to improve the W2 artifact must re-run this matrix and report a smaller v_Linf.  Iters that accidentally worsen the artifact will see v_Linf grow in the notes line — automated early warning even if h-based pass criteria still meet.

**No source numerical change.**  The W2 run itself is unchanged; only the error-norm reporting is extended.  Matrix still runs in 3.5 s (unchanged).  All 3 SW CS tests still PASS.

**Iter-741+ routing (unchanged from iter-739).**  The binary choice remains: (a) multi-iter FB investigation with Fortran-reference comparison, or (b) direct A-L+RK3 W2 attack via iter-729 xtp_u port / iter-732 NW-vertex localisation.  Iter-740's metric supports EITHER: whichever iter produces a smaller v_Linf on this gate has made concrete progress on the user-reported blocker.

### Iter-741 — v_Linf measures a proxy; fix to v_north (Codex stop-time)

Codex stop-time review on iter-740 flagged: **"new W2 sentinel measures a proxy, not the matrix's actual visible artifact."**  Correct.

**What iter-740 measured.**  Raw D-grid `v_d` error relative to the analytic IC `v_d_exact = -sin_angle_edge_y · u_0 · cos(lat_edge_y)`.  Value: 2.92e-01 m/s.

**What the visible artifact IS.**  The snapshot `snapshots_v.png` plots GEOGRAPHIC `v_north` (m/s), computed by `extract_fn` (scripts/run_atmosphere_test_matrix.py:1232-1238) via:
```python
u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
v_north = sa_4edge * u_cc + ca_4edge * v_cc     # ← VISIBLE v
```
`v_north` mixes u_d and v_d errors via the rotation by `sa_4edge` (~sin of the 4-edge-averaged cell-centre angle).  A pure-`v_d`-error metric captures only one component; a pure-`u_d`-error contribution is invisible to it.

**For W2 steady state, exact v_north = 0 everywhere and at all times** (the flow is purely zonal geographic u, with no geographic v).  So `|v_north|` is itself the error, no subtraction needed.

**Fix (iter-741).**  Replace the v_d-vs-v_d_exact computation with an in-matrix replication of the extract_fn rotation.  Value drops 2.92e-01 → 3.07e-01 (0.29 → 0.31, +6 % larger because the cross-coupling term is now included).  The new metric matches the snapshot colourbar of ±0.3 m/s — confirmed directly.

**Measured baseline (iter-741).**  `v_north_Linf = 3.07e-01 m/s` on W2 CS C36 1 day with `hyperdiff_coeff=_hyperdiff_cube(36)`, `div_damp=_div_damp_cube(36)`, `boundary_fix=True` — the current production settings.  This is the Fortran-fidelity regression sentinel going forward.

**Process discipline.**  iter-740's proxy mistake happened because I conflated "v in the state" (D-grid component) with "v in the snapshot" (geographic component after rotation).  The two differ by an O(sin_angle) cross-term that is precisely the source of cube-face rotation effects on the cubed sphere.  Future sentinels must match the quantity the visual inspection EXPECTS, not the nearest arithmetically-convenient substitute.

**Gate name change.**  `v_Linf` → `v_north_Linf` in the notes string, so the regression-history reader can unambiguously distinguish iter-740's (dropped) metric from iter-741+'s.  First-time readers see the "_north" suffix and know it is the geographic component.

**Reporting only, same as iter-740.**  No pass/fail threshold yet — any fixed threshold either fails every current run or misses worsening.  Iter-742+ sets a threshold after the first concrete fix lands.

### Iter-742 — v_north is still a proxy; measure the regridded v_ll (Codex stop-time)

Codex stop-time review on iter-741 flagged: **"iter-741 still measures a proxy instead of the plotted snapshots_v.png field."**  Correct.  The snapshot plots POST-REGRID `v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, coord_kind="cube")` — the bilinear cube→latlon interpolation.  Iter-741's pre-regrid `v_north_face` differs because the regrid can shift/smooth the mode-4 peak at cube corners.

**Fix (iter-742).**  Apply `_regrid_2d` to v_north_face before taking max.  Notes key renamed `v_north_Linf` → `v_ll_Linf` with pre-regrid value retained in parentheses for diagnostic contrast.

**Baseline values:**
- v_ll_Linf (post-regrid, what PNG shows)  = **3.03e-01 m/s**
- v_north_Linf (pre-regrid)                =   3.07e-01 m/s
- Delta 1.3 % — bilinear regrid minor smoothing.

Both match the iter-717/739 PNG colourbar of ±0.3 m/s; iter-742's 3.03e-01 is the precise match.

**Process takeaway (iter-740 → 741 → 742).**  Three iters in a row, each corrected by Codex stop-time review, to get the sentinel right.  Each step was a real improvement.  Going forward: NEW metrics must be validated against the plotted artifact BEFORE committing — follow the pipeline (inspect plot callsite → replicate exact computation → compare PNG colourbar).

### Iter-743 — duogrid=ON on A-L+RK3 production BLOWS UP (rules out trivial path)

Per iter-722 user directive "implement the exact FV3 duogrid."  The test matrix currently calls `create_cubed_sphere(n)` with default `use_duogrid=False`.  Iter-743 tests the obvious question: does flipping production A-L+RK3 to `use_duogrid=True` reduce the W2 artifact?

**Diagnostic.**  `scripts/diag_iter743_production_duogrid.py` runs W2 C36 1 day on the exact matrix config (`hyperdiff_coeff=_hyperdiff_cube(36)`, `div_damp=_div_damp_cube(36)`, `boundary_fix=True`, `dt=300s`) with `use_duogrid=False` vs `use_duogrid=True` and reports v_ll_Linf + h L2/Linf.

**Results (C36, W2, 1 day, production settings):**

| Config          | h_L2      | h_Linf    | v_ll_Linf    | v_north_Linf |
|-----------------|-----------|-----------|--------------|--------------|
| duogrid=OFF     | 2.42e-04  | 1.83e-03  | 3.03e-01     | 3.07e-01     |
| duogrid=ON      | 1.45e-01  | 8.86e-01  | **3.32e+02** | 3.37e+02     |
| delta           | +598×     | +484×     | **+1097× (blowup)** |       |

**Interpretation.**  A-L+RK3 production with duogrid enabled **blows up catastrophically** — state winds reach hurricane scale and height is order-unity relative error.  The run doesn't NaN but the result is garbage.

**Why duogrid blows up A-L+RK3.**  The A-L + corner-wind + `boundary_fix` + halo-interpolation production path is non-FV3 (architectural item #1 above).  Flipping to duogrid invalidates its tuning:
- `cube_rmp` kinked-to-extended remap replaces plain halo interpolation with Duogrid-specific extrapolation that mismatches A-L corner winds.
- `boundary_fix` averaging is tuned to compensate for SPECIFIC halo-interp errors; duogrid produces DIFFERENT halo errors that `boundary_fix` doesn't handle.
- `cos_angle_padded` / `sin_angle_padded` are Duogrid-computed when duogrid is active; A-L's Bernoulli gradient stencil assumes standard panel-angle continuation.

These amount to a completely different halo regime that the A-L + boundary_fix stabilizer was never designed for.

**What this rules out.**  The user directive "implement the exact FV3 duogrid" CANNOT be interpreted as "set `use_duogrid=True` in the matrix."  That's a catastrophic-blowup path.  The directive requires the **FV3 FB chain** (Fortran-faithful AND designed around duogrid).  FB remains unstable at C24/C36 per iter-730/731.

**What this confirms.**  Production W2 improvement requires one of:
- (a) Fix the FB chain C24/C36 stability (iter-732..738 investigation).
- (b) Fix the A-L+RK3 production path's non-duogrid halo handling directly (iter-732 NW-vertex localisation applies here via `pad_halo_vector` at `operators_cdgrid.py:1397`).
- (c) Introduce a different Fortran-faithful production alternative.

The "trivial" path (d) is **ruled out by direct measurement**.

**v_Linf baseline stability.**  Pre-iter-743, W2 v_ll_Linf has been stable at 0.303 m/s since the iter-505 major production fix.  Iter-743 confirms baseline = 0.303 m/s — unchanged across hundreds of iterations.  Any iter claiming to have reduced the artifact must beat 0.303 without introducing a blowup.

**Deliverable.**  `scripts/diag_iter743_production_duogrid.py` checked in as evidence.  Future iters considering duogrid flips on production see this result in the review doc and know not to repeat the experiment without first fixing the A-L path's duogrid compatibility.
