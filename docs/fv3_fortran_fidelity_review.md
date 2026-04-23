# FV3 Fortran Fidelity Review

Baselined 2026-04-14. Older prose is aggressively condensed to
save tokens. Only the newest Ralph-loop tail remains in full
form below.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Live status

- **W2 v-wind artifact at C36 remains unresolved.**
  Production still runs `FV3EdgeShallowWaterModel` ->
  `fv3_sw_tendencies` (Arakawa-Lamb + RK3 + `boundary_fix`), not
  the FV3 FB chain. Current canonical W2 baseline after iter-505:
  `L2=2.42e-04`, `Linf=1.83e-03`, `max|v_ll|≈3.03e-01 m/s`.
- **FB-path C24/C36 stability remains unresolved.**
  Halo=3 scaffolding is partly in place, but the FB chain still
  needs full h=3 caller rollout and the remaining stability work.

## Closed priorities

- **Panel-edge corner metrics**: resolved. `cosa_corner` /
  `sina_corner` / `rsin2_corner` now match the Fortran seam
  construction at all 24 seams.
- **d_sw3 BGRID_NE sync**: resolved. Python now routes the corner
  sync through the geographic-frame seam handler instead of the old
  scalar KE fallback.
- **Non-duogrid `_d2a2c_vect` cube-vertex gap**: isolated, not
  fixed. It is architectural and requires halo=3, but it does not
  affect the default A-L production path.
- **Old W2 polar face-4 vs face-5 asymmetry**: resolved by the
  iter-505 PPM-axis fix.

## Key production fix

Iter-505 fixed the major production-path bug in
`cgrid_mass_flux_divergence`: x-direction strips were being passed
to `_ppm_reconstruct_1d` with the wrong active axis. Impact on
canonical W2 C36 dt=300s 1d:

- `L2`: `1.53e-3 -> 2.42e-4`
- `Linf`: `4.07e-3 -> 1.83e-3`
- `max|v_ll|`: `0.557 -> 0.303 m/s`

## Historical archive

Everything before `iter-772` is intentionally compressed here.

- `iter-1..174`: core FV3 metric/operator port, seam/sync work,
  regression expansion, FB-chain brought up but still unstable.
- `iter-505..510`: production-path PPM-axis fix; old polar
  asymmetry closed.
- `iter-511..729`: W2/W5 artifact characterization, formula locks,
  halo=3 plumbing, FB diagnostics, structural test hardening, and
  production-vs-FV3 routing clarified.
- `iter-730..741`: FB C24 stability sweep plus production W2
  artifact measurement pipeline tightened from raw `v_d` to
  post-regrid `v_ll`.
- `iter-742..745`: locked the correct user-visible `v_ll`
  sentinel, proved `use_duogrid=True` is a catastrophic production
  blowup, localized the remaining W2 artifact to face-4/5
  near-pole cells, and showed the polar mode requires
  `boundary_fix` plus `hyperdiff`.
- `iter-745b..751`: several mechanism stories were falsified, but
  the stable conclusion held: the production hyperdiff block is
  structurally non-Fortran and the fix direction is vorticity-form
  damping, not scalar bilaplacian on geographic wind components.
- `iter-752..755b`: ported `_del6_vt_flux`, fixed its metrics,
  units, and post-step semantics, and established the real
  Fortran-faithful production best as `damp_v=0.06, nord_v=2`
  with `v_ll_Linf≈2.14e-01` (about 29% below the old `0.303`
  baseline). Earlier tendency-form `0.168` / `-45%` claims were
  retracted.
- `iter-756..759`: visual check confirmed reduced polar bands but
  surviving cube-corner seams; `div_damp` metric use was aligned
  to B-grid `da_min_c`; the isolated `*dt` Smag transplant was
  reverted; and a standalone `d_sw5` corner-divergence helper was
  started.
- `iter-760..763`: switched the matrix default to the
  Fortran-faithful del6 post-step path, reducing canonical W2
  `v_ll_Linf` to about `1.59e-01`; localized the remaining mode A
  artifact to cube corners; and gathered sensitivity evidence that
  corner-fill choice matters at the right order of magnitude.
- `iter-764..767`: three Fortran-inspired cube-corner fill ideas
  were tested on the A-L production path and all made W2 worse:
  directional `fill_4corners`, `a2b_ord4` 3-point scalar average,
  and `fill_corners_agrid_r8` vector swap. The result was that
  these Fortran recipes are not drop-in compatible with the
  current rotate-pad-rotate A-L pipeline.
- `iter-768..771`: the remaining W2 artifact was re-measured as
  mainly dynamical, `boundary_fix` corner smoothing was shown to
  be load-bearing, face-local peak localization was tightened, and
  `grad_c10` sensitivity around the peak-updating D-grid corners
  was identified as a high-leverage remaining production-path
  knob.

Use git history if you need the full older narrative.

## Latest Ralph-loop iterations (full form)

### Iter-772 — Ablation: grad_c10 = 0 at peak-updating D-grid corners

Per iter-771b's revised candidate list ("Ablation: force grad_c10 = 0 at the 16 D-grid corners surrounding the top-4 peak cells"), iter-772 runs the canonical W2 C36 matrix config with the ablation in place and measures v_ll_Linf vs unablated.

**Method** (`scripts/diag_iter772_ablate_c10_peak_corners.py`).  Constructs cdgrid, mutates `grad_c10` to 0 at the 16 D-grid corner positions that surround the iter-770 top-4 peak cells: `{face0 (34,0), face0 (34,35), face2 (34,0), face2 (34,35)}` × 4 surrounding corners each.  Runs the standard W2 loop with the modified metric.

**Measurement** (C36 dt=300s 1d):

| Config                             | h_L2     | h_Linf   | v_ll_Linf |
|------------------------------------|----------|----------|-----------|
| OFF (unablated default)            | 2.07e−4  | 1.53e−3  | 1.59e−01  |
| ON (`c10 = 0` at 16 peak corners)  | **2.18e−3**| **1.54e−2**| **4.96e+00 (31× BLOWUP)** |

**Observation.**  Zeroing `grad_c10` at the 16 peak-updating D-grid corners makes W2 v_ll_Linf ~31× worse, h_L2 ~10.5× worse.  The ablation shows that the ZERO substitution at these corners is strictly worse than the current value.  It does NOT show that the current magnitude of c10 is geometrically optimal — only that the specific substitution `c10 → 0` is harmful.

**Scope limits (iter-772b, Codex stop-time review).**  The ablation tested ONE substitution (full zero) at ONE set of 16 corners.  It does NOT establish:
- That the current c10 magnitude is the best possible at these corners (e.g. a smaller non-zero value might be better).
- That other substitutions (reducing c10 to 0.5× current, zeroing c01 instead, modifying c00/c11) would also be worse.
- That the off-diagonal coupling at these corners is "the correct geometric representation" of the cubed-sphere non-orthogonality — verifying that requires comparing our derivation against the Fortran oracle, which iter-772 did not do.

**What iter-772 established (combined with 765/766/767/769/771).**  Five distinct structural candidates at cube-corner / near-corner D-grid positions have now been tested and found to be either load-bearing or already Fortran-faithful in magnitude:

| Iter | Candidate                                 | Outcome                       |
|------|-------------------------------------------|-------------------------------|
| 765  | directional cube-corner halo fill         | 12× blowup — FALSIFIED        |
| 766  | a2b_ord4 3-pt corner avg                  | 1.89× blowup — FALSIFIED      |
| 767  | vector-swap cube-corner halo fill         | 16× blowup — FALSIFIED        |
| 769  | skip boundary_fix corner smoothing        | 6.5× blowup — CONFIRMED LOAD-BEARING |
| 772  | zero grad_c10 at peak-updating corners    | 31× blowup — CONFIRMED LOAD-BEARING |

These five specific single-knob substitutions each make W2 worse.  The combined result does NOT prove that "no single-knob fix exists" (many other knobs — c01, c00, c11, partial-zero c10, alternative corner-fill recipes, different smoothing weights — are untested), nor that the current values are "geometrically correct".  What the evidence DOES support: the specific substitutions tested across iter-765/766/767/769/772 all fail to reduce mode A.  Further investigation should either (a) test a substitution that has not yet been ablated, or (b) step outside the single-knob-at-a-time paradigm.

**Iter-773+ next directions.**
- Circulation-based vorticity stencil at cube-vertex D-grid corners (iter-768 candidate #2, still untested).  This is the only remaining low-cost structural diagnostic.
- Structural port of Fortran c_sw + d_sw FB chain (blocked on ng=3 halo infrastructure).  The long-term Fortran-faithful solution.
- Document mode-A v_ll_Linf ≈ 0.159 m/s as the stabilized research baseline of the A-L + RK3 + boundary_fix production path.  This is a provisional conclusion pending iter-773 vorticity check.

**Deliverable.**  `scripts/diag_iter772_ablate_c10_peak_corners.py` with in-place cdgrid metric mutation via `._replace(grad_c10=...)`.  No source-code change.  No new sentinel (the iter-769 sentinel `test_boundary_fix_skip_corners_is_known_worse` already pattern-matches this kind of "zero a load-bearing knob" failure).  All 10 TestW2BoundaryErrorBudget sentinels pass.

**Process.**  36th iter in iter-752-772 chain.  Completes the ablation from iter-771b's revised candidate list.  Fifth single-knob substitution tested; fifth one that made W2 worse than the default.

### Iter-773 — Ranged c10 ablation at peak-updating D-grid corners

Per iter-772b's scope-limits note ("a smaller non-zero value might be better"), iter-773 runs a ranged ablation sweep on `grad_c10` at the 16 peak-updating D-grid corners.

**Method** (`scripts/diag_iter773_c10_range_sweep.py`; committed output at `diagnostics/iter773_output/iter773_c10_range.txt`).  For each scale factor α ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}, set `c10 = α × c10_default` at the 16 D-grid corner positions surrounding the iter-770 top-4 peak cells, and run the canonical W2 C36 1-day config.  All other metric entries unchanged.

**Result** (C36):

| α    | h_L2     | v_ll_Linf | relative to α=1 |
|------|----------|-----------|-----------------|
| 0.00 | 2.18e−3  | 4.96e+00  | 31.3×           |
| 0.25 | 1.77e−3  | 3.35e+00  | 21.1×           |
| 0.50 | 1.32e−3  | 1.96e+00  | 12.3×           |
| 0.75 | 6.01e−4  | 7.92e−01  | 5.0×            |
| **1.00** | **2.07e−4** | **1.59e−01** | **1.000× [default]** |
| 1.25 | 6.91e−4  | 8.91e−01  | 5.6×            |
| 1.50 | 1.04e−3  | 1.42e+00  | 9.0×            |

**Observation.**  Of the 7 discrete α values tested (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5), α=1.0 produced the smallest v_ll_Linf (0.159 m/s).  The measured points are sorted in ascending α and the v_ll_Linf values form a V shape with the smallest value at the center of the tested range.  The sweep does NOT prove that v_ll_Linf is continuously monotone between the sample points, or that no smaller value exists outside the tested range [0, 1.5] or at a finer-grained α near 1.0 (e.g., α=0.95 or α=1.05 were not measured).

**Scope limits.**  The sweep tested ONE kind of modification: a UNIFORM scalar multiplier applied to all 16 corner c10 values, at 7 discrete α values.  It did NOT test:
- Finer-grained α near 1.0 (only 0.25 steps were used).
- α outside the tested range [0, 1.5].
- Non-uniform scale factors (each corner with a different α).
- Modifications to c01, c00, c11 at these corners.
- Modifications to cells outside these 16 corners.
- Modifications to the circulation-vorticity stencil (iter-768 candidate #2, still untested).

**Implication.**  Among the 7 α values sampled, the default (α=1.0) gives the smallest W2 v_ll_Linf.  A simple uniform-scale rescaling by any of the 6 other tested factors would not reduce mode A.  This narrows the search space for iter-774+ but does not close it — finer-grained α near 1.0 and non-uniform / multi-coefficient changes remain candidates.

**Deliverable.**  `scripts/diag_iter773_c10_range_sweep.py` + committed output `diagnostics/iter773_output/iter773_c10_range.txt`.  No source-code change.  No new sentinel.  All 10 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  37th iter in iter-752-773 chain.  First iter that sampled a parameterized sweep of a single knob: among 7 discrete α values of a uniform scale factor on c10 at the 16 corners, the default (α=1.0) produced the smallest W2 v_ll_Linf.  Iter-774+ candidates: circulation-vorticity stencil at cube corners (iter-768 candidate #2, still untested), finer-grained α near 1.0, or non-uniform / multi-coefficient ablation.

### Iter-774 — Fine-grained c10 sweep near α=1.0 (potential 30% mode-A reduction, BUT NOT Fortran-faithful)

Per iter-773b's explicit iter-774+ candidate "finer-grained α near 1.0 (e.g. α=0.95, 1.05) were not measured", iter-774 runs the same sweep on 9 alpha values near 1.0.

**Method** (`scripts/diag_iter774_c10_fine_sweep.py`; committed output at `diagnostics/iter774_output/iter774_c10_fine.txt`).  Same protocol as iter-773 but α ∈ {0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10}.

**Result** (C36):

| α     | h_L2     | v_ll_Linf | relative to α=1 |
|-------|----------|-----------|-----------------|
| 0.900 | 2.04e−4  | 2.66e−1   | 1.675×          |
| 0.950 | 1.59e−4  | 1.43e−1   | 0.899×          |
| **0.980** | **1.79e−4**  | **1.11e−1**   | **0.697× (−30%)** |
| 0.990 | 1.92e−4  | 1.33e−1   | 0.838×          |
| 1.000 | 2.07e−4  | 1.59e−1   | 1.000× [default]|
| 1.010 | 2.24e−4  | 1.85e−1   | 1.166×          |
| 1.020 | 2.41e−4  | 2.17e−1   | 1.369×          |
| 1.050 | 2.99e−4  | 3.13e−1   | 1.972×          |
| 1.100 | 3.99e−4  | 4.66e−1   | 2.942×          |

**Observation.**  Among these 9 α values, the smallest W2 v_ll_Linf occurs at α=0.98 (not α=1.00).  Applying a uniform 0.98× scale to `grad_c10` at the 16 peak-updating D-grid corners reduces v_ll_Linf by ~30 % (0.159 → 0.111 m/s).  h_L2 also reduces at α=0.95 (1.59e−4) and α=0.98 (1.79e−4) relative to default (2.07e−4).

**CRITICAL CAVEAT — NOT Fortran-faithful.**  A 2 % uniform scalar correction to the metric coefficient at specific D-grid corners is a NUMERICAL TUNING, not a port of a Fortran formula.  The Fortran oracle computes `c10` (or its analog) from geometric constructs derived from the cubed-sphere grid vertices.  Our default `c10` is the Python analog of that same derivation.  A 0.98× correction that happens to reduce mode A on one test case is NOT automatically Fortran-faithful — the right Fortran-faithful direction is to compare our Python `c10` formula against Fortran's analog and identify any systematic derivation difference.

**What iter-774 does and does NOT establish.**
- DOES: uniformly scaling `c10` by 0.98 at the 16 corners reduces C36 1-d W2 v_ll_Linf by ~30 %.
- DOES NOT: establish this scaling generalizes to W5, cosine bell, ocean rest state, or other C resolutions.
- DOES NOT: establish this scaling is Fortran-faithful.
- DOES NOT: establish that α=0.98 is the true continuous minimum — the 0.01-step sweep only samples 9 points.
- DOES NOT: explain the mechanism — a 30 % reduction from a 2 % corner coefficient perturbation suggests a sensitivity that our current metric derivation may be mis-capturing, but this is inferential.

**Iter-775+ candidates.**
- Fortran-oracle comparison of `grad_c10` derivation (`src/legoesm/grids/cubed_sphere_cdgrid.py::create_cubed_sphere_cdgrid` vs Fortran `fv_grid_utils.F90` metric construction).  If our Python derivation has a systematic +2 % bias at peak-updating corners, the correction is to fix the derivation, not to hardcode α=0.98.
- Test whether α=0.98 at 16 corners affects W5, cosine bell, ocean rest.  If it harms any of them, the correction is not universal.
- Non-uniform per-corner optimization (vs uniform scalar).

**Deliverable.**  `scripts/diag_iter774_c10_fine_sweep.py` + committed output at `diagnostics/iter774_output/iter774_c10_fine.txt`.  NO production default change — pending Fortran-oracle comparison, the default stays at α=1.00 (unchanged grad_c10).  No new sentinel.  All 10 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  38th iter in iter-752-774 chain.  First iter in the 760+ chain to identify a parameter setting that DOES reduce W2 mode A below the current baseline — but only on this single 1D slice and with a caveat that a fixed-scalar metric correction is not Fortran-faithful until the underlying derivation difference is identified.

### Iter-775 — W5 cross-test of iter-774's α=0.98 c10 correction

Per iter-774's iter-775+ candidate "test α=0.98 impact on W5, cosine bell, ocean rest", iter-775 runs the W5 canonical C36 1-day test at α=1.00 (default) vs α=0.98 (iter-774 candidate) and compares the W5 diagnostics.

**Method** (`scripts/diag_iter775_w5_cross_test.py`; committed output at `diagnostics/iter775_output/iter775_w5_cross.txt`).  Same ablation as iter-774 (scale `grad_c10` by α at the 16 peak-updating D-grid corners), but run W5 instead of W2.  Cosine bell is NOT re-tested here because the matrix script NOTE (`run_atmosphere_test_matrix.py:1568-1576`) confirms cosine bell uses `transport_step` directly, not `model.step`, so the A-L gradient (where our c10 edit takes effect) is not on the cosine-bell path.

**Result** (W5, C36, dt=300s, 1-day):

| α     | mass_drift | \|h − h_ic\|_Linf | h_range (min, max)     |
|-------|------------|--------------------|------------------------|
| 1.000 | 6.71e−07   | 1.960e+02          | (3896, 5967)           |
| 0.980 | 7.67e−07   | 1.960e+02          | (3896, 5967)           |

**Observation.**  W5 mass drift and `|h - h_ic|_Linf` at t=1 day are essentially identical between α=1.00 and α=0.98 (matches to 4 significant figures on the h diagnostic, and both are ~7e-7 on mass drift — dominated by floating-point noise, not the c10 change).

**Implication.**  The α=0.98 correction from iter-774 does NOT degrade W5 within the measurement precision of the iter-775 diagnostic.  Combined with:

- iter-774: α=0.98 reduces W2 v_ll_Linf by 30 % (0.159 → 0.111 m/s) at C36 1-day.
- iter-775 (this iter): α=0.98 leaves W5 unchanged to measurement precision.
- cosine bell is structurally unaffected (does not use A-L gradient).

the two-test cross-validation is that α=0.98 reduces W2 without harming W5 or cosine bell.

**CRITICAL CAVEAT (unchanged from iter-774).**  α=0.98 is a NUMERICAL TUNING, not a Fortran-faithful fix.  A 2 % uniform scaling of a metric coefficient at specific D-grid corners does not derive from any Fortran formula.  The Fortran-faithful path remains: compare our Python `grad_c10` derivation against Fortran's metric-construction code (`fv_grid_utils.F90`) and identify any systematic derivation bias.  If a derivation bias is found, the correction is to fix the derivation, not hardcode α=0.98.

**What iter-775 does NOT establish.**
- That α=0.98 generalizes beyond C36 (C16, C24, C48, C96 untested).
- That α=0.98 generalizes beyond 1-day integration horizons.
- That α=0.98 is Fortran-faithful.
- Ocean rest state is not re-tested here (the ocean dycore uses `ocean_pe_cdgrid`, a different A-L caller; iter-766's scope note flagged it as unaffected by cube-corner halo tweaks, but iter-775 does not confirm the same for c10-at-peak-corners).

**Iter-776+ candidates.**
- Fortran-oracle comparison of the `grad_c10` derivation (`src/legoesm/grids/cubed_sphere_cdgrid.py` vs Fortran `fv_grid_utils.F90`).  This is the proper Fortran-faithful path, deferred repeatedly from iter-774 and iter-775+ candidate lists.
- C16 / C48 / C96 cross-resolution validation of α=0.98.
- Ocean rest test of α=0.98.

**Deliverable.**  `scripts/diag_iter775_w5_cross_test.py` + committed output `diagnostics/iter775_output/iter775_w5_cross.txt`.  No source-code change.  No production default change.  No new sentinel.  All 10 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  39th iter in iter-752-775 chain.  Second positive finding in the chain: α=0.98 correction reduces W2 without harming W5.  Default still α=1.00 pending Fortran-oracle derivation comparison.

### Iter-776 — Cosine bell C36 1-day transport distortion quantified

User visual inspection (provided mid-iter-775) reported "slight distortion near 1 day" on the cosine bell height snapshot.  Iter-776 quantifies that observation.

**Method** (`scripts/diag_iter776_cosine_bell_1day.py`; committed output at `diagnostics/iter776_output/iter776_cb_1day.txt`).  Runs the canonical matrix cosine bell config (C36, dt=1800s, β=π/4, `transport_step` + frozen `_d2a2c_vect` winds) for 1 day, then compares to the analytic exact solution (`cosine_bell_exact` — Eq. 25 of PL07 evaluated at reverse-rotated coordinates).  Cosine bell does NOT use the A-L gradient, so iter-774's `grad_c10` correction is irrelevant here.

**Result** (C36, 1-day):

| Metric        | Value      |
|---------------|------------|
| L1            | 1.198e−01  |
| L2            | 1.168e−01  |
| Linf          | 1.227e−01  |
| Peak (exact)  | 990.43 m   |
| Peak (C36 1d) | 896.29 m   |
| Undershoot    | 9.50 %     |
| Peak cell shift | 1 cell in j direction |

**Observation.**  At t=1 day the cosine bell has lost ~9.5 % of its peak amplitude and the peak cell position has shifted by 1 cell (~2.4° lat).  The L1/L2/Linf error norms are ≈ 0.12.

**Not a regression.**  These values match the matrix script's canonical PASS output (`L1=1.20e-01, L2=1.17e-01, Linf=1.23e-01`) — they have been the reported matrix-PASS values since at least iter-760.  The matrix PASS threshold does not fire on this level of distortion, but the user's visual confirms it is observable and worth tracking.

**Mechanism candidates** (untested in iter-776 — this iter is purely quantitative; candidate list is deliberately broad and not exhaustive).

- PPM truncation in the Lin-Rood operator-split sweep (`fv_tp_2d` at `src/legoesm/core/fv_tp_2d.py:459`).  PPM has inherent shape preservation limits; 10 % amplitude loss per day is high for a linear PPM but plausible at C36.
- Halo handling at cube-corner passages via `pad_halo(halo=2)` in `fv_tp_2d`.  The bell crosses cube edges during the 1-day integration (start lon=−90°, end lat/lon depending on β=π/4 flow).
- `_d2a2c_vect` contravariant-velocity computation at cube-corner cells.  Frozen winds are computed once at t=0; any cube-corner error in that computation propagates for the whole integration.
- `fv_tp_2d`'s optional del-n damping (`nord`, `damp_c`) is NOT used by the matrix cosine bell path (both are None).  If Fortran's `sw_core.F90:886-887` applies a 4th-order smoother in its transport path, enabling it in our cosine bell could change the error distribution — untested.
- The initial-condition projection onto the cubed-sphere cells.  `cosine_bell_cubesphere` in `tests/test_cases/cosine_bell.py` evaluates the analytic bell at cell centres; any cell-average vs point-value discrepancy could contribute.
- `transport_step`'s `mass_target` renormalization path (`fv_tp_2d.py:568`).  Mass-conservation clipping can redistribute error between cells; this is active in the matrix cosine bell config.
- Relation to W2 mode-A: the two are structurally distinct code paths, but both involve cube-corner cells, and any `pad_halo` / `_d2a2c_vect` / metric-coefficient finding from the W2 investigation might also be relevant to the transport path.  iter-765-775's cube-corner work did NOT modify `fv_tp_2d` or `_d2a2c_vect`, but future fixes in those paths could affect cosine bell.

**Iter-777+ candidates (broad, not prioritized).**

- Compare `fv_tp_2d` PPM sweep boundary handling to Fortran `tp_core.F90` at cube-corner cells.  Our Python `pad_halo(halo=2)` vs Fortran's `copy_corners` (tp_core.F90:229) — is the boundary-halo treatment at cube corners faithful?
- Measure the bell's peak amplitude at t=0.5, 1, 2, 3 days to characterize the amplitude-loss rate.  If it's uniform (not jump at cube-corner passage), the distortion has more diffuse origins; if it's stepped at cube-corner crossings, that localizes the cause.
- Test cosine bell at C48, C72, C96 to verify PPM convergence order.
- Enable `nord` / `damp_c` in the cosine bell matrix config (currently both None).  Fortran's d_sw1 calls `fv_tp_2d(delp, ..., nord=nord_v, damp_c=damp_v)` per `sw_core.F90:886-887`.  A Fortran-faithful cosine bell run would include that 4th-order smoother.
- Port the Fortran c_sw+d_sw FB transport chain (blocked on ng=3 halo infrastructure, same as the W2 long-term fix).
- Audit `_d2a2c_vect` cube-corner behaviour: the "Non-duogrid `_d2a2c_vect` cube-vertex gap" is listed as `isolated, not fixed` in the Closed priorities section — it may be contributing to the frozen-wind error on cosine bell.

**Deliverable.**  `scripts/diag_iter776_cosine_bell_1day.py` + committed output.  No source-code change.  No new sentinel.  All 12 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  40th iter in iter-752-776 chain.  First iter to quantify the cosine bell visual artifact the user flagged.  Shifts iter-777+ investigation from W2 mode-A (A-L gradient) to cosine bell transport (PPM).

### Iter-777 — Cosine bell `nord`/`damp_c` sweep at 3 configs (small / negligible change on this test)

Per iter-776b's candidate "Enable `nord`/`damp_c` in cosine bell matrix config", iter-777 measures the cosine bell C36 1-day error at three `transport_step` configurations and reports the results.  Iter-777 is narrowly scoped: 3 discrete samples, 1 test case, 1 resolution, 1 day.

**Note on order nomenclature** (iter-777b, Codex): per `CDGridShallowWaterConfig.nord` docstring, `nord=0` is del-2, `nord=1` is del-4, `nord=2` is del-6.  The iter-776b candidate description as "4th-order smoother" was imprecise; the Fortran `sw_core.F90:886-887` call uses `nord=nord_v`, which with our Config default `nord_v=2` is a del-6 smoother, not del-4.

**Method** (`scripts/diag_iter777_cb_nord_dampc.py`; committed output at `diagnostics/iter777_output/iter777_cb_smoother.txt`).  Same canonical cosine bell C36 1-day setup as iter-776.  Three configs:

- matrix default: `nord=None, damp_c=None` (current)
- matches `sw_core.F90:886-887` with our Config's `nord_v=2`, `damp_v=0.06`: `nord=2, damp_c=0.06`
- cross-check: `nord=1, damp_c=0.16` (a different knob; `d4_bg`, not `damp_v`)

**Result** (C36, 1-day, cosine bell):

| nord | damp_c | L1        | L2        | Linf      | undershoot |
|------|--------|-----------|-----------|-----------|------------|
| None | None   | 1.198e−01 | 1.168e−01 | 1.227e−01 | 9.50 %     |
| 2    | 0.06   | 1.198e−01 | 1.168e−01 | 1.226e−01 | 9.47 %     |
| 1    | 0.16   | 1.563e−01 | 1.373e−01 | 1.331e−01 | 11.32 %    |

**Observation.**  Between rows 1 and 2, Linf changes from 0.1227 to 0.1226 and the undershoot from 9.50 % to 9.47 % — a change within single-precision-style rounding for these metrics.  Between rows 1 and 3, Linf rises from 0.123 to 0.133 and the undershoot from 9.50 % to 11.32 %.

**Scope limits.**
- Only 3 `(nord, damp_c)` points were sampled.
- Only C36, only 1 day, only β=π/4.
- Iter-777 does NOT verify that our `_deln_flux` implementation produces the same flux output as Fortran's `deln_flux` on shared inputs; the negligible change at `(nord=2, damp_c=0.06)` is consistent with (a) our implementation matching Fortran at that coefficient strength, (b) our implementation being weaker than Fortran, or (c) PPM truncation dominating on this test regardless of the smoother strength.  A Fortran-oracle comparison is required to distinguish.
- Iter-777 does NOT establish that "missing smoothing" is ruled out as a contributor.  It establishes only that the specific `(nord=2, damp_c=0.06)` config produces no measurable change and the specific stronger `(nord=1, damp_c=0.16)` config makes this particular test worse.

**Implication.**  On this specific configuration (C36, 1 day, β=π/4, these 3 sample points), enabling the `(nord=2, damp_c=0.06)` setting does not materially change the cosine bell error norms.  Further iter-778+ candidates from iter-776b remain open:

- Compare our `_deln_flux` implementation against Fortran's `deln_flux` derivation (the outcome above does not distinguish matched implementations from weak ones).
- Port the Fortran c_sw+d_sw FB transport chain (blocked on ng=3 halo infrastructure).
- Audit the non-duogrid `_d2a2c_vect` cube-vertex gap (listed `isolated, not fixed` in Closed priorities).
- Sweep cosine bell at C48/C72/C96 to measure PPM convergence order.
- Larger `nord`/`damp_c` grid sweep at multiple horizons.

**Decision.**  Matrix cosine bell default `nord=None, damp_c=None` is retained pending further investigation.  Production path UNCHANGED.

**Deliverable.**  `scripts/diag_iter777_cb_nord_dampc.py` + committed output.  No source-code change.  No new sentinel.  All 12 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  41st iter in iter-752-777 chain.  Three discrete `(nord, damp_c)` samples were tested on the cosine bell C36 1-day case; the `(2, 0.06)` sample produced no measurable change and the `(1, 0.16)` sample increased the error.  Iter-778+ candidates from iter-776b remain open.  Iter-777b (Codex stop-time) retracted the "4th-order smoother" mis-nomenclature, the "too weak" mechanism attribution, and the "falsifying 'missing smoothing'" strong conclusion from only 3 discrete points.

### Iter-778 — Cosine bell cross-resolution convergence: error plateaus, not converging

Per iter-777b's iter-778+ candidate "Sweep cosine bell at C48/C72/C96 to measure PPM convergence order", iter-778 runs the canonical cosine bell setup at C16, C24, C36, C48 and measures observed convergence order.

**Method** (`scripts/diag_iter778_cb_convergence.py`; committed output at `diagnostics/iter778_output/iter778_cb_convergence.txt`).  Same β=π/4, 1 day integration.  `dt = 1800 × (36/n)` (CFL-scaled from the C36 matrix default).  Reports L1/L2/Linf and observed order `p = log(err(n1)/err(n2)) / log(n2/n1)` between consecutive resolutions.

**Result**:

| n  | dt     | steps | L1        | L2        | Linf      | undershoot |
|----|--------|-------|-----------|-----------|-----------|------------|
| 16 | 4114   | 21    | 1.722e−01 | 1.517e−01 | 1.727e−01 | 15.00 %    |
| 24 | 2700   | 32    | 1.239e−01 | 1.156e−01 | 1.174e−01 | 11.20 %    |
| 36 | 1800   | 48    | 1.198e−01 | 1.168e−01 | 1.227e−01 | 9.50 %     |
| 48 | 1350   | 64    | 1.214e−01 | 1.195e−01 | 1.267e−01 | 8.75 %     |

**Convergence order** (log ratio / log resolution ratio):

| pair     | p(L1)  | p(L2)  | p(Linf) |
|----------|--------|--------|---------|
| 16→24    | +0.811 | +0.671 | +0.953  |
| 24→36    | +0.083 | −0.026 | −0.110  |
| 36→48    | −0.045 | −0.077 | −0.111  |

**Observation.**  The error norms decrease from C16 to C24 (sub-1st-order convergence on this transition).  Above C24, L2 and Linf INCREASE at higher resolution — negative "convergence order" on the last two transitions.  L1 is approximately flat after C24.  The peak-undershoot fraction decreases monotonically with n, which is the only monotone trend observed.

**Scope limits.**
- 4 resolutions tested.  Further points (C72, C96) would refine the picture.
- 1 test case (β=π/4 cosine bell), 1 horizon (1 day).  Different ICs or horizons might behave differently.
- The dt scaling with (36/n) is a choice; a different dt policy would shift per-step error budgets and the measured p could change.  An additional sweep at fixed dt or fixed Courant number would distinguish spatial-truncation from dt-dependent error.
- PPM formal accuracy is ~3rd order for smooth linear advection; the cosine bell is smooth except where the C² term reaches the bell radius.  This is not a fully regular advection test.

**What iter-778 does and does not support.**
- DOES support: between C16 and C24, error decreases.  Above C24, error norms are roughly flat-to-increasing.
- DOES NOT support (yet): a blanket claim that "convergence is broken" or "there is a structural bug."  A non-monotone error plateau has multiple candidate mechanisms:
  (a) a resolution-invariant structural error (candidate bug).
  (b) dt scaling vs spatial-error crossover (as resolution increases, time-discretization error grows relative to spatial).
  (c) cube-vertex-localized error whose magnitude is set by face-count (constant 8) and whose shape gets finer but does not get smaller at higher n.
  (d) PPM shape-preserving limiter kicking in at different ratios with n.

Distinguishing among these requires further diagnostics (iter-779+).

**Iter-779+ candidates.**
- Run the same sweep at FIXED dt across all resolutions (e.g. dt=450s — CFL-safe at C64).  If error still plateaus, dt scaling is not the cause.
- Measure where in the field the error concentrates at each resolution — if the peak error stays near a cube-vertex passage independent of n, that's (c).
- Audit `_d2a2c_vect` non-duogrid cube-vertex gap (iter-776b's repair candidate still unaudited).
- Port Fortran c_sw+d_sw FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter778_cb_convergence.py` + committed output.  No source-code change.  No new sentinel.  All 12 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  42nd iter in iter-752-778 chain.  Measures the question iter-777b deferred ("test C48/C72/C96 to verify PPM convergence order"); result is a non-monotone error plateau that has multiple candidate mechanisms.

### Iter-779 — Cosine bell fixed-dt convergence: plateau persists

Per iter-778 candidate (b) discriminator, iter-779 re-runs the cosine bell 4-resolution sweep at FIXED dt=1350s (CFL-safe at C48).

**Result** (fixed-dt; for full detail see `diagnostics/iter779_output/iter779_cb_fixed_dt.txt`):

| n  | Linf (fixed dt=1350s) | Linf (iter-778 CFL-scaled) |
|----|-----------------------|-----------------------------|
| 16 | 1.79e−01              | 1.73e−01                    |
| 24 | 1.18e−01              | 1.17e−01                    |
| 36 | 1.23e−01              | 1.23e−01                    |
| 48 | 1.27e−01              | 1.27e−01                    |

Observed p(Linf) in iter-779: 16→24: +1.03, 24→36: −0.09, 36→48: −0.12.

**Observation.**  The plateau above C24 PERSISTS at fixed dt.  Candidate (b) dt-scaling-crossover is ruled out by this test at these 4 resolutions.

**Remaining iter-778 candidates open.** (a) structural error, (c) cube-vertex-localized error set by face count, (d) PPM limiter activation.

**Iter-779b/c sentinel notes.**  Iter-779b adds a file-content sentinel (`test_iter778_779_cb_convergence_artifacts`, 0.4s) that locks the committed iter-778/iter-779 output files and the plateau pattern.  A subprocess sentinel that actually executes the iter-779 script was attempted locally (3:36 runtime) but NOT committed — Codex stop-time review flagged "CI timeout conflict" for unit tests.  Trade-off accepted: file-content sentinel catches drift; rerunnability is verified locally but not in CI.

**Deliverable.**  `scripts/diag_iter779_cb_fixed_dt.py` + committed output + iter-779b sentinel.  No source-code change.  All 13 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  43rd iter in chain; rules out iter-778 candidate (b).  iter-780+ candidates: measure error LOCATION at each resolution; audit `_d2a2c_vect` cube-vertex gap.
