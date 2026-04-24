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

### Iter-780 — Cosine bell peak-error location across C16/C24/C36/C48

Per iter-779's iter-780+ candidate "Measure where in the field the error concentrates at each resolution", iter-780 runs the same 4-resolution sweep and reports the face-local cell where `|h - h_exact|` is maximum.

**Method** (`scripts/diag_iter780_cb_error_location.py`; committed output at `diagnostics/iter780_output/iter780_cb_error_location.txt`).  Same β=π/4, 1 day, dt=1350s as iter-779.  At each resolution reports peak-error face + `(i, j)` + lat/lon + great-circle distance to nearest cube vertex + cell-distance to nearest face edge + peak |err| amplitude.

**Result** (C16/C24/C36/C48, fixed dt=1350s):

| n  | face | (i, j)    | lat      | lon       | GC to vertex | cell to edge | peak \|err\| |
|----|------|-----------|----------|-----------|--------------|--------------|--------------|
| 16 | 3    | (12, 12)  | 23.15°   | −64.69°   | 20.96°       | 3            | 1.68e+02     |
| 24 | 3    | (18, 17)  | 18.92°   | −65.62°   | 24.48°       | 5            | 1.15e+02     |
| 36 | 3    | (30, 32)  | 32.08°   | −58.75°   | 11.87°       | 3            | 1.21e+02     |
| 48 | 3    | (39, 43)  | 32.95°   | −60.94°   | 13.38°       | 4            | 1.26e+02     |

Summary (from committed output): GC-distance to the nearest cube vertex ranges `11.87° – 24.48°` (span 12.62°); cell-distance to face edge ranges `3 – 5`; peak `|err|` ranges `115 – 168`.

**Observation.**  At each of the 4 tested resolutions, the peak-error cell is on face 3 but at a lat/lon position in the middle of that face (not on face edges, not at face corners).  The cell-distance to the nearest face edge is 3–5 cells.  The GC-distance to the nearest of 8 cube vertices varies by 12°+ across the 4 resolutions.

**What iter-780 DOES show** (observational only).
- The peak-error cell position in face-local indices at each n.
- The peak-error great-circle distance to the nearest cube vertex.
- The peak-error cell-distance to the face edge.

**What iter-780 does NOT establish.**
- Whether the peak-error cell is also where the error was ACCUMULATED (vs a downstream collection of errors propagated from upstream).
- Whether the peak-error position is stable in time (this is a single end-of-day snapshot).
- Which of iter-778's candidate mechanisms (a)/(c)/(d) is supported.  Candidate (c) "cube-vertex-localized error set by face count" would predict the peak-error location to stay within a small GC-distance of a cube vertex at all resolutions; iter-780's 12°-24° range does NOT directly match that prediction, but iter-780 also does not DISPROVE (c) because the peak-error LOCATION at t=1 day is not the same as the error-ACCUMULATION location at cube-vertex CROSSINGS earlier in the trajectory.

**Iter-781+ candidates**:
- Trajectory diagnostic: measure the error location every ~0.1 day during the 1-day integration.  If the peak-error cell moves with the bell and its distance to cube vertices is transient but pronounced during cube crossings, that is consistent with (c).
- Audit `_d2a2c_vect` non-duogrid cube-vertex gap — still unaudited.
- Port Fortran c_sw+d_sw FB transport chain (blocked on ng=3 halos).

**Deliverable.**  `scripts/diag_iter780_cb_error_location.py` + committed output.  No source-code change.  No new sentinel.  All 13 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  44th iter in iter-752-780 chain.  Observational-only location reporting with explicit scope-limits; does not distinguish candidates (a)/(c)/(d) but provides data for iter-781+ trajectory-vs-snapshot follow-up.

### Iter-781 — Cosine bell peak-error trajectory at 0.1-day intervals (C36)

Per iter-780's iter-781+ candidate "Trajectory diagnostic: measure the error location every ~0.1 day during the 1-day integration", iter-781 samples peak-error position + GC distance to nearest cube vertex + peak |err| amplitude every 0.1 days during a C36 1-day run.

**Method** (`scripts/diag_iter781_cb_error_trajectory.py`; committed output at `diagnostics/iter781_output/iter781_cb_error_trajectory.txt`).  Same β=π/4 and C36 as iter-779/780.  **dt=1440s** (changed from iter-780's 1350s) so that each 0.1-day sample is exactly 6 timesteps of 1440s = 8640s = 0.1 day; an integer-step assertion in the script guarantees this alignment.  The earlier iter-781a draft used dt=1350s, which advances 6×1350s = 8100s = 0.0938 day per "0.1 day" label — Codex flagged the time misalignment and the pre-alignment numbers were retracted before the committed output was written.  11 samples at t = 0.00, 0.10, ..., 1.00 days.  Also reports the bell centre's GC-distance to nearest cube vertex (taken from the argmax of the exact solution on the grid) as a proxy for where the bell is in its trajectory.

**Result** (full 11-row trajectory; committed output `diagnostics/iter781_output/iter781_cb_error_trajectory.txt`):

| t (d) | peak cell | GC to vertex | peak \|err\| | bell centre GC to vertex |
|-------|-----------|--------------|--------------|---------------------------|
| 0.00  | face 0 (0, 0)   |  1.19°  | 0.000e+00 | 52.97° |
| 0.10  | face 3 (17, 17) | 52.97°  | 1.208e+01 | 52.97° |
| 0.20  | face 3 (18, 18) | 52.97°  | 2.286e+01 | 49.44° |
| 0.30  | face 3 (20, 20) | 45.93°  | 2.747e+01 | 45.93° |
| 0.40  | face 3 (21, 21) | 42.46°  | 3.721e+01 | 42.46° |
| 0.50  | face 3 (22, 22) | 39.02°  | 4.472e+01 | 39.02° |
| 0.60  | face 3 (23, 23) | 35.65°  | 5.186e+01 | 35.65° |
| 0.70  | face 3 (24, 24) | 32.33°  | 5.976e+01 | 32.33° |
| 0.80  | face 3 (27, 30) | 18.99°  | 7.453e+01 | 32.33° |
| 0.90  | face 3 (29, 31) | 14.49°  | 9.478e+01 | 29.09° |
| 1.00  | face 3 (30, 32) | 11.87°  | 1.213e+02 | 25.93° |

**Observation — numerical reportage only.**  Peak |err| grows monotonically from 0 at t=0 to 121.3 at t=1.0.  Step-to-step increments (t=0.1→1.0): 10.78, 4.61, 9.74, 7.51, 7.14, 7.90, 14.77, 20.25, 26.52 — the increment rises through the second half of the day (14.8 at t=0.7→0.8, 20.3 at t=0.8→0.9, 26.5 at t=0.9→1.0).  No sharp spike at a single sample point.  For t ≤ 0.7 the peak-error cell and the bell centre are at the same face-local (i, j) and same GC-to-vertex; for t ≥ 0.8 the peak-error cell is closer to the nearest cube vertex than the bell centre is (e.g. t=1.0: peak at 11.87°, bell centre at 25.93°).

**Scope limit — the bell does NOT pass over any cube vertex in this 1-day run.**  The bell centre's GC-to-vertex remains ≥ 25.93° throughout (minimum at t=1.0).  A cube-vertex-crossing event would require GC-distance ~2–3° transiently, which this 1-day trajectory does not contain.  Therefore iter-781 does NOT directly test candidate (c) ("cube-vertex-localized error set by face count") at a cube-crossing event — the bell never crosses one in this window.

**What iter-781 DOES show.**
- 11 time-aligned sample points of peak-error location and amplitude during a 1-day cosine bell trajectory at C36.
- Peak |err| grows smoothly without visible spikes across all 10 step-to-step increments.
- Bell centre approaches a cube vertex (from 52.97° to 25.93°) but does not cross one.
- For t ≥ 0.8 the peak-error cell is closer to the nearest cube vertex than the bell centre (peak-cell "leads" bell-centre toward the vertex).

**What iter-781 does NOT establish.**
- Candidate (c) cannot be directly tested here because no cube-vertex-crossing occurs in 1 day.
- Longer horizons (3–12 days, including multiple cube crossings) are needed to see whether peak |err| spikes at crossings.
- A smooth growth curve is consistent with multiple mechanisms including (a), (d), or (c)-that-simply-doesn't-fire-in-this-window.
- The "peak-cell leads bell-centre toward the vertex" pattern seen for t ≥ 0.8 is a single-trajectory observation; it does not by itself attribute the leading offset to any specific mechanism and it does not persist backwards in time (t ≤ 0.7 has peak and bell at the same cell).

**Iter-782+ candidates.**
- Extend the trajectory to 3 days (bell rotates 90° of its 12-day period — may cross a cube vertex if β=π/4 path passes near (45°, ±35.26°)).
- 12-day trajectory sampled every 0.5 day (would catch all cube crossings in a full revolution).
- Audit `_d2a2c_vect` non-duogrid cube-vertex gap (still unaudited from iter-776b).
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter781_cb_error_trajectory.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  45th iter in iter-752-781 chain.  Observational-only trajectory report.  dt was changed from 1350s (iter-779/780) to 1440s specifically so that 0.1-day sample labels correspond to an integer number of dt steps; a script-level assertion prevents future drift.  Explicit scope-limit: the 1-day window doesn't include a cube-vertex crossing, so candidate (c) remains untested by iter-781.

### Iter-782 — `_d2a2c_vect` cube-vertex error audit on solid-body rotation IC (C36)

Per iter-781's iter-782+ candidate "Audit `_d2a2c_vect` non-duogrid cube-vertex gap (still unaudited from iter-776b)", iter-782 measures the absolute error in `_d2a2c_vect`'s output (ut, vt) on the ANALYTICAL solid-body rotation initial condition at t=0, i.e., BEFORE any time integration.

**Rationale.**  For the cosine bell test, `_d2a2c_vect` is called once at t=0 and the resulting ut/vt are held constant for the whole integration.  Any error in (ut, vt) at t=0 therefore imprints a CONSTANT bias on the transport winds for all 60 timesteps of a 1-day C36 run.  Iter-782 quantifies that bias and localises it.

**Method** (`scripts/diag_iter782_d2a2c_fidelity.py`; committed output at `diagnostics/iter782_output/iter782_d2a2c_fidelity.txt`).  C36, β=π/4.  Build state via `cosine_bell_cubesphere` (which uses the analytical solid-body rotation winds for u_d, v_d).  Call `_d2a2c_vect(state.u_d, state.v_d, cdgrid)` to obtain numerical (ua, va, uc, vc, ut, vt).  Compute ut_exact / vt_exact at each C-grid stagger position by the same formula `_d2a2c_vect` uses at line 400 (`ut = (uc - v_d * cosa_u) * rsin_u` and the vt analog) but substituting the analytical covariant u/v at that stagger for the interpolated uc/vc.  This isolates the A→C interpolation error, amplified by rsin_u / rsin_v.

**Result.**

| field | peak \|err\| | face, (i, j)    | lat     | lon     | GC-to-vertex | cell-to-edge |
|-------|--------------|-----------------|---------|---------|--------------|--------------|
| ut    | 5.740 m/s    | 0, (36, 35)     | +34.10° | +45.00° | 1.16°        | 0            |
| vt    | 6.174 m/s    | 1, (0, 36)      | +35.84° | +46.24° | 1.16°        | 0            |

- max \|ut_exact\| = 49.09 m/s → max \|ut - ut_exact\| = 11.69% relative
- max \|vt_exact\| = 54.17 m/s → max \|vt - vt_exact\| = 11.40% relative
- Hot points (\|err\| > 0.5 * peak): 60 in ut, 30 in vt.  100% of ut hot points and 100% of vt hot points are within 10° GC of a cube vertex; 46.7% / 53.3% are within 5° GC.
- Peak-error cells are at C-grid stagger positions exactly adjacent to cube-vertex corners (cell-to-face-edge = 0, GC to nearest cube vertex = 1.16°).

**Observation — numerical reportage only.**  At C36 on the analytical solid-body rotation IC, `_d2a2c_vect` produces ut/vt with peak absolute error ~6 m/s localised at all 8 cube vertices (GC ≈ 1°), falling off with GC distance such that all points with \|err\| > 0.5 * peak lie within 10° GC of a cube vertex.  The relative error is ~11% of peak wind.  This measurement is taken at t=0, with no time integration performed.

**What iter-782 DOES show.**
- `_d2a2c_vect`'s output on the exact solid-body rotation IC has nontrivial error at cube vertices — ~6 m/s absolute, ~11% relative, all within 10° GC of the 8 cube vertices.
- The error is imprinted on the transport winds for the entire cosine bell run (because ut/vt are computed once and held constant).
- The error is localised; most of each face has much smaller error.

**What iter-782 does NOT establish.**
- Whether this ~6 m/s localised error is large enough to CAUSE the cosine bell Linf ≈ 121 at t=1 day.  That requires advecting the h-field through the error-biased wind and comparing to the clean case — iter-782 does not run this test.
- Whether the error comes from `fill_corner_region` (Lagrange corner fill) or from the 4th-order A→C stencil's unresolved neighbours at cube vertices — iter-782 cannot decompose those contributions.
- Whether the cosine bell trajectory at β=π/4 actually PASSES THROUGH a 10°-radius zone of a cube vertex during 1 day.  From iter-781's trajectory, the bell centre's GC-to-vertex minimum at t=1 day is 25.93°, well OUTSIDE the 10° error zone — so the bell centre itself does not advect through the error zone in 1 day.  However, the bell's FOOTPRINT (radius R/3 ≈ 33° on a sphere) overlaps the error zone for much of the run.
- Whether a single-trajectory cosine bell tests this mechanism (since iter-781 showed no crossing happens).

**Iter-783+ candidates.**
- Targeted test: replace ut/vt with analytical values at cube-vertex cells only (override `_d2a2c_vect` output where GC < 10°), re-run 1-day cosine bell, measure Linf vs baseline.  If Linf drops ≥ 50%, strong evidence this mechanism is primary.
- Investigate whether the error comes from `fill_corner_region` (Lagrange corner fill) or from the A→C stencil by toggling the corner fill method on a controlled input.
- Measure the same `_d2a2c_vect` error on W2 IC (zonal wind) to check whether the cube-vertex error also correlates with the W2 mode-A artifact.
- Cross-check Python `fill_corner_region_2d` against Fortran `fv_duogrid.F90:1719-1903` for ordering differences (Python pass-2 diagonal cells use CURRENT padded; Fortran uses `veltemp`/`veltempp` snapshots captured AFTER pass-1 non-diagonal fills).

**Deliverable.**  `scripts/diag_iter782_d2a2c_fidelity.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  46th iter in iter-752-782 chain.  First observational measurement of `_d2a2c_vect` absolute error since the iter-776b audit flag was raised.  The 11% relative cube-vertex error is a concrete, quantified artifact that iter-783+ can attempt to repair or rule out.

### Iter-783 — Targeted `_d2a2c_vect` override rules out cube-vertex error as cosine bell driver

Per iter-782's iter-783+ candidate "targeted test: replace ut/vt with analytical values at cube-vertex cells only, re-run 1-day cosine bell, measure Linf vs baseline", iter-783 runs the 1-day C36 cosine bell four times:

- **BASELINE**: unmodified `_d2a2c_vect` output.
- **OVERRIDE GC ≤ 5°**: 144 cells per field replaced with analytical ut_exact / vt_exact.
- **OVERRIDE GC ≤ 10°**: 576 cells per field replaced.
- **OVERRIDE GC ≤ 20°**: 2136 cells per field replaced.

The override is a DIAGNOSTIC-ONLY analytical injection (not a production fix).  ut_exact / vt_exact are computed from the solid-body rotation geographic winds and the same covariant→contravariant formula `_d2a2c_vect` uses internally.

**Method** (`scripts/diag_iter783_cb_mask_override.py`; committed output at `diagnostics/iter783_output/iter783_cb_mask_override.txt`).  C36, β=π/4, dt=1440s (time-aligned per iter-781b), 1 day.  Same `CDGridShallowWaterConfig` as iter-779/780/781.

**Result.**

| case              | L_inf     | L2        | peak cell       | overrides        | ΔLinf vs baseline |
|-------------------|-----------|-----------|-----------------|------------------|-------------------|
| BASELINE          | 1.213e+02 | 1.165e-01 | face 3 (30, 32) | 0                | —                 |
| OVERRIDE GC ≤ 5°  | 1.213e+02 | 1.165e-01 | face 3 (30, 32) | ut=144, vt=144   | +0.0%             |
| OVERRIDE GC ≤ 10° | 1.213e+02 | 1.165e-01 | face 3 (30, 32) | ut=576, vt=576   | +0.0%             |
| OVERRIDE GC ≤ 20° | 1.260e+02 | 1.184e-01 | face 3 (30, 32) | ut=2136, vt=2136 | +3.9%             |

**Observation — numerical reportage only.**  Overriding `_d2a2c_vect`'s output at cube-vertex-proximal cells with analytical values does not reduce the 1-day C36 cosine bell Linf:
- At GC ≤ 5° (144 cells, where iter-782 measured the largest absolute errors of ~5–6 m/s): Linf unchanged to 4 significant digits.
- At GC ≤ 10° (576 cells, covering 100% of hot points from iter-782): Linf unchanged.
- At GC ≤ 20° (2136 cells, a large halo around each cube vertex): Linf slightly WORSE (+3.9%), consistent with the override introducing artificial discontinuity at the mask boundary where ut/vt jumps from analytical to interpolated.
- Peak-error cell position is unchanged at face 3 (30, 32), which corresponds to GC ≈ 11.87° from the nearest cube vertex (iter-780 finding) — outside the 10° override zone.

**What iter-783 DOES show.**
- The ~6 m/s / 11% relative `_d2a2c_vect` cube-vertex error measured in iter-782 is NOT the dominant driver of the C36 1-day cosine bell Linf at 121.  Removing it entirely (via analytical injection) does not reduce Linf.
- The cosine bell's trajectory in 1 day does not pass through the 10°-radius cube-vertex error zone, so the localised transport bias does not imprint on the bell's actual path.
- Introducing the override over an even larger zone (20°) WORSENS Linf because it creates artificial discontinuity.

**What iter-783 does NOT establish.**
- Which OTHER mechanism causes the observed cosine bell distortion.  Remaining iter-778 candidates: (a) resolution-invariant structural error, (c) cube-vertex-localized error set by face count, (d) PPM limiter.  iter-778/779's plateau above C24 is consistent with (a) or (d).
- Whether the `_d2a2c_vect` cube-vertex error becomes important for LONGER integrations (multi-day, multi-cube-crossing trajectories).
- Whether a production-quality fix to `_d2a2c_vect` (e.g., ordering of `fill_corner_region` to match Fortran `veltemp`/`veltempp` snapshot semantics) would improve W2 or ocean-rest-state.

**Iter-784+ candidates.**
- Test the A→C stencil directly: toggle 4th-order vs 2nd-order A→C interpolation; measure Linf impact.
- Test the PPM limiter: swap the PPM limiter in `fv_tp_2d` for an unlimited polynomial; measure impact (candidate d).
- Test fill_corner_region ordering: port Fortran `veltemp`/`veltempp` snapshot semantics and re-run iter-782 to see if the ~6 m/s error drops (does not help the C36 1-day cosine bell per iter-783, but would improve Fortran fidelity for longer horizons).
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter783_cb_mask_override.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  47th iter in iter-752-783 chain.  Rules OUT iter-782's candidate (cube-vertex `_d2a2c_vect` error) as cause of the C36 1-day cosine bell distortion.  Points iter-784+ attention toward candidates (a) and (d).

### Iter-784 — PPM limiter bypass rules out candidate (d) as cosine bell driver

Per iter-783's iter-784+ candidate "test the PPM limiter: swap the PPM limiter in `fv_tp_2d` for an unlimited polynomial; measure impact (candidate d)", iter-784 runs the 1-day C36 cosine bell twice:

- **BASELINE**: normal `_pert_ppm_iv0` (positive-definite, FV3 hord=9) + face-boundary `_pert_ppm(iv=1)`.
- **NO_LIMITER**: monkey-patched so both limiters return bl, br unchanged.

The monkey-patch is diagnostic-only.  Without the limiter, h can go negative; the `mass_target` fix then re-clips negatives to zero and rescales — so the reported Linf is on the re-clipped field.

**Method** (`scripts/diag_iter784_cb_limiter_bypass.py`; committed output at `diagnostics/iter784_output/iter784_cb_limiter_bypass.txt`).  Same C36, β=π/4, dt=1440s, 1 day.

**Result.**

| case                  | L_inf     | L2        | peak cell       | peak err signed | h_min | h_max  | Δ vs base         |
|-----------------------|-----------|-----------|-----------------|-----------------|-------|--------|-------------------|
| BASELINE (limiter ON) | 1.213e+02 | 1.165e-01 | face 3 (30, 32) | +1.213e+02      | 0.00  | 895.40 | —                 |
| NO_LIMITER            | 1.204e+02 | 1.172e-01 | face 3 (30, 32) | +1.204e+02      | 0.00  | 893.42 | Linf −0.8%, L2 +0.6% |

**Observation — numerical reportage only.**  Bypassing the PPM limiter changes Linf by −0.8% (virtually unchanged) and L2 by +0.6% (slightly worse).  Peak-error sign is POSITIVE (+121 m) in both cases, indicating the numerical bell has h > h_exact at the peak-error cell — consistent with phase or shape distortion rather than monotone clipping.  h_max remains ~895 in both cases (about 10.5% amplitude loss from H0=1000 due to numerical diffusion).

**What iter-784 DOES show.**
- The PPM limiter (`_pert_ppm_iv0` + face-boundary `_pert_ppm`) is NOT the dominant driver of the C36 1-day cosine bell distortion.  Turning it off entirely gives essentially the same Linf.
- h_min = 0 in both cases (the mass-target fix clips negatives even when the limiter is off), so physical positivity is preserved.
- Peak-error sign is positive in both cases — the numerical bell is displaced or distorted, not simply limiter-clipped.

**What iter-784 does NOT establish.**
- Whether a DIFFERENT limiter (e.g. hord=8 2*dm monotone, hord=10 pmp/lac) would outperform hord=9.  iter-784 only tests OFF vs ON.
- Whether the mass-target fix's "clip then rescale" is contributing to the observed error.
- Which specific part of the advection scheme (PPM reconstruction stencil, flux integration, time-splitting ordering, cross-term coupling) is the main contributor.

**Iter-785+ candidates.**
- Test time-splitting: run with Strang vs 1st-order operator splitting in `fv_tp_2d` (candidate a).
- Test cross-term handling: compare flux-form Lin-Rood (current) against fully 2D PPM (if available).
- Test mass-target fix: run WITHOUT the clip-and-rescale step and measure impact.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter784_cb_limiter_bypass.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  48th iter in iter-752-784 chain.  Rules OUT candidate (d) "PPM limiter" as primary driver.  With (b) "dt scaling" (iter-779), (c)-in-1-day (iter-781), `_d2a2c_vect` cube-vertex (iter-783), and (d) PPM limiter (iter-784) all ruled out, remaining candidate (a) "resolution-invariant structural error" is the most likely residual driver.  iter-785+ should probe time-splitting, cross-term handling, and the mass-target fix.

### Iter-785 — mass_target fix bypass rules out Python-specific post-fix

Per iter-784's iter-785+ candidate "test mass-target fix: run WITHOUT the clip-and-rescale step", iter-785 runs the C36 1-day cosine bell three times:

- **BASELINE**: normal mass-target fix (`mass_target=mass_init` → clip h < 0 to 0, then rescale positive values to total mass).
- **NO_MASS_FIX**: `mass_target=None` — no clip, no rescale.  Raw PPM output.
- **CLIP_ONLY**: clip h < 0 to 0, but no rescale.

The mass-target fix is a Python-specific, non-Fortran-faithful post-step correction (FV3 `fv_tp_2d` does not apply this rescale).

**Method** (`scripts/diag_iter785_cb_mass_target_bypass.py`; committed output at `diagnostics/iter785_output/iter785_cb_mass_target_bypass.txt`).  Same C36, β=π/4, dt=1440s, 1 day.

**Result.**

| case         | L_inf     | L2        | peak cell       | peak err signed | h_min  | h_max  | mass drift   |
|--------------|-----------|-----------|-----------------|-----------------|--------|--------|--------------|
| BASELINE     | 1.213e+02 | 1.165e-01 | face 3 (30, 32) | +1.213e+02      |  0.00  | 895.40 | 0.000e+00    |
| NO_MASS_FIX  | 1.214e+02 | 1.164e-01 | face 3 (30, 32) | +1.214e+02      | −0.01  | 895.70 | +3.39e-04    |
| CLIP_ONLY    | 1.214e+02 | 1.164e-01 | face 3 (30, 32) | +1.214e+02      |  0.00  | 895.70 | +3.39e-04    |

**Observation — numerical reportage only.**  Bypassing the mass-target fix changes Linf by only +0.1% and L2 by −0.1%.  Mass drift without the fix is +3.39e-4 (0.034% gain in 1 day — truncation-level, indicating the PPM scheme is inherently nearly-conservative).  h_min in NO_MASS_FIX is only −0.01 (negligible compared to h_max ≈ 896).

**What iter-785 DOES show.**
- The non-Fortran-faithful mass-target fix is COSMETIC for the cosine bell: its removal changes Linf by only +0.1%.
- The PPM scheme is inherently nearly-conservative (mass drift ~3e-4 over 1 day at C36 without any post-fix).
- Raw PPM output produces only trivial negatives (h_min = −0.01), so positivity is essentially preserved without the clip.

**What iter-785 does NOT establish.**
- Whether the mass-target fix should be removed from production.  It's cosmetic for cosine bell but may matter for more demanding cases or longer horizons.  That's a fidelity question requiring W2/W5/ocean-rest-state cross-checks.
- Which remaining part of the transport pipeline produces the residual 121 m peak error.
- Whether `fv_tp_2d`'s Lin-Rood cross-term handling, face-boundary PPM interpolation, or the underlying time-splitting ordering is the dominant contributor.

**Iter-786+ candidates.**
- Compare the Python duogrid path against the legacy path by forcing `dg.ng < 2` on the state.  If legacy is similar to duogrid, both paths have the same residual error; if different, the duogrid path introduces an extra error source.
- Inspect `fv_tp_2d` cross-term (Lin-Rood) timestep splitting for ordering differences vs Fortran.
- Run the cosine bell at even higher resolution (C72, C96) to pin the structural error floor more precisely.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter785_cb_mass_target_bypass.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  49th iter in iter-752-785 chain.  Rules OUT the mass-target fix as contributor to the cosine bell distortion.  Five candidates are now ruled out or non-dominant: (b) dt scaling (iter-779), (c)-in-1-day (iter-781), `_d2a2c_vect` cube-vertex (iter-783), (d) PPM limiter (iter-784), mass-target fix (iter-785).  The residual ~121 m at t=1 day C36 comes from the underlying Lin-Rood / PPM scheme + time-splitting on the cubed sphere at moderate resolution — consistent with iter-778's "plateau above C24" and with candidate (a) "resolution-invariant structural error".

### Iter-786 — `_d2a2c_vect` legacy vs duogrid path: duogrid is 3-4× WORSE at cube vertices

Follow-up to iter-782 — a re-audit revealed that iter-782's C36 measurement was taken with `create_cubed_sphere(n)` using the DEFAULT `use_duogrid=False`, so it measured the LEGACY (non-duogrid) `_d2a2c_vect` path.  The W2 production sentinels (`test_w2_alpha0_c36_1day_iter761_matrix_config`, `test_cdgrid_fv3_regression.py:3916`) also use `use_duogrid=False`, so they too hit the LEGACY path.  iter-786 compares the LEGACY path against the DUOGRID path side-by-side on the same solid-body rotation IC.

**Method** (`scripts/diag_iter786_d2a2c_legacy_vs_duogrid.py`; committed output at `diagnostics/iter786_output/iter786_d2a2c_legacy_vs_duogrid.txt`).  Same C36, β=π/4.  Two grids: `create_cubed_sphere(n, use_duogrid=False)` and `create_cubed_sphere(n, use_duogrid=True)` (ng=3).  For each, call `_d2a2c_vect` and compare against the analytical ut_exact / vt_exact.

**Result.**

| field | LEGACY peak \|err\| (rel %) | LEGACY peak location     | DUOGRID peak \|err\| (rel %) | DUOGRID peak location     | D/L ratio |
|-------|-----------------------------|--------------------------|-------------------------------|---------------------------|-----------|
| ut    | 5.740 m/s (11.69%)          | face 0 (36, 35), GC 1.16° | 1.931e+01 m/s (38.30%)       | face 0 (0, 34), GC 3.45°   | 3.363     |
| vt    | 6.174 m/s (11.40%)          | face 1 (0, 36), GC 1.16°  | 2.785e+01 m/s (49.44%)       | face 5 (0, 0), GC 1.16°    | 4.511     |

**Observation — numerical reportage only.**  Enabling `use_duogrid=True` INCREASES the peak absolute error in `_d2a2c_vect` by 3.4× (ut) and 4.5× (vt).  Relative errors jump from ~11% to 38–49%.  Both paths place peak errors at GC ≲ 3.5° of a cube vertex.  The DUOGRID path — which uses Lagrange polynomial corner fill (`fill_corner_region` in `src/legoesm/grids/duogrid.py:868`) in the `c2l_ord2 + pad_halo + cubed_a2d_halo` chain — produces LARGER cube-vertex errors than the LEGACY path's simpler 2-point corner averaging (`pad_halo_vector` + `_fill_corners_h2`).

**What iter-786 DOES show.**
- iter-782's earlier C36 measurement was on the LEGACY path (because `create_cubed_sphere(n)` defaults to `use_duogrid=False`), NOT on the duogrid path as the code comment in `fv3_sw_core.py:477-478` suggests (that comment says "Duogrid path (via `_d2a2c_vect_duogrid`, which Fortran also skips via `dg%is_initialized`) is unaffected by this gap" — but iter-786 shows the duogrid path has 3-4× MORE cube-vertex error, not less).
- The production W2 sentinels run the LEGACY path and observe v_ll_Linf ≈ 0.159 m/s at 8 cube vertices — a mode-A artifact that is consistent with the 11% `_d2a2c_vect` cube-vertex error in the LEGACY path.
- The DUOGRID path's Lagrange corner fill does NOT reduce the cube-vertex error; it increases it.

**What iter-786 does NOT establish.**
- WHY the duogrid path has larger cube-vertex error.  Candidates: (i) Lagrange corner fill overshoots at high-curvature corners; (ii) compounding error in the c2l_ord2 + scalar halo + cubed_a2d_halo chain; (iii) 4th-order A→C stencil amplifies error at non-orthogonal cells; (iv) `fill_corner_region` ordering difference from Fortran (Python pass-2 diagonal cells use updated padded; Fortran uses `veltemp`/`veltempp` snapshots).
- Whether switching W2 sentinels to `use_duogrid=True` would reduce or increase the W2 mode-A artifact.  Given duogrid's 3-4× larger cube-vertex error in `_d2a2c_vect`, a switch would likely make W2 worse, not better.
- Whether the duogrid path changes OTHER parts of the shallow-water pipeline (halo exchanges for scalars, momentum tendencies) in ways that offset the `_d2a2c_vect` degradation.

**Iter-787+ candidates.**
- Run a W2 1-hour integration under LEGACY vs DUOGRID and compare v_ll_Linf.  If DUOGRID W2 is much worse than LEGACY W2, that confirms duogrid's cube-vertex error dominates W2 behaviour.
- Decompose the duogrid `_d2a2c_vect` chain: measure the intermediate fields (A-grid lat/lon after `pad_halo`, D-grid after `cubed_a2d_halo`) to find which step introduces the cube-vertex error.
- Cross-check `fill_corner_region` ordering against Fortran `fv_duogrid.F90:1719-1903` — Python's pass-2 diagonal loop reads from padded which already contains pass-1 AND earlier pass-2 diagonal updates, while Fortran `veltemp`/`veltempp` snapshot only pass-1.
- Audit `_d2a2c_vect_duogrid` against FV3 sw_core.F90:3419-3454 for the 4th-order A→C stencil at non-orthogonal cube-vertex cells.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter786_d2a2c_legacy_vs_duogrid.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  50th iter in iter-752-786 chain.  Major finding: iter-782's 11% cube-vertex error was on the LEGACY path (the W2 sentinels' actual path), not the duogrid path.  The DUOGRID path has 3-4× LARGER cube-vertex error than the LEGACY path, contrary to the expectation that the more sophisticated Lagrange corner fill would improve accuracy.  iter-787+ should decompose the duogrid `_d2a2c_vect` chain to isolate which component introduces the degradation, and cross-check the `fill_corner_region` ordering against Fortran.

### Iter-787 — W2 under LEGACY vs DUOGRID: duogrid is 800× WORSE (production-broken)

Per iter-786's iter-787+ candidate "Run a W2 1-hour integration under LEGACY vs DUOGRID and compare v_ll_Linf", iter-787 runs the full W2 alpha=0 1-day C36 integration under both paths with the iter-761 canonical config (boundary_fix=True, damp_v=0.06, nord_v=2, div_damp = 8 × _div_damp_cube(n), hyperdiff=0).

**Method** (`scripts/diag_iter787_w2_legacy_vs_duogrid.py`; committed output at `diagnostics/iter787_output/iter787_w2_legacy_vs_duogrid.txt`).  Same n=36, dt=300s, 1 day.  Only difference: `create_cubed_sphere(n, use_duogrid=False)` vs `create_cubed_sphere(n, use_duogrid=True)`.  Measure L2 in h-drift and v_ll_Linf (iter-765 mode-A metric).

**Result.**

| path            | L2 (h drift)   | v_ll_Linf   | v_cc_Linf   |
|-----------------|----------------|-------------|-------------|
| LEGACY          | 2.176e-04      | 1.585e-01   | 1.879e-01   |
| DUOGRID         | 1.757e-01      | 9.996e+01   | 1.087e+02   |
| ratio (D / L)   | 807            | 630         | 578         |

**Observation — numerical reportage only.**  DUOGRID W2 is catastrophically worse than LEGACY W2 across all three metrics:
- L2 h-drift: 2.18e-4 → 1.76e-1 (807× worse).
- v_ll_Linf: 0.159 m/s → 100 m/s (630× worse).
- v_cc_Linf: 0.188 m/s → 109 m/s (578× worse).

The DUOGRID v_ll_Linf of ~100 m/s is comparable to the solid-body rotation wind magnitude (u0 ≈ 40 m/s), indicating gross spurious v-wind oscillation, not a small-amplitude mode-A artifact.

**What iter-787 DOES show.**
- Enabling `use_duogrid=True` in the full shallow-water pipeline at C36 produces an 800× degradation in W2 L2 and a 600× degradation in v_ll_Linf.
- The production W2 sentinels' LEGACY path gives v_ll_Linf = 0.159 m/s (iter-765 documented baseline, recovered exactly here).
- The ~100 m/s v_ll_Linf under DUOGRID indicates the duogrid path, as currently wired in the full SW operators, is PRODUCTION-BROKEN — not merely less accurate than legacy.

**What iter-787 does NOT establish.**
- Which specific operator in the duogrid shallow-water chain breaks W2.  Candidates: (a) `_d2a2c_vect_duogrid` (iter-786: 3-4× cube-vertex error); (b) `dgrid_to_cgrid` or mass-flux divergence operator in duogrid mode; (c) momentum tendencies (`cdgrid_momentum_tendencies`) in duogrid mode; (d) halo exchange chain (`pad_halo` with duogrid + `cube_rmp_vectorized` + `fill_corner_region`) for vector fields; (e) config knob incompatibility (e.g., `boundary_fix=True` under duogrid has an unintended interaction).
- Whether the DUOGRID failure mode is fixable by a known change to any ONE of those operators.
- Whether the Fortran-faithful intent of duogrid (`dg%is_initialized` branch in sw_core.F90) can be recovered by isolating and repairing the broken component.

**Iter-788+ candidates.**
- Binary-search the duogrid chain: disable duogrid-mode dispatch in individual operators (e.g., force `_d2a2c_vect_duogrid` off while keeping duogrid elsewhere) and measure which single operator's duogrid dispatch causes the W2 blowup.
- Compare `cdgrid_momentum_tendencies` and `fv3_sw_tendencies` under duogrid vs legacy at t=0 on W2 IC to find where NaN/blowup begins.
- Verify the critical duogrid constraint from the Ralph loop: "Legacy edge handling must be disabled in duogrid mode via `bounded_domain = .true.`".  Check `_bounded_domain = (base.duogrid is not None) or _is_single_face` at `cubed_sphere_cdgrid.py:890` is being consulted by every relevant operator.
- Cross-check `cube-edge flux synchronization` constraint: "Flux computation split across d_sw1/d_sw3/d_sw5 and updates across d_sw2/d_sw4/d_sw6 requires mandatory cube-edge flux synchronization before update, with synchronized flux = average(face_A_to_B, face_B_to_A)".  Audit the duogrid path for this synchronization step.

**Deliverable.**  `scripts/diag_iter787_w2_legacy_vs_duogrid.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass (they exercise LEGACY only).

**Process.**  51st iter in iter-752-787 chain.  Major production-relevant finding: the DUOGRID path is catastrophically broken for W2 (800× worse than LEGACY).  This clarifies that the production W2 sentinels' 0.159 m/s mode-A artifact is the LEGACY-path residual; the DUOGRID path — intended as the Fortran-faithful branch — is not presently a production alternative.  iter-788+ should decompose the duogrid shallow-water chain to isolate the broken component and audit the Ralph-loop-flagged duogrid constraints (bounded_domain, cube-edge flux synchronization).

### Iter-788 — W2 DUOGRID config sweep: brokenness is invariant to damping/boundary knobs

Per iter-787's iter-788+ candidate "verify Ralph-loop-flagged duogrid constraints" and "binary-search the duogrid chain", iter-788 runs the W2 alpha=0 C36 6-hour integration under the DUOGRID path with six different config knob combinations to narrow down which knob (if any) drives the blowup.

**Method** (`scripts/diag_iter788_w2_duogrid_config_sweep.py`; committed output at `diagnostics/iter788_output/iter788_w2_duogrid_config_sweep.txt`).  n=36, dt=300s, 6 hours.  Configs tested:
1. iter-761 canonical (boundary_fix=True, damp_v=0.06, nord_v=2, div_damp=8×base, hyperdiff=0).
2. boundary_fix=False, otherwise canonical.
3. damp_v=0, nord_v=0, otherwise canonical.
4. div_damp=0, otherwise canonical.
5. div_damp=1×base (unscaled), otherwise canonical.
6. All damping off (boundary_fix=False, damp_v=0, nord_v=0, div_damp=0, hyperdiff=0).

**Result.**

| DUOGRID config         | L2        | v_cc_Linf   | status  |
|------------------------|-----------|-------------|---------|
| iter-761 canonical     | 8.894e-02 | 4.165e+01   | broken  |
| boundary_fix=False     | 1.078e-01 | 8.922e+01   | broken  |
| damp_v=0 nord_v=0      | 8.894e-02 | 4.259e+01   | broken  |
| div_damp=0             | 7.588e-02 | 8.602e+01   | broken  |
| div_damp=base (1×)     | 7.102e-02 | 7.568e+01   | broken  |
| all damping off        | 1.147e-01 | 2.556e+02   | broken  |

Reference: LEGACY iter-761 canonical at 6h gives L2 = 1.225e-4, v_cc_Linf = 4.31e-2 m/s.

**Observation — numerical reportage only.**  All six DUOGRID config variants produce v_cc_Linf > 30 m/s at 6 hours (classified "broken").  The LEGACY baseline at 6h is 4.3e-2 m/s (~1000× smaller).  Disabling boundary_fix makes DUOGRID WORSE (89 m/s vs canonical 42 m/s), consistent with boundary_fix providing at least some partial stabilisation.  Disabling all damping is the worst (256 m/s).  Even the "all damping off" case has L2 ~ 0.11 (~900× LEGACY's 1.2e-4).

**What iter-788 DOES show.**
- The DUOGRID brokenness is INVARIANT to the 5 damping/boundary knobs tested — no config toggle recovers near-LEGACY performance.
- boundary_fix, damp_v/nord_v, and div_damp each provide partial damping of the duogrid failure mode (turning them off makes things worse), but none of them CAUSES the failure — the failure exists at baseline config and persists across variants.
- The failure is not in the config, it's in the duogrid-path operator wiring or dispatch.

**What iter-788 does NOT establish.**
- WHICH operator's duogrid dispatch is broken.  Candidates remain: `_d2a2c_vect_duogrid`, `dgrid_to_cgrid` duogrid branch, `cgrid_mass_flux_divergence` duogrid branch, `cdgrid_momentum_tendencies` duogrid branch, or the halo-exchange chain for vectors.
- Whether the brokenness is a missing operator change (new code needed for Fortran-faithful behaviour) or a bug (existing code is mis-wired).
- Whether the "critical duogrid constraint" from the Ralph loop brief — "Flux computation split across d_sw1/d_sw3/d_sw5 and updates across d_sw2/d_sw4/d_sw6 requires mandatory cube-edge flux synchronization before update, with synchronized flux = average(face_A_to_B, face_B_to_A)" — is correctly implemented in the duogrid path.

**Iter-789+ candidates.**
- First-step diagnosis: measure `cdgrid_shallow_water_tendencies(h, u_d, v_d, ...)` output under DUOGRID vs LEGACY at t=0 on the W2 IC; compare dh_dt, du_dt, dv_dt.  If DUOGRID tendencies have large spurious values at t=0, the wiring bug is in the tendency operator.
- Audit `_d2a2c_vect_duogrid` corner fills vs Fortran: port `veltemp`/`veltempp` snapshot semantics and re-run iter-786 + iter-787 to see if cube-vertex error drops.
- Audit the "cube-edge flux synchronization" constraint in d_sw1/d_sw3/d_sw5 against Fortran.
- Verify `bounded_domain = True` is consistently used in every duogrid-active operator.

**Deliverable.**  `scripts/diag_iter788_w2_duogrid_config_sweep.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  52nd iter in iter-752-788 chain.  Rules out config-level fixes for the DUOGRID W2 blowup: the brokenness is invariant across 6 damping/boundary variants.  The failure is in operator dispatch/wiring, not config.  iter-789+ should measure t=0 tendencies under DUOGRID vs LEGACY to isolate the broken operator.

### Iter-789 — Monkey-patch of `_d2a2c_vect` fires ZERO times in W2; function is not in production W2 path

Per iter-787/788's iter-789+ candidate "replace ut/vt with analytical at every step and measure v_ll_Linf", iter-789 monkey-patches `legoesm.core.fv3_sw_core._d2a2c_vect` to inject the analytical solid-body rotation ut/vt, then runs W2 alpha=0 C36 24h under LEGACY and compares to BASELINE.

**Result (null).**

| case             | L2         | v_ll_Linf   | v_cc_Linf   |
|------------------|------------|-------------|-------------|
| BASELINE         | 2.176e-04  | 1.585e-01   | 1.879e-01   |
| ANALYTICAL ut/vt | 2.176e-04  | 1.585e-01   | 1.879e-01   |
| Δ                | +0.0%      | +0.0%       | +0.0%       |

Debug probe inside the patched function (`_patched_trace_hits[0]`) reports **zero** calls during the W2 integration.  The patch is installed on both `fv3_sw_core._d2a2c_vect` and `operators_cdgrid._d2a2c_vect` (the latter does not import the name), so the call site is not in either module's dispatch.

**Key correction to iter-782/786 framing.**  Inspection of the call graph (`src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py:526-539` + `src/legoesm/core/operators_cdgrid.py:1187`) shows:
- `FV3EdgeShallowWaterModel.step` → `fv3_sw_tendencies` (from `operators_cdgrid`) → `dgrid_to_cgrid` (also in `operators_cdgrid`, line 298) → RK3 integrator.
- `_d2a2c_vect` (in `fv3_sw_core.py`) is NOT in this path.  It IS in the `fv3_fb_sw_step` path (used by `FV3FBShallowWaterModel`) but W2 production uses the RK3 edge-midpoint `FV3EdgeShallowWaterModel`.
- The W2 sentinels (`test_w2_alpha0_c36_1day_*`) use `FV3EdgeShallowWaterModel`, so the RK3 path with `dgrid_to_cgrid` is what actually runs.

**What iter-789 DOES establish (observational only).**
- `_d2a2c_vect` fires ZERO times during a W2 production run.  The 11% cube-vertex error measured in iter-782/786 on that function does NOT affect the W2 mode-A artifact.
- The W2 mode-A at v_ll_Linf = 0.159 m/s must originate in a different operator: `dgrid_to_cgrid`, `cdgrid_momentum_tendencies`, or downstream ops in `fv3_sw_tendencies`.

**What iter-789 does NOT establish.**
- Which operator in the `fv3_sw_tendencies` pipeline produces the W2 mode-A — iter-782/786's measurement was of the WRONG function.
- Whether `dgrid_to_cgrid` has an analogous cube-vertex error.  That is iter-790's test.

**Corrections to earlier iter claims (must re-visit).**
- iter-782 documented "`_d2a2c_vect` on the analytical solid-body rotation IC at C36 has ~6 m/s (~11% relative) error at cube vertices".  This observation IS correct for `_d2a2c_vect` in isolation, but the IMPLICATION that it's the cause of the W2 artifact was wrong — W2 doesn't call `_d2a2c_vect`.
- iter-783 override test's null result for cosine bell is unaffected (cosine bell uses `transport_step` → `fv_tp_2d`, not `_d2a2c_vect`... wait, actually the cosine bell script DOES call `_d2a2c_vect` explicitly at the top, then passes ut/vt to `transport_step`).  So iter-783's null result is valid for the cosine bell.
- iter-786's DUOGRID-vs-LEGACY `_d2a2c_vect` ratio is still valid data about that function, but the inference that DUOGRID would break W2 via this mechanism was wrong — W2 doesn't call `_d2a2c_vect`.  The DUOGRID W2 breakage observed in iter-787 must come from a different duogrid dispatch (in `dgrid_to_cgrid`, `fv3_sw_tendencies`, or their downstream ops).

**Iter-790+ candidates.**
- Re-run iter-782-style fidelity test for `dgrid_to_cgrid` (the actual W2 production D→C operator).  If `dgrid_to_cgrid` has cube-vertex error, THAT is the candidate mechanism for W2 mode-A.
- Run iter-789-style monkey-patch on `dgrid_to_cgrid` to test the analytical-injection-vs-computed W2 result.
- Re-examine DUOGRID's `dgrid_to_cgrid` and `fv3_sw_tendencies` branches for the source of the iter-787 800× blowup.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter789_w2_analytical_ut_vt.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  53rd iter in iter-752-789 chain.  KEY CORRECTION: `_d2a2c_vect` (the function measured in iter-782/786) is NOT in the production W2 pipeline.  W2 uses `dgrid_to_cgrid` in `operators_cdgrid.py`.  iter-790+ must redo the fidelity audit for `dgrid_to_cgrid` to identify the actual W2 mode-A source.

### Iter-790 — `dgrid_to_cgrid` shape-mismatch: W2 production uses `fv3_d2cc` + `fv3_cc2c`

Attempted to redo the iter-782 fidelity measurement for `dgrid_to_cgrid`, but the function raised a shape-mismatch error: `dgrid_to_cgrid` expects u_d/v_d at shape `(6, n+1, n+1)` (corner-D convention), whereas `FV3EdgeShallowWaterState.u_d` is `(6, n, n+1)` and `v_d` is `(6, n+1, n)` (edge-midpoint-D convention).  `dgrid_to_cgrid` as called directly cannot process the production state.

**Call-graph trace (`operators_cdgrid.py:1547-1584`).**  The ACTUAL D→C pipeline in `fv3_sw_tendencies` (which is what `FV3EdgeShallowWaterModel.step` uses) is:

```
fv3_sw_tendencies(h, u_d, v_d, h_s, cdgrid, ...)
  ↓
  u_cc, v_cc = fv3_d2cc(u_d, v_d, cdgrid)         # operators_cdgrid.py:1414
  u_c,  v_c  = fv3_cc2c(u_cc, v_cc, cdgrid)        # operators_cdgrid.py:1437
  dh_dt      = cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)
  # [+ KE, Bernoulli gradient, Arakawa-Lamb momentum tendencies]
```

`fv3_d2cc` is a simple 2-point average to cell centres (no non-orthogonality correction).  `fv3_cc2c` goes cell centre → C-grid via `pad_halo_vector` + 2nd-order interpolation + non-orthogonality correction via `cosa_u` / `sina_u`.  Neither operator matches the `_d2a2c_vect` 4th-order-with-Lagrange path.

The `dgrid_to_cgrid` name in `operators_cdgrid.py:298` is a DIFFERENT operator used in the "corner-D" path (`FV3FBShallowWaterModel`), not the "edge-midpoint-D" path (`FV3EdgeShallowWaterModel`) used by W2 sentinels.

**What iter-790 DOES establish (corrective only).**
- The W2 production D→C operators are `fv3_d2cc` + `fv3_cc2c` (not `_d2a2c_vect` or `dgrid_to_cgrid`).
- The stagger convention in `FV3EdgeShallowWaterState` is edge-midpoint, not corner-D, so the function named `dgrid_to_cgrid` is inapplicable.

**Iter-791+ candidates.**
- Measure `fv3_cc2c`'s cube-vertex fidelity: at each C-grid stagger cell, compare computed u_c / v_c against analytical projection of solid-body rotation (u_east, v_north) onto C-grid edge normals.  Report peak |err| + GC-to-vertex.
- Monkey-patch `fv3_cc2c` (not `_d2a2c_vect`) to inject analytical u_c / v_c and measure W2 v_ll_Linf.
- Audit the `cosa_u` / `sina_u` metrics at cube vertices — if they have extreme non-orthogonality, `fv3_cc2c`'s 2-point-avg-with-correction will amplify that error.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter790_dgrid_to_cgrid_fidelity.py` (fails at the shape-mismatch line, output shows error) + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  54th iter in iter-752-790 chain.  Inconclusive but CORRECTIVE: rules out `dgrid_to_cgrid` as the W2 D→C operator (it's shape-incompatible with `FV3EdgeShallowWaterState`).  Identifies `fv3_d2cc` + `fv3_cc2c` as the actual W2 D→C operators; iter-791+ should audit their cube-vertex fidelity.

### Iter-791 — `fv3_cc2c` cube-vertex fidelity: peak error is near the POLE, not cube vertices

Per iter-790's iter-791+ candidate "measure fv3_cc2c's cube-vertex fidelity", iter-791 feeds analytical cell-centre (u_cc, v_cc) for solid-body rotation into `fv3_cc2c` and measures the error in (u_c, v_c) against the analytical C-grid stagger values computed via the same non-orthogonality formula used inside `fv3_cc2c`.

**Method** (`scripts/diag_iter791_fv3_cc2c_fidelity.py`; committed output at `diagnostics/iter791_output/iter791_fv3_cc2c_fidelity.txt`).  C36, β=π/4.  Analytical input: u_east/v_north at cell centres projected to grid-axis (u_cc, v_cc).  Expected output: u_c_exact = u_cov_at_uc * sina_u - v_cov_at_uc * cosa_u at the u_c stagger (shape (6, n+1, n)), and v_c_exact at the v_c stagger (shape (6, n, n+1)) with no non-orthogonality correction (y-edge is along e_perp).

**Result.**

| path    | u_c peak (rel %)      | u_c location                         | v_c peak (rel %)      | v_c location                         |
|---------|-----------------------|--------------------------------------|-----------------------|--------------------------------------|
| LEGACY  | 9.05e-01 m/s (2.34%)  | face 4 (18, 18), lat +88.75°, lon -180°, GC 53.86° | 3.64e+00 m/s (10.88%) | face 4 (17, 17), lat +87.21°, lon -26.55°, GC 52.09° |
| DUOGRID | 9.05e-01 m/s (2.34%)  | face 4 (18, 18), same location       | 3.64e+00 m/s (10.88%) | face 4 (17, 17), same location       |

**Observation — numerical reportage only.**  Peak |u_c - u_c_exact| is 0.90 m/s (2.3% rel) at lat=+88.75°, lon=-180° — that's on face 4 (north face) near the north pole.  Peak |v_c - v_c_exact| is 3.64 m/s (10.9% rel) at lat=+87.21°, lon=-26.55° — also on face 4 near the pole.  Both peak locations are 52–54° GC from the nearest cube vertex (which is at arcsin(1/√3) ≈ 35.26° lat), NOT adjacent to a cube vertex.  LEGACY and DUOGRID give BYTE-IDENTICAL results for this analytical-input test.

**What iter-791 DOES show (observational).**
- `fv3_cc2c`'s peak error on analytical solid-body rotation input at C36 lives near the POLE on face 4, NOT at cube vertices.
- The LEGACY and DUOGRID halo paths give identical outputs for this particular analytical input — any difference from duogrid dispatch does not appear in `fv3_cc2c` alone on a smooth cell-centre-analytical input.

**What iter-791 does NOT establish.**
- Whether `fv3_cc2c`'s BEHAVIOUR on production W2 data (h/u_d/v_d fields that have accumulated corner-specific errors from momentum tendencies over many timesteps) shows different error localisation.  iter-791 only tests `fv3_cc2c` on smooth analytical input at t=0.
- Whether downstream operators (momentum tendencies `cdgrid_momentum_tendencies`, KE gradient via `_arakawa_lamb_gradient`, vorticity flux) amplify small upstream errors into the W2 mode-A at cube vertices.
- Whether the pole-region error in `fv3_cc2c` (~1–4 m/s) propagates into the pole region of the W2 field (which would be a DIFFERENT artifact than the cube-vertex mode-A).

**Iter-792+ candidates.**
- Measure the W2 field's v_north at t=0 right after ONE call to `fv3_sw_tendencies`: compare to zero to see which OPERATOR first produces a cube-vertex spike.  Options:
  (i) `cdgrid_momentum_tendencies` (called as part of the pipeline or separately).
  (ii) `_arakawa_lamb_gradient` on the Bernoulli function B = KE + g*(h + h_s) at D-grid corners.
  (iii) `cgrid_mass_flux_divergence` (PPM mass transport — less likely since mass conserves).
- Monkey-patch `fv3_cc2c` with analytical u_c / v_c injection and see if W2 v_ll_Linf changes (should not, given iter-791's pole-not-cube-vertex finding).
- Audit the KE gradient computation — nonlinear in cell-centre winds, so small cube-vertex noise in u_cc/v_cc amplifies quadratically in KE.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter791_fv3_cc2c_fidelity.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  55th iter in iter-752-791 chain.  Rules out `fv3_cc2c` as the direct W2 cube-vertex mode-A source: its peak error on analytical input localises near the pole, not at cube vertices.  iter-792+ should audit `_arakawa_lamb_gradient` on the Bernoulli function and `cdgrid_momentum_tendencies` for cube-vertex amplification of the quadratic KE term.

### Iter-792 — W2 t=0 tendency audit: dv/dt peak is AT a cube vertex (smoking gun for W2 mode-A)

Per iter-791's iter-792+ candidate "measure t=0 tendency of v_d right after ONE fv3_sw_tendencies call on W2 IC; locate the operator that first produces a cube-vertex spike", iter-792 calls `fv3_sw_tendencies` once at t=0 on the exact W2 solid-body rotation IC and reports the peak tendency magnitudes and locations.  For steady-state W2, analytical tendencies are ALL ZERO, so any non-zero numerical tendency is error.

**Method** (`scripts/diag_iter792_w2_t0_tendency_audit.py`; committed output at `diagnostics/iter792_output/iter792_w2_t0_tendency_audit.txt`).  C36, β=0 (polar-axis rotation), iter-761 canonical config.  Apply `fv3_sw_tendencies` once, report peak |dh/dt|, |du/dt|, |dv/dt| with location + GC-to-vertex + hot-cell (|tend| > 0.5*peak) cube-vertex / pole proximity counts.

**Result.**

| field | peak           | face, (i, j)    | lat       | lon       | GC-to-vertex | hot | near_vertex (GC<10°) | near_pole (|lat|>80°) |
|-------|----------------|------------------|-----------|-----------|--------------|-----|----------------------|-----------------------|
| dh/dt | 1.329e-04 m/s  | face 4 (18, 0)  | +46.24°   | +1.31°    | 34.38°       | 216 | 40 (18.5%)          | 0                     |
| du/dt | 1.404e-05 m/s² | face 4 (0, 25)  | +44.77°   | −108.23°  | 22.46°       | 100 | 8 (8.0%)            | 0                     |
| dv/dt | **1.903e-05 m/s²** | **face 0 (2, 34)** | **+33.90°** | **−40.00°** | **4.34°**    | **328** | **136 (41.5%)**     | **0**                 |

**Observation — numerical reportage only.**  The v-wind tendency dv/dt at t=0 peaks at (face 0, i=2, j=34) with lat=+33.90°, lon=−40.00°, which is 4.34° GC from the nearest cube vertex at (+arcsin(1/√3)≈+35.26°, −45°).  41.5% (136 of 328) of hot cells (|dv/dt| > 0.5 × peak) are within 10° GC of a cube vertex — the strongest cube-vertex concentration among the three tendency fields.  dh/dt and du/dt have weaker cube-vertex concentration (18.5% / 8.0%) and peak farther from vertices (34° / 22°).

**Crude accumulation estimate.**  If dv/dt = 1.9e-5 m/s² held constant, integrated over 1 day (86400 s), Δv = 1.64 m/s.  The observed W2 v_ll_Linf is 0.159 m/s — about 10% of the raw accumulation.  The factor-10 reduction is consistent with the damp_v=0.06 vorticity damping providing partial suppression of the accumulating mode-A.

**What iter-792 DOES show.**
- The W2 mode-A artifact at v_ll_Linf = 0.159 m/s has a CONCRETE t=0 origin: dv/dt peaks at 4.34° GC from a cube vertex on face 0, with 41.5% of hot cells near cube vertices.
- The v-tendency operator is more cube-vertex-localised than the h- or u-tendency operators.
- The artifact magnitude is quantitatively consistent with accumulating the t=0 tendency over 1 day with partial damping by damp_v.

**What iter-792 does NOT establish.**
- Which SPECIFIC operator INSIDE `fv3_sw_tendencies` produces the dv/dt cube-vertex peak.  Candidates:
  (i) `_arakawa_lamb_gradient(B, ...)` with B = KE + g*(h+h_s) at D-grid corners (nonlinear in u_cc/v_cc).
  (ii) Corner winds from halo-exchanged cell-centre velocities (line 1607).
  (iii) Vorticity transport (Coriolis × u_d cross-product).
  (iv) Divergence damping (div_damp = 8 × _div_damp_cube(n)).
- Whether a Fortran-faithful fix to any ONE of those sub-operators would reduce dv/dt at cube vertices and propagate to a W2 v_ll_Linf reduction.

**Iter-793+ candidates.**
- Decompose dv/dt into contributions from each term of `fv3_sw_tendencies` (lines 1601-1700-ish): gradient-of-B, Coriolis-u_d, vorticity transport, divergence damping.  Report each term's peak and cube-vertex concentration.
- Toggle `fortran_dir_aware_corners`, `fortran_a2b_corner_avg`, `fortran_vector_corner_fill` individually and measure dv/dt cube-vertex peak change.
- Audit `_arakawa_lamb_gradient`'s halo handling vs Fortran `sw_core.F90:3856-3915`.
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter792_w2_t0_tendency_audit.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  56th iter in iter-752-792 chain.  Major mechanistic finding: dv/dt at t=0 for W2 solid-body rotation has a cube-vertex peak (GC=4.34°) with 41.5% hot-cell cube-vertex concentration.  Magnitude (1.9e-5 m/s²) is quantitatively consistent with the observed 0.159 m/s v_ll_Linf after 1 day with partial damping.  iter-793+ should decompose the v-tendency terms to identify which operator produces the cube-vertex signal.

### Iter-793 — dv_cc decomposition: cube-vertex signal is INCOMPLETE cancellation of Coriolis+pressure+KE

Per iter-792's iter-793+ candidate "decompose dv/dt into contributions from each term of fv3_sw_tendencies", iter-793 evaluates each individual dv_cc term at t=0 on the W2 IC and reports peak + GC-to-vertex + hot-cell cube-vertex concentration.

**Method** (`scripts/diag_iter793_w2_dv_decomposition.py`; committed output at `diagnostics/iter793_output/iter793_w2_dv_decomposition.txt`).  C36, β=0, LEGACY iter-761 canonical config.  Replicate `fv3_sw_tendencies` internals (minus boundary_fix) to extract each dv_cc contribution separately.

**Result.**

| term                      | peak          | face,(i,j)     | lat      | lon       | GC      | hot  | near_vert     |
|---------------------------|---------------|----------------|----------|-----------|---------|------|---------------|
| dv total (with div damp)  | 3.966e-05     | (4, 35, 35)    | +36.45°  | +135.00°  | **1.19°** | 24   | 20 (**83.3%**) |
| dv total (NO div damp)    | 2.516e-05     | (5, 0, 0)      | −36.45°  | −135.00°  | **1.19°** | 8    | 8 (**100.0%**) |
| dv Coriolis (−zeta*u)     | 3.045e-03     | (4, 17, 0)     | +46.24°  | −1.31°    | 34.38°  | 4896 | 528 (10.8%)   |
| dv pressure (−∂ᵧ gh)      | 2.926e-03     | (0, 17, 0)     | −43.74°  | −1.25°    | 34.38°  | 4896 | 528 (10.8%)   |
| dv KE-grad (−∂ᵧ KE)       | 1.167e-04     | (1, 18, 0)     | −43.74°  | +91.25°   | 34.38°  | 4896 | 528 (10.8%)   |
| dv balance (Cor+press)    | 1.210e-04     | (4, 17, 0)     | +46.24°  | −1.31°    | 34.38°  | 4832 | 528 (10.9%)   |
| dv div-damp               | 1.825e-05     | (4, 33, 0)     | +38.68°  | +39.98°   | **5.27°** | 288  | 160 (**55.6%**) |
| zeta (rel vort)           | 1.210e-05     | (4, 17, 18)    | +88.23°  | −135.00°  | 52.97°  | 4000 | 432 (10.8%)   |

**Observation — numerical reportage only.**  The individual constituent terms (Coriolis, pressure gradient, KE gradient, the "balance" Cor+press) each peak at mid-face (GC ≈ 34°) with only ~11% hot-cell cube-vertex concentration — their own truncation-level noise is NOT cube-vertex-localised.  But when summed, the residual dv total peaks at GC = 1.19° with 83–100% cube-vertex concentration.  This is the signature of INCOMPLETE CANCELLATION: the constituent terms almost cancel for geostrophic W2, but the cancellation FAILS at cube vertices where halo-interpolation bias differs between terms.

**Quantitative breakdown.**
- Coriolis, pressure, KE-grad each have peak ~3e-3 m/s² at mid-face — O(1) large.
- Their numerical sum (dv total WITHOUT div damp) is 2.5e-5 m/s² at cube vertices — 100× smaller.  The cancellation is 99%+ effective at mid-face but only ~99% effective at cube vertices (leaving the 2.5e-5 residual).
- Div damping contributes 1.8e-5 m/s² at cube vertices, raising the total to 4.0e-5.
- The "dv balance" (Cor + press, no KE grad) peaks at mid-face — so KE grad is REQUIRED for the cancellation.  KE grad provides a ~O(10⁻⁴) correction that fixes the mid-face imbalance but introduces cube-vertex-specific error.

**What iter-793 DOES show.**
- The W2 mode-A cube-vertex signal is NOT caused by any single operator with a bug.  Each term (Coriolis, pressure, KE, div damp) has uniform-magnitude mid-face error; the cube-vertex localisation emerges from INCOMPLETE CANCELLATION between them.
- The KE gradient is a participant in the cancellation: including it moves the error pattern, doesn't eliminate it.
- Divergence damping adds ~37% to the cube-vertex total (turning it off reduces peak from 4.0e-5 to 2.5e-5).
- The `zeta` (relative vorticity) is tiny (1.2e-5) — consistent with solid-body rotation having analytical zero vorticity — so the Coriolis term's ~3e-3 peak is dominated by `f * u_cc`, not `zeta * u_cc`.

**What iter-793 does NOT establish.**
- A specific Fortran-faithful fix.  The cancellation failure is structural, arising from halo-interpolation differences between the three gradient terms at cube vertices.  A proper fix requires matching Fortran's cube-vertex halo treatment for `h`, `u_cc`, `v_cc`, and `KE` simultaneously — or replacing the A-L+RK3 pipeline with the Fortran d_sw chain (blocked on ng=3 halo infrastructure).
- Whether the iter-761 boundary_fix stabiliser partially masks this error (it does — iter-511 measured 2.4× L2 worsening when boundary_fix=False).

**Iter-794+ candidates.**
- Port ng=3 halo infrastructure to unblock the FB chain (Fortran-faithful d_sw path that intrinsically handles cube-vertex cancellation).
- Test whether applying `_fortran_agrid_vector_corner_fill` simultaneously to `u_cc_pad`, `v_cc_pad`, AND the `h_pad` used by `_arakawa_lamb_gradient` internally reduces the cube-vertex residual.  Currently only the vector pair is treated consistently (via `fortran_vector_corner_fill`); the scalar `h` halo may use a different corner convention.
- Compare the `_arakawa_lamb_gradient` halo treatment for scalars (h, KE) vs vectors (u_cc, v_cc) at cube vertices.
- Decompose dv_cc contributions on a W2 RUN (not just t=0) — see if the imbalance grows, stays constant, or damps via damp_v.

**Deliverable.**  `scripts/diag_iter793_w2_dv_decomposition.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  57th iter in iter-752-793 chain.  Mechanistic decomposition: W2 cube-vertex mode-A is INCOMPLETE cancellation between Coriolis, pressure gradient, and KE gradient at cube vertices (with 37% additional contribution from div damping).  No single operator is individually "broken"; the issue is halo-interpolation inconsistency between the three gradient terms.  iter-794+ should either unblock the FB chain (ng=3) or audit cross-term halo consistency at cube vertices.

### Iter-794 — W2 LEGACY div_damp sweep: damping SUPPRESSES mode-A, doesn't amplify it

Per iter-793's iter-794+ candidate, iter-794 sweeps `div_damp` on the LEGACY path to measure v_ll_Linf sensitivity.  iter-793 showed `dv div-damp` contributes ~37% of the cube-vertex residual at t=0; iter-794 tests whether this carries through to the 1-day v_ll_Linf.

**Method** (`scripts/diag_iter794_w2_divdamp_sweep.py`; committed output at `diagnostics/iter794_output/iter794_w2_divdamp_sweep.txt`).  C36, 24h, dt=300s, iter-761 config but varying div_damp ∈ {0, 1×, 4×, 8×, 16×} × `_div_damp_cube(n)`.

**Result.**

| div_damp      | L2          | v_ll_Linf   | v_cc_Linf   | Δ vs 8× (iter-761)     |
|---------------|-------------|-------------|-------------|------------------------|
| 0 (none)      | 3.438e-04   | 2.337e-01   | 2.707e-01   | L2 +58%, v_ll +47%     |
| 1× base       | 3.092e-04   | 2.139e-01   | 2.447e-01   | L2 +42%, v_ll +35%     |
| 4× base       | 2.506e-04   | 1.757e-01   | 2.069e-01   | L2 +15%, v_ll +11%     |
| 8× (iter-761) | 2.176e-04   | 1.585e-01   | 1.879e-01   | —                      |
| 16× base      | NaN         | NaN         | NaN         | blow-up (unstable)      |

**Observation — numerical reportage only.**  v_ll_Linf DECREASES monotonically with INCREASING div_damp (from 0.234 m/s at div_damp=0 to 0.159 m/s at 8×).  L2 also decreases.  16× blows up to NaN (too much damping destabilises).  iter-761's 8× is indeed near the sweet spot — both L2 and v_ll_Linf are minimised at 8×, with L2 degrading 15% / v_ll_Linf degrading 11% at 4×, and both degrading ~50% at div_damp=0.

**Reconciliation with iter-793.**  Iter-793 reported `dv div-damp` contributes +37% to the t=0 cube-vertex residual.  Iter-794 shows div_damp REDUCES the 1-day v_ll_Linf by ~47%.  These are CONSISTENT: div_damp at t=0 has a non-zero cube-vertex signal (a TRANSIENT amplification), but over many timesteps its DISSIPATIVE nature dominates — damping out the cube-vertex mode-A faster than it accumulates.  iter-793's t=0 snapshot over-emphasised div_damp's role; the time-integrated behaviour is NET suppression.

**What iter-794 DOES show.**
- iter-761's 8× div_damp scaling is near-optimal for W2 at C36.  No improvement from smaller or larger values within stability.
- div_damp is NET STABILISING: more damping → smaller artifact, until 16× destabilises.
- The residual W2 mode-A at 0.159 m/s cannot be meaningfully reduced by tuning div_damp alone.

**What iter-794 does NOT establish.**
- Whether a DIFFERENT stabiliser (e.g. del-4 vorticity damping, a different Smag cap) would further reduce mode-A without the 16× blow-up.
- Whether the 8× sweet spot at C36 is also optimal at C48/C72.
- Whether reducing mode-A below 0.159 m/s requires structural operator changes (FB chain port, consistent cross-term halo fills), not config tuning.

**Iter-795+ candidates.**
- Probe the cube-vertex cancellation hypothesis directly: compute B at cube-vertex halo cells from HALO'd (u_cc, v_cc, h, h_s) components rather than halo'ing the combined B as a scalar.  If this changes the cube-vertex residual in the predicted direction, implement a Fortran-faithful halo ordering for B.
- Audit Fortran `c_sw` KE construction (sw_core.F90:295-370) — uses `ua*ke + va*vort` with upwind sin_sg blending at face boundaries.  This is NOT the same as our cell-centre `0.5*(u² + v²)` + A-L gradient.  A direct port of Fortran's c_sw KE term would change the KE-gradient semantics.
- Investigate whether the DUOGRID path in `fv3_sw_tendencies` (iter-787 blowup) has a different cross-term halo convention that could be backported to LEGACY without the blowup.

**Deliverable.**  `scripts/diag_iter794_w2_divdamp_sweep.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  58th iter in iter-752-794 chain.  Confirms iter-761's 8× div_damp is near-optimal for W2 at C36; the residual 0.159 m/s mode-A cannot be reduced by div_damp tuning alone.  Further improvement requires structural cross-term halo consistency (iter-795+) or the FB chain port.

### Iter-795 — Component-consistent B-halo hypothesis REFUTED for smooth W2 IC

Per iter-794's iter-795+ candidate "probe the cube-vertex cancellation hypothesis directly: compute B at cube-vertex halo cells from HALO'd (u_cc, v_cc, h, h_s) components rather than halo'ing the combined B as a scalar", iter-795 tests this directly at t=0 on the W2 IC.

**Method** (`scripts/diag_iter795_w2_consistent_B_halo.py`; committed output at `diagnostics/iter795_output/iter795_w2_consistent_B_halo.txt`).  C36, β=0.  Construct B_pad_consistent: start from `pad_halo(B)` (default), then at the 4 cube-vertex halo cells per face, overwrite with `0.5*(u_halo² + v_halo²) + g*(h_halo + h_s_halo)` where u_halo, v_halo come from `pad_halo_vector` and h_halo, h_s_halo come from `pad_halo`.  Pass this padded B to `_arakawa_lamb_gradient(B, cdgrid, padded=B_pad_consistent)` via its explicit `padded` parameter (already supported for stage-level packing).  Compare dv_cc residual (without div damp) to the default path.

**Result.**

| variant              | max \|dB_dy_cc_cons − dB_dy_cc_def\| | max \|dv_cc\| (no div damp) | peak location           |
|----------------------|---------------------------------------|-----------------------------|-------------------------|
| default halo         | —                                     | 2.516e-05                   | (5, 0, 0), −36.45°, −135°  |
| consistent B halo    | 3.83e-07 m/s²                         | 2.516e-05                   | (4, 35, 35), +36.45°, +135° |

Ratio (consistent / default) = 1.000.

**Observation — numerical reportage only.**  At t=0 for the smooth W2 IC, the difference between scalar B halo and component-consistent B halo at cube-vertex cells is ~4e-7 m/s² — four orders of magnitude smaller than the dv_cc residual (2.5e-5).  The dv_cc peak is UNCHANGED to 4 significant digits; only the peak location shifts from one cube vertex to the symmetric mirror (SW ↔ NE).  The hypothesis that B-halo inconsistency drives the W2 mode-A is REFUTED for smooth steady-state input.

**Why the hypothesis failed.**  For solid-body rotation, u_cc, v_cc, h, h_s are all smooth across cube vertices.  The scalar 2-pt average `0.5*(B_A + B_B)` at a cube-vertex halo cell is nearly identical to `0.5*(u_A² + v_A² + u_B² + v_B²) + g*0.5*(h_A + h_B + h_s_A + h_s_B)` because the quadratic terms u² and v² are also smooth.  The difference would show up for non-smooth fields (e.g. cosine bell near its edge) but not for solid-body rotation.

**What iter-795 DOES show.**
- B-halo choice (scalar vs component-consistent) does not drive the W2 cube-vertex mode-A for smooth IC.
- The cube-vertex residual (2.5e-5) comes from another mechanism, NOT from the B halo treatment.

**What iter-795 does NOT establish.**
- Whether the component-consistent halo matters for NON-SMOOTH cases (cosine bell, breaking waves).
- What the actual cube-vertex mechanism is.  Remaining candidates:
  (i) Corner-wind interpolation — u_corner, v_corner from 4-point avg of halo'd u_cc, v_cc; this enters zeta which enters Coriolis.
  (ii) A-L gradient stencil ITSELF has cube-vertex behaviour that differs from interior.
  (iii) `_interp_corner_to_center` (corner → cell centre projection, used for dB_dx_cc and dB_dy_cc).

**Iter-796+ candidates.**
- Zero-out `u_corner`, `v_corner` contribution at cube vertices — compute zeta from analytical (exactly zero for solid body) and see if dv_cc residual drops.
- Replace `_interp_corner_to_center` with a smoother/different operator at cube-vertex-adjacent cells.
- Direct W2 measurement at C48/C72 to see if residual scales with grid spacing (would indicate truncation-level, otherwise structural).
- Port Fortran FB transport chain (blocked on ng=3).

**Deliverable.**  `scripts/diag_iter795_w2_consistent_B_halo.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  59th iter in iter-752-795 chain.  Rules out scalar-vs-component B-halo inconsistency as the W2 mode-A mechanism for smooth solid-body rotation IC (the candidate contributed only 4e-7 m/s² difference, 4 orders below the residual).  Remaining candidate mechanisms: corner-wind interpolation, A-L gradient stencil itself, `_interp_corner_to_center`.

### Iter-796 — Numerical zeta is REQUIRED for Cor+press+KE cancellation (reversed sign!)

Per iter-795's iter-796+ candidate "zero-out u_corner, v_corner contribution at cube vertices — compute zeta from analytical (exactly zero for solid body) and see if dv_cc residual drops", iter-796 tests this directly at t=0 on the W2 IC.

**Method** (`scripts/diag_iter796_w2_zeta_zero.py`; committed output at `diagnostics/iter796_output/iter796_w2_zeta_zero.txt`).  C36, β=0, LEGACY path, no div damp applied.  Compute dv_cc with numerical zeta (default) and with zeta replaced by analytical ZERO.

**Result.**

| variant                   | peak       | face,(i,j) | lat     | lon      | GC      | hot  | near_vert      |
|---------------------------|------------|------------|---------|----------|---------|------|----------------|
| default (numerical zeta)  | 2.516e-05  | (5, 0, 0)  | −36.45° | −135.00° | **1.19°** | 8    | 8 (**100%**)   |
| zeta = 0 (analytical)     | 2.307e-04  | (1, 17, 0) | −43.74° | +88.75°  | 34.38°  | 4896 | 528 (10.8%)    |

Ratio (zeta_zero / default) = **9.17×** — zeroing zeta makes the residual ~10× LARGER!

**Observation — numerical reportage only.**  REVERSED expectation: setting zeta = 0 (analytical for solid-body) makes the dv_cc residual peak 10× LARGER (2.5e-5 → 2.3e-4), not smaller.  Also, the peak location MOVES from cube vertex (GC=1.19°) to mid-face (GC=34.38°), and hot-cell cube-vertex concentration drops from 100% to 10.8%.

**Interpretation.**  Numerical zeta (relative vorticity) provides CRUCIAL CANCELLATION of the Coriolis + pressure + KE gradient imbalance produced by `_arakawa_lamb_gradient` at cell centres.  Specifically:
- dv_Cor+press+KE (without zeta*u) = 2.3e-4 at mid-face.
- Adding −zeta*u_cc (numerical zeta) provides an opposite-signed mid-face term that cancels 90%+ of the imbalance.
- The cancellation is near-perfect at mid-face but fails at cube vertices, leaving the 2.5e-5 cube-vertex residual.

This is a concrete mechanistic understanding: the A-L+RK3 pipeline achieves geostrophic balance for W2 through a FORTUITOUS CANCELLATION between the gradient-of-B truncation error and the corner-wind-derived zeta truncation error.  The cancellation is grid-dependent and only fully works at mid-face; cube vertices are where the cancellation breaks down, producing the observed 0.159 m/s v_ll_Linf mode-A.

**What iter-796 DOES show.**
- The W2 mode-A at 0.159 m/s is NOT a single-operator bug.  It's the CUBE-VERTEX BREAKDOWN of a grid-global cancellation between two independent truncation errors (gradient-of-B and zeta).
- Numerical zeta is REQUIRED for the current balance to work; removing it breaks W2 by an order of magnitude.
- The remaining cube-vertex residual cannot be repaired by fixing either operator in isolation — they must be made CONSISTENT at cube vertices.

**What iter-796 does NOT establish.**
- A specific Fortran-faithful fix.  The Fortran c_sw chain achieves the balance differently (via upwind ke formula with sin_sg/cos_sg blending at face boundaries, sw_core.F90:295-339), avoiding the A-L+RK3 cancellation dependency.  Porting this is blocked on ng=3 and FB-chain stability (iter-787 showed current FB chain is broken at C36).
- Whether a different discretisation of zeta (e.g., via circulation form, which is used in `dgrid_vorticity` — check its cube-vertex behaviour) would improve the cancellation.

**Iter-797+ candidates.**
- Audit `dgrid_vorticity` (circulation form of zeta) against a Fortran-equivalent vorticity operator — check if the cube-vertex cancellation can be improved.
- Test whether the Fortran `fill_corners_agrid_r8` applied to both u_cc/v_cc halo (for zeta) AND h halo (for pressure gradient) simultaneously could shift the cancellation sweet spot from mid-face to uniform.
- Visual inspection of the current W2 v-wind at C36 — is 0.159 m/s v_ll_Linf actually VISIBLE in the snapshot, or is it below the display threshold?  Update `diagnostics/fv3_visual/*.png` if visual inspection is ambiguous.
- Port Fortran FB transport chain (blocked on ng=3 and FB-chain stability — iter-787 showed current FB is broken at C36).

**Deliverable.**  `scripts/diag_iter796_w2_zeta_zero.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  60th iter in iter-752-796 chain.  Mechanistic clarification: the W2 mode-A emerges from a grid-global Cor+press+KE+zeta cancellation whose cube-vertex breakdown is the 0.159 m/s residual.  Numerical zeta is a LOAD-BEARING component of this cancellation; zeroing it makes W2 10× worse.  A Fortran-faithful fix requires either (a) porting Fortran's c_sw KE/vort construction that avoids the A-L cancellation dependency, or (b) making zeta and the gradient-of-B operators cube-vertex-consistent.

### Iter-797 — Fix `diag_w2_visual.py` to use production config + accurate IC; new visual observations

Per iter-796's iter-797+ candidate "Visual inspection of the current W2 v-wind at C36 — is 0.159 m/s v_ll_Linf actually VISIBLE in the snapshot, or is it below the display threshold?", iter-797 audited the `scripts/diag_w2_visual.py` harness and found two bugs that made the committed visual plots UNREPRESENTATIVE of production:

1. **Config drift**: The script used `CDGridShallowWaterConfig(use_experimental_csw=use_csw, boundary_fix=True)` which defaults `div_damp=0.0`, `damp_v=0.0`, `nord_v=-1`.  The production matrix uses iter-761 canonical: `div_damp=8*_div_damp_cube(n)`, `damp_v=0.06`, `nord_v=2`.
2. **IC construction inaccuracy**: The script built u_d, v_d by rotating cell-centre (u_east, v_north) to grid axes, padding, and averaging to edges — a cell-centre-mediated approximation.  The production sentinel uses the analytical formulas `u_d = cos_angle_edge_x * u0 * cos(lat_edge_x)` and `v_d = -sin_angle_edge_y * u0 * cos(lat_edge_y)` directly at edge positions (accurate to machine precision).
3. **dt mismatch**: The script used dt=600s, 144 steps; production uses dt=300s, 288 steps.

**Fix.**  Updated `diag_w2_visual.py` to use the iter-761 canonical config, match the production IC construction, and match dt=300s.  Regenerated all plots (`w2_vnorth_production.png`, `w2_herr_production.png`, `w2_herr_csw_dg.png`, `w2_vnorth_csw_dg.png`, `w2_herr_improvement.png`).

**Before vs after.**

| metric                     | before fix     | after fix    |
|----------------------------|----------------|--------------|
| C24 day 1 h_err max        | 225 m          | 5 m          |
| C24 day 1 v_north max      | 57.2 m/s       | 0.6 m/s      |

The prior plots were displaying a mis-configured run, not production.  The new plots reflect the actual production state at C24.

**New visual observations on the corrected plots.**
- **v_north**: colour scale ~±0.6 m/s.  Equatorial faces (0, 1, 2, 3) show subtle edge striping near cube vertices (the mode-A we've characterised).  Polar faces (4, 5) show a distinct X-shaped pattern at the pole where face-local coordinates become singular — this is a pole-coordinate artifact, NOT a cube-vertex one.
- **h_err**: colour scale ~±4 m.  Equatorial faces show a clear 2dx CHECKERBOARD pattern in the interior — a grid-scale mode that the iter-761 config does not fully damp.  Cube-vertex-adjacent cells have stronger checkerboard + edge striping.  Polar faces show edge-striping along cube-vertex rows.

**The 2dx checkerboard in h_err is a NEW visible artifact that earlier iterations did not surface.**  It was masked in the prior plot by the much larger ±225 m scaling.  At C24 with iter-761 config, this grid-scale h noise exists at ~1-2 m amplitude in the interior.  It is likely driven by insufficient dissipation of 2dx modes in the PPM mass transport or the boundary_fix smoothing pattern.

**What iter-797 DOES show.**
- The committed production W2 visual was a mis-configured run, not production state.  After fix, C24 v_north max drops from 57 to 0.6 m/s.
- The REAL production W2 visual artifacts at C24 are: (i) mild cube-vertex edge striping in v_north (~0.5 m/s), (ii) pole-coordinate X-pattern in v_north at the geographic pole (~0.5 m/s), (iii) 2dx checkerboard in h_err interior (~1-2 m).
- The W2 mode-A at 0.159 m/s v_ll_Linf (C36) or equivalent at C24 IS visible but mild.

**What iter-797 does NOT establish.**
- Whether the 2dx h checkerboard is always present or specific to C24.  Should check C36 / C48 plots to see resolution scaling.
- The origin of the h checkerboard: PPM transport, boundary_fix cascaded smoothing, or aliasing from the Arakawa-Lamb operator.
- Whether the pole X-pattern is related to the cube-vertex mode-A or a distinct pole-singularity artifact.

**Iter-798+ candidates.**
- Regenerate visual plots at C36 and C48 (matching the production grid) — document the artifact scaling.
- Investigate the 2dx h checkerboard origin (is it from PPM transport's inherent 2dx null mode, or from the boundary_fix cascaded smoothing pattern?).
- Audit the pole-coordinate X-pattern on faces 4, 5 — document as known pole artifact vs investigate if it's tied to mode-A.
- Port Fortran FB transport chain (blocked on ng=3 and FB-chain stability).

**Deliverable.**  Updated `scripts/diag_w2_visual.py` + regenerated PNG plots in `diagnostics/fv3_visual/`.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  61st iter in iter-752-797 chain.  Fixes the W2 visual diagnostic to accurately reflect production behaviour.  Surfaces 3 distinct visible artifacts previously masked by the misconfig: cube-vertex v_north edge striping (the mode-A we've been tracking), pole-coordinate X-pattern, and interior 2dx h checkerboard.  The checkerboard is a new observation and should be investigated in iter-798+.

### Iter-798 — W2 h-checkerboard amplitude vs resolution: small and non-monotone

Per iter-797's iter-798+ candidate "investigate 2dx h checkerboard origin and scaling", iter-798 measures the 2dx checkerboard amplitude in W2 1-day h_err at C24/C36/C48 with identical iter-761 config.

**Method** (`scripts/diag_iter798_w2_checkerboard_scaling.py`; committed output at `diagnostics/iter798_output/iter798_w2_checkerboard_scaling.txt`).  W2 alpha=0 1-day, iter-761 canonical.  At each resolution extract face-averaged checkerboard amplitude via `sign = (-1)^(i+j)` applied to interior cells (3-cell boundary buffer), then absolute mean.

**Result.**

| n  | dx [km] | max \|h_err\| [m] | face-avg checker amp | face-4 max \|h_err\| |
|----|---------|------------------|----------------------|-----------------------|
| 24 | 417.0   | 4.515            | 2.486e−03            | 4.515                 |
| 36 | 278.0   | 4.588            | 4.833e−04            | 4.569                 |
| 48 | 208.5   | 6.598            | 1.464e−03            | 6.551                 |

Scaling ratios:
- C24→C36 (dx ratio 0.67×): checkerboard amp ratio = 0.19 (faster than dx² = 0.44).
- C36→C48 (dx ratio 0.75×): checkerboard amp ratio = 3.03 (INCREASES with resolution).

**Observation — numerical reportage only.**  The 2dx checkerboard amplitude is small (~1e-3) compared to max |h_err| (~5 m) at all three resolutions — i.e. the "checkerboard" I described visually in iter-797 is not the dominant h_err pattern.  The dominant pattern is a broader equatorial/polar bell-shaped deviation with max ~5 m amplitude.  The checkerboard amplitude scales non-monotonically (shrinks C24→C36, grows C36→C48), suggesting grid-resolution-specific interactions with the boundary_fix cascaded smoothing or PPM transport, rather than a pure truncation-level error.

Max |h_err| is NON-CONVERGENT: 4.5 at C24, 4.6 at C36, 6.6 at C48.  This is consistent with iter-778/779's cosine-bell plateau finding — the structural W2 h error does not reduce with grid refinement, indicating the residual comes from the discretisation, not resolution-dependent truncation.

**What iter-798 DOES show.**
- The 2dx checkerboard in h_err is small (amp ~ 1e-3) compared to the overall h_err (~5 m).
- Max |h_err| does NOT decrease with resolution from C24 to C48 — consistent with structural, not truncation, error.
- The checkerboard does not scale as dx² at C36→C48 (it grows 3× instead of shrinking to 0.56×) — not pure truncation.

**What iter-798 does NOT establish.**
- Whether the checkerboard comes from PPM's 2dx null mode, the boundary_fix smoothing, or A-L aliasing.
- Whether the C48 checkerboard growth is a genuine structural increase or a specific-grid fluke.

**Iter-799+ candidates.**
- Investigate why h_err is non-convergent from C24 to C48.  The residual h_err is ~5-7 m regardless of grid — same structural signature as cosine bell plateau.
- Test the Fortran FB chain at C48 — if it's stable there, it could be used as a production path for finer grids even if broken at C36.
- Combine multi-knob Fortran corner fills (vector + a2b scalar) — currently only single-knob tests exist, all making W2 worse.
- Port Fortran FB transport chain (blocked on ng=3 and FB-chain stability).

**Deliverable.**  `scripts/diag_iter798_w2_checkerboard_scaling.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  62nd iter in iter-752-798 chain.  Downgrades iter-797's "2dx checkerboard" concern: the checkerboard amplitude is ~1000× smaller than max |h_err|, and max |h_err| is non-convergent across resolutions — consistent with the structural-error plateau we've documented for cosine bell (iter-778/779) and W2 (iter-782 through iter-796).  The mode-A and h_err residuals are all manifestations of the same underlying discretisation issue: the A-L+RK3 pipeline cannot fully eliminate cube-vertex cancellation residuals without the Fortran c_sw construction.

### Iter-799 — DUOGRID dh/dt at t=0 is 1172× LEGACY (mass transport is catastrophically broken)

Per iter-787's finding that DUOGRID W2 is 800× worse than LEGACY after 1 day, iter-799 localises the broken operator by measuring t=0 tendencies.

**Method** (`scripts/diag_iter799_duogrid_t0_tendency.py`; committed output at `diagnostics/iter799_output/iter799_duogrid_t0_tendency.txt`).  C36, iter-761 config, analytical W2 IC.  Compute tendencies ONCE at t=0 under `use_duogrid=False` and `use_duogrid=True`; report peak magnitudes.

**Result.**

| path    | dh/dt peak    | du/dt peak    | dv/dt peak    |
|---------|---------------|---------------|---------------|
| LEGACY  | 1.329e−04     | 1.404e−05     | 1.903e−05     |
| DUOGRID | 1.557e−01     | 4.700e−05     | 8.446e−05     |
| ratio   | **1171.57×**  | 3.35×         | 4.44×         |

**Observation — numerical reportage only.**  At t=0 for steady-state W2, analytical tendencies are ALL ZERO.  The LEGACY numerical tendencies are tiny (~1e−4 to 1e−5).  The DUOGRID numerical tendencies are **three orders of magnitude larger for dh/dt** (0.16 m/s mass tendency at t=0, which over 86400s gives 1.4e4 m h-drift — approaching the mean h0 ~ 3000 m).  du/dt and dv/dt are only 3-4× worse, suggesting the momentum side is less broken than mass transport.

**Implication.**  The DUOGRID path's `cgrid_mass_flux_divergence(h, u_c, v_c, cdgrid)` produces a catastrophic non-zero divergence at t=0 on steady-state input.  Since dh/dt is driven by the mass flux divergence, and u_c / v_c come from `fv3_cc2c(u_cc, v_cc, cdgrid)` with duogrid halo (line 1458: `offsets = None if dg is not None else grid.halo_interp_offsets`), the DUOGRID halo of (u_cc, v_cc) must produce u_c, v_c whose divergence is LARGE near cube vertices.

This is a CONCRETE t=0 signature of the duogrid path breakage — it's not from time-step accumulation, the halo-exchange chain is broken immediately.

**What iter-799 DOES show.**
- DUOGRID's t=0 dh/dt peak is 1172× LEGACY's.  Catastrophic at the mass transport step.
- The breakage is INSTANT — not from accumulation.  The duogrid halo dispatch in `fv3_cc2c` (or upstream via `pad_halo_vector(duogrid=dg)`) is producing u_c, v_c that have large spurious divergence near cube vertices for steady-state input.

**What iter-799 does NOT establish.**
- WHICH specific duogrid-mode component is broken: (a) `pad_halo_vector(duogrid=dg)` at cube corners, (b) the A-grid-like scalar halo for h inside `cgrid_mass_flux_divergence`, (c) a missing `bounded_domain` gate in one of the downstream operators.
- Whether the du/dt, dv/dt 3-4× worsening is a separate issue or downstream of the mass-transport breakage.

**Iter-800+ candidates.**
- Measure u_c, v_c from `fv3_cc2c` at t=0 under LEGACY vs DUOGRID directly; compare to analytical.  If DUOGRID u_c, v_c have cube-vertex spikes, the `pad_halo_vector(duogrid=dg)` is confirmed broken.
- Check the divergence of analytical solid-body (u_c, v_c) vs numerical at cube vertices under both paths.
- Audit `cgrid_mass_flux_divergence`'s handling of `base.duogrid is not None` (does it switch halo paths?).
- Audit the Ralph-loop-flagged critical duogrid constraint: "Legacy edge handling must be disabled in duogrid mode via `bounded_domain = .true.`" — confirm every relevant operator consults `_bounded_domain`.

**Deliverable.**  `scripts/diag_iter799_duogrid_t0_tendency.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  63rd iter in iter-752-799 chain.  Major DUOGRID diagnostic finding: the mass-transport chain (`fv3_cc2c` → `cgrid_mass_flux_divergence`) produces a t=0 dh/dt 1172× larger than LEGACY on steady-state W2.  This is an INSTANT halo/operator breakage, not time-step accumulation.  iter-800+ should localise which specific duogrid halo or operator produces the spurious divergence.

### Iter-800 — DUOGRID blowup is in `cgrid_mass_flux_divergence` PPM halo at cube vertices

Per iter-799's finding, iter-800 decomposes the DUOGRID mass-transport chain at t=0 on W2 IC, measuring u_c, v_c, ∇·u_c (raw divergence), and mass flux divergence dh/dt separately.

**Method** (`scripts/diag_iter800_duogrid_uc_vc_divergence.py`; committed output at `diagnostics/iter800_output/iter800_duogrid_uc_vc_divergence.txt`).  C36 W2 IC.  Under LEGACY and DUOGRID separately: compute `u_cc, v_cc = fv3_d2cc(u_d, v_d)`; `u_c, v_c = fv3_cc2c(u_cc, v_cc)`; `div = cgrid_divergence(u_c, v_c)`; `mass_div = cgrid_mass_flux_divergence(h, u_c, v_c)`.

**Result.**

| quantity                     | LEGACY peak             | DUOGRID peak            | ratio     | peak location (DUOGRID)     |
|------------------------------|-------------------------|-------------------------|-----------|-----------------------------|
| u_c                          | 38.60 m/s (GC=34°)     | 38.60 m/s (GC=34°)      | 1.00×     | face 0 (0, 17), mid-face    |
| v_c                          | 26.68 m/s (GC=35°)     | 26.68 m/s (GC=35°)      | 1.00×     | face 4 (0, 18), mid-face    |
| ∇·u_c (bare divergence)      | 3.30e−08 (GC=5°)       | 3.60e−08 (GC=5°)        | 1.09×     | face 0 (2, 0)               |
| **mass_div (∇·(h·u))**       | **1.33e−04 (GC=34°)**  | **1.57e−01 (GC=1.19°)** | **1172×** | **face 2 (0, 0), GC=1.19°** |

**Observation — numerical reportage only.**  u_c and v_c from `fv3_cc2c` are BYTE-IDENTICAL between LEGACY and DUOGRID at t=0 on the smooth W2 IC.  Their bare divergence is also identical (both ~3e-8, truncation-level).  But the PPM mass flux divergence `cgrid_mass_flux_divergence` gives:
- LEGACY: peak 1.33e-4 at mid-face (GC=34°, face 4 (18, 0)).
- DUOGRID: peak 1.57e-1 at CUBE VERTEX (GC=1.19°, face 2 (0, 0)) — exactly adjacent to cube-vertex position.

**Localisation.**  The blowup is NOT in u_c, v_c, or bare divergence.  It IS in `cgrid_mass_flux_divergence`'s PPM stencil applied to h with halo=2 via `_pad_halo_auto_h2(h, cdgrid)` → `pad_halo(h, halo=2, duogrid=dg)` → `cube_rmp_vectorized + fill_corner_region`.  Under DUOGRID, the h-halo at cube-vertex cells produces a PPM reconstruction that gives a large spurious face value, which the flux divergence picks up.

**What iter-800 DOES show.**
- The DUOGRID t=0 dh/dt blowup is ISOLATED to the scalar h-halo path with halo=2 (PPM stencil).  Vector halo (for u_c, v_c) is fine.
- The blowup peak is at GC=1.19° from a cube vertex — confirms it's a halo-corner-specific issue.
- Identical u_c, v_c between paths means `pad_halo_vector(duogrid=dg)` for vector halo in `fv3_cc2c` is not the culprit.

**What iter-800 does NOT establish.**
- WHICH specific routine in the duogrid h-halo chain produces the spurious value: `cube_rmp_vectorized` Lagrange remap, `fill_corner_region` Lagrange corner fill at halo=2, or the underlying `_pad_halo_local_h2` dispatch.
- Whether the DUOGRID halo=2 scalar path works correctly for non-smooth h (the smooth W2 h already fails; non-smooth is worse).

**Iter-801+ candidates.**
- Inspect h_pad from `pad_halo(h, halo=2, duogrid=dg)` at the 4 cube-vertex halo cells on each face.  Report values vs LEGACY.  A large mismatch confirms the halo is broken.
- Toggle `cube_rmp_vectorized` and `fill_corner_region` individually (e.g., skip one and keep the other) to isolate which produces the spurious cube-vertex h halo values.
- Cross-check the Fortran equivalent: at halo=2, what cube-corner fill does FV3 apply to a scalar A-grid field?
- Port Fortran FB transport chain (blocked on ng=3 and FB-chain stability).

**Deliverable.**  `scripts/diag_iter800_duogrid_uc_vc_divergence.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  64th iter in iter-752-800 chain.  Localises the DUOGRID t=0 blowup to `cgrid_mass_flux_divergence`'s halo=2 scalar h-halo path at cube vertices.  Vector halo (u_cc/v_cc → u_c/v_c) is fine.  iter-801+ should inspect the h_pad cube-corner values directly and compare LEGACY vs DUOGRID to identify the specific broken routine.
