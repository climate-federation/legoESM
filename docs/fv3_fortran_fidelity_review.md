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

Everything before `iter-764` is intentionally compressed here.

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

Use git history if you need the full older narrative.

## Latest Ralph-loop iterations (full form)

### Iter-764 — Fortran `fill_4corners` (halo=1-reduced subset only)

Per iter-763b's Codex-flagged shortcoming, iter-764 evaluates a SUBSET of Fortran's `fill_4corners` formula (`sw_core.F90:3856-3915`) that applies to our halo=1 infrastructure.

**Caveat (iter-764b, important).**  Fortran's fill_4corners writes TWO halo cells per corner per direction:
```
dir=1 SW:  q(-1, 0) = q(0, 2)   <- OUTER fill, needs halo>=2 to exist
           q( 0, 0) = q(0, 1)   <- INNER fill
```
Our Python halo=1 padded array has ONLY ONE corner halo cell per face (padded[0, 0]); there is no "outer" cell analogous to Fortran's q(-1, 0). Iter-764's evaluation covers ONLY the inner fill `q(0, 0) = q(0, 1)` mapped to `padded[0, 0] = padded[0, 1]`, not the outer fill. A full Fortran-faithful port would require running at halo>=2 and implementing both fills.

Thus iter-764 evaluates a REDUCED-SCOPE version of the Fortran formula, NOT the full two-cell directional fill. The measurement below is the INNER-fill discrepancy only.

For the halo=1 inner-fill reduction:
- dir=1 (x-sweep): `padded[0, 0] = padded[0, 1]`  (y-inward halo value)
- dir=2 (y-sweep): `padded[0, 0] = padded[1, 0]`  (x-inward halo value)

These DIFFER from each other — Fortran's directional asymmetry is explicit. Python's `_fill_corners_h1` uses `0.5*(padded[0, 1] + padded[1, 0])` — the average of the two dir-specific inner values.

**Measurement on W2-exact KE field** (`scripts/diag_iter764_fortran_fill_4corners.py`):

| Face    | 2-pt-avg | dir=1 (y-inner) | dir=2 (x-inner) | diff (dir1 − py) | diff (dir2 − py) |
|---------|----------|------------------|------------------|------------------|------------------|
| face 0-3| 496.62   | 518.28           | 474.95           | **+21.66**        | **−21.66**        |
| face 4-5| 497.56   | 497.56           | 497.56           | 0                | 0                |

**Key findings:**
- **Fortran dir=1 differs from Python by 21.66 m²/s²** (max, equatorial faces); dir=2 differs by −21.66. dir=1 and dir=2 differ from EACH OTHER by 43.33 m²/s² — Fortran's explicit directional asymmetry.
- **2-pt-avg = 0.5*(dir1 + dir2) verified to machine precision** (symmetry check 5.7e−14) — our averaging IS the midpoint of Fortran's two directional values.
- **Polar faces 4/5 have zero discrepancy** on this smooth IC — an accident where `padded[0, 1]` and `padded[1, 0]` happen to share the same value at polar-face cube corners.
- **Relative discrepancy**: 2.91 % of B scale (745 m²/s²), matching the iter-763b proxy estimate.

**Implication (scope-limited to halo=1 inner fill).**  A Fortran-faithful A-L gradient AT HALO=1 would use the dir=1 inner fill for the x-component (`dB_raw_x`) and dir=2 inner fill for the y-component (`dB_raw_y`), with cube-corner halo values 22 m²/s² different from Python's current averaged value. The SIGN of the error is opposite between dir=1 and dir=2, so the x-component and y-component of the gradient at cube-corner cells both shift in a direction-dependent manner. A halo>=2 port would additionally implement the outer fill which iter-764 does NOT cover.

**Iter-765+ concrete port.**  Modify `_arakawa_lamb_gradient` to accept two `B_pad` variants (one with dir=1 inner fill, one with dir=2 inner fill) — or equivalently, construct them internally by selectively overwriting the 4 cube-corner cells per face after pad_halo. Apply dir=1 values to the x-stencil, dir=2 values to the y-stencil. This is the halo=1-scoped fix; a future halo=2+ port can add the outer fill. Run W2 matrix + regression tests to measure mode A reduction.

**Iter-764 deliverable.**  `scripts/diag_iter764_fortran_fill_4corners.py` checked in. Evaluates the halo=1 INNER subset of Fortran's `fill_4corners` formula (the `q(0,0) = q(0,1)` fill only; Fortran's companion `q(-1,0) = q(0,2)` outer fill is NOT evaluated because our halo=1 layout has no corresponding outer cell). Measures the inner-fill discrepancy vs Python's 2-pt-avg. Corroborates iter-763b's sensitivity evidence with the best Fortran-formula comparison achievable at halo=1. No source-code change to production. Matrix + ocean baselines unchanged.

**Process.**  24th-25th iter in iter-752-764b chain. Two Codex stop-time catches on iter-764 — first (iter-763b) for using a proxy instead of computing the Fortran formula, second (iter-764b) for overclaiming a halo=1 subset as "the Fortran formula." Lesson: when porting a Fortran subroutine that uses multi-cell halo fills, be explicit about which cells your shallower-halo environment can cover.

### Iter-765 — dir-aware Fortran corner fill HYPOTHESIS FALSIFIED

Per iter-764d's plan, iter-765 implements the halo=1 Fortran inner dir-aware corner fill in `_arakawa_lamb_gradient` via a new `fortran_dir_aware_corners` kwarg (default False). When True, the x-gradient stencil uses dir=1 fill (`padded[0, 0] <- padded[0, 1]`) at the 4 cube-corner halo cells per face, and the y-gradient stencil uses dir=2 fill (`padded[0, 0] <- padded[1, 0]`).

**Direct measurement on W2 C36 1d, iter-761 matrix config:**

| Config                         | h_L2     | v_ll_Linf |
|--------------------------------|----------|-----------|
| default (2-pt-avg corners)     | 2.07e−4  | 1.59e−01  |
| dir-aware fills (iter-765 test)| **7.75e−4**| **1.90e+00 (BLOWUP)** |

**Dir-aware fills make W2 12× WORSE, not better.** The h_L2 also degrades 3.7×. Hypothesis iter-763/764's sensitivity evidence -> Fortran-faithful -> mode-A reduction is **FALSIFIED**.

**Why the hypothesis failed.**  Fortran's fill_4corners is used **before PPM sweep operators** — `dir=1` is applied before the X-SWEEP so the PPM 4-point stencil (q(-2), q(-1), q(0), q(1)) has a consistent cube-corner value CONSISTENT with the sweep direction. Fortran's `dir=2` is applied before Y-SWEEP. These fills are SWEEP-SPECIFIC — they encode "what value should the cube corner take IF I'm about to slide a 1D PPM stencil across it in direction X".

The A-L gradient is **not a sweep** — it's a simultaneous 4-point 2D stencil at each D-grid corner. Applying a sweep-specific fill to a non-sweep stencil is algorithmically inconsistent: the stencil expects a "direction-neutral" value (which is precisely what the 2-pt-avg provides as the midpoint of the two sweep-specific values), not a direction-biased one.

**Iter-765 source-code status.**  The `fortran_dir_aware_corners` kwarg is RETAINED in `_arakawa_lamb_gradient` and `fv3_sw_tendencies` (default False) as an opt-in diagnostic path — future iters investigating cube-corner halo behaviour can toggle it. The matrix default and production path are UNCHANGED (still 2-pt-avg). Matrix + ocean + 173 regression tests all PASS.

**Broader lesson.**  Sensitivity evidence (iter-763/764: fill choice matters by ~3% of B scale) does NOT imply "Fortran's fill recipe is drop-in compatible with our A-L operator." Fortran's fill was designed for an operator-split PPM (dir=1 then dir=2, NEVER simultaneously), while our A-L is simultaneous in both directions. Adopting Fortran's fill in a non-Fortran operator context introduces algorithmic mismatch rather than fixing the artifact.

**Iter-766+ next directions.**
- Investigate whether the W2 cube-corner artifact has a different structural driver than halo fill. Candidates: the non-orthogonal metric treatment at the 3-face vertex, halo-rotated vector components via `pad_halo_vector`, or A-L gradient stencil itself at cube corners.
- Alternative: design a DIRECTION-NEUTRAL Fortran-inspired fill (e.g., the mean of dir=1 and dir=2 values — which is our existing 2-pt-avg). iter-765 confirms 2-pt-avg was actually well-chosen for the A-L operator.
- Consider that mode A at 0.159 m/s may be a FUNDAMENTAL LIMIT of the A-L production path, and further reduction requires adopting the FB chain (currently blocked on h=3 halo infrastructure).

**Iter-765 deliverable.**  Source: `fortran_dir_aware_corners` kwarg added to `_arakawa_lamb_gradient` and `fv3_sw_tendencies`, default False. Matrix + ocean baselines unchanged. All regression tests pass.

**Process.**  28th iter in iter-752-765 chain. First iter with an actual code path WIRED (even if opt-in), and first clean negative result (no Codex stop-time catch). Falsification of the hypothesis is a useful outcome that constrains iter-766+ directions.

### Iter-765b — thread dir-aware flag through BOTH A-L calls (Codex stop-time)

Codex stop-time review on iter-765 flagged: **"the new flag is only partially threaded, so the claimed falsification is based on a mixed code path."** Correct. My iter-765 threading only reached the Bernoulli A-L gradient call (line 1451) but not the div_damp A-L call (line 1517). The claimed falsification compared:
- default (both calls 2-pt-avg)
- "dir-aware" (Bernoulli dir-aware, div_damp still 2-pt-avg)

This is a MIXED code path — not a clean iter-765 falsification.

**Fix.**  Threaded `fortran_dir_aware_corners` through both A-L gradient calls inside `fv3_sw_tendencies` (Bernoulli at line 1451 AND div_damp at line 1517). Now when the flag is True, BOTH calls use dir-aware fills consistently.

**Re-measured with fully consistent threading:**

| Config                          | h_L2     | v_ll_Linf |
|---------------------------------|----------|-----------|
| default (both 2-pt-avg)         | 2.07e−4  | 1.59e−01  |
| dir-aware BOTH A-L calls        | 7.69e−4  | **1.88e+00** (BLOWUP) |

Essentially identical to iter-765's mixed-path result (1.88 vs 1.90 m/s). The conclusion holds: Fortran dir-aware fill in A-L gradient blows up by 12×. Adding the div_damp A-L call to the dir-aware treatment does NOT materially rescue (or worsen) the failure.

**Falsification stands.**  Hypothesis was: Fortran dir=1/dir=2 inner fill in A-L gradient reduces mode A. Evidence with FULLY consistent flag threading: FALSIFIED at 12× amplitude, 3.7× h_L2. Consistency check complete.

**Iter-765b deliverable.**  One-line edit to thread kwarg through the second A-L call in `fv3_sw_tendencies`. Matrix + ocean + 173 regression tests all PASS (default path unchanged).

**Process.**  29th iter in iter-752-765b chain. Codex-flagged partial threading fixed so the falsification claim rests on a clean single-path comparison. Lesson: when adding a config flag that controls multiple code paths, verify it reaches ALL of them before drawing conclusions.

### Iter-766 — Fortran `a2b_ord4` 3-pt corner average HYPOTHESIS FALSIFIED

Per iter-765b's iter-766+ directions ("the A-L gradient stencil itself at cube corners" candidate) and to continue the Fortran-faithful audit initiated by iter-762-765, iter-766 investigates a DIFFERENT Fortran scalar corner fill: `a2b_ord4` (model/a2b_edge.F90:385-388).

**Fortran reference.**  `a2b_ord4` interpolates A-grid scalars to B-grid vertices for the pressure-gradient machinery. The interior bulk stencil is a 4-point average of the 4 surrounding A-cells (a2b_edge.F90:380):

```
qout(i,j) = 0.25*(qin(i-1,j-1) + qin(i,j-1) + qin(i-1,j) + qin(i,j))
```

At the 4 CUBE VERTICES (lines 385-388), Fortran OVERRIDES this with a 3-point formula that EXCLUDES the cube-corner A-halo cell:

```
if ( sw_corner ) qout(1,1) = r3*(qin(1,1) + qin(1,0) + qin(0,1))
```

where `r3 = 1.0/3.0`. The 3 included cells are the first interior A-cell adjacent to the cube vertex, and the two halo strips adjacent to the cube vertex (NOT the diagonal halo cell qin(0,0)). This 3-point formula is DIRECTION-NEUTRAL — it does not have a sweep-specific variant (unlike the PPM-related `fill_corners` family investigated iter-763/764/765).

**Iter-766 candidate.**  Use Fortran's a2b_ord4 3-pt formula as the cube-corner halo value in our A-L gradient: replace the 2-pt-avg `padded[0,0] = 0.5*(padded[0,1] + padded[1,0])` with `padded[0,0] = (1/3)*(padded[0,1] + padded[1,0] + padded[1,1])` (and symmetric formulas for SE, NE, NW). Because this fill is direction-neutral, it should in principle be compatible with our direction-neutral A-L operator (unlike the directional fills iter-765 falsified).

**Magnitude measurement** (`scripts/diag_iter766_fortran_a2b_corner_avg.py`): at cube corners on the W2-exact KE field, 3-pt-avg differs from 2-pt-avg by up to 5.1 m²/s² (0.68% of B scale), RMS 3.6 m²/s². Equatorial-face discrepancy +2.5 m²/s², polar-face −5.1 m²/s². Roughly 4× smaller than iter-764's directional-fill discrepancy (22 m²/s² max).

**Direct measurement on W2 C36 1d, iter-761 matrix config:**

| Config                          | h_L2     | v_ll_Linf |
|---------------------------------|----------|-----------|
| default (2-pt-avg corners)      | 2.07e−4  | 1.59e−01  |
| Fortran a2b 3-pt-avg (iter-766) | **4.50e−4**| **3.00e−01** |

**Fortran a2b 3-pt-avg makes W2 1.89× worse, not better.** h_L2 degrades 2.17×. Hypothesis FALSIFIED.

**Why the hypothesis failed.**  Although Fortran's a2b_ord4 3-pt formula IS direction-neutral (unlike `fill_corners`/`fill_4corners`), it was designed for a DIFFERENT operator: A-grid -> B-grid SCALAR INTERPOLATION. Our A-L is a 2D GRADIENT STENCIL that reads 4 surrounding A-cells per D-grid corner. The 3-pt-avg lowers the weight of the two edge halos (from 0.5 each to 1/3 each) and adds the diagonal interior cell (weight 1/3). When this substitution is made in the A-L gradient coefficient structure, the gradient at the cube-vertex D-grid corner picks up a different mix of near-field vs far-field values than the 2-pt-avg — and this different mix evidently amplifies the pre-existing mode A rather than damping it.

Working out the stencil coefficients for `dB_raw_x = (B_se + B_ne) - (B_sw + B_nw)` at the cube-corner D-grid corner:
- With 2-pt-avg: dB_raw_x = +0.5·padded[1,0] − 1.5·padded[0,1] + 1·padded[1,1]
- With 3-pt-avg: dB_raw_x = +(2/3)·padded[1,0] − (4/3)·padded[0,1] + (2/3)·padded[1,1]

The 2-pt-avg is MORE asymmetric in its treatment of the x-halo (+0.5) vs y-halo (−1.5), while the 3-pt-avg is more balanced. Apparently our A-L metric coefficients at the cube-corner D-grid corner expect the more-asymmetric 2-pt-avg structure.

**Iter-766 source-code status.**  A `fortran_a2b_corner_avg` kwarg is RETAINED in `_arakawa_lamb_gradient`, `fv3_sw_tendencies`, and `CDGridShallowWaterConfig` (default False) as an opt-in diagnostic path. The matrix default and production path are UNCHANGED (still 2-pt-avg). Matrix + ocean + all `tests/unit/test_cdgrid_fv3_regression.py` tests PASS. (Running test count is not pinned here because the count changes with each new sentinel; see the pytest output of that file for the current pass count.)

**Broader lesson (combining iter-765 and iter-766).**  Two Fortran-inspired cube-corner fills — directional `fill_4corners` (iter-765: 12× W2 blowup) and direction-neutral `a2b_ord4` 3-pt-avg (iter-766: 1.89× W2 blowup) — have both made W2 worse than the 2-pt-avg on the canonical C36 1-day matrix config. The 2-pt-avg was not tuned for the A-L operator, but both of the Fortran recipes we have direct-measured are strictly worse than it. The evidence base is two specific Fortran fills, not every conceivable fill — so "2-pt-avg is provably optimal" is OVERREACH. What we can say: these two Fortran recipes designed for different operators (PPM sweeps and A-to-B scalar interpolation) are not drop-in replacements.

**Scope note (accurate caller inventory, function-name references).**  The `fortran_a2b_corner_avg` flag is implemented in `_arakawa_lamb_gradient` and is THREADED through these callers (file-local grep: `fortran_a2b_corner_avg=`):

- `fv3_sw_tendencies` Bernoulli A-L call (operators_cdgrid.py)
- `fv3_sw_tendencies` div-damp A-L call (operators_cdgrid.py)

It is NOT threaded through these other direct `_arakawa_lamb_gradient(...)` call sites:

- `cdgrid_momentum_tendencies` in operators_cdgrid.py — KE/p/B/div_field calls (the legacy non-FV3-faithful path; not on the canonical matrix W2 path)
- `_overlapped_arakawa_lamb_gradient` 3D fallback in operators_cdgrid.py — used by `primitive_eq_cdgrid` only
- `primitive_eq_cdgrid.py` — direct A-L calls (plus one through `_overlapped_arakawa_lamb_gradient`)
- `compressible_euler_cdgrid.py` — direct A-L calls (NOT through `_overlapped_arakawa_lamb_gradient`)
- `ocean_pe_cdgrid.py` — direct A-L calls

(Line numbers intentionally omitted because they drift whenever unrelated edits shift code. Use `grep -n "_arakawa_lamb_gradient(" src/` for the up-to-date list.)

The iter-766 W2 falsification therefore applies to the shallow-water production path exercised by the matrix (the two `fv3_sw_tendencies` call sites). Non-SW callers were intentionally left at the default 2-pt-avg because (a) mode A is not visible on those paths, (b) adding a diagnostic-only flag to their signatures would be churn without scientific motivation at this iteration. A future iter investigating cube-corner mode A on one of those dycores can thread the flag as needed.

**Iter-767+ next directions.**
- Mode A at v_ll_Linf ≈ 0.159 m/s (−48% from iter-752 legacy 0.303) appears to be a fundamental limit of the A-L + 2-pt-avg production path. Further reduction requires a STRUCTURAL change, not a corner-fill tweak.
- Candidates: (a) port Fortran's c_sw + d_sw forward-backward scheme holistically (blocked on ng=3 halo infrastructure); (b) redesign the A-L operator's metric coefficients so that a different corner-fill recipe becomes natural; (c) investigate whether `pad_halo_vector` rotation at cube vertices has a separable contribution to mode A (iter-765b candidate #2, still unexplored).
- The iter-766 `fortran_a2b_corner_avg` opt-in kwarg is useful for future ablation studies of metric-coefficient alternatives.

**Iter-766 deliverable.**  Source: `fortran_a2b_corner_avg` kwarg added to `_arakawa_lamb_gradient`, `fv3_sw_tendencies`, and `CDGridShallowWaterConfig`, default False. Runtime guard in `_arakawa_lamb_gradient` rejects `fortran_a2b_corner_avg=True` + `fortran_dir_aware_corners=True` simultaneously (the two opt-ins overwrite the same cells; iter-765's construction wins silently if both are on — the guard prevents that misuse). Diagnostic scripts `diag_iter766_fortran_a2b_corner_avg.py` (magnitude measurement) and `diag_iter766b_w2_fortran_a2b.py` (W2 impact measurement). Matrix + ocean + `tests/unit/test_cdgrid_fv3_regression.py` all PASS.

**Process.**  30th iter in iter-752-766 chain. Second Fortran-corner-fill hypothesis falsified after iter-765. Combined iter-765/766 result constrains iter-767+: corner-fill tweaks cannot break mode A below ~0.159 m/s; structural change required.

### Iter-767 — Fortran `fill_corners_agrid_r8` VECTOR cube-corner fill HYPOTHESIS FALSIFIED

Per iter-765b candidate #2 ("halo-rotated vector components via `pad_halo_vector`") which had never been checked against the Fortran oracle, iter-767 investigates the Fortran oracle for VECTOR cube-corner halo fills.

**Fortran reference** (`../atmos_cubed_sphere-symmetryclean/tools/fv_mp_mod.F90:1433-1457`). `fill_corners_agrid_r8` with `mySign=-1` (VECTOR):

```
if (sw_corner) x(0, 0) = mySign*y(0, 1)
if (sw_corner) y(0, 0) = mySign*x(1, 0)
```

And analogous formulas for SE/NE/NW with sign pattern `{SW: -, SE: +, NW: +, NE: -}`. This is a DIRECT cross-component swap with sign flip — x at the cube-corner halo cell equals ± y at the adjacent edge-halo cell of the OTHER component. Fortran uses this because the face-local grid angle is discontinuous at the 3-face cube vertex, and a direct swap sidesteps the angle altogether.

**Python current path.**  `pad_halo_vector` (`src/legoesm/grids/halo.py:1565-1711`) uses a rotate-pad-rotate chain: (1) rotate grid-aligned `(u, v)` to geographic `(u_east, v_north)` via face-local cos/sin angle, (2) pad each as SCALAR via `pad_halo` (2-pt edge-halo-only average at cube corners), (3) rotate back to grid-aligned via the PADDED face-local angle `cos/sin_angle_padded`. The `compute_padded_angle` at halo cells is face-local and discontinuous across the 3-face cube vertex.

**Iter-767 candidate.**  Overwrite the 4 cube-vertex halo cells of `(u_cc_pad, v_cc_pad)` output by `pad_halo_vector` with Fortran's VECTOR swap formula. New helper `_fortran_agrid_vector_corner_fill(u_pad, v_pad)` in `operators_cdgrid.py`; opt-in via `fortran_vector_corner_fill` kwarg on `fv3_sw_tendencies` and `CDGridShallowWaterConfig`, default False. Fill applied to BOTH `pad_halo_vector` calls inside `fv3_sw_tendencies` (step (e) wind halo and step (k) tendency projection) for consistency.

**Direct measurement on W2 C36 1d, iter-761 matrix config:**

| Config                                    | h_L2     | v_ll_Linf |
|-------------------------------------------|----------|-----------|
| default (rotate-pad-rotate, 2-pt-avg)     | 2.07e−4  | 1.59e−01  |
| Fortran agrid VECTOR fill (iter-767)      | **2.07e−3**| **2.56e+00 (16× BLOWUP)** |

**Fortran VECTOR swap makes W2 16× worse, h_L2 10× worse.** Hypothesis FALSIFIED.

**Why the hypothesis failed (interpretation, not fully oracle-verified).**  The most plausible reading is that Fortran's `fill_corners_agrid_r8` formula expects the halo exchange pipeline to supply RAW face-local u/v in the edge-halo cells, so its swap `u_corner = ±v_edge_halo` expresses the geometric alignment between neighboring face axes at the cube vertex.  Cross-corroboration: `sw_core.F90:3568-3581,3640-3654` shows `d2a2c_vect` hardcoding the same component-swap pattern on local `ua/va` before any rotation step.  The exact wiring of `mpp_update_domains` is internal to the MPP library and not directly visible in the oracle tree here — so "edge halos contain RAW grid-aligned values" is an inferred contract, consistent with the in-tree evidence but not line-by-line verified.

In our Python path, `pad_halo_vector` ALREADY rotates through geographic components before padding, so the edge-halo cells `v_pad[0, 1]` etc. do NOT contain neighbor-face raw-grid-aligned values — they contain back-rotated-to-this-face values that have already absorbed the face-angle transformation. Applying Fortran's swap on top of these back-rotated values double-applies the rotation and produces values inconsistent with the neighbor face's interior.

**Broader lesson (combining iter-765, iter-766, and iter-767).**  Three Fortran-inspired cube-corner fills have now been falsified:

- iter-765: directional `fill_4corners` (12× W2 blowup)
- iter-766: direction-neutral `a2b_ord4` 3-pt-avg (1.89× W2 blowup)
- iter-767: direct VECTOR swap `fill_corners_agrid_r8` (16× W2 blowup)

All three assume source-cell semantics that our `pad_halo_vector` / `pad_halo` pipeline has already altered. The Python rotate-pad-rotate chain is INTERNALLY CONSISTENT (its output is what our A-L operator's metric coefficients were calibrated against); applying a Fortran fill on top of the rotated output creates double-counted rotations.

**Structural implication.**  To adopt ANY Fortran cube-corner fill faithfully, we would also need to replace `pad_halo_vector` with a pipeline that DOES NOT rotate through geographic components — i.e., copy raw grid-aligned u/v directly from neighbor faces, THEN apply Fortran's swap formula at the cube corners. This is a much bigger refactor touching every caller of `pad_halo_vector`. Deferred until a structural dycore port (Fortran c_sw+d_sw FB scheme) is planned.

**Iter-767 source-code status.**  `fortran_vector_corner_fill` kwarg RETAINED in `fv3_sw_tendencies` (default False) and exposed on `CDGridShallowWaterConfig` (default False) as an opt-in diagnostic path. The matrix default and production path are UNCHANGED. Helper `_fortran_agrid_vector_corner_fill` defined in `operators_cdgrid.py` for future reuse if the rotate-pad-rotate chain is replaced.

**Iter-768+ next directions.**  Three Fortran cube-corner-fill recipes have now all made W2 worse — "corner-fill tweaks via Fortran formulas overlaid on the current pipeline are exhausted" is the defensible claim; "only structural paths remain" is a stronger inference that is consistent with the current evidence but not proven by three ablations alone.  Concrete candidates:

- Port Fortran's c_sw+d_sw forward-backward scheme holistically (blocked on ng=3 halo infrastructure — see review-doc item #2).
- Redesign the A-L operator's metric coefficients or replace the entire `pad_halo_vector` rotation chain with raw-copy + Fortran corner swap — either change is LARGE and cross-cutting.
- Continue narrow-scope ablations: try a non-Fortran cube-corner value engineered to match the A-L operator's coefficient structure (e.g. a weighted combo of the 2-pt-avg with the adjacent interior value), or investigate whether the issue is in the `boundary_fix` blending at cube-vertex rows rather than in the halo fill itself.
- Accept mode A at 0.159 m/s as the current best of the A-L + rotate-pad-rotate production path until one of the structural options lands; document as the stabilized-research result baseline.

**Iter-767 deliverable.**  Source: `_fortran_agrid_vector_corner_fill` helper + `fortran_vector_corner_fill` kwarg in `operators_cdgrid.py`, config field in `shallow_water_fv3_cdgrid.py`. Diagnostic script `scripts/diag_iter767_fortran_vector_corner_fill.py`. Sentinel `test_fortran_vector_corner_fill_is_known_worse` pins OFF baseline < 0.20, ON < 5.0, and ON/OFF ratio > 5.0 (iter-767 measured ~16×). Matrix + ocean + all regression tests PASS.

**Process.**  31st iter in iter-752-767 chain. Third Fortran-corner-fill hypothesis falsified. Combined iter-765/766/767 result strongly suggests mode A requires a structural dycore change, not a cube-corner-halo tweak.

### Iter-768 — Two-point v_ll_Linf observable measurement (t=0 vs t=1 day)

Per iter-767's finding that three Fortran cube-corner halo fills all fail, iter-768 measures `v_ll_Linf` at t=0 (from the analytic W2 IC) and at t=1 day (after 288 RK3 steps in the canonical matrix config) and reports both as observables.  Iter-768 does NOT decompose either observable into causal components.

**Scope clarification (iter-768b, Codex stop-time review).**  This is a two-point observable measurement, not a causal decomposition.  The original iter-768 framing ("IC contributes 5 %, dynamics 95 %") treated a simple ratio of two `v_ll_Linf` values as a causal split — it is not.  At t=0, `u_d err Linf` and `v_d err Linf` are both 0 (state exactly matches the analytic reference), while `v_ll_Linf = 8.0e−3 m/s`; at t=1 day, `v_ll_Linf = 1.59e−1 m/s`.  The ratio 1-day / t=0 = 19.79× is a single scalar observable; it does not by itself identify which of the dycore, the measurement pipeline, the IC construction, or couplings among them is responsible for the t=1 day value.  Causal separation requires follow-up ablations (iter-769+).

**Method** (`scripts/diag_iter768_mode_a_at_t0.py`).  Runs the canonical matrix W2 C36 config with n_steps=0 (IC only) and n_steps=288 (1 day), measures `v_ll_Linf` at each.  Uses identical IC, config, and v_north/regrid conventions to iter-766c/767 diagnostics.  Also prints `u_d err Linf` / `v_d err Linf` at t=0 to verify the D-grid state itself has zero error vs the analytic IC.

**Result.**

| Snapshot           | v_ll_Linf (m/s) |
|--------------------|-----------------|
| t=0                | 8.01e−03        |
| t=1 day            | 1.59e−01        |
| ratio (t=1d / t=0) | **19.79×**      |

At t=0 the per-field errors `u_d err Linf` and `v_d err Linf` are both exactly 0 — the stored state matches the analytic IC — while `v_ll_Linf` is 0.008 m/s.  The two numbers are reported as observables; iter-768 does not attempt a causal model of how the latter arises.

**Interpretation.**  The 1-day `v_ll_Linf` observable is ~20× the t=0 observable.  The measurement does not by itself distinguish among candidate mechanisms for the 1-day value: dycore drift at cube vertices, amplification under the dynamics of whatever is captured by the t=0 observable, interactions between the dynamics and the measurement pipeline, or combinations of those.  Separating them requires follow-up ablation studies (deferred to iter-769+) — e.g. re-run with a pipeline change only, re-run with a dycore change only, and compare 1-day `v_ll_Linf` values.

**Peak locator.**  The 10 largest |v_ll| values at t=1 day are all at (lat ≈ ±35°, lon ≈ ±41°-±43° or ±137°-±139°), i.e. within 3.33° GREAT-CIRCLE distance of the 8 cube vertices at (±arcsin(1/√3) ≈ ±35.26°, ±45°/±135°).  This confirms iter-762's localization claim even after iter-752-767 iteration chain.  The ~3° GC offset from the exact cube vertex is explained by (a) 1° regrid binning, (b) cell-centre positions being 1-2 cells inside the cube-vertex corner (~1.3°-2.5° per cell at C36).  The committed script output is in `diagnostics/iter768_output/iter768_mode_a_at_t0.txt`.

**Implication for iter-769+.**  The two-point observable measurement is consistent with multiple candidate mechanisms and does not prefer any.  Candidate investigations for iter-769+ include:

- The A-L 4-point stencil at the cube-vertex D-grid corner (reads cube-corner halo padded[0, 0] + 3 near-interior cells).  All three Fortran halo fills here failed (iter-765/766/767), so the issue may be in the STENCIL STRUCTURE itself or the METRIC coefficients, not the halo values.
- The circulation-based vorticity at the cube-vertex D-grid corner (reads the same 4 cells).  Geostrophic balance requires `zeta * v - dB/dx = 0`; an asymmetric error in zeta vs dB/dx at cube vertex would produce a net v-tendency.
- The `boundary_fix` smoothing applied to cells at cube-vertex row/column.  The current cascaded `row 0 → column 0` update applies a 4-point average at cube-corner cells that might be either over- or under-correcting.
- The measurement pipeline itself: at t=0 the stored state matches the analytic IC with zero per-field error but the t=0 `v_ll_Linf` observable is 0.008 m/s.  One candidate mechanism is that the D-grid → cell-centre-avg → rotate → regrid chain does not exactly recover `v_north = 0`; iter-769+ ablations would test whether this is a contributor.
- The structural dycore port (c_sw+d_sw FB) remains the long-term goal; iter-768 does not change that priority.

**Iter-768 deliverable.**  `scripts/diag_iter768_mode_a_at_t0.py` — reproducible two-point `v_ll_Linf` measurement (t=0 and t=1 day), with committed output at `diagnostics/iter768_output/iter768_mode_a_at_t0.txt`.  No source-code change.  No new sentinel (the existing iter-766/767 sentinels already cover the ON/OFF gaps; iter-768 is purely diagnostic).

**Process.**  32nd iter in iter-752-768 chain.  First iter in the chain that is purely DIAGNOSTIC — no new code path, no new kwarg.  Produces a concrete two-point measurement (t=0 and t=1 day, with committed reproducible output) that establishes the t=1 day observable is ~20× the t=0 observable; mechanism separation is deferred to iter-769+ ablations.  Iter-768b (Codex stop-time) retracted the original "IC 5 % / dynamics 95 %" causal framing and successive residual causal attributions — a ratio of two `v_ll_Linf` observables is not a causal decomposition without an ablation.
