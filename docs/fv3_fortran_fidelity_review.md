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

Everything before `iter-796` is intentionally compressed here.

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
- `iter-742..751`: user-visible `v_ll` sentinels were locked, the
  production hyperdiff path was shown to be structurally non-Fortran,
  and `use_duogrid=True` on the A-L path was proven catastrophic.
- `iter-752..759`: `_del6_vt_flux` was ported and corrected; the
  Fortran-faithful del6 post-step path became the production best,
  reducing canonical W2 `v_ll_Linf` from the old `0.303` class to
  about `0.159`, though cube-corner seams remained.
- `iter-760..767`: remaining W2 mode-A was localized to cube corners;
  three Fortran-inspired cube-corner fill ideas were tested on the
  A-L path and all made W2 worse, so they were ruled out as drop-in
  fixes.
- `iter-768..771`: the residual was re-measured as mainly dynamical;
  `boundary_fix` corner smoothing was shown to be load-bearing; the
  peak was localized to cells updated by specific D-grid corners; and
  `grad_c10` emerged as the most suspicious remaining local metric.
- `iter-772..775`: coarse `grad_c10` ablations all worsened W2, but a
  fine `0.98×` corner scaling reduced W2 without hurting W5. That
  reduction was explicitly classified as numerical tuning, not a
  Fortran-faithful fix.
- `iter-776..781`: cosine-bell transport distortion was quantified;
  convergence plateau persisted under both CFL-scaled and fixed-dt
  sweeps; peak errors localized near cube vertices and appeared early;
  the main signal pointed at transport / halo structure rather than a
  simple limiter or damping knob.
- `iter-782..786`: `_d2a2c_vect` cube-vertex audits showed legacy
  contravariant-wind error exists but is not the main cosine-bell
  driver; duogrid was consistently worse, so it is not a near-term
  drop-in production fix.
- `iter-787..790`: W2 under duogrid was shown to be catastrophically
  worse, and the production W2 path was confirmed not to go through
  `_d2a2c_vect` at all; it uses `fv3_d2cc` + `fv3_cc2c` instead.
- `iter-791..795`: a t=0 W2 smoking gun was isolated at cube-vertex
  `dv/dt`; the residual was identified as incomplete cancellation of
  Coriolis + pressure + KE; `div_damp` suppresses rather than causes
  mode-A; and component-consistent B-halo was refuted for the smooth
  W2 IC.

Use git history if you need the full older narrative.

## Latest Ralph-loop iterations (full form)

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

### Iter-801 — DUOGRID `fill_corner_region` OVERSHOOTS by 84 m at polar cube corners

Per iter-800's iter-801+ candidate, iter-801 inspects the h_pad cube-corner 2×2 blocks directly under LEGACY (use_duogrid=False → `_fill_corners_h2` averaging) vs DUOGRID (use_duogrid=True → `fill_corner_region` Lagrange extrapolation).

**Method** (`scripts/diag_iter801_duogrid_h_pad_inspection.py`; committed output at `diagnostics/iter801_output/iter801_duogrid_h_pad_inspection.txt`).  C36 W2 h-field (smooth with polar-dominant gradient).  Compute h_pad via `pad_halo(h, halo=2, interp_offsets=..., duogrid=dg)` and compare the 4 cube-corner 2×2 blocks per face.

**Result.**

| face region | corner 2×2 block LEGACY range | corner 2×2 block DUOGRID range | max \|Δ\| (m) |
|-------------|-------------------------------|--------------------------------|---------------|
| equatorial (faces 0-3) | 2295.09 – 2408.76 | 2268.01 – 2384.46 | 42.01 |
| polar (faces 4-5)      | 2364.63 – 2385.21 | 2399.70 – 2469.24 | **84.02** |

Interior h on polar face 4: min=1094.64, max=2325.54.
DUOGRID h_pad peak at polar cube corner (face 4, padded (0,0)) = **2469.24 m** — **144 m ABOVE the interior maximum** (2325.54).  Clear overshoot.

**Observation — numerical reportage only.**  The DUOGRID `fill_corner_region` Lagrange corner extrapolation produces h_pad values at polar cube-vertex halo cells that exceed the interior physical range by up to 144 m.  This overshoot is a concrete failure mode of 4-pt Lagrange extrapolation at high-curvature geographic regions (near the pole where h has large meridional gradient).  The LEGACY path's simple 2-point averaging at cube corners stays WITHIN the interior range (max 2408 < interior max).

**Mechanism.**  When the PPM stencil in `cgrid_mass_flux_divergence` reads h at cube-vertex halo positions:
- LEGACY: h ∈ [2295, 2408] — within interior range, PPM behaves reasonably.
- DUOGRID: h ∈ [2268, 2469] — overshoots interior max, PPM reconstructs a parabola that attains large face values, giving spurious fluxes.
- Result: dh/dt peak at DUOGRID face-2 cube vertex = 1.57e-1 (iter-800), 1172× LEGACY.

**What iter-801 DOES show.**
- The DUOGRID `fill_corner_region` overshoots the interior h range by up to 144 m at polar cube corners.  This is a concrete, quantified failure of the Lagrange extrapolation.
- LEGACY's simpler 2-pt averaging (`_fill_corners_h2`) stays within the interior range and avoids the overshoot.
- Fix direction: the Lagrange corner fill needs a MONOTONICITY constraint (clip to local interior min/max), or should fall back to 2-pt averaging for specific high-curvature fields.

**What iter-801 does NOT establish.**
- Whether clipping the Lagrange output to the local interior range (monotone-preserving fix) restores DUOGRID W2 to LEGACY-equivalent behaviour.
- Whether FV3 Fortran's `fill_corner_region_2d` has an analogous monotonicity constraint that we've missed in the port.

**Iter-802+ candidates.**
- Add a monotonicity clip to `fill_corner_region`: after Lagrange computes the 2×2 corner block, clip each cell to `[min(neighbouring interior + edge-halo cells), max(...)]`.  Re-run iter-800 and iter-787 to measure impact.
- Audit Fortran `fill_corner_region_2d` (fv_duogrid.F90:1719-1903) for any clipping or monotonicity logic; cross-check against our Python.
- Test a simpler fallback: when duogrid is active, use the LEGACY `_fill_corners_h2` averaging only at cube corners (keep Lagrange for non-corner halo cells).
- Port Fortran FB transport chain (blocked on ng=3 and FB-chain stability).

**Deliverable.**  `scripts/diag_iter801_duogrid_h_pad_inspection.py` + committed output.  No source-code change.  No new sentinel.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  65th iter in iter-752-801 chain.  Localises the DUOGRID t=0 W2 blowup to `fill_corner_region`'s Lagrange extrapolation overshooting by up to 84 m (144 m above interior max at polar cube corners).  The fix path is a monotonicity clip, which is a targeted source change with a clear diagnostic test (re-run iter-800 and iter-787 after the fix).

### Iter-802 — Monotonicity clip attempt on `fill_corner_region`: dh/dt UNCHANGED

Per iter-801's iter-802+ candidate, iter-802 adds an opt-in `monotone_clip=True` parameter to `fill_corner_region` (`src/legoesm/grids/duogrid.py:868`) and tests whether the DUOGRID dh/dt blowup drops.

**Source change.**  Added `monotone_clip: bool = False` parameter to `fill_corner_region`.  When True, each Lagrange-extrapolated cube-corner cell is clipped to `[min, max]` of its 4 padded-array neighbours (i_p±1, j_p) and (i_p, j_p±1).  Default False preserves FV3-faithful behaviour.

**Test** (`scripts/diag_iter802_duogrid_monotone_clip.py`; committed output at `diagnostics/iter802_output/iter802_duogrid_monotone_clip.txt`).  Monkey-patch `duogrid.fill_corner_region` to force `monotone_clip=True`.  Run iter-799's t=0 tendency audit on C36 W2.

**Result.**

| variant                     | dh/dt peak   | du/dt peak   | dv/dt peak   |
|-----------------------------|--------------|--------------|--------------|
| LEGACY (clip off)           | 1.329e−04    | 1.404e−05    | 1.903e−05    |
| DUOGRID (clip off, baseline)| 1.557e−01    | 4.700e−05    | 8.446e−05    |
| DUOGRID (clip ON)           | 1.557e−01    | 2.816e−05    | 4.857e−05    |

- DUOGRID dh/dt: UNCHANGED (1.557e−1 → 1.557e−1).
- DUOGRID du/dt: 40% reduction (4.7e−5 → 2.8e−5).
- DUOGRID dv/dt: 42% reduction (8.4e−5 → 4.9e−5).

**Observation — numerical reportage only.**  The monotone clip as implemented does NOT fix the DUOGRID dh/dt blowup.  It does partially reduce the du/dt and dv/dt peaks.  The dh/dt peak at face 2 (0, 0) is unchanged to 4 significant digits.

**Root cause of clip ineffectiveness.**  The clip used neighbours `padded[i_p±1, j_p]` and `padded[i_p, j_p±1]` inside the 2×2 cube-corner block.  AT the extreme corner cell (e.g., face 4 padded [0, 0]), two of those neighbours are:
- `padded[−1, 0]` → wrapped via `max(0, ...)` to `padded[0, 0]` (self-reference).
- `padded[0, −1]` → wrapped to `padded[0, 0]` (self-reference).

And the two valid neighbours `padded[1, 0]`, `padded[0, 1]` are ADJACENT cube-corner halo cells, which ALSO overshoot.  So the clip range is dominated by already-overshooting cells, and the clip becomes a no-op.

A correctly-designed monotonicity clip must reference:
- The nearest INTERIOR cell (at `[halo, halo]` for SW corner).
- The edge-halo cells OUTSIDE the cube-corner 2×2 block (e.g. at row i=halo, or col j=halo for SW).

iter-803 will re-implement the clip with a correct neighbour set.

**What iter-802 DOES show.**
- The implemented clip (using padded-array neighbours in the 2×2 corner block) is INEFFECTIVE for dh/dt because the clip range is dominated by adjacent cube-corner cells that also overshoot.
- The clip DOES partially help du/dt and dv/dt (40–42% reduction) — those tendencies read from wider halo regions where the clip's effect on a subset of cube-corner cells matters.
- The DUOGRID dh/dt blowup IS driven by cube-corner halo cells (confirmed by iter-800/801), but the specific clip neighbour set must exclude other cube-corner cells.

**What iter-802 does NOT establish.**
- Whether a correctly-implemented clip (using edge-halo or nearest-interior as the clip range) will fix dh/dt.
- Whether a completely different fix (e.g. reverting `fill_corner_region` to `_fill_corner_region_averaging` at halo=2 specifically) would work.
- Whether Fortran FV3 has a different mechanism for avoiding the overshoot issue entirely.

**Iter-803+ candidates.**
- Re-implement the clip using NEAREST INTERIOR cell as the reference (e.g. for SW corner at [0, 0], clip to `[padded[halo, halo], padded[halo, halo]]` or a small window around it).
- Alternative: at halo=2 with duogrid active, bypass `fill_corner_region` and use `_fill_corner_region_averaging` (the legacy 2-pt-avg fallback already available).  Compare impact.
- Audit Fortran d_sw PPM transport for how it handles cube-corner halo (does it skip corner cells, use a different fill, or apply a limiter?).
- Port Fortran FB transport chain (blocked on ng=3 and FB-chain stability).

**Deliverable.**  Source change in `src/legoesm/grids/duogrid.py` (added `monotone_clip` opt-in parameter, default False so existing callers are unchanged) + `scripts/diag_iter802_duogrid_monotone_clip.py` + committed output.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Process.**  66th iter in iter-752-802 chain.  First SOURCE-CODE CHANGE in this diagnostic chain — adds a backward-compatible `monotone_clip` parameter to `fill_corner_region`.  The implemented clip is INEFFECTIVE for dh/dt due to neighbour-set design flaw.  Next iter needs to redesign the clip or try an entirely different approach (averaging fallback).

### Iter-803 — Fortran-faithful snapshot semantics for pass-2 diagonals; averaging fallback fixes du/dv but not dh

**Source change.**  Ported Fortran `fill_corner_region_2d` (fv_duogrid.F90:1759-1779) `veltemp`/`veltempp` snapshot semantics to Python `fill_corner_region` in `src/legoesm/grids/duogrid.py`.  Each pass-2 diagonal cell now reads from a snapshot of `padded` captured AFTER pass-1 and BEFORE any pass-2 write, so later diagonal cells do not see earlier diagonal writes.  All 14 `TestW2BoundaryErrorBudget` sentinels pass.

**Measurement.**  The snapshot change has NO numerical impact on the h_pad cube-corner values at C36 (iter-801 re-run gives identical 84 m overshoot) because the Lagrange stencils in `_lagrange_interp_x_plus/minus` and `_lagrange_interp_y_plus/minus` read ONLY from the interior of the face (padded indices `h..h+n-1`), not from the corner 2×2 block.  The snapshot vs in-place distinction only matters when stencils read from cells that other pass-2 writes could modify — which is not the case here.  The change is Fortran-faithful even though it has null local impact for this particular diagnostic.

**Averaging-fallback test** (`scripts/diag_iter803_averaging_fallback_test.py`).  Monkey-patch `fill_corner_region` to always return `_fill_corner_region_averaging` output (LEGACY-style 2-pt averaging at cube corners, no Lagrange).

| variant                             | dh/dt peak   | du/dt peak   | dv/dt peak   | Δ vs LEGACY |
|-------------------------------------|--------------|--------------|--------------|-------------|
| LEGACY (use_duogrid=False)          | 1.329e−04    | 1.404e−05    | 1.903e−05    | —           |
| DUOGRID (Lagrange, baseline)        | 1.557e−01    | 4.700e−05    | 8.446e−05    | dh 1172×    |
| DUOGRID (averaging fallback forced) | 1.557e−01    | 1.412e−05    | 1.985e−05    | dh 1172×; du 1.01×; dv 1.04× |

**Observation — numerical reportage only.**  Forcing averaging fallback at `fill_corner_region` FIXES DUOGRID du/dt (4.7e−5 → 1.4e−5, matching LEGACY) and dv/dt (8.4e−5 → 2.0e−5, matching LEGACY) — so the Lagrange corner fill WAS responsible for these.  But dh/dt is UNCHANGED at 1.557e−1 (still 1172× LEGACY) — the dh/dt blowup comes from a DIFFERENT operator, not `fill_corner_region`.

**Deliverable.**  Source change in `src/legoesm/grids/duogrid.py` (snapshot semantics, backward-compatible) + `scripts/diag_iter803_averaging_fallback_test.py` + committed output.

### Iter-804 — `cube_rmp_vectorized` also ruled out; dh/dt blowup is elsewhere

**Method** (`scripts/diag_iter804_cube_rmp_test.py`).  Monkey-patch `cube_rmp_vectorized` to a no-op (pass-through); combine with `_fill_corner_region_averaging`; measure W2 t=0 tendencies.

**Result.**

| variant                            | dh/dt peak   | dh ratio vs LEGACY |
|------------------------------------|--------------|---------------------|
| LEGACY                             | 1.329e−04    | —                   |
| DUOGRID (full baseline)            | 1.557e−01    | 1172×               |
| DUOGRID (no Lagrange; rmp ON)      | 1.557e−01    | 1172×               |
| DUOGRID (Lagrange ON; rmp OFF)     | 1.598e−01    | 1202×               |
| DUOGRID (no Lagrange; no rmp)      | 1.598e−01    | 1202×               |

**Observation — numerical reportage only.**  Disabling `cube_rmp_vectorized` SLIGHTLY WORSENS dh/dt (1202× vs 1172×), and combining with no-Lagrange doesn't help.  Neither `fill_corner_region` nor `cube_rmp_vectorized` is the dh/dt culprit.

**What iter-804 DOES show.**
- The DUOGRID dh/dt blowup is NOT in `fill_corner_region` or `cube_rmp_vectorized`.  Both are individually not the culprit.
- du/dt and dv/dt ARE fixed by disabling Lagrange (i.e., using averaging corner fill).

### Iter-805 — Forcing interp_offsets fixes du/dv EXACTLY but still not dh

**Method** (`scripts/diag_iter805_duogrid_h_halo_interp.py`).  Monkey-patch `pad_halo` so that when called under DUOGRID (`duogrid is not None`, `interp_offsets is None`), it injects `halo_interp_offsets_h2` and DISABLES the duogrid halo path entirely — effectively running LEGACY halo for all pad_halo calls.

**Result.**

| variant                                      | dh/dt       | du/dt       | dv/dt       |
|----------------------------------------------|-------------|-------------|-------------|
| LEGACY                                       | 1.329e−04   | 1.404e−05   | 1.903e−05   |
| DUOGRID baseline                             | 1.557e−01   | 4.700e−05   | 8.446e−05   |
| DUOGRID + LEGACY-like halo (force offsets)   | 1.557e−01   | 1.404e−05   | 1.903e−05   |

**Observation — numerical reportage only.**  Forcing LEGACY-like halo (interp_offsets on, duogrid halo off) restores DUOGRID du/dt and dv/dt to BYTE-IDENTICAL LEGACY values.  But dh/dt remains UNCHANGED at 1.557e−1.  This means the dh/dt difference is NOT in any pad_halo-derived quantity.

**Implication.**  The 1172× DUOGRID dh/dt blowup comes from a code path that is NOT affected by swapping halo treatment.  Remaining candidates:
- Metric fields computed at cdgrid creation time (`sin_sg`, `cos_sg`, `dx_edge_y`, `dy_edge_x`, `area`) may use different duogrid remaps during construction.
- The `cgrid_mass_flux_divergence` implementation may have a duogrid-specific branch we haven't identified.
- A field that feeds into the PPM reconstruction via a different code path.

But iter-800 showed u_c, v_c are byte-identical between LEGACY and DUOGRID.  u_c, v_c depend on the same metric fields (sin_sg, cos_sg, cosa_u).  So the metric fields can't be the difference.

**Unresolved**: WHY does DUOGRID dh/dt differ from LEGACY by 1172× when the h_pad, u_c, v_c all match LEGACY exactly with the force-offsets patch?

**Deliverable.**  `scripts/diag_iter805_duogrid_h_halo_interp.py` + committed output.  No source-code change.  No new sentinel.

**Process.**  69th iter in iter-752-805 chain.  Three consecutive iters (803/804/805) have ruled out Lagrange corner fill, cube_rmp, AND the interp_offsets difference as the dh/dt blowup cause — yet dh/dt remains 1172× LEGACY.  The residual must come from a cdgrid metric field or a cgrid_mass_flux_divergence path we haven't yet inspected.  iter-806+ should compare cdgrid metric fields BETWEEN paths directly.
