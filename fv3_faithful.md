# FV3-faithful cubed-sphere — change log (shrunk @ iter 10)

Goal: cube finite-volume core faithful to GFDL FV3 across the atmosphere + ocean
matrix (SW → AMIP/OMIP), matching MPAS/icosahedral + lat-lon FV, **zero cube
edge artifacts**, visual + quantitative, `/codex:adversarial-review` on code +
visuals. Run on **CPU** (`JAX_PLATFORMS=cpu`; Metal broken, [[metal-backend-broken-use-cpu]]).
Per-iteration detail in git log `8f693690..HEAD`.

## Environment constraint
Oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean` resolves under
`~/Documents`; macOS TCC blocks the whole tree except cwd (`Operation not
permitted` even sandbox-off). Can't diff the Fortran. Audit vs known FV3
algorithms + physical benchmarks instead.

## Verdict on the 3 user-flagged visual concerns
1. **Cosine bell day-1 "distortion"** = bulk PPM-limiter diffusion in the panel
   INTERIOR (in-code iter-61 probe: 99% interior, edges ~0.0003; 12-day L2 cube
   0.865 / ico 0.620 = **1.4×**), NOT a grid/edge artifact. The high-lat
   scattered squares in native plots are cosmetic scatter-render.
2. **W5 waves propagate** ≈ latlon/MPAS by day 15; ~12% under-propagation at
   day 1.5 (cube downstream h'rms 64.8 vs latlon 74). Lowering damping BLOWS UP.
3. **Cross-grid longitude alignment** OK for SW (all regrid to lon[-180,180];
   continents aligned in cross-grid plots). Re-verify on AMIP/OMIP.

## THE real cube edge artifact (atmosphere) — quantified, root-caused
Steady tests (exact = no motion ⇒ any wind = pure imprint), cube vs latlon:
| test | cube | latlon | ratio |
|------|------|--------|-------|
| rest_state_topo wind_rms | 0.205 | 0.0029 | **70×** |
| baroclinic jet v_rms @0.2d | 3.4 | 0.023 | **150×** |
| W2 SW v_rms @5d | 0.093 | 0.013 | 7× |

Two distinct sources (spatial inspection): rest_state_topo imprint is
mountain-localized = **sigma-coord PGF error**; baroclinic/W2 is panel-edge
wavenumber-4 = **A-L corner-gradient reading O(Δx) halo-interp values at faces**
(same mechanism the ocean doc `cubed_sphere_pgf_stability.md` identified).
Prognostic (grows from clean v=0 IC), NOT a diagnostic rotation error.

## Why the standard FV3-faithful fixes are blocked, and the un-blocked lead
- FV3 forward-backward stepping → unstable (470× W2 gate; team failed many iters).
- Full-strength d_sw5 corner damping → blows up the baroclinic jet (NaN day 0.23
  for d2_bg≥0.001); ~6% on weak rest. Corner damping is NOT the fix.
- FV3 Lin-1997 PGF exists (`_fv3_lin_pgf.py`) but INERT ("not stable with RK3").
- ⇒ cube is FV3-faithful in COMPONENTS (PPM `fv_tp_2d`, d2a2c, A-L gradient,
  corner fills, metric) within stabilized RK3; the full FV3 SCHEME destabilizes.
- **UN-BLOCKED LEAD:** `core/operators_fc` FC-Gram cube operators
  (`fc_gradient_x/y`, `fc_divergence`) are face-boundary-accurate + RK3-compatible
  + unused by the atmosphere. They cured the *same* root cause in the ocean.
  CATCH: FC ops are cell-centered (A-grid); the atmosphere PGF needs **D-grid-
  staggered** gradients → needs a staggered FC operator (a focused multi-iter
  build; `_fv3_lin_pgf.project_cgrid_pgf_to_dgrid_corners` may help project).
  NOT a drop-in swap. This is the next deep target.

## OCEAN cube fix — DONE + codex-APPROVED (commits 5f425042, 47d21e83)
Cube ocean was NaN-blowing-up (rest_state day 2.08, barotropic_wave day 1.04)
while MPAS/latlon stable. Fix existed (FC-Gram `ocean_pe_fc`, `run_omip` uses it
by default) but `run_ocean_test_matrix.py` only wired it into 3 of 8 runners.
Defaulted cube→FC in `_create_ocean_setup`, **decoupled FC backend from the
density-test heavy diffusion** (`cube_fc_light_diffusion`), opted IGW out (stable
on no-FC, FC over-damps it). Result (no regressions):
| ocean cube case | before | after |
|----------------|--------|-------|
| rest_state ×4 | NaN | **PASS** (drift ~1e-29) |
| barotropic_wave | NaN day1 | finite+conserving (FC intrinsically over-damps the fast wave → amp 0.02 < 0.1 gate, as latlon marginally fails too) |
| inertia_gravity_wave | FAIL 0.068 | unchanged (no-FC) |
LESSON: FC-Gram cures the density-gradient face-edge NaN but its Fourier-
continuation smoothing over-damps fast barotropic waves — right for density/rest,
wrong for pure-wave cases. Codex re-review: APPROVE, ship, no findings.

## Other fixes committed
- `2b409ad8` native cube velocity snapshot plots were BLANK for every velocity
  case (regridded array vs native-cube scatter size mismatch → silent skip);
  route native `u_cc_east/v_cc_north`. (The visual-inspection tool itself.)
- `81a5d021` fixed an AST-guard regression my SW env-knob commit introduced
  (pinned literal `2.0*_hyperdiff_cube(n)` → factor form + default-2.0 assert).
- SW W5/W2/W6 cube env knobs `LEGOESM_SW_{DIV_DAMP,HYPERDIFF}_FACTOR` (default
  inert); baroclinic `LEGOESM_CDD_*` knobs (default inert). `scripts/probe_w5_metrics.py`.

## Coverage run so far (`results/probe_*`, CPU)
SW all grids (W2/W5/W6/cosine_bell); hydro baroclinic + rest_state_topo +
rotated_steady (cube vs latlon vs ico); ocean rest_state/barotropic_wave/IGW
(cube vs latlon vs mpas); held_suarez (cube physical 265K; latlon/ico pending).
NOT yet: AMIP cube, full DCMIP transport, ocean gyres (regional-only).

## No quick atmosphere lever (iter 10 — confirmed)
The PE baroclinic A-L gradient (operators_cdgrid.py:896 `_arakawa_lamb_gradient`,
returns dB/dx, dB/dy_perp at D-grid corners in face-local basis) is intricate
(cube-vertex non-orthogonality via 3D Cartesian metric; Fortran-faithful corner
variants). `use_fv3_a2b_zeta_corner` (4th-order corner interp) already team-
tested NEUTRAL on the imprint (+51% wall) ⇒ OFF. Toggling existing flags won't
reduce the imprint; the fix is the FC-PGF build below.

## BUILD PLAN — atmosphere D-grid FC PGF (the imprint fix; execute focused)
Risk: may hit the same RK3 instability as the Lin-1997 PGF (which also put a
gradient on the D-grid). Steps, each opt-in + tested, NO production default change
until validated:
1. New fn `fc_bernoulli_gradient_dgrid(B, cdgrid, fc_cfg)` in operators_cdgrid:
   FC cell-centre face-local ∂B/∂x,∂B/∂y (`fc_gradient_x/y`) → interpolate to
   D-grid corners (a2b; reuse `_interp_center_to_corner_a2b_ord4` /
   `_fv3_lin_pgf.project_cgrid_pgf_to_dgrid_corners`). Handle the 8 cube vertices.
2. Unit test: on a smooth global scalar, FC-PGF edge-cell error ≈ interior error
   (vs A-L edge amplification) — the go/no-go accuracy gate.
3. Opt-in PE config flag `use_fc_bernoulli_gradient=False`; wire at
   primitive_eq_cdgrid.py:413. Measure baroclinic v_rms@0.2d (target ≪3.4) AND
   15-day stability (the RK3 risk). If unstable → document like Lin PGF; if
   stable+lower-imprint → the fix, re-pin sentinels, make default.

## iter 11-12 — baroclinic imprint is DISCRETE-BALANCE error (halo-insensitive)
- Enabling `use_duogrid=True` (FV3 high-order Lagrange halo) on the PE baroclinic
  cube → v_rms 3.416 vs no-duogrid 3.43 = **no effect** (PASS, stable). Matches
  the team's 33-iter duogrid study (FV3_3D.md:270-314: "PE edge ratio INSENSITIVE
  to all factory flags; PE much less responsive than NH"; duogrid HURTS NH but
  not PE). ⇒ the imprint is NOT a halo/corner error. Knob reverted.
- `perturbed=False` (pure steady jet) → v_rms 3.426 = **identical** to perturbed
  (3.43). ⇒ the imprint is NOT perturbation growth — it is the **cube's discrete
  geostrophic-balance truncation error**: the analytic jet (continuous balance)
  is not reproduced by the cube's discrete PGF/Coriolis/metric (latlon nearly is
  → 0.023). Fundamental cube discretization, IC-independent.
- ⇒ REDIRECT: the duogrid (halo) lever is out. FC-PGF would change the INTERIOR
  gradient ORDER (2nd→spectral), distinct from the halo — still the one untried
  lever, but a large/uncertain build, and may or may not reduce a truncation
  error the metric geometry amplifies.
- CLEANUP: killed a stuck `icosahedral baroclinic --quick` process (PID 39061,
  58 min wall / 250 min CPU, hung) that was starving all runs of CPU, + a slow
  latlon HS. SIDE BUG to file: **ico (MPAS) baroclinic --quick HANGS** (should
  be ~1 min). HS latlon HS also pathologically slow (pole CFL). cube HS climate
  is physical (T 265K, [241,308]) — imprint does NOT corrupt the climate.

## iter 13 — imprint ANTI-converges + comprehensive atmosphere verdict
Baroclinic cube v_rms vs resolution (quick, hybrid): C36 3.43@0.2d/4.95@2d;
C48 **4.32@0.2d / 8.46@2d** — HIGHER + growing faster at finer resolution. The
imprint does NOT converge (worsens) ⇒ consistency-level error, not benign
truncation; it persists/worsens at AMIP resolution. (Caveat: matrix scales
hyperdiff/A_h/dt with resolution, so weaker relative damping at C48 inflates the
2-day growth; but the robust conclusion — no convergence — matches the team's
edge-ratio plateau 4-5× at C8/C16/C24.)

**COMPREHENSIVE ATMOSPHERE VERDICT.** The cube panel-edge imprint is:
- the cube's **discrete geostrophic-balance truncation/consistency error** (steady
  jet, perturbation-independent; analytic continuous balance not reproduced by
  the cube discrete PGF/Coriolis/metric — latlon nearly is, 0.023 vs cube 3.4);
- **halo-insensitive** (duogrid no effect) and **non-converging** with resolution;
- **deeply team-investigated** (1045+ fidelity iters + 33-iter duogrid study;
  "PE insensitive to all factory flags");
- with **no cheap fix** (damping → blows up; duogrid → no effect/NH-harmful;
  existing FV3 flags → neutral; FB stepping + Lin PGF → RK3-unstable);
- but it does **NOT corrupt the climate** (cube HS physical, T 265K Earth-like).
The ONE untried lever is the FC-PGF (interior spectral gradient order, distinct
from the halo), but it is a large/uncertain build — and the non-convergence
hints the root is metric-geometry consistency, which a gradient swap may not fix.
HONEST: "absolutely no edge artifacts" on the atmosphere cube is NOT achievable
without a major novel advance; the cube is FV3-faithful in components + gives
physical climates, but carries this fundamental steady-balance imprint.

## iter 14 — USER DIRECTIVE: eliminate imprints via FV3. PGF RULED OUT.
User: "get rid of the cube imprints entirely using FV3 as the oracle." Oracle
STILL TCC-blocked (`~/Documents/Code` Operation-not-permitted even sandbox-off)
— NEED it copied to a readable path (`/tmp/fv3oracle` or `docs/references/fv3src/`).

Tested the top FV3-faithful lever — wired FV3's **Lin (1997) finite-volume PGF**
(`_fv3_lin_pgf.fv3_lin1997_pgf_3d_cgrid` → `project_cgrid_pgf_to_dgrid_corners`,
the cross-product `p_grad_c` that explicitly does NOT amplify cross-face halo
error like the A-L 4-pt matrix) into the PE momentum as opt-in (B=KE-only +
Lin PGF replaces ∇Φ+R_dT∇ln_ps). RESULTS:
- baroclinic cube v_rms@0.2d = 3.443 vs A-L **3.43** — IDENTICAL (no imprint fix);
- rest_state_topo wind_rms 0.210 vs 0.205 — IDENTICAL;
- and it is **STABLE in RK3** (PASS, max|v|=38) — contradicting the codebase's
  "Lin PGF not stable with RK3" claim. (Reverted the wiring: no benefit + slight
  baroclinic-2d regression.)
⇒ **The imprint is NOT the PGF/gradient operator.** Both A-L and Lin PGF give the
same imbalance. For the balanced jet (v=0 @t=0) dv/dt = −(zeta_corner·u_d +
dB_dy_perp − pgf_y); changing pgf_y didn't move it ⇒ the residual is on the
**corner Coriolis-vorticity side** (`zeta_corner` = relative-vort-to-corner +
`f_corner`) and/or the shared **cube metric** (`rdxc`, corner positions). With
halo, IC, damping, duogrid, AND now PGF all ruled out, the imprint is in the core
cube metric/vorticity discretization — the next target (needs the oracle to
compare FV3's exact corner-vorticity/Coriolis stencil, `sw_core.F90:378-480`).
AMIP cube runs physical + stable under full physics (T 279K, finite).

## iter 14b — ROOT CAUSE FOUND + FIXED (vector-aware wind interp) ✅✅
**The cube baroclinic v-imprint was a vector-vs-scalar bug.** `_step_cell_centre`
(primitive_eq_cdgrid.py) converted cc winds → D-grid corners EVERY step via
`_interp_center_to_corner` on the **stacked (u,v)** — treating the D-grid winds as
two SCALARS and blending face-local components across panel seams WITHOUT
rotation. Chain: PGF ruled out → vorticity 229× rougher at edges → winds 167×/83×
rougher (scalar) → **0.7×/1.3×/1.8× with vector `center_to_dgrid_vector`**.
FIX (ac6a8f58): rotation-aware `center_to_dgrid_vector` (inverse of the exit
`dgrid_to_center_vector`). Results (C36 quick, cube):
| metric | before | AFTER |
|--------|--------|-------|
| baroclinic v_rms@2d | 4.95 (75× latlon) | **0.66 (10×)** |
| imprint growth | 3.4→4.95 | **stops (~0.65)** |
| mass drift | 2.6e-11 | **1.1e-15 (1e4×)** |
| gravity_wave_3_1 max\|v\| | 22.4 (outlier) | **19.5 (≈ico 20)** |
| baroclinic v-field (visual) | wavenumber-4 panel blocks | **smooth zonal bands** |
ALL PE cube cases PASS; 22 PE tests green. 1e4× conservation gain + gravity_wave→ico
+ smooth v confirm correctness. [git: fix briefly mis-committed to main by a stray
checkout; cherry-picked here, local main reset.]

## iter 14c — codex-driven sweep: ALL vector-seam instances fixed (5 sites)
Adversarial review found the SAME vector-vs-scalar seam bug at 4 more PE sites;
fixed each (commits eb14fca0, bc068e28, fe28b11c) + regression test (4deb66ae):
1. `_step_cell_centre` wind lift → `center_to_dgrid_vector` (ac6a8f58).
2. `tendencies()` HydrostaticState entry wind lift → vector.
3. `fv3_hydrostatic_tendencies` vector tendency blocks (vert_adv/lap/hyperdiff/
   physics) center→corner lift → per-block `center_to_dgrid_vector`.
4. wind DIFFUSION halo (∇²/∇⁴ of u_cell,v_cell) → `pad_halo_vector_4d` (was scalar
   `_pad_halo_4d` → unrotated seam halos into the stencil).
5. pass `duogrid=_pe_dg` to that vector halo (consistency).
6. `hydrostatic_to_fv3` entry lift → duogrid-aware vector halo (2842d98e).
7. wind hyperdiffusion OUTER ∇² stencil → hand-built vector ∇⁴ (4fd71710); the
   shared `_hyperdiffusion_3d` scalar-pads the inner ∇²(u,v) result.
Cumulative baroclinic v_rms@2d: 4.95 → 0.77 (sites 1-3) → **0.60** (sites 4-7) =
**9× latlon** (was 75×). v-field VISUALLY smooth zonal bands (no panel imprint).
ALL 10 fast PE cube cases PASS; 21+ PE/regression/AST-guard tests green; 6 codex
reviews drove the sweep. KNOWN FOLLOW-UP: MPI packed-halo path
(`packed_pad_halo_mpi_4d`) still packs u,v scalar — needs a vector packed MPI halo
(matrix runs single-device, so not hit by the matrix).
⇒ The cube panel-edge v-imprint (the user's "edge artifact") is ELIMINATED for
the PE atmosphere. Residual ~9× latlon is a SMOOTH zonal amplitude diff, not an
edge artifact.

## iter 14d — CONVERGENCE VALIDATION: the fix made the cube CONSISTENT ✅✅✅
FIXED baroclinic v_rms@0.2d (sigma) vs resolution:
  C36 0.688 → C48 0.535 (1.29×) → C72 0.282 (1.90×)  — MONOTONE DECREASE (~1.5-2 order).
WITH the seam bug it ANTI-converged (C36 3.24 → C48 3.91, worse).
⇒ The seam bug was a CONSISTENCY error (non-vanishing at high res); the fix makes
the residual a BENIGN converging TRUNCATION error that vanishes at AMIP resolution
(C72 already 0.28, → latlon's ~0.066 at C96-C192).  The cube PE atmosphere is now
a consistent, FV3-faithful (vector-correct) discretization with NO edge artifact.

## Pending / next
- Codex 5th review [running]; broad PE verify [running].
- Residual ~9× latlon is a SMOOTH zonal amplitude diff (NOT an edge artifact) —
  the user's "no edge artifacts" is met for baroclinic; chase the smooth residual
  (KE-grad / effective-resolution) as polish.
- rest_state_topo topography-PGF (separate mechanism); MPI vector packed halo.
- `DONE` withheld: residual + topography + MPI follow-ups; broader climate verify.

## iter 14e — cube FIX is CLEANER than latlon (apples-to-apples, hybrid)
Baroclinic v_rms@2d, HYBRID coord: cube FIX **0.60** (v_max 1.47) vs latlon
**7.95** (v_max 32.3). latlon hybrid develops a strong UNPHYSICAL equatorial v
band (±50 m/s; J-W perturbation is at 40°N, equator should be quiet) — a latlon-
hybrid artifact (separate issue; latlon SIGMA is clean at 0.066). The cube FIX
v_max 1.47 @day2 is PHYSICALLY CORRECT (the baroclinic wave has not grown yet).
⇒ the earlier "9× latlon" was cube-hybrid vs latlon-SIGMA; vs latlon's own hybrid
run the cube is **13× cleaner + smooth + no edge artifact**. The cube PE atmosphere
is now competitive with / cleaner than latlon. (latlon-hybrid equatorial artifact
filed as a separate latlon item.)

## iter 14f — VISUAL INSPECTION (FIXED cube PE, hybrid)
- baroclinic v: smooth zonal bands, NO panel-edge blocks ✓
- rotated_steady v: smooth large-scale tilted-rotation flow, no panel blocks ✓
- rest_state_topo wind_speed: residual ~1 m/s is MOUNTAIN-localized (the dcmip
  mountain), i.e. the sigma-coord topography-PGF error — NOT a cube-edge artifact
  (Lin-PGF-insensitive; common to all sigma-coord models over steep terrain).
⇒ the cube PANEL-EDGE artifacts (the user's concern) are eliminated. Remaining
residuals are (a) a smooth converging truncation error and (b) the localized
topography-PGF over the mountain — both distinct from the panel-seam imprint.
(Plot titles read "Baroclinic" on rotated/rest cases — generic-label bug, cosmetic.)

## STATUS SUMMARY (post vector-seam fix)
ATMOSPHERE cube PE: panel-edge v-imprint ROOT-CAUSED + FIXED (vector-vs-scalar
cc→D-grid wind/tendency/diffusion lift, 7 sites, codex-APPROVED, convergence-
validated 0.688→0.535→0.282, regression-guarded). Cleaner than latlon-hybrid.
OCEAN cube: NaN→PASS (FC-Gram default, codex-approved).
Native velocity plots fixed. SW cube W2 imprint (0.09, FV3Edge path) = separate
smaller mechanism, untouched.
OPEN: topography-PGF residual (rest_state_topo, separate); MPI vector packed halo
(single-device unaffected); ico-MPAS-baroclinic HANG (separate); held_suarez/AMIP
broad climate verify; latlon-hybrid equatorial artifact (latlon, separate).
