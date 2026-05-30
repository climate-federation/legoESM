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

## Pending / next
- HS climate match (cube vs latlon) — latlon slow [running, monitor armed].
- AMIP cube vs latlon (user wants it + continent-longitude check).
- Atmosphere FC-PGF build (per plan above) — the imprint fix.
- `DONE` withheld: atmosphere panel-edge imprint is real + the FC fix is a build.
