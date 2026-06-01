# FV3-faithful cubed-sphere — change log (shrunk @ iter140; prior detail in git + memory `cube-fv3-faithfulness-state`)

Goal (target, NOT all achieved): cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
≈ MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **minimal cube edge artifacts** (small documented residuals
remain — "zero" not reached), visual + quantitative, codex/oracle-reviewed. CPU only (Metal broken).
**Directive: never A-grid; be FV3-faithful; use the Fortran as oracle; do NOT improvise.**
Branch `latlon-fv-amip-verify`; fv3 work pushed to `fv3-faithful-cube`.

## ✅ STABLE + CROSS-GRID-CLOSE + COMPONENT-AUDITED (+ FIRST MEASURED FV3 bit-match iter143)
**SCOPE (honest):** "faithful" = (a) component oracle source-audits + (b) cube ≈ MPAS/latlon/ico cross-grid
+ (c) stable/conservative. Mostly INFERRED (not yet a full field-by-field RUNNING-FV3 match); one default
production operator is a KNOWN mismatch (centered vs FV3-upwind vorticity). Residual edge imprints exist
(NOT "zero edge artifacts"): W2 v-wind imprint (v_ll_Linf=0.34 on an analytically-zero field) + slow C96 W5
vertex mode (~1.0007) — small + stable but nonzero coherent residuals.
**iter143-144 — *MEASURED* FV3 bit-matches (codex iter137's "inferred not measured" cracked):** gfortran-
extract FV3 subroutine standalone (mocked minimal types, NO mpp/MPI/NetCDF) → feed the SAME inputs as my
Python port → bit-compare. PROVEN TOOLCHAIN (verbatim FV3 loop bodies + identical-input + negative-control).
MEASURED-faithful so far (all machine-precision, independently re-verified):
  (1) iter143 `divergence_corner_duo` ↔ `_divergence_corner_duo`: interior max|diff|=4.2e-22 (N=8/12).
  (2) iter144 `a2b_ord4` (duogrid) ↔ `_interp_center_to_corner_a2b_ord4`: 2.2e-16=1 ULP (N=24/48); in BOTH
      the FB Phase-4 PGF AND the production 3D PGF gz_b. (neg-control: swap a1↔a2 → 0.53 diff.)
  (3) iter145 d_sw3 B-grid KE ↔ `_bgrid_ke_transport` (the DOMINANT FB-mode source term, +1.47e-3): full
      ke_corner interior max|diff|=3.6e-12 (N=24/48), sub-terms vb/ub Courant 4.5e-13 + ytp_v/xtp_u PPM
      jord=9 1.8e-15 + corner-combine all faithful. (neg-control: perturb PPM r3 1e-6 → 3e-8, 7 orders up.)
**KEY (iter145) — FB bug MEASURED-narrowed to the CROSS-FACE EDGE coupling, NOT the interior dynamics:**
the d_sw3 KE matches bit-for-bit to edge-distance 1; the ONLY divergence (full max|diff|≈42) is the
OUTERMOST corner ring = the BGRID_NE cross-face sync (`synchronize_bgrid_ne_corner_geo` — my port's
geographic-frame avg vs FV3's per-seam rotation). So the 3 dominant FB-mode formulas are all MEASURED-
faithful on the interior ⇒ the edge instability lives in the cross-face edge coupling / halo (BGRID_NE sync
+ the duogrid cross-face fill), now the measured prime suspect. NEXT: measure the BGRID_NE sync / duogrid
halo (multi-tile compare) vs FV3 — the crux. Assets /tmp/fv3_poc{,_a2b,_dsw3}/.
- **area_corner #faces-scaling** (iter84-91, codex-OK, NET-ZERO): edges ×2, vertices ×3, C1 guard n>=1
  (oracle fv_grid_tools.F90). W2 L2=1.76e-4, ocean 9/9 rest machine-zero, 4 golds re-pinned. CAVEAT: sg_area
  is PLANAR CHORD (only the ×2/×3 scaling faithful; chord→spherical deferred). iter142 QUANTIFIED the chord
  caveat vs FV3 get_area (spherical excess, l'Huilier): O(dx²), rel err ~2e-3 @C8 → ~1e-4 @C36 (mean);
  cube-VERTEX quads ~7e-5 @C36 = slightly BETTER than the mean, NOT worse ⇒ the chord area is BENIGN at
  production res + decisively RULED OUT as the vertex-artifact source (≪ the 0.34/1.0019 artifact scales).
  Upgrade worth it only for C1-C8 fidelity/cleanliness. gnomonic_ed grid wired (iter73).
- **atm SW cross-grid** (iter92/135, 16/16 PASS): cube W2 L2 1.76e-4 ≈ latlon 2.67e-4 ≈ ico 9.9e-5; W5/W6/
  cosine_bell pass. cosine_bell (codex iter109): cube xppm boundary, 99% residual = panel-INTERIOR PPM
  dissipation plateau (cube≈ico), not edge. iter139: ATM cross-grid projection VERIFIED CLEAN (localized
  features ALIGN: cosine_bell -68° + W5 mountain -90° on cube/latlon/ico; consistent [-180,180)).
- **3D rest_state_topo "13× artifact" = FLOAT32 precision** (iter97-101): cube 3D PGF EXACTLY well-balanced
  in f64; PE runs f32 by design. Optional f32 fix.
- **ocean cross-grid** (iter105/111): geostrophic cube 0.0143 ≈ latlon 0.0159 ≈ mpas 0.0169; IGW omega
  1.09e-4 identical 3 grids; rest 9/9 machine-zero. barotropic_wave (iter136-138): a 180° lon-projection
  bug in the OCEAN comparison harness (cube/latlon mislabeled vs mpas) — user+codex flagged, **FIXED iter138**
  (run_ocean_test_matrix `_regrid_2d`: cube regridder [-180,180) rolled +n_lon//2 → [0,360]; latlon no longer
  mis-rolled). Post-fix IC aligns at 180°E on all grids; cube RETAINS amplitude (~0.86 vs latlon/mpas ~0.21
  dissipated) = the test's EXPECTED retention (cube less dissipative, edge-clean), not an artifact.
  codex-reviewed iter141: added a loud-fail guard so the cube branch rejects a non-[0,360] target_lon
  (prevents silent 180° re-break); latlon branch confirmed self-consistent (native data+label; staggered
  fields legitimately differ in size, so no guard). barotropic_wave 3/3 PASS + 17/17 regrid tests post-guard.
  Pending: resolution-convergence check.

## ❗ THE ONE REMAINING GAP — experimental SW FB-port edge instability (production SW is STABLE + faithful-in-results)
CONFIRMED FV3 MISMATCH (codex iter109): production SW (operators_cdgrid.py:1167) + 3D PE (:445) use CENTERED
`zeta_corner*v_d`; FV3 upwind donor-cell. The faithful staggered c_sw→d_sw port (`fv3_fb_sw_step`, upwind
`_vorticity_flux`) is EXPERIMENTAL + has a weak edge instability (W2 C36 day2 NaN). Production's co-located
D→A-avg scheme suppresses it. FULLY CHARACTERIZED (iter123-134, two codex reviews drove it to ground truth):
- **TYPE (iter128 genuine Arnoldi, dim 23760):** weakly-UNSTABLE spectrum **ρ(M')≈1.0019>1** (real eigenvalue
  cluster) + strong NON-NORMAL TRANSIENT growth (K=50 optimal ~1.007/step ≫ the eigenvalue). The earlier
  "λ≈1.007" was the TRANSIENT rate, NOT the eigenvalue (‖M^K‖^{1/K} ≠ spectral radius for a non-normal op —
  codex [high] right twice: conditioning iter123 + Arnoldi iter128). 2Δx grid-scale, edge-localized (j0,
  Nyquist power 0.79-0.91, edge-energy→0.997), 91% v_d, a near-neutral KE-grad↔PGF gravity-wave edge exchange.
- **RULED OUT (exhaustive):** float32/dt/resolution stability-limit; EVERY component reads FV3-faithful
  (a2b duogrid path iter129, d_sw zeta 114, vort-flux 110b, halos 94/107/115, B-grid KE 118, d_sw3 vb/ub
  formulas 132 + the vb/ub EDGE decisively ruled out by Arnoldi iter133, assembly 119, d_sw4/5 120); BOTH
  damping operators (divergence damping FAITHFUL incl. FV3's deliberate edge-zeroing iter131 — so FV3 nulls
  edge div-damp too, can't be the fix); FB Phase 1→4 assembly matches dyn_core (iter134). The ONE genuinely
  MISSING term (iter134) = d_ext external-mode div-damping in one_grad_p (FV3 d_ext=0.02); adding it reduces
  ρ (1.00187→1.00157@0.1) but does NOT cure (edge-zeroed divg_d). ⇒ EMERGENT assembly-level — now
  MEASURED-NARROWED (iter145, see the measured-bit-match list at top): the 3 dominant FB-mode formulas
  (divergence damping, a2b PGF, d_sw3 KE) are all bit-for-bit faithful to FV3 on the INTERIOR; the only
  divergence is the CROSS-FACE EDGE ring (the BGRID_NE corner sync). ⇒ the bug is in the cross-face edge
  coupling / halo (BGRID_NE sync `synchronize_bgrid_ne_corner_geo` [geographic-frame avg vs FV3 per-seam
  rotation] + the duogrid cross-face fill) — the MEASURED prime suspect.
- **iter146 narrowing (reading-based) WITHIN the cross-face suspect:** (a) the BGRID_NE sync
  `synchronize_bgrid_ne_corner_geo` is a geographic-frame average of the FRAME-INVARIANT (east,north)
  components → mathematically correct for vector sync (≡ FV3 per-seam rotation if the cos/sin_angle_corner
  are right) ⇒ the sync FORMULA is not the bug. (b) the FB wind halo `_pad_halo_dgrid_for_ppm` uses
  `ext_vector_dgrid` WITH cos_sg5+rsin2 → it IS non-orthogonality-aware, so the memory's "dropped sin_sg(5)"
  issue (which is in the 3D-PE `pad_halo_vector_4d`) does NOT apply to the FB SW path. ⇒ the remaining
  unmeasured suspect = does `ext_vector_dgrid` (my duogrid cross-face wind/uc-vc halo) MATCH FV3's duogrid
  halo (duogrid_mod: fill_corner_region + lagrange_poly_interp) bit-for-bit?
- **NEXT:** measure `ext_vector_dgrid` vs FV3 `duogrid_mod` cross-face fill (multi-tile compare) — the crux,
  de-risked by the proven toolchain (duogrid_mod is a Lagrange-interp module, likely extractable sans MPI).
  HARNESS/ASSETS (validated, f64): /tmp/fb_arnoldi.py (eigs ρ), fb_arnoldi_damp.py, fb_localize2.py (energy
  5-term decomp), fb_nonnormal.py (K-trend). FIX-VALIDATION = re-run Arnoldi (ρ<1?). NaN mechanism
  (transient→nonlinear) is a HYPOTHESIS. Separate from the production C96 W5 mode (slow ~1.0007).

## CPU-PROHIBITIVE / OUT-OF-REACH (rely on recorded audits + representative cross-grid + the spectral harness)
3D atm baroclinic cross-grid (>21 min/case); full ocean matrix (57 cases, multi-hr); climate/AMIP.
RUNNING-FORTRAN: gfortran available (GCC 15.2). iter143 PROVEN the toolchain on divergence_corner_duo
(machine-precision bit-match, /tmp/fv3_poc/). Single self-contained subroutines are now a SOLVED, reusable
pattern (extract → mock minimal types → feed identical metrics to both → compare). The DECISIVE assembly-
level compare (full c_sw→d_sw→one_grad_p step, where the FB instability lives) still needs the coupled
modules (tp_core + a2b_edge + duogrid + fv_mp + fv_arrays) + the cross-face halo — a dedicated multi-step
build, but now de-risked by the proven toolchain. Next subroutines to measure: c_sw KE, a2b_ord4, the d_sw
vorticity flux (build up to the assembly).

## OPEN QUEUE
1. FB edge instability: DECISIVE = standalone-dycore running-FV3 step compare (gfortran avail; dedicated
   extraction effort, see above) OR dedicated edge-dynamics debug. Emergent/assembly ⇒ needs the full step.
2. barotropic_wave resolution-convergence check (cube amplitude vs res); chord→spherical sg_area; f32 PGF.
3. d_ext term: optionally add to the FB code (faithfulness completeness, default-off, net-zero) — found iter134.
State persisted: memory `cube-fv3-faithfulness-state` (codex-vetted, iter89-139).
