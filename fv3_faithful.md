# FV3-faithful cubed-sphere — change log (shrunk @ iter140; prior detail in git + memory `cube-fv3-faithfulness-state`)

Goal (target, NOT all achieved): cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
≈ MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **minimal cube edge artifacts** (small documented residuals
remain — "zero" not reached), visual + quantitative, codex/oracle-reviewed. CPU only (Metal broken).
**Directive: never A-grid; be FV3-faithful; use the Fortran as oracle; do NOT improvise.**
Branch `latlon-fv-amip-verify`; fv3 work pushed to `fv3-faithful-cube`.

## ✅ STABLE + CROSS-GRID-CLOSE + COMPONENT-AUDITED (NOT a measured FV3 run-match — codex iter137 [high])
**SCOPE (honest):** "faithful" = (a) component oracle source-audits + (b) cube ≈ MPAS/latlon/ico cross-grid
+ (c) stable/conservative. NOT a field-by-field match vs a RUNNING FV3 (no GFDL build — out of reach); and
one default production operator is a KNOWN mismatch (centered vs FV3-upwind vorticity). Residual edge
imprints exist (NOT "zero edge artifacts"): W2 v-wind imprint (v_ll_Linf=0.34 on an analytically-zero field)
+ slow C96 W5 vertex mode (~1.0007) — small + stable but nonzero coherent residuals.
- **area_corner #faces-scaling** (iter84-91, codex-OK, NET-ZERO): edges ×2, vertices ×3, C1 guard n>=1
  (oracle fv_grid_tools.F90). W2 L2=1.76e-4, ocean 9/9 rest machine-zero, 4 golds re-pinned. CAVEAT: sg_area
  is PLANAR CHORD (only the ×2/×3 scaling faithful; chord→spherical deferred). gnomonic_ed grid wired (iter73).
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
  dissipated) = the test's EXPECTED retention (cube less dissipative, edge-clean), not an artifact. Pending:
  resolution-convergence check.

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
  ρ (1.00187→1.00157@0.1) but does NOT cure (edge-zeroed divg_d). ⇒ EMERGENT assembly-level; single-
  component/assembly/missing-term avenues EXHAUSTED.
- **NEXT (out of loop reach):** field-by-field compare vs a RUNNING FV3 Fortran step (needs a GFDL build).
  HARNESS/ASSETS (validated, f64): /tmp/fb_arnoldi.py (eigs ρ), fb_arnoldi_damp.py, fb_localize2.py (energy
  5-term decomp), fb_nonnormal.py (K-trend). FIX-VALIDATION = re-run Arnoldi (ρ<1?). NaN mechanism
  (transient→nonlinear) is a HYPOTHESIS. Separate from the production C96 W5 mode (slow ~1.0007).

## CPU-PROHIBITIVE / OUT-OF-REACH (rely on recorded audits + representative cross-grid + the spectral harness)
3D atm baroclinic cross-grid (>21 min/case); full ocean matrix (57 cases, multi-hr); climate/AMIP; running-
Fortran field compare (no GFDL build).

## OPEN QUEUE
1. FB edge instability: needs a running-FV3 field compare (the decisive step) OR dedicated edge-dynamics debug.
2. barotropic_wave resolution-convergence check (cube amplitude vs res); chord→spherical sg_area; f32 PGF.
3. d_ext term: optionally add to the FB code (faithfulness completeness, default-off, net-zero) — found iter134.
State persisted: memory `cube-fv3-faithfulness-state` (codex-vetted, iter89-139).
