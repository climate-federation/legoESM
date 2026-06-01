# FV3-faithful cubed-sphere — change log (shrunk @ iter130; prior detail in git + memory `cube-fv3-faithfulness-state`)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
≈ MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge artifacts**, visual + quantitative,
codex/oracle-reviewed. CPU only (Metal broken). **Directive: never A-grid; be FV3-faithful; use the
Fortran as oracle; do NOT improvise.** Branch `latlon-fv-amip-verify`; fv3 work pushed to `fv3-faithful-cube`.

## ✅ VERIFIED FAITHFUL (codex-vetted iter109; close to MPAS/latlon across the feasible scope)
- **area_corner (#faces)-junction SCALING** (iter84-91, codex-APPROVED, NET-ZERO): edges ×2, vertices
  ×3, C1 guard n>=1 (oracle fv_grid_tools.F90 edge=2*get_area, vertex=3*get_area; replaces iter-670
  interior-copy). W2 L2=1.76e-4 unchanged, ocean 9/9 rest machine-zero, 4 SW gold fingerprints re-pinned.
  CAVEAT: `sg_area` is PLANAR CHORD — only the ×2/×3 SCALING is faithful; chord→spherical is a tracked follow-up.
- **gnomonic_ed grid** WIRED + halo-collapse fixed (iter73) + codex-approved; equiangular byte-identical.
- **atm SW cross-grid** (iter92, 16/16): cube W2 L2 1.76e-4 ≈ latlon 2.67e-4 ≈ ico 9.9e-5. baroclinic clean.
  iter135 RE-CONFIRMED production SW matrix 4/4 PASS (W2/W5/cosine_bell/W6, unchanged); W2 v_ll_Linf=0.344
  = the known tiny v-imprint (centered-vs-upwind vort tradeoff, [[the upwind FB fix is the unstable path]]).
  Surfaced W2 v-wind native + cross-grid wind_speed PNGs to user for visual edge-artifact verdict.
- **cosine_bell** (codex-vetted iter109): cube uses apply_fortran_xppm_boundary=True; 12-day cube/ico=1.4×
  (cube best mass-cons), 99% residual panel-INTERIOR bulk PPM dissipation ⇒ faithful PPM plateau, not a bug.
- **3D rest_state_topo "13× artifact" = FLOAT32 precision** (iter97-101): cube 3D PGF EXACTLY well-balanced
  in float64 ((Φ_k−phis) std=0.0); PE runs f32 by design. Optional f32 fix.
- **ocean cross-grid dynamics** (iter105/111): geostrophic cube 0.0143 ≈ latlon 0.0159 ≈ mpas 0.0169;
  IGW omega 1.09e-4 IDENTICAL all 3 grids. ocean cube 9/9 rest machine-zero. ⇒ ocean ≈ MPAS/latlon (OMIP).

## ❗ THE ONE REMAINING GAP — experimental SW FB-port edge instability (production SW is STABLE + faithful-in-results)
CONFIRMED FV3 MISMATCH (codex iter109): production SW (operators_cdgrid.py:1167) + 3D PE (:445) use CENTERED
`zeta_corner*v_d`; FV3 sw_core.F90 upwind donor-cell. The faithful staggered c_sw→d_sw port (FB chain,
`fv3_fb_sw_step`, uses upwind `_vorticity_flux`) is EXPERIMENTAL and has a weak edge instability (W2 C36:
day1 max|u_d| 48.6 vs 38.6, day2 NaN). Production's co-located D→A-avg scheme suppresses it.

**FULLY CHARACTERIZED (iter123-129, two codex adversarial reviews drove it to ground truth):**
- **TYPE = weakly-UNSTABLE spectrum + NON-NORMAL TRANSIENT (iter128 genuine Arnoldi eigs on the linearized
  1-step M', dim 23760):** spectral radius **ρ(M')≈1.0019>1** — a CLUSTER of REAL unstable eigenvalues
  (~1.0015–1.0019) — WITH strong non-normal transient growth on top (K=50-window optimal ~1.007/step ≈ ×1.4
  over a few hours ≫ the 1.0019 eigenvalue). σ_asymp≈0.5/day. The earlier "per-step λ≈1.007" (iter123-125)
  is the TRANSIENT rate, NOT the eigenvalue; ‖M^K‖^(1/K) ≠ spectral radius for a non-normal operator (the
  hard-won meta-lesson — codex [high] was right both times: conditioning iter123, Arnoldi iter128).
- **2Δx GRID-SCALE EDGE-LOCALIZED (iter125, codex [medium] #2):** modal-ID C24/C36/C48 — argmax on a panel
  edge row (j0, face 2/3), Nyquist(2Δx) power 0.79–0.91, edge-energy frac →0.997 as res refines, σ GROWS
  with resolution. Mode energy 91% v_d. Driven by a near-neutral KE-grad↔PGF gravity-wave exchange at the edge.
- **RULED OUT (robust):** float32 (survives f64), dt & resolution stability-limit (iter124-125); EVERY
  component reads FV3-faithful (a2b_ord4 edge = FV3 duogrid plain-stencil path a2b_edge.F90:98 iter129; d_sw
  zeta 114, vort-flux 110b, corner-vort/fv_tp_2d/circulation halos 94/107/115, B-grid KE rsin2 118 + d_sw3
  vb/ub Courant FORMULAS faithful iter132 [sw_core.F90:1270-1275 duogrid form]; c_sw/d_sw assembly 119,
  d_sw4/5 duogrid no-ops 120); BOTH FV3 damping operators (vorticity iter125 + divergence
  dddmp=0.2 iter129: ρ only 1.00188→1.00167, needs unphysical ~10× to stabilize) ⇒ the mode is in their
  JOINT NULL SPACE (resolves codex [medium] #4: damping can't fix it). **iter131: divergence damping is
  FAITHFUL incl. the deliberate EDGE ZEROING** — `_divergence_corner_duo` matches FV3 sw_core.F90:2427-2440
  (grid_type<3 else-branch): cross-velocity correction + face-boundary zeroing (i/j=0,n →0) + 0.25
  attenuation (i/j=1,n-1). So FV3 ITSELF nulls the corner divergence at the panel edge ⇒ divergence damping
  can't control an edge mode in FV3 either (faithful, NOT a bug). The iter-132 mode='edge' in the iterated
  Laplacian has limited impact BECAUSE divg_d is already zeroed/0.25-attenuated there.
- **PRIME REMAINING SUSPECT + FIX (iter131 CORRECTION — moved off the damping):** since both FV3 damping
  operators are faithful AND FV3 zeroes edge divergence damping yet is edge-STABLE, the bug must be in the
  **c_sw/d_sw DYNAMICS edge treatment** (FV3's dynamics are edge-stable without edge damping; the port's
  create the weakly-unstable mode). NOT the damping-edge-halo (iter129's suspect, ruled out). d_sw3 vb/ub
  EDGE computation RULED OUT (iter133, decisive Arnoldi test, inlined KE validated max|diff|=0): replacing
  the edge vb/ub (this-face edge metric) with interior extrapolation does NOT drop ρ (1.00189→1.00194). ⇒
  the edge KE Courant is not the cause. **NET: NO single component or edge candidate localizes it** — every
  dynamics FORMULA + both dampings + a2b + the d_sw3 vb/ub edge are all faithful/ruled-out, yet ρ=1.0019.
  The instability is an EMERGENT assembly-level interaction. NaN mechanism (transient→nonlinear) is a
  HYPOTHESIS (not causally shown).
- **iter134 ASSEMBLY AUDIT (dyn_core oracle) + a genuinely MISSING TERM:** FB Phase 1→4 sequencing is
  FAITHFUL — FV3 dyn_core.F90 SW orchestration is c_sw(489)→p_grad_c(629)→d_sw1-6(831-1256)→one_grad_p
  (1531), matching my FB. BUT FV3's one_grad_p adds the **d_ext EXTERNAL-MODE divergence damping**
  (dyn_core.F90:2415-2476: u += rdx*(divg2(i,j)-divg2(i+1,j)), divg2=d_ext*da_min_c*divg_d, d_ext=0.02
  operational fv_arrays.F90:399) — my FB Phase 4 OMITTED it. Added + Arnoldi-tested (d_ext=0 matches
  model.step to 1e-14): ρ = 1.00187(0) → 1.00176(0.02) → 1.00157(0.1) → 1.292(0.25 over-damps). ⇒ the
  missing d_ext term is a REAL faithfulness gap that REDUCES ρ in the right direction but does NOT stabilize
  at operational values (divg_d is edge-zeroed → d_ext weak at the edge; 0.25 over-damps). Worth adding to
  the FB code as a faithfulness fix, but NOT the cure for the edge mode. ⇒ single-component + assembly +
  missing-term avenues now EXHAUSTED; decisive next step = field-by-field compare vs a RUNNING FV3 Fortran
  step (full GFDL build, beyond this loop). HARNESS/ASSETS (validated, all float64): /tmp/fb_arnoldi.py (eigs ρ),
  /tmp/fb_arnoldi_damp.py (ρ-vs-damping), /tmp/fb_localize2.py (energy-norm 5-term decomp), /tmp/fb_nonnormal.py
  (K-trend). FIX-VALIDATION = re-run Arnoldi (does ρ drop <1?) + transient gain. Separate from the production
  C96 W5 mode (slow ~1.0007).

## CPU-PROHIBITIVE here (rely on recorded audits + representative cross-grid + t=0 probes + the spectral harness)
3D atm baroclinic cross-grid (>21 min/case); full ocean matrix --grid all (57 cases, multi-hr); climate/AMIP;
jvp-AD eigen-analysis through the FB chain (use finite-diff + Arnoldi); running-Fortran field compare (no build).

## OPEN QUEUE
1. FB edge instability: structural cube-edge c_sw→d_sw coupling / damping-edge-halo fix → re-run Arnoldi
   (ρ<1?) → FB→production. Needs running-Fortran field compare or dedicated edge-halo debug (fresh context).
2. (optional, low-pri) chord→spherical sg_area; f64/well-balanced PGF (f32 rest-state); 3D PE upwind vort.
3. human visual PNG verdict (W2 v / W5 wind_speed / cross-grid atm+ocean — surfaced); pre-existing 17-test
   swamp in test_cdgrid_fv3_regression.py (deleted results/ + concurrent churn).
State persisted: memory `cube-fv3-faithfulness-state` (codex-vetted, iter89-129).
