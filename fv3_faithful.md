# FV3-faithful cubed-sphere — change log (shrunk @ iter122; prior detail in git + memory `cube-fv3-faithfulness-state`)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
≈ MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge artifacts**, visual + quantitative,
codex/oracle-reviewed. CPU only (Metal broken). **Directive: never A-grid; be FV3-faithful; use the
Fortran as oracle; do NOT improvise.** Branch `latlon-fv-amip-verify`.

## ✅ VERIFIED FAITHFUL (codex-vetted iter109; close to MPAS/latlon across the feasible scope)
- **area_corner (#faces)-junction SCALING** (iter84-91, codex-APPROVED, NET-ZERO): edges ×2, vertices
  ×3, C1 guard n>=1 (oracle fv_grid_tools.F90 edge=2*get_area, vertex=3*get_area; replaces iter-670
  interior-copy). W2 L2=1.76e-4 unchanged, ocean 9/9 rest machine-zero, 4 SW gold fingerprints re-pinned.
  CAVEAT: `sg_area` is PLANAR CHORD — only the ×2/×3 SCALING is faithful; chord→spherical is a tracked
  follow-up (trips the gold-fingerprint surface).
- **gnomonic_ed grid** WIRED + halo-collapse fixed (iter73) + codex-approved; equiangular byte-identical.
- **atm SW cross-grid** (iter92, 16/16): cube W2 L2 1.76e-4 ≈ latlon 2.67e-4 ≈ ico 9.9e-5. baroclinic clean.
- **cosine_bell** (codex-vetted iter109): cube uses apply_fortran_xppm_boundary=True; iter-61/62 12-day
  cube/ico=1.4× (cube best mass-cons), 99% residual panel-INTERIOR bulk PPM dissipation (NOT edge) ⇒
  faithful PPM plateau (cube≈ico), not a bug.
- **3D rest_state_topo "13× artifact" = FLOAT32 precision, not a bug** (iter97-101): cube 3D PGF EXACTLY
  well-balanced in float64 (forcing f64 ⇒ (Φ_k−phis) std=0.0); PE runs f32 by design. Optional f32 fix.
- **ocean cross-grid dynamics** (iter105/111): geostrophic cube 0.0143 ≈ latlon 0.0159 ≈ mpas 0.0169;
  IGW omega 1.09e-4 IDENTICAL all 3 grids. ocean cube 9/9 rest machine-zero. ⇒ ocean ≈ MPAS/latlon (OMIP).

## ❗ THE ONE REMAINING GAP — SW FB-port eigenmode (precisely typed; harness-equipped; structural fix pending)
CONFIRMED FV3 MISMATCH (codex iter109): production SW (operators_cdgrid.py:1167) + 3D PE
(primitive_eq_cdgrid.py:445) use CENTERED `zeta_corner*v_d`; FV3 sw_core.F90 upwind-selects (donor-cell).
Faithful path = the FB staggered c_sw→d_sw scheme (uses upwind `_vorticity_flux`); the FB chain has a
RESIDUAL (W2 C36 day1 48.6 vs 38.6, day2 NaN).
EIGEN-ANALYSIS (iter115b-121, finite-diff K=50 power iteration — jvp CPU-prohibitive):
- **It IS a genuine growing eigenmode: per-step λ=1.0072** (K=8 was masked by the neutral W2 modes).
- **TYPE: a 2Δx-along-edge COMPUTATIONAL MODE in u_d** at the cube panel edge (argmax face3 i=0 row,
  alternating sign `-+-+-+` in j; 2D-checkerboard corr 0.10 ⇒ 1D-along-edge, not vertex/2D).
- **STRUCTURAL, NOT dampable** (harness: damp_v×8→NaN over-damp limit; nord_v/hyperdiff neutral — in the
  del-n null space at the edge; confirms iter82-83) and **NOT any single component** (all FV3-faithful:
  d_sw zeta(114), vorticity-flux sina_u(110b), corner-vort halos(94/107), fv_tp_2d halo(115), B-grid KE
  rsin2_corner(118), dissipation/Phase4/wind-halo(82-83); FB assembly READS structurally correct(119);
  FV3 d_sw4/d_sw5 corner fixes are duogrid no-ops(120)).
- ⇒ the bug is the STRUCTURAL cube-EDGE c_sw→d_sw coupling that should suppress the 2Δx-along-edge u_d
  mode (production co-located scheme kills it via the D→A avg; the FB staggered port doesn't at the edge).
- **FIX-VALIDATION HARNESS (iter117, validated iter120):** K=50 finite-diff power iteration → any candidate
  fix → re-run → does per-step λ drop below 1? (~min/test). NEXT (dedicated): localize the i=0-edge d_sw
  term injecting the 2Δx-in-j mode (decompose ke_diff_u vs fy_vort at i=0) → match the FV3 edge coupling →
  test against λ. The FB residual eigenmode (fast, λ=1.0072, W2 C36) is SEPARATE from the production C96 W5
  eigenmode (slow, ~1.0007). Best in a FRESH context (reading/component/metric avenues exhausted).

## CPU-PROHIBITIVE here (rely on recorded audits + representative cross-grid + t=0 probes + the λ harness)
3D atm baroclinic cross-grid (>21 min/case); full ocean matrix --grid all (57 cases, multi-hr); climate/AMIP;
jvp-AD eigen-analysis through the FB chain (use finite-diff instead).

## OPEN QUEUE
1. FB 2Δx-edge eigenmode: localize the i=0-edge d_sw term vs the λ harness → structural cube-edge coupling
   fix → FB→production. The genuine remaining faithfulness gap (dedicated/fresh-context).
2. (optional, low-pri) chord→spherical sg_area; f64/well-balanced PGF (f32 rest-state); 3D PE upwind vort.
3. human visual PNG verdict (W2 v / W5 wind_speed / cross-grid atm+ocean — surfaced); pre-existing 17-test
   swamp in test_cdgrid_fv3_regression.py (deleted results/ + concurrent churn).
State persisted: memory `cube-fv3-faithfulness-state` (codex-vetted, iter89-121).
