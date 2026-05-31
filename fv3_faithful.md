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
EIGEN-ANALYSIS (iter115b-123, finite-diff power iteration — jvp CPU-prohibitive; codex-challenged iter123):
- **Genuine growing mode, per-step λ≈1.007 ± 0.001 — CONDITIONING-VALIDATED (iter123, answers codex
  [high]):** robust across eps∈{1e-2,1e-3} × K∈{30,50,70} × 4 random restarts (range 1.0058–1.0072) ⇒
  NOT a finite-diff/float32 artifact (eps/restart-stable, real eigenvalue >1). CAVEAT: residual
  ‖Mv−λv‖/‖λv‖≈3–5% (best: eps=1e-2, K≥50) ⇒ the DOMINANT mode in a near-unit CLUSTER (W2 has many
  neutral λ≈1 modes), NOT a cleanly isolated 6-figure eigenvalue; eps=1e-4 too small (roundoff → resid
  0.45). State as a BAND (1.007±0.001), never "1.0072".
- **TYPE = HYPOTHESIS, not rigorously established (codex [medium]):** edge-localized alternating-in-j
  component in u_d (argmax face3 i=0 row, sign `-+-+-+` in j; 2D-checkerboard corr 0.10 argues against a
  GLOBAL 2D checkerboard — but does NOT prove a 1D mode). Full modal ID (1D Fourier along the edge, phase
  across all equivalent cube edges, per-term linear-operator projection) NOT done ⇒ "2Δx-along-edge
  computational mode" is the LEADING hypothesis only; could be a non-normal edge-localized transient.
- **DAMPING doesn't kill it** (harness: damp_v×8→NaN; nord_v/hyperdiff neutral) and every COMPONENT reads
  FV3-faithful (d_sw zeta 114, vort-flux sina_u 110b, corner-vort halos 94/107, fv_tp_2d halo 115, B-grid
  KE rsin2 118, dissipation/Phase4/wind-halo 82-83; assembly reads correct 119; d_sw4/d_sw5 duogrid
  no-ops 120). LEADING HYPOTHESIS (codex [medium] — "not dampable ⇒ structural" is NOT a sound inference
  on its own): a structural cube-EDGE c_sw→d_sw coupling the FB port misses (production co-located scheme
  kills the edge mode via the D→A avg).
- **FALSIFICATION BATTERY (iter124, answers codex [medium] #3 — float32 & dt stability-limit RULED OUT):**
  (a) float64 — λ=1.0067 PERSISTS in genuine float64 (FB output dtype=float64) ⇒ NOT a float32-roundoff
  artifact. (b) dt-scaling at fixed window K·dt=15000s, C36: dt=600/300/150 → λ−1=1.43e-2/6.64e-3/3.28e-3
  (HALVES with dt; ratios 0.46/0.49) ⇒ (λ−1) ∝ dt, and σ=(λ−1)/dt CONVERGES to ~1.89/day (2.06→1.91→1.89).
  This is the signature of a REAL continuous growth rate (λ=e^{σΔt}≈1+σΔt), NOT a CFL/timestep limit (which
  would STABILIZE, σ→0, as dt drops). ⇒ a genuine growing mode of the FB SPATIAL discretization, σ≈1.9/day
  (e-fold ~0.5 day), precision- AND timestep-independent. The "structural cube-edge coupling" is now the
  WELL-SUPPORTED leading hypothesis (a real spatial-operator mode, not a numerical artifact). STILL OPEN:
  resolution scaling (grid-scale vs large-scale) + the modal-ID type-proof + a direct oracle W2-C36 compare.
- **HARNESS (conditioned, iter123):** finite-diff power iteration WITH the eps/K/restart sweep + residual
  → one diagnostic reported as a BAND, NOT a hard λ<1 gate. NEXT (dedicated/fresh): (a) resolution scaling
  (C24/C48) + oracle W2-C36 step compare to finish the typing; (b) THEN localize the i=0-edge d_sw term
  (ke_diff_u vs fy_vort) → match the FV3 edge coupling → re-test vs the λ band. Separate from the
  production C96 W5 eigenmode (slow ~1.0007). Reading/component/metric avenues exhausted.

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
