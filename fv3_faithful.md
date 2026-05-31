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

## ❗ THE ONE REMAINING GAP — SW FB-port edge instability = WEAKLY-UNSTABLE spectrum (ρ≈1.0019) + NON-NORMAL TRANSIENT growth (iter128 Arnoldi; fix pending)
CONFIRMED FV3 MISMATCH (codex iter109): production SW (operators_cdgrid.py:1167) + 3D PE
(primitive_eq_cdgrid.py:445) use CENTERED `zeta_corner*v_d`; FV3 sw_core.F90 upwind-selects (donor-cell).
Faithful path = the FB staggered c_sw→d_sw scheme (uses upwind `_vorticity_flux`); the FB chain has a
RESIDUAL (W2 C36 day1 48.6 vs 38.6, day2 NaN).
EIGEN-ANALYSIS (iter115b-123, finite-diff power iteration — jvp CPU-prohibitive; codex-challenged iter123):
- **⚠ COMPOSITE TRUTH (iter128 Arnoldi, see the iter128 bullet below — supersedes the iter127 "stable
  spectrum" reframe):** the FB scheme is GENUINELY but WEAKLY UNSTABLE — Arnoldi ρ(M')≈1.0019>1 (real
  eigenvalue cluster) — WITH strong NON-NORMAL TRANSIENT growth on top (K-window optimal ~1.007/step ≫
  the 1.0019 eigenvalue). The quoted "per-step λ≈1.007" is the TRANSIENT rate, NOT the eigenvalue (1.0019);
  the single-step 0.9997 was the transient singular vector, masking the weakly-projected unstable eigenvector.
- **Growth amplitude robust, per-step λ≈1.007 ± 0.001 — CONDITIONING-VALIDATED (iter123, answers codex
  [high]):** robust across eps∈{1e-2,1e-3} × K∈{30,50,70} × 4 random restarts (range 1.0058–1.0072) ⇒
  NOT a finite-diff/float32 artifact (eps/restart-stable). The K-DEPENDENCE (1.0072@K50 → 1.0065@K70)
  was the early hint it's a transient (window-peaking) rate, not a K-independent eigenvalue (iter127
  confirmed). State as a BAND (1.007±0.001), never "1.0072".
- **TYPE = 2Δx GRID-SCALE EDGE MODE — PROVEN (iter125, resolves codex [medium] #2):** modal-ID across
  C24/C36/C48 (eigenvector du): argmax ALWAYS on a panel-edge row (j0, face 2/3); 1D Fourier along the
  dominant edge line peaks at the NYQUIST (2Δx) wavenumber (power frac 0.79/0.91/0.88) with sign-flip
  frac 0.83–0.89; edge-ring energy fraction 0.567→0.908→0.997 (→1.0 as resolution refines, strongly
  edge-localized). ⇒ a 2Δx grid-scale COMPUTATIONAL MODE localized at the cube panel edge — NOT a 2D
  checkerboard, large-scale, or non-normal transient.
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
  (e-fold ~0.5 day), precision- AND timestep-independent.
  RESOLUTION SCALING (iter125): σ GROWS with resolution (C24/C36/C48 → 0.43/2.01/3.33 /day) ⇒ a GRID-SCALE
  instability (a marginal-RESOLUTION stability limit would IMPROVE/converge at finer Δx, not worsen) ⇒
  the last stability-limit alternative is ruled out too. NET (iter123-125): the FB mode is a REAL,
  precision/dt/resolution-robust, 2Δx grid-scale, edge-localized growing mode of the FB SPATIAL operator
  — a structural cube-PANEL-EDGE computational mode (production co-located scheme suppresses it via the
  D→A avg; the FB staggered port does not). All 3 codex findings resolved.
- **HARNESS (conditioned, iter123):** finite-diff power iteration WITH the eps/K/restart sweep + residual
  → one diagnostic reported as a BAND.
- **iter126→127 LOCALIZATION → the growth is NON-NORMAL TRANSIENT, not an unstable eigenvalue:** built +
  VALIDATED a single-step FB replication (calls _c_sw/_p_grad_c/_d_sw_native internals; matches model.step
  to float32-eps; runs float64). iter126 (wind-only, fixed-h) Rayleigh decomp did NOT close (wrong sign).
  iter127 FIX — full (h,u,v) coupled eigenvector + energy-norm (√g·η,√H̄·u,√H̄·v) 5-term decomposition:
  - **iter128 ARNOLDI RESOLUTION (genuine eigs on the linearized 1-step M' — the DEFINITIVE answer; codex
    [high] was right AGAIN):** spectral radius **ρ(M') ≈ 1.0019 > 1** — a CLUSTER of REAL unstable
    eigenvalues (~1.0015–1.0019; LM & LR agree). So the spectrum is GENUINELY UNSTABLE; my iter127
    "asymptotically stable (radius ≤1)" was WRONG — the unstable eigenvalue is real but WEAKLY-PROJECTED,
    masked by the single-step measurement (0.9997) on the dominant TRANSIENT singular vector (so "w is a
    near-eigenvector" was also misleading — 1.1% residual ≫ the 3e-4 margin; the unstable eigenvectors are
    different). **COMPOSITE TRUTH (reconciles iter123-127):** a WEAKLY UNSTABLE spectrum (ρ≈1.0019,
    σ_asymp≈0.5/day, e-fold ~1.8 d) WITH strong NON-NORMAL TRANSIENT amplification layered on top
    (K=50-window optimal ~1.0071/step ≈ ×1.4 over a few hours ≫ the 1.0019 eigenvalue). iter123-125's
    "growing mode" is REAL (radius>1) but the eigenvalue is 1.0019 not 1.007; iter127's transient growth is
    REAL and dominates EARLY but the spectrum is not stable. K-trend λK 1.0071→1.0031 = the optimal-window
    rate relaxing from the transient peak toward ρ (still >1).
  - **Mode energy:** 91% v_d, 8% u_d, 1% h. Single-step term balance (decomp CLOSES, Σ=-3.95e-4 ≈ λ1-1
    =-3.38e-4): **KE-grad +1.47e-3 (Bernoulli SOURCE) ↔ PGF -1.72e-3 (SINK) dominate and nearly cancel**;
    VORT -1.9e-4, DAMP ~0, MASS +5e-5. ⇒ a near-neutral gravity-wave KE↔PGF exchange at the panel edge.
  - **NaN MECHANISM = HYPOTHESIS (codex [medium], not demonstrated):** transient ×1.4 + the 1.0019
    eigenvalue → nonlinear breakdown by day2 is PLAUSIBLE but not causally shown; would need projection
    tracking onto the growth subspace during the nonlinear run + showing a fix delays/removes the NaN.
  - **FIX STRATEGY (codex [medium] — testable, NOT pre-excluding damping):** evaluate candidate fixes by
    BOTH ρ(M') (Arnoldi) AND max_K‖M^K‖ (transient gain); include targeted edge/Nyquist filtering AND the
    edge c_sw→d_sw coupling change in the comparison (the damp_v×8 null was ONE config, not damping as a
    class). Validated replication /tmp/fb_arnoldi.py (eigs) + /tmp/fb_localize2.py (decomp) are the assets.
    NEXT (dedicated/fresh): candidate edge-coupling fix → re-run Arnoldi (does ρ drop <1?) + transient gain
    + oracle W2-C36 step compare. Separate from the production C96 W5 eigenmode (slow ~1.0007).

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
