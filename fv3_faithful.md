# FV3-faithful cubed-sphere — change log (shrunk @ iter113; prior detail in git + memory `cube-fv3-faithfulness-state`)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
≈ MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge artifacts**, visual + quantitative,
codex/oracle-reviewed. CPU only (Metal broken). **Directive: never A-grid; be FV3-faithful; use the
Fortran as oracle; do NOT improvise.** Branch `latlon-fv-amip-verify`.

## ✅ VERIFIED FAITHFUL (codex-vetted iter109; close to MPAS/latlon across the feasible scope)
- **area_corner (#faces)-junction SCALING** (iter84-91, codex-APPROVED, NET-ZERO): edges ×2, vertices
  ×3, C1 guard n>=1 (oracle fv_grid_tools.F90 edge=2*get_area, vertex=3*get_area; replaces the iter-670
  interior-copy). W2 L2=1.76e-4 unchanged, ocean 9/9 rest machine-zero, 4 SW gold fingerprints re-pinned
  (da_min_c now the correct smaller vertex area). CAVEAT: `sg_area` is PLANAR CHORD — only the ×2/×3
  SCALING is faithful; chord→spherical is a tracked follow-up (trips the gold-fingerprint surface).
- **gnomonic_ed grid** WIRED + halo-collapse fixed (iter73) + codex-approved; equiangular byte-identical.
- **atm SW cross-grid** (iter92, 16/16): cube W2 L2 1.76e-4 ≈ latlon 2.67e-4 ≈ ico 9.9e-5 (dynamics
  FAITHFUL). baroclinic clean (C36→C72 ≤1.24×).
- **cosine_bell** (codex-vetted iter109): cube path uses apply_fortran_xppm_boundary=True (FV3-faithful
  boundary) + hord=10; iter-61/62 12-day audit cube/ico=1.4× (cube BEST mass-cons, drift 1.28e-9), 99%
  of residual is panel-INTERIOR bulk PPM-limiter dissipation (NOT edge), resolution-independent ⇒
  faithful transport, the cube PPM-dissipation plateau (cube≈ico), not a bug.
- **3D rest_state_topo "13× artifact" = FLOAT32 precision, not a bug** (iter97-101): the cube 3D PGF is
  EXACTLY well-balanced in float64 (compute_geopotential uses only σ-derived ln_ratio/alpha ⇒
  Φ=phis+per-level-const ⇒ −∇Φ cancels −R_d·T·∇ln_ps; forcing f64 ⇒ (Φ_k−phis) std=0.0). PE runs f32 by
  design; the large Φ≈2.5e5 vs phis≈1.8e4 add loses ~7 digits → spurious ~1e-7 m/s² PGF (7× at edges).
  Optional f32-only fix: f64 geopotential gradient / reference-subtraction (f64-identical).
- **ocean cross-grid dynamics** (iter105/111): geostrophic_adjustment cube 0.0143 ≈ latlon 0.0159 ≈ mpas
  0.0169 m/s (T drift machine-zero); inertia_gravity_wave omega 1.09e-4 IDENTICAL all 3 grids, cube L2
  best. ocean cube matrix 9/9, rest machine-zero. ⇒ ocean cube ≈ MPAS/latlon (OMIP).

## ❗ THE ONE REMAINING GAP — SW upwind-vorticity FB PORT BUG (confirmed mismatch; root unproven)
CONFIRMED FV3 MISMATCH (codex iter109): production SW (operators_cdgrid.py:1167) + 3D PE
(primitive_eq_cdgrid.py:445) use CENTERED `zeta_corner*v_d`; FV3 sw_core.F90:416-480 upwind-selects the
transported absolute vorticity (donor-cell). Leading suspect for the C96 W5 eigenmode (38→79,
mass-conserving, dt-independent) + the W2 v-imprint (0.344 m/s — real, NOT float32).
FAITHFUL PATH = FV3's staggered c_sw→d_sw FB scheme (uses the UPWIND `_vorticity_flux`). The FB chain
(`fv3_fb_sw_step`/`FV3FBShallowWaterModel`) ports it (Phase4 one_grad_p PGF + duogrid done, 8.5×) but
has a RESIDUAL: W2 C36 day1 max|u_d|=48.6 (vs 38.6), day2 NaN.
KEY (iter109-112): FV3-upwind (oracle) is STABLE ⇒ the residual is a FINDABLE PORT BUG, not a scheme
flaw. It is a GROWTH-RATE eigenmode (iter79/108b) ⇒ single-step probes can rule out but NOT pinpoint it.
RULED OUT (single-step): dissipation(82), Phase4(81), wind-halo(83), corner-vort metric halo(94),
corner-vort circulation/uc-vc reconstruction halo(107: faithful d2a2c halo NEUTRAL/worse), `_vorticity
_flux` vertex metric sina_u(110b: not degenerate). FIELD localization (iter112): the mode is a
ROTATIONAL WIND/vorticity vertex mode (|Δu|~43%, |Δv|~54% vs |Δh|~8%), faces 0/4, h follows — NOT
gravity-wave. ⇒ GENUINE METHOD = linearized-FB EIGEN-ANALYSIS focused on the wind/vorticity coupling at
the face-0/4 vertices (OR a careful FV3-oracle re-port of the c_sw/d_sw vorticity) — a DEDICATED/
fresh-context effort (autonomous-loop single-step probes exhausted; the codex ablation is confounded
because the upwind FB port also has the residual).

## CPU-PROHIBITIVE here (rely on recorded audits + representative cross-grid + t=0 probes)
3D atm baroclinic cross-grid (C36×L40, >21 min/case); full ocean matrix --grid all (57 cases × grids,
multi-hour). climate/AMIP equilibrium.

## OPEN QUEUE
1. SW FB port-bug eigenmode — linearized-FB eigen-analysis (target scoped: wind/vorticity, faces 0/4)
   OR oracle re-port → FB→production (gated, eigenmode-validated). The genuine remaining faithfulness gap.
2. (optional, low-pri) chord→spherical sg_area (gold regen); f64/well-balanced PGF (f32 rest-state);
   3D PE upwind PPM vort; FV3 sin_sg(5) seam rotation.
3. human visual PNG verdict (W2 v / W5 wind_speed / cross-grid atm+ocean — surfaced); pre-existing
   17-test swamp in test_cdgrid_fv3_regression.py (deleted results/ + concurrent churn).
State persisted: memory `cube-fv3-faithfulness-state` (codex-vetted, iter89-112).


## iter114 — d_sw zeta is FAITHFUL (corrects iter83); ruled out as the FB residual root
Investigated iter83's lead ("port d_sw zeta uses covariant u_d·dx with NO corner correction; oracle
d_sw5 wk uses contravariant ut/vt + fill_corners"). READING THE ORACLE (sw_core.F90:1584-1596):
  vt(i,j)=u(i,j)*dx(i,j); ut(i,j)=v(i,j)*dy(i,j)   ← COVARIANT circulation (NOT contravariant)
  wk(i,j)=rarea*(vt(i,j)-vt(i,j+1)-ut(i,j)+ut(i+1,j))   ← "volume-mean" relative vorticity
The port (_d_sw_native:1893-1898: vt_circ=u_d*dx_u, ut_circ=v_d*dy_v, zeta=rarea*(vt_circ[j]-vt_circ[j+1]
+ut_circ[i+1]-ut_circ[i])) MATCHES this EXACTLY (same covariant curl, same indices). And fill_corners
is NOT applied to wk in the oracle (only to divg_d at 1746/1754 and vc/uc at 1762). ⇒ iter83's claim was
WRONG on both counts; the d_sw RELATIVE VORTICITY is FV3-faithful. RULED OUT as the residual root.
Then the d_sw transports vort=zeta+f via fv_tp_2d(hord_vt) (port Step 7, line 1931 = oracle line 1861) —
matching. Remaining FB-residual candidates: the fv_tp_2d CUBE-CORNER halo of vort (oracle computes vort
over isd:ied extended w/ f0 halo'd; port relies on fv_tp_2d's internal halo — the f-part cross-face at
the cube corners is the subtle open item), the B-grid KE transport (`_bgrid_ke_transport`), the
c_sw→d_sw coupling. Still a growth-rate eigenmode (eigen-analysis the genuine method). This iteration
CORRECTED a stale wrong lead (iter83) + ruled out the d_sw zeta — net narrowing.


## iter115 — fv_tp_2d vort halo FAITHFUL too ⇒ ALL readable FB components faithful; residual is in the COUPLING
Checked the last iter114 candidate: does fv_tp_2d cross-face-halo the transported zeta_abs at the cube
corners?  YES — fv_tp_2d (fv_tp_2d.py:923) uses the duogrid halo (`dg=grid.duogrid; halo_dg=dg if
use_duogrid else None`, ng=3 active in the FB chain) ⇒ the vort transport halo is FV3-faithful, NOT
edge-mode.  RULED OUT.
CAPSTONE of the FB-residual investigation: EVERY individually-readable/testable FB component is now
verified FV3-FAITHFUL — d_sw zeta (iter114), vorticity-flux metric sina_u (iter110b), corner-vort
metric (iter94) + circulation/reconstruction halo (iter107), fv_tp_2d vort halo (iter115),
dissipation/Phase4/wind-halo (iter82-83).  Yet the W2 C36 residual (day1 48.6, day2 NaN) + the C96 W5
eigenmode persist as a vertex-localized WIND/vorticity GROWTH-RATE mode (iter112: faces 0/4).
⇒ DEFINITIVE CONCLUSION: the bug is NOT in any single component but in their COUPLING/ASSEMBLY (the
c_sw→p_grad_c→d_sw forward-backward sequencing, the KE-vorticity-PGF balance at the vertices, or a
sign/time-level subtlety) — a coupled growth mode that ONLY a linearized-FB eigen/singular-vector
analysis can isolate (JAX jvp/vjp of FB^K about steady W2 → the optimal growing perturbation localizes
the unstable coupling).  This is a dedicated research-level debug, best in a FRESH context (my context
is now ~26 iterations large).  The single-component-read avenue is DEFINITIVELY exhausted.


## iter115b — EIGEN-ANALYSIS DONE (finite-diff power iteration): FB blow-up is MARGINAL-mode forcing amp, not a fast eigenmode
jvp eigen-analysis was CPU-prohibitive (AD graph through 8 FB steps; 32 min, killed). FINITE-DIFFERENCE
power iteration (Mδ ≈ [FB^8(s0+εδ)-FB^8(s0)]/ε, 20 iters, C36 W2) WORKED:
  dominant λ_8step: 0.635→0.949→0.970→0.982→0.989 (rising, per-step 0.945→0.998), converging to λ≈1.0
  (MARGINAL — neither fast-growing nor strongly-decaying). Eigenvector: WIND mode (|u|≈|v|), vertex
  ratio 1.8× (mild), per-face mass ~uniform [0.19,0.18,0.19,0.19,0.18,0.07] across faces 0-4 (face 5
  low) — NOT sharply face-4-localized.
⇒ REFRAME: the FB W2 blow-up (day2 NaN) is NOT a clean linear growing eigenmode (dominant λ≈1.0,
marginally stable). It is the near-marginal (I-M)^-1 ≈ 1/(1-0.998) ≈ 500× AMPLIFICATION of the CONSTANT
single-step FORCING r = FB(W2)-W2 (the c_sw+p_grad_c+d_sw ASSEMBLY inconsistency at the vertices; r is
what iter95/108 measured as the vertex-localized deviation). The near-marginal damping is FV3-like
CORRECT (FV3 is ~non-dissipative for balanced flow); a tiny assembly inconsistency r is amplified ~500×
→ the offset → nonlinear NaN.
⇒ FIX TARGET (now clear): REDUCE the single-step assembly residual r = FB(W2)-W2 at the vertices — the
non-cancellation of (c_sw vortflux+KE) + p_grad_c PGF + d_sw, NOT any single component (all faithful,
iter114/115) and NOT the damping (marginal is correct). Since every component is individually faithful,
r is an ASSEMBLY/sequencing subtlety (FB time-level ordering, the half-step vs full-step staggering, or
a sign at the vertex). NEXT (dedicated): compute the full r=FB(W2)-W2 + decompose by FB STAGE
(c_sw vs p_grad_c vs d_sw contribution to r at the vertices) to find which stage's assembly injects r.
Eigen-analysis avenue is now FEASIBLE (finite-diff, ~min); the marginal-amplification reframe is the
key new understanding.
