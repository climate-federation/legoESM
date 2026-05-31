# FV3-faithful cubed-sphere — change log (shrunk @ iter103; prior detail in git + memory `cube-fv3-faithfulness-state`)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`),
≈ MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge artifacts**, visual + quantitative,
codex/oracle-reviewed. CPU only (Metal broken). **Directive: never A-grid; be FV3-faithful; use the
Fortran as oracle; do NOT improvise.** Branch `latlon-fv-amip-verify`.

## ✅ FAITHFUL / VALIDATED / RESOLVED
- area_corner (#faces)-junction SCALING (iter84-91, codex-APPROVED, NET-ZERO regression): edges ×2,
  vertices ×3, C1 guard n>=1 (oracle fv_grid_tools.F90 edge=2*get_area, vertex=3*get_area). Replaces
  the iter-670 interior-copy (which mirrored FV3's HALO extrap, not in-domain grid_area). W2 L2=1.76e-4
  unchanged, ocean 9/9 rest machine-zero, 4 SW gold fingerprints re-pinned (da_min_c=min(1/rarea_c) now
  = the correct smaller vertex area). CAVEAT: `sg_area` is PLANAR CHORD — only the ×2/×3 SCALING is
  faithful; chord→spherical is a tracked follow-up (trips the gold-fingerprint surface).
- gnomonic_ed grid WIRED + halo-collapse fixed (iter73) + codex-approved; equiangular byte-identical.
- cross-grid SW (iter92, 16/16 all grids): cube W2 L2 1.76e-4 ≈ latlon 2.67e-4 ≈ ico 9.9e-5 (dynamics
  FAITHFUL). cosine_bell cube 0.131 vs latlon 0.025 (~5×, ~1.8× MPAS) = inherent cube advection cost
  (faithful fv_tp_2d). ocean cube matrix 9/9, rest machine-zero. baroclinic clean (C36→C72 ≤1.24×).
- **3D rest_state_topo "13× edge artifact" = FLOAT32 precision, NOT a faithfulness bug** (iter97-101):
  the cube 3D hydrostatic PGF is EXACTLY well-balanced in float64 (compute_geopotential uses only
  σ-derived ln_ratio/alpha ⇒ Φ=phis+per-level-const ⇒ −∇Φ cancels −R_d·T·∇ln_ps to machine zero;
  forcing f64 ⇒ (Φ_k−phis) std=0.0). PE runs FLOAT32 by design; the large Φ≈2.5e5 vs phis≈1.8e4 add
  loses ~7 digits → spurious ~1e-7 m/s² PGF (7× at panel edges). Optional f32-only fix: f64 geopotential
  gradient / reference-subtraction well-balanced PGF (f64-identical). Matrix audit comment corrected.

## ❗ THE ONE GENUINE REMAINING FAITHFULNESS GAP — SW co-located CENTERED vorticity
Both production SW (operators_cdgrid.py:1167) AND 3D PE (primitive_eq_cdgrid.py:445) use CENTERED
`zeta_corner*v_d` vorticity advection (energy-conserving but dispersive), NOT FV3's UPWIND staggered
donor-cell flux. Root of: the C96 W5 vertex eigenmode (38→79, mass-conserving, dt-independent) AND the
W2 v-imprint (0.344 m/s — a REAL discretization effect, NOT float32: f32-of-38 ~ 4e-6). iter87: faithful
vertex area/grid make the vertex vorticity MORE accurate → EXPOSE the mode MORE ⇒ the fix is the UPWIND
ADVECTION (implicit dissipation), NOT vorticity accuracy or more explicit damping (sweeps ruled out
iter82-83).

The faithful path = FV3's staggered c_sw→d_sw FB scheme (which uses the UPWIND `_vorticity_flux`,
fv3_sw_core.py:1260, sw_core.F90:416-480: fy1=(v_d-uc*cosa_u)/sina_u contravariant, vort_x=upwind).
The FB chain (`fv3_fb_sw_step`/`FV3FBShallowWaterModel`) ports it; fixes done: D-grid backward PGF
(Phase4 one_grad_p) + duogrid (8.5× stability). RESIDUAL: W2 C36 day1 max|u_d|=48.6 (vs 38.6), day2 NaN,
VERTEX-localized from step 1 (iter95: 5-7× edge), isolated to the `_corner_vorticity` duogrid uc/vc
RECONSTRUCTION at the 8 vertices (NOT the metric — iter94 ruled out: edge-copy vs linear-extrap dxc/dyc
<0.2% Δ).

### BLUEPRINT — staggered C-grid vector halo (the infra blocker; oracle-grounded iter103)
FV3 c_sw (sw_core.F90:374-408): vort = fx(i,j-1)−fx(i,j)−fy(i-1,j)+fy(i,j), fx=uc·dxc (needs j-halo
rows js-1/je+1), fy=vc·dyc (needs i-halo rows is-1/ie+1). For DUOGRID the corner correction is SKIPPED
(line 395) and FV3 relies on uc/vc being halo'd AT THE EDGES (mpp_update CGRID with vector rotation).
legoESM's `_corner_vorticity` instead does a lossy uc→cell-center(2-pt avg)→pad_halo_vector→re-stagger
detour (FORCED by `pad_halo_vector` being A-grid/cell-centred ONLY) → smears the 3-face-vertex
circulation → the residual.
FIX: a NEW `pad_halo_cgrid_vector` that rotates uc (i-edge, j-halo) + vc (j-edge, i-halo) across cube
seams at their NATIVE edge staggering (the CGRID_NE rotation: at a seam the i-component maps to the
neighbour's ±j-component etc.), reusing the duogrid cross-face mapping + the cos_theta/sin_theta
rotation that `pad_halo_vector` already has for cell-centred vectors. Then fx_halo=uc_halo·dxc_halo
directly (no reconstruction). Validate: FB W2 C36 residual (day1 48.6→? toward 38.6, day2 NaN→?) +
re-pin the corner_vorticity fingerprint. Then FB-chain→production (gated, eigenmode-validated) → the
upwind `_vorticity_flux` damps the eigenmode.
RISK/STATUS: DEFERRED MAJOR infra (per memory: "sanctioned major-effort, needs a proper validation
budget, NOT a safe loop-iteration hack" — concurrent session does destructive git reset). Build with
the oracle, validate incrementally, codex-review.

## OPEN QUEUE (priority)
1. Staggered C-grid vector halo → fix `_corner_vorticity` reconstruction → FB residual → FB→production
   (the genuine SW faithfulness gap; major, oracle-grounded blueprint above).
2. (optional) chord→spherical sg_area (gold regen); f64/well-balanced PGF (f32 rest-state).
3. 3D PE upwind PPM vort (same upwind theme as #1, 3D). thread FV3 sin_sg(5) seam rotation.
4. human visual PNG verdict (W2 v / W5 wind_speed / cross-grid — surfaced). pre-existing 17-test swamp.
5. ocean/AMIP cross-grid (CPU-infeasible to block on; rely on recorded audits + t=0 probes).
State persisted: memory `cube-fv3-faithfulness-state` (full iter89-102 resolution).


## iter104 — staggered C-grid halo blueprint SHARPENED: it needs uc/vc on the duogrid extension
Read `pad_halo_vector` (halo.py:2065-2134): the cross-face vector rotation works by (u_grid,v_grid)→
(u_east,v_north) geographic→pad-as-scalars→back.  CRUCIAL: this needs BOTH components AT ONE POINT —
but C-grid uc (i-edges) and vc (j-edges) only COEXIST at cell centres.  That is precisely WHY
`_corner_vorticity` detours uc/vc through cell-centres (2-pt avg) before the rotation; the lossy
averaging is INHERENT to co-locating staggered components.  ⇒ a clean staggered C-grid halo cannot
just be a new halo function — it requires uc/vc available on the DUOGRID EXTENSION directly (compute
the C-grid winds via `d2a2c` ON the extended grid so the halo edges carry true cross-face uc/vc, no
co-location averaging).  That is a d2a2c/dycore structural change = the "deferred major algorithmic
upgrade" the project flags as needing a dedicated validation budget (NOT a safe autonomous-loop hack;
concurrent session does destructive git reset).  Confirms the FB-residual fix is structural, not a
bounded halo.  The faithful path (staggered c_sw→d_sw with extended-grid uc/vc + upwind _vorticity_
flux) stands as the genuine remaining SW faithfulness item, blueprinted + oracle-grounded, awaiting a
dedicated build.


## iter105 — OCEAN cross-grid (OMIP-relevant): cube dynamics faithful, close to MPAS/latlon
geostrophic_adjustment (ocean balanced-flow test, analog of W2) cross-grid quick, 3/3 PASS:
  max_speed_final: cube C24 0.0143 m/s | latlon 36×72 0.0159 | mpas ico3 0.0169  (all ~0.015, within
  18%; cube LOWEST = best balance).  T drift machine-zero all 3 (7.5e-14 / 1.3e-14 / 2.6e-15).
⇒ the ocean CUBE dynamics is FAITHFUL + CLOSE TO MPAS/latlon for OMIP-relevant geostrophic balance
(complements the iter89 ocean cube matrix 9/9 + rest machine-zero).  Cross-grid eta/SST PNGs surfaced
for the human visual verdict.  Confirms the directive's "close to MPAS and lat-lon" holds for ocean
dynamics, as it does for atm SW dynamics (W2 cube≈latlon≈ico).  The residual SW-vorticity gap is
atm-SW-core-specific (the co-located centered vorticity); ocean barotropic uses the gated FV3 fv3sw
path, not the same centered-vorticity production SW.
