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
  FAITHFUL). ocean cube matrix 9/9, rest machine-zero. baroclinic clean (C36→C72 ≤1.24×).
  cosine_bell: bulk PPM-limiter dissipation on the cube grid, NOT an edge/boundary bug
  (iter109: codex [high] worried 0.131 was inflated by a non-oracle boundary + mass-fixing; REFUTED by
  the existing iter-61/62 evidence the run uses).  The cube CB path (run_atmosphere_test_matrix.py:2872)
  DOES pass `apply_fortran_xppm_boundary=True` (the FV3-faithful xppm cube-edge boundary), hord=10,
  n_sub=6.  iter-61/62 apples-to-apples 12-day audit: cube L2=0.865 | latlon 0.133 | ico 0.620 |
  spectral 0.382 → cube/latlon 6.5×, cube/ico 1.4×, cube/spectral 2.3× (cube IS the L2 outlier but BEST
  at mass conservation, drift 1.28e-9 ⇒ the rescale is negligible).  iter-61 spatial decomposition
  (_probe_iter61_cb_error_map.py): 99% of the residual lives in panel-INTERIOR cells of the bell's
  face; panel-edge cells contribute ~3e-4 ⇒ bulk PPM limiter dissipation, NOT panel-coupling/edge.
  Resolution-independent (the cube PPM transport-accuracy plateau).  ⇒ faithful transport (oracle
  boundary + fv_tp_2d), error is the cube PPM dissipation characteristic, cube≈ico — defensible.
- **3D rest_state_topo "13× edge artifact" = FLOAT32 precision, NOT a faithfulness bug** (iter97-101):
  the cube 3D hydrostatic PGF is EXACTLY well-balanced in float64 (compute_geopotential uses only
  σ-derived ln_ratio/alpha ⇒ Φ=phis+per-level-const ⇒ −∇Φ cancels −R_d·T·∇ln_ps to machine zero;
  forcing f64 ⇒ (Φ_k−phis) std=0.0). PE runs FLOAT32 by design; the large Φ≈2.5e5 vs phis≈1.8e4 add
  loses ~7 digits → spurious ~1e-7 m/s² PGF (7× at panel edges). Optional f32-only fix: f64 geopotential
  gradient / reference-subtraction well-balanced PGF (f64-identical). Matrix audit comment corrected.

## ❗ THE ONE LIKELY REMAINING FAITHFULNESS GAP — SW co-located CENTERED vorticity (CONFIRMED MISMATCH; root NOT proven)
CONFIRMED FV3 MISMATCH (code+oracle, codex-validated iter109): production SW (operators_cdgrid.py:1167)
AND 3D PE (primitive_eq_cdgrid.py:445) use CENTERED `zeta_corner*v_d` vorticity advection
(energy-conserving but dispersive); FV3 sw_core.F90 upwind-selects the transported absolute vorticity
(donor-cell). This IS a real algorithmic mismatch + the LEADING SUSPECT for the C96 W5 eigenmode
(38→79, mass-conserving, dt-independent) + the W2 v-imprint (0.344 m/s — a real discretization effect,
NOT float32: f32-of-38 ~ 4e-6).
NOT PROVEN as the root (iter109 codex [high], correcting an earlier overclaim): the causal link to the
eigenmode is UNVERIFIED — the `_corner_vorticity` halo hypothesis was tested+reverted (iter107), the
residual is a growth-rate mode (iter79/108b) with eigen-analysis NOT done, and the unstable coupling
(centered-vort vs KE vs d_sw zeta vs c_sw→d_sw corner coupling) is NOT isolated. Replacing the centered
term with the upwind flux MIGHT still fail the growth mode. BLOCK the causal claim until an ablation
(centered→upwind, does the W2 imprint + W5 growth vanish without a new residual?) OR the linearized-FB
eigen-analysis settles it. iter87: faithful vertex area/grid EXPOSE the mode more (⇒ suspect the
advection scheme, not vorticity accuracy; dissipation sweeps ruled out iter82-83).

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


## iter106 — DISCOVERY: FB cross-face uc/vc halo machinery EXISTS + evolved (iter-946/947/948)
Reading fv3_sw_core.py:87-230 (re-examining the FB chain) found the cross-face uc/vc halo problem is
FURTHER ALONG than the iter95-104 blueprint assumed (which said "build a staggered C-grid halo from
scratch"):
  • `_pad_halo_uc_vc_via_d2a2c` (iter-946, line 87): computes the faithful cross-face uc/vc halo via
    d2a2c on the duogrid extension (ng>=3) — NO co-location averaging. NEGATIVE alone (OLD time-level
    → C36 W2 1d |v|=156 vs 81 baseline, OLD/NEW mismatch with the c_sw+p_grad_c increments).
  • `_pad_halo_uc_vc_new_via_old_delta` (iter-947, line ~160): FIXES it — NEW(incremented) anchor +
    OLD cross-face-rotation delta (uc_new = uc[NEW] + (uc_old_jhalo - uc_old_int)). Time-level
    consistent. WIRED into `_d_sw1_recompute_ut_vt` (line 226, when duogrid ng>=3 + u_d_old/v_d_old).
    iter-948: linear-extrap variant worse (75→87); the d2a2c-delta is the kept path.
⇒ the d_sw ut/vt path ALREADY has the time-level-consistent cross-face halo. The c_sw
`_corner_vorticity` (my iter95 vertex-residual focus) STILL uses the lossy center-avg reconstruction.
REVISED FIX (more tractable than build-from-scratch): apply the SAME NEW+OLD-delta cross-face halo
technique to `_corner_vorticity`'s uc/vc (replacing the lossy reconstruction). CAVEAT: c_sw is the
HALF-step (uc/vc BEFORE the c_sw increment), so the "NEW" anchor differs from d_sw1's — need the
c_sw time-level's uc/vc + the OLD-delta. NEXT: read `fv3_fb_sw_step` wiring (does it pass u_d_old to
d_sw1? is the residual measured WITH the iter-947 halo?), then extend the technique to `_corner_vorticity`
+ re-measure the FB W2 C36 residual. This is a BOUNDED extension of existing validated machinery,
not the major from-scratch build — re-prioritizes the genuine SW gap as tractable.


## iter107 — ATTEMPTED + RULED OUT: `_corner_vorticity` cross-face HALO is NOT the FB residual root
Implemented the iter106 bounded fix: routed `_corner_vorticity`'s cross-face circulation halo through
the faithful d2a2c-extended uc/vc (`_pad_halo_uc_vc_new_via_old_delta`, the validated d_sw1 technique)
instead of the lossy cell-centre reconstruction; both c_sw call sites pass u_d/v_d (OLD-derived, ng=3
confirmed). DECISIVE TEST (FB W2 C36, same harness): day1 max|u_d| = 49.71 vs edge-copy/reconstruction
baseline 48.6 (marginally WORSE), day2 NaN. ⇒ the `_corner_vorticity` cross-face HALO (neither the
metric iter94 NOR the circulation/uc-vc reconstruction iter107) is the FB residual root.  REVERTED
(no benefit; FB-only + fingerprint uses the fallback, intact).
This CORRECTS iter95-106: the residual is vertex-LOCALIZED but is NOT in `_corner_vorticity`'s halo.
Re-narrowed to: the vorticity FLUX (`_vorticity_flux` upwind donor-cell selection at the vertex), the
KE (`_ke_upwind`), the d_sw zeta (`_d_sw_native`:1885 covariant circulation), or the c_sw→d_sw
coupling.  NEXT: probe which vertex-localized TERM (flux vs KE vs d_sw zeta) produces the residual —
e.g. zero each in turn (FB-only) and re-measure the day1 vertex deviation.  The genuine SW fix remains
open; the bounded-halo-extension hypothesis is disproven.


## iter108 — c_sw uc-increment decomposition: KE-grad is the most VERTEX-amplified term (new lead)
Decomposed the c_sw half-step uc increment (uc_new = uc + fy1*vort_x + dke_x) for steady W2 C36:
  vort-flux fy1*vort_x: edge(i=0,n) 0.295 vs interior 0.126  → 2.3× edge-enhanced
  KE-grad   dke_x:      edge        0.089 vs interior 0.0091 → 9.7× EDGE-ENHANCED (most vertex-amplified)
The KE gradient (from `_ke_upwind`, the upwind KE at cell centres) is FAR more vertex-concentrated than
the vorticity flux ⇒ NEW candidate for the vertex-localized FB residual: the KE reconstruction at the
3-face junction (`_ke_upwind`).  CAVEAT (not conclusive): the c_sw HALF-step increment is NOT supposed
to cancel alone — the C-grid PGF (`_p_grad_c`, the -∇Φ) balances it in the next FB stage; the true
residual is the non-cancellation of (c_sw vortflux+KE) + p_grad_c PGF + d_sw.  So this localizes a
SUSPECT (vertex KE) but the decisive test is the full c_sw+p_grad_c+d_sw balance, or zeroing/faithful-
ifying the vertex KE and re-measuring.  FB-residual candidates ruled out: dissipation(82), Phase4(81),
wind-halo(83), corner-vort metric halo(94), corner-vort circulation halo(107).  NEW suspect: vertex KE
(`_ke_upwind`).  The FB residual remains a deep, slowly-narrowing structural mode — genuine SW
faithfulness gap, needs continued isolation or a dedicated debugging budget.


## iter108b — META: FB residual is a GROWTH-RATE eigenmode → single-step probes are at their limit
iter79 already established the FB residual is a GROWTH-RATE mode (not a t=0 single-step amplitude:
duogrid has LARGER t=0 dh yet SLOWER growth). So the iter94/107/108 single-step probes (metric halo,
circulation halo, c_sw increment decomposition) can RULE OUT candidates (6 ruled out) but
fundamentally CANNOT pinpoint a growth mode — the iter108 "KE-grad 9.7× edge" lead is a single-step
artifact, not necessarily the growth driver.  The genuine isolation needs EIGEN-ANALYSIS of the
linearized FB step (power-iterate the FB step on a perturbation about steady W2 → the dominant growing
eigenvector localizes the unstable coupling).  That is the proper deep-debug method, flagged since
iter79, and is a dedicated focused effort (not an autonomous-loop single-step probe; my context is
also very large now — fresh-context eigen-analysis would be more effective).
CONSOLIDATED FB STATE: vertex-localized growth-rate eigenmode in the FB c_sw→d_sw scheme; 6 single-step
candidates ruled out (dissipation, Phase4, wind-halo, corner-vort metric+circulation halos);
not-yet-tested via the RIGHT method (eigen-analysis): the c_sw→d_sw corner coupling / vorticity-flux /
KE / d_sw zeta as a COUPLED growing mode.  The cube is comprehensively faithful otherwise (atm SW,
3D PGF f64, ocean dynamics all ≈ MPAS/latlon; area_corner FV3-faithful).  The FB→production upgrade
(the eigenmode fix) needs the eigen-analysis + a dedicated validation budget.
