# FV3-faithful cubed-sphere — change log (shrunk @ iter ~67)

Goal: cube faithful to GFDL FV3 (oracle `../Code/FV3/atmos_cubed_sphere-symmetryclean/`,
70 F90), matching MPAS/ico + lat-lon FV across SW→AMIP/OMIP, **zero cube edge
artifacts**, visual+quantitative, codex/oracle-reviewed. CPU only (Metal broken).
**User directive: never A-grid; be FV3-faithful; use the Fortran as oracle; do NOT
improvise.** Branch `latlon-fv-amip-verify`, ~90 commits ahead. Per-iter detail in git.

## ✅ FIXES (done, validated, codex-reviewed)
- ATM panel-edge imprint: cc→D-grid lift (u,v) as scalars (7 sites). baroclinic v_rms
  4.95→0.60, mass→1e-15. Test `test_vector_cc_to_dgrid_wind_lift.py`.
- CROSS-GRID alignment: canonical `_canvas_lat/_canvas_lon` (fixed translation + node/cell
  drift; unblocked latlon SW). Test `test_latlon_regrid_alignment.py`.
- OCEAN geostrophic artifact 40%→1.45%: cube barotropic A-grid → FV3 SW core `fv3sw`
  (never A-grid). T drift 7.5e-14, max_speed≈latlon/mpas. + FC viscosity vector-halo.
  Faithful upwind option `fv3edge` (gated, 8/9). Tests `test_fv3{sw,edge}_barotropic.py`.
- [iter57] Stale iter-707 KE-fingerprint test: places=4 abs → rtol=1e-6 L2-scaled.
- [iter62] `hord==9` iv mislabel: scalar `_ppm_1d` hord=9 used iv=1; FV3 iord=9=iv=0
  (tp_core.F90:610) → `_pert_ppm_iv0`. Unexercised (live uses hord=12 iv=0; momentum
  ytp_v/xtp_u uses iv=1 correctly). Cube SW 4/4 unchanged. `test_hord9_iv0_scalar`.

## EDGE-ARTIFACT TRUTH (iter57-61; earlier "PROVEN clean" was OVERSTATED)
Reliable metric = `compute_cross_face_continuity` (halo.py): cross-seam first-diff / same-
face interior first-diff via real `pad_halo_4d` (≈1 continuous, 5-50× = real jump).
SUPERSEDES pooled same-face `compute_edge_artifact_metric` (amplifies curvature, unreliable
both ways: gave 5.95× on W5 v where true continuity is 1.30×). Codex caught + fixed a
denominator bug (N/E halo row leaked into `gi`); re-review APPROVED. Validated vs the REAL
C96 eigenmode (tracks onset 1.57→3.38). Tests `test_cross_face_continuity_metric.py` (4/4).
- BAROCLINIC (3D PE): genuinely clean, converges C36→C72 (≤1.24× incl rotated).
- SHALLOW-WATER: all PHYSICAL fields CONTINUOUS — C36/C48/C96 FLAT (W2 u 1.57→1.55; W5 v
  1.30, wind_speed 1.58). No large edge artifact. (W2 v 2.23× is the sub-0.34 m/s error.)
- **OPEN — C96 W5 edge eigenmode:** W5 max|wind| tracks C36/C48 (~38) to day3.5 then BLOWS
  UP 38→79 (edge/corner faces 0,4). CFL REFUTED (dt=300/150/100 all blow → grid-scale
  spatial mode). Diagnosed (iter59-61): NOT damping (dddmp inert in prod RK3; lives only in
  the unstable FB chain) and NOT flux (mass exactly conserved). FB chain (`fv3_fb_sw_step`)
  has the SAME mode: mass-conserving h-drift piling at i=0 edge. Every exposed knob
  (dddmp/xppm/d_sw5_corner) inert; FB d_sw1 already uses sin_sg/cosa. ⇒ root = the GRID
  (see below) and/or core c_sw/d_sw edge treatment. `hyperdiff=6×` masks it but is
  improvising (rejected). FAITHFUL fix = gnomonic_ed grid (likely) + FB-chain stabilization.
- STILL MISSING: human visual PNG inspection (assistant barred from Read images):
  `results/atmosphere/shallow_water/williamson{2,5}/cubed_sphere/C36/snapshots_{v,wind_speed}_native.png`.

## FAITHFULNESS SCORECARD (12-agent oracle workflow iter57; Fortran-vs-port direct read)
**FAITHFUL but ORPHANED:** gnomonic_ed grid (1 ULP, but create runs equiangular);
get_area/cell_center3 (prod uses equiv l'Huilier rel<1e-6); upwind vort-flux/`fv_tp_2d`
(3D PE doesn't call it).
**MINOR:** `d2a2c_vect` (live duogrid path faithful); `a2b_ord4` (bit-faithful, inert
default); cross-face vector halo `pad_halo_vector_4d` ORTHOGONAL — drops FV3 `1/sin_sg(5)`
→ O(cosθ) seam err; faithful `pad_halo_dgrid_vector_4d` (12/12) gated off.
**MAJOR:** SW core production = single-stage SSP-RK3 (1 Arakawa-Lamb tendency), NOT FV3 FB
c_sw→d_sw (`fv3_fb_sw_step` exists but unstable/unreachable). 3D PE D-grid vort/KE =
centered `zeta_corner*v_d` (primitive_eq_cdgrid:445), NOT FV3 upwind PPM `hord_vt`.

## GRID gap → gnomonic_ed wiring (LIKELY fixes the C96 eigenmode; iter62-66)
create runs EQUIANGULAR (gt=2); FV3 operational = gnomonic_ed (gt=0). **Conditioning C96:
equiangular max corner aspect 1.4027 vs gnomonic_ed 1.0583 (=FV3 1.06089, dx ratio √2)** —
1.33× worse; FV3 uses gnomonic_ed to suppress high-res corner/edge modes ⇒ wiring it likely
fixes the C96 eigenmode (unifies grid gap + eigenmode). Signature locked
`test_fv3_gnomonic_ed_signature_iter62`.
**WIRING — bricks done (verify-first; equiangular default UNTOUCHED; 4 shortcuts rejected:
separable-same-dist→1.40 aspect, naive-θ-ext→moved diagonal, per-face-separable→breaks
seams, must use mirror construction):**
- ✅ `_gnomonic_ed_construct(theta_w, alpha)` — halo-extensible face core, bit-matches
  gnomonic_ed; diagonal pinned to in-domain ±α (interior invariant to halo ext). Tests 4/4.
- ✅ `_gnomonic_ed_remap_to_create` — perm [0,1,3,4,5,2] + D4 rot [0,0,1,1,0,3] mapping
  make_fv3_native_grid's FV3 numbering → create's (derived by matching equiangular both
  sides). SEAM-CONTINUITY crux SOLVED: validated 6 faces at create positions + cross-face
  1.22 through create's halo tables.
- ✅ `_compute_gnomonic_ed_lonlat` (centers, remapped → create-numbered + seam-continuous).
- ✅ centers `_compute_gnomonic_ed_lonlat` = cell_center2 of corners (defn A, FV3 agrid;
  √2 1.31 ratio, seam-continuous, create-numbered). iter67: construct-at-cell-centre-θ
  (defn B) is WRONG — gnomonic_ed's construct is range-dependent (sub-range → 1.80 ratio),
  not parametric like equiangular's tan(α).
- ✅ `compute_padded_{half_metrics,angle}_ed` + `_compute_exact_cell_areas_ed` — ALL reworked
  to defn A (corner-based): `_gnomonic_ed_padded_centers` = cell_center2 of extended corners,
  per-face 2-cell-chord dx + centred-diff angle. In-domain centres match grid.lon/lat to
  1.4e-15 (consistency-lock test). Fixed the 7th subtlety (defn-B was range-dependent/~½-cell
  off). 22/22 gnomonic_ed brick tests. grid.lon/lat + all metrics now share ONE centre defn.
- ✅ `compute_padded_angle_ed` (angle 2-valued: equatorial 0-3 / polar 4-5 differ by π — a
  REAL N/S flip, in equiangular too; within 0.076 rad of equiangular). Tests 5/5.
- ✅ `_compute_exact_cell_areas_ed` (FV3 get_area on remapped corners; total 4πR² exact,
  face-indep; area max/min 2.27 = √2-edge geometry, aspect still 1.06).
- CODEX-REVIEWED (iter67, base 2b17eda7^): the 6 builders (construct/remap/centers/half-
  metrics/angle/area) found SOUND (no bug; codex ran the tests). Sole finding (high) =
  exactly the scoped gap: must NOT gate gnomonic="ed" until `compute_halo_interp_offsets_ed`
  exists (else halo interpolates at equiangular fractional positions = wrong seam geometry)
  + a test exercising the actual `grid.halo_interp_offsets` path (not just nearest pad_halo).
- ✅ `compute_halo_interp_offsets_ed` (halo.py) — position-matching each face's first-halo
  cell to the neighbour strip on the extended defn-A centres (parabola-vertex on great-circle
  dist). gnomonic_ed-specific (codex's gating blocker addressed), bounded (maxabs 0.223 vs
  eq 0.498), helps vs no-offset. Tests 3/3. CAVEAT: sub-cell correction; field tests
  dominated by cross-face linear-interp error so eq-vs-ed optimality is below noise — final
  validation = gated W5 C96 eigenmode run.
- ✅ GATE DONE (iter68): `create_cubed_sphere(gnomonic="ed")` builds the FV3 gnomonic_ed grid
  in production — selects the *_ed builders (centres, padded angle/half-metrics h1/2/3, areas,
  halo-offsets h1/2/3 `_compute_halo_interp_offsets_ed_hN`). Equiangular default BYTE-IDENTICAL
  (cube SW 4/4 unchanged, W2 v_ll 0.339, mass machine-zero). ed grid: area=4πR², dx ratio
  1.40≈√2, cell aspect **1.05≈1.06** (vs equiangular 1.40), all fields finite, rejects
  Schmidt/shift. Gate tests 4/4 (`test_gnomonic_ed_gate_iter68`).
- ⏳ REMAINING for the eigenmode test: the DYNAMICS use `create_cubed_sphere_cdgrid` (C/D
  supergrid, ALSO equiangular-hardcoded `alpha_edges=linspace`) — needs an ed-consistent
  cdgrid before running W5 C96 on the ed grid. Then validate (dynamical suite) + re-calibrate
  + the W5 C96 eigenmode-fix test (the payoff).
- (superseded) earlier `compute_halo_interp_offsets_ed` notes: reuse equiangular
  CONNECTIVITY + `_face_to_xyz_np`/`_xyz_to_gnomonic_np` (pass gnomonic_ed angles since
  tan(arctan(coord))=coord) with `frac=interp(neighbor_angle, alpha_ed, range(n))`. The
  gnomonic_ed gnomonic-angle distribution `alpha_ed` is i↔j-symmetric, spans [-π/4,π/4],
  spacing ratio ~1.31 (in-domain extraction validated, matches corners 2.6e-4). BUG found
  (iter67, verify-first): extracting the angle via construct→mirror→remap→arctan2(y,x)
  SATURATES the halo cells at ±π/4 (the mirror wraps them cross-face) instead of the
  same-face extension BEYOND ±π/4 that the offset's across-edge halo position needs. (The
  angle BUILDER is unaffected — it uses centered differences, robust to this.) ⇒ correct
  approach. ISOLATED (iter67): the CONSTRUCT extends cleanly (halo cells reach 45.02° vs
  in-domain 43.45° — beyond the edge), so the saturation was an EXTRACTION artifact
  (arctan2(y,x) on the mirrored/remapped face), NOT the grid. ⇒ robust offset path =
  POSITION-MATCHING on the cleanly-extending xyz: build the extended 6-face create-numbered
  centers (construct→shift→mirror→remap, halo=1), then for each (face,edge,j) match the
  halo cell's xyz to the neighbour edge-strip cells via great-circle interp → fractional
  index → offset=frac−j. No fragile analytic angle inversion. VALIDATE with cross-face
  continuity USING the offsets (codex's required test) for halo=1/2/3. Then gate
  `create_cubed_sphere(gnomonic="ed")` + validate + re-calibrate + W5 C96 eigenmode test.

## VERIFICATION
- SW 16/16 all grids ≤1 cell. Full fast cube atm dynamical suite PASS + NH dcmip_tc1, mass
  machine-zero. Ocean matrix cube 9/9; rest_state ×12 machine-zero.
- Cross-grid CLOSENESS (W2 height L2 vs exact): cube 1.76e-4 BETWEEN ico 9.9e-5 and latlon
  2.67e-4 (cube > latlon). Regression-clean 44/44. Health 23/24 (1 = collaborator's latlon
  `implicit_cn`, not cube scope).

## ACTION QUEUE
SAFE-BOUNDED: [DONE iter62] hord==9 iv0 + cross-face denominator bug. [TODO] numerical
regressions locking orphaned ports (gnomonic_ed already has signature test; a2b_ord4); human
visual PNG inspection.
FOUNDATIONAL (multi-day; per user no-improvise): finish gnomonic_ed wiring (angle+gate+
validate+recalibrate — IN PROGRESS, likely fixes C96 eigenmode); stabilize+wire FB chain
(FV3 c_sw/d_sw 2-stage + dddmp); 3D PE upwind PPM vort; thread FV3 sin_sg(5) seam rotation;
climate-equilibrium cross-grid CPU-INFEASIBLE here.
State persisted: memory `cube-fv3-faithfulness-state`.
