# FV3 Fortran Fidelity Review (compacted; pre-2026-04-16 history removed)

## CURRENT STATE (post iter-67)

Scope:
- Python: `src/legoesm/grids/cubed_sphere_cdgrid.py`, `src/legoesm/core/fv3_sw_core.py`, `src/legoesm/core/fv_tp_2d.py`, `src/legoesm/atmosphere/dynamics/shallow_water_fv3_cdgrid.py`, `src/legoesm/core/operators_cdgrid.py`
- Fortran oracle (read-only): `../atmos_cubed_sphere-symmetryclean/model/{sw_core,dyn_core,tp_core,fv_grid_utils,fv_arrays}.F90`

### Fidelity status
All FV3 duogrid-branch operator formulas are implemented and verified against the oracle:
- metrics: `cosa_u/rsin_u` from `sin_sg`; supergrid `area_corner/dxc/dyc/rdxa/rdya/divg_u/divg_v`; `rsin_u/rsin_v` mixed 1/sin² interior + 1/sin at non-duogrid panel edges (iter 66); bounded_domain gating covers duogrid AND single-face-panel regional path (iter 66 follow-up #3).
- `c_sw`: D→A→C 4th-order + face-boundary `sin_sg/cos_sg` conversion, corner vorticity, vorticity flux.
- `d2a2c_vect` duogrid: cross-axis D-grid halo via `ext_vector_dgrid`, length-weighted c2l_ord2 (iter 60).
- `d_sw1`-`d_sw6`: adjacent-strip + corner 2x2 solve; PPM hord=9 B-grid KE transport (iter 32/39); `d_sw5` corner divergence damping with `divg_u/divg_v` metrics (iter 51/53); `d_sw6` vorticity damping via `_del6_vt_flux` (iter 52).
- `fv_tp_2d`: Lin-Rood operator split + CGRID flux sync + `pert_ppm(iv=0)` + `pert_ppm(iv=1)` at Fortran interior cells gated on non-duogrid (iter 42/62/63); offset-based dm rescaling and al edge corrections gated on non-duogrid (iter 64); duogrid halo actually routed to `pad_halo` when available (iter 63 follow-up).
- Plumbing: FB wrapper forwards `d2_bg/dddmp/d4_bg/nord/damp_v` and auto-derives `nord_v = min(2, nord)` at step time (iter 62/62 follow-up).

### Unresolved (stopping-condition blockers)
1. **W2 v-wind cube-face imprint at C36** — architectural. Production path `fv3_sw_tendencies` uses Arakawa-Lamb gradient, not FV3's `c_sw/p_grad_c/d_sw` chain. Forward-backward (FB) path IS FV3-faithful but unstable at C36 (first-order upwind in `c_sw` amplifies halo-induced face-boundary divergence).
2. **FB path C36 stability** — requires ng=3-equivalent halo. Iter 58 extended `pad_halo_dgrid` to `halo=2`; further widening requires implementing halo=3 in pad_halo + all downstream stencils. Not a formula fix.
3. **d_sw3 scalar KE sync vs Fortran BGRID_NE component sync** — the Fortran syncs x/y components before forming KE via MPI BGRID_NE vector exchange with cross-axis rotation. Python's scalar KE sync matches the Fortran commented-out alternative. Proper fix requires cross-axis vector rotation infrastructure (naive swap-without-sign attempted iter 36, reverted as rotation-unsafe).
4. **Non-duogrid `_d2a2c_vect` corner 2x2 solve** — 4-point adjacent-strip PORTED (iter 68). Fortran sw_core.F90:739-811 four corner systems remain unimplemented; require halo i-columns / j-rows that Python's interior-only layout does not expose. Affects non-duogrid FB path (experimental, unstable anyway).

### Evaluation metrics (post iter-66 follow-up #3)
- Williamson 2: L2=1.53e-03, Linf=4.07e-03 (C36, 1 day)
- Williamson 5: mass drift=1.42e-05 (C36, 1 day)
- Cosine bell: L1=1.20e-01, L2=1.17e-01, Linf=1.23e-01
- Ocean rest state: all cubed-sphere variants machine-precision
- 124 regression + audit-harness tests pass

---

## Iteration history (2026-04-16 onward)

### iter 31 (2026-04-16) — investigated
- Unconditional flux sync tested and rejected: applying sync to non-duogrid path causes 110x W2 regression (PPM boundary asymmetry destroyed by averaging).
- Boundary vs interior error analysis: W2 boundary error ≤1.38x interior; production at A-L accuracy limit.
- C36 rest state machine-precision; C16 h_err=3.37e-03 at cube vertex corners (symmetric, converges with resolution).

### iter 32 (2026-04-16) — d_sw3 B-grid KE transport ported
- `_bgrid_ke_transport()` in fv3_sw_core.py: B-grid contravariant velocities from cosa_corner/rsin2_corner; operator-split 1D first-order upwind transport; KE = 0.5*(transported_y * vb + ub * transported_x) (Lin-Rood); KE gradient at D-grid edges. Matches sw_core.F90:1201-1388.
- d_sw4 confirmed no-op for duogrid (corner KE fix gated on `.not. duogrid`).

### iter 35 (2026-04-16) — diagnosis
- **FB instability root cause**: c_sw first-order upwind redistributes height because d2a2c_vect produces transport velocities with non-zero face-boundary divergence (halo exchange quality issue at cube vertex corners). FV3 avoids via ng=3+ MPI halo + explicit face/corner handling.
- d_sw6 replacement formula u_new = u_old*dx + ke_diff + fy_vort verified algebraically equivalent to incremental form.
- D-grid vorticity in production path tested and rejected: breaks geostrophic cancellation (3x W2 regression) despite 4x W5 conservation improvement.

### iter 36 (2026-04-16) — BGRID_NE sync + dedup
- `synchronize_bgrid_ne()` in halo.py: syncs x-component (ubb) at WEST/EAST and y-component (vbbtemp) at SOUTH/NORTH boundaries; KE then computed from synced components. Matches dyn_core.F90:969-1011. NOTE: removed in subsequent iters as not rotation-safe at cross-axis seams; see blocker #3.
- Extracted shared helpers `_ke_upwind`, `_corner_vorticity`, `_vorticity_flux` (~120 lines deduped between _c_sw and fv3_csw_tendencies).

### iter 37 (2026-04-16) — duogrid+production stable
- W2 C16 with duogrid+production survives 1 full day (288 steps, dt=300s); tendency dropped 130x to max|dh/dt|=1.2e-3.
- FB instability confirmed: c_sw first-order upwind on cubed sphere creates 13.78 m h_err per step at C16 — fundamental, not flux-sync-fixable.
- All 6 rest state paths (production/CSW/FB × no-DG/DG) machine-precision.

### iter 38 (2026-04-16) — line-by-line oracle trace
Verified against Fortran sw_core.F90:
- d2a2c_vect duogrid: D→A 4th-order (a1=0.5625, a2=-0.0625), covariant→contravariant `(utmp - vtmp*cosa_s)*rsin2`, A→C 4th-order, all boundary overrides correctly gated on `(.not. dg%is_initialized)`.
- c_sw transport scaling `ut = dt2 * ut * dy * sin_sg(upwind)` ✓; KE/vorticity via shared helpers ✓.
- d_sw5 vorticity, d_sw6 wind update verified.
- **Remaining infrastructure gap**: FV3 ng=3 MPI DGRID_NE halo vs Python ng=1 pad_halo_dgrid → ~0.3% transport velocity asymmetry at face boundaries (root of FB c_sw mass error).

### iter 39 (2026-04-16) — d2a2c_vect halo + PPM hord=9
- d2a2c_vect face-boundary uses `pad_halo(sin_sg)` (cross-face halo) instead of clamping to nearest interior cell. Matches sw_core.F90:3589-3607.
- `_ppm_transport_1d()` matches Fortran ytp_v/xtp_u jord>=8 with rdy/rdx CFL scaling.

### iter 40 (2026-04-16) — Courant + xppm/yppm verified
- `crx = (dt*ut) * rdxa(upwind_cell)` with upwind-selected rdxa from mean of adjacent rdxc + pad_halo. Cosine bell L1 0.8% improvement.
- d_sw4 corner KE fix correctly skipped for duogrid; xppm/yppm dxa metric unused for hord>=8 in duogrid path.

### iter 42 (2026-04-16) — fv_tp_2d hord=9
- Upgraded from hord=8 to hord=9. Cosine bell improved 5-7%.

### iter 43 (2026-04-16) — true hord=9 (pert_ppm iv=0)
- Corrected from hord=10 (pmp/lac) to true hord=9 = `pert_ppm(iv=0)` positive-definite (tp_core.F90:610). Added `_pert_ppm_iv0(q, bl, br)` matching tp_core.F90:1169-1192.
- Fixed v_c indexing in `_ppm_transport_1d` (h3-1 instead of h3 for bl/br alignment).

### iter 44 (2026-04-16) — Courant scaling + exact rdxa
- Removed non-Fortran `crx/(1-offset)` boundary scaling in `_xppm`.
- `rdxa/rdya` now computed exactly from supergrid (was approximated as 0.5*(rdxc[i]+rdxc[i+1])).

### iter 46 (2026-04-16) — _ppm_transport_1d limiter fix
- Codex caught: pert_ppm(iv=0) was incorrectly applied to B-grid KE transport of signed D-grid winds (zeroes reconstruction for q≤0 → destroys negative wind values).
- Restored pmp/lac limiter in `_ppm_transport_1d` (sw_core.F90 ytp_v jord=9 path uses pmp/lac, not pert_ppm). `_ppm_1d` correctly keeps pert_ppm(iv=0) for fv_tp_2d (positive-definite mass transport).

### iter 51 (2026-04-16) — d_sw5 corner divergence damping wired
- `_d_sw5_corner_divergence()`: nord=0 (del-2) duogrid formula with cosa/sina_u/v from sin_sg; nord>0 uses `_divergence_corner_duo` + 5-point Laplacian. Matches sw_core.F90:1641-1821.
- Defaults: d2_bg=0.0, dddmp=0.0, d4_bg=0.16, nord=1.

### iter 52 (2026-04-16) — vorticity damping wired (d_sw6)
- `_del6_vt_flux` (sw_core.F90:2008-2121) called from `_d_sw_native` after wind update. `damp4 = (damp_v * da_min_c)^(nord_v+1)` matches d_sw6 (sw_core.F90:1948-2000).
- Default `damp_v=0.0` (off) matches Fortran `vtdm4=0.0`.

### iter 53-54 (2026-04-16) — divg_u/divg_v + duogrid halo
- Replaced 5-point Laplacian with Fortran's metric-weighted divergence-of-gradient using proper `divg_u = sina_v * dyc / dx` and `divg_v = sina_u * dxc / dy` (fv_grid_utils.F90:717,724).
- `_del6_vt_flux` `use_duogrid` parameter matches Fortran `bounded_domain` gating; `pad_halo` is equivalent to Fortran's MPI scalar exchange for duogrid.

### iter 54 — SW fidelity completeness summary
All SW-relevant FV3 operator formulas verified matching Fortran oracle for the duogrid path. The Fortran `SW_DYNAMICS` compile-time gate (sw_core.F90:2002-2003) means divergence heating is NOT in SW path. Remaining gaps are infrastructure-level (halo width ng=1 vs ng=3) and 3D-only.

### iter 55 (2026-04-16) — Codex adversarial review
Four findings, all resolved as non-bugs or documented limitations:
1. d_sw1 duogrid early return matches Fortran (boundary overrides gated on `.not. dg%is_initialized`).
2. d_sw3 C-grid padding: infrastructure halo gap.
3. PPM edge-repair NOT a simplification — exact Fortran behavior (xppm/yppm boundary formulas all gated on `(.not. (bounded_domain .or. duogrid))`).
4. d_sw5 edge-padded halos: infrastructure limitation masked by face-boundary zeroing (`divg_d=0` at boundaries).

### iter 56 (2026-04-16) — pert_ppm iv=1 fix
- Fortran `ar=-2*al` when `a6da<-da2`, `al=-2*ar` when `a6da>da2`. Python had swapped variables; corrected.
- iv=1 kept unconditional (deferred to iter-63 for proper duogrid gating).

### iter 58 (2026-04-17) — `pad_halo_dgrid` halo=2 support
- Added `halo` parameter; for halo=2, neighbor edges at depth=1 and depth=2 pulled per-edge, rotated via exact edge angles, written to outer/inner halo layers.
- Halo=1 output bitwise-identical to halo=2 inner layer.
- Wiring into `_d2a2c_vect_duogrid` deferred — also requires cross-axis padding (handled in iter 60).

### iter 58 — production+duogrid regression observed
- Enabling `use_duogrid=True` on production path destabilizes W2 at C36 (max|v_north|=347 vs 0.557 m/s). Root cause: `synchronize_cgrid_fluxes` interacts with PPM boundary reconstruction (replaces boundary flux with `0.5*(local + neighbor)` while adjacent interior left untouched → breaks local mass balance in A-L gradient path).
- Pre-existing; documented to explain why `use_duogrid=True` not switched on in production test matrix.

### iter 59 (2026-04-17) — verifications
- Duogrid flux sync: 24 (face, edge) pairs all agree to 1e-14 post-sync.
- Legacy edge handling: 15 guard sites audited; all correctly bypass legacy when duogrid active.
- Iter-58 deferred halo=2 in d2a2c_vect: numerically zero effect on uc/vc — same-axis halo populates wrong axis for the 4th-order A→C stencil. Cross-axis halo needed (iter 60).
- FB+duogrid W2 C36: blows up after 288 steps (consistent with iter-35).

### iter 60 (2026-04-17) — cross-axis D-grid halo
- `_d2a2c_vect_duogrid` rewired to use `ext_vector_dgrid` (fv_duogrid.F90:741-826 equivalent) as single halo source. Produces u_d_full / v_d_full with halos in BOTH axes via c2l_ord2 + scalar lat/lon halo + cubed_a2d_halo.
- Interior overwritten with exact original u_d/v_d (mirrors FV3 mpp_update_domains).
- Length-weighted c2l_ord2: `utmp = (u_j*dx_j + u_{j+1}*dx_{j+1}) / (dx_j + dx_{j+1})` using `cdgrid.dx_edge_y` / `dy_edge_x`. Drops FV3's leading factor of 2 (Fortran compensates via a11/a22 with 0.5; our ext_vector_dgrid is uncompensated).
- Codex rounds: (R1) flagged stitched halo paths → unified to ext_vector_dgrid. (R2) plain 0.5 seed only matches in uniform-dx limit → length-weighted. (R3) factor-of-2 doubling bug → dropped leading 2. (R4) test sliced past seams → added explicit boundary assertions.
- 5 seam regression tests added in `TestD2a2cVectDuogridSeams`.

### iter 61 (2026-04-17) — c_sw/d_sw audit
Full chain ALREADY IMPLEMENTED at `_c_sw` (1029), `_p_grad_c`, `_d_sw_native` (1720), `_d_sw1_recompute_ut_vt` (43), `_bgrid_ke_transport` (1638), `_d_sw5_corner_divergence` (800), `_del6_vt_flux`, `fv3_fb_sw_step` (1835).

c_sw mass error per step on W2 balanced state (dt=300):
| Grid | max\|dh\| |
|------|-----------|
| C8   | 16.0 m |
| C16  | 40.0 m |
| C24  | 64.6 m |
| C36  | 102.0 m |

Error GROWS with resolution at face boundaries — halo-quality issue. FV3's ng=3 MPI halo gives stability where ours doesn't. Cosine bell FB path stable (no exponential growth) but mass drifts due to first-order upwind. W2 FB C36 blows up ~99 steps regardless of damping (vtdm4=0.06 nord_v=1/2, d4_bg=0.32) — root cause is NOT damping shortage.

### iter 62 (2026-04-17) — pert_ppm iv=1 indices + FB plumbing
- pert_ppm iv=1 cell positions corrected from `[0,1,2,-3,-2,-1]` (off-by-one, included halo) to `[1,2,3,-4,-3,-2]` (interior cells matching Fortran tp_core.F90:629,648).
- `CDGridShallowWaterConfig` extended with 6 fields matching Fortran defaults: `d2_bg=0.0`, `dddmp=0.0`, `d4_bg=0.16`, `nord=1`, `damp_v=0.0`, `nord_v=-1` (sentinel; `FV3FBShallowWaterModel.step` substitutes `min(2, nord)` per dyn_core.F90:757,1258).
- `FV3FBShallowWaterModel.step` now forwards all 6 controls (previously only `div_damp`).

### iter 62 — investigated but not implemented
- d_sw3 BGRID_NE component sync: requires FMS `mpp_get_boundary` vector rotation semantics at cross-panel seams; current scalar KE sync matches Fortran's commented-out alternative.
- pert_ppm iv=1 unconditional vs gated: see iter 63 (now properly gated).
- Production path uses A-L/RK3, not FB: architectural; FB path unstable at W2 C36 due to halo gap.
- D-grid vector halo halo=2 vs ng=3: infrastructure-level.

### iter 63 (2026-04-17) — pert_ppm iv=1 duogrid gate
- `_ppm_1d` now accepts `use_duogrid` and skips iv=1 when set (Fortran tp_core.F90:612 gates on `.not. (bounded_domain .or. duogrid)`).
- Codex stop-time follow-up: switched `fv_tp_2d`'s three `pad_halo` calls to canonical `_pad_halo_auto_h2` pattern (`interp_offsets=None, duogrid=dg` when active). Now iv=1 gate and halo used line up.

### iter 64 (2026-04-17) — `_ppm_1d` boundary corrections gated
- Offset-derived `dm` rescaling and position-aware `al` edge corrections now gated on `not use_duogrid`. For duogrid, standard monotone `dm` + standard `al` apply (matches Fortran uniform-spacing path); for non-duogrid the offset corrections compensate `interp_offsets` halo semantics.

### iter 65 (2026-04-17) — non-duogrid d2a2c_vect regression test
- Verified Python `fv3_sw_core.py:534-539,590-595` already matches sw_core.F90:660-668,677-684,696-703,714-721 (face-boundary ut/vt sin_sg upwind override).
- New test `test_face_boundary_ut_divides_uc_by_upwind_sin_sg` locks correspondence at machine precision.

### iter 66 (2026-04-17) — rsin_u/rsin_v Fortran-faithful mixed convention restored
- Codex flagged: iter-17's "1/sin² everywhere" was a numerical-smoothness deviation, NOT Fortran-faithful. Restored mixed convention (interior `1/sina_u²`, panel edges `1/sina_u` — only when `.not. bounded_domain`).
- Gated on `base.duogrid is None` initially, then extended (follow-up #2) to `bounded_domain = (base.duogrid is not None) or (base.lat.shape[0] == 1)` to cover regional/nested single-face panels.
- Follow-up #3: `create_cubed_sphere_panel(return_cdgrid=True)` post-processes extracted cdgrid to recompute `rsin_u = rsin_v = 1/sin²` everywhere (bounded_domain). Also fixed pre-existing missing `duogrid=None` field.
- All 29 `test_fv3_audit_harness.py` tests pass.

### iter 68 (2026-04-17) — non-duogrid `_d2a2c_vect` 4-pt adjacent-strip ported
- Fortran sw_core.F90:670-691 (W/E) and 701-722 (S/N) recomputes vt/ut at interior-adjacent strip via 4-point contravariant cross-velocity average: `vt(1,j) = vc(1,j) - 0.25*cosa_v(1,j)*(ut(1,j-1)+ut(2,j-1)+ut(1,j)+ut(2,j))`.
- Python applies at `vt[:, 0, j]`, `vt[:, n-1, j]`, `ut[:, i, 0]`, `ut[:, i, n-1]` for `j_face/i_face ∈ [2, n-2]`. Gated on `n >= 4`.
- Halo columns and four corner 2x2 solves (sw_core.F90:739-811) remain deferred (interior-only layout limitation).

### iter 69 (2026-04-17) — investigated `_fill_corners_h1/h2`
- Python's 2-point average vs Fortran `copy_corners` directional rotated copy (`q(i,j) = q(j, 1-i)` for X-sweep, `q(1-j, i)` for Y-sweep).
- `fv_tp_2d`'s operator-split PPM slices `q_full[:, 2:-2, :]` (y-sweep) and `q_i_pad[:, :, 2:-2]` (x-sweep) — never includes both i-halo and j-halo. Cube-vertex 2x2 corner blocks NEVER dereferenced by FV3-faithful transport. Only consumer is non-FV3 A-L gradient.
- Decision: do NOT port directional copy_corners. Added inline doc + `TestFvTp2dCornerInvariant::test_fv_tp_2d_independent_of_corner_ghost_values`.

### iter 70 (2026-04-17) — investigated `_deln_flux` corners
- Behavioral mock-patch test confirms `_deln_flux` at nord=1 is bit-identical when 2x2 cube-vertex corner blocks of `pad_halo` output are NaN-poisoned (slices `d2_pad[:, :-1, 1:-1]` etc. all exclude i-halo∪j-halo cells).
- Decision: do NOT port directional `copy_corners`. Added `test_deln_flux_output_unchanged_when_corner_ghosts_nan`.

### iter 74 (2026-04-17) — planetary vorticity at corners via `f_corner`
- `cdgrid_momentum_tendencies` (SW branch): interpolate only ζ to corners, then add `cdgrid.f_corner = 2Ω sin(lat_corner)` directly. Removes O(dx²) interpolation error from `interp(sin(lat_cc)) ≠ sin(lat_corner)`.
- Codex stop-time follow-up: `FV3EdgeShallowWaterModel` calls `fv3_sw_tendencies` (NOT `cdgrid_momentum_tendencies`). At `operators_cdgrid.py:1458`, `zeta` and `v_cc` are BOTH at cell centres → `cdgrid.base.f` (cell-centre f) IS correct. Initial misapplication reverted; added inline comment.

### Deferred from Codex iter-64 review
- d_sw3 `_ppm_transport_1d` non-duogrid edge repair: gated on `(.not. bounded_domain .or. .not. duogrid_initialized)` — SKIPPED for duogrid in Fortran; non-duogrid FB path is experimental and unstable regardless.
- `cos_sg` midpoint geometry: Fortran uses `mid_pt3_cart` + `inner_prod(ec1,ec2)`; Python uses centred-difference tangent on uniformly-spaced gnomonic supergrid. Both produce valid metrics; structural choice, not a formula bug.

---

## Sanity checks
- `JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/unit/test_cdgrid_fv3_regression.py tests/unit/test_duogrid.py` → 86 passed
- `JAX_ENABLE_X64=1 scripts/run_atmosphere_test_matrix.py --only sw --grid cubed_sphere --quick` → 3 PASS
  - Williamson 2: L2=1.94e-03, Linf=8.66e-03
  - Williamson 5: mass drift=1.56e-05
  - Cosine bell: L1=1.42e-01, L2=1.35e-01, Linf=1.49e-01
- Ocean rest state: all 4 cubed-sphere variants PASS (eta drift 1e-14 to 1e-18)
- No visible edge artifacts in v-wind, wind speed, or cosine bell snapshots
