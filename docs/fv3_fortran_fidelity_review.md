# FV3 Fortran Fidelity Review

Baselined 2026-04-14.  Live audit log (per-iteration entries) in
"Latest iteration work" section further down; earlier frozen
snapshots at iter-67 and iter-112.

> **Metadata convention (iter-174)**: the earlier "updated through
> Ralph iter N" dateline in this title was removed because each
> iteration bumping it triggered a Codex-flagged inconsistency
> (the number would lag by 1-4 iterations as soon as the NEXT
> stop-hook committed anything).  The authoritative current
> iteration count is the commit count on branch plus the HEAD
> commit message's iter-N tag; there is no need to duplicate that
> metadata in the title.


## Architectural unresolved items (carried through all iterations)

1. **W2 v-wind cube-face imprint at C36** — production path
   (`FV3EdgeShallowWaterModel` → `fv3_sw_tendencies`) uses
   Arakawa-Lamb + RK3, NOT the FV3 FB chain.  iter-505 reduced
   amplitude from ~0.56 m/s to ~0.30 m/s by fixing a PPM x-axis
   bug; the residual 4-fold mode-4/8/12 signature at ±30° is
   inherent to A-L + halo-interpolation + `boundary_fix` and is
   NOT eliminable within the production path.  Fix requires
   either (a) stabilize FV3 FB chain at C36 (needs ng=3 halo
   infrastructure — partially scaffolded, see iter-496..598
   below), or (b) accept and lock the imprint magnitude.
   Iter-592 chose (b) as short-term guard.

2. **FB-path C36 instability** — `_c_sw` first-order upwind
   amplifies face-boundary halo divergence; requires ng=3
   MPI DGRID_NE halo infrastructure for `_d2a2c_vect`.  Scalar
   `pad_halo(halo=3)` + `halo_interp_offsets_h3` + padded grid
   angle + half-metrics + vector `pad_halo_vector(halo=3)` on
   non-MPI backend ALL wired (iter-496..598).  Remaining:
   `pad_halo_mpi_4d(halo=3)` + flipping `_d2a2c_vect` and
   `fv3_fb_sw_step` callers from h=2 to h=3.

## Priority resolution history (iter-96..131)

- **Priority 1 (panel-edge corner metrics)**: RESOLVED iter-96/98/99.
  Python `cdgrid.cosa_corner`/`sina_corner`/`rsin2_corner` matches
  Fortran halo-averaged formula with cross-face sign-flip rotation
  at all 24 seams to 1e-10.  iter-91..95 "convention gap" claim
  was wrong (naive halo-copy model).  Locked by
  `test_cosa_corner_panel_edge_fortran_match_all_24_seams` +
  `test_rsin2_corner_matches_fortran_at_interior`.

- **Priority 2 (d_sw3 BGRID_NE component sync)**: RESOLVED iter-102/103.
  `synchronize_bgrid_ne_corner_geo` routes through geographic frame,
  handling all 24 seams + 8 cube vertices via
  `synchronize_corner_scalar`.  Wired into `_bgrid_ke_transport`
  replacing scalar-KE fallback, matching `dyn_core.F90:968-1019`.
  Iter-140 removed dead 12-seam legacy (−233 lines).

- **Priority 3 (non-duogrid `_d2a2c_vect` cube-vertex gap)**:
  GAP ISOLATED AND LOCKED iter-107/108/128..131.  Fortran
  `sw_core.F90:3527-3545,3620-3640` writes 3 halo cells per
  corner-axis; Python's `_fill_corners_h2` can represent only 2
  per axis (architecturally bound, needs halo=3).  Reachability:
  only experimental FB chain + `fv3_csw_tendencies`; default
  production A-L path does NOT call `_d2a2c_vect`.  Runtime
  tripwires (iter-129/130) pin the call graph.  iter-569/570
  extracted exact ndsl / pyFV3 `fill_corners_3cells_mult_x`
  recipe for future h=3 port.

- **Priority 4 (FB-path diagnostic)**: RESOLVED iter-109..111
  with `test_fb_path_component_vs_scalar_sync_propagates_to_wind`
  pinning end-to-end propagation at ±2e-3 of measured diffs.

## Polar-face asymmetry investigation (iter-112..127, closed iter-510)

Pre-iter-505 baseline showed 16% N-S mass-tendency asymmetry
between faces 4 and 5 on W2 alpha=0 balanced state.  Bisection
localized to `cgrid_mass_flux_divergence` PPM upwind at polar-face
halo cells.  **Iter-505 axis fix eliminated this entirely** —
faces 4/5 now identical tendency magnitudes to 1e-4 of 1.0 and
exact N-S reflection symmetry at machine precision.  Locked by
`TestFv3SwTendenciesPolarFaceSymmetry` (iter-510).

## ITER-505 major production-path bug fix (2026-04-19)

**Root cause**: `cgrid_mass_flux_divergence` (`operators_cdgrid.py`)
and `_cgrid_fct_fluxes_2d` (iter-506) passed x-direction strips of
shape `(6, n+4, n)` directly to `_ppm_reconstruct_1d`, which
reconstructs along the LAST axis.  The halo-padded i-axis was
axis 1 but the 8-cell interior j-axis was LAST — so reconstruction
ran along the wrong axis.  **Fix**: `swapaxes(1, 2)` before
reconstruction, swap back after.

**Impact on canonical W2 alpha=0 C36 dt=300s 1 day** (matrix path):

| Quantity                        | Pre-iter-505 | Post-iter-505 | Change |
| ------------------------------- | ------------ | ------------- | ------ |
| Williamson 2 L2                 | 1.53e-3      | **2.42e-4**   | 6.3× ↓ |
| Williamson 2 Linf               | 4.07e-3      | **1.83e-3**   | 2.2× ↓ |
| Williamson 2 max\|v_ll\|        | 0.557 m/s    | **0.303 m/s** | 1.8× ↓ |
| Williamson 5 mass drift         | 1.42e-5      | **1.74e-5**   | within order |
| Cosine bell L1                  | 1.20e-1      | **1.20e-1**   | unchanged |
| Ocean rest state                | machine prec | **machine prec** | invariant |

Cosine bell uses `transport_step` directly (Lin-Rood), not
`cgrid_mass_flux_divergence`.  Iter-508/509 refactored
`_ppm_reconstruct_1d` to take mandatory `axis` kwarg; AST guard
`test_no_future_caller_passes_non_halo_last_axis_to_ppm` enforces
contract.

## ITER-67 historical baseline

**Scope**: Python `src/legoesm/grids/cubed_sphere_cdgrid.py`,
`core/{fv3_sw_core,fv_tp_2d,operators_cdgrid}.py`,
`atmosphere/dynamics/shallow_water_fv3_cdgrid.py`; Fortran oracle
`../atmos_cubed_sphere-symmetryclean/model/*.F90`.

All FV3 duogrid-branch operator formulas verified against oracle:
metrics (sin_sg-based cosa_u/rsin_u, supergrid area_corner/dxc/dyc,
mixed 1/sin² interior + 1/sin edge for non-duogrid); `c_sw`
(4th-order D→A→C, sin_sg/cos_sg boundary conversions, corner
vorticity, 1/sin vorticity flux); `d2a2c_vect` duogrid
(cross-axis halo via `ext_vector_dgrid`, length-weighted c2l_ord2);
`d_sw1`-`d_sw6` (adjacent-strip + corner 2x2 solve, PPM hord=9
B-grid KE transport, corner divergence damping, vorticity damping);
`fv_tp_2d` (Lin-Rood operator split + CGRID flux sync +
`pert_ppm(iv=0/1)` gated on non-duogrid).  FB wrapper forwards
`d2_bg/dddmp/d4_bg/nord/damp_v`; auto-derives `nord_v=min(2,nord)`.

## Historical findings (baseline 2026-04-14, pre-iter-1)

Six baseline gaps identified vs Fortran oracle, all RESOLVED by
iter-1..iter-66: (1) metric construction from `sin_sg/cos_sg`;
(2) supergrid `area_corner/dxc/dyc`; (3) `_d2a2c_vect_duogrid`
covariant convention; (4) `_c_sw` boundary KE/vorticity specials;
(5) FB chain ported (iter-51..64) but C36-unstable (arch item #2);
(6) regression tests updated to FV3 formulas.  Gaps F3-1..F3-5:
F3-1 closed as not applicable; F3-2..F3-4 resolved by iter-8/25/29;
F3-5 partially ported iter-68, full corner 2x2 architecturally
blocked by halo infrastructure.

## Condensed iteration log (iter-1..iter-589)

### Iter-1..iter-46 (SW fidelity core)
- iter-1: cosa_u/rsin_u from sin_sg/cos_sg; cell-centre metrics.
- iter-2: supergrid area_corner/dxc/dyc.
- iter-3: sin_sg transport metrics haloed in `_c_sw`.
- iter-5: `c_sw` vorticity flux uses `1/sin`.
- iter-8/25: `_c_sw` boundary KE/vorticity specials; iter-9/11/25
  boundary ut/vt override with cross-velocity term.
- iter-13/14: edge-padded 4th-order D→A; halo=2 for
  `edge_interpolate4`.
- iter-15..24: `_d2a2c_vect_duogrid` rewrite; physical-frame KE
  (−24% boundary residual); vector halo (−27× uc smoothness).
- iter-17: `rsin_u` uniform 1/sin² — REVERTED iter-66.
- iter-25: ut override to positions {0,n}; vorticity flux narrowed.
- iter-26: 4-edge mean angle diagnostic (−47× v_north at t=0).
- iter-28: CGRID flux sync; W5 drift 1.42e-5 → 1.09e-6 with duogrid.
- iter-29: direct corner vorticity in `_c_sw`.
- iter-31: unconditional flux sync on non-duogrid rejected (110× W2 regression).
- iter-32: `_bgrid_ke_transport` ported.
- iter-36: `synchronize_bgrid_ne` — LATER REMOVED iter-102/140;
  `_ke_upwind`/`_corner_vorticity`/`_vorticity_flux` helpers
  extracted (−120 lines).
- iter-37: duogrid+production stable at C16 1d (130× improvement).
- iter-38: line-by-line Fortran trace verified duogrid formulas.
- iter-39: d2a2c_vect face-boundary sin_sg halo fix.
- iter-40: upwind-selected rdxa; PPM hord=9.
- iter-42..44: PPM limiter refinements; pert_ppm(iv=0) per
  `tp_core.F90:610`; exact rdxa/rdya from supergrid.
- iter-46: `_ppm_transport_1d` restored to pmp/lac for signed winds.

### Iter-51..iter-67 (SW fidelity completion + damping)
- iter-51/52/53/54: corner divergence damping + vorticity damping
  wired; proper `divg_u/divg_v` metrics.
- iter-56: `_pert_ppm iv=1` bl/br fix.
- iter-58: `pad_halo_dgrid` halo=2 (ng≥2 infrastructure first half).
- iter-60: cross-axis D-grid halo via `ext_vector_dgrid` + length-
  weighted c2l_ord2 (single unified path).
- iter-61: FB chain audited line-by-line; c_sw mass error scales
  with resolution (halo-quality root cause).
- iter-62: pert_ppm iv=1 cell indices corrected to interior;
  FB damping controls plumbed.
- iter-63: pert_ppm iv=1 gated on `use_duogrid`; fv_tp_2d
  switched to canonical `_pad_halo_auto_h2`.
- iter-64: dm-rescaling + al edge-correction gated on non-duogrid.
- iter-65: non-duogrid `_d2a2c_vect` regression test.
- iter-66: rsin_u RESTORED to Fortran-faithful mixed convention
  (iter-17 had flattened to 1/sin²); panel build for regional.

### Iter-68..iter-90 (polish + duogrid halo audit)
- iter-68: non-duogrid `_d2a2c_vect` 4-point adjacent-strip ported.
- iter-69/70: `_fill_corners_h1/h2` vs Fortran `copy_corners` —
  closed as not-needed (Python stencils never dereference both
  i/j halos simultaneously in fv_tp_2d/_deln_flux).
- iter-74/77: planetary vorticity via `f_corner` at corners.
- iter-78/79: `_del6_vt_flux`/`_deln_flux`/`compute_transport_
  quantities`/`_c_sw` sin_sg halos silently non-duogrid — fixed.
- iter-80: regression test.
- iter-81: `_arakawa_lamb_gradient` metric matrix duogrid routing.
- iter-82..86: PE MPI packed halo duogrid extension.
- iter-87..90: `_vorticity_flux` from sin_sg (not `sqrt(1-cosa²)`);
  cdgrid.cosa_corner verified Fortran-faithful at interior +
  panel-edge + cube-vertex corners.

### Iter-91..iter-131 (priority work + reconciliation + gap hardening)
- iter-91..95: "convention gap" claim — LATER CORRECTED iter-96
  (naive halo-copy model ignored sub-grid sign-flip rotation).
- iter-96..99: Python IS Fortran-faithful; 24-seam coverage
  including 8 reversed seams.
- iter-100..111: iter-102 `synchronize_bgrid_ne_corner_geo` via
  geographic frame (24 seams + 8 vertices); iter-103
  `_bgrid_ke_transport` wired; iter-104..111 regression tests +
  FB-path end-to-end sync propagation lock.
- iter-112: evaluation snapshot unchanged from iter-77 baseline.
- iter-113..127: polar asymmetry bisection — CLOSED BY ITER-505.
- iter-128..131: Priority-3 gap hardening — AST architectural
  bound, runtime reachability tripwires.

### Iter-132..iter-174 (runtime fixes + cleanup)
- iter-132/133: `_d_sw5_corner_divergence` mode='edge' halo gap
  documented (FB chain only).
- iter-135/139: plateau + stale validation_report.md check.
- iter-136/137/138: ocean latlon quiver C-grid→A-grid fix
  (4 ERROR → PASS); phillips/inertia-gravity reverted.
- iter-140: removed dead `synchronize_bgrid_ne_corner` (−233 lines).
- iter-142..147: JAX shard_map fix, unused import cleanup (−195 lines).
- iter-149..151: TPU bootstrap crash on JAX 0.9 (`jax_spmd_mode`);
  regression test.
- iter-154: CPU bootstrap crash fix
  (`--intra_op_parallelism_threads`); regression test.
- iter-155..174: XLA flag audit, test file fixes, session
  checkpoints, doc metadata convention normalization.

### Iter-496..iter-510 (ng=3 halo infrastructure + MAJOR bug fix)
- iter-496: `compute_halo_interp_offsets_h3(n)` helper.
- iter-497: `_fill_corners_h3` 3×3 inside-out corner fill.
- iter-498: `_pad_halo_local_h3` generalizing h2 to 3 depths.
- iter-499: `pad_halo(halo=3)` dispatch with NotImplementedError
  guards for MPI/SPMD/Duogrid.
- iter-500/501: halo=3 public-API guardrails + shape check.
- iter-502/503/504: AST lock for Critical Duogrid Constraint #1
  (flux sync call sites in 3 hosts); host-function pairing +
  nested-scope/comprehension escape hatches closed.
- **iter-505**: MAJOR axis bug fix (see section above).
- **iter-506**: sibling bug in `_cgrid_fct_fluxes_2d`; AST guard.
- iter-507: strengthened FCT behavioral guard.
- iter-508/509: `_ppm_reconstruct_1d` explicit `axis` kwarg,
  keyword-only + required; AST guard enforces literal int ≥ 1.
- iter-510: polar-face asymmetry CLOSED by iter-505.

### Iter-511..iter-528 (W2 lock + boundary_fix characterization)
- iter-511..515: W2 end-to-end error-budget lock at canonical
  C36 dt=300s 1d (L2 < 5.0e-4; True/False ratio < 0.7 at 0.525).
- iter-516/517: Constraint #2 (5 production gates) locked under
  duogrid bypass; FULL production propagation chain mock-patched.
- iter-518: `fv3_cc2c v_c` asymmetry INTENTIONAL (mixed-orthogonal
  D-grid); AST lock; iter-126 polar baseline test renamed.
- iter-519..523: W2 v-wind imprint lock pinned to canonical
  lat-lon regridded (max|v_ll| < 0.4 m/s at all 11 snapshots).
- iter-524: visual inspection — CLEAN except W2 v-wind residual.
- iter-525/526: cosine bell positivity lock (every step +
  NaN/Inf).
- iter-528/529: `cell_centre_angles_from_4edge(cdgrid)` helper.

### Iter-530..iter-549 (ng=3 readiness + behavior locks)
- iter-530..535: halo=3 grid-angle + half-metrics + interp_offsets
  wired as precomputed attributes; duogrid halo=3 combination
  supported; outer-ring third-ring edge strips verified.
- iter-536..539: halo=3 corner-cells outer ring coverage.
- iter-540: W5 initializer Fortran planar distance (clipped).
- iter-541: behavioral lock on iter-518 convention.
- iter-542: **documented Python CW vs FV3 iord==8/10 limiter
  divergence**.  AST-asserts `smt5/smt6` symbols ABSENT.
- iter-543..549: source-level locks for BGRID_NE sync,
  `_interp_corner_to_center`, `synchronize_corner_scalar`
  (including Pass-1 contamination guard),
  `_interp_center_to_corner`, `_broadcast_metric`.

### Iter-550..iter-575 (corner vorticity + operator-level hardening)
- iter-550/551: `_corner_vorticity` regression + interior
  circulation formula lock.
- **iter-552**: PRODUCTION NUMERICAL CHANGE — gate
  `_corner_vorticity` linear-extrapolation on `not use_duogrid`
  per `sw_core.F90:396-400`.
- iter-553..557: `_divergence_corner_duo` locks (boundary zeroing,
  0.25 factor bit-exact in float32, double-attenuated corners).
- iter-558/559: `_extrapolate_boundary_corners` lock (du + dv).
- iter-560..564: `_laplacian_dgrid`, `cgrid_tracer_advection_fct`
  (no-op guards + 3D/4D level-routing).
- iter-565: `_pad_halo_auto` + `_pad_halo_auto_h2` lock.
- iter-566..571: pyFV3 cross-references (xppm.py iord==8,
  d2a2c_vect.py, c_sw.py); CW vs iord==8 empirical divergence
  test; exact ndsl cell mapping for future h=3 port.
- iter-572..575: `_arakawa_lamb_gradient` + `cgrid_divergence`
  comprehensive behavior locks (shape, no-op, anti-symmetry,
  linearity, per-level, exact flux-form formula).

### Iter-576..iter-589 (exact-formula locks)
- iter-576: exact `_arakawa_lamb_gradient` 4-point stencil.
- iter-577/578: `cdgrid_momentum_tendencies` lock (non-zero
  output + pressure-gradient sign convention).
- iter-579/580: `dgrid_to_center_vector`/`center_to_dgrid_vector`
  + c2d_vector exact halo + 4-point formula locks.
- **iter-581**: AST-level lock for 4 Constraint #2 legacy-path
  gates (total 5 production gates).
- iter-582..588: iter-581 strengthening + `_corner_vorticity`
  linear-extrapolation gate + 4 cube-vertex correction body lock
  + `_vorticity_flux` fy1/fx1 dual-edge under/non-duogrid +
  `_ke_upwind` 4 boundary override with exact sin_sg/cos_sg.
- iter-589: zero-sign coverage for `_ke_upwind` (strict vs
  inclusive inequality).

---

## Latest iterations (full form)

### Iter-590 (2026-04-19): symmetric ke_v zero-sign coverage for Constraint #2

**Codex stop-time review (iter-589):** "Iter-589 claims exhaustive zero-sign coverage, but it only tests `ke_u` and leaves the symmetric `ke_v` drift unguarded."  The prior iteration only locked the `ua > 0` / `ua <= 0` strict/inclusive boundary on the west (`ke_u[:, 0, :]`) and east (`ke_u[:, n-1, :]`) edges; it did not verify the matching `va > 0` / `va <= 0` strict/inclusive boundary on the south (`ke_v[:, :, 0]`) and north (`ke_v[:, :, n-1]`) edges.  A silent `>` → `>=` refactor on either south or north would therefore pass.

**Change:** extended `test_ke_upwind_legacy_boundary_override_gated` in `tests/unit/test_duogrid.py` with two zero-sign ke_v checks:
1. **South edge with va=0 at j=0**: under `duogrid=False`, `ke_v[:, :, 0]` MUST equal pure downwind `vc[:, :, 1]` at every (i, k).  The south override (`va > 0`) is strict, so at va=0 the override must NOT fire.  A `va >= 0` refactor would silently apply the override at zero-wind cells (24.87 unit deviation measured during sanity check).
2. **North edge with va=0 at j=n-1**: under `duogrid=False`, `ke_v[:, :, n-1]` MUST equal `ke_bdy_t = vc[:, :, n]*sin_sg[N] + u_d[:, :, n]*cos_sg[N]`.  The north override (`va <= 0`) is inclusive, so at va=0 the override MUST fire.  A `va < 0` refactor would silently leave zero-wind cells on the upwind branch.

Both ke_v checks use the same construction pattern as the earlier ke_u south/north checks: synthetic inputs with controlled sign on va, plus sin_sg/cos_sg metric tensors that make `ke_bdy_t` deterministic.

**Sanity check fired:** patched `fv3_sw_core.py:745` `va[:, :, 0] > 0` → `>= 0`.  Test fired with `AssertionError: Under duogrid=False with va=0, ke_v[:,:,0] deviates from PURE UPWIND (vc[:,:,1]) by 2.487e+01.  The south override should NOT fire at va=0 (condition `va > 0` is strict).`  Production code restored; all 107 test_duogrid.py tests pass.

**Constraint #2 coverage summary across 5 gates:**
- `_ke_upwind` (fv3_sw_core.py:731): 4 boundary overrides (W/E on ke_u; S/N on ke_v) × {gate-removal, gate-inversion, body-relocation, body-deletion, zero-sign strict vs inclusive}.  All 6 regression modes now locked in `test_ke_upwind_legacy_boundary_override_gated`.
- `_corner_vorticity` (fv3_sw_core.py:1129): 4 cube-vertex corrections locked in `test_corner_vorticity_legacy_correction_not_applied_under_duogrid` with AST + behavioral checks across 4 regression modes.
- `_vorticity_flux` (fv3_sw_core.py:1156, 1162): fy1/fx1 panel-edge overrides locked in `test_vorticity_flux_legacy_overrides_gated_both_sides` across dual-edge + body-reorder regressions.
- `_ppm_transport_1d` iord==8 pert_ppm path (fv3_sw_core.py): bounded_domain gating locked in `test_pert_ppm_iv1_not_called_under_duogrid_via_production_path`.
- `rsin_u` panel-edge override (operators_cdgrid.py): locked in `test_rsin_u_panel_edge_override_only_in_non_bounded_domain`.

With iter-590, all five Constraint #2 gates have both AST-level and behavioral coverage that fires on realistic regression patterns (gate flip, gate removal, body move, body delete, operator refactor).

### Iter-591 (2026-04-19): prove edge-copy ≡ halo-exchange for `_divergence_corner_duo`

**Codex pre-iteration review** identified a potential fidelity gap in `_divergence_corner_duo` (`src/legoesm/core/fv3_sw_core.py:827-922`): Python uses `jnp.pad(..., mode='edge')` at lines 854-855 and 872-894 to fill the halo needed for `va(i, j-1)` and `ua(i-1, j)` reads, whereas Fortran `sw_core.F90:2413-2425` reads from the duogrid halo (true cross-face-neighbor values).

**Investigation (iter-591)**:
1. Diagnostic print: under duogrid, `pad_halo_vector(ua, va, ...)` at the south halo row differs from edge-copy by up to 16.5 units for random ua/va — the neighbor-face halo value is genuinely different.
2. But propagation analysis shows the halo-vs-edge distinction at `va(i, -1)` only affects `uf(i, 0)` (south-outermost u-flux row), which then only feeds `divg_d(i, 0)` via the stencil `divg_d(i, j) = vf(i, j-1) - vf(i, j) + uf(i-1, j) - uf(i, j)`.  Similarly for the other three face-outer strips.
3. All four face-outer `divg_d` strips (j=0, j=n, i=0, i=n) are unconditionally zeroed at `fv3_sw_core.py:911-914` (per `sw_core.F90:2431-2434`).  Therefore the halo-vs-edge choice has ZERO impact on any non-zero `divg_d` cell.

**Conclusion**: `mode='edge'` padding in `_divergence_corner_duo` is SEMANTICALLY EQUIVALENT to the Fortran halo read under the current gate structure.  **NOT a fidelity gap.**

**Lock added**: `test_divergence_corner_duo_edge_halo_equivalence` in `tests/unit/test_cdgrid_fv3_regression.py:2003-2200`.  Three assertions:
1. Production matches mode='edge' reproduction bit-exact (float32).
2. Halo-exchange reproduction vs. mode='edge' reproduction differ globally by < 1e-4 (within float64 round-trip error at face-outer cells; zeroed at interior cells).
3. Interior divg_d diff between halo and edge reproductions is < 1e-10 (pure float64 rotation round-trip noise).

**Sanity verified**: disabling face-boundary zeroing lines 911-914 in production fires assertion #1 with diff 5.19e-06 — meaning if anyone removes the zeroing, the test catches the regression with a specific error magnitude.  Production restored; all 286 regression tests pass.

**Why this lock matters**: a future refactor that drops the face-boundary zeroing would (a) allow halo-vs-edge to leak into non-zero `divg_d`, breaking Fortran fidelity invisibly; (b) pass all existing float32 bit-exact tests that use the same mode='edge' reproduction.  The new equivalence lock ties the zeroing to the halo-choice semantics and forces the refactor decision to become explicit.

### Iter-592 (2026-04-19): user work plan — W2 v-imprint audit and lock

User directive: the production shallow-water path (`FV3EdgeShallowWaterModel` at `shallow_water_fv3_cdgrid.py:426`) is an explicit Arakawa-Lamb + RK3 stabilized path, NOT the FV3 forward-backward c_sw + p_grad_c + d_sw chain.  The FV3 FB model (`FV3FBShallowWaterModel` at `shallow_water_fv3_cdgrid.py:329`) remains unstable at C36.  The remaining W2 v-wind artifact is therefore a real cube-face mode-4 signature generated by the non-FV3 production operator chain — NOT the old polar bug, NOT a t=0 diagnostic-angle bug, and NOT mainly a raw regridding mistake.

**Audit conclusion (step 1-4 of user work plan)**:
1. **Artifact structure confirmed**: FFT along longitude at lat=±15°, ±30°, ±45° on the saved C36 W2 1-day latlon output shows modes 0, 4, 8, 12 dominate (4-fold cubed-sphere symmetry).  At lat=±30° t=1d: mode-4 amplitude = 5.25e-2 m/s.  At equator: essentially zero (2.6e-5 m/s) as expected for zonal flow.  North/south faces (±15°, ±30°, ±45°) have IDENTICAL mode-4 amplitudes → confirms NOT the old face-4/5 asymmetry.
2. **max|v| growth**: t=0: 8.0e-3 → t=1d: 3.03e-1 m/s, matching the user's observation.  Growth is dynamical (linear-ish) not diagnostic.
3. **Source**: `fv3_sw_tendencies` (`operators_cdgrid.py:1343`) uses the Arakawa-Lamb 4-point gradient, corner winds from `pad_halo_vector(u_cc, v_cc)`, and projects cell-centre tendencies back to D-grid edges via a second halo exchange.  Each halo crossing at face boundaries is a potential O(dx) imprint source.  The `boundary_fix` smoother at `operators_cdgrid.py:1475-1483` masks but does not eliminate the imprint.
4. **Decision**: the production path CANNOT really become FV3-faithful without replacing A-L + RK3 with c_sw + p_grad_c + d_sw (i.e., adopting the FB path).  The FB path needs ng=3 halo infrastructure (still pending) before it is stable enough.  This is a deferred architectural item (not a small-diff fix).

**Locks added (step 5 of user work plan)** in `tests/unit/test_cdgrid_fv3_regression.py::TestW2CubeFaceImprintCharacterization`:
1. `test_w2_t0_v_north_diagnostic_angle_bounded`: at t=0 with canonical W2 IC, max|v_cc_north| < 0.01 m/s.  Measured: 8.2e-3 (iter-26's 4-edge-angle fix).  Sanity-verified: replacing 4-edge angles with single cell-centre angle fires at 0.386 m/s (38x the ceiling).
2. `test_w2_short_run_v_north_N_S_mirror_symmetry`: after 10 steps at C36/dt=300, |v_north[face=4] - (-v_north[face=5][:, ::-1])| / ref < 1e-2.  Measured: 2.3e-3 (4x headroom).  Pre-iter-505 was 0.16 (16x the ceiling).  Guards against PPM polar-axis regression.

**Not yet locked**: mode-4 amplitude ceiling at lat=±30° after 1 day.  Requires running a full 1-day integration (~3s) and regridding to lat-lon (adds binning dependency).  Deferred to a follow-up iteration — the existing `test_w2_alpha0_c36_1day_canonical_l2_post_iter505` already guards the height L2; the N/S mirror + t=0 diagnostic-angle cover the two historic regression classes most likely to silently return.

**Remaining unresolved issue** (added to active backlog): the mode-4/8/12 v_north imprint at ±30° is fundamentally from the A-L + RK3 operator chain and CANNOT be eliminated within the production path.  Two paths forward:
(a) stabilize the FV3 FB path at C36 (needs ng=3 halo port) and switch production default — this is the ONLY route to a fully FV3-faithful W2.
(b) live with the imprint and lock its current magnitude so architecture changes cannot silently make it worse.

Iteration 592 chose (b) as the realistic short-term guard; (a) is tracked as the long-term architectural item.

### Iter-593 (2026-04-19): AST lock on `_d2a2c_vect` interp_offsets shape

**Codex pre-iter-593 pointed at `fv3_sw_core.py:475-481`** — `_d2a2c_vect` requests `halo=2` in `pad_halo_vector` and the offset-table argument must be `halo_interp_offsets_h2` (shape `(6, 4, 2, n)`).  A prior bug (fixed in commit `959454d`, April 2026) passed `halo_interp_offsets` (shape `(6, 4, n)`) to the halo=2 path; the h2 halo implementation indexes `interp_offsets[face, edge_idx, depth]` which on the wrong-shape table yields a SCALAR broadcast per edge rather than the required per-cell array.  Result: halo interpolation degenerated to a uniform shift, injecting O(Δα) position error at face-boundary halos.

**Verification**: the fix is in place at the current HEAD, but no structural test guards against a refactor silently reverting it.

**Lock added**: `test_d2a2c_vect_interp_offsets_match_halo_depth` in `tests/unit/test_cdgrid_fv3_regression.py` (new test inside `TestFvTp2dCornerInvariant`).  Parses the AST of `_d2a2c_vect`, finds every `pad_halo_vector(...)` call, and for any call with `halo=2` asserts `interp_offsets=grid.halo_interp_offsets_h2`.

**Sanity verified**: reverting line 479 to `interp_offsets=grid.halo_interp_offsets` (the buggy version) fires the assertion with a clear pointer to the prior bug and the Fortran anchor (`sw_core.F90:3587` + `3528-3530` for `edge_interpolate4`).  Production restored; all 68 `test_cdgrid_fv3_regression` tests pass.

**Why this matters**: without an AST-level lock, a well-intentioned "attribute cleanup" refactor that removed `halo_interp_offsets_h2` (reading the code superficially as a duplicate of `halo_interp_offsets`) could quietly regress the halo=2 interpolation accuracy in `_d2a2c_vect`, which is reached by both the experimental csw path AND the FB shallow-water model.  The test pins down the shape invariant structurally, not just numerically.

### Iter-595 (2026-04-20): enable halo=3 vector pad on non-MPI backend (FB-path prereq)

**Motivation**: review-doc item #2 — the FB chain (FV3 forward-backward `c_sw + p_grad_c + d_sw`) is unstable at C36 because it needs ng=3 halo depth.  `pad_halo_vector(halo=3)` was explicitly blocked at `halo.py:1474-1479` citing two missing pieces:
(a) h=3 padded grid-angle + half-metrics,
(b) `pad_halo_mpi_4d(halo=3)`.

Piece (a) is small: `compute_padded_angle(n, halo=3)` and `compute_padded_half_metrics(n, radius, halo=3)` already support halo=3 (since iter-530).  The missing piece was wiring the precomputed values into `CubedSphereGrid` and exposing them to the vector halo path.

**Changes (this iteration)**:
1. Added 4 new fields to `CubedSphereGrid` (`src/legoesm/grids/cubed_sphere.py`): `cos_angle_padded_h3`, `sin_angle_padded_h3`, `hx_ext_h3`, `hy_ext_h3` — each shape `(6, n+6, n+6)`.
2. Computed them in `create_cubed_sphere` via the existing `compute_padded_angle(halo=3)` / `compute_padded_half_metrics(halo=3)` helpers.
3. Wired them into the single-face panel constructor (wall-BC path) at `cubed_sphere.py:735` using the existing `_repad` helper for Neumann boundaries.
4. Relaxed the halo=3 `NotImplementedError` in `pad_halo_vector` (`src/legoesm/grids/halo.py:1474-1483`) to raise ONLY under the MPI backend (piece (b) is still missing).  Non-MPI now accepts halo=3.

**Locks added**:
- `test_pad_halo_vector_halo3_works_on_local_backend` in `tests/unit/test_scale_halo.py::TestPadHaloH3Guardrails`: constructs a C8 grid, runs a constant-field round-trip through `pad_halo_vector(halo=3)` using the new `cos_angle_padded_h3`/`sin_angle_padded_h3`, and asserts the interior is preserved within 1e-10 (float64 rotation round-trip noise).
- `test_pad_halo_vector_halo3_rejects_mpi_backend` (replaces the iter-500 unconditional-raise lock): monkey-patches `_halo_backend='mpi'` and verifies the NotImplementedError message cites "MPI backend".

**Regression**: 343 tests pass across `test_scale_halo.py`, `test_cdgrid_fv3_regression.py`, `test_duogrid.py`, `test_cdgrid.py`.

**Status of FB-chain ng=3 unlock (review-doc item #2)**:
- ✅ Scalar `pad_halo(halo=3)` (since iter-499).
- ✅ Halo interp offsets at h=3 (`halo_interp_offsets_h3`, since iter-532).
- ✅ Padded grid-angle + half-metrics at h=3 (THIS ITERATION).
- ✅ Vector `pad_halo_vector(halo=3)` on non-MPI backend (THIS ITERATION).
- ❌ `pad_halo_mpi_4d(halo=3)` — remaining piece for the MPI path.
- ❌ Wiring h=3 into `_d2a2c_vect` and `fv3_fb_sw_step` — the ng=3 callers still use halo=2.  Iter-596+ can now flip callers without hitting the infrastructure wall.

The halo=3 infrastructure is no longer the blocking dependency on the non-MPI single-device path, which is what the W2/W5 test matrix uses.

### Iter-596 (2026-04-20): validate the halo cells iter-595 exposed

**Codex stop-time review on iter-595 (commit e087f33)**: the iter-595 test `test_pad_halo_vector_halo3_works_on_local_backend` removed the halo=3 guard without validating the halo cells it just exposed — it only checked interior preservation, which is the trivial identity portion of the rotation round-trip.  The new halo cells (indices [0..2] and [-3..-1] in the padded output) had ZERO explicit validation.

**Enhanced test coverage**:
1. **Interior preservation** (iter-595 invariant, kept): `up[:, 3:-3, 3:-3]` ≈ input u within 1e-10.
2. **Inner two halo rings match halo=2 output** (new): for random inputs, the INNER TWO rings of the halo=3 output (indices [1, 2] and [-3, -2]) must agree with the validated halo=2 output at the overlapping physical cells — per direction (W/E/S/N) and per component (u/v) — to within 1e-10.  This anchors the h=3 path to the already-validated h=2 path at the two overlapping rings.
3. **Outermost halo ring is non-trivial** (new): for random unit-variance inputs, the outermost halo ring (index [0] and [-1]) must be finite AND have max magnitude > 0.1 on each of the 4 sides — catches bugs where the outer ring is zero, NaN, or silently dropped.

**Sanity-verified both new checks fire**:
- Zeroing `u_east_padded[:, 0/-1, :]` / `v_north_padded[:, :, 0/-1]` at h=3 produces outermost-ring magnitude 0.0 on face 0, firing check-3 with the message "h=3 outermost W ring has near-zero magnitude (0.000e+00)".
- Perturbing the inner ring `u_east_padded[:, 1, :] += 10.0` at h=3 produces 9.96 diff vs h=2, firing check-2 with the message "h=3 inner halo rings on W u diverge from validated h=2 output by 9.963e+00".

Production halo.py restored after each sanity-check.  All 54 halo tests pass.

**Note**: check-2 has a subtle implicit invariant — the halo=2 reference path must itself be correct.  `TestPadHaloH3Dispatch` elsewhere in the file validates the halo=2 pad (and the iter-103 BGRID_NE sync + iter-107 duogrid constant-field checks extend that coverage).  The h=3 → h=2 anchor chain makes iter-596 a tight composition lock rather than an independent check.

### Iter-597 (2026-04-20): exact correctness check on halo=3 outer ring

**Codex stop-time review on iter-596 (commit 810d089)**: the iter-596 outer-ring check only verified NON-TRIVIALITY (finite, |max| > 0.1), not CORRECTNESS.  A bug that returned `0.5 * expected` everywhere in the outermost ring would produce magnitudes still > 0.1 and pass iter-596 silently.

**Exact-correctness lock added**: `test_pad_halo_vector_halo3_outer_ring_exact_constant_geographic_wind` in `tests/unit/test_scale_halo.py::TestPadHaloH3Guardrails`.

**Key insight**: for a CONSTANT geographic wind (u_east = 1, v_north = 0 everywhere on the sphere), the grid-aligned input is `u_grid = cos_angle`, `v_grid = -sin_angle`.  After `pad_halo_vector(halo=3)`, the padded output must equal:
- `u_padded[f, i, j] = cos_angle_padded_h3[f, i, j]`
- `v_padded[f, i, j] = -sin_angle_padded_h3[f, i, j]`

at EVERY padded cell including the outermost halo ring.  This is an EXACT analytical prediction, not a round-trip-to-reference, so it validates the absolute correctness of the outer ring.

**Two assertions**:
1. Full-array `max|u_out - cos_angle_padded_h3| < 1e-6` (and same for v).  Tolerance 1e-6 reflects the float32 storage precision of `halo_interp_offsets_h3` — 1e-6 is ~8x above the 1.2e-7 precision floor.
2. Per-side outermost-ring isolation check (W/E/S/N × u/v): same 1e-6 bound, isolates which side a regression is in if it fires.

**Sanity-verified**: halving the outermost ring (`u_padded = u_padded.at[:, 0, :].multiply(0.5)` inside `pad_halo_vector`) fires the test with diff 0.498 vs the 1e-6 ceiling — 5 × 10⁵ × the tolerance.  This kind of amplitude bug would have passed iter-596's non-triviality check (the halved values are still > 0.1) and this iter-597 lock catches it.

Production restored.  3 h=3 tests pass (interior + h2-anchor + exact-correctness).  The outermost halo=3 vector ring is now locked to an EXACT analytical reference, closing the correctness gap Codex flagged.

### Iter-598 (2026-04-20): outer-ring CONNECTIVITY + rotation correctness

**Codex stop-time review on iter-597 (commit 616aa83)**: the constant-u_east test doesn't exercise interpolation coefficients — `interp(constant) = constant` regardless of offset weights.  A bug in the neighbour-face dispatch or axis-reversal for the outermost ring would pass iter-597 silently.

**New lock**: `test_pad_halo_vector_halo3_outer_ring_face_unique_connectivity` uses FACE-UNIQUE u_east values (face f has u_east = f+1, v_north = 0).  For each face and each of 4 sides, the test looks up `CONNECTIVITY[face][side]` to get the expected neighbour face and asserts:
- `u_grid_out[face, outer_ring, :] == cos_angle_padded_h3[face, outer_ring, :] * (nbr_face + 1)`
- `v_grid_out[face, outer_ring, :] == -sin_angle_padded_h3[face, outer_ring, :] * (nbr_face + 1)`

This validates:
1. **CONNECTIVITY correctness**: the outermost halo pulls from the correct neighbour face (a bug swapping faces would read the wrong constant).
2. **Rotation correctness**: the INVERSE rotation uses the correct padded-angle cell (not a transverse position, not a wrong face angle).
3. **Axis-reversal correctness**: for reversed connections (e.g., face-1 SOUTH → face-5 EAST reversed), the halo still reads the right face's constant (since the test uses constant-per-face inputs, reversal of a constant strip is still the same constant).

**Sanity-verified**: scaling u_padded outer ring by 1.25 on the WEST side fires the test with diff 0.996 (vs 1e-6 ceiling).  Catches amplitude/CONNECTIVITY errors iter-597 would miss.

**What this iter does NOT cover**: interpolation of NON-constant fields (varying along the strip).  This would require spatially-varying geographic inputs with a known analytical halo prediction and is deferred as a separate harder test — combined with iter-596's h=3 → h=2 anchor (which validates that the inner rings match h=2 for random non-constant inputs), the three iter-596/597/598 locks cover (a) interior preservation, (b) inner-ring consistency with h=2, (c) outer-ring rotation correctness, (d) outer-ring CONNECTIVITY correctness.  Only non-constant interpolation weights at the outermost ring remain unvalidated structurally; they are exercised by actual W2/W5 production tests at halo=2 (where they matter for the current model) and would be locked separately if/when halo=3 is wired into the dycore.

4 h=3 tests pass.  Production restored.

### Iter-599 (2026-04-20): axis-reversal + neighbour-strip correctness at h=3 outer ring

**Codex stop-time review on iter-598 (commit 8970b35)**: the face-unique-but-constant-per-face test CAN'T catch bugs in `is_reversed` handling, because reversing a constant-along-strip field gives back the same constant.  A refactor that dropped `strip = strip[::-1]` from `_pad_halo_local_h3` would pass iter-598 silently.

**Lock added**: `test_pad_halo_vector_halo3_outer_ring_axis_reversal` uses a field that varies along BOTH axes with face-unique offsets:
```
u_east[f, i, j] = 100*f + 0.5*i + 0.1*j     v_north = 0
```
With `interp_offsets=None` (nearest-cell copy), the outermost h=3 halo at each side pulls from the neighbour strip at depth=2 via CONNECTIVITY; the strip is reversed if `is_reversed=True`.

For each of 6 faces × 4 sides, the test:
1. Reads `(nbr_face, nbr_edge, is_reversed)` from `CONNECTIVITY[face][side]`.
2. Extracts the neighbour strip via `_extract_edge_strip_at_depth(..., depth=2)`.
3. Applies reversal if needed.
4. Computes expected grid-aligned output: `cos_angle_padded_h3 * strip` (and `-sin_angle_padded_h3 * strip` for v).
5. Asserts max diff < 1e-4 (accounts for `u_east ~ O(600)` max times float32 storage noise ~1e-7 → 6e-5 floor).

**Sanity-verified**: replacing `if is_reversed:` with `if False and is_reversed:` in `_pad_halo_local_h3:1005` fires the test on face=1 side=S with diff 0.547 — 5400× the 1e-4 ceiling.  This is the exact regression Codex worried about, and iter-598 would have missed it entirely.

**Coverage summary across iter-595/596/597/598/599** for the halo=3 vector path:
- (a) **interior preservation** (iter-595): trivially `up[:, 3:-3, 3:-3] == u_grid` to 1e-10.
- (b) **inner-ring h=3 ≡ h=2** (iter-596): random non-constant inputs agree at overlapping rings.
- (c) **outer-ring exact rotation** (iter-597): constant geographic wind predicts `cos_angle_padded_h3 * u_east`.
- (d) **outer-ring CONNECTIVITY** (iter-598): face-unique constants verify correct neighbour selection.
- (e) **outer-ring axis reversal** (iter-599): varying-along-strip field verifies `is_reversed` handling.

Production restored.  5 h=3 vector tests pass.  Only non-constant *interpolation-weight* correctness at the outer ring (which only matters when interp_offsets_h3 is used in a spatially-varying context) remains unvalidated — deferred until h=3 is actually wired into a dycore path.

### Iter-601 (2026-04-20): ordering lock — sync must precede divergence consumption (Constraint #1)

**Motivation**: audit of Critical Duogrid Constraint #1 flux synchronization coverage found that the existing `TestFluxSyncCallSitesWired` class verifies (a) each REQUIRED_SITES function calls `synchronize_cgrid_fluxes`, and (b) the call is gated on duogrid.  Missing: a check that the sync call happens LEXICALLY BEFORE the flux-divergence stencil that consumes the synced fluxes.  A silent refactor that moved the sync AFTER `h_new = h + (fx[:-1] - fx[1:] + ...)` would be a no-op on the current timestep — the stencil reads the UNSYNCED values — but would still pass (a) and (b).

**Lock added**: `test_flux_sync_call_lexically_precedes_divergence_consumption` in `tests/unit/test_duogrid.py::TestFluxSyncCallSitesWired`.  For each SAME_BODY_SITE (`cgrid_mass_flux_divergence`, `_c_sw`), the test:
1. Walks the function AST.
2. Finds every `synchronize_cgrid_fluxes` call lineno.
3. Finds every `BinOp(Sub)` where both sides are `Subscript(Name)` on the same name (flux-difference stencil) matching `fx`, `fy`, `flux_*`, `fx_*`, `fy_*`.
4. Asserts the earliest sync call lineno strictly less than the earliest consumption lineno.

`fv_tp_2d` is excluded from the SAME_BODY check because consumption happens in the CALLER (`transport_step`, `_d_sw_native`, etc.) — the function returns already-synced fluxes.

**Sanity-verified**: reordering the `_c_sw` sync call to AFTER the `h_star = h + (fx[:-1] - fx[1:] + ...)` computation fires the test with the explicit message "`synchronize_cgrid_fluxes` call at line 1254 happens AT/AFTER the earliest flux-difference consumption at line 1249".  Production restored; 108 `test_duogrid` tests pass.

**Constraint #1 coverage now spans three orthogonal locks**:
- **Site presence** (`test_each_required_site_calls_synchronize_cgrid_fluxes`): the call exists in each required function body.
- **Gate correctness** (`test_every_flux_sync_site_is_duogrid_gated`): the call is guarded by `duogrid`/`dg` condition.
- **Ordering** (`test_flux_sync_call_lexically_precedes_divergence_consumption`, THIS): the call precedes the divergence consumption.

All three must pass simultaneously for Constraint #1 to be structurally honored.

### Iter-602 (2026-04-20): widen ordering test to catch fy consumption (Codex)

**Codex stop-time review on iter-601 (commit 4951c49)**: the iter-601 ordering test used a narrow pattern — `BinOp(Sub)` with `Subscript` on BOTH sides on the same Name — which missed `_c_sw`'s `fy` divergence term.  The reason is Python parsing: `(fx[a] - fx[b] + fy[c] - fy[d])` parses as `((fx[a] - fx[b]) + fy[c]) - fy[d]`, so the OUTERMOST `BinOp(Sub)` has a `BinOp(Add)` on the left (not a Subscript).  The iter-601 test only detected the INNER `fx - fx` pattern and was blind to `fy - fy`.

**Fix**: widened detection to ANY `Subscript` of a rebinded flux name.  The test now:
1. Locates every `Assign` whose value is `synchronize_cgrid_fluxes(...)`.
2. Extracts the LHS tuple names (e.g., `(fx, fy)` or `(flux_x, flux_y)`) — these are the names the sync rebinds and the only names subsequent consumptions should use.
3. Finds all Subscript nodes whose base Name matches one of those LHS names.
4. Asserts every Subscript lineno ≥ sync call lineno.

This catches ANY pre-sync read of a rebinded flux array — including subscripts that are arguments to other operations, function-call arguments, or inside compound expressions.  A flux name is only rebinded by the sync, so any Subscript of that name has a meaningful consumption semantic.

**Sanity-verified**: inserting `_peek_fy = fy[:, :, 0]` immediately BEFORE the sync in `_c_sw` fires the test with message `"does NOT precede these Subscript consumptions of rebinded names: [('fy', 1249)]"`.  This is the exact regression class Codex flagged — the iter-601 test would have missed it.

**Constraint #1 ordering coverage** now detects both:
- Structural pattern violations (like the `fy` BinOp(Add) parse bug above).
- Direct pre-sync subscript reads of any rebinded flux name.

108 `test_duogrid` tests pass.  Production restored.

### Iter-603 (2026-04-20): close nested-scope escape hatch in ordering test (Codex)

**Codex stop-time review on iter-602 (commit 761f3c3)**: the iter-602 ordering test used `ast.walk(func)` which descends into nested `FunctionDef` / `Lambda` / comprehension scopes — the exact "iter-504 escape-hatch class" the surrounding tests explicitly close via the `_nested_scope_types()` helper.  A refactor that hid the sync call inside a `def _deadhelper():` that is never invoked would satisfy iter-602's test while leaving the direct-body consumption unsynced.

**Fix**: replaced `ast.walk` with a pre-order walk that tracks `inside_nested` using the class's existing `_nested_scope_types()` helper (covers `FunctionDef`, `AsyncFunctionDef`, `Lambda`, `GeneratorExp`, `ListComp`, `SetComp`, `DictComp`).  The test now:
1. Enumerates DIRECT-body nodes only (stopping at nested scopes).
2. Restricts sync-call detection to `Assign` nodes in the direct body.
3. Restricts `Subscript` consumption detection to the direct body.

**Sanity-verified**: relocating the `_c_sw` sync into a nested `def _deadhelper():` inside `_c_sw` fires the test with the explicit message "no `synchronize_cgrid_fluxes` call found inside an Assign in the DIRECT function body.  The call must be used as `(fx, fy) = synchronize_cgrid_fluxes(...)` at function scope (not inside a nested helper, lambda, or comprehension)."  Iter-602 would have PASSED this regression silently.

**Constraint #1 ordering coverage** now matches the nested-scope rigor of the surrounding tests:
- Site presence (iter-502/503): direct-body-only via `_iter_calls_in_function`.
- Gate correctness (iter-504): direct-body-only via `_nested_scope_types()`.
- **Ordering** (iter-601/602/603, THIS): direct-body-only via the same `_nested_scope_types()` helper.

No divergence between tests' nested-scope handling.  108 `test_duogrid` tests pass; production restored.

### Iter-604 (2026-04-20): polar-cap v-wind artifact — root cause isolated

**User-reported visible artifact** on W2 C36 1-day `snapshots_v.png`: strong alternating ±0.3 m/s patches at high latitudes (|lat| > 75°) visible from t=0.5d onwards — dominates over the mid-latitude mode-4 cube-face imprint analyzed in iter-592.

**Diagnostic** on the saved `snapshots_latlon.npz` + `snapshots_native.npz`:
1. **Polar-cap signal is 3× the mid-latitude mode-4**: at t=1d, `rms|v|` for |lat|>75° reaches 0.092 m/s with max 0.303 m/s.  Mid-latitude (±30°) mode-4 at the same time was 0.053 m/s.
2. **Face-by-face at t=1d**: faces 0-3 (equatorial) have max|v_cc_north| ≈ 0.087 m/s.  Faces 4 (north polar) and 5 (south polar) have max|v_cc_north| = 0.307 m/s — **3.5× the equatorial faces**.  N-S mirror-symmetry (face 4 vs face 5 reflected): 1.1% relative — NOT the old polar-axis bug.
3. **FFT at lat=±85°** shows dominant mode-2 (amplitude 0.20 m/s at t=1d), not mode-4.  Mode-4 at pole is 0.06.  This is different from the ±30° signature (mode-4 dominant).
4. **Root cause localized to the 4 pole-adjacent cells of face 4** (and face 5 by symmetry).  Face 4 has n=36 cells; the north pole sits at the MID-CELL INTERSECTION (i, j) ∈ {17, 18} × {17, 18}.  At t=1d:
   - Cells (17,17), (18,18): u_east_actual = 1.178 m/s vs expected 1.191 m/s; v_north_actual = +0.229 m/s.
   - Cells (17,18), (18,17): u_east_actual = 1.332 m/s vs expected 1.191 m/s; v_north_actual = −0.195 m/s.
   - **Checkerboard pattern**: u_east error = ±0.14 m/s (12% relative), v_north = ±0.23 m/s at 4 cells at lon = ±45°, ±135°.
5. **The 4-cell checkerboard at 4 quadrant longitudes produces mode-2 in longitude FFT** at fixed high latitude (alternating +,−,+,− around the circle = cos(2λ)).  This matches the observed mode-2 dominance at lat=±85°.

**Mechanism**: the cubed-sphere discretization places the pole at the intersection of 4 face-4 cells.  Each cell's grid-local orientation differs by π/2 from its neighbours (gnomonic angles collapse at the pole).  The dycore's momentum integration produces grid-aligned `(u_cc, v_cc)` that are not quite consistent between the 4 cells — a checkerboard velocity error accumulates at ~O(dx²·u₀/R·t).  When diagnostically rotated to geographic `(u_east, v_north)` via cell-centre angles, the checkerboard becomes a visible mode-2 polar-cap artifact.

**This is DISTINCT from the mid-latitude mode-4 cube-face imprint** (iter-592): mid-latitude is at face seams (lat ≈ ±30°), polar is at the pole singularity inside face 4/5.  Both are production A-L + RK3 path artifacts; neither eliminates within the non-FV3 framework.

**W5 C36 comparison**: face 4 max|v_cc_north| = 7.75 m/s (physical Rossby wave dominates), face 5 = 0.31 m/s.  The 7.75 is mostly the W5 mountain-induced wave signal on face 4; the pole-cell 4-checkerboard adds ~0.15 m/s on top.  Face 5's 0.31 m/s is consistent with the pure polar-cell artifact.

**Actionable fix paths** (architectural, not iterable in one Ralph iteration):
- (a) Replace A-L + RK3 with FV3 FB chain (needs ng=3 halo — iter-595..603 unblocked the infrastructure).  The c_sw + d_sw boundary handling at polar faces specifically treats the pole singularity via cube-vertex corner overrides (`sw_corner`, `ne_corner` etc. in `sw_core.F90:3528-3545`).  Python's current `_d2a2c_vect` is architecturally bound from representing these (see Priority 3 history).
- (b) Add a pole-cell diagnostic smoother (non-FV3) that averages the 4 pole-adjacent cells — similar pattern to the `boundary_fix` at face seams.  Would eliminate the visible artifact at cost of diagnostic-only smoothing.
- (c) Accept and lock the magnitude, matching the iter-592 pattern for mid-latitude.

**Iter-604 deliverable**: diagnostic breakdown + root-cause localization (above).  No code fix committed — the fix choice is an architectural decision.  The analysis provides the grounds to revisit Priority 3 (polar-face cube-vertex overrides) once ng=3 halo is wired into the dycore.

### Iter-605 (2026-04-20): regression lock for W2 pole-cell magnitude

**Motivation**: iter-604 identified the polar-cap v artifact as an inherent A-L + RK3 pole-singularity feature.  Without an architectural fix, the magnitude can only get worse with future dycore tweaks (hyperdiff coefficient drift, halo refactors, limiter changes).  Iter-605 adds a regression lock that catches any future amplification.

**Lock added**: `test_w2_pole_cell_v_north_ceiling_at_1day` in `TestW2CubeFaceImprintCharacterization`.  Runs canonical W2 C36 dt=300s 1-day integration, extracts face-native `v_cc_north` via the 4-edge-angle helper, and asserts:
1. `max|v_cc_north|` on face 4 < 0.40 m/s (iter-604 baseline: 0.307 m/s; ~25% headroom).
2. Face 4 / face 5 max|v| N-S mirror symmetry: relative diff < 5e-3 (currently machine precision — the pole artifact is N-S symmetric by grid topology).

**Sanity-verified**: tightening the ceiling to 0.25 m/s fires the test with the observed 0.307 m/s measurement — confirms the lock catches amplifications within ~20% of the current baseline.  Production restored to 0.40 m/s ceiling.

**Cost**: ~11 seconds per run at C36 dt=300s 1day (one JAX trace + 288 time steps).  3 `TestW2CubeFaceImprintCharacterization` tests pass.

**Coverage status for user-visible W2 artifacts**:
- t=0 diagnostic-angle bound: `test_w2_t0_v_north_diagnostic_angle_bounded` (iter-592).
- N/S mirror symmetry after 10 steps: `test_w2_short_run_v_north_N_S_mirror_symmetry` (iter-592).
- Pole-cell magnitude after 1 day: `test_w2_pole_cell_v_north_ceiling_at_1day` (iter-605, THIS).

These three locks together cover the top-three user-visible failure modes (pre-iter-26 diagnostic bug, pre-iter-505 PPM polar asymmetry, and the current pole-cell artifact).  A new failure mode would appear visually in the snapshots and the test matrix would continue to PASS — but the magnitude of that specific failure mode would be outside the locked ceilings.  Future iterations can then add targeted locks for any new failure mode discovered.

### Iter-606 (2026-04-20): field-level mirror-symmetry fix for iter-605 (Codex)

**Codex stop-time review on iter-605 (commit 6efd914)**: the "N-S mirror symmetry" assertion in `test_w2_pole_cell_v_north_ceiling_at_1day` compared scalar `max|v|` between face 4 and face 5 only.  Two completely different field patterns can share the same scalar max while being non-mirror-images — a scalar comparison cannot detect that regression class.

**Fix**: replaced the scalar-max comparison with a FIELD-LEVEL check identical in convention to iter-592's `test_w2_short_run_v_north_N_S_mirror_symmetry`:
```
field_diff = max|v_north[4] - (-v_north[5][:, ::-1])|
rel = field_diff / max|v_north[4]|
assert rel < 3e-2
```

The mirror convention `-v[:, ::-1]` is the one iter-592 validated at the 10-step mark, and the 1-day baseline measures `rel = 1.139e-2`.  Ceiling 3e-2 allows ~2.6× headroom.

**Sanity-verified**: tightening the ceiling to 1e-3 fires the test with measured 1.139e-2 — confirms field-level comparison sees the actual pattern difference (the scalar maxima were identical at 0.307 m/s both faces and would have PASSED iter-605's original assertion regardless).  Production ceiling restored to 3e-2.

**Scalar-max vs field-level distinction**: iter-605 baselined face 4 max = face 5 max = 0.3068 (identical to 4 sig figs from grid-topology mirror).  If a hypothetical future bug introduced a perturbation that preserved face-4 max but DISTORTED its shape (e.g., moved the peak to a different cell), the old scalar-max assertion would not fire.  The new field-level comparison fires if the shape itself drifts, not just the max magnitude.

3 W2-characterization tests pass.  Coverage for user-visible W2 artifacts now uses field-level comparisons throughout.

### Iter-607 (2026-04-20): W5 face-4 polar magnitude regression lock

**Motivation**: iter-604 measured W5 C36 1-day face-4 max|v_cc_north| = 7.75 m/s (mountain-induced Rossby wave on face 4 + pole-cell artifact), face-5 = 0.31 m/s (pure pole-cell, same as W2).  Without a lock, a dycore regression that amplifies polar-face response (numerical dispersion, wrong hyperdiff scaling, broken pole-singularity handling) would go undetected.

**Lock added**: `test_w5_face4_v_north_ceiling_at_1day` in new `TestW5PolarFaceMagnitude` class.  Runs canonical W5 C36 dt=300s 1-day integration and asserts:
1. `max|v_cc_north|` on face 4 < 10.0 m/s (baseline 7.75; ~29% headroom).
2. `max|v_cc_north|` on face 5 < 1.0 m/s (baseline 0.31; ~3× headroom).

The two ceilings probe different physics:
- Face 4 ceiling catches Rossby-wave amplification (polar-face nonlinear dispersion from mountain forcing).
- Face 5 ceiling catches spurious wave propagation, face-to-face reflection, or pole-cell artifact amplification in a quiet polar cap.

**Sanity-verified**: tightening the face-4 ceiling to 5 m/s fires with measured 7.75 m/s.  Production restored to 10.0 ceiling.

**Cost**: ~12 seconds per run at C36 dt=300s 1-day (one JAX trace + 288 time steps).

**User-visible artifact coverage now spans W2 + W5**:
- `test_w2_t0_v_north_diagnostic_angle_bounded` (iter-592): t=0 angle helper.
- `test_w2_short_run_v_north_N_S_mirror_symmetry` (iter-592): 10-step N/S mirror.
- `test_w2_pole_cell_v_north_ceiling_at_1day` (iter-605/606): 1-day W2 pole-cell magnitude + field-level mirror.
- `test_w5_face4_v_north_ceiling_at_1day` (iter-607, THIS): 1-day W5 polar-face magnitude (both face 4 active wave + face 5 quiet cap).

**Rejected Codex suggestion**: the proposed `qsmith` moist-pressure correction in `src/legoesm/thermo.py` does not apply to the current Python callers.  Fortran `qsmith` is a free-atmosphere saturation formula that takes `q_v` as input (microphysics-internal); Python callers (land slab, multilayer land, simple ocean) compute SURFACE saturation where no q_v is available — they need a (T, p) formula.  Adopting the Fortran formula would make Python less accurate in context.  Documented for future reference if a microphysics scheme is ported.

### Iter-608 (2026-04-20): W5 Rossby-wave signal floor (Codex)

**Codex stop-time review on iter-607 (commit b442dea)**: the W5 face-4 lock has only an UPPER bound (`f4_max_v < 10.0 m/s`).  A regression that over-damps the mountain-induced Rossby wave (excessive hyperdiff, wrong mountain forcing, broken vorticity-gradient coupling) could drop the face-4 signal from 7.75 m/s to e.g. 0.5 m/s and the test would PASS silently — the Rossby wave is the PHYSICAL signal of W5 and its loss is a fidelity regression.

**Fix**: added a lower-bound assertion `f4_max_v > 4.0 m/s` (~50% of the 7.75 m/s baseline).  A drop below 4 m/s indicates the wave has been substantially damped or its generation has been broken.

**Sanity-verified**: tightening the floor to 8.0 fires with measured 7.75 m/s.  Production restored to 4.0.

**Bounded face-5 intentionally one-sided**: face 5 has only an upper bound (< 1.0 m/s) because its baseline of 0.31 m/s is a pure pole-cell artifact.  A drop to zero there would be an IMPROVEMENT (the artifact is gone), not a regression — so there's no corresponding floor.

**Physical vs artifactual signal distinction**: the iter-608 two-sided bound on face 4 expresses a fidelity claim — W5's Rossby wave IS a physical signal that the dycore must produce AND must not amplify.  Face 5 is the opposite: a non-physical pole-cell artifact that we tolerate but never want to grow.

4 user-visible-artifact regression tests pass (3 W2 + 1 W5).  Coverage class hierarchy now:
- W2 diagnostic-angle / N-S mirror / 1-day pole magnitude + mirror (iter-592/605/606).
- W5 mountain-wave preservation + pole-cell ceiling (iter-607/608).

