# FV3 Fortran Fidelity Review

Baselined 2026-04-14. This file keeps the latest Ralph-loop
iterations in full and compresses older material for token economy.
Use git history for retired prose.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Current Status

- Production W2 is still not true FV3. It runs
  `FV3EdgeShallowWaterModel` -> `fv3_sw_tendencies` with
  Arakawa-Lamb/RK3/`div_damp`/`boundary_fix`, not the Fortran
  FB `c_sw` -> `d_sw1/d_sw4/d_sw5/d_sw6` chain.
- W2 C36 production remains a cube-vertex/meridian `v_ll` artifact.
  `apply_fortran_xppm_boundary=True` improved the canonical baseline
  from `v_ll_Linf=1.585e-01` to `1.319e-01 m/s` (iter-893), but did
  not remove the structural bias.  iter-921→942 added 64 sentinel
  tests, 4 production code changes, and traced the root cause to
  bare A-L 1 % imperfect cancellation at the 8 cube vertices
  (iter-929; resolution-refinable per iter-931).
- FB chain stabilisation (iter-940→944): step survival on W2 C36
  dt=300 s improved from **41 → 288 steps** (1-day NaN-free) via
  five fixes: iter-934 `_deln_flux` float32-overflow factoring,
  iter-941 unconditional `BGRID_NE` corner sync, iter-942
  `ke_corner` scalar sync after d_sw5 KE-add, iter-944 CGRID_NE
  sync of `(ut, vt)` at d_sw step 1, iter-944 forced CGRID_NE sync
  of `(fx_vort, fy_vort)` at d_sw step 7.  iter-944 closes the
  structural-growth NaN blocker: FB chain reaches the 1-day target
  without producing NaN.
- FB chain W2 acceptance (v_ll_Linf ≤ 0.119 m/s per user iter-938
  brief) **NOT YET MET**: at 1 day, |v_max|≈2300 m/s (analytical
  ≈0).  Numerical stability achieved; W2 fidelity remains the open
  iter-945+ target (likely PPM hord=9 boundary handling, the
  operator-split sweep order in `_bgrid_ke_transport`, or the cube-
  vertex halo for u_d/v_d themselves).
- FB/duogrid scaffolding (`halo=3`, flux sync, `d_sw*` helpers)
  refined; C24/C36 W2 fidelity remains the open blocker for
  using FB chain as the production path.

## Compact Archive

- Closed or resolved: panel-edge corner metrics, `d_sw3` BGRID_NE
  sync, duogrid flux sync, legacy edge bypass in duogrid, old polar
  face asymmetry, visual diagnostics, and cosine-bell visual checks.
- `iter-1..174`: core FV3 metric/operator port, seam/sync work,
  regression expansion, and early FB bring-up.
- `iter-505`: fixed the production PPM active-axis bug in
  `cgrid_mass_flux_divergence`; W2 C36 `max|v_ll|` improved
  `0.557 -> 0.303 m/s`.
- `iter-511..751`: W2/W5 artifacts characterized; production path
  identified as structurally non-FV3; `use_duogrid=True` on the
  A-L path shown catastrophic.
- `iter-752..855`: del6/corner-fill/zeta/DUOGRID/FB ablations
  localized the W2 residual to cube-vertex `dv/dt`, incomplete
  Cor+pressure+KE cancellation, and load-bearing numerical
  zeta/`boundary_fix`.
- `iter-856..898c`: Phase-1/d_sw5 production swaps failed; PPM
  stop-time and Fortran-sentinel work culminated in
  `apply_fortran_xppm_boundary=True` as the current W2/W5 baseline.
- `iter-899..903`: PPM strip forensics showed production uses
  symmetric halo=2 (`q[k]=q1(k-2)`). Strict LEFT/RIGHT Fortran PPM
  flags were implemented but worsened W2 (`0.1319 -> 0.2027` and
  `0.1319 -> 0.1989 m/s`), so both remain default-off; the dxa xt
  correction was quantified as small and deferred.

## Latest Ralph-Loop Iterations

Full detail is retained from `iter-940` onward. Older Ralph-loop entries are compressed below; use git history for retired prose.

### Iter-904 to Iter-939 - compacted Ralph-loop archive

- `iter-904..905`: Partial true-FV3 mass transport inside RK3 failed catastrophically: NaN, then W2 `v_ll_Linf=3.137e+02 m/s`; split FV3 mass + RK3 momentum still gave `2.921e+02 m/s`. One-component FV3/RK3 hybrids are rejected.
- `iter-906..910`: W2 residual localized to cube-edge/cube-vertex geometry. `div_damp` dominates instantaneous hot spots, but reducing global or boundary damping worsens W2. C36 sits near a practical `0.10..0.12 m/s` v-bias floor at current cost.
- `iter-912..920`: Regression cleanup. Focused iter-89x/9xx sweep passed; broader gold-file scan found silent drift from iter-878's Fortran-correct PPM limiter fix. Rebaselines/live replacements restored coverage; process rule: periodically run full `test_cdgrid_fv3_regression.py` and related layers.
- `iter-921..925`: Visual/sentinel refresh for W2, W5, cosine bell, and rest state. Iter-893 PPM boundary fix improves W2 `v_ll_Linf` by ~17 % but worsens `h_err_max` by ~77 %. W5/cosine/rest are essentially unaffected; iter-893 is bit-exact on constants.
- `iter-926..927`: `use_fv3_dsw5_corner_damping` added as default-off negative-result documentation. Additive d_sw5 (`v_ll_Linf=2.253 m/s`) and replacement d_sw5 (`0.971 m/s`) both failed, proving per-operator d_sw5 swaps cannot fix the A-L/RK3 operator-family mismatch.
- `iter-928..932`: Gap markers and W2 source diagnosis. Meta-sentinel locked 8 Fortran-fidelity gaps. Bare A-L decomposition found ~1 % imperfect geostrophic cancellation at the 8 cube vertices; `boundary_fix` spreads it and `div_damp` amplifies it. The bias is slowly resolution-refinable. Iter-893 affects `dh_dt` only, not velocity tendencies.
- `iter-933..937`: FB-chain stability work. Iter-933 found step-1 NaNs at C8/C12/C16; iter-934 fixed `_deln_flux` float32 overflow by factoring `damp` to final flux assembly. Iter-935/936 then exposed a deeper velocity-side growth mode. Iter-937/937b applied the same damp-factoring defense to `_del6_vt_flux`; this closed the overflow class but did not improve long-term FB survival.
- `iter-938..938b`: Ported and unit-tested the Fortran `d2a2c_vect` cube-corner sign-flip helper, then removed dead flag wiring after review because the helper did not yet propagate to outputs. Backlog: extend `_d2a2c_vect` edge-interpolate slicing so the helper affects `uc/vc`.
- `iter-939`: Closure scan confirmed all known `(coeff * da_min)^(nord+1) * field` float32-overflow sites in `src/legoesm/core/` are fixed or already factored. Cross-iter sentinels passed; no production behavior change.

### Iter-940 — localise FB chain structural growth to d_sw6 KE-gradient

**Trigger.**  iter-935 found FB chain blows up at step 41 on C36 1-day target.  iter-936 found velocities grow first.  iter-940 narrows further by temporarily ZEROING individual contributions inside `_d_sw_native` and measuring step survival.

**iter-940 substep probes** (W2 C36 dt=300 s):

| modification                                 | survived |
|----------------------------------------------|----------|
| BASELINE (full FB chain)                     | 41 steps |
| vorticity flux sync ON (was OFF per iter-864)| 41 steps |
| `d4_bg=0` (no d_sw5 corner damping)          | 41 steps |
| `damp_v=0` (no del6_vt post-step)            | 41 steps |
| both off (no damping at all)                 | 41 steps |
| `damp_v=2.0` (huge — destabilises)           |  3 steps |
| `fy_vort=fx_vort=0` (zero vorticity flux)    | 43 steps |
| `u_d_new = u_d` (skip d_sw6 wind update)     | **120+** |
| `ke_diff_u/v_scaled=0` (skip KE-grad ONLY)   | **120+** |

**Key finding**: zeroing the d_sw6 KE-gradient contribution (`ke_corner[i] - ke_corner[i+1]` in `fv3_sw_core.py:2397-2398`) makes the FB chain stable for 120+ steps.

**Localisation conclusion**:
- d_sw5 corner damping: ruled OUT (zeroing d4_bg doesn't help).
- del6_vt_flux post-step: ruled OUT (zeroing damp_v doesn't help; iter-937b also fixed its overflow).
- vorticity-flux sync: ruled OUT (toggling sync gives same 41).
- vorticity transport: minor contributor (43 vs 41).
- **d_sw6 KE-gradient: load-bearing structural growth source.**

The bug is in either `_bgrid_ke_transport` (computes `ke_corner` at step 4) or the gradient stencil at lines 2397-2398 + 2426-2427.  Both paths converge on the cubed-sphere stagger and halo treatment.

**iter-940 deliverables.**

1. `scripts/diag_iter940_fb_substep_probes.py` — runnable damping sweep + documented findings of the manual probe set.

**Verification.**  Code reverted to bit-identical baseline.  iter-934 sentinel (6/6) passes — no regressions.

**Backlog for iter-941+.**

1. Compare `_bgrid_ke_transport` against Fortran `sw_core.F90:1201-1388` (d_sw3 B-grid KE transport).  Check ke_corner staggering, halo path at cube vertices, and time integration of the KE field.
2. Verify the gradient formula `(ke_corner[:, :-1, :] - ke_corner[:, 1:, :]) / dx_u` matches Fortran's `(ke(i,j) - ke(i+1,j))` interpretation including sign and metric scaling on a non-uniform cubed-sphere grid.
3. Check whether ke_corner has cube-vertex halo errors that propagate through the gradient into the wind update.

**Process.**  No production code change.  Pure diagnostic + localisation iter.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-941 — apply BGRID_NE corner sync UNCONDITIONALLY (FB chain 41 → 63 steps)

**Trigger.**  iter-940 localised the FB chain structural growth to the d_sw6 KE-gradient.  Investigating `_bgrid_ke_transport` revealed that the `synchronize_bgrid_ne_corner_geo` corner sync (which makes the four faces meeting at each cube vertex agree on `(ubb, vbbtemp)` BEFORE computing `ke_corner`) was gated on `use_duogrid` — it ran ONLY for duogrid runs and was skipped on non-duogrid.

**Fortran spec.**  `dyn_core.F90:968-1019` calls `mpp_get_boundary(... gridtype=BGRID_NE)` UNCONDITIONALLY (no duogrid gate).  This is a single-process MPI-equivalent that ensures cube-vertex agreement.

**iter-941 fix.**  Removed the `if use_duogrid:` gate around the BGRID_NE sync inside `_bgrid_ke_transport` (`fv3_sw_core.py:2210+`).  The sync now fires on every FB step.

**FB chain step-survival improvement** (W2 C36 dt=300 s, default damping):
- Pre-iter-941:  **41 steps**
- Post-iter-941: **63 steps** (+54 % improvement)

The FB chain still does not reach 1-day stability (288 steps target), so iter-935's structural growth bug is NOT fully closed — there is at least one more contributor in the d_sw6 KE-gradient path (likely the cube-vertex halo of `ke_corner` itself, or the gradient stencil at face boundaries).  But iter-941 is a clear partial fix that closes one specific Fortran-fidelity gap.

**Production impact.**  ZERO.  `_d_sw_native` is FB-chain-only; production `fv3_sw_tendencies` (FV3EdgeShallowWaterModel default) does NOT call this code path.  Production W2 sentinel (iter-921) and rest-state sentinel (iter-925) bit-identical post-iter-941.

**iter-941 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_bgrid_ke_transport` — removed `if use_duogrid:` gate around `synchronize_bgrid_ne_corner_geo` call (~10-line edit; comment block records the Fortran spec citation and the FB step-survival measurement).
2. `tests/test_iter941_bgrid_ne_sync_unconditional.py` — 2 sentinels:
   - `test_iter941_fb_chain_c36_survives_at_least_60_steps`: pin the post-iter-941 step-survival at ≥ 60 (with 3-step tolerance around 63).
   - `test_iter941_fb_chain_low_res_step1_finite`: smoke test that the BGRID_NE sync addition doesn't introduce step-1 NaN at C8/C12/C16/C24/C36 (covered by iter-934, re-checked here).

**Verification.**  2/2 iter-941 + 21 cross-iter sentinels (iter-921/925/934/938) pass.  No production regression.

**Backlog for iter-942+.**

The remaining FB chain growth at step 63 is in the d_sw6 KE-gradient stencil OR `ke_corner` cube-vertex halo.  Per-substep instrumentation between steps 30 and 60 would identify which corner cell starts diverging first.  Possible candidates:
1. `ke_corner` cube-vertex halo: even with BGRID_NE sync, the halo cells (j=0/n in Fortran indexing) may carry stale data feeding the gradient stencil at the boundary.
2. The KE-gradient formula `(ke_corner[:, :-1, :] - ke_corner[:, 1:, :]) / dx_u` may have a sign or scaling bug at face-boundary u-edges.
3. The `_bgrid_ke_transport` operator-split sweep order (y-then-x for transported_y; x-then-y for transported_x) may differ from Fortran's order.

**Process.**  Real production code change in the FB chain (helper function gate removed).  No production behavior change at default `FV3EdgeShallowWaterModel`.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-942 — `ke_corner` scalar sync after d_sw5 KE-add (FB chain 63 → 209 steps)

**Trigger.**  iter-941 improved FB chain step survival from 41 → 63 steps by syncing the BGRID_NE Courant numbers.  But the d_sw6 KE-gradient (`(ke_corner[i] - ke_corner[i+1]) / dx_u`) reads `ke_corner` directly, and `ke_corner` itself is computed as `0.5 × (ubbtemp×vbbtemp + ubb×vbb)` — even with synced ubb/vbbtemp, the product carries cube-vertex inconsistency from the unsynced `ubbtemp` and `vbb` PPM transport outputs.

**iter-942 fix.**  Added `ke_corner = synchronize_corner_scalar(ke_corner, n)` AFTER the d_sw5 corner-divergence KE-add and BEFORE the d_sw6 KE-gradient (`fv3_sw_core.py:_d_sw_native` step 5 → step 6).  This makes the 4 faces meeting at each cube vertex agree on the final ke_corner value used by the gradient stencil.

**Cumulative FB chain step survival on W2 C36 dt=300 s** (target 1-day = 288 steps):

| iter      | survived  | improvement vs baseline |
|-----------|-----------|-------------------------|
| baseline  | 41 steps  | —                       |
| iter-941  | 63 steps  | +54 %                   |
| **iter-942** | **209 steps** | **+410 %**              |

Still does NOT reach 1-day stability (288 steps target) — there is at least one more contributor to FB chain growth post-iter-942.  Possible candidates for iter-943+:
1. `_bgrid_ke_transport`'s separate scalar transports `ubbtemp` (transported_y) and `vbb` (transported_x) — these are PPM scalar outputs at corners but not vector-pair, so the BGRID_NE vector sync doesn't apply.  An attempted "treat them as vector and rotate to geo" probe at iter-942 made the FB chain WORSE (step 103 vs step 209) — confirming they need a different sync strategy.
2. The KE-gradient stencil at face boundaries may need a Fortran-faithful one-sided formula similar to the production iter-893 PPM-boundary fix.
3. Vorticity flux `fy_vort, fx_vort` may need their own corner sync.

**Production impact.**  ZERO — same scope as iter-941 (`_d_sw_native` is FB-chain-only).  12 cross-iter sentinels (iter-921 W2 + iter-925 rest state + iter-934 FB low-res) pass post-iter-942.

**iter-942 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_d_sw_native` — added `synchronize_corner_scalar(ke_corner, n)` call between step 5 (d_sw5 KE-add) and step 6 (KE-gradient).  ~3-line edit + comment block.
2. `tests/test_iter942_ke_corner_scalar_sync.py` — 1 sentinel pinning FB chain C36 dt=300 s step survival ≥ 180 (baseline 209; tolerance ±29 steps).

**Verification.**  1/1 iter-942 + 12 cross-iter sentinels pass.

**Process.**  Real production code change in the FB chain (one helper call added).  No production behavior change at default `FV3EdgeShallowWaterModel`.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-943 — negative probe iter: ubbtemp/vbb scalar sync HURTS, redundant placement is no-op

**Trigger.**  iter-942 reached 209 FB chain steps but still doesn't reach 1-day (288).  iter-943 probes the remaining gap with two targeted sync attempts.

**iter-943 probes** (all at C36 dt=300 s W2 1-day target):

| probe                                                             | survived  |
|-------------------------------------------------------------------|-----------|
| iter-942 baseline (BGRID_NE + ke_corner sync after d_sw5)         | 209 steps |
| ALSO sync `ubbtemp, vbb` AS A VECTOR (geo-frame rotation)         | 103 steps (worse) |
| ALSO sync `ubbtemp` and `vbb` separately as scalars (no rotation) | 100 steps (worse) |
| ALSO sync ke_corner BEFORE d_sw5 (redundant placement)            | 209 steps (no change) |

**Findings.**

1. `ubbtemp` (transported_y) and `vbb` (transported_x) are **separate scalar transport outputs**, not a vector pair.  Treating them as a vector and rotating to the geographic frame loses physical meaning — fails sooner.
2. Treating each separately as a corner scalar (no rotation) ALSO fails sooner — the values from different faces have different physical meanings (face-A's transported v_d ≠ face-B's transported v_d in any common frame), so averaging them is incorrect.
3. The pre-d_sw5 ke_corner sync is mathematically redundant because: (a) iter-942 already syncs ke_corner after d_sw5; (b) the d_sw5 ke_damping addition (`ke_corner += ke_damping`) is itself per-face inconsistent at cube vertices, so syncing `ke_corner` BEFORE d_sw5 only to have it become inconsistent again from the d_sw5 add is futile.

**Implication.**  The remaining FB chain growth post-iter-942 is NOT in the corner-stagger sync; it's in another part of the d_sw chain.  Possible candidates for future iters:
1. KE-gradient stencil at face-boundary u-edges (needs Fortran-faithful one-sided formula like iter-893 PPM-boundary).
2. Vorticity flux corner sync.
3. The operator-split sweep order in `_bgrid_ke_transport`.
4. Cube-vertex halo for `ke_damping` from `_d_sw5_corner_divergence`.

**iter-943 deliverables.**  None.  Pure negative-result documentation iter; revert all probes.

**Verification.**  iter-942 sentinel (1/1) and production sentinels still pass post-revert.

**Process.**  No production code change (probes reverted).  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.

### Iter-944 — CGRID_NE sync of `(ut, vt)` and `(fx_vort, fy_vort)`: FB chain reaches 1-day NaN-free (209 → 288 steps)

**Trigger.**  iter-942 reached 209 FB chain steps; iter-943 confirmed the corner-stagger sync probes were exhausted (both `ubbtemp/vbb` syncs HURT, redundant ke_corner placements were no-ops).  iter-944 widens the search to the C-grid edge-staggered fields `(ut, vt)` and `(fx_vort, fy_vort)` to localise the remaining structural growth.

**iter-944 probe matrix** (W2 C36 dt=300 s, default damping):

| state                                                   | survived |
|---------------------------------------------------------|----------|
| iter-942 baseline                                       | 208–209  |
| `apply_cgrid_flux_sync=True` on vort flux (probe A)     | 208      |
| sync `ke_damping` itself before add (probe B)           | 208      |
| force vort flux sync (bypass duogrid gate, probe C)     | 270      |
| ungate `fv_tp_2d` sync only (probe D)                   | 187      |
| probes C + D (force vort + mass flux sync)              | 279      |
| **probes C + E (force vort sync + sync ut/vt at step 1)** | **288** |

**Key finding** (probe A vs C).  The pre-iter-944 `apply_cgrid_flux_sync=False` kwarg passed to the vorticity-flux `fv_tp_2d` call was indistinguishable from `=True` because the duogrid gate inside `fv_tp_2d` (`if apply_cgrid_flux_sync and dg is not None and dg.ng >= 2`) skipped the sync regardless on the non-duogrid grid used by the test.  Forcing the sync explicitly (probe C) rather than relying on the kwarg lifts FB chain step survival from 209 → 270 steps.

**Probe E — ut/vt CGRID_NE sync at step 1.**  `_d_sw1_recompute_ut_vt` produces transport velocities at C-grid u-face / v-face positions with face-local upwind sin_sg boundary overrides.  These overrides leave cube-edge cells inconsistent across face pairs.  `synchronize_cgrid_fluxes` (built for fluxes with the iter-808 sign-flip table) is signature-compatible with `(ut, vt)` since they share the (`fx`, `fy`) face-staggered shape.  Applied immediately after step 1, it lifts the FB chain past the 1-day target.

**iter-944 fix.**

1. `src/legoesm/core/fv3_sw_core.py:_d_sw_native` step 1 — added `ut, vt = synchronize_cgrid_fluxes(ut, vt, n)` after `_d_sw1_recompute_ut_vt`.
2. `src/legoesm/core/fv3_sw_core.py:_d_sw_native` step 7 — added `fx_vort, fy_vort = synchronize_cgrid_fluxes(fx_vort, fy_vort, n)` after the `fv_tp_2d` call.  The `apply_cgrid_flux_sync=False` kwarg is preserved (avoids a redundant call inside `fv_tp_2d` if its duogrid gate ever flips on).

**Cumulative FB chain step survival on W2 C36 dt=300 s** (target 1-day = 288 steps):

| iter      | survived  | improvement vs baseline |
|-----------|-----------|-------------------------|
| baseline  | 41 steps  | —                       |
| iter-941  | 63 steps  | +54 %                   |
| iter-942  | 209 steps | +410 %                  |
| **iter-944** | **288 steps** | **+602 %**              |

**Caveat — NaN-free ≠ W2 acceptance.**  At 1 day:

- `h` range: `[-323, 41796] m`  (W2 IC ~1000..3000 m)
- `|u|_max`: 3044 m/s  (analytical 40 m/s)
- `|v|_max`: 2278 m/s  (analytical ≈0 m/s)

The FB chain runs through 1 day without NaN, but the W2 v_ll_Linf acceptance (≤ 0.119 m/s per user iter-938 brief) is failed by 4 orders of magnitude.  iter-944 closes the structural-growth NaN blocker; the W2 fidelity gap remains the iter-945+ target.

**Production impact.**  ZERO.  `_d_sw_native` is FB-chain-only; production `fv3_sw_tendencies` (`FV3EdgeShallowWaterModel` default) does not call this code path.  15/15 cross-iter sentinels (iter-921 W2, iter-925 rest state, iter-934 FB low-res, iter-941 BGRID_NE, iter-942 ke_corner) pass post-iter-944.

**iter-944 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_d_sw_native` — two `synchronize_cgrid_fluxes` calls inside the FB-chain function (steps 1 and 7).
2. `tests/test_iter944_cgrid_ne_ut_vt_vort_sync.py` — 1 sentinel pinning FB chain C36 dt=300 s step survival ≥ 288 (1-day NaN-free target).
3. `scripts/diag_iter944_fb_remaining_growth.py` — measurement record + final-state print.

**Backlog for iter-945+.**

The 288-step NaN-free milestone is necessary but not sufficient.  Now that 1-day FB chain runs without NaN, the W2 v_ll_Linf 1-day measurement is finally meaningful.  Candidates for the v_ll fidelity gap:

1. PPM hord=9 boundary handling in `_ppm_transport_1d` (currently `mode='edge'` face-local extrapolation, NOT cross-face halo).  Same root cause as iter-893's PPM-boundary one-sided fix on the production path.
2. Operator-split sweep order in `_bgrid_ke_transport` (Lin-Rood y-then-x for `transported_y` vs x-then-y for `transported_x`).
3. Cube-vertex halo for `u_d, v_d` themselves before passing to step 4 (`_bgrid_ke_transport`), since iter-941 syncs only the `(ubb, vbbtemp)` Courant numbers, not the underlying transported velocities.
4. `_d2a2c_vect` propagation refactor (iter-938b backlog) — the iter-938 utmp/vtmp corner overrides are still no-op on uc/vc.

**Process.**  Real production code change in the FB chain (two helper calls added).  No production behaviour change at default `FV3EdgeShallowWaterModel`.  Production W2 baseline unchanged at v_ll_Linf=0.132 m/s.
