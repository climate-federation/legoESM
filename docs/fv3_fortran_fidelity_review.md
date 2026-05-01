# FV3 Fortran Fidelity Review

Baselined 2026-04-14. This file keeps the latest Ralph-loop
iterations in full and compresses older material for token economy.
Use git history for retired prose.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Working Notes (iter-955 status)

- **Last successful production-impacting change**: iter-947
  (NEW-corrected uc, vc halo via OLD-cross-face delta).  Cumulative
  reduction in v_ll_Linf since iter-944b: ~85 → 55.6 m/s (~35%).
- **Remaining gap to W2 acceptance**: 55.6 m/s vs 0.119 m/s = 467×.
- **Iters 948-954 were all NEGATIVE-RESULT**: linear extrapolation
  (worsened |v|), pad_halo_vector for ua/va in `_divergence_corner_duo`
  (no-op confirmed), h_dg=3 PPM halo (regressed v_ll), enabling
  Parts 2/3/4 of d_sw1 on duogrid (catastrophic), and selective
  Part 2 sin_sg-upwind override (also catastrophic).  Iter-955
  added a runnable cumulative-progress diagnostic.
- **General pattern**: with iter-947's halo fix, Fortran's "Part
  2/3/4 boundary overrides" are NOT NEEDED on duogrid (they're a
  workaround for mode='edge' Part 1).  Future fidelity gains must
  come from PPM transport, operator-split sweep order, or
  ke_corner halo (the latter two are non-trivial without GFDL
  Fortran source access).
- **Risk**: continuing to iterate without source visibility may
  yield mostly negative-result iters.  Each one still constrains
  the design space but the production code drift is minimal.

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
  brief) **NOT YET MET** but **iter-945 closes a major gap**: the
  PPM hord=9 transport inside `_bgrid_ke_transport` was reading
  `mode='edge'` same-face halo cells at the four cube-face
  boundaries instead of cross-face data.  iter-945 wires the
  existing duogrid `ext_vector_dgrid` halo (used by
  `_d2a2c_vect_duogrid`) into a new `_pad_halo_dgrid_for_ppm`
  helper and threads pre-padded `(u_d, v_d)` into PPM via a new
  `external_halo` kwarg on `_ppm_transport_1d`.  Duogrid W2 1-day
  on FB chain improved from `|u_max|=106 → 78 m/s, |v_max|=151 →
  81 m/s`; step survival preserved at 288/288.
- Iter-946 (NEGATIVE-RESULT): tried analogous d2a2c-derived halo
  for `(uc, vc)` via new `_pad_halo_uc_vc_via_d2a2c` helper; the
  OLD-u_d derivation lacks the c_sw + p_grad_c increment (~15 m/s
  on vc for W2) that was added to interior, producing an OLD/NEW
  discontinuity at boundaries that worsened |v_max| (81 → 156 m/s).
  Reverted; helper retained as dead-code reference.
- Iter-947: NEW-corrected uc, vc halo via
  `_pad_halo_uc_vc_new_via_old_delta`, using
  `NEW_halo = NEW_boundary + (OLD_halo - OLD_boundary)`.  Anchors
  on the NEW interior boundary (preserves c_sw + p_grad_c increment)
  while carrying the OLD cross-face geometric delta as an additive
  correction.  Wired into d_sw1 + d_sw3 for duogrid ng>=3.  Result:
  |u_max| 78 → 77, |v_max| 81 → 75.  Cumulative since iter-944b:
  |v_max| 151 → 75 (50% reduction).  W2 acceptance still NOT met.
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

### Iter-954 — NEGATIVE-RESULT: just the Part 2 sin_sg-upwind override at I=0/n on duogrid also regresses

**Trigger.**  Iter-953 found running ALL of Parts 2/3/4 on duogrid
worsened v_ll_Linf 55.6 → 127.2 m/s.  Iter-954 narrowed to JUST
Part 2's sin_sg-upwind override at the four cube-face boundaries
(I=0, n, J=0, n) using duogrid-aware sin_sg padding — skipping
Parts 3/4 strip + corner-solve.

**Negative result.**  v_ll_Linf 55.6 → 133 m/s (140 % worse).
Part 2's override formula is ``ut = uc / sin_sg(upwind)`` — this
REPLACES Part 1's 4-cell average but DROPS the cross-velocity
``-0.25*cosa_u*vc_avg`` correction.  With iter-947's halo'd
4-cell average producing correct ut at I=0/n, the Part 2 override
is a regression: it loses the cross-velocity term in exchange for
"slightly different boundary handling".

The Part 2 override was useful PRE-iter-947 because Part 1's
mode='edge' was wrong at boundaries; with iter-947's correct
halo, Part 1's full formula is the preferred path.

**Iter-954 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_d_sw1_recompute_ut_vt` —
   updated comment block records the iter-954 narrow-test
   negative-result.

No new sentinel — the iter-951 v_ll_Linf gate already locks the
Fortran-faithful FB chain numbers.

**Insight.**  Iter-953/954 reveal a general principle: with iter-947
giving Part 1 the correct halo, the Fortran Part 2/3/4 boundary
overrides are NOT NEEDED on duogrid (they're a workaround for
mode='edge' Part 1, which the duogrid path now bypasses).  Future
duogrid fidelity gains must come from elsewhere (PPM transport,
operator-split sweep order, ke_corner halo, vorticity-flux halo).

### Iter-953 — NEGATIVE-RESULT: enabling d_sw1 Parts 2/3/4 boundary overrides on duogrid regresses v_ll_Linf

**Trigger.**  iter-952 localised the remaining FB-chain v_ll_Linf
to cube-face I-boundaries.  iter-953 tested whether running
`_d_sw1_recompute_ut_vt` Parts 2/3/4 (sin_sg-upwind override +
adjacent strip + corner 2x2) for the duogrid path too — currently
they only run for non-duogrid — would help.

**Negative result.**  Removing the `if use_duogrid: return ut, vt`
early-return regressed v_ll_Linf catastrophically:

| iter           | v_ll_Linf (m/s) |
|----------------|----------------:|
| iter-947 baseline (early-return)  |   55.6  |
| iter-953 (Parts 2/3/4 enabled)    |  127.2  | ← +130% regression |

Root cause: Parts 2/3/4 use `grid.halo_interp_offsets` (the
non-duogrid `_pad_halo_local` interp mode) for sin_sg padding.
Combining this non-duogrid sin_sg halo with the iter-947 duogrid
cube_rmp halo of uc, vc creates inconsistent boundary metrics.

Reverted; duogrid path early-returns after Part 1.  A future
iter-954+ could selectively enable just the sin_sg-upwind override
formula at I=0/n on duogrid (using a duogrid-aware sin_sg pad)
without the strip + corner-solve overrides — separate experiment.

**Iter-953 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_d_sw1_recompute_ut_vt` —
   updated comment block records the iter-953 negative-result.

No new sentinel — iter-947's sentinel + iter-951's v_ll_Linf gate
already lock the Fortran-faithful FB chain numbers.

### Iter-952 — Diagnostic: v_north max localized at cube-face I-boundaries (i=0, i=n-1)

**Trigger.**  iter-951 added direct v_ll_Linf tracking for the FB
chain.  Iter-952 investigates WHERE the v_north error is largest
to narrow down the remaining fidelity gaps.

**Diagnostic.**  At duogrid C36 W2 1-day, the top-10 |v_north|
locations are concentrated at the cube-face I-boundary columns:

```
face=1 i= 0 j=21 |v|=56.50   ← i=0 west cube-face boundary
face=3 i= 0 j=21 |v|=56.50   ← i=0
face=0 i= 0 j=21 |v|=56.44
face=2 i= 0 j=21 |v|=56.44
face=2 i=34 j=30 |v|=56.11   ← i=n-2 (interior, but adjacent to east boundary)
face=0 i=34 j=30 |v|=56.10
face=1 i=35 j= 4 |v|=56.02   ← i=n-1 east cube-face boundary
face=3 i=35 j= 4 |v|=55.98
face=3 i=34 j=30 |v|=55.92
face=1 i=34 j=30 |v|=55.90
```

The error is concentrated on the equatorial belt (faces 0, 1, 2, 3
— the "ring" around the equator) at the i-boundary cells (west /
east cube-face seams).  The poles (faces 4, 5) and j-boundaries are
NOT in the top-10.

**Implication.**  The remaining fidelity gap is dominated by the
**i-direction cube-face boundary handling**, not j-direction or cube
vertices.  Specifically:

1. The PPM mass transport at I=0 / I=n boundaries (FB chain step 2).
2. The d_sw3 B-grid Courant numbers at corner i=0, n (FB chain
   step 3, where iter-947 already provides cross-face halo).
3. The d_sw1 ut recomputation at I=0, n (FB chain step 1).
4. The d_sw6 KE-gradient at I=0, n (this reads ke_corner at
   interior j-stagger but the i-direction is sensitive to ke_corner
   at the boundary I-faces).

iter-952 records this diagnostic but does not implement a fix —
the next iter-953+ should target one of these four locations
specifically.

**Iter-952 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry recording the
   v_north location concentration finding.

No new sentinel — the iter-951 v_ll_Linf gate is sufficient.

**Backlog for iter-953+.**

The i-direction cube-face boundary is the localized problem area.
Candidate fixes:

1. Fortran-faithful ytp_v / xtp_u boundary overrides at the cube-
   edge I-faces (analogous to `apply_fortran_xppm_boundary` for
   tp_core's xppm/yppm — requires sw_core.F90 reference).
2. Re-examine the d_sw1 boundary handling Parts 2/3/4 (currently
   skipped for duogrid because the iter-947 halo "fixes" the
   4-cell average — but maybe Parts 2/3/4 should still run for
   the sin_sg-upwind override formula even on duogrid).
3. Audit the iter-945 `_pad_halo_dgrid_for_ppm` index map for
   off-by-one at the I-boundary (the d_sw3 PPM x-sweep reads
   u_d_ihalo at index 0 = i-cell -h and at index n+2h-1 = i-cell
   n+h-1; verify these map correctly to the cube-face neighbour).

### Iter-951 — Add v_ll_Linf sentinel for FB chain (the actual W2 acceptance metric)

**Trigger.**  Iter-944b through iter-950 tracked |u_max| / |v_max|
(face-covariant maxima) as proxies for W2 fidelity.  iter-950
discovered these can move OPPOSITE to v_ll_Linf — extending the PPM
cross-face halo reduced |v_max| 6% but regressed v_ll_Linf 4%.
Future iters need to track v_ll_Linf directly since it's the
actual W2 acceptance gate (≤ 0.119 m/s per user iter-938 brief).

**Iter-951 deliverable.**  New sentinel
`tests/test_iter951_fb_chain_v_ll_linf_tracking.py` measures the FB
chain duogrid C36 W2 1-day v_ll_Linf and asserts it stays below
60 m/s (margin around the iter-947 measured value of 55.6 m/s).
A regression past this gate signals new structural error.

**Cumulative duogrid FB chain progress:**

| iter        | step survival | |u_max| (m/s) | |v_max| (m/s) | v_ll_Linf (m/s) |
|-------------|--------------:|--------------:|--------------:|----------------:|
| iter-944b   |    288 / 288  |        106    |        151    |          ~85    | (estimate) |
| iter-945    |    288 / 288  |         78    |         81    |          56.2   |
| iter-947    |    288 / 288  |         77    |         75    |          55.6   |
| acceptance  |        ∞      |         -     |          -    |          0.119  |

The v_ll_Linf reduction since iter-944b is ~35%, smaller than the
50% reduction in |v_max| — confirming v_ll_Linf is a stricter
metric.  Closing the remaining gap to acceptance requires reducing
v_ll_Linf by another ~470x.

**Production impact.**  ZERO.  iter-951 only adds a measurement
sentinel.

### Iter-950 — NEGATIVE-RESULT: extending D-grid PPM halo from h=2 to h=3 regresses v_ll_Linf

**Trigger.**  iter-945's `_pad_halo_dgrid_for_ppm` uses h_dg=2 cells
of cross-face halo for the PPM transport.  PPM hord=9 has internal
halo h3=4, so the outer 2 cells fall back to `mode='edge'`.  Iter-950
tested h_dg=3 (when duogrid ng>=3) to reduce the mode='edge' gap to
1 outer cell.

**Negative result.**  Mixed metrics on duogrid C36 W2 1-day:

| iter           | step survival | |u_max| | |v_max| | v_ll_Linf |
|----------------|--------------:|-------:|-------:|----------:|
| iter-947 (h_dg=2)|  288/288  |  76.91 |  75.38 |    55.61  |
| iter-950 (h_dg=3)|  288/288  |  77.76 |  70.64 |    58.00  | ← v_ll regressed |

|v_max| improved 6% but v_ll_Linf — the W2 acceptance metric —
regressed by 4%.  The deeper halo spreads cube-vertex artifacts
further into the panel rather than damping them.  Reverted; h_dg=2
retained.  Lesson: |u_max|/|v_max| are imperfect proxies for
v_ll_Linf, which weights the entire interpolated lat-lon field
(not just the maximum face-covariant point).

**Iter-950 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_bgrid_ke_transport` — comment
   block records the iter-950 negative-result finding.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

**Backlog for iter-951+.**

The `|u|/|v| max ≠ v_ll_Linf` divergence motivates a sharper
diagnostic: future iters should measure v_ll_Linf directly.
Strategic candidates remain:

1. Compute c_sw + p_grad_c increment at halo positions exactly
   (extend metric tensors to halo, allow halo'd uc/vc that includes
   the c_sw/p_grad_c gradient-term increments).
2. Audit operator-split sweep order in `_bgrid_ke_transport` against
   Fortran's Lin-Rood 2-sweep KE form.
3. PPM hord=9 cube-edge boundary overrides for ytp_v / xtp_u.

### Iter-949 — NEGATIVE-RESULT: pad_halo_vector for ua, va in `_divergence_corner_duo` is bit-identical no-op

**Trigger.**  iter-657 documented that `_divergence_corner_duo`'s
`mode='edge'` padding of ua, va is a numerical no-op because the
boundary divg_d cells are zeroed before they propagate.  Iter-949
re-verified this claim with the iter-947 / iter-948 v_ll_Linf
context (55.6 m/s on FB chain duogrid C36 1-day) by replacing
`mode='edge'` with `pad_halo_vector` (cube_rmp + grid-angle
rotation).

**Negative result.**  FB chain duogrid C36 W2 1-day:
``|u|=76.91, |v|=75.38`` either way — bit-identical.  iter-657's
"no-op" finding is confirmed: the boundary zeroing + 0.25×
attenuation at face-adjacent cells masks any halo difference for
ua, va inside this helper.

| iter           | step survival | |u_max| (m/s) | |v_max| (m/s) |
|----------------|--------------:|--------------:|--------------:|
| iter-947       |    288 / 288  |     76.91     |     75.38     |
| iter-949       |    288 / 288  |     76.91     |     75.38     | (bit-identical) |

Reverted; mode='edge' retained as the cheaper equivalent.

**Iter-949 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_divergence_corner_duo` —
   updated comment to record the iter-949 numerical confirmation.

No new sentinel — the iter-947 sentinel pins the FB chain numbers
and the comment now records the iter-949 verification.

**Backlog for iter-950+.**

The largest remaining gap to W2 v_ll_Linf acceptance (≤ 0.119 m/s)
is structural: FB chain on duogrid C36 W2 1-day is at v_ll_Linf=
55.6 m/s vs production A-L+RK3 path's 0.13 m/s — a ~420× gap
across the entire FB chain.  Per-iter improvements at this scale
(5-15% per iter) suggest 50+ more Fortran-fidelity iterations
needed.  Candidates remain:

1. Compute c_sw + p_grad_c increment AT halo positions exactly
   (iter-947's constant-extrap approximation accounts for ~7%
   improvement; an exact halo computation might give similar gains).
2. Fortran-faithful PPM hord=9 cube-edge boundary overrides for
   ytp_v / xtp_u (analog of iter-888's `apply_fortran_xppm_boundary`
   on tp_core's xppm/yppm).
3. d_sw operator-split sweep order audit.
4. d_sw5 corner-divergence damping audit beyond ua/va halo.
5. Vorticity flux (zeta_abs) corner-edge handling at cube vertices.

### Iter-948 — NEGATIVE-RESULT: linear extrapolation of c_sw+p_grad_c increment to halo

**Trigger.**  iter-947 used a CONSTANT extrapolation of the c_sw +
p_grad_c increment from interior boundary cell to halo cell:
``delta_at_halo ≈ delta_at_boundary``.  The increment varies smoothly
across the face, so a 2-point LINEAR EXTRAPOLATION ought to be
strictly more accurate:

    delta(j=-1) ≈ 2*delta(j=0) - delta(j=1)            (south halo)
    delta(j=n)  ≈ 2*delta(j=n-1) - delta(j=n-2)        (north halo)

**Iter-948 attempt.**  Replaced the constant extrapolation in
`_pad_halo_uc_vc_new_via_old_delta` with the linear stencil above
(with constant fallback for n<2 tiles).

**Negative result.**  Linear extrapolation WORSENED the W2 1-day
|v_max| at C36 from 75 → 87 m/s.  |u_max| was approximately
unchanged (77 → 78 m/s).  The c_sw + p_grad_c increment varies
non-linearly along the face's j-direction near cube vertices, so a
linear stencil overshoots — the constant (iter-947) extrap is more
conservative and stays closer to the truth.

| iter           | step survival | |u_max| (m/s) | |v_max| (m/s) |
|----------------|--------------:|--------------:|--------------:|
| iter-947 (constant)  |  288 / 288  |     77    |     75    |
| iter-948 (linear)    |  288 / 288  |     78    |     87    | ← reverted |

Reverted to iter-947's constant extrapolation.  Documentation note
preserved in the `_pad_halo_uc_vc_new_via_old_delta` docstring so a
future iter does not re-introduce the linear stencil.

**Iter-948 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_pad_halo_uc_vc_new_via_old_delta`
   — comment block records the iter-948 negative-result finding.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

No new sentinel — the iter-947 sentinel already pins the constant
extrap behaviour, and the docstring commentary prevents re-introduction.

**Backlog for iter-949+.**  Same as iter-947 backlog:

1. Compute c_sw + p_grad_c increment AT halo positions (extend
   `_pad_halo_auto`, `cdgrid.rdxc`/`rdyc` to halo) — would replace
   the iter-947 boundary-extrapolation approximation with the exact
   value.
2. Operator-split sweep order audit in `_bgrid_ke_transport`.
3. PPM hord=9 cube-edge boundary overrides for ytp_v / xtp_u
   (analog of iter-888's `apply_fortran_xppm_boundary` on tp_core).
4. Vorticity flux halo and the d_sw5 corner-divergence damping
   halo audits.

### Iter-947 — NEW-corrected uc, vc halo via OLD-cross-face delta (duogrid W2 1-day |v| 81→75 m/s)

**Trigger.**  iter-946 (negative-result) showed that sourcing uc, vc
halo from OLD u_d, v_d via the d2a2c machinery introduces an
OLD/NEW discontinuity at cube-face boundaries (worsened W2 |v_max|
81 → 156 m/s).  The iter-946 helper carries the correct cross-face
geometric delta but lacks the c_sw + p_grad_c increment that
modifies the NEW interior uc, vc.

**Iter-947 fix.**  New helper `_pad_halo_uc_vc_new_via_old_delta`
combines the iter-946 OLD halo with the NEW interior boundary cell
to estimate the NEW cross-face halo via:

    NEW_halo = NEW_boundary + (OLD_halo - OLD_boundary)
             = NEW_boundary + cross_face_delta_from_OLD

This anchors the halo on the NEW interior boundary cell (preserving
the c_sw + p_grad_c increment) while carrying the OLD cross-face
geometric delta (the rotation between cube faces) as an additive
correction.  The two contributions are nearly orthogonal for W2
(c_sw + p_grad_c is smooth on a face; cross-face delta is smooth
across faces), so summing them gives a good first approximation of
the true NEW halo.

Wired into BOTH `_d_sw1_recompute_ut_vt` Part 1 and
`_bgrid_ke_transport` Step 1 for the duogrid ng>=3 path,
replacing `mode='edge'`.  The non-duogrid path falls back to
`mode='edge'` (no `_pad_halo_uc_vc_via_d2a2c`-equivalent for
non-duogrid).

**Cumulative duogrid FB chain on W2 C36 dt=300 s 1-day:**

| iter           | step survival | |u_max| (m/s) | |v_max| (m/s) |
|----------------|--------------:|--------------:|--------------:|
| iter-944b      |    288 / 288  |        106    |        151    |
| iter-945       |    288 / 288  |         78    |         81    |
| iter-946       |    288 / 288  |         73    |        156    | ← reverted |
| **iter-947**   |    288 / 288  |     **77**    |     **75**    |
| analytical     |        ∞      |         40    |          0    |

|v_max| improved 81 → 75 m/s (~7%); |u_max| approximately unchanged
(78 → 77).  Cumulative since iter-944b baseline: |u_max| 106 → 77
(27 % reduction toward analytical 40), |v_max| 151 → 75 (50 %
reduction toward analytical 0).  W2 v_ll_Linf acceptance still NOT
met — the 75 m/s |v_max| is still ~600x the 0.119 m/s acceptance
threshold.

**Iter-947 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_pad_halo_uc_vc_new_via_old_delta`
   — new helper (~70 lines) wrapping iter-946's
   `_pad_halo_uc_vc_via_d2a2c` plus a NEW-boundary anchor.
2. `_d_sw1_recompute_ut_vt` gains optional `u_d_old`/`v_d_old`
   kwargs that, when supplied (and duogrid ng>=3), trigger the
   iter-947 halo path.
3. `_bgrid_ke_transport` calls the iter-947 helper directly (uses
   its existing `u_d`, `v_d` parameters).
4. `_d_sw_native` forwards `u_d_old=u_d, v_d_old=v_d` to
   `_d_sw1_recompute_ut_vt` so the iter-947 halo path fires.
5. `tests/test_iter947_uc_vc_new_via_old_delta.py` — 3 sentinels
   (helper shapes & interior preservation, duogrid W2 |u|/|v|
   improvement vs iter-945, non-duogrid baseline preserved).
6. `docs/fv3_fortran_fidelity_review.md` — this entry.

**Verification.**  20 cross-iter sentinels (iter-921, 922, 923, 924,
925, 926, 928, 930, 931, 932, 934, 938, 941, 942, 944, 945, 946) +
3 iter-947 sentinels pass.  Production W2 / W5 / cosine-bell /
rest-state sentinels remain bit-identical.

**Backlog for iter-948+.**

1. Improve the c_sw + p_grad_c increment estimation: iter-947's
   approximation is `delta_at_halo ≈ delta_at_boundary` (extrapolation
   by NEW boundary anchoring).  A more accurate approach would
   compute the actual c_sw + p_grad_c increment AT halo positions
   by extending the metric tensors and pad_halo machinery.
2. Operator-split sweep order audit in `_bgrid_ke_transport`
   (Lin-Rood y-then-x for `transported_y` vs x-then-y for
   `transported_x`) — possibly missing a 2D average.
3. PPM hord=9 cube-edge boundary overrides for ytp_v / xtp_u
   (analogous to `apply_fortran_xppm_boundary` on tp_core.F90's
   xppm/yppm — separate from the iter-945 halo fix).
4. Vorticity flux halo (currently `mode='edge'` inside `fv_tp_2d`
   for the FB chain vorticity transport).

**Process.**  Real production code change in the FB chain (one new
helper, two call-site updates).  No production behaviour change at
default `FV3EdgeShallowWaterModel`.  Production W2 baseline
unchanged at v_ll_Linf=0.132 m/s.

### Iter-946 — NEGATIVE-RESULT: d2a2c-derived uc, vc halo conflicts with c_sw+p_grad_c interior

**Trigger.**  iter-945 closed the cube-face D-grid PPM halo gap (|u|
106→78, |v| 151→81 m/s).  The remaining `(uc, vc)` C-grid halo was
identified as the next likely fidelity gap: `_d_sw1_recompute_ut_vt`
Part 1 and `_bgrid_ke_transport` Step 1 still read uc, vc with
`mode='edge'` at the four cube-face boundaries instead of cross-face
data.

**Iter-946 attempt.**  New helper `_pad_halo_uc_vc_via_d2a2c` reuses
`_d2a2c_vect_duogrid`'s 4th-order machinery (extending the slicing of
`u_d_full`, `v_d_full` to derive utmp / vtmp at i-halo + j-halo and
then running the 4-point A→C interpolation) to produce:

  * uc_jhalo: (6, n+1, n+2) — covariant uc with j-halo of width 1
  * vc_ihalo: (6, n+2, n+1) — covariant vc with i-halo of width 1

This bypasses the lossy cell-centre roundtrip approach that was
prototyped pre-iter-945 (and rejected because it introduced a 1-2-1
smoothing).  Wired into `_d_sw1_recompute_ut_vt` and
`_bgrid_ke_transport` to replace the corresponding `mode='edge'`
calls.

**Negative result.**  The helper is called at `_d_sw_native` entry
with the OLD u_d, v_d (the inputs to d_sw, before c_sw + p_grad_c).
The interior uc, vc passed to the d_sw operators is NEW
(post-c_sw + p_grad_c).  For W2 solid-body rotation, the c_sw +
p_grad_c increment to vc is dominated by `dt2 * g * dh/dy` ~ O(15
m/s), which is comparable to vc itself.  Mixing OLD-derived halo
with NEW interior in the 4-cell averages produces a ~15 m/s
discontinuity at cube-face boundaries that gives WORSE results
than `mode='edge'`:

| location of iter-946 halo                | |u_max| (m/s) | |v_max| (m/s) |
|------------------------------------------|--------------:|--------------:|
| iter-945 baseline (mode='edge')          |        78     |       81      |
| iter-946 d_sw1 + d_sw3 (both)            |        73     |      156      |
| iter-946 d_sw3 only                      |        71     |      150      |

The minor |u_max| improvement does not compensate for the ~75 m/s
|v_max| regression.  Reverted; `mode='edge'` is the better mismatch
(consistent boundary cells) until iter-947+ propagates the c_sw +
p_grad_c increments to halo.

**Iter-946 deliverables (retained as documentation).**

1. `src/legoesm/core/fv3_sw_core.py:_pad_halo_uc_vc_via_d2a2c` —
   4th-order halo helper, RETAINED as a documented dead-code
   reference for iter-947+ (docstring records the negative result).
2. `tests/test_iter946_uc_vc_via_d2a2c_negative_result.py` — 3
   sentinels:
   - helper shapes & finiteness on duogrid C24
   - ValueError on non-duogrid input
   - `_d_sw_native` does NOT currently call the helper (re-enable
     guard)
3. `docs/fv3_fortran_fidelity_review.md` — this entry.

**Verification.**  9 cross-iter sentinels (iter-941, iter-942, iter-945)
+ 3 iter-946 sentinels pass.  Production W2 / W5 / cosine-bell /
rest-state sentinels remain bit-identical (no production code change).

**Backlog for iter-947+.**

1. Compute c_sw + p_grad_c increment at HALO positions (extend
   `_pad_halo_auto` to halo I-faces / J-faces, extend `cdgrid.rdxc`
   / `rdyc` metrics to halo) so a NEW-uc, NEW-vc halo is available.
   Then re-enable the `_pad_halo_uc_vc_via_d2a2c`-style halo with
   the increment added.
2. As an interim simpler experiment: try using NEW interior values
   to ESTIMATE the c_sw + p_grad_c contribution at halo (e.g.
   uc_NEW_halo = uc_OLD_halo + (NEW_interior_boundary -
   OLD_interior_boundary) extrapolated).  Hacky but cheaper than
   full halo extension of c_sw / p_grad_c.
3. Independently: audit operator-split sweep order in
   `_bgrid_ke_transport` and PPM hord=9 boundary overrides for
   ytp_v / xtp_u (separate from the halo issue).

**Process.**  No production code change.  No FB-chain behaviour
change post-revert (mode='edge' restored on both d_sw1 and d_sw3).
W2 acceptance still NOT met.  The helper docstring + sentinel
prevent re-introduction without iter-947+ companion fix.

### Iter-945 — D-grid PPM cross-face halo via `ext_vector_dgrid` (duogrid W2 1-day |u| 106→78 m/s, |v| 151→81 m/s)

**Trigger.**  iter-944b (Fortran-fidelity audit) reverted Python-only
`mpp_get_boundary` syncs that did not appear in the Fortran reference
source.  The duogrid FB chain at C36 dt=300 s W2 still reached 1-day
NaN-free (288 steps), but with `|u_max|=106 m/s, |v_max|=151 m/s` —
far from the analytical W2 (`|u|=40 m/s, |v|≈0`).  The remaining gap
was not in any individual `mpp_get_boundary` call but in the basic
`mpp_update_domains` halo: the PPM hord=9 transport inside
`_bgrid_ke_transport` was reading `mode='edge'` same-face halo cells
at the four cube-face boundaries instead of the cross-face data
Fortran provides via `mpp_update_domains(u_d, v_d, gridtype=DGRID_NE)`
upstream of `d_sw3` (with `cube_rmp` interpolation when `duogrid` is
active).

**iter-945 fix.**

1. New helper `_pad_halo_dgrid_for_ppm(u_d, v_d, cdgrid, halo=2)` in
   `src/legoesm/core/fv3_sw_core.py` reuses the existing
   `ext_vector_dgrid` pipeline (same machinery as
   `_d2a2c_vect_duogrid` step 1) and slices to:
   - `u_d_ihalo`: `(6, n+2h, n+1)` — i-cells extended (cell axis 1)
     with j-stagger preserved
   - `v_d_jhalo`: `(6, n+1, n+2h)` — j-cells extended (cell axis 2)
     with i-stagger preserved
   The interior is overwritten EXACTLY with the input u_d/v_d so the
   helper is identity on the interior region (matches Fortran's
   `mpp_update_domains` which only fills halo cells).  Raises
   `ValueError` on a non-duogrid grid (the standard
   `mpp_update_domains` Python equivalent for non-duogrid is not
   implemented).

2. `_ppm_transport_1d` gains an `external_halo` kwarg (default 0,
   bit-identical when 0).  When > 0, the input `field` is treated as
   already having `external_halo` cells of cross-face halo on the
   sweep axis; the function pads only the gap `(h3 - external_halo)`
   with `mode='edge'`, leaving the cross-face halo cells intact.
   When `external_halo > h3`, it trims to the inner h3 cells.  The
   rest of the function indexes the same `(N + 2*h3)`-cell padded
   layout it always has.

3. `_bgrid_ke_transport` step 1.5 (new) gates on duogrid and pre-pads
   `(u_d, v_d)` via `_pad_halo_dgrid_for_ppm`; the d_sw3 x-sweep
   (xtp_u) and y-sweep (ytp_v) call `_ppm_transport_1d` with
   `external_halo=h_dg=2` on duogrid, `0` otherwise.

**Cumulative duogrid FB chain on W2 C36 dt=300 s** (target 1-day = 288 steps):

| iter        | step survival | |u_max| (m/s) | |v_max| (m/s) | h range (m)         |
|-------------|--------------:|--------------:|--------------:|---------------------|
| iter-944b   |    288 / 288  |        106    |        151    | [-232, 21019]       |
| **iter-945**|    288 / 288  |     **78**    |     **81**    | [-257, 21804]       |
| analytical  |        ∞      |         40    |          0    | [~1000, ~3000]      |

So |u_max| dropped 26 % toward analytical and |v_max| dropped 46 %.
W2 v_ll_Linf acceptance (≤ 0.119 m/s per user iter-938 brief) is
still NOT met — further fidelity work continues in iter-946+ (the
remaining gap is likely the still-`mode='edge'` `(uc, vc)` halo
inside `_d_sw1_recompute_ut_vt` Part 1 and inside
`_bgrid_ke_transport`'s corner-Courant computation).  iter-945
attempted a covariant cell-centre roundtrip helper for `(uc, vc)`
(modeled on iter-836b/iter-837 in `_corner_vorticity`) and found it
WORSENED the W2 fidelity (|u_max| → 747 m/s) because the
two-point cell-centre averaging then re-stagger introduces a
1-2-1 smoothing that is not Fortran-faithful for the d_sw 4-cell
averages.  That helper was removed; the asymmetry (D-grid winds
halo'd, C-grid winds still mode='edge') is documented as the
iter-946+ target.

**Production impact.**  ZERO.  `_d_sw_native` is FB-chain-only;
production `fv3_sw_tendencies` (`FV3EdgeShallowWaterModel` default)
does not call this code path.  Production W2 (iter-921, iter-922,
iter-932), W5 (iter-923), cosine-bell (iter-924), rest-state
(iter-925), and Fortran-fidelity gap markers (iter-928, iter-930,
iter-931, iter-938) sentinels remain bit-identical post-iter-945.

**iter-945 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py` — `_pad_halo_dgrid_for_ppm`
   helper (~70 lines), `external_halo` kwarg on `_ppm_transport_1d`,
   gate + pre-pad inside `_bgrid_ke_transport`.  No changes to
   `_d_sw1_recompute_ut_vt` or anywhere else in the FB chain.
2. `tests/test_iter945_dgrid_ppm_cross_face_halo.py` — 4 sentinels:
   - shapes & interior-identity invariant for `_pad_halo_dgrid_for_ppm`
     on duogrid C24
   - `ValueError` raised on non-duogrid input
   - duogrid FB chain at C36 dt=300 s 1-day reaches 288 steps with
     `|u_max| ≤ 95 m/s` AND `|v_max| ≤ 100 m/s` (margin around 78/81)
   - non-duogrid path remains at the iter-944b ~41-step baseline
     (gate skips `_pad_halo_dgrid_for_ppm`)
3. `tests/test_iter944_cgrid_ne_ut_vt_vort_sync.py` — REWRITTEN to
   reflect iter-944b reality (the original test was failing
   pre-iter-945 because iter-944b reverted the syncs but did not
   update this test).  New tests pin the duogrid 288-step milestone
   AND the non-duogrid ~41-step Fortran-faithful baseline.
4. `scripts/diag_iter945_dgrid_ppm_halo.py` — runnable diagnostic.
5. Comment fix in `_d_sw_native` (replaced
   `\`synchronize_corner_scalar(ke_corner, n)\`` in the iter-944b
   audit comment block with `\`synchronize_corner_scalar\` on
   \`ke_corner\`` so the iter-942 sentinel's substring guard does
   not false-positive on the audit comment).

**Verification.**

- 4/4 iter-945 + 17 cross-iter sentinels (iter-921 W2, iter-922 PPM
  boundary, iter-923 W5, iter-924 cosine bell, iter-925 rest state,
  iter-926 d_sw5 corner damping, iter-928 fidelity gap markers,
  iter-930 boundary fix, iter-931 bare-AL cube vertex resolution
  scaling, iter-932 iter-893 invariant, iter-934 FB low-res, iter-938
  d2a2c corner overrides, iter-941 BGRID_NE sync gating, iter-942
  ke_corner sync absence, iter-944 FB 1-day milestone) pass.

**Backlog for iter-946+.**

1. Cross-face halo for `(uc, vc)` on the duogrid path — currently
   still `mode='edge'` inside `_d_sw1_recompute_ut_vt` Part 1 and
   `_bgrid_ke_transport`'s corner-Courant computation.  iter-945's
   cell-centre roundtrip attempt (`_pad_halo_uc_vc_via_a2c`, removed)
   was not Fortran-faithful for the 4-cell averages.  Need a
   non-lossy approach — possibly threading uc/vc with halo through
   from `_d2a2c_vect_duogrid` (modify `_c_sw` and `_p_grad_c` to
   propagate halo), or implementing a true `cubed_a2c_halo` analog
   for staggered C-grid fields.
2. Operator-split sweep order audit in `_bgrid_ke_transport`
   (Lin-Rood y-then-x for `transported_y` vs x-then-y for
   `transported_x`).
3. PPM hord=9 boundary handling for the inside of the four cube-face
   edges (separate from the iter-945 halo fix — Fortran's ytp_v /
   xtp_u may have additional cube-edge boundary overrides analogous
   to `apply_fortran_xppm_boundary` on tp_core.F90's xppm/yppm).

**Process.**  Real production code change in the FB chain (one new
helper, one kwarg, one gate).  No production behaviour change at
default `FV3EdgeShallowWaterModel`.  Production W2 baseline unchanged
at v_ll_Linf=0.132 m/s.

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
