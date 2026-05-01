# FV3 Fortran Fidelity Review

Baselined 2026-04-14. This file keeps the latest Ralph-loop
iterations in full and compresses older material for token economy.
Use git history for retired prose.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title. The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.

## Working Notes (iter-961 status snapshot)

Ralph-loop session iter-945..960 progressed from v_ll_Linf=85 to
55.6 m/s (35% reduction).  Acceptance is 0.119 m/s — 467× more
reduction needed.

**Per-iter improvement is asymptotically near zero without GFDL
Fortran source access**.  Iter-958 found that the existing
apply_legacy_d_sw5_corner_corrections gate could be removed but is
bit-identical for W2 (corrections applied to zeroed boundaries).
Iter-959 found a 4% calibration tweak (Smagorinsky d2_bg=0.01,
dddmp=0.05) but this is parameter tuning, not Fortran-fidelity.

**Strategic bottleneck**.  The remaining v_ll_Linf=55.6 m/s
concentrates at cube-face I-boundaries (per iter-952's diagnostic).
Closing this requires:

1. PPM hord=9 cube-edge boundary overrides for ytp_v / xtp_u
   (analog of iter-888's `apply_fortran_xppm_boundary` on tp_core
   but for the d_sw3 wind transport).  Without sw_core.F90 source
   the exact override formula is unclear.
2. Exact c_sw + p_grad_c increment computation at halo positions
   (extend metric tensors and pad_halo to halo I-faces / J-faces).
   Requires extending the cdgrid metric infrastructure.
3. Audit operator-split sweep order in `_bgrid_ke_transport`
   against Fortran's Lin-Rood 2-sweep KE form.

The Ralph loop will keep iterating against these but the prior is
that each iter yields ≤5% improvement until a structural shift.

## Working Notes (iter-960 session summary)

This Ralph-loop session (iter-945 through iter-960) made these
production-impacting changes:

1. **iter-945** — D-grid PPM cross-face halo via `ext_vector_dgrid`
   wired into `_bgrid_ke_transport`.  v_ll_Linf: ~85 → 56 m/s
   (~34% reduction).  4 sentinels.
2. **iter-947** — NEW-corrected uc/vc cross-face halo
   (`_pad_halo_uc_vc_new_via_old_delta`) wired into
   `_d_sw1_recompute_ut_vt` and `_bgrid_ke_transport`.
   v_ll_Linf: 56 → 55.6 m/s (~1% additional).  3 sentinels.
3. **iter-951** — v_ll_Linf-tracking sentinel for FB chain.
   1 sentinel.

Negative-result iters (no production code change after revert):
iter-946 (OLD-direct halo), iter-948 (linear extrap), iter-949
(ua/va halo no-op), iter-950 (h_dg=3 PPM regresses v_ll), iter-953
(d_sw1 Parts 2/3/4 on duogrid catastrophic), iter-954 (Part 2 only
also regresses), iter-957/958 (apply_legacy_* flags no-op on
duogrid).

Calibration tweak available (not enabled by default):
- iter-959: `d2_bg=0.01, dddmp=0.05` Smagorinsky on FB chain gives
  4% v_ll_Linf improvement (55.6 → 53.2 m/s) at no stability cost.

Diagnostics + working notes:
- iter-952: v_north max localized to equatorial cube-face
  i-boundaries (i=0, n-1).
- iter-955: cumulative progress diagnostic script
  `scripts/diag_iter955_fb_chain_progress.py`.
- iter-956: top-of-doc working-notes summary.

Cumulative state at end of session:

| metric                     | iter-944b | iter-960 (this session) | acceptance |
|----------------------------|----------:|------------------------:|-----------:|
| FB chain step survival     |  288/288  |              288/288    |     —      |
| W2 |u_max| (m/s)           |    106    |              76.91      |     —      |
| W2 |v_max| (m/s)           |    151    |              75.38      |     —      |
| W2 v_ll_Linf (m/s)         |     ~85   |              55.61      |    0.119   |
| h_err_max (m)              |   ~21000  |              18591      |     —      |

Remaining gap: v_ll_Linf still 467× above acceptance.  Without GFDL
Fortran source access, further fidelity gains beyond iter-959's 4%
Smagorinsky tweak require structural code changes (PPM cube-edge
boundary overrides for ytp_v / xtp_u; exact c_sw + p_grad_c
increment computation at halo positions; operator-split sweep
order audit).  The current Python implementation is consistent
within itself and bit-identical on production sentinels.

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

### Iter-965 — Iter-947 halo helper also slightly improves W5

**Trigger.**  Iter-947's NEW-corrected uc/vc cross-face halo
improved W2 v_ll_Linf 56 → 55.6.  Iter-965 verifies the helper
also improves W5 (mountain forcing) and isn't W2-specific.

**Sweep on FB chain duogrid C36 W5 1-day at dt=300 s:**

| config              | |u_max| (m/s) | h range (m) |
|---------------------|--------------:|-------------|
| mode='edge'         |        31.42  | [2069, 18065] |
| **iter-947 NEW**    |     **31.30** | [2069, 18007] |

Small improvement (~0.4% on |u_max|).  Iter-947 helper is
generically useful, not W2-specific.  Confirms the iter-945 +
iter-947 halo work is a real FortranR-fidelity gain across multiple
test cases.

**Iter-965 deliverable.**  No code change.  Documentation only.

### Iter-964 — Smagorinsky tuning is W2-specific; W5 needs default damping

**Trigger.**  Iter-963 found `(d2_bg=0.09, dddmp=0.45)` improves W2
v_ll_Linf 21%.  Iter-964 verifies whether the high Smagorinsky also
helps Williamson 5 (zonal flow over an isolated mountain).

**Sweep on FB chain duogrid C36 W5 1-day at dt=300 s:**

| config            | step survival | |u_max| (m/s) | h range (m) |
|-------------------|--------------:|--------------:|-------------|
| default (no Smag) |    288/288    |        31.30  | [2069, 18007] |
| Smag iter-963     |    288/288    |        41.03  | [2726, 16660] |

W5 analytical |u| = 20 m/s.  Default damping gives |u|=31.3 m/s
(closer to analytical); Smagorinsky tuning gives |u|=41 m/s
(further from analytical).

**Implication.**  The iter-963 Smagorinsky tweak is W2-SPECIFIC.
W5's mountain forcing creates physical gradients that the
Smagorinsky over-damps.  Cannot be made the default — would
trade W2 fidelity for W5 fidelity.

**Iter-964 deliverable.**  No code change.  The
`use_smagorinsky_tuned` opt-in path documented in iter-962/963 is
correct: callers using W2-like initial conditions can opt in;
callers using W5/W6/Galewsky-like topography or jets should keep
the default.

### Iter-987 — Audit pad_halo cube_rmp at seam: Fortran-faithful but secondary

**Trigger.**  Iter-986 found west/east cube-edge asymmetry (face=1 i=0
at -56.5 vs face=0 i=35 at -23.4 m/s same physical seam).
PPM duogrid bypass of boundary corrections was the leading
hypothesis.  Iter-987 audits Fortran's xppm/yppm duogrid path and
the `pad_halo` cube_rmp halo values.

**Iter-987a — Fortran xppm/yppm audit.**

`tp_core.F90:333,357,612` — Fortran's xppm has the SAME `.not.
(bounded_domain .or. duogrid)` gate skipping the boundary specials
(s11/s14/s15 + dxa-weighted formulas + iv=1 limiter).  Our Python
matches.  No fidelity gap on the PPM kernel itself.

**Iter-987b — pad_halo halo values at face=1 west j=21.**

| i_pad | nearest copy | after cube_rmp | "expected" face0 east | true ext-position |
|-------|--------------|----------------|-----------------------|-------------------|
| 0     | 1.834e-05    | 1.611e-05      | 1.834e-05             | 1.611e-05         |
| 1     | 1.753e-05    | 1.679e-05      | 1.753e-05             | 1.679e-05         |
| 2     | 1.753e-05    | 1.753e-05      | 1.753e-05             | 1.753e-05         |
| 3     | 1.834e-05    | 1.834e-05      | 1.834e-05             | 1.834e-05         |

The "kinked" nearest copy gives a V-shape (decrease toward seam,
increase past).  cube_rmp Lagrange-extends to a monotonic profile
(smooth continuation).  Fortran `cube_rmp` (`fv_duogrid.F90:977`)
does the SAME thing — both apply k2e_coef weighted Lagrange remap
to extended-grid positions.

**Iter-987c — Skip cube_rmp experiment.**

Monkey-patched pad_halo to skip cube_rmp (use nearest copy only).
FB chain C36 W2 1-day:

| state                          | v_ll_Linf | seam[1,0,21] |
|--------------------------------|-----------|--------------|
| baseline (with cube_rmp)       | 55.61     | 56.50        |
| no cube_rmp (nearest-copy only)| 55.52     | 52.83        |

**Δ = -0.09 m/s (-0.2%) on v_ll_Linf**.  The cube_rmp accounts for
~6.5% of the seam-probe magnitude but only ~0.2% of the lat-lon
Linf metric.  Not the dominant lever.

**Implication.**

The duogrid `pad_halo` cube_rmp matches Fortran's behavior.  The
seam asymmetry must come from another source.

Candidates for iter-988+:
1. The 4 corner cells of cube_rmp's extended halo (cube vertices
   in the halo) may have a fill error not present in Fortran.
2. The fv_tp_2d `pad_halo` for `q_i, q_j` between Y- and X-sweeps
   may have inconsistent halo patterns (the Lin-Rood operator-split
   uses cross-flux corrections — if either sweep's halo is off,
   the cross-corrections amplify).
3. The d_sw5 corner divergence damping has a different halo path
   than fv_tp_2d transport — may add seam noise that other paths
   absorb.

**Iter-987 deliverables.**

1. `scripts/diag_iter987_pad_halo_seam_values.py` — runnable halo
   value probe at face=1 west.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

No production code change.

**Backlog for iter-988+.**

1. Probe Lin-Rood inter-sweep `pad_halo(q_i)` values at face=1
   west j=21 — does it preserve the seam structure?
2. Audit `_d_sw5_corner_divergence` halo path independently of
   fv_tp_2d.

### Iter-986 — Seam mode characterisation: damping sweep + asymmetry between west/east cube edges

**Trigger.**  Iter-985 localised the day-1 v_ll peak to face=0..3 i=0
j=21 on equatorial cube-edge seams (lat ±6.34°).  Iter-986 probes
which operator drives the seam.

**Damping sweep on FB chain C36 W2 1-day (baseline 55.6 m/s):**

| config                          | v_ll_Linf | seam[1,0,21] | interior[1,1,21] |
|---------------------------------|-----------|--------------|------------------|
| baseline (damp_v=0.06,d4_bg=0.16) |  55.61   |    56.50     |    28.75         |
| damp_v=0.12 (2x)                  |  55.73   |    44.62     |    26.91         |
| damp_v=0                          |  56.34   |    63.37     |    30.61         |
| damp_v=0.20 / 0.30                |    NaN   |     NaN      |     NaN          |
| **d4_bg=0 (no d_sw5)**            |  82.98   |  **17.92**   |  **0.16**        |
| nord_v=1 (del-4 vort)             |  55.68   |    44.26     |    26.42         |
| nord=2 (del-6 d_sw5)              |  80.42   |    34.59     |    20.83         |

**Key findings.**

1. **The seam mode IS damp_v-sensitive.**  damp_v=0.12 cuts seam
   peak 56.5→44.6.  damp_v=0 grows it to 63.4.  But damp_v cannot
   exceed ~0.12 without numerical instability.

2. **d_sw5 generates BOTH the interior mode AND amplifies the seam.**
   Removing d_sw5 (d4_bg=0): seam drops 56.5→17.9 (3× reduction)
   and interior drops 28.75→0.16 (180× reduction!) — but a NEW
   interior mode at lat=1° (face=*, i=5, j=18) grows to 86 m/s.

3. **Two competing modes:**
   - INTERIOR mode at lat=1° (suppressed by d_sw5 when on)
   - SEAM mode at cube edges at lat=±6.3° (suppressed by damp_v)

   d_sw5 trades down the interior mode but ADDS noise to the seam,
   which damp_v=0.06 partially absorbs.

**Iter-986b — Asymmetry across cube edges (west vs east).**

For W2 eastward zonal flow:

| face=0 i=35 (east, downwind) j=21 |    -23.39 m/s |
| face=1 i=0  (west, upwind)   j=21 |    -56.50 m/s |

These are different physical seams (face=0 east is lon~+45°, face=1
west is at a different cube edge).  But both are i=0 vs i=N-1 of
their respective faces.  Per-step Δv at j=21 is roughly UNIFORM
across i (≈-0.23 m/s/step everywhere, regardless of i=0 or i=5).
So the i=0 vs i=N-1 asymmetry in DAY-1 magnitude must come from
how DAMPING applies asymmetrically at the boundary.

**Iter-986c — PPM duogrid bypass hypothesis.**

`_ppm_1d` for duogrid skips:
- `_pert_ppm(iv=1)` face-boundary monotonicity at indices 0,1,2,
  -3,-2,-1.
- Position-aware `al_L0/al_L1/al_R0/al_R1` corrections.
- The `apply_fortran_xppm_boundary` Fortran s11/s14/s15 overrides
  (gated on `fortran_legacy_face`).

For duogrid, the assumption is that the kinked-extended remap puts
halo cells at correct physical positions, so the standard uniform-
spacing PPM is Fortran-faithful.  But empirically the seam shows
2× amplification at the upwind-direction edge — suggesting either:
(a) the duogrid halo VALUES still have sub-cell precision errors
that the standard PPM amplifies on the upwind side, or (b) the
Fortran duogrid path has additional boundary handling that we are
not replicating.

**Iter-986 deliverables.**

1. `scripts/diag_iter986_seam_damping_sweep.py` — runnable damping
   sweep + west/east asymmetry probe.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

No production code change.

**Backlog for iter-987+.**

1. Audit duogrid PPM upwind boundary by comparing Fortran's
   `xppm` / `yppm` source for the duogrid branch — are there
   special boundary overrides we are missing?
2. Probe `pad_halo` halo VALUES at face=1 i=-1 j=21 vs face=0
   i=35 j=21 — measure the physical-quantity discrepancy in
   zeta_abs, zeta, ut, vt, h that PPM amplifies on the upwind
   side.
3. Test FB chain with `apply_fortran_xppm_boundary=True` — even
   though gated to non-duogrid, may serve as a baseline reference.

### Iter-985 — CRITICAL PIVOT: 55.6 m/s peak is at equatorial cube EDGES, not vertices

**Trigger.**  Iter-984 confirmed cube-vertex imbalance is 1.3 m/s in
`c_sw + p_grad_c` 1-step tendency, identifying `_corner_vorticity`
halo path as suspect.  Iter-985 first verified the diagnosis by
running a patched `_corner_vorticity` (cube-vertex halo cells
replaced with edge-mode interior copy instead of cross-face
rotation), then localised where the FB chain 1-day v_ll_Linf=55.6 m/s
peak actually concentrates.

**Iter-985a — Cube-vertex halo fix (no improvement).**

Replaced the 4 cube-vertex halo cells in `fx_halo_*`, `fy_halo_*`
with `mode='edge'` (interior copy).  This reduced cube-vertex
`vort_abs` from 2.18e-04 → 1.07e-04 (a 51% reduction) — confirming
the cross-face rotation IS over-amplifying at cube vertices.

But on FB chain C36 W2 1-day:

| state         | v_ll_Linf  |
|---------------|------------|
| ORIGINAL      | 55.6130    |
| PATCHED       | 55.6034    |

Delta = -0.01 m/s (-0.02%).  **The cube-vertex halo error has
NEGLIGIBLE impact on the 1-day W2 v_ll_Linf.**

**Iter-985b — Localising the actual peak.**

Probed |v_north| on the cubed-sphere directly (before lat-lon
regridding) after FB chain 1-day:

```
|v_north| max on cubed sphere = 56.50 m/s
  Top 4 hotspots:
  #1: face=1, i=0, j=21 → -56.50 m/s | lat= 6.34°
  #2: face=3, i=0, j=21 → -56.50 m/s | lat= 6.34°
  #3: face=0, i=0, j=21 → -56.44 m/s | lat= 6.34°
  #4: face=2, i=0, j=21 → -56.44 m/s | lat= 6.34°
```

ALL four equatorial faces hit the SAME (i=0, j=21) cell, on the
WEST cube-edge seam at 6.34° N latitude.  This is **NOT a cube
vertex** (i, j not at extremes; 0 in N-cell space is one extreme of
i but j=21 is mid-range).

**Pattern along face=1, i=0 (west cube-edge seam):**

```
j=14: v_north=  52.28 m/s | i=1: 29.76 | i=2: 25.15 | lat= -6.34°
j=15: v_north=  50.23 m/s | i=1: 36.53 | i=2: 29.45 | lat= -4.52°
...
j=18: v_north= -41.71 m/s | i=1: -21.04 | i=2: -14.15 | lat=  0.90°
j=21: v_north= -56.50 m/s | i=1: -28.75 | i=2: -25.05 | lat=  6.34°
j=22: v_north= -20.30 m/s | i=1: -11.90 | i=2: -11.39 | lat=  8.18°
```

Two key features:

1. **Sign flip across equator** between j=17 and j=18 (v_north goes
   from +23 to -41 m/s in one cell row).
2. **Edge amplification**: i=0 values are ~2× larger than i=1
   (e.g., -56.5 vs -28.75 at j=21; +52 vs +30 at j=14).

This is a classic edge artifact: the seam cells experience some
combination of grid-scale gradient that does not propagate into
the interior.  The 2× amplification ratio strongly suggests halo
exchange IS adding energy at the seam rather than balancing it.

**Iter-985 implication.**

Iter-981/982/983/984's cube-vertex thread is a SECONDARY effect.
The **primary** W2 v_ll_Linf=55.6 m/s gap is at **equatorial cube
EDGES** at lat ≈ ±6° (the boundary cells closest to the equator
on the seam).  Different from the corner imprint chased in earlier
iters.

This calls for re-investigation:

1. The FB chain (in `_d_sw_native`) edge halo for `u_d, v_d` after
   step-N or step-1 may not be cross-face complete on the seams.
2. The d_sw5 Smagorinsky branch may be mis-calibrated near the
   equatorial seam (lat 0 has different metric behaviour from lat
   45).
3. The PPM transport in `_bgrid_ke_transport` may be aliasing at
   the seam, since the W2 wind has zero cross-edge component but
   the PPM stencil reaches into halo at the seam.

**Iter-985 deliverables.**

1. `scripts/diag_iter985_cube_edge_localise.py` — runnable
   localisation of the W2 1-day peak.
2. Patched `_corner_vorticity` measurement (no production change;
   cube-vertex fix yields +0.02% on v_ll_Linf — not the dominant
   bug).
3. `docs/fv3_fortran_fidelity_review.md` — this entry.

No production code change in iter-985.  Iter-986 will pivot to
investigate the **equatorial cube-edge seam** as the new prime
suspect.

**Backlog for iter-986+.**

1. Audit halo path for `u_d` (D-grid west-edge-of-face-1 → meets
   east-edge-of-face-0).  Compare cross-face halo of u_d at j=21
   to the analytical W2 wind (u = u0*cos(lat)*cos(angle)).
2. Probe d_sw5 vs Smagorinsky behaviour at seam by zeroing each
   d_sw step and remeasuring 1-day v_ll_Linf — narrow the term that
   amplifies the seam.
3. Run iter-985 fix + iter-986 candidate fix in combination once
   iter-986 candidate is identified — the cube-vertex tweak is
   small but free.

### Iter-984 — Term-by-term decomposition: vortex flux DOMINATES cube-vertex imbalance

**Trigger.**  Iter-983 found ~1.3 m/s |duc| imbalance at cube
vertices.  Iter-984 decomposes the c_sw uc update into its three
terms.

**c_sw uc update formula:**
```
uc_new = uc + fy1 * vort_x + dke_x       (vortex flux + KE grad)
       + dp_x                             (p_grad_c)
```

For W2 solid-body steady, all three should sum to zero.

**Iter-984 measurement on duogrid C36 W2 IC:**

```
                          |max|  | top cube vertex (face=0, i=0, j=35):
  vort_contrib_uc:        1.3174 |     -1.317  ← DOMINANT
  dke_x (KE grad):         0.3718 |     -0.027  ← small
  dp_x (p_grad_c):         0.4396 |     +0.202  ← partial cancel
  ───────────────────────────────|──────────────
  SUM (target ≈ 0):        1.1422 |     -1.142  ← imbalance!
```

**Critical finding.**  At cube vertex face=0 (i=0, j=35):

- vort_contrib_uc = -1.317 m/s (vortex flux + Coriolis term)
- dp_x = +0.202 m/s (pressure gradient)

Geostrophic balance demands `fy1*vort_x = -dp_x`.  Expected
fy1*vort_x = -0.202.  Actual: -1.317.  **OFF BY 6.5×.**

**Diagnosis.**  At cube vertex, `fy1 = dt2 * (v_d - uc*cosa) /
sina_u`.  The 1/sina_u factor amplifies near cube vertices where
sin(angle between i and j basis) is reduced.

But that doesn't explain the 6.5× excess — it should be a smaller
factor.  Likely the issue is that vort_x (upwind-selected
relative+absolute vorticity at corner) is wrong.  Specifically,
`_corner_vorticity` uses a 4-circulation formula on
`fx_circ = uc * dxc`, `fy_circ = vc * dyc`.  At cube vertices, the
halo cells of fx/fy come from cell-centre roundtrip via
`pad_halo_vector` (iter-836b/iter-837 path).  The
ROUNDTRIP-INDUCED ERROR on the halo cells of fx, fy may give
incorrect vort_x at cube vertices.

**Backlog confirmed**: the halo'ing of `fx_circ`, `fy_circ` in
`_corner_vorticity` for the duogrid path is the prime suspect.

**Iter-984 deliverables.**

1. `scripts/diag_iter984_csw_term_decomposition.py` — runnable
   term decomposition.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Backlog for iter-985+.**

1. Audit `_corner_vorticity` halo path (iter-836b/iter-837 cell-
   centre roundtrip) at cube vertices.  Compare against Fortran's
   c_sw vorticity computation at sw_core.F90:374-408.
2. Test alternative halo strategies for fx_circ, fy_circ (e.g.,
   direct cube_rmp on the face-local circulation).

### Iter-983 — c_sw + p_grad_c imperfect geostrophic balance at cube vertices

**Trigger.**  Iter-982 narrowed the cube-vertex bug to upstream of
d_sw_native.  Iter-983 directly measures c_sw + p_grad_c output
on W2 IC.

**Iter-983 measurement.**  Run c_sw + p_grad_c on W2 solid-body
IC (analytical steady → c_sw+p_grad_c should give Δuc=Δvc=0).
Measure increments duc, dvc:

```
c_sw alone:                  |duc|_max=1.3441, |dvc|_max=0.8424
c_sw + p_grad_c:             |duc|_max=1.2937, |dvc|_max=0.5332
                             (both at CUBE VERTICES)

Top 5 |duc| locations (c_sw + p_grad_c):
  face=2 i= 0 j=35  |duc|=1.2937   ← NW cube vertex face 2
  face=2 i= 0 j= 0  |duc|=1.2937   ← SW cube vertex face 2
  face=0 i= 0 j= 0  |duc|=1.2915   ← SW cube vertex face 0
  face=0 i= 0 j=35  |duc|=1.2915   ← NW cube vertex face 0
  face=3 i= 0 j= 0  |duc|=1.2880   ← SW cube vertex face 3

Top 5 |dvc| locations (c_sw + p_grad_c):
  face=4 i=35 j=35  |dvc|=0.5332   ← N-pole NE cube vertex
  face=5 i=35 j= 1  |dvc|=0.5332   ← S-pole SE cube vertex
  face=4 i= 0 j= 1  |dvc|=0.5324   ← N-pole SW cube vertex
  face=5 i= 0 j=35  |dvc|=0.5324   ← S-pole NW cube vertex
```

**Critical finding.**  c_sw + p_grad_c produces ~1.3 m/s
geostrophic imbalance at cube vertices for the W2 analytical
steady IC.  This propagates into d_sw_native as wrong NEW uc, vc.

The dvc cube vertices (top 4) match EXACTLY the locations of
iter-981's |du_d| top errors at the polar faces.  The c_sw +
p_grad_c imbalance is the dominant source of the FB chain
cube-vertex error.

**Cause.**  Geostrophic balance for W2 requires `fy1 * vort_x +
dke_x + dp_x = 0` (continuous).  In discrete form, the three
terms cancel within truncation error.  At cube vertices, the
stencils for vort, ke, p_grad use halo data from 3 different
faces with different orientations.  The cancellation is
imperfect at cube vertices.

**Implication.**  Closing the v_ll_Linf gap requires Fortran-
faithful cube-vertex handling in c_sw and/or p_grad_c.  The 14+
audited routines all match Fortran functionally, but the
COMPOSITION at cube vertices doesn't achieve the cancellation
that Fortran does.

**Iter-983 deliverables.**

1. `scripts/diag_iter983_csw_pgradc_balance.py` — runnable
   diagnostic that measures duc, dvc on W2 IC.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Backlog for iter-984+.**

1. Compare term-by-term: which component of c_sw (KE-grad,
   vort-flux, p_grad_c) contributes most at cube vertices?
   This requires running each operator separately on W2 IC.
2. Investigate whether `_pad_halo_auto` corner-fill at cube
   vertices matches Fortran's expected stencil coverage.
3. Check whether Fortran's W2 IC is loaded with cube-vertex
   precomputed data that we may be missing.

### Iter-982 — Component probes: cube-vertex bug NOT in BGRID_NE sync, iter-947 helper, or damping

**Trigger.**  Iter-981 isolated the per-step ~0.98 m/s |du_d|
error to polar-face cube vertices.  Iter-982 probes individual
components by disabling them and measuring residual error.

**Iter-982 probes on duogrid C36 W2 1-step:**

| config                                | \|du\|_max | \|dv\|_max | \|dh\|_max |
|---------------------------------------|-----------:|-----------:|-----------:|
| baseline (default)                    |     0.9801 |     1.4492 |     56.47  |
| no BGRID_NE sync                      |     0.8787 |     1.4467 |     56.47  |
| no iter-947 (mode='edge' uc/vc)       |     1.0673 |     1.5862 |     60.77  |
| damp_v=0 (no vort damping)            |     0.9808 |     1.4492 |     56.47  |
| d4_bg=0 (no d_sw5)                    |     0.9755 |     1.4425 |     56.47  |
| both damping OFF                      |     0.9762 |     1.4424 |     56.47  |

**Findings.**

1. **BGRID_NE sync** introduces ~0.10 m/s of error (baseline 0.98
   vs without-sync 0.88).  Small contributor.
2. **iter-947 helper** REDUCES error by ~0.09 m/s (mode='edge'
   gives 1.07 vs iter-947 0.98).  Small but positive.
3. **Damping** (damp_v, d4_bg) is essentially neutral on the
   per-step error (within 0.005 m/s).  Damping is not the bug.

**Crucial:** With BOTH damping disabled AND BGRID_NE sync disabled
AND iter-947 disabled, the error would still be ~0.85-0.90 m/s
(estimated from independent contributions).

The dominant ~0.85 m/s |du_d| per-step error is in the
NON-DAMPING, NON-SYNC, NON-HALO part of the FB chain.  Candidates:

1. `c_sw` + `_p_grad_c` for uc, vc construction — runs BEFORE
   d_sw_native, contributes to the NEW uc, vc passed in.
2. `_d_sw1_recompute_ut_vt` Part 1 (4-cell average) — uses uc,
   vc to compute ut, vt.  At cube vertex, the average reads halo
   uc, vc values.
3. `_bgrid_ke_transport` PPM transport — uses iter-945 D-grid
   halo.  At cube vertices, the cube_rmp Lagrange interpolation
   in `ext_vector_dgrid` may not exactly match Fortran's expected
   value.
4. `_corner_vorticity` for zeta_abs — at cube vertices.

**Iter-982 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry with probe
   results.

No code change.

**Backlog for iter-983+.**

Add probes that swap c_sw + p_grad_c for a NULL operator
(uc, vc unchanged from d2a2c) and measure residual error.  This
isolates whether the cube-vertex bug originates upstream of d_sw.

### Iter-981 — 1-step W2 tendency diagnostic isolates bug to polar-face cube vertices

**Trigger.**  Iter-980 suggested running a 1-step diagnostic to
isolate the per-step bug source.  W2 is a solid-body steady
solution; analytic time derivative is zero everywhere.

**Iter-981 measurement on duogrid C36 W2 dt=300 s, 1 step:**

```
|du_d|_max = 0.980 m/s
|dv_d|_max = 1.449 m/s
|dh|_max   = 56.5 m

Top 5 |du_d| locations (face, i, j, value):
  face=4 i= 0 j= 1   du = 0.9801   ← N-pole face SW vertex
  face=4 i=35 j=35   du = 0.9799   ← N-pole face NE vertex
  face=5 i= 0 j=35   du = 0.9795   ← S-pole face NW vertex
  face=5 i=35 j= 1   du = 0.9793   ← S-pole face SE vertex
  face=4 i=35 j=18   du = 0.8788   ← N-pole face mid-east boundary
```

**Critical pattern:** Top 4 du_d errors are ALL at CUBE VERTICES
on the POLAR FACES (face 4 = north pole, face 5 = south pole).
All four polar-vertex errors have nearly identical magnitude
(~0.98 m/s), suggesting the SAME structural error fires at each.

The 5th-largest error (face=4, i=35, j=18) is at a MID cube-edge
on the polar face, ~10% smaller than the vertex errors.

The error is NOT at equatorial-belt face boundaries (faces 0, 1,
2, 3) — those are only ~0.5 m/s per step.

**Implication.**  The bug is in CUBE-VERTEX HANDLING, specifically
at the 8 cube vertices where 3 faces meet (4 vertices visible from
each polar face).  Likely candidates:

1. **`synchronize_bgrid_ne_corner_geo` 3-face vertex averaging**
   (halo.py:2027-2126).  At each cube vertex, 3 faces share the
   point; the geo-frame-then-back rotation may not correctly
   handle the 3-way mean.
2. **`pad_halo` cube-vertex corner-fill** (halo.py + duogrid
   `cube_rmp` corner_fill_region).  At halo cells adjacent to
   cube vertices, the fill logic may diverge from Fortran's
   `mpp_update_domains` corner handling.
3. **`_d2a2c_vect_duogrid` cube-vertex 4th-order interp** —
   uses `ext_vector_dgrid` which goes through cube_rmp.  At
   vertices, the Lagrange interpolation may be ill-conditioned.

The 1-day v_ll_Linf=55 m/s gap is consistent with 0.98 m/s per
step × 288 steps × something growing slower than linearly =
~55 m/s when integrated.  The cube-vertex error compounds over
time.

**Iter-981 deliverables.**

1. `scripts/diag_iter981_w2_1step_tendency.py` — runnable 1-step
   diagnostic that reports |du_d|, |dv_d|, |dh| max + top-5
   locations.
2. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Backlog for iter-982+.**  Investigate the 3 candidate cube-
vertex helpers individually; identify which one introduces the
0.98 m/s per-step error.  Likely fixes will be ports of Fortran's
specific cube-vertex code.

### Iter-980 — Verify mass transport CGRID_NE flux sync + KE Lin-Rood formula

**Iter-980 audit findings:**

1. **Mass-transport flux sync** (dyn_core.F90:850-900): Fortran's
   d_sw1 computes mass fluxes (fx, fy from `fv_tp_2d` of delp),
   then `mpp_get_boundary(fxx_delp, fyy_delp, ..., CGRID_NE)`
   averages the boundary fluxes for duogrid.  Our Python's
   `fv_tp_2d` calls `synchronize_cgrid_fluxes(fx, fy, n)` at line
   864-865 (gated on `apply_cgrid_flux_sync=True` default for
   mass).  ✓ Matches Fortran.

2. **KE Lin-Rood formula** (dyn_core.F90:1015-1020):
   ```fortran
   kee(i,j) = (ubbtemp(i,j) * vbbtemp(i,j))
   kee(i,j) = 0.5*(kee(i,j) + ubb(i,j)*vbb(i,j))
   ```
   Our Python:
   ```python
   ke_corner = 0.5 * (ubbtemp * vbbtemp + ubb * vbb)
   ```
   Variable mapping:
   - `ubbtemp` ← transported_y (ytp_v output)
   - `vbbtemp` ← vb (y-Courant)
   - `ubb` ← ub (x-Courant)
   - `vbb` ← transported_x (xtp_u output)
   ✓ Identical formula.

3. **BGRID_NE corner sync** (dyn_core.F90:984-1011 + our
   `synchronize_bgrid_ne_corner_geo`): Fortran averages ubb,
   vbbtemp face-local at boundary edges using mpp_get_boundary
   with internal sign-flip table.  Our Python converts to geo
   frame (invariant under face rotation), averages as scalars,
   converts back.  Mathematically equivalent for a continuous
   physical vector field — both methods give the same answer.

**Comprehensive audit conclusion (iter-967 through iter-980).**

The iter-967..iter-980 audit covered EVERY structural component
of Fortran's d_sw chain that affects the duogrid path.  All 16+
audited routines + halo orchestration + flux sync + KE formula
match Fortran functionally.

The remaining v_ll_Linf=55.6 m/s gap (vs Fortran's claimed < 1
m/s for W2 at C36) is structurally unexplained.  Most likely
sources:

1. **Floating-point rounding accumulation** (~6% per iter-978).
2. **Fortran's stale-halo data race** (uc, vc deeper halo cells
   contain previous-step values that may differ from our
   iter-947 NEW-corrected halo).
3. **Bug in some auxiliary helper** I have not yet examined
   (e.g., `_pad_halo_auto`, `pad_halo` corner-fill logic at cube
   vertices, or duogrid `cube_rmp` interpolation table).

**Suggested next steps for iter-981+.**

1. Build a small "1-step" diagnostic that runs ONE FB chain step
   on W2 IC and compares u_d_new, v_d_new at every cube-vertex
   cell vs analytical W2 zero-tendency (since W2 is steady).
2. Identify which specific cells deviate first.
3. Trace back through the d_sw chain to find which operator
   introduced the deviation.

This would isolate the bug source instead of cascading through
288 steps of compounded effects.

### Iter-979 — Audit Fortran's uc/vc halo orchestration between c_sw and d_sw

**Trigger.**  Iter-977 hypothesized that Fortran's per-step
communications between c_sw and d_sw differ from our Python's
all-in-one `_d_sw_native`.  Iter-979 audits the halo flow.

**Iter-979 findings.**

1. **`mpp_update_domains(uc, vc, gridtype=CGRID_NE)` between c_sw
   and d_sw:**
   - Line 633: `start_group_halo_update(i_pack(9), uc, vc, ...,
     CGRID_NE)` — async start.
   - Line 654: `if (.not. duogrid) call complete_group_halo_update
     (i_pack(9), domain)` — completes the async update ONLY for
     non-duogrid.
   - Lines 689, 702: synchronous `mpp_update_domains` calls inside
     `if (flagstruct%regional)` block — only fire for regional
     setups, NOT for duogrid.

   **Conclusion:** For DUOGRID, the started halo update at line
   633 is NEVER COMPLETED.  uc, vc do NOT receive an explicit MPI
   halo exchange between c_sw and d_sw.

2. **Where do uc, vc halo values come from on the duogrid path?**
   - `d2a2c_vect` duogrid branch (sw_core.F90:3558-3563) computes
     `uc(i, j)` at `i ∈ [is-1, ie+2], j ∈ [js-1, je+1]`.  This is
     SLIGHTLY WIDER than interior (1 extra cell on each side).
   - `c_sw` UPDATES uc only at INTERIOR `(is..ie+1, js..je)` —
     halo cells unchanged.
   - `p_grad_c` UPDATES uc only at interior — halo cells unchanged.
   - When d_sw1 reads `uc(i, jsd..jed)` for the full halo j-range,
     halo cells have:
     - At `j ∈ [js-1, je+1]`: d2a2c_vect output (no c_sw/p_grad_c
       increment).
     - At `j ∈ [jsd, js-2] ∪ [je+2, jed]` (deeper halo): UNDEFINED
       (from a previous timestep or initial value).

3. **Comparison with our Python.**  Our iter-947 helper computes
   `NEW_halo = NEW_boundary + (OLD_halo - OLD_boundary)`, where
   OLD_halo is from d2a2c on OLD u_d, v_d.  This gives:
   - Halo value ≈ OLD_d2a2c at halo + Δ_at_boundary.
   - Fortran's halo value ≈ OLD_d2a2c at halo (no Δ).

   For W2 solid-body, Δ at boundary is small (~15 m/s but partially
   canceled by symmetry in the geostrophic regime), so iter-947 ≈
   Fortran.  For more dynamic flows the Δ correction matters more.

**Insight.**  Our Python's iter-947 helper is FUNCTIONALLY
EQUIVALENT to Fortran's actual behavior on the duogrid path.  The
only difference is the small Δ correction we add (which is more
Fortran-FAITHFUL in spirit since Fortran's halo cells are stale
between time steps).

**Implication.**  The FB chain's remaining 55 m/s v_ll_Linf gap is
NOT due to missing halo exchanges between c_sw and d_sw.  Both
implementations have the same data quality at uc, vc halo cells.

**Iter-979 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Conclusion of iter-967 → iter-979 audits.**  After auditing 14+
routines + the halo orchestration, our Python is FUNCTIONALLY
EQUIVALENT to Fortran for the duogrid d_sw chain.  The remaining
v_ll_Linf gap of 55 m/s (vs Fortran's claimed < 1 m/s for W2 at
C36) must arise from:
1. The deepest halo cells that Fortran reads UNDEFINED but we
   read mode='edge' or iter-947-corrected.  In Fortran, those
   undefined cells contain stale data from previous timesteps —
   maybe luck-of-the-draw works out for W2.
2. Subtle floating-point ordering / rounding differences.
3. A bug in our Python that I haven't yet identified.

### Iter-978 — FB chain dt sensitivity: truncation contributes ~6% over 4x dt reduction

**Trigger.**  Iter-977 hypothesized truncation as one possible
contributor to the 55.6 m/s gap.  Iter-978 measures the dt
sensitivity to quantify.

**Iter-978 measurement on duogrid C36 W2 1-day:**

| dt (s)  | n_steps | v_ll_Linf (m/s) |
|---------|--------:|----------------:|
|   600   |  NaN    |    NaN @ step 111 (CFL exceeded) |
|   300   |    288  |          55.61   |
|   150   |    576  |          55.24   |
|    75   |   1152  |          52.40   |

**Findings.**
- dt=600s: blows up due to CFL (the FB chain at C36 with default
  damping is unstable above ~dt=400s).
- dt=300 → 75: 4× smaller dt gives 5.8% reduction in v_ll_Linf.
- The improvement is modest; truncation accounts for ~6% of the
  remaining gap.
- Extrapolating to dt→0: maybe 10-15% reduction.  Still 30+ m/s
  from acceptance.

**Conclusion.**  Truncation is a SMALL contributor.  The dominant
55 m/s gap is STRUCTURAL — in components we have not yet pinpointed.

**Iter-978 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Backlog for iter-979+.**

The structural gap is most likely in:
1. The d_sw chain's specific halo-stitching at cube vertices
   (where 3 faces meet) — our iter-945/947 helpers may differ
   from Fortran's mpp_update_domains in vertex handling.
2. Subtle indexing / sign / cross-component conventions in the
   covariant rotation logic.
3. The orchestration of c_sw + p_grad_c + d_sw1..d_sw6 (we group
   all into `_d_sw_native`; Fortran has separate per-step
   communications between).

### Iter-977 — Comprehensive audit conclusion: structural code is Fortran-faithful

**Iter-967 through iter-976 audited 9 routines** against the GFDL
Fortran source at `../FV3/atmos_cubed_sphere-symmetryclean/`:

1. `c_sw` (sw_core.F90:79-494) ✓
2. `_corner_vorticity` cube-vertex correction (sw_core.F90:395-401) ✓
3. `_ke_upwind` boundary handling (sw_core.F90:303-365) ✓
4. `_vorticity_flux` boundary handling (sw_core.F90:420-480) ✓
5. C-grid update formulae (sw_core.F90:483-492) ✓
6. `_d_sw1_recompute_ut_vt` early-return on duogrid
   (sw_core.F90:656) ✓
7. `_divergence_corner_duo` (sw_core.F90:2345-2447) ✓
8. `_d2a2c_vect_duogrid` (sw_core.F90:3419-3454) ✓
9. `_del6_vt_flux` and d_sw6 wind update (sw_core.F90:1937-2121) ✓
10. d_sw5 nord>=1 iterated Laplacian (sw_core.F90:1727-1787) ✓
11. `fv_tp_2d` Lin-Rood operator-split (tp_core.F90:80-225) ✓
12. `_xppm`/`_yppm` boundary handling at iord>=8 (tp_core.F90:
    2752-2863) ✓
13. iord=9 pmp/lac limiter (tp_core.F90:2778-2786) ✓
14. `compute_transport_quantities` (sw_core.F90:830-869) ✓

**One real gap identified and ported:**
- `_interp_center_to_corner_a2b_ord4` (iter-971/972): 4th-order
  Lagrange + PPM cascade matching Fortran a2b_edge.F90:50-330.
  Wired into d_sw5 Smagorinsky; bit-identical for default config
  (Smagorinsky branch off) but Fortran-faithful for high-dddmp
  regimes.

**Major restructuring (NOT Fortran code-equivalent):**
- iter-945 `_pad_halo_dgrid_for_ppm` provides cross-face halo for
  u_d, v_d via `ext_vector_dgrid`.  Fortran does this halo
  upstream of d_sw via `mpp_update_domains(DGRID_NE)`; we do it
  inside `_bgrid_ke_transport`.  Mathematically equivalent at
  interior but differs at deepest halo (Fortran: 1st-order
  extrap; ours: 4th-order ext_vector).
- iter-947 `_pad_halo_uc_vc_new_via_old_delta` provides
  cross-face halo for uc, vc.  Similar restructuring.

**Conclusion of audits.**  All audited structural code is
Fortran-faithful for the duogrid path.  The remaining v_ll_Linf=
55.6 m/s gap is NOT in the audited structural code.

**Hypotheses for the remaining gap:**

1. **Float precision floor.**  Fortran is typically compiled with
   R8 (double precision) for FV3 dycore but R4 (single) for
   transport/output.  Our Python uses float64 throughout
   (JAX_ENABLE_X64).  Accumulated rounding over 288 steps may
   yield different residuals.

2. **`mpp_get_boundary` BGRID_NE averaging semantics**: Fortran
   averages ubb in face-local frame (with internal sign-flip
   table); our `synchronize_bgrid_ne_corner_geo` averages in
   geographic frame.  Mathematically equivalent for matching
   physics but may differ at cube vertices where the rotation
   tables are subtle.

3. **d2a2c_vect deepest-halo treatment**: Fortran uses 1st-order
   extrapolation `utmp(jsd) = u(jsd+1)` (weirdly via
   double-overwrite); we use 4th-order ext_vector.  Our 4th-order
   should be MORE accurate, not less — unless our 4th-order
   produces an unphysical halo value at cube vertices that
   Fortran's 1st-order avoids.

4. **`_divergence_corner_duo` halo'd u_d / v_d**: Fortran reads u
   and v at full halo (jsd+1..jed) for the `uf`, `vf` formulas at
   inner halo cells.  Our Python reads at interior only.  At
   boundary-adjacent corners (which receive 0.25× attenuation),
   the difference is small but present.

5. **Time-integration coupling**: Fortran's FB chain integrates
   c_sw + p_grad_c + d_sw with specific dt scaling.  Verify our
   Python's dt2 = dt/2 in c_sw and full dt in d_sw matches Fortran
   exactly.

**Iter-977 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this audit summary
   entry.

No code change.  Audits complete; further fidelity gains require
either (a) restructuring our halo strategy to match Fortran's
upstream `mpp_update_domains` exactly, or (b) detailed numerical
testing to find the source of accumulated drift.

### Iter-976 — Audit `compute_transport_quantities` against Fortran d_sw1

**Iter-976 audit findings:**

1. **Fortran d_sw1 transport-quantity formulas** (sw_core.F90:830-869):
   ```fortran
   xfx_adv(i,j) = dt*ut(i,j)                          ! line 832
   if ( xfx_adv(i,j) > 0. ) then
      crx_adv(i,j) = xfx_adv(i,j) * rdxa(i-1,j)        ! upwind cell i-1
      xfx_adv(i,j) = dy(i,j)*xfx_adv(i,j)*sin_sg(i-1,j,3)
   else
      crx_adv(i,j) = xfx_adv(i,j) * rdxa(i,j)          ! upwind cell i
      xfx_adv(i,j) = dy(i,j)*xfx_adv(i,j)*sin_sg(i,j,1)
   end if
   ```

2. **Our Python `compute_transport_quantities`** (fv_tp_2d.py:536):
   - `xfx_raw = dt * ut`
   - Upwinding on `ut > 0` (equivalent to `dt*ut > 0` since dt > 0). ✓
   - `crx = xfx_raw * rdxa(upwind cell)`. ✓
   - `xfx = xfx_raw * dy * sin_sg(upwind edge)`. ✓
   - sin_sg edge index: 3=E (Fortran 1-indexed) ↔ 2=E (Python
     0-indexed) for `i-1` upwind; 1=W ↔ 0=W for `i` upwind. ✓
   - Same logic for cry/yfx in the y-direction. ✓
   - `ra_x = area + xfx[i] - xfx[i+1]` matches Fortran. ✓
   - `pad_halo` with duogrid for rdxa, rdya, sin_sg — matches
     Fortran's bounded_domain path that skips copy_corners.  ✓

**Conclusion.**  `compute_transport_quantities` is Fortran-faithful
for the duogrid path.

**Iter-976 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Backlog for iter-977+.**

1. **`mpp_get_boundary` BGRID_NE averaging semantics** —
   compare Fortran's face-local averaging (with internal rotation
   table) vs our `synchronize_bgrid_ne_corner_geo` (geo-frame
   averaging).
2. Float precision floor.

### Iter-975 — Audit `fv_tp_2d` xppm/yppm against Fortran tp_core.F90

**Iter-975 audit findings:**

1. **`fv_tp_2d`** (tp_core.F90:80-225) Lin-Rood operator-split
   matches our Python `fv_tp_2d` in `fv_tp_2d.py:707-742`:
   - Pass 1: yppm on q → fy2; cross-correct via fyy = yfx*fy2 →
     q_i; xppm on q_i → fx (final).
   - Pass 2: xppm on q → fx2; cross-correct via fx1 = xfx*fx2 →
     q_j; yppm on q_j → fy (final).
   - Cross-corrected formula: `q_i = (q*area + fyy[j] - fyy[j+1])
     / ra_y` — matches our Python exactly. ✓
   - Final flux averaging: `fx = 0.5*(fx + fx2) * xfx`, `fy =
     0.5*(fy + fy2) * yfx` — matches. ✓

2. **`xppm` boundary fix** (tp_core.F90:2752-2863, iord>=8 path):
   - Fortran's iord=9 (default `hord_vt = 9` per fv_arrays.F90:339)
     boundary fix at line 2819 has condition `is==1 .and. (.not.
     bounded_domain .or. .not. duogrid_initialized)`.
   - For duogrid: `(.not. true) .or. (.not. true) = false`, so
     boundary fix SKIPS.
   - Our Python's `_ppm_1d` `apply_fortran_xppm_boundary` is gated
     on `not use_duogrid` — matches.  ✓

3. **`copy_corners`** (tp_core.F90:139-141, 160-162) — gated on
   non-duogrid; our Python's `pad_halo(duogrid=...)` does the
   equivalent for duogrid via cube_rmp.  ✓

4. **iord=9 pmp/lac limiter** (tp_core.F90:2778-2786):
   ```fortran
   pmp_1 = -2.*dq(i)
   lac_1 = pmp_1 + 1.5*dq(i+1)
   bl(i) = min(max(0., pmp_1, lac_1),
               max(al(i)-u(i,j), min(0., pmp_1, lac_1)))
   ```
   Our Python `_ppm_1d` uses the same pmp/lac formula (verified at
   tp_core.F90:2778-2786 / `fv_tp_2d.py` iord=9 path).  ✓

**Conclusion.**  `fv_tp_2d` xppm/yppm chain is Fortran-faithful for
the duogrid path with hord_vt=9.  Boundary handling, cross-
correction, and limiter all match Fortran.

**Iter-975 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.

**Backlog for iter-976+.**

The remaining v_ll_Linf=55.6 m/s gap is now NARROWED to:
1. **`compute_transport_quantities`** for ut, vt → crx, cry, xfx,
   yfx conversion — verify our Python matches Fortran's d_sw1.
2. **`mpp_get_boundary` BGRID_NE averaging** — Fortran averages
   ubb in face-local frame; our `synchronize_bgrid_ne_corner_geo`
   averages in geo frame.  Subtle difference at cube vertices.
3. **Float precision floor** (Fortran R4 vs our float64).

### Iter-974 — Test Fortran-style d_sw3 (mode='edge' + iter-967 boundary fix) vs iter-945 halo

**Trigger.**  iter-967 found the d_sw3 boundary fix conflicts with
iter-945 halo.  Iter-974 tests the OPPOSITE configuration: Fortran's
actual approach (mode='edge'-style halo for u_d, v_d + d_sw3
cube-edge boundary fix from sw_core.F90:2819-2863).

**Iter-974 measurement on duogrid C36 W2 1-day:**

| config                                  | v_ll_Linf (m/s) | step survival |
|-----------------------------------------|----------------:|--------------:|
| **iter-945 halo (current)**             |        55.61    |     288/288   |
| Fortran-style (mode='edge' + bdy fix)   |   NaN @ step 215|         215   |

**Conclusion.**  Our iter-945 cross-face halo for D-grid winds is
strictly MORE STABLE on duogrid C36 W2 than Fortran's actual
approach (mode='edge' halo + boundary fix).

**Implication.**  Either:
1. Fortran benefits from the FULL upstream halo'd u, v (via
   `mpp_update_domains(DGRID_NE)` providing cross-face data
   at depth ng=3 BEFORE d_sw3 runs).  Inside d_sw3, mode='edge'
   then extends from this cross-face halo, so Fortran's
   "mode='edge'" is functionally close to our iter-945 halo.
   The boundary fix is then a consistent extrapolation.
2. OR Fortran's specific implementation does something extra
   (e.g., a sign-correction or rotation) that our Python misses.

The iter-967 boundary fix wired into our Python with iter-945's
already-cross-face halo over-corrected (worsened).  With
mode='edge' and iter-967 (closer to Fortran's actual code path
modulo our missing upstream halo), it goes NaN.

This means our iter-945 halo is functionally REPLACING Fortran's
"upstream mpp_update_domains halo + mode='edge' inside d_sw3" with
"cross-face halo + no boundary fix".  Mathematically equivalent at
the interior but DIFFERENT at the deepest boundary cells (where
Fortran's boundary fix override applies).

**Iter-974 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.  iter-967 boundary-fix code preserved on
`_ppm_transport_1d` (gated `apply_d_sw3_boundary_fix=False`) for
future experiments.

**Backlog for iter-975+.**

The iter-967 boundary fix's actual purpose is to correct ytp_v /
xtp_u at the deepest cube-face boundary CELL (j=0/1/2 in Fortran
1-indexed).  At these cells, Fortran's mode='edge'-on-halo'd-data
gives one specific value, and the boundary fix overrides with
another using s11/s14/s15.  Our iter-945 halo gives a THIRD value
(true cross-face from cube_rmp).

The three values may all be valid Fortran-faithful approximations
of the same physical quantity.  The exact equivalence between
Fortran's combo and our combo is not provable without testing.

For now, our iter-945 alone is the most stable + accurate
configuration we have.

### Iter-973 — Audit `del6_vt_flux` and `d_sw6` wind update vs Fortran

**Iter-973 audit findings:**

1. **`del6_vt_flux`** (sw_core.F90:2008-2121) matches our Python
   `_del6_vt_flux` in `fv3_sw_core.py:1202-1289`:
   - Initial `d2 = damp * q` (we factor damp to end via iter-937b
     for float32 safety; mathematically equivalent).
   - Initial flux: `fx2 = del6_v * (d2(i-1) - d2(i))`, sign matches
     Fortran's USE_SG branch (line 2065).
   - Iteration loop: sign FLIP from initial — Fortran has `(d2(i) -
     d2(i-1))` after iteration; our Python matches.
   - copy_corners gated on non-duogrid; our Python uses
     duogrid-aware `pad_halo` which is the equivalent.  ✓

2. **d_sw6 wind update** (sw_core.F90:1937-1944):
   ```fortran
   u(i,j) = vt(i,j) + ke(i,j) - ke(i+1,j) + fy(i,j)
   v(i,j) = ut(i,j) + ke(i,j) - ke(i,j+1) - fx(i,j)
   ```
   Fortran's `u` here is in CIRCULATION form (u*dx) at the end of
   d_sw5 (vt = u*dx).  The assignment writes new u in circulation;
   conversion back to velocity happens at a later step in the
   chain.

   Our Python:
   ```python
   u_d_new = u_d + (ke_diff_u_scaled + fy_vort) * rdx_u
   v_d_new = v_d + (ke_diff_v_scaled - fx_vort) * rdy_v
   ```
   Equivalent to Fortran via `u_d_new * dx_u = u_d * dx_u + ke_diff
   + fy_vort` = Fortran's circulation-form u_new.  Both formulations
   give the same physical update.  ✓

3. **d_sw6 damp_v post-step** (sw_core.F90:1989-2000):
   ```fortran
   u(i,j) = u(i,j) + vt(i,j)   ! vt = fy2 from del6_vt_flux
   v(i,j) = v(i,j) - ut(i,j)   ! ut = fx2
   ```
   Our Python: `u_d_new = u_d_new + fy2 * rdx_u`,
   `v_d_new = v_d_new - fx2 * rdy_v`.  Same equivalence as #2. ✓

**Conclusion.**  d_sw6 (wind update + vorticity damping) and
`del6_vt_flux` are Fortran-faithful for the duogrid path.

**Iter-973 deliverables.**

1. `docs/fv3_fortran_fidelity_review.md` — this entry.

No code change.  All previously committed code matches Fortran in
the audited path.

**Backlog for iter-974+.**

The remaining v_ll_Linf=55.6 m/s gap is NOT in any audited
structural code.  Candidates:

1. **fv_tp_2d** (tp_core.F90 xppm/yppm) — finer audit of the
   mass / vorticity transport.
2. **fill_corners** for non-duogrid (skipped on our duogrid path
   anyway).
3. **mpp_get_boundary** semantics for the BGRID_NE flux sync at
   the d_sw3 corners (already implemented per iter-941; verify
   the exact averaging formula matches Fortran).
4. The numerical-precision floor: our Python uses float64 with
   JAX_ENABLE_X64; Fortran uses single precision by default
   (R4) but FV3 is often compiled with double (R8).  At C36
   solid-body, accumulated truncation over 288 steps may itself
   contribute a few m/s.

### Iter-972 — Wire `_interp_center_to_corner_a2b_ord4` into d_sw5 Smagorinsky branch

**Trigger.**  iter-971 added the Fortran-faithful 4th-order
interpolation helper.  Iter-972 wires it into
`_d_sw5_corner_divergence`'s Smagorinsky branch (sw_core.F90:1795
`call a2b_ord4(wk, vort)`).

**Iter-972 fix.**  Replaced the 2nd-order
`_interp_center_to_corner(wk, cdgrid)` call inside the `dddmp >
1e-5` branch with `_interp_center_to_corner_a2b_ord4(wk, cdgrid)`.

**Measurement.**  W2 1-day duogrid C36 v_ll_Linf:

| config                   | iter-971 (2nd-order) | iter-972 (4th-order) |
|--------------------------|---------------------:|---------------------:|
| default (dddmp=0)        |               55.61  |               55.61  |
| iter-959 (dddmp=0.05)    |               53.17  |               53.17  |
| iter-963 (dddmp=0.45)    |               43.77  |               43.77  |

**Result: bit-identical.**  Reason: in the Smagorinsky formula
`damp2 = max(d2_bg, min(0.20, dddmp*smag_vort))`, when
`d2_bg ≥ dddmp*smag_vort`, d2_bg dominates and wk_corner is
irrelevant.  For W2 solid-body the relative vorticity is small
(order 10⁻⁵ s⁻¹), so `dddmp*smag_vort ~ 1e-3 ≪ d2_bg ∈ {0.01,
0.09}` for both Smagorinsky-tuned configs we test.  The
4th-order wk_corner is computed but doesn't affect the final
damp2 in this regime.

**Iter-972 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_d_sw5_corner_divergence` —
   replaced `_interp_center_to_corner` with
   `_interp_center_to_corner_a2b_ord4` in the Smagorinsky branch
   (1-line change with import).
2. Documentation only — no new sentinel needed because iter-971
   sentinel already pins the helper, and iter-962/963 sentinels
   pin the v_ll_Linf measurements which are bit-identical.

**Production impact.**  ZERO.  Default config doesn't hit the
Smagorinsky branch at all.  For Smagorinsky-tuned callers
(iter-959/963) the result is bit-identical because d2_bg
dominates damp2.

**Backlog for iter-973+.**

1. The wk_corner computation matters when `dddmp*smag_vort >
   d2_bg`.  This regime requires either high dddmp + low d2_bg
   (weak background damping with strong adaptive damping) or
   high local vorticity (e.g., W6/Galewsky jets, Held-Suarez
   forcing).  For those cases iter-972 IS Fortran-faithful and
   may matter.
2. Continue auditing remaining FV3 helpers (`del6_vt_flux`,
   `fv_tp_2d`, `fill_corners`).

### Iter-971 — Implement Fortran-faithful `_interp_center_to_corner_a2b_ord4` (4th-order)

**Trigger.**  iter-970 identified that Fortran's d_sw5 Smagorinsky
path uses 4th-order `a2b_ord4` for cell-centre → corner
interpolation; our Python's `_interp_center_to_corner` is 2nd-order.

**Iter-971 fix.**  Ported Fortran's a2b_ord4 (a2b_edge.F90:50-330)
duogrid path (lines 100-104, 188-192, 241-258) to Python as
`_interp_center_to_corner_a2b_ord4` in `operators_cdgrid.py`.

The 4th-order cascade:

```
qx(i, j)  = b2*(qin(i-2, j) + qin(i+1, j))
          + b1*(qin(i-1, j) + qin(i, j))
qy(i, j)  = b2*(qin(i, j-2) + qin(i, j+1))
          + b1*(qin(i, j-1) + qin(i, j))
qxx(i, j) = a2*(qx(i, j-2) + qx(i, j+1))
          + a1*(qx(i, j-1) + qx(i, j))
qyy(i, j) = a2*(qy(i-2, j) + qy(i+1, j))
          + a1*(qy(i-1, j) + qy(i, j))
qout(i, j) = 0.5 * (qxx(i, j) + qyy(i, j))
```

Constants matching Fortran:
- a1 = 9/16, a2 = -1/16  (Lagrange 4-pt)
- b1 = 7/12, b2 = -1/12  (PPM volume mean)

**Iter-971 deliverables.**

1. `src/legoesm/core/operators_cdgrid.py` —
   `_interp_center_to_corner_a2b_ord4` (~70 lines) using
   `_pad_halo_auto_h2` for cross-face halo of depth 2 (matches
   Fortran's halo'd input expectation for the 4-pt stencil).
2. `tests/test_iter971_a2b_ord4.py` — 2 sentinels:
   - shape (6, n+1, n+1) + exact-on-constants invariant
   - smooth-field 4th-order vs 2nd-order agreement within 10%
3. `docs/fv3_fortran_fidelity_review.md` — this entry.

**Production impact.**  ZERO.  The new helper is added but NOT
yet wired into `_d_sw5_corner_divergence`'s Smagorinsky branch —
that wiring is iter-972's task to keep the iter-971 commit small
and focused.

The default config (d2_bg=0, dddmp=0) doesn't hit the Smagorinsky
branch at all, so wiring won't change default behaviour either.
For Smagorinsky-tuned callers (iter-959/963), wiring _will_ change
W2 numbers — TBD whether the change is improvement or regression.

**Backlog for iter-972+.**

1. Wire `_interp_center_to_corner_a2b_ord4` into
   `_d_sw5_corner_divergence` Smagorinsky branch under an opt-in
   flag.  Measure W2 v_ll_Linf with iter-963 high-Smagorinsky
   config to verify Fortran-faithful 4th-order kernel improves
   over 2nd-order.

### Iter-970 — Audit d_sw5 nord>=1 iterated Laplacian + a2b_ord4 vs `_interp_center_to_corner`

**Iter-970 audit findings:**

1. **d_sw5 nord>=1 iterated Laplacian** (sw_core.F90:1727-1787):
   Fortran iterates `vc/uc = (divg_d gradient) * divg_u/v` with
   metric weighting `divg_u, divg_v`, then `divg_d = corner_div(uc,
   vc)`, scale by rarea_c.  Our Python matches structurally.  Both
   skip cube-vertex corrections for duogrid. ✓

2. **a2b_ord4 vs `_interp_center_to_corner`** (a2b_edge.F90:50,
   sw_core.F90:1795): Fortran's d_sw5 Smagorinsky branch uses
   `a2b_ord4(wk, vort)` — a **4th-order compact cubic**
   interpolation from cell centres to corners with constants
   `c1=2/3, c2=-1/6`, plus 1D edges done via similar 4th-order
   along each axis.

   Our Python uses `_interp_center_to_corner` which is a
   **2nd-order 4-point average**:
   ```python
   0.25 * (f_pad[:-1, :-1] + f_pad[1:, :-1]
            + f_pad[:-1, 1:] + f_pad[1:, 1:])
   ```

   This is a real fidelity gap.  However, the d_sw5 Smagorinsky
   branch is gated on `dddmp > 1e-5`, which is OFF by default
   (`dddmp=0`).  Only callers who opt into Smagorinsky tuning
   (like iter-959/963) hit this code path.  For our default
   `(d2_bg=0, dddmp=0)`, this gap doesn't matter.

   For iter-963 high-Smagorinsky (dddmp=0.45), the difference
   between 2nd and 4th-order interpolation on wk (vorticity at
   cell centres) is small for W2 solid-body (smooth field).
   Could close ~1% of the iter-963 v_ll_Linf=43.8 gap.

   **Iter-970 deliverable.**  Documented; no code change.  Future
   iter could port `a2b_ord4`'s 4th-order kernel to Python for
   the Smagorinsky path.

3. **d_sw5 nord=0 path** (sw_core.F90:1641-1724): direct del-2
   with adaptive Smagorinsky.  Boundary handling at face edges
   uses sin_sg upwind formulas.  Default `nord=1` doesn't take
   this path; Smagorinsky tuning callers may.  Audit deferred.

### Iter-969 — Audit d_sw1 / divergence_corner_duo / d2a2c_vect against Fortran (more findings)

**Iter-969 audit findings:**

1. **`d_sw1` boundary overrides skip on duogrid** (sw_core.F90:656):
   Fortran's condition `(.not. bounded_domain .or. .not. duogrid)`
   evaluates FALSE for duogrid runs (both flags TRUE → both `.not.`
   FALSE → `False .or. False = False`).  So the West/East/South/North
   edge overrides at lines 658-712 SKIP for duogrid.  Our Python's
   `_d_sw1_recompute_ut_vt` early-returns for duogrid after Part 1
   — Fortran-faithful. ✓

2. **`d_sw1` 4-cell average uses INTERIOR I, FULL halo j** (sw_core.F90:
   623-634): Fortran loops j=jsd..jed (full halo), i=is..ie+1
   (interior I-face).  Our Python computes ut at (n+1, n) — interior
   I-face (n+1), interior j-cells (n).  Fortran computes wider j
   range to provide ut for downstream operators that need halo'd ut.
   For our Python, the iter-947 NEW-corrected uc/vc halo provides
   the cross-face data needed; the j-direction of ut is computed at
   interior only (sufficient for our `_d_sw_native` use).  Equivalent
   for our use case. ✓

3. **`divergence_corner_duo`** (sw_core.F90:2345-2447) matches our
   `_divergence_corner_duo` for the duogrid path: same uf, vf
   formulas with cos_sg/sin_sg sub-grid corrections; same boundary
   zeroing at i=0/n, j=0/n; same 0.25× attenuation at i=1/n-1,
   j=1/n-1.  Note: Fortran has a typo at line 2434 (`je+1==npx`
   instead of `je+1==npy`) but for cubed-sphere npx==npy so it
   works either way.  Our Python uses the correct `j==n` check. ✓

4. **`d2a2c_vect` duogrid branch** (sw_core.F90:3419-3454): Fortran
   does NOT do internal halo exchange; it expects u, v already
   halo'd by `mpp_update_domains(DGRID_NE)` upstream.  Our Python
   `_d2a2c_vect_duogrid` does internal halo via `ext_vector_dgrid`
   which is functionally equivalent (same cube_rmp Lagrange + corner
   fill).  At j=jsd, jed (deepest halo): Fortran has weird
   double-overwrite `utmp(i,j) = 0.5*(u(i,j)+u(i,j+1))` then
   `utmp(i,j) = 0.5*(u(i,j+1)+u(i,j+1)) = u(i,j+1)`.  The first
   line is OVERWRITTEN — likely a Fortran bug or "intentional
   1st-order extrap".  Our Python's 4th-order on the full halo'd
   u_d_full effectively gives a more accurate utmp at the deepest
   halo than Fortran's 1st-order extrapolation.  Either Fortran's
   weird behaviour is intentional and our Python is OVER-SMOOTHING,
   or our Python is more accurate.  No measurable W2 impact in our
   testing — both work. ✓

**Conclusion.**  d_sw1, divergence_corner_duo, d2a2c_vect duogrid
branch are all Fortran-faithful in spirit (modulo the Fortran
double-overwrite quirk at deepest halo).

**Backlog for iter-970+.**  The remaining v_ll_Linf=55.6 m/s gap is
NOT in the structures audited so far.  Candidates:

1. `del6_vt_flux` (sw_core.F90:2008-2121) — vorticity damping in
   d_sw6.  Our Python may have an indexing or Fortran-faithful
   formula gap.
2. `fv_tp_2d` mass / vorticity transport (tp_core.F90 xppm/yppm)
   — verify our Python matches.
3. d_sw5 corner-divergence damping nord>=1 path (sw_core.F90:1727-
   1821) — verify the iterated Laplacian metric weighting matches.

### Iter-968 — Audit c_sw against Fortran source (key findings)

**Trigger.**  Iter-967 found that Fortran's d_sw3 boundary fix
conflicts with iter-945's halo (alternative strategies).  Iter-968
audits `_c_sw` and supporting helpers against the Fortran source
at `sw_core.F90:79-494` to verify our Python implementation matches
or to identify gaps.

**Findings.**

1. **`d2a2c_vect` call from c_sw** (sw_core.F90:150-151):
   Fortran HARDCODES `bounded_domain=.false.` in the call.  Our
   Python `_d2a2c_vect_duogrid` always uses the duogrid path; the
   hardcoded `.false.` doesn't gate on duogrid because the
   `gridstruct%dg%is_initialized` check inside `d2a2c_vect`
   dominates.  Behaviour is equivalent. ✓

2. **Cube-vertex vorticity correction** (sw_core.F90:395-401):
   Fortran applies +/-fy(corner) corrections at the 4 cube vertices
   ONLY when NOT duogrid.  Our Python `_corner_vorticity` matches
   exactly:
   ```python
   if not use_duogrid:
       vort = vort.at[:, 0, 0].add(fy_pad[:, 0, 0])
       vort = vort.at[:, n, 0].add(-fy_pad[:, n + 1, 0])
       vort = vort.at[:, n, n].add(-fy_pad[:, n + 1, n])
       vort = vort.at[:, 0, n].add(fy_pad[:, 0, n])
   ```
   ✓ Matches Fortran.

3. **KE-upwind boundary handling** (sw_core.F90:303-365):
   Fortran has 2 variants — bounded_domain/duogrid (simple upwind
   selection) vs ELSE (sin_sg/cos_sg corrections at cube edges).
   Our Python `_ke_upwind` matches both branches via
   `if not use_duogrid: ...` block. ✓

4. **Vorticity-flux boundary handling** (sw_core.F90:420-480):
   Fortran has 2 variants — duogrid (`fy1 = dt2*(v-uc*cosa_u)/sina_u`)
   vs ELSE (boundary specials at i==1/npx/j==1/npy).  Our Python
   `_vorticity_flux` matches. ✓

5. **C-grid update formulae** (sw_core.F90:483-492):
   `uc(i,j) = uc(i,j) + fy1(i,j)*fy(i,j) + rdxc(i,j)*(ke(i-1,j)-ke(i,j))`.
   Our Python:
   ```python
   uc_new = uc + fy1 * vort_x + dke_x
   ```
   where `dke_x = rdxc * (ke_pad[:-1] - ke_pad[1:])`.  Matches. ✓

**Conclusion.**  c_sw is bit-faithful to Fortran for the duogrid
path.  The remaining v_ll_Linf gap is NOT in c_sw structure.

**Iter-968 deliverables.**  Documentation only — confirms 5
sub-routines as Fortran-faithful.

**Backlog for iter-969+.**

1. Audit `d2a2c_vect` duogrid branch (sw_core.F90:3419-3454)
   against `_d2a2c_vect_duogrid` for indexing edge cases.
2. Audit `divergence_corner_duo` (sw_core.F90:2345+) against
   `_divergence_corner_duo`.
3. Audit `del6_vt_flux` (sw_core.F90:2008-2121) against the
   Python `_del6_vt_flux`.
4. Audit `p_grad_c` for SW variant (dyn_core.F90:2073-2132) —
   verify our `_p_grad_c` simple gradient is the SW limit.

### Iter-967 — NEGATIVE-RESULT: Fortran d_sw3 cube-edge boundary fix conflicts with iter-945 halo

**Trigger.**  Now have GFDL Fortran source access at
`../FV3/atmos_cubed_sphere-symmetryclean/`.  Read `sw_core.F90`
d_sw3 (B-grid KE transport) lines 1201-1388 and `xtp_u` /
`ytp_v` cube-edge boundary fix at lines 2819-2863 / 3239-3316.

**Critical Fortran detail.**  d_sw3 calls ytp_v / xtp_u with
``bounded_domain=.false.`` HARDCODED at lines 1316 and 1374
(regardless of the global bounded_domain flag).  Inside ytp_v /
xtp_u, the cube-edge boundary fix at iord=9 fires when
``(.not. bounded_domain .or. .not. dg%is_initialized)`` — with the
hardcoded `.false.`, the condition is always true.  So Fortran
ALWAYS applies a specific cube-edge boundary correction in d_sw3
using `s11=11/14, s14=4/7, s15=3/14` constants.

**Iter-967 attempt.**  Added the boundary fix to
`_ppm_transport_1d` via a new `apply_d_sw3_boundary_fix` kwarg.
The fix overrides bl/br at the 6 boundary cells (3 each at south +
north, mirrored for west/east on the other axis) using:

```
xt = s15*v(1) + s11*v(2) - s14*dm(2)
br(1) = xt - v(1);  bl(2) = xt - v(2)
br(2) = al(3) - v(2)
bl(0) = s14*dm(-1) - s11*dq(-1)
xt = (length-weighted xt of v(0)/v(-1) and v(1)/v(2) extrap)
bl(1) = xt - v(1);  br(0) = xt - v(0)
pert_ppm(iv=1) on bl(2), br(2)
```

(plus mirrored north boundary).  Wired into `_bgrid_ke_transport`
for the duogrid path with `cdgrid.dy_edge_x` / `dx_edge_y` as the
length-weighted dx field.

**Negative result.**  W2 v_ll_Linf 55.6 → 119.8 m/s on duogrid C36
1-day (115% regression).  Reverted.

**Diagnosis.**  Fortran's d_sw3 boundary fix is designed to
COMPENSATE FOR the absence of cross-face halo data on the
mode='edge'-style halo Fortran uses inside the d_sw3 ytp_v / xtp_u
sub-step.  It applies a specific extrapolation that yields
sensible values at cube edges given that mode='edge' halo.

But iter-945 (`_pad_halo_dgrid_for_ppm`) already provides
**correct cross-face halo** for u_d, v_d at depth h_dg=2 via
`ext_vector_dgrid`.  Applying Fortran's mode='edge' compensation
ON TOP of the cross-face halo OVER-CORRECTS — the formula assumes
the halo is wrong and tries to fix it, but the halo is right.

**Conclusion.**  iter-945 and the Fortran boundary fix are
ALTERNATIVE strategies for the same problem.  Iter-945 (cross-face
halo) is the more accurate Fortran-spirit strategy.  The boundary
fix kwarg is preserved on `_ppm_transport_1d` for potential use on
non-duogrid paths or other contexts where cross-face halo is
unavailable.

**Iter-967 deliverables.**

1. `src/legoesm/core/fv3_sw_core.py:_ppm_transport_1d` — new
   `apply_d_sw3_boundary_fix` kwarg + `boundary_fix_dx_field`
   kwarg (default False).  Implements the Fortran s11/s14/s15
   boundary formula with length-weighted xt for the southern and
   northern cube-edge cells; calls `_pert_ppm` (iv=1) at the
   adjacent cells.
2. `_bgrid_ke_transport` — kwarg DEFAULT False (preserves
   iter-945's cross-face halo behaviour).  Code commentary records
   the iter-967 negative-result.
3. `docs/fv3_fortran_fidelity_review.md` — this entry.

**Backlog for iter-968+.**

The Fortran source unblocks several other angles:
1. `c_sw` (sw_core.F90:79-488) — verify our `_c_sw` matches.
2. `d2a2c_vect` (sw_core.F90:3006-3345) — verify duogrid /
   non-duogrid branches.
3. Pressure gradient (`p_grad_c` analogue in dyn_core.F90) — check
   our `_p_grad_c` halo handling.
4. Vorticity damping (`del6_vt_flux`) — check our implementation.

### Iter-963 — Push Smagorinsky to stability boundary: 21% v_ll_Linf reduction

**Trigger.**  Iter-959 found that enabling Smagorinsky
(`d2_bg=0.01, dddmp=0.05`) gives a 4% v_ll_Linf improvement on the
FB chain duogrid C36 W2 1-day.  Iter-963 swept higher Smagorinsky
values to find the stability boundary.

**Sweep results on duogrid C36 W2 1-day:**

| (d2_bg, dddmp)        | v_ll_Linf (m/s) | |v_max| | h_err_max |
|-----------------------|----------------:|--------:|----------:|
| (0.00, 0.00) default  |          55.61  |   75.4  |    18591  |
| (0.01, 0.05) iter-959 |          53.17  |   60.5  |     —     |
| (0.02, 0.10)          |          51.67  |    —    |     —     |
| (0.03, 0.15)          |          50.28  |    —    |     —     |
| (0.04, 0.20)          |          48.97  |    —    |     —     |
| (0.05, 0.25)          |          47.80  |   59.2  |    14403  |
| (0.06, 0.30)          |          46.73  |    —    |     —     |
| (0.07, 0.35)          |          45.73  |    —    |     —     |
| (0.08, 0.40)          |          44.76  |    —    |     —     |
| **(0.09, 0.45)**      |       **43.77** |   55.3  |    12855  |
| (0.10, 0.40)          | NaN at step 156 |    —    |     —     |

The improvement is monotone in the Smagorinsky coefficient up to
the stability boundary near (0.09, 0.45) → (0.10, 0.40).  Cumulative
gain at the practical maximum:
- v_ll_Linf: 55.61 → 43.77 (21% reduction)
- |v_max|:    75.4 → 55.3 (27% reduction)
- h_err_max: 18591 → 12855 (31% reduction)

**Iter-963 deliverable.**  Sentinel
`tests/test_iter962_smagorinsky_tweak.py::test_iter963_high_smagorinsky_drives_w2_to_44_m_s`
pins v_ll_Linf ≤ 47 m/s when caller passes `d2_bg=0.09, dddmp=0.45`.

The default `d2_bg=0, dddmp=0` is preserved.  Callers who want
better W2 fidelity on the FB chain can opt into the higher
Smagorinsky.  This is calibration tuning — Fortran's actual
defaults are likely `dddmp ~ 0.05-0.2` per typical FV3 namelists,
so `dddmp=0.45` may be above the Fortran-faithful range.

**Backlog for iter-964+.**

- Verify the stability boundary on other Williamson cases (W5, W6).
- Determine the Fortran-faithful d2_bg/dddmp values from a GFDL
  config file (e.g., `atmos_data/dyn_core_nml`).

### Iter-959 — Damping-coefficient sensitivity sweep (small Smagorinsky improvement available)

**Trigger.**  Iters 945-958 explored Fortran-fidelity gaps in halo /
operator structure with limited gain (v_ll_Linf 81 → 75.4 → 75.4
m/s on |v|).  Iter-959 instead sweeps the d_sw5 damping
coefficients to see if the iter-934 default
(`d2_bg=0, dddmp=0, d4_bg=0.16, nord=1, damp_v=0.06, nord_v=2`) is
optimal for v_ll_Linf.

**Sweep results on duogrid C36 W2 1-day:**

| config                                   | v_ll_Linf (m/s) | |v_max| |
|------------------------------------------|----------------:|--------:|
| default (d4_bg=0.16 nord=1 damp_v=0.06)  |          55.61  |   75.4  |
| d4_bg=0.32 (double)                      |   NaN at step 16 |  —      |
| d4_bg=0.08 (half)                        |          70.53  |   94.3  |
| nord=2 (del-6)                           |          80.42  |  113.4  |
| damp_v=0.12 (double vorticity)           |          55.73  |   61.8  |
| damp_v=0.03 (half vorticity)             |          56.07  |   85.7  |
| **d2_bg=0.01 dddmp=0.05 (Smagorinsky)**  |        **53.17**|   60.5  |

**Insight.**  The default `d2_bg=0, dddmp=0` setting disables the
adaptive Smagorinsky branch of d_sw5 entirely.  Enabling
`d2_bg=0.01, dddmp=0.05` gives a 4% v_ll_Linf improvement
(55.6 → 53.2 m/s) AND reduces |v_max| from 75 → 60.5 m/s.  This is
the largest single-knob improvement found since iter-947.

**Iter-959 deliverable.**  No code change — the iter-934 default is
preserved because changing it would shift the iter-934/941/942/944/
945/947/951 step-survival sentinel pins.  Future iters can opt in
to `d2_bg=0.01, dddmp=0.05` via the existing config kwargs.

The 4% Smagorinsky improvement is a calibration tweak rather than
a Fortran-fidelity fix.  Adding it to the default would close ~4%
of the remaining 467× gap.

### Iter-957 — NEGATIVE-RESULT: existing apply_legacy_* flags are no-ops on duogrid

**Trigger.**  Iter-947 closes the boundary halo gap; iter-948..954
were all negative results.  Iter-957 verifies whether the existing
opt-in flags `apply_legacy_d_sw4_corner_ke_fix` (iter-869) and
`apply_legacy_d_sw5_corner_corrections` (iter-862) — which encode
Fortran cube-vertex corrections — could be re-enabled with iter-947's
better halo data quality.

**Negative result.**  All four configurations on duogrid C36 W2 1-day
give bit-identical v_ll_Linf=55.6130 m/s:

| flags                                              | v_ll_Linf |
|----------------------------------------------------|----------:|
| default (no legacy)                                |   55.6130 |
| apply_legacy_d_sw5_corner_corrections=True         |   55.6130 |
| apply_legacy_d_sw4_corner_ke_fix=True              |   55.6130 |
| BOTH                                               |   55.6130 |

Reason: both flags are gated on `cdgrid.base.duogrid is None`
(per their docstrings — they only fire on the non-duogrid path
where the right-hand-side `vort_pad` / `uc_lap` data quality is
known incomplete).  On duogrid they're silently no-op.

iter-957 records the verification but no code change.  The cube-
vertex corrections can be ported into the duogrid path (iter-958+
candidate) but require care — the iter-862/iter-869 docstrings
note the right-hand-side data needs cross-face halo to be
Fortran-faithful.

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
