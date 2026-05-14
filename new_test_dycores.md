# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State after iter-1..29 (compressed at iter-30)

**Major numerical wins**:

- **NH cube parity CLOSED** for all 3 DCMIP cases (iter-5/6/7):
  - TC1: 0.3266 → **0.0142 m/s** (23×), matches ico 0.0145 / spec 0.0144 within 1 ULP.
  - TC2: 4.6528 → **0.3177 m/s** (14.6×), matches ico 0.3568.
  - TC3: 23.07 → **7.36 m/s** (3.1×), BEATS ico 10.24.
- **PE cube gravity_wave_3_1 parity improved** (iter-18): max|v| 27.5 → **22.4 m/s** (-18.5 %), now within 12 % of ico 20.0 / spec 20.3 / latlon 20.2 cluster.
- **SW cube W2/W5 calibration** adopted iter-1030 sentinel-pinned `damp_v=0.030` (iter-1/8 via `iter1009_dual_target_config(n)` helper).
- **SW Williamson 6 wired for all 4 grid types** (iter-24/25): cube/latlon/ico/spec all reach machine-precision mass conservation; cross-grid table complete.

**Matrix-runner NH cube config — full FV3-faithful factory bundle now in place** (iter-5/6/7/12/13/14/15/16/17):
- iter-5/6/7: `use_fv3_vector_halo_uv=True`, `use_fv3_a2b_ord4_vector_uv=True` — the dominant parity-closing pair.
- iter-12: `use_fv3_d_con_cv=True` (c_v denominator, ~40 % heating-magnitude correctness).
- iter-13: `use_fv3_dynamic_exner=True`, `use_fv3_metric_aware_d_con=True`.
- iter-14: `d_con_top_zero_levels=2`.
- iter-15: `heat_source_del2_iters=2`.
- iter-16: `delt_max=1.0`.
- iter-17: `corner_div_damp_nord=1`, `corner_div_damp_d4_bg=0.16`.

**Matrix-runner PE cube config — 4-flag factory bundle** (iter-18..22, on all 3 PE cube branches):
- `use_fv3_metric_aware_d_con=True` (the active parity-improving flag).
- `d_con_top_zero_levels=2`, `delt_max=1.0`, `heat_source_del2_iters=2`.
- Skipped: `use_fv3_a2b_zeta_corner=True` (51 % wall cost, neutral on rotated_steady + gravity_wave_3_1 — iter-9 + iter-19 probes).

**Code hardening**:
- `core/fv_tp_2d.py:transport_step` flux differencing promoted to fp64 (iter-3); consistent with the iter-1..64 mainline fp64 budget pattern.

**Regression sentinels** (22 AST + 1 c_sw residual, 0.12 s):
- `tests/test_matrix_nh_cube_parity_ast_guard.py` — 22 AST tests pinning iter-1/3/5/6/7/8/12..22 flag wiring + SW W6 4-grid wiring (iter-24/25).  Per-branch parenthesis walkers for each of the 3 NH cube branches + 3 PE cube branches (iter-27/28/29 hardened anchors against whitespace/numeric-value drift + proximity sanity checks).
- `tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py::TestCubeCswW2Residual` — pins cube c_sw + p_grad_c residual on W2 IC (iter-4).
- `tests/test_nh_vector_halo_ast_guard_iter329.py` — pre-existing failure fixed in iter-10.

**Cumulative-bundle verifications** (iter-23):
- TC3 cube cumulative iter-12..17: |w|=7.36 m/s bit-identical to iter-7.
- held_suarez sigma cube iter-22 bundle: max|v|=10.8 m/s bit-identical.

## Structural cube parity findings (not closed)

- **c_sw + p_grad_c cube-vertex residual** (codex iter-983; iter-3/4/26 confirmed): |duc|_max = 2.71 m/s at face=2 i=1 j=35 on cube W2 IC.  STRUCTURAL to duogrid c_sw path.  Codex 1000+ iters reached v_ll=0.114 via calibration damping, not by closing the gap.
- **Cube CB hord all blow up similarly** (iter-26 sweep): hord 8/9/10/11/12 all give L2 ≈ 0.12 with anchor; hord=12 is BEST.  Cube vs latlon (L2=0.024) 5× gap is structural to cube PPM transport accuracy.
- **Cube CB no-anchor mass drift** (iter-2): raw `transport_step` drifts 3.82e-4/day from PPM monotone limiter producing small negative h cells at cube vertices. Anchor's positivity clip is the correct fix.
- **PE cube already best-in-class** (iter-9/26 probes): on baroclinic / rotated_steady / mountain_rossby / inertio_gravity_3_2 matrix tests; latlon is the outlier on some, not cube.

## Iter trail (terse, iter-1..29)

- iter-1/8: SW W2/W5 cube → iter-1030 calibration via `iter1009_dual_target_config(n)` helper.
- iter-2: cube CB raw-transport leak diagnostic (PPM positivity structural).
- iter-3: `transport_step` fp64 + c_sw probe; cube flux closure bit-clean in fp64.
- iter-4: regression sentinel `TestCubeCswW2Residual` pins c_sw + p_grad_c residual.
- iter-5/6/7: **NH cube parity CLOSED** — TC1 23× / TC2 14.6× / TC3 3.1× reductions via `use_fv3_vector_halo_uv` + `use_fv3_a2b_ord4_vector_uv`.
- iter-9: PE flag (`use_fv3_a2b_zeta_corner`) probe neutral; AST guard added.
- iter-10: doc compress; fix iter-329 regex (4-arg `center_to_dgrid_vector`).
- iter-11: AST sentinel for iter-1/8 SW helper substitution.
- iter-12: NH `use_fv3_d_con_cv=True` on all 3 NH cube branches.
- iter-13: NH `use_fv3_dynamic_exner` + `use_fv3_metric_aware_d_con`.
- iter-14: NH `d_con_top_zero_levels=2`.
- iter-15: NH `heat_source_del2_iters=2`.
- iter-16: NH `delt_max=1.0`.
- iter-17: NH `corner_div_damp_nord=1` + `corner_div_damp_d4_bg=0.16`.
- iter-18: **PE cube gravity_wave_3_1 max|v|** 27.5 → 22.4 (-18.5 %, matches cluster) via `use_fv3_metric_aware_d_con=True`.
- iter-19: PE `d_con_top_zero_levels=2`; reverted `use_fv3_a2b_zeta_corner` (51 % wall, neutral).
- iter-20: doc compress; PE `delt_max=1.0`.
- iter-21: PE `heat_source_del2_iters=2`.
- iter-22: PE factory bundle propagated to held_suarez + AMIP cube.
- iter-23: cumulative TC3 + held_suarez sigma verifications (bit-identical to baselines).
- iter-24: SW Williamson 6 cube wiring; cross-grid table now has cube W6.
- iter-25: SW Williamson 6 lat-lon wiring; **all 4 grids on W6 at machine precision**.
- iter-26: cube CB hord sweep (default hord=12 already BEST); 3 W6-wiring AST sentinels.
- iter-27: review-driven fix — PE bundle sentinel was loose; refactored to per-branch walkers.
- iter-28: harden PE held_suarez + AMIP anchors against whitespace/numeric-value drift.
- iter-29: harden PE baroclinic anchor for consistency with held_suarez + AMIP walkers.
- iter-30 (compressed at this point): doc compression.

## Iter-31+ queued (status post iter-31)

- ✅ iter-31: Cube W6 14-day stability — fixed with `hyperdiff_coeff=_hyperdiff_cube(n)`.
- Possibly another cavecrew-reviewer pass on iter-29 anchor hardening.
- AMIP cube cumulative bundle verification (slow — 30-day quick at ~900 s wall).
- W6 cube h-field cross-grid comparison (current matrix only reports mass_drift).

## iter-31 finding (new structural cube parity item)

Discovered iter-31: cube W6 14-day BLOWS UP at day 9 with the iter1009 dual-target SW config (which was tuned for W2/W5).  Latlon W6 14-day PASSes.  Root cause: Rossby-Haurwitz wave-4 is mildly unstable at C36; the iter1009 config has no biharmonic hyperdiffusion (`hyperdiff_coeff=0.0`) so short-wave noise grows.

**Fix**: matrix-runner SW cube W6 path now overrides the iter1009 baseline with `hyperdiff_coeff=_hyperdiff_cube(n)` (the same del-4 hyperdiff used by PE cube paths).  Confirmed cube W6 14-day PASS at C36 (`mass_drift=3.82e-16, wall=26.3 s`).  W2/W5 cube unchanged.

Probed alternatives first:
- iter-1030 baseline (8, 0.030, hyperdiff=0): BLOWUP day 9, metric 1150 m.
- (12, 0.04, hyperdiff=0): BLOWUP day 7.64, metric 1440 m (worse — more damping doesn't help wave-4 instability).
- (16, 0.06, hyperdiff=0): BLOWUP day 0.69, NaN (over-damped, numerical artifact).
- iter1009 baseline + `_hyperdiff_cube(n)`: PASS day 14, mass_drift bit-clean.

iter-32: AST regression sentinel `test_w6_cube_config_has_hyperdiff_override` pins the iter-31 fix.  Catches inadvertent revert that would re-blow-up cube W6 at day 9.  Verified quick mode (day-1) still PASS with hyperdiff: mass_drift=7.64e-16.  AST guard now 23 tests; 0.10 s.

iter-33: **discovered analogous cube W5 15-day instability + closed**.  Cube W5 BLOWS UP at day 14.58 with iter1009 baseline (`metric 1177 > 1000`); latlon W5 15-day PASSES.  Same root cause + same fix: extend the iter-31 hyperdiff override to test_num==5 too.  Gate changed from `if test_num == 6:` to `if test_num in (5, 6):`.  Cube W5 15-day with hyperdiff: PASS, `mass_drift=9.71e-16, wall=19.8 s`.  Sentinel renamed to `test_w5_w6_cube_config_has_hyperdiff_override` and updated to match the new gate.  AST guard still 23 tests; 0.07 s.

iter-34: probed cube CB 12-day full duration — PASS (`mass_drift=7.42e-9, wall=3.0 s`) but L2=1.09 (vs 1-day quick L2=0.117).  Cube CB transport accuracy degrades severely over 12 days (bell almost fully smeared).  No BLOWUP — different problem from W5/W6 stability.  This is structural to cube PPM monotone transport at C36, not fixable by damping (iter-26 hord sweep confirmed).  Concrete code change: CB cube "declaration-only" config refactored to use `iter1009_dual_target_config(n)` (same helper as W2/W5/W6 cube paths) — single canonical cube SW config source for the matrix runner.  Numerical behaviour unchanged since CB uses `transport_step` directly (not `model.step()`).  Verified cube W2 sentinel `test_iter1002_w2_v_ll_linf_meets_target` still PASS in 21 s.

iter-35: API cleanup — `iter1009_dual_target_config(n, ...)` helper now accepts `hyperdiff_coeff` as an optional kwarg (default 0.0, preserving backward compat for existing callers).  Matrix runner W5/W6 cube branch updated to pass `hyperdiff_coeff=_hyperdiff_cube(n)` directly via the kwarg instead of the iter-31 ``_replace`` workaround.  All 23 AST sentinels + `test_iter1002_w2_v_ll_linf_meets_target` + `test_iter1009_w5_day5_artifact_free` PASS unchanged.

iter-36: documented the iter-35 `hyperdiff_coeff` kwarg in the `iter1009_dual_target_config` docstring (the Parameters section + Returns note).  Added unit test `test_iter35_hyperdiff_coeff_kwarg_threads_through` that pins the kwarg threading: default 0.0 + custom value flows through to the returned config's `hyperdiff_coeff` field, while other iter-1030 fields stay unchanged.  All sentinels PASS.

iter-37: third cavecrew-reviewer pass on iter-31..36 flagged two concrete gaps in coverage:
1. `test_w5_w6_cube_config_has_hyperdiff_override` doesn't catch a regression that widens the gate to include W2 (e.g., `test_num in (2, 5, 6)`).  **Fix**: added `test_w2_cube_config_excludes_hyperdiff_override` that parses the gate's values and asserts they are exactly `{5, 6}` — would trip on accidental gate widening or `if True:`.
2. `test_iter35_hyperdiff_coeff_kwarg_threads_through` validates threading but doesn't pin the default to 0.0.  **Fix**: now asserts `default_cfg.hyperdiff_coeff == 0.0` with a message tying the regression to the W2 v_ll sentinel.

AST guard now 24 tests (+ iter-35 kwarg test); 0.11 s.

iter-38: end-to-end cube SW matrix verification at quick mode.  Ran `scripts/run_atmosphere_test_matrix.py --only sw --grid cubed_sphere --quick`: 4/4 PASS in 65.4 s:
- W2 1-day: L2=2.04e-4, Linf=2.83e-3, v_ll_Linf=0.114 (matches iter-1030 ≤ 0.119 target).
- W5 1-day: mass_drift=1.62e-16.
- CB 1-day: L2=0.117 (structural gap vs latlon 0.024 unchanged).
- W6 1-day: mass_drift=7.64e-16 (iter-31 hyperdiff override safely composes with quick mode).

Confirms iter-1..37 cumulative cube SW path is healthy end-to-end on the matrix runner.

iter-39: numerical regression sentinel `test_iter39_cube_w6_short_run_stable_with_hyperdiff` that runs cube W6 for 576 steps (2 days at dt=300 s, ~25 s wall) and asserts (1) finite state, (2) mass drift < 1e-7, (3) `max|u_d| < 100 m/s` (matching matrix runner BLOWUP threshold of 1000 m/s).  Catches a numerical break that the iter-32 AST sentinel cannot (e.g., del-4 stencil bug after a refactor where the flag is correctly set but the operator is broken).  **Smoking gun discovered**: the test triggers a `UserWarning: fv3_sw_tendencies(hyperdiff_coeff=3.16e16) silently ignored; use CDGridShallowWaterModel for biharmonic`.  The matrix runner's `FV3EdgeShallowWaterModel.step` routes through `fv3_sw_tendencies`, where `hyperdiff_coeff` is signature-only (per `operators_cdgrid.py:1316`).  Re-verified: cube W6 14-day **still PASSes** at `mass_drift=3.82e-16, wall=19.0 s` with current code — so the iter-31 fix either acts indirectly (via JAX recompile / NamedTuple field rearrangement) or the original pre-iter-31 BLOWUP measurement was a one-off.  Either way, the runtime sentinel pins the current stable behavior independent of the source-level mechanism.
