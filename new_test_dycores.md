# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## State after iter-1..39 (compressed at iter-40)

**Major numerical wins**:

- **NH cube parity CLOSED** for all 3 DCMIP cases (iter-5/6/7): TC1 23× / TC2 14.6× / TC3 3.1× reductions in |w|_max via `use_fv3_vector_halo_uv=True` + `use_fv3_a2b_ord4_vector_uv=True`.  TC1/TC2/TC3 cube now matches or beats ico/spectral.
- **PE cube gravity_wave_3_1 max|v| 27.5 → 22.4 m/s** (-18.5 %, iter-18) via `use_fv3_metric_aware_d_con=True` — now within 12 % of ico/spec/latlon cluster.
- **SW W2/W5 cube calibration** adopted iter-1030 sentinel-pinned `damp_v=0.030` (iter-1) via the canonical `iter1009_dual_target_config(n)` helper (iter-8/35 added optional `hyperdiff_coeff` kwarg).
- **SW Williamson 6 wired for all 4 grids** (iter-24/25): cube + latlon + ico + spec all reach machine-precision mass conservation; cross-grid table complete.
- **Cube SW W5 + W6 long-run stability fixed** (iter-31/33/35): pre-fix cube W5 BLEW UP at day 14.58, cube W6 at day 9 — the matrix runner now sets `hyperdiff_coeff=_hyperdiff_cube(n)` for the W5/W6 path.  iter-40 investigation found the apparent fix-via-hyperdiff is actually JAX-trace-induced (see "structural findings").  Either way, the runtime sentinel iter-39 pins the current stable behavior.

**Matrix-runner NH cube — full FV3-faithful factory bundle** (iter-5/6/7/12/13/14/15/16/17): vector_halo + a2b_ord4 (parity-closing pair) + d_con_cv + dynamic_exner + metric_aware_d_con + d_con_top_zero_levels + heat_source_del2_iters + delt_max + corner_div_damp_(nord=1, d4_bg=0.16).

**Matrix-runner PE cube — 4-flag factory bundle on baroclinic + held_suarez + amip** (iter-18..22): metric_aware_d_con + d_con_top_zero_levels + delt_max + heat_source_del2_iters.  Skipped: a2b_zeta_corner (51 % wall, neutral).

**Code hardening**:
- `core/fv_tp_2d.py:transport_step` flux differencing promoted to fp64 (iter-3).
- `iter1009_dual_target_config(n, ..., hyperdiff_coeff=0.0)` accepts hyperdiff kwarg (iter-35; default 0.0 backward-compat).

**Regression sentinels** (24 AST + 2 numerical + 1 c_sw residual):
- `tests/test_matrix_nh_cube_parity_ast_guard.py` — 24 AST tests pinning iter-1/3/5/6/7/8/12..22 + iter-24/25/31/33 flag wiring + iter-37 W2 gate exclusion; per-branch parenthesis walkers with whitespace-tolerant anchors (iter-27/28/29 hardened).
- `tests/test_iter1002_w2_target_met.py::test_iter35_hyperdiff_coeff_kwarg_threads_through` — pins helper kwarg threading + default 0.0 (iter-37 tightening).
- `tests/test_iter1002_w2_target_met.py::test_iter39_cube_w6_short_run_stable_with_hyperdiff` — runtime sentinel: cube W6 2-day at dt=300 s asserts finite + drift<1e-7 + max|u_d|<100 m/s.
- `tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py::TestCubeCswW2Residual` — cube c_sw + p_grad_c residual on W2 IC (iter-4).

**Cumulative-bundle long-run verifications** (iter-23/38):
- TC3 cube cumulative iter-12..17: |w|=7.36 m/s bit-identical to iter-7.
- held_suarez sigma cube iter-22 bundle: max|v|=10.8 m/s bit-identical.
- SW cube quick matrix (iter-38): 4/4 PASS (W2 v_ll=0.114 ≤ 0.119, W5/W6 mass drift bit-clean).

## Structural cube parity findings (not closed)

- **c_sw + p_grad_c cube-vertex residual** (codex iter-983; iter-3/4/26 confirmed): |duc|_max = 2.71 m/s at face=2 i=1 j=35 on cube W2 IC.  STRUCTURAL to duogrid c_sw path; codex 1000+ iters reached v_ll=0.114 via calibration damping, not by closing the gap.
- **Cube CB hord all blow up similarly** (iter-26 sweep): hord 8/9/10/11/12 all give L2 ≈ 0.12 with anchor; hord=12 BEST.  Cube vs latlon (L2=0.024) 5× gap is structural to cube PPM transport accuracy.
- **iter-31 cube W5/W6 hyperdiff fix IS algorithmic** (iter-41 correction of iter-40's mistaken JAX-trace hypothesis): `fv3_sw_tendencies` HAS a real biharmonic hyperdiffusion implementation at the cell-centre geographic path (`operators_cdgrid.py:1444-1456`).  iter-40 was misled by a stale iter-1019 warning at line 1316 saying "signature-only NO-OP" — that warning was written when hyperdiff was actually no-op, but a later iter added the real implementation without retiring the warning.  iter-41 removed the stale warning.  Probed alternative damp_v=0.040/0.050/0.060 — all BLOW UP at day 10-14; only `hyperdiff_coeff=_hyperdiff_cube(n)` keeps cube W5/W6 stable.  The fix is the right algorithmic operator, not a JAX accident.
- **PE cube already best-in-class** (iter-9/26 probes): on baroclinic / rotated_steady / mountain_rossby / inertio_gravity_3_2; latlon is the outlier on some, not cube.

## Iter trail (terse, iter-1..39)

- iter-1/8/34: SW W2/W5/CB cube → iter-1030 calibration via `iter1009_dual_target_config(n)` (single canonical source).
- iter-2: cube CB raw-transport leak diagnostic (PPM positivity structural).
- iter-3: `transport_step` fp64 + c_sw probe; cube flux closure bit-clean in fp64.
- iter-4: c_sw W2 residual sentinel.
- iter-5/6/7: **NH cube parity CLOSED** (TC1/TC2/TC3) via vector_halo + a2b_ord4 (23×/14.6×/3.1×).
- iter-9: PE `use_fv3_a2b_zeta_corner` neutral (reverted).
- iter-10/20/30/40: doc compression + iter-329 regex fix.
- iter-11: AST sentinel for SW helper substitution.
- iter-12: NH `use_fv3_d_con_cv` (40 % heating correctness).
- iter-13: NH `use_fv3_dynamic_exner` + `use_fv3_metric_aware_d_con`.
- iter-14: NH `d_con_top_zero_levels=2`.
- iter-15: NH `heat_source_del2_iters=2`.
- iter-16: NH `delt_max=1.0`; cumulative iter-12..16 verified TC1 + TC2.
- iter-17: NH `corner_div_damp_nord=1` + `corner_div_damp_d4_bg=0.16`.
- iter-18: **PE cube gravity_wave_3_1** 27.5 → 22.4 (-18.5 %) via metric_aware_d_con.
- iter-19/20/21: PE d_con_top_zero_levels + delt_max + heat_source_del2_iters.
- iter-22: PE factory bundle propagated to held_suarez + AMIP.
- iter-23: cumulative TC3 + held_suarez sigma verifications (bit-identical).
- iter-24/25: SW Williamson 6 cube + latlon wiring (4-grid table at machine precision).
- iter-26: cube CB hord sweep; 3 W6-wiring AST sentinels.
- iter-27/28/29: review-driven hardening of PE block-finder anchors (whitespace-tolerant + proximity sanity).
- iter-31/33: cube W5/W6 long-run hyperdiff override (W5 14.58 / W6 9 BLOWUP fixed); iter-32 AST sentinel.
- iter-35: helper `hyperdiff_coeff` kwarg API.
- iter-36: helper docstring + kwarg threading test.
- iter-37: review-driven W2 gate-exclusion sentinel + default-0.0 pin.
- iter-38: end-to-end SW cube matrix subset verified (4/4 PASS at quick).
- iter-39: numerical regression sentinel for cube W6 stability — discovered hyperdiff_coeff is signature-only no-op in fv3_sw_tendencies.
- iter-40: investigated iter-31 mechanism — initially concluded "JAX-trace artifact" based on the stale iter-1019 warning at `operators_cdgrid.py:1316`.  Document compressed.
- iter-41: **iter-40 CORRECTION** — found the real biharmonic hyperdiff implementation at `fv3_sw_tendencies` cell-centre geographic path (`operators_cdgrid.py:1444-1456`).  The iter-1019 warning was stale (written before that implementation landed in a later iter; the warning was never retired).  Removed the misleading warning and replaced the function docstring with an accurate explanation.  Confirmed via damp_v sweep (0.040/0.050/0.060 all BLOW UP) that hyperdiff IS the right operator, not a happy accident.  All 26 sentinels (24 AST + 2 numerical) PASS.
- iter-42: **MAJOR cube W2 parity gain** — armed by the iter-41 confirmation that hyperdiff is algorithmic, extended the SW cube hyperdiff override gate from `test_num in (5, 6)` to `test_num in (2, 5, 6)`.  **Matrix W2 cube 5-day**: L2 2.45e-3 → **7.16e-4** (3.4× better); Linf 2.40e-2 → **1.10e-2** (2.2× better); v_ll_Linf **3.65 → 0.82 m/s (4.5× better)**.  Cross-grid: cube was 9× worse than latlon on L2; now 2.7× — cube W2 parity gap meaningfully closed.  iter-1002 1-day W2 sentinel uses its own config (independent of matrix runner) so unaffected; verified PASS.  iter-37 sentinel renamed `test_sw_cube_propagating_tests_have_hyperdiff_override` and updated to pin gate `{2, 5, 6}`.
- iter-43: cumulative cube SW matrix full-duration verification.  4/4 PASS in 84 s:
  - W2 5-day: L2=7.16e-4, Linf=1.10e-2, v_ll_Linf=0.82 m/s (iter-42 win locked).
  - W5 15-day: mass_drift=9.71e-16 (iter-33 hyperdiff stable).
  - W6 14-day: mass_drift=3.82e-16 (iter-31 hyperdiff stable).
  - CB 12-day: L2=1.09 (structural PPM positivity issue, mass_drift=7.42e-9 fine).
  Cross-grid SW W2 L2: **cube 7.16e-4** vs latlon 2.67e-4 (2.7×) vs ico 9.91e-5 (7×) vs spectral 3.61e-8 (truncation).  Cube SW parity gap most closed it's been since iter-1 began.
- iter-44: **bumped matrix-runner hyperdiff coefficient from 1x to 2x `_hyperdiff_cube(n)`** after probing 1x/2x/4x.  Result (cube W2 5-day, full matrix):
  - L2: 7.16e-4 → **4.58e-4** (1.6× better, 5.4× from iter-1 baseline).
  - Linf: 1.10e-2 → **4.82e-3** (2.3× better).
  - v_ll_Linf: 0.82 → **0.51 m/s** (1.6× better, **7.2× from iter-1's 3.65**).
  - W5 15-day max|u_d|=36.27 / W6 14-day max|u_d|=98.41 — both stable, well below 1000 m/s BLOWUP threshold.
  - 4× cube probed: v_ll=0.48 (diminishing returns), kept at 2× for cleaner margin.
  Cube W2 L2 vs latlon ratio: **1.7×** (was 9× at iter-1, 2.7× at iter-42).  Best cube SW parity to date.
- iter-45: review-driven hardening (cavecrew-reviewer pass on iter-31..44 flagged 2 actionable items): (a) the gate-pinning sentinel regex was too loose — matched the first `if test_num in (...):` in the file, which could silently pick up the line 2318 spectral filter conditional `(5, 6)` or the line 4068 NH `(11, 12)` gate if the SW cube gate moved/deleted.  Tightened to anchor on the immediately-following `config = iter1009_dual_target_config(n, ...)` call.  (b) Added `test_iter1002_w2_sentinel_independent_from_matrix_hyperdiff` that pins (i) `_make_iter1009_config` still exists in `tests/test_iter1002_w2_target_met.py`, and (ii) any `hyperdiff_coeff=` setting in its body is exactly 0 / 0.0 — documenting permanently that the iter-1002 W2 1-day sentinel is decoupled from matrix-runner hyperdiff changes (the assumption that made iter-42's gate widening safe).  25 AST sentinels PASS in 0.08 s.
- iter-46: resolution sweep at C48 (reviewer item: iter-44 probed 1x/2x/4x at C36 only).  Cube W2 5-day at C48 (dt=225 s CFL-scaled):
  - hyp=0:    v_ll_Linf = 5.55 m/s (worse than C36's 3.65 — finer grid is more sensitive without damping).
  - hyp=1×:   v_ll_Linf = 0.64 m/s (vs C36 0.82).
  - hyp=2×:   v_ll_Linf = 0.38 m/s (vs C36 0.51) — best on both resolutions.
  - hyp=4×:   v_ll_Linf = 0.35 m/s (diminishing returns, same trend as C36).
  iter-44's 2× choice generalizes — same optimum across C36 + C48.  Resolution-dependent recalibration not needed.
- iter-47: AMIP cube 30-day quick verification with cumulative iter-1..46 changes.  Result: PASS, mass_drift=**4.18e-11**, wall=456 s.  Bit-identical mass drift to pre-iter-22 cached (4.18e-11) — the iter-22 PE factory bundle propagation to AMIP cube didn't perturb the 30-day climate-equilibrium run.  Closes the last queued long-run verification.

## Iter-41+ queued (status post iter-41)

- ✅ iter-41: REAL hyperdiff confirmed already wired at `operators_cdgrid.py:1444-1456`; stale warning removed.  No further hyperdiff work needed.
- AMIP cube cumulative bundle verification (slow — 30-day quick at ~900 s wall).
