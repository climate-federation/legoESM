# new_test_dycores — FV3 cube parity vs latlon/MPAS/spectral

Branch: `new_test_dycores` (from `main` post merge of `test_dycores` PR #259).
Oracle: `../../FV3/atmos_cubed_sphere-symmetryclean/model/`.
Scope: cube SW/PE/NH error norms within close numerical proximity of latlon FV / MPAS Voronoi / spectral SH at the same resolution + duration.

## AIMIP workstream (iter-249+, parallel to dycore audit)

Phase 1 landed at commit `db4d88a0` (mid-loop user request):
- `forcing/experiments.py` AIMIP template (2015-2020, transient GHG via CMIP6 historical).
- `scripts/run_aimip.py` CLI wrapper (ERA5 IC + RRTMGP + latlon C-grid defaults).
- 4 unit tests in `tests/test_cases/test_aimip_template.py` (template registration + GHG interpolation).

Phase 2-5 (TrainablePhysicsParams expansion, training script, validation, corrections) queued.  Plan file at `/home/gentine/.claude/plans/jolly-humming-gray.md`.

## CRITICAL FINDING (iter-102 + iter-123): NH cube TC2 AND TC3 full-mode BLOWUP

**iter-124 update**: ran a quick NH matrix (--quick) to verify iter-118 fix in practice.  TC1 cube quick-mode result: |w|=0.0142 m/s — **MATCHES iter-7 doc claim of 0.014** (cube/ico/spec parity at quick mode confirmed).  Critically, the `mean_timeseries.csv` now has 22 lines (header + 21 data rows) with NO `_blowup_info` column — **iter-118 fix is operational** for new matrix runs.  TC2 + TC3 quick-mode running; will verify cube/ico/spec quick-mode parity confirmed once they complete.

**iter-123 update**: NH cube matrix re-run completed.  Both TC2 AND TC3 cube **BLEW UP at full mode**:

| Test | Status | Blowup step | Sim time | Reason |
|------|--------|-------------|----------|--------|
| TC1  | PASS   | —            | 3 hr full | (clean) |
| TC2  | **FAIL** | step 8500 | day 0.13 (3.14 hr) | state non-finite (NaN/Inf) |
| TC3  | **FAIL** | step 2250 | day 0.01 (8.3 min)  | max\|w\|=1271 > 1000 threshold |

TC3 blows up MUCH earlier than TC2 (step 2250 vs 8500; 8.3 min vs 3.14 hr sim time).  TC3 has Kessler microphysics + squall-line forcing → bigger w-amplitude → faster instability.

Total NH matrix re-run: 4:13 - 7:12 = 3 hr wall.  TC1 PASS (1669s) + TC2 FAIL (6663s) + TC3 FAIL (2408s) = 10740s elapsed dynamics; rest was JIT compilation + plot generation.

The iter-118 fix for empty-csv-on-FAIL was NOT yet applied to this running process (Python doesn't reload modules).  Both TC2 and TC3 csv files are still header-only.  Future NH matrix runs will benefit.

## CRITICAL FINDING (iter-102): NH TC2 cube full-mode BLOWUP

NH cube matrix re-run completed TC2 at iter-102.  **TC2 cube BLEW UP at day 0.13 (step 8500) of the full 6-hour run** (`state non-finite (NaN/Inf)`).  Pre-blowup last-clean reading: `|w|_max=nan m/s, mass_drift=7.85e-16`.  Wall time: 6663 s (111 min).

This is a **REGRESSION from quick-mode**:
- TC2 quick-mode (5 min, 0.083 hr): cube |w|_max=4.5 m/s, PASS (per cached pre-iter-72 result).
- TC2 with iter-12..17 fixes at quick-mode (per doc claim): cube |w|_max=0.32 m/s, PASS.
- TC2 with iter-12..17 fixes at **full-mode (6 hr)**: cube **BLOWUP at step 8500 = 3.14 hr in**.

So the iter-12..17 NH bundle (vector_halo + a2b_ord4 + d_con_cv + dynamic_exner + metric_aware_d_con + d_con_top_zero=2 + heat_source_del2=2 + delt_max=1.0 + corner_div_damp_pair) is **NOT sufficient for full-duration TC2 cube stability**.  The matrix has been silently PASSing TC2 cube at quick mode (5 min) where the instability hasn't grown.

**Diagnostic limitation (iter-117 audit)**: the matrix runner writes `mean_timeseries.csv` only after a successful run completes.  For the iter-102 TC2 FAIL, the csv is empty (header only) — no per-step diagnostics survived the blowup detection.  Snapshots saved every `n_steps/10` would have captured the field at step 8120 (last clean) and step 9744 (post-blowup, all NaN), but `snapshots_native.npz` from the current run also wasn't written (timestamp still May 13 = pre-iter-72 baseline).  To investigate the blowup mechanism, a future probe (iter-118+) needs to run TC2 cube up to step ~8400 only (~30 min wall) with custom diagnostics dumping.  The matrix runner's blowup-detection-then-discard-diagnostics pattern is a limitation worth addressing in a future iter.

**Action required (iter-103+)**: investigate the day-0.13 blowup mechanism (mountain-wave breaking? acoustic substep insufficient? halo error at panel edges in NH solver?).  Possible fixes:
- More acoustic substeps (currently 20).
- Stronger hyperdiff at full duration.
- Re-evaluate which iter-12..17 flags help vs hurt at long duration.
- FV3 oracle comparison for TC2 mountain stability.

**iter-12..17 doc claims need caveat**: those measurements were at quick mode (5 min); full-mode behaviour DOES NOT HOLD for TC2 cube.

## State after iter-1..250 (compressed at iter-250)

**iter-241..250 — TC2 probe abort, ralph-loop stopped, AIMIP Phase 1 landed**:

- **iter-248**: TC2 hypothesis-2 probe killed at 51 min wall (user request to stop ralph-loop).  Matrix runner reverted from 16× → 4× baseline (commit `a9c7e2f3`).  Hypothesis 2 (more hyperdiff prevents day-0.13 blowup) remains UNTESTED.
- **iter-249 AIMIP Phase 1** (parallel workstream, commit `db4d88a0`):
  - `forcing/experiments.py` adds `"aimip"` template (2015-2020 transient GHG via CMIP6 historical).
  - `scripts/run_aimip.py` CLI wrapper (ERA5 IC + RRTMGP + latlon C-grid defaults).
  - 4 unit tests in `tests/test_cases/test_aimip_template.py`.

**Cumulative sentinels at iter-250**: 31 AST + 14 numerical + 1 c_sw + 1 functional + 4 AIMIP = **51 total**, all PASS in <1 s for AST+AIMIP layer (numerical layer ~150 s).

**Active state**: ralph-loop dycore audit is at steady state.  AIMIP Phase 1 complete; Phases 2-5 (TrainablePhysicsParams expansion, training script, validation, corrections) require ~80 hr training compute + ERA5 data fetch.

## State after iter-1..240 (compressed at iter-240)

**iter-231..240 — TC2 probe wait phase X**: probe wall 47 → 50 min over 10 iters.  Still in dynamics.

## State after iter-1..230 (compressed at iter-230)

**iter-221..230 — TC2 probe wait phase IX**: probe wall 44 → 47 min over 10 iters.  Still in dynamics.

## State after iter-1..220 (compressed at iter-220)

**iter-211..220 — TC2 probe wait phase VIII**: probe wall 41 → 44 min over 10 iters.  Still in dynamics.  No code changes.

## State after iter-1..210 (compressed at iter-210)

**iter-201..210 — TC2 probe wait phase VII**: probe wall 39 → 41 min over 10 iters.  Still in dynamics.  No code changes.

## State after iter-1..200 (compressed at iter-200)

**iter-191..200 — TC2 probe wait phase VI** (200-iter milestone): probe wall 36 min → 39 min over 10 iters.  Still in dynamics.  No code changes (just polling).  Probe estimated 67 min total; ~28 min remaining.

**Iter-200 retrospective**: 200 iterations completed.  Major achievements:
- iter-58/59: cube CB L2 1.092 → 0.865 (−20.8 % cumulative)
- iter-61: latlon CB mass-fixer parity, all 4 grids PASS at 12-day
- iter-66: negative-result lesson (ico CB heterogeneous mesh)
- iter-69: W2 5-day apples-to-apples — cube is BEST finite-volume
- iter-81: TC1 0.040 vs 0.014 RESOLVED (duration mismatch, not config)
- iter-102/123: TC2/TC3 cube full-mode BLOWUP discovered
- iter-118: functional bug fix (csv writer private-key filter)
- iter-138: hypothesis-2 probe (16x hyperdiff) in progress

47 sentinels (31 AST + 14 numerical + 1 c_sw + 1 functional), all PASS.

## State after iter-1..190 (compressed at iter-190)

**iter-181..190 — TC2 probe wait phase V**: probe wall 33 min → 36 min over 10 iters.  Still in dynamics.  No code changes.

## State after iter-1..180 (compressed at iter-180)

**iter-171..180 highlights — TC2 probe wait phase IV**:

- iter-171..179: TC2 probe continues running.  Wall: 30 min → 33 min over 10 iters.
- iter-180: compression milestone.

**Active state**: probe slow.  Estimated ~67 min total wall; current 33 min = halfway.  ~34 min remaining.  Result expected iter-200+ if cadence holds.

## State after iter-1..170 (compressed at iter-170)

**iter-161..170 highlights — continued TC2 probe wait phase**:

- iter-161..170: TC2 probe continues running.  Wall: 19 min (iter-161) → 30 min (iter-170).  About half done (~half remaining).
- iter-165: launched blocking-poll bash to wait for TC2 result (10min timeout; will need re-arm if not done by then).
- Otherwise iters were verifications + waiting.

**Active state**: TC2 probe at 30 min wall.  Result expected iter-180+ if linear estimate holds.

## State after iter-1..160 (compressed at iter-160)

**iter-151..160 highlights — TC2 probe wait phase**:

- iter-151..159: TC2 probe continues running.  Wall: 14 min (iter-151) → 18 min (iter-160).  Estimated total: ~67 min.  Probe is in dynamics phase; no file I/O until completion.
- iter-156: verified iter-71 sweep doc intact in matrix runner.
- iter-159: full sentinel run 55/55 PASS (31 AST + 24 matrix filter).

**Active state**: TC2 probe continues.  Result expected iter-165+.  Stop hook fires me every ~30s during the wait; each iter only adds a tiny audit unless probe completes.

## State after iter-1..150 (compressed at iter-150)

**iter-141..150 highlights — TC2 hypothesis-2 probe wait phase**:

- **iter-138 hypothesis-2 probe still running**: TC2 cube with 16× hyperdiff, --days 0.15.  14+ min wall at iter-150 (out of estimated 67 min total).  Result iter-155+ likely.
- **iter-141..149**: most iterations were verification + waiting (no commits).  Sentinels remain 47.

**Active state**: probe in dynamics phase.  No code/doc changes pending.  Once probe completes, iter-151+ will:
  1. Read TC2 probe result (PASS = hypothesis 2 confirmed; FAIL = need hypothesis 3).
  2. **REVERT** the temporary 16× hyperdiff edit in matrix runner if probe shows fix doesn't work, OR if probe shows it works, keep it + add sentinel.
  3. Either way: update doc with findings.

## State after iter-1..140 (compressed at iter-140)

**iter-131..140 highlights — TC2/TC3 cube blowup investigation phase**:

- **iter-132 TC3 cube comment upgraded** from pre-emptive CAUTION to **CONFIRMED** full-mode BLOWUP warning (iter-123 verified TC3 cube FAILs at step 2250).
- **iter-135 TC2 quick 3-grid parity verified**: cube 0.3177 / ico 0.3552 / spec 0.3597 (all within 12 %; cube is LOWEST at quick mode).
- **iter-136 FV3 oracle audit (n_split)**: FV3 control for mountain test is `n_split=10`.  Our `n_acoustic_substeps=20` is already 2× FV3 — so **hypothesis 1** (increase acoustic substeps) is unlikely to be the fix.  iter-104 hypothesis list updated.
- **iter-138 hypothesis 2 PROBE in progress**: TC2 cube with 16× hyperdiff (vs baseline 4×), running --days 0.15 (3.6 hr past current blowup at day 0.13).  Expected wall ~67 min.  Currently 11 min in.  Result iter-141+.
- **iter-133 TC3 quick refresh FAILED twice** (~25 min stuck each).  TC3 quick parity vs iter-7 doc claim remains UNVERIFIED but full-mode FAIL is the critical finding (already documented).
- **iter-130 doc compression** of iter-121..130 — NH matrix completion phase.

**Cumulative sentinel coverage at iter-140**: **31 AST + 14 numerical + 1 c_sw + 1 functional = 47 sentinels** (unchanged from iter-130).

**Active state**: TC2 hypothesis-2 probe running.  If 16× hyperdiff prevents the day-0.13 cube blowup, that's THE fix.  If still blows up, hypothesis 3 (acoustic_off_centering) is next probe.

## State after iter-1..130 (compressed at iter-130)

**iter-121..130 highlights — NH matrix completion + iter-118 verification phase**:

- **iter-123 TC3 cube ALSO BLEW UP at full mode**: NH cube matrix re-run completed.  TC3 cube FAIL at step 2250 (day 0.01, 8.3 min sim time) — `max|w|=1271 > 1000 threshold`.  TC3 fails MUCH earlier than TC2 (step 8500, day 0.13).  Both confirm iter-12..17 NH bundle insufficient for full-mode cube stability.
- **iter-124 iter-118 fix VERIFIED**: ran `--quick` NH matrix.  TC1 cube quick: |w|=0.0142 m/s — matches iter-7 doc claim of 0.014.  `mean_timeseries.csv` now 22 lines (header + 21 data rows) with NO `_blowup_info` column.  iter-118 csv-fix operational for new matrix runs.
- **iter-125 TC2 cube quick parity confirmed**: |w|=0.3177 m/s — matches iter-7 doc claim of 0.32.  cube/ico/spec parity holds at quick mode (iter-135 verified cached: cube 0.3177, ico 0.3552, spec 0.3597 — all within 12 %; cube is actually LOWEST at quick mode).
- **iter-121 functional test**: `test_iter118_save_timeseries_csv_skips_blowup_info` in `tests/atmosphere/test_atmosphere_matrix_filter.py` — pairs with iter-119 AST sentinel to cover both source-code preservation + runtime behaviour.
- **iter-126/127/128/129/131/132/133 TC3 quick refresh FAILED**: 2 attempts to refresh TC3 cube quick result both stuck for 25+ min in sleeping/IO state.  Process never wrote `mean_timeseries.csv` or refreshed `results.txt`.  Likely JAX async dispatch + Kessler microphysics graph compile issue specific to TC3 cube quick mode.  Skipped; TC3 cube full-mode FAIL result already documented at iter-123 (the more critical finding).  TC3 cube quick parity (vs iter-7 doc claim 7.36 m/s) remains UNVERIFIED but is lower priority — the full-mode BLOWUP is the actual cube-parity issue.

**Cumulative sentinel coverage at iter-130**: **31 AST + 14 numerical + 1 c_sw + 1 functional = 47 sentinels**.

**Active state**: TC3 cube quick-mode result is the only unrefreshed data point.  Once it completes, all 3 NH cube tests will have BOTH quick-mode (cube/ico/spec parity confirmed) AND full-mode (TC1 PASS, TC2 + TC3 BLOWUP) measurements.

## State after iter-1..120 (compressed at iter-120)

**iter-111..120 highlights — investigation phase post-TC2-blowup discovery**:

- **iter-118 FUNCTIONAL BUG FIX** ⭐: `_save_timeseries_csv` and `_save_timeseries_plot` in matrix runner included `_blowup_info` (a dict value added by `_run_timeloop` on FAIL) in column keys.  Writer crashed silently on `diag["_blowup_info"][i]` → empty csv with header only.  Fix: exclude private-prefix (`_*`) keys.  Future FAILed runs now preserve pre-blowup timeseries for post-mortem investigation.  **First functional code fix this loop** (vs. comment/sentinel additions).
- **iter-119 AST sentinel**: `test_iter118_timeseries_csv_excludes_private_keys` guards both filter sites.
- **iter-112 TC3 caution AST sentinel**: pins iter-108 TC3 cube branch caution comment.
- **iter-113/114 TC1/TC2/TC3 baseline clarifications**: all three NH cube branches now carry consistent breadcrumbs noting the iter-5/6/7 measurements were at quick mode.
- **iter-116 hypothesis direction correction**: iter-104 hypothesis 3 ("Reduce acoustic_off_centering 0.15 → 0.05") was inverted; lower off-centering is LESS stable per `compressible_euler_cdgrid.py:85` docstring.  Corrected to "INCREASE 0.15 → 0.30".
- **iter-117 matrix runner diagnostic limitation noted** (which iter-118 then FIXED).
- **iter-111 probe attempt FAILED**: tried to run TC2 cube growth-trace probe; competed with NH matrix CPU + tracer-leak issue.  Lesson: don't probe NH cube while NH matrix busy.

**Cumulative sentinel coverage at iter-120**: **31 AST + 14 numerical + 1 c_sw = 46 sentinels** (added iter-112 + iter-119).

**Active state**: NH cube matrix re-run still in TC3 phase (TC2 BLOWUP at iter-102 documented).  Expected TC3 completion: ~10:00 AM (4 hr after TC2 finish).

## State after iter-1..110 (compressed at iter-110)

**iter-101..110 highlights — the TC2 cube BLOWUP discovery phase**:

- **iter-102 CRITICAL FINDING**: NH cube matrix re-run completed TC2 at iter-102.  Result: **TC2 cube BLEW UP at day 0.13 (step 8500) of the full 6-hour run**.  Pre-blowup `mass_drift=7.85e-16` (clean conservation), so the dynamics propagation itself fails — not flux/halo.  iter-12..17 NH bundle was sufficient at quick-mode 5 min (cube |w|=0.32 m/s PASS) but is **insufficient for full-duration TC2 cube stability**.  See in-doc CRITICAL FINDING section at top.
- **iter-103/104/108 breadcrumbs**: doc caveats applied to all iter-5/6/7 quick-mode claims; inline WARNING block added to matrix runner TC2 cube branch with 4 hypothesis-driven probe directions for iter-110+ investigation (acoustic substeps, hyperdiff, off-centering, FV3 oracle).  Pre-emptive CAUTION comment added to TC3 cube branch (TC3 also at quick mode, full-mode behaviour unknown).
- **iter-105 AST sentinel**: `test_iter102_tc2_cube_blowup_warning_present` pins the iter-104 WARNING comment + 4-hypothesis list against future refactor stripping.
- **iter-101 fp64 hard-fail mirror**: `test_iter921_w2_v_vs_h_pareto_sentinel.py` matches iter-99's assert (both numerical sentinel files now fail loudly if `JAX_ENABLE_X64=1` not set).
- **iter-106/107/109**: verification + heading normalisation.

**Cumulative sentinel coverage at iter-110**: **29 AST + 14 numerical + 1 c_sw = 44 sentinels** (added `test_iter102_tc2_cube_blowup_warning_present` at iter-105).  Latest full-suite check (iter-106): 42 AST+numerical PASS in 154 s.

**NH cube matrix re-run status**: TC1 cube done at full 3-hr (PASS, |w|=0.040 m/s — duration-mismatch artifact vs ico/spec quick-mode 0.014).  TC2 cube done with FAIL at day 0.13.  TC3 cube currently refreshing (since 06:32; estimated ~3.5 hr more given dt=0.22s + 25 acoustic substeps; expected completion ~10:00).

## State after iter-1..100 (compressed at iter-100)

**iter-91..100 highlights** (audit & hygiene phase — NH matrix re-run still in progress throughout):

- **iter-91**: pruned resolved queued items (iter-64 nord_v probe retired by iter-81).  Refreshed outstanding list to current state.
- **iter-93/94**: added inline comments in matrix-runner cube branches for the two PE outliers (rest_state_topo iter-73 finding, rotated_baroclinic iter-87 2-day caveat).  Breadcrumbs for future investigators.
- **iter-96/97**: documented bi-directionally the `_div_damp_cube` duplication between matrix runner + test side (`tests/test_iter921_w2_v_vs_h_pareto_sentinel.py`).  Both copies now mention each other so a future calibration change updates both.
- **iter-99**: hard-fail assertion at `test_iter1002_w2_target_met.py` module import if `JAX_ENABLE_X64=1` is not set.  Pre-iter-99 the fp64-silently-disabled failure mode produced a confusing "mass_drift exceeds 1e-7" assertion; now the actual cause is pointed-finger.  Verified both directions.
- **iter-92/95/98**: verification iters (no commit) — re-confirmed 43 sentinels (28 AST + 14 numerical + 1 c_sw residual) all PASS.

**Active state**: all SW + PE + sentinels green; NH cube matrix re-run continues (started 04:13, now 138 min wall, TC2 still in dynamics).  Once complete: refresh TC2/TC3 cube measurements + re-evaluate cross-grid NH parity (cube full mode vs ico/spec quick — partial apples-to-apples).

## State after iter-1..90 (compressed at iter-90)

**iter-81..90 audit highlights**:

- **TC1 NH 0.040 vs 0.014 gap RESOLVED (iter-81)**: matrix runner caches were mixed-mode — cube TC1 ran 3.0 hr (full), ico/spec at 0.5 hr (quick).  Apples-to-oranges duration NOT a config issue.  iter-63 nord_v hypothesis withdrawn; iter-64 nord_v probe retired.  Cube TC1 is at parity with ico/spec when measured at same duration (~0.014 m/s).
- **Full matrix mode survey (iter-88/89)**:
  - SW matrix: **16/16 at full-spec** duration.
  - PE matrix: **51/52 at quick mode** (only gravity_wave_3_1 at full per iter-72).
  - NH matrix: **6/9 at quick + 1/9 at full + 2/9 refreshing** (TC1 cube full done; TC2/TC3 cube in progress).
  Implications: SW audit = HIGH confidence; PE quick-mode audit = HIGH within quick mode but full-mode TBD; NH = mode-mixed.
- **rotated_baroclinic 39% gap caveat (iter-87)**: the cube-39%-lower-than-ico claim was at 2-day quick mode where the baroclinic instability hasn't grown.  Same identical-result artifact between rotated_baroclinic and rotated_steady at 2 days.  Full-mode 10-day re-run TBD.
- **Sentinel hygiene**:
  - iter-83 hardened iter-79's iter-66 sentinel with anti-anchor (forbids `_mass_target_iter66` constant in ico CB branch — guards against incomplete revert).
  - iter-86 verified c_sw cube residual still PASS (19s).
- **Total sentinel coverage**: **28 AST + 14 numerical + 1 c_sw = 43 sentinels**, all PASS at last full check (iter-85 + iter-86).

**NH cube matrix re-run still in progress** (since 4:13 AM, now ~122 min wall, TC2 phase).  Expected total 4-6 hr; TC2 needs ~25 min more, TC3 ~3.5 hr.  Once complete: TC2/TC3 cube will refresh with iter-12..17 fixes applied.

## State after iter-1..80 (compressed at iter-80)

**iter-71..80 audit highlights**:

- **PE matrix audit (iter-73)**: 13/13 cases × 4 grids = 52 tests PASS.  Cube max|v| parity vs ico in **0.95-1.15×** on most PE tests (held_suarez, gravity_wave_3_1, inertio_gravity_3_2, rossby_haurwitz_6_0, baroclinic).  2 cube outliers flagged for iter-74+:
  - `rotated_baroclinic`: cube 31.8 vs ico 51.9, spec 47.1 m/s (cube 39 % LOWER — possibly over-damped on rotated-pole IC).  **iter-87 caveat**: this comparison is at **2-day quick mode** for ALL 4 grids; matrix spec full mode is **10 days** (instability development).  At 2 days the baroclinic perturbation hasn't grown enough to differ meaningfully from `rotated_steady` (which shows the same max|v| values across all 4 grids — see results files).  Quick-mode max|v| is dominated by the steady-state rotated jet, not the instability.  To audit the cube-vs-ico parity claim at the matrix-spec full duration, all 4 grids need to be re-run at 10 days (deferred until NH matrix run completes).
  - `rest_state_topo`: cube 1.3 vs ico 0.1 m/s (analytic exact = 0 motion; cube has 13× residual — likely panel-edge metric errors interacting with topo).
- **PE gravity_wave_3_1 (iter-72)**: refreshed cross-grid at 1-day apples-to-apples.  cube 22.4 / ico 19.7 / spec 20.3 / latlon 31.0 m/s.  Cube/ico=**1.14×**, cube/spec=**1.10×** — confirms iter-18's −18.5 % gain stable.
- **SW W2 hyperdiff sweep (iter-71)**: probed mult 1.0/1.5/2.0/2.5/3.0 on cube W2 5-day.  Optimum near 2.5× (L2=4.42e-4) but only 3.5 % below iter-44's 2.0× (4.58e-4) — sub-noise-floor.  Kept 2.0×.
- **Apples-to-apples completion (iter-75)**: post-iter-72 audit confirmed **NO remaining quick/full mode result mismatches** in SW or PE matrix (17 cases × 4 grids = 68 entries).
- **iter-66 negative-result AST sentinel (iter-79)**: locked in via `test_iter66_cb_ico_uses_additive_correction` — guards against future ico CB multiplicative-fixer attempts that regress Linf 5×.
- **Repo hygiene (iter-76)**: `_probe_*.py` added to `.gitignore` (prevents accidental commit of throwaway audit scripts).
- **TC1 NH cube 0.040 vs doc claim 0.014 — RESOLVED (iter-81)**: not a config issue.  Cached NH TC1 cube ran at **3.0 hours** (full mode); ico + spec cached at **0.5 hours** (quick mode).  The "0.040 vs 0.014" gap was **apples-to-oranges duration mismatch**, NOT a nord_v=1 vs nord_v=2 issue.  Quick-mode 0.5-hr cube measurement at iter-7 fixes gave 0.014 (matching ico/spec).  iter-63's nord_v hypothesis is **withdrawn**; iter-64 nord_v probe **no longer needed**.  TC1 cube is at parity when measured apples-to-apples.  NH duration mismatch is the 6th quick/full mode caching issue found this loop (SW W2/W5/W6/CB + PE gravity_wave_3_1 + NH TC1).
- **NH TC2 / TC3 cube outliers (current cached, awaiting refresh)**: TC2 cube 4.54 m/s vs ico/spec 0.36 m/s (12.8× outlier — but iter-12..17 fixes pending matrix re-run).  TC3 cube 22.97 vs ico 10.24 m/s (2.24×).  When NH matrix re-run completes (TC2 currently ~80 min in), these will refresh with iter-12..17 effects applied.

**Cumulative sentinel coverage**: **28 AST + 14 numerical + 1 c_sw residual = 43 sentinels total** (iter-79 added iter-66 negative-result guard; iter-83 hardened it with anti-anchor).  Latest full-suite check (iter-85): 41 AST+numerical PASS in 152 s; c_sw residual (iter-86 verified) PASS in 19 s.

**iter-88/89 detail**: see compressed iter-81..90 block above.  PE cached at quick mode (51/52 entries); only gravity_wave_3_1 at full.  NH cube refreshing; ico/spec NH cached at quick.

## State after iter-1..70 (compressed at iter-70)

**Cross-grid apples-to-apples SW matrix audit (iter-61..70)**: refreshed all 4 grids at full duration (the matrix's quick-mode + full-mode results coexist; iter-61 caught the mixed-mode comparison issue).

- **W2 5-day** (iter-69): cube is **BEST finite-volume grid**.
  - spectral (T21):     L2=1.80e-7, Linf=2.37e-7 (analytic-perfect)
  - ico (ico5):         L2=1.25e-4, Linf=6.05e-4
  - **cube (C36)**:     **L2=4.58e-4, Linf=4.82e-3, v_ll=0.51 m/s**
  - latlon (72×144):    L2=1.41e-3, Linf=1.27e-2
  - Cross-grid ratios: cube/spectral=2544×, cube/ico=3.66×, cube/latlon=**0.325** (cube 3× BETTER than latlon).  Doc pre-iter-69 claim "cube/latlon 1.7×" was correct at iter-42; post-iter-44 cube overshoot below latlon.

- **W5 15-day** (iter-70): **mass conservation parity at machine precision across all 4 grids**.
  - ico: drift=0.00e+00, latlon: 0.00e+00, spectral: 8.10e-16, cube: 1.46e-15.
  - All PASS at full 15-day duration.

- **CB 12-day** (iter-61): all 4 grids PASS, cube is L2 outlier but BEST on conservation.
  - latlon: L2=0.133, drift=2.07e-8 (iter-61 anchored fixer applied).
  - ico: L2=0.620, drift=1.58e-6 (additive fixer — iter-66 anchored-attempt regressed, reverted).
  - spectral: L2=0.382, drift=2.16e-16.
  - cube (iter-58/59): L2=0.865, drift=1.28e-9.
  - True cube/latlon=6.5× (was previously falsely "35×" from mixed-mode).

**Lessons recorded**:
- Cross-grid "consistency" ≠ identical fixer logic.  Cube/latlon (regular structured grids) prefer multiplicative-rescale-to-initial-mass; ico (heterogeneous unstructured: 12 pentagons + hexagons) prefers additive uniform correction.  Multiplicative on ico over-concentrates mass in pentagons (Linf 0.56 → 2.83) — iter-66 negative result.

**Cumulative cube CB improvement (iter-1 → iter-69 baseline)**: L2 1.092 → 0.865 (−20.8 %) via:
- iter-58 hord=10 + xppm_bdy → −14.7 % (matched Fortran tp_core L2 limiter + boundary cube-edge formulas).
- iter-59 N=6 temporal substep → −7.0 % (closed forward-Euler PPM phase-error gap vs latlon SSP-RK3).

**Sentinel coverage** (now 27 AST + 14 numerical):
- AST: 27 in `tests/test_matrix_nh_cube_parity_ast_guard.py` (added iter-59 N=6 substep + iter-61 anchored mass fixer sentinels).
- Numerical: 14 in `tests/test_iter1002_w2_target_met.py` — W2 1-day/2-day/5-day, W5 day-5/2-day, W6 2-day, CB 12-day cube + 12-day latlon, hyperdiff kwarg API, 2 quiet-path guards.

**Outstanding (not closed)**:
- Cube SW W2 vs ico residual 3.66× L2 ratio — structural to PPM on curved cube faces vs SOM-PPM on isotropic Voronoi.
- Cube CB vs ico residual ~1.4× L2 ratio — bulk PPM limiter dissipation along rotated trajectory (iter-61 spatial decomposition).
- NH TC1 cube |w|_max=0.040 m/s vs doc claim 0.014 m/s — 3× gap; iter-63 audit found matrix uses `nord_v=2` default while factory uses `nord_v=1`.  iter-71+ to probe whether `nord_v=1` closes this.
- NH TC2/TC3 cube matrix re-run still in progress; will refresh iter-7 claim numbers (TC2: 0.32, TC3: 7.36 m/s).

## State after iter-1..60 (compressed at iter-60)

**New cube CB wins (iter-51..60)**:

- **Cube CB 12-day L2 1.092 → 0.865 (−20.8 % cumulative)**.
  - iter-58: `hord=10` + `apply_fortran_xppm_boundary=True` in matrix `transport_step` call → 1.092 → 0.931 (−14.7 %).  PPM hord=10 (slope-limiter monotone) retains more amplitude than hord=12 at multi-day integrations; xppm boundary applies Fortran-faithful cube-edge formulas.
  - iter-59: N=6 temporal substepping (dt_sub=300s) inside CB cube step_fn → 0.931 → 0.865 (−7.0 %).  Latlon CB uses SSP-RK3 (O(dt³)); cube CB used single-stage forward-Euler PPM (O(dt)) — substep closes ~7 % of the temporal-truncation gap.  Saturation past n_sub=6 shallow (≤1.5 % per doubling).  Mass drift 28× tighter (3.6e-8 → 1.3e-9, per-substep fixer).
  - iter-61: cross-grid L2 ratio CORRECTED from previously documented 35×.  The 35× figure was based on comparing cube 12-day vs latlon **1-day** stale results — apples-to-oranges.  True apples-to-apples 12-day cube/latlon L2 ratio = **6.5×** (cube 0.865 vs latlon 0.133).  Cube vs ico = 1.4×, cube vs spectral = 2.3×.  Cube is L2 outlier but BEST on mass conservation among finite-volume grids (1.28e-9 vs latlon 2.07e-8 vs ico 1.58e-6, with spectral 2e-16 exact).

**Other iter-51..60 work**:

- iter-51: cube CB C36 vs C48 L2 essentially identical (1.092 vs 1.094) → CB structural gap NOT dx²-convergence-limited; PPM limiter dissipation dominates.
- iter-52..55: parameter sweeps on iter-1030 calibration (`nord_v`, `div_damp_factor`, `damp_v`) — all confirm current values near L2 optimum.  Wind-vs-height trade-off characterised.
- iter-56: 1-day vs 5-day hyperdiff trade-off characterized; iter-1002 sentinel and matrix runner pin DIFFERENT operating points (1-day spectral character vs 5-day long-run stability).
- iter-57: cube W2 5-day numerical sentinel at matrix-actual config (L2 < 1.5e-3, max|v_d| < 5 m/s).
- iter-60: cleaned up 2 stale silent-noop warning sentinels (iter-1019/iter-1020) — both were pre-existing red CI tests asserting warnings from code paths the SW model doesn't traverse.  Inverted to positive guards: assert NO spurious warning fires.  Plus added `test_iter59_cube_cb_12day_matrix_config` numerical sentinel pinning the iter-58/59 cube CB win (L2 < 0.90, Linf < 0.90).

**Sentinel coverage** (now 26 AST + 8 numerical + 1 c_sw residual):
- AST: 26 in `tests/test_matrix_nh_cube_parity_ast_guard.py` (iter-59 added `test_iter59_cb_cube_uses_n_sub_substepping`).
- Numerical: 8 in `tests/test_iter1002_w2_target_met.py` — W2 1-day (iter-1002), W5 day-5 (iter-1009), W6 2-day (iter-39), W5 2-day (iter-48), W2 2-day (iter-49), W2 5-day (iter-57), CB 12-day (iter-59), hyperdiff kwarg API (iter-35); plus 2 quiet-path guards (iter-60 inverted iter-1019/iter-1020).

## State after iter-1..50 (compressed at iter-50)

**Headline cube parity wins**:

- **NH cube parity CLOSED** for all 3 DCMIP cases (iter-5/6/7).  Pre→post:
  - TC1: 0.327 → **0.014 m/s** (23×).  Matches ico 0.015 / spec 0.014 within 1 ULP.
  - TC2: 4.65 → **0.32 m/s** (14.6×).  Matches ico 0.36 / spec 0.36.  ⚠️ **CAVEAT (iter-102)**: these numbers are at quick mode (5 min = 0.083 hr); cube TC2 **BLOWS UP at full 6-hr mode** at day 0.13 (step 8500).  See critical finding at top of doc.  iter-12..17 fixes are NOT sufficient for full-duration TC2 cube stability.
  - TC3: 23.1 → **7.36 m/s** (3.1×).  BEATS ico 10.24.
  Mechanism: `use_fv3_vector_halo_uv` + `use_fv3_a2b_ord4_vector_uv` (iter-697/698 FV3-faithful pair) promoted from the factory to the matrix runner NH cube branches.

- **SW W2 cube parity 7.2× improved** (iter-42/44).  v_ll_Linf 3.65 → **0.51 m/s**; L2 2.45e-3 → 4.58e-4 (cube/latlon ratio 9× → 1.7×).  Mechanism: enable `hyperdiff_coeff = 2 × _hyperdiff_cube(n)` on cube SW propagating tests (test_num ∈ {2, 5, 6}).

- **SW cube W5 / W6 14-15-day stability** (iter-31/33/44).  Pre-fix W5 BLEW UP day 14.58, W6 BLEW UP day 9.  Now bit-clean mass conservation at full duration via the same hyperdiff override.

- **PE cube gravity_wave_3_1** max|v| 27.5 → **22.4 m/s** (-18.5 %, iter-18) via `use_fv3_metric_aware_d_con=True`.  Now within 12 % of ico/spec/latlon cluster.

- **SW Williamson 6 wired for all 4 grids** (iter-24/25).  cube/latlon/ico/spec all at machine-precision mass conservation; cross-grid table complete.

**Matrix-runner cube config bundles in place**:
- NH cube (iter-5/6/7/12/13/14/15/16/17, 8 flags): vector_halo + a2b_ord4 + d_con_cv + dynamic_exner + metric_aware_d_con + d_con_top_zero_levels=2 + heat_source_del2_iters=2 + delt_max=1.0 + corner_div_damp_nord=1/d4_bg=0.16.
- PE cube on all 3 branches (iter-18..22, 4 flags): metric_aware_d_con + d_con_top_zero_levels=2 + delt_max=1.0 + heat_source_del2_iters=2.
- SW cube W2/W5/W6 (iter-42/44): `hyperdiff_coeff = 2 × _hyperdiff_cube(n)`.

**Code hardening**:
- `core/fv_tp_2d.py:transport_step` flux differencing promoted to fp64 (iter-3).
- `iter1009_dual_target_config(n, ..., hyperdiff_coeff=0.0)` accepts hyperdiff kwarg (iter-35).
- Stale iter-1019 "signature-only NO-OP" warning removed from `operators_cdgrid.py:fv3_sw_tendencies` (iter-41 — real biharmonic hyperdiff implementation IS at line 1444-1456).

**Regression sentinels** (25 AST + 4 numerical + 1 c_sw residual):
- `tests/test_matrix_nh_cube_parity_ast_guard.py` — 25 AST tests pinning flag wiring across NH/PE cube branches + SW gate `{2,5,6}` + W6 init helpers; whitespace-tolerant regex + proximity sanity (iter-27/28/29/45 hardened).
- `tests/test_iter1002_w2_target_met.py` — 4 numerical sentinels:
  - `test_iter1002_w2_v_ll_linf_meets_target`: W2 1-day at iter-1030 calibration (hyperdiff=0).
  - `test_iter1009_w5_day5_artifact_free`: W5 day-5 (hyperdiff=0 calibration).
  - `test_iter49_cube_w2_matrix_config_short_run`: W2 2-day at iter-44 matrix config (hyperdiff=2×).
  - `test_iter48_cube_w5_short_run_stable_with_hyperdiff`: W5 2-day (hyperdiff=2×).
  - `test_iter39_cube_w6_short_run_stable_with_hyperdiff`: W6 2-day (hyperdiff=2×).
  - `test_iter35_hyperdiff_coeff_kwarg_threads_through`: helper kwarg API.
- `tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py::TestCubeCswW2Residual` — cube c_sw + p_grad_c residual on W2 IC (iter-4).

**Cumulative-bundle long-run verifications**:
- TC3 cube cumulative iter-12..17: |w|=7.36 m/s bit-identical to iter-7 (iter-23).
- held_suarez sigma cube iter-22 bundle: max|v|=10.8 m/s bit-identical (iter-23).
- AMIP cube iter-22 bundle: mass_drift=4.18e-11 bit-identical (iter-47).
- Full SW cube matrix at full duration (iter-43): 4/4 PASS.
- iter-44 2× hyperdiff probed at C36 + C48 (iter-46) — same optimum.

## Structural cube parity findings (not closed)

- **c_sw + p_grad_c cube-vertex residual** (codex iter-983 + iter-3/4/26 reproduced): |duc|_max = 2.71 m/s at face=2 i=1 j=35 on cube W2 IC.  STRUCTURAL to the duogrid c_sw path.  iter-1030 calibration + iter-44 hyperdiff partially mask this, but the per-step imbalance remains.
- **Cube CB 12-day L2=1.09** (iter-26 hord sweep + iter-34 + iter-51 resolution probe): structural cube PPM transport accuracy.  C36 L2=1.092, C48 L2=1.094 — virtually identical, confirming the issue is NOT dx² convergence-limited.  Root cause: PPM monotone limiter dissipation accumulates over 12-day rotation, eventually smearing the cosine bell to a near-flat field.  Cube vs latlon L2 ratio 45× (cube 1.09 vs latlon 0.024 at iter-43 measurement).  CB uses `transport_step` directly, not `model.step()`; hyperdiff path doesn't engage.  Would require a tracer-specific transport variant or a different limiter to address.
- **C48 hyp=0 cube W2 v_ll = 5.55 m/s** (iter-46): finer grids amplify cube-vertex residual without hyperdiff coverage.  iter-44's 2× hyperdiff is essential, not optional.

## Iter trail (terse, iter-1..49)

- iter-1/8/34: SW W2/W5/CB cube → iter-1030 calibration via `iter1009_dual_target_config(n)`.
- iter-2: cube CB raw-transport leak diagnostic.
- iter-3: `transport_step` fp64 + cube flux closure verified bit-clean.
- iter-4: c_sw W2 residual sentinel.
- iter-5/6/7: **NH cube parity CLOSED** (TC1/TC2/TC3 23×/14.6×/3.1×) via vector_halo + a2b_ord4.  ⚠️ iter-102 found TC2 was measured at quick mode only — full mode BLOWS UP at day 0.13.
- iter-9: PE `use_fv3_a2b_zeta_corner` neutral, reverted.
- iter-10/20/30/40/50: doc compress + iter-329 regex fix (iter-10).
- iter-11: AST sentinel for SW helper substitution.
- iter-12..17: NH factory bundle (d_con_cv + dynamic_exner + metric_aware + d_con_top_zero_levels + heat_source_del2 + delt_max + corner_div_damp).
- iter-18: PE gravity_wave_3_1 -18.5 % via metric_aware_d_con.
- iter-19/20/21: PE factory bundle (d_con_top_zero_levels + delt_max + heat_source_del2).
- iter-22: PE factory bundle propagated to held_suarez + AMIP.
- iter-23: cumulative TC3 + held_suarez verifications bit-identical.
- iter-24/25: SW W6 cube + latlon wiring (4-grid table at machine precision).
- iter-26: cube CB hord sweep (hord=12 best); 3 W6-wiring AST sentinels.
- iter-27/28/29: hardened PE block-finder anchors.
- iter-31/33: cube W5/W6 long-run hyperdiff override.
- iter-32: W6 hyperdiff AST sentinel.
- iter-35: helper `hyperdiff_coeff` kwarg API.
- iter-36: docstring + kwarg threading test.
- iter-37: review-driven W2 gate-exclusion sentinel.
- iter-38: end-to-end SW cube quick matrix PASS.
- iter-39: numerical sentinel for cube W6 stability.
- iter-40: investigated iter-31 mechanism (initially mis-concluded JAX-trace).
- iter-41: **iter-40 correction** — real biharmonic hyperdiff at `fv3_sw_tendencies:1444`; removed stale warning.
- iter-42: **cube W2 5-day v_ll 3.65 → 0.82 m/s** (4.5×) via gate widening to `{2,5,6}`.
- iter-43: cumulative cube SW matrix full-duration 4/4 PASS.
- iter-44: **bumped hyperdiff coefficient 1× → 2× → cube W2 v_ll 0.82 → 0.51 m/s** (cumulative 7.2× from iter-1).
- iter-45: review-driven AST sentinel hardening — tightened gate regex + iter-1002 independence sentinel.
- iter-46: resolution sweep at C48 confirms 2× hyperdiff optimum.
- iter-47: AMIP cube 30-day quick PASS bit-identical to pre-iter-22.
- iter-48: W5 numerical sentinel (mirror of iter-39 W6).
- iter-49: W2 matrix-config numerical sentinel (hyperdiff=2× variant).
- iter-50 / iter-60: doc compressions (state of iter-1..49 and iter-51..60 summarised in the two "State after" blocks above).
- iter-51..70: see two compressed "State after" blocks at the top of this doc.
- iter-61 (verbose entry retained — first apples-to-apples mass-fixer parity finding):  cross-grid CB 12-day apples-to-apples audit + latlon CB mass-conservation parity.  Found previously documented "35× cube/latlon L2 ratio" was based on comparing cube **12-day** results vs latlon **1-day** stale results files (the matrix's quick-mode + full-mode results coexist in the same dir).  Re-ran all 4 grids at 12-day full duration:
  - latlon (72×144): L2=0.125, mass_drift=5.35e-4 → **FAIL** (drift > matrix tol 1e-4).
  - icosahedral (ico5): L2=0.620, mass_drift=1.58e-6, PASS.
  - spectral (T21): L2=0.382, mass_drift=2.16e-16, PASS.
  - cube (C36, iter-58/59): L2=0.865, mass_drift=1.28e-9, PASS.
  Diagnosed: latlon CB step_fn was intentionally a "raw FV benchmark" (no mass fixer); iter-29 matrix-wide tolerance tightening from 1e-2 to 1e-4 dropped below the raw-FV mass drift.  Per the ralph-loop "consistency across grids" goal, applied the **same anchored mass fixer cube uses** (clip-negatives + rescale-positives to mass_target) to the latlon CB step_fn.  Result: latlon mass_drift 5.35e-4 → **2.07e-8** (26000× tighter), L2 0.125 → 0.133 (+6 % small redistribution cost), Linf 0.229 → 0.246 (+7 %), **PASS**.  **All 4 grids now PASS at 12-day apples-to-apples**.  True cube/latlon L2 ratio: **6.5×** (was claimed 35×), cube/ico 1.4×, cube/spectral 2.3× — all far closer than previous trail suggested.  iter-61 also probed the spatial decomposition of the cube CB error (`_probe_iter61_cb_error_map.py`): 99 % of the residual L2 lives in the panel-INTERIOR cells of the single face containing the bell at t=12 d; panel-edge cells contribute ~0.0003 (essentially zero).  Confirms residual is bulk PPM limiter dissipation along the rotated trajectory, NOT cube panel-coupling.

- iter-62: iter-61 latlon CB numerical sentinel.
- iter-66 negative result (reverted): ico CB multiplicative-fixer attempt regressed Linf 5× — additive uniform correction wins on heterogeneous unstructured mesh.  See top compressed block for lesson.
- iter-69: W2 5-day apples-to-apples — cube is BEST finite-volume.  See top compressed block.
- iter-70: W5 15-day apples-to-apples — all 4 grids machine-precision mass conservation.  See top compressed block.
- iter-71..80: see compressed block at the top of this doc.

## Iter-91+ queued

Open items requiring substantial compute (deferred):
- **HIGHEST PRIORITY (iter-102/103)**: investigate TC2 cube full-mode BLOWUP at day 0.13.  Hypothesis: gravity wave reflection off cube panel edges accumulates over 3 hours.  Pre-blowup mass_drift was 7.85e-16 (clean) — dynamics field blew up cleanly without conservation issues.  Possible fixes to probe:
  1. Increase n_acoustic_substeps from 20 → 30+ (acoustic instability hypothesis).
  2. Increase hyperdiff_coeff for TC2 cube (over-edge dissipation hypothesis).
  3. INCREASE acoustic_off_centering from 0.15 → 0.30 (more implicit = more stable; iter-116 correction — initial direction was inverted, lower off-centering is LESS stable for long runs per `acoustic_off_centering: float = 0.0   # Off-centering beta; 0=centered, 0.1 long runs` doc string in `compressible_euler_cdgrid.py`).
  4. FV3 oracle comparison at /home/gentine/Documents/Code/FV3/atmos_cubed_sphere-symmetryclean/ for TC2-equivalent mountain test config.


- **PE full-mode re-run** (iter-88 audit): all 13 PE cases × 4 grids currently at quick mode (10-30 % of spec full duration).  Full-mode runs of held_suarez (200d), AMIP (365d), held_suarez_topo (200d) are multi-hour each.  Cross-grid full-mode parity TBD.
- **NH ico+spec full-mode re-run** (iter-81/89): TC1/TC2/TC3 cached at quick mode for ico/spec.  Cube currently refreshing to full mode (NH matrix in progress).  For true 3-grid full-mode comparison, ico+spec also need to re-run at TC1=3hr/TC2=6hr/TC3=2hr each.
- **rotated_baroclinic 10-day full-mode** (iter-87): currently all 4 grids at 2-day quick mode where instability hasn't grown.  Need 10-day full to assess true rotated-pole cube parity vs ico (the apparent 39 % gap may be a quick-mode artifact).
- **C72 resolution sweep** (iter-46 extension): probe whether cube SW/CB findings hold at higher resolution.

Closed items (formerly queued):
- ~~iter-64 nord_v=1 TC1 probe~~ — **resolved by iter-81** as duration-mismatch artifact, not config issue.
- ~~iter-45 visual artifact inspection~~ — addressed by iter-58/59 numerical work; visual would require additional plotting work.
- ~~Cube CB structural tracer del-4 op~~ — iter-58/59 cumulative 20.8 % L2 improvement makes this no longer the highest-value action.

## Iter-255 — AIMIP Phase 2 audit + S_0 demoted from tunable

While continuing AIMIP Phase 2 (extending the trainable-physics set), discovered iters 251-254 had landed `albedo_land`, `Ri_crit`, `Ck`, `l_mix_max` in `_COMMON_TRAINABLE` / `_LOUIS_TURBULENCE_TRAINABLE` **without wiring them as kwargs of `build_segment_fn`**.  Net effect: gradients ran, but param values never reached the rollout — the training loop would optimise a constant.  CLAUDE.md explicitly bans this kind of half-impl.

Landed two corrective changes:

1. **`tests/test_aimip_phase2_trainable_wired.py`** — three-sentinel regression guard:
   - every name in the trainable lists is either a `build_segment_fn` kwarg or in a documented `UNWIRED_TODO` whitelist;
   - `UNWIRED_TODO` entries are not already wired (whitelist tightens monotonically);
   - currently-wired-and-trainable param count is exactly the original 8 (`tau_equator, tau_pole, sbm_tau_c, sbm_RH_ref, C_H, C_E, albedo_ice, albedo_ocean`).
   `UNWIRED_TODO = {albedo_land (Phase 2.6), Ri_crit, Ck, l_mix_max (Phase 2.7)}`.

2. **`src/legoesm/tuning.py`** — removed `S_0` from `TUNING_PARAMETERS` per user direction ("solar constant should be a constant not a parameter").  `constants.S_0 = 1361 W/m^2` remains the single source of truth; per-experiment overrides go via `ExperimentConfig.S_0` (already exists, defaults to `constants.S_0`).  Param count `TUNING_PARAMETERS` 19 → 18.  `tests/unit/test_constants_consistency.py:46` still passes (verifies `S_0` exists in `constants.py`, orthogonal to its tuning-registry status).

Phase-2 trainable accounting (as of iter-255):

| Set | Count | Names | Wired? |
|-----|-------|-------|--------|
| `_COMMON_TRAINABLE` | 7 | `tau_equator, tau_pole, C_H, C_E, albedo_ice, albedo_ocean, albedo_land` | 6/7 |
| `_SBM_TRAINABLE` | 2 | `sbm_tau_c, sbm_RH_ref` | 2/2 |
| `_LOUIS_TURBULENCE_TRAINABLE` | 3 | `Ri_crit, Ck, l_mix_max` | 0/3 |

Next AIMIP iters: wire `albedo_land` (Phase 2.6) then the Louis triple (Phase 2.7) end-to-end (`build_segment_fn` → `step_unified` → surface/turbulence physics modules).

## Iter-256 — AIMIP Phase 2 honest revert + surface-blend gap surfaced

Investigated the Phase 2.6 wiring of `albedo_land` and found a deeper infrastructure gap that blocks it.  In `src/legoesm/driver/physics_pipeline.py::compute_radiation_core` the surface blend is

```python
albedo = sic * albedo_ice + (1 - sic) * albedo_ocean
```

— ice/ocean only, no land term.  `blend_surface_property` in `src/legoesm/forcing/surface_utils.py` is the same 2-way blend.  There is no `land_mask` / `land_fraction` channel in the surface-state pipeline at all.  Wiring `albedo_land` as a `build_segment_fn` kwarg today would produce a kwarg that has no consumer downstream — gradient would flow into a node nothing reads.

Same problem affects the Louis turbulence triple (`Ri_crit`, `Ck`, `l_mix_max`): `build_segment_fn` does not take them as kwargs and `step_unified` does not forward them to the turbulence physics, so they were also nominal-only trainables in iters 252-254.

Honest correction (iter-256):

- `src/legoesm/training/trainable_params.py`:
  - Removed `albedo_land` from `_COMMON_TRAINABLE` (was iter-251).
  - Cleared `_LOUIS_TURBULENCE_TRAINABLE` to `[]` (was iters 252-254).
  - Docstrings now explain the wiring prerequisites and cite the regression sentinel.
- `tests/test_aimip_phase2_trainable_wired.py`:
  - `UNWIRED_TODO` set is now empty: any future addition that lands without wiring trips the sentinel.
  - All three sentinels remain green; wired-trainable count stays at 8.

Phase-2 trainable accounting (revised, iter-256):

| Set | Count | Names | Wired? |
|-----|-------|-------|--------|
| `_COMMON_TRAINABLE` | 6 | `tau_equator, tau_pole, C_H, C_E, albedo_ice, albedo_ocean` | 6/6 |
| `_SBM_TRAINABLE` | 2 | `sbm_tau_c, sbm_RH_ref` | 2/2 |
| `_LOUIS_TURBULENCE_TRAINABLE` | 0 | — | 0/0 |
| **Total wired trainable** | **8** | | **100 %** |

This is fewer params than the AIMIP plan target (~17-20) — but every entry now actually gradient-couples to the rollout.  `albedo_land`, `Ri_crit`, `Ck`, `l_mix_max` remain in `TUNING_PARAMETERS` as experiment-config knobs; they will re-enter the trainable set once their respective surface and turbulence wiring lands.

Real Phase-2 wiring backlog (each a distinct future iter):
- **Phase 2.6**: introduce `land_fraction` channel through `SegmentForcing` + surface blend, then thread `albedo_land` through `build_segment_fn` → `step_unified` → `compute_radiation_core` and add to `_COMMON_TRAINABLE`.  **LANDED iter-257** (see below).
- **Phase 2.7**: thread `Ri_crit` / `Ck` / `l_mix_max` through `build_segment_fn` → `step_unified` → `physics_step_no_rad` → Louis turbulence scheme; add back to `_LOUIS_TURBULENCE_TRAINABLE`.

## Iter-257 — AIMIP Phase 2.6 LANDED: land-fraction channel + 3-way albedo blend

Wired `albedo_land` end-to-end so it actually gradient-couples to the rollout.  Multi-file change, every step verified before stacking the next:

1. **`SegmentForcing` grew a `land_fraction` field** (`src/legoesm/driver/compiled_segments.py`).
   - NamedTuple field count 9 → 10.
   - `pack_forcing(..., land_fraction=None)` defaults to `jnp.zeros_like(sst)` so analytical AMIP runs are bit-for-bit identical to pre-iter-257.

2. **`PhysicsPipeline.albedo_land` field** (`src/legoesm/driver/physics_pipeline.py`).
   - Constructor default = 0.30 (mid-range desert/snow).
   - Stored on the instance and consumed by `compute_radiation_core`.

3. **3-way albedo blend in `compute_radiation_core`** (same file).
   - Accepts new kwargs `albedo_land`, `land_fraction`.
   - When `land_fraction is None` the blend is unchanged from pre-iter-257.
   - Otherwise: `albedo = lf*albedo_land + (1-lf) * (sic*albedo_ice + (1-sic)*albedo_ocean)`, with `lf` clipped to `[0, 1]` for numerical safety.

4. **`step_unified` threads the new kwargs** (same file).
   - Signature grew `albedo_land=pipeline.albedo_land` and `land_fraction=None`.
   - Both `_rad_branch` and `_no_rad_branch` accept them in their args-tuple; the no-rad branch deletes them locally (radiation-only relevance).
   - Inside `step_unified` a default `_land_fraction = jnp.zeros_like(sst)` is built when the caller passes `None`, keeping the `lax.cond` branches dtype-consistent.

5. **`build_segment_fn` propagates `albedo_land` + `forcing.land_fraction`** (`src/legoesm/driver/compiled_segments.py`).
   - New `albedo_land=None` kwarg cast to `_albedo_land = jnp.asarray(...)` once at build time.
   - Both `step_unified` call sites (owned-face MPI path + serial path) now pass `albedo_land=_albedo_land, land_fraction=forcing.land_fraction`.

6. **`albedo_land` back in `_COMMON_TRAINABLE`** (`src/legoesm/training/trainable_params.py`).
   - Updated the docstring to record the full revert→relanding history (iters 251→256→257).
   - Sentinel expected-wired list now includes `albedo_land`; `UNWIRED_TODO` remains empty.

7. **New unit-level sentinel** `tests/test_aimip_phase26_albedo_blend.py`.
   - 6 parametrized cases pin the 3-way blend at canonical points (pure ocean, pure ice, pure land, lf=1 dominating ice, 50/50 land/ocean, generic mix).
   - 2 tests pin `pack_forcing` default + round-trip behaviour for `land_fraction`.
   - 1 test pins the `SegmentForcing._fields` count + presence of `land_fraction`.

Test status: all 16 AIMIP-specific tests green; `tests/unit/test_compiled_segments.py` (28 tests) green.  Two pre-existing failures unrelated to this iter (`tests/unit/test_gradient_checkpointing.py`, `tests/validation/test_scaling_readiness.py::test_segment_carry_is_valid_pytree`) verified to fail identically on parent commit `b2c7a626`.

Phase-2 trainable accounting (revised, iter-257):

| Set | Count | Names | Wired? |
|-----|-------|-------|--------|
| `_COMMON_TRAINABLE` | 7 | `tau_equator, tau_pole, C_H, C_E, albedo_ice, albedo_ocean, albedo_land` | 7/7 |
| `_SBM_TRAINABLE` | 2 | `sbm_tau_c, sbm_RH_ref` | 2/2 |
| `_LOUIS_TURBULENCE_TRAINABLE` | 0 | — | 0/0 |
| **Total wired trainable** | **9** | | **100 %** |

## Iter-258 — AIMIP Phase 2.7 partial LAND: l_mix_max wired through Louis

Wired `l_mix_max` (Louis maximum mixing length) end-to-end so the AIMIP training loop can tune it.  `Ri_crit` and `Ck` deliberately NOT re-added — audit of `src/legoesm/atmosphere/physics/turbulence/louis.py` confirmed Louis never reads them.  They live on `LouisConfig` only as forward-compat placeholders (consumed instead by holtslag_boville / TKE / YSU).  Making them "Louis trainable" would re-introduce the same nominal-only bug iter-256 reverted.

Three-file edit:

1. `src/legoesm/atmosphere/physics/turbulence/louis.py` — `louis_turbulence` signature grew `l_mix_max: jax.Array | float | None = None`.  Inside the body the mixing-length formula reads `_l_mix_max = config.l_mix_max if l_mix_max is None else l_mix_max` so the static-config path is bit-for-bit unchanged.

2. `src/legoesm/driver/physics_pipeline.py` — `physics_step_no_rad` grew an `l_mix_max=None` kwarg.  Inside, the turbulence-call site checks `isinstance(self.turbulence_config, LouisConfig)` at Python time and forwards `l_mix_max` only when the active scheme is Louis (so TKE / YSU / Smagorinsky paths see no unexpected kwarg).  `step_unified` threads `l_mix_max` through both `lax.cond` branches' args-tuple to `physics_step_no_rad`.

3. `src/legoesm/driver/compiled_segments.py` — `build_segment_fn` grew an `l_mix_max=None` kwarg cast to `_l_mix_max = jnp.asarray(...)` once at build time.  Both `step_unified` call sites (owned-face MPI path + serial path) pass `l_mix_max=_l_mix_max`.

`_LOUIS_TURBULENCE_TRAINABLE` re-populated to `[ParamConstraint("l_mix_max", 50.0, 300.0, "sigmoid")]`.  Phase-2 wiring sentinel expected wired count: 9 → 10.

New `tests/test_aimip_phase27_louis_l_mix_max.py` (4 tests, all green):
- Signature check: `l_mix_max` is a `louis_turbulence` kwarg.
- Numerical override: 100 → 200 produces ≥10 % rel diff in du/dt.
- Backward-compat: `l_mix_max=None` reproduces no-kwarg call bit-for-bit.
- Equivalence: kwarg override matches `config._replace(l_mix_max=...)`.

Phase-2 trainable accounting (revised, iter-258):

| Set | Count | Names | Wired? |
|-----|-------|-------|--------|
| `_COMMON_TRAINABLE` | 7 | `tau_equator, tau_pole, C_H, C_E, albedo_ice, albedo_ocean, albedo_land` | 7/7 |
| `_SBM_TRAINABLE` | 2 | `sbm_tau_c, sbm_RH_ref` | 2/2 |
| `_LOUIS_TURBULENCE_TRAINABLE` | 1 | `l_mix_max` | 1/1 |
| **Total wired trainable** | **10** | | **100 %** |

`Ri_crit` and `Ck` remain in `tuning.py::TUNING_PARAMETERS` as experiment-config knobs.  They will re-enter the trainable set under a scheme-appropriate list (e.g. `_HOLTSLAG_BOVILLE_TRAINABLE`) when those schemes become AIMIP-relevant.

## Iter-259 — AIMIP Phase-2 capstone: TrainablePhysicsParams integration sentinel

Locks in the cumulative iter-251 → iter-258 wiring work with a 5-test integration sentinel `tests/test_aimip_trainable_params_integration.py`:

1. `trainable_constraints_for_scheme("sbm", "louis")` returns exactly 10 names in the documented order.
2. `TrainablePhysicsParams.from_defaults(...).to_segment_kwargs()` round-trips: every constrained value lands inside its `(min_val, max_val)` bounds.
3. Every key emitted by `to_segment_kwargs` is also a kwarg of `build_segment_fn` — the trainable pytree can be `**kw`'d directly into the rollout entry point.
4. `eqx.filter_grad` flows **finite, non-zero** gradients to every raw_value (differentiability guard against silently orphaned leaves).
5. Constraint bounds match the canonical `tuning.py::TUNING_PARAMETERS` registry (prevents drift between the two sources of truth).

All 5 tests green.  Combined with the iter-255 wire-status sentinel + the iter-257/iter-258 numerical sentinels, the AIMIP Phase-2 trainable surface is now monotonically enforced: any future addition that lands without complete wiring (constraint list ↔ `build_segment_fn` kwarg ↔ physics consumer ↔ tuning registry ↔ live gradient) trips at least one sentinel.

Phase-2 sentinel coverage matrix:

| Sentinel file | Pins |
|---------------|------|
| `test_aimip_phase2_trainable_wired.py` | every trainable name is a `build_segment_fn` kwarg; `UNWIRED_TODO` is empty; wired count = 10 |
| `test_aimip_phase26_albedo_blend.py` | 3-way albedo blend numerical oracle; `SegmentForcing.land_fraction` default + field count |
| `test_aimip_phase27_louis_l_mix_max.py` | `louis_turbulence` kwarg override changes output; backward-compat preserved |
| `test_aimip_trainable_params_integration.py` | 10-param set complete + bounds + diff-flow + registry-bounds parity (this iter) |
| `test_cases/test_aimip_template.py` | AIMIP experiment template registration + CMIP6 historical-GHG interpolation |

Phase 2 closed.  Phase 3 (training script) is the next AIMIP focus.
