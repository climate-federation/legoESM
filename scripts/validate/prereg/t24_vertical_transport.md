# T24 vertical tracer transport census — preregistration

Before measurement: extend `diag_hybrid_tracer_smoothing.py` using
`--all-vertical-paths --out DIR`; preserve its original smoothing mode.
Run only PBS develop@desched1, UCLB0017, CPU, fp64 policy.

Instantaneous per-column budget: `sum(m*dq + q*dm)`, scaled by
`sum(abs(m*dq)+abs(q*dm))`. Pressure uses `m=dp`, `dm=-diff(F)`;
sigma uses the equivalent unit-surface-pressure weight `dsigma`.
Height uses `m=rho*J*dz`, with `dm` obtained from the native acoustic
density JVP at dt=0, tangent 1, one substep (including off-centering).
Closed lids and positive-up w. Acoustic-only finite updates are also scored.
This is a vertical subsystem diagnostic, not total continuity.

Finite updates use exact `m_new*q_new-m_old*q_old`, scaled by the sum
of absolute layer product increments. Report separately from instantaneous
budgets; these are finite changes, not an estimate of a semidiscrete operator.
Each column is scored separately; no horizontal cancellation or area average.
Zero numerator/denominator gives zero; nonfinite model output invalidates run.
CONSERVES iff maximum relative residual <= 1e-12; otherwise LEAKS.
Nonzero lid flux invalidates the closed-boundary fixture, not a leak verdict.

Known-conserving controls: native hybrid SB and smooth van Leer. Negative
control: native advective hybrid. Planted violation: add `0.01*q*rate` to
nonzero interior layer 24, where rate is each column's maximum interior
`abs(F_face)/min(dp_above,dp_below)`. All four must classify as registered.

Coverage: native pressure tracer callers on MPAS/latlon/spectral hybrid and
sigma, standalone prescribed-wind MPAS/latlon/spectral/cube sigma; MPAS and
spectral NH slow callers plus full split steps; LES serial caller/full step
and both serial/halo local vertical operators; actual FV3 Lagrangian remap
on its existing native fixture/window; explicit and implicit ocean vertical
operators and spectral ocean (report only). Native vmap traces are observed
using restored spies and live callbacks. Local halo columns make no MPI claim.
The table is incomplete if a listed route did not execute. A full height fix
requiring new slow/acoustic time coupling is a task-mandated design STOP.

User approved SB as the spectral production default; FV3 SB only if it leaks.
Other design/default decisions remain reserved to user. No ocean edits.

---

## Execution evidence — 2026-11-22 (post-registration)

# T24 phase 1 — CONFIRMED fixture census

Committed probe 94e936281c924267bb314dc8ca4994299e3b182e; PBS 8278711.desched1; fp64 CPU; complete=True.
Artifact: `/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/a263900a-1926-4ecc-96da-3478d090de9d/scratchpad/codex/t24/census_8278711.desched1`. Script SHA256 `6bef142db0e1d36f6b5f45afe123fc46bf7a43ef294271c1ec7be447afc042ef`.

Per-column vertical-subsystem budgets, zero lids, fixed 1e-12 gate. Maximum relative residual, never a horizontal/global cancellation. Instantaneous denominator = sum(abs(m*dq)+abs(q*dm)); finite denominator = sum(abs(layer exact mass increments)). Ratios are fractions of gross tendencies/redistribution, NOT percent of inventory lost. Zero columns are retained; each observed route has nonzero support. No production-climate or whole-pressure-model claim.

| Path | Scheme/operator | Statistic | Max relative residual | Verdict |
|---|---|---|---:|---|
| control | vertical_advection_hybrid_sb | instantaneous | 9.221715203e-17 | CONSERVES |
| control | vertical_advection_hybrid_van_leer | instantaneous | 6.065527697e-17 | CONSERVES |
| control | vertical_advection_hybrid | instantaneous | 5.362106302e-02 | LEAKS |
| control | planted | instantaneous | 1.040453466e-02 | LEAKS |
| MPAS/hybrid | vertical_advection_hybrid_van_leer | instantaneous | 1.436964062e-16 | CONSERVES |
| latlon/hybrid/sb | vertical_advection_hybrid_sb | instantaneous | 2.023278783e-17 | CONSERVES |
| latlon/hybrid/van_leer | vertical_advection_hybrid_van_leer | instantaneous | 6.065527697e-17 | CONSERVES |
| spectral PE/hybrid | vertical_advection_hybrid | instantaneous | 6.347705887e-02 | LEAKS |
| MPAS/sigma | upwind | instantaneous | 3.785740293e-02 | LEAKS |
| latlon/sigma/sb | vertical_advection | instantaneous | 1.893030853e-02 | LEAKS |
| latlon/sigma/van_leer | vertical_advection | instantaneous | 1.893030853e-02 | LEAKS |
| MPAS/sigma/van_leer | van_leer | instantaneous | 2.114313574e-05 | LEAKS |
| spectral PE/sigma | vertical_advection | instantaneous | 1.584504178e-02 | LEAKS |
| standalone MPAS/sigma | vertical_advection | instantaneous | 1.893030853e-02 | LEAKS |
| standalone latlon/sigma | vertical_advection | instantaneous | 1.893030853e-02 | LEAKS |
| standalone spectral/sigma | vertical_advection | instantaneous | 1.893030853e-02 | LEAKS |
| standalone cube/sigma | vertical_advection | instantaneous | 1.893030853e-02 | LEAKS |
| MPAS NH/height | _vertical_advection_height_1d | instantaneous | 2.588129377e-02 | LEAKS |
| MPAS NH/height | acoustic-only step; tracers unchanged | finite product increment | 4.474670080e-01 | LEAKS |
| MPAS NH/height | full split step | finite product increment | 3.339413424e-02 | LEAKS |
| spectral NH/height | vertical_advection_height | instantaneous | 4.911034246e-02 | LEAKS |
| spectral NH/height | acoustic-only step; tracers unchanged | finite product increment | 4.474670080e-01 | LEAKS |
| spectral NH/height | full split step | finite product increment | 5.755400007e-02 | LEAKS |
| LES serial | _vertical_advection_plane | instantaneous | 2.922159111e-02 | LEAKS |
| LES serial | vertical_advection_van_leer_plane | instantaneous | 1.656134295e-03 | LEAKS |
| LES halo local | _vertical_advection_plane | instantaneous | 2.922159111e-02 | LEAKS |
| LES halo local | vertical_advection_van_leer_plane | instantaneous | 1.656134295e-03 | LEAKS |
| LES serial/height | acoustic-only step; tracers unchanged | finite product increment | 8.548696576e-02 | LEAKS |
| LES serial native caller/centered | _vertical_advection_plane | instantaneous | 2.922159111e-02 | LEAKS |
| LES halo native caller/centered | _vertical_advection_plane | instantaneous | 2.922159111e-02 | LEAKS |
| LES serial native caller/van_leer | vertical_advection_van_leer_plane | instantaneous | 1.656134295e-03 | LEAKS |
| LES halo native caller/van_leer | vertical_advection_van_leer_plane | instantaneous | 1.656134295e-03 | LEAKS |
| LES serial/height | full split step | finite product increment | 2.446005468e-02 | LEAKS |
| FV3 remap | 0 | finite product increment | 1.070652474e-15 | CONSERVES |
| FV3 remap | 1 | finite product increment | 7.336441281e-16 | CONSERVES |
| ocean variable-thickness subsystem | vertical_advection_ocean | instantaneous | 1.112350252e-01 | LEAKS |
| ocean variable-thickness subsystem | _vertical_advection_spectral | instantaneous | 1.112350252e-01 | LEAKS |
| ocean | implicit flux form fixed thickness | finite product increment | 7.858615046e-17 | CONSERVES |

All four controls met their preregistered classifications. Native callback spies executed on every listed caller; local halo results are single-rank vertical columns, not an MPI horizontal budget. FV3 uses the native remap fixture/compute window, not a climate integration. Ocean explicit advective variable-thickness and implicit fixed-thickness budgets differ by design and are not an A/B; REPORT ONLY, no ocean edits.

## Phase-2 design gate

CONFIRMED: MPAS NH, spectral NH and LES acoustic-only updates change weighted tracer mass while carrying tracer mixing ratios unchanged. Full native split steps also leak. Replacing only the slow tracer reconstruction cannot supply the missing acoustic-stage tracer mass flux. A correct full-step fix needs a decision about tracer mass transport during acoustic/RK coupling. The task ballot item8 explicitly requires STOP for a new time-coupling algorithm. No model edits made.

Source: compressible_euler_mpas.py:474-538 updates rho with rho_w, returns state.tracers unchanged; spectral_nh.py:550-624 returns only w/theta/rho, with tracers retained by acoustic_update_fn:848-903; compressible_euler_plane.py:2441-2522 returns w/theta/rho with original tracers. Native acoustic JVPs and finite steps supply measured evidence above.

ASKED: spectral T31 production default SB (approved); FV3 does not leak in this remap fixture, so no conditional FV3 default change. OPEN: height slow/acoustic tracer time coupling; default selections for other formerly leaking lanes beyond the expressly approved spectral production setting, if any new production default would be required. No default selected silently.

Required branch+ratchet suites: 1207 passed, 2 skipped, 1 pre-existing failure; exact failure/value reproduced on origin/main and detached baseline worktree removed. See tests.md for exact pytest summary lines. Phase2-specific tests not run because no phase2 implementation exists. Result-claim GLM CONFIRM and Claude CONFIRM (claude_result_review2.md); prior code review Codex PASS.

## Concrete design decision reserved to user

Authorize continuity-paired tracer mass transport through height-core acoustic substeps AND mass-weighted RK stage combinations. Recommended conservation target: each substep updates m*q with the same density-continuity face flux times the selected tracer face value, then recovers q using the resulting m; stage blends preserve m*q rather than separately blending m and q. This describes the needed invariant and coupling scope, not an implemented or selected algorithm. It also removes double counting of vertical transport in slow tendencies. All existing explicit/semi-implicit acoustic variants would need matching treatment and tests before calling the height fix complete. Task/ballot item8 forbids silently choosing this new time coupling.

No new prognostic field, production default or time-coupling algorithm was selected. No model/production/ocean code edited; no push/PR edits/merge.

## Spectral-row adversarial challenge

Claude result reviewer queried whether spectral PE ratios were roundoff-over-roundoff, because the displayed raw pair came from the maximum-relative column at a longitude node. Discriminating check on the ALREADY RETAINED raw arrays (no new gate/statistic/run): at the largest-denominator column, hybrid numerator1.0827879349821793e-4 / denominator2.106522142582237e-3 =0.051401687791179165, sigma numerator1.1291262800886032e-9 / denominator8.636146315169947e-8 =0.013074422767771085. Both meaningful-scale columns independently LEAK far above1e-12. The original max-relative verdicts remain valid.

Tracer mixing-ratio storage verified in core/state.py:478 (MPAS kg/kg), spectral_nh.py:1205, compressible_euler_plane.py:505. Spectral acoustic-only equality with MPAS is expected: both call their actual grid acoustic routines on deliberately identical frozen rho/w/q profiles and heightcoord. The spectral full-step row independently includes the native model spectral round trips.

Measured height acoustic configurations are EXPLICIT: SpectralNHConfig.semi_implicit_acoustic defaults False (spectral_nh.py:106); LES fixture explicitly sets False (test_plane_nh_tracers.py:52); MPAS uses its explicit native substeps. No inference that all production acoustic variants were exercised. Claude final caveat about a possibly different default is therefore refuted for these fixtures.

# T24 test record

All jobs PBS develop@desched1 / UCLB0017 / CPU / JAX_ENABLE_X64=1; explicit PYTHONPATH uses this checkout. Each test module runs in a fresh Python process. No model code changed.

## t24_branch.log

=================== 1 failed, 73 passed in 255.05s (0:04:15) ===================
======================== 35 passed in 61.99s (0:01:01) =========================
============================= 18 passed in 37.25s ==============================
======================= 386 passed, 2 skipped in 30.22s ========================
======================== 90 passed, 1 warning in 22.84s ========================
============================== 6 passed in 0.81s ===============================

## t24_ratchets.log

============================= 24 passed in 26.31s ==============================
============================== 76 passed in 8.91s ==============================
============================== 81 passed in 9.68s ==============================
============================== 7 passed in 7.44s ===============================
============================= 405 passed in 12.34s =============================
============================== 6 passed in 7.56s ===============================

## t24_probe_unit.log

============================== 3 passed in 1.65s ===============================
============================== 5 passed in 1.66s ===============================
============================== 6 passed in 2.77s ===============================

## t24_origin_main.log

=================== 1 failed, 39 passed in 95.45s (0:01:35) ====================

Combined required branch+ratchet run: 1207 passed, 2 skipped, 1 failed (known origin/main failure). The isolated six-test instrument validation is a repeat, not added to that total.

Failure: TestTracerTransport.test_physics_tracer_tendency_applied expected q_v=1.0, observed 0.5025125558252623. EXACT SAME failed assertion/value on origin/main 3c40954272 in detached /glade/derecho/scratch/pg2328/t24_origin_main_baseline (PBS8278708); origin/main summary 1 failed,39 passed in95.45s, fail-fast. Baseline worktree removed after clean-status verification. Original branch summary 1 failed,73 passed in255.05s. This is pre-existing; no unrelated physics fix.

Phase-2 paired-revert, selector roundtrip, unchanged-production bitwise fixtures and new-path fp32 gradients NOT RUN: no physics/selector implementation exists because the explicit height time-coupling design STOP was reached. Do not represent phase2 as complete.

Reviews: GLM full-diff PASS; independent Codex code PASS; Claude code conditional callback HOLD refuted by six-test PBS gate. GLM and Claude measured-result reviews CONFIRM the finding and mandated design STOP.
