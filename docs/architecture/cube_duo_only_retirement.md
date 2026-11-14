# Cubed-sphere atmosphere retirement

User decision: 2026-11-13. FV3 duo is the only supported cubed-sphere
atmosphere dynamical core. Retired selectors raise ValueError naming fv3_duo,
including registered plugins using the twelve retired flat names. Other plugin
names retain their existing registry precedence. Passive tracer transport is
retained: it prescribes winds rather than implementing an atmosphere dycore.

## Defaults changed

| Entry point | Old cube selection | New cube selection |
| --- | --- | --- |
| driver DycoreConfig | cdgrid | fv3_duo |
| AMIP grid resolver | centered | fv3_duo |
| Coupled grid resolver | cdgrid | fv3_duo |
| Nested YAML default/fallback | cdgrid | fv3_duo |
| Experiment-template resolver | cdgrid | fv3_duo |
| RCE grid-aware default | cdgrid | fv3_duo |
| Levante segment benchmark resolver | cdgrid | fv3_duo |
| Held-Suarez/RRTMGP all-grids deck | cdgrid | fv3_duo |
| AMIP all-grids smoke cube labels | centered / finite_volume / cdgrid | fv3_duo |
| amip_sota.yaml | cdgrid | fv3_duo |
| amip_realism_c12.yaml | implicit legacy default | explicit fv3_duo |
| Atmosphere matrix cube adapters | legacy model constructors / production / FB | fv3_duo; unsupported adapters refuse |

Non-cube resolver rows and numerical implementations are preserved. The RCE
parser uses a sentinel to change only the cube default; its old non-cube
fallback remains cdgrid. Historical smoke labels are retained as labels, with
all cube labels explicitly selecting fv3_duo.

## Parity backlog: configurations that now refuse

- Cube shallow-water atmosphere: Williamson, cosine bell, modons, CLI
  test/benchmark, legacy SW benchmarks and visual-regression adapters.
- Existing cube atmosphere matrix adapters: Held-Suarez, baroclinic/AMIP,
  moist and nonhydrostatic cases. Legacy cube benchmark, scaling, profiler,
  and scan-equivalence adapters also have no duo port in this change.
- Main-path full-physics AMIP/RCE and coupled setups requiring that path.
  The two historical AMIP decks still carry incompatible main-path
  physics/levels/options and deliberately fail. The existing hydrostatic
  duo column physics remains available under its existing contract.
- fp32/float32 and mixed precision with fp32 storage. Existing uniform fp64,
  including mixed_fp64_storage, remains supported.
- MPI/mpi4jax atmosphere decomposition. Distributed duo retains its existing
  SPMD and window-layout policies.
- Vertical counts outside 5/10 on the main lane or 5/10/32 on the column lane.
  The existing CAM6 L32 table is retained; no vertical table is added.
- Height-coordinate compressible Euler, semi-implicit/selectable time
  integrators, and implicit-gravity-wave switches. The existing dry native
  pressure-coordinate duo NH arm remains supported.
- Main-path dycore energy-fixer use where it is not wired; the existing
  factory refuses rather than accepting an open energy sink silently.
- Legacy persistent_dgrid, enable_tiled_dycore, moisture_advection and
  GridConfig.use_duogrid switches. Both duo lanes refuse these retired
  switches. Native duo winds, halos and tracer transport remain valid.
- Other legacy customizations rejected by the duo default-deny wall. Existing
  HS+Kessler, NH HS/Kessler, and NH column-physics restrictions remain.

No physics, numerics, tolerance, non-cube production default, forcing, or
vertical table is ported or changed.

## Shared and ambiguous modules retained

- atmosphere/dynamics/gcm/shallow_water_fv3_cdgrid.py, including its classes:
  OceanModel uses its fv3sw/fv3edge provider builders. Atmosphere dispatch and
  re-exports are removed; explicit ocean provider imports remain.
- core/fv3_sw_core.py, core/fv3_del6_vt_flux.py and
  core/_fv3_divergence_corner.py: shared SW/ocean dependencies.
- atmosphere/_future/_fv3_lin_pgf.py: quarantined research helper with separate
  tests, retained as ambiguous rather than expanding the retirement scope.
- Shared cubed-sphere/CD-grid/duogrid grids, halos, native FV3 kernels,
  operators, vertical coordinates, precision, field/state and reductions.

## Side decisions

- User-directed: reject retired selections instead of aliasing them silently.
- User-directed: retain shared/ambiguous modules; the ocean registry is the
  verified source dependency requiring the shallow-water module to stay.
- User-directed: refuse unported matrix/benchmark adapters instead of adding
  physics coupling, vertical tables or numerical ports.
- Preserve existing accumulated config diagnostics; factory lookup refuses
  before constructing a retired solver.
- Preserve generic test coverage by using the existing lat-lon pipeline for
  manifest/replay bookkeeping and the existing Gaussian branch for generic
  numeric CLI validation. Scientific inputs and tolerances stay unchanged.
- Reject only the twelve exact retired plugin names; custom plugin names
  retain their existing behavior. Tests include a registered retired-name
  sentinel and a valid custom plugin control.
- Two independent read-only Codex reviewers approved dispatch and deletion
  scope, including the test-fixture and exact-name registry choices. The
  named Claude/GLM review pair in CLAUDE.md was not run. A disputed claim
  about the RCE file was checked against actual source: all three non-cube
  smoke cases and three non-cube nightly cases remain; the HOLD was withdrawn.

## Validation environment and mutation controls

All Python execution and tests ran via SLURM CPU, account bb1596, partition
compute, JAX_ENABLE_X64=1, using the requested virtualenv. The namespace guard
restricts legoesm to this worktree, prints imported paths and is inherited by
subprocesses. No tests ran on the login node.

Implementation commit before the first controls: e137cbf996.

- Driver-selector mutation: restored the original driver_solver_name function
  from 296752ec11117e42eb055e3c9bf5e9767e0b9f80. The tool printed MUTATION APPLIED;
  the legacy-selection test produced `1 failed, 1 warning in 13.22s`.
- Default mutation: restored DycoreConfig.discretization=cdgrid. The tool
  printed MUTATION APPLIED; the defaults test produced
  `1 failed, 1 warning in 0.95s`.
- Each mutation restored saved bytes in finally and asserted a clean tracked
  status. The restored check produced `177 passed, 2 warnings in 59.54s`.
- Final controls on c9bdc40c26 repeated both applied reversions: the legacy
  test produced `1 failed, 1 warning in 14.15s`; the defaults test produced
  `1 failed, 1 warning in 0.88s`. Saved bytes were restored, git status was
  clean, and the restored suite produced
  `177 passed, 2 warnings in 59.56s` (`POST-MUTATION CHECK PASSED`).

Local raw logs and control scripts are retained, ignored, under
`.pytest_cache/cube-retirement/`.

## Decisive pytest summaries

- Initial core/factory/federation/dispatch/validation suites:
  `194 passed, 2 warnings in 77.52s (0:01:17)`.
- AMIP CLI: `367 passed, 1 warning in 34.20s`.
- Latest retirement/plugin/legacy-option controls plus corrected bookkeeping
  predicates and shrunk-import minimality:
  `22 passed, 1 warning in 15.53s`.
- RCE CLI validation: `19 passed, 1 warning in 76.95s (0:01:16)`.
- Existing duo grid/factory/refusal/default-deny suites:
  `29 passed, 1 warning in 235.62s (0:03:55)`.
- Retained duo Kessler, lat-lon wind, coupled guard, standard-IC, and import
  minimality coverage: `43 passed, 1 warning in 51.18s`.
- Grid-independent bookkeeping/replay suite after fixture correction:
  `1 failed, 29 passed, 3 skipped, 1 warning in 116.12s (0:01:56)`.
  The sole failure is the independently reproduced original Kessler hard-guard
  fixture; all surviving generic bookkeeping/replay checks pass.
- Retained vapour-floor warning cases: `5 passed, 1 warning in 1.91s`.
- CLI/driver/tracer/ratchet sweep before the final allowance shrink:
  `3 failed, 6386 passed, 2 skipped, 4 warnings in 186.89s (0:03:06)`.
  The stale-allowance failure was subsequently corrected by removing existing
  allowances only; its minimality test passed in the focused checks. The
  remaining import and literal failures reproduce on original tests/sources.
- Affected-file collection before the supplementary cleanup:
  `6625 tests collected in 34.58s`.
- Remaining RCE smoke/nightly collection:
  `3/6 tests collected (3 deselected) in 0.08s` under the repository's default
  slow-test selection. The three nightly cases remain in source.

### Inherited failures: verified against original tests/sources

No allowance or tolerance is increased for these failures.

- Inline-import ratchet, original HEAD sources/budgets:
  `25 failed, 41 passed, 1 warning in 11.74s`.
  Current tree: `23 failed, 41 passed, 1 warning in 13.05s`.
  Both removed entries were failing retired modules; surviving over-budget
  counts are unchanged or reduced.
- Original-test replays each reproduced the same failure and restored a clean
  tree: shared vector-lift roughness; spectral PE first-mass snapshot inside
  grad; two conservation-pattern ratchets in unchanged ocean probes; the
  lake-freeze fixture's hardcoded constant; production test imports in five
  unchanged scripts; and module-top arrays in the unchanged land canopy solver.
- Idealized Kessler recording test: original test and unchanged recording/config
  functions reproduced `1 failed, 1 warning in 2.50s`. It asks for hard
  saturation adjustment that Kessler already refuses; this is not ported here.

### Unverified

- Full repository suite, GPU scaling, and non-cube nightly integrations were
  not run. The existing one-day Gaussian/slab coupled test timed out at its
  unchanged 600-second limit; no timeout was increased.
- Named Claude/GLM review pair was not run; the two available independent
  reviewers were Codex agents.

### Local commits

- e137cbf996: primary retirement implementation and initial controls.
- c9bdc40c26: completed entrypoint, legacy-option, plugin and shared-test audit.
- The final documentation-only commit records the completed controls. No push
  or PR was performed.

### Per-file mixed checks on the retired tree

Each file ran in a fresh Python process after a JAX symbol-materialization error
in the initial combined run. Failing checks below are covered by the original
baseline replays above; their numerical code is unchanged.

- `tests/atmosphere/dycore/regression/test_nh_rayleigh_fast_iter448.py`: `2 passed, 1 warning in 2.08s`.
- `tests/atmosphere/dycore/unit/test_fv3_naming_phase5.py`: `2 passed, 1 warning in 0.48s`.
- `tests/atmosphere/hydrostatic/unit/test_gw_oro_cam6_followups.py`: `15 passed, 3 warnings in 45.84s`.
- `tests/atmosphere/nonhydrostatic/integration/test_nh_mass_conservation_anchored.py`: `4 passed, 1 warning in 30.01s`.
- `tests/atmosphere/test_cube_moist_tracer_advection.py`: `1 passed, 1 warning in 17.99s`.
- `tests/parallel/test_cube_tile_native_segment.py`: `1 passed, 1 warning in 0.65s`.
- `tests/parallel/test_tiled_fv3_hydrostatic_tracer.py`: `1 passed, 1 warning in 27.76s`.
- `tests/parallel/test_warmup_tiled_cube_comms.py`: `1 skipped, 1 warning in 0.48s`.
- `tests/unit/test_diff_atmosphere_dynamics.py`: `12 passed, 1 warning in 95.36s (0:01:35)`.
- `tests/unit/test_moisture_advection_wiring.py`: `8 passed, 1 warning in 13.78s`.
- `tests/unit/test_pr_f_fixes.py`: `1 passed, 1 warning in 2.99s`.
- `tests/validation/test_differentiability_regression.py`: `8 passed, 1 warning in 186.15s (0:03:06)`.
- `tests/atmosphere/dycore/regression/test_smagorinsky_grad_at_zero_iter181.py`: `3 passed, 1 warning in 28.80s`.
- `tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py`: `3 passed, 1 warning in 0.53s`.
- `tests/atmosphere/hydrostatic/unit/test_primitive_eq.py`: `29 passed, 1 warning in 27.80s`.
- `tests/atmosphere/nonhydrostatic/integration/test_nh_terrain_jacobian_mass.py`: `3 passed, 1 warning in 10.30s`.
- `tests/distributed/test_cube_face_scatter_mpi.py`: `1 passed, 1 warning in 72.03s (0:01:12)`.
- `tests/parallel/test_sharded_step_sharding_tripwire.py`: `1 passed, 1 skipped, 1 warning in 0.54s`.
- `tests/parallel/test_tiled_halo_grouping.py`: `1 passed, 1 skipped, 1 warning in 0.83s`.
- `tests/test_vector_cc_to_dgrid_wind_lift.py`: `1 failed, 1 warning in 29.59s`.
- `tests/unit/test_diff_coupled_system.py`: `8 passed, 1 warning in 220.81s (0:03:40)`.
- `tests/unit/test_no_module_top_jax_alloc.py`: `1 failed, 53 passed, 1 warning in 30.22s`.
- `tests/unit/test_precision_leaf_dtypes.py`: `6 passed, 2 warnings in 56.35s`.
- `tests/validation/validation_differentiability_all.py`: `1 xfailed, 1 warning in 25.42s`.
- `tests/atmosphere/dycore/regression/test_te_drift_iter599.py`: `2 passed, 1 warning in 15.63s`.
- `tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py`: `4 passed, 1 warning in 49.21s`.
- `tests/atmosphere/hydrostatic/validation/test_mass_conservation_anchored.py`: `6 passed, 1 warning in 29.26s`.
- `tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py`: `25 passed, 1 warning in 27.16s`.
- `tests/distributed/test_mpi_driver.py`: `2 passed, 1 warning in 1.88s`.
- `tests/parallel/test_tiled_center_to_dgrid_vector.py`: `1 passed, 1 warning in 27.62s`.
- `tests/parallel/test_tiled_operator_split_step.py`: `1 skipped, 1 warning in 0.47s`.
- `tests/unit/test_conservation_drift.py`: `2 failed, 60 passed, 1 warning in 16.90s`.
- `tests/unit/test_dycore_energy_fixer.py`: `17 passed, 1 warning in 104.32s (0:01:44)`.
- `tests/unit/test_physical_balances.py`: `4 passed, 1 warning in 43.85s`.
- `tests/validation/test_cross_grid_nh_consistency.py`: `8 passed, 1 warning in 11.61s`.
- `tests/atmosphere/dycore/test_persistent_dgrid_1028.py`: `3 passed, 1 warning in 29.30s`.
- `tests/atmosphere/hydrostatic/unit/test_cross_grid_water_positivity.py`: `5 passed, 1 warning in 10.23s`.
- `tests/atmosphere/hydrostatic/validation/test_pe_flux_form_continuity.py`: `6 passed, 1 warning in 37.86s`.
- `tests/atmosphere/test_anchor_mass_api.py`: `1 failed, 20 passed, 1 warning in 41.35s`.
- `tests/parallel/test_cube_face_scatter.py`: `4 passed, 1 warning in 25.20s`.
- `tests/parallel/test_tiled_fv3_hydrostatic_momentum.py`: `2 passed, 1 warning in 33.13s`.
- `tests/parallel/test_tiled_vertical_pe_stage.py`: `1 passed, 1 warning in 2.92s`.
- `tests/unit/test_convection_config_for.py`: `7 passed, 1 warning in 13.34s`.
- `tests/unit/test_issue_fixes.py`: `12 passed, 1 warning in 37.82s`.
- `tests/unit/test_physics_state_carry.py`: `39 passed, 2 warnings in 187.58s (0:03:07)`.
- `tests/validation/test_cross_grid_rce_smoke.py`: `3 passed, 1 warning in 25.95s`.

## Deleted files

96 files: six implementation modules, eleven scripts, and 79 exclusive tests.

- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_cdgrid.py`
- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py`
- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/semi_implicit_cdgrid.py`
- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py`
- `packages/core/legoesm/parallel/tiled_production_cdgrid.py`
- `packages/coupler/legoesm/driver/tiled_operator_split_step.py`
- `scripts/bench/bench_cube_shardmap_halo.py`
- `scripts/bench/bench_cube_tiled_step_scaling.py`
- `scripts/bench/bench_fv3_sw_fb_vs_production.py`
- `scripts/bench/profile_cs_dycore.py`
- `scripts/run/example_fv3_clip_helper.py`
- `scripts/run/run_w2_w5_cosine_bell_iter1030.py`
- `scripts/validate/cg_helmholtz_mixed_precision_1675.py`
- `scripts/validate/run_colliding_modons.py`
- `scripts/validate/validate_cubed_sphere_fv3_atmos.py`
- `scripts/validate/validate_driver_tiled_dycore_parity.py`
- `scripts/validate/validate_tiled_fv3_sw_multinode.py`
- `tests/atmosphere/dycore/regression/test_corner_div_d4_bg_scaling_iter282.py`
- `tests/atmosphere/dycore/regression/test_corner_div_damp_d_con_quantitative_iter228.py`
- `tests/atmosphere/dycore/regression/test_corner_div_damp_nord1_quantitative_iter198.py`
- `tests/atmosphere/dycore/regression/test_cube_imprint_d_con_iter233.py`
- `tests/atmosphere/dycore/regression/test_cube_imprint_d_con_nh_iter234.py`
- `tests/atmosphere/dycore/regression/test_d_con_global_energy_balance_iter238.py`
- `tests/atmosphere/dycore/regression/test_damp_v_quantitative_iter175.py`
- `tests/atmosphere/dycore/regression/test_damp_w_quantitative_iter195.py`
- `tests/atmosphere/dycore/regression/test_div_damp_quantitative_iter174.py`
- `tests/atmosphere/dycore/regression/test_factory_sponge_defaults_iter444.py`
- `tests/atmosphere/dycore/regression/test_fidelity_flags_default_off.py`
- `tests/atmosphere/dycore/regression/test_full_toolkit_100step_stability_iter244.py`
- `tests/atmosphere/dycore/regression/test_fv3_faithful_config_factories_iter392.py`
- `tests/atmosphere/dycore/regression/test_fv3_full_toolkit_ad_at_rest_iter184.py`
- `tests/atmosphere/dycore/regression/test_iter1039_3d_cube_edge_smoothness.py`
- `tests/atmosphere/dycore/regression/test_long_run_mass_iter539.py`
- `tests/atmosphere/dycore/regression/test_mass_conservation_sbr_iter523.py`
- `tests/atmosphere/dycore/regression/test_metric_aware_d_con_ast_guard_iter340.py`
- `tests/atmosphere/dycore/regression/test_metric_d_con_edge_concentration_iter345.py`
- `tests/atmosphere/dycore/regression/test_nh_d_con_cv_global_energy_iter321.py`
- `tests/atmosphere/dycore/regression/test_nh_duogrid_reduces_cube_imprint_iter326.py`
- `tests/atmosphere/dycore/regression/test_nh_full_fv3_stack_imprint_iter330.py`
- `tests/atmosphere/dycore/regression/test_pe_aam_iter604.py`
- `tests/atmosphere/dycore/regression/test_pe_d_con_global_conservation_iter243.py`
- `tests/atmosphere/dycore/regression/test_pe_duogrid_reduces_imprint_iter335.py`
- `tests/atmosphere/dycore/regression/test_pe_full_toolkit_ad_at_rest_iter185.py`
- `tests/atmosphere/dycore/regression/test_pe_mass_conservation_iter524.py`
- `tests/atmosphere/dycore/regression/test_pe_te_boundary_sign_iter603.py`
- `tests/atmosphere/dycore/regression/test_smag_vort_cap_formula_iter249.py`
- `tests/atmosphere/dycore/regression/test_smag_vort_cap_formula_nh_iter250.py`
- `tests/atmosphere/dycore/regression/test_te_correction_pe_iter602.py`
- `tests/atmosphere/dycore/regression/test_total_energy_pe_iter598.py`
- `tests/atmosphere/dycore/unit/test_cdgrid_flux_form_moisture.py`
- `tests/atmosphere/hydrostatic/integration/test_amip_smoke.py`
- `tests/atmosphere/hydrostatic/integration/test_amip_stability.py`
- `tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py`
- `tests/atmosphere/hydrostatic/unit/test_semi_implicit_cdgrid.py`
- `tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py`
- `tests/atmosphere/shallow_water/unit/test_run_colliding_modons_cli.py`
- `tests/bench/test_bench_cube_shardmap_halo.py`
- `tests/bench/test_bench_cube_tiled_step_scaling.py`
- `tests/distributed/test_mpi_fv3_nh_step_fidelity.py`
- `tests/distributed/test_mpi_fv3_step_fidelity.py`
- `tests/parallel/test_cube_moist_spmd_parity.py`
- `tests/parallel/test_cubed_sphere_spmd_step.py`
- `tests/parallel/test_operator_split_tiled_cube_driver_parity.py`
- `tests/parallel/test_tiled_3d_velocity.py`
- `tests/parallel/test_tiled_al_gradient.py`
- `tests/parallel/test_tiled_blocked_loop.py`
- `tests/parallel/test_tiled_box_interps.py`
- `tests/parallel/test_tiled_cc_step_adapter.py`
- `tests/parallel/test_tiled_cube_spmd_driver.py`
- `tests/parallel/test_tiled_dgrid_vorticity.py`
- `tests/parallel/test_tiled_dp_s_dt.py`
- `tests/parallel/test_tiled_fix_ps_mass.py`
- `tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py`
- `tests/parallel/test_tiled_fv3_hydrostatic_step.py`
- `tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py`
- `tests/parallel/test_tiled_fv3_hydrostatic_thermo.py`
- `tests/parallel/test_tiled_fv3_sw_full.py`
- `tests/parallel/test_tiled_fv3_sw_momentum.py`
- `tests/parallel/test_tiled_geopotential.py`
- `tests/parallel/test_tiled_mass_divergence.py`
- `tests/parallel/test_tiled_mass_flux.py`
- `tests/parallel/test_tiled_vadv_omega.py`
- `tests/parallel/test_tiled_velocity_transforms.py`
- `tests/parallel/test_tiled_zero_mean_tendency.py`
- `tests/test_a2b_zeta_corner_nh.py`
- `tests/test_async_halo_nh.py`
- `tests/test_corner_div_damp_nh.py`
- `tests/test_damp_v_nh.py`
- `tests/test_div_damp_adaptive.py`
- `tests/test_div_damp_nh.py`
- `tests/unit/test_bench_fv3_sw_fb_vs_production.py`
- `tests/unit/test_cube_dgrid_wind_restart.py`
- `tests/unit/test_matrix_fv3_native_grid_cli.py`
- `tests/unit/test_matrix_sw_core_cli.py`
- `tests/unit/test_tiled_dycore_driver_dispatch.py`
- `tests/unit/test_validate_tiled_dycore_parity_cli.py`
