# Cubed-sphere atmosphere retirement (2026-11-13)

The user selected FV3 duo as the only cubed-sphere atmosphere dynamical core.
Legacy selectors raise ValueError naming fv3_duo; they do not alias to duo or
fall back to another core. Atmospheric registry entries and re-exports are
removed. Non-cube dispatch and numerical implementations are preserved.

## Defaults

- Driver DycoreConfig: cdgrid -> fv3_duo.
- AMIP cube grid resolver: centered -> fv3_duo.
- Coupled cube grid resolver: cdgrid -> fv3_duo.
- Nested YAML cube default and fallback: cdgrid -> fv3_duo.
- Experiment-template cube resolver: cdgrid -> fv3_duo.
- RCE cube CLI default: cdgrid -> fv3_duo.
- amip_sota.yaml: cdgrid -> fv3_duo.
- amip_realism_c12.yaml: implicit legacy default -> explicit fv3_duo.
- Atmosphere matrix cube selector: production/FB -> fv3_duo; the old matrix
  adapters refuse execution because they have no duo implementation.

## Parity backlog

These configurations now fail rather than execute the retired atmosphere:

- Cube shallow-water atmosphere, including Williamson, cosine bell, modons,
  CLI test/benchmark, legacy SW benchmarks and visual-regression adapters.
  Shared ocean SW remains available through its ocean registry.
- The existing cube atmosphere matrix adapters (Held-Suarez, baroclinic/AMIP,
  moist and nonhydrostatic cases), legacy cube benchmark/scaling adapters,
  and the old cube scan-equivalence validator. Their non-cube branches remain.
- Main-path full-physics AMIP/RCE and coupled setups requiring that path.
  The existing hydrostatic duo column lane retains its existing physics;
  no physics coupling is ported here. The two historical AMIP decks still
  carry incompatible main-path physics/levels/options and deliberately fail.
- fp32/float32 and mixed precision with fp32 storage. Existing uniform fp64
  (including mixed_fp64_storage) behavior remains.
- MPI/mpi4jax atmosphere decomposition. The supported distributed duo lane
  remains SPMD, including its existing window layout.
- Unsupported vertical counts: main lane outside 5/10; column lane outside
  5/10/32. The column 32-level lane keeps the existing CAM6 table.
- Height-coordinate compressible Euler; the dry duo NH pressure-coordinate
  arm remains a distinct supported lane within its existing restrictions.
- Semi-implicit/selectable time integrators and implicit-gravity-wave knobs.
- Legacy persistent_dgrid and tiled C/D-grid atmosphere step/operator-split
  adapters, plus non-default options already refused by duo's default-deny
  contract. Held-Suarez plus Kessler, NH Held-Suarez/Kessler, and NH column
  physics remain refused under their existing contracts.

## Shared and ambiguous modules retained

- atmosphere/dynamics/gcm/shallow_water_fv3_cdgrid.py: OceanModel uses its
  fv3sw/fv3edge registry builders, including both model classes. Its module
  and shared tests stay; atmosphere dispatch and re-exports are removed.
- core/fv3_sw_core.py, core/fv3_del6_vt_flux.py, core/_fv3_divergence_corner.py:
  shared SW/ocean dependencies.
- atmosphere/_future/_fv3_lin_pgf.py: quarantined research helper with separate
  tests; retained as ambiguous rather than expanding the retirement scope.
- All shared cubed-sphere/CD-grid/duogrid grids, halo code, native FV3 kernels,
  operators, vertical coordinates, precision, field/state and reductions.

## Side decisions

- ASKED: retire rather than silently alias the old cube selections; use
  ValueError naming fv3_duo, as explicitly requested.
- ASKED: preserve shared/ambiguous modules; the ocean SW registry is the
  concrete reason the shallow-water module is retained.
- ASKED: refuse unported matrix/benchmark adapters instead of inventing ports.
- Existing validation's accumulated-error behavior is retained; the driver
  lookup refuses immediately before construction. No physics, numerics,
  tolerance, non-cube default or allow-list budget is changed.
- Two independent read-only reviewers approved dispatch and deletion scope.
  The named Claude/GLM tools were unavailable; reviewers were independent
  Codex agents. Their runtime checks were not independent reruns.

## Validation

All Python execution and tests use SLURM CPU, account bb1596, partition compute,
JAX_ENABLE_X64=1, and the specified virtualenv. A namespace guard restricts
legoesm to this worktree and prints imported paths; subprocess checks inherit
that guard. No tests execute on the login node.

- Initial corrected core suites: 194 passed, 2 warnings in 77.52s (0:01:17).
- Updated AMIP CLI suite: 367 passed, 1 warning in 34.20s.
- Affected-file collection: 6625 tests collected in 34.58s.
- Inline-import ratchet on original HEAD sources/budgets: 25 failed, 41 passed,
  1 warning in 11.74s. Current tree: 23 failed, 41 passed, 1 warning in 13.05s.
  The removed entries were both failing legacy modules. Every surviving
  over-budget module's count is unchanged or reduced; no budget is increased.

Final validation, mutation controls and commit identifiers are recorded below
once their SLURM runs finish.

## Deleted files

94 files:

- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_cdgrid.py`
- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py`
- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/semi_implicit_cdgrid.py`
- `packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py`
- `packages/core/legoesm/parallel/tiled_production_cdgrid.py`
- `packages/coupler/legoesm/driver/tiled_operator_split_step.py`
- `scripts/bench/bench_cube_shardmap_halo.py`
- `scripts/bench/bench_cube_tiled_step_scaling.py`
- `scripts/bench/bench_fv3_sw_fb_vs_production.py`
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
