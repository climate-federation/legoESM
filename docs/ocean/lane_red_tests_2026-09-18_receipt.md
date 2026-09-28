# GYRE lane pre-existing red-test receipt — 2026-09-18

## Scope and result

This receipt covers the 87 node IDs in
`merge_main_2026-09-17/tests/preexisting_red_ids_on_both_trees.txt`: 71 test
failures and 16 setup errors that reproduced both at the lane tip and after the
2026-09-17 main merge.  The first-pass repair is commit `6cb7419da` (with its
immediately following shared-diagnostics completion `eb8bbb20f`).  The
independent-review repair was performed in
`/tmp/redtests87-fix-3024223959`, on branch
`fix/gyre-lane-red-tests-87`, from that branch tip.

Result: **70 fixed, 5 held, and 12 skipped with an explicit environment
reason, of 87**.  Every listed node was first reproduced from the unmodified
base with one pytest process per source file.  At the repaired tip, every held
node remains an explicit failure and every environment node reports an explicit
reason rather than an xfail or bare skip.

No GYRE trajectory comparison was required: this change does not alter a
configuration choice or any model statement executed by
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py`.  The production
fixes are confined to the MPAS adapter/barotropic path, an offline tendency
probe, and matrix-runner diagnostics.  The only edits under the protected
`nemo_testcase_l2_gyre_*` glob are the nine metadata-only `worktree_stamp()`
calls explicitly required by the worktree-stamp ratchet; they do not execute in
the identity-card trajectory.

## Commit chain

The source checkout's `.git` metadata was read-only, so the second pass used
the required shared clone.  Each review finding has one commit:

| Finding | Commit |
|---|---|
| A1 | `8ef8d964d` |
| A2 | `ae20871d6` |
| A3 | `27df4811b` |
| A4 (held) | `ca5f2ec86` |
| A5 (held) | `7c8421129` |
| B3 | `bd13e4ef3` |
| B2 | `514bf7ac0` |
| B7 | `7a8a0644d` |
| B4 | this receipt's final commit |

## Isolated proof lines

The before logs are retained in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/parallel/redtests87/baseline_isolated/`;
the first repaired batch is retained in the sibling `after_batch1/` directory.
The remaining after-lines were captured directly from isolated pytest
processes using the same CPU-only environment.

| Proof | Before (change reverted/base) | After |
|---|---|---|
| P01 stamp | `1 failed, 9 passed in 2.32s` | `10 passed in 1.96s` |
| P02 advection AD | assertions removed: `11 passed in 81.51s` | assertions restored: `4 failed, 7 passed in 135.57s` (HELD) |
| P03 baroclinic golden | `1 failed, 1 passed in 15.73s` | `2 passed in 23.69s` |
| P04 frozen tide | `1 failed, 24 passed in 125.99s` | target: `1 passed in 18.16s` |
| P05 seasonal artifacts | `1 failed, 5 passed in 1.21s` | `6 passed, 1 skipped in 1.58s` |
| P06 vertex area | `2 failed, 9 passed, 3 errors in 13.61s` | `14 passed in 13.83s` |
| P07 v-face width | `2 failed, 9 passed, 2 errors in 16.05s` | `13 passed in 14.83s` |
| P08 EKE goldens | `3 failed, 1 passed in 15.65s` | `4 passed in 15.91s` |
| P09 freshwater/MPAS scan | `7 failed, 44 passed in 49.02s` | `51 passed in 67.19s` |
| P10 GM capture | `1 failed, 42 passed in 83.06s` | `43 passed in 104.04s` |
| P11 implicit-face capture | `3 failed, 1 passed in 6.59s` | `4 passed in 30.49s` |
| P12 K-zeta anchor | `1 failed, 9 passed in 1.58s` | `10 passed in 1.64s` |
| P13 mass-flux schema | `1 failed, 79 passed, 3 skipped in 155.53s` | `80 passed, 3 skipped in 158.10s` |
| P14 MPAS TKE | `14 failed, 22 passed, 9 warnings in 24.97s` | `36 passed, 6 warnings in 43.96s` |
| P15 QCO operands | `1 failed, 4 passed in 11.08s` | `5 passed in 11.33s` |
| P16 SCO step PGF | `5 failed, 3 passed in 10.78s` | `1 failed, 7 passed in 22.45s` |
| P17 Wicker package | `1 failed, 3 passed in 3.77s` | `4 passed in 4.00s` |
| P18 FESOM optional dependency | `20 passed, 67 skipped, 11 errors in 0.96s` | `20 passed, 78 skipped in 0.49s` |
| P19 overflow parity | `1 failed, 7 passed, 9 warnings in 161.07s` | target: `1 passed, 5 warnings in 114.33s` |
| P20 PGF seamount | `4 failed, 1 passed in 6.57s` | `5 passed in 34.28s` |
| P21 PGF tier envelopes | `3 failed, 20 passed, 2 deselected in 332.85s` | targets: `3 passed in 49.40s` |
| P22 prescribed-flow schema | `1 failed, 15 passed in 14.05s` | `16 passed in 14.55s` |
| P23 SCM twins | `11 failed, 34 passed in 2.81s` | `45 passed in 19.04s` |
| P24 generated wiring | `1 failed, 11 passed in 0.37s` | `12 passed in 0.24s` |
| P25 TKE N2 routing | `1 failed, 4 passed in 5.27s` | `5 passed in 6.12s` |
| P26 Veros basic probe | `1 failed, 10 passed in 28.88s` | `11 passed in 32.09s` |
| P27 Veros recipe probe | `2 failed, 28 passed in 17.85s` | `30 passed in 20.22s` |
| P28 literal-TKE probe timestep | `2 failed, 17 deselected in 14.74s` | `2 passed, 17 deselected in 16.42s` |
| P29 MPAS partial-cell surface mask | `1 failed in 9.68s` | `1 passed in 10.44s` |
| P30 matrix PE scoring | MPAS LOCK_EXCHANGE: `5.290259116013614e-05`, FAIL; lat-lon OVERFLOW: `-8.37695888705242e-08`, PASS | MPAS LOCK_EXCHANGE: `-1.4936222653221853e-08`, PASS; lat-lon OVERFLOW: `-2.0206019137654794e-08`, PASS |

The required second-pass, whole-file isolated reruns at the final code state
reported:

| Touched test file | Final isolated summary |
|---|---|
| `test_dino_basin_seasonal_decomp.py` | `6 passed, 1 skipped in 1.33s` |
| `test_pgf_tiers.py` | `23 passed, 2 deselected in 398.79s (0:06:38)` |
| `test_nemo_sco_step_pgf.py` | `1 failed, 7 passed in 19.89s` (A4 held) |
| `test_advection_grad_underflow.py` | `4 failed, 7 passed in 130.22s (0:02:10)` (A5 held) |
| `test_tendency_probe.py` | `19 passed in 23.11s` |
| `test_mpas_tke.py` | `37 passed, 6 warnings in 65.30s (0:01:05)` |

`tests/ocean/unit/test_scm_column_twins.py` exits normally in isolation; its
base line is `11 failed, 34 passed in 2.81s`.  Its previously reported exit 134
is therefore compiler/resource pressure when combined with many JAX-heavy
files, not an abort performed by the test.

## Per-ID disposition

| ID | Class | Cause commit or reason | Fix commit | Proof line |
|---|---|---|---|---|
| `tests/ocean/fidelity/test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree` | TEST-INFRASTRUCTURE | Nine report emitters omitted the required provenance stamp. | `6cb7419da` | P01 |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[dst3]` | REAL DEFECT | **HELD:** direct rollout JVP still raises `TypeError` at the `custom_vjp` Thomas solver.  The restored gate records a finite reverse directional derivative of 3.128107381e+01, but no forward derivative is available for the required agreement check. | `7c8421129` | P02 |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[ppm_fct]` | REAL DEFECT | **HELD:** direct rollout JVP still raises `TypeError` at the `custom_vjp` Thomas solver.  The restored gate records a finite reverse directional derivative of 3.088927824e+01, but no forward derivative is available for the required agreement check. | `7c8421129` | P02 |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[superbee]` | REAL DEFECT | **HELD:** direct rollout JVP still raises `TypeError` at the `custom_vjp` Thomas solver.  The restored gate records a finite reverse directional derivative of 3.128046442e+01, but no forward derivative is available for the required agreement check. | `7c8421129` | P02 |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[tvd]` | REAL DEFECT | **HELD:** direct rollout JVP still raises `TypeError` at the `custom_vjp` Thomas solver.  The restored gate records a finite reverse directional derivative of 3.128023910e+01, but no forward derivative is available for the required agreement check. | `7c8421129` | P02 |
| `tests/ocean/unit/test_baroclinic_decomposition.py::test_baroclinic_decomposition_bit_identical` | STALE EXPECTATION | `9caa61f3e` intentionally changed the faithful DINO decomposition; the 81-array frozen oracle was regenerated with provenance. | `6cb7419da` | P03 |
| `tests/ocean/unit/test_barotropic_accuracy.py::test_the_frozen_tide_still_uses_loop_start_sampling` | STALE EXPECTATION | `385d2410a` changed the tide to per-substep sampling; the test now asserts the landed clock rather than the removed loop-start behavior. | `6cb7419da` | P04 |
| `tests/ocean/unit/test_dino_basin_seasonal_decomp.py::test_probe_self_checks_pass` | ENVIRONMENT | Required campaign members and tiled NEMO restart artifacts are absent on this machine.  The skip now uses the probe's real member paths and `RUN_VERDICT360_M%d/DINO_<kt>_restart_*.nc` resolver; a temporary-artifact control proves the self-check runs when those resolved inputs exist. | `8ef8d964d` | P05 |
| `tests/ocean/unit/test_dino_vertex_area_nemo.py::TestConstructionMatchesNemo::test_area_at_hand_quoted_nemo_points` | STALE EXPECTATION | `aa010f143` landed NEMO's true 199x52 DINO grid; quoted indices move by two columns. | `6cb7419da` | P06 |
| `tests/ocean/unit/test_dino_vertex_area_nemo.py::TestConstructionMatchesNemo::test_gap_is_the_midpoint_rule_not_something_else` | STALE EXPECTATION | `aa010f143`; analytic fixture and midpoint expectation now use the faithful grid. | `6cb7419da` | P06 |
| `tests/ocean/unit/test_dino_vertex_area_nemo.py::TestConstructionMatchesNemo::test_wall_rows_keep_the_exact_cap` | STALE EXPECTATION | `aa010f143`; wall-row fixture now has the faithful 199-column shape. | `6cb7419da` | P06 |
| `tests/ocean/unit/test_dino_vertex_area_nemo.py::TestConstructionMatchesNemo::test_whole_interior_not_only_the_quoted_rows` | STALE EXPECTATION | `aa010f143`; full-interior oracle now matches the faithful grid. | `6cb7419da` | P06 |
| `tests/ocean/unit/test_dino_vertex_area_nemo.py::TestPairAnalysisPins::test_the_two_halves_have_the_predicted_analytic_forms` | STALE EXPECTATION | `aa010f143`; pair-analysis medians and shapes were pinned to the obsolete grid. | `6cb7419da` | P06 |
| `tests/ocean/unit/test_dino_vface_zonal_width_nemo.py::TestConstructionMatchesNemo::test_exact_convention_still_carries_the_gap` | STALE EXPECTATION | `aa010f143`; v-face oracle now uses NEMO's faithful 199x52 geometry. | `6cb7419da` | P07 |
| `tests/ocean/unit/test_dino_vface_zonal_width_nemo.py::TestConstructionMatchesNemo::test_vface_latitudes_are_nemos_gphiv` | STALE EXPECTATION | `aa010f143`; latitude fixture shape/indexing was obsolete. | `6cb7419da` | P07 |
| `tests/ocean/unit/test_dino_vface_zonal_width_nemo.py::TestConstructionMatchesNemo::test_whole_interior_not_only_the_quoted_rows` | STALE EXPECTATION | `aa010f143`; full-interior oracle now matches the faithful grid. | `6cb7419da` | P07 |
| `tests/ocean/unit/test_dino_vface_zonal_width_nemo.py::TestConstructionMatchesNemo::test_width_matches_nemo_e1v_at_the_walls` | STALE EXPECTATION | `aa010f143`; wall-width fixture now matches the faithful grid. | `6cb7419da` | P07 |
| `tests/ocean/unit/test_eke_regression.py::test_eke_rhines_kiso_step_regression_bit_identical` | STALE EXPECTATION | `66ad4bf7f` corrected the barotropic averaging window; the frozen EKE output was regenerated. | `6cb7419da` | P08 |
| `tests/ocean/unit/test_eke_regression.py::test_eke_rhines_step_regression_bit_identical` | STALE EXPECTATION | `66ad4bf7f`; regenerated frozen output. | `6cb7419da` | P08 |
| `tests/ocean/unit/test_eke_regression.py::test_eke_step_regression_bit_identical` | STALE EXPECTATION | `66ad4bf7f`; regenerated frozen output. | `6cb7419da` | P08 |
| `tests/ocean/unit/test_freshwater.py::TestOceanModelWithFreshwater::test_freshwater_none_closure_ignores_forcing` | REAL DEFECT | MPAS split precision seeded a `lax.scan` barotropic carry from f64 `u_bar` although the body returns eta-precision f32 transport.  The carry is now initialized in `eta.dtype`. | `6cb7419da` | P09 |
| `tests/ocean/unit/test_freshwater.py::TestOceanModelWithFreshwater::test_multi_step_stability` | REAL DEFECT | Same MPAS scan-carry dtype defect. | `6cb7419da` | P09 |
| `tests/ocean/unit/test_freshwater.py::TestOceanModelWithFreshwater::test_step_with_evap_increases_S` | REAL DEFECT | Same MPAS scan-carry dtype defect. | `6cb7419da` | P09 |
| `tests/ocean/unit/test_freshwater.py::TestOceanModelWithFreshwater::test_step_with_precip_decreases_S` | REAL DEFECT | Same MPAS scan-carry dtype defect. | `6cb7419da` | P09 |
| `tests/ocean/unit/test_gm_resolution_function.py::TestEKEBudgetCoupling::test_signed_iso_sink_gets_raw_kappa_split_call` | TEST-INFRASTRUCTURE | A monkeypatch spy cannot observe calls through the already-jitted wrapper.  The capture test now invokes the same production `_step_impl` eagerly. | `6cb7419da` | P10 |
| `tests/ocean/unit/test_implicit_vmix_face_control_volume.py::test_no_face_carries_water_on_a_level_neither_column_has` | TEST-INFRASTRUCTURE | Same already-jitted-wrapper spy problem; capture uses the production implementation directly. | `6cb7419da` | P11 |
| `tests/ocean/unit/test_implicit_vmix_face_control_volume.py::test_u_face_column_depth_is_the_nemo_value` | TEST-INFRASTRUCTURE | Same already-jitted-wrapper spy problem. | `6cb7419da` | P11 |
| `tests/ocean/unit/test_implicit_vmix_face_control_volume.py::test_v_face_column_depth_is_the_nemo_value` | TEST-INFRASTRUCTURE | Same already-jitted-wrapper spy problem. | `6cb7419da` | P11 |
| `tests/ocean/unit/test_k_zeta_bih_resolution_scaling.py::test_anchor_mesh_is_bit_identical_to_the_tuned_value` | REAL DEFECT | A float32 mesh-spacing mean was used to derive an allegedly exact anchor.  Accumulation and the reference value now use the float64 mesh mean.  This intentionally changes the `K_zeta_bih=None` opt-in path as recorded under B7. | `6cb7419da` | P12 |
| `tests/ocean/unit/test_mass_flux_store.py::test_store_mass_flux_is_the_last_config_field` | STALE EXPECTATION | `d80df3ed7` intentionally appended a config field; the positional schema pin now names the current tail. | `6cb7419da` | P13 |
| `tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_eice0_no_ice_is_fine_and_default_card_unchanged` | REAL DEFECT | `21e85d252` made `surface_tmask` part of faithful NEMO TKE; the MPAS adapter did not forward its reconstructed surface wet mask. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_eice3_quarter_ice_maps_to_full_attenuation` | REAL DEFECT | Same missing MPAS `surface_tmask` bridge. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_eice_full_ice_attenuates_vs_eice0` | REAL DEFECT | Same missing MPAS `surface_tmask` bridge. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_kernel_receives_degrees_and_e3t_inputs` | REAL DEFECT | Same missing MPAS `surface_tmask` bridge. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_orca1_card_runs_on_mpas` | REAL DEFECT | Same missing MPAS `surface_tmask` bridge. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestNemoSurfaceTermsOnMPAS::test_partial_cell_zeroes_subseafloor_interfaces` | REAL DEFECT | Same missing MPAS `surface_tmask` bridge. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticCarryHardening::test_carry_land_masking_forced_land` | REAL DEFECT | Same missing MPAS `surface_tmask` bridge; the tests also pin the required fp64 policy explicitly. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticCarryHardening::test_carry_partial_cell_subseafloor_zeroed` | REAL DEFECT | Same MPAS TKE bridge defect. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticCarryHardening::test_scan_carry_stable_treedef_dtype_shape` | REAL DEFECT | Same MPAS TKE bridge defect. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticCarryHardening::test_split_policy_tke_dtype_pinned` | REAL DEFECT | Same MPAS TKE bridge defect; test policy is now explicit. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticTKECarryOnMPAS::test_carry_evolves_and_feeds_back` | REAL DEFECT | Same MPAS TKE bridge defect. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticTKECarryOnMPAS::test_carry_masked_on_land` | REAL DEFECT | Same MPAS TKE bridge defect. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticTKECarryOnMPAS::test_diagnostic_mode_unchanged_two_tuple` | REAL DEFECT | Same MPAS TKE bridge defect. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_mpas_tke.py::TestPrognosticTKECarryOnMPAS::test_model_seed_and_step_carry` | REAL DEFECT | Same MPAS TKE bridge defect. | `6cb7419da` | P14 |
| `tests/ocean/unit/test_nemo_qco_generic_mesh_operands.py::test_arm_is_constructible_on_the_certified_l1_cards` | STALE EXPECTATION | `03f0a1a07` intentionally made partial raw-QCO operand bundles fail closed; the test now supplies all raw operands or asserts the partial-bundle error. | `6cb7419da` | P15 |
| `tests/ocean/unit/test_nemo_sco_step_pgf.py::test_eta_zero_reduces_to_adcroft_bitwise` | STALE EXPECTATION | `7f729c488` requires raw W-grid geometry and `0da38492e` landed the dimensionless recurrence; fixture/expected recurrence were stale. | `6cb7419da` | P16 |
| `tests/ocean/unit/test_nemo_sco_step_pgf.py::test_f90_recurrence_oracle_nonuniform_rho` | REAL DEFECT | **HELD:** the recurrence expectation was updated for `7f729c488` / `0da38492e`, but the original `atol=1e-19` oracle now measures maximum absolute PGF discrepancies of 2.34866651e-15 (u) and 1.46611239e-15 m/s² (v) on CPU/fp64.  The 30,000× tolerance widening is removed; the numerical discrepancy remains visible. | `ca5f2ec86` | P16 |
| `tests/ocean/unit/test_nemo_sco_step_pgf.py::test_staircase_rest_stays_at_rest` | STALE EXPECTATION | Same `7f729c488` / `0da38492e` faithful-operand change. | `6cb7419da` | P16 |
| `tests/ocean/unit/test_nemo_sco_step_pgf.py::test_two_column_step_form_stress_x` | STALE EXPECTATION | Same `7f729c488` / `0da38492e` faithful-operand change. | `6cb7419da` | P16 |
| `tests/ocean/unit/test_nemo_sco_step_pgf.py::test_two_column_step_form_stress_y` | STALE EXPECTATION | Same `7f729c488` / `0da38492e` faithful-operand change. | `6cb7419da` | P16 |
| `tests/ocean/unit/test_nemo_wicker_aimp_package.py::test_literal_tracer_matrix_fuses_diffusion_and_implicit_transport` | STALE EXPECTATION | `101d3383f` intentionally split the multiply and subtraction to reproduce gfortran rounding; the literal oracle moved by one binary64 ULP. | `6cb7419da` | P17 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9FacadeBasics::test_step_updates_uv_node` | ENVIRONMENT | Optional `fesom_jax` package is not installed in this CPU test environment; module fixture uses `importorskip(..., reason=...)`. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9FacadeBasics::test_T_field_shape_drops_padding` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9FacadeBasics::test_uv_node_is_zero_at_rest` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9FacadeBasics::test_uv_node_shape` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9NlevelsNod2DMin::test_nlevels_nod2D_min_equals_fesom_formula` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9NlevelsNod2DMin::test_wet_nodes_incident_to_dry_have_min_one` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9PytreeDesign::test_jit_compiles_once_across_two_states` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9PytreeDesign::test_two_value_identical_states_share_treedef` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9PytreeDesign::test_u_and_v_share_one_uv_node` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9RunIsEager::test_run_matches_eager_loop_and_threads_flag` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_ocean_model_fesom.py::TestR9RunIsEager::test_run_rejects_zero_steps` | ENVIRONMENT | Same absent optional `fesom_jax` dependency. | `6cb7419da` | P18 |
| `tests/ocean/unit/test_overflow_runner_parity.py::TestModularOverflowStable::test_modular_matches_monolithic_pe_rel` | TEST-INFRASTRUCTURE | Modular and monolithic runners had drifted duplicate PE/RPE diagnostic implementations.  Both now call one shared wet-cell, moving-volume helper. | `6cb7419da` | P19 |
| `tests/ocean/unit/test_pgf_seamount_at_rest.py::test_smc03_partial_cell_rest_balance` | STALE EXPECTATION | `b8552523e` requires the seeded slow/barotropic forcing carry; the fixture now enters through `model.seed_scan_carry`. | `6cb7419da` | P20 |
| `tests/ocean/unit/test_pgf_seamount_at_rest.py::test_smc03_zstar_meridional_slope_balance` | STALE EXPECTATION | Same `b8552523e` carry contract. | `6cb7419da` | P20 |
| `tests/ocean/unit/test_pgf_seamount_at_rest.py::test_smc03_zstar_rest_balance` | STALE EXPECTATION | Same `b8552523e` carry contract. | `6cb7419da` | P20 |
| `tests/ocean/unit/test_pgf_seamount_at_rest.py::test_zstar_uncorrected_pgf_is_large_selftest` | STALE EXPECTATION | Same `b8552523e` carry contract. | `6cb7419da` | P20 |
| `tests/ocean/unit/test_pgf_tiers.py::TestTier4::test_tier4_seamount_strong_wind[adcroft]` | STALE EXPECTATION | `9caa61f3e` corrected the AL81 triad/flux pairing.  The corrected CPU/fp64 value is pinned at 25.217152071275 m/s with `rtol=1e-4` (0.01%); the rejected 30 m/s ceiling is gone. | `ae20871d6` | P21 |
| `tests/ocean/unit/test_pgf_tiers.py::TestTier4::test_tier4_seamount_strong_wind[smc03]` | STALE EXPECTATION | Same `9caa61f3e` change; the measured 21.796509983933 m/s value is pinned at `rtol=1e-4`. | `ae20871d6` | P21 |
| `tests/ocean/unit/test_pgf_tiers.py::TestTier4::test_tier4_wind_flat_bottom_reference` | STALE EXPECTATION | Same `9caa61f3e` change; the measured 2.467487967708 m/s value is pinned at `rtol=1e-4` instead of widening the former 2 m/s gate to 3 m/s. | `ae20871d6` | P21 |
| `tests/ocean/unit/test_prescribed_flow.py::test_positional_construction_unshifted_by_tail_field` | STALE EXPECTATION | `9caa61f3e` and later `d80df3ed7` extended/nested the config tail; positional schema expectation was obsolete. | `6cb7419da` | P22 |
| `tests/ocean/unit/test_scm_column_twins.py::test_build_jitted_step_rejects_bad_tier_sw_mode_and_channels` | REAL DEFECT | `a90bd1935` nested `rho_0` under `LatLonCGridOceanConfig._field_defaults["constants"]`, but the shared SCM harness still read `d["rho_0"]` and raised `KeyError`. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_build_scm_rejects_unknown_tier` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_extract_point_forcing_series_channels_and_values` | STALE EXPECTATION | `a90bd1935` changed nearest-cell forcing to bilinear interpolation; the expected forcing values/channels were stale. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_forcing_closures_finite_over_run_window` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_jitted_step_matches_closure_reference[T0-top]` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_jitted_step_matches_closure_reference[T1-penetrate]` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_jitted_step_salt_is_3d_virtual_salt_closure` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_run_point_tier_daily_history_and_meta_shapes` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_run_point_tier_forcing_and_coriolis_at_selected_cell` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_t0_has_no_coriolis_no_wind_and_uv_stay_exactly_zero` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_scm_column_twins.py::test_t1_has_coriolis_and_wind_produces_motion` | REAL DEFECT | Same stale shared-harness `rho_0` lookup. | `6cb7419da` | P23 |
| `tests/ocean/unit/test_step_wiring_generated.py::test_committed_doc_matches_regeneration` | TEST-INFRASTRUCTURE | Generated wiring inventory drifted from its generator; regenerated result is 148 calls, 51 live, 58 dead, 39 unresolved. | `6cb7419da` | P24 |
| `tests/ocean/unit/test_tke_n2_before_advection.py::test_set_diffusivities_routes_exactly_to_n2_source` | STALE EXPECTATION | `21e85d252` added the faithful `surface_tmask` operand; the fixture now supplies it. | `6cb7419da` | P25 |
| `tests/ocean/unit/test_veros_acc_basic_recipe.py::test_frozen_state_probe_compatible` | REAL DEFECT | The `dt_tke` requirement introduced with the newer TKE contract (traceable to `080ce674a`) was not forwarded by the offline tendency probe.  Factored TKE continues to receive momentum `dt`; literal NEMO TKE now uses the dedicated channel documented under B3. | `6cb7419da` | P26 |
| `tests/ocean/unit/test_veros_acc_recipe.py::test_compare_momentum_emits_all_processes` | REAL DEFECT | Same stale tendency-probe caller; factored TKE now receives the probe timestep as `dt_tke`. | `6cb7419da` | P27 |
| `tests/ocean/unit/test_veros_acc_recipe.py::test_end_to_end_recipe_probe_round_trip` | REAL DEFECT | Same stale tendency-probe caller. | `6cb7419da` | P27 |

The table contains exactly 87 IDs: 31 stale expectations, 37 real defects, 7
test-infrastructure defects, and 12 environment skips.

## Second-pass review dispositions

| Finding | Disposition and evidence |
|---|---|
| B3 | **FIXED.** The probe now has a dedicated `tke_rn_dt` channel.  For `scheme="tke"` plus `tke_matrix_evaluation="nemo_literal"`, it forwards that base timestep exactly as production resolves it at `ocean_model_latlon_cgrid.py:10018-10022`; all committed DINO tendency-probe callers pass the card's 2700 s `rn_Dt`.  A literal card without the channel raises a named `ValueError` instead of silently substituting momentum `dt`.  P28 proves both refusal and forwarding. |
| B2 | **FIXED.** The MPAS TKE bridge now mirrors the lat-lon mask-owner rule: it prefers `z_coord.is_active[..., 0]` and falls back to the reconstructed 2-D column mask.  P29 plants a surface-inactive partial cell whose column mask remains wet, so the old proxy fails and the preferred operand passes. |
| B7 | **DOCUMENTED SHIPPED BEHAVIOUR CHANGE.** The float64 ico6 anchor changes `K_zeta_bih_ref_dx_m` from 120194.609375 m to 120194.60581296285 m.  At a fixed non-anchor mesh spacing, the cubed scaling therefore multiplies a derived (`K_zeta_bih=None`) coefficient by 1.00000008890675, a +8.890675e-8 relative shift (0.08890675 ppm); the runtime mean is now accumulated in float64 as well, so its last-bit contribution is mesh-specific.  The corrected ico6 anchor remains exactly 1.0e14 m⁴/s.  The default `MPASOceanConfig` pins `K_zeta_bih=0.0`, so this opt-in change does not affect it.  P12 is the fail/pass control for the exact anchor. |
| B4 | **MEASURED SCORING CHANGE.** The committed `matrix_pe_gate_rescore.py` probe ran the matrix's registered quick cases on CPU/fp64, swapping only the scorer and proving all saved physical snapshot arrays bit-identical (8 LOCK_EXCHANGE keys and 11 OVERFLOW keys).  The legacy scorer is the pre-`6cb7419da` unmasked, reference-thickness `nansum`; the shared scorer uses the wet mask and live thickness.  P30 gives the exact before/after gate values.  MPAS LOCK_EXCHANGE flips FAIL→PASS; lat-lon OVERFLOW remains PASS. |

DECISION_NEEDED: B4 changes the registered MPAS LOCK_EXCHANGE acceptance verdict from FAIL to PASS (`PE_rel_final` +5.290259116013614e-05 → -1.4936222653221853e-08) without changing its trajectory.  The user must decide whether to accept the new wet-mask/live-thickness gate definition.

## Whole-tree runs

Both required commands used the prescribed CPU-only environment and
`pytest -n 12`.  The fidelity command completed; the unit command did not
produce a pytest summary because its JAX compiler workers repeatedly aborted
and the coordinator stopped making progress.

### Fidelity tree

```text
3 failed, 1324 passed, 8 skipped in 2773.57s (0:46:13)
```

Every remaining fidelity failure is in the protected citation-map lane, which
this task explicitly forbids changing.  The map audit reports eight stale
anchors (six generated NEMO spans and two round-46 Python spans).

| Remaining fidelity ID | Reason |
|---|---|
| `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py::test_every_map_entry_is_shift_sensitive` | Eight protected citation-map entries are no longer anchored at their recorded lines. |
| `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py::test_the_audit_itself_can_fail` | Its planted-entry expectation is preceded by those same eight existing map-audit failures. |
| `tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py::test_the_gate_runs_clean_on_the_real_receipt` | The real-receipt gate correctly fails because `map_entries_failing_audit` contains those eight protected entries. |

### Unit tree

The required command collected 6,823 tests and reached 96%.  Multiple xdist
workers aborted in JAX's CPU `backend_compile_and_load`; after the coordinator
emitted no output for more than 20 minutes, it was interrupted and exited 130.
Therefore **pytest emitted no unit-tree summary line**.  The worker-abort
locations captured before the stall were:

| Worker-abort location | Classification |
|---|---|
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[...]` | JAX compiler-process abort under multi-file pressure; the four assertion outcomes are separately pinned and held below. |
| `tests/ocean/unit/test_ocean.py::test_longrun_stability_without_fixer` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_polar_filter_ocean.py::test_step_gating_off_vs_on` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_nemo_recipe.py::test_nemo_iso_lap_card_one_step_is_finite` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_momentum_friction_additive.py::test_tracers_bit_identical_across_modes` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_ocean_scm.py::test_coriolis_rotation_conserves_kinetic_energy` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_vertical_momentum_scheme.py::test_default_off_bit_identical` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_nemo_ws_stage1_transport.py::test_stage1_transport_is_inert_from_rest` | JAX compiler-process abort; no pytest assertion outcome. |
| `tests/ocean/unit/test_partial_cells_phase7.py::test_partial_cells_implicit_mixing_conserves_heat` | JAX compiler-process abort; no pytest assertion outcome. |

Bounded four-worker recovery chunks showed the same load sensitivity.  Every
chunk-only assertion was rerun in a fresh process.  The FCT file reported
`26 passed in 26.06s`, baroclinic decomposition reported
`2 passed in 21.57s`, the regional-audit file reported
`45 passed, 1 skipped in 1.45s`, the EKE differentiability target reported
`1 passed in 32.11s`, and all eight targets exposed by the mass-flux,
friction, surface-buoyancy, and momentum-diagnostics chunk reported one pass
each.  The five required held assertions and four unrelated assertions
remained reproducibly red in fresh processes; neither the unrelated tests nor
their failing production path changed in this second pass:

| Remaining unit ID | Isolated result and reason |
|---|---|
| `tests/ocean/unit/test_nemo_sco_step_pgf.py::test_f90_recurrence_oracle_nonuniform_rho` | A4 HELD: `1 failed, 7 passed in 19.89s`; max absolute PGF errors are u=2.34866651e-15 and v=1.46611239e-15 m/s² against the restored `atol=1e-19`. |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[ppm_fct]` | A5 HELD: direct JVP reaches the `custom_vjp` Thomas solver and raises `TypeError`; reverse directional derivative is 3.088927824e+01. |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[tvd]` | A5 HELD: same direct-JVP defect; reverse directional derivative is 3.128023910e+01. |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[superbee]` | A5 HELD: same direct-JVP defect; reverse directional derivative is 3.128046442e+01. |
| `tests/ocean/unit/test_advection_grad_underflow.py::test_model_rollout_grads_finite_f32[dst3]` | A5 HELD: same direct-JVP defect; reverse directional derivative is 3.128107381e+01.  The file summary is `4 failed, 7 passed in 130.22s (0:02:10)`. |
| `tests/ocean/unit/test_freesurface_helmholtz_adjoint_mpas.py::test_production_step_grad_finite_f32_and_scan` | Out-of-scope existing debt: `1 failed in 4.42s`; the CG `while_loop` enters with a float32 carry and returns float64. |
| `tests/ocean/unit/test_nemo_match_recipe.py::test_mpas_factory_builds_valid_model_and_one_step_is_finite` | Same existing MPAS Helmholtz CG dtype defect: `1 failed, 1 warning in 2.67s`. |
| `tests/ocean/unit/test_partial_cells_phase1.py::TestFlatBottomBitExact::test_flat_bottom_nonzero_eta` | Out-of-scope existing exact-equality debt: `1 failed in 2.37s`; maximum absolute difference 1.9378662e-05 (relative 6.12422374e-08). |
| `tests/ocean/unit/test_partial_cells_phase2.py::TestFlatBottomBitExact::test_nonzero_eta_flat_bottom` | Out-of-scope existing exact-equality debt: `1 failed in 2.81s`; maximum absolute difference 1.30672296 (relative 4.40189547e-08). |

The recovery evidence is supplementary and does not invent a summary for the
interrupted required unit command.

## Review

Second-pass dispositions are complete.  A4 and A5 remain deliberately red;
B4 requires the scoring-definition decision recorded above.

## Amendment — Decision 44 (user, 2026-09-19)

The one shared matrix energy scorer (`scripts/matrix/ocean_test_matrix/energy_diagnostics.py`, monolithic semantics:
land-masked, live layer thickness) is adopted for both runners.  Consequence accepted by the user: the registered
MPAS LOCK_EXCHANGE verdict flips FAIL (5.290259116013614e-05 under the retired modular scorer, which summed dry
cells with reference thicknesses) to PASS (-1.4936222653221853e-08); the lat-lon OVERFLOW verdict stays PASS
(-8.37695888705242e-08 -> -2.0206019137654794e-08).  Every saved physical snapshot is byte-identical between the two
scorings, so this is a change of what is scored, not of the model.  Follow-up: record on the ocean case board that the
earlier FAIL was never a conservation failure.
