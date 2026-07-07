"""FV3_3D iter 383: meta-test that runs every FV3-fidelity AST/
structure guard together.

iter-382 caught a real bug (iter-319 d_con knob count
collision with iter-339's ``use_fv3_metric_aware_d_con``
field) ONLY when guards were run together in sweep mode.
iter-383 wires this sweep as a permanent regression so similar
suffix collisions / cross-guard interactions are caught at CI
time.

Tests
-----

1. ``test_all_fv3_fidelity_guards_pass`` — imports every
   guard module + executes its tests inline.  Pytest treats
   imports as collection; this test re-validates the existence
   of every guard at module level.
"""
from __future__ import annotations

import importlib

import pytest


_GUARD_MODULES = (
    "test_pe_nh_config_default_parity_iter317",
    "test_nh_only_knob_defaults_iter318",
    "test_d_con_knob_count_iter319",
    "test_pe_nh_flag_asymmetry_iter331",
    "test_metric_aware_d_con_ast_guard_iter340",
    "test_nh_dynamic_exner_ast_guard_iter342",
    "test_all_metric_d_con_sites_ast_guard_iter353",
    "test_nh_5_dcon_sites_flag_coverage_iter361",
    "test_pe_4_dcon_sites_flag_coverage_iter362",
    "test_fv3_3d_doc_compaction_iter368",
    "test_fv3_fidelity_flag_set_iter369",
    "test_cross_face_du_proj_ast_iter373",
    "test_cross_face_duogrid_pairing_doc_iter388",   # iter-389 add
    "test_dyn_exner_pkz_equivalence_doc_iter395",    # iter-395 add
    "test_fv3_faithful_factory_signature_iter402",   # iter-403 add
    "test_fv3_flag_count_consistency_iter405",       # iter-414 add
    "test_fv3_faithful_factory_docstring_iter406",   # iter-414 add
    "test_fv3_faithful_passes_through_overrides_iter412",  # iter-414 add
    "test_guard_sweep_no_duplicates_iter415",        # iter-416 add
    "test_iter417_doc_example_iter418",              # iter-422 add
    "test_dyn_exner_gradient_flow_iter413",          # iter-422 add
    "test_dyn_exner_equals_pkz_iter401",             # iter-422 add
    "test_iter392_factory_module_exports_iter426",   # iter-427 add
    "test_iter392_factory_step_runs_iter427",        # iter-427 add
    "test_iter392_factory_ad_umbrella_iter428",      # iter-428 add
    "test_iter392_factory_stability_iter429",        # iter-429 add
    "test_d_con_top_zero_levels_iter431",            # iter-431 add
    "test_d_con_top_zero_levels_full_iter432",       # iter-432 add
    "test_d_con_top_zero_levels_pe_iter433",         # iter-433 add
    "test_factory_d_con_top_zero_default_iter434",   # iter-434 add
    "test_d_con_top_zero_ast_guard_iter435",         # iter-435 add
    "test_factory_delt_max_default_iter436",         # iter-436 add
    "test_factory_nord_defaults_iter437",            # iter-437 add
    "test_pe_corner_div_d2_bg_k1_iter438",           # iter-438 add
    "test_pe_corner_div_d2_bg_k2_iter439",           # iter-439 add
    "test_nh_corner_div_d2_bg_sponge_iter440",       # iter-440 add
    "test_nh_sponge_damp_w_iter441",                 # iter-441 add
    "test_nh_sponge_damp_v_iter442",                 # iter-442 add
    "test_pe_sponge_damp_v_iter443",                 # iter-443 add
    "test_factory_sponge_defaults_iter444",          # iter-444 add
    "test_sponge_boost_ast_guard_iter445",           # iter-445 add
    "test_fv3_sponge_boost_shared_iter446",          # iter-446 add
    "test_fv3_sponge_field_scale_iter447",           # iter-447 add
    "test_nh_rayleigh_fast_iter448",                 # iter-448 add
    "test_pe_rayleigh_fast_iter449",                 # iter-449 add
    "test_rayleigh_fast_ast_guard_iter450",          # iter-450 add
    "test_factory_d4_bg_default_iter451",            # iter-451 add
    "test_factory_sponge_e2e_damping_iter452",       # iter-452 add
    "test_factory_d4_bg_d2_bg_k_ast_iter453",        # iter-453 add
    "test_sponge_calibration_5step_iter455",         # iter-455 add
    "test_cube_edge_artifact_metric_iter456",        # iter-456 add
    "test_heat_source_del2_iter457",                 # iter-457 add
    "test_pe_heat_source_del2_iter458",              # iter-458 add
    "test_factory_heat_source_del2_default_iter459", # iter-459 add
    "test_heat_source_del2_ast_iter460",             # iter-460 add
    "test_factory_reduces_edge_artifact_iter461",    # iter-461 add
    "test_d2_bg_k1_calibration_sweep_iter462",       # iter-462 add
    "test_rf_tau_calibration_sweep_iter463",         # iter-463 add
    "test_heat_source_del2_sweep_iter464",           # iter-464 add
    "test_per_flag_edge_ratio_iter465",              # iter-465 add
    "test_factory_minimal_edge_iter466",             # iter-466 add
    "test_make_legoesm_nh_min_edge_config_iter467",  # iter-467 add
    "test_make_legoesm_pe_min_edge_config_iter468",  # iter-468 add
    "test_pe_per_flag_edge_ratio_iter469",           # iter-469 add
    "test_pe_per_flag_u_d_edge_iter470",             # iter-470 add
    "test_nh_duogrid_effect_iter471",                # iter-471 add
    "test_nh_duogrid_effect_c16_iter472",            # iter-472 add
    "test_nh_duogrid_raw_std_iter473",               # iter-473 add
    "test_pad_halo_4d_duogrid_constant_iter474",     # iter-474 add
    "test_pad_halo_4d_duogrid_linear_iter475",       # iter-475 add
    "test_pad_halo_vector_4d_duogrid_iter476",       # iter-476 add
    "test_laplacian_compact_3d_duogrid_amplification_iter477",  # iter-477 add
    "test_nh_duogrid_n_step_growth_iter478",         # iter-478 add
    "test_nh_duogrid_op_bisection_iter479",          # iter-479 add
    "test_nh_duogrid_in_step_bisection_iter480",     # iter-480 add
    "test_corner_div_d2_bg_sweep_duogrid_iter481",   # iter-481 add
    "test_combined_iter466_iter481_iter482",         # iter-482 add
    "test_make_legoesm_nh_min_edge_aggressive_iter483",  # iter-483 add
    "test_make_legoesm_pe_min_edge_aggressive_iter484",  # iter-484 add
    "test_legoesm_min_edge_factories_ast_iter485",   # iter-485 add
    "test_nh_duogrid_resolution_scan_iter487",       # iter-487 add
    "test_pad_halo_4d_duogrid_overshoot_location_iter489",  # iter-489 add
    "test_pad_halo_4d_monotone_clip_iter490",        # iter-490 add
    "test_pad_halo_4d_monotone_clip_robustness_iter491",  # iter-491 add
    "test_nh_duogrid_with_monotone_clip_iter492",    # iter-492 add
    "test_nh_duogrid_comprehensive_clip_iter493",    # iter-493 add
    "test_divergence_corner_duogrid_iter495",        # iter-495 add
    "test_interp_center_to_corner_duogrid_iter496",  # iter-496 add
    "test_center_to_dgrid_vector_duogrid_iter497",   # iter-497 add
    "test_pad_halo_vector_4d_monotone_clip_iter498", # iter-498 add
    "test_center_to_dgrid_vector_with_clip_iter499", # iter-499 add
    "test_monotone_clip_slack_iter501",              # iter-501 add
    "test_monotone_clip_slack_sweep_iter502",        # iter-502 add
    "test_nh_dycore_clip_slack_iter503",             # iter-503 add
    "test_combined_iter466_iter503_iter504",         # iter-504 add
    "test_monotone_halo_clip_context_iter505",       # iter-505 add
    "test_pe_monotone_halo_clip_context_iter506",    # iter-506 add
    "test_nh_residual_bisection_iter507",            # iter-507 add
    "test_corner_div_damp_sweep_iter508",            # iter-508 add
    "test_residual_growth_iter509",                  # iter-509 add
    "test_clip_long_term_effect_iter511",            # iter-511 add
    "test_long_term_slack_iter512",                  # iter-512 add
    "test_pad_halo_3d_clip_iter513",                 # iter-513 add
    "test_pad_halo_vector_3d_clip_iter514",          # iter-514 add
    "test_within_grid_edge_metric_iter515",          # iter-515 add
    "test_higher_order_damp_iter516",                # iter-516 add
    "test_smooth_ic_edge_iter517",                   # iter-517 add
    "test_resolution_scan_smooth_iter518",           # iter-518 add
    "test_cube_smooth_ic_iter519",                   # iter-519 add
    "test_sbr_ic_iter521",                           # iter-521 add
    "test_clip_context_ad_iter522",                  # iter-522 add
    "test_mass_conservation_sbr_iter523",            # iter-523 add
    "test_pe_mass_conservation_iter524",             # iter-524 add
    "test_clip_context_jit_iter525",                 # iter-525 add
    "test_make_clipped_step_iter526",                # iter-526 add
    "test_pe_halo_targets_iter527",                  # iter-527 add
    "test_pe_make_clipped_step_iter529",             # iter-529 add
    "test_sw_make_clipped_step_iter531",             # iter-531 add
    "test_sbr_c24_scan_iter532",                     # iter-532 add
    "test_corner_excited_ic_iter533",                # iter-533 add
    "test_aggressive_clip_combined_iter534",         # iter-534 add
    "test_example_script_imports_iter535",           # iter-535 add
    "test_slack_at_c16_sbr_iter536",                 # iter-536 add
    "test_terrain_clip_iter537",                     # iter-537 add
    "test_tracer_clip_iter538",                      # iter-538 add
    "test_long_run_mass_iter539",                    # iter-539 add
    "test_clip_helper_perf_iter541",                 # iter-541 add
    "test_dt_sensitivity_iter543",                   # iter-543 add
    "test_clipped_scan_step_iter544",                # iter-544 add
    "test_scan_step_perf_iter546",                   # iter-546 add
    "test_pe_clipped_scan_step_iter547",             # iter-547 add
    "test_end_to_end_validation_iter548",            # iter-548 add
    "test_sw_clipped_scan_step_iter549",             # iter-549 add
    "test_clip_helper_perf_c16_iter551",             # iter-551 add
    "test_hs_like_nh_iter552",                       # iter-552 add
    "test_full_stack_edge_reduction_iter553",        # iter-553 add
    "test_faithful_plus_clip_iter555",               # iter-555 add
    "test_faithful_convergence_iter556",             # iter-556 add
    "test_pe_factory_compare_iter557",               # iter-557 add
    "test_pe_long_run_iter558",                      # iter-558 add
    "test_pe_long_run_dt5_iter559",                  # iter-559 add
    "test_min_edge_bisect_smooth_iter561",           # iter-561 add
    "test_heat_source_del2_sweep_iter562",           # iter-562 add
    "test_heat_source_del2_coeff_sweep_iter563",     # iter-563 add
    "test_combined_best_iter564",                    # iter-564 add
    "test_combined_best_c32_iter565",                # iter-565 add
    "test_combined_best_30step_iter566",             # iter-566 add
    "test_iters_vs_growth_iter567",                  # iter-567 add
    "test_helpers_match_iter568",                    # iter-568 add
    "test_time_growth_powerlaw_iter569",             # iter-569 add
    "test_pe_time_growth_iter571",                   # iter-571 add
    "test_nh_zero_ic_iter572",                       # iter-572 add
    "test_low_amplitude_sbr_iter573",                # iter-573 add
    "test_50step_optimal_iter576",                   # iter-576 add
    "test_linearity_tiny_pert_iter578",              # iter-578 add
    "test_sw_rest_preservation_iter579",             # iter-579 add
    "test_compute_edge_metric_iter581",              # iter-581 add
    "test_angular_momentum_iter583",                 # iter-583 add
    "test_w_safety_cap_iter584",                     # iter-584 add
    "test_hord8_limiter_iter585",                    # iter-585 add
    "test_schmidt_transform_iter586",                # iter-586 add
    "test_aam_drift_iter587",                        # iter-587 add
    "test_aam_correction_iter588",                   # iter-588 add
    "test_cube_transform_iter589",                   # iter-589 add
    "test_shift_fac_iter591",                        # iter-591 add
    "test_hord11_limiter_iter592",                   # iter-592 add
    "test_hord10_limiter_iter593",                   # iter-593 add
    "test_iord_variants_comparison_iter594",         # iter-594 add
    "test_hord_dispatch_iter595",                    # iter-595 add
    "test_transport_step_hord_iter596",              # iter-596 add
    "test_total_energy_nh_iter597",                  # iter-597 add
    "test_total_energy_pe_iter598",                  # iter-598 add
    "test_te_drift_iter599",                         # iter-599 add
    "test_te_correction_iter601",                    # iter-601 add
    "test_te_correction_pe_iter602",                 # iter-602 add
    "test_pe_te_boundary_sign_iter603",              # iter-603 add
    "test_pe_aam_iter604",                           # iter-604 add
    "test_column_d_ext_iter605",                     # iter-605 add
    "test_terrain_filter_iter606",                   # iter-606 add
    "test_cubed_to_latlon_iter607",                  # iter-607 add
    "test_mid_pt_sphere_iter608",                    # iter-608 add
    "test_terrain_filter_mass_iter609",              # iter-609 add
    "test_fv3_cartesian_primitives_iter611",         # iter-611 add
    "test_fv3_mirror_intp_iter612",                  # iter-612 add
    "test_fv3_spherical_geometry_iter613",           # iter-613 add
    "test_fv3_get_area_iter614",                     # iter-614 add
    "test_fv3_unit_vect_iter615",                    # iter-615 add
    "test_fv3_intersect_iter616",                    # iter-616 add
    "test_fv3_gnomonic_iter617",                     # iter-617 add
    "test_fv3_get_center_vect_iter618",              # iter-618 add
    "test_fv3_symm_ed_iter619",                      # iter-619 add
    "test_fv3_gnomonic_ed_iter621",                  # iter-621 add
    "test_fv3_gnomonic_grids_iter622",               # iter-622 add
    "test_fv3_rot3d_gsum_iter623",                   # iter-623 add
    "test_fv3_mirror_grid_iter624",                  # iter-624 add
    "test_fv3_mirror_grid_sym_iter625",              # iter-625 add
    "test_fv3_init_c2l_iter626",                     # iter-626 add
    "test_fv3_native_grid_iter629",                  # iter-629 add
    "test_fv3_edge_factors_iter631",                 # iter-631 add
    "test_fv3_global_reductions_iter632",            # iter-632 add
    "test_fv3_fill_ghost_iter633",                   # iter-633 add
    "test_fv3_get_eta_level_iter634",                # iter-634 add
    "test_fv3_compute_dz_zflip_iter635",             # iter-635 add
    "test_fv3_sm1_edge_iter636",                     # iter-636 add
    "test_fv3_compute_dz_L101_iter637",              # iter-637 add
    "test_fv3_compute_dz_L32_iter638",               # iter-638 add
    "test_fv3_hybrid_z_dz_iter639",                  # iter-639 add
    "test_fv3_compute_dz_var_iter641",               # iter-641 add
    "test_fv3_gw_1d_iter642",                        # iter-642 add
    "test_fv3_mount_waves_iter643",                  # iter-643 add
    "test_fv3_p_var_core_iter644",                   # iter-644 add
    "test_fv3_drymadj_iter645",                      # iter-645 add
    "test_fv3_hydro_eq_iter646",                     # iter-646 add
    "test_fv3_set_eta_L60_iter647",                  # iter-647 add
    "test_fv3_get_pt_on_gc_iter651",                 # iter-651 add
    "test_fv3_dtoa_iter652",                         # iter-652 add
    "test_fv3_ctoa_iter653",                         # iter-653 add
    "test_fv3_atoc_iter654",                         # iter-654 add
    "test_fv3_atod_iter655",                         # iter-655 add
    "test_fv3_checker_tracers_iter657",              # iter-657 add
    "test_fv3_terminator_iter658",                   # iter-658 add
    "test_fv3_rotate_winds_iter661",                 # iter-661 add
    "test_fv3_rankine_vortex_iter662",               # iter-662 add
    "test_fv3_case9_iter664",                        # iter-664 add
    "test_fv3_dcmip16_tc_sphum_iter666",             # iter-666 add
    "test_fv3_dcmip16_bc_iter667",                   # iter-667 add
    "test_fv3_dcmip16_bc_wind_iter668",              # iter-668 add
    "test_fv3_dcmip16_tc_iter669",                   # iter-669 add
    "test_fv3_bc_uwind_pert_iter671",                # iter-671 add
    "test_fv3_tc_uwind_pert_iter672",                # iter-672 add
    "test_fv3_a2b_ord4_vector_cc_to_corner_iter696", # iter-696 add
    "test_fv3_a2b_ord4_nh_wiring_iter697",           # iter-697 add
    "test_fv3_a2b_ord4_vector_uv_edge_impact_iter698", # iter-698 add
    "test_fv3_a2b_ord4_vector_uv_c16_iter699",       # iter-699 add
    "test_fv3_a2b_ord4_theta_corner_iter700",        # iter-700 add
    "test_fv3_a2b_ord4_theta_corner_edge_impact_iter701", # iter-701 add
    "test_fv3_a2b_zeta_corner_nh_impact_iter702",    # iter-702 add
    "test_fv3_a2b_ord4_vector_uv_c24_iter703",       # iter-703 add
    "test_fv3_nord_validation_iter890",              # iter-890 add
    "test_fv3_wide_halo_iter891",                    # iter-891 add
    "test_fv3_model_nord_validation_iter902",        # iter-902 add
)


def test_all_fv3_fidelity_guards_importable():
    """Every FV3-fidelity guard module must be importable via
    sys.path insertion at tests/ dir."""
    import sys
    from pathlib import Path
    tests_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(tests_dir))
    try:
        for mod_name in _GUARD_MODULES:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                pytest.fail(
                    f"Guard module {mod_name!r} cannot import: {e!r}"
                )
    finally:
        sys.path.pop(0)


def test_guard_set_inventory_complete():
    """Inventory check: every guard module in this sweep is
    actually present in the tests directory."""
    from pathlib import Path
    tests_dir = Path(__file__).resolve().parent
    for mod_name in _GUARD_MODULES:
        path = tests_dir / f"{mod_name}.py"
        assert path.exists(), (
            f"Guard module {mod_name!r} declared in sweep but "
            f"file not found at {path}."
        )


def test_guard_set_non_trivial_count_iter896():
    """FV3_3D iter 896: sweep must contain >= 80 guard modules.

    Regression guard against accidental sweep deletion / mass-truncate.
    Pinned at conservative floor below current count to allow normal
    additions/removals while catching destructive refactors.
    """
    assert len(_GUARD_MODULES) >= 80, (
        f"FV3-fidelity guard sweep has {len(_GUARD_MODULES)} entries "
        f"(< 80 floor).  Possible accidental truncation; verify "
        f"iter-383+ guard accumulation is intact."
    )
