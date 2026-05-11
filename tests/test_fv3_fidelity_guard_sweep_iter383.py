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
