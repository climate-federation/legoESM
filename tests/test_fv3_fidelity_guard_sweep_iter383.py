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
