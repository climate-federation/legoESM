4a64c4746 fix(micro): morrison-wiring round 3 — one tolerance everywhere + schema-version migration gate
ERROR tests/unit/test_run_manifest.py::test_no_digest_recorded_for_failed_run
ERROR tests/unit/test_run_manifest.py::test_record_and_read_state_digest - Fi...
ERROR tests/unit/test_run_manifest.py::test_record_state_digest_rejects_invalid_manifest
ERROR tests/unit/test_run_manifest.py::test_atomic_write_leaves_no_temp_file
ERROR tests/unit/test_run_manifest.py::test_manifest_for_a_shipped_config - F...
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_uses_resolved_normalized_config
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_records_dataset_provenance
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_records_input_config_not_setup_mutated
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_is_write_once - F...
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_rejects_reuse_with_different_config
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_failure_is_fatal
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_fails_closed_on_corrupt_existing
ERROR tests/unit/test_run_manifest.py::test_driver_manifest_fails_closed_on_schema_invalid_existing
ERROR tests/unit/test_run_manifest.py::test_write_run_manifest_exclusive_raises_on_existing
ERROR tests/run/test_run_correction_campaign.py::test_surface_flux_flag_wired_into_les_config
ERROR tests/run/test_run_correction_campaign.py::test_load_base_config_and_grid_roundtrip
ERROR tests/run/test_run_correction_campaign.py::test_effective_config_sidecar_persists_runtime_injections
ERROR tests/run/test_run_correction_campaign.py::test_configure_jax_compilation_cache
=========== 21 failed, 84 passed, 141 deselected, 25 errors in 9.54s ===========

codex
VERDICT: FIX-FIRST
tokens used
105,343
VERDICT: FIX-FIRST
