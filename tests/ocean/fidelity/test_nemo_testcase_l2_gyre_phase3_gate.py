"""Non-vacuity and fail-closed tests for the lane-2 GYRE Phase 3 gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_phase3_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_phase3_gate", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_score_exact_and_planted_violation():
    values = np.array([1.0, 2.0], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    assert gate.score("exact", values, values.copy(), mask)["status"] == "AT-BAR"
    assert gate.score("plant", values, values.copy(), mask, plant=True)["status"] == "DEBT"


def test_kt1_at_rest_controls_are_uninformative_but_debt_stays_red():
    for field in ("u", "v", "ssh"):
        row = gate._mark_kt1_uninformative({"status": "AT-BAR"}, field, 1)
        assert row["status"] == "UNINFORMATIVE"
        assert "at-rest" in row["reason"]
        assert gate._mark_kt1_uninformative({"status": "DEBT"}, field, 1)["status"] == "DEBT"


def test_one_variable_manifest_control_fires():
    arms = {"omit_stage_barotropic_correction": {"changed_operands": ["one"]}}
    assert gate.validate_one_variable_arms(arms)[0]["status"] == "VERIFIED"
    assert gate.validate_one_variable_arms(arms, plant=True)[0]["status"] == "DEBT"


def test_gate_reuses_registry_and_lane1_growth_instrument():
    source = PATH.read_text()
    assert "time_level_for_dump" in source
    assert 'with_name("nemo_testcase_phase3_trajectory_gate.py")' in source
    assert '"continue_after_first": True' in source
    assert '"UNMEASURED_AFTER_REGISTERED_ARMS"' in source


def test_full_rk3_inventory_and_scaling_precede_owner_labels():
    source = PATH.read_text()
    for token in ("read_stage", "read_transport", "read_rhs", "read_bt"):
        assert token in source
    assert '"scaling_check_before_owner_label": True' in source
    assert '"CONFIRMED_OWNER"' in source
    assert '"PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"' in source
    assert '"REFUTED_AS_PRIMARY_OWNER"' in source
    assert '"operator_scaling_before_owner"' in source
    assert '"UNMEASURED_SCALING_ONLY"' in source


def test_resolved_program_coverage_parses_runtime_namelist(tmp_path):
    blocks = (
        "namdyn_adv",
        "namdyn_vor",
        "namdyn_hpg",
        "namdyn_spg",
        "namdyn_ldf",
        "namtra_adv",
        "namtra_ldf",
        "namtra_eiv",
        "namtra_qsr",
        "namtra_dmp",
        "namtra_mle",
        "namzdf",
        "namzdf_tke",
    )
    resolved = tmp_path / "output.namelist.dyn"
    resolved.write_text("".join(f"&{block.upper()}\n /\n" for block in blocks))
    assert gate.resolved_namelist_blocks(resolved) == set(blocks)
    rows = gate.resolved_program_coverage_rows(
        resolved, {block: True for block in blocks}
    )
    assert all(row["status"] == "VERIFIED" for row in rows)

    # A future fourteenth group is discovered from the runtime artifact and
    # fails because the static card-disposition map has no row for it.
    resolved.write_text(
        resolved.read_text() + "&NAMTRA_FUTURE\n /\n"
    )
    rows = gate.resolved_program_coverage_rows(
        resolved, {block: True for block in blocks}
    )
    future = next(row for row in rows if row["name"].endswith("namtra_future"))
    assert future["status"] == "DEBT"
    assert future["runtime_namelist_present"] is True
    assert future["card_disposition_present"] is False


def test_resolved_program_coverage_gate_and_planted_control_are_wired():
    source = PATH.read_text()
    assert '"resolved_program_coverage"' in source
    assert "resolved_program_coverage_rows" in source
    assert 'coverage_checks["namdyn_vor"] = False' in source
    assert '"--plant-coverage"' in source


def test_momentum_scaling_arms_are_near_null_not_exonerated():
    source = PATH.read_text()
    assert '"NEAR-NULL_AT_KT2"' in source
    assert '"UNMEASURED_SCALING_ONLY"' in source
    assert "EXONERATED" not in source


def test_seasonal_sbc_controls_split_surface_and_freshwater_inputs():
    source = PATH.read_text()
    assert '"changed_operands": ["surface_forcing"]' in source
    assert '"changed_operands": ["freshwater"]' in source
    assert "surface_and_freshwater_forcing_pytree" not in source


def test_causal_oracle_operands_and_substep_control_are_wired():
    source = PATH.read_text()
    for token in (
        "read_zdf_entry",
        "read_qsr_stage3",
        "read_bt_substeps",
        '"DIAGNOSTIC_ONE_VARIABLE_CAUSAL_ARM"',
        '"CAUSAL_NEAR_NULL_AFTER_DIRECT_MATCH"',
        '"CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER"',
        '"barotropic_substep_boundary"',
        '"--plant-barotropic"',
    ):
        assert token in source
    assert '"scaling_check_before_owner_label": True' in source


def test_bt_trace_native_stagger_mapping():
    eta = np.zeros((2, 3, 4), dtype=np.float64)
    u = np.zeros((2, 3, 5), dtype=np.float64)
    v = np.zeros((2, 4, 4), dtype=np.float64)
    assert gate._trace_native(eta, "eta_entry").shape == (2, 3, 4)
    assert gate._trace_native(u, "u_entry").shape == (2, 3, 4)
    assert gate._trace_native(v, "v_entry").shape == (2, 3, 4)


def test_causal_arm_near_null_is_not_an_owner_claim():
    source = PATH.read_text()
    assert '"CAUSAL_NEAR_NULL_AFTER_DIRECT_MATCH"' in source
    assert "EXONERATED" not in source
