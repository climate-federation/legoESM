"""Non-vacuity and fail-closed tests for the lane-2 GYRE Phase 3 gate."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np

PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_phase3_gate.py"
)
TRACER_PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_phase3_tracer_gate.py"
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
    assert 'drag_causal_scaling["combined_trd_owner_label"]' in source


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
        "namdrg",
        "namdrg_bot",
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

    # A future group is discovered from the runtime artifact and
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


def test_ene_coefficient_reader_layout_and_planted_violation(tmp_path):
    nx, ny = gate.DIMS[:2]
    shape = (nx - 4, ny - 4)
    blocks = [
        np.asarray(index + np.arange(np.prod(shape)).reshape(shape, order="F"),
                   dtype=np.float64)
        for index in range(len(gate.ENE_COEFFICIENT_NAMES))
    ]
    path = tmp_path / "oracle_bt_ene_coeff_kt00000001.bin"
    path.write_bytes(
        b"NEMO_L2_ENECO_1 "
        + struct.pack("=7i", 1, 1, 1, 2, nx, ny, 64)
        + b"".join(block.ravel(order="F").tobytes() for block in blocks)
    )
    got = gate.read_ene_coefficients(path)
    for name, expected in zip(gate.ENE_COEFFICIENT_NAMES, blocks):
        np.testing.assert_array_equal(got[name], expected.T)

    oracle = got["ffu_nw"]
    active = np.ones_like(oracle, dtype=bool)
    planted = oracle.copy()
    planted[0, 0] += 1.0
    row = gate.score("planted_ene", oracle, planted, active)
    assert row["status"] == "DEBT"
    assert row["absolute_max"] == 1.0
    source = PATH.read_text()
    assert '"--plant-ene-coefficient"' in source
    assert "planted ENE coefficient did not fire" in source


def test_drag_operand_reader_layout_and_planted_violation(tmp_path):
    nx, ny = gate.DIMS[:2]
    full = np.arange(nx * ny, dtype=np.float64).reshape((nx, ny), order="F")
    path = tmp_path / "oracle_bt_drag_operands_kt00000001.bin"
    payload = [
        b"NEMO_L2_BTDRG_1 ",
        struct.pack("=6i", 1, 1, 50, nx, ny, 64),
        full.ravel(order="F").tobytes(),
        (full + 1).ravel(order="F").tobytes(),
    ]
    for jn in range(1, 51):
        payload.append(struct.pack("=i", jn))
        for index in range(len(gate.BT_DRAG_NAMES)):
            payload.append((full + 10 * jn + index).ravel(order="F").tobytes())
    path.write_bytes(b"".join(payload))

    got = gate.read_bt_drag_operands(path)
    np.testing.assert_array_equal(got["coefficient_u"], full[2:-2, 2:-2].T)
    expected = (full + 10)[2:-2, 2:-2].T
    np.testing.assert_array_equal(got["u_entry"][0], expected)

    oracle = got["coefficient_u"]
    planted = oracle.copy()
    planted[0, 0] += 1.0
    row = gate.score("planted_drag", oracle, planted, np.ones_like(oracle, dtype=bool))
    assert row["status"] == "DEBT"
    assert row["absolute_max"] == 1.0
    source = PATH.read_text()
    assert '"--plant-drag-coefficient"' in source
    assert "planted bottom-drag coefficient did not fire" in source


def test_stage2_operand_reader_layout_and_planted_violation(tmp_path):
    nx, ny, nz = gate.DIMS
    count3 = nx * ny * nz
    count2 = nx * ny
    blocks3 = [
        np.arange(count3, dtype=np.float64) + 1000 * index
        for index in range(10)
    ]
    blocks2 = [
        np.arange(count2, dtype=np.float64) + 100 * index
        for index in range(2)
    ]
    native_count2 = (nx - 4) * (ny - 4)
    native_blocks2 = [
        np.arange(native_count2, dtype=np.float64) + 100 * (index + 2)
        for index in range(2)
    ]
    path = tmp_path / "oracle_rkstage2_operands_kt00000001.bin"
    payload = [
        b"NEMO_L2_RKSTG_1 ",
        struct.pack("=11i", 1, 1, 2, 1, 3, 2, 2, nx, ny, nz, 64),
    ]
    payload.extend(block.tobytes() for block in blocks3[:8])
    payload.extend(block.tobytes() for block in blocks2)
    payload.extend(block.tobytes() for block in native_blocks2)
    payload.extend(block.tobytes() for block in blocks3[8:])
    path.write_bytes(b"".join(payload))

    got = gate.read_stage2_operands(path)
    expected_rhs_u = blocks3[4].reshape((nx, ny, nz), order="F")[
        2:-2, 2:-2
    ].transpose(1, 0, 2)
    np.testing.assert_array_equal(got["aliased_krhs_kaa_u"], expected_rhs_u)
    expected_correction_v = native_blocks2[1].reshape(
        (nx - 4, ny - 4), order="F"
    ).T
    np.testing.assert_array_equal(got["correction_v"], expected_correction_v)
    active = np.ones_like(got["aliased_krhs_kaa_u"], dtype=bool)
    row = gate.score(
        "planted_stage2_rhs",
        got["aliased_krhs_kaa_u"],
        got["aliased_krhs_kaa_u"],
        active,
        plant=True,
    )
    assert row["status"] == "DEBT"
    assert row["absolute_max"] == 1.0
    source = PATH.read_text()
    assert '"--plant-stage2-rhs"' in source
    assert "planted stage-2 RHS violation did not fire" in source


def test_stage2_term_reader_preserves_source_order_and_planted_control(tmp_path):
    nx, ny, nz = gate.DIMS
    count = nx * ny * nz
    blocks = [
        np.arange(count, dtype=np.float64) + 1000 * index
        for index in range(8)
    ]
    path = tmp_path / "oracle_rkstage2_terms_kt00000001.bin"
    path.write_bytes(
        b"NEMO_L2_RKTRM_1 "
        + struct.pack("=9i", 1, 1, 2, 3, 2, nx, ny, nz, 64)
        + b"".join(block.tobytes() for block in blocks)
    )
    got = gate.read_stage2_terms(path)
    expected = blocks[5].reshape((nx, ny, nz), order="F")[
        2:-2, 2:-2
    ].transpose(1, 0, 2)
    np.testing.assert_array_equal(got["after_vorticity_v"], expected)
    active = np.ones_like(expected, dtype=bool)
    row = gate.score("planted_hpg", expected, expected, active, plant=True)
    assert row["status"] == "DEBT"
    assert row["absolute_max"] == 1.0
    source = PATH.read_text()
    assert '"source_order": ["hpg", "vorticity", "advection"]' in source
    assert '"--plant-stage2-hpg"' in source
    assert "planted stage-2 HPG violation did not fire" in source


def test_stage_reader_includes_ssh_and_geometry_control_is_wired(tmp_path):
    nx, ny, nz = gate.DIMS
    count3 = nx * ny * nz
    count2 = nx * ny
    blocks3 = [
        np.arange(count3, dtype=np.float64) + 1000 * index
        for index in range(4)
    ]
    ssh = np.arange(count2, dtype=np.float64)
    path = tmp_path / "oracle_stage_kt00000001_s1.bin"
    path.write_bytes(
        b"NEMO_L1_STAGE_1 "
        + struct.pack("=9i", 1, 1, 1, 3, nx, ny, nz, 2, 64)
        + b"".join(block.tobytes() for block in blocks3)
        + ssh.tobytes()
    )
    got = gate.read_stage(path, 1)
    expected = ssh.reshape((nx, ny), order="F")[2:-2, 2:-2].T
    np.testing.assert_array_equal(got["ssh"], expected)
    source = PATH.read_text()
    assert '"--plant-stage1-ssh"' in source
    assert "planted stage-1 SSH violation did not fire" in source


def test_tracer_stage_operand_reader_and_planted_control_are_wired(tmp_path):
    nx, ny, nz = gate.DIMS
    count3 = nx * ny * nz
    count2 = nx * ny
    blocks3 = [
        np.arange(count3, dtype=np.float64) + 1000 * index
        for index in range(15)
    ]
    blocks2 = [
        np.arange(count2, dtype=np.float64) + 20000 * index
        for index in range(3)
    ]
    path = tmp_path / "oracle_rktracer_operands_kt00000001_s1.bin"
    path.write_bytes(
        b"NEMO_L2_RKTRA_1 "
        + struct.pack("=11i", 1, 1, 1, 1, 1, 3, 3, nx, ny, nz, 64)
        + b"".join(block.tobytes() for block in blocks3 + blocks2)
    )
    got = gate.read_tracer_stage_operands(path, 1)
    expected = blocks3[12].reshape((nx, ny, nz), order="F")[2:-2, 2:-2, :].transpose(1, 0, 2)
    np.testing.assert_array_equal(got["kmm_S"], expected)
    source = TRACER_PATH.read_text()
    assert '"--plant-tracer-sbc"' in source
    assert "planted tracer post-SBC violation did not fire" in source


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
