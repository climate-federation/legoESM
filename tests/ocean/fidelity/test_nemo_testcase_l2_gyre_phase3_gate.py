"""Non-vacuity and fail-closed tests for the lane-2 GYRE Phase 3 gate."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path

import jax.numpy as jnp
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe_module
import numpy as np
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry
from legoesm.ocean.physics.shortwave_penetration import (
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)

PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_phase3_gate.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_phase3_gate", PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

ADMISSION_PATH = PATH.with_name("nemo_testcase_l2_gyre_round21_admission.py")
ADMISSION_SPEC = importlib.util.spec_from_file_location(
    "gyre_round21_admission", ADMISSION_PATH)
assert ADMISSION_SPEC and ADMISSION_SPEC.loader
admission = importlib.util.module_from_spec(ADMISSION_SPEC)
ADMISSION_SPEC.loader.exec_module(admission)


GNX, GNY, GNZ = gate.DIMS
GN3 = GNX * GNY * GNZ


def test_scalar_math_root_identity_defaults_and_planted_mismatch(
    tmp_path, monkeypatch
):
    assert gate.ROOT == gate.SCALAR_MATH_ROOT
    assert gate.STAGE2_ROOT == gate.SCALAR_MATH_ROOT
    assert gate.STAGE3_ROOT == gate.SCALAR_MATH_ROOT

    name = "oracle_step_entry_kt00000002.bin"
    canonical = tmp_path / "canonical"
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    for root in (canonical, good, bad):
        root.mkdir()
    payload = b"certified scalar-math kt2"
    (canonical / name).write_bytes(payload)
    (good / name).write_bytes(payload)
    (bad / name).write_bytes(payload + b" planted violation")
    monkeypatch.setattr(gate, "SCALAR_MATH_ROOT", canonical)
    monkeypatch.setitem(
        gate.BIT_IDENTITY_EXPECTED, name, hashlib.sha256(payload).hexdigest())

    rows = gate.require_scalar_math_roots(good, good, good)
    assert all(row["status"] == "VERIFIED" for row in rows)
    try:
        gate.require_scalar_math_roots(good, good, bad)
    except gate.GateError as exc:
        assert "stage3 root is not certified scalar-math v2" in str(exc)
    else:
        raise AssertionError("planted vectorized/mismatched root was admitted")


def _transport_record(values: np.ndarray) -> bytes:
    return b"NEMO_L1_TRANSP_1" + struct.pack(
        "=8i", 1, 1, 1, 1, GNX, GNY, GNZ, 64) + values.tobytes()


ZFW_WAIVED = {("NEMO_L1_TRANSP_1", "zFw"): "GYRE resolves ln_dynadv_vec = T"}


def test_admission_admits_only_the_undefined_zfw_slot(tmp_path):
    """On GYRE's arm zFw is undefined at this record's write point; zFu is not.

    The waiver is passed in explicitly, because it is a property of the RUN
    (ln_dynadv_vec) and not of the record kind -- on the flux-form tanks the
    same slot is filled before the same write and IS bit-tested.
    """
    values = np.zeros(3 * GN3, dtype=np.float64)
    baseline = tmp_path / "baseline.bin"
    candidate = tmp_path / "candidate.bin"
    baseline.write_bytes(_transport_record(values))
    changed = values.copy()
    # An OWNED cell of the undefined zFw slot: admitted on GYRE's arm.
    changed[2 * GN3 + (2 + GNX * 2)] = 1.0
    candidate.write_bytes(_transport_record(changed))
    assert admission.compare_record(
        baseline, candidate, [True], waived=ZFW_WAIVED)["consumed_equal"]
    # ... and NOT admitted with no waiver, which is the tanks' case.
    assert not admission.compare_record(
        baseline, candidate, [True])["consumed_equal"]
    # The same OWNED cell of the DEFINED zFu slot: a violation either way.
    changed[2 + GNX * 2] = 1.0
    candidate.write_bytes(_transport_record(changed))
    assert not admission.compare_record(
        baseline, candidate, [True], waived=ZFW_WAIVED)["consumed_equal"]


def test_stage_ww_reader_discards_unowned_nonfinite_halo(tmp_path):
    path = tmp_path / "oracle_rkstage_ww_kt00000001_s1.bin"
    header = b"NEMO_L2_STGWW_1 " + struct.pack(
        "=10i", 1, 1, 1, 1, 1, 3, gate.DIMS[0], gate.DIMS[1], gate.DIMS[2], 64)
    values = np.zeros(1 + 2 * GN3, dtype=np.float64)
    values[0] = 4800.0
    values[1] = np.nan
    values[1 + GN3] = np.nan
    path.write_bytes(header + values.tobytes())
    record = gate.read_stage_ww(path, 1)
    assert np.all(np.isfinite(record["ww"]))
    assert np.all(np.isfinite(record["pFw"]))


def test_score_exact_and_planted_violation():
    values = np.array([1.0, 2.0], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    assert gate.score("exact", values, values.copy(), mask)["status"] == "AT-BAR"
    assert gate.score("plant", values, values.copy(), mask, plant=True)["status"] == "DEBT"


def test_bt_reader_exposes_all_four_prognostic_and_transport_fields(tmp_path):
    nx, ny = gate.DIMS[:2]
    count = nx * ny
    path = tmp_path / "oracle_bt_frames_kt00000001.bin"
    header = b"NEMO_L1_BTFRM_1 " + struct.pack("=6i", 1, 1, 3, nx, ny, 64)
    values = np.arange(4 * count, dtype=np.float64)
    path.write_bytes(header + values.tobytes())
    record = gate.read_bt(path, 1)
    assert set(record) == {
        "kt", "Kaa", "registry_level", "uu_b", "vv_b", "un_adv", "vn_adv"}
    assert record["uu_b"].shape == (ny - 4, nx - 4)
    assert record["vn_adv"].shape == (ny - 4, nx - 4)


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
    assert '"BOTTOM_DRAG_CONFIRMED_AT_SUBSTEP2; NEXT_BOUNDARY_FROM_REGISTER"' in source


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
        "namdrg",
        "namdrg_bot",
    )
    resolved = tmp_path / "output.namelist.dyn"
    resolved.write_text("".join(f"&{block.upper()}\n /\n" for block in blocks))
    assert gate.resolved_namelist_blocks(resolved) == set(blocks)
    rows = gate.resolved_program_coverage_rows(
        resolved, {block: True for block in blocks}
    )
    assert all(row["status"] == "VERIFIED" for row in rows)

    # A future dynamics/tracer/drag group is discovered from the runtime artifact and
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


def test_stage_ww_reader_and_one_cell_plant(tmp_path):
    nx, ny, nz = gate.DIMS
    count = nx * ny * nz
    ww = np.arange(count, dtype=np.float64)
    pfw = ww + np.float64(0.25)
    path = tmp_path / "oracle_rkstage_ww_kt00000001_s1.bin"
    path.write_bytes(
        b"NEMO_L2_STGWW_1 "
        + struct.pack("=10i", 1, 1, 1, 1, 1, 3, nx, ny, nz, 64)
        + np.asarray([4800.0], dtype=np.float64).tobytes()
        + ww.tobytes()
        + pfw.tobytes()
    )
    record = gate.read_stage_ww(path, 1)
    assert record["rDt_s"] == 4800.0
    expected = ww.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)
    np.testing.assert_array_equal(record["ww"], expected)
    mask = np.ones(expected.shape, dtype=bool)
    exact = gate.score("stage_ww", expected, record["ww"], mask)
    planted = gate.score("stage_ww", expected, record["ww"], mask, plant=True)
    assert exact["n_unequal"] == 0 and exact["status"] == "AT-BAR"
    assert planted["n_unequal"] == 1 and planted["status"] == "DEBT"


def test_bottom_drag_boundary_is_source_reconstructed_and_fail_closed():
    source = PATH.read_text()
    for token in (
        '"bottom_drag_operand_walk"',
        '"coefficient_time_level": "Kmm_once_per_whole_step"',
        '"substep_velocity_time_level": "entry_un_e_vn_e"',
        '"omit_barotropic_substep_drag"',
        '"--plant-drag"',
        "planted drag violation did not fire",
    ):
        assert token in source
    assert source.index('"scaling_check_before_owner_label": True') < source.index(
        '"owner_label": drag_owner_label')


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


def test_stage3_tracer_helper_returns_the_content_it_actually_advanced():
    grid = create_beta_plane_cgrid_geometry(
        2, 3, dx_m=8.0, dy_m=8.0, f0=0.0, beta=0.0,
        cartesian_pseudo_lat=True)
    shape = (2, 3, 2)
    tracer = jnp.arange(12, dtype=jnp.float64).reshape(shape)
    salt = tracer + 30.0
    h = jnp.ones(shape, dtype=jnp.float64)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    source = jnp.full(shape, 3.0, dtype=jnp.float64)
    result = model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, salt, "centered",
        jnp.zeros((2, 4, 2), dtype=jnp.float64),
        jnp.zeros((3, 3, 2), dtype=jnp.float64),
        jnp.zeros((2, 3, 3), dtype=jnp.float64),
        h, h, jnp.ones((2, 4, 2), dtype=jnp.float64),
        jnp.ones((3, 3, 2), dtype=jnp.float64), grid, 2.0,
        jnp.ones(shape, dtype=jnp.float64),
        stage_source_rates=((zero, zero), (zero, zero), (source, zero)),
        return_final_content=True,
    )
    out_t, out_s, content_t, content_s, adv_t, adv_s = result
    np.testing.assert_array_equal(np.asarray(adv_t), np.asarray(tracer))
    np.testing.assert_array_equal(np.asarray(adv_s), np.asarray(salt))
    np.testing.assert_array_equal(np.asarray(content_t), np.asarray(tracer + 6.0))
    np.testing.assert_array_equal(np.asarray(content_s), np.asarray(salt))
    np.testing.assert_array_equal(np.asarray(out_t), np.asarray(content_t))
    np.testing.assert_array_equal(np.asarray(out_s), np.asarray(content_s))


def test_stage3_qsr_replacement_removes_kbb_and_adds_kmm():
    tendency = jnp.asarray([[[11.0]]], dtype=jnp.float64)
    qsr_kbb = jnp.asarray([[[3.0]]], dtype=jnp.float64)
    qsr_kmm = jnp.asarray([[[5.0]]], dtype=jnp.float64)
    result = model_module._nemo_qsr_stage3_rate(
        tendency, qsr_kbb, qsr_kmm,
        jnp.asarray([[[2.0]]]), jnp.asarray([[[4.0]]]))
    np.testing.assert_array_equal(np.asarray(result), np.asarray([[[9.0]]]))


def test_source_named_two_band_selector_owns_full_qsr_composition():
    sw = jnp.asarray([100.0], dtype=jnp.float64)
    nemo = pe_module._shortwave_surface_composition(
        sw, "nemo_qsr_2bd", jnp.float64)
    generic = pe_module._shortwave_surface_composition(
        sw, "jerlov_2band", jnp.float64)
    np.testing.assert_array_equal(np.asarray(nemo), np.asarray([100.0]))
    np.testing.assert_array_equal(np.asarray(generic), np.asarray([94.0]))


def test_two_band_live_kmm_ladder_changes_the_stage3_profile():
    sw = jnp.asarray([[100.0]], dtype=jnp.float64)
    dz = jnp.asarray([10.0, 20.0], dtype=jnp.float64)
    z_half = jnp.asarray([0.0, -10.0, -30.0], dtype=jnp.float64)
    cfg = ShortwavePenetrationConfig(
        scheme="nemo_qsr_2bd", water_type="I", nemo_time_step_s=14400.0
    )
    base = shortwave_penetration_tendency(
        sw, dz, z_half, jnp.ones((1, 1)), cfg)
    live = shortwave_penetration_tendency(
        sw, dz, z_half, jnp.ones((1, 1)), cfg,
        z_half_stretch=jnp.asarray([[[1.1]]]))
    assert not np.array_equal(np.asarray(base), np.asarray(live))
