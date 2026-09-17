"""Hermetic fail-closed checks for the round-46 acquisition package."""

from __future__ import annotations

import importlib.util
import struct
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round46_kt2_stage_gate",
    TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

NX, NY, NZ = gate.DIMS
SCALARS = {f"has_{x}" for x in ("hpg", "vor", "keg", "zad", "ldf", "zdf")} | {"r1_Dt"}
TWO_D = {
    name
    for name in gate.REQUIRED
    if name.startswith("r3")
    or name.startswith("ssh_")
    or name.startswith("uu_b_")
    or name.startswith("vv_b_")
} | {
    "e1e2t",
    "e1e2u",
    "e1e2v",
    "r1_e1e2u",
    "r1_e1e2v",
    "r1_e1u",
    "r1_e2v",
    "r1_e1e2t",
    "e2u",
    "e1v",
}


def _payload(value: np.ndarray, rank: int) -> bytes:
    if rank == 2:
        return np.asarray(value).T.ravel(order="F").tobytes()
    return np.asarray(value).transpose(1, 0, 2).ravel(order="F").tobytes()


def _write(path: Path, *, stage=1, truncate=False) -> Path:
    names = set(gate.REQUIRED) | {
        "pre_baro_u",
        "pre_baro_v",
        "post_update_u",
        "post_update_v",
        "after_ldf_u",
        "after_ldf_v",
    }
    names -= {"post_zdf_u", "post_zdf_v", "pre_zdf_rhs_u", "pre_zdf_rhs_v"}
    with path.open("wb") as f:
        f.write(gate.MAGIC.ljust(16).encode())
        f.write(struct.pack("=16i", 1, 2, stage, 1, 1, 2, 3, NX, NY, NZ, 30, 3, 34, 3, 24, 64))
        for index, name in enumerate(sorted(names)):
            f.write(name.ljust(16).encode())
            if name in SCALARS:
                f.write(struct.pack("=4i", 0, 1, 1, 1))
                value = gate.PRESENCE[stage].get(name[4:], 1) if name.startswith("has_") else 1
                raw = np.asarray([value], dtype=np.float64).tobytes()
            elif name in TWO_D:
                f.write(struct.pack("=4i", 2, NX, NY, 1))
                fill = 1.0 if name.startswith(("e1", "r1_e1", "r3", "ssh_")) else 0.0
                raw = _payload(np.full((NY, NX), fill), 2)
            else:
                owned = name in gate.OWNED_3D_FIELDS
                nx, ny = gate.OWNED_DIMS[:2] if owned else (NX, NY)
                f.write(struct.pack("=4i", 3, nx, ny, NZ))
                fill = 1.0 if name.startswith(("e3", "tmask", "umask", "vmask", "wmask")) else 0.0
                raw = _payload(np.full((ny, nx, NZ), fill), 3)
            f.write(raw[:-8] if truncate and index == 0 else raw)
            if truncate and index == 0:
                break
    return path


def _write_tke_statement(path: Path, *, trailing=False) -> tuple[Path, dict]:
    nx, ny, nz = gate.OWNED_DIMS
    arrays = {
        name: (np.arange(nx * ny * nz, dtype=np.float64)
               .reshape((nx, ny, nz), order="F") + index)
        for index, name in enumerate(gate.TKE_STATEMENT_FIELDS)
    }
    payload = bytearray(gate.TKE_STATEMENT_MAGIC)
    payload.extend(struct.pack(
        "=13i", 1, 2, 3, 3, NX, NY, NZ, 30, 3, 34, 3, 24, 64))
    for name in gate.TKE_STATEMENT_FIELDS:
        payload.extend(arrays[name].ravel(order="F").tobytes())
    if trailing:
        payload.extend(np.float64(1.0).tobytes())
    path.write_bytes(payload)
    return path, arrays


def test_reader_and_zero_source_replays(tmp_path):
    rec = gate.read_stage(_write(tmp_path / "stage.bin"))
    records = {key: rec for key in gate.STAGES}
    assert all(value == 0 for value in gate._calibrate(records, None).values())


def test_header_and_truncation_plants_are_red(tmp_path):
    path = _write(tmp_path / "stage.bin")
    with pytest.raises(Exception, match="wrong kt/stage"):
        gate.read_stage(path, plant="header")
    with pytest.raises(Exception, match="short payload"):
        gate.read_stage(_write(tmp_path / "short.bin", truncate=True))


def test_round101_tke_statement_reader_closes_layout_and_eof(tmp_path):
    path, expected = _write_tke_statement(tmp_path / "tke.bin")
    record = gate.read_tke_statement_walk_record(path)
    assert path.stat().st_size == 873028
    assert record["header"]["kt"] == 2
    assert record["header"]["bits"] == 64
    for name in gate.TKE_STATEMENT_FIELDS:
        np.testing.assert_array_equal(record["arrays"][name], expected[name])
    with pytest.raises(Exception, match="bad magic"):
        gate.read_tke_statement_walk_record(path, plant="header")
    with pytest.raises(Exception, match="truncated"):
        gate.read_tke_statement_walk_record(path, plant="truncation")
    trailing, _ = _write_tke_statement(tmp_path / "trailing.bin", trailing=True)
    with pytest.raises(Exception, match="trailing bytes"):
        gate.read_tke_statement_walk_record(trailing)


def test_round101_tke_duplicate_rows_and_one_ulp_plant(tmp_path):
    path, arrays = _write_tke_statement(tmp_path / "tke.bin")
    record = gate.read_tke_statement_walk_record(path)
    legacy = {"arrays": {
        name: np.array(arrays[name], copy=True)
        for name in ("en_entry", "rhs_pre_sweep", "en_post_sweep")
    }}
    clean = gate._tke_statement_duplicate_rows(record, legacy)
    assert len(clean) == 9
    assert all(row["classification"] == "BIT" for row in clean)
    planted = gate._tke_statement_duplicate_rows(
        record, legacy, plant_ulp=True)
    target = next(
        row for row in planted
        if row["field"] == "rhs_pre_sweep"
        and row["domain"] == "compiled_consumed_1_jpkm1"
    )
    assert target["classification"] == "DEBT"
    assert target["n_unequal"] == 1


def test_round101_tke_rhs_unconsumed_sentinel_is_reported_not_binding(tmp_path):
    path, arrays = _write_tke_statement(tmp_path / "tke.bin")
    record = gate.read_tke_statement_walk_record(path)
    legacy_arrays = {
        name: np.array(arrays[name], copy=True)
        for name in ("en_entry", "rhs_pre_sweep", "en_post_sweep")
    }
    legacy_arrays["rhs_pre_sweep"][:, :, -1] = 0.0
    rows = gate._tke_statement_duplicate_rows(
        record, {"arrays": legacy_arrays})
    full = next(
        row for row in rows
        if row["field"] == "rhs_pre_sweep"
        and row["domain"] == "complete_stored_array"
    )
    consumed = next(
        row for row in rows
        if row["field"] == "rhs_pre_sweep"
        and row["domain"] == "compiled_consumed_1_jpkm1"
    )
    sentinel = next(
        row for row in rows
        if row["field"] == "rhs_pre_sweep"
        and row["domain"] == "unconsumed_jpk_sentinel"
    )
    assert full["classification"] == "DEBT"
    assert sentinel["classification"] == "DEBT"
    assert not full["admission_binding"]
    assert not sentinel["admission_binding"]
    assert consumed["classification"] == "BIT"
    assert consumed["admission_binding"]


def test_round102_tke_production_rows_and_ulp_control(tmp_path):
    path, arrays = _write_tke_statement(tmp_path / "tke.bin")
    record = gate.read_tke_statement_walk_record(path)
    production = SimpleNamespace(
        en_entry=arrays["en_entry"].swapaxes(0, 1)[..., 1:30],
        en_after_boundaries=(
            arrays["en_after_boundaries"].swapaxes(0, 1)[..., :30]),
        en_after_langmuir=(
            arrays["en_after_langmuir"].swapaxes(0, 1)[..., :30]),
        rhs_pre_sweep=(
            arrays["rhs_pre_sweep"].swapaxes(0, 1)[..., :30]),
        en_post_sweep=(
            arrays["en_post_sweep"].swapaxes(0, 1)[..., :30]),
    )
    trace = SimpleNamespace(tke_statement_trace=production)
    clean = gate._tke_production_statement_rows(
        trace, record, "NEMO_RECORDED")
    assert clean["first_nonbit"] is None
    assert all(row["classification"] == "BIT" for row in clean["rows"])
    assert all(row["classification"] == "BIT"
               for row in clean["boundary_block_rows"])
    assert sum(row["selected_cells"]
               for row in clean["boundary_block_rows"]) == 22 * 32 * 30
    planted = gate._tke_production_statement_rows(
        trace, record, "NEMO_RECORDED", "stage-tke-production-ulp")
    target = next(
        row for row in planted["rows"]
        if row["field"] == "en_after_boundaries")
    assert target["n_unequal"] == 1
    assert target["clean_n_unequal"] == 0
    assert planted["plant_target"] == target["name"]


def test_round102_tke_surface_operand_attribution_uses_live_trace(tmp_path):
    path, arrays = _write_tke_statement(tmp_path / "tke.bin")
    record = gate.read_tke_statement_walk_record(path)
    rho0 = np.float64(1026.0)
    rn_ebb = np.float64(67.83)
    rn_emin0 = np.float64(1.0e-4)
    taum_raw = np.full((32, 22), np.float64(0.02))
    surface_raw = np.maximum(rn_emin0, rn_ebb / rho0 * taum_raw)
    record["arrays"]["en_after_boundaries"][:, :, 0] = surface_raw
    operand = {"arrays": {
        "taum_entry": taum_raw,
        "rn_ebb": rn_ebb,
        "rn_emin0": rn_emin0,
    }}
    exact_trace = SimpleNamespace(tke_statement_trace=SimpleNamespace(
        taum_surface=taum_raw.swapaxes(0, 1),
        surface_dirichlet=surface_raw.swapaxes(0, 1),
    ))
    exact = gate._tke_surface_operand_rows(
        exact_trace, record, operand, "NEMO_TKE_RECORDED", rho0)
    assert all(row["classification"] == "BIT" for row in exact["rows"])

    moved_taum = taum_raw.swapaxes(0, 1).copy()
    moved_taum[0, 0] = np.nextafter(moved_taum[0, 0], np.float64(np.inf))
    moved_trace = SimpleNamespace(tke_statement_trace=SimpleNamespace(
        taum_surface=moved_taum,
        surface_dirichlet=np.maximum(rn_emin0, rn_ebb / rho0 * moved_taum),
    ))
    moved = gate._tke_surface_operand_rows(
        moved_trace, record, operand, "MODEL_TAUM", rho0)
    rows = {row["field"]: row for row in moved["rows"]}
    assert rows["production_taum_vs_recorded"]["n_unequal"] == 1
    assert rows["recorded_operand_scalar_replay"]["classification"] == "BIT"


def test_round101_tke_writer_is_additive_write_only_and_fixed_layout():
    package = TESTCASES / "nemo_testcase_l2_gyre_round101_tke_walk"
    writer = (package / "l2_r101_tke_walk.F90").read_text()
    patch = (package / "zdftke_round101.patch").read_text()
    assert "ACTION='WRITE'" in writer
    assert "STATUS='REPLACE'" in writer
    assert "kt == nit000+1" in writer
    assert "STORAGE_SIZE(1) /= 32" in writer
    assert "STORAGE_SIZE(1._wp) /= 64" in writer
    assert "r101_counts /= expected_rows" in writer
    assert writer.count("missing, duplicated, or out of order") == 4
    assert "r101_entry_field, r101_boundaries, r101_langmuir" in writer
    assert "r101_rhs, r101_post_sweep" in writer
    assert not any(line.startswith("-") and not line.startswith("---")
                   for line in patch.splitlines())
    assert "CALL r101_after_boundaries_row" in patch
    assert "CALL r101_after_langmuir_row" in patch
    assert "CALL r101_rhs_row" in patch
    assert "CALL r101_post_sweep_row" in patch


def test_instrument_contract_is_zero_first_write_only_and_widened():
    package = TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage"
    writer = (package / "l2_r46_stage.F90").read_text()
    assert "ACTION='WRITE'" in writer
    assert "STATUS='REPLACE'" in writer
    assert "z(:,:,:)=0._wp" in writer or "z(:,:,:) = 0._wp" in writer
    assert "kt <= nit000+1" in writer
    assert "PUBLIC   dissl" in (package / "zdftke_round46.patch").read_text()
    stg = (package / "stprk3_stg_round46.patch").read_text()
    adv = (package / "dynadv_round46.patch").read_text()
    assert "kstp <= nit000 + 1" in stg
    assert 'oracle_rkstage3_terms_kt",I8.8' in stg
    assert "kt <= nit000 + 1" in adv
    assert 'oracle_dynadv_split_kt",I8.8' in adv


def test_every_declared_plant_has_a_nonzero_exit_contract():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    run = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage/run.sh").read_text()
    for plant in ("header", "truncation", "calibration", "given", "trajectory",
                  "twin", "stage-entry-ulp", "stage-context-ulp",
                  "stage-rhs-ulp", "stage-w-transport-ulp",
                  "stage-w-carry-ulp", "stage-tke-record-header",
                  "stage-tke-record-truncation", "stage-tke-record-stamp",
                  "stage-tke-record-ulp", "stamp"):
        assert f'"{plant}"' in source
    assert 'return 1 if args.plant or report["status"] != "PASS" else 0' in source
    for plant in ("header", "truncation", "calibration", "twin", "stamp"):
        assert plant in run


def test_retracted_round48_owner_prediction_cannot_abort_current_measurement():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert '"interpretation": (' in source
    assert '"POSTHOC_AFTER_ROUND49_ROUND50" if kt == 2' in source
    assert "round48 first-non-bit prediction REFUTED" not in source


def test_stage_output_reference_uses_next_handoff_and_next_step():
    shape3 = (NY, NX, NZ)
    shape2 = (NY, NX)

    def arrays(fill):
        return {
            "post_baro_u": np.full(shape3, fill),
            "post_baro_v": np.full(shape3, fill + 1),
            "T_Kmm": np.full(shape3, fill + 2),
            "S_Kmm": np.full(shape3, fill + 3),
            "ssh_Kmm": np.full(shape2, fill + 4),
        }

    records = {(1, stage): {"arrays": arrays(10 * stage)}
               for stage in (1, 2, 3)}
    next_entries = {2: {
        "T": np.full((NY - 4, NX - 4, NZ), 70.0),
        "S": np.full((NY - 4, NX - 4, NZ), 71.0),
        "ssh": np.full((NY - 4, NX - 4), 72.0),
    }}
    stage1 = gate._stage_reference(records, next_entries, 1, 1)
    stage3 = gate._stage_reference(records, next_entries, 1, 3)
    assert np.all(stage1["T"] == 22.0)
    assert np.all(stage1["ssh"] == 24.0)
    assert np.all(stage3["T"] == 70.0)
    assert np.all(stage3["ssh"] == 72.0)
    assert stage3["T"].shape[-1] == 30


def test_stage_twin_private_overrides_are_off_by_default():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    hooks = _NEMOWSRK3TestHooks()
    assert hooks.stage_barotropic_output_override is None
    assert hooks.stage_entry_override is None


def test_stage_twin_refuses_unmeasured_required_rows():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert 'report["status"] = "UNMEASURED"' in source
    assert 'row.get("classification") == "UNMEASURED_WITH_SPEC"' in source
    assert 'report["stage_twin"]["first_owned_nonbit"] = None' in source
    assert 'report["stage_twin"]["first_measured_nonbit"]' in source


def test_stage_twin_scores_consumed_rhs_and_reuses_operator_walk():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert 'f"{rhs_boundary}_{face}"' in source
    assert '"field": f"momentum_rhs_{face}"' in source
    assert "kt=1, stages=(1,)" in source
    assert '"compiled_order": ("hpg", "ldf", "vor", "adv")' in source
    assert "stage1_operator_rows" in source
    assert '"first_nonbit_model_statement"' in source


def test_stage_twin_separates_isolated_jit_from_the_production_step():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert '"isolated-closure eager"' in source
    assert '"isolated-closure JIT"' in source
    assert '"production step"' in source
    assert "trace.stage_raw_velocities" in source
    assert "trace.stage_rhs" in source
    assert '"production_fusion_discriminator_fired"' in source
    assert '"stage-assignment-output-ulp"' in source


def test_stage_twin_uses_the_direct_post_transport_w_reference():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert 'read_stage_ww(' in source
    assert '"post_tra_adv_trp_transport_form"' in source
    assert '"pre_external_zad_operand_w"' in source
    assert '"stage1_w_walk"' in source


def test_stage1_w_acquisition_reader_is_exact_and_fail_closed(tmp_path):
    path = tmp_path / "oracle_stage1_w_walk_kt00000001.bin"
    n2, n3 = NX * NY, NX * NY * NZ
    local_n3 = (NX - 2) * (NY - 2) * NZ
    values = np.arange(2 * local_n3 + 2 * n3 + 2 * n2 + 1,
                       dtype=np.float64)
    path.write_bytes(
        b"NEMO_L2_R98W_1  "
        + struct.pack("=9i", 1, 1, 1, 1, 3, NX, NY, NZ, 64)
        + values.tobytes())
    record = gate.read_stage1_w_walk_record(path)
    assert record["header"]["Kaa"] == 3
    assert record["hdiv"].shape == (NY - 2, NX - 2, NZ)
    assert record["e3div"].shape == (NY - 2, NX - 2, NZ)
    assert record["r3_kaa"].shape == (NY, NX)
    assert record["ww"].shape == (NY, NX, NZ)
    producer = "a" * 40
    (tmp_path / "producer_commit.txt").write_text(producer + "\n")
    digest = gate.sha256(path)
    (tmp_path / (path.name + ".stamp")).write_text(
        f"{digest} {producer} {path.name}\n")
    admitted = gate.read_admitted_stage1_w_walk(tmp_path)
    assert admitted["sha256"] == digest
    assert admitted["producer_commit"] == producer
    with pytest.raises(Exception, match="producer commit mismatch"):
        gate.read_admitted_stage1_w_walk(tmp_path, plant_stamp=True)
    path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(Exception, match="wrong payload size"):
        gate.read_stage1_w_walk_record(path)
    path.write_bytes(
        b"NEMO_L2_R98W_1  "
        + struct.pack("=9i", 1, 1, 1, 1, 3, NX, NY, NZ, 64)
        + np.append(values, np.float64(0.0)).tobytes())
    with pytest.raises(Exception, match="wrong payload size"):
        gate.read_stage1_w_walk_record(path)


def test_stage1_r3_operand_reader_is_exact_and_fail_closed(tmp_path):
    path = tmp_path / "oracle_stage1_r3_operands_kt00000001.bin"
    values = np.arange(6 * NX * NY, dtype=np.float64)
    payload = (
        b"NEMO_L2_R99R3_1 "
        + struct.pack("=9i", 1, 1, 1, 1, 3, NX, NY, NZ, 64)
        + values.tobytes()
    )
    path.write_bytes(payload)
    record = gate.read_stage1_r3_operand_record(path)
    assert record["header"]["Kaa"] == 3
    assert tuple(record) == (
        "ssh_kaa", "r1_ht_0", "r3_kaa", "ssh_kbb", "r3_kbb", "ht_0",
        "header",
    )
    assert all(record[name].shape == (NY, NX) for name in (
        "ssh_kaa", "r1_ht_0", "r3_kaa", "ssh_kbb", "r3_kbb", "ht_0",
    ))
    producer = "b" * 40
    (tmp_path / "producer_commit.txt").write_text(producer + "\n")
    digest = gate.sha256(path)
    (tmp_path / (path.name + ".stamp")).write_text(
        f"{digest} {producer} {path.name}\n")
    admitted = gate.read_admitted_stage1_r3_operands(tmp_path)
    assert admitted["sha256"] == digest
    assert admitted["producer_commit"] == producer
    with pytest.raises(Exception, match="producer commit mismatch"):
        gate.read_admitted_stage1_r3_operands(tmp_path, plant_stamp=True)
    path.write_bytes(payload[:-8])
    with pytest.raises(Exception, match="wrong payload size"):
        gate.read_stage1_r3_operand_record(path)
    path.write_bytes(payload + np.float64(0.0).tobytes())
    with pytest.raises(Exception, match="wrong payload size"):
        gate.read_stage1_r3_operand_record(path)


def test_rk3_vector_assignment_scalar_preserves_source_order():
    before = np.asarray([1.0, -2.0], dtype=np.float64)
    rhs = np.asarray([0.25, 0.5], dtype=np.float64)
    mask = np.asarray([1.0, 0.0], dtype=np.float64)
    got = gate._rk3_vector_assignment_scalar(before, rhs, 2.0, mask)
    expected = np.multiply(
        np.add(before, np.multiply(np.float64(2.0), rhs)), mask)
    assert np.array_equal(got.view(np.uint64), expected.view(np.uint64))


def test_transport_w_scalar_replay_preserves_zero_state():
    import jax
    import jax.numpy as jnp

    shape3 = (NY, NX, NZ)
    arrays = {
        "tmask": np.ones(shape3),
        "e3t_Kmm": np.ones(shape3),
        "e3t_0": np.ones(shape3),
        "r1_e1e2t": np.ones((NY, NX)),
        "r3t_Kbb": np.zeros((NY, NX)),
        "r3t_Kaa": np.zeros((NY, NX)),
        "r1_Dt": np.float64(1.0),
    }
    transport = np.zeros((NY - 4, NX - 4, NZ))
    got = gate._stage1_transport_w_scalar_reference(
        arrays, transport, transport)
    assert all(np.count_nonzero(got[name]) == 0
               for name in gate._STAGE1_W_ORDER)

    recurrence = gate._stage1_w_scalar_recurrence(
        np.zeros((NY - 4, NX - 4, NZ - 1)),
        np.ones((NY - 4, NX - 4, NZ - 1)),
        np.zeros((NY - 4, NX - 4)),
        np.zeros((NY - 4, NX - 4)),
        np.float64(1.0),
        np.ones((NY - 4, NX - 4, NZ - 1)),
    )
    assert all(np.count_nonzero(value) == 0 for value in recurrence.values())

    planted = jax.device_get(jax.jit(
        lambda: gate._stage1_w_recurrence_trace(
            jnp.zeros((NY - 4, NX - 4, NZ - 1)),
            jnp.ones((NY - 4, NX - 4, NZ - 1)),
            jnp.zeros((NY - 4, NX - 4)),
            jnp.zeros((NY - 4, NX - 4)),
            jnp.asarray(1.0),
            jnp.ones((NY - 4, NX - 4, NZ - 1)),
            source_round=True, plant_carry_at=(0, 0, NZ - 2),
        ))())
    assert np.count_nonzero(planted["incoming_carry"]) == 1


def test_stage_w_walk_retains_retracted_and_compiled_r3_associations():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert '"RETRACTED_NOT_COMPILED_ORDER"' in source
    assert '"COMPILED_ORDER"' in source
    assert '"r3_kaa_association_rows"' in source


def test_final_external_history_uses_last_pre_swap_current_and_before():
    arrays = {}
    for index, name in enumerate(
        ("u_entry", "u_b", "v_entry", "v_b", "eta_entry", "eta_b"),
        start=1,
    ):
        arrays[name] = np.asarray([[index], [10 + index]], dtype=np.float64)
    histories = gate._final_history_from_btstep(arrays)
    assert tuple(float(value[0]) for value in histories) == (
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0)


def test_kt2_transport_uses_recorded_rotated_pointer_slots(tmp_path):
    from nemo_testcase_l2_gyre_phase3_gate import GateError, read_transport

    path = tmp_path / "oracle_tracer_transport_kt00000002_s3.bin"
    header = b"NEMO_L2_TRTRP_1 " + struct.pack(
        "=11i", 1, 2, 3, 3, 2, 1, 1, NX, NY, NZ, 64)
    values = np.zeros(3 * NX * NY * NZ, dtype=np.float64)
    path.write_bytes(header + values.tobytes())
    got = read_transport(
        path, 3, expected_kt=2, expected_slots=(3, 2, 1, 1))
    assert got["stage"] == 3 and got["Kmm"] == 2
    with pytest.raises(GateError, match="wrong stage indices"):
        read_transport(
            path, 3, expected_kt=2, expected_slots=(1, 2, 3, 3))


def test_qco_face_layout_drops_redundant_edge_and_singleton_level():
    u = np.zeros((NY - 4, NX - 3, 1))
    v = np.zeros((NY - 3, NX - 4, 1))
    assert u[:, 1:, 0].shape == (NY - 4, NX - 4)
    assert v[1:, :, 0].shape == (NY - 4, NX - 4)


def test_missing_model_context_is_fail_closed_not_an_exception():
    class Context:
        tke = None
        tke_avm = None
        tke_avt = None
        tke_dissl = None
        tke_avm_surface = None

    arrays = {
        "tke_en": np.zeros((NY - 4, NX - 4, NZ)),
        "tke_avm_k": np.zeros((NY, NX, NZ)),
        "tke_avt_k": np.zeros((NY - 4, NX - 4, NZ)),
        "tke_dissl": np.zeros((NY - 4, NX - 4, NZ)),
        "wmask": np.ones((NY, NX, NZ)),
    }
    masks = {"ssh": np.ones((NY - 4, NX - 4), dtype=bool)}
    rows = gate._closure_rows(
        Context(), None, arrays, masks, 1, 1, "LEGO_CHAINED", "output")
    assert len(rows) == 5
    assert {row["classification"] for row in rows} == {"UNMEASURED_WITH_SPEC"}
    assert all(row["entry_mode"] == "LEGO_CHAINED" for row in rows)


def test_chained_observer_scores_the_public_step_seeded_context():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    seed = "state = model._seed_tke_preclosure_carry(state)"
    step = "trace = jax.device_get(model.step("
    output = "np.asarray(card.recipe.grid.area_T), state, kt,"
    assert source.index(seed) < source.index(step) < source.rindex(output)
