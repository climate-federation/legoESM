from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import jax
import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round83_slow_forcing_walk.py"
)
SPEC = importlib.util.spec_from_file_location("round83_slow_forcing_walk", SCRIPT)
assert SPEC and SPEC.loader
WALK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WALK)


def _round139_payload(arrays: dict[str, np.ndarray]) -> bytes:
    header = struct.pack(
        "=11i", 1, WALK.ROUND139_KT, WALK.ROUND139_KMM,
        WALK.round81.JPI, WALK.round81.JPJ, 64,
        WALK.round81.NTSI, WALK.round81.NTEI,
        WALK.round81.NTSJ, WALK.round81.NTEJ,
        len(WALK.ROUND139_FIELDS),
    )
    body = b"".join(
        np.asarray(arrays[name], dtype=np.float64).T.tobytes(order="F")
        for name in WALK.ROUND139_FIELDS
    )
    return WALK.ROUND139_MAGIC.ljust(16).encode("ascii") + header + body


def _round139_fields() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    shape = (WALK.round81.NY, WALK.round81.NX)
    base = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 10.0
    cor_u = np.full(shape, 0.25, dtype=np.float64)
    cor_v = np.full(shape, -0.5, dtype=np.float64)
    u_mask = np.ones(shape, dtype=np.float64)
    v_mask = np.ones(shape, dtype=np.float64)
    u_mask[0, 0] = 0.0
    v_mask[-1, -1] = 0.0
    fields = {
        "incoming_u": base,
        "incoming_v": -base,
        "coriolis_u": cor_u,
        "coriolis_v": cor_v,
        "final_u": base - cor_u * u_mask,
        "final_v": -base - cor_v * v_mask,
    }
    return fields, u_mask, v_mask


def _round140_rhs_payload() -> bytes:
    nx, ny, nz = 36, 26, 31
    header = struct.pack(
        "=8i", WALK.ROUND140_RHS_VERSION, WALK.ROUND139_KT,
        1, 3, nx, ny, nz, 64)
    sizes = struct.pack(
        "=7i", *((nx * ny * nz,) * 6 + ((nx - 4) * (ny - 4),)))
    zero3 = np.zeros((ny, nx, nz), dtype=np.float64)
    mask3 = np.ones((ny, nx, nz), dtype=np.float64)
    zero_full = np.zeros((ny, nx), dtype=np.float64)
    one_full = np.ones((ny, nx), dtype=np.float64)
    zero_interior = np.zeros((ny - 4, nx - 4), dtype=np.float64)

    def field3(values):
        return np.asarray(values).transpose(1, 0, 2).tobytes(order="F")

    def full2(values):
        return np.asarray(values).T.tobytes(order="F")

    def interior2(values):
        return np.asarray(values).T.tobytes(order="F")

    body = b"".join((
        field3(zero3), field3(zero3), field3(mask3),
        field3(zero3), field3(zero3), field3(mask3),
        interior2(zero_interior), interior2(zero_interior),
        full2(one_full), full2(one_full),
        interior2(zero_interior), interior2(zero_interior),
        full2(zero_full), full2(zero_full),
        np.asarray([1.0], dtype=np.float64).tobytes(),
        full2(zero_full), full2(zero_full),
        full2(one_full), full2(one_full),
        interior2(zero_interior), interior2(zero_interior),
    ))
    return (WALK.ROUND140_RHS_MAGIC.ljust(16).encode("ascii")
            + header + sizes + body)


def test_bottom_value_selects_deepest_wet_face_level() -> None:
    values = np.array([[[1.0, 2.0, 99.0], [3.0, 88.0, 77.0]]])
    mask = np.array([[[True, True, False], [True, False, False]]])
    np.testing.assert_array_equal(WALK.bottom_value(values, mask), [[2.0, 3.0]])


def test_source_chain_preserves_compiled_statement_association() -> None:
    # The final model level is NEMO's non-contributing jpk slot.
    rhs = np.array([[[1.0, 2.0, 99.0]]])
    e3 = np.array([[[3.0, 4.0, 99.0]]])
    mask3 = np.ones_like(rhs)
    chain = WALK.source_chain(
        rhs=rhs,
        e3=e3,
        mask3=mask3,
        reciprocal_ref=np.array([[0.5]]),
        inverse_depth=np.array([[0.25]]),
        drag_coefficient=np.array([[2.0]]),
        bottom_velocity=np.array([[5.0]]),
        barotropic_velocity=np.array([[1.0]]),
        rho_reciprocal=np.float64(0.1),
        stress=np.array([[8.0]]),
        coriolis=np.array([[0.75]]),
        mask2=np.ones((1, 1)),
    )
    # depth=(3*1 + 4*2)*.5=5.5; drag=(.25*2)*(5-1)=2;
    # wind=(.1*8)*.25=.2; final=5.5+2+.2-.75=6.95.
    np.testing.assert_array_equal(chain["depth_mean"], [[5.5]])
    np.testing.assert_array_equal(chain["post_drag"], [[7.5]])
    np.testing.assert_allclose(chain["post_wind"], [[7.7]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(chain["final"], [[6.95]], rtol=0.0, atol=0.0)


def test_direct_record_replay_retains_noncontributing_bottom_slot() -> None:
    recorded = np.arange(6 * 7 * 31.0).reshape(6, 7, 31)
    assert WALK.owned3(recorded).shape == (2, 3, 30)
    retained = WALK.owned3_with_bottom(recorded)
    assert retained.shape == (2, 3, 31)
    np.testing.assert_array_equal(retained[..., -1], recorded[2:-2, 2:-2, -1])


def test_comparison_detects_one_ulp_on_an_active_cell() -> None:
    oracle = np.array([[1.0, 2.0]])
    candidate = oracle.copy()
    candidate[0, 1] = np.nextafter(candidate[0, 1], np.inf)
    row = WALK.comparison(candidate, oracle, np.array([[True, True]]))
    assert not row["bit_exact"]
    assert row["differing_cells"] == 1
    assert row["absolute_max"] > 0.0


def test_round139_record_reader_and_replay_cover_all_six_fields() -> None:
    fields, u_mask, v_mask = _round139_fields()
    payload = _round139_payload(fields)
    assert len(payload) == WALK.ROUND139_EXPECTED_SIZE
    parsed = WALK.read_round139_record_bytes(payload)
    for name in WALK.ROUND139_FIELDS:
        np.testing.assert_array_equal(parsed[name], fields[name])
    replay = WALK.validate_round139_record(parsed, u_mask, v_mask)
    assert replay == {
        "u_bit_exact": True,
        "v_bit_exact": True,
        "u_wet_faces": int(np.count_nonzero(u_mask)),
        "v_wet_faces": int(np.count_nonzero(v_mask)),
    }


def test_round139_record_reader_refuses_header_truncation_and_trailing_byte() -> None:
    fields, _, _ = _round139_fields()
    payload = _round139_payload(fields)
    bad_header = bytearray(payload)
    bad_header[16:20] = struct.pack("=i", 2)
    with pytest.raises(RuntimeError, match="bad Round-139 header"):
        WALK.read_round139_record_bytes(bytes(bad_header))
    with pytest.raises(RuntimeError, match="truncated Round-139 final_v"):
        WALK.read_round139_record_bytes(payload[:-1])
    with pytest.raises(RuntimeError, match="trailing Round-139 payload"):
        WALK.read_round139_record_bytes(payload + b"x")


def test_round139_replay_control_detects_one_ulp_in_consumed_result() -> None:
    fields, u_mask, v_mask = _round139_fields()
    parsed = WALK.read_round139_record_bytes(_round139_payload(fields))
    planted = np.array(parsed["final_u"], copy=True)
    location = tuple(int(value) for value in np.argwhere(u_mask != 0.0)[0])
    planted[location] = np.nextafter(planted[location], np.float64(np.inf))
    with pytest.raises(RuntimeError, match="does not replay bit for bit"):
        WALK.validate_round139_record(
            parsed, u_mask, v_mask, final_u=planted)


def test_round139_closed_run_manifest_and_markers_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in WALK.ROUND139_MANIFEST_MEMBERS:
        path = tmp_path / name
        if name == "round139_parent_record_validation.json":
            path.write_text(json.dumps({"status": "AT-BAR"}))
        elif name == "round139_record_validation.json":
            path.write_text(json.dumps({"status": "PASS"}))
        elif name == "round139_passive_admission_plant.log":
            path.write_text("STATUS PLANT-FIRED: passive-admission\n")
        elif name.endswith("_plant.log"):
            path.write_text("ROUND139 STATUS PLANT-FIRED\n")
        else:
            path.write_bytes((name + "\n").encode())
    (tmp_path / "run.user.time.log").write_text("NEMO_DONE\nRUN_DONE\n")
    (tmp_path / "run.user.stdout.log").write_text("STOP 0\n")
    inherited = {
        "ROUND139_PARENT_RESTART_SHA256":
            "GYRE_OMIP_L2_P3_00001080_restart.nc",
        "ROUND139_PARENT_PROCESS_SHA256":
            "oracle_process_budget_kt00001081.bin",
        "ROUND139_PARENT_EXTERNAL_SHA256": WALK.round81.DEVELOPED_RECORD,
        "ROUND139_PARENT_QCO_SHA256": WALK.round81.DEVELOPED_QCO_RECORD,
    }
    for constant, name in inherited.items():
        monkeypatch.setattr(WALK, constant, WALK.sha256(tmp_path / name))
    manifest = "".join(
        f"{WALK.sha256(tmp_path / name)}  {name}\n"
        for name in WALK.ROUND139_MANIFEST_MEMBERS
    )
    (tmp_path / WALK.ROUND139_MANIFEST).write_text(manifest)
    report = WALK._verify_round139_closed_run(tmp_path)
    assert report["manifest_members"] == len(WALK.ROUND139_MANIFEST_MEMBERS)

    changed = tmp_path / WALK.ROUND139_MANIFEST_MEMBERS[0]
    changed.write_bytes(changed.read_bytes() + b"plant")
    with pytest.raises(RuntimeError, match="manifest member changed"):
        WALK._verify_round139_closed_run(tmp_path)


def test_round140_registry_rejects_a_missing_operand() -> None:
    WALK._validate_round140_registry()
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_round140_registry(WALK.ROUND140_OPERAND_REGISTRY[:-1])


def test_round140_rhs_reader_replay_and_registry_are_fail_closed() -> None:
    payload = _round140_rhs_payload()
    assert len(payload) == WALK.ROUND140_RHS_EXPECTED_SIZE
    record = WALK.read_round140_rhs_bytes(payload)
    assert tuple(record["fields"]) == WALK.ROUND140_RHS_FIELDS
    assert all(
        row["bit_exact"]
        for row in WALK.validate_round140_rhs_replay(record).values())
    WALK._validate_round140_rhs_registry()
    with pytest.raises(RuntimeError, match="developed-RHS registry changed"):
        WALK._validate_round140_rhs_registry(WALK.ROUND140_RHS_REGISTRY[:-1])


def test_round140_rhs_reader_and_replay_plants_fire() -> None:
    payload = _round140_rhs_payload()
    bad_header = bytearray(payload)
    bad_header[16:20] = struct.pack("=i", WALK.ROUND140_RHS_VERSION + 1)
    with pytest.raises(RuntimeError, match="bad Round-140 RHS header"):
        WALK.read_round140_rhs_bytes(bytes(bad_header))
    with pytest.raises(RuntimeError, match="record size changed"):
        WALK.read_round140_rhs_bytes(payload[:-1])
    record = WALK.read_round140_rhs_bytes(payload)
    planted = np.array(record["fields"]["post_wind_u"], copy=True)
    planted[0, 0] = np.nextafter(planted[0, 0], np.float64(np.inf))
    with pytest.raises(RuntimeError, match="source replay is not bit exact"):
        WALK.validate_round140_rhs_replay(record, post_wind_u=planted)


def test_round140_rhs_replay_includes_the_deepest_physical_level() -> None:
    e3 = np.zeros((1, 1, 30), dtype=np.float64)
    rhs = np.zeros_like(e3)
    mask = np.ones_like(e3)
    e3[..., -1] = 2.0
    rhs[..., -1] = 3.0
    result = WALK._round140_source_sum(
        e3, rhs, mask, np.asarray([[0.5]], dtype=np.float64))
    np.testing.assert_array_equal(result, [[3.0]])


def test_round117_compiled_accumulator_order_keeps_after_adv_identity() -> None:
    terms = [np.asarray([value], dtype=np.float64) for value in range(1, 11)]
    rows = jax.device_get(jax.jit(WALK.round117_source_order_accumulators)(
        *terms))
    assert rows["after_hpg_u"][0] == 1.0
    assert rows["after_ldf_u"][0] == 4.0
    assert rows["after_vor_u"][0] == 9.0
    assert rows["after_keg_u"][0] == 16.0
    assert rows["after_zad_u"][0] == 25.0
    np.testing.assert_array_equal(rows["after_adv_u"], rows["after_zad_u"])


def test_round117_first_nonbit_obeys_boundary_then_face_order() -> None:
    exact = {"bit_exact": True, "absolute_max": 0.0}
    rows = {
        face: {boundary: dict(exact) for boundary in WALK.ROUND117_BOUNDARIES}
        for face in ("u", "v")
    }
    rows["v"]["after_hpg"] = {"bit_exact": False, "absolute_max": 2.0}
    rows["u"]["after_ldf"] = {"bit_exact": False, "absolute_max": 3.0}
    first = WALK.round117_first_nonbit(rows)
    assert first["boundary"] == "after_hpg"
    assert first["face"] == "v"


def test_round117_native_override_retains_excluded_halo() -> None:
    full_u = np.arange(8.0).reshape(2, 4)
    native_u = np.full((2, 3), 42.0)
    replaced_u = WALK._full_from_native(full_u, native_u, "u")
    np.testing.assert_array_equal(replaced_u[:, 0], full_u[:, 0])
    np.testing.assert_array_equal(replaced_u[:, 1:], native_u)

    full_v = np.arange(8.0).reshape(4, 2)
    native_v = np.full((3, 2), -7.0)
    replaced_v = WALK._full_from_native(full_v, native_v, "v")
    np.testing.assert_array_equal(replaced_v[0], full_v[0])
    np.testing.assert_array_equal(replaced_v[1:], native_v)


def test_round117_one_ulp_plant_survives_subtract() -> None:
    incoming = np.asarray([[1.0, 2.0]], dtype=np.float64)
    coriolis = np.asarray([[0.25, 0.5]], dtype=np.float64)
    mask = np.asarray([[True, True]])
    planted, location, _ = WALK._round117_propagating_ulp(
        incoming, coriolis, mask)
    assert np.count_nonzero(planted.view(np.uint64) != incoming.view(np.uint64)) == 1
    assert ((planted - coriolis)[location].view(np.uint64)
            != (incoming - coriolis)[location].view(np.uint64))


def test_round119_hpg_ulp_plant_survives_every_source_boundary() -> None:
    hpg = np.asarray([[[1.0, 2.0]]], dtype=np.float64)
    zero = np.zeros_like(hpg)
    active = np.asarray([[[True, False]]])
    planted, location, _ = WALK._round119_propagating_hpg_ulp(
        hpg, zero, zero, zero, zero, active)
    assert location == (0, 0, 0)
    assert np.count_nonzero(planted.view(np.uint64) != hpg.view(np.uint64)) == 1
    value = planted[location]
    ordinary = hpg[location]
    for term in (zero, zero, zero, zero):
        assert value.view(np.uint64) != ordinary.view(np.uint64)
        value = value + term[location]
        ordinary = ordinary + term[location]
    assert value.view(np.uint64) != ordinary.view(np.uint64)


def test_round119_source_order_arm_is_private_and_default_off() -> None:
    hook_name = "nemo_stage_rhs_accumulation_order_arm"
    hooks = WALK.model_module._NEMOWSRK3TestHooks()
    assert getattr(hooks, hook_name) is False
    assert hook_name not in WALK.model_module.LatLonCGridOceanConfig._fields


def test_round120_float_words_preserve_signed_ulp_order() -> None:
    one = np.float64(1.0)
    above = np.nextafter(one, np.float64(np.inf))
    below = np.nextafter(one, np.float64(-np.inf))
    assert WALK._signed_ulp_difference(above, one) == 1
    assert WALK._signed_ulp_difference(below, one) == -1
    negative = np.float64(-1.0)
    negative_above = np.nextafter(negative, np.float64(np.inf))
    negative_below = np.nextafter(negative, np.float64(-np.inf))
    assert WALK._signed_ulp_difference(negative_above, negative) == 1
    assert WALK._signed_ulp_difference(negative_below, negative) == -1
    assert WALK._float64_word(one)["hex"] == "0x3ff0000000000000"


def test_round120_keg_ulp_plant_survives_both_boundaries() -> None:
    addend = np.asarray([[[1.0]]], dtype=np.float64)
    after_vor = np.zeros_like(addend)
    zad = np.zeros_like(addend)
    planted, _ = WALK._round120_keg_ulp_addend(
        addend, after_vor, zad, (0, 0, 0))
    assert np.count_nonzero(
        planted.view(np.uint64) != addend.view(np.uint64)) == 1
    ordinary_keg = after_vor + addend
    planted_keg = after_vor + planted
    assert planted_keg[0, 0, 0].view(np.uint64) != (
        ordinary_keg[0, 0, 0].view(np.uint64))
    assert (planted_keg + zad)[0, 0, 0].view(np.uint64) != (
        (ordinary_keg + zad)[0, 0, 0].view(np.uint64))


def test_round121_w_override_is_private_and_default_off() -> None:
    hook_name = "stage1_zad_w_override"
    hooks = WALK.model_module._NEMOWSRK3TestHooks()
    assert getattr(hooks, hook_name) is None
    assert hook_name not in WALK.model_module.LatLonCGridOceanConfig._fields


def test_round121_proxy_executes_directed_model_only_at_kt2() -> None:
    calls = []

    class FakeModel:
        def __init__(self, *args, _nemo_ws_test_hooks=None, **kwargs):
            self.directed = (
                _nemo_ws_test_hooks is not None
                and _nemo_ws_test_hooks.stage1_zad_w_override is not None)

        def prime_step_caches(self, state):
            calls.append(("prime", self.directed, state))

        def step(self, state):
            calls.append(("step", self.directed, state))
            return state + 1

    audit = []
    proxy_type = WALK._round121_proxy_class(
        FakeModel, np.ones((1, 1, 1)), audit)
    proxy = proxy_type()
    proxy.prime_step_caches(0)
    state = 0
    for _ in range(4):
        state = proxy.step(state)
    assert state == 4
    assert proxy._round121_interventions == [2]
    assert [directed for kind, directed, _ in calls if kind == "step"] == [
        False, True, False, False]
    assert len(audit) == 1


def test_round141_rhs_observer_is_private_default_off_and_registered() -> None:
    hook_name = "slow_forcing_rhs_observer"
    hooks = WALK.model_module._NEMOWSRK3TestHooks()
    assert getattr(hooks, hook_name) is None
    assert hooks.slow_forcing_rhs_observer_face == ""
    assert hook_name not in WALK.model_module.LatLonCGridOceanConfig._fields
    assert "slow_forcing_rhs_observer_face" not in (
        WALK.model_module.LatLonCGridOceanConfig._fields)
    WALK._validate_round140_rhs_registry(WALK.ROUND141_RHS_REGISTRY)
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_round140_rhs_registry(WALK.ROUND141_RHS_REGISTRY[:-1])


def test_round141_depth_reduction_uses_the_production_stacked_sum() -> None:
    h_u = np.asarray([[[2.0, 3.0]]], dtype=np.float64)
    h_v = np.asarray([[[4.0, 1.0]]], dtype=np.float64)
    rhs_u = np.asarray([[[5.0, 7.0]]], dtype=np.float64)
    rhs_v = np.asarray([[[11.0, 13.0]]], dtype=np.float64)
    mask = np.ones((1, 1), dtype=np.float64)
    got_u, got_v = jax.device_get(WALK._round141_depth_reduction(
        h_u, h_v, rhs_u, rhs_v, mask, mask))
    assert got_u[0, 0] == (2.0 * 5.0 + 3.0 * 7.0) / 5.0
    assert got_v[0, 0] == (4.0 * 11.0 + 1.0 * 13.0) / 5.0


def test_round142_rhs_override_is_private_default_off_and_registered() -> None:
    hook_name = "slow_forcing_rhs_override"
    hooks = WALK.model_module._NEMOWSRK3TestHooks()
    assert getattr(hooks, hook_name) is None
    assert hook_name not in WALK.model_module.LatLonCGridOceanConfig._fields
    WALK._validate_round142_registry(WALK.ROUND142_DIRECTED_REGISTRY)
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_round142_registry(
            WALK.ROUND142_DIRECTED_REGISTRY[:-1])


def test_round142_rhs_ulp_plant_changes_one_consumed_word() -> None:
    rhs = np.zeros((1, 1, 30), dtype=np.float64)
    rhs[0, 0, 0] = 1.0
    fields = {
        "rhs_u": rhs,
        "e3u": np.ones_like(rhs),
        "umask": np.ones_like(rhs),
        "r1_hu0": np.ones((1, 1), dtype=np.float64),
    }
    planted, location = WALK._round142_propagating_rhs_ulp(
        fields, np.ones_like(rhs, dtype=bool), np.ones_like(rhs))
    assert location == (0, 0, 0)
    assert np.count_nonzero(
        planted.view(np.uint64) != rhs.view(np.uint64)) == 1
    baseline = np.asarray(WALK._round142_u_depth_reduction(
        np.ones_like(rhs), rhs, np.ones((1, 1), dtype=np.float64)))
    changed = np.asarray(WALK._round142_u_depth_reduction(
        np.ones_like(rhs), planted, np.ones((1, 1), dtype=np.float64)))
    assert changed[0, 0].view(np.uint64) != baseline[0, 0].view(np.uint64)


def test_round143_overrides_are_private_default_off_and_registered() -> None:
    hooks = WALK.model_module._NEMOWSRK3TestHooks()
    for hook_name in (
            "slow_forcing_depth_override", "slow_forcing_drag_override"):
        assert getattr(hooks, hook_name) is None
        assert hook_name not in WALK.model_module.LatLonCGridOceanConfig._fields
    WALK._validate_round143_registry(WALK.ROUND143_DIRECTED_REGISTRY)
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_round143_registry(WALK.ROUND143_DIRECTED_REGISTRY[:-1])


def test_round143_depth_ulp_changes_exactly_one_active_word() -> None:
    depth = np.asarray([[0.0, 2.0], [3.0, 1.0]], dtype=np.float64)
    active = np.asarray([[False, True], [True, True]])
    planted, location = WALK._round143_depth_ulp(depth, active)
    assert location == (1, 0)
    assert np.count_nonzero(
        planted.view(np.uint64) != depth.view(np.uint64)) == 1
    assert planted[location] == np.nextafter(depth[location], np.inf)


def test_round144_override_is_private_default_off_and_registered() -> None:
    hook_name = "slow_forcing_wind_operand_override"
    hooks = WALK.model_module._NEMOWSRK3TestHooks()
    assert getattr(hooks, hook_name) is None
    assert hook_name not in WALK.model_module.LatLonCGridOceanConfig._fields
    WALK._validate_round144_registry(WALK.ROUND144_WIND_REGISTRY)
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_round144_registry(WALK.ROUND144_WIND_REGISTRY[:-1])


def test_round144_stress_ulp_changes_one_consumed_word() -> None:
    fields = {
        "r1_rho0": 1.0,
        "wind_tau_u": np.asarray([[1.0]], dtype=np.float64),
        "wind_r1_hu": np.asarray([[1.0]], dtype=np.float64),
        "post_drag_u": np.asarray([[0.0]], dtype=np.float64),
    }
    planted, location = WALK._round144_stress_ulp(
        fields, np.asarray([[True]]))
    assert location == (0, 0)
    assert np.count_nonzero(
        planted.view(np.uint64)
        != fields["wind_tau_u"].view(np.uint64)) == 1


def test_round145_inverse_depth_ulp_changes_one_consumed_word() -> None:
    fields = {
        "r1_rho0": 1.0,
        "wind_tau_u": np.asarray([[1.0]], dtype=np.float64),
        "wind_r1_hu": np.asarray([[1.0]], dtype=np.float64),
        "post_drag_u": np.asarray([[0.0]], dtype=np.float64),
    }
    planted, location = WALK._round145_inverse_depth_ulp(
        fields, np.asarray([[True]]))
    assert location == (0, 0)
    assert np.count_nonzero(
        planted.view(np.uint64)
        != fields["wind_r1_hu"].view(np.uint64)) == 1


def test_round147_registry_and_adjacent_family_differences() -> None:
    WALK._validate_round147_registry(WALK.ROUND147_FAMILY_REGISTRY)
    with pytest.raises(RuntimeError, match="registry changed"):
        WALK._validate_round147_registry(WALK.ROUND147_FAMILY_REGISTRY[:-1])
    shape = (WALK.round146_family.NY, WALK.round146_family.NX,
             WALK.round146_family.NZ)
    fields = {}
    for value, family in enumerate(WALK.round146_family.FAMILIES, start=1):
        for face in ("u", "v"):
            fields[f"after_{family}_{face}"] = np.full(
                shape, float(value), dtype=np.float64)
    addends = WALK._round147_oracle_addends(fields)
    assert tuple(addends) == WALK.round146_family.FAMILIES
    for pair in addends.values():
        assert np.array_equal(pair[0], np.ones((22, 32, 30)))
        assert np.array_equal(pair[1], np.ones((22, 32, 30)))


def test_round147_directed_rhs_plant_changes_one_consumed_word() -> None:
    shape = (2, 2, 1)
    rhs = np.ones(shape, dtype=np.float64)
    thickness = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=bool)
    planted, location = WALK._round147_directed_rhs_ulp(
        rhs, thickness, active)
    assert len(location) == 3
    assert np.count_nonzero(
        planted.view(np.uint64) != rhs.view(np.uint64)) == 1
