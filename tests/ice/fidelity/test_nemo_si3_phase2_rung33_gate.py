"""Controls for the partial SI3 rung-3.3 gate."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest

_GATE_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung33_gate.py"
)
_SPEC = importlib.util.spec_from_file_location("nemo_si3_phase2_rung33_gate", _GATE_PATH)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)
_AGRID_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung33_agrid_guard.py"
)
_AGRID_SPEC = importlib.util.spec_from_file_location("rung33_agrid_guard", _AGRID_PATH)
assert _AGRID_SPEC and _AGRID_SPEC.loader
agrid_guard = importlib.util.module_from_spec(_AGRID_SPEC)
_AGRID_SPEC.loader.exec_module(agrid_guard)
_REPLAY_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_rung33_replay.py"
)
_REPLAY_SPEC = importlib.util.spec_from_file_location("rung33_replay", _REPLAY_PATH)
assert _REPLAY_SPEC and _REPLAY_SPEC.loader
replay = importlib.util.module_from_spec(_REPLAY_SPEC)
_REPLAY_SPEC.loader.exec_module(replay)


def test_status_and_exit_inputs_are_derived_from_rows_and_coverage():
    at_bar = [{"status": "AT-BAR"}]
    debt = [{"status": "AT-BAR"}, {"status": "DEBT"}]
    assert gate._status_from_rows(at_bar) == "AT-BAR"
    assert gate._status_from_rows(at_bar, {"later": "UNMEASURED"}) == "UNMEASURED"
    assert gate._status_from_rows(debt, {"later": "UNMEASURED"}) == "DEBT"


def test_partial_gate_has_exact_registry_and_loud_unmeasured_boundary():
    report, code = gate.run_gate()
    assert code == 1
    assert report["status"] == "UNMEASURED"
    assert report["numeric_status"] == "AT-BAR"
    assert len(report["rows"]) == 68
    assert report["debt"] == []
    for name in ("u_ice", "v_ice", "stress1_i", "stress2_i", "stress12_i"):
        row = next(
            item
            for item in report["rows"]
            if item["name"] == f"trajectory.post_step_00000001.{name}"
        )
        assert row["bitwise_nonzero_over_n"] == "0 / 9801"
    assert report["binding_control"]["status"] == "RED_AS_REQUIRED"
    assert report["binding_control"]["scored_row"]["status"] == "DEBT"
    assert report["unmeasured"]


def test_geometry_plant_goes_red_and_registry_omission_is_fatal():
    report, _ = gate.run_gate(plant_geometry=True)
    row = next(item for item in report["rows"] if item["name"] == "geometry.e1t")
    assert row["status"] == "DEBT"
    with pytest.raises(gate.GateError, match="tripwire registry mismatch"):
        gate._validate_registry(report["rows"][:-1])


def test_small_velocity_row_reports_relative_error_beside_gate_metric():
    rows = []
    dtypes = {}
    oracle = np.asarray([0.024, -0.012], dtype=np.float64)
    candidate = oracle.copy()
    candidate[0] += 2.4e-15
    row = gate._score(rows, dtypes, "control.v_ice", oracle, candidate)
    assert row["normalized_max_abs"] < row["relative_max_abs"]
    assert row["relative_max_abs"] == pytest.approx(
        row["max_abs"] / row["oracle_max_abs"]
    )


def test_existing_a_grid_solver_is_byte_identical_to_preregister_boundary():
    report = agrid_guard.run_guard()
    assert report["byte_identical"] is True
    assert set(report["schemes"]) == {"evp", "mevp"}
    for result in report["schemes"].values():
        assert result["before_sha256"] == result["after_sha256"]


def test_source_rounded_replay_is_bit_exact_from_oracle_entry():
    report, code = replay.run_replay()
    assert code == 0
    assert report["status"] == "BIT-EXACT"
    predicates = report["classification_predicates"]
    assert predicates["all_written_order_active_stress_rows_at_most_two_ulp"] is True
    assert predicates["all_production_carries_bit_exact_after_100"] is True
    assert predicates["all_replay_carries_bit_exact_after_100"] is True
    written = report["first_active_stress_subcycle_2_replay_vs_written_order_production"]
    assert all(written[name]["max_ulp"] == 0 for name in ("stress1", "stress2", "stress12"))
    assert all(
        row["bitwise_nonzero_over_n"] == "0 / 9801"
        for row in report["hundred_subcycles_production_vs_oracle"].values()
    )


def test_rung33_restart_requires_stresses_and_moments_with_fp64_dtype():
    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
        build_ice_adv2d_rhg_card,
        load_ice_adv2d_rhg_restart,
        save_ice_adv2d_rhg_restart,
    )

    _, entry = gate.oracle_gate.read_frame(
        gate.ROOT / "oracle_ice_step_entry_kt00000001.bin"
    )
    card = build_ice_adv2d_rhg_card(
        gate.oracle_surface_temperature_c(gate.ROOT), entry
    )
    with tempfile.TemporaryDirectory(prefix="rung33-test-") as directory:
        path = Path(directory) / "state.npz"
        save_ice_adv2d_rhg_restart(path, card, card.initial_state, completed_steps=0)
        restored, clock = load_ice_adv2d_rhg_restart(path, card)
        assert clock == 0
        assert all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(card.initial_state, restored, strict=True)
        )
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload.pop("stress12_i")
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="missing=.*stress12_i"):
            load_ice_adv2d_rhg_restart(path, card)

        save_ice_adv2d_rhg_restart(path, card, card.initial_state, completed_steps=0)
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: archive[name].copy() for name in archive.files}
        payload["moment_4"] = payload["moment_4"].astype(np.float32)
        np.savez(path, **payload)
        with pytest.raises(ValueError, match="moment_4 shape/dtype mismatch"):
            load_ice_adv2d_rhg_restart(path, card)
