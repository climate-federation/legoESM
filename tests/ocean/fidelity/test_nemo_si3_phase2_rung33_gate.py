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


def test_partial_gate_has_exact_registry_and_loud_debt_boundary():
    report, code = gate.run_gate()
    assert code == 1
    assert report["status"] == "UNMEASURED"
    assert report["numeric_status"] == "DEBT"
    assert len(report["rows"]) == 68
    names = {row["name"] for row in report["debt"]}
    assert "trajectory.post_step_00000001.stress1_i" in names
    assert "trajectory.post_step_00000001.stress2_i" in names
    assert "trajectory.post_step_00000001.stress12_i" in names
    assert report["binding_control"]["status"] == "RED_AS_REQUIRED"
    assert report["unmeasured"]


def test_geometry_plant_goes_red_and_registry_omission_is_fatal():
    report, _ = gate.run_gate(plant_geometry=True)
    row = next(item for item in report["rows"] if item["name"] == "geometry.e1t")
    assert row["status"] == "DEBT"
    with pytest.raises(gate.GateError, match="tripwire registry mismatch"):
        gate._validate_registry(report["rows"][:-1])


def test_existing_a_grid_solver_is_byte_identical_to_preregister_boundary():
    report = agrid_guard.run_guard()
    assert report["byte_identical"] is True
    assert report["before_sha256"] == report["after_sha256"]


def test_written_order_replay_classifies_stress_debt_as_reassociation():
    report, code = replay.run_replay()
    assert code == 0
    assert report["status"] == "RE-ASSOCIATION"
    predicates = report["classification_predicates"]
    assert predicates["all_written_order_active_stress_rows_at_most_two_ulp"] is True
    assert predicates["all_replayed_stress_rows_in_1e-14_oracle_debt_class_after_100"] is True
    written = report["first_active_stress_subcycle_2_replay_vs_written_order_production"]
    assert all(written[name]["max_ulp"] == 0 for name in ("stress1", "stress2", "stress12"))


def test_rung33_restart_requires_stresses_and_moments_with_fp64_dtype():
    from legoesm.ice.fidelity.nemo_adv2d_rhg_testcase_recipe import (
        build_ice_adv2d_rhg_card,
        load_ice_adv2d_rhg_restart,
        save_ice_adv2d_rhg_restart,
    )

    card = build_ice_adv2d_rhg_card(gate.oracle_surface_temperature_c(gate.ROOT))
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
