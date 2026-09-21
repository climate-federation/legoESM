"""Non-vacuous controls for the rung-3.2 written-order replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPLAY_PATH = Path(__file__).parents[3] / (
    "scripts/validate/ocean_fidelity/testcases/nemo_si3_phase2_adv2d_replay.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_phase2_adv2d_replay", REPLAY_PATH)
assert SPEC and SPEC.loader
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_ulp_distance_is_exact_and_non_vacuous():
    value = np.asarray([1.0, -1.0, 0.0], dtype=np.float64)
    neighbor = np.nextafter(value, np.asarray([np.inf, -np.inf, np.inf]))
    assert np.array_equal(replay._ulp_distance(value, value), np.zeros(3, dtype=np.uint64))
    assert np.array_equal(replay._ulp_distance(value, neighbor), np.ones(3, dtype=np.uint64))


def test_written_order_replay_classifies_and_plant_changes_output():
    report = replay.run_replay(replay.ROOT)
    assert report["classification"] in {
        "RE-ASSOCIATION",
        "INHERITED_STEP_ENTRY_DEBT",
        "IMPLEMENTATION_OR_UNMEASURED_MOMENT_INPUT_DEBT",
    }
    # The source-exact Prather arm moved this historical row below the gate
    # bar, but its 24-ULP residual remains the replay's nonzero owner.
    assert report["target"]["production_normalized_max_abs"] > 0.0
    assert report["target"]["production_max_ulp"] > replay.ULP_LIMIT
    assert len(report["tracer_rows"]) == 16
    assert len(report["input_tracer_rows"]) == 16
    assert len(report["target_input_history"]) == replay.ORACLE_INPUT_KT
    assert report["first_target_input_over_two_ulp"] is not None
    assert report["initial_salt_reciprocal_order"]["name"] == replay.TARGET_TRACER
    assert len(report["moment_rows"]) == 80
    assert report["planted_control"]["status"] == "RED"
    assert report["planted_control"]["changed_v_i_max_ulp"] > 0
