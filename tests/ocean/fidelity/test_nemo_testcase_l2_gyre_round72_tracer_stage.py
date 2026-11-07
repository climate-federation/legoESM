from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round72_tracer_stage.py"
)
SPEC = importlib.util.spec_from_file_location("round72_tracer_stage", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _fields() -> dict[str, np.ndarray]:
    shape = (3, 2, 2)
    kbb = np.arange(np.prod(shape), dtype=np.float64).reshape(shape, order="F")
    rhs = np.full(shape, 0.25, dtype=np.float64)
    kaa = kbb + np.float64(2.0) * rhs
    return {
        "r3t_Kbb": np.zeros(shape[:2], dtype=np.float64),
        "r3t_Kmm": np.zeros(shape[:2], dtype=np.float64),
        "r3t_Kaa": np.zeros(shape[:2], dtype=np.float64),
        "Kbb_T": kbb,
        "after_sbc_T": rhs,
        "Kaa_T": kaa,
    }


def test_replay_update_matches_written_association_and_plant_fires() -> None:
    wet = np.ones((2, 3, 2), dtype=bool)
    replay, expected = GATE.replay_update(
        _fields(), "T", np.float64(2.0), wet)
    np.testing.assert_array_equal(replay, expected)

    replay, planted = GATE.replay_update(
        _fields(), "T", np.float64(2.0), wet, plant_kaa_ulp=True)
    assert np.count_nonzero(replay != planted) == 1


def test_stage2_transport_hook_is_default_inert() -> None:
    hooks = GATE.model_module._NEMOWSRK3TestHooks()
    assert hooks.stage2_tracer_transport_override is None
    target = tuple(np.ones((1, 1, 1), dtype=np.float64) for _ in range(3))
    assert hooks._replace(
        stage2_tracer_transport_override=target
    ).stage2_tracer_transport_override is target
