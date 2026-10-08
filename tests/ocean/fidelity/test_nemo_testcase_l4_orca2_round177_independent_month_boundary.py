"""Controls for the round-177 independent month boundary gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round177_independent_month_boundary as gate,
)


def _report() -> dict[str, object]:
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "unmeasured_features": ["linear_implicit_bottom_drag"],
        "initial_entry": {
            name: {"bit_exact": True, "unequal": 0}
            for name in ("T", "S", "u", "v", "ssh")
        },
        "steps_completed": 36,
        "first_nonfinite": {
            "step": 36, "field": "T", "index": [86, 159, 0],
            "value": float("nan"),
        },
        "runtime_refusal": None,
    }


def test_classify_accepts_predicted_boundary() -> None:
    result = gate.classify(_report())
    assert result["status"] == "MEASURED_R177_FIRST_NONFINITE"
    assert result["prediction_ledger"]["R177-P4"] == "CONFIRMED"


def test_changed_boundary_is_retained_as_refutation() -> None:
    report = _report()
    report["steps_completed"] = 40
    report["first_nonfinite"]["step"] = 40
    result = gate.classify(report)
    assert result["prediction_ledger"]["R177-P4"] == "REFUTED"


def test_complete_finite_month_is_valid_refutation() -> None:
    report = _report()
    report["steps_completed"] = 240
    report["first_nonfinite"] = None
    result = gate.classify(report)
    assert result["status"] == "MEASURED_R177_MONTH_COMPLETE"
    assert result["prediction_ledger"]["R177-P4"] == "REFUTED"


def test_live_thickness_runtime_refusal_is_measured() -> None:
    report = _report()
    report["steps_completed"] = 90
    report["first_nonfinite"] = None
    report["runtime_refusal"] = {
        "step": 91,
        "message": "raw-mesh e3w_int must contain only finite values > 0",
    }
    result = gate.classify(report)
    assert result["status"] == "MEASURED_R177_RUNTIME_REFUSAL"
    assert result["prediction_ledger"]["R177-P4"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_control_plant_refuses(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)
