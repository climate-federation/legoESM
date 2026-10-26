from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
from netCDF4 import Dataset

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round71_independent_month_gate as gate,
)


def _score(value: float = 1.0) -> dict[str, object]:
    return {
        "bit_identical": False,
        "unequal": 1,
        "count": 2,
        "max_abs": value,
        "mean_abs_over_unequal": value,
        "rms": value / 2.0,
        "first_unequal_index": [0],
    }


def _report() -> dict[str, object]:
    rows = {name: _score(5.0 - index) for index, name in enumerate(gate.FIELD_ORDER)}
    ranking_max = [
        {"field": name, "units": gate.FIELD_UNITS[name], **rows[name]} for name in gate.FIELD_ORDER
    ]
    ranking_rms = copy.deepcopy(ranking_max)
    return {
        "format": "test",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed": 240,
        "surface_frames_consumed": 480,
        "unmeasured_features": list(gate.ladder.EXPECTED_UNMEASURED),
        "ten_step_calibration": {"exact": True},
        "chlorophyll_clock": {"status": "RESOLVED_CENTRES_MATCH"},
        "chlorophyll_clock_retraction": {"round66_metrics_reproduced": False},
        "terminal_restart": {
            "status": "BIT_EXACT_ORIENTATION",
            "latitude_bit_identical": True,
            "longitude_bit_identical": True,
            "files": [{"sha256": "1" * 64}, {"sha256": "2" * 64}],
        },
        "terminal": {
            "rows": rows,
            "ranking_by_max_abs": ranking_max,
            "ranking_by_rms": ranking_rms,
        },
    }


def test_classify_accepts_complete_independent_month() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_INDEPENDENT_MONTH_RANKING"
    assert result["prediction_ledger"]["all_five_non_bit"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["temperature_largest_max_abs"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["original_round66_metric_reproduction"] == {
        "status": "REFUTED"
    }


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_classification_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_prediction_refutation_is_recorded_not_refused() -> None:
    report = _report()
    report["terminal"]["ranking_by_max_abs"][0]["field"] = "S"
    report["terminal"]["ranking_by_max_abs"][1]["field"] = "T"
    result = gate.classify(report)
    prediction = result["prediction_ledger"]["temperature_largest_max_abs"]
    assert prediction == {"status": "REFUTED", "observed_largest": "S"}


def test_exact_terminal_row_is_valid_and_refutes_all_non_bit_prediction() -> None:
    report = _report()
    report["terminal"]["rows"]["ssh"] = {
        "bit_identical": True,
        "unequal": 0,
        "count": 2,
        "max_abs": 0.0,
        "mean_abs_over_unequal": 0.0,
        "rms": 0.0,
        "first_unequal_index": None,
    }
    result = gate.classify(report)
    assert result["prediction_ledger"]["all_five_non_bit"]["status"] == "REFUTED"


def test_chlorophyll_clock_switches_month_pair_at_step_125(tmp_path) -> None:
    path = tmp_path / "chlorophyll.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 12)
        dataset.createDimension("y", 148)
        dataset.createDimension("x", 180)
        variable = dataset.createVariable("CHLA", "f8", ("time", "y", "x"))
        variable[:] = np.arange(12, dtype=np.float64)[:, None, None]
    wet = np.ones((148, 180), dtype=bool)

    step2 = gate.ladder.chlorophyll_at_step(path.parent, 2, wet, dt_s=10800.0)
    expected2 = np.float64(11.0) * (np.float64(1.0) - np.float64(251) / 496)
    assert np.array_equal(step2, np.full((148, 180), expected2))

    step240 = gate.ladder.chlorophyll_at_step(path.parent, 240, wet, dt_s=10800.0)
    expected240 = np.float64(231) / np.float64(472)
    assert np.array_equal(step240, np.full((148, 180), expected240))

    with pytest.raises(gate.ladder.GateError):
        gate.ladder.chlorophyll_at_step(path.parent, 241, wet, dt_s=10800.0)


def test_surface_schema_calibration_registers_legacy_only_fields() -> None:
    fields = {
        name: np.full((2, 3), index, dtype=np.float64)
        for index, name in enumerate(gate.surface_gate.FIELDS)
    }
    old_fields = {**fields, "passive_debug_stream": np.ones((2, 3))}

    comparisons, old_only = gate.validate_surface_schema_calibration(
        fields, old_fields, kt=1
    )

    assert comparisons == len(gate.surface_gate.FIELDS)
    assert old_only == ["passive_debug_stream"]


def test_surface_schema_calibration_refuses_changed_consumed_operand() -> None:
    fields = {
        name: np.full((2, 3), index, dtype=np.float64)
        for index, name in enumerate(gate.surface_gate.FIELDS)
    }
    old_fields = {name: values.copy() for name, values in fields.items()}
    old_fields[gate.surface_gate.FIELDS[0]][0, 0] = np.nextafter(
        old_fields[gate.surface_gate.FIELDS[0]][0, 0], np.inf
    )

    with pytest.raises(gate.GateError):
        gate.validate_surface_schema_calibration(fields, old_fields, kt=1)


def test_month_progress_interval_is_diagnostic_only_and_defaults_to_40() -> None:
    parameter = inspect.signature(gate.run_month).parameters["progress_interval"]
    assert parameter.default == 40
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
