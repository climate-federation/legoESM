from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round131_step16_walk_gate as gate,
)


def _summary(nonfinite: bool = False) -> dict[str, object]:
    first = ({"field": "T", "index": list(gate.TARGET), "value": "nan"}
             if nonfinite else None)
    return {
        "fields": ["T", "S"],
        "nonfinite": {"T": int(nonfinite), "S": 0},
        "nonfinite_total": int(nonfinite),
        "first_nonfinite": first,
        "finite_max_abs": {"T": 1.0, "S": 1.0},
        "target_jik": list(gate.TARGET),
        "target_values": {"T": "nan" if nonfinite else "0.0", "S": "0.0"},
    }


def _report(first: str = "stage3_advection_content") -> dict[str, object]:
    boundaries = {
        name: _summary(gate.BOUNDARIES.index(name) >= gate.BOUNDARIES.index(first))
        for name in gate.BOUNDARIES
    }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 15,
        "boundary_order": list(gate.BOUNDARIES),
        "boundaries": boundaries,
        "first_nonfinite_boundary": first,
        "ordinary_repeat_state_equal": {name: True for name in gate.FIELDS},
        "instrument_limit": "synthetic fixture",
    }


def test_classify_accepts_source_ordered_walk() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND131_STEP16_WALK"
    assert result["prediction_ledger"]["R131-P2"]["status"] == "CONFIRMED"


def test_earlier_boundary_is_retained_as_refutation() -> None:
    result = gate.classify(_report("stage2_tracer"))
    assert result["prediction_ledger"]["R131-P2"]["status"] == "REFUTED"
    assert result["first_nonfinite_boundary"] == "stage2_tracer"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_boundary_summary_finds_frozen_field_and_index_order() -> None:
    temperature = np.zeros((2, 50, 1), dtype=np.float64)
    salinity = np.zeros_like(temperature)
    temperature[1, 49, 0] = np.nan
    salinity[0, 0, 0] = np.inf
    row = gate.boundary_summary({"T": temperature, "S": salinity})
    assert row["nonfinite"] == {"T": 1, "S": 1}
    assert row["first_nonfinite"] == {
        "field": "T", "index": [1, 49, 0], "value": "nan"}


def test_bit_comparison_includes_nan_payload() -> None:
    left = np.array([np.nan, 1.0], dtype=np.float64)
    right = copy.deepcopy(left)
    assert gate._bit_equal(left, right)
    right[1] = np.nextafter(right[1], np.inf)
    assert not gate._bit_equal(left, right)
