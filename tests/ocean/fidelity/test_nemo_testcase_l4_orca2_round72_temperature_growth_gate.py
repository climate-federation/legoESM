from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round72_temperature_growth_gate as gate,
)


def _weighted(value: float) -> dict[str, object]:
    return {
        "bit_identical": value == 0.0,
        "unequal": 0 if value == 0.0 else 1,
        "count": 2,
        "max_abs": value,
        "weighted_rms": value,
        "weighted_squared_error": value * value,
        "weight_m3": 1.0,
        "support_max_flat_index": 0,
    }


def _growth(names: list[str], owner: str) -> dict[str, object]:
    ordered = [owner, *[name for name in names if name != owner]]
    rows = [
        {"name": name, "delta_weighted_squared_error": float(len(ordered) - index),
         "start": _weighted(0.0), "end": _weighted(1.0)}
        for index, name in enumerate(ordered)
    ]
    return {"intervals": {"0->10": copy.deepcopy(rows),
                           "10->240": copy.deepcopy(rows)},
            "total_0->240": rows}


def _report() -> dict[str, object]:
    source_rows = {
        "entry": {"bit_exact": True},
        "after_advection": {"bit_exact": False},
        "after_sbc": {"bit_exact": False},
        "qco_rk": {"bit_exact": False},
    }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed": 240,
        "unmeasured_features": list(gate.ladder.EXPECTED_UNMEASURED),
        "weighting": {"wet_only": True,
                      "formula": "area_T * dz_ref * is_active",
                      "dtype": "float64"},
        "restart_calibration": {"bit_exact": True},
        "round71_terminal_temperature_reproduced": True,
        "round71_terminal_temperature": copy.deepcopy(gate.ROUND71_T),
        "global_checkpoints": {"0": {}, "10": {}, "240": {}},
        "interval_ranking": [
            {"name": "10->240", "delta_weighted_squared_error": 2.0},
            {"name": "0->10", "delta_weighted_squared_error": 1.0},
        ],
        "depth_growth": _growth(["k=0", "k=1"], "k=0"),
        "region_growth": _growth(
            ["atlantic", "pacific", "indian", "other"], "pacific"),
        "region_partition": {"partition_exact": True,
                             "overlap_wet_cells": 0,
                             "uncovered_wet_cells_after_other": 0},
        "decomposition_closure": {
            "depth": {"absolute_error": 0.0, "tolerance": 1e-12},
            "region": {"absolute_error": 0.0, "tolerance": 1e-12},
        },
        "source_walk": {"rows": source_rows,
                        "first_non_bit_boundary": "after_advection"},
    }


def test_classify_accepts_complete_growth_report() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_TEMPERATURE_GROWTH_RANKING"
    assert all(row["status"] == "CONFIRMED"
               for row in result["prediction_ledger"].values())


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_scientific_prediction_is_refuted_not_refused() -> None:
    report = _report()
    report["depth_growth"] = _growth(["k=0", "k=1"], "k=1")
    result = gate.classify(report)
    assert result["prediction_ledger"]["surface_level_largest"] == {
        "status": "REFUTED", "observed": "k=1"}
