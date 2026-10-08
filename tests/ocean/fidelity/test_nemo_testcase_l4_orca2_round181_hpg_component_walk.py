from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round181_hpg_component_walk as gate,
)


def _row(*, at_floor: bool, cells: int = 1) -> dict:
    return {
        "at_floor": at_floor, "bit_exact": at_floor,
        "differing_cells": 0 if at_floor else cells,
        "compared_cells": cells, "absolute_max": 0.0 if at_floor else 1.0,
        "rms": 0.0 if at_floor else 1.0, "argmax": [0],
        "candidate_at_argmax": 0.0, "oracle_at_argmax": 0.0,
    }


def _report() -> dict:
    components = {name: _row(at_floor=name != "zhpj")
                  for name in gate.COMPONENT_ORDER}
    inputs = {name: _row(at_floor=name != "north_e3w")
              for name in gate.INPUT_ORDER}
    return {
        "claim_label": "independent hierarchy rung 0",
        "record_census": {"rank_coverage": "exactly-once"},
        "record_control": {"bit_exact": False, "differing_cells": 1},
        "component_order": list(gate.COMPONENT_ORDER),
        "input_order": list(gate.INPUT_ORDER),
        "target": {"cells": 68, "row": 147, "wet_levels": 1319},
        "recorded_input_self_replay": {
            "rank0": {name: _row(at_floor=True) for name in gate.COMPONENT_ORDER},
            "rank1": {name: _row(at_floor=True) for name in gate.COMPONENT_ORDER},
        },
        "candidate_calibration": {"reproduces_round179": True},
        "component_rows": components,
        "first_component": {"boundary": "zhpj", **components["zhpj"]},
        "input_rows": inputs,
        "first_input": {"boundary": "north_e3w", **inputs["north_e3w"]},
        "north_only_replay": _row(at_floor=True),
        "endpoint_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_classifier_accepts_registered_component_result():
    result = gate.classify(_report())
    assert result["status"] == "HELD_FIRST_HPG_V_COMPONENT"
    assert result["prediction_ledger"]["R181-P3"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_later_component_refutes_prediction_without_changing_selector():
    report = _report()
    report["component_rows"]["zhpj"] = _row(at_floor=True)
    report["component_rows"]["zvap"] = _row(at_floor=False)
    report["first_component"] = {
        "boundary": "zvap", **report["component_rows"]["zvap"]}
    result = gate.classify(report)
    assert result["prediction_ledger"]["R181-P3"] == "REFUTED"


def test_local_input_refutes_north_prediction():
    report = _report()
    report["input_rows"]["north_e3w"] = _row(at_floor=True)
    report["input_rows"]["current_e3w"] = _row(at_floor=False)
    report["first_input"] = {
        "boundary": "current_e3w", **report["input_rows"]["current_e3w"]}
    result = gate.classify(report)
    assert result["prediction_ledger"]["R181-P4"] == "REFUTED"
