"""Known-answer controls for round 294's raw-hv substep continuation."""

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round294_raw_hv_substep as gate,
)


def _score(unequal=0, maximum=0.0, count=100):
    return {
        "bit_exact": unequal == 0,
        "unequal": unequal,
        "count": count,
        "max_abs": maximum,
        "rms": maximum,
        "argmax": [0, 0],
    }


def _regions(unequal=0, maximum=0.0):
    return {
        "complete": _score(unequal, maximum),
        "interior": _score(),
        "fold_band": _score(unequal, maximum),
    }


def _row(substep, order, name, unequal=0, maximum=0.0):
    return {
        "substep": substep,
        "source_order": order,
        "name": name,
        "face": "V",
        "regions": _regions(unequal, maximum),
    }


def _report():
    rows = []
    for substep in range(1, 4):
        for order, name in enumerate(gate.r292.SOURCE_ORDER):
            unequal = 35 if (substep, name) == (3, "face_depth_v_mid") else 0
            maximum = 0.003 if unequal else 0.0
            rows.append(_row(substep, order, name, unequal, maximum))
    depth = next(row for row in rows
                 if row["substep"] == 1
                 and row["name"] == "face_depth_v_mid")
    first = next(row for row in rows
                 if row["regions"]["complete"]["max_abs"] > gate.FLOOR)
    return {
        "status": "PASS_R294_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1",
        "admission": {"status": "PASS_R280_OMT4_STAGE1_ADMISSION"},
        "passivity": {"T": True, "S": True, "u": True,
                      "v": True, "eta": True},
        "arm_isolation": {
            "u_override_vs_production": _score(),
            "v_override_vs_raw_nemo": _score(),
            "v_override_vs_production": _regions(13, 719.0),
        },
        "source_order": list(gate.r292.SOURCE_ORDER),
        "rows": rows,
        "substep1_depth": depth,
        "substep1_predecessors": [
            row for row in rows if row["substep"] == 1
            and row["source_order"] < depth["source_order"]],
        "first_valued": copy.deepcopy(first),
        "one_bit_control": _score(1, 5e-324, 1),
    }


def test_classifier_names_registered_pair_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R294_SUBSTEP3_FACE_DEPTH_V_MID"
    assert result["prediction_ledger"]["R294-P4"] == "CONFIRMED"
    assert "ROUND245" in result["prediction_ledger"]["R294-P5"]
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)


def test_refuted_prediction_is_retained_with_measured_boundary():
    report = _report()
    for row in report["rows"]:
        row["regions"] = _regions()
    measured = next(row for row in report["rows"]
                    if row["substep"] == 2 and row["name"] == "v_exit")
    measured["regions"] = _regions(4, 1e-4)
    report["first_valued"] = copy.deepcopy(measured)
    result = gate.classify(report)
    assert result["prediction_ledger"]["R294-P4"].startswith("REFUTED")
    assert result["prediction_ledger"]["R294-P5"] == "NOT_CREDITED"


def test_raw_hv_arm_must_not_move_u_or_the_interior():
    report = _report()
    report["arm_isolation"]["v_override_vs_production"]["interior"][
        "unequal"] = 1
    with pytest.raises(gate.GateError, match="interior"):
        gate.classify(report)
