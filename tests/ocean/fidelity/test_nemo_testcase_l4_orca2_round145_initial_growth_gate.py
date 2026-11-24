import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round145_initial_growth_gate as gate,
)


def _row(delta=0.0, nonfinite=None):
    return {
        "count": 1,
        "bit_exact": delta == 0.0 and nonfinite is None,
        "max_abs": delta,
        "argmax_k": 0,
        "lego_at_argmax": delta,
        "nemo_at_argmax": 0.0,
        "first_nonfinite_k": nonfinite,
        "over_floor": nonfinite is not None or delta > gate.shared.FLOOR,
    }


def _report():
    steps = {}
    for step in gate.STEPS:
        steps[str(step)] = {
            "rows": {name: _row() for name in gate.shared.ROW_ORDER},
            "passivity": {name: True for name in (*gate.shared.rung0.FIELDS,
                                                    "uu_b", "vv_b")},
        }
    steps["1"]["rows"]["ssh_after"] = _row(gate.shared.FLOOR * 2.0)
    first = {"step": 1, "row": "ssh_after",
             **steps["1"]["rows"]["ssh_after"]}
    return {
        "claim_label": "independent", "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "target_ji": list(gate.shared.TARGET), "floor": gate.shared.FLOOR,
        "row_order": list(gate.shared.ROW_ORDER),
        "admission": {"status": "PASS_R144_INITIAL_GROWTH_RECORD"},
        "steps": steps, "first_over_floor": first,
    }


def test_classify_accepts_bracketed_step_one_boundary():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND145_INITIAL_GROWTH_WALK"
    assert result["prediction_ledger"]["R145-P2"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["R145-P3"]["status"] == "CONFIRMED"


def test_different_boundary_is_retained_as_refutation():
    report = _report()
    report["steps"]["1"]["rows"]["ssh_after"] = _row()
    report["steps"]["2"]["rows"]["r3t_entry"] = _row(1.0)
    report["first_over_floor"] = {
        "step": 2, "row": "r3t_entry",
        **report["steps"]["2"]["rows"]["r3t_entry"],
    }
    result = gate.classify(report)
    assert result["prediction_ledger"]["R145-P3"]["status"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_first_over_floor_is_step_then_source_order():
    report = _report()
    altered = copy.deepcopy(report["steps"])
    altered["1"]["rows"]["r3t_after"] = _row(1.0)
    assert gate.first_over_floor(altered)["row"] == "ssh_after"
