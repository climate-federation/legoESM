import copy
import inspect

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round143_growth_walk as gate,
)


def _row(*, delta=0.0, nonfinite=None):
    return {
        "count": 1,
        "bit_exact": delta == 0.0 and nonfinite is None,
        "max_abs": delta,
        "argmax_k": 0,
        "lego_at_argmax": delta,
        "nemo_at_argmax": 0.0,
        "first_nonfinite_k": nonfinite,
        "over_floor": nonfinite is not None or delta > gate.FLOOR,
    }


def _report():
    steps = {}
    for step in gate.STEPS:
        rows = {name: _row() for name in gate.ROW_ORDER}
        steps[str(step)] = {
            "rows": rows,
            "passivity": {name: True for name in (*gate.rung0.FIELDS, "uu_b", "vv_b")},
        }
    steps["30"]["rows"]["ssh_entry"] = _row(delta=gate.FLOOR * 2.0)
    steps["36"]["rows"]["zFw_stage1"] = _row(delta=0.0, nonfinite=4)
    first = {"step": 30, "row": "ssh_entry", **steps["30"]["rows"]["ssh_entry"]}
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "target_ji": list(gate.TARGET),
        "floor": gate.FLOOR,
        "row_order": list(gate.ROW_ORDER),
        "admission": {"status": "PASS_R141_GROWTH_RECORD"},
        "steps": steps,
        "first_over_floor": first,
    }


def test_classify_accepts_source_ordered_growth_walk():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND143_GROWTH_WALK"
    assert result["prediction_ledger"]["R143-P3"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["R143-P4"]["status"] == "CONFIRMED"


def test_earlier_transport_row_is_retained_as_refutation():
    report = _report()
    report["steps"]["30"]["rows"]["ssh_entry"] = _row()
    report["steps"]["31"]["rows"]["un_adv_w"] = _row(delta=1.0)
    report["first_over_floor"] = {
        "step": 31, "row": "un_adv_w",
        **report["steps"]["31"]["rows"]["un_adv_w"],
    }
    result = gate.classify(report)
    assert result["prediction_ledger"]["R143-P3"]["status"] == "REFUTED"
    assert result["prediction_ledger"]["R143-P4"]["status"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_score_values_does_not_hide_nonfinite():
    row = gate.score_values(np.array([1.0, np.inf]), np.array([1.0, 2.0]))
    assert row["first_nonfinite_k"] == 1
    assert row["over_floor"] is True


def test_first_over_floor_uses_step_then_source_order():
    report = _report()
    altered = copy.deepcopy(report["steps"])
    altered["30"]["rows"]["r3t_entry"] = _row(delta=1.0)
    assert gate.first_over_floor(altered)["row"] == "ssh_entry"


def test_live_trace_requires_tke_payload_only_when_tke_is_prognostic():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    source = inspect.getsource(LatLonCGridOceanModel._step_impl)
    guard = source.index("self._tke_prognostic_active()")
    tke_entry = source.index("_nemo_ws_live_tke_entry is None", guard)
    incomplete = source.index(
        'raise ValueError("live WS-RK3 operand trace is incomplete")',
        tke_entry,
    )
    assert guard < tke_entry < incomplete
