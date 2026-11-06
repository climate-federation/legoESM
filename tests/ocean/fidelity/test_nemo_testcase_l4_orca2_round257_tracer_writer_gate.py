from __future__ import annotations

import ast
import copy
import inspect
import textwrap

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round257_tracer_writer_gate as gate,
)


def _row(max_abs=0.0):
    return {"max_abs": max_abs, "unequal": int(max_abs != 0.0)}


def _report(label):
    rows = {name: _row() for name in gate.ORDER}
    rows["zFv"] = _row(1.0e-6)
    rows["qco_stage_T"] = _row(0.17735225873138316)
    rows["qco_stage_S"] = _row(3.283688696376231)
    rows["post_stage_association"] = {"unmeasured": True}
    return {
        "status": "PASS_R257_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved",
        "private_unit": True,
        "passivity": {"T": True, "S": True},
        "support": "T-pivot physical half, j=147, i=0..89, k=0..29",
        "source_order": list(gate.ORDER),
        "rows": rows,
        "zfv_signature": {
            "unequal": 1, "nonzero_ratio_count": 1,
            "ratio_min": 2.0, "ratio_max": 2.0, "ratio_median": 2.0,
            "sign_mismatch": 0,
        },
        "helper_calibration": {"T": _row(), "S": _row()},
        "recorded_transport_closure": {
            "after_advection_T": _row(1.0e-5),
            "after_advection_S": _row(1.0e-5),
            "qco_stage_T": _row(1.0e-5),
            "qco_stage_S": _row(1.0e-5),
        },
        "current_tree_reproduction": {
            "T_max_abs": 0.17735225873138316,
            "S_max_abs": 3.283688696376231,
            "round256_T": 0.17735225873138316,
            "round256_S": 3.283688696376231,
        },
        "one_bit_control": {"unequal": 1},
    }


def test_classifier_names_zfv_and_preserves_both_labels():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["first_numerical_statement"] == "zFv"
    assert result["predictions"]["R257-P3"] == "CONFIRMED"


def test_production_cen2_call_selects_certified_nemo_storage(monkeypatch):
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as model_module

    sentinel = object()

    def fake_cen2(*args, **kwargs):
        assert len(args) == 7
        assert kwargs == {"_certified_nemo_storage": True}
        return sentinel

    monkeypatch.setattr(model_module, "_nemo_cen2_tracer_rhs", fake_cen2)
    assert gate._certified_cen2_rhs(*range(7)) is sentinel


def test_measure_routes_every_direct_cen2_call_through_certified_wrapper():
    """Plant the round-257 direct caller, not only the wrapper it should use."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(gate.measure)))
    calls = [
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("_certified_cen2_rhs") == 4
    assert "_nemo_cen2_tracer_rhs" not in calls


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    reports = [_report("independent"), _report("given_nemo_entry")]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
