from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round260_stage2_pivot_gate as gate,
)


def _row(max_abs=0.0, *, support=5400):
    return {"unequal": 0, "max_abs": max_abs, "rms": max_abs,
            "support": support, "nonfinite": 0}


def _field(max_abs=0.0):
    return {"complete": _row(max_abs), "fold_band": _row(max_abs),
            "pivot_row": _row(max_abs)}


def _report(label):
    fields = {name: _field() for name in ("u", "v", "T", "S", "ssh")}
    raw_entry = copy.deepcopy(fields)
    raw_entry["v"]["pivot_row"] = _row(0.027227136128767482)
    baseline_stage2 = copy.deepcopy(fields)
    baseline_stage2["S"]["fold_band"] = _row(6.817836451535307)
    return {
        "status": "PASS_R260_MEASUREMENT", "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": {"T": True, "S": True},
        "source_order": ["stage1_update", "post_stage_lbc_lnk",
                         "stage2_transport", "stage2_tracer"],
        "baseline_stage1": copy.deepcopy(fields),
        "associated_stage1": copy.deepcopy(fields),
        "baseline_stage2_entry": raw_entry,
        "associated_stage2_entry": copy.deepcopy(fields),
        "baseline_stage2": baseline_stage2,
        "arm_stage2": copy.deepcopy(fields),
        "one_bit_control": {"unequal": 1},
    }


def test_classifier_accepts_both_labels():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["status"] == "PASS_R260_POST_STAGE_V_OWNER"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    reports = [_report("independent"), _report("given_nemo_entry")]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
