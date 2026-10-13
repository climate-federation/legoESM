"""Controls for the round-264 stage-ratio placement reconciliation."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round264_ratio_placement_gate as gate,
)


def _score(max_abs=0.0, unequal=0, support=100):
    return {
        "argmax": [0, 0, 0], "first_nonfinite": None,
        "max_abs": max_abs, "nonfinite": 0, "rms": max_abs,
        "support": support, "unequal": unequal,
    }


def _mode(name: str):
    stage1 = {
        field: {"complete": _score(), "fold_band": _score(),
                "pivot_row": _score()}
        for field in ("u", "v", "T", "S", "ssh")
    }
    if name == "off":
        t_max, s_max = 6.435500132795369e-09, 1.3842782919937235e-08
    elif name == "round262":
        t_max, s_max = 2.220446049250313e-16, 7.105427357601002e-15
    else:
        t_max, s_max = 6.435500132795369e-09, 1.3842782919937235e-08
    stage2 = copy.deepcopy(stage1)
    stage2["T"]["fold_band"] = _score(t_max, 10)
    stage2["S"]["fold_band"] = _score(s_max, 10)
    return {
        "mode": name,
        "passivity": {field: True for field in ("T", "S", "u", "v", "eta")},
        "raw_stage1": copy.deepcopy(stage1),
        "matched_stage1": stage1,
        "matched_stage2": stage2,
        "stage1_zFv_sha256": "a" * 64,
    }


def _report(label: str, mode: str):
    return {
        "status": "PASS_R264_MEASUREMENT",
        "label": label,
        "mode": mode,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "record_identity": {"frame.bin": "a" * 64},
        "result": _mode(mode),
        "compiled_order": [
            "dom_qco_r3c_RK3", "scalar_v_fold", "store_stage_ratio",
            "stage2_zvb", "stage2_zFv", "stage2_tracer",
        ],
        "one_bit_control": _score(2.220446049250313e-16, 1, 1),
    }


def test_round264_accepts_controlled_placement_reconciliation():
    result = gate.classify([
        _report(label, mode) for label in gate.LABELS for mode in gate.MODES])
    assert result["status"] == "PASS_R264_RATIO_PLACEMENT_RECONCILIATION"
    assert result["predictions"]["R264-P2"].startswith("REFUTED")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round264_plants_fire(plant):
    reports = [
        _report(label, mode) for label in gate.LABELS for mode in gate.MODES]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
