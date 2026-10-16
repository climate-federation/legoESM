"""Known-answer controls for the round-293 V-depth operand split."""

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round293_v_depth_operands as gate,
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


def _pair(depth_unequal, depth_maximum, transport_unequal=None):
    if transport_unequal is None:
        transport_unequal = depth_unequal
    return {
        "v_mid": _regions(),
        "face_depth_v_mid": _regions(depth_unequal, depth_maximum),
        "metric_transport_v": _regions(
            transport_unequal, depth_maximum if transport_unequal else 0.0),
    }


def _report():
    operands = {name: _score() for name in
                ("hv_0", "r1_e1e2v", "e1e2t", "ssvmask")}
    operands["hv_0"] = _score(13, 719.0)
    operands["ssvmask"] = _score(35, 1.0)
    return {
        "status": "PASS_R293_MEASUREMENT",
        "claim_label": "independent_omt4_kt1_stage1_substep1",
        "operand_order": list(gate.OPERAND_ORDER),
        "arms": list(gate.ARMS),
        "passivity": {
            name: {field: True for field in ("T", "S", "u", "v", "ssh")}
            for name in gate.ARMS
        },
        "operand_rows": operands,
        "baseline_replay_vs_production": _score(),
        "baseline_vs_nemo": _score(13, 719.0),
        "cumulative_substitution": {},
        "direct_arms": {
            "baseline": _pair(13, 719.0, transport_unequal=0),
            "raw_hv_only": _pair(0, 0.0),
            "raw_v_surface_mask": _pair(0, 0.0),
            "mask_t_neighbor_pair": _pair(0, 0.0),
        },
        "one_bit_control": _score(1, 5.0e-324, 1),
    }


def test_classifier_names_pair_and_every_plant_fires():
    result = gate.classify(_report())
    assert result["status"] == "HELD_R293_HV0_SINGLE_STATEMENT_CLOSES_BOUNDARY"
    assert result["first_direct_operand"] == "hv_0"
    assert result["cancelling_partner"]["partner"] is None
    assert result["cancelling_partner"]["pair_depth_unequal"] == 0
    for plant in gate.PLANTS[1:]:
        with pytest.raises(gate.GateError):
            gate.classify(_report(), plant=plant)


def test_raw_hv_arm_must_close_the_boundary():
    report = _report()
    report["direct_arms"]["raw_hv_only"] = _pair(1, 1.0)
    with pytest.raises(gate.GateError, match="raw hv_0"):
        gate.classify(report)


def test_pair_must_close_the_immediate_consumer():
    report = _report()
    report["direct_arms"]["mask_t_neighbor_pair"]["metric_transport_v"] = (
        _regions(1, 1.0e-3))
    with pytest.raises(gate.GateError, match="immediate V transport"):
        gate.classify(report)
