import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round265_stage2_u_gate as gate,
)


def _score(max_abs=0.0, unequal=0):
    return {
        "support": 1, "unequal": unequal, "max_abs": max_abs,
        "rms": max_abs, "argmax": [0], "first_nonfinite": None,
        "nonfinite": 0,
    }


def _report(label):
    rows = {
        name: {"complete": _score(), "fold_band": _score()}
        for name in gate.ORDER
    }
    rows["Krhs_u"]["fold_band"] = _score(1.0, 1)
    rows["raw_Kaa_u"]["fold_band"] = _score(1.0, 1)
    rows["post_lbc_Kaa_u"]["fold_band"] = _score(
        0.003382014663707235, 1)
    return {
        "format": "nemo-testcase-l4-orca2-round265-stage2-u-v1",
        "status": "PASS_R265_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "passivity": {name: True for name in ("T", "S", "u", "v", "ssh")},
        "source_order": list(gate.ORDER),
        "rows": rows,
        "rhs_closure": {
            "raw_Kaa_u": _score(), "post_mean_Kaa_u": _score(),
            "post_lbc_Kaa_u": _score(1.0, 1),
        },
        "matched_stage2": {"T": _score(), "S": _score()},
        "record_identity": {"record": "a" * 64, "stage1": "b" * 64,
                            "stage2": "c" * 64},
        "one_bit_control": _score(1.0, 1),
    }


def _prefix(label):
    return {
        "status": "PASS_R264_PREFIX_MEASUREMENT", "label": label,
        "completed": [{"kt": 1, "nonfinite": 0},
                      {"kt": 2, "nonfinite": 0}],
        "refusal": None,
    }


def test_round265_accepts_source_ordered_rhs_owner():
    result = gate.classify(
        [_report(label) for label in gate.LABELS],
        [_prefix(label) for label in gate.LABELS])
    assert result["status"] == "PASS_R265_STAGE2_U_OWNER"
    assert result["predictions"]["R265-P3"] == "REFUTED_CANCELLING_UNIT_REMAINS"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round265_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(
            [copy.deepcopy(_report(label)) for label in gate.LABELS],
            [copy.deepcopy(_prefix(label)) for label in gate.LABELS],
            plant=plant)
