"""Controls for the round-263 scalar-ratio and matched-point gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round263_matched_v_gate as gate,
)


def _score(max_abs=0.0, unequal=0, support=16200):
    return {"argmax": [0, 0, 0], "first_nonfinite": None,
            "max_abs": max_abs, "nonfinite": 0, "rms": max_abs,
            "support": support, "unequal": unequal}


def _report(label: str):
    matched = {
        "status": "PASS_R260_MEASUREMENT",
        "passivity": {name: True for name in ("T", "S", "u", "v", "eta")},
        "baseline_stage1": {"v": {"fold_band": _score(0.027227136128767482, 6091)}},
        "associated_stage1": {"v": {"fold_band": _score(3.469446951953614e-18, 1718)}},
        "baseline_stage2": {
            "T": {"fold_band": _score(0.27211560747403896, 100)},
            "S": {"fold_band": _score(5.037097791516988, 100)},
        },
        "arm_stage2": {
            "T": {"fold_band": _score(0.09829516579631314, 100)},
            "S": {"fold_band": _score(1.8200702777361712, 100)},
        },
    }
    ratio = {
        "unassociated_pivot_row": {}, "scalar_plus_pivot_row": {},
        "vector_minus_pivot_row": {},
        "plus_vs_unassociated": _score(1.0, 10, 180),
        "minus_vs_plus": _score(2.0, 10, 180),
        "association": {"point_type": "V", "sign": 1.0,
                        "destination": "pivot row", "source": "south row via perm_v"},
    }
    records = {"record.bin": "abc"}
    return {
        "status": "PASS_R263_MEASUREMENT", "label": label,
        "ratio_table": ratio,
        "ratio_oracle_closure": {
            "e3v": _score(), "r1_hv": _score(), "zvb": _score(),
            "zFv": _score(0.0, 181), "candidate_replay": _score(),
            "oracle_replay": _score(), "records": records,
        },
        "matched_stage": matched,
        "compiled_contract": {
            "scalar_ratio": {"point_type": "V", "sign": 1.0,
                             "citation": "stprk3_stg.f90:163-180"},
            "completed_velocity": {"point_type": "V", "sign": -1.0,
                                   "citation": "stprk3_stg.f90:779-802"},
            "fold_rule": "lbcnfd.f90:684-721",
        },
        "one_bit_control": _score(2.220446049250313e-16, 1, 1),
    }


def test_round263_accepts_scalar_plus_and_matched_point():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "PASS_R263_PROGRAM_POINT_RECONCILIATION"
    assert result["predictions"]["R263-P2"].startswith("CONFIRMED")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round263_plants_fire(plant):
    reports = [_report(label) for label in gate.LABELS]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
