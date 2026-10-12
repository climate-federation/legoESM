from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round260_stage2_transport_split as gate,
)


def _row(max_abs=0.0, unequal=0):
    return {"max_abs": max_abs, "unequal": unequal, "support": 1,
            "rms": max_abs, "nonfinite": 0}


def _report(label):
    return {
        "status": "PASS_R260_TRANSPORT_SPLIT_MEASUREMENT",
        "label": label,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "operand_route": "production-d114-resolved-complete-private-unit",
        "entry_rows": {name: _row() for name in ("u", "v", "T", "S")},
        "transport_rows": {"zFu": _row(), "zFv": _row(1.0), "zFw": _row()},
        "transport_replay": {"oracle_zFv_self": _row()},
        "recorded_transport_stage2": {"T": _row(), "S": _row()},
        "prior_stage2": {"T": _row(6.435500132795369e-9),
                         "S": _row(1.3842782919937235e-8)},
        "one_bit_control": _row(unequal=1),
        "record": {"sha256": "a" * 64},
    }


def test_classifier_accepts_transport_owner():
    result = gate.classify([
        _report("independent"), _report("given_nemo_entry")])
    assert result["status"] == "PASS_R260_STAGE2_TRANSPORT_OWNER"
    assert result["first_non_floor_transport"] == "zFv"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    reports = [_report("independent"), _report("given_nemo_entry")]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
