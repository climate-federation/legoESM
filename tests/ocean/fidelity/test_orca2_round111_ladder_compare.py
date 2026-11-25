import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round111_ladder_compare as gate,
)


def _flat():
    return {
        "rows": [
            {"kt": kt, "checkpoint": checkpoint, "field": field,
             "bit_identical": True, "unequal": 0, "first_unequal_index": None,
             "max_abs": 0.0, "rms": 0.0, "mean_abs_over_unequal": 0.0}
            for kt in range(1, 11)
            for checkpoint in ("entry", "stage1", "stage2", "stage3")
            for field in ("T", "S", "u", "v", "ssh")
        ],
        "first_non_bit_checkpoint": None,
        "first_non_bit_source_statement": None,
    }


def test_registers_moved_row_without_losing_exact_row():
    base = _flat()
    candidate = copy.deepcopy(base)
    candidate["rows"][-1].update(bit_identical=False, unequal=1,
                                  first_unequal_index=[0], max_abs=1.0,
                                  rms=0.5, mean_abs_over_unequal=1.0)
    base["rows"][-1] = copy.deepcopy(candidate["rows"][-1])
    candidate["rows"][-1]["max_abs"] = 0.5
    result = gate.compare(base, candidate)
    assert result["moved_row_count"] == 1
    assert result["moved_rows"][0]["metrics"]["max_abs"]["direction"] == "toward"


def test_refuses_bit_exact_loss():
    base = _flat()
    candidate = copy.deepcopy(base)
    candidate["rows"][0].update(bit_identical=False, unequal=1,
                                 first_unequal_index=[0], max_abs=1.0,
                                 rms=1.0, mean_abs_over_unequal=1.0)
    with pytest.raises(gate.GateError, match="left the bar"):
        gate.compare(base, candidate)


def test_refuses_first_statement_move():
    base = _flat()
    candidate = copy.deepcopy(base)
    candidate["first_non_bit_checkpoint"] = {"kt": 0}
    with pytest.raises(gate.GateError, match="first non-bit"):
        gate.compare(base, candidate)


def test_salinity_veto_rejects_larger_kt10_stage3_error():
    base = _flat()
    candidate = copy.deepcopy(base)
    target = next(row for row in candidate["rows"]
                  if (row["kt"], row["checkpoint"], row["field"])
                  == (10, "stage3", "S"))
    target["max_abs"] += 1.0
    with pytest.raises(gate.GateError, match="salinity maximum increased"):
        gate.require_salinity_veto(base, candidate)
