"""Controls for the round-215 Decision-96 gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_decision96_gate as gate,
)


def _row(kt: int, checkpoint: str, field: str, value: float) -> dict:
    return {
        "kt": kt, "checkpoint": checkpoint, "field": field,
        "bit_identical": False, "unequal": 1, "first_unequal_index": [0],
        "rms": value, "max_abs": value, "mean_abs_over_unequal": value,
    }


def _ladder(scale: float) -> dict:
    rows = []
    for kt in range(1, 9):
        for checkpoint in ("entry", "stage1", "stage2", "stage3"):
            for field in ("T", "S", "u", "v", "ssh"):
                rows.append(_row(kt, checkpoint, field, scale))
    return {"independent": {
        "rows": rows,
        "first_non_bit_checkpoint": {"kt": 1, "checkpoint": "stage1", "field": "T"},
    }}


def _pair() -> dict:
    return {
        "status": "PASS_R215_OMT1_VECTOR_PAIR_REPLAY",
        "operand_rows": {
            "zv_frc": {"differing_cells": 68},
            "ssvmask": {"differing_cells": 68},
        },
        "replays": {
            "pair": {"bit_exact": True},
            "zv_frc_only": {"bit_exact": False},
            "ssvmask_only": {"bit_exact": False},
        },
    }


def test_eligible_pair_and_all_controls_fire() -> None:
    base, candidate, pair = _ladder(2.0), _ladder(1.0), _pair()
    result = gate.classify(base, candidate, pair)
    assert result["decision96_eligible"] is True
    for plant in ("pair-closure", "exact-loss", "false-majority"):
        with pytest.raises(gate.GateError):
            gate.classify(base, candidate, pair, plant=plant)


def test_score_equal_rows_are_registered_but_not_votes() -> None:
    base, candidate = _ladder(2.0), _ladder(2.0)
    candidate = copy.deepcopy(candidate)
    rows = candidate["independent"]["rows"]
    rows[5]["unequal"] = 2
    rows[6]["rms"] = rows[6]["max_abs"] = rows[6]["mean_abs_over_unequal"] = 1.0
    result = gate.classify(base, candidate, _pair())
    census = result["ladder"]["rms_direction_census"]
    assert census == {"toward": 1, "away": 0, "equal": 1}
    assert result["ladder"]["rms_score_moved_row_count"] == 1
