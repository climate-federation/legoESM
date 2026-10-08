from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round168_exit_depth_operands as gate,
)


def _row(*, exact: bool = True, differing: int = 0) -> dict:
    return {"bit_exact": exact, "differing_cells": differing}


def _report() -> dict:
    return {
        "admission": {
            "rank_coverage": "exactly-once",
            "records": [
                {"icycle": 65, "groups": 2106},
                {"icycle": 65, "groups": 2106},
            ],
            "terminal_restart_comparisons": [{} for _ in range(20)],
        },
        "completed_kt": 7,
        "registered_count": 41,
        "source_order": list(gate.SOURCE_ORDER),
        "upstream_order": list(gate.UPSTREAM_ORDER),
        "registered_rows": {
            "raw_plus_oracle_face_ssh": _row(),
            "candidate_sum_replay": _row(),
            "oracle_eta_face_ssh_replay": _row(),
            "candidate_face_ssh": _row(exact=False, differing=41),
            "candidate_after_ssh_stencil": _row(exact=False, differing=41),
        },
        "first_upstream_nonbit": {"name": "transport_v"},
    }


def test_clean_report_passes_and_disposes_every_prediction():
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND168_EXIT_DEPTH_OPERAND_WALK"
    assert set(result["prediction_dispositions"]) == {
        "R168-P1", "R168-P2", "R168-P3", "R168-P4", "R168-P5",
    }
    assert set(result["prediction_dispositions"].values()) == {"CONFIRMED"}


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant: str):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_registered_row_uses_bits_and_nonzero_values():
    candidate = np.array([[2.0, -0.0]])
    oracle = np.array([[2.0, 0.0]])
    row = gate._registered_row(candidate, oracle, np.ones((1, 2), dtype=bool))
    assert row["differing_cells"] == 1
    assert row["candidate_max"] == 2.0
    assert not row["bit_exact"]


def test_first_nonbit_preserves_source_order():
    rows = [
        {"name": "a", "bit_exact": True},
        {"name": "b", "bit_exact": False},
        {"name": "c", "bit_exact": False},
    ]
    assert gate._first_nonbit(rows)["name"] == "b"
