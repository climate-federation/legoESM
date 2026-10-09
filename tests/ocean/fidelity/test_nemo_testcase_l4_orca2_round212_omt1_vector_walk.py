from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round212_omt1_vector_walk as gate,
)


def _report() -> dict:
    row = {"comparison_bit_exact": True, "at_floor": True}
    return {
        "claim_label": "independent OMT-1",
        "record_admission": {"stream_count": 2, "streams": [
            {"header_valid": True, "defined_twin_status": "EXACT_DEFINED_BYTES"},
            {"header_valid": True, "defined_twin_status": "EXACT_DEFINED_BYTES"},
        ]},
        "source_order": list(gate.r205.ENTRY_ORDER),
        "offline_replay_passivity": {
            "ssh": True, "u_barotropic": True, "v_barotropic": True,
            "transport_u": True, "transport_v": True,
        },
        "source_rows": [copy.deepcopy(row)],
        "substep_table": [{} for _ in range(65)],
        "slow_v_arm": {
            "input": {"comparison_bit_exact": True},
            "substep_table": [{} for _ in range(65)],
        },
        "first_nonbit": None,
        "first_over_floor": None,
        "terminal_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_round212_classifier_accepts_complete_report() -> None:
    assert gate.classify(_report())["status"] == "PASS_R212_OMT1_VECTOR_WALK"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round212_plants_fire(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)
