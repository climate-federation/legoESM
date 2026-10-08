"""Controls for the round-66 independent-start ORCA2 ladder gate."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round66_independent_start_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round66_independent", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _row(*, unequal=0, count=4, max_abs=0.0, rms=0.0):
    return {
        "bit_identical": unequal == 0,
        "unequal": unequal,
        "count": count,
        "max_abs": max_abs,
        "mean_abs_over_unequal": max_abs if unequal else 0.0,
        "rms": rms,
        "first_unequal_index": [0] if unequal else None,
    }


def _ladder():
    initial_rows = {field: _row() for field in gate.FIELD_ORDER}
    initial_rows["ssh"] = _row(
        unequal=gate.EXPECTED_ENTRY_SSH["unequal"],
        count=gate.EXPECTED_ENTRY_SSH["count"],
        max_abs=gate.EXPECTED_ENTRY_SSH["max_abs"],
        rms=0.01,
    )
    initial = {
        "rows": initial_rows,
        "first_non_bit_field": "ssh",
        "ranked_non_bit_by_max_abs": [],
    }
    checkpoints = []
    for kt in range(1, 11):
        for checkpoint in gate.CHECKPOINT_ORDER:
            rows = {field: _row() for field in gate.FIELD_ORDER}
            if kt == 1 and checkpoint == "entry":
                rows = copy.deepcopy(initial_rows)
            checkpoints.append({
                "kt": kt,
                "checkpoint": checkpoint,
                "rows": rows,
                "first_non_bit_field": "ssh" if rows["ssh"]["unequal"] else None,
                "ranked_non_bit_by_max_abs": [],
            })
    return {
        "status": "LADDER_MEASURED",
        "trajectory_claim": "MEASURED_INDEPENDENT_CARD_OWN_STATE",
        "unmeasured_features": list(gate.EXPECTED_UNMEASURED),
        "candidate_trajectory": {
            "claim_label": "INDEPENDENT",
            "initial_mode": "card_own_state",
            "decision52_bridge": None,
            "execution": "production-jit-cpu-fp64-x64-libm",
            "executed_initial_state_vs_nemo": initial,
            "first_non_bit_statement": {
                "kt": 1,
                "checkpoint": "entry",
                "field": "ssh",
                "source_citation": gate.SSH_CITATION,
            },
            "checkpoints": checkpoints,
        },
    }


def test_classifies_complete_independent_ladder():
    report = gate.classify(_ladder())
    assert report["status"] == "PASS_INDEPENDENT_LADDER"
    assert report["claim_label"] == "independent"
    assert report["checkpoints"] == 40
    assert report["field_rows"] == 200
    assert report["first_non_bit_statement"]["field"] == "ssh"


@pytest.mark.parametrize("plant", ["bridge", "entry-count", "checkpoint-drop"])
def test_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_ladder(), plant=plant)


def test_nonfinite_metric_refuses():
    ladder = _ladder()
    ladder["candidate_trajectory"]["checkpoints"][3]["rows"]["T"]["rms"] = float("nan")
    with pytest.raises(gate.GateError, match="non-finite rms"):
        gate.classify(ladder)


def test_json_sorted_field_order_is_not_a_registry_change():
    ladder = _ladder()
    for checkpoint in ladder["candidate_trajectory"]["checkpoints"]:
        checkpoint["rows"] = dict(sorted(checkpoint["rows"].items()))
    assert gate.classify(ladder)["field_rows"] == 200


def test_first_statement_citation_is_bound():
    ladder = _ladder()
    ladder["candidate_trajectory"]["first_non_bit_statement"][
        "source_citation"
    ] = "iceistate.f90:441-465"
    with pytest.raises(gate.GateError, match="citation changed"):
        gate.classify(ladder)
