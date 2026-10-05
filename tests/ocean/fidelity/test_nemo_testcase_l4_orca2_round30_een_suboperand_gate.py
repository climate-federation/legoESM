import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round30_een_suboperand_gate as gate,
)


def _trajectory():
    return {
        "status": "LADDER_MEASURED",
        "trajectory_claim": "MEASURED_INDEPENDENT_WITH_DECISION52_SSH",
        "candidate_trajectory": {
            "checkpoints": [{"kt": 1, "checkpoint": "entry", "rows": {
                "T": {"unequal": 0, "bit_identical": True}}}],
            "first_non_bit_statement": "kt=1:stage1:T",
        },
    }


def test_round30_trajectory_gate_accepts_exact_owner():
    owner = _trajectory()
    result = gate.evaluate_trajectory(
        owner, copy.deepcopy(owner),
        "REFUSE: the production step raised an unregistered refusal\n"
        "raw-mesh e3w_int must contain only finite values > 0\n")
    assert result["status"] == "HELD"
    assert result["kt1_through_kt3_score_document_bit_identical"]


@pytest.mark.parametrize("plant", ["score", "refusal"])
def test_round30_trajectory_gate_plants_fire(plant):
    owner = _trajectory()
    with pytest.raises(gate.GateError):
        gate.evaluate_trajectory(
            owner, copy.deepcopy(owner),
            "REFUSE: the production step raised an unregistered refusal\n"
            "raw-mesh e3w_int must contain only finite values > 0\n",
            plant=plant)
