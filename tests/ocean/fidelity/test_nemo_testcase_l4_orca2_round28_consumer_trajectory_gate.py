import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round28_consumer_trajectory_gate as gate,
)


def _score(value=0.0, exact=False):
    return {"bit_identical": exact, "unequal": 0 if exact else 1,
            "count": 1, "max_abs": value,
            "mean_abs_over_unequal": value,
            "first_unequal_index": None if exact else [0, 0, 0]}


def _document(count, commit):
    checkpoints = []
    for index in range(count):
        kt = index // 4 + 1
        checkpoint = ("entry", "stage1", "stage2", "stage3")[index % 4]
        checkpoints.append({
            "kt": kt, "checkpoint": checkpoint,
            "rows": {name: _score(index + 1.0) for name in
                     ("T", "S", "u", "v", "ssh")},
        })
    checkpoints[0]["rows"]["u"] = _score(0.0, exact=True)
    return {
        "status": "LADDER_MEASURED", "card": {"id": "orca2"},
        "worktree": {"clean": True, "commit": commit},
        "candidate_trajectory": {
            "checkpoints": checkpoints,
            "first_non_bit_statement": {"kt": 1, "checkpoint": "stage1",
                                        "field": "T"},
        },
    }


def _fixtures():
    parent = _document(40, "parent")
    een = copy.deepcopy(parent)
    een["worktree"]["commit"] = "een"
    een["candidate_trajectory"]["checkpoints"] = (
        een["candidate_trajectory"]["checkpoints"][:12])
    een["candidate_trajectory"]["checkpoints"][2]["rows"]["u"] = _score(99.0)
    een["candidate_trajectory"]["checkpoints"][2]["rows"]["v"] = _score(98.0)
    shared = copy.deepcopy(een)
    shared["worktree"]["commit"] = "shared"
    ldf = copy.deepcopy(parent)
    ldf["worktree"]["commit"] = "ldf"
    direct = {
        "consumer": "dynldf_lev F-curl thickness",
        "base_vs_carried_raw_f": {
            "u": {"bit_identical": False},
            "v": {"bit_identical": False},
        },
    }
    vor = {"stage2_vorticity": {"digest": "same"}}
    refusal = ("raw-mesh e3w_int must contain only finite values > 0\n"
               "REFUSE: the production step raised an unregistered refusal")
    return parent, een, ldf, shared, refusal, direct, vor, copy.deepcopy(vor)


def test_round28_gate_accepts_registered_held_outcome():
    result = gate.evaluate(*_fixtures())
    assert result["status"] == "HELD"
    assert set(result["predictions"].values()) == {"CONFIRMED"}


@pytest.mark.parametrize("plant", ["exact_row", "consumer_labels"])
def test_round28_gate_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.evaluate(*_fixtures(), plant=plant)
