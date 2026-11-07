import argparse
import copy
import json

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round25_outcome_gate as gate,
)


def _score(*, exact=True, maximum=0.0):
    return {
        "bit_identical": exact,
        "unequal": 0 if exact else 1,
        "count": 1,
        "max_abs": maximum,
        "mean_abs_over_unequal": maximum,
        "first_unequal_index": None if exact else [0],
    }


def _document(commit, checkpoints):
    return {
        "status": "LADDER_MEASURED",
        "worktree": {"commit": commit},
        "candidate_trajectory": {
            "first_non_bit_statement": {
                "kt": 1, "checkpoint": "stage1", "field": "T"},
            "checkpoints": checkpoints,
        },
    }


def _checkpoints(count):
    result = []
    names = ("entry", "stage1", "stage2", "stage3")
    for index in range(count):
        checkpoint = names[index % 4]
        rows = {name: _score() for name in ("T", "S", "u", "v", "ssh")}
        if checkpoint == "stage1":
            rows["T"] = _score(exact=False, maximum=1.0)
        if checkpoint == "stage2":
            rows["u"] = _score(exact=False, maximum=1.0)
        result.append({"kt": index // 4 + 1, "checkpoint": checkpoint,
                       "rows": rows})
    return result


def test_round25_gate_accepts_three_live_arms_and_rejects_plant(tmp_path):
    baseline = _document("base", _checkpoints(40))
    arms = {}
    for name, count in (("single", 40), ("thickness", 12), ("metric", 40)):
        arm = _document(name, copy.deepcopy(baseline["candidate_trajectory"]["checkpoints"][:count]))
        if name != "single":
            arm["candidate_trajectory"]["checkpoints"][2]["rows"]["u"] = (
                _score(exact=False, maximum=2.0))
        arms[name] = arm

    paths = {}
    for name, document in (("baseline", baseline), *arms.items()):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        paths[name] = path
    refusal = tmp_path / "refusal.log"
    refusal.write_text(
        "raw-mesh e3w_int must contain only finite values > 0\n"
        "REFUSE: the production step raised an unregistered refusal\n")

    args = argparse.Namespace(
        baseline=paths["baseline"], single_mask=paths["single"],
        live_thickness=paths["thickness"], native_f_metrics=paths["metric"],
        refusal_log=refusal, json_out=None, plant=False)
    result = gate.run(args)
    assert result["isolated_instability_owner"] == "live_thickness"
    assert result["arms"]["single_mask"]["moved_row_count"] == 0
    args.plant = True
    with pytest.raises(gate.GateError, match="formerly AT-BAR rows left"):
        gate.run(args)
