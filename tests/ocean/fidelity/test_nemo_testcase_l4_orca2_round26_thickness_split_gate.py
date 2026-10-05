import argparse
import copy
import json

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round26_thickness_split_gate as gate,
)


def _score(*, exact=True, maximum=0.0):
    return {
        "bit_identical": exact, "unequal": 0 if exact else 1, "count": 1,
        "max_abs": maximum, "mean_abs_over_unequal": maximum,
        "first_unequal_index": None if exact else [0],
    }


def _checkpoints(count):
    rows = []
    names = ("entry", "stage1", "stage2", "stage3")
    for index in range(count):
        checkpoint = names[index % 4]
        fields = {name: _score() for name in ("T", "S", "u", "v", "ssh")}
        if checkpoint == "stage1":
            fields["T"] = _score(exact=False, maximum=1.0)
        if checkpoint == "stage2":
            fields["u"] = _score(exact=False, maximum=1.0)
        rows.append({"kt": index // 4 + 1, "checkpoint": checkpoint,
                     "rows": fields})
    return rows


def _document(commit, count):
    return {
        "status": "LADDER_MEASURED", "card": {"name": "orca2"},
        "worktree": {"commit": commit, "clean": True},
        "candidate_trajectory": {
            "first_non_bit_statement": {
                "kt": 1, "checkpoint": "stage1", "field": "T"},
            "checkpoints": _checkpoints(count),
        },
    }


def test_round26_gate_names_f_curl_owner_and_rejects_plant(tmp_path):
    baseline = _document("base", 40)
    f_curl = _document("f", 12)
    kbb = _document("kbb", 40)
    kmm = copy.deepcopy(kbb)
    kmm["worktree"]["commit"] = "kmm"
    for document in (f_curl, kbb, kmm):
        document["candidate_trajectory"]["checkpoints"][2]["rows"]["u"] = (
            _score(exact=False, maximum=2.0))
    combined = _document("combined", 12)
    combined["candidate_trajectory"]["checkpoints"][2]["rows"]["u"] = (
        _score(exact=False, maximum=0.10080009966621735))
    combined["candidate_trajectory"]["checkpoints"][2]["rows"]["v"] = (
        _score(exact=False, maximum=0.11456680400114852))

    paths = {}
    for name, document in (("baseline", baseline), ("f", f_curl),
                           ("kbb", kbb), ("kmm", kmm),
                           ("combined", combined)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(document))
        paths[name] = path
    refusal = tmp_path / "refusal.log"
    refusal.write_text(
        "raw-mesh e3w_int must contain only finite values > 0\n"
        "REFUSE: the production step raised an unregistered refusal\n")
    args = argparse.Namespace(
        baseline=paths["baseline"], f_curl=paths["f"],
        f_curl_refusal=refusal, kbb_divergence=paths["kbb"],
        kmm_divisor=paths["kmm"], combined=paths["combined"],
        json_out=None, plant=False)
    result = gate.run(args)
    assert result["isolated_refusal_owner"] == "f_curl"
    assert result["kbb_kmm_score_documents_identical"]
    assert result["predictions"]["R26-P3"] == "REFUTED"
    args.plant = True
    with pytest.raises(gate.GateError, match="formerly AT-BAR rows left"):
        gate.run(args)
