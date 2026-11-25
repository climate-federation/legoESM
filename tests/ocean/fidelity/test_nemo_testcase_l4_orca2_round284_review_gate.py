from __future__ import annotations

import copy
import json
import sys

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round284_review_gate as gate,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_receipt_citation_gate as citations,
)


def _stage(label, arm, error):
    score = {"max_abs": 0.0, "rms": 0.0, "unequal": 0, "count": 1,
             "bit_exact": True, "argmax": [0]}
    return {
        "label": label, "round284_arm": arm,
        "target_19_33_3": {"candidate": error, "nemo": 0.0,
                            "abs_error": error},
        "passivity": {"T": True, "S": True, "u": True, "v": True},
        "boundary_rows": {"after_vorticity_u": {
            "complete": copy.deepcopy(score), "fold_band": copy.deepcopy(score)}},
    }


def test_missing_falsifier_requires_area_to_move_target_toward(tmp_path):
    paths = []
    for label in gate.LABELS:
        for arm, error in (("area_off", 2.0), ("area_on", 1.0),
                           ("top_f_halo", 0.5)):
            path = tmp_path / f"{label}_{arm}.json"
            path.write_text(json.dumps(_stage(label, arm, error)))
            paths.append(path)
    result = gate.classify_stage(paths)
    assert result["prediction_ledger"]["R284-P3"] == "CONFIRMED"
    with pytest.raises(gate.GateError, match="did not move"):
        gate.classify_stage(paths, plant=True)


def test_npcrv_citation_is_pinned_to_vor_een_and_context_plant_fires():
    key = "ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766"
    assert citations.check(key, citations.CITATION_MAP[key])["status"] == "OK"
    assert citations.check(
        key, citations.CITATION_MAP[key], routine_plant=True
    )["status"] == "ROUTINE-CONTEXT-MISMATCH"


def test_cli_distinguishes_fired_from_green_plant(monkeypatch, capsys):
    argv = ["round284", "--classify-stage", *("unused.json",) * 6,
            "--plant", "stage"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        gate, "classify_stage",
        lambda *_args, **_kwargs: {"status": "PASS_BROKEN_PLANT"},
    )
    assert gate.main() == gate.PLANT_STAYED_GREEN_EXIT
    output = capsys.readouterr().out
    assert "STATUS PLANT-STAYED-GREEN: stage plant stayed green" in output
    assert "STATUS PLANT-FIRED" not in output

    def fired(*_args, **_kwargs):
        raise gate.GateError("known-answer difference detected")

    monkeypatch.setattr(gate, "classify_stage", fired)
    assert gate.main() == gate.PLANT_FIRED_EXIT
    output = capsys.readouterr().out
    assert "STATUS PLANT-FIRED: known-answer difference detected" in output
    assert "STATUS PLANT-STAYED-GREEN" not in output
