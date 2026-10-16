"""Controls for the round-286 resolved WS-RK3 mask gate."""

import copy
import json
from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round286_stage_mask_gate as gate,
)


ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")


def _docs():
    before = json.loads((ROOT / "round283/measurement_raw.json").read_text())
    after = json.loads((ROOT / "round286/stage1_mask_measurement_raw.json").read_text())
    return before, after


def test_replay_gate_closes_only_the_executing_consumer():
    result = gate.replay_gate(*_docs())
    assert result["consumer_unequal_before"] == 1319
    assert result["consumer_unequal_after"] == 0
    assert result["stale_diagnostic_unequal"] == 1319


def test_consumer_bit_plant_fires():
    with pytest.raises(gate.GateError, match="consumer did not close"):
        gate.replay_gate(*_docs(), plant="consumer-bit")


def test_decision96_exact_loss_plant_fires():
    control = {
        "status": "PASS_R276_SOURCE_LADDER", "arm": "control",
        "rung": "rung0", "label": "independent",
        "worktree": {"commit": "same"},
        "ladder": {"first_non_bit_checkpoint": {
            "kt": 1, "checkpoint": "stage1", "field": "v"},
            "rows": [{"kt": 1, "checkpoint": "stage1", "field": "v",
                      "bit_identical": True, "unequal": 0,
                      "rms": 1.0, "max_abs": 1.0}], "row_count": 1},
    }
    candidate = copy.deepcopy(control)
    candidate["arm"] = gate.ARM
    candidate["selected_statements"] = [gate.ARM]
    with pytest.raises(Exception):
        gate.r276.classify_documents(control, candidate, plant="exact-loss")
