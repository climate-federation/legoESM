from __future__ import annotations

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round161_debt_ranking_gate as gate,
)


def _row(kt: int, checkpoint: str, delta: float) -> dict[str, object]:
    before = 0.4363833806287545 if (kt, checkpoint) == (10, "stage3") else 1.0
    after = before + delta
    return {
        "kt": kt,
        "checkpoint": checkpoint,
        "field": "v",
        "metrics": {
            "max_abs": {
                "before": before,
                "after": after,
                "delta": delta,
                "direction": "away",
            },
        },
    }


def _document() -> dict[str, object]:
    target = _row(10, "stage3", 0.862704843805445)
    target["metrics"]["max_abs"]["after"] = 1.2990882244341995
    return {
        "status": "PASS_R111_ORCA2_LADDER_COMPARE",
        "rung7": {"moved_rows": [target, _row(10, "stage2", 0.8)]},
    }


def test_target_ranks_first_within_v_maximum_rows() -> None:
    result = gate.rank_v_maximum_regressions(_document())
    assert result["target_rank"] == 1
    assert result["away_v_maximum_rows"] == 2


def test_target_delta_plant_fires() -> None:
    with pytest.raises(gate.GateError, match="non-positive delta"):
        gate.rank_v_maximum_regressions(_document(), plant="target-delta")
