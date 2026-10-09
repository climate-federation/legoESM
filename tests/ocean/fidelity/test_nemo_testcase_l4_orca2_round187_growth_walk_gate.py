"""Unit controls for the round-187 deterministic growth selectors."""

from __future__ import annotations

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round187_growth_walk_gate as gate,
)


def _score(value: float) -> dict:
    return {"max_abs": value, "argmax_jik": [0, 0, 0]}


def test_coarse_growth_starts_from_frozen_floor() -> None:
    rows = []
    for step in gate.COARSE_STEPS:
        value = 1.0 if step == 10 else 1.5
        rows.append({
            "step": step,
            "error_rows": {name: _score(value if name == "T" else 0.5)
                           for name in gate.FIELDS},
        })
    rows.append({"step": 95, "error_rows": None})
    result = gate._growth_rows(rows, gate.COARSE_STEPS)
    assert result[0]["over_10x"]
    assert result[0]["ratio"] == 1.0 / gate.FLOOR
    assert not result[1]["over_10x"]


def test_stage_growth_preserves_compiled_boundary_order() -> None:
    rows = []
    for kt in range(1, 11):
        for checkpoint in gate.CHECKPOINTS:
            for field in gate.FIELDS:
                value = 1.0 if (kt, checkpoint, field) == (1, "stage1", "ssh") else 0.0
                rows.append({
                    "kt": kt, "checkpoint": checkpoint, "field": field,
                    "max_abs": value, "first_unequal_index": None,
                })
    result = gate._stage_rows({"rows": rows})
    first = next(row for row in result if row["over_10x"])
    assert (first["kt"], first["checkpoint"], first["field"]) == (
        1, "stage1", "ssh")
