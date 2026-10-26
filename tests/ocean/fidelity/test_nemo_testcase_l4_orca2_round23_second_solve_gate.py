from __future__ import annotations

from copy import deepcopy

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round23_second_solve_gate as gate,
)


def _row(value=0.0):
    return {
        "bit_identical": value == 0.0,
        "count": 1,
        "first_unequal_index": None if value == 0.0 else [0],
        "max_abs": value,
        "mean_abs_over_unequal": value,
        "unequal": int(value != 0.0),
    }


def _document(stage2_u=1.0):
    checkpoints = []
    for kt in range(1, 11):
        for checkpoint in gate.ladder_gate.CHECKPOINTS:
            rows = {field: _row() for field in gate.ladder_gate.FIELDS}
            if (kt, checkpoint) == (1, "stage1"):
                rows["T"] = _row(2.0)
            if (kt, checkpoint) == (1, "stage2"):
                rows["u"] = _row(stage2_u)
            checkpoints.append({"kt": kt, "checkpoint": checkpoint, "rows": rows})
    return {
        "status": "LADDER_MEASURED",
        "trajectory_claim": "MEASURED_INDEPENDENT_WITH_DECISION52_SSH",
        "candidate_trajectory": {
            "checkpoints": checkpoints,
            "first_non_bit_statement": {
                "kt": 1, "checkpoint": "stage1", "field": "T"},
        },
    }


def _comparison():
    return {
        "status": "PASS", "violations": [],
        "n_certified_rows_compared": 70,
        "largest_oracle_residual_worsening_ulps": 0,
    }


def _analyze(before, after):
    return gate.analyze(
        before, after, _comparison(), residuals_equal=True,
        residual_arrays=210, daily_equal=True, daily_snapshots=30,
        package_diff_exact=True,
    )


def test_gate_lands_only_the_stage2_momentum_change():
    result = _analyze(_document(1.0), _document(0.5))
    assert result["status"] == "LANDED"
    assert result["orca2_rows_moved"] == 1
    assert len(result["moved_rows"]) == result["orca2_rows_moved"]
    assert result["moved_rows"][0]["direction_by_max_abs"] == "toward_nemo"
    assert result["first_moved_row"]["checkpoint"] == "stage2"
    assert set(result["predictions"].values()) == {"CONFIRMED"}


def test_one_ulp_like_entry_plant_holds_the_gate():
    before = _document(1.0)
    after = _document(0.5)
    planted = deepcopy(after)
    planted["candidate_trajectory"]["checkpoints"][0]["rows"]["T"][
        "max_abs"] = 5e-324
    result = _analyze(before, planted)
    assert result["status"] == "HELD"
    assert result["kt1_entry_or_stage1_rows_moved"] == [[1, "entry", "T"]]
    assert result["predictions"]["R23-P2"] == "REFUTED"


def test_gyre_snapshot_or_residual_movement_holds_the_gate():
    result = gate.analyze(
        _document(1.0), _document(0.5), _comparison(),
        residuals_equal=False, residual_arrays=210,
        daily_equal=True, daily_snapshots=30, package_diff_exact=True,
    )
    assert result["status"] == "HELD"
    assert result["predictions"]["R23-P5"] == "REFUTED"
