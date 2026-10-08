"""Controls for the round-186 month growth and live-thickness gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round186_month_growth_gate as gate,
)


def _report(*, private: bool = False, missing: bool = False) -> dict:
    rows = []
    for step in gate.CHECKPOINTS:
        error = {
            name: {"max_abs": float(step)} for name in ("T", "S", "u", "v", "ssh")
        }
        rows.append({
            "step": step,
            "claim_label": "independent",
            "candidate_max_abs": {
                name: 1.0 for name in ("T", "S", "u", "v", "ssh")
            },
            "error_rows": None if missing else error,
        })
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "private_halo_unit": private,
        "initial_entry": {
            name: {"bit_exact": True, "unequal": 0}
            for name in ("T", "S", "u", "v", "ssh")
        },
        "steps_completed": 95 if not private else 7,
        "runtime_refusal": {
            "step": 96 if not private else 8,
            "message": gate.EXPECTED_ERROR,
        },
        "live_thickness": {
            "invalid_count": 3,
            "shape": [148, 180, 29],
            "first_invalid": {"index_jik": [86, 159, 3]},
        },
        "growth_table": rows,
        "missing_oracle_steps": list(gate.CHECKPOINTS) if missing else [],
    }


def test_complete_production_report_confirms_frozen_boundary() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_R186_MONTH_BOUNDARY"
    assert result["prediction_ledger"]["R186-P1"] == "CONFIRMED"
    assert result["prediction_ledger"]["R186-P2"] == "CONFIRMED"


def test_missing_streams_stop_for_record_without_inventing_scores() -> None:
    result = gate.classify(_report(missing=True))
    assert result["status"] == "STOPPED_FOR_RECORD"
    assert result["prediction_ledger"]["R186-P3"] == "REFUTED"


def test_private_boundary_is_separate_and_earlier() -> None:
    result = gate.classify(_report(private=True))
    assert result["prediction_ledger"]["R186-P1"] == "NOT_APPLICABLE_PRIVATE"
    assert result["prediction_ledger"]["R186-P5"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_refuses(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_partial_rank_coverage_refuses(tmp_path) -> None:
    (tmp_path / "ORCA2_00000010_restart_0000.nc").touch()
    with pytest.raises(gate.GateError, match="rank coverage is partial"):
        gate._read_restart(tmp_path, 10, object())
