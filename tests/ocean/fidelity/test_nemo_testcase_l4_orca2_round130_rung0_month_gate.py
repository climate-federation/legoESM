from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round130_rung0_month_gate as gate,
)


def _row(value: float) -> dict[str, object]:
    return {
        "bit_identical": False,
        "unequal": 1,
        "count": 2,
        "rms": value,
        "max_abs": value,
        "argmax_jik": [0],
        "candidate_at_argmax": value,
        "oracle_at_argmax": 0.0,
    }


def _report() -> dict[str, object]:
    rms_values = {name: float(len(gate.FIELDS) - index)
                  for index, name in enumerate(gate.EXPECTED_RMS_ORDER)}
    max_values = {name: float(len(gate.FIELDS) - index)
                  for index, name in enumerate(gate.EXPECTED_MAX_ORDER)}
    rows = {
        name: {**_row(rms_values[name]), "max_abs": max_values[name]}
        for name in gate.FIELDS
    }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed": 240,
        "first_nonfinite": None,
        "unmeasured_features": ["linear_implicit_bottom_drag"],
        "record_admission": {"status": "PASS_RUNG0_RECORD"},
        "terminal_restart": {
            "status": "BIT_EXACT_ORIENTATION",
            "files": [{"sha256": "1" * 64}, {"sha256": "2" * 64}],
        },
        "terminal": {
            "rows": rows,
            "ranking_by_rms": gate.rank_rows(rows, "rms"),
            "ranking_by_max_abs": gate.rank_rows(rows, "max_abs"),
        },
    }


def test_classify_accepts_complete_independent_month() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_RUNG0_INDEPENDENT_MONTH"
    assert all(row["status"] == "CONFIRMED"
               for row in result["prediction_ledger"].values())


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_terminal_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_changed_ranking_is_retained_as_a_refutation() -> None:
    report = _report()
    report["terminal"]["ranking_by_rms"] = list(reversed(
        report["terminal"]["ranking_by_rms"]))
    result = gate.classify(report)
    assert result["prediction_ledger"]["rms_order"]["status"] == "REFUTED"


def test_first_nonfinite_uses_frozen_field_and_index_order() -> None:
    fields = {name: np.zeros((2, 2), dtype=np.float64) for name in gate.FIELDS}
    fields["S"][1, 1] = np.inf
    fields["T"][0, 1] = np.nan
    result = gate.first_nonfinite(fields)
    assert result is not None
    assert result["field"] == "T"
    assert result["index"] == [0, 1]


def test_score_field_reports_argmax_and_bit_census() -> None:
    oracle = np.zeros((2, 2), dtype=np.float64)
    candidate = copy.deepcopy(oracle)
    candidate[1, 0] = 3.0
    row = gate.score_field(candidate, oracle)
    assert row["unequal"] == 1
    assert row["argmax_jik"] == [1, 0]
    assert row["max_abs"] == 3.0


def test_terminal_ledger_selects_full_month_paths_despite_twin_basenames(
    tmp_path,
) -> None:
    month = tmp_path / "month"
    twin = tmp_path / "twin"
    month.mkdir()
    twin.mkdir()
    names = [f"ORCA2_00000240_restart_{rank:04d}.nc" for rank in (0, 1)]
    ledger = tmp_path / "outputs.sha256"
    ledger.write_text("".join(
        [f"{'a' * 64}  {twin / names[0]}\n",
         f"{'b' * 64}  {month / names[0]}\n",
         f"{'c' * 64}  {month / names[1]}\n"]
    ))
    assert gate.terminal_ledger_rows(ledger, month) == {
        names[0]: "b" * 64,
        names[1]: "c" * 64,
    }
