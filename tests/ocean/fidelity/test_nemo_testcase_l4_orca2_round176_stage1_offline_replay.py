"""Controls for the round-176 rank-complete offline replay gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round176_stage1_offline_replay as gate,
)


def _row(name: str, active: int = 7) -> dict[str, object]:
    return {
        "name": name, "bit_exact": True, "differing_cells": 0,
        "active_cells": active, "max_abs": 0.0, "rms": 0.0,
        "at_floor": True, "argmax": [0, 0, 0], "registered_cells": [],
    }


def _report() -> dict[str, object]:
    rows = [_row(name) for name in gate.SOURCE_ORDER]
    return {
        "claim_label": "independent",
        "admission": {"status": "PASS_R175_STAGE1_ADMISSION",
                      "rank_coverage": "exactly-once"},
        "source_order": list(gate.SOURCE_ORDER),
        "cells": [list(cell) for cell in gate.CELLS],
        "active_counts": {name: 7 for name in ("T", "S", "u", "v", "ssh")},
        "rows": rows, "first_replayed_debt": None,
        "recorded_adv_equals_sbc": True,
        "one_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_classify_accepts_derived_clean_report() -> None:
    result = gate.classify(_report())
    assert result["status"] == "HELD_FIRST_UNAVAILABLE_CANDIDATE_EXTERNAL_MODE_AND_RHS"
    assert result["prediction_ledger"]["R176-P5"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_control_plant_refuses(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_first_real_replayed_debt_is_selected() -> None:
    report = _report()
    report["rows"][8]["at_floor"] = False
    report["first_replayed_debt"] = copy.deepcopy(report["rows"][8])
    result = gate.classify(report)
    assert result["status"] == "PASS_FIRST_REPLAYED_DEBT"
    assert result["prediction_ledger"]["R176-P3"] == "REFUTED"


def test_independent_entry_debt_stops_before_operator_claim() -> None:
    report = _report()
    report["rows"][0]["at_floor"] = False
    report["first_replayed_debt"] = copy.deepcopy(report["rows"][0])
    report["terminal_boundary"] = "independent_initial_state"
    result = gate.classify(report)
    assert result["status"] == "HELD_FIRST_NONBIT_INDEPENDENT_INITIAL_STATE"
    assert result["prediction_ledger"]["R176-P3"] == "UNMEASURED"
    assert result["prediction_ledger"]["R176-P5"] == "UNMEASURED"
