"""Controls for the ORCA2 round-178 external-stage SSH walk."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as gate,
)


def synthetic_report() -> dict:
    source_order = list(gate.ENTRY_ORDER)
    rows = [
        {"name": name, "at_floor": True, "bit_exact": True}
        for name in gate.ENTRY_ORDER
    ]
    rows[1] = {"name": "u_forcing", "at_floor": False, "bit_exact": False}
    return {
        "claim_label": "independent hierarchy rung 0",
        "record_census": {"coverage": "exactly-once"},
        "record_control": {"bit_exact": False, "differing_cells": 1},
        "observer_passivity": {"all": True},
        "source_order": source_order,
        "arm_order": list(gate.ARM_ORDER),
        "endpoint_ulp_control": {"bit_exact": False, "differing_cells": 1},
        "baseline_source_rows": rows,
        "first_over_floor": rows[1],
        "entry_and_histories_exact": True,
        "history_arm_null": True,
        "slow_arm_improves_endpoint": True,
    }


def test_classifier_uses_compiled_source_order() -> None:
    report = gate.classify(synthetic_report())
    assert report["first_over_floor"]["name"] == "u_forcing"
    assert report["prediction_ledger"]["R178-P4"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round178_plant_refuses(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(synthetic_report()), plant)


def test_forcing_prediction_is_refuted_by_a_later_first_debt() -> None:
    report = synthetic_report()
    report["baseline_source_rows"][1]["at_floor"] = True
    report["baseline_source_rows"][1]["bit_exact"] = True
    later = {"name": gate.SUBSTEP_ORDER[0], "substep": 1,
             "at_floor": False, "bit_exact": False}
    report["baseline_source_rows"].append(later)
    report["source_order"].append(f"001:{gate.SUBSTEP_ORDER[0]}")
    report["first_over_floor"] = later
    classified = gate.classify(report)
    assert classified["prediction_ledger"]["R178-P4"] == "REFUTED"


def test_source_score_includes_masked_values_that_feed_active_stencils() -> None:
    candidate = np.zeros((2, 2), dtype=np.float64)
    oracle = candidate.copy()
    oracle[0, 1] = 1.0e-3
    active = np.array([[True, False], [True, True]])

    active_row = gate._score(candidate, oracle, active)
    source_row = gate._score(
        candidate, oracle, active, complete_domain=True)

    assert active_row["at_floor"]
    assert not source_row["at_floor"]
    assert source_row["verdict"] == "DEBT"
    assert not source_row["comparison_bit_exact"]
    assert source_row["full_domain_argmax_jik"] == [0, 1]
    assert source_row["comparison_domain"] == "complete-recorded"
