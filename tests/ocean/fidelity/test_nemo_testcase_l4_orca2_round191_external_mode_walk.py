"""Controls for the ORCA2 round-191 exact-forcing external-mode walk."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round178_external_ssh_walk as r178,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round191_external_mode_walk as gate,
)


def synthetic_report() -> dict:
    forcing_rows = [
        {"name": name, "at_floor": True, "comparison_bit_exact": True}
        for name in r178.ENTRY_ORDER
    ]
    forcing_rows.append({
        "name": "u_mid", "substep": 1, "at_floor": True,
        "comparison_bit_exact": True,
    })
    forcing_rows.append({
        "name": "v_mid", "substep": 1, "at_floor": False,
        "comparison_bit_exact": False,
    })
    baseline_rows = [
        {"name": "ssh_forcing", "at_floor": True, "bit_exact": True},
        {"name": "u_forcing", "at_floor": False, "bit_exact": False},
    ]
    digest = "same"
    return {
        "claim_label": "independent hierarchy rung 0",
        "record_census": {"coverage": "exactly-once"},
        "record_control": {"bit_exact": False, "differing_cells": 1},
        "observer_passivity": {"all": True},
        "source_order": ["ssh_forcing", "u_forcing"],
        "arm_order": list(r178.ARM_ORDER),
        "endpoint_ulp_control": {"bit_exact": False, "differing_cells": 1},
        "baseline_source_rows": baseline_rows,
        "first_over_floor": baseline_rows[-1],
        "entry_and_histories_exact": True,
        "history_arm_null": True,
        "slow_arm_improves_endpoint": False,
        "endpoint_scores": {
            "slow_only": {"trace_digest": digest},
            "slow_and_history": {"trace_digest": digest},
        },
        "source_rows_by_arm": {
            "baseline": baseline_rows,
            "history_only": baseline_rows,
            "slow_only": forcing_rows,
            "slow_and_history": copy.deepcopy(forcing_rows),
        },
    }


def test_classifier_stops_at_exact_forcing_first_debt() -> None:
    source = synthetic_report()
    source["source_rows_by_arm"] = dict(
        reversed(list(source["source_rows_by_arm"].items())))
    report = gate.classify(source)
    assert report["round191"]["first_over_floor"]["name"] == "v_mid"
    assert report["round191"]["history_arm_null"]


@pytest.mark.parametrize("plant", ("exact-arm-order", "exact-arm-selector"))
def test_round191_plants_refuse(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(synthetic_report(), plant=plant)


def test_history_digest_and_rows_must_agree() -> None:
    report = synthetic_report()
    report["source_rows_by_arm"]["slow_and_history"][-1]["at_floor"] = True
    with pytest.raises(gate.GateError):
        gate.classify(report)
