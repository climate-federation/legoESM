from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round80_baseline_location_gate as gate,
)


def _report() -> dict:
    rows = {
        "T": {"max_abs": 1.2367457128331782},
        "S": {"max_abs": 0.2871061346986039},
    }
    return {
        "claim_label": "given NEMO's entry",
        "comparison": {"rows": rows},
        "locations": {
            "T": {"index_jik": [1, 2, 3], "max_abs": rows["T"]["max_abs"]},
            "S": {"index_jik": [4, 5, 6], "max_abs": rows["S"]["max_abs"]},
        },
    }


def test_classifier_accepts_exact_locations() -> None:
    assert gate.classify(_report())["status"] == "PASS_ROUND80_BASELINE_LOCATION"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant: str) -> None:
    with pytest.raises(gate.extremes.GateError):
        gate.classify(copy.deepcopy(_report()), plant)
