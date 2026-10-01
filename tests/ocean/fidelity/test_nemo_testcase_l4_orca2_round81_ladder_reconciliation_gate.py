import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round81_ladder_reconciliation_gate as gate,
)


def _report():
    return {
        "claim_label": "given NEMO's entry",
        "reference": {"producer_commit":
                      "9b27d1b3ad198a275b639018ea4ce007f786e9eb"},
        "round80_observer_refusal": {"sha256": "1" * 64},
        "checkpoint_rows_compared": 200,
        "moved_rows": [],
        "first_non_bit_statement_equal": True,
    }


def test_real_shape_report_passes():
    assert gate.classify(_report())["status"] == "PASS_ROUND81_LADDER_RECONCILIATION"


@pytest.mark.parametrize("plant", ["candidate-row", "producer", "round80-log", "claim"])
def test_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)
