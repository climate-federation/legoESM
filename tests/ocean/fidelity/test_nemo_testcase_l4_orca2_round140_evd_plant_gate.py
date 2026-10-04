"""Controls for the round-140 rung-0 EVD effectiveness readout."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round140_evd_plant_gate as gate,
)


def _report():
    return {
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "scheme": "enhanced_diffusion",
        "base_K_conv": 100.0,
        "planted_K_conv": 1.0e6,
        "profile_dtype": {"avt": "float64", "avm": "float64"},
        "profiles_equal": {"avt": True, "avm": True},
        "profile_delta": {"avt": {"moved": 0}, "avm": {"moved": 0}},
    }


def test_valid_inert_report_confirms_prediction():
    admitted = gate.classify(_report())
    assert admitted["evd_active"] is False
    assert admitted["prediction_ledger"]["R140-P4"]["status"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_planted_violation_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
