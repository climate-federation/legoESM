from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round77_external_transport_pair_gate as gate,
)


def _row(*, exact: bool, rms: float) -> dict[str, object]:
    return {
        "count": 8568,
        "max_abs": 0.0 if exact else 1.0,
        "max_ulp": 0 if exact else 1,
        "rms_abs": rms,
        "status": "AT_BAR" if exact else "DEBT",
        "unequal": 0 if exact else 1,
    }


def _report(*, owner: str = "ua_e") -> dict[str, object]:
    if owner == "ua_e":
        velocity_rms, depth_rms = 0.1, 2.0
    else:
        velocity_rms, depth_rms = 2.0, 0.1
    arms = {
        "live_pair": {
            "input_sources": list(gate.ARM_SOURCES["live_pair"]),
            **_row(exact=False, rms=gate.EXPECTED_BASELINE["rms_abs"]),
        },
        "oracle_velocity_only": {
            "input_sources": list(gate.ARM_SOURCES["oracle_velocity_only"]),
            **_row(exact=False, rms=velocity_rms),
        },
        "oracle_depth_only": {
            "input_sources": list(gate.ARM_SOURCES["oracle_depth_only"]),
            **_row(exact=False, rms=depth_rms),
        },
        "oracle_pair": {
            "input_sources": list(gate.ARM_SOURCES["oracle_pair"]),
            **_row(exact=True, rms=0.0),
        },
    }
    return {
        "claim_label": "independent",
        "execution": {
            "backend": "cpu", "production_jit": True,
            "dtype": "float64", "transcendentals": "libm",
        },
        "card_scope": {},
        "external_card_scope": copy.deepcopy(
            gate.round76.EXPECTED_EXTERNAL_CARD_SCOPE),
        "round76_status": "PASS_EXTERNAL_TRANSPORT_WALK",
        "round76_first_boundary": {
            "substep": 2, "boundary": "metric_transport",
            **copy.deepcopy(gate.EXPECTED_BASELINE),
        },
        "arms": arms,
        "dominant_owner": owner,
        "support": {"active_u_columns": 8568},
    }


def test_classifier_accepts_predicted_velocity_owner():
    result = gate.classify(_report())
    assert result["status"] == "PASS_EXTERNAL_TRANSPORT_PAIR"
    assert result["prediction_ledger"]["ua_e_is_dominant_owner"][
        "status"] == "CONFIRMED"


def test_classifier_keeps_refuted_owner_prediction():
    result = gate.classify(_report(owner="zhup2_e"))
    assert result["status"] == "PASS_EXTERNAL_TRANSPORT_PAIR"
    assert result["prediction_ledger"]["ua_e_is_dominant_owner"][
        "status"] == "REFUTED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)
