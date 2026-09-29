from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round74_metric_u_transport_gate as gate,
)


def _row(*, exact: bool) -> dict[str, object]:
    return {
        "count": 10,
        "max_abs": 0.0 if exact else 1.0,
        "max_ulp": 0 if exact else 1,
        "status": "AT_BAR" if exact else "DEBT",
        "unequal": 0 if exact else 1,
    }


def _report() -> dict[str, object]:
    rows = {name: _row(exact=True) for name in gate.ORDER}
    rows["un_adv"] = _row(exact=False)
    rows["zub"] = _row(exact=False)
    rows["e3u_Kmm"] = _row(exact=False)
    rows["corrected_velocity"] = _row(exact=False)
    rows["metric_thickness"] = _row(exact=False)
    rows["zFu"] = {
        **_row(exact=False),
        **gate.ROUND73_ZFU,
    }
    return {
        "claim_label": "independent",
        "card_scope": {
            name: list(values) for name, values in gate.EXPECTED_CARD_SCOPE.items()
        },
        "execution": {
            "backend": "cpu", "production_jit": True, "dtype": "float64",
            "transcendentals": "libm",
        },
        "first_non_bit": "un_adv",
        "first_non_bit_derived": "zub",
        "order": list(gate.ORDER),
        "production_exposure_calibration": {
            "derived_thickness_matches_exposure": _row(exact=True),
            "derived_corrected_matches_exposure": _row(exact=True),
            "derived_zFu_matches_exposure": _row(exact=True),
        },
        "record_replay": [_row(exact=True), _row(exact=True)],
        "rows": rows,
    }


def test_classifier_accepts_frozen_report():
    result = gate.classify(_report())
    assert result["status"] == "PASS_METRIC_U_TRANSPORT_WALK"
    assert result["prediction_ledger"]["un_adv_first"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["uu_Kmm_exact"]["status"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_failed_first_input_predictions_are_retained():
    report = _report()
    report["rows"]["un_adv"] = _row(exact=True)
    report["rows"]["inverse_depth"] = _row(exact=False)
    report["first_non_bit"] = "inverse_depth"
    result = gate.classify(report)
    assert result["prediction_ledger"]["un_adv_first"] == {
        "status": "REFUTED", "observed": "inverse_depth"}


def test_failed_velocity_and_thickness_predictions_are_retained():
    report = copy.deepcopy(_report())
    report["rows"]["uu_Kmm"] = _row(exact=False)
    report["rows"]["e3u_Kmm"] = _row(exact=True)
    result = gate.classify(report)
    assert result["prediction_ledger"]["uu_Kmm_exact"]["status"] == "REFUTED"
    assert result["prediction_ledger"]["e3u_Kmm_non_bit"]["status"] == "REFUTED"
