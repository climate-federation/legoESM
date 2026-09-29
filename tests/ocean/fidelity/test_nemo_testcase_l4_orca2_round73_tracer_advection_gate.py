from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round73_tracer_advection_gate as gate,
)


def _row(*, exact: bool, count: int, maximum: float, unequal: int) -> dict:
    return {
        "bit_exact": exact,
        "count": count,
        "max_abs": maximum,
        "status": "AT_BAR" if exact else "DEBT",
        "unequal": unequal,
    }


def _report() -> dict:
    rows = {
        name: _row(exact=False, count=10, maximum=1.0, unequal=10)
        for name in gate.ORDER
    }
    rows["entry_T"] = _row(exact=True, count=228641, maximum=0.0, unequal=0)
    rows["after_advection_T"] = copy.deepcopy(gate.ROUND72_AFTER_ADV)
    return {
        "claim_label": "independent",
        "card_scope": {name: list(values)
                       for name, values in gate.EXPECTED_CARD_SCOPE.items()},
        "compiled_dispatch": "CEN2 at RK3 stages 1-2; FCT2 at stage 3",
        "execution": {
            "backend": "cpu", "production_jit": True, "dtype": "float64",
            "transcendentals": "libm",
        },
        "first_non_bit_arithmetic": "u_face_flux_line153",
        "first_non_bit_operand": "metric_pU",
        "order": list(gate.ORDER),
        "recorded_transport_only_replay": _row(
            exact=False, count=228641, maximum=1.0e-9, unequal=10),
        "recorded_full_operand_replay": _row(
            exact=True, count=228641, maximum=0.0, unequal=0),
        "rows": rows,
    }


def test_classifier_accepts_frozen_report():
    result = gate.classify(_report())
    assert result["status"] == "PASS_TRACER_ADVECTION_WALK"
    assert result["prediction_ledger"]["pU_first_operand"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["u_flux_first_arithmetic"]["status"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_failed_first_statement_predictions_are_retained():
    report = _report()
    report["first_non_bit_operand"] = "metric_pV"
    report["first_non_bit_arithmetic"] = "v_face_flux_line154"
    result = gate.classify(report)
    assert result["prediction_ledger"]["pU_first_operand"]["status"] == "REFUTED"
    assert result["prediction_ledger"]["u_flux_first_arithmetic"]["status"] == "REFUTED"
