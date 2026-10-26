from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round75_hybrid_pair_gate as gate,
)


def _row(*, exact: bool, rms: float) -> dict[str, object]:
    return {
        "count": 10,
        "max_abs": 0.0 if exact else 1.0,
        "max_ulp": 0 if exact else 1,
        "rms_abs": rms,
        "status": "AT_BAR" if exact else "DEBT",
        "unequal": 0 if exact else 1,
    }


def _report() -> dict[str, object]:
    arms = {}
    for name, sources in gate.ARM_SOURCES.items():
        exact_pair = name == "oracle_pair"
        rows = {
            "zub": _row(exact=exact_pair, rms=0.0 if exact_pair else 2.0),
            "corrected_velocity": _row(
                exact=exact_pair, rms=0.0 if exact_pair else 2.0),
            "zFu": _row(exact=False, rms=2.0),
        }
        arms[name] = {"input_sources": list(sources), "rows": rows}
    arms["baseline"]["rows"]["zub"] = {
        **_row(exact=False, rms=1.0),
        **gate.EXPECTED_BASELINE["zub"],
    }
    arms["baseline"]["rows"]["zFu"] = {
        **_row(exact=False, rms=1.0),
        **gate.EXPECTED_BASELINE["zFu"],
    }
    return {
        "claim_label": "independent",
        "card_scope": {
            name: list(values)
            for name, values in gate.round74.EXPECTED_CARD_SCOPE.items()
        },
        "arm_sources": copy.deepcopy(gate.ARM_SOURCES),
        "arms": arms,
        "execution": {
            "backend": "cpu", "production_jit": True, "dtype": "float64",
            "transcendentals": "libm",
        },
        "round74_gate_status": "PASS_METRIC_U_TRANSPORT_WALK",
        "row_order": list(gate.ROW_ORDER),
        "support": {
            "oracle_active_columns": 8568,
            "oracle_active_3d": 226236,
        },
    }


def test_classifier_accepts_two_sided_cancellation():
    result = gate.classify(_report())
    assert result["status"] == "PASS_HYBRID_CORRECTION_PAIR"
    assert result["owner"] == "paired_un_adv_and_inverse_depth"
    assert result["prediction_ledger"]["two_sided_cancellation"]["status"] == (
        "CONFIRMED")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_single_arm_refutation_is_retained_and_names_owner():
    report = _report()
    report["arms"]["oracle_un_adv"]["rows"]["zub"]["rms_abs"] = 0.5
    result = gate.classify(report)
    assert result["owner"] == "un_adv"
    assert result["prediction_ledger"]["un_adv_single_worsens"]["status"] == (
        "REFUTED")
    assert result["prediction_ledger"]["two_sided_cancellation"]["status"] == (
        "REFUTED")


def test_exact_paired_zfu_refutes_only_downstream_prediction():
    report = _report()
    report["arms"]["oracle_pair"]["rows"]["zFu"] = _row(
        exact=True, rms=0.0)
    result = gate.classify(report)
    assert result["prediction_ledger"]["paired_zFu_remains_debt"]["status"] == (
        "REFUTED")
