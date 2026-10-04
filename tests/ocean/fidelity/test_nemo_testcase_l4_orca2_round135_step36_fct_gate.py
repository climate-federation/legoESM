from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round135_step36_fct_gate as gate,
)


def _group(fields: tuple[str, ...], nonfinite: bool = False) -> dict[str, object]:
    counts = {"T": int(nonfinite), "S": 0}
    target = {"T": int(nonfinite), "S": 0}
    support = {name: 2 for name in fields}
    return {
        "fields": list(fields),
        "support_count": support,
        "active_count": sum(support.values()),
        "nonfinite": counts,
        "nonfinite_total": sum(counts.values()),
        "target_nonfinite": target,
        "target_nonfinite_total": sum(target.values()),
        "details": {},
    }


def _report(first: str = "antidiffusive_flux") -> dict[str, object]:
    first_index = gate.GROUP_ORDER.index(first)
    groups = {
        name: _group(fields, gate.GROUP_ORDER.index(name) >= first_index)
        for name, fields in gate.rejected.TRACE_GROUPS
    }
    return {
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "steps_completed_before_walk": 35,
        "trace_field_order": list(gate.TRACE_FIELD_ORDER),
        "group_order": list(gate.GROUP_ORDER),
        "groups": groups,
        "first_nonfinite_group": first,
        "first_target_nonfinite_group": first,
        "ordinary_repeat_state_equal": {name: True for name in gate.FIELDS},
        "observer_state_equal": {"T": True, "S": True, "u": True},
        "returned_first_nonfinite": {
            "field": "T", "index": list(gate.TARGET), "value": "nan"},
        "side_output_type": "_NEMOWSFCTInputTrace",
        "side_output_field": "mass_flux_w",
    }


def test_classify_accepts_passive_antidiffusive_boundary() -> None:
    result = gate.classify(_report())
    assert result["status"] == "PASS_ROUND135_STEP36_FCT_WALK"
    assert result["prediction_ledger"]["R135-P4"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["R135-P5"]["status"] == "CONFIRMED"
    assert result["prediction_ledger"]["R135-P6"]["status"] == "CONFIRMED"


def test_earlier_boundary_is_retained_as_refutation() -> None:
    result = gate.classify(_report("averaged_upstream_flux"))
    assert result["prediction_ledger"]["R135-P5"]["status"] == "REFUTED"
    assert result["first_nonfinite_group"] == "averaged_upstream_flux"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant: str) -> None:
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant=plant)


def test_target_disagreement_is_retained_as_refutation() -> None:
    report = copy.deepcopy(_report())
    report["first_target_nonfinite_group"] = "limiter_coefficient"
    for name in gate.GROUP_ORDER:
        row = report["groups"][name]
        value = int(gate.GROUP_ORDER.index(name) >=
                    gate.GROUP_ORDER.index("limiter_coefficient"))
        row["target_nonfinite"] = {"T": value, "S": 0}
        row["target_nonfinite_total"] = value
    result = gate.classify(report)
    assert result["prediction_ledger"]["R135-P6"]["status"] == "REFUTED"


def test_private_hook_defaults_off() -> None:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    assert _NEMOWSRK3TestHooks().expose_stage3_fct_inputs is False
