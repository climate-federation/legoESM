import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round193_stage_growth_gate as gate,
)


def _score(value=1.0):
    return {
        "bit_exact": False, "differing_cells": 1, "cells": 1,
        "candidate_nonfinite": 0, "oracle_nonfinite": 0,
        "max_abs": value, "argmax": [0], "candidate_max_abs": value,
        "oracle_max_abs": 0.0,
    }


def _report():
    boundaries = []
    for kt in gate.STEPS:
        for stage in gate.STAGES:
            boundaries.append({
                "kt": kt, "stage": stage,
                "rows": {field: _score() for field in gate.FIELDS},
            })
    growth = gate._growth_rows(boundaries)
    return {
        "claim_label": "independent", "initial_mode": "card_own_state",
        "decision52_bridge": None,
        "execution": "production-jit-cpu-fp64-x64-libm",
        "admission": {"record_count": 80},
        "entry": {field: {"bit_exact": True, "unequal": 0}
                  for field in gate.FIELDS},
        "private_arm": {
            "slow_depth_source_order": True,
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "passivity": {
            "detached_stage_outputs_not_carried": True,
            "ordinary_duplicate": {field: True for field in gate.FIELDS},
        },
        "field_order": list(gate.FIELDS),
        "growth_field_order": list(gate.GROWTH_FIELDS),
        "floor": float(gate.FLOOR),
        "growth_threshold": float(gate.GROWTH),
        "boundaries": boundaries, "growth": growth,
        "first_growth": gate._first_growth(growth), "terminal": None,
        "one_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_clean_report_selects_first_boundary():
    result = gate.classify(_report())
    assert result["first_growth"]["kt"] == 1
    assert result["first_growth"]["stage"] == 1
    assert result["prediction_ledger"]["R193-P3"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)


def test_terminal_must_be_first_missing_boundary():
    report = _report()
    report["boundaries"] = report["boundaries"][:2]
    report["growth"] = gate._growth_rows(report["boundaries"])
    report["first_growth"] = gate._first_growth(report["growth"])
    report["terminal"] = {"kt": 1, "stage": 3, "status": "REFUSED",
                          "error": "planted"}
    assert gate.classify(report)["status"] == "PASS_R193_STAGE_GROWTH_BOUNDARY"


def test_growth_ignores_ssh_by_frozen_registry():
    report = _report()
    report["boundaries"][0]["rows"]["ssh"] = _score(1.0e30)
    report["growth"] = gate._growth_rows(report["boundaries"])
    report["first_growth"] = gate._first_growth(report["growth"])
    assert gate.classify(report)["first_growth"]["max_abs"] == 1.0
