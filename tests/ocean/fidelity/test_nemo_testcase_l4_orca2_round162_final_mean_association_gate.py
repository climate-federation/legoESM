"""Controls for the ORCA2 round-162 final external-mode association gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round162_final_mean_association_gate as gate,
)


def _row(value: float, *, exact: bool = False) -> dict:
    return {
        "bit_identical": exact, "unequal": 0 if exact else 1,
        "max_abs": value, "rms": value,
        "mean_abs_over_unequal": value,
        "first_unequal_index": None if exact else [0],
    }


def _ladder(salinity: float = 0.4) -> dict:
    rows = []
    for kt in range(1, 11):
        for checkpoint in ("entry", "stage1", "stage2", "stage3"):
            for field in ("T", "S", "u", "v", "ssh"):
                value = salinity if (kt, checkpoint, field) == (10, "stage3", "S") else 0.1
                rows.append({"kt": kt, "checkpoint": checkpoint,
                             "field": field, **_row(value)})
    return {
        "rows": rows,
        "first_non_bit_checkpoint": {"kt": 1, "checkpoint": "stage1", "field": "T"},
        "first_non_bit_source_statement": {"source": "test"},
    }


def _boundary() -> dict:
    exact_row = {
        "candidate_finite": True, "differing_cells": 4,
        "bit_exact": False,
    }
    return {
        "status": "PASS_R162_FINAL_ASSOCIATION_BOUNDARY",
        "claim_label": "independent",
        "production_repeat_passive": True,
        "movement_against_production": {
            "final_only": {"uu_b": exact_row, "vv_b": exact_row},
        },
        "rows_against_nemo": {
            "substep_only": {
                "uu_b": {**exact_row, "differing_cells": 10},
                "vv_b": {**exact_row, "differing_cells": 10},
            },
            "pair": {
                "uu_b": {**exact_row, "differing_cells": 2},
                "vv_b": {**exact_row, "differing_cells": 2},
            },
        },
    }


def test_classification_registers_reduced_boundary_debt() -> None:
    base = _ladder(0.4)
    pair = copy.deepcopy(base)
    result = gate.classify(_boundary(), base, copy.deepcopy(base), pair)
    assert result["prediction_ledger"]["R162-P2"]["status"] == "CONFIRMED"
    assert result["salinity_veto"]["after_max_abs"] == 0.4


@pytest.mark.parametrize("plant", ("passivity", "boundary-bit", "salinity"))
def test_each_plant_fires(plant: str) -> None:
    base = _ladder(0.4)
    with pytest.raises((gate.GateError, gate.compare_gate.GateError)):
        gate.classify(_boundary(), base, copy.deepcopy(base), copy.deepcopy(base),
                      plant=plant)


def test_private_hook_is_false_by_default() -> None:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    assert _NEMOWSRK3TestHooks().barotropic_final_mean_association is False
