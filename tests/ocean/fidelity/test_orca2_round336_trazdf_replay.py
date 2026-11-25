"""Fail-closed controls for round 336's statement replay."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round336_trazdf_replay as replay,
)


def _score(value=0, maximum=0.0):
    return {"support": 18, "bit_unequal": value, "value_unequal": value,
            "max_abs": maximum, "rms": maximum}


def _report():
    rows = {name: {"all_wet": _score(), "target": _score()}
            for name in replay.ROW_ORDER}
    rows["trazdf_forward_T"]["target"] = _score(2, 1e-15)
    rows["trazdf_Kaa_T_update"]["target"] = _score(18, 7.7e-12)
    return {
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_admission": "PASS_R336_TRAZDF_ACQUISITION",
        "observer_passivity": {"bit_identical": True, "changed_leaves": 0},
        "target_count_T": 18,
        "row_order": list(replay.ROW_ORDER),
        "rows": rows,
        "first_differing_target_statement": "trazdf_forward_T",
    }


def test_classification_passes_complete_source_order_walk():
    result = replay.classify(_report())
    assert result["status"] == "PASS_R336_TRAZDF_STATEMENT_REPLAY"
    assert result["claim_label"] == "CONFIRMED"


@pytest.mark.parametrize("plant", replay.PLANTS)
def test_each_registered_plant_refuses(plant):
    with pytest.raises(replay.GateError):
        replay.classify(copy.deepcopy(_report()), plant=plant)


def test_first_row_is_recomputed_not_trusted():
    report = _report()
    report["first_differing_target_statement"] = "trazdf_diagonal"
    with pytest.raises(replay.GateError, match="not source ordered"):
        replay.classify(report)


def test_first_plant_moves_metric_derived_first_row():
    with pytest.raises(replay.GateError, match="not source ordered"):
        replay.classify(_report(), plant="first")


def test_target_faces_exclude_dry_w_faces():
    targets = np.zeros((1, 1, 4), dtype=bool)
    targets[..., 1] = True
    wet_faces = np.array([[[False, True, True]]])
    assert np.array_equal(
        replay._target_faces(targets, wet_faces),
        np.array([[[False, True, False]]]),
    )
