"""Controls for the ORCA2 round-267 HPG accumulator carry."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round267_hpg_carry_gate as gate,
)


def _score(value=0.0, unequal=0):
    return {"max_abs": value, "unequal": unequal}


def _arm_report(active):
    boundary = {
        name: {"complete": _score(), "fold_band": _score()}
        for name in gate.r266.BOUNDARY_ORDER
    }
    replay = {name: _score() for name in (
        "after_hpg_u", "after_vorticity_u", "after_keg_u",
        "after_advection_u")}
    if active:
        boundary["after_vorticity_u"]["complete"] = _score(1.0e-6, 2)
        boundary["after_advection_u"]["complete"] = _score(1.0e-6, 2)
    else:
        boundary["after_vorticity_u"]["complete"] = _score(1.0e-6, 2)
        boundary["after_advection_u"]["complete"] = _score(1.0e-6, 2)
    return {
        "literal_hpg_accumulator": active,
        "source_order_arm": True,
        "record_identity": {"record": "a" * 64},
        "passivity": {"T": True},
        "boundary_order": list(gate.r266.BOUNDARY_ORDER),
        "boundary_rows": boundary,
        "accumulator_self_replay": replay,
        "matched_stage2": {"T": _score(), "S": _score()},
    }


def _report(label):
    return {
        "status": "PASS_R267_MEASUREMENT",
        "label": label,
        "baseline": _arm_report(False),
        "candidate": _arm_report(True),
        "one_bit_control": _score(1.0e-16, 1),
    }


def test_classification_retracts_hpg_and_names_vorticity():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["first_next_boundary"] == "after_vorticity_u"
    assert result["completed_rhs_closes"] is False
    assert result["predictions"]["R267-P2"] == "REFUTED_ARM_UNNEEDED_BASELINE_CLOSES"


@pytest.mark.parametrize("plant", [name for name in gate.PLANTS if name != "none"])
def test_every_plant_refuses(plant):
    reports = [_report(label) for label in gate.LABELS]
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(reports), plant=plant)
