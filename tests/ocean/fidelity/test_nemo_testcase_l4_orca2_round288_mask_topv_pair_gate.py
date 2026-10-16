"""Controls for the round-288 cancelling-pair gate."""

from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round288_mask_topv_pair_gate as gate,
)


def _score(*, rms=0.0, maximum=0.0, exact=True):
    return {
        "bit_identical": exact,
        "unequal": 0 if exact else 1,
        "max_abs": maximum,
        "rms": rms,
    }


def _boundary(kt, stage, *, v_rms=0.0, v_max=0.0, exact=False):
    rows = {field: _score(exact=exact) for field in gate.FIELDS}
    rows["v"] = _score(rms=v_rms, maximum=v_max, exact=exact)
    return {"kt": kt, "stage": stage, "rows": rows}


def _documents():
    control_rows = []
    for kt, checkpoint in ((1, "stage1"), (10, "stage3")):
        for field in gate.FIELDS:
            control_rows.append({"kt": kt, "checkpoint": checkpoint,
                                 "field": field, "bit_identical": False})
    control = {
        "status": "PASS_R276_SOURCE_LADDER", "arm": "control",
        "rung": "rung0", "label": "independent",
        "ladder": {"row_count": 200, "rows": control_rows},
    }
    common = {
        "format": "nemo-testcase-l4-orca2-round288-stage-table-v1",
        "status": "PASS_R288_STAGE_TABLE",
        "claim_label": "independent_rung0",
        "admission": {"record_count": 80},
        "one_bit_control": {"unequal": 1},
    }
    single = {
        **common, "arm": "resolved-mask-only",
        "boundaries": [_boundary(1, 1, v_rms=2.0, v_max=2.0)],
        "terminal": {"kt": 4, "stage": 1, "error": gate.EXPECTED_REFUSAL},
    }
    pair = {
        **common, "arm": "resolved-mask-plus-top-v",
        "boundaries": [
            _boundary(1, 1, v_rms=1.0, v_max=1.0),
            _boundary(10, 3, v_rms=0.5, v_max=0.5),
        ],
        "terminal": None,
    }
    return control, single, pair


def test_atomic_pair_qualifies_only_after_both_single_controls_are_registered():
    result = gate.classify(*_documents())
    assert result["status"] == "QUALIFIED_R288_FULL_GATES"
    assert result["prediction_ledger"] == {
        "R288-P1": "CONFIRMED",
        "R288-P2": "CONFIRMED",
        "R288-P3": "CONFIRMED",
        "R288-P4": "READY_FOR_FULL_GATES",
        "R288-P5": "CONFIRMED",
    }


def test_pair_refusal_is_a_measured_hold_not_a_gate_crash():
    control, single, pair = _documents()
    pair["terminal"] = {"kt": 5, "stage": 2, "error": gate.EXPECTED_REFUSAL}
    pair["boundaries"] = pair["boundaries"][:1]
    result = gate.classify(control, single, pair)
    assert result["status"] == "HELD_R288_PAIR"
    assert result["prediction_ledger"]["R288-P2"] == "REFUTED"


@pytest.mark.parametrize(
    "plant", ("one-bit", "missing-single-failure", "fake-pair-pass"))
def test_every_known_answer_plant_fires(plant):
    documents = copy.deepcopy(_documents())
    with pytest.raises(gate.GateError):
        gate.classify(*documents, plant=plant)
