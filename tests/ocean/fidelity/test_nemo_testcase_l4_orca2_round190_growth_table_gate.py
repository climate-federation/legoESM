from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round190_growth_table_gate as gate,
)


def _fixtures():
    admission = {
        "status": "PASS_R189_GROWTH_RECORD",
        "claim_label": "independent",
        "admitted_steps": list(gate.STEPS),
        "missing_steps": [],
        "rank_count": 2,
        "terminal_sentinel": 96,
        "terminal_overwrites": [],
        "comparisons": [
            {"step": step, "rank": rank, "unequal": {name: 0 for name in
             ("tn", "sn", "un", "vn", "sshn")}}
            for step in gate.STEPS for rank in (0, 1)
        ],
        "sentinel_comparisons": [
            {"step": 96, "rank": rank, "unequal": {name: 0 for name in
             ("tn", "sn", "un", "vn", "sshn")}}
            for rank in (0, 1)
        ],
    }
    growth = []
    maximum = 1.0
    for step in gate.STEPS:
        if step == 95:
            maximum = 100.0
        scores = {}
        for index, field in enumerate(gate.prior.FIELDS):
            value = maximum if field == "v" else maximum / (index + 2)
            scores[field] = {
                "max_abs": value,
                "rms": value / 10.0,
                "candidate_at_argmax": value,
                "oracle_at_argmax": 0.0,
                "count": 10,
                "unequal": 1,
                "bit_identical": False,
                "argmax_jik": [0, 0] if field == "ssh" else [0, 0, 0],
            }
        growth.append({"step": step, "claim_label": "independent",
                       "error_rows": scores})
    month = {
        "status": "PASS_R186_MONTH_BOUNDARY",
        "claim_label": "independent",
        "initial_mode": "card_own_state",
        "missing_oracle_steps": [],
        "runtime_refusal": {"step": 96, "message": "registered"},
        "steps_completed": 95,
        "initial_entry": {
            name: {"bit_exact": True, "unequal": 0}
            for name in gate.prior.FIELDS
        },
        "growth_table": growth,
    }
    rows = []
    for kt in range(1, 11):
        for checkpoint in gate.prior.CHECKPOINTS:
            for field in gate.prior.FIELDS:
                rows.append({"kt": kt, "checkpoint": checkpoint,
                             "field": field, "max_abs": 1.0,
                             "first_unequal_index": [0]})
    ladder = {
        "status": "PASS_RUNG0_TEN_STEP_LADDER",
        "claim_label": "independent",
        "rows": rows,
        "first_non_bit_source_statement": copy.deepcopy(
            gate.prior.EXPECTED_STATEMENT),
    }
    return admission, month, ladder


def test_complete_growth_table_selects_entry_to_step10():
    result = gate.classify(*_fixtures())
    assert result["first_coarse_growth"]["step"] == 10
    assert result["coarse_growth"][-1]["step"] == 95
    assert result["coarse_growth"][-1]["over_10x"]


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant):
    with pytest.raises((gate.GateError, gate.prior.GateError)):
        gate.classify(*_fixtures(), plant=plant)
