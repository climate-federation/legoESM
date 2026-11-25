import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round173_hpg_operand_walk as gate,
)


def _score(value=0.0, *, exact=True):
    return {
        "bit_exact": exact,
        "differing_cells": 0 if exact else 1,
        "wet_cells": 1,
        "full_domain_differing_cells": 0 if exact else 1,
        "full_domain_cells": 1,
        "absolute_max": abs(value),
        "rms": abs(value),
        "argmax_jik": [0, 0, 0],
        "candidate_at_argmax": value,
        "oracle_at_argmax": 0.0,
        "candidate_max_abs": abs(value),
        "reference_max_abs": 0.0,
        "explosive": abs(value) >= 1.0e20,
    }


def _report():
    rows = {name: _score() for name in gate.WALK_ORDER}
    rows["rhd"] = _score(1.0e30, exact=False)
    return {
        "admission": {"rank_coverage": "exactly-once", "records": [{}, {}]},
        "recorded_self_replay": {name: _score() for name in gate.STATEMENT_ORDER},
        "operand_capture_matches_unexposed_hpg": {"u": _score(), "v": _score()},
        "candidate_literal_matches_live_hpg": {"u": _score(), "v": _score()},
        "one_ulp_control": {"differing_cells": 1, "bit_exact": False},
        "rows": rows,
        "first_nonbit": gate._first_nonbit(rows),
        "recorded_rhd_arm": {
            "u": {"candidate_max_abs": 1.0, "absolute_max": 1.0},
            "v": {"candidate_max_abs": 1.0, "absolute_max": 1.0},
            "u_improvement_factor": 1.0e30,
            "v_improvement_factor": 1.0e30,
        },
    }


def test_clean_report_selects_rhd_and_confirms_frozen_predictions():
    result = gate.classify(_report())
    assert result["first_nonbit"]["boundary"] == "rhd"
    assert result["prediction_dispositions"]["R173-P3"] == "CONFIRMED"
    assert result["prediction_dispositions"]["R173-P4"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_finite_earlier_difference_owns_before_explosive_later_row():
    report = _report()
    report["rows"]["rhd"] = _score(1.0, exact=False)
    report["rows"]["e3w"] = _score(1.0e30, exact=False)
    report["first_nonbit"] = gate._first_nonbit(report["rows"])
    result = gate.classify(report)
    assert result["first_nonbit"]["boundary"] == "rhd"
    assert result["prediction_dispositions"]["R173-P3"] == "REFUTED"


def test_self_replay_is_mandatory():
    report = _report()
    report["recorded_self_replay"]["zuap_v"]["bit_exact"] = False
    with pytest.raises(gate.GateError, match="recorded operands"):
        gate.classify(report)


def test_operand_capture_must_be_passive():
    report = _report()
    report["operand_capture_matches_unexposed_hpg"]["u"]["bit_exact"] = False
    with pytest.raises(gate.GateError, match="operand exposure"):
        gate.classify(report)
