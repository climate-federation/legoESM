import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round172_passive_rhs_replay as gate,
)


def _score(candidate, reference, *, exact=False):
    return {
        "bit_exact": exact,
        "differing_cells": 0 if exact else 1,
        "wet_cells": 1,
        "full_domain_differing_cells": 0 if exact else 1,
        "full_domain_cells": 1,
        "absolute_max": abs(candidate - reference),
        "rms": abs(candidate - reference),
        "argmax_jik": [0, 0, 0],
        "candidate_at_argmax": candidate,
        "oracle_at_argmax": reference,
        "candidate_max_abs": abs(candidate),
        "reference_max_abs": abs(reference),
        "explosive": abs(candidate) >= 1.0e20,
        "reference_explosive": False,
    }


def _report():
    rows = {face: {} for face in gate.FACES}
    for face in gate.FACES:
        for boundary in gate.BOUNDARIES:
            value = 1.0
            if face == "u" and boundary in (
                    "after_vor", "after_keg", "after_zad"):
                value = 1.0e30
            rows[face][boundary] = _score(value, 0.5)
    return {
        "admission": {"rank_coverage": "exactly-once", "records": [{}, {}]},
        "source_order": list(gate.BOUNDARIES),
        "trace_passivity": {
            str(kt): {
                "state": {name: True for name in ("T", "S", "u", "v", "ssh")},
                "same_graph_closure_u": True,
                "same_graph_closure_v": True,
                "offline_bridge": {"all": True},
            }
            for kt in range(1, 8)
        },
        "offline_closure": {
            "after_ldf_to_total_u": True,
            "after_ldf_to_total_v": True,
        },
        "rows": rows,
        "first_nonbit_accumulator": gate._first_nonbit(rows),
        "first_explosive_u": gate._first_explosive(rows, "u"),
        "one_ulp_control": {"differing_cells": 1, "bit_exact": False},
    }


def test_clean_report_keeps_finite_first_nonbit_and_explosive_vor_apart():
    result = gate.classify(_report())
    assert result["first_nonbit_accumulator"]["boundary"] == "after_hpg"
    assert result["first_explosive_u"]["boundary"] == "after_vor"
    assert result["prediction_dispositions"]["R172-P4"] == "CONFIRMED"
    assert result["prediction_dispositions"]["R172-P2"] == "REFUTED"
    assert result["prediction_dispositions"]["R172-P2a"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_wrong_explosive_prediction_is_retained_as_refuted():
    report = _report()
    report["rows"]["u"]["after_vor"] = _score(1.0, 0.5)
    report["rows"]["u"]["after_keg"] = _score(1.0e30, 0.5)
    report["first_explosive_u"] = gate._first_explosive(report["rows"], "u")
    result = gate.classify(report)
    assert result["first_explosive_u"]["boundary"] == "after_keg"
    assert result["prediction_dispositions"]["R172-P4"] == "REFUTED"


def test_one_ulp_control_is_mandatory():
    report = _report()
    report["one_ulp_control"]["differing_cells"] = 0
    with pytest.raises(gate.GateError, match="one-ULP"):
        gate.classify(report)
