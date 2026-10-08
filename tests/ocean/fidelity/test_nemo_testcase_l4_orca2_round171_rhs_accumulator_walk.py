import copy

import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _NEMOWSBarotropicRHSComponentTrace,
    _NEMOWSRK3TestHooks,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round171_rhs_accumulator_walk as gate,
)


def test_return_only_rhs_trace_is_private_and_default_off():
    hooks = _NEMOWSRK3TestHooks()
    assert hooks.expose_barotropic_rhs_component == ""
    selected = hooks._replace(expose_barotropic_rhs_component="vorticity")
    assert selected.expose_barotropic_rhs_component == "vorticity"
    assert _NEMOWSBarotropicRHSComponentTrace._fields[-2:] == (
        "operator_component_u", "operator_component_v")


def _score(candidate, reference, *, exact=False):
    return {
        "bit_exact": exact, "differing_cells": 1, "wet_cells": 1,
        "full_domain_differing_cells": 1, "full_domain_cells": 1,
        "absolute_max": abs(candidate - reference), "rms": abs(candidate - reference),
        "argmax_jik": [0, 0, 0], "candidate_at_argmax": candidate,
        "oracle_at_argmax": reference, "candidate_max_abs": abs(candidate),
        "reference_max_abs": abs(reference),
        "explosive": abs(candidate) >= 1.0e20,
        "reference_explosive": False,
    }


def _report():
    rows = {face: {} for face in gate.FACES}
    for face in gate.FACES:
        for boundary in gate.BOUNDARIES:
            value = 1.0
            if face == "u" and boundary in ("after_vor", "after_keg", "after_zad"):
                value = 1.0e30
            rows[face][boundary] = _score(value, 0.5)
    return {
        "admission": {"rank_coverage": "exactly-once", "records": [{}, {}]},
        "source_order": list(gate.BOUNDARIES), "rows": rows,
        "source_order_closure": {"u": {"bit_exact": True}, "v": {"bit_exact": True}},
        "observer_passivity": {
            "full_step_prediction": "REFUTED_IN_PRIOR_RUN",
            "kt1_to_7_barotropic": {str(k): {"T": True} for k in range(1, 8)},
            "kt8_barotropic": {"T": True, "completed_rhs_u": True,
                               "completed_rhs_v": True},
        },
    }


def test_clean_report_names_vor_and_retains_first_nonbit():
    result = gate.classify(_report())
    assert result["first_explosive_u"]["boundary"] == "after_vor"
    assert result["first_nonbit_accumulator"]["boundary"] == "after_hpg"
    assert result["prediction_dispositions"]["R171-P3"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_all_plants_fire(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_wrong_prediction_is_retained_as_refuted():
    report = _report()
    for boundary in gate.BOUNDARIES:
        report["rows"]["u"][boundary] = _score(1.0, 0.5)
    report["rows"]["u"]["after_keg"] = _score(1.0e30, 0.5)
    report["rows"]["u"]["after_zad"] = _score(1.0e30, 0.5)
    result = gate.classify(report)
    assert result["first_explosive_u"]["boundary"] == "after_keg"
    assert result["prediction_dispositions"]["R171-P3"] == "REFUTED"


def test_source_order_closure_failure_is_retained_not_called_observer_movement():
    report = _report()
    report["source_order_closure"]["u"]["bit_exact"] = False
    result = gate.classify(report)
    assert result["prediction_dispositions"]["R171-P2"] == "REFUTED"
    assert all(result["observer_passivity"]["kt8_barotropic"].values())
