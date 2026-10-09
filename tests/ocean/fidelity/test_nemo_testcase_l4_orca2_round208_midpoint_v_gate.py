import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round208_midpoint_v_gate as gate,
)


def _score(exact=True):
    return {"comparison_bit_exact": exact, "bit_exact": exact}


def _fixture():
    return {
        "claim_label": "independent OMT-0",
        "record_admission": {"streams": [
            {"header_valid": True, "defined_twin_status": "EXACT_DEFINED_BYTES"},
            {"header_valid": True, "defined_twin_status": "EXACT_DEFINED_BYTES"},
        ]},
        "coefficients": {
            "candidate": [1.0, 0.0, 0.0], "oracle": [1.0, 0.0, 0.0]},
        "offline_trace_passivity": {"ssh": True, "u": True, "v": True},
        "history_rotation": {
            "candidate": _score(), "oracle": _score()},
        "midpoint_replay": {
            "candidate_written_association": _score(),
            "candidate_midpoint_equals_now": _score(),
            "all_recorded": _score(),
            "first_effective_input": "v_entry",
        },
        "substep1_flux_form_update": {
            "status": "UNMEASURED_WITH_SPEC",
            "missing_oracle_operands": [
                "hv_e", "zhv_bck", "hv_0_times_1_plus_r3v_Kmm"],
        },
    }


def test_classifier_accepts_frozen_contract():
    assert gate.classify(_fixture())["status"] == "PASS_R208_MIDPOINT_V_SPLIT"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_each_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_fixture()), plant)
