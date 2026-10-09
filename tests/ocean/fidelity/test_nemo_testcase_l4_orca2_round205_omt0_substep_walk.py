import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round205_omt0_substep_walk as gate,
)


def _row(name="eta_entry", *, exact=True):
    return {
        "substep": 1,
        "name": name,
        "bit_exact": exact,
        "comparison_bit_exact": exact,
        "at_floor": exact,
    }


def _report():
    exact = _row()
    return {
        "claim_label": "independent OMT-0",
        "record_admission": {
            "stream_count": 2,
            "streams": [
                {"header_valid": True, "defined_twin_status": "EXACT_DEFINED_BYTES"},
                {"header_valid": True, "defined_twin_status": "EXACT_DEFINED_BYTES"},
            ],
        },
        "offline_replay_passivity": {
            "ssh": True,
            "u_barotropic": True,
            "v_barotropic": True,
            "transport_u": True,
            "transport_v": True,
        },
        "source_order": list(gate.ENTRY_ORDER),
        "terminal_ulp_control": {"bit_exact": False, "differing_cells": 1},
        "slow_v_arm": {
            "input": {"comparison_bit_exact": True},
            "substep_table": [{"substep": index + 1} for index in range(65)],
        },
        "slow_v_association_unit": {
            "substep_table": [{"substep": index + 1} for index in range(65)],
        },
        "substep_table": [{"substep": index + 1} for index in range(65)],
        "source_rows": [exact],
        "substep_rows": [exact],
        "first_nonbit": None,
        "first_over_floor": None,
    }


def test_clean_report_passes_and_preserves_source_order():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R205_OMT0_SUBSTEP_WALK"
    assert tuple(result["source_order"]) == gate.ENTRY_ORDER


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)


def test_first_selector_names_earliest_source_ordered_row():
    report = _report()
    report["substep_rows"] = [
        _row("eta_entry"),
        _row("u_entry", exact=False),
        _row("eta_mid", exact=False),
    ]
    report["source_rows"] = report["substep_rows"]
    report["first_nonbit"] = report["source_rows"][1]
    report["first_over_floor"] = report["source_rows"][1]
    result = gate.classify(report)
    assert result["first_nonbit"]["name"] == "u_entry"
