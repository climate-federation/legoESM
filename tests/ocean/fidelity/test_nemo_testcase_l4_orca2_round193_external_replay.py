import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round193_external_replay as gate,
)


def _row(name, at_floor, *, substep=None):
    row = {"name": name, "at_floor": at_floor}
    if substep is not None:
        row["substep"] = substep
    return row


def _report():
    rows = []
    for name in gate.r178.ENTRY_ORDER:
        rows.append(_row(name, True))
    for name in gate.r178.SUBSTEP_ORDER:
        rows.append(_row(name, True, substep=1))
    for name in gate.r178.SUBSTEP_ORDER:
        rows.append(_row(
            name, name != "transport_sum_v", substep=2))
        if name == "transport_sum_v":
            break
    exact = {"comparison_bit_exact": True}
    return {
        "claim_label": "independent hierarchy rung 0",
        "growth_report": {"sha256": gate.GROWTH_SHA256,
                          "first_kt": 1, "first_stage": 1},
        "record_census": {"coverage": "exactly-once"},
        "record_control": {"bit_exact": False, "differing_cells": 1},
        "observer_passivity": {"ssh": True},
        "private_arm": {
            "slow_forcing": "recorded_source_exact",
            "external_mode_association": True,
            "raw_reference_depth": True,
            "unmasked_v_transport": True,
            "materialize_v_transport": True,
        },
        "independent_entry": {field: {"bit_exact": True}
                              for field in ("T", "S", "u", "v", "ssh")},
        "source_rows": rows,
        "first_over_floor": copy.deepcopy(rows[-1]),
        "accumulation_split": {
            "previous_sum": exact,
            "completed_transport": exact,
            "weight": exact,
            "candidate": {"full_domain_differing_cells": 68},
            "unmasked_reciprocal_replay": exact,
        },
    }


def test_clean_report_names_first_statement():
    result = gate.classify(_report())
    assert result["first_statement"]["name"] == "transport_sum_v"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_report(), plant)


def test_compiled_source_registry_covers_substep_rows():
    assert set(gate.r178.SUBSTEP_ORDER) <= set(gate.COMPILED_SOURCE)
