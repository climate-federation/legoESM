import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round194_v_reciprocal_gate as gate,
)


def _exact():
    return {"bit_exact": True, "differing_cells": 0, "cells": 1, "max_abs": 0.0}


def _report():
    return {
        "claim_label": "independent hierarchy rung 0",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": {"coverage": "exactly-once", "rank_records": 2,
                           "substeps": 65},
        "private_arm": {
            "external_mode_association": True, "raw_reference_depth": True,
            "unmasked_v_transport": True, "materialize_v_transport": True,
            "unmasked_v_reciprocal": True,
        },
        "passivity": {"ssh": True},
        "independent_entry": {name: _exact() for name in ("T", "S", "u", "v", "ssh")},
        "rows": [{"substep": index, **{
            name: _exact() for name in ("incoming", "transport", "weight", "exit")}}
            for index in range(1, 66)],
        "masked_control": {"differing_substeps": 64,
                           "first_differing_substep": 2},
        "one_ulp_control": {"bit_exact": False, "differing_cells": 1},
    }


def test_clean_report_passes():
    assert gate.classify(_report())["status"] == "PASS_R194_V_RECIPROCAL_65_SUBSTEPS"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant)
