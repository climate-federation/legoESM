import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round240_r239_reconciliation as gate,
)


def _row(*, exact):
    return {
        "bit_exact": exact, "at_floor": exact,
        "differing_cells": 0 if exact else 68, "compared_cells": 68,
        "absolute_max": 0.0 if exact else 1.0, "rms": 0.0 if exact else 1.0,
        "argmax": [147, 0], "candidate_at_argmax": 0.0,
        "oracle_at_argmax": 0.0 if exact else 1.0,
    }


def _report():
    return {
        "claim_label": "independent hierarchy rung 0",
        "target": {"cells": 68, "row": 147, "wet_levels": 1319},
        "raw_hpg": _row(exact=True),
        "depth_endpoints": {
            "legacy_diagnostic": _row(exact=False),
            "production": _row(exact=True),
        },
        "routes": {
            "legacy_diagnostic": "nemo_qco_card_mesh_operands",
            "production": "nemo_qco_resolved_mesh_operands",
        },
        "geometry_route_rows": {"e3v": _row(exact=False)},
    }


def test_classifier_reconciles_the_stale_geometry_route():
    result = gate.classify(_report())
    assert result["status"] == "PASS_R240_R239_STALE_GEOMETRY_DIAGNOSTIC"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_report()), plant=plant)
