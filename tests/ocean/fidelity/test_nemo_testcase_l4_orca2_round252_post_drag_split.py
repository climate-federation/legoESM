from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round252_post_drag_split as gate,
)


def _row(exact: bool) -> dict[str, object]:
    return {
        "bit_exact": exact, "at_floor": exact,
        "unequal": 0 if exact else 35, "scored": 35,
        "max_abs": 0.0 if exact else 1.0, "argmax": [147, 0],
    }


def _report(label: str) -> dict[str, object]:
    exact = {name: _row(True) for name in gate.SOURCE_ORDER}
    exact["live_reciprocal"] = _row(False)
    exact["compact_surface_mask"] = _row(False)
    exact["observed_post_drag"] = _row(False)
    return {
        "status": "PASS_R252_POST_DRAG_SPLIT",
        "label": label,
        "operand_route": "production-d114-resolved",
        "record_admission": [{"status": "EXACT_DEFINED_BYTES"}],
        "passivity": {"T": True, "S": True, "u": True, "v": True},
        "override_absence": {
            "slow_forcing_incoming_override": True,
            "barotropic_slow_forcing_override": True,
            "slow_forcing_drag_override": True,
        },
        "source_order": list(gate.SOURCE_ORDER),
        "support": {
            "owner_unequal": 35, "mask_unequal": 35, "changed_unequal": 35,
            "mask_outside_owner": 0, "changed_outside_owner": 0,
        },
        "rows": exact,
        "replays": {
            "compact_mask_to_observed": _row(True),
            "unmasked_to_oracle": _row(True),
            "resolved_post_drag_to_oracle": _row(True),
            "resolved_final_to_oracle": _row(True),
        },
        "first_nonbit_output": "observed_post_drag",
    }


def test_classifier_names_compact_drag_mask_owner():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "CONFIRMED_R252_COMPACT_DRAG_MASK_OWNER"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_classifier_plants_fail_closed(plant):
    reports = copy.deepcopy([_report(label) for label in gate.LABELS])
    with pytest.raises(gate.GateError):
        gate.classify(reports, plant=plant)


def test_mask_product_control_is_non_vacuous():
    import numpy as np

    value = np.asarray([[2.0]], np.float64)
    assert np.array_equal(gate._mask_product(value, [[1.0]]), value)
    assert not np.array_equal(gate._mask_product(value, [[0.0]]), value)
