from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round250_drag_split as gate,
)


def _row(exact: bool) -> dict[str, object]:
    return {
        "bit_exact": exact,
        "at_floor": exact,
        "unequal": 0 if exact else 35,
        "scored": 35,
        "max_abs": 0.0 if exact else 1.0,
        "argmax": [147, 0],
    }


def _report(label: str) -> dict[str, object]:
    rows = {name: _row(True) for name in gate.OPERAND_ORDER}
    rows["drag_coefficient"] = _row(False)
    rows["bottom_residual"] = _row(False)
    rows["drag_increment"] = _row(False)
    rows["post_drag"] = _row(False)
    return {
        "status": "PASS_R250_DRAG_SPLIT",
        "label": label,
        "operand_route": "production-d114-resolved",
        "passivity": {name: True for name in ("T", "S", "u", "v", "eta")},
        "record_admission": [{"status": "EXACT_DEFINED_BYTES"}],
        "source_order": list(gate.OPERAND_ORDER),
        "owner_support": {"unequal": 35},
        "rows": rows,
        "first_unequal": "drag_coefficient",
        "self_replay": {"candidate": _row(True), "oracle": _row(True)},
        "substitutions": {"through_drag_coefficient": _row(True)},
    }


def _reports() -> list[dict[str, object]]:
    return [_report(label) for label in gate.LABELS]


def test_classifier_names_drag_coefficient_owner():
    result = gate.classify(_reports())
    assert result["status"] == "CONFIRMED_R250_DRAG_COEFFICIENT_OWNER"
    assert result["predictions"]["R250-P2"] == "CONFIRMED"


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round250_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_reports()), plant=plant)
