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
    rows["live_reciprocal"] = _row(False)
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
        "first_unequal": "live_reciprocal",
        "self_replay": {"candidate": _row(False), "oracle": _row(True)},
        "substitutions": {"through_live_reciprocal": _row(True)},
    }


def _reports() -> list[dict[str, object]]:
    return [_report(label) for label in gate.LABELS]


def test_classifier_retracts_non_source_exact_drag_owner():
    result = gate.classify(_reports())
    assert result["status"] == "REFUTED_R250_DRAG_OWNER_NON_SOURCE_EXACT_ARM"
    assert result["predictions"]["R250-P2"].startswith("REFUTED")


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_round250_plant_refuses(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy(_reports()), plant=plant)
