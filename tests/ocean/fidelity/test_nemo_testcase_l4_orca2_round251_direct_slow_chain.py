from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round251_direct_slow_chain as gate,
)


def _row(exact: bool) -> dict[str, object]:
    return {
        "bit_exact": exact, "at_floor": exact,
        "unequal": 0 if exact else 35, "scored": 35,
        "max_abs": 0.0 if exact else 1.0, "argmax": [147, 0],
    }


def _report(label: str, first: str | None = None) -> dict[str, object]:
    rows = {name: _row(name != first) for name in gate.SOURCE_ORDER}
    return {
        "status": "PASS_R251_DIRECT_SLOW_CHAIN",
        "label": label,
        "passivity": {name: True for name in ("T", "S", "u", "v", "eta")},
        "override_absence": {
            "slow_forcing_incoming_override": True,
            "barotropic_slow_forcing_override": True,
            "slow_forcing_drag_override": True,
        },
        "source_order": list(gate.SOURCE_ORDER),
        "support": {"unequal": 35, "fold_only": True},
        "rows": rows,
        "self_replay": {"candidate": _row(True), "oracle": _row(True)},
        "first_unequal": first,
    }


def test_classifier_accepts_an_exact_direct_chain():
    result = gate.classify([_report(label) for label in gate.LABELS])
    assert result["status"] == "PASS_R251_DIRECT_CHAIN_EXACT"
    assert result["first_unequal"] is None


def test_classifier_names_a_real_direct_boundary():
    result = gate.classify([
        _report(label, "surface_v_mask") for label in gate.LABELS])
    assert result["status"] == "HELD_R251_DIRECT_CHAIN_FIRST_SURFACE_V_MASK"


@pytest.mark.parametrize("plant", ("label-coverage", "source-order"))
def test_classifier_plants_fail_closed(plant):
    with pytest.raises(gate.GateError):
        gate.classify(copy.deepcopy([_report(label) for label in gate.LABELS]),
                      plant=plant)


def test_final_statement_replay_is_sensitive_to_every_operand():
    import numpy as np

    incoming = np.asarray([[1.0]], np.float64)
    coriolis = np.asarray([[0.25]], np.float64)
    mask = np.asarray([[1.0]], np.float64)
    baseline = gate.r236._source_final(incoming, coriolis, mask)
    assert not np.array_equal(
        gate.r236._source_final(incoming * 2, coriolis, mask), baseline)
    assert not np.array_equal(
        gate.r236._source_final(incoming, coriolis * 2, mask), baseline)
    assert not np.array_equal(
        gate.r236._source_final(incoming, coriolis, mask * 0), baseline)
