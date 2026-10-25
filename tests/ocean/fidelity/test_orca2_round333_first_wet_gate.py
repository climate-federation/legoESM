"""Controls for the round-333 first-wet-value locator."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round333_first_wet_gate as gate,
)


def test_score_separates_signed_zero_from_a_value_difference() -> None:
    actual = np.array([[[-0.0], [2.0 + 8.0e-15]]], dtype=np.float64)
    expected = np.array([[[0.0], [2.0]]], dtype=np.float64)
    row = gate.located_score(actual, expected)
    try:
        assert row["unequal"] == 2
        assert row["value_unequal"] == 1
        assert row["classification"] == "OVER_BAR"
    finally:
        gate._DETAILS.clear()


def test_self_test_plant_fires() -> None:
    with pytest.raises(gate.GateError, match="plant stayed green"):
        gate.self_test(plant=True)


def test_checkpoint_mask_distinguishes_entry_and_stage_geometry() -> None:
    full = np.ones((148, 180, 30), dtype=bool)
    assert gate.checkpoint_mask(full, "entry").shape == (148, 180, 30)
    assert gate.checkpoint_mask(full, "stage3").shape == (148, 90, 30)
    with pytest.raises(gate.GateError, match="unknown checkpoint"):
        gate.checkpoint_mask(full, "after")
