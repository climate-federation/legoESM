"""Skill scoring of the standalone ERA5 demo lane (masked RMSE/bias)."""

from __future__ import annotations

import numpy as np

from scripts.validate.validate_wave_standalone import _score


def test_score_uses_the_valid_mask_and_persistence_baseline():
    predicted = np.asarray([1.0, 2.0, 100.0])
    observed = np.asarray([1.0, 4.0, 100.0])
    initial = np.asarray([0.0, 4.0, 0.0])
    valid = np.asarray([True, True, False])
    rmse, bias, persist, count = _score(predicted, observed, initial, valid)
    assert count == 2
    assert np.isclose(rmse, np.sqrt((0.0 + 4.0)/2.0))
    assert np.isclose(bias, -1.0)
    assert np.isclose(persist, np.sqrt((1.0 + 0.0)/2.0))
