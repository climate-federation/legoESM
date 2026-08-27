"""Unit controls for the pre-registered EUC mechanism reducers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH = (Path(__file__).parents[3] / "scripts" / "validate" / "ocean_fidelity"
         / "dino_1226" / "euc_mechanism.py")
_SPEC = importlib.util.spec_from_file_location("_euc_mechanism_test", _PATH)
E = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(E)


def test_weighted_nrms_is_thickness_weighted():
    lego = np.array([[2.0, 2.0]])
    nemo = np.array([[1.0, 1.0]])
    weights = np.array([[1.0, 3.0]])
    wet = np.ones_like(lego, dtype=bool)
    assert E.weighted_nrms(lego, nemo, weights, wet) == pytest.approx(1.0)


def test_empty_mask_fails_closed():
    a = np.ones((2, 2))
    with pytest.raises(ValueError, match="empty wet comparison"):
        E.weighted_nrms(a, a, a, np.zeros_like(a, dtype=bool))


def test_offline_bars_are_mutually_discriminating():
    assert E.classify_offline(0.05, np.array([0.98, 1.02])) == "REFUTE_CLOSURE_DIFFERENCE"
    assert E.classify_offline(0.30, np.array([0.70, 0.72])) == "CONFIRM_CLOSURE_DIFFERENCE"
    assert E.classify_offline(0.20, np.array([0.80, 1.20])) == "UNRESOLVED_CLOSURE_DIFFERENCE"


def test_wrong_shift_control_changes_vertical_mapping():
    a = np.arange(6.0).reshape(2, 3)
    shifted = E.wrong_shift(a)
    np.testing.assert_array_equal(shifted[:, :-1], a[:, 1:])
    assert np.isnan(shifted[:, -1]).all()
