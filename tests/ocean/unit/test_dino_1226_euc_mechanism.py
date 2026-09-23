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
    # Non-uniform error: an unweighted implementation returns sqrt(1/2), while
    # the 3x-thicker exact cell reduces the registered answer to 1/2.
    lego = np.array([[2.0, 1.0]])
    nemo = np.array([[1.0, 1.0]])
    weights = np.array([[1.0, 3.0]])
    wet = np.ones_like(lego, dtype=bool)
    assert E.weighted_nrms(lego, nemo, weights, wet) == pytest.approx(0.5)


def test_empty_mask_fails_closed():
    a = np.ones((2, 2))
    with pytest.raises(ValueError, match="empty wet comparison"):
        E.weighted_nrms(a, a, a, np.zeros_like(a, dtype=bool))


def test_offline_bars_are_mutually_discriminating():
    assert E.classify_offline(0.05, np.array([0.98, 1.02])) == "REFUTE_CLOSURE_DIFFERENCE"
    assert E.classify_offline(0.30, np.array([0.70, 0.72])) == "CONFIRM_CLOSURE_DIFFERENCE"
    assert E.classify_offline(0.20, np.array([0.80, 1.20])) == "UNRESOLVED_CLOSURE_DIFFERENCE"


def test_review_bar_allows_single_shear_setting_level():
    ratios = np.array([2.05, 0.99, 1.01])
    depths = np.array([10.14, 20.59, 31.43])
    assert E.classify_offline_review(0.689, ratios, depths) == "CONFIRM_VISCOSITY_PRIME_SUSPECT"


def test_attribution_names_energy_when_energy_owns_exact_factorization():
    nemo_e = np.ones(4)
    lego_e = 4.0 * nemo_e
    mxl = np.ones(4)
    nemo_avm = 0.1 * mxl * np.sqrt(nemo_e)
    lego_avm = 0.1 * mxl * np.sqrt(lego_e)
    got = E.closure_attribution(
        lego_avm, nemo_avm, lego_e, nemo_e, mxl, mxl,
        np.ones(4, dtype=bool), c_lego=0.1, c_nemo=0.1,
        floor_lego=1e-4, floor_nemo=1e-4)
    assert got["label"] == "TKE_ENERGY_CARRIES_10M_EXCESS"
    assert got["max_direct_reconstruction_relative_error"] < 1e-14


def test_single_root_requires_buoyancy_limb_and_normalized_length_closure():
    nemo_en = np.ones(4)
    lego_en = np.full(4, 2.25)
    n2 = np.full(4, 2.0)
    nemo_mxl = np.sqrt(2.0 * nemo_en / n2)
    lego_mxl = np.sqrt(2.0 * lego_en / n2)
    nemo_avm = 0.1 * nemo_mxl * np.sqrt(nemo_en)
    lego_avm = 0.1 * lego_mxl * np.sqrt(lego_en)
    got = E.single_root_energy_test(
        lego_avm, nemo_avm, lego_en, nemo_en, lego_mxl, nemo_mxl,
        n2, n2, np.ones(4, dtype=bool), lego_mxl_min=1e-6,
        floor_lego=1e-4, floor_nemo=1e-4)
    assert got["verdict"] == "CONFIRM_TKE_ENERGY_SINGLE_ROOT"
    assert got["normalized_mxl_over_sqrt_en_ratio"]["geometric_mean"] == pytest.approx(1.0)
    assert got["active_limb"]["nemo_buoyancy_limited_fraction"] == 1.0


def test_wrong_shift_control_changes_vertical_mapping():
    a = np.arange(6.0).reshape(2, 3)
    shifted = E.wrong_shift(a)
    np.testing.assert_array_equal(shifted[:, :-1], a[:, 1:])
    assert np.isnan(shifted[:, -1]).all()
