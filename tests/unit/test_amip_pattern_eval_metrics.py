"""Metric math of scripts/plot/plot_amip_pattern_eval.py (2026-07-24 ladder).

The pattern-correlation / centered-RMSE / area-weight helpers are the load-
bearing numerics of the AMIP pattern evaluation — pin them against analytic
cases so a silent weighting or centering slip cannot grade a run wrong.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "plot_amip_pattern_eval",
    Path(__file__).resolve().parents[2]
    / "scripts" / "plot" / "plot_amip_pattern_eval.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


LAT = np.linspace(-87.5, 87.5, 36)


def test_area_weights_mean_one_and_cosine_shape():
    w = _mod.area_weights(LAT)
    assert w.shape == (36,)
    assert w.mean() == pytest.approx(1.0)
    # equator heavier than poles, symmetric
    assert w[18] > w[0] and w[0] == pytest.approx(w[-1])


def test_identical_fields_perfect_pattern():
    rng = np.random.default_rng(0)
    f = rng.normal(280.0, 10.0, size=(36, 72))
    st = _mod.pattern_stats(f, f, LAT)
    assert st["bias"] == pytest.approx(0.0, abs=1e-12)
    assert st["rmse_centered"] == pytest.approx(0.0, abs=1e-9)
    assert st["pattern_corr"] == pytest.approx(1.0)


def test_mean_offset_is_bias_only():
    """A uniform +2 offset is PURE bias: centered RMSE and correlation must
    be unaffected (the Taylor centering contract)."""
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, size=(36, 72))
    st = _mod.pattern_stats(ref + 2.0, ref, LAT)
    assert st["bias"] == pytest.approx(2.0)
    assert st["rmse_centered"] == pytest.approx(0.0, abs=1e-9)
    assert st["pattern_corr"] == pytest.approx(1.0)


def test_anticorrelated_pattern():
    rng = np.random.default_rng(2)
    ref = rng.normal(0.0, 1.0, size=(36, 72))
    st = _mod.pattern_stats(-ref, ref, LAT)
    assert st["pattern_corr"] == pytest.approx(-1.0)


def test_weighting_is_latitudinal_not_uniform():
    """A perturbation at the pole must matter LESS than the same perturbation
    at the equator (cos-lat weighting actually applied)."""
    ref = np.zeros((36, 72))
    pole = ref.copy();  pole[0, :] = 5.0     # ~87.5S row
    eq = ref.copy();    eq[18, :] = 5.0      # ~2.5N row
    st_pole = _mod.pattern_stats(pole, ref, LAT)
    st_eq = _mod.pattern_stats(eq, ref, LAT)
    assert st_eq["rmse_centered"] > st_pole["rmse_centered"]


def test_nan_reference_cells_excluded_pairwise():
    rng = np.random.default_rng(3)
    ref = rng.normal(0.0, 1.0, size=(36, 72))
    ref_nan = ref.copy()
    ref_nan[5:8, 10:20] = np.nan
    st = _mod.pattern_stats(ref, ref_nan, LAT)
    assert st["pattern_corr"] == pytest.approx(1.0)
    assert st["rmse_centered"] == pytest.approx(0.0, abs=1e-9)


def test_weighted_global_mean_uniform_field():
    f = np.full((36, 72), 3.25)
    assert _mod.weighted_global_mean(f, LAT) == pytest.approx(3.25)
