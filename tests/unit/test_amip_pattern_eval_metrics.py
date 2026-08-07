"""Metric math of scripts/plot/plot_amip_pattern_eval.py (2026-07-24 ladder).

The pattern-correlation / centered-RMSE / area-weight helpers are the load-
bearing numerics of the AMIP pattern evaluation — pin them against analytic
cases so a silent weighting or centering slip cannot grade a run wrong.
"""

from __future__ import annotations

import re
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


# ---------------------------------------------------------------------------
# --months window selector: comparing two runs of DIFFERENT length requires
# pinning the calendar window on both, else the difference is a sampling
# confound.  A typo must fail loudly rather than silently reshape the window.
# ---------------------------------------------------------------------------

def test_parse_months_none_means_every_month_written():
    assert _mod.parse_months("") is None
    assert _mod.parse_months(None) is None


def test_parse_months_range_and_list():
    assert _mod.parse_months("1-8") == [1, 2, 3, 4, 5, 6, 7, 8]
    assert _mod.parse_months("1,2,12") == [1, 2, 12]
    assert _mod.parse_months("3") == [3]
    # union, de-duplicated and sorted
    assert _mod.parse_months("6-8,1,7") == [1, 6, 7, 8]


@pytest.mark.parametrize("spec", ["0-5", "12-13", "13", "0", "8-2"])
def test_parse_months_rejects_out_of_range_or_reversed(spec):
    with pytest.raises(ValueError):
        _mod.parse_months(spec)


# ---------------------------------------------------------------------------
# Meridional mass streamfunction (Hadley panel): pin the integral against the
# closed-form uniform-wind solution and the stated sign convention, so a
# flipped sign or dropped cos(lat) can never mislabel the Hadley cells.
# ---------------------------------------------------------------------------

from legoesm import constants  # noqa: E402

_A = constants.R_earth
_G = constants.g

PLEV = np.array([100e2, 300e2, 500e2, 700e2, 900e2])  # Pa, TOA first


def test_streamfunction_uniform_v_matches_closed_form():
    """[v] = v0 everywhere: psi(phi, p) = 2 pi a cos(phi) v0 (p - ptop) / g."""
    v0 = 2.0
    v = np.full((PLEV.size, LAT.size), v0)
    psi = _mod.meridional_streamfunction(v, PLEV, LAT, _A, _G)
    expect = (2.0 * np.pi * _A * np.cos(np.deg2rad(LAT))[None, :]
              * v0 * (PLEV[:, None] - PLEV[0]) / _G)
    np.testing.assert_allclose(psi, expect, rtol=1e-12)


def test_streamfunction_zero_at_top_and_sign_convention():
    """Positive (northward) v gives psi > 0 below TOA — the NH-Hadley sense
    documented in the helper.  psi at the top level is identically zero."""
    v = np.full((PLEV.size, LAT.size), 1.0)
    psi = _mod.meridional_streamfunction(v, PLEV, LAT, _A, _G)
    np.testing.assert_array_equal(psi[0], 0.0)
    assert np.all(psi[1:, 1:-1] > 0.0)   # (poles excluded: cos ~ 0)


def test_streamfunction_level_order_invariance():
    """Descending-pressure input returns the same field in the caller's
    order — CMOR plev direction must not flip the answer."""
    rng = np.random.default_rng(7)
    v = rng.normal(0.0, 5.0, size=(PLEV.size, LAT.size))
    psi = _mod.meridional_streamfunction(v, PLEV, LAT, _A, _G)
    psi_rev = _mod.meridional_streamfunction(
        v[::-1], PLEV[::-1], LAT, _A, _G)
    np.testing.assert_allclose(psi_rev, psi[::-1], rtol=1e-12)


def test_streamfunction_nan_below_ground_contributes_zero():
    """NaN [v] (below-ground plev) must act as zero mass flux, not poison
    the column: identical to explicitly zeroed input."""
    v = np.full((PLEV.size, LAT.size), 3.0)
    v_nan = v.copy()
    v_nan[-2:, :10] = np.nan
    v_zero = v.copy()
    v_zero[-2:, :10] = 0.0
    psi_nan = _mod.meridional_streamfunction(v_nan, PLEV, LAT, _A, _G)
    psi_zero = _mod.meridional_streamfunction(v_zero, PLEV, LAT, _A, _G)
    assert np.all(np.isfinite(psi_nan))
    np.testing.assert_allclose(psi_nan, psi_zero, rtol=1e-12)


# ---------------------------------------------------------------------------
# Near-sea-level mask (psl-vs-ERA5-ps row): the mask is what keeps orography
# out of the circulation comparison, so its polarity must be pinned.
# ---------------------------------------------------------------------------

def test_low_elevation_mask_polarity_and_nan():
    psl = np.array([[1013.0, 1013.0, 1013.0, np.nan]])
    ps = np.array([[1012.0, 950.0, 1013.0, 1013.0]])
    mask = _mod.low_elevation_mask(psl, ps, tol=3.0)
    # sea-level cell kept, mountain cell (63 hPa apart) dropped, exact kept,
    # NaN dropped
    np.testing.assert_array_equal(mask, [[True, False, True, False]])


# --------------------------------------------------------------------------
# --years window pin (guards the short-run vs long-run sampling confound)
# --------------------------------------------------------------------------

def test_parse_years_single_range_and_list():
    parse_years = _mod.parse_years
    assert parse_years("1979") == [1979]
    assert parse_years("1979-1981") == [1979, 1980, 1981]
    assert parse_years("1979,1981") == [1979, 1981]
    assert parse_years("") is None
    assert parse_years(None) is None


def test_parse_years_rejects_reversed_range():
    parse_years = _mod.parse_years
    with pytest.raises(ValueError, match="reversed"):
        parse_years("1983-1979")


def test_parse_years_rejects_non_integer():
    parse_years = _mod.parse_years
    with pytest.raises(ValueError):
        parse_years("nineteen-seventy-nine")


def test_open_cmor_accepts_years_argument():
    """The year filter must be threaded to every consumer, not just one."""
    import inspect
    m = _mod
    assert "years" in inspect.signature(m._open_cmor).parameters
    src = inspect.getsource(m.main)
    assert "parse_years(args.years)" in src
    # every _open_cmor call inside main must pass the year window through
    for call in re.findall(r"_open_cmor\([^)]*\)", src):
        assert "years" in call, f"year window not threaded: {call}"
