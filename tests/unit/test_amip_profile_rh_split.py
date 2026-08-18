"""Direct tests for scripts/validate/amip_bias/profile_rh_split.py.

Only the pure reductions are exercised — the probe's data paths point at a
campaign run tree that is not part of the repo, so ``main`` is out of scope.
What IS in scope is every piece of arithmetic a wrong number could hide in:
the RH identity the whole attribution rests on, the humidity-convention
conversion, and the masked area mean that replaced an unmasked one after it
was found to be averaging below-ground pressure levels into the answer.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prs():
    return _load("profile_rh_split")


@pytest.fixture(scope="module")
def rb():
    return _load("regional_bias")


def test_rh_split_is_an_exact_identity(prs):
    """RH_m - RH_e must equal (RH_m - RH_swapT) + (RH_swapT - RH_e) EXACTLY.

    The whole temperature-vs-humidity attribution is this telescoping sum. If
    it only held approximately, the two parts would be a regression rather
    than a decomposition and could not be quoted separately.
    """
    rng = np.random.default_rng(0)
    plev = np.array([85000.0, 70000.0, 50000.0])
    T_m = 280.0 + rng.normal(0.0, 5.0, (3, 4, 5))
    T_e = T_m + rng.normal(0.0, 3.0, (3, 4, 5))
    w_m = np.abs(rng.normal(5e-3, 1e-3, (3, 4, 5)))
    w_e = np.abs(rng.normal(5e-3, 1e-3, (3, 4, 5)))

    rh_m = prs._rh(T_m, w_m, plev)
    rh_e = prs._rh(T_e, w_e, plev)
    rh_swap = prs._rh(T_e, w_m, plev)

    residual = (rh_m - rh_e) - ((rh_m - rh_swap) + (rh_swap - rh_e))
    assert np.max(np.abs(residual)) == 0.0


def test_rh_responds_to_temperature_at_fixed_humidity(prs):
    """Non-vacuity for the identity above: the dT part must not be ~0.

    Saturation falls with temperature, so a colder column at unchanged mixing
    ratio is MORE saturated. A test that only checked the sum would pass on a
    probe whose two parts were both zero.
    """
    plev = np.array([70000.0])
    w = np.full((1, 1, 1), 5e-3)
    rh_warm = prs._rh(np.full((1, 1, 1), 280.0), w, plev)
    rh_cold = prs._rh(np.full((1, 1, 1), 275.0), w, plev)
    assert rh_cold[0, 0, 0] > rh_warm[0, 0, 0]
    assert rh_cold[0, 0, 0] - rh_warm[0, 0, 0] > 0.05      # ~7 %/K over 5 K


def test_specific_humidity_to_mixing_ratio(prs):
    """w = q / (1 - q).  ERA5 publishes q; the model's saturation curve, and
    hence its RH, is in w.  Skipping this biases the reference RH low by
    ~(1 - q), which at a tropical q of 0.02 is the same size as the signal."""
    assert prs._to_mixing_ratio(0.02) == pytest.approx(0.02 / 0.98)
    assert prs._to_mixing_ratio(0.0) == 0.0
    # Monotone and always >= q.
    q = np.array([1e-6, 1e-3, 1e-2, 3e-2])
    w = prs._to_mixing_ratio(q)
    assert np.all(w >= q)
    assert np.all(np.diff(w) > 0)


def test_region_mean_valid_mask_renormalises_weights(rb):
    """A masked cell must be REMOVED from the denominator, not counted as zero.

    This is the defect the mask was added for: averaging below-ground levels in
    reported a global 1000 hPa cold bias of -3.9 K where the masked value is
    -1.3 K.
    """
    lat = np.array([-45.0, 45.0])
    lon = np.array([90.0, 270.0])
    field = np.array([[10.0, 10.0], [2.0, 2.0]])
    box = (-90, 90, 0, 360)

    # cos(45) is equal in both rows, so the unmasked mean is 6.0.
    assert rb.region_mean(field, lat, lon, box) == pytest.approx(6.0)
    # Masking the value-10 row must give 2.0, NOT 3.0 (which is what treating
    # the masked cells as zeros in a fixed denominator would give).
    valid = np.array([[False, False], [True, True]])
    assert rb.region_mean(field, lat, lon, box, valid) == pytest.approx(2.0)


def test_region_mean_refuses_an_empty_region(rb):
    """An all-masked region must raise, not return NaN.

    A silent NaN propagates into a printed table and reads as a measurement.
    """
    lat = np.array([-45.0, 45.0])
    lon = np.array([90.0, 270.0])
    field = np.ones((2, 2))
    with pytest.raises(SystemExit):
        rb.region_mean(field, lat, lon, (-90, 90, 0, 360), np.zeros((2, 2), bool))


def test_region_mean_nan_outside_the_valid_mask_cannot_leak(rb):
    """NaN in a masked-out cell must not poison the result.

    The probe writes NaN into below-ground cells before reducing, so the mean
    has to be immune to NaN wherever ``valid`` is False.
    """
    lat = np.array([-45.0, 45.0])
    lon = np.array([90.0, 270.0])
    field = np.array([[np.nan, np.nan], [2.0, 2.0]])
    valid = np.array([[False, False], [True, True]])
    got = rb.region_mean(field, lat, lon, (-90, 90, 0, 360), valid)
    assert np.isfinite(got)
    assert got == pytest.approx(2.0)


def test_bin_to_model_conserves_a_uniform_field(rb):
    """Area-weighted binning of a constant must return that constant."""
    rlat = np.linspace(-89.5, 89.5, 180)
    rlon = np.linspace(0.5, 359.5, 360)
    mlat = np.linspace(-87.5, 87.5, 36)
    mlon = np.linspace(2.5, 357.5, 72)
    out = rb.bin_to_model(np.full((180, 360), 7.0), rlat, rlon, mlat, mlon)
    assert out.shape == (36, 72)
    assert np.allclose(out, 7.0)


def test_bin_to_model_names_the_variable_when_a_cell_has_no_reference(rb):
    """The fatal message must identify WHICH variable left cells empty."""
    rlat = np.array([0.0])
    rlon = np.array([0.0])
    mlat = np.linspace(-87.5, 87.5, 36)
    mlon = np.linspace(2.5, 357.5, 72)
    with pytest.raises(SystemExit, match="rsut"):
        rb.bin_to_model(np.array([[1.0]]), rlat, rlon, mlat, mlon, label="rsut")


@pytest.fixture(scope="module")
def kern():
    return _load("rlutcs_kernel")


def test_on_columns_interpolates_in_log_pressure(kern):
    """A profile given on the (descending) plev axis must land on p_full at
    the right pressures, interpolated in log-p."""
    plev = np.array([100000.0, 50000.0, 10000.0])
    prof = np.array([0.0, 4.0, 8.0])
    p_full = np.array([100000.0, np.sqrt(100000.0 * 50000.0), 50000.0, 10000.0])
    got = kern._on_columns(plev, prof, p_full, fill=0.0)
    assert got[0] == pytest.approx(0.0)
    assert got[2] == pytest.approx(4.0)
    assert got[3] == pytest.approx(8.0)
    # geometric midpoint in pressure = arithmetic midpoint in log-p
    assert got[1] == pytest.approx(2.0)


def test_on_columns_holds_the_fill_outside_the_measured_range(kern):
    """Outside the plev range the perturbation must be the IDENTITY.

    Extrapolating the measured +11 K upper-level bias into the model's lid
    would manufacture the answer the probe exists to measure.
    """
    plev = np.array([100000.0, 10000.0])
    prof = np.array([0.0, 10.0])
    p_full = np.array([120000.0, 1000.0])          # below the base, above the top
    assert np.all(kern._on_columns(plev, prof, p_full, fill=0.0) == 0.0)
    assert np.all(kern._on_columns(plev, prof, p_full, fill=1.0) == 1.0)
