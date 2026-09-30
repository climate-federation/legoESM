"""Direct test for ``scripts/validate/amip_bias/aerosol_ccn_audit.py`` (#1521).

The audit's number went into an issue, so its two pieces of real logic get
pinned: the area weighting that turns a zonal field into a global mean, and
the guarantee that the droplet number comes from the MODEL's inversion rather
than a retyped fit.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "amip_bias" / "aerosol_ccn_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("_aodaudit", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_aodaudit"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cos_weights_sum_to_one_and_favour_the_tropics():
    mod = _load()
    lat = np.array([-60.0, 0.0, 60.0])
    w = mod.cos_lat_weights(lat)
    assert np.isclose(w.sum(), 1.0)
    assert w[1] > w[0] and w[1] > w[2], "the equator must outweigh 60 degrees"
    np.testing.assert_allclose(w[0], w[2], rtol=1e-12)


def test_area_weighting_is_not_a_plain_mean():
    """The distinction is load-bearing: the poles are the CLEANEST rows on
    this field, so an unweighted mean would flatter the answer."""
    mod = _load()
    lat = np.array([-60.0, 0.0, 60.0])
    field = np.array([[10.0, 100.0, 10.0]])          # tropics-heavy signal
    got = mod.area_weighted_annual_mean(field, lat)
    assert not np.isclose(got, field.mean()), "weighting had no effect"
    w = np.cos(np.deg2rad(lat)); w = w / w.sum()
    np.testing.assert_allclose(got, float((field[0] * w).sum()), rtol=1e-12)


def test_annual_mean_precedes_the_area_mean():
    """Two latitudes symmetric about the equator carry EQUAL cos weights, so
    the expected value is the plain mean and only the ordering is under test.

    (A degenerate [0.0, 0.0] axis would also give equal weights, but it now
    trips the radians guard -- correctly, since no real latitude axis spans
    less than 2*pi degrees.)
    """
    mod = _load()
    lat = np.array([-30.0, 30.0])
    field = np.array([[1.0, 3.0], [3.0, 5.0]])
    np.testing.assert_allclose(mod.area_weighted_annual_mean(field, lat), 3.0)


def test_transposed_or_mismatched_input_is_rejected():
    """A silently transposed array would weight months by cos(lat)."""
    mod = _load()
    lat = np.array([-60.0, 0.0, 60.0])
    with pytest.raises(ValueError, match="wrong file or transposed"):
        mod.area_weighted_annual_mean(np.zeros((3, 5)), lat)
    with pytest.raises(ValueError, match="expected"):
        mod.area_weighted_annual_mean(np.zeros(3), lat)


def test_droplet_number_is_the_models_own_inversion():
    """Not a retyped fit: the probe must agree with ``ccn_from_aod`` exactly."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.microphysics.aerosol_activation import (
        CCNFromAODConfig, ccn_from_aod)
    mod = _load()
    aod = np.array([[0.02, 0.05, 0.15, 0.25]])
    got = mod.implied_ccn_cm3(aod)
    ref = np.asarray(ccn_from_aod(jnp.asarray(aod), CCNFromAODConfig())) / 1.0e6
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)
    # And it must actually vary with AOD, or the comparison above is vacuous.
    assert got[0, -1] > got[0, 0] * 2.0


def test_a_radian_latitude_axis_is_rejected():
    """The real hazard: radians would give a UNIFORM mean wearing the name of
    an area-weighted one, with no error at all."""
    mod = _load()
    with pytest.raises(ValueError, match="RADIANS"):
        mod.cos_lat_weights(np.linspace(-np.pi / 2, np.pi / 2, 8))


def test_poles_only_axis_raises():
    """cos(90 deg) is 6.1e-17, not 0, so this needs a tolerance rather than a
    positivity test -- the first version of this guard let it through."""
    mod = _load()
    with pytest.raises(ValueError, match="every row sits at a pole"):
        mod.cos_lat_weights(np.array([90.0, -90.0]))


def test_a_real_latitude_axis_is_accepted():
    """Non-vacuity partner: the guards must not reject the actual file."""
    mod = _load()
    w = mod.cos_lat_weights(np.linspace(-89.0, 89.0, 96))
    assert np.isclose(w.sum(), 1.0)
