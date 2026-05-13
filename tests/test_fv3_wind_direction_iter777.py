"""FV3_3D iter 777: wind_direction_fv3 (meteorological convention).

Returns wind direction in degrees on [0, 360).  Convention 'from'
(default) is the meteorological standard (METAR, sondes, wind-rose);
'to' is the mathematical/oceanographic convention.

Tests
-----

1. ``test_wdir_pure_east_wind``: u>0, v=0 → from=270° (from west).
2. ``test_wdir_pure_north_wind``: u=0, v>0 → from=180° (from south).
3. ``test_wdir_pure_west_wind``: u<0, v=0 → from=90° (from east).
4. ``test_wdir_calm``: u=v=0 → 0°.
5. ``test_wdir_from_to_complement``: from = (to + 180) mod 360.
6. ``test_wdir_shapes_3d_finite``: 3-D shapes, finite, range [0,360).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import pytest

from legoesm.grids.cubed_sphere import wind_direction_fv3


def test_wdir_pure_east_wind():
    """Wind blowing eastward (ua=10, va=0) comes from west: 270°."""
    ua = jnp.array([10.0])
    va = jnp.array([0.0])
    wdir = wind_direction_fv3(ua, va, convention="from")
    np.testing.assert_allclose(np.asarray(wdir), [270.0], atol=1e-12)


def test_wdir_pure_north_wind():
    """Wind blowing northward (ua=0, va=10) comes from south: 180°."""
    ua = jnp.array([0.0])
    va = jnp.array([10.0])
    wdir = wind_direction_fv3(ua, va, convention="from")
    np.testing.assert_allclose(np.asarray(wdir), [180.0], atol=1e-12)


def test_wdir_pure_west_wind():
    """Wind blowing westward (ua=-10, va=0) comes from east: 90°."""
    ua = jnp.array([-10.0])
    va = jnp.array([0.0])
    wdir = wind_direction_fv3(ua, va, convention="from")
    np.testing.assert_allclose(np.asarray(wdir), [90.0], atol=1e-12)


def test_wdir_calm():
    """Calm air (ua=va=0) returns 0° by convention."""
    ua = jnp.array([0.0])
    va = jnp.array([0.0])
    wdir_from = wind_direction_fv3(ua, va, convention="from")
    wdir_to = wind_direction_fv3(ua, va, convention="to")
    # atan2(0,0) is 0; 'to' = 0°, 'from' = 180°
    np.testing.assert_allclose(np.asarray(wdir_to), [0.0], atol=1e-12)
    np.testing.assert_allclose(np.asarray(wdir_from), [180.0], atol=1e-12)


def test_wdir_from_to_complement():
    """For any wind, 'from' = ('to' + 180) mod 360."""
    rng = np.random.default_rng(seed=777)
    ua = jnp.asarray(rng.uniform(-30.0, 30.0, size=(50,)))
    va = jnp.asarray(rng.uniform(-30.0, 30.0, size=(50,)))
    wdir_to = wind_direction_fv3(ua, va, convention="to")
    wdir_from = wind_direction_fv3(ua, va, convention="from")
    expected_from = jnp.mod(wdir_to + 180.0, 360.0)
    np.testing.assert_allclose(np.asarray(wdir_from), np.asarray(expected_from), atol=1e-10)


def test_wdir_shapes_3d_finite():
    """3-D shapes, finite, range [0, 360)."""
    rng = np.random.default_rng(seed=778)
    n_x, n_y, km = 4, 5, 20
    ua = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.uniform(-50.0, 50.0, size=(n_x, n_y, km)))
    wdir = wind_direction_fv3(ua, va)
    assert wdir.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(wdir))
    assert jnp.all(wdir >= 0.0)
    assert jnp.all(wdir < 360.0)


def test_wdir_invalid_convention():
    """Invalid convention string raises ValueError."""
    ua = jnp.array([1.0])
    va = jnp.array([1.0])
    with pytest.raises(ValueError):
        wind_direction_fv3(ua, va, convention="bogus")
