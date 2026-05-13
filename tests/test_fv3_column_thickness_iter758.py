"""FV3_3D iter 758: column_geopotential_thickness_fv3 port.

thickness = sum_k (-delz[k]).

Tests
-----

1. ``test_thickness_uniform_delz``.
2. ``test_thickness_known_value``.
3. ``test_thickness_zero_delz``.
4. ``test_thickness_positive``.
5. ``test_thickness_shapes_3d``.
6. ``test_thickness_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import column_geopotential_thickness_fv3


def test_thickness_uniform_delz():
    """km=10, delz=-500 → thickness = 5000 m."""
    km = 10
    delz = jnp.full((km,), -500.0)
    thickness = column_geopotential_thickness_fv3(delz)
    assert abs(float(thickness) - 5000.0) < 1e-10


def test_thickness_known_value():
    """Sum of −delz exactly."""
    delz = jnp.array([-100.0, -200.0, -300.0, -400.0])
    thickness = column_geopotential_thickness_fv3(delz)
    assert abs(float(thickness) - 1000.0) < 1e-10


def test_thickness_zero_delz():
    """delz=0 → thickness=0."""
    delz = jnp.zeros((5,))
    thickness = column_geopotential_thickness_fv3(delz)
    assert abs(float(thickness)) < 1e-15


def test_thickness_positive():
    """FV3 delz<0 → thickness>0."""
    rng = np.random.default_rng(seed=758)
    delz = jnp.asarray(rng.uniform(-500.0, -50.0, size=(20,)))
    thickness = column_geopotential_thickness_fv3(delz)
    assert float(thickness) > 0.0


def test_thickness_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=759)
    n_x, n_y, km = 4, 5, 20
    delz = jnp.asarray(rng.uniform(-500.0, -100.0, size=(n_x, n_y, km)))
    thickness = column_geopotential_thickness_fv3(delz)
    assert thickness.shape == (n_x, n_y)


def test_thickness_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=760)
    delz = jnp.asarray(rng.uniform(-1000.0, -50.0, size=(4, 4, 30)))
    thickness = column_geopotential_thickness_fv3(delz)
    assert jnp.all(jnp.isfinite(thickness))
