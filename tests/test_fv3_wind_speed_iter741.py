"""FV3_3D iter 741: wind_speed_fv3 port.

|V| = sqrt(ua^2 + va^2 [+ w^2]).

Tests
-----

1. ``test_speed_zero_wind``.
2. ``test_speed_unit_vec``.
3. ``test_speed_pythagoras_3_4_5``.
4. ``test_speed_w_optional``.
5. ``test_speed_matches_sqrt_2ke``.
6. ``test_speed_shapes_3d``.
7. ``test_speed_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import kinetic_energy_fv3, wind_speed_fv3


def test_speed_zero_wind():
    """Zero wind → speed = 0."""
    n = 5
    z = jnp.zeros((n,))
    speed = wind_speed_fv3(z, z, z)
    assert jnp.all(speed == 0.0)


def test_speed_unit_vec():
    """ua=1, va=0, w=0 → speed=1."""
    n = 5
    one = jnp.ones((n,))
    zero = jnp.zeros((n,))
    speed = wind_speed_fv3(one, zero, zero)
    assert jnp.allclose(speed, 1.0, atol=1e-12)


def test_speed_pythagoras_3_4_5():
    """ua=3, va=4 → speed=5 (Pythagoras)."""
    ua = jnp.full((5,), 3.0)
    va = jnp.full((5,), 4.0)
    speed = wind_speed_fv3(ua, va)
    assert jnp.allclose(speed, 5.0, atol=1e-12)


def test_speed_w_optional():
    """w=None → horizontal speed only."""
    ua = jnp.full((5,), 3.0)
    va = jnp.full((5,), 4.0)
    w = jnp.full((5,), 12.0)
    speed_2d = wind_speed_fv3(ua, va)
    speed_3d = wind_speed_fv3(ua, va, w)
    assert jnp.allclose(speed_2d, 5.0, atol=1e-12)
    assert jnp.allclose(speed_3d, 13.0, atol=1e-12)


def test_speed_matches_sqrt_2ke():
    """|V| = sqrt(2 · KE) (consistency with iter-740)."""
    rng = np.random.default_rng(seed=741)
    ua = jnp.asarray(rng.normal(scale=10.0, size=(10,)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(10,)))
    w = jnp.asarray(rng.normal(scale=2.0, size=(10,)))
    speed = wind_speed_fv3(ua, va, w)
    ke = kinetic_energy_fv3(ua, va, w)
    assert jnp.allclose(speed, jnp.sqrt(2.0 * ke), atol=1e-12)


def test_speed_shapes_3d():
    """3-D shapes."""
    rng = np.random.default_rng(seed=742)
    n_x, n_y, km = 4, 5, 20
    ua = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    va = jnp.asarray(rng.normal(scale=10.0, size=(n_x, n_y, km)))
    w = jnp.asarray(rng.normal(scale=0.5, size=(n_x, n_y, km)))
    speed = wind_speed_fv3(ua, va, w)
    assert speed.shape == (n_x, n_y, km)


def test_speed_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=743)
    ua = jnp.asarray(rng.normal(scale=20.0, size=(4, 30)))
    va = jnp.asarray(rng.normal(scale=20.0, size=(4, 30)))
    speed = wind_speed_fv3(ua, va)
    assert jnp.all(jnp.isfinite(speed))
