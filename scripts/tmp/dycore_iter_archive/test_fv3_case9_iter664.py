"""FV3_3D iter 664: case9_B + case9_AofT ports.

Faithful JAX ports of FV3 Williamson test 9 forcing
(tools/test_cases.F90:4361-4424).

Tests
-----

1. ``test_case9_B_shape``.
2. ``test_case9_B_southern_zero``.
3. ``test_case9_B_peak_at_pi4``.
4. ``test_case9_B_zero_at_zero_lon``.
5. ``test_case9_AofT_ramp_up``.
6. ``test_case9_AofT_peak``.
7. ``test_case9_AofT_ramp_down``.
8. ``test_case9_AofT_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import case9_AofT, case9_B


def test_case9_B_shape():
    """B has shape of lon/lat input."""
    lon = jnp.linspace(0, 2 * jnp.pi, 10)
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 10)
    lon_2d, lat_2d = jnp.meshgrid(lon, lat, indexing="ij")
    B = case9_B(lon_2d, lat_2d)
    assert B.shape == lon_2d.shape


def test_case9_B_southern_zero():
    """B = 0 in southern hemisphere (and at equator)."""
    lons = jnp.linspace(0, 2 * jnp.pi, 8)
    lats = jnp.linspace(-jnp.pi / 2 + 0.1, 0.0, 5)
    lon_2d, lat_2d = jnp.meshgrid(lons, lats, indexing="ij")
    B = case9_B(lon_2d, lat_2d)
    assert jnp.allclose(B, 0.0, atol=1e-30)


def test_case9_B_peak_at_pi4():
    """At lat=π/4, cot²(lat)=1, exp(1-1)=1 → max amplitude = gh0·sin(lon)."""
    gh0 = 720.0 * constants.g
    lon = jnp.asarray(jnp.pi / 2)  # sin(π/2) = 1 → max
    lat = jnp.asarray(jnp.pi / 4)  # cot²(π/4) = 1
    B = case9_B(lon, lat, gh0=gh0)
    assert abs(float(B) - gh0) / gh0 < 1e-10


def test_case9_B_zero_at_zero_lon():
    """B = 0 at lon=0 (sin(0) factor)."""
    lats = jnp.linspace(0.1, jnp.pi / 2 - 0.1, 10)
    lon = jnp.zeros_like(lats)
    B = case9_B(lon, lats)
    assert jnp.allclose(B, 0.0, atol=1e-30)


def test_case9_AofT_ramp_up():
    """tday=0 → A=0; tday=4 → A=1 (ramp up)."""
    assert abs(float(case9_AofT(jnp.asarray(0.0)))) < 1e-14
    assert abs(float(case9_AofT(jnp.asarray(4.0))) - 1.0) < 1e-12


def test_case9_AofT_peak():
    """4 < tday ≤ 16 → A = 1 (peak)."""
    for t in (5.0, 10.0, 15.0, 16.0):
        assert abs(float(case9_AofT(jnp.asarray(t))) - 1.0) < 1e-12


def test_case9_AofT_ramp_down():
    """tday=16 → A=1; tday=20 → A=0 (ramp down)."""
    assert abs(float(case9_AofT(jnp.asarray(16.0))) - 1.0) < 1e-12
    assert abs(float(case9_AofT(jnp.asarray(20.0)))) < 1e-12


def test_case9_AofT_finite():
    """AofT(tday) finite for any positive time."""
    times = jnp.linspace(0, 30, 100)
    A = case9_AofT(times)
    assert jnp.all(jnp.isfinite(A))
    assert jnp.all(A >= -1e-12)
    assert jnp.all(A <= 1.0 + 1e-12)
