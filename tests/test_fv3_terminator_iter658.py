"""FV3_3D iter 658: terminator_tracers port.

Faithful JAX port of FV3 ``terminator_tracers``
(tools/test_cases.F90:4136-4205).  DCMIP 2016 terminator chemistry
IC: paired Cl / Cl2 tracers under photolysis at localized "sun".

Tests
-----

1. ``test_terminator_shape``.
2. ``test_terminator_total_chlorine_conserved``.
3. ``test_terminator_nonnegative``.
4. ``test_terminator_uniform_in_k``.
5. ``test_terminator_sun_max_cl``.
6. ``test_terminator_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import terminator_tracers


def _make_lat_lon(n=20):
    lons = jnp.linspace(0.0, 2 * jnp.pi, n)
    lats = jnp.linspace(-jnp.pi / 2 + 0.1, jnp.pi / 2 - 0.1, n)
    lon_2d, lat_2d = jnp.meshgrid(lons, lats, indexing="ij")
    return lon_2d, lat_2d


def test_terminator_shape():
    """Output shapes (..., n_x, n_y, km)."""
    lon, lat = _make_lat_lon(20)
    Cl, Cl2 = terminator_tracers(lon, lat, km=5)
    assert Cl.shape == (20, 20, 5)
    assert Cl2.shape == (20, 20, 5)


def test_terminator_total_chlorine_conserved():
    """Cl + 2·Cl2 = qcly everywhere (chlorine atom balance)."""
    lon, lat = _make_lat_lon(20)
    qcly = 4.0e-6
    Cl, Cl2 = terminator_tracers(lon, lat, km=3, qcly=qcly)
    total = Cl + 2.0 * Cl2
    assert jnp.allclose(total, qcly, atol=1e-15), (
        f"total chlorine: max diff = {float(jnp.max(jnp.abs(total - qcly)))}"
    )


def test_terminator_nonnegative():
    """Cl, Cl2 ≥ 0 everywhere."""
    lon, lat = _make_lat_lon(30)
    Cl, Cl2 = terminator_tracers(lon, lat, km=2)
    assert jnp.all(Cl >= -1e-30)
    assert jnp.all(Cl2 >= -1e-30)


def test_terminator_uniform_in_k():
    """Same pattern at every k."""
    lon, lat = _make_lat_lon(15)
    Cl, Cl2 = terminator_tracers(lon, lat, km=4)
    for k in range(4):
        assert jnp.allclose(Cl[..., k], Cl[..., 0])
        assert jnp.allclose(Cl2[..., k], Cl2[..., 0])


def test_terminator_sun_max_cl():
    """At sun position (lc, thc): max k1 → max Cl, min Cl2."""
    lc = 5.0 * jnp.pi / 3.0
    thc = jnp.pi / 9.0
    # Build coarse grid; sun position at one cell
    lons = jnp.asarray([0.0, jnp.pi / 2, lc, 3 * jnp.pi / 4])
    lats = jnp.asarray([-jnp.pi / 4, 0.0, thc, jnp.pi / 4])
    lon_2d, lat_2d = jnp.meshgrid(lons, lats, indexing="ij")
    Cl, Cl2 = terminator_tracers(lon_2d, lat_2d, km=1)
    Cl_flat = Cl[..., 0]
    # Sun position has the largest k1 → smallest Cl2, largest Cl
    sun_i, sun_j = 2, 2  # (lc, thc)
    assert float(Cl_flat[sun_i, sun_j]) == float(jnp.max(Cl_flat))


def test_terminator_finite():
    """No NaN/Inf."""
    lon, lat = _make_lat_lon(20)
    Cl, Cl2 = terminator_tracers(lon, lat, km=3)
    assert jnp.all(jnp.isfinite(Cl))
    assert jnp.all(jnp.isfinite(Cl2))
