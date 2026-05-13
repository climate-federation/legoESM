"""FV3_3D iter 681: get_height_field_fv3 port.

Faithful JAX port of FV3 ``get_height_field``
(tools/fv_diagnostics.F90:3911-3945).

Tests
-----

1. ``test_height_shape``.
2. ``test_height_surface``.
3. ``test_height_monotonic_increase``.
4. ``test_height_hydrostatic_isothermal``.
5. ``test_height_nonhydrostatic_branch``.
6. ``test_height_nonhydrostatic_raises_no_delz``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import get_height_field_fv3


def test_height_shape():
    """Output shape (..., km+1)."""
    n_x, n_y, km = 4, 4, 8
    pt = jnp.full((n_x, n_y, km), 250.0)
    q = jnp.zeros((n_x, n_y, km))
    peln = jnp.broadcast_to(
        jnp.linspace(jnp.log(100.0), jnp.log(1.0e5), km + 1),
        (n_x, n_y, km + 1),
    )
    zsurf = jnp.zeros((n_x, n_y))
    wz = get_height_field_fv3(pt, q, peln, zsurf)
    assert wz.shape == (n_x, n_y, km + 1)


def test_height_surface():
    """wz[..., km] = zsurf."""
    n_x, n_y, km = 4, 4, 8
    pt = jnp.full((n_x, n_y, km), 250.0)
    q = jnp.zeros((n_x, n_y, km))
    peln = jnp.broadcast_to(
        jnp.linspace(jnp.log(100.0), jnp.log(1.0e5), km + 1),
        (n_x, n_y, km + 1),
    )
    zsurf = jnp.full((n_x, n_y), 500.0)
    wz = get_height_field_fv3(pt, q, peln, zsurf)
    assert jnp.allclose(wz[..., km], zsurf, atol=1e-12)


def test_height_monotonic_increase():
    """wz decreases from top to bottom (top has higher altitude)."""
    n_x, n_y, km = 4, 4, 8
    pt = jnp.full((n_x, n_y, km), 250.0)
    q = jnp.zeros((n_x, n_y, km))
    peln = jnp.broadcast_to(
        jnp.linspace(jnp.log(100.0), jnp.log(1.0e5), km + 1),
        (n_x, n_y, km + 1),
    )
    zsurf = jnp.zeros((n_x, n_y))
    wz = get_height_field_fv3(pt, q, peln, zsurf)
    # wz[k] > wz[k+1] (k=0 is top, km is bottom)
    diffs = wz[..., 1:] - wz[..., :-1]
    assert jnp.all(diffs < 0)


def test_height_hydrostatic_isothermal():
    """Isothermal: wz[0] - wz[km] = (R_d/g) · T · ln(p_s/p_top)."""
    km = 8
    T = 250.0
    pt = jnp.full((km,), T)
    q = jnp.zeros((km,))
    p_top, p_s = 100.0, 1.0e5
    peln = jnp.linspace(jnp.log(p_top), jnp.log(p_s), km + 1)
    zsurf = jnp.asarray(0.0)
    wz = get_height_field_fv3(pt, q, peln, zsurf)
    expected = (constants.R_d / constants.g) * T * jnp.log(p_s / p_top)
    actual = float(wz[0] - wz[km])
    assert abs(actual - float(expected)) / float(expected) < 1e-10


def test_height_nonhydrostatic_branch():
    """Non-hydrostatic: wz from delz."""
    km = 8
    pt = jnp.zeros((km,))
    q = jnp.zeros((km,))
    peln = jnp.zeros((km + 1,))
    zsurf = jnp.asarray(100.0)
    # delz = -500 m for each layer (negative by FV3 convention)
    delz = jnp.full((km,), -500.0)
    wz = get_height_field_fv3(
        pt, q, peln, zsurf, delz=delz, hydrostatic=False,
    )
    # wz[km] = 100; wz[km-1] = 100 + 500; ...
    assert abs(float(wz[km]) - 100.0) < 1e-12
    assert abs(float(wz[0]) - (100.0 + km * 500.0)) < 1e-10


def test_height_nonhydrostatic_raises_no_delz():
    """hydrostatic=False without delz raises ValueError."""
    pt = jnp.zeros((4,))
    q = jnp.zeros((4,))
    peln = jnp.zeros((5,))
    zsurf = jnp.asarray(0.0)
    with pytest.raises(ValueError):
        get_height_field_fv3(pt, q, peln, zsurf, hydrostatic=False)
