"""FV3_3D iter 678: interpolate_z_fv3 port.

Faithful JAX port of FV3 ``interpolate_z`` (tools/fv_diagnostics.F90:
4776-4810).  Linear interpolation of 3D field to single z-level.

Tests
-----

1. ``test_interpz_shape``.
2. ``test_interpz_above_top_clamp``.
3. ``test_interpz_below_bot_clamp``.
4. ``test_interpz_exact_at_midpoint``.
5. ``test_interpz_linear_interp``.
6. ``test_interpz_constant_field``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import interpolate_z_fv3


def _setup(n_x=4, n_y=4, km=8):
    """km layers, top-down decreasing heights."""
    # hght[k] from 20000 (top) down to 0 (bottom)
    heights_1d = jnp.linspace(20000.0, 0.0, km + 1)
    hght = jnp.broadcast_to(heights_1d, (n_x, n_y, km + 1))
    return hght


def test_interpz_shape():
    """Output shape (n_x, n_y)."""
    n_x, n_y, km = 4, 4, 8
    hght = _setup(n_x, n_y, km)
    a3 = jnp.zeros((n_x, n_y, km))
    a2 = interpolate_z_fv3(a3, hght, jnp.asarray(5000.0))
    assert a2.shape == (n_x, n_y)


def test_interpz_above_top_clamp():
    """zl above top → a2 = a3[..., 0]."""
    n_x, n_y, km = 4, 4, 8
    hght = _setup(n_x, n_y, km)   # heights 20000..0
    a3 = jnp.broadcast_to(jnp.arange(km, dtype=jnp.float64), (n_x, n_y, km))
    a2 = interpolate_z_fv3(a3, hght, jnp.asarray(25000.0))
    assert jnp.allclose(a2, a3[..., 0])


def test_interpz_below_bot_clamp():
    """zl below bottom → a2 = a3[..., -1]."""
    n_x, n_y, km = 4, 4, 8
    hght = _setup(n_x, n_y, km)
    a3 = jnp.broadcast_to(jnp.arange(km, dtype=jnp.float64), (n_x, n_y, km))
    a2 = interpolate_z_fv3(a3, hght, jnp.asarray(-100.0))
    assert jnp.allclose(a2, a3[..., -1])


def test_interpz_exact_at_midpoint():
    """At zm[k] exactly, a2 = a3[..., k]."""
    n_x, n_y, km = 4, 4, 8
    hght = _setup(n_x, n_y, km)
    a3 = jnp.broadcast_to(jnp.arange(km, dtype=jnp.float64), (n_x, n_y, km))
    # zm[2] = 0.5·(hght[2] + hght[3]) = 0.5·(15000 + 12500) ≈ 13750
    zm_2 = 0.5 * (hght[0, 0, 2] + hght[0, 0, 3])
    a2 = interpolate_z_fv3(a3, hght, zm_2)
    assert jnp.allclose(a2, a3[..., 2])


def test_interpz_linear_interp():
    """Linear field a3 = lin(zm) → exact interp at any zl."""
    n_x, n_y, km = 4, 4, 10
    hght = _setup(n_x, n_y, km)
    zm = 0.5 * (hght[0, 0, :-1] + hght[0, 0, 1:])
    # a3 = 2·zm + 3
    a3 = jnp.broadcast_to((2.0 * zm + 3.0), (n_x, n_y, km))
    target_z = 5000.0
    expected = 2.0 * target_z + 3.0
    a2 = interpolate_z_fv3(a3, hght, jnp.asarray(target_z))
    assert jnp.allclose(a2, expected, atol=1e-10)


def test_interpz_constant_field():
    """Constant a3 → constant a2 = a3."""
    n_x, n_y, km = 4, 4, 8
    hght = _setup(n_x, n_y, km)
    a3 = jnp.full((n_x, n_y, km), 7.5)
    a2 = interpolate_z_fv3(a3, hght, jnp.asarray(5000.0))
    assert jnp.allclose(a2, 7.5, atol=1e-12)
