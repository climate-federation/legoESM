"""FV3_3D iter 679: interpolate_vertical_fv3 port.

Faithful JAX port of FV3 ``interpolate_vertical`` (tools/
fv_diagnostics.F90:4735-4774).  Linear log-pressure-level
interpolation.

Tests
-----

1. ``test_interpvert_shape``.
2. ``test_interpvert_above_top_clamp``.
3. ``test_interpvert_below_bot_clamp``.
4. ``test_interpvert_exact_at_midpoint``.
5. ``test_interpvert_linear_in_logp``.
6. ``test_interpvert_constant_field``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import interpolate_vertical_fv3


def _setup(n_x=4, n_y=4, km=8):
    """km layers, log-p increasing with k (top-down)."""
    # log-p from log(100) (top) up to log(1e5) (bottom)
    peln_1d = jnp.linspace(jnp.log(100.0), jnp.log(1.0e5), km + 1)
    peln = jnp.broadcast_to(peln_1d, (n_x, n_y, km + 1))
    return peln


def test_interpvert_shape():
    """Output shape (n_x, n_y)."""
    n_x, n_y, km = 4, 4, 8
    peln = _setup(n_x, n_y, km)
    a3 = jnp.zeros((n_x, n_y, km))
    a2 = interpolate_vertical_fv3(a3, peln, jnp.asarray(5.0e4))
    assert a2.shape == (n_x, n_y)


def test_interpvert_above_top_clamp():
    """plev far below ptop (model top): a2 = a3[..., 0]."""
    n_x, n_y, km = 4, 4, 8
    peln = _setup(n_x, n_y, km)
    a3 = jnp.broadcast_to(jnp.arange(km, dtype=jnp.float64), (n_x, n_y, km))
    a2 = interpolate_vertical_fv3(a3, peln, jnp.asarray(50.0))   # above top
    assert jnp.allclose(a2, a3[..., 0])


def test_interpvert_below_bot_clamp():
    """plev above ps (below model bottom): a2 = a3[..., -1]."""
    n_x, n_y, km = 4, 4, 8
    peln = _setup(n_x, n_y, km)
    a3 = jnp.broadcast_to(jnp.arange(km, dtype=jnp.float64), (n_x, n_y, km))
    a2 = interpolate_vertical_fv3(a3, peln, jnp.asarray(1.0e6))
    assert jnp.allclose(a2, a3[..., -1])


def test_interpvert_exact_at_midpoint():
    """At pm[k] exactly, a2 = a3[k]."""
    n_x, n_y, km = 4, 4, 8
    peln = _setup(n_x, n_y, km)
    a3 = jnp.broadcast_to(jnp.arange(km, dtype=jnp.float64), (n_x, n_y, km))
    # pm[2] = 0.5·(peln[2] + peln[3]) → exp(pm[2]) is target
    pm_2 = 0.5 * (peln[0, 0, 2] + peln[0, 0, 3])
    plev = jnp.exp(pm_2)
    a2 = interpolate_vertical_fv3(a3, peln, plev)
    assert jnp.allclose(a2, a3[..., 2], atol=1e-10)


def test_interpvert_linear_in_logp():
    """Linear field in log-p: a3 = 2·pm + 3 reconstructs exactly."""
    n_x, n_y, km = 4, 4, 10
    peln = _setup(n_x, n_y, km)
    pm = 0.5 * (peln[0, 0, :-1] + peln[0, 0, 1:])
    a3 = jnp.broadcast_to((2.0 * pm + 3.0), (n_x, n_y, km))
    target_p = 5.0e4
    expected = 2.0 * jnp.log(target_p) + 3.0
    a2 = interpolate_vertical_fv3(a3, peln, jnp.asarray(target_p))
    assert jnp.allclose(a2, expected, atol=1e-10)


def test_interpvert_constant_field():
    """Constant a3 → constant a2."""
    n_x, n_y, km = 4, 4, 8
    peln = _setup(n_x, n_y, km)
    a3 = jnp.full((n_x, n_y, km), 7.5)
    a2 = interpolate_vertical_fv3(a3, peln, jnp.asarray(5.0e4))
    assert jnp.allclose(a2, 7.5, atol=1e-12)
