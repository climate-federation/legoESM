"""FV3_3D iter 676: wind_max_fv3 port.

Faithful JAX port of FV3 ``wind_max`` (tools/fv_diagnostics.F90:
3843-3874).  Max wind speed in 7×7 neighborhood.

Tests
-----

1. ``test_wind_max_shape``.
2. ``test_wind_max_uniform``.
3. ``test_wind_max_isolated_peak``.
4. ``test_wind_max_zero_field``.
5. ``test_wind_max_3d_input``.
6. ``test_wind_max_smaller_window``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import wind_max_fv3


def test_wind_max_shape():
    """Output shape matches input."""
    n_x, n_y = 10, 12
    us = jnp.zeros((n_x, n_y))
    vs = jnp.zeros((n_x, n_y))
    out = wind_max_fv3(us, vs)
    assert out.shape == (n_x, n_y)


def test_wind_max_uniform():
    """Uniform u=U, v=V → ws_max = sqrt(U²+V²) everywhere."""
    n_x, n_y = 10, 10
    U, V = 10.0, 5.0
    us = jnp.full((n_x, n_y), U)
    vs = jnp.full((n_x, n_y), V)
    expected = jnp.sqrt(U * U + V * V)
    out = wind_max_fv3(us, vs)
    assert jnp.allclose(out, expected, atol=1e-12)


def test_wind_max_isolated_peak():
    """Single non-zero cell propagates to all cells in 7×7 neighborhood."""
    n_x, n_y = 15, 15
    us = jnp.zeros((n_x, n_y))
    vs = jnp.zeros((n_x, n_y))
    # Place 100 m/s at (7, 7)
    us = us.at[7, 7].set(100.0)
    out = wind_max_fv3(us, vs)
    # Within 7×7 neighborhood (rows 4..10, cols 4..10), max = 100
    assert jnp.allclose(out[4:11, 4:11], 100.0, atol=1e-12)
    # Outside, max = 0
    assert jnp.allclose(out[0, :], 0.0)
    assert jnp.allclose(out[11:, :], 0.0)


def test_wind_max_zero_field():
    """Zero wind everywhere → output 0."""
    n_x, n_y = 10, 10
    us = jnp.zeros((n_x, n_y))
    vs = jnp.zeros((n_x, n_y))
    out = wind_max_fv3(us, vs)
    assert jnp.allclose(out, 0.0, atol=1e-14)


def test_wind_max_3d_input():
    """3D (n_face, n_x, n_y) input works (max over (i, j) only)."""
    n_face, n_x, n_y = 6, 12, 12
    rng = np.random.default_rng(seed=676)
    us = jnp.asarray(rng.uniform(0, 20, size=(n_face, n_x, n_y)))
    vs = jnp.asarray(rng.uniform(0, 20, size=(n_face, n_x, n_y)))
    out = wind_max_fv3(us, vs)
    assert out.shape == (n_face, n_x, n_y)
    assert jnp.all(jnp.isfinite(out))


def test_wind_max_smaller_window():
    """half_window=1 → 3×3 max-pool."""
    n_x, n_y = 5, 5
    us = jnp.zeros((n_x, n_y))
    vs = jnp.zeros((n_x, n_y))
    us = us.at[2, 2].set(50.0)
    out = wind_max_fv3(us, vs, half_window=1)
    # 3×3 neighborhood around (2, 2) is (1..3, 1..3)
    assert float(out[1, 1]) == 50.0
    assert float(out[3, 3]) == 50.0
    # Cell (0, 0) outside 3×3 → 0
    assert float(out[0, 0]) == 0.0
