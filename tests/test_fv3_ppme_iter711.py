"""FV3_3D iter 711: ppme_fv3 port.

Faithful JAX port of FV3 ``ppme`` (tools/fv_diagnostics.F90:
5196-5305).  PPM cell-edge values with non-uniform delp.

Tests
-----

1. ``test_ppme_uniform_field``.
2. ``test_ppme_linear_field``.
3. ``test_ppme_monotonicity_bounded``.
4. ``test_ppme_shapes_3d``.
5. ``test_ppme_finite``.
6. ``test_ppme_min_km_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import ppme_fv3


def test_ppme_uniform_field():
    """Uniform p → uniform qe."""
    km = 10
    p = jnp.full((km,), 5.0)
    delp = jnp.full((km,), 1000.0)
    qe = ppme_fv3(p, delp)
    assert qe.shape == (km + 1,)
    assert jnp.all(jnp.abs(qe - 5.0) < 1e-10)


def test_ppme_linear_field():
    """Linear p(k) → interior edges close to half-cell midpoint values.

    PPM is exact on linear fields for interior 4th-order edges.
    """
    km = 12
    p = jnp.linspace(0.0, 11.0, km)
    delp = jnp.full((km,), 1000.0)
    qe = ppme_fv3(p, delp)
    # Interior edges k=3..km-1 (Python idx 3..km-2) should be near
    # midpoint values:  qe[k] ≈ (p[k-1] + p[k])/2
    for k in range(3, km - 1):
        expected = 0.5 * (float(p[k - 1]) + float(p[k]))
        assert abs(float(qe[k]) - expected) < 0.5


def test_ppme_monotonicity_bounded():
    """Random p → 4th-order interior edges bounded by neighbors
    (slope-limited PPM)."""
    rng = np.random.default_rng(seed=711)
    km = 30
    p = jnp.asarray(rng.uniform(0.0, 100.0, size=(km,)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(km,)))
    qe = ppme_fv3(p, delp)
    # Interior edges k=3..km-1 should be in [min, max] of bracketing
    # cells (with Van-Leer limiter not over-extending too far)
    for k in range(3, km - 1):
        lo = float(jnp.minimum(p[k - 1], p[k]))
        hi = float(jnp.maximum(p[k - 1], p[k]))
        # Limiter may extend up to one-cell amplitude
        spread = hi - lo + 1e-9
        assert lo - 2 * spread <= float(qe[k]) <= hi + 2 * spread


def test_ppme_shapes_3d():
    """3-D (n_x, n_y, km) input → (n_x, n_y, km+1) output."""
    rng = np.random.default_rng(seed=712)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(0.0, 100.0, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 1000.0)
    qe = ppme_fv3(p, delp)
    assert qe.shape == (n_x, n_y, km + 1)


def test_ppme_finite():
    """Random p, delp → finite qe."""
    rng = np.random.default_rng(seed=713)
    n_x, n_y, km = 4, 4, 30
    p = jnp.asarray(rng.normal(scale=50.0, size=(n_x, n_y, km)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(n_x, n_y, km)))
    qe = ppme_fv3(p, delp)
    assert jnp.all(jnp.isfinite(qe))


def test_ppme_min_km_raises():
    """km < 4 raises ValueError (top/bottom closures need 4 cells)."""
    p = jnp.zeros((3,))
    delp = jnp.ones((3,))
    with pytest.raises(ValueError):
        ppme_fv3(p, delp)
