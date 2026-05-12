"""FV3_3D iter 750: area_weighted_mean_fv3 helper port.

Tests
-----

1. ``test_awm_uniform``.
2. ``test_awm_weighted``.
3. ``test_awm_masked_subset``.
4. ``test_awm_empty_mask_returns_sentinel``.
5. ``test_awm_custom_sentinel``.
6. ``test_awm_shapes_3d``.
7. ``test_awm_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import area_weighted_mean_fv3


def test_awm_uniform():
    """Uniform field → mean = field value."""
    field = jnp.full((10,), 7.5)
    area = jnp.ones((10,))
    mean = area_weighted_mean_fv3(field, area)
    assert abs(float(mean) - 7.5) < 1e-12


def test_awm_weighted():
    """Cells with larger area contribute more."""
    field = jnp.array([10.0, 0.0])
    area = jnp.array([3.0, 1.0])
    mean = area_weighted_mean_fv3(field, area)
    expected = 30.0 / 4.0
    assert abs(float(mean) - expected) < 1e-12


def test_awm_masked_subset():
    """Mask selects subset; mean = subset average."""
    field = jnp.array([1.0, 2.0, 3.0, 4.0])
    area = jnp.ones((4,))
    mask = jnp.array([True, False, True, False])
    mean = area_weighted_mean_fv3(field, area, mask=mask)
    expected = 0.5 * (1.0 + 3.0)
    assert abs(float(mean) - expected) < 1e-12


def test_awm_empty_mask_returns_sentinel():
    """Empty mask → returns -1.0 sentinel."""
    field = jnp.array([1.0, 2.0])
    area = jnp.ones((2,))
    mask = jnp.array([False, False])
    mean = area_weighted_mean_fv3(field, area, mask=mask)
    assert float(mean) == -1.0


def test_awm_custom_sentinel():
    """empty_band_sentinel parameter overrides default."""
    field = jnp.array([1.0])
    area = jnp.array([0.5])   # area sum 0.5 < 1.0 → sentinel
    mean = area_weighted_mean_fv3(field, area, empty_band_sentinel=-999.0)
    assert float(mean) == -999.0


def test_awm_shapes_3d():
    """Multi-dim flatten via field/area broadcasting works."""
    rng = np.random.default_rng(seed=750)
    field = jnp.asarray(rng.uniform(0.0, 10.0, size=(10, 20)))
    area = jnp.asarray(rng.uniform(0.5, 1.5, size=(10, 20)))
    mean = area_weighted_mean_fv3(field, area)
    # Result is scalar (sums over all axes)
    assert mean.shape == ()


def test_awm_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=751)
    field = jnp.asarray(rng.normal(scale=100.0, size=(50,)))
    area = jnp.asarray(rng.uniform(0.5, 2.0, size=(50,)))
    mean = area_weighted_mean_fv3(field, area)
    assert jnp.isfinite(mean)
