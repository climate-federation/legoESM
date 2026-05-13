"""FV3_3D iter 884: aridity_index_fv3.

AI = P_annual / PET_annual  [dimensionless].

Tests
-----

1. ``test_hyper_arid_sahara``: P=20, PET=2500 → AI=0.008 (hyper-arid).
2. ``test_semi_arid_sahel``: P=500, PET=1500 → AI=0.33.
3. ``test_humid_temperate``: P=800, PET=1000 → AI=0.80 (humid).
4. ``test_zero_precip_zero``: P=0 → AI=0.
5. ``test_pet_zero_floored``: PET=0 → finite (no NaN).
6. ``test_unep_thresholds``: 4 sample AI values map to correct band.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import aridity_index_fv3


def test_hyper_arid_sahara():
    """Sahara P=20 mm, PET=2500 mm → AI=0.008 (hyper-arid, <0.05)."""
    ai = aridity_index_fv3(
        precip_annual=jnp.array([20.0]),
        pet_annual=jnp.array([2500.0]),
    )
    assert float(ai[0]) < 0.05


def test_semi_arid_sahel():
    """Sahel P=500 mm, PET=1500 mm → AI=0.33 (semi-arid)."""
    ai = aridity_index_fv3(
        precip_annual=jnp.array([500.0]),
        pet_annual=jnp.array([1500.0]),
    )
    np.testing.assert_allclose(np.asarray(ai), [500.0 / 1500.0], rtol=1e-12)
    assert 0.20 <= float(ai[0]) < 0.50


def test_humid_temperate():
    """Humid P=800, PET=1000 → AI=0.80 (humid, ≥0.65)."""
    ai = aridity_index_fv3(
        precip_annual=jnp.array([800.0]),
        pet_annual=jnp.array([1000.0]),
    )
    assert float(ai[0]) >= 0.65


def test_zero_precip_zero():
    """P=0 → AI=0."""
    ai = aridity_index_fv3(
        precip_annual=jnp.array([0.0]),
        pet_annual=jnp.array([1500.0]),
    )
    np.testing.assert_allclose(np.asarray(ai), [0.0], atol=1e-14)


def test_pet_zero_floored():
    """PET=0 → finite via floor (polar regime)."""
    ai = aridity_index_fv3(
        precip_annual=jnp.array([100.0]),
        pet_annual=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(ai))


def test_unep_thresholds():
    """4 sample AI values → correct UNEP category."""
    p = jnp.array([30.0, 200.0, 600.0, 1500.0])
    pet = jnp.array([2000.0, 1500.0, 1200.0, 1000.0])
    ai = aridity_index_fv3(p, pet)
    # 0.015, 0.133, 0.5, 1.5 → hyper-arid, arid, dry-sub-humid, humid
    assert float(ai[0]) < 0.05
    assert 0.05 <= float(ai[1]) < 0.20
    assert 0.50 <= float(ai[2]) < 0.65
    assert float(ai[3]) >= 0.65


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=884)
    n_x, n_y = 6, 8
    p = jnp.asarray(rng.uniform(0.0, 2500.0, size=(n_x, n_y)))
    pet = jnp.asarray(rng.uniform(50.0, 3000.0, size=(n_x, n_y)))
    ai = aridity_index_fv3(p, pet)
    assert ai.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(ai))
    assert jnp.all(ai >= 0.0)
