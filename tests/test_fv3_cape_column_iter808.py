"""FV3_3D iter 808: cape_column_fv3.

CAPE = Σ max(b_mid, 0) · Δz   (J/kg).

Composes iter-807 ``parcel_buoyancy_fv3``.

Tests
-----

1. ``test_cape_pure_positive``: uniform b > 0 → CAPE = b·Δz_total.
2. ``test_cape_pure_negative``: uniform b < 0 → CAPE = 0.
3. ``test_cape_mixed_sign``: only positive contributes.
4. ``test_cape_zero_b``: b = 0 → CAPE = 0.
5. ``test_cape_realistic``: tropical-like profile → 500-6000 J/kg.
6. ``test_cape_shapes_batched``: 3-D (n_x, n_y, km) → (n_x, n_y).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import cape_column_fv3


def test_cape_pure_positive():
    """Uniform b = 0.1 m/s², 5 levels evenly spaced over 10 km
    → CAPE ≈ 0.1 · 10000 = 1000 J/kg."""
    z = jnp.linspace(0.0, 10_000.0, 5)
    b = jnp.full((5,), 0.1)
    cape = cape_column_fv3(b, z)
    np.testing.assert_allclose(np.asarray(cape), [1000.0], rtol=1e-12)


def test_cape_pure_negative():
    """Uniform b<0 → CAPE = 0."""
    z = jnp.linspace(0.0, 10_000.0, 5)
    b = jnp.full((5,), -0.05)
    cape = cape_column_fv3(b, z)
    np.testing.assert_allclose(np.asarray(cape), [0.0], atol=1e-12)


def test_cape_mixed_sign():
    """Bottom-half negative, top-half positive → only positive counted."""
    z = jnp.array([0.0, 2500.0, 5000.0, 7500.0, 10_000.0])
    b = jnp.array([-0.05, -0.02, 0.05, 0.10, 0.05])
    cape = cape_column_fv3(b, z)
    # b_mid: -0.035, 0.015, 0.075, 0.075
    # Layers: 2500, 2500, 2500, 2500
    # positive_b_mid · dz: 0, 0.015·2500, 0.075·2500, 0.075·2500
    # = 0 + 37.5 + 187.5 + 187.5 = 412.5
    np.testing.assert_allclose(np.asarray(cape), [412.5], rtol=1e-12)


def test_cape_zero_b():
    """b ≡ 0 → CAPE = 0."""
    z = jnp.linspace(0.0, 10_000.0, 6)
    b = jnp.zeros((6,))
    cape = cape_column_fv3(b, z)
    np.testing.assert_allclose(np.asarray(cape), [0.0], atol=1e-15)


def test_cape_realistic():
    """Tropical-like profile: small CIN below LFC, positive b in mid-trop."""
    z = jnp.array([100.0, 1000.0, 2000.0, 5000.0, 9000.0, 12_000.0])
    # Surface CIN, positive aloft, decay toward EL
    b = jnp.array([-0.02, -0.01, 0.02, 0.08, 0.06, 0.01])
    cape = cape_column_fv3(b, z)
    # Tropical CAPE typically 500-5000 J/kg for this kind of profile
    assert 100.0 < float(cape) < 6000.0


def test_cape_shapes_batched():
    """3-D batched: (n_x, n_y, km) → (n_x, n_y)."""
    rng = np.random.default_rng(seed=808)
    n_x, n_y, km = 4, 5, 30
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(100.0, 600.0, size=(n_x, n_y, km))), axis=-1
    )
    b = jnp.asarray(rng.uniform(-0.05, 0.15, size=(n_x, n_y, km)))
    cape = cape_column_fv3(b, z)
    assert cape.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(cape))
    assert jnp.all(cape >= 0.0)
