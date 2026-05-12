"""FV3_3D iter 809: cin_column_fv3.

CIN = -Σ min(b_mid, 0) · Δz   (J/kg, ≥ 0 magnitude).

Companion to iter-808 ``cape_column_fv3``.

Tests
-----

1. ``test_cin_pure_negative``: uniform b<0 → CIN = |b|·Δz_total.
2. ``test_cin_pure_positive``: uniform b>0 → CIN = 0.
3. ``test_cin_mixed_sign``: only negative contributes.
4. ``test_cin_zero_b``: b=0 → CIN=0.
5. ``test_cin_realistic``: surface inversion → CIN 50-500 J/kg.
6. ``test_cin_shapes_batched``: 3-D → 2-D.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import cin_column_fv3


def test_cin_pure_negative():
    """Uniform b = -0.05, Δz_total = 10 km → CIN = 500 J/kg."""
    z = jnp.linspace(0.0, 10_000.0, 5)
    b = jnp.full((5,), -0.05)
    cin = cin_column_fv3(b, z)
    np.testing.assert_allclose(np.asarray(cin), [500.0], rtol=1e-12)


def test_cin_pure_positive():
    """Uniform b>0 → CIN = 0."""
    z = jnp.linspace(0.0, 10_000.0, 5)
    b = jnp.full((5,), 0.1)
    cin = cin_column_fv3(b, z)
    np.testing.assert_allclose(np.asarray(cin), [0.0], atol=1e-12)


def test_cin_mixed_sign():
    """Bottom 2 layers negative, top 2 positive → only negative counted."""
    z = jnp.array([0.0, 2500.0, 5000.0, 7500.0, 10_000.0])
    b = jnp.array([-0.05, -0.02, 0.05, 0.10, 0.05])
    cin = cin_column_fv3(b, z)
    # b_mid: -0.035, 0.015, 0.075, 0.075
    # negative_b_mid · dz: -0.035·2500, 0, 0, 0 = -87.5
    # CIN = -(-87.5) = 87.5
    np.testing.assert_allclose(np.asarray(cin), [87.5], rtol=1e-12)


def test_cin_zero_b():
    """b ≡ 0 → CIN = 0."""
    z = jnp.linspace(0.0, 10_000.0, 6)
    b = jnp.zeros((6,))
    cin = cin_column_fv3(b, z)
    np.testing.assert_allclose(np.asarray(cin), [0.0], atol=1e-15)


def test_cin_realistic():
    """Surface inversion profile → CIN 50-500 J/kg."""
    z = jnp.array([10.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0])
    # Bottom layers strongly negative (capping inversion), positive aloft
    b = jnp.array([-0.10, -0.05, -0.02, 0.05, 0.10, 0.02])
    cin = cin_column_fv3(b, z)
    assert 50.0 < float(cin) < 500.0


def test_cin_shapes_batched():
    """3-D batched: (n_x, n_y, km) → (n_x, n_y)."""
    rng = np.random.default_rng(seed=809)
    n_x, n_y, km = 4, 5, 30
    z = jnp.cumsum(
        jnp.asarray(rng.uniform(100.0, 600.0, size=(n_x, n_y, km))), axis=-1
    )
    b = jnp.asarray(rng.uniform(-0.05, 0.15, size=(n_x, n_y, km)))
    cin = cin_column_fv3(b, z)
    assert cin.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(cin))
    assert jnp.all(cin >= 0.0)
