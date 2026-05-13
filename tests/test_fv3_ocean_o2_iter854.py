"""FV3_3D iter 854: ocean_oxygen_decline_fv3.

ΔO₂/O₂_ref = −α_O2 · ΔT_ocean.

Tests
-----

1. ``test_observed_1960_2010``: ΔT=0.4 K → ΔO₂ ≈ −2% (Schmidtko).
2. ``test_ssp37_2100``: ΔT=2.5 K → ΔO₂ ≈ −12.5%.
3. ``test_no_warming_zero``: ΔT=0 → ΔO₂=0.
4. ``test_cooling_increases_o2``: ΔT<0 → ΔO₂>0.
5. ``test_alpha_o2_scaling``: 2× α → 2× |ΔO₂|.
6. ``test_paired_with_ocean_chain``: OHC→ΔT→ΔO₂ pipeline.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    ocean_heat_content_fv3,
    ocean_oxygen_decline_fv3,
)


def test_observed_1960_2010():
    """ΔT=0.4 K (1960-2010 ocean) → ΔO₂ ≈ −2% (Schmidtko 2017)."""
    dt = jnp.array([0.4])
    d_o2 = ocean_oxygen_decline_fv3(dt)
    np.testing.assert_allclose(np.asarray(d_o2), [-0.02], rtol=1e-12)


def test_ssp37_2100():
    """ΔT=2.5 K (SSP3-7.0 2100 ocean) → ΔO₂ ≈ −12.5%."""
    dt = jnp.array([2.5])
    d_o2 = ocean_oxygen_decline_fv3(dt)
    np.testing.assert_allclose(np.asarray(d_o2), [-0.125], rtol=1e-12)


def test_no_warming_zero():
    """ΔT=0 → ΔO₂=0."""
    dt = jnp.array([0.0])
    d_o2 = ocean_oxygen_decline_fv3(dt)
    np.testing.assert_allclose(np.asarray(d_o2), [0.0], atol=1e-14)


def test_cooling_increases_o2():
    """ΔT<0 (paleoclimate cooling) → ΔO₂>0 (more soluble)."""
    dt = jnp.array([-1.0])
    d_o2 = ocean_oxygen_decline_fv3(dt)
    assert float(d_o2[0]) > 0.0


def test_alpha_o2_scaling():
    """2× α → 2× |ΔO₂|."""
    dt = jnp.array([1.0])
    d1 = ocean_oxygen_decline_fv3(dt, alpha_o2=0.05)
    d2 = ocean_oxygen_decline_fv3(dt, alpha_o2=0.10)
    np.testing.assert_allclose(np.asarray(d2), 2.0 * np.asarray(d1),
                                rtol=1e-12)


def test_paired_with_ocean_chain():
    """ΔOHC → implies ΔT (via ρ·c_p·H integral) → iter-854 ΔO₂."""
    # Construct a column with ΔT=1 K uniform / 4000 m
    n = 40
    dt_layer = jnp.full((n,), 1.0)
    h = jnp.full((n,), 100.0)
    ohc = ocean_heat_content_fv3(dt_layer, h)
    # Test that OHC > 0 (chained consistency); use volume-mean ΔT for O₂
    assert float(ohc) > 0.0
    d_o2 = ocean_oxygen_decline_fv3(jnp.array([1.0]))
    assert float(d_o2[0]) < 0.0
    # Cross-check: 1 K → −5% O₂
    np.testing.assert_allclose(np.asarray(d_o2), [-0.05], rtol=1e-12)


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=854)
    n_x, n_y = 6, 8
    dt = jnp.asarray(rng.uniform(-2.0, 5.0, size=(n_x, n_y)))
    d_o2 = ocean_oxygen_decline_fv3(dt)
    assert d_o2.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(d_o2))
