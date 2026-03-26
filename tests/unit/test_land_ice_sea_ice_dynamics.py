"""Category 9: Sea Ice Dynamics -- EVP & Rheology (unit-level tests).

Tests the ice strength and rheology functions that do not require
a cubed-sphere grid. Grid-dependent tests (strain rates, EVP solver)
are covered in integration tests.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ice.rheology import ice_strength


SHAPE = (4,)


class Test9a_IceStrength:
    def test_positive_when_ice_exists(self):
        P = ice_strength(jnp.full(SHAPE, 1.0), jnp.full(SHAPE, 0.9))
        assert jnp.all(P > 0)

    def test_zero_no_ice(self):
        P = ice_strength(jnp.zeros(SHAPE), jnp.full(SHAPE, 0.9))
        assert jnp.allclose(P, 0.0)

    def test_zero_no_concentration(self):
        P = ice_strength(jnp.full(SHAPE, 1.0), jnp.zeros(SHAPE))
        # With A=0: exp(-C * 1.0) is very small
        assert jnp.all(P < 1.0)  # essentially zero

    def test_increases_with_h(self):
        P_thin = ice_strength(jnp.full(SHAPE, 0.5), jnp.full(SHAPE, 0.9))
        P_thick = ice_strength(jnp.full(SHAPE, 2.0), jnp.full(SHAPE, 0.9))
        assert jnp.all(P_thick > P_thin)

    def test_increases_with_A(self):
        P_low = ice_strength(jnp.full(SHAPE, 1.0), jnp.full(SHAPE, 0.5))
        P_high = ice_strength(jnp.full(SHAPE, 1.0), jnp.full(SHAPE, 0.95))
        assert jnp.all(P_high > P_low)

    def test_formula(self):
        """Check the formula: P = P* * h * exp(-C * (1 - A))."""
        h = jnp.array([2.0])
        A = jnp.array([0.8])
        P = ice_strength(h, A)
        P_expected = 2.75e4 * 2.0 * jnp.exp(-20.0 * 0.2)
        assert jnp.allclose(P, P_expected, rtol=1e-10)
