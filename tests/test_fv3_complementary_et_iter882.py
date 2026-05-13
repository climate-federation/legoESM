"""FV3_3D iter 882: complementary_relationship_et_fv3.

λE_actual = 2·λE_wet − λE_pot   (Bouchet 1963).

Tests
-----

1. ``test_saturated_identity``: λE_wet = λE_pot → λE_actual = λE_wet.
2. ``test_drying_decreases_actual``: λE_pot > λE_wet → λE_actual < λE_wet.
3. ``test_fully_dry_zero_floored``: λE_pot = 2·λE_wet → λE_actual=0.
4. ``test_floor_zero_default``: extreme-drought negative raw → 0.
5. ``test_floor_false_negative``: floor_zero=False returns raw value.
6. ``test_chain_with_priestley_taylor``: PT λE_wet → iter-882.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    complementary_relationship_et_fv3,
    priestley_taylor_le_fv3,
)


def test_saturated_identity():
    """λE_wet = λE_pot (saturated surface) → λE_actual = λE_wet."""
    le_wet = jnp.array([350.0])
    le_pot = jnp.array([350.0])
    le_act = complementary_relationship_et_fv3(le_wet, le_pot)
    np.testing.assert_allclose(np.asarray(le_act), [350.0], rtol=1e-12)


def test_drying_decreases_actual():
    """λE_pot > λE_wet → λE_actual < λE_wet (drying regime)."""
    le_wet = jnp.array([300.0])
    le_pot = jnp.array([400.0])
    le_act = complementary_relationship_et_fv3(le_wet, le_pot)
    # 2·300 − 400 = 200
    np.testing.assert_allclose(np.asarray(le_act), [200.0], rtol=1e-12)
    assert float(le_act[0]) < float(le_wet[0])


def test_fully_dry_zero_floored():
    """λE_pot = 2·λE_wet → λE_actual = 0 (fully dry)."""
    le_wet = jnp.array([200.0])
    le_pot = jnp.array([400.0])
    le_act = complementary_relationship_et_fv3(le_wet, le_pot)
    np.testing.assert_allclose(np.asarray(le_act), [0.0], atol=1e-14)


def test_floor_zero_default():
    """Extreme drought: λE_pot=500, λE_wet=200 → raw=−100 → 0."""
    le_wet = jnp.array([200.0])
    le_pot = jnp.array([500.0])
    le_act = complementary_relationship_et_fv3(le_wet, le_pot)
    np.testing.assert_allclose(np.asarray(le_act), [0.0], atol=1e-14)


def test_floor_false_negative():
    """floor_zero=False returns raw 2·λE_wet − λE_pot."""
    le_wet = jnp.array([200.0])
    le_pot = jnp.array([500.0])
    le_act = complementary_relationship_et_fv3(
        le_wet, le_pot, floor_zero=False,
    )
    # 2·200 − 500 = −100
    np.testing.assert_allclose(np.asarray(le_act), [-100.0], rtol=1e-12)


def test_chain_with_priestley_taylor():
    """iter-881 PT → iter-882 CR using PM-derived λE_pot."""
    le_wet = priestley_taylor_le_fv3(
        available_energy=jnp.array([400.0]),
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
    )
    # λE_pot from Penman (caller-supplied)
    le_pot = jnp.array([450.0])
    le_act = complementary_relationship_et_fv3(le_wet, le_pot)
    assert jnp.all(jnp.isfinite(le_act))
    assert float(le_act[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=882)
    n_x, n_y = 6, 8
    le_wet = jnp.asarray(rng.uniform(100.0, 500.0, size=(n_x, n_y)))
    le_pot = jnp.asarray(rng.uniform(100.0, 700.0, size=(n_x, n_y)))
    le_act = complementary_relationship_et_fv3(le_wet, le_pot)
    assert le_act.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(le_act))
    assert jnp.all(le_act >= 0.0)
