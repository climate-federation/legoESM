"""FV3_3D iter 810: lifted_index_fv3.

LI = T_env(500 mb) − T_parcel_lifted(500 mb).

Tests
-----

1. ``test_li_neutral``: parcel = env → LI = 0.
2. ``test_li_unstable``: warmer parcel → LI < 0.
3. ``test_li_stable``: cooler parcel → LI > 0.
4. ``test_li_severe_threshold``: LI ≤ −5 caught by inequality.
5. ``test_li_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import lifted_index_fv3


def test_li_neutral():
    """T_env=T_parcel → LI=0."""
    t_env = jnp.array([250.0, 252.0, 248.0])
    li = lifted_index_fv3(t_env, t_env)
    np.testing.assert_allclose(np.asarray(li), jnp.zeros((3,)), atol=1e-15)


def test_li_unstable():
    """Warmer parcel → LI<0 (unstable)."""
    t_env = jnp.array([250.0])
    t_parcel = jnp.array([255.0])
    li = lifted_index_fv3(t_env, t_parcel)
    np.testing.assert_allclose(np.asarray(li), [-5.0], rtol=1e-12)
    assert float(li[0]) < 0.0


def test_li_stable():
    """Cooler parcel → LI>0 (stable)."""
    t_env = jnp.array([255.0])
    t_parcel = jnp.array([250.0])
    li = lifted_index_fv3(t_env, t_parcel)
    np.testing.assert_allclose(np.asarray(li), [5.0], rtol=1e-12)
    assert float(li[0]) > 0.0


def test_li_severe_threshold():
    """LI ≤ -5 marks severe-storm threshold."""
    t_env = jnp.array([250.0, 250.0, 250.0])
    t_parcel = jnp.array([252.0, 256.0, 258.0])
    li = lifted_index_fv3(t_env, t_parcel)
    severe = li <= -5.0
    # LI = -2, -6, -8 → [False, True, True]
    np.testing.assert_array_equal(
        np.asarray(severe), np.array([False, True, True])
    )


def test_li_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=810)
    n_x, n_y = 6, 8
    t_env = jnp.asarray(rng.uniform(240.0, 260.0, size=(n_x, n_y)))
    t_parcel = jnp.asarray(rng.uniform(240.0, 270.0, size=(n_x, n_y)))
    li = lifted_index_fv3(t_env, t_parcel)
    assert li.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(li))
