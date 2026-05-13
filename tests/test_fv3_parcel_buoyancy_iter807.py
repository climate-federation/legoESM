"""FV3_3D iter 807: parcel_buoyancy_fv3.

b = g·(θ_v_parcel − θ_v_env)/θ_v_env

Tests
-----

1. ``test_b_neutral``: θ_v_p = θ_v_env → b = 0.
2. ``test_b_warm_parcel``: warmer parcel → b > 0.
3. ``test_b_cool_parcel``: cooler parcel → b < 0.
4. ``test_b_magnitude_symmetric``: |Δθ_v| symmetric.
5. ``test_b_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import parcel_buoyancy_fv3


def test_b_neutral():
    """θ_v_p = θ_v_env → b = 0."""
    theta_v = jnp.array([300.0, 290.0, 310.0])
    b = parcel_buoyancy_fv3(theta_v, theta_v)
    np.testing.assert_allclose(np.asarray(b), jnp.zeros((3,)), atol=1e-15)


def test_b_warm_parcel():
    """Parcel 5 K warmer than env: b > 0, magnitude ≈ g·5/300 ≈ 0.16 m/s²."""
    env = jnp.array([300.0])
    p = env + 5.0
    b = parcel_buoyancy_fv3(p, env)
    expected = constants.g * 5.0 / 300.0
    np.testing.assert_allclose(np.asarray(b), [expected], rtol=1e-12)


def test_b_cool_parcel():
    """Parcel 3 K cooler → b < 0."""
    env = jnp.array([290.0])
    p = env - 3.0
    b = parcel_buoyancy_fv3(p, env)
    assert float(b[0]) < 0.0


def test_b_magnitude_symmetric():
    """+Δ and −Δ give symmetric magnitude (same |b|)."""
    env = jnp.array([300.0])
    p_up = env + 2.0
    p_dn = env - 2.0
    b_up = parcel_buoyancy_fv3(p_up, env)
    b_dn = parcel_buoyancy_fv3(p_dn, env)
    np.testing.assert_allclose(
        np.asarray(b_up), -np.asarray(b_dn), rtol=1e-12
    )


def test_b_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=807)
    n_x, n_y, km = 4, 5, 20
    env = jnp.asarray(rng.uniform(280.0, 320.0, size=(n_x, n_y, km)))
    parcel = env + jnp.asarray(rng.uniform(-5.0, 5.0, size=(n_x, n_y, km)))
    b = parcel_buoyancy_fv3(parcel, env)
    assert b.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(b))
