"""FV3_3D iter 774: bulk_richardson_fv3 (Ri_b).

Ri_b(z) = g·z·(θ_v(z) − θ_v_surf) / (θ_v_surf · (u² + v²))

Used by PBL schemes for boundary-layer height detection.

Tests
-----

1. ``test_rib_surface_zero``: at z=0 with θ_v=θ_v_surf → Ri_b=0.
2. ``test_rib_stable_positive``: θ_v(z) > θ_v_surf → Ri_b > 0.
3. ``test_rib_unstable_negative``: θ_v(z) < θ_v_surf → Ri_b < 0.
4. ``test_rib_calm_air_floor``: zero wind → bounded by wind_sq_floor.
5. ``test_rib_shapes_3d``: km dim preserved.
6. ``test_rib_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import bulk_richardson_fv3


def test_rib_surface_zero():
    """z=0 and θ_v(0)=θ_v_surf → Ri_b=0."""
    theta_v = jnp.array([[300.0, 302.0, 305.0]])
    z = jnp.array([[0.0, 100.0, 500.0]])
    u = jnp.array([[2.0, 5.0, 10.0]])
    v = jnp.array([[1.0, 3.0, 5.0]])
    theta_v_surf = theta_v[..., 0]
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    # At z=0 with θ_v=θ_v_surf, Ri_b is 0 (numerator vanishes)
    assert ri_b.shape == (1, 3)
    np.testing.assert_allclose(np.asarray(ri_b[..., 0]), [0.0], atol=1e-14)


def test_rib_stable_positive():
    """θ_v(z) > θ_v_surf (stable) → Ri_b > 0."""
    theta_v_surf = jnp.array([295.0])
    theta_v = jnp.array([[295.0, 297.0, 300.0]])
    z = jnp.array([[0.0, 200.0, 500.0]])
    u = jnp.array([[1.0, 2.0, 3.0]])
    v = jnp.zeros_like(u)
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    # Index 0 is zero (z=0); 1 and 2 should be positive
    assert jnp.all(ri_b[..., 1:] > 0.0)


def test_rib_unstable_negative():
    """θ_v(z) < θ_v_surf (unstable / superadiabatic) → Ri_b < 0."""
    theta_v_surf = jnp.array([300.0])
    theta_v = jnp.array([[300.0, 298.0, 295.0]])
    z = jnp.array([[0.0, 200.0, 500.0]])
    u = jnp.array([[1.0, 2.0, 3.0]])
    v = jnp.zeros_like(u)
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    assert jnp.all(ri_b[..., 1:] < 0.0)


def test_rib_calm_air_floor():
    """Zero wind → wind_sq_floor caps the denominator → finite Ri_b."""
    theta_v_surf = jnp.array([295.0])
    theta_v = jnp.array([[295.0, 300.0, 305.0]])  # stable
    z = jnp.array([[0.0, 100.0, 500.0]])
    u = jnp.zeros((1, 3))
    v = jnp.zeros((1, 3))
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    # With wind_sq_floor=0.01 and stable Δθ_v, Ri_b is large but finite
    assert jnp.all(jnp.isfinite(ri_b))
    # Aloft is significantly positive
    assert ri_b[0, 2] > 100.0


def test_rib_shapes_3d():
    """3-D shapes: theta_v_surf broadcasts across km."""
    rng = np.random.default_rng(seed=774)
    n_x, n_y, km = 4, 5, 20
    theta_v_surf = jnp.asarray(rng.uniform(295.0, 305.0, size=(n_x, n_y)))
    theta_v = jnp.asarray(rng.uniform(285.0, 320.0, size=(n_x, n_y, km)))
    z = jnp.cumsum(jnp.asarray(rng.uniform(50.0, 500.0, size=(n_x, n_y, km))), axis=-1)
    u = jnp.asarray(rng.uniform(-15.0, 15.0, size=(n_x, n_y, km)))
    v = jnp.asarray(rng.uniform(-15.0, 15.0, size=(n_x, n_y, km)))
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    assert ri_b.shape == (n_x, n_y, km)


def test_rib_finite():
    """No NaN/Inf for realistic surface-PBL inputs."""
    rng = np.random.default_rng(seed=775)
    km = 30
    theta_v_surf = jnp.asarray(rng.uniform(290.0, 305.0, size=(4,)))
    theta_v = jnp.asarray(rng.uniform(280.0, 325.0, size=(4, km)))
    z = jnp.cumsum(jnp.asarray(rng.uniform(50.0, 500.0, size=(4, km))), axis=-1)
    u = jnp.asarray(rng.uniform(-25.0, 25.0, size=(4, km)))
    v = jnp.asarray(rng.uniform(-25.0, 25.0, size=(4, km)))
    ri_b = bulk_richardson_fv3(theta_v_surf, theta_v, u, v, z)
    assert jnp.all(jnp.isfinite(ri_b))
