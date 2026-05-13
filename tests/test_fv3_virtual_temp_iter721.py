"""FV3_3D iter 721: virtual_temp_fv3 helper + iter-718 duplicate cleanup.

Iter-718 had accidentally redefined ``get_vorticity_fv3`` that
iter-656 already ported.  iter-721 removes the duplicate and adds
``virtual_temp_fv3`` (T_v = T·(1 + zvir·q)) helper used throughout
FV3 dyn_core and fv_diagnostics.

Tests
-----

1. ``test_virtual_temp_dry_returns_T``.
2. ``test_virtual_temp_moist_increases``.
3. ``test_virtual_temp_custom_zvir``.
4. ``test_virtual_temp_iter656_get_vorticity_still_works``.
5. ``test_virtual_temp_shapes_3d``.
6. ``test_virtual_temp_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import get_vorticity_fv3, virtual_temp_fv3


def test_virtual_temp_dry_returns_T():
    """q = 0 → T_v = T."""
    T = jnp.full((5,), 280.0)
    q = jnp.zeros((5,))
    Tv = virtual_temp_fv3(T, q)
    assert jnp.allclose(Tv, T, atol=1e-12)


def test_virtual_temp_moist_increases():
    """q > 0 → T_v > T (water vapor is lighter than dry air)."""
    T = jnp.full((5,), 290.0)
    q = jnp.full((5,), 0.015)
    Tv = virtual_temp_fv3(T, q)
    assert jnp.all(Tv > T)
    # Δ should be ~ T·zvir·q ≈ 290·0.608·0.015 ≈ 2.65 K
    expected_delta = 290.0 * (constants.R_v / constants.R_d - 1.0) * 0.015
    assert jnp.allclose(Tv - T, expected_delta, atol=1e-10)


def test_virtual_temp_custom_zvir():
    """Custom zvir overrides the default."""
    T = jnp.full((5,), 280.0)
    q = jnp.full((5,), 0.01)
    Tv = virtual_temp_fv3(T, q, zvir=1.0)  # Non-physical but tests plumbing
    expected = 280.0 * (1.0 + 1.0 * 0.01)
    assert jnp.allclose(Tv, expected, atol=1e-10)


def test_virtual_temp_iter656_get_vorticity_still_works():
    """iter-721 removed duplicate get_vorticity_fv3 from iter-718;
    the iter-656 canonical definition must still be importable and
    functional."""
    n = 6
    u = jnp.full((n, n + 1), 5.0)
    v = jnp.full((n + 1, n), 3.0)
    dx = jnp.full((n, n + 1), 1000.0)
    dy = jnp.full((n + 1, n), 1000.0)
    rarea = jnp.full((n, n), 1.0 / (1000.0 * 1000.0))
    vort = get_vorticity_fv3(u, v, dx, dy, rarea)
    # Uniform wind → vort = 0
    assert jnp.all(jnp.abs(vort) < 1e-12)


def test_virtual_temp_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=721)
    n_x, n_y, km = 4, 5, 20
    T = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    Tv = virtual_temp_fv3(T, q)
    assert Tv.shape == (n_x, n_y, km)


def test_virtual_temp_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=722)
    T = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 30)))
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 30)))
    Tv = virtual_temp_fv3(T, q)
    assert jnp.all(jnp.isfinite(Tv))
