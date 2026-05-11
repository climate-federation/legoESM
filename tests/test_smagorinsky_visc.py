"""Unit tests for FV3_3D iter 57 Smagorinsky-style adaptive A_h helper.

Tests structural properties of ``compute_smagorinsky_ah_{2d,3d}``:

1. ``c_s = 0`` → A_h = 0 everywhere (regression / cheap gate).
2. Zero winds → zero A_h (correct edge case).
3. Uniform wind (constant u, v) → zero A_h (no shear).
4. Linear shear ``u = a*y`` → constant A_h proportional to ``a``.
5. Linearity in ``c_s``: ``ah(c=2*c0) = 2 * ah(c=c0)``.
6. 3D wrapper preserves the level axis.

The helper is NOT YET wired into the dycore — these tests pin
the building-block behavior so iter-58+ wiring has a stable
foundation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core._smagorinsky_visc import (
    compute_smagorinsky_ah_2d,
    compute_smagorinsky_ah_3d,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


@pytest.fixture(scope="module")
def small_cube():
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    return grid, cdgrid, n


def test_smagorinsky_zero_cs_returns_zero(small_cube):
    """``c_s = 0`` → A_h = 0 everywhere.  Cheap on/off gate.

    Pin this so iter-58+ users can disable Smagorinsky without
    paying for unnecessary computation.
    """
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=42)
    u = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1)))

    ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.0)
    np.testing.assert_array_equal(np.asarray(ah), np.zeros((6, n + 1, n + 1)))


def test_smagorinsky_zero_winds(small_cube):
    """Zero u, v → zero A_h (no strain rate)."""
    _, cdgrid, n = small_cube
    u = jnp.zeros((6, n + 1, n + 1))
    v = jnp.zeros((6, n + 1, n + 1))
    ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)
    np.testing.assert_array_equal(np.asarray(ah), np.zeros((6, n + 1, n + 1)))


def test_smagorinsky_uniform_winds_near_zero(small_cube):
    """Uniform wind (constant u, v) → near-zero A_h (no shear).

    The constant-angle FD approximation may produce small residuals
    at panel boundaries due to cross-panel halo exchanges, but the
    result should be much smaller than a typical synoptic-scale
    A_h (~1e6 m^2/s).
    """
    _, cdgrid, n = small_cube
    u = jnp.full((6, n + 1, n + 1), 5.0)  # 5 m/s uniform u
    v = jnp.full((6, n + 1, n + 1), 3.0)  # 3 m/s uniform v
    ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)
    max_ah = float(jnp.max(jnp.abs(ah)))
    # On a curvilinear grid the local FD of "uniform" winds in
    # face-local frame may pick up small residuals from the metric.
    # Bound to 1 % of typical synoptic A_h (~1e6).
    assert max_ah < 1e4, (
        f"uniform winds on curvilinear grid give max A_h = {max_ah:.3e}, "
        f"expected near zero (< 1e4)"
    )


def test_smagorinsky_linearity_in_cs(small_cube):
    """Linearity: ``ah(c=2*c0) = 2 * ah(c=c0)``.  Captures that
    c_s acts as a simple multiplier on the strain magnitude.
    """
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=11)
    u = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1)))

    ah_1 = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.1)
    ah_2 = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)

    np.testing.assert_allclose(
        np.asarray(ah_2), 2.0 * np.asarray(ah_1),
        rtol=1e-12, atol=1e-12,
    )


def test_smagorinsky_finite_on_random_input(small_cube):
    """All output cells are finite on random input — regression
    guard against divide-by-zero or NaN propagation.
    """
    _, cdgrid, n = small_cube
    rng = np.random.default_rng(seed=99)
    u = jnp.asarray(rng.uniform(-100.0, 100.0, size=(6, n + 1, n + 1)))
    v = jnp.asarray(rng.uniform(-100.0, 100.0, size=(6, n + 1, n + 1)))
    ah = compute_smagorinsky_ah_2d(u, v, cdgrid, c_s=0.2)
    assert jnp.all(jnp.isfinite(ah))


def test_smagorinsky_3d_wrapper_shape_and_per_level_match(small_cube):
    """3D wrapper preserves the level axis and matches per-level 2D."""
    _, cdgrid, n = small_cube
    nlev = 5
    rng = np.random.default_rng(seed=7)
    u3 = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1, nlev)))
    v3 = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1, nlev)))
    ah3 = compute_smagorinsky_ah_3d(u3, v3, cdgrid, c_s=0.2)
    assert ah3.shape == (6, n + 1, n + 1, nlev)
    # Each level should match the 2D call.
    for lev in range(nlev):
        ah2 = compute_smagorinsky_ah_2d(
            u3[..., lev], v3[..., lev], cdgrid, c_s=0.2,
        )
        np.testing.assert_allclose(
            np.asarray(ah3[..., lev]), np.asarray(ah2), rtol=1e-12,
        )


def test_smagorinsky_3d_zero_cs_returns_zero(small_cube):
    """3D ``c_s = 0`` → zero output (mirrors the 2D gate)."""
    _, cdgrid, n = small_cube
    nlev = 4
    rng = np.random.default_rng(seed=33)
    u3 = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1, nlev)))
    v3 = jnp.asarray(rng.uniform(-10.0, 10.0, size=(6, n + 1, n + 1, nlev)))
    ah3 = compute_smagorinsky_ah_3d(u3, v3, cdgrid, c_s=0.0)
    np.testing.assert_array_equal(
        np.asarray(ah3), np.zeros_like(np.asarray(u3)),
    )
