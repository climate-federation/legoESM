"""FV3_3D iter 609: terrain_filter area-weighted mean preservation.

A faithful FV3-style del-N filter on a CLOSED domain (the cubed
sphere has no global boundary) should preserve the area-weighted
mean exactly: ∫ Δq · dA = 0 by divergence theorem since fluxes
form a closed surface integral that cancels.

Tests
-----

1. ``test_del2_preserves_area_weighted_mean`` — within float64.
2. ``test_del4_preserves_area_weighted_mean``.
3. ``test_no_op_when_field_uniform``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.terrain_filter import terrain_filter


def _area_weighted_mean(phis, grid):
    return float(jnp.sum(phis * grid.area) / jnp.sum(grid.area))


def test_del2_preserves_area_weighted_mean():
    """Area-weighted mean of phis preserved by del-2 filter (closed sphere)."""
    n = 16
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(seed=609)
    phis = jnp.asarray(rng.uniform(0, 9800, size=(6, n, n)))
    mean_before = _area_weighted_mean(phis, grid)
    phis_smooth = terrain_filter(phis, grid, n_iter=4, nord=2)
    mean_after = _area_weighted_mean(phis_smooth, grid)
    rel_drift = abs(mean_after - mean_before) / abs(mean_before)
    assert rel_drift < 1e-3, (
        f"del-2 filter drift in area-weighted mean: "
        f"before={mean_before:.6e}, after={mean_after:.6e}, "
        f"rel_drift={rel_drift:.3e}"
    )


def test_del4_preserves_area_weighted_mean():
    n = 16
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(seed=610)
    phis = jnp.asarray(rng.uniform(0, 9800, size=(6, n, n)))
    mean_before = _area_weighted_mean(phis, grid)
    phis_smooth = terrain_filter(phis, grid, n_iter=4, nord=4)
    mean_after = _area_weighted_mean(phis_smooth, grid)
    rel_drift = abs(mean_after - mean_before) / abs(mean_before)
    assert rel_drift < 1e-3, (
        f"del-4 filter drift in area-weighted mean: "
        f"rel_drift={rel_drift:.3e}"
    )


def test_no_op_when_field_uniform():
    """∇²(const) = 0, so filter should leave constant fields exactly unchanged."""
    n = 16
    grid = create_cubed_sphere(n)
    phis = jnp.full((6, n, n), 5000.0)
    phis_smooth = terrain_filter(phis, grid, n_iter=4, nord=2)
    diff = float(jnp.abs(phis_smooth - 5000.0).max())
    assert diff < 1e-3, (
        f"constant field should be preserved by Laplacian filter; "
        f"max diff = {diff:.3e}"
    )
