"""FV3_3D iter 631: edge_factor_along_axis_nonortho port.

Faithful port of FV3 ``edge_factors`` non-ortho branch
(fv_grid_utils.F90:1212-1289).

Tests
-----

1. ``test_edge_factor_shape``.
2. ``test_edge_factor_endpoints_nan``.
3. ``test_edge_factor_uniform_grid_half``.
4. ``test_edge_factor_in_unit_interval``.
5. ``test_edge_factor_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import edge_factor_along_axis_nonortho


def _make_random_inputs(n, seed):
    rng = np.random.default_rng(seed)
    # Cells outside: 1D strip of (lon, lat)
    agrid_out_lon = jnp.asarray(rng.uniform(0.1, 2 * jnp.pi - 0.1, size=n))
    agrid_out_lat = jnp.asarray(rng.uniform(-0.5, 0.5, size=n))
    agrid_in_lon = jnp.asarray(rng.uniform(0.1, 2 * jnp.pi - 0.1, size=n))
    agrid_in_lat = jnp.asarray(rng.uniform(-0.5, 0.5, size=n))
    grid_corner_lon = jnp.asarray(rng.uniform(0.1, 2 * jnp.pi - 0.1, size=n + 1))
    grid_corner_lat = jnp.asarray(rng.uniform(-0.5, 0.5, size=n + 1))
    return (agrid_out_lon, agrid_out_lat,
            agrid_in_lon, agrid_in_lat,
            grid_corner_lon, grid_corner_lat)


def test_edge_factor_shape():
    """Output shape (n+1,)."""
    n = 8
    inputs = _make_random_inputs(n, seed=631)
    e = edge_factor_along_axis_nonortho(*inputs)
    assert e.shape == (n + 1,)


def test_edge_factor_endpoints_nan():
    """Endpoints (0 and n) must be NaN (FV3 big_number stays as fill)."""
    n = 6
    inputs = _make_random_inputs(n, seed=632)
    e = edge_factor_along_axis_nonortho(*inputs)
    assert jnp.isnan(e[0])
    assert jnp.isnan(e[-1])
    # Interior must be finite
    assert jnp.all(jnp.isfinite(e[1:-1]))


def test_edge_factor_uniform_grid_half():
    """If midpoints are equidistant from corner, edge_factor = 0.5."""
    # Build collinear points along the equator
    n = 6
    dl = 0.1
    j_idx = jnp.arange(n, dtype=jnp.float64)
    # Outside cells slightly south, inside cells slightly north
    agrid_out_lon = dl * j_idx + 0.5 * dl
    agrid_out_lat = jnp.full((n,), -0.01)
    agrid_in_lon = dl * j_idx + 0.5 * dl
    agrid_in_lat = jnp.full((n,), 0.01)
    # Grid corners between cells: midpoint of adjacent agrid_in cells
    # at the equator
    grid_corner_lon = dl * jnp.arange(n + 1, dtype=jnp.float64)
    grid_corner_lat = jnp.full((n + 1,), 0.0)
    e = edge_factor_along_axis_nonortho(
        agrid_out_lon, agrid_out_lat,
        agrid_in_lon, agrid_in_lat,
        grid_corner_lon, grid_corner_lat,
    )
    # For symmetrically-placed corner between two midpoints,
    # d1 = d2 → edge_factor = 0.5
    interior = e[1:-1]
    assert jnp.allclose(interior, 0.5, atol=1e-3), (
        f"interior factors = {interior}, expected ≈ 0.5"
    )


def test_edge_factor_in_unit_interval():
    """All interior weights must be in [0, 1]."""
    n = 8
    inputs = _make_random_inputs(n, seed=633)
    e = edge_factor_along_axis_nonortho(*inputs)
    interior = e[1:-1]
    assert jnp.all(interior >= 0.0)
    assert jnp.all(interior <= 1.0)


def test_edge_factor_finite():
    """Interior values are finite (no NaN/Inf)."""
    n = 12
    inputs = _make_random_inputs(n, seed=634)
    e = edge_factor_along_axis_nonortho(*inputs)
    assert jnp.all(jnp.isfinite(e[1:-1]))
