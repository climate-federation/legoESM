"""FV3_3D iter 674: get_staggered_grid_fv3 port.

Faithful JAX port of FV3 ``get_staggered_grid`` (tools/
fv_treat_da_inc.F90:444-475).  B-grid corners → C/D-grid
edge midpoints.

Tests
-----

1. ``test_staggered_shapes``.
2. ``test_staggered_unit_sphere``.
3. ``test_staggered_equator_midpoints``.
4. ``test_staggered_finite``.
5. ``test_staggered_batched``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    get_staggered_grid_fv3,
    latlon2xyz,
    make_fv3_native_grid,
)


def test_staggered_shapes():
    """C-grid (n+1, n); D-grid (n, n+1)."""
    n = 8
    rng = np.random.default_rng(seed=674)
    pt_b_lon = jnp.asarray(rng.uniform(0.1, 2 * jnp.pi - 0.1, size=(n + 1, n + 1)))
    pt_b_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n + 1, n + 1)))
    pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat = get_staggered_grid_fv3(
        pt_b_lon, pt_b_lat,
    )
    assert pt_c_lon.shape == (n + 1, n)
    assert pt_c_lat.shape == (n + 1, n)
    assert pt_d_lon.shape == (n, n + 1)
    assert pt_d_lat.shape == (n, n + 1)


def test_staggered_unit_sphere():
    """Edge midpoints should remain on unit sphere."""
    n = 8
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    pt_b_lon = lons[0]
    pt_b_lat = lats[0]
    pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat = get_staggered_grid_fv3(
        pt_b_lon, pt_b_lat,
    )
    # All midpoints on unit sphere
    for lon, lat in [(pt_c_lon, pt_c_lat), (pt_d_lon, pt_d_lat)]:
        x, y, z = latlon2xyz(lon, lat)
        norm = jnp.sqrt(x * x + y * y + z * z)
        assert jnp.allclose(norm, 1.0, atol=1e-12)


def test_staggered_equator_midpoints():
    """For uniform equator grid, midpoints at exact uniform spacing."""
    # 5 corners along equator
    pt_b_lon = jnp.asarray([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4],
    ])
    pt_b_lat = jnp.zeros((5, 2))
    pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat = get_staggered_grid_fv3(
        pt_b_lon, pt_b_lat,
    )
    # D-grid midpoints in i-direction: average lon = (0+0.1)/2, (0.1+0.2)/2, ...
    expected_d_lon = jnp.asarray([0.05, 0.15, 0.25, 0.35])
    # Across i axis, both rows j=0 and j=1 should give same lon (since lat=0)
    assert jnp.allclose(pt_d_lon[:, 0], expected_d_lon, atol=1e-12)


def test_staggered_finite():
    """No NaN/Inf in any outputs."""
    n = 6
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    pt_b_lon = lons[0]
    pt_b_lat = lats[0]
    pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat = get_staggered_grid_fv3(
        pt_b_lon, pt_b_lat,
    )
    for arr in (pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat):
        assert jnp.all(jnp.isfinite(arr))


def test_staggered_batched():
    """Leading axes (e.g., face) preserved."""
    n = 4
    lons, lats = make_fv3_native_grid(n, grid_type=0)
    pt_c_lon, pt_c_lat, pt_d_lon, pt_d_lat = get_staggered_grid_fv3(
        lons, lats,
    )
    assert pt_c_lon.shape == (6, n + 1, n)
    assert pt_d_lon.shape == (6, n, n + 1)
