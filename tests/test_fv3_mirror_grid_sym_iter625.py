"""FV3_3D iter 625: mirror_grid_face1_symmetrize port.

Faithful port of FV3 ``mirror_grid`` first loop (intra-face-1
SIGN-averaging symmetrization, fv_grid_tools.F90:2774-2807).

Tests
-----

1. ``test_symmetrize_shape_preserved``.
2. ``test_symmetrize_idempotent``.
3. ``test_symmetrize_i_mirror_lon_abs``.
4. ``test_symmetrize_j_mirror_lat_abs``.
5. ``test_symmetrize_odd_npx_center_zero``.
6. ``test_symmetrize_on_already_symmetric``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gnomonic_grids,
    mirror_grid_face1_symmetrize,
)


def test_symmetrize_shape_preserved():
    """Output shape matches input."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=0)
    lon_s, lat_s = mirror_grid_face1_symmetrize(lon, lat)
    assert lon_s.shape == (im + 1, im + 1)
    assert lat_s.shape == (im + 1, im + 1)
    assert jnp.all(jnp.isfinite(lon_s))
    assert jnp.all(jnp.isfinite(lat_s))


def test_symmetrize_idempotent():
    """Applying twice gives same result."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=0)
    lon_1, lat_1 = mirror_grid_face1_symmetrize(lon, lat)
    lon_2, lat_2 = mirror_grid_face1_symmetrize(lon_1, lat_1)
    assert jnp.allclose(lon_1, lon_2, atol=1e-14)
    assert jnp.allclose(lat_1, lat_2, atol=1e-14)


def test_symmetrize_i_mirror_lon_abs():
    """After symmetrize: |lon[i, j]| == |lon[npx-1-i, j]| for all (i, j)."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=0)
    lon_s, _ = mirror_grid_face1_symmetrize(lon, lat)
    abs_lon = jnp.abs(lon_s)
    assert jnp.allclose(abs_lon, abs_lon[::-1, :], atol=1e-14)


def test_symmetrize_j_mirror_lat_abs():
    """After symmetrize: |lat[i, j]| == |lat[i, npy-1-j]| for all (i, j)."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=0)
    _, lat_s = mirror_grid_face1_symmetrize(lon, lat)
    abs_lat = jnp.abs(lat_s)
    assert jnp.allclose(abs_lat, abs_lat[:, ::-1], atol=1e-14)


def test_symmetrize_odd_npx_center_zero():
    """For odd npx, central column lon = 0."""
    # Use im=8 so npx = 9 (odd)
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=0)
    lon_s, _ = mirror_grid_face1_symmetrize(lon, lat)
    center_i = (im + 1 - 1) // 2  # = 4
    assert jnp.allclose(lon_s[center_i, :], 0.0, atol=1e-14)


def test_symmetrize_on_already_symmetric():
    """gnomonic_grids output is already nearly symmetric → diff small."""
    im = 16
    lon, lat = gnomonic_grids(im, grid_type=0)
    lon_s, lat_s = mirror_grid_face1_symmetrize(lon, lat)
    # Should change by at most O(numerical-precision-of-input) per cell
    max_diff_lon = float(jnp.max(jnp.abs(lon_s - lon)))
    max_diff_lat = float(jnp.max(jnp.abs(lat_s - lat)))
    # gnomonic_grids already calls symm_ed → near-symmetric input
    # Diff should be << π
    assert max_diff_lon < 1e-3, f"max lon diff = {max_diff_lon}"
    assert max_diff_lat < 1e-3, f"max lat diff = {max_diff_lat}"
