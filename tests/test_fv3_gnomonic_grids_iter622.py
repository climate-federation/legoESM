"""FV3_3D iter 622: gnomonic_grids dispatcher port.

Faithful port of FV3 ``gnomonic_grids`` (fv_grid_utils.F90:
1290-1311) — dispatcher for the three FV3 grid generators
(grid_type 0, 1, 2) with symm_ed + lon shift post-processing.

Tests
-----

1. ``test_gnomonic_grids_shape``.
2. ``test_gnomonic_grids_dispatch_0_ed``.
3. ``test_gnomonic_grids_dispatch_1_dist``.
4. ``test_gnomonic_grids_dispatch_2_angl``.
5. ``test_gnomonic_grids_corners_on_unit_sphere``.
6. ``test_gnomonic_grids_lon_shifted_by_pi``.
7. ``test_gnomonic_grids_invalid_type_raises``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gnomonic_angl,
    gnomonic_dist,
    gnomonic_ed,
    gnomonic_grids,
    latlon2xyz,
    symm_ed,
)


def test_gnomonic_grids_shape():
    """Output shapes (im+1, im+1) for each grid_type."""
    im = 8
    for gt in (0, 1, 2):
        lon, lat = gnomonic_grids(im, grid_type=gt)
        assert lon.shape == (im + 1, im + 1), f"gt={gt}"
        assert lat.shape == (im + 1, im + 1), f"gt={gt}"
        assert jnp.all(jnp.isfinite(lon)), f"gt={gt}"
        assert jnp.all(jnp.isfinite(lat)), f"gt={gt}"


def test_gnomonic_grids_dispatch_0_ed():
    """grid_type=0 must dispatch to gnomonic_ed + symm_ed + shift."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=0)
    # Reproduce expected via direct calls
    lon0, lat0 = gnomonic_ed(im)
    lon0, lat0 = symm_ed(lon0, lat0)
    lon0 = lon0 - jnp.pi
    assert jnp.allclose(lon, lon0, atol=1e-12)
    assert jnp.allclose(lat, lat0, atol=1e-12)


def test_gnomonic_grids_dispatch_1_dist():
    """grid_type=1 dispatches to gnomonic_dist."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=1)
    lon0, lat0 = gnomonic_dist(im)
    lon0, lat0 = symm_ed(lon0, lat0)
    lon0 = lon0 - jnp.pi
    assert jnp.allclose(lon, lon0, atol=1e-12)
    assert jnp.allclose(lat, lat0, atol=1e-12)


def test_gnomonic_grids_dispatch_2_angl():
    """grid_type=2 dispatches to gnomonic_angl."""
    im = 8
    lon, lat = gnomonic_grids(im, grid_type=2)
    lon0, lat0 = gnomonic_angl(im)
    lon0, lat0 = symm_ed(lon0, lat0)
    lon0 = lon0 - jnp.pi
    assert jnp.allclose(lon, lon0, atol=1e-12)
    assert jnp.allclose(lat, lat0, atol=1e-12)


def test_gnomonic_grids_corners_on_unit_sphere():
    """All grid points on unit sphere for all grid_types."""
    im = 6
    for gt in (0, 1, 2):
        lon, lat = gnomonic_grids(im, grid_type=gt)
        x, y, z = latlon2xyz(lon, lat)
        norm = jnp.sqrt(x * x + y * y + z * z)
        assert jnp.allclose(norm, 1.0, atol=1e-12), f"gt={gt}"


def test_gnomonic_grids_lon_shifted_by_pi():
    """Dispatcher output lon must equal grid generator + symm_ed
    output minus π."""
    im = 8
    lon_disp, _ = gnomonic_grids(im, grid_type=0)
    lon_raw, lat_raw = gnomonic_ed(im)
    lon_sym, _ = symm_ed(lon_raw, lat_raw)
    assert jnp.allclose(lon_disp, lon_sym - jnp.pi, atol=1e-12)


def test_gnomonic_grids_invalid_type_raises():
    """Unsupported grid_type should raise ValueError."""
    with pytest.raises(ValueError):
        gnomonic_grids(8, grid_type=99)
