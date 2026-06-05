"""FV3_3D iter 617: gnomonic_angl + gnomonic_dist face-2 grid ports.

Faithful ports of FV3 fv_grid_utils.F90:
- ``gnomonic_angl`` (F90:1531) — equi-angular gnomonic grid (FV3 default)
- ``gnomonic_dist`` (F90:1558) — equi-distance gnomonic grid

Tests
-----

1. ``test_gnomonic_angl_shape``.
2. ``test_gnomonic_angl_corners_on_face2_x_negative``.
3. ``test_gnomonic_angl_center_on_face2_axis``.
4. ``test_gnomonic_dist_shape``.
5. ``test_gnomonic_dist_corners_unit_norm``.
6. ``test_gnomonic_angl_vs_dist_center_agreement``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gnomonic_angl,
    gnomonic_dist,
    latlon2xyz,
)


def test_gnomonic_angl_shape():
    """Output shapes are (im+1, im+1)."""
    im = 12
    lon, lat = gnomonic_angl(im)
    assert lon.shape == (im + 1, im + 1)
    assert lat.shape == (im + 1, im + 1)
    assert jnp.all(jnp.isfinite(lon))
    assert jnp.all(jnp.isfinite(lat))


def test_gnomonic_angl_corners_on_face2_x_negative():
    """All face-2 points have Cartesian x < 0 (face 2 is -x face)."""
    im = 16
    lon, lat = gnomonic_angl(im)
    x, y, z = latlon2xyz(lon, lat)
    assert jnp.all(x < 0.0), f"face 2 should be -x; max x = {float(x.max())}"


def test_gnomonic_angl_center_on_face2_axis():
    """Center of face 2 is at Cartesian (-1, 0, 0)."""
    im = 8
    lon, lat = gnomonic_angl(im)
    # Center point of (im+1, im+1) grid is at (im/2, im/2)
    c = im // 2
    x, y, z = latlon2xyz(lon[c, c], lat[c, c])
    assert abs(float(x) - (-1.0)) < 1e-10, f"x = {float(x)}, expected -1"
    assert abs(float(y)) < 1e-10, f"y = {float(y)}"
    assert abs(float(z)) < 1e-10, f"z = {float(z)}"


def test_gnomonic_dist_shape():
    """Output shapes are (im+1, im+1)."""
    im = 10
    lon, lat = gnomonic_dist(im)
    assert lon.shape == (im + 1, im + 1)
    assert lat.shape == (im + 1, im + 1)
    assert jnp.all(jnp.isfinite(lon))
    assert jnp.all(jnp.isfinite(lat))


def test_gnomonic_dist_corners_unit_norm():
    """All grid points lie on unit sphere after latlon2xyz."""
    im = 8
    lon, lat = gnomonic_dist(im)
    x, y, z = latlon2xyz(lon, lat)
    norm = jnp.sqrt(x * x + y * y + z * z)
    assert jnp.allclose(norm, 1.0, atol=1e-12)


def test_gnomonic_angl_vs_dist_center_agreement():
    """At the face center, both grids must give (-1, 0, 0)."""
    im = 8
    c = im // 2
    lon_a, lat_a = gnomonic_angl(im)
    lon_d, lat_d = gnomonic_dist(im)
    xa, ya, za = latlon2xyz(lon_a[c, c], lat_a[c, c])
    xd, yd, zd = latlon2xyz(lon_d[c, c], lat_d[c, c])
    assert abs(float(xa) - float(xd)) < 1e-10
    assert abs(float(ya) - float(yd)) < 1e-10
    assert abs(float(za) - float(zd)) < 1e-10
