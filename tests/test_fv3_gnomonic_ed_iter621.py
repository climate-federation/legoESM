"""FV3_3D iter 621: gnomonic_ed port (FV3 canonical grid).

Faithful port of FV3 ``gnomonic_ed`` (fv_grid_utils.F90:1313-1407).

Tests
-----

1. ``test_gnomonic_ed_shape``.
2. ``test_gnomonic_ed_finite``.
3. ``test_gnomonic_ed_corners_on_unit_sphere``.
4. ``test_gnomonic_ed_w_edge_longitude``.
5. ``test_gnomonic_ed_e_edge_longitude``.
6. ``test_gnomonic_ed_n_s_symmetry``.
7. ``test_gnomonic_ed_aspect_ratio_bounded``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gnomonic_ed,
    great_circle_distance,
    latlon2xyz,
)


def test_gnomonic_ed_shape():
    """Output shapes are (im+1, im+1)."""
    im = 12
    lon, lat = gnomonic_ed(im)
    assert lon.shape == (im + 1, im + 1)
    assert lat.shape == (im + 1, im + 1)


def test_gnomonic_ed_finite():
    """All grid points are finite."""
    im = 16
    lon, lat = gnomonic_ed(im)
    assert jnp.all(jnp.isfinite(lon))
    assert jnp.all(jnp.isfinite(lat))


def test_gnomonic_ed_corners_on_unit_sphere():
    """All corner positions lie on the unit sphere."""
    im = 8
    lon, lat = gnomonic_ed(im)
    x, y, z = latlon2xyz(lon, lat)
    norm = jnp.sqrt(x * x + y * y + z * z)
    assert jnp.allclose(norm, 1.0, atol=1e-12)


def test_gnomonic_ed_w_edge_longitude():
    """West edge (i=0) at lon = 0.75π for all j (FV3 line 1346)."""
    im = 8
    lon, _ = gnomonic_ed(im)
    assert jnp.allclose(lon[0, :], 0.75 * jnp.pi, atol=1e-12)


def test_gnomonic_ed_e_edge_longitude():
    """East edge (i=im) at lon = 1.25π for all j (FV3 line 1347)."""
    im = 8
    lon, _ = gnomonic_ed(im)
    assert jnp.allclose(lon[im, :], 1.25 * jnp.pi, atol=1e-12)


def test_gnomonic_ed_n_s_symmetry():
    """N and S edges symmetric in latitude (FV3 line 1358 → -theta)."""
    im = 12
    _, lat = gnomonic_ed(im)
    # S edge at j=0, N edge at j=im, opposite lat
    for i in range(1, im):
        assert abs(float(lat[i, 0]) + float(lat[i, im])) < 1e-12, (
            f"i={i}: lat[i,0] + lat[i,im] = "
            f"{float(lat[i, 0]) + float(lat[i, im])}"
        )


def test_gnomonic_ed_aspect_ratio_bounded():
    """FV3 docstring claims max(dx,dy)/min(dx,dy) = √2; check < 1.5 at C32."""
    im = 32
    lon, lat = gnomonic_ed(im)
    # Sample dx along i-direction at j=im/2
    j_eq = im // 2
    d_arr = []
    for i in range(im):
        d = float(great_circle_distance(
            lon[i, j_eq], lat[i, j_eq],
            lon[i + 1, j_eq], lat[i + 1, j_eq],
            radius=1.0,
        ))
        d_arr.append(d)
    d_arr = np.asarray(d_arr)
    aspect = float(np.max(d_arr) / np.min(d_arr))
    assert aspect < 1.5, f"aspect ratio along equator = {aspect}, expected < √2"
