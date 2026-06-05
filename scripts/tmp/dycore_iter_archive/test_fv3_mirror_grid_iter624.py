"""FV3_3D iter 624: mirror_grid_faces port (FV3 6-face construction).

Faithful port of FV3 ``mirror_grid`` faces-2-to-6 rotation sequence
(fv_grid_tools.F90:2809-2897).

Tests
-----

1. ``test_mirror_grid_faces_shape``.
2. ``test_mirror_grid_face1_unchanged``.
3. ``test_mirror_grid_all_unit_sphere``.
4. ``test_mirror_grid_face2_center_geographic``.
5. ``test_mirror_grid_finite_everywhere``.
6. ``test_mirror_grid_face_orthogonality``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gnomonic_grids,
    latlon2xyz,
    mirror_grid_faces,
)


def test_mirror_grid_faces_shape():
    """Output shape (6, im+1, im+1)."""
    im = 8
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons, lats = mirror_grid_faces(lon1, lat1)
    assert lons.shape == (6, im + 1, im + 1)
    assert lats.shape == (6, im + 1, im + 1)


def test_mirror_grid_face1_unchanged():
    """Face 1 must equal input."""
    im = 8
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons, lats = mirror_grid_faces(lon1, lat1)
    assert jnp.allclose(lons[0], lon1, atol=1e-14)
    assert jnp.allclose(lats[0], lat1, atol=1e-14)


def test_mirror_grid_all_unit_sphere():
    """All face corners on the unit sphere."""
    im = 6
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons, lats = mirror_grid_faces(lon1, lat1)
    x, y, z = latlon2xyz(lons, lats)
    norm = jnp.sqrt(x * x + y * y + z * z)
    assert jnp.allclose(norm, 1.0, atol=1e-12), (
        f"max |norm - 1| = {float(jnp.max(jnp.abs(norm - 1.0)))}"
    )


def test_mirror_grid_face2_center_geographic():
    """Face 2 is face 1 rotated by -90° about z.  Face-1 center is
    at (lon=0, lat=0); face-2 center should be at (lon=π/2, lat=0).

    rot_z(-90°) maps (cos λ, sin λ, 0) → (cos λ' , sin λ', 0) with
    λ' = λ + π/2 (FV3 sign convention).
    """
    im = 8
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons, _ = mirror_grid_faces(lon1, lat1)
    c = im // 2
    face1_center_lon = float(lon1[c, c])
    face2_center_lon = float(lons[1, c, c])
    diff = face2_center_lon - face1_center_lon
    # Wrap diff to [-π, π]
    diff = diff - 2 * jnp.pi * jnp.round(diff / (2 * jnp.pi))
    assert abs(abs(diff) - jnp.pi / 2) < 1e-6, (
        f"face2 - face1 lon shift = {diff} (expected ±π/2)"
    )


def test_mirror_grid_finite_everywhere():
    """No NaN/Inf in any of the 6 faces."""
    im = 12
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons, lats = mirror_grid_faces(lon1, lat1)
    assert jnp.all(jnp.isfinite(lons))
    assert jnp.all(jnp.isfinite(lats))


def test_mirror_grid_face_orthogonality():
    """Faces 3 (top) and 6 (bottom) should have opposite z signs at
    their centers (top maps near +z, bottom near -z)."""
    im = 8
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons, lats = mirror_grid_faces(lon1, lat1)
    c = im // 2
    x3, y3, z3 = latlon2xyz(lons[2, c, c], lats[2, c, c])
    x6, y6, z6 = latlon2xyz(lons[5, c, c], lats[5, c, c])
    assert float(z3) * float(z6) < 0.0, (
        f"face 3 z={float(z3)}, face 6 z={float(z6)} — should have opp signs"
    )
