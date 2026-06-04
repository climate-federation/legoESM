"""FV3_3D iter 629: make_fv3_native_grid integration wrapper.

End-to-end FV3 native cubed-sphere grid construction:
  gnomonic_grids → mirror_grid_face1_symmetrize → mirror_grid_faces

Tests
-----

1. ``test_native_grid_shape``.
2. ``test_native_grid_unit_sphere``.
3. ``test_native_grid_finite``.
4. ``test_native_grid_equals_manual_pipeline``.
5. ``test_native_grid_symmetrize_flag_off``.
6. ``test_native_grid_all_grid_types``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    gnomonic_grids,
    latlon2xyz,
    make_fv3_native_grid,
    mirror_grid_face1_symmetrize,
    mirror_grid_faces,
)


def test_native_grid_shape():
    """Output shape (6, im+1, im+1) for each grid_type."""
    im = 8
    lons, lats = make_fv3_native_grid(im, grid_type=0)
    assert lons.shape == (6, im + 1, im + 1)
    assert lats.shape == (6, im + 1, im + 1)


def test_native_grid_unit_sphere():
    """All 6 faces on unit sphere."""
    im = 8
    lons, lats = make_fv3_native_grid(im, grid_type=0)
    x, y, z = latlon2xyz(lons, lats)
    norm = jnp.sqrt(x * x + y * y + z * z)
    assert jnp.allclose(norm, 1.0, atol=1e-12), (
        f"max |norm - 1| = {float(jnp.max(jnp.abs(norm - 1.0)))}"
    )


def test_native_grid_finite():
    """No NaN/Inf anywhere."""
    im = 12
    lons, lats = make_fv3_native_grid(im, grid_type=0)
    assert jnp.all(jnp.isfinite(lons))
    assert jnp.all(jnp.isfinite(lats))


def test_native_grid_equals_manual_pipeline():
    """Wrapper output equals manually-composed pipeline."""
    im = 8
    # Wrapper
    lons_w, lats_w = make_fv3_native_grid(im, grid_type=0)
    # Manual
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lon1, lat1 = mirror_grid_face1_symmetrize(lon1, lat1)
    lons_m, lats_m = mirror_grid_faces(lon1, lat1)
    assert jnp.allclose(lons_w, lons_m, atol=1e-14)
    assert jnp.allclose(lats_w, lats_m, atol=1e-14)


def test_native_grid_symmetrize_flag_off():
    """symmetrize_face1=False skips iter-625 step."""
    im = 8
    lons_off, lats_off = make_fv3_native_grid(
        im, grid_type=0, symmetrize_face1=False,
    )
    lon1, lat1 = gnomonic_grids(im, grid_type=0)
    lons_m, lats_m = mirror_grid_faces(lon1, lat1)
    assert jnp.allclose(lons_off, lons_m, atol=1e-14)
    assert jnp.allclose(lats_off, lats_m, atol=1e-14)


def test_native_grid_all_grid_types():
    """All 3 grid_types build finite 6-face grids."""
    im = 6
    for gt in (0, 1, 2):
        lons, lats = make_fv3_native_grid(im, grid_type=gt)
        assert lons.shape == (6, im + 1, im + 1), f"gt={gt}"
        assert jnp.all(jnp.isfinite(lons)), f"gt={gt}"
        assert jnp.all(jnp.isfinite(lats)), f"gt={gt}"
