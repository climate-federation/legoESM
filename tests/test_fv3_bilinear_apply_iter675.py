"""FV3_3D iter 675: bilinear_interp_apply port.

Faithful JAX port of FV3 ``apply_inc_on_3d_scalar`` core
(tools/fv_treat_da_inc.F90:339-360).  Apply iter-673 bilinear
remap weights to interpolate src field.

Tests
-----

1. ``test_apply_shape``.
2. ``test_apply_constant_field``.
3. ``test_apply_linear_field_exact``.
4. ``test_apply_3d_levels``.
5. ``test_apply_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    bilinear_interp_apply,
    remap_coef_fv3,
)


def _src_grid(im=16, jm=12):
    src_lon = jnp.linspace(0.0, 2 * jnp.pi - 2 * jnp.pi / im, im)
    src_lat = jnp.linspace(-1.3, 1.3, jm)
    return src_lon, src_lat


def test_apply_shape():
    """Output shape matches target."""
    src_lon, src_lat = _src_grid()
    target_lon = jnp.asarray([[0.5, 1.0], [1.5, 2.0]])
    target_lat = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    src_field = jnp.ones((16, 12))
    out = bilinear_interp_apply(src_field, id1, id2, jc, s2c)
    assert out.shape == (2, 2)


def test_apply_constant_field():
    """Constant src field → constant output (interp preserves constants)."""
    src_lon, src_lat = _src_grid()
    target_lon = jnp.asarray([0.5, 1.5, 2.0])
    target_lat = jnp.asarray([0.1, -0.3, 0.6])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    src_field = jnp.full((16, 12), 3.7)
    out = bilinear_interp_apply(src_field, id1, id2, jc, s2c)
    assert jnp.allclose(out, 3.7, atol=1e-12)


def test_apply_linear_field_exact():
    """Linear field f(lon, lat) = a·lon + b·lat: bilinear interp exact."""
    src_lon = jnp.linspace(0.0, 2 * jnp.pi - 0.3, 16)
    src_lat = jnp.linspace(-1.3, 1.3, 12)
    lon_2d, lat_2d = jnp.meshgrid(src_lon, src_lat, indexing="ij")
    a, b = 2.0, 3.0
    src_field = a * lon_2d + b * lat_2d
    target_lon = jnp.asarray([1.0, 2.5, 0.5, 4.0])
    target_lat = jnp.asarray([0.1, -0.5, 0.8, -0.2])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    out = bilinear_interp_apply(src_field, id1, id2, jc, s2c)
    expected = a * target_lon + b * target_lat
    assert jnp.allclose(out, expected, atol=1e-12)


def test_apply_3d_levels():
    """3D src field (im, jm, km): output (..., km)."""
    src_lon, src_lat = _src_grid()
    target_lon = jnp.asarray([0.5, 1.5])
    target_lat = jnp.asarray([0.1, 0.3])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    rng = np.random.default_rng(seed=675)
    src_field = jnp.asarray(rng.normal(size=(16, 12, 5)))
    out = bilinear_interp_apply(src_field, id1, id2, jc, s2c)
    assert out.shape == (2, 5)
    assert jnp.all(jnp.isfinite(out))


def test_apply_finite():
    """No NaN/Inf on random inputs."""
    src_lon, src_lat = _src_grid()
    rng = np.random.default_rng(seed=676)
    target_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=20))
    target_lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=20))
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    src_field = jnp.asarray(rng.normal(size=(16, 12)))
    out = bilinear_interp_apply(src_field, id1, id2, jc, s2c)
    assert jnp.all(jnp.isfinite(out))
