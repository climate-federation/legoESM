"""FV3_3D iter 673: remap_coef_fv3 port.

Faithful JAX port of FV3 ``remap_coef`` (tools/fv_treat_da_inc.F90:
366-442).  Bilinear remap weights from regular lat-lon to
arbitrary target grid (typically cubed-sphere).

Tests
-----

1. ``test_remap_shape``.
2. ``test_remap_weights_sum_to_one``.
3. ``test_remap_weights_nonneg``.
4. ``test_remap_at_src_grid_point``.
5. ``test_remap_linear_interp_correct``.
6. ``test_remap_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import remap_coef_fv3


def _make_src_grid(im=12, jm=8):
    """Regular lat-lon grid."""
    src_lon = jnp.linspace(0.0, 2 * jnp.pi - 2 * jnp.pi / im, im)
    src_lat = jnp.linspace(-jnp.pi / 2 + 0.1, jnp.pi / 2 - 0.1, jm)
    return src_lon, src_lat


def test_remap_shape():
    """Outputs shape match target."""
    src_lon, src_lat = _make_src_grid()
    target_lon = jnp.asarray([[0.5, 1.0], [1.5, 2.0]])
    target_lat = jnp.asarray([[0.1, 0.2], [0.3, 0.4]])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    assert id1.shape == (2, 2)
    assert id2.shape == (2, 2)
    assert jc.shape == (2, 2)
    assert s2c.shape == (2, 2, 4)


def test_remap_weights_sum_to_one():
    """4 bilinear weights sum to 1."""
    src_lon, src_lat = _make_src_grid()
    rng = np.random.default_rng(seed=673)
    target_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=20))
    target_lat = jnp.asarray(rng.uniform(-1.4, 1.4, size=20))
    _, _, _, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    sums = jnp.sum(s2c, axis=-1)
    assert jnp.allclose(sums, 1.0, atol=1e-12), (
        f"weights sum: min={float(jnp.min(sums))}, max={float(jnp.max(sums))}"
    )


def test_remap_weights_nonneg():
    """All 4 bilinear weights ≥ 0 within source range."""
    src_lon, src_lat = _make_src_grid()
    rng = np.random.default_rng(seed=674)
    target_lon = jnp.asarray(rng.uniform(float(src_lon[0]) + 0.01, float(src_lon[-1]) - 0.01, size=20))
    target_lat = jnp.asarray(rng.uniform(float(src_lat[0]) + 0.01, float(src_lat[-1]) - 0.01, size=20))
    _, _, _, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    assert jnp.all(s2c >= -1e-12)


def test_remap_at_src_grid_point():
    """Target exactly at src[i1, jc]: weight[0] = 1, others = 0."""
    src_lon, src_lat = _make_src_grid()
    target_lon = jnp.asarray([float(src_lon[3])])
    target_lat = jnp.asarray([float(src_lat[2])])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    assert int(id1[0]) == 3
    assert int(jc[0]) == 2
    # Weight[0] = 1 (SW corner)
    assert abs(float(s2c[0, 0]) - 1.0) < 1e-12
    assert abs(float(s2c[0, 1])) < 1e-12
    assert abs(float(s2c[0, 2])) < 1e-12
    assert abs(float(s2c[0, 3])) < 1e-12


def test_remap_linear_interp_correct():
    """Bilinear interp of a linear field f(lon, lat) = a·lon + b·lat
    reproduces target value exactly."""
    src_lon = jnp.linspace(0.0, 2 * jnp.pi - 0.3, 16)
    src_lat = jnp.linspace(-1.3, 1.3, 12)
    # Source field f(lon, lat) = 2·lon + 3·lat
    lon_2d, lat_2d = jnp.meshgrid(src_lon, src_lat, indexing="ij")
    f_src = 2.0 * lon_2d + 3.0 * lat_2d
    # Target points
    target_lon = jnp.asarray([1.0, 2.5, 0.5])
    target_lat = jnp.asarray([0.1, -0.5, 0.8])
    id1, id2, jc, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    # Bilinear reconstruction
    f_target = (
        s2c[..., 0] * f_src[id1, jc]
        + s2c[..., 1] * f_src[id2, jc]
        + s2c[..., 2] * f_src[id2, jc + 1]
        + s2c[..., 3] * f_src[id1, jc + 1]
    )
    expected = 2.0 * target_lon + 3.0 * target_lat
    assert jnp.allclose(f_target, expected, atol=1e-12)


def test_remap_finite():
    """No NaN/Inf for valid inputs."""
    src_lon, src_lat = _make_src_grid()
    rng = np.random.default_rng(seed=675)
    target_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=30))
    target_lat = jnp.asarray(rng.uniform(-1.5, 1.5, size=30))
    _, _, _, s2c = remap_coef_fv3(target_lon, target_lat, src_lon, src_lat)
    assert jnp.all(jnp.isfinite(s2c))
