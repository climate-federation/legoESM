"""FV3_3D iter 612: mirror_xyz / mirror_latlon / intp_great_circle / slerp.

Faithful ports of FV3 fv_grid_utils.F90 helpers built on iter-611
Cartesian primitives:

- ``mirror_xyz``        (F90:1668)
- ``mirror_latlon``     (F90:1705)
- ``intp_great_circle`` (F90:1896)
- ``slerp`` / ``spherical_linear_interpolation`` (F90:1927)

Tests
-----

1. ``test_mirror_xyz_preserves_unit_norm``.
2. ``test_mirror_xyz_involutive``.
3. ``test_mirror_xyz_fixed_on_plane``.
4. ``test_mirror_latlon_matches_xyz``.
5. ``test_intp_great_circle_endpoints``.
6. ``test_intp_great_circle_midpoint_matches_mid_pt_sphere``.
7. ``test_slerp_endpoints``.
8. ``test_slerp_arc_length_uniform``.
9. ``test_slerp_colocated_safe``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    great_circle_distance,
    intp_great_circle,
    latlon2xyz,
    mid_pt_sphere,
    mirror_latlon,
    mirror_xyz,
    slerp,
    xyz2latlon,
)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(jnp.asarray(lon), jnp.asarray(lat))
    return jnp.stack([x, y, z], axis=-1)


def test_mirror_xyz_preserves_unit_norm():
    """Mirror of unit vector is unit vector."""
    rng = np.random.default_rng(seed=612)
    p1 = _xyz(float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1)))
    p2 = _xyz(float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1)))
    p0 = _xyz(float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1)))
    p3 = mirror_xyz(p1, p2, p0)
    assert abs(float(jnp.linalg.norm(p3)) - 1.0) < 1e-12


def test_mirror_xyz_involutive():
    """Mirror twice across same plane is identity."""
    rng = np.random.default_rng(seed=613)
    p1 = _xyz(float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1)))
    p2 = _xyz(float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1)))
    p0 = _xyz(float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1)))
    p3 = mirror_xyz(p1, p2, p0)
    p4 = mirror_xyz(p1, p2, p3)
    assert jnp.allclose(p4, p0, atol=1e-12)


def test_mirror_xyz_fixed_on_plane():
    """Point on mirror plane maps to itself."""
    p1 = _xyz(0.0, 0.0)             # equator longitude 0
    p2 = _xyz(jnp.pi / 2, 0.0)      # equator longitude 90
    # mid point lies on the great circle through (p1, p2) — fixed
    pm = _xyz(jnp.pi / 4, 0.0)
    pm_mirror = mirror_xyz(p1, p2, pm)
    assert jnp.allclose(pm_mirror, pm, atol=1e-12)


def test_mirror_latlon_matches_xyz():
    """mirror_latlon should produce same result as mirror_xyz roundtripped."""
    rng = np.random.default_rng(seed=614)
    lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
    lat1 = float(rng.uniform(-1.0, 1.0))
    lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
    lat2 = float(rng.uniform(-1.0, 1.0))
    lon0 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
    lat0 = float(rng.uniform(-1.0, 1.0))
    lon3, lat3 = mirror_latlon(
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
        jnp.asarray(lon0), jnp.asarray(lat0),
    )
    # Compare against direct xyz path
    p1 = _xyz(lon1, lat1)
    p2 = _xyz(lon2, lat2)
    p0 = _xyz(lon0, lat0)
    p3 = mirror_xyz(p1, p2, p0)
    lon3_xyz, lat3_xyz = xyz2latlon(p3[..., 0], p3[..., 1], p3[..., 2])
    assert abs(float(lon3) - float(lon3_xyz)) < 1e-12
    assert abs(float(lat3) - float(lat3_xyz)) < 1e-12


def test_intp_great_circle_endpoints():
    """beta=0 → p1, beta=1 → p2."""
    lon1, lat1 = 0.3, 0.5
    lon2, lat2 = 1.7, -0.2
    lon_a, lat_a = intp_great_circle(
        jnp.asarray(0.0),
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
    )
    assert abs(float(lon_a) - lon1) < 1e-12
    assert abs(float(lat_a) - lat1) < 1e-12
    lon_b, lat_b = intp_great_circle(
        jnp.asarray(1.0),
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
    )
    assert abs(float(lon_b) - lon2) < 1e-12
    assert abs(float(lat_b) - lat2) < 1e-12


def test_intp_great_circle_midpoint_matches_mid_pt_sphere():
    """β=0.5 must agree with mid_pt_sphere."""
    rng = np.random.default_rng(seed=615)
    for _ in range(5):
        lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat2 = float(rng.uniform(-1.0, 1.0))
        lon_intp, lat_intp = intp_great_circle(
            jnp.asarray(0.5),
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        lon_mid, lat_mid = mid_pt_sphere(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        assert abs(float(lon_intp) - float(lon_mid)) < 1e-12
        assert abs(float(lat_intp) - float(lat_mid)) < 1e-12


def test_slerp_endpoints():
    """β=0 → p1, β=1 → p2."""
    lon1, lat1 = 0.1, 0.2
    lon2, lat2 = 1.4, -0.3
    lon_a, lat_a = slerp(
        jnp.asarray(0.0),
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
    )
    assert abs(float(lon_a) - lon1) < 1e-10
    assert abs(float(lat_a) - lat1) < 1e-10
    lon_b, lat_b = slerp(
        jnp.asarray(1.0),
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
    )
    assert abs(float(lon_b) - lon2) < 1e-10
    assert abs(float(lat_b) - lat2) < 1e-10


def test_slerp_arc_length_uniform():
    """slerp at β=0.5 puts the point equidistant via arc length."""
    rng = np.random.default_rng(seed=616)
    for _ in range(5):
        lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat2 = float(rng.uniform(-1.0, 1.0))
        lon_b, lat_b = slerp(
            jnp.asarray(0.5),
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        d1 = float(great_circle_distance(
            jnp.asarray(lon1), jnp.asarray(lat1),
            lon_b, lat_b,
        ))
        d2 = float(great_circle_distance(
            jnp.asarray(lon2), jnp.asarray(lat2),
            lon_b, lat_b,
        ))
        d12 = float(great_circle_distance(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        ))
        # Arc-length-uniform: d1 = d2 = d12 / 2
        assert abs(d1 - d2) / max(d12, 1.0) < 1e-10
        assert abs(d1 + d2 - d12) / max(d12, 1.0) < 1e-10


def test_slerp_colocated_safe():
    """slerp called on colocated points must not produce NaN."""
    lon_b, lat_b = slerp(
        jnp.asarray(0.5),
        jnp.asarray(0.3), jnp.asarray(0.5),
        jnp.asarray(0.3), jnp.asarray(0.5),
    )
    assert jnp.all(jnp.isfinite(lon_b))
    assert jnp.all(jnp.isfinite(lat_b))
    assert abs(float(lon_b) - 0.3) < 1e-12
    assert abs(float(lat_b) - 0.5) < 1e-12
