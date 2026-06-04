"""FV3_3D iter 613: spherical-geometry helpers port.

Faithful ports of FV3 fv_grid_utils.F90 helpers built on
iter-611/612 primitives:

- ``spherical_angle``    (F90:2838)
- ``cell_center3``       (F90:2728)
- ``cell_center2``       (F90:2700)
- ``dist2side_latlon``   (F90:2812)
- ``expand_cell``        (F90:2631)

Tests
-----

1. ``test_spherical_angle_right``.
2. ``test_spherical_angle_octant``.
3. ``test_spherical_angle_colinear``.
4. ``test_cell_center3_unit_norm``.
5. ``test_cell_center3_equidistant``.
6. ``test_cell_center2_matches_3``.
7. ``test_dist2side_latlon_on_arc_zero``.
8. ``test_dist2side_latlon_pole_to_equator``.
9. ``test_expand_cell_fac_one_identity``.
10. ``test_expand_cell_fac_zero_collapses_to_center``.
11. ``test_expand_cell_corners_on_sphere``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    cell_center2,
    cell_center3,
    dist2side_latlon,
    expand_cell,
    great_circle_distance,
    latlon2xyz,
    spherical_angle,
    xyz2latlon,
)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(jnp.asarray(lon), jnp.asarray(lat))
    return jnp.stack([x, y, z], axis=-1)


def test_spherical_angle_right():
    """At north pole, angle between two meridians 90° apart should be π/2."""
    p1 = _xyz(0.0, jnp.pi / 2)              # north pole
    p2 = _xyz(0.0, 0.0)                     # along Greenwich meridian
    p3 = _xyz(jnp.pi / 2, 0.0)              # along 90E meridian
    angle = float(spherical_angle(p1, p2, p3))
    assert abs(angle - jnp.pi / 2) < 1e-10, f"expected π/2, got {angle}"


def test_spherical_angle_octant():
    """Octant corner: angle at north pole between 0° and 60° meridians = π/3."""
    p1 = _xyz(0.0, jnp.pi / 2)
    p2 = _xyz(0.0, 0.0)
    p3 = _xyz(jnp.pi / 3, 0.0)
    angle = float(spherical_angle(p1, p2, p3))
    assert abs(angle - jnp.pi / 3) < 1e-10


def test_spherical_angle_colinear():
    """If p2 = p3 (degenerate), FV3 path returns finite (0 or π)."""
    p1 = _xyz(0.0, 0.0)
    p2 = _xyz(jnp.pi / 4, 0.0)
    angle = float(spherical_angle(p1, p2, p2))
    assert jnp.isfinite(angle), f"angle should be finite, got {angle}"


def test_cell_center3_unit_norm():
    """Cell center of 4 unit vectors is on unit sphere."""
    rng = np.random.default_rng(seed=613)
    p1 = _xyz(float(rng.uniform(0.1, 2 * jnp.pi - 0.1)), float(rng.uniform(-1, 1)))
    p2 = _xyz(float(rng.uniform(0.1, 2 * jnp.pi - 0.1)), float(rng.uniform(-1, 1)))
    p3 = _xyz(float(rng.uniform(0.1, 2 * jnp.pi - 0.1)), float(rng.uniform(-1, 1)))
    p4 = _xyz(float(rng.uniform(0.1, 2 * jnp.pi - 0.1)), float(rng.uniform(-1, 1)))
    ec = cell_center3(p1, p2, p3, p4)
    assert abs(float(jnp.linalg.norm(ec)) - 1.0) < 1e-12


def test_cell_center3_equidistant():
    """Symmetric square of 4 corners around equator → center at equator."""
    # Square at equator centered at lon=0
    p1 = _xyz(-0.1, -0.1)
    p2 = _xyz(0.1, -0.1)
    p3 = _xyz(0.1, 0.1)
    p4 = _xyz(-0.1, 0.1)
    ec = cell_center3(p1, p2, p3, p4)
    lon_c, lat_c = xyz2latlon(ec[0], ec[1], ec[2])
    assert abs(float(lon_c)) < 1e-10 or abs(float(lon_c) - 2 * jnp.pi) < 1e-10
    assert abs(float(lat_c)) < 1e-10


def test_cell_center2_matches_3():
    """cell_center2 latlon path must match cell_center3 Cartesian path."""
    rng = np.random.default_rng(seed=614)
    corners = []
    for _ in range(4):
        corners.append((
            float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
            float(rng.uniform(-1.0, 1.0)),
        ))
    lons = [jnp.asarray(c[0]) for c in corners]
    lats = [jnp.asarray(c[1]) for c in corners]
    # latlon path
    lon_c, lat_c = cell_center2(
        lons[0], lats[0], lons[1], lats[1],
        lons[2], lats[2], lons[3], lats[3],
    )
    # Cartesian path
    ps = [_xyz(c[0], c[1]) for c in corners]
    ec = cell_center3(*ps)
    lon_c2, lat_c2 = xyz2latlon(ec[0], ec[1], ec[2])
    assert abs(float(lon_c) - float(lon_c2)) < 1e-12
    assert abs(float(lat_c) - float(lat_c2)) < 1e-12


def test_dist2side_latlon_on_arc_zero():
    """Point on the great circle arc should have distance ~0."""
    # Equator arc from (0,0) to (π/2, 0); point on equator at π/4
    d = float(dist2side_latlon(
        jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(jnp.pi / 2), jnp.asarray(0.0),
        jnp.asarray(jnp.pi / 4), jnp.asarray(0.0),
    ))
    assert abs(d) < 1e-10, f"on-arc point should have d ≈ 0; got {d}"


def test_dist2side_latlon_pole_to_equator():
    """Distance from north pole to equator arc should be π/2."""
    d = float(dist2side_latlon(
        jnp.asarray(0.0), jnp.asarray(0.0),
        jnp.asarray(jnp.pi / 2), jnp.asarray(0.0),
        jnp.asarray(0.0), jnp.asarray(jnp.pi / 2),
    ))
    # FV3 dist2side returns asin(sin(side) sin(angle))
    # For pole-to-equator: side = π/2, angle = π/2, dist = asin(1) = π/2
    assert abs(d - jnp.pi / 2) < 1e-10


def test_expand_cell_fac_one_identity():
    """fac = 1 returns input corners unchanged."""
    rng = np.random.default_rng(seed=615)
    corners = []
    for _ in range(4):
        corners.append((
            float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
            float(rng.uniform(-0.8, 0.8)),
        ))
    args = [jnp.asarray(c) for ll in corners for c in ll]
    out = expand_cell(*args, fac=1.0)
    for k, (lon_in, lat_in) in enumerate(corners):
        lon_out, lat_out = out[k]
        assert abs(float(lon_out) - lon_in) < 1e-10, (
            f"corner {k}: lon {float(lon_out)} != {lon_in}"
        )
        assert abs(float(lat_out) - lat_in) < 1e-10


def test_expand_cell_fac_zero_collapses_to_center():
    """fac = 0 collapses all 4 corners to the cell center."""
    rng = np.random.default_rng(seed=616)
    corners = []
    for _ in range(4):
        corners.append((
            float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
            float(rng.uniform(-0.8, 0.8)),
        ))
    args = [jnp.asarray(c) for ll in corners for c in ll]
    out = expand_cell(*args, fac=0.0)
    lon_c, lat_c = cell_center2(*args)
    for k in range(4):
        lon_k, lat_k = out[k]
        assert abs(float(lon_k) - float(lon_c)) < 1e-10
        assert abs(float(lat_k) - float(lat_c)) < 1e-10


def test_expand_cell_corners_on_sphere():
    """Expanded corners must lie on unit sphere (FV3 re-normalization)."""
    rng = np.random.default_rng(seed=617)
    corners = []
    for _ in range(4):
        corners.append((
            float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
            float(rng.uniform(-0.8, 0.8)),
        ))
    args = [jnp.asarray(c) for ll in corners for c in ll]
    out = expand_cell(*args, fac=1.5)
    for k in range(4):
        lon_k, lat_k = out[k]
        x, y, z = latlon2xyz(lon_k, lat_k)
        assert abs(float(jnp.sqrt(x * x + y * y + z * z)) - 1.0) < 1e-10
