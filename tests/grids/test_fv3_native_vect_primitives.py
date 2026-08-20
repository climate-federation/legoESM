"""Vector primitives for the DCMIP16_BC wind assembly.

fv_grid_utils.F90: vect_cross :1781, inner_prod :984, normalize_vect :1880,
get_unit_vect2 :1848, get_latlon_vector :3326.

These are pure geometry, so they admit exact analytic checks -- which is the
only kind of check available before the full IC exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_native_metrics import (
    get_latlon_vector,
    get_unit_vect2,
    inner_prod,
    latlon2xyz,
    normalize_vect,
    vect_cross,
)


def test_vect_cross_matches_the_right_hand_rule():
    x, y, z = np.eye(3)
    np.testing.assert_array_equal(vect_cross(x, y), z)
    np.testing.assert_array_equal(vect_cross(y, z), x)
    np.testing.assert_array_equal(vect_cross(z, x), y)
    # antisymmetry
    np.testing.assert_array_equal(vect_cross(y, x), -z)


def test_normalize_vect_returns_unit_length():
    for v in ([3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [-2.0, 0.5, 7.0]):
        e = normalize_vect(np.array(v))
        assert np.linalg.norm(e) == pytest.approx(1.0, rel=1e-15)


def test_inner_prod_is_the_dot_product():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-4.0, 5.0, 6.0])
    assert inner_prod(a, b) == pytest.approx(np.dot(a, b), rel=1e-15)
    # orthogonal unit vectors give exactly 0
    assert inner_prod(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == 0.0


def test_get_latlon_vector_is_an_orthonormal_east_north_pair():
    """elon and elat must be unit, mutually orthogonal, and both tangent to
    the sphere at pp (i.e. orthogonal to the position vector)."""
    for lon, lat in [(0.0, 0.0), (0.7, 0.3), (-1.2, -0.9), (3.0, 1.1)]:
        pp = np.array([lon, lat])
        elon, elat = get_latlon_vector(pp)
        r = latlon2xyz(pp)
        assert np.linalg.norm(elon) == pytest.approx(1.0, rel=1e-15)
        assert np.linalg.norm(elat) == pytest.approx(1.0, rel=1e-15)
        assert np.dot(elon, elat) == pytest.approx(0.0, abs=1e-15)
        assert np.dot(elon, r) == pytest.approx(0.0, abs=1e-15)
        assert np.dot(elat, r) == pytest.approx(0.0, abs=1e-15)


def test_get_latlon_vector_is_the_right_hand_convention():
    """:3336 uses elat(3) = +cos(lat); the left-hand variant at :3338 is
    commented out. At the equator/prime meridian north must be +z."""
    elon, elat = get_latlon_vector(np.array([0.0, 0.0]))
    np.testing.assert_allclose(elon, [0.0, 1.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(elat, [0.0, 0.0, 1.0], atol=1e-15)


def test_get_unit_vect2_points_from_e1_toward_e2():
    """Along the equator the tangent from lon0 to lon1 (lon1 > lon0) must be
    due EAST, i.e. parallel to elon at the midpoint.

    This is the test that catches the p2 x p1 operand order at :1859 --
    swapping it gives exactly -uc, which is still a unit tangent and still
    passes every norm check.
    """
    e1 = np.array([0.0, 0.0])
    e2 = np.array([0.2, 0.0])
    uc = get_unit_vect2(e1, e2)
    mid = np.array([0.1, 0.0])
    elon, _ = get_latlon_vector(mid)
    assert inner_prod(uc, elon) == pytest.approx(1.0, rel=1e-12)


def test_get_unit_vect2_reverses_when_the_endpoints_swap():
    e1 = np.array([0.0, 0.3])
    e2 = np.array([0.4, 0.3])
    np.testing.assert_allclose(get_unit_vect2(e1, e2),
                               -get_unit_vect2(e2, e1), atol=1e-14)


def test_get_unit_vect2_is_unit_and_tangent():
    e1 = np.array([0.1, 0.2])
    e2 = np.array([0.3, 0.5])
    uc = get_unit_vect2(e1, e2)
    assert np.linalg.norm(uc) == pytest.approx(1.0, rel=1e-15)
    # tangent at the midpoint => orthogonal to the midpoint position vector
    from legoesm.grids.fv3_native_metrics import mid_pt_sphere
    r = latlon2xyz(mid_pt_sphere(e1, e2))
    assert inner_prod(uc, r) == pytest.approx(0.0, abs=1e-14)


def test_a_meridional_edge_projects_to_zero_zonal_component():
    """A due-north edge has no east component, so e . elon = 0.

    This is the geometry behind the IC: v-points project the ZONAL wind onto
    the +j tangent, so on a due-north edge the stored v is 0 -- and on a
    cubed sphere most v-edges are NOT due north, which is why the oracle's
    V max (20.347) exceeds its U max (20.141) at t=0.
    """
    e1 = np.array([0.5, 0.1])
    e2 = np.array([0.5, 0.4])          # same longitude => due north
    uc = get_unit_vect2(e1, e2)
    from legoesm.grids.fv3_native_metrics import mid_pt_sphere
    elon, _ = get_latlon_vector(mid_pt_sphere(e1, e2))
    assert inner_prod(uc, elon) == pytest.approx(0.0, abs=1e-14)
