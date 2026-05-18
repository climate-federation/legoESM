"""Unit tests for legoesm.ocean.fidelity.regrid_veros."""

from __future__ import annotations

import math

import numpy as np
import pytest

from legoesm.ocean.fidelity import regrid_veros, report_grid


def test_identity_regrid_preserves_field():
    src = report_grid.get_report_grid("2deg")
    rng = np.random.default_rng(0)
    field = rng.standard_normal((src.n_lat, src.n_lon))
    out = regrid_veros.regrid_2d_latlon(
        field,
        src.lat_edges,
        src.lon_edges,
        src.lat_edges,
        src.lon_edges,
    )
    np.testing.assert_allclose(out, field, rtol=1e-12, atol=1e-12)


def test_regrid_to_coarser_preserves_area_mean():
    src = report_grid.get_report_grid("1deg")
    dst = report_grid.get_report_grid("2deg")
    field = np.ones((src.n_lat, src.n_lon)) * 3.5
    out = regrid_veros.regrid_2d_latlon(
        field,
        src.lat_edges,
        src.lon_edges,
        dst.lat_edges,
        dst.lon_edges,
    )
    # Loose rtol because conservative_regrid casts weights through jnp
    # which truncates to float32 when JAX_ENABLE_X64 is unset (e.g. on
    # Metal). The full-precision path is exercised by the underlying
    # conservative_regrid unit tests.
    np.testing.assert_allclose(out, 3.5, rtol=1e-5)


def test_regrid_field_shape_mismatch_raises():
    src = report_grid.get_report_grid("2deg")
    with pytest.raises(ValueError):
        regrid_veros.regrid_2d_latlon(
            np.zeros((10, 10)),
            src.lat_edges,
            src.lon_edges,
            src.lat_edges,
            src.lon_edges,
        )


def test_regrid_field_must_be_2d():
    src = report_grid.get_report_grid("2deg")
    with pytest.raises(ValueError):
        regrid_veros.regrid_2d_latlon(
            np.zeros((src.n_lat, src.n_lon, 2)),
            src.lat_edges,
            src.lon_edges,
            src.lat_edges,
            src.lon_edges,
        )


def test_bgrid_velocity_to_cgrid_known_average():
    u_corner = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [5.0, 6.0, 7.0],
        ]
    )
    v_corner = u_corner.copy()
    u_face, v_face = regrid_veros.bgrid_velocity_to_cgrid(u_corner, v_corner)
    assert u_face.shape == (2, 3)
    assert v_face.shape == (3, 2)
    expected_u = np.array([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    expected_v = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
    np.testing.assert_allclose(u_face, expected_u)
    np.testing.assert_allclose(v_face, expected_v)


def test_bgrid_velocity_validates_shapes():
    with pytest.raises(ValueError):
        regrid_veros.bgrid_velocity_to_cgrid(np.zeros((3, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        regrid_veros.bgrid_velocity_to_cgrid(np.zeros((1, 3)), np.zeros((1, 3)))


def test_interp_to_target_z_recovers_linear_profile():
    z_src = np.linspace(0.0, 100.0, 11)
    horiz = (4, 5)
    profile = z_src[:, None, None] * 2.5
    field = np.broadcast_to(profile, (11, *horiz)).copy()
    z_dst = np.linspace(5.0, 95.0, 7)
    out = regrid_veros.interp_to_target_z(field, z_src, z_dst)
    expected = (z_dst[:, None, None] * 2.5) * np.ones((1, *horiz))
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_interp_to_target_z_flips_descending_source():
    z_src = np.linspace(100.0, 0.0, 11)
    field = z_src[:, None] * 2.0
    z_dst = np.array([25.0, 75.0])
    out = regrid_veros.interp_to_target_z(field, z_src, z_dst)
    np.testing.assert_allclose(out.squeeze(), z_dst * 2.0)


def test_interp_to_target_z_validates_inputs():
    with pytest.raises(ValueError):
        regrid_veros.interp_to_target_z(np.zeros((3,)), np.array([[0.0]]), np.array([0.0]))
    with pytest.raises(ValueError):
        regrid_veros.interp_to_target_z(np.zeros((3,)), np.array([0.0, 1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        regrid_veros.interp_to_target_z(
            np.zeros((3,)), np.array([0.0, 1.0, 1.0]), np.array([0.0])
        )


def test_report_grid_edges_from_centers_evenly_spaced():
    centers = np.array([1.0, 3.0, 5.0, 7.0])
    edges = regrid_veros.report_grid_edges_from_centers(centers)
    np.testing.assert_allclose(edges, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))


def test_report_grid_edges_validates():
    with pytest.raises(ValueError):
        regrid_veros.report_grid_edges_from_centers(np.array([1.0]))
    with pytest.raises(ValueError):
        regrid_veros.report_grid_edges_from_centers(np.array([[1.0, 2.0]]))
