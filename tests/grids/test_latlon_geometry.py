"""Unit tests for ``LatLonCGridGeometry`` + ``ensure_geometry``.

Addresses PR #268 slopbuster WARN #8: ``ensure_geometry`` had no direct
test coverage, and the field-by-field parity between ``LatLonGrid`` and
``LatLonCGridGeometry`` was relied on but never asserted.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.latlon import (
    LatLonCGridGeometry,
    create_latlon_geometry,
    create_latlon_grid,
    ensure_geometry,
)


@pytest.fixture
def regular_grid_pair():
    n_lat, n_lon = 16, 32
    return create_latlon_grid(n_lat, n_lon), create_latlon_geometry(n_lat, n_lon)


class TestEnsureGeometry:
    def test_passthrough_for_geometry(self):
        geom = create_latlon_geometry(8, 16)
        out = ensure_geometry(geom)
        assert out is geom

    def test_converts_latlongrid(self):
        grid = create_latlon_grid(8, 16)
        out = ensure_geometry(grid)
        assert isinstance(out, LatLonCGridGeometry)
        assert out.n_lat == grid.n_lat
        assert out.n_lon == grid.n_lon
        assert float(out.radius) == pytest.approx(grid.radius)

    def test_default_omega_matches_constants(self):
        grid = create_latlon_grid(8, 16)
        out = ensure_geometry(grid)
        out_explicit = ensure_geometry(grid, omega=constants.Omega)
        assert jnp.allclose(out.f_T, out_explicit.f_T)


class TestLatLonGridParity:
    """LatLonGrid ↔ LatLonCGridGeometry field-by-field agreement.

    Operators duck-type between both; if the analytical 2D metric arrays
    in ``LatLonCGridGeometry`` ever drifted from the legacy 1D
    ``LatLonGrid`` fields, the dispatch would silently pick the wrong
    branch.  These tests pin the parity.
    """

    def test_scalar_fields(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert geom.n_lat == grid.n_lat
        assert geom.n_lon == grid.n_lon
        assert float(geom.dlon) == pytest.approx(float(grid.dlon))
        assert float(geom.dlat) == pytest.approx(float(grid.dlat))
        assert float(geom.radius) == pytest.approx(float(grid.radius))

    def test_lat_lon_arrays(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.lat, grid.lat, atol=1e-12)
        assert jnp.allclose(geom.lon, grid.lon, atol=1e-12)
        assert jnp.allclose(geom.cos_lat, grid.cos_lat, atol=1e-12)
        assert jnp.allclose(geom.sin_lat, grid.sin_lat, atol=1e-12)

    def test_coriolis(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.f, grid.f, atol=1e-12)
        assert jnp.allclose(geom.f_T, grid.f, atol=1e-12)

    def test_area_parity(self, regular_grid_pair):
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.area, grid.area, atol=1e-8)
        assert float(geom.total_area) == pytest.approx(
            float(grid.total_area), rel=1e-10,
        )

    def test_dx_dy_two_cell_parity(self, regular_grid_pair):
        """LatLonGrid.dx/dy are 2-cell spans; LatLonCGridGeometry exposes
        them via convenience properties that double the single-cell dx_T/dy_T."""
        grid, geom = regular_grid_pair
        assert jnp.allclose(geom.dx, grid.dx, atol=1e-8)
        assert jnp.allclose(geom.dy, grid.dy, atol=1e-8)

    def test_dy_v_constant_for_regular_grid(self, regular_grid_pair):
        _, geom = regular_grid_pair
        ref = float(geom.dy_v[0, 0])
        assert jnp.allclose(geom.dy_v, ref, atol=1e-6)


class TestFoldInactiveOnRegularGeometry:
    def test_fold_is_inactive(self):
        geom = create_latlon_geometry(8, 16)
        assert geom.fold.is_active is False

    def test_perm_T_length(self):
        geom = create_latlon_geometry(8, 16)
        assert geom.fold.perm_T.shape == (16,)


# --------------------------------------------------------------------------
# #1455: the vertex-Coriolis PLACEMENT option.
#
# legoESM built f at the v-point as the average of the two adjacent tracer
# rows; NEMO (and any C-grid model that defines its Coriolis at the F-point)
# evaluates it AT the v-face latitude.  These tests pin the default as
# bit-identical, pin the new branch against values computed by hand, and pin
# the ONE case where the two conventions are indistinguishable -- because that
# case bounds what the option can ever be measured to do.
# --------------------------------------------------------------------------
class TestCoriolisPlacement:
    def test_default_is_the_cell_average_and_is_bit_identical(self):
        """The default must reproduce the old construction EXACTLY, or every
        recorded number on every card moves."""
        geom = create_latlon_geometry(16, 32)
        f_T = geom.f_T
        want_int = 0.5 * (f_T[:-1] + f_T[1:])
        assert jnp.array_equal(geom.f_v[1:-1], want_int)
        assert jnp.array_equal(geom.f_v[0], f_T[0])
        assert jnp.array_equal(geom.f_v[-1], f_T[-1])
        # and the explicit spelling is the same object-for-object answer
        same = create_latlon_geometry(16, 32,
                                      coriolis_placement="cell_average")
        assert jnp.array_equal(same.f_v, geom.f_v)

    def test_face_latitude_is_hand_computable_at_a_known_latitude(self):
        """f at 30 degrees is exactly Omega, because sin(30) = 1/2.

        The rotation rate is chosen as 1e-4 so the expected value is a number
        a reader can check without running anything: f_v at the 30-degree face
        must be 2 * 1e-4 * 0.5 = 1e-4 exactly.
        """
        # Faces placed so that one of them sits EXACTLY on 30 degrees north.
        lat_face = jnp.deg2rad(jnp.array([-30.0, 0.0, 30.0, 60.0]))
        lat_c = 0.5 * (lat_face[:-1] + lat_face[1:])
        geom = create_latlon_geometry(
            3, 4, omega=1.0e-4, dtype=jnp.float64,
            lat_1d=lat_c, lat_face_1d=lat_face,
            coriolis_placement="face_latitude")
        got = jnp.asarray(geom.f_v)[:, 0]
        assert float(got[2]) == pytest.approx(1.0e-4, rel=1e-12)   # +30 deg
        assert float(got[1]) == pytest.approx(0.0, abs=1e-20)      # equator
        assert float(got[0]) == pytest.approx(-1.0e-4, rel=1e-12)  # -30 deg
        # 60 degrees: 2*1e-4*sin(60) = 1e-4*sqrt(3)
        assert float(got[3]) == pytest.approx(1.0e-4 * 3.0 ** 0.5, rel=1e-12)

    def test_the_cell_average_at_the_SAME_faces_is_measurably_different(self):
        """The control for the test above: the default convention must NOT
        return the hand-computed values, or that test proves nothing."""
        lat_face = jnp.deg2rad(jnp.array([-30.0, 0.0, 30.0, 60.0]))
        lat_c = 0.5 * (lat_face[:-1] + lat_face[1:])
        kw = dict(omega=1.0e-4, dtype=jnp.float64, lat_1d=lat_c,
                  lat_face_1d=lat_face)
        avg = create_latlon_geometry(3, 4, coriolis_placement="cell_average",
                                     **kw)
        face = create_latlon_geometry(3, 4,
                                      coriolis_placement="face_latitude", **kw)
        # the 30-degree face is INTERIOR here, so the average is a real average
        assert float(jnp.abs(avg.f_v[2, 0] - face.f_v[2, 0])) > 1e-6 * 1.0e-4

    def test_on_a_UNIFORM_grid_the_two_differ_by_exactly_cos_half_dphi(self):
        """The boundary of what this option can be measured to do.

        With constant latitude spacing the cell average is cos(dphi/2) times
        the face value -- a UNIFORM factor, indistinguishable from a change of
        rotation rate.  Only a stretched grid separates them, and that is the
        whole reason the DINO measurement could split its Coriolis gap into a
        constant part and a placement part.
        """
        n_lat, n_lon = 12, 4
        avg = create_latlon_geometry(n_lat, n_lon, dtype=jnp.float64)
        face = create_latlon_geometry(n_lat, n_lon, dtype=jnp.float64,
                                      coriolis_placement="face_latitude")
        dphi = float(avg.dlat)
        ratio = jnp.asarray(avg.f_v)[1:-1, 0] / jnp.asarray(face.f_v)[1:-1, 0]
        finite = jnp.abs(jnp.asarray(face.f_v)[1:-1, 0]) > 1e-12
        got = jnp.asarray(ratio)[finite]
        assert float(jnp.max(jnp.abs(got - jnp.cos(dphi / 2.0)))) < 1e-12

    def test_face_latitude_puts_the_true_wall_value_on_the_polar_rows(self):
        """The cell average CARRIES OVER the nearest tracer row at the poles;
        the face convention uses the wall latitude itself, where sin = +-1."""
        geom = create_latlon_geometry(12, 4, omega=1.0e-4, dtype=jnp.float64,
                                      coriolis_placement="face_latitude")
        assert float(geom.f_v[0, 0]) == pytest.approx(-2.0e-4, rel=1e-9)
        assert float(geom.f_v[-1, 0]) == pytest.approx(2.0e-4, rel=1e-9)
        avg = create_latlon_geometry(12, 4, omega=1.0e-4, dtype=jnp.float64)
        assert float(jnp.abs(avg.f_v[0, 0])) < float(jnp.abs(geom.f_v[0, 0]))

    def test_f_T_and_f_u_are_UNTOUCHED_by_the_option(self):
        """The option is about the v-point only: on a lat-lon grid the u-point
        shares the tracer row's latitude, so f_u is already f at its own point
        and must not move."""
        a = create_latlon_geometry(16, 32, dtype=jnp.float64)
        b = create_latlon_geometry(16, 32, dtype=jnp.float64,
                                   coriolis_placement="face_latitude")
        assert jnp.array_equal(a.f_T, b.f_T)
        assert jnp.array_equal(a.f_u, b.f_u)

    def test_unknown_placement_raises_rather_than_defaulting(self):
        with pytest.raises(ValueError, match="coriolis_placement"):
            create_latlon_geometry(8, 16, coriolis_placement="f_point")

    def test_ensure_geometry_forwards_the_placement(self):
        grid = create_latlon_grid(12, 4)
        geom = ensure_geometry(grid, coriolis_placement="face_latitude")
        direct = create_latlon_geometry(
            12, 4, omega=grid.omega, radius=grid.radius,
            lat_1d=grid.lat, lon_1d=grid.lon,
            lat_face_1d=getattr(grid, "lat_v", None),
            coriolis_placement="face_latitude")
        assert jnp.array_equal(geom.f_v, direct.f_v)

    def test_ensure_geometry_does_NOT_apply_it_to_a_prebuilt_geometry(self):
        """Documented footgun, pinned: a pre-built geometry passes through
        UNCHANGED, so the convention has to be chosen where it is built.  The
        ocean model refuses this case rather than running the default while
        its config says otherwise."""
        pre = create_latlon_geometry(12, 4)
        out = ensure_geometry(pre, coriolis_placement="face_latitude")
        assert out is pre
        assert jnp.array_equal(out.f_v, pre.f_v)
