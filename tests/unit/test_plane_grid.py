"""Unit tests for ``src/legoesm/grids/plane.py``.

Covers construction, input validation, the three Coriolis modes, area
totals, GridProtocol surface, and a JIT round-trip to confirm the
pytree static/dynamic split is correct.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.plane import PlaneGrid, create_plane_grid


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


def test_create_plane_grid_basic_shapes_and_metrics():
    g = create_plane_grid(
        nx=8, ny=6, nlev=10, dx=1.0e3, dy=2.0e3,
        lat0=15.0, lon0=120.0,
    )
    assert isinstance(g, PlaneGrid)
    assert (g.nx, g.ny, g.nlev) == (8, 6, 10)
    assert g.Lx == 8.0e3
    assert g.Ly == 12.0e3
    assert g.xc.shape == (8,)
    assert g.yc.shape == (6,)
    assert g.xu.shape == (8,)
    assert g.yv.shape == (6,)
    assert g.area_T.shape == (6, 8)
    assert g.f_y.shape == (6, 8)
    assert g.surface_mask.shape == (6, 8)
    assert jnp.allclose(g.area_T, 1.0e3 * 2.0e3)
    assert float(g.total_area) == pytest.approx(g.Lx * g.Ly)
    assert jnp.all(g.surface_mask == 1.0)
    assert jnp.allclose(g.xc, (jnp.arange(8) + 0.5) * 1.0e3)
    assert jnp.allclose(g.yc, (jnp.arange(6) + 0.5) * 2.0e3)
    assert jnp.allclose(g.xu, jnp.arange(8) * 1.0e3)
    assert jnp.allclose(g.yv, jnp.arange(6) * 2.0e3)


def test_create_plane_grid_default_coriolis_is_zero():
    g = create_plane_grid(nx=8, ny=8, nlev=4, dx=1.0, dy=1.0)
    assert g.f0 == 0.0
    assert g.beta == 0.0
    assert jnp.all(g.f_y == 0.0)


# --------------------------------------------------------------------- #
# Coriolis modes                                                        #
# --------------------------------------------------------------------- #


def test_coriolis_f_plane_is_constant():
    g = create_plane_grid(
        nx=4, ny=4, nlev=2, dx=1.0e3, dy=1.0e3,
        coriolis_mode="f_plane", f0=1.0e-4,
    )
    # Compare at the grid's native storage dtype: ``f_y`` follows the active
    # precision policy (float32 by default, float64 under the fp64 policy), so a
    # hardcoded ``jnp.float32(1e-4)`` reference spuriously mismatches a float64
    # ``f_y`` (``float64(1e-4) != float32(1e-4)``).
    assert jnp.all(g.f_y == jnp.asarray(1.0e-4, dtype=g.f_y.dtype))


def test_coriolis_beta_plane_linear_in_y():
    f0 = 1.0e-4
    beta = 2.0e-11
    g = create_plane_grid(
        nx=4, ny=8, nlev=2, dx=1.0e3, dy=1.0e3,
        coriolis_mode="beta_plane", f0=f0, beta=beta,
    )
    # Reference: f(y) = f0 + beta * (yc - Ly/2). Same in every column.
    expected_1d = f0 + beta * (g.yc - 0.5 * g.Ly)
    assert jnp.allclose(g.f_y, expected_1d[:, None], atol=1.0e-12)
    # Equal value across every column.
    assert jnp.all(g.f_y[:, 0:1] == g.f_y)


# --------------------------------------------------------------------- #
# Input validation                                                      #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"nx": 3, "ny": 4, "nlev": 2, "dx": 1.0, "dy": 1.0}, "nx must be >= 4"),
        ({"nx": 4, "ny": 3, "nlev": 2, "dx": 1.0, "dy": 1.0}, "ny must be >= 4"),
        ({"nx": 4, "ny": 4, "nlev": 1, "dx": 1.0, "dy": 1.0}, "nlev must be >= 2"),
        ({"nx": 4, "ny": 4, "nlev": 2, "dx": 0.0, "dy": 1.0}, "dx must be positive"),
        ({"nx": 4, "ny": 4, "nlev": 2, "dx": 1.0, "dy": -1.0}, "dy must be positive"),
        ({"nx": 4, "ny": 4, "nlev": 2, "dx": 1.0, "dy": 1.0, "lat0": 200.0},
         "lat0 must lie in"),
        ({"nx": 4, "ny": 4, "nlev": 2, "dx": 1.0, "dy": 1.0,
          "coriolis_mode": "not_a_mode"},
         "coriolis_mode must be one of"),
        ({"nx": 4, "ny": 4, "nlev": 2, "dx": 1.0, "dy": 1.0,
          "coriolis_mode": "beta_plane", "f0": 1.0e-4, "beta": 0.0},
         "requires nonzero beta"),
    ],
)
def test_create_plane_grid_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_plane_grid(**kwargs)


# --------------------------------------------------------------------- #
# GridProtocol surface                                                  #
# --------------------------------------------------------------------- #


def test_grid_protocol_properties():
    g = create_plane_grid(
        nx=8, ny=4, nlev=3, dx=10.0, dy=20.0,
        lat0=30.0, lon0=-90.0,
    )
    assert g.grid_n_columns == 32
    assert g.grid_shape_2d == (4, 8)
    assert g.grid_area.shape == (4, 8)
    assert float(g.grid_total_area) == pytest.approx(8 * 10.0 * 4 * 20.0)
    lat_field = g.grid_lat
    lon_field = g.grid_lon
    assert lat_field.shape == (4, 8)
    assert lon_field.shape == (4, 8)
    assert jnp.allclose(lat_field, jnp.deg2rad(30.0))
    assert jnp.allclose(lon_field, jnp.deg2rad(-90.0))
    # grid_radius is documented metadata only — it should equal the
    # canonical Earth radius from ``constants.R_earth`` so that
    # downstream protocol consumers see a consistent tangent-plane
    # placeholder. Hardcoded thresholds here would drift if the
    # constants module ever updated R_earth.
    assert float(g.grid_radius) == pytest.approx(float(constants.R_earth))


def test_columns_round_trip():
    g = create_plane_grid(nx=5, ny=4, nlev=2, dx=1.0, dy=1.0)
    field = jnp.arange(g.ny * g.nx * 3, dtype=jnp.float32).reshape(g.ny, g.nx, 3)
    cols = g.to_columns(field)
    assert cols.shape == (g.ny * g.nx, 3)
    assert jnp.array_equal(g.from_columns(cols), field)


def test_to_columns_rejects_wrong_shape():
    g = create_plane_grid(nx=5, ny=4, nlev=2, dx=1.0, dy=1.0)
    # Transposed (nx, ny) input — last axes wrong.
    bad = jnp.zeros((g.nx, g.ny, 3))
    with pytest.raises(ValueError, match="expects shape"):
        g.to_columns(bad)
    # 1D input.
    with pytest.raises(ValueError, match="ndim >= 2"):
        g.to_columns(jnp.zeros((g.ny,)))


def test_from_columns_rejects_wrong_leading_axis():
    g = create_plane_grid(nx=5, ny=4, nlev=2, dx=1.0, dy=1.0)
    bad = jnp.zeros((g.ny * g.nx + 1, 3))
    with pytest.raises(ValueError, match="expects leading axis"):
        g.from_columns(bad)


# --------------------------------------------------------------------- #
# Compatibility aliases                                                 #
# --------------------------------------------------------------------- #


def test_area_alias_matches_area_T():
    g = create_plane_grid(nx=5, ny=4, nlev=2, dx=2.5, dy=3.0)
    assert jnp.array_equal(g.area, g.area_T)


def test_dx_face_dy_face_uniform_2d():
    g = create_plane_grid(nx=5, ny=4, nlev=2, dx=2.5, dy=3.0)
    assert g.dx_face.shape == (g.ny, g.nx)
    assert g.dy_face.shape == (g.ny, g.nx)
    assert jnp.all(g.dx_face == 2.5)
    assert jnp.all(g.dy_face == 3.0)


def test_land_mask_is_complement_of_surface_mask():
    g = create_plane_grid(nx=5, ny=4, nlev=2, dx=1.0, dy=1.0)
    assert g.land_mask.shape == (g.ny, g.nx)
    assert jnp.all(g.land_mask == 0.0)
    assert jnp.array_equal(g.land_mask + g.surface_mask,
                            jnp.ones((g.ny, g.nx), dtype=g.surface_mask.dtype))


# --------------------------------------------------------------------- #
# Coriolis-mode decoding                                                #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mode, code",
    [("none", 0), ("f_plane", 1), ("beta_plane", 2)],
)
def test_coriolis_mode_round_trip(mode, code):
    kwargs = dict(nx=4, ny=4, nlev=2, dx=1.0, dy=1.0, coriolis_mode=mode)
    if mode == "beta_plane":
        kwargs.update(f0=1.0e-4, beta=1.0e-11)
    g = create_plane_grid(**kwargs)
    assert g.coriolis_mode_code == code
    assert g.coriolis_mode == mode


# --------------------------------------------------------------------- #
# JIT / pytree smoke                                                    #
# --------------------------------------------------------------------- #


def test_grid_is_jit_argument():
    """The grid is a valid pytree leaf set; JIT should accept it without
    flattening errors and Coriolis sum should round-trip."""

    @jax.jit
    def coriolis_sum(g: PlaneGrid) -> jax.Array:
        return g.f_y.sum()

    g = create_plane_grid(
        nx=8, ny=8, nlev=2, dx=1.0e3, dy=1.0e3,
        coriolis_mode="f_plane", f0=1.0e-4,
    )
    out = coriolis_sum(g)
    assert float(out) == pytest.approx(64 * 1.0e-4)
