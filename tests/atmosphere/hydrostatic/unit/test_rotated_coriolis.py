"""Rotated-planet Coriolis (DCMIP 2008 rotated JW tests) on every grid.

The rotated JW state is steady only if the planet's rotation axis is tilted
with the flow: f = 2 omega S, S = -cos(lon) cos(lat) sin(alpha) + sin(lat)
cos(alpha).  Checks: (1) one shared formula, alpha=0 exact; (2) each grid
builder's Coriolis equals it, and alpha=0 leaves the grid unchanged; (3) the
rotated JW initial state uses the SAME frame (sin(lat_r) == S) and its wind
blows along the rotated latitude circles (v . grad S == 0).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.williamson_sw_analytic import rotated_coriolis, rotated_sin_lat

ALPHA = math.pi / 4.0

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="fp64 tolerances: run with JAX_ENABLE_X64=1")


def test_alpha_zero_is_standard_coriolis():
    lat = np.linspace(-1.5, 1.5, 7)[:, None]
    lon = np.linspace(0.0, 6.0, 5)[None, :]
    np.testing.assert_array_equal(
        rotated_coriolis(lon, lat, omega=constants.Omega, alpha=0.0),
        2.0 * constants.Omega * np.sin(lat) * np.ones_like(lon))


def test_gaussian_grid_f():
    from legoesm.grids.gaussian import create_gaussian_grid
    g0 = create_gaussian_grid(10)
    g = create_gaussian_grid(10, coriolis_alpha=ALPHA)
    np.testing.assert_array_equal(np.asarray(create_gaussian_grid(10, coriolis_alpha=0.0).f),
                                  np.asarray(g0.f))
    np.testing.assert_allclose(
        np.asarray(g.f),
        rotated_coriolis(np.asarray(g.lon2d), np.asarray(g.lat2d),
                         omega=constants.Omega, alpha=ALPHA), rtol=0, atol=1e-18)


def test_voronoi_mesh_f():
    from legoesm.grids.voronoi import create_voronoi_mesh
    m0 = create_voronoi_mesh(2)
    m = create_voronoi_mesh(2, coriolis_alpha=ALPHA)
    np.testing.assert_array_equal(np.asarray(m.lonEdge), np.asarray(m0.lonEdge))
    for f, lon, lat in ((m.fEdge, m.lonEdge, m.latEdge),
                        (m.fVertex, m.lonVertex, m.latVertex)):
        np.testing.assert_allclose(
            np.asarray(f), rotated_coriolis(np.asarray(lon, np.float64),
                                            np.asarray(lat, np.float64),
                                            omega=constants.Omega, alpha=ALPHA),
            rtol=1e-6, atol=0)


def test_latlon_vertex_f_and_default_unchanged():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis, latlon_vertex_coriolis)
    g0 = create_latlon_grid(12, 24)
    g = create_latlon_grid(12, 24, coriolis_alpha=ALPHA)
    fv = np.asarray(latlon_vertex_coriolis(g))
    assert fv.shape == (13, 25)
    # poles: f = +-2 omega cos(alpha), independent of longitude
    np.testing.assert_allclose(fv[0], -2 * constants.Omega * math.cos(ALPHA), rtol=1e-6)
    np.testing.assert_allclose(fv[-1], 2 * constants.Omega * math.cos(ALPHA), rtol=1e-6)
    np.testing.assert_allclose(fv[:, -1], fv[:, 0], rtol=1e-6)  # periodic column
    # interior vertex (row 3, col 5) at (lon[5] - dlon/2, mid-latitude)
    lat_v = 0.5 * float(g.lat[2] + g.lat[3])
    lon_v = float(g.lon[5]) - 0.5 * float(g.dlon)
    assert fv[3, 5] == pytest.approx(
        rotated_coriolis(lon_v, lat_v, omega=constants.Omega, alpha=ALPHA), rel=1e-6)
    # alpha = 0: identical Coriolis tendency to the historical 1-D path
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.normal(size=(12, 25, 3)))
    v = jnp.asarray(rng.normal(size=(13, 24, 3)))
    a = absolute_vorticity_coriolis(u, v, g0)
    b = absolute_vorticity_coriolis(u, v, create_latlon_grid(12, 24, coriolis_alpha=0.0))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # rotated grid really changes the tendency
    c = absolute_vorticity_coriolis(u, v, g)
    assert float(jnp.max(jnp.abs(c[0] - a[0]))) > 1e-6


def test_rotated_jw_frame_matches_coriolis_frame():
    from tests.test_cases.dcmip2008.jablonowski_rotated import (
        _build_rotated_state_dense_grid, _rotate_to_rotated_frame)
    from legoesm.grids.vertical import create_sigma_coordinate
    lat, lon = np.meshgrid(np.linspace(-1.4, 1.4, 15), np.linspace(0, 6.2, 31),
                           indexing="ij")
    _, lat_r = _rotate_to_rotated_frame(jnp.asarray(lon), jnp.asarray(lat), ALPHA)
    np.testing.assert_allclose(np.sin(np.asarray(lat_r)),
                               rotated_sin_lat(lon, lat, alpha=ALPHA), atol=1e-13)
    u, v, _, _ = _build_rotated_state_dense_grid(
        jnp.asarray(lon), jnp.asarray(lat), create_sigma_coordinate(4), False, ALPHA)
    # grad S on the sphere (east, north) components, times a
    dS_dlon = np.cos(lat) * np.sin(lon) * math.sin(ALPHA) / np.cos(lat)
    dS_dlat = (np.cos(lon) * np.sin(lat) * math.sin(ALPHA) + np.cos(lat) * math.cos(ALPHA))
    for k in range(4):
        along = np.asarray(u[..., k]) * dS_dlon + np.asarray(v[..., k]) * dS_dlat
        speed = np.hypot(np.asarray(u[..., k]), np.asarray(v[..., k]))
        assert speed.max() > 1.0
        np.testing.assert_allclose(along, 0.0, atol=1e-9 * speed.max())


def test_latlon_traced_alpha_zero_bit_identical_and_rotated_vertex_exact():
    """Grid passed as a dynamic JIT argument (alpha becomes a tracer) takes the
    rotated formula, which at alpha = 0 equals the historical rows to within one
    rounding of the fp32-stored geometry (the rotated rows are evaluated in fp64
    and rounded once; the historical rows take sin in the grid dtype).  A Python
    0.0 tilt -- every existing grid -- never reaches that path.  The rotated
    field has an exact periodic seam and exact pole scalars."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis, latlon_vertex_coriolis)
    g0 = create_latlon_grid(12, 24)
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.normal(size=(12, 25, 3)))
    v = jnp.asarray(rng.normal(size=(13, 24, 3)))
    ref = absolute_vorticity_coriolis(u, v, g0)
    traced = jax.jit(absolute_vorticity_coriolis)(u, v, g0)
    eps = float(np.finfo(np.asarray(g0.lat).dtype).eps)
    for x, y in zip(ref, traced):
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), rtol=0,
                                   atol=4 * eps * float(jnp.max(jnp.abs(x))))
    two_om = 2.0 * g0.omega
    legacy = np.pad(np.asarray(two_om * jnp.sin(0.5 * (g0.lat[:-1] + g0.lat[1:]))),
                    (1, 1), constant_values=(-two_om, two_om))
    f0 = np.asarray(latlon_vertex_coriolis(g0))
    np.testing.assert_allclose(f0, np.broadcast_to(legacy[:, None], f0.shape),
                               rtol=eps, atol=eps * two_om)
    fv = np.asarray(latlon_vertex_coriolis(create_latlon_grid(12, 24, coriolis_alpha=ALPHA)))
    np.testing.assert_array_equal(fv[:, -1], fv[:, 0])
    assert np.all(fv[-1] == fv[-1, 0]) and np.all(fv[0] == -fv[-1, 0])


def test_voronoi_cache_does_not_leak_rotation():
    from legoesm.grids.voronoi import create_voronoi_mesh
    a = create_voronoi_mesh(2)
    create_voronoi_mesh(2, coriolis_alpha=ALPHA)
    b = create_voronoi_mesh(2, coriolis_alpha=0.0)
    np.testing.assert_array_equal(np.asarray(a.fEdge), np.asarray(b.fEdge))
    np.testing.assert_array_equal(np.asarray(a.fVertex), np.asarray(b.fVertex))


def test_rotated_jw_jet_is_westerly_in_rotated_frame():
    """Signed check: at the rotated-frame jet latitude the wind points along
    pole x position (eastward in the rotated frame), not against it."""
    from tests.test_cases.dcmip2008.jablonowski_rotated import (
        _build_rotated_state_dense_grid)
    from legoesm.grids.vertical import create_sigma_coordinate
    lat, lon = np.meshgrid(np.linspace(-1.4, 1.4, 29), np.linspace(0, 6.2, 63),
                           indexing="ij")
    u, v, _, _ = _build_rotated_state_dense_grid(
        jnp.asarray(lon), jnp.asarray(lat), create_sigma_coordinate(8), False, ALPHA)
    pole = np.array([-math.sin(ALPHA), 0.0, math.cos(ALPHA)])
    pos = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    east_r = np.cross(pole, pos)  # rotated-frame eastward (unnormalised)
    e_geo = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    n_geo = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], -1)
    k = 3  # upper-mid level, jet present
    vel = np.asarray(u[..., k])[..., None] * e_geo + np.asarray(v[..., k])[..., None] * n_geo
    proj = np.sum(vel * east_r, -1)
    speed = np.linalg.norm(vel, axis=-1)
    jet = speed > 0.5 * speed.max()
    assert jet.any() and np.all(proj[jet] > 0.0)


@pytest.mark.parametrize("form", ["as83", "legacy"])
def test_latlon_pe_forms_use_rotated_coriolis(form):
    """Both lat-lon PE forms (energy-conserving AS83 PV term and the legacy
    vector-invariant term) pick up the tilt: alpha = 0 is bit-identical to the
    default grid, a tilt changes only the wind tendencies through the
    wind-coupled Coriolis term (zero effect at rest), and T/p_s are untouched."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        cgrid_latlon_pe_tendencies)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState, CGridLatLonPrimitiveEquationConfig)
    nlev = 4
    from legoesm.grids.vertical import standard_hybrid_levels
    coord = (standard_hybrid_levels(nlev) if form == "as83"
             else create_sigma_coordinate(nlev))
    cfg = CGridLatLonPrimitiveEquationConfig()
    g0 = create_latlon_grid(16, 32)
    gz = create_latlon_grid(16, 32, coriolis_alpha=0.0)
    gr = create_latlon_grid(16, 32, coriolis_alpha=ALPHA)
    r = np.random.RandomState(0)
    v = r.randn(17, 32, nlev) * 8.0
    v[0] = v[-1] = 0.0
    moving = CGridLatLonHydrostaticState(
        u=jnp.asarray(r.randn(16, 33, nlev) * 10.0), v=jnp.asarray(v),
        T=jnp.asarray(250.0 + 20.0 * r.rand(16, 32, nlev)),
        p_s=jnp.asarray(1.0e5 + 2000.0 * r.randn(16, 32)),
        phis=jnp.zeros((16, 32)), tracers={})
    rest = moving._replace(u=jnp.zeros_like(moving.u), v=jnp.zeros_like(moving.v))

    def tend(s, g):
        return cgrid_latlon_pe_tendencies(s, g, coord, cfg, dt=600.0, vertical_form=form)

    a, z, b = tend(moving, g0), tend(moving, gz), tend(moving, gr)  # (du, dv, dT, dps, tr)
    for x, y in zip(a[:4], z[:4]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert float(jnp.max(jnp.abs(b[0] - a[0]))) > 1e-6
    assert float(jnp.max(jnp.abs(b[1] - a[1]))) > 1e-6
    np.testing.assert_array_equal(np.asarray(a[2]), np.asarray(b[2]))   # dT
    np.testing.assert_array_equal(np.asarray(a[3]), np.asarray(b[3]))   # dp_s
    ra, rb = tend(rest, g0), tend(rest, gr)
    for x, y in zip(ra[:4], rb[:4]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
