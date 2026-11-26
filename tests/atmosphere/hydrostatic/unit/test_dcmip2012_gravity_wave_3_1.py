"""DCMIP 2012 test 3-1 initial state and App. F.3 levels vs the document.

The checks are PHYSICAL (finite differences of the analytic fields), not a
re-statement of the builder's formulas: the background must be hydrostatic,
cyclostrophically balanced with u = u0 cos(lat) on the non-rotating planet,
and have Brunt-Vaisala frequency N; the levels must sit at z = 0, 1, ...,
10 km at the equator.  Needs JAX_ENABLE_X64=1.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.grids.vertical import (
    DCMIP2012_GW31_PEQ,
    dcmip2012_gw31_eta_table,
    dcmip2012_gw31_pressure,
    dcmip2012_transport_eta_table,
    make_dcmip2012_gw31_levels,
    pressure_from_hybrid,
)
from tests.test_cases.dcmip2012 import gravity_wave_3_1 as gw

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
                                reason="needs JAX_ENABLE_X64=1")


def test_f3_table_levels_are_1km_apart_at_the_equator():
    ak, bk, ptop = dcmip2012_gw31_eta_table()
    assert ak.shape == bk.shape == (11,)
    p = ak + bk * DCMIP2012_GW31_PEQ
    T_s = jnp.asarray([gw.T_EQ])
    p_s = jnp.asarray([DCMIP2012_GW31_PEQ])
    z = np.asarray(gw.height_from_pressure(jnp.asarray(p)[None], T_s, p_s))[0]
    np.testing.assert_allclose(z, np.arange(10, -1, -1) * 1000.0,
                               rtol=0, atol=1e-6)
    # document p_top 273.759 hPa with its Table III constants; ours uses
    # the shared constants (c_pd 1004.64 vs 1004.5, R_d 287.05 vs 287.0)
    # (Eq. 232 with them: 273.831410 hPa)
    assert ptop == pytest.approx(27383.141, abs=1e-3)
    assert bk[0] == 0.0 and bk[-1] == 1.0 and ak[-1] == 0.0
    # Eq. (236), c = 1: B linear in eta between lid and surface
    eta = p / DCMIP2012_GW31_PEQ
    np.testing.assert_allclose(bk, (eta - eta[0]) / (1 - eta[0]), atol=1e-15)
    coord = make_dcmip2012_gw31_levels()
    np.testing.assert_allclose(np.asarray(coord.B_full),
                               0.5 * (bk[1:] + bk[:-1]), atol=1e-15)


def test_transport_table_unchanged_by_the_f3_factoring():
    """The L30 transport table, recomputed with its pre-factoring
    arithmetic, is bit-identical."""
    from legoesm.grids.vertical import (
        DCMIP2012_TRANSPORT_RD, DCMIP2012_TRANSPORT_T0)
    H = DCMIP2012_TRANSPORT_RD * DCMIP2012_TRANSPORT_T0 / float(constants.g)
    z = 12000.0 * (1.0 - np.arange(31, dtype=np.float64) / 30)
    eta = np.exp(-z / H)
    ptop = 1.0e5 * eta[0]
    bk = (eta - eta[0]) / (1.0 - eta[0])
    bk[0], bk[-1] = 0.0, 1.0
    ak = ptop * (1.0 - bk)
    ak[-1] = 0.0
    a2, b2, p2 = dcmip2012_transport_eta_table(30)
    assert np.array_equal(a2, ak) and np.array_equal(b2, bk) and p2 == ptop


def _bg(lat, p):
    """Background T and z at (lat, p) arrays (no perturbation)."""
    T_s = gw.surface_temperature(lat)
    p_s = gw.surface_pressure(lat)
    return (gw.background_temperature(p, T_s, p_s),
            gw.height_from_pressure(p, T_s, p_s))


def test_background_is_hydrostatic_and_has_constant_N():
    lat = jnp.asarray([0.0, 0.4, 1.0])
    p = jnp.linspace(30000.0, 95000.0, 27)[None, :] * jnp.ones((3, 1))
    h = 1.0
    T, z = _bg(lat, p)
    _, zp = _bg(lat, p + h)
    _, zm = _bg(lat, p - h)
    # hydrostatic: T = -(g p / R_d) dz/dp
    T_hyd = -(constants.g * p / constants.R_d) * (zp - zm) / (2 * h)
    np.testing.assert_allclose(np.asarray(T_hyd), np.asarray(T), rtol=1e-8)
    # N^2 = g dln(theta)/dz
    theta = T * (gw.P0 / p) ** constants.kappa
    thp = _bg(lat, p + h)[0] * (gw.P0 / (p + h)) ** constants.kappa
    thm = _bg(lat, p - h)[0] * (gw.P0 / (p - h)) ** constants.kappa
    N2 = constants.g * (jnp.log(thp) - jnp.log(thm)) / (zp - zm)
    np.testing.assert_allclose(np.asarray(N2), gw.N_BV ** 2, rtol=1e-6)
    assert bool(jnp.all(jnp.isfinite(theta)))


def test_background_is_cyclostrophic_on_the_nonrotating_planet():
    """Omega = 0 gradient-wind balance on pressure surfaces:
    u^2 tan(lat) = -g dz/dlat |_p with u = u0 cos(lat) (radius cancels)."""
    lat = jnp.linspace(-1.2, 1.2, 25)
    p = jnp.asarray([35000.0, 60000.0, 90000.0])[None, :] * jnp.ones((25, 1))
    h = 1e-5
    _, zp = _bg(lat + h, p)
    _, zm = _bg(lat - h, p)
    lhs = (gw.U0 * jnp.cos(lat)) ** 2 * jnp.tan(lat)
    rhs = -constants.g * (zp - zm) / (2 * h)
    np.testing.assert_allclose(np.asarray(rhs), np.asarray(lhs)[:, None]
                               * np.ones((1, 3)), rtol=1e-6, atol=1e-9)


def test_eq94_inverts_eq96():
    lat = jnp.asarray([0.3])
    T_s, p_s = gw.surface_temperature(lat), gw.surface_pressure(lat)
    z = jnp.linspace(0.0, 10000.0, 11)
    p = dcmip2012_gw31_pressure(z, T_s[:, None], p_s[:, None])
    np.testing.assert_allclose(
        np.asarray(gw.height_from_pressure(p, T_s, p_s)), np.asarray(z)[None],
        atol=1e-7)


def test_columns_perturbation_matches_eq107_109():
    coord = make_dcmip2012_gw31_levels()
    a = gw.small_planet_radius()
    lon = jnp.asarray([gw.LON_C, gw.LON_C + 5000.0 / a, 0.0])
    lat = jnp.zeros(3)
    p_s, T = gw.columns(lon, lat, coord, a)
    p = pressure_from_hybrid(coord, p_s, full=True)
    Tb, z = _bg(lat, p)
    dth = (T - Tb) * (gw.P0 / p) ** constants.kappa
    expect = np.sin(2 * np.pi * np.asarray(z) / 20000.0)
    np.testing.assert_allclose(np.asarray(dth[0]), expect[0], atol=1e-12)
    # r = d: s = 1/2
    np.testing.assert_allclose(np.asarray(dth[1]), 0.5 * expect[1],
                               atol=1e-9)
    # antipode-ish (120 deg away, r ~ 107 km): s = d^2/(d^2+r^2)
    r = a * 2 * math.pi / 3
    np.testing.assert_allclose(np.asarray(dth[2]),
                               expect[2] * 25e6 / (25e6 + r * r), atol=1e-12)
    # equatorial p_s = p_eq exactly
    assert float(p_s[0]) == pytest.approx(gw.P_EQ, rel=0, abs=1e-9)


def test_small_planet_radius_and_refusals():
    a = gw.small_planet_radius()
    assert a == pytest.approx(float(constants.R_earth) / 125.0, rel=1e-15)
    coord = make_dcmip2012_gw31_levels()
    from types import SimpleNamespace
    lat = jnp.zeros((6, 2, 2))
    earth = SimpleNamespace(n=2, lon=lat, lat=lat, angle=lat,
                            radius=float(constants.R_earth), omega=0.0)
    with pytest.raises(ValueError, match="a/X"):
        gw.gravity_wave_init(earth, coord)
    spinning = SimpleNamespace(n=2, lon=lat, lat=lat, angle=lat, radius=a,
                               omega=float(constants.Omega))
    with pytest.raises(ValueError, match="non-rotating"):
        gw.gravity_wave_init(spinning, coord)
    from legoesm.grids.voronoi import create_voronoi_mesh
    spinning_mesh = create_voronoi_mesh(2, radius=a)   # Earth's Omega
    assert not hasattr(spinning_mesh, "omega")
    with pytest.raises(ValueError, match="non-rotating"):
        gw.gravity_wave_init_mpas(spinning_mesh, coord)
    bare = SimpleNamespace(n=2, lon=lat, lat=lat, angle=lat, radius=a)
    with pytest.raises(ValueError, match="both fEdge and fVertex"):
        gw.gravity_wave_init(bare, coord)
    for k in ("fEdge", "fVertex"):
        one = SimpleNamespace(n=2, lon=lat, lat=lat, angle=lat, radius=a,
                              **{k: jnp.zeros(3)})
        with pytest.raises(ValueError, match="both fEdge and fVertex"):
            gw.gravity_wave_init(one, coord)
    # a valid non-rotating mesh validates on the host even under jit
    still = create_voronoi_mesh(2, radius=a, omega=0.0)
    T = jax.jit(lambda: gw.gravity_wave_init_mpas(still, coord).T.data)()
    assert T.shape == (still.nCells, 10)
    from legoesm.grids.vertical import create_sigma_coordinate
    with pytest.raises(TypeError):
        gw.columns(lat, lat, create_sigma_coordinate(10), a)


def test_builders_on_all_grids():
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis_3d
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.voronoi import create_voronoi_mesh
    a = gw.small_planet_radius()
    coord = make_dcmip2012_gw31_levels()
    ll = create_latlon_grid(24, 48, radius=a, omega=0.0)
    s = gw.gravity_wave_init_latlon(ll, coord)
    assert s.T.data.shape == (24, 48, 10)
    m = create_voronoi_mesh(3, radius=a, omega=0.0)
    s = gw.gravity_wave_init_mpas(m, coord)
    assert s.T.data.shape == (m.nCells, 10)
    g = create_gaussian_grid(21, radius=a, omega=0.0)
    s = gw.gravity_wave_init_spectral(g, coord)
    T = np.asarray(sh_synthesis_3d(g, s.T_hat.data))
    assert np.all(np.isfinite(T)) and 225.0 < T.min() and T.max() < 305.0
