"""mpas_land_forcing: compiled (as the driver runs it) == eager, per leaf and dtype."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.model_driver import mpas_land_forcing
from legoesm.grids.vertical import make_cam6_l32_levels
from legoesm.grids.voronoi import create_voronoi_mesh

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(2)


@pytest.mark.parametrize("has_precip", [True, False])
@pytest.mark.parametrize("has_qv", [True, False])
def test_compiled_matches_eager(mesh, has_precip, has_qv):
    rng = np.random.default_rng(4)
    nc, ne, nlev = mesh.nCells, mesh.nEdges, 32
    arr = lambda *shape, lo=0.0, hi=1.0: jnp.asarray(rng.uniform(lo, hi, shape))
    args = (arr(nc, lo=0, hi=900), arr(nc, lo=150, hi=450),
            arr(nc, hi=1e-3) if has_precip else None,
            arr(nc, nlev, lo=200, hi=310), arr(nc, nlev, hi=0.02) if has_qv else None,
            arr(nc, lo=5e4, hi=1.05e5), arr(ne, nlev, lo=-30, hi=30),
            np.float64(172.0), np.float64(45000.0))
    kw = dict(mesh=mesh, sigma=make_cam6_l32_levels(),
              lat=jnp.asarray(mesh.latCell), lon=jnp.asarray(mesh.lonCell), co2_ppmv=412.0)
    eager = mpas_land_forcing(*args, **kw)
    fn = jax.jit(functools.partial(mpas_land_forcing, **kw))
    compiled = fn(*args)
    fn(*args[:-2], np.float64(173.0), np.float64(46000.0))  # new day/time: no retrace
    assert fn._cache_size() == 1
    for name, a, b in zip(eager._fields, eager, compiled):
        assert a.dtype == b.dtype, name
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-12, err_msg=name)
    assert compiled.cos_zenith.dtype == jnp.float64
    assert float(jnp.max(compiled.cos_zenith)) > 0.0  # daytime somewhere
    if not has_precip:
        assert float(jnp.max(jnp.abs(compiled.precip_total))) == 0.0


def test_rain_snow_split_identical_at_freezing(mesh):
    """precip_snow is a step in T; lowest-level T is sliced, never computed,
    so compiled and eager see identical bits at, and 1 ulp around, T_freeze."""
    from legoesm import constants
    nc, ne, nlev = mesh.nCells, mesh.nEdges, 32
    tf = np.float64(constants.T_freeze)
    T_low = np.full(nc, tf)
    T_low[1::3] = np.nextafter(tf, 0.0)
    T_low[2::3] = np.nextafter(tf, 1e3)
    T3 = jnp.asarray(np.repeat(T_low[:, None], nlev, axis=1))
    args = (jnp.full(nc, 400.0), jnp.full(nc, 300.0), jnp.full(nc, 1e-4), T3, None,
            jnp.full(nc, 1e5), jnp.zeros((ne, nlev)), np.float64(10.0), np.float64(0.0))
    kw = dict(mesh=mesh, sigma=make_cam6_l32_levels(),
              lat=jnp.asarray(mesh.latCell), lon=jnp.asarray(mesh.lonCell), co2_ppmv=412.0)
    eager = mpas_land_forcing(*args, **kw).precip_snow
    compiled = jax.jit(functools.partial(mpas_land_forcing, **kw))(*args).precip_snow
    np.testing.assert_array_equal(np.asarray(eager).view(np.uint64), np.asarray(compiled).view(np.uint64))
    assert 0 < int(np.sum(np.asarray(compiled) > 0)) < nc  # both sides of the step present


def test_cell_winds_match_driver_helper(mesh):
    """Winds follow the driver's _cell_winds(level=-1) for both state kinds:
    edge-normal u on the Voronoi lane, carried cell (u, v) on the column lane."""
    from types import SimpleNamespace
    from legoesm.driver.model_driver import _cell_winds
    rng = np.random.default_rng(7)
    nc, ne, nlev = mesh.nCells, mesh.nEdges, 32
    kw = dict(mesh=mesh, sigma=make_cam6_l32_levels(),
              lat=jnp.asarray(mesh.latCell), lon=jnp.asarray(mesh.lonCell), co2_ppmv=412.0)
    fn = jax.jit(functools.partial(mpas_land_forcing, **kw))
    base = (jnp.full(nc, 400.0), jnp.full(nc, 300.0), None,
            jnp.full((nc, nlev), 280.0), None, jnp.full(nc, 1e5))
    t = (np.float64(10.0), np.float64(0.0))
    u_e = jnp.asarray(rng.uniform(-20, 20, (ne, nlev)))
    edge = SimpleNamespace(u=SimpleNamespace(data=u_e), v=None)
    got = fn(*base, u_e, *t)
    for a, b in zip(_cell_winds(edge, mesh, level=-1), (got.u_lowest, got.v_lowest)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-12)
    u_c = jnp.asarray(rng.uniform(-20, 20, (nc, nlev)))
    v_c = jnp.asarray(rng.uniform(-20, 20, (nc, nlev)))
    col = SimpleNamespace(u=SimpleNamespace(data=u_c), v=SimpleNamespace(data=v_c))
    got = fn(*base, u_c, *t, v3=v_c)
    for a, b in zip(_cell_winds(col, mesh, level=-1), (got.u_lowest, got.v_lowest)):
        np.testing.assert_array_equal(np.asarray(b), np.asarray(a))
