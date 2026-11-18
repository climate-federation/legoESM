"""Shared shallow-water core: operator routing and a live-oracle regression.

The oracle is the same core with the free-function B2 adapters forced in
(``LatLonCGridOperators`` / ``MPASEdgeOperators``), i.e. the operators the
lat-lon and MPAS models used before the neighbour layer.  Tolerances (fp64,
voted): lat-lon tendencies 1e-13 relative to the field's max, MPAS tendencies
bit-identical, 10-step states 1e-11 on both.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm import shallow_water_generic as gen
from legoesm.atmosphere.dynamics.gcm import shallow_water_latlon_cgrid as ll
from legoesm.atmosphere.dynamics.gcm import shallow_water_mpas as mp
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_latlon_grid, create_regional_latlon_grid
from legoesm.grids.neighbour_connectivity import latlon_neighbour_supported
from legoesm.grids.operator_adapters import (
    LatLonCGridOperators,
    LatLonNeighbourOperators,
    MPASNeighbourOperators,
    latlon_sw_grid_operators,
    mpas_edge_operators,
)
from legoesm.grids.voronoi import create_voronoi_mesh

from tests.atmosphere.shallow_water.test_cases.williamson_mpas import williamson_test2_mpas

TEND_RTOL = 1e-13
TRAJ_RTOL = 1e-11
LATLON_CONFIGS = {
    "ppm": dict(),
    "centered": dict(use_ppm_transport=False),
    "A_h": dict(A_h=1e5),
    "del4": dict(nu_del4=1e15),
}
MPAS_CONFIGS = {
    "energy": dict(),
    "enstrophy": dict(pv_scheme="enstrophy"),
    "order3": dict(thickness_order=3),
    "apvm": dict(apvm_scale=0.5),
    "del2_del4": dict(nu_del2=1e4, nu_del4=1e14),
}
DT_LL, DT_MPAS = 300.0, 600.0


@pytest.fixture(scope="module", autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def grid(_x64):
    return create_latlon_grid(16, 32)


@pytest.fixture(scope="module")
def mesh(_x64):
    return create_voronoi_mesh(3, lloyd_iterations=50)


def _rel(new, old):
    err = float(jnp.max(jnp.abs(new - old)))
    scale = float(jnp.max(jnp.abs(old)))
    assert np.isfinite(err) and scale > 0
    return err / scale


def _latlon_state(grid):
    s = ll.williamson_test2_cgrid(grid)
    lat, lon = grid.lat[:, None], grid.lon[None, :]
    du = 3.0 * jnp.cos(lat) * jnp.sin(lon - 0.5 * grid.dlon)
    return s._replace(
        h=s.h + 40.0 * jnp.cos(lat) ** 2 * jnp.cos(2 * lon),
        u=s.u + jnp.concatenate([du, du[:, :1]], axis=1),
        v=(s.v + 2.0 * jnp.cos(grid.lat_v[:, None]) * jnp.sin(3 * lon))
        .at[0].set(0.0).at[-1].set(0.0))


def _mpas_state(mesh):
    s = williamson_test2_mpas(mesh)
    h = s.h.data + 40.0 * jnp.cos(mesh.latCell) ** 2 * jnp.cos(2 * mesh.lonCell)
    u = s.u.data + 3.0 * jnp.cos(mesh.latEdge) * jnp.sin(mesh.lonEdge)
    return s._replace(h=s.h.replace(data=h), u=s.u.replace(data=u))


def _force_free_function_ops(m):
    m.setattr(gen, "latlon_sw_grid_operators", LatLonCGridOperators)
    m.setattr(gen, "mpas_neighbour_operators", mpas_edge_operators)


# ------------------------------------------------------------------ routing

def test_routing_by_grid(grid, monkeypatch):
    assert latlon_neighbour_supported(grid)
    assert isinstance(latlon_sw_grid_operators(grid), LatLonNeighbourOperators)
    regional, _ = create_regional_latlon_grid(
        8, 16, lat_south=30.0, lat_north=50.0, lon_west=0.0, lon_east=40.0,
        periodic_x=True, dtype=jnp.float64)
    assert not latlon_neighbour_supported(regional)
    ops = latlon_sw_grid_operators(regional)
    assert type(ops) is LatLonCGridOperators
    import legoesm.grids.halo as halo
    monkeypatch.setattr(halo, "get_halo_backend", lambda: "mpi")
    assert not latlon_neighbour_supported(grid)
    assert type(latlon_sw_grid_operators(grid)) is LatLonCGridOperators


def test_regional_tendency_uses_free_function_ops(monkeypatch):
    regional, _ = create_regional_latlon_grid(
        8, 16, lat_south=30.0, lat_north=50.0, lon_west=0.0, lon_east=40.0,
        periodic_x=True, dtype=jnp.float64)
    rng = np.random.default_rng(0)
    nj, ni = regional.n_lat, regional.n_lon
    state = ll.CGridLatLonShallowWaterState(
        h=jnp.asarray(1000.0 + rng.random((nj, ni))),
        u=jnp.asarray(rng.standard_normal((nj, ni + 1))),
        v=jnp.zeros((nj + 1, ni)),
        h_s=jnp.zeros((nj, ni)))
    cfg = ll.CGridLatLonShallowWaterConfig(use_ppm_transport=False)
    routed = gen.cgrid_latlon_sw_tendencies(state, regional, cfg, DT_LL)
    with monkeypatch.context() as m:
        m.setattr(gen, "latlon_sw_grid_operators", LatLonCGridOperators)
        forced = gen.cgrid_latlon_sw_tendencies(state, regional, cfg, DT_LL)
    for a, b in zip(routed, forced):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_mpas_uses_neighbour_ops(mesh):
    assert isinstance(gen.mpas_sw_operators(mesh, mp.MPASShallowWaterConfig()).grid_ops,
                      MPASNeighbourOperators)


# ------------------------------------------------- live-oracle regression

@pytest.mark.parametrize("cname", list(LATLON_CONFIGS))
def test_latlon_matches_free_function_ops(cname, grid, monkeypatch):
    cfg = ll.CGridLatLonShallowWaterConfig(**LATLON_CONFIGS[cname])
    state = _latlon_state(grid)
    new = gen.cgrid_latlon_sw_tendencies(state, grid, cfg, DT_LL)
    with monkeypatch.context() as m:
        _force_free_function_ops(m)
        old = gen.cgrid_latlon_sw_tendencies(state, grid, cfg, DT_LL)
    for name, a, b in zip(("dh", "du", "dv"), new, old):
        assert _rel(a, b) <= TEND_RTOL, (name, _rel(a, b))


@pytest.mark.parametrize("cname", list(MPAS_CONFIGS))
def test_mpas_matches_free_function_ops(cname, mesh, monkeypatch):
    cfg = mp.MPASShallowWaterConfig(**MPAS_CONFIGS[cname])
    state = _mpas_state(mesh)
    new = gen.mpas_shallow_water_tendencies(state, mesh, cfg, dt=DT_MPAS)
    with monkeypatch.context() as m:
        _force_free_function_ops(m)
        old = gen.mpas_shallow_water_tendencies(state, mesh, cfg, dt=DT_MPAS)
    np.testing.assert_array_equal(np.asarray(new.dh_dt.data), np.asarray(old.dh_dt.data))
    np.testing.assert_array_equal(np.asarray(new.du_dt.data), np.asarray(old.du_dt.data))


def test_model_trajectories_match_free_function_ops(grid, mesh, monkeypatch):
    def run_ll():
        model = ll.CGridLatLonShallowWaterModel(
            grid, ll.CGridLatLonShallowWaterConfig(fix_mass=False), dt=DT_LL)
        s = _latlon_state(grid)
        for _ in range(10):
            s = model.step(s, DT_LL)
        return s

    def run_mpas():
        model = mp.MPASShallowWaterModel(mesh, mp.MPASShallowWaterConfig(fix_mass=False))
        s = _mpas_state(mesh)
        for _ in range(10):
            s = model.step(s, DT_MPAS)
        return s

    ll_new, mp_new = run_ll(), run_mpas()
    with monkeypatch.context() as m:
        _force_free_function_ops(m)
        ll_old, mp_old = run_ll(), run_mpas()
    for f in ("h", "u", "v"):
        assert _rel(getattr(ll_new, f), getattr(ll_old, f)) <= TRAJ_RTOL, f
    for f in ("h", "u"):
        assert _rel(getattr(mp_new, f).data, getattr(mp_old, f).data) <= TRAJ_RTOL, f
    # Mass: both conserve without the fixer (halo-3 PPM, TRiSK flux form).
    m0 = float(jnp.sum(_latlon_state(grid).h * grid.area))
    assert abs(float(jnp.sum(ll_new.h * grid.area)) - m0) <= 1e-13 * m0


@pytest.mark.parametrize("cname", ["ppm", "centered"])
def test_traced_grid_falls_back_to_free_function_ops(cname, grid, monkeypatch):
    """A grid passed as a jit argument has traced geometry: the scope check
    cannot run, so the free-function adapter is used (no crash)."""
    cfg = ll.CGridLatLonShallowWaterConfig(**LATLON_CONFIGS[cname])
    state = _latlon_state(grid)
    traced = jax.jit(lambda s, g: gen.cgrid_latlon_sw_tendencies(s, g, cfg, DT_LL))(state, grid)
    with monkeypatch.context() as m:
        _force_free_function_ops(m)
        eager = gen.cgrid_latlon_sw_tendencies(state, grid, cfg, DT_LL)
    for a, b in zip(traced, eager):
        assert _rel(a, b) <= TEND_RTOL
