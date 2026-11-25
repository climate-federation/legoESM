"""The OM4p5 form of the OM4 lateral friction (MOM6-examples OM4_05, MOM_hor_visc.F90,
MOM_lateral_mixing_coeffs.F90): the WKB first baroclinic wave speed, MOM6's step resolution
function, the latitude background, the Coriolis bound and the caps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    om4_resolution_function, om4p25_lateral_friction_tendency_cgrid, wkb_first_baroclinic_wave_speed)

OM4P5 = dict(C2=0.15, Cu2=0.01, C4=0.06, Cu4=0.01, Kh_sin_lat=2000.0, Kh_sin_power=4.0,
             bound_coriolis_vel=6.0, bound_coef=0.8, dt=1800.0)


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _grid(n_lat=24, n_lon=24):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)


def _half_degree_grid():
    """0.5 degree, 0 to 30 N: the deformation radius is resolved near the equator, not at 30 N."""
    from legoesm.grids.factory import create_regional_grid
    grid, _ = create_regional_grid("latlon", n_lat=60, n_lon=8, lat_south=0.0, lat_north=30.0,
                                   lon_west=140.0, lon_east=144.0)
    return grid


def _fields(grid, nlev=2, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, nlev)))
    v = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, nlev)))
    return u, v.at[0].set(0.0).at[-1].set(0.0)


def test_wkb_wave_speed_of_uniform_stratification():
    """Constant N with a linear equation of state: c1 = N H / pi exactly (the WKB integral)."""
    rho0, g, alpha, N = 1025.0, constants.g, 2e-4, 5e-3
    dz = np.full(40, 100.0)
    z = np.cumsum(dz) - dz / 2
    T = 20.0 - (N ** 2 / (g * alpha)) * z                       # dT/dz = -N^2 / (g alpha)
    eos = lambda T, S, p: rho0 * (1.0 - alpha * (T - 10.0))     # noqa: E731
    h = jnp.asarray(np.broadcast_to(dz, (3, 2, 40)))
    act = jnp.ones((3, 2, 40), bool)
    c1 = wkb_first_baroclinic_wave_speed(jnp.asarray(np.broadcast_to(T, (3, 2, 40))), jnp.zeros((3, 2, 40)),
                                         h, act, eos, rho0, g, width=0.0)
    H_int = z[-1] - z[0]                                         # the interfaces span centre to centre
    assert np.allclose(np.asarray(c1), N * H_int / np.pi, rtol=1e-12), (np.asarray(c1)[0, 0], N * H_int / np.pi)


def test_wkb_ignores_unstable_and_inactive_interfaces():
    rho0, g = 1025.0, constants.g
    eos = lambda T, S, p: rho0 - T                              # noqa: E731  (warmer is lighter)
    T = jnp.asarray([[[10.0, 12.0, 8.0, 0.0]]])                # unstable top pair, stable second, last cell dry
    act = jnp.asarray([[[True, True, True, False]]])
    h = jnp.full((1, 1, 4), 50.0)
    c1 = float(wkb_first_baroclinic_wave_speed(T, jnp.zeros_like(T), h, act, eos, rho0, g, width=0.0)[0, 0])
    expect = np.sqrt(g / rho0 * 4.0 / 50.0) * 50.0 / np.pi       # only the stable 12 -> 8 interface
    assert abs(c1 - expect) < 1e-12 * expect


def test_resolution_function_is_the_mom6_step():
    """R = 0 where c1^2 > (dx^2 + dy^2)(f^2 + 2 c1 |grad f|), else 1, cell by cell."""
    grid = _half_degree_grid()
    c1 = jnp.full((grid.n_lat, grid.n_lon), 2.5)
    R_h, R_q = om4_resolution_function(c1, grid, width=0.0)
    lat = np.asarray(grid.lat)
    om = grid.omega
    dx = constants.R_earth * np.cos(lat) * grid.dlon
    dy = np.asarray(grid.area)[:, 0] / dx
    f, beta = 2 * om * np.sin(lat), 2 * 2 * om * np.cos(lat) / constants.R_earth
    expect = np.where(2.5 ** 2 > (dx ** 2 + dy ** 2) * (f ** 2 + 2.5 * beta), 0.0, 1.0)
    assert np.array_equal(np.asarray(R_h)[:, 0], expect)
    assert expect.min() == 0.0 and expect.max() == 1.0          # both regimes occur on this grid
    assert np.asarray(R_q).shape == (grid.n_lat + 1, grid.n_lon + 1)


def test_om4p5_dissipates_energy_and_is_zero_at_rest():
    grid = _grid()
    u, v = _fields(grid, seed=3)
    R = om4_resolution_function(jnp.full((grid.n_lat, grid.n_lon), 2.5), grid, width=0.0)
    tu, tv = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **OM4P5)
    area = np.asarray(grid.area)[..., None]
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)
    dE = float(np.sum(np.asarray(u) * np.asarray(tu) * A_u) + np.sum(np.asarray(v) * np.asarray(tv) * A_v))
    E = float(np.sum(np.asarray(u) ** 2 * A_u) + np.sum(np.asarray(v) ** 2 * A_v))
    assert dE < -1e-12 * E                                       # dissipates, and does something
    z = jnp.zeros_like(u), jnp.zeros_like(v)
    tz = om4p25_lateral_friction_tendency_cgrid(*z, grid, resolution_fn=R, **OM4P5)
    assert float(jnp.max(jnp.abs(tz[0]))) == 0.0 and float(jnp.max(jnp.abs(tz[1]))) == 0.0


def test_each_om4p5_term_changes_the_tendency():
    """Each term acts where it should: the latitude background poleward of about 46 degrees under weak
    flow (it must beat 0.01 m/s times the grid spacing), the Coriolis bound under grid-scale shear of
    metres per second, the resolution function where the deformation radius is resolved."""
    from legoesm.grids.factory import create_regional_grid

    def diff(grid, u, v, R, change):
        a = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **OM4P5)
        b = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **{**OM4P5, **change})
        return float(jnp.max(jnp.abs(a[0] - b[0]))) / float(jnp.max(jnp.abs(a[0])))

    high, _ = create_regional_grid("latlon", n_lat=40, n_lon=8, lat_south=40.0, lat_north=60.0,
                                   lon_west=140.0, lon_east=144.0)
    u, v = _fields(high, seed=5)
    R = om4_resolution_function(jnp.full((high.n_lat, high.n_lon), 2.5), high, width=0.0)
    assert diff(high, 1e-4 * u, 1e-4 * v, R, dict(Kh_sin_lat=0.0)) > 1e-3
    grid = _half_degree_grid()
    u, v = _fields(grid, seed=5)
    R = om4_resolution_function(jnp.full((grid.n_lat, grid.n_lon), 2.5), grid, width=0.0)
    assert diff(grid, 15.0 * u, 15.0 * v, R, dict(bound_coriolis_vel=0.0)) > 1e-3
    a = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **OM4P5)
    b = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=tuple(jnp.ones_like(r) for r in R),
                                               **OM4P5)
    assert float(jnp.max(jnp.abs(a[0] - b[0]))) > 1e-3 * float(jnp.max(jnp.abs(a[0])))


def test_caps_bound_the_tendency():
    """With a tiny dt the caps bind: the tendency equals the capped one, smaller than uncapped."""
    grid = _grid()
    u, v = _fields(grid, seed=7)
    R = om4_resolution_function(jnp.full((grid.n_lat, grid.n_lon), 2.5), grid, width=0.0)
    capped = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **{**OM4P5, 'dt': 1e9})
    free = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **{**OM4P5, 'bound_coef': 0.0})
    assert float(jnp.abs(capped[0]).max()) < float(jnp.abs(free[0]).max())


def test_om4p5_options_need_the_resolution_function():
    grid = _grid()
    u, v = _fields(grid)
    with pytest.raises(ValueError):
        om4p25_lateral_friction_tendency_cgrid(u, v, grid, Kh_sin_lat=2000.0)


def test_om4p5_config_dispatch_and_full_step():
    """lateral_friction_scheme='om4p25' with the OM4p5 options (resolution_function='mom6_step_wkb')
    builds, takes a full ocean step from the model's own T, S and layer thicknesses, stays finite, and
    differs from the legacy taper; an unknown resolution function raises."""
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OMp25Config

    r = build_eady_uniform_setup(n_lat=16, n_lon=16, momentum_advection="vector_invariant")
    lv0 = r.model_config.lateral_viscosity._replace(A_h=0.0, B_h=0.0, C_smag=0.0, C_leith=0.0)
    om4p5 = OMp25Config(Kh_sin_lat=2000.0, Kh_sin_power=4.0, bound_coriolis_vel=6.0,
                        resolution_function="mom6_step_wkb", bound_coef=0.8)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev", dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S), u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    out = {}
    for name, om in (("om4p5", om4p5), ("taper", OMp25Config())):
        cfg = r.model_config._replace(lateral_friction_scheme="om4p25", omp25=om, lateral_viscosity=lv0)
        out[name] = LatLonCGridOceanModel(r.grid, r.z_coord, cfg).step(s, 300.0)
        assert bool(jnp.all(jnp.isfinite(out[name].u.data)))
    assert float(jnp.max(jnp.abs(out["om4p5"].u.data - out["taper"].u.data))) > 0.0
    bad = r.model_config._replace(lateral_friction_scheme="om4p25", lateral_viscosity=lv0,
                                  omp25=OMp25Config(resolution_function="hallberg_smooth"))
    with pytest.raises(ValueError, match="resolution_function"):
        LatLonCGridOceanModel(r.grid, r.z_coord, bad).step(s, 300.0)
