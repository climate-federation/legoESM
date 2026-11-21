"""RULE 5 for the ported open-boundary radiation and the OM4p5 lateral friction.

The hard limit (width 0) is the reference form; the tests of the reference forms
(test_open_boundary_*.py, test_om4p5_*.py) run there.  Here, at the default width
0.01: the measured per-case gap from the hard form (stored, KNOWN DIFFERENCE),
and a finite derivative through each smoothed switch.
"""
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import (  # noqa: E402
    om4_resolution_function, om4p25_lateral_friction_tendency_cgrid, wkb_first_baroclinic_wave_speed)
from legoesm.ocean.dynamics.open_boundary_latlon import _orlanski, _orlanski_1d  # noqa: E402
from legoesm.ocean.eos import make_eos_fn  # noqa: E402
from legoesm.ocean.state import LatLonCGridOceanConfig, OMp25Config  # noqa: E402

W = 0.01
RI, RO = 0.05, 0.001
OM4P5 = dict(C2=0.15, Cu2=0.01, C4=0.06, Cu4=0.01, Kh_sin_lat=2000.0, Kh_sin_power=4.0,
             bound_coriolis_vel=6.0, bound_coef=0.8, dt=1800.0)

# Max |smooth - hard| per case at W, measured on this fixture (KNOWN DIFFERENCE, RULE 5).
GAPS = {
    "orlanski": [5.551115123125783e-17, 4.440892098500626e-16, 0.0011095395193931878],
    "orlanski_1d": [2.220446049250313e-16, 1.1102230246251565e-16, 2.220446049250313e-16],
    "resolution_fn": [0.0, 0.9769121715676555, 0.0],
    "wkb": [0.0012680457993169414, 2.2288693464989073e-05, 2.2288693829031203e-05],
    "om4p5_tendency": [4.290172723640188e-10, 4.731819003420793e-10, 3.590434841460404e-10],
}


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _rows(seed, n=5):
    rng = np.random.default_rng(seed)
    return [jnp.asarray(rng.standard_normal((3, 16))) for _ in range(n)]


def _grid():
    from legoesm.grids.factory import create_regional_grid
    grid, _ = create_regional_grid("latlon", n_lat=60, n_lon=8, lat_south=0.0, lat_north=30.0,
                                   lon_west=140.0, lon_east=144.0)
    return grid


def _uv(grid, seed):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, 2)))
    v = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, 2)))
    return u, v.at[0].set(0.0).at[-1].set(0.0)


def _column(seed):
    rng = np.random.default_rng(seed)
    T = 25.0 - jnp.cumsum(jnp.asarray(np.abs(rng.standard_normal((2, 3, 20))) * 0.5), axis=-1)
    T = T.at[..., 7].set(T[..., 6] - 1e-7)          # a near-neutral interface, inside the sqrt bridge
    h = jnp.full(T.shape, 25.0)
    return T, jnp.full(T.shape, 35.0), h, jnp.ones(T.shape, bool)


def _eos():
    return make_eos_fn("linear", None)


def _cases():
    out = {}
    out["orlanski"] = [float(jnp.max(jnp.abs(_orlanski(*_rows(s), RI, RO, width=W)
                                             - _orlanski(*_rows(s), RI, RO, width=0.0)))) for s in range(3)]
    out["orlanski_1d"] = [float(jnp.max(jnp.abs(
        _orlanski_1d(*_rows(s), RI, RO, jnp.asarray(1.0), width=W)
        - _orlanski_1d(*_rows(s), RI, RO, jnp.asarray(1.0), width=0.0)))) for s in range(3)]
    grid = _grid()
    out["resolution_fn"] = []
    for c in (2.0, 2.5, 3.0):
        c1 = jnp.full((grid.n_lat, grid.n_lon), c)
        a = om4_resolution_function(c1, grid, width=W)
        b = om4_resolution_function(c1, grid, width=0.0)
        out["resolution_fn"].append(float(max(jnp.max(jnp.abs(a[0] - b[0])), jnp.max(jnp.abs(a[1] - b[1])))))
    out["wkb"] = []
    for s in range(3):
        T, S, h, act = _column(s)
        args = (T, S, h, act, _eos(), constants.rho_ocean, constants.g)
        out["wkb"].append(float(jnp.max(jnp.abs(wkb_first_baroclinic_wave_speed(*args, width=W)
                                                 - wkb_first_baroclinic_wave_speed(*args, width=0.0)))))
    R = om4_resolution_function(jnp.full((grid.n_lat, grid.n_lon), 2.5), grid, width=0.0)
    out["om4p5_tendency"] = []
    for s in range(3):
        u, v = _uv(grid, s)
        a = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, smooth_width=W, **OM4P5)
        b = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, smooth_width=0.0, **OM4P5)
        out["om4p5_tendency"].append(float(max(jnp.max(jnp.abs(a[0] - b[0])), jnp.max(jnp.abs(a[1] - b[1])))))
    return out


def test_default_widths_are_the_approved_value():
    assert LatLonCGridOceanConfig().open_boundary_smooth_width == W
    assert OMp25Config().smooth_width == W


def test_stored_gaps_at_the_default_width():
    measured = _cases()
    for name, gaps in GAPS.items():
        np.testing.assert_allclose(measured[name], gaps, rtol=1e-6, atol=1e-12, err_msg=name)


def test_gradients_through_the_switches():
    rows = _rows(0)
    check_grads(lambda f1n: _orlanski(rows[0], rows[1], f1n, rows[3], rows[4], RI, RO, width=W),
                (rows[2],), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
    check_grads(lambda f1n: _orlanski_1d(rows[0], rows[1], f1n, rows[3], rows[4], RI, RO,
                                         jnp.asarray(1.0), width=W),
                (rows[2],), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)
    # inside the inflow/outflow band (dfx*dft just above 0): d(out)/d(f1_new) finite and non-zero
    one = [jnp.asarray([[x]]) for x in (0.3, 1.0, 0.999, 0.5, 0.0)]
    g = jax.grad(lambda x: _orlanski(one[0], one[1], x, one[3], one[4], RI, RO, width=W).sum())(one[2])
    assert np.isfinite(float(g[0, 0])) and float(g[0, 0]) != 0.0
    grid = _grid()
    gR = jax.grad(lambda c: sum(r.sum() for r in om4_resolution_function(
        jnp.full((grid.n_lat, grid.n_lon), c), grid, width=W)))(2.5)
    assert np.isfinite(float(gR)) and float(gR) != 0.0      # the hard step has zero derivative
    T, S, h, act = _column(0)
    gN = jax.grad(lambda t: wkb_first_baroclinic_wave_speed(t, S, h, act, _eos(), constants.rho_ocean, constants.g,
                                                            width=W).sum())(T)
    assert np.all(np.isfinite(np.asarray(gN)))
    R = om4_resolution_function(jnp.full((grid.n_lat, grid.n_lon), 2.5), grid, width=W)
    u, v = _uv(grid, 0)
    check_grads(lambda uu: om4p25_lateral_friction_tendency_cgrid(uu, v, grid, resolution_fn=R,
                                                                  smooth_width=W, **OM4P5)[0],
                (u,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)
