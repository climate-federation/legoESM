"""Sponge viscosity next to open sides (CROCO SPONGE_VIS2, set_nudgcof.F) added to the OM4p5 lateral
friction: SpongeForcing.visc2 at cell centres, q points the mean of the four cells around them, added
after the bound caps and not scaled by the resolution function.
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
    om4_resolution_function, om4p25_lateral_friction_tendency_cgrid, viscous_tendency_cgrid)

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


def _setup(seed=0, nlev=3):
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(n_lat=20, n_lon=24, radius=constants.R_earth)
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, nlev)))
    v = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, nlev)))
    v = v.at[0].set(0.0).at[-1].set(0.0)
    R = om4_resolution_function(jnp.full((grid.n_lat, grid.n_lon), 2.5), grid)
    n = np.arange(grid.n_lat)                         # a southern sponge, 10 cells, CROCO's cosine
    w = np.where(n < 10, 0.5 * (np.cos(np.pi * n / 10) + 1.0), 0.0)
    zon = 1.0 + 0.5 * np.cos(2 * np.pi * np.arange(grid.n_lon) / grid.n_lon)   # varies zonally: tests the wrap
    sp = jnp.asarray(5.0e3 * w[:, None] * zon[None, :])
    return grid, u, v, R, sp


def _q_mean(sp):
    """Reference corner values: the mean of the four cells around each q point, zonally periodic as the
    model's strain-rate corners (cell i-1 of column 0 is n_lon-1), edge rows repeated meridionally."""
    s = np.asarray(sp)
    nj, ni = s.shape
    q = np.zeros((nj + 1, ni + 1))
    for j in range(nj + 1):
        for i in range(ni + 1):
            jj = [min(max(j - 1, 0), nj - 1), min(j, nj - 1)]
            ii = [(i - 1) % ni, i % ni]
            q[j, i] = 0.25 * sum(s[a, b] for a in jj for b in ii)
    return q


def test_zero_sponge_is_bit_identical():
    grid, u, v, R, sp = _setup()
    a = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **OM4P5)
    b = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, nu2_sponge_h=jnp.zeros_like(sp), **OM4P5)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_sponge_adds_exactly_a_laplacian_with_croco_corners():
    """The difference with and without the sponge is the stress-operator Laplacian with nu_h = visc2 and
    nu_q = the four-cell mean (it is added after the caps, so the difference is linear in visc2)."""
    grid, u, v, R, sp = _setup(seed=1)
    a = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **OM4P5)
    b = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, nu2_sponge_h=sp, **OM4P5)
    ref = viscous_tendency_cgrid(u, v, grid, jnp.asarray(sp)[..., None], jnp.asarray(_q_mean(sp))[..., None])
    for k in range(2):
        d = np.asarray(b[k]) - np.asarray(a[k])
        assert np.max(np.abs(d - np.asarray(ref[k]))) <= 1e-12 * np.max(np.abs(np.asarray(ref[k])))
        assert np.max(np.abs(d)) > 0.0
    # nothing changes beyond the sponge (cells 10 and up have visc2 = 0; their faces see only zero corners)
    assert np.max(np.abs(np.asarray(b[0]) - np.asarray(a[0]))[12:]) == 0.0


def test_sponge_dissipates_more():
    grid, u, v, R, sp = _setup(seed=2)
    area = np.asarray(grid.area)[..., None]
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)

    def dE(t):
        return float(np.sum(np.asarray(u) * np.asarray(t[0]) * A_u) + np.sum(np.asarray(v) * np.asarray(t[1]) * A_v))
    e0 = dE(om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, **OM4P5))
    e1 = dE(om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, nu2_sponge_h=sp, **OM4P5))
    assert e1 < e0 < 0.0


def test_sponge_needs_the_om4p5_form():
    grid, u, v, R, sp = _setup()
    with pytest.raises(ValueError, match="nu2_sponge_h"):
        om4p25_lateral_friction_tendency_cgrid(u, v, grid, nu2_sponge_h=sp)


def test_gradient_through_the_sponge_is_finite():
    grid, u, v, R, sp = _setup(seed=3)

    def loss(s):
        t = om4p25_lateral_friction_tendency_cgrid(u, v, grid, resolution_fn=R, nu2_sponge_h=s, **OM4P5)
        return jnp.sum(t[0] ** 2) + jnp.sum(t[1] ** 2)
    g = np.asarray(jax.grad(loss)(sp))
    assert np.all(np.isfinite(g)) and np.abs(g).max() > 0.0


def test_model_step_applies_the_sponge_and_rejects_other_schemes():
    """SpongeForcing.visc2 reaches the model step with the om4p25 friction (the step changes); with
    another friction scheme it raises instead of being dropped."""
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.sponge import SpongeForcing
    from legoesm.ocean.state import OMp25Config

    r = build_eady_uniform_setup(n_lat=16, n_lon=16, momentum_advection="vector_invariant")
    lv0 = r.model_config.lateral_viscosity._replace(A_h=0.0, B_h=0.0, C_smag=0.0, C_leith=0.0)
    om4p5 = OMp25Config(Kh_sin_lat=2000.0, Kh_sin_power=4.0, bound_coriolis_vel=6.0,
                        resolution_function="mom6_step_wkb", bound_coef=0.8)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev", dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S), u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    shape2 = s.T.data.shape[:2]
    base = SpongeForcing(gamma=jnp.zeros(shape2), T_ref=s.T.data, S_ref=s.S.data)
    sp = base._replace(visc2=jnp.full(shape2, 5.0e3))
    cfg = r.model_config._replace(lateral_friction_scheme="om4p25", omp25=om4p5, lateral_viscosity=lv0)
    m = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    a, b = m.step(s, 300.0, sponge=base), m.step(s, 300.0, sponge=sp)
    assert bool(jnp.all(jnp.isfinite(b.u.data)))
    assert float(jnp.max(jnp.abs(a.u.data - b.u.data))) > 0.0
    other = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    assert other.config.lateral_friction_scheme != "om4p25"
    with pytest.raises(ValueError, match="visc2"):
        other.step(s, 300.0, sponge=sp)
