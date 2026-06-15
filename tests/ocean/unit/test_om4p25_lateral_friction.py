"""OM4p25 lateral-friction closure (Silvestri et al. 2024 "SM2"; GFDL OM4.0).

Exercises ``om4p25_lateral_friction_tendency_cgrid`` directly: the max(Smag,
static) Laplacian + biharmonic combination, the deformation-radius taper F,
energy dissipation (the stress-tensor operator is the exact strain adjoint),
zero-on-rest, finiteness/AD, and the config dispatch through a full model step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _grid(n_lat=12, n_lon=24):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)


def _fields(grid, nlev=3, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, nlev)))
    v = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, nlev)))
    v = v.at[0].set(0.0).at[-1].set(0.0)   # no flow through the poles
    return u, v


def test_rest_state_zero_tendency():
    """u=v=0 → OM4p25 tendency is exactly zero (no strain → no viscosity)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        om4p25_lateral_friction_tendency_cgrid,
    )
    grid = _grid()
    u = jnp.zeros((grid.n_lat, grid.n_lon + 1, 2))
    v = jnp.zeros((grid.n_lat + 1, grid.n_lon, 2))
    tu, tv = om4p25_lateral_friction_tendency_cgrid(u, v, grid)
    assert float(jnp.max(jnp.abs(tu))) == 0.0
    assert float(jnp.max(jnp.abs(tv))) == 0.0


def test_finite_and_shapes():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        om4p25_lateral_friction_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid)
    tu, tv = om4p25_lateral_friction_tendency_cgrid(u, v, grid)
    assert tu.shape == u.shape and tv.shape == v.shape
    assert bool(jnp.all(jnp.isfinite(tu))) and bool(jnp.all(jnp.isfinite(tv)))


def test_energy_dissipation():
    """The closure removes kinetic energy: Σ(u·tend_u·area_u + v·tend_v·area_v) ≤ 0
    (the stress-tensor operator is the exact discrete adjoint of strain → the
    max(Smag,static) non-negative coefficients guarantee dissipation)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        om4p25_lateral_friction_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=2)
    tu, tv = om4p25_lateral_friction_tendency_cgrid(u, v, grid)
    area = np.asarray(grid.area)[..., None]
    # u-face / v-face area weights (avg of adjacent cells).
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)
    dE = float(np.sum(np.asarray(u) * np.asarray(tu) * A_u)
               + np.sum(np.asarray(v) * np.asarray(tv) * A_v))
    assert dE <= 1e-12, f"OM4p25 injected energy: dE/dt = {dE:.3e}"


def test_taper_reduces_laplacian():
    """The deformation-radius factor F = 1/(1+0.25(L_d/Δ)⁴) makes a LARGER L_d
    (better-resolved eddies) damp LESS via the Laplacian part. With the
    biharmonic-only coefficients zeroed, a bigger deformation_radius ⇒ smaller
    |tendency|."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        om4p25_lateral_friction_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=4)
    # Isolate the Laplacian (zero the biharmonic coefficients).
    kw = dict(C4=0.0, Cu4=0.0)
    tu_small, _ = om4p25_lateral_friction_tendency_cgrid(
        u, v, grid, deformation_radius=1.0e3, **kw)
    tu_big, _ = om4p25_lateral_friction_tendency_cgrid(
        u, v, grid, deformation_radius=500.0e3, **kw)
    assert float(jnp.sum(jnp.abs(tu_big))) < float(jnp.sum(jnp.abs(tu_small))), \
        "larger L_d should taper the Laplacian friction down"


def test_max_static_floor():
    """For a near-zero strain field, the Smagorinsky term vanishes and the
    static floors (Cu2·Δ Laplacian, Cu4·Δ³ biharmonic) keep a nonzero
    viscosity → a nonzero tendency on a structured but weak field."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        om4p25_lateral_friction_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=6)
    u_weak, v_weak = u * 1e-8, v * 1e-8       # strain ~1e-8 → Smag ~1e-16, static dominates
    tu, _ = om4p25_lateral_friction_tendency_cgrid(u_weak, v_weak, grid)
    # Static-floor viscosity (Cu·Δ) is independent of strain magnitude, so the
    # tendency scales ~linearly with velocity and is clearly nonzero.
    assert float(jnp.max(jnp.abs(tu))) > 0.0


def test_differentiable():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        om4p25_lateral_friction_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=8)

    def loss(u_):
        tu, _ = om4p25_lateral_friction_tendency_cgrid(u_, v, grid)
        return jnp.sum(tu ** 2)

    g = jax.grad(loss)(u)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_config_dispatch_and_full_step():
    """momentum_advection=vector_invariant + lateral_friction_scheme='om4p25'
    builds, validates, and a full ocean step stays finite (SM2 recipe path)."""
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OMp25Config

    r = build_eady_uniform_setup(n_lat=16, n_lon=16,
                                 momentum_advection="vector_invariant")
    cfg = r.model_config._replace(
        lateral_friction_scheme="om4p25", omp25=OMp25Config(),
        A_h=0.0, B_h=0.0, C_smag=0.0, C_leith=0.0)
    model = LatLonCGridOceanModel(r.grid, r.z_coord, cfg)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 300.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data)))
    assert bool(jnp.all(jnp.isfinite(nxt.v.data)))


def test_invalid_lateral_friction_scheme_rejected():
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = build_eady_uniform_setup(n_lat=12, n_lon=12)
    bad = r.model_config._replace(lateral_friction_scheme="bogus")
    with pytest.raises(ValueError, match="lateral_friction_scheme"):
        LatLonCGridOceanModel(r.grid, r.z_coord, bad)
