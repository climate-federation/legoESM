"""QG-Leith harmonic viscosity (Bachman et al. 2017; Silvestri et al. 2024 "QG2").

Exercises ``qg_leith_viscosity_tendency_cgrid`` (the barotropic ν=(C·Δ/π)³·
√(|∇(ζ+f)|²+|∇δ|²) Laplacian closure) and the ``bound_qg_pv_gradient`` helper
(the Bu/Ro min bound, ready for the future buoyancy-stretching wiring):
energy dissipation, rest-zero, absolute-vs-relative vorticity distinction,
finiteness/AD, the bound math, and the config dispatch through a model step.
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


def _fields(grid, nlev=2, seed=0):
    rng = np.random.default_rng(seed)
    u = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, nlev)))
    v = jnp.asarray(0.2 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, nlev)))
    v = v.at[0].set(0.0).at[-1].set(0.0)
    return u, v


def test_rest_state_zero_tendency():
    """u=v=0 → ζ=0, δ=0; |∇(ζ+f)| from f alone is constant in lon so its h-point
    gradient magnitude is the β term only. With zero velocity the stress operator
    sees zero strain → zero tendency."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u = jnp.zeros((grid.n_lat, grid.n_lon + 1, 2))
    v = jnp.zeros((grid.n_lat + 1, grid.n_lon, 2))
    tu, tv = qg_leith_viscosity_tendency_cgrid(u, v, grid)
    assert float(jnp.max(jnp.abs(tu))) == 0.0
    assert float(jnp.max(jnp.abs(tv))) == 0.0


def test_finite_and_shapes():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid)
    tu, tv = qg_leith_viscosity_tendency_cgrid(u, v, grid)
    assert tu.shape == u.shape and tv.shape == v.shape
    assert bool(jnp.all(jnp.isfinite(tu))) and bool(jnp.all(jnp.isfinite(tv)))


def test_energy_dissipation():
    """QG-Leith removes kinetic energy (energy-stable stress-tensor operator)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=3)
    tu, tv = qg_leith_viscosity_tendency_cgrid(u, v, grid)
    area = np.asarray(grid.area)[..., None]
    a_uc = 0.5 * (area + np.roll(area, 1, axis=1))
    A_u = np.concatenate([a_uc, a_uc[:, 0:1]], axis=1)
    a_vc = 0.5 * (area[:-1] + area[1:])
    A_v = np.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)
    dE = float(np.sum(np.asarray(u) * np.asarray(tu) * A_u)
               + np.sum(np.asarray(v) * np.asarray(tv) * A_v))
    assert dE <= 1e-12, f"QG-Leith injected energy: dE/dt = {dE:.3e}"


def test_absolute_vorticity_floor():
    """QG-Leith uses Q=ζ+f, so for WEAK flow the planetary gradient |∇f|=β sets a
    velocity-INDEPENDENT viscosity floor ν≈(C·Δ/π)³·β. The tendency then scales
    ~LINEARLY with velocity (ν floored, strain ∝ u), unlike a pure-relative-
    vorticity Leith where ν∝|∇ζ|∝u gives a QUADRATIC tendency. Halving a weak u
    halves the tendency (ratio→2), not quarters it (ratio→4)."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=5)
    eps = 1e-6                                   # weak flow → β-floor dominates ν
    tu1, _ = qg_leith_viscosity_tendency_cgrid(eps * u, eps * v, grid)
    tu2, _ = qg_leith_viscosity_tendency_cgrid(2 * eps * u, 2 * eps * v, grid)
    ratio = float(jnp.max(jnp.abs(tu2))) / (float(jnp.max(jnp.abs(tu1))) + 1e-300)
    assert 1.8 < ratio < 2.2, f"expected ~linear (f-floor), got ratio {ratio:.3f}"


def test_differentiable():
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        qg_leith_viscosity_tendency_cgrid,
    )
    grid = _grid()
    u, v = _fields(grid, nlev=1, seed=7)

    def loss(u_):
        tu, _ = qg_leith_viscosity_tendency_cgrid(u_, v, grid)
        return jnp.sum(tu ** 2)

    g = jax.grad(loss)(u)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0


class TestBoundQGPVGradient:
    """The Bachman Bu/Ro min-bound helper (ready for the stretching wiring)."""

    def test_min_of_three(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import bound_qg_pv_gradient
        grad_q = jnp.array(1.0)
        # stretching makes the full gradient large; Bu/Ro bounds cap it.
        grad_q_stretch = jnp.array(10.0)
        Bu = jnp.array(1.0)        # gq2 = 1·(1+1) = 2
        Ro = jnp.array(1.0)        # gq3 = 1·(1+1) = 2
        out = float(bound_qg_pv_gradient(grad_q, grad_q_stretch, Bu, Ro))
        assert abs(out - 2.0) < 1e-12, out   # min(10, 2, 2) = 2 (bound active)

    def test_no_bound_when_stretch_small(self):
        from legoesm.ocean.dynamics.latlon_cgrid_operators import bound_qg_pv_gradient
        grad_q = jnp.array(1.0)
        grad_q_stretch = jnp.array(1.0)        # no enhancement
        Bu = jnp.array(1e6)                     # gq2 ≈ 1
        Ro = jnp.array(1e6)                     # gq3 ≈ 1
        out = float(bound_qg_pv_gradient(grad_q, grad_q_stretch, Bu, Ro))
        assert abs(out - 1.0) < 1e-6, out       # min ≈ 1 (no bound)


def test_config_dispatch_and_full_step():
    """lateral_friction_scheme='qg_leith' builds, validates, full step finite."""
    from legoesm.core.field import Field
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    r = build_eady_uniform_setup(n_lat=16, n_lon=16,
                                 momentum_advection="vector_invariant")
    cfg = r.model_config._replace(
        lateral_friction_scheme="qg_leith", qg_leith_coeff=2.0,
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


def test_qg_leith_with_nonzero_other_friction_rejected():
    from legoesm.ocean.experiments.eady_uniform import build_eady_uniform_setup
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = build_eady_uniform_setup(n_lat=12, n_lon=12,
                                 momentum_advection="vector_invariant")
    bad = r.model_config._replace(lateral_friction_scheme="qg_leith", A_h=1.0e4)
    with pytest.raises(ValueError, match="sole lateral friction"):
        LatLonCGridOceanModel(r.grid, r.z_coord, bad)
