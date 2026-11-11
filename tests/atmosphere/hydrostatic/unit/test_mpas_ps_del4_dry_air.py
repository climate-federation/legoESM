"""Surface-pressure hyperdiffusion moves air, not water.

The del4 filter on p_s changes each layer's mass by dB_k * dps_del4 with no
tracer flux.  Unless the tracer tendency carries that air as dry, every
tracer's layer mass q*dp changes with it: at res 6 it removed ~0.009 mm/day
of water (10-day production restart, physics and fixers off).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import mpas_pe_tendencies
from legoesm.core.field import Field
from legoesm.grids.vertical import create_sigma_coordinate, standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh

NU = 1.0e16   # m^4/s, order of the production res-6 value


@pytest.fixture(autouse=True)
def fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _state(coord):
    from legoesm.atmosphere.idealized.held_suarez_topo import (
        held_suarez_topo_init_mpas)
    mesh = create_voronoi_mesh(3)
    s = held_suarez_topo_init_mpas(mesh, coord)
    n, k = s.T.data.shape
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.uniform(1e-6, 1e-2, size=(n, k)))
    tracers = {"q_v": Field(data=q, name="q_v", dims=("nCells", "nlev")),
               "q_c": Field(data=0.1 * q[:, ::-1], name="q_c", dims=("nCells", "nlev"))}
    return mesh, s._replace(tracers=tracers)


def _dB(coord, ps):
    """d(layer mass)/d(p_s), by forward-mode AD of the coordinate's own layer
    thickness -- independent of how the dycore forms its dB."""
    return jax.jvp(coord.layer_thickness_dp, (ps,), (jnp.ones_like(ps),))[1]


def _tracer_mass_rates(state, mesh, coord, nu):
    """Global d/dt of sum(area*q*dp) per tracer, and its gross del4 scale."""
    cfg = MPASPrimitiveEquationConfig(fix_mass=False, nu_del4_ps=nu)
    t = mpas_pe_tendencies(state, mesh, coord, cfg, vertical_form="legacy")
    ps = state.p_s.data
    dp = coord.layer_thickness_dp(ps)
    dB = _dB(coord, ps)
    area = jnp.asarray(mesh.areaCell)[:, None]
    out = {}
    for k, f in state.tracers.items():
        out[k] = float(jnp.sum(area * (dp * t.tracer_tendencies[k].data
                                       + f.data * dB * t.dp_s_dt.data[:, None])))
    return out, t


@pytest.mark.parametrize("coord_fn", [lambda: standard_hybrid_levels(10),
                                      lambda: create_sigma_coordinate(10)],
                         ids=["hybrid", "sigma"])
def test_ps_del4_leaves_tracer_mass_unchanged(coord_fn):
    coord = coord_fn()
    mesh, state = _state(coord)
    r0, _ = _tracer_mass_rates(state, mesh, coord, 0.0)
    r1, t1 = _tracer_mass_rates(state, mesh, coord, NU)
    _, t0 = _tracer_mass_rates(state, mesh, coord, 0.0)
    ps = state.p_s.data
    dB = _dB(coord, ps)
    dps_del4 = t1.dp_s_dt.data - t0.dp_s_dt.data
    area = jnp.asarray(mesh.areaCell)[:, None]
    for k, f in state.tracers.items():
        gross = float(jnp.sum(area * jnp.abs(f.data * dB * dps_del4[:, None])))
        assert gross > 0.0, k                       # control: the filter acts
        # the filter's net tracer-mass change, with the fix, is rounding
        assert abs(r1[k] - r0[k]) <= 1e-12 * gross, (k, r1[k] - r0[k], gross)
        # and it cancels in every cell and layer, not only in the global sum
        dp = coord.layer_thickness_dp(ps)
        local = (dp * (t1.tracer_tendencies[k].data - t0.tracer_tendencies[k].data)
                 + f.data * dB * dps_del4[:, None])
        scale = jnp.abs(f.data * dB * dps_del4[:, None])
        assert bool(jnp.all(jnp.abs(local) <= 1e-12 * jnp.max(scale))), k


def test_air_mass_still_filtered_and_gradient_finite():
    coord = standard_hybrid_levels(10)
    mesh, state = _state(coord)
    _, t0 = _tracer_mass_rates(state, mesh, coord, 0.0)
    _, t1 = _tracer_mass_rates(state, mesh, coord, NU)
    # the p_s filter itself is unchanged by the fix and still acts
    assert float(jnp.max(jnp.abs(t1.dp_s_dt.data - t0.dp_s_dt.data))) > 0.0

    def loss(qv):
        s = state._replace(tracers={**state.tracers,
                                    "q_v": state.tracers["q_v"].replace(data=qv)})
        cfg = MPASPrimitiveEquationConfig(fix_mass=False, nu_del4_ps=NU)
        return jnp.sum(mpas_pe_tendencies(s, mesh, coord, cfg, vertical_form="legacy")
                       .tracer_tendencies["q_v"].data ** 2)

    g = jax.grad(loss)(state.tracers["q_v"].data)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0
