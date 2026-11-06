"""The column network as an MPAS physics_fn (WB MPAS core).

Pinned: the network's cell (east, north) wind tendency reaches the edge
normals as u_e cos(angleEdge) + v_n sin(angleEdge) (no swap, no sign flip);
the network is fed the cell winds reconstructed from the edge normals; tracer
tendencies only for the state's tracers; no surface-pressure tendency.
Needs JAX_ENABLE_X64=1.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics import neural_physics as nph  # noqa: E402
from tests.unit.test_learned_wind_heads import (  # noqa: E402
    NLEV, RAW_U, RAW_V, _expected_wind, _net, _with_wind_bias,
)


@pytest.fixture(scope="module")
def setup():
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    nc, ne = int(mesh.nCells), int(mesh.nEdges)
    rng = np.random.default_rng(0)
    prof = np.linspace(0.0, 1.0, NLEV)
    state = MPASHydrostaticState(
        u=Field(jnp.asarray(5.0 * rng.normal(size=(ne, NLEV))), name="u"),
        T=Field(jnp.asarray(np.tile(250.0 + 40.0 * prof, (nc, 1))), name="T"),
        p_s=Field(jnp.full((nc,), 1.0e5), name="p_s"),
        phis=Field(jnp.zeros((nc,)), name="phis"),
        tracers={"q_v": Field(jnp.full((nc, NLEV), 5e-3), name="q_v"),
                 "q_c": Field(jnp.zeros((nc, NLEV)), name="q_c")})
    forcing = {"sst": jnp.full((nc,), 295.0), "sic": jnp.zeros((nc,)),
               "day_of_year": jnp.asarray(80.0),
               "seconds_of_day": jnp.asarray(43200.0),
               "s_0": jnp.asarray(constants.S_0), "land_frac": None}
    return mesh, create_sigma_coordinate(NLEV), state, forcing


def _cell_out(net, mesh, state, forcing, uu=None, vv=None):
    """The network's cell outputs, fed the reconstructed cell winds."""
    from legoesm.core.grid_adapters import make_adapter
    from legoesm.grids.voronoi import reconstruct_cell_velocity

    u, v = reconstruct_cell_velocity(state.u.data, mesh)
    uu = u if uu is None else uu
    vv = v if vv is None else vv
    nc = int(mesh.nCells)
    z3, z2 = jnp.zeros((nc, NLEV)), jnp.zeros((nc,))
    su = nph.make_neural_step_unified(net, make_adapter(mesh))
    return su(True, state.T.data, state.p_s.data, state.tracers["q_v"].data,
              state.tracers["q_c"].data, z3, uu, vv, forcing["sst"],
              forcing["sic"], mesh.latCell, mesh.lonCell,
              forcing["day_of_year"], forcing["seconds_of_day"], None, None,
              forcing["s_0"], None, None, z3, z2, z2, z2, z2, z2,
              phis=state.phis.data)[0]


def test_cell_wind_tendency_projects_onto_the_edge_normals(setup):
    """Edge tendency = (two-cell mean of the east tendency) cos(angleEdge)
    + (two-cell mean of the north tendency) sin(angleEdge), written out
    here in numpy; the biased heads make east and north distinct and ~1e-4."""
    mesh, sigma, state, forcing = setup
    net = _with_wind_bias(_net())
    t = nph.make_neural_mpas_physics_fn(net, mesh)(state, mesh, sigma,
                                                   forcing=forcing)
    out = _cell_out(net, mesh, state, forcing)
    du, dv = np.asarray(out.du_dt), np.asarray(out.dv_dt)
    # the heads carry the bias (the projection below is not of zeros)
    np.testing.assert_allclose(du.mean(0), _expected_wind(RAW_U), rtol=1e-3)
    np.testing.assert_allclose(dv.mean(0), _expected_wind(RAW_V), rtol=1e-3)
    c0, c1 = (np.asarray(mesh.cellsOnEdge)[i] for i in (0, 1))
    a = np.asarray(mesh.angleEdge)[:, None]
    want = (0.5 * (du[c0] + du[c1]) * np.cos(a)
            + 0.5 * (dv[c0] + dv[c1]) * np.sin(a))
    # The mesh stores angleEdge in float32: jnp and numpy float32 cos differ
    # by ~1 ulp (measured max |diff| 5.9e-12 on ~1e-4 tendencies), so the
    # bound is 1e-6 of the tendency scale; a swap or sign error is O(1e-4).
    np.testing.assert_allclose(np.asarray(t.du_dt.data), want,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(t.tracer_tendencies["q_v"].data),
                               np.asarray(out.dq_v_dt), rtol=1e-12, atol=0)
    assert t.dv_dt is None
    assert set(t.tracer_tendencies) == {"q_v", "q_c"}
    assert float(jnp.abs(t.dp_s_dt.data).max()) == 0.0


def test_network_sees_the_reconstructed_cell_winds(setup):
    mesh, sigma, state, forcing = setup
    net = _net()
    # every weight live, so the T tendency depends on the input winds
    leaves, tree = jax.tree_util.tree_flatten(eqx.filter(net, eqx.is_array))
    rng = np.random.default_rng(3)
    net = eqx.combine(jax.tree_util.tree_unflatten(
        tree, [jnp.asarray(0.3 * rng.normal(size=x.shape)) for x in leaves]),
        eqx.filter(net, lambda x: not eqx.is_array(x)))
    got = nph.make_neural_mpas_physics_fn(net, mesh)(
        state, mesh, sigma, forcing=forcing).dT_dt.data
    from legoesm.grids.voronoi import reconstruct_cell_velocity
    u, v = reconstruct_cell_velocity(state.u.data, mesh)
    ref = _cell_out(net, mesh, state, forcing).dT_dt
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                               rtol=1e-12, atol=1e-18)
    # the comparison resolves the wind inputs (a swap is visible)
    swapped = _cell_out(net, mesh, state, forcing, uu=v, vv=u).dT_dt
    assert float(jnp.abs(swapped - ref).max()) > 0.0
