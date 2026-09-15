"""Bechtold convective momentum transport reaches the MPAS edge winds.

Before 2026-09-15 the hydrostatic convection bridge handed a CMT-capable
scheme ZERO winds on MPAS (edge-normal staggering) and then zeroed du_dt
through a try/except reshape, so CMT was silently dead on the production
lane. Now the bridge rebuilds cell winds (Perot reconstruction) and projects
the scheme's (du, dv) back onto the edge normals through the shared
``cell_vector_to_edge_normal`` helper, which the turbulence bridge also uses.

``test_convection_cmt_reaches_mpas_edge_winds`` fails on the old bridge
(zero winds in -> zero CMT out); the helper tests pin the projection to the
turbulence bridge's previous inline formula bit for bit.
"""
from __future__ import annotations

import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from legoesm.atmosphere.physics.convection.config import BechtoldConfig, ConvectionConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.integration import make_convection_physics  # noqa: E402
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import MPASHydrostaticState  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.grids.voronoi import cell_vector_to_edge_normal, create_voronoi_mesh  # noqa: E402
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402

NLEV = 20


def test_helper_projects_a_uniform_field_onto_the_edge_normals():
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    u = jnp.full((mesh.nCells, 4), 3.0)
    v = jnp.full((mesh.nCells, 4), -2.0)
    got = cell_vector_to_edge_normal(u, v, mesh)
    expect = (3.0 * jnp.cos(mesh.angleEdge) - 2.0 * jnp.sin(mesh.angleEdge))[:, None]
    np.testing.assert_allclose(np.asarray(got), np.broadcast_to(np.asarray(expect), (mesh.nEdges, 4)), rtol=1e-12)


def test_helper_is_the_turbulence_bridges_former_inline_formula():
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    rng = np.random.default_rng(0)
    du = jnp.asarray(rng.standard_normal((mesh.nCells, 4)))
    dv = jnp.asarray(rng.standard_normal((mesh.nCells, 4)))
    c0, c1 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    angle = mesh.angleEdge[:, None]
    inline = (0.5 * (du[c0] + du[c1])) * jnp.cos(angle) + (0.5 * (dv[c0] + dv[c1])) * jnp.sin(angle)
    np.testing.assert_array_equal(np.asarray(inline), np.asarray(cell_vector_to_edge_normal(du, dv, mesh)))


def _state(mesh):
    sigma = create_sigma_coordinate(NLEV)
    s_full = np.asarray(sigma.sigma_full, dtype=np.float64)
    p_s = 1.0e5
    z = -8500.0 * np.log(s_full)
    T_col = np.maximum(302.0 - 7.5e-3 * z, 200.0)
    q_sfc = 0.8 * float(saturation_mixing_ratio(jnp.asarray(302.0), jnp.asarray(p_s)))
    q_col = q_sfc * np.exp(-z / 3000.0)
    u_cell = jnp.asarray(np.broadcast_to(10.0 * s_full[None, :], (mesh.nCells, NLEV)))
    u_edge = cell_vector_to_edge_normal(u_cell, jnp.zeros_like(u_cell), mesh)
    state = MPASHydrostaticState(
        u=Field(u_edge), T=Field(jnp.asarray(np.broadcast_to(T_col, (mesh.nCells, NLEV)))),
        p_s=Field(jnp.full((mesh.nCells,), p_s)), phis=Field(jnp.zeros((mesh.nCells,))),
        tracers={"q_v": Field(jnp.asarray(np.broadcast_to(q_col, (mesh.nCells, NLEV))))})
    return state, sigma


def _run(enable_cmt, n_calls=8):
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    state, sigma = _state(mesh)
    fn = make_convection_physics(
        ConvectionConfig(scheme="bechtold", bechtold=BechtoldConfig(enable_cmt=enable_cmt)),
        model_type="mpas", dt=600.0)
    prog, tend = None, None
    for _ in range(n_calls):
        ps_obj = None if prog is None else types.SimpleNamespace(
            conv_prog_profile=jnp.asarray(prog), conv_stoch_state=jnp.zeros((mesh.nCells,)))
        tend, out = fn(state, mesh, sigma, phys_state=ps_obj)
        prog = out["conv_prog_profile"]
    return mesh, np.asarray(tend.du_dt.data), np.asarray(out["conv_precip"])


def test_convection_cmt_reaches_mpas_edge_winds():
    mesh, du, pconv = _run(True)
    assert du.shape == (mesh.nEdges, NLEV)
    assert np.all(np.isfinite(du))
    assert pconv.max() > 1e-7, "fixture must convect or the test is vacuous"
    assert np.abs(du).max() > 0.0


def test_convection_cmt_off_gives_exact_zeros_on_the_edges():
    mesh, du, _ = _run(False)
    assert du.shape == (mesh.nEdges, NLEV)
    assert np.all(du == 0.0)
