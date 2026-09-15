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


def test_mass_weighted_projection_conserves_column_momentum_on_hybrid_levels():
    """codex P1: with differing surface pressures the plain two-cell average of
    a cell-conservative tendency is NOT conservative against the dycore's
    edge layer mass (the two-cell mean of dp); the mass-weighted form is."""
    from legoesm.grids.vertical import create_hybrid_coordinate
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    nlev = 12
    s_h = np.linspace(0.02, 1.0, nlev + 1)
    a = 0.4 * s_h * (1.0 - s_h)
    coord = create_hybrid_coordinate(nlev, jnp.asarray(a), jnp.asarray(s_h - a))
    rng = np.random.default_rng(1)
    p_s = jnp.asarray(7.0e4 + 3.0e4 * rng.random(mesh.nCells))
    p_half = np.asarray(coord.pressure_at_half(p_s), dtype=np.float64)
    dp = p_half[:, 1:] - p_half[:, :-1]
    du = rng.standard_normal((mesh.nCells, nlev))
    dv = rng.standard_normal((mesh.nCells, nlev))
    du -= (du * dp).sum(1, keepdims=True) / dp.sum(1, keepdims=True)   # mass-weighted column mean zero
    dv -= (dv * dp).sum(1, keepdims=True) / dp.sum(1, keepdims=True)
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    dp_edge = 0.5 * (dp[c0] + dp[c1])
    plain = np.asarray(cell_vector_to_edge_normal(jnp.asarray(du), jnp.asarray(dv), mesh))
    weighted = np.asarray(cell_vector_to_edge_normal(jnp.asarray(du), jnp.asarray(dv), mesh, dp_cell=jnp.asarray(dp)))
    scale = np.abs(dp_edge * weighted).sum(1).max()
    assert np.abs((dp_edge * weighted).sum(1)).max() < 1e-12 * scale
    assert np.abs((dp_edge * plain).sum(1)).max() > 1e-6 * scale, "the plain average must fail this, or the test is vacuous"


def test_cmt_kernel_conserves_column_momentum():
    """codex P1 (pre-existing): the kernel repeated the last flux at the bottom
    and clipped per level, leaving -flux_last in the column.  Now the bottom
    flux is zero and the cap is column-uniform: sum dp*du_dt = 0 to round-off
    for a sheared, convecting column, before and after the cap binds."""
    from legoesm.atmosphere.physics.convection._plume import cmt_gregory_1997
    ncol, nlev = 4, 20
    rng = np.random.default_rng(2)
    p_half = np.linspace(1.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1]); dp = p_half[:, 1:] - p_half[:, :-1]
    from legoesm import constants
    rho = p_full / (constants.R_d * 280.0)
    u = 10.0 * np.linspace(1.0, 0.0, nlev)[None, :] + rng.standard_normal((ncol, nlev))
    v = 3.0 * np.linspace(0.0, 1.0, nlev)[None, :]
    M_u = 0.05 * np.exp(-((np.arange(nlev) - 12) / 4.0) ** 2)[None, :] * np.ones((ncol, 1))
    for amp in (1.0, 200.0):   # the second drives the cap
        du, dv = cmt_gregory_1997(jnp.asarray(u * amp), jnp.asarray(v * amp), jnp.asarray(M_u), None,
                                  jnp.asarray(p_full), jnp.asarray(p_half), jnp.asarray(rho), c_u=0.7, c_d=0.7)
        du, dv = np.asarray(du), np.asarray(dv)
        assert np.abs(du).max() > 0.0
        col = (dp * du).sum(1); scale = np.abs(dp * du).sum(1).max()
        assert np.abs(col).max() < 1e-12 * scale, col
        assert np.abs((dp * dv).sum(1)).max() < 1e-12 * np.abs(dp * dv).sum(1).max()


def test_cmt_kernel_is_down_gradient_and_lands_on_the_sheared_interface():
    """Layer placement and sign (GLM review): a single wind step between
    layers 11 and 12 (faster below, trade-like) must accelerate the slow
    layer above and decelerate the fast layer below by the same momentum,
    and touch no other layer."""
    from legoesm import constants
    from legoesm.atmosphere.physics.convection._plume import cmt_gregory_1997
    nlev = 20
    p_half = np.linspace(1.0e4, 1.0e5, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1]); dp = p_half[:, 1:] - p_half[:, :-1]
    rho = p_full / (constants.R_d * 280.0)
    u = np.where(np.arange(nlev)[None, :] >= 12, 5.0, 0.0)
    du, _ = cmt_gregory_1997(jnp.asarray(u), jnp.zeros_like(jnp.asarray(u)), jnp.asarray(np.full((1, nlev), 0.05)),
                             None, jnp.asarray(p_full), jnp.asarray(p_half), jnp.asarray(rho), c_u=0.7, c_d=0.7)
    du = np.asarray(du)[0]
    assert du[11] > 0.0 and du[12] < 0.0
    np.testing.assert_allclose(du[11] * dp[0, 11], -du[12] * dp[0, 12], rtol=1e-12)
    others = np.delete(du, [11, 12])
    assert np.all(others == 0.0)
