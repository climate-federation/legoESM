"""THE closure gate for the interior/rim split (codex blocker ii).

The FULL hydrostatic tendency — vertex PV with kite areas, energy PV
flux, APVM, del4 hyperdiffusion — evaluated on the rim submesh must
reproduce the GLOBAL-mesh tendency at every rim entity. Divergence and
gradient passing (test_rim_plan) does NOT establish this: the submesh
keeps a vertex when ANY incident cell is local, so partially-closed
vertices read -1-masked neighbours against a global kite/area
denominator, and edgesOnEdge tangential paths can leave the closure.

Parameterized by closure (stencil) depth: the MINIMUM green depth is
the poison-verified width the wiring must use. A depth expected to be
insufficient is asserted RED so the gate is provably non-vacuous.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

N_DEV = 6
SUBDIV = 3
HALO_DEPTH = 6      # device partition depth: room for rim+closure
RIM_WIDTH = 2
NLEV = 4


@pytest.fixture(scope="module")
def global_setup():
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import (
        _build_rim_rings, _build_voronoi_partition_infra,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    mesh = create_voronoi_mesh(subdivision_level=SUBDIV)
    if mesh.nCells % N_DEV or mesh.nEdges % N_DEV:
        pytest.skip("mesh not divisible")
    sigma = create_sigma_coordinate(NLEV)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     _co) = _build_voronoi_partition_infra(
        mesh, N_DEV, halo_depth=HALO_DEPTH)
    cell_rim, _ = _build_rim_rings(mesh, partitions, max_lc, max_le,
                                   RIM_WIDTH)
    return mesh, sigma, state, partitions, cell_rim


def _sub_state(state, part, plan):
    """Gather the submesh state out of the device's local buffers —
    the same data path the wired split will use."""
    import jax.numpy as jnp

    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)

    def sub(field, gather_local, local_ents):
        return field.replace(
            data=jnp.asarray(np.asarray(field.data)[local_ents])[
                jnp.asarray(gather_local)])

    return state._replace(
        u=sub(state.u, plan["edge_gather"], le),
        T=sub(state.T, plan["cell_gather"], lc),
        p_s=sub(state.p_s, plan["cell_gather"], lc),
        phis=sub(state.phis, plan["cell_gather"], lc),
    )


def _tendency(state, mesh, sigma):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, mpas_hydrostatic_tendencies,
    )
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=False,
        pv_scheme="energy", time_integrator="ssp_rk3",
    )
    return mpas_hydrostatic_tendencies(state, mesh, sigma, cfg, dt=600.0)


def _max_rim_err(depth, global_setup):
    from legoesm.parallel.sharded_dynamics import _build_rim_plan

    mesh, sigma, state, partitions, cell_rim = global_setup
    plans = _build_rim_plan(mesh, partitions, cell_rim, RIM_WIDTH, depth)
    tend_g = _tendency(state, mesh, sigma)
    du_g = np.asarray(tend_g.du_dt.data)
    dT_g = np.asarray(tend_g.dT_dt.data)

    worst = 0.0
    for part, plan in zip(partitions, plans):
        if plan["n_rim_cells"] == 0:
            continue
        s_sub = _sub_state(state, part, plan)
        tend_s = _tendency(s_sub, plan["sub_mesh"], sigma)
        rim_cells_g = np.asarray(part.local_cells)[plan["cell_scatter"]]
        rim_edges_g = np.asarray(part.local_edges)[plan["edge_scatter"]]
        eT = np.max(np.abs(
            np.asarray(tend_s.dT_dt.data)[: plan["n_rim_cells"]]
            - dT_g[rim_cells_g]))
        eu = np.max(np.abs(
            np.asarray(tend_s.du_dt.data)[: plan["n_rim_edges"]]
            - du_g[rim_edges_g]))
        worst = max(worst, float(eT), float(eu))
    return worst


def test_shallow_closure_is_red(global_setup):
    """Depth 1 must FAIL — a gate that cannot go red proves nothing."""
    err = _max_rim_err(1, global_setup)
    print(f"depth-1 closure err: {err:.3e}")
    assert err > 1e-8, (
        f"depth-1 closure unexpectedly exact (err {err:.2e}) — the gate "
        f"is vacuous or the RHS stencil is narrower than believed")


def test_full_tendency_closure_depth(global_setup):
    """Find and pin the minimum green closure depth for the FULL RHS."""
    # Threshold calibrated from measurement (2026-08-18): depths 2-4
    # PLATEAU at 1.39e-10 — the f64 re-association floor of a tendency
    # with nu_del4=1e16 — while depth 1 sits orders above. A missing
    # dependency would keep improving with depth; a plateau is closure.
    errs = {}
    for depth in (2, 3, 4):
        errs[depth] = _max_rim_err(depth, global_setup)
        if errs[depth] < 1e-9:
            break
    green = [d for d, e in errs.items() if e < 1e-9]
    assert green, (
        f"no closure depth up to 4 reproduces the full tendency at rim "
        f"entities: errs={ {d: f'{e:.2e}' for d, e in errs.items()} } — "
        f"closure needs dependency-graph growth, not just depth")
    print(f"closure errs by depth: { {d: f'{e:.2e}' for d, e in errs.items()} }; "
          f"min green depth = {min(green)}")
