"""Phase-1 tests for ``_build_rim_plan`` — the compact rim submesh the
interior/rim split runs the UNCHANGED RHS operators on.

The pinned property: an operator evaluated on the submesh with fields
gathered from the device's local buffers reproduces the GLOBAL-mesh
values at every rim entity. (float64; tolerance is re-association only.)
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def rim_setup():
    return _setup()


def _setup(n_dev=6, subdivision=3, halo_depth=4, rim_width=2,
           stencil_depth=2):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.sharded_dynamics import (
        _build_rim_plan, _build_rim_rings,
        _build_voronoi_partition_infra,
    )

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    if mesh.nCells % n_dev or mesh.nEdges % n_dev:
        pytest.skip("mesh not divisible")
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     _cell_owner) = _build_voronoi_partition_infra(
        mesh, n_dev, halo_depth=halo_depth)
    cell_rim, edge_rim = _build_rim_rings(mesh, partitions, max_lc,
                                          max_le, rim_width)
    plans = _build_rim_plan(mesh, partitions, cell_rim, edge_rim,
                            rim_width, stencil_depth)
    return mesh, partitions, plans


def test_divergence_on_submesh_matches_global_at_rim_cells(rim_setup):
    import jax.numpy as jnp
    from legoesm.core.operators_voronoi import divergence_cell_3d

    mesh, partitions, plans = rim_setup
    nlev = 3
    rng = np.random.default_rng(0)
    u_global = jnp.asarray(rng.normal(size=(mesh.nEdges, nlev)))
    div_global = np.asarray(divergence_cell_3d(u_global, mesh))

    checked = 0
    for d, (part, plan) in enumerate(zip(partitions, plans)):
        if plan["n_rim_cells"] == 0:
            continue
        # Device-local edge buffer (owned+halo), then gather the submesh
        # ordering out of it — the same data path the split will use.
        u_local = u_global[np.asarray(part.local_edges)]
        u_sub = u_local[plan["edge_gather"]]
        div_sub = np.asarray(divergence_cell_3d(u_sub, plan["sub_mesh"]))

        rim_globals = np.asarray(
            part.local_cells)[plan["cell_scatter"]]
        np.testing.assert_allclose(
            div_sub[: plan["n_rim_cells"]], div_global[rim_globals],
            rtol=1e-12, atol=1e-12,
            err_msg=f"device {d}: submesh divergence != global at rim")
        checked += plan["n_rim_cells"]
    assert checked > 0, "no rim cells checked — setup degenerate"


def test_gradient_on_submesh_matches_global_at_rim_edges(rim_setup):
    import jax.numpy as jnp
    from legoesm.core.operators_voronoi import gradient_edge_3d

    mesh, partitions, plans = rim_setup
    nlev = 2
    rng = np.random.default_rng(1)
    f_global = jnp.asarray(rng.normal(size=(mesh.nCells, nlev)))
    g_global = np.asarray(gradient_edge_3d(f_global, mesh))

    checked = 0
    for d, (part, plan) in enumerate(zip(partitions, plans)):
        if plan["n_rim_edges"] == 0:
            continue
        f_local = f_global[np.asarray(part.local_cells)]
        f_sub = f_local[plan["cell_gather"]]
        g_sub = np.asarray(gradient_edge_3d(f_sub, plan["sub_mesh"]))

        rim_edge_globals = np.asarray(
            part.local_edges)[plan["edge_scatter"]]
        np.testing.assert_allclose(
            g_sub[: plan["n_rim_edges"]], g_global[rim_edge_globals],
            rtol=1e-12, atol=1e-12,
            err_msg=f"device {d}: submesh gradient != global at rim")
        checked += plan["n_rim_edges"]
    assert checked > 0


def test_closure_overflow_raises():
    """rim_width + stencil_depth beyond the partition halo must raise,
    not silently gather garbage rows."""
    from legoesm.parallel.sharded_dynamics import (
        _build_rim_plan, _build_rim_rings,
        _build_voronoi_partition_infra,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(subdivision_level=3)
    n_dev = 6
    if mesh.nCells % n_dev or mesh.nEdges % n_dev:
        pytest.skip("mesh not divisible")
    (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
     _co) = _build_voronoi_partition_infra(mesh, n_dev, halo_depth=1)
    cell_rim, edge_rim = _build_rim_rings(mesh, partitions, max_lc,
                                          max_le, 1)
    with pytest.raises(ValueError, match="closure leaves"):
        _build_rim_plan(mesh, partitions, cell_rim, edge_rim, 1, 4)


def test_edge_scatter_is_device_owned_and_rim_predicated(rim_setup):
    """codex blocker (i): scatter targets must be DEVICE-owned edge rows
    selected by the rim predicate 0 <= edge_rim <= width — never the
    synthetic partition's edge-ownership, never a halo row."""
    from legoesm.parallel.sharded_dynamics import _build_rim_rings

    mesh, partitions, plans = rim_setup
    # recompute edge_rim exactly as _setup did (width 2)
    max_lc = max(p_.n_local_cells for p_ in partitions)
    max_le = max(p_.n_local_edges for p_ in partitions)
    _, edge_rim = _build_rim_rings(mesh, partitions, max_lc, max_le, 2)

    for d, (part, plan) in enumerate(zip(partitions, plans)):
        es = plan["edge_scatter"]
        assert (es < part.n_owned_edges).all(), (
            f"device {d}: edge_scatter names a halo row")
        want = np.sort(np.where(
            (edge_rim[d, :part.n_owned_edges] >= 0)
            & (edge_rim[d, :part.n_owned_edges] <= 2))[0])
        got = np.sort(es)
        np.testing.assert_array_equal(got, want, err_msg=(
            f"device {d}: edge_scatter != rim-predicated owned rows"))


def test_stack_rim_plans_shapes_and_sentinels(rim_setup):
    """Stacked plans: uniform shapes, -1 pad sentinels only in padded
    lanes, true rows preserved verbatim."""
    import numpy as np
    from legoesm.parallel.sharded_dynamics import _stack_rim_plans

    mesh, partitions, plans = rim_setup
    st = _stack_rim_plans(plans)
    n_dev = len(plans)
    assert st["cell_gather"].shape[0] == n_dev
    for d, plan in enumerate(plans):
        ncg = len(plan["cell_gather"])
        np.testing.assert_array_equal(
            st["cell_gather"][d, :ncg], plan["cell_gather"])
        assert (st["cell_gather"][d, ncg:] == -1).all()
        nsc = len(plan["cell_scatter"])
        np.testing.assert_array_equal(
            st["cell_scatter"][d, :nsc], plan["cell_scatter"])
        assert (st["cell_scatter"][d, nsc:] == -1).all()
        assert st["n_rim_cells"][d] == plan["n_rim_cells"]
    # stacked submesh leading dim = device axis, uniform trailing shapes
    assert st["sub_mesh"].cellsOnEdge.shape[0] == n_dev
