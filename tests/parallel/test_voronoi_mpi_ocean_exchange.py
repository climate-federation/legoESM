"""Direct tests for ``voronoi_mpi.exchange_state_mpas_ocean`` (M3d inc-1).

Single-process coverage of the packed MPAS-OCEAN state halo refresh.
The np=1 empty-neighbor schedule makes ``batched_halo_exchange`` return
before packing, so identity tests alone CANNOT detect an omitted or
mis-wired field (codex finding 4) — the five-field wiring (u edge; T, S,
eta, w cells, in that order) is therefore locked by a MOCKED exchange
returning distinct per-field sentinels, plus the np=1 identity,
static-field pass-through, schema tripwire, hand-built-layout schedule
rebuild, and jit-compatibility tests.  What this file does NOT cover:
real multi-rank transport semantics — those ride
``tests/distributed/test_voronoi_batched_halo.py`` (the generic batched
transport) and the np>=2 bench-lane smoke runs
(``scripts/bench/bench_ocean_mpas_scaling.py --halo-refresh per_step``),
which are cluster jobs, not CI gates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.parallel.voronoi_mpi import (
    exchange_state_mpas_ocean,
    make_voronoi_partition_layout,
)

# After the imports (E402): arrays are only built inside the fixture,
# which runs long after this module-scope config takes effect.
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def single_rank_setup():
    mesh = create_voronoi_mesh(2)
    z = create_ocean_z_star(n_levels=3)
    state = rest_state_mpas_ocean(mesh, z)
    rng = np.random.default_rng(3)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            rng.standard_normal(state.u.data.shape))),
        T=state.T.replace(data=state.T.data + jnp.asarray(
            rng.standard_normal(state.T.data.shape))),
        S=state.S.replace(data=state.S.data + jnp.asarray(
            0.1 * rng.standard_normal(state.S.data.shape))),
        eta=state.eta.replace(data=jnp.asarray(
            0.01 * rng.standard_normal(state.eta.data.shape))),
        w=state.w.replace(data=jnp.asarray(
            1e-4 * rng.standard_normal(state.w.data.shape))),
    )
    layout = make_voronoi_partition_layout(mesh, rank=0, n_ranks=1,
                                           method="geometric")
    return mesh, state, layout


def test_np1_identity_all_fields(single_rank_setup):
    _mesh, state, layout = single_rank_setup
    # np=1: the union-neighbor schedule is empty -> documented identity.
    assert layout.batched_comm is not None
    assert len(layout.batched_comm.neighbor_ranks) == 0
    out = exchange_state_mpas_ocean(state, layout)
    for name in state._fields:
        want = getattr(state, name)
        got = getattr(out, name)
        if want is None:
            assert got is None, name
            continue
        np.testing.assert_array_equal(
            np.asarray(got.data), np.asarray(want.data),
            err_msg=f"np=1 exchange must be the identity on {name}")


def test_static_fields_pass_through_by_reference(single_rank_setup):
    _mesh, state, layout = single_rank_setup
    out = exchange_state_mpas_ocean(state, layout)
    # H_bathy / land_mask / rho_ref_z are static after scatter — the
    # helper must not rebuild (or worse, exchange) them.
    assert out.H_bathy is state.H_bathy
    assert out.land_mask is state.land_mask
    assert out.rho_ref_z is state.rho_ref_z


def test_hand_built_layout_rebuilds_schedule(single_rank_setup):
    _mesh, state, layout = single_rank_setup
    # Layouts constructed by hand may carry batched_comm=None — the
    # helper rebuilds the schedule (mirrors make_voronoi_mpi_step).
    bare = layout._replace(batched_comm=None)
    out = exchange_state_mpas_ocean(state, bare)
    np.testing.assert_array_equal(
        np.asarray(out.T.data), np.asarray(state.T.data))


def test_schema_drift_tripwire(single_rank_setup):
    _mesh, _state, layout = single_rank_setup

    class _Drifted:
        _fields = ("u", "T", "S", "eta", "w", "H_bathy", "land_mask",
                   "rho_ref_z", "new_prognostic")

    with pytest.raises(ValueError, match="schema changed"):
        exchange_state_mpas_ocean(_Drifted(), layout)


def test_jit_compatible(single_rank_setup):
    _mesh, state, layout = single_rank_setup
    fn = jax.jit(lambda s: exchange_state_mpas_ocean(s, layout))
    out = fn(state)
    np.testing.assert_array_equal(
        np.asarray(out.S.data), np.asarray(state.S.data))


def test_mocked_exchange_routes_all_five_fields(single_rank_setup,
                                                monkeypatch):
    """Kill the omitted-field blind spot (codex finding 4).

    np=1 pass-through makes every identity test insensitive to the field
    tuple at the ``batched_halo_exchange`` call site: dropping ``w`` or
    ``eta`` (or swapping T/S) would still return the unchanged state.
    Mock the exchange to return a DISTINCT sentinel per slot and assert
    (a) all five prognostic fields are handed over, in the documented
    (u | T, S, eta, w) order, by array identity, and (b) each output
    state slot lands ITS OWN sentinel — an omitted field breaks the
    tuple arity, a swapped field lands the wrong sentinel.
    """
    _mesh, state, layout = single_rank_setup
    import legoesm.parallel.voronoi_mpi as vm

    calls = []

    def fake_exchange(edge_fields, cell_fields, sched, rank):
        calls.append((tuple(edge_fields), tuple(cell_fields), sched, rank))
        edge_out = tuple(f + 1000.0 * (i + 1)
                         for i, f in enumerate(edge_fields))
        cell_out = tuple(f + 1000.0 * (len(edge_fields) + j + 1)
                         for j, f in enumerate(cell_fields))
        return edge_out, cell_out

    monkeypatch.setattr(vm, "batched_halo_exchange", fake_exchange)
    out = vm.exchange_state_mpas_ocean(state, layout)

    # (a) exactly one packed call, all five fields, documented order.
    assert len(calls) == 1
    edge_in, cell_in, sched, rank = calls[0]
    assert len(edge_in) == 1 and len(cell_in) == 4
    assert edge_in[0] is state.u.data
    assert cell_in[0] is state.T.data
    assert cell_in[1] is state.S.data
    assert cell_in[2] is state.eta.data
    assert cell_in[3] is state.w.data
    assert sched is layout.batched_comm
    assert rank == layout.rank

    # (b) every state slot received its own sentinel (u:+1000, T:+2000,
    # S:+3000, eta:+4000, w:+5000).
    for name, offset in (("u", 1000.0), ("T", 2000.0), ("S", 3000.0),
                         ("eta", 4000.0), ("w", 5000.0)):
        np.testing.assert_allclose(
            np.asarray(getattr(out, name).data),
            np.asarray(getattr(state, name).data) + offset,
            err_msg=f"{name} did not receive its own exchange sentinel")
    # Static fields still pass through by reference, never exchanged.
    assert out.H_bathy is state.H_bathy
    assert out.land_mask is state.land_mask
    assert out.rho_ref_z is state.rho_ref_z


# ------------------------------------------------------------------
# make_mpas_ocean_halo_refresh — the IN-STEP stage-frontier refresh
# factory (stage-correctness lever).  Same coverage doctrine as above:
# np=1 identity alone cannot see a mis-routed entity class, so the
# packed-call routing is locked with a mocked exchange.
# ------------------------------------------------------------------

def test_halo_refresh_factory_np1_identity(single_rank_setup):
    from legoesm.parallel.voronoi_mpi import make_mpas_ocean_halo_refresh

    _mesh, state, layout = single_rank_setup
    refresh = make_mpas_ocean_halo_refresh(layout)
    (u_out,) = refresh.edges(state.u.data)
    np.testing.assert_array_equal(np.asarray(u_out),
                                  np.asarray(state.u.data))
    T_out, S_out = refresh.cells(state.T.data, state.S.data)
    np.testing.assert_array_equal(np.asarray(T_out),
                                  np.asarray(state.T.data))
    np.testing.assert_array_equal(np.asarray(S_out),
                                  np.asarray(state.S.data))
    (e_out,), (c_out,) = refresh.both((state.u.data,), (state.eta.data,))
    np.testing.assert_array_equal(np.asarray(e_out),
                                  np.asarray(state.u.data))
    np.testing.assert_array_equal(np.asarray(c_out),
                                  np.asarray(state.eta.data))
    # Vertex channel (K_zeta_bih T3 site): np=1 identity through the real
    # per-field VoronoiHaloExchange machinery.
    v_field = jnp.arange(float(_mesh.nVertices))[:, None] * jnp.ones((1, 3))
    (v_out,) = refresh.vertices(v_field)
    np.testing.assert_array_equal(np.asarray(v_out), np.asarray(v_field))


def test_halo_refresh_factory_routes_entities(single_rank_setup, monkeypatch):
    """edges() must pack fields as EDGE fields, cells() as CELL fields, and
    both() must keep the (edge, cell) split — a swapped entity class would
    scatter halos with the wrong connectivity and the np=1 identity test
    would never notice."""
    import legoesm.parallel.voronoi_mpi as vm

    _mesh, state, layout = single_rank_setup
    calls = []

    def fake_exchange(edge_fields, cell_fields, sched, rank):
        calls.append((tuple(edge_fields), tuple(cell_fields)))
        return (tuple(f + 10.0 for f in edge_fields),
                tuple(f + 20.0 for f in cell_fields))

    monkeypatch.setattr(vm, "batched_halo_exchange", fake_exchange)
    refresh = vm.make_mpas_ocean_halo_refresh(layout)

    (u_out,) = refresh.edges(state.u.data)
    assert len(calls[-1][0]) == 1 and len(calls[-1][1]) == 0
    np.testing.assert_allclose(np.asarray(u_out),
                               np.asarray(state.u.data) + 10.0)

    (T_out,) = refresh.cells(state.T.data)
    assert len(calls[-1][0]) == 0 and len(calls[-1][1]) == 1
    np.testing.assert_allclose(np.asarray(T_out),
                               np.asarray(state.T.data) + 20.0)

    (e_out,), (c_out,) = refresh.both((state.u.data,), (state.eta.data,))
    assert len(calls[-1][0]) == 1 and len(calls[-1][1]) == 1
    np.testing.assert_allclose(np.asarray(e_out),
                               np.asarray(state.u.data) + 10.0)
    np.testing.assert_allclose(np.asarray(c_out),
                               np.asarray(state.eta.data) + 20.0)
