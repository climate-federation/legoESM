"""Direct tests for ``voronoi_mpi.exchange_state_mpas_ocean`` (M3d inc-1).

Single-process coverage of the packed MPAS-OCEAN state halo refresh:
np=1 (empty union-neighbor schedule) is the documented identity path of
``batched_halo_exchange`` and needs no MPI stack, so the wiring —
field packing order, static-field pass-through, schema tripwire,
hand-built-layout schedule rebuild, jit compatibility — is locked here.
The np=2 exchange semantics ride the existing distributed machinery
(``tests/distributed/test_voronoi_batched_halo.py`` for the transport;
``scripts/bench/bench_ocean_mpas_scaling.py --halo-refresh per_step``
drives this helper multi-rank).
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
