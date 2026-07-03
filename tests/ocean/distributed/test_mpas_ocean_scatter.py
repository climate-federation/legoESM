"""np=2 scatter/gather roundtrip for the MPAS OCEAN state.

Run:  mpirun -np 2 python -m pytest \
          tests/ocean/distributed/test_mpas_ocean_scatter.py

Covers the state plumbing the distributed-MPAS milestone needs:
``scatter_state_mpas_ocean`` slices each rank's (owned + halo)
entities; ``gather_state_mpas_ocean`` reassembles the global state from
owned entities only — the composition must be the identity on every
field.  The full distributed model STEP additionally needs per-RK-stage
halo refreshes (the atmosphere's ``make_voronoi_mpi_step`` pattern) and
is the scoped next milestone — see
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

mpi4py = pytest.importorskip("mpi4py")
pytest.importorskip("mpi4jax")

from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK, SIZE = comm.Get_rank(), comm.Get_size()

pytestmark = pytest.mark.skipif(SIZE < 2, reason="needs mpirun -np 2")

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.vertical import create_ocean_z_star


def test_scatter_gather_roundtrip_identity():
    from legoesm.parallel.voronoi_mpi import (
        gather_state_mpas_ocean,
        initialize_voronoi_mpi,
        scatter_state_mpas_ocean,
    )

    mesh = create_voronoi_mesh(3)
    z = create_ocean_z_star(n_levels=4)
    state = rest_state_mpas_ocean(mesh, z)
    # Deterministic structure on every prognostic field so the
    # roundtrip is non-trivial (rank-identical construction).
    rng = np.random.default_rng(11)
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            rng.standard_normal(state.u.data.shape))),
        T=state.T.replace(data=state.T.data + jnp.asarray(
            rng.standard_normal(state.T.data.shape))),
        S=state.S.replace(data=state.S.data + jnp.asarray(
            0.1 * rng.standard_normal(state.S.data.shape))),
        eta=state.eta.replace(data=jnp.asarray(
            0.01 * rng.standard_normal(state.eta.data.shape))),
    )

    _r, _n, layout = initialize_voronoi_mpi(mesh)
    local = scatter_state_mpas_ocean(state, layout.partition)

    # Local shapes: leading dim = local entity counts.
    assert local.u.data.shape[0] == layout.partition.n_local_edges
    assert local.T.data.shape[0] == layout.partition.n_local_cells

    gathered = gather_state_mpas_ocean(local, layout.partition)
    for name in ("u", "T", "S", "eta", "w", "H_bathy", "land_mask"):
        got = np.asarray(getattr(gathered, name).data)
        want = np.asarray(getattr(state, name).data)
        np.testing.assert_array_equal(
            got, want, err_msg=f"roundtrip mismatch on {name}")
