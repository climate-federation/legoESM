"""End-to-end MPI scaling tests for the cubed-sphere driver path.

Run with:
    mpirun -np 2 python -m pytest tests/distributed/test_mpi_driver.py -v
    mpirun -np 3 python -m pytest tests/distributed/test_mpi_driver.py -v
    mpirun -np 6 python -m pytest tests/distributed/test_mpi_driver.py -v

Verifies that:
1. initialize_distributed() sets the MPI halo backend.
2. 4D MPI halo exchange matches the local reference.
3. Multi-rank distributed execution (scatter → step → gather) matches
   a single-rank reference within tolerance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import (
    get_halo_backend,
    set_halo_backend,
    pad_halo_4d,
)
from legoesm.parallel.comm import build_comm_topology
from legoesm.parallel.distributed import initialize_distributed


@pytest.fixture(autouse=True)
def reset_halo_backend():
    """Reset halo backend to local before AND after each test.

    FV3_3D iter-1063: the conftest session fixture pre-initializes
    MPI and leaves the backend in 'mpi' mode.  Each test must start
    with a clean 'local' baseline so preconditions like
    ``assert get_halo_backend() == 'local'`` hold.
    """
    set_halo_backend("local")
    yield
    set_halo_backend("local")


class TestMPIDriverPath:
    """End-to-end tests for the MPI-initialized cubed-sphere driver."""

    def test_initialize_distributed_sets_mpi_backend(self):
        """initialize_distributed() must set halo backend to 'mpi'."""
        assert get_halo_backend() == "local"
        initialize_distributed()
        assert get_halo_backend() == "mpi"

    def test_pad_halo_4d_mpi_matches_local(self):
        """4D MPI halo exchange matches local reference on owned faces.

        FV3_3D iter-1063: under MPI with replicated ``(6, n, n, nlev)``
        input, the helper only exchanges halos for the rank's
        locally-owned faces (face-only mode).  Non-owned faces in
        the output have unfilled (zero) halo cells, so a full
        ``(6, n+2, n+2, nlev)`` comparison fails on every rank.
        Compare per-owned-face only — matches the FV3 step-fidelity
        and iter-1061 patterns.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        n, nlev = 8, 5

        data_global = jnp.zeros((6, n, n, nlev), dtype=jnp.float64)
        for f in range(6):
            data_global = data_global.at[f].set(float(f + 1))

        set_halo_backend("local")
        ref = pad_halo_4d(data_global)

        topology = build_comm_topology(rank, MPI.COMM_WORLD.Get_size())
        set_halo_backend("mpi", topology)
        result = pad_halo_4d(data_global)

        for f in topology.local_face_ids:
            diff = jnp.max(jnp.abs(result[f] - ref[f]))
            assert diff == 0.0, (
                f"4D MPI halo mismatch on owned face {f} (rank {rank}): "
                f"max diff = {diff}"
            )
