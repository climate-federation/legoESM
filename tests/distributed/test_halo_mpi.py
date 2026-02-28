"""MPI halo exchange correctness tests.

Run with:
    mpirun -np 6 python -m pytest tests/distributed/test_halo_mpi.py -v
    mpirun -np 2 python -m pytest tests/distributed/test_halo_mpi.py -v

These tests verify that MPI halo exchange produces the same results
as the local (single-process) implementation.
"""

import jax
import jax.numpy as jnp
import pytest

# Guard: skip all tests if mpi4jax/mpi4py are not installed.
mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.parallel.comm import build_comm_topology
from legoesm.parallel.distributed import partition_state, gather_state
from legoesm.parallel.reductions import global_sum_mpi
from legoesm.grids.halo import pad_halo, set_halo_backend, _pad_halo_local


@pytest.fixture(autouse=True)
def reset_halo_backend():
    """Reset halo backend to local after each test."""
    yield
    set_halo_backend("local")


@pytest.fixture
def topology():
    """Build topology for this MPI rank."""
    rank = MPI.COMM_WORLD.Get_rank()
    n_processes = MPI.COMM_WORLD.Get_size()
    return build_comm_topology(rank, n_processes)


class TestMPIHaloExchange:
    """MPI halo exchange correctness."""

    def test_pad_halo_mpi_matches_local(self, topology):
        """MPI halo exchange matches local reference on rank 0."""
        n = 8
        # Create a known field: face i has value i+1.
        data = jnp.zeros((6, n, n), dtype=jnp.float32)
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        # Reference: local halo exchange (no MPI).
        reference = _pad_halo_local(data)

        # MPI halo exchange.
        set_halo_backend("mpi", topology)
        result = pad_halo(data)

        # Gather results to rank 0 for comparison.
        if topology.rank == 0:
            assert jnp.allclose(result, reference), (
                f"MPI halo mismatch (max diff: "
                f"{jnp.max(jnp.abs(result - reference))})"
            )

    def test_pad_halo_mpi_random(self, topology):
        """MPI halo exchange with random data matches local."""
        n = 16
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float32)

        reference = _pad_halo_local(data)

        set_halo_backend("mpi", topology)
        result = pad_halo(data)

        if topology.rank == 0:
            assert jnp.allclose(result, reference, atol=1e-6)


class TestMPIReductions:
    """MPI reduction operations."""

    def test_global_sum(self, topology):
        """Global sum across ranks equals expected value."""
        # Each rank contributes its rank number.
        local = jnp.array(float(topology.rank))
        expected = jnp.array(
            float(sum(range(topology.n_processes)))
        )
        result = global_sum_mpi(local)
        assert jnp.allclose(result, expected)


class TestPartitionGather:
    """State partitioning and gathering."""

    def test_partition_zeros_non_local(self, topology):
        """partition_state zeros out non-local faces."""
        data = jnp.ones((6, 4, 4), dtype=jnp.float32)
        result = partition_state(data, topology)

        for f in range(6):
            if f in topology.local_face_ids:
                assert jnp.allclose(result[f], 1.0)
            else:
                assert jnp.allclose(result[f], 0.0)

    def test_gather_recovers_full(self, topology):
        """gather_state(partition_state(x)) == x."""
        data = jnp.ones((6, 4, 4), dtype=jnp.float32)
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        partitioned = partition_state(data, topology)
        gathered = gather_state(partitioned, topology)

        assert jnp.allclose(gathered, data)
