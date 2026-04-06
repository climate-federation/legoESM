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
from legoesm.parallel.distributed import scatter_to_local, gather_to_global
from legoesm.parallel.reductions import global_sum_mpi
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import (
    pad_halo,
    pad_halo_vector,
    set_halo_backend,
    _pad_halo_local,
    _pad_halo_local_h2,
)
from legoesm.ocean.conservation import ocean_conservation_fixer
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


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


def _zero_non_owned(data, topology):
    """Zero non-owned faces in a (6, ...) array for MPI halo exchange.

    pad_halo_mpi expects a full (6, n, n) array where non-local faces
    are zeroed.  scatter_to_local returns (n_local, n, n) which is
    incompatible with the MPI halo exchange functions.
    """
    owned = set(topology.local_face_ids)
    mask = jnp.array([1.0 if f in owned else 0.0 for f in range(6)])
    # Broadcast mask to match data shape: (6,) → (6, 1, 1, ...)
    for _ in range(data.ndim - 1):
        mask = mask[..., jnp.newaxis]
    return data * mask


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

        # Zero non-owned faces then apply MPI halo exchange.
        partitioned = _zero_non_owned(data, topology)
        set_halo_backend("mpi", topology)
        result = pad_halo(partitioned)

        # All ranks participate in MPI exchange; rank 0 checks result.
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

        partitioned = _zero_non_owned(data, topology)
        set_halo_backend("mpi", topology)
        result = pad_halo(partitioned)

        if topology.rank == 0:
            assert jnp.allclose(result, reference, atol=1e-6)

    def test_pad_halo_vector_mpi_matches_local(self, topology):
        """MPI vector halo exchange matches local reference after gather."""
        n = 8
        grid = create_cubed_sphere(n)
        key = jax.random.PRNGKey(123)
        key_u, key_v = jax.random.split(key)
        u_data = jax.random.normal(key_u, (6, n, n), dtype=jnp.float32)
        v_data = jax.random.normal(key_v, (6, n, n), dtype=jnp.float32)

        set_halo_backend("local")
        ref_u, ref_v = pad_halo_vector(
            u_data,
            v_data,
            grid.cos_angle,
            grid.sin_angle,
            grid.cos_angle_padded,
            grid.sin_angle_padded,
        )

        u_part = _zero_non_owned(u_data, topology)
        v_part = _zero_non_owned(v_data, topology)
        set_halo_backend("mpi", topology)
        out_u, out_v = pad_halo_vector(
            u_part,
            v_part,
            grid.cos_angle,
            grid.sin_angle,
            grid.cos_angle_padded,
            grid.sin_angle_padded,
        )

        if topology.rank == 0:
            assert jnp.allclose(out_u, ref_u, atol=1e-6)
            assert jnp.allclose(out_v, ref_v, atol=1e-6)


    def test_pad_halo_mpi_h2_matches_local(self, topology):
        """MPI halo=2 exchange matches local reference."""
        n = 8
        key = jax.random.PRNGKey(99)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float32)

        reference = _pad_halo_local_h2(data)

        partitioned = _zero_non_owned(data, topology)
        set_halo_backend("mpi", topology)
        result = pad_halo(partitioned, halo=2)

        if topology.rank == 0:
            assert result.shape == (6, n + 4, n + 4)
            assert jnp.allclose(result, reference, atol=1e-6), (
                f"MPI halo=2 mismatch (max diff: "
                f"{jnp.max(jnp.abs(result - reference))})"
            )


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
        """scatter_to_local extracts only local faces."""
        data = jnp.ones((6, 4, 4), dtype=jnp.float32)
        result = scatter_to_local(data)

        n_local = len(topology.local_face_ids)
        assert result.shape == (n_local, 4, 4), (
            f"Expected ({n_local}, 4, 4), got {result.shape}"
        )
        # All local faces should have value 1.0
        assert jnp.allclose(result, 1.0)

    def test_gather_recovers_full(self, topology):
        """gather_to_global(scatter_to_local(x)) == x on rank 0."""
        data = jnp.ones((6, 4, 4), dtype=jnp.float32)
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        partitioned = scatter_to_local(data)
        gathered = gather_to_global(partitioned)

        if topology.rank == 0:
            assert gathered.shape == (6, 4, 4), f"Expected (6,4,4), got {gathered.shape}"
            assert jnp.allclose(gathered, data)


class TestMPIOceanConservation:
    """MPI conservation-fixer behavior on partitioned ocean state."""

    def test_ocean_conservation_fixer_matches_local(self, topology):
        """MPI conservation fixers should match local-global reference."""
        grid = create_cubed_sphere(8)
        z_coord = create_ocean_z_star(n_levels=6, H_max=4000.0)
        state_old = rest_state_ocean(
            grid,
            z_coord,
            H_max=4000.0,
            land_lat_threshold=70.0,
        )
        mask = state_old.land_mask.data

        state_new = state_old._replace(
            eta=state_old.eta.replace(
                data=state_old.eta.data + 0.03 * mask,
            ),
            T=state_old.T.replace(
                data=state_old.T.data + 0.2 * mask[..., jnp.newaxis],
            ),
            S=state_old.S.replace(
                data=state_old.S.data - 0.1 * mask[..., jnp.newaxis],
            ),
        )

        config = OceanConfig(
            fix_volume=True,
            fix_heat=True,
            fix_salt=True,
            min_water_column_m=0.5,
        )

        set_halo_backend("local")
        ref_state = ocean_conservation_fixer(
            state_new,
            state_old,
            grid,
            z_coord,
            config,
        )

        # For MPI conservation: use _zero_non_owned on state arrays
        # but keep grid global (conservation fixer needs global areas
        # for the area-weighted correction).
        set_halo_backend("mpi", topology)
        old_part = jax.tree.map(
            lambda x: _zero_non_owned(x, topology)
            if isinstance(x, jnp.ndarray) and x.ndim >= 3 and x.shape[0] == 6
            else x,
            state_old,
        )
        new_part = jax.tree.map(
            lambda x: _zero_non_owned(x, topology)
            if isinstance(x, jnp.ndarray) and x.ndim >= 3 and x.shape[0] == 6
            else x,
            state_new,
        )
        fixed_part = ocean_conservation_fixer(
            new_part,
            old_part,
            grid,
            z_coord,
            config,
        )

        if topology.rank == 0:
            assert jnp.allclose(fixed_part.eta.data, ref_state.eta.data, atol=1e-6)
            assert jnp.allclose(fixed_part.T.data, ref_state.T.data, atol=1e-6)
            assert jnp.allclose(fixed_part.S.data, ref_state.S.data, atol=1e-6)
