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
    pad_halo_local,
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
    """MPI halo exchange correctness.

    Tests verify that MPI halo exchange (face-only mode) produces the
    same results as the local single-process implementation.
    """

    def test_pad_halo_mpi_matches_local(self, topology):
        """MPI halo exchange matches local reference for owned faces."""
        n = 8
        # Create a known field: face i has value i+1.
        data = jnp.zeros((6, n, n), dtype=jnp.float32)
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        # Reference: local halo exchange (no MPI).
        reference = pad_halo_local(data)

        # Zero non-owned faces then apply MPI halo exchange.
        partitioned = _zero_non_owned(data, topology)
        set_halo_backend("mpi", topology)
        result = pad_halo(partitioned)

        # Compare only owned faces (non-owned are zeroed on each rank).
        owned = jnp.array(list(topology.local_face_ids))
        if topology.rank == 0:
            assert jnp.allclose(result[owned], reference[owned]), (
                f"MPI halo mismatch on owned faces (max diff: "
                f"{jnp.max(jnp.abs(result[owned] - reference[owned]))})"
            )

    def test_pad_halo_mpi_random(self, topology):
        """MPI halo exchange with random data matches local for owned faces."""
        n = 16
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float32)

        reference = pad_halo_local(data)

        partitioned = _zero_non_owned(data, topology)
        set_halo_backend("mpi", topology)
        result = pad_halo(partitioned)

        owned = jnp.array(list(topology.local_face_ids))
        if topology.rank == 0:
            assert jnp.allclose(result[owned], reference[owned], atol=1e-6)

    def test_pad_halo_vector_mpi_matches_local(self, topology):
        """MPI vector halo exchange matches local reference for owned faces."""
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

        owned = jnp.array(list(topology.local_face_ids))
        if topology.rank == 0:
            assert jnp.allclose(out_u[owned], ref_u[owned], atol=1e-6)
            assert jnp.allclose(out_v[owned], ref_v[owned], atol=1e-6)


    def test_pad_halo_mpi_h2_matches_local(self, topology):
        """MPI halo=2 exchange matches local reference for owned faces."""
        n = 8
        key = jax.random.PRNGKey(99)
        data = jax.random.normal(key, (6, n, n), dtype=jnp.float32)

        reference = _pad_halo_local_h2(data)

        partitioned = _zero_non_owned(data, topology)
        set_halo_backend("mpi", topology)
        result = pad_halo(partitioned, halo=2)

        owned = jnp.array(list(topology.local_face_ids))
        if topology.rank == 0:
            assert result.shape == (6, n + 4, n + 4)
            assert jnp.allclose(result[owned], reference[owned], atol=1e-6), (
                f"MPI halo=2 mismatch on owned faces (max diff: "
                f"{jnp.max(jnp.abs(result[owned] - reference[owned]))})"
            )


class TestCanonicalOrderingLogic:
    """Verify canonical ordering in MPI halo exchange for both modes.

    These tests check the sorting logic WITHOUT requiring many MPI ranks.
    They simulate the entry lists that two ranks would produce and verify
    that canonical sorting makes them agree on strip order.
    """

    def test_face_only_canonical_ordering_agreement(self, topology):
        """Sender and receiver canonical orderings agree for face-only entries."""
        # entry format: (face, edge, nbr_face, nbr_edge, is_reversed, nbr_rank)
        WEST, EAST, SOUTH, NORTH = 0, 1, 2, 3
        # Rank 0 owns faces [0,1,2], Rank 1 owns [3,4,5].
        # Rank 0's remote entries for rank 1:
        entries_r0 = [
            (0, WEST, 4, EAST, True, 1),    # face0 WEST <-> face4 EAST
            (0, SOUTH, 5, NORTH, False, 1),  # face0 SOUTH <-> face5 NORTH
            (1, EAST, 4, SOUTH, True, 1),    # face1 EAST <-> face4 SOUTH
        ]
        # Rank 1's remote entries for rank 0 (reverse perspective):
        entries_r1 = [
            (4, EAST, 0, WEST, True, 0),     # face4 EAST <-> face0 WEST
            (5, NORTH, 0, SOUTH, False, 0),   # face5 NORTH <-> face0 SOUTH
            (4, SOUTH, 1, EAST, True, 0),     # face4 SOUTH <-> face1 EAST
        ]

        # Canonical sort: send by (nbr_face, nbr_edge), recv by (face, edge)
        send_r0 = sorted(entries_r0, key=lambda e: (e[2], e[3]))
        recv_r0 = sorted(entries_r0, key=lambda e: (e[0], e[1]))
        send_r1 = sorted(entries_r1, key=lambda e: (e[2], e[3]))
        recv_r1 = sorted(entries_r1, key=lambda e: (e[0], e[1]))

        # R0 sends in order of (nbr_face, nbr_edge): r1's face/edge
        # R1 unpacks in order of (face, edge): r1's own face/edge
        r0_send_keys = [(e[2], e[3]) for e in send_r0]
        r1_recv_keys = [(e[0], e[1]) for e in recv_r1]
        assert r0_send_keys == r1_recv_keys, (
            f"R0 send {r0_send_keys} != R1 recv {r1_recv_keys}"
        )

        # R1 sends in order of (nbr_face, nbr_edge): r0's face/edge
        # R0 unpacks in order of (face, edge): r0's own face/edge
        r1_send_keys = [(e[2], e[3]) for e in send_r1]
        r0_recv_keys = [(e[0], e[1]) for e in recv_r0]
        assert r1_send_keys == r0_recv_keys, (
            f"R1 send {r1_send_keys} != R0 recv {r0_recv_keys}"
        )

    def test_tiled_canonical_ordering_agreement(self, topology):
        """Sender and receiver canonical orderings agree for tiled entries."""
        # Simulate a corner tile that shares TWO edges with the same neighbor.
        # Rank A has edges (WEST, SOUTH) going to rank B.
        # Rank B has edges (EAST, NORTH) coming from rank A.
        # entry format: (edge, nbr_rank, nbr_edge, is_reversed, is_tile_nbr)
        WEST, EAST, SOUTH, NORTH = 0, 1, 2, 3
        rank_B = 99

        # Rank A's entries for rank B (two edges):
        entries_A = [
            (SOUTH, rank_B, NORTH, False, True),  # A's SOUTH <-> B's NORTH
            (WEST, rank_B, EAST, False, True),     # A's WEST <-> B's EAST
        ]
        # Rank B's entries for rank A (reverse perspective):
        rank_A = 42
        entries_B = [
            (NORTH, rank_A, SOUTH, False, True),   # B's NORTH <-> A's SOUTH
            (EAST, rank_A, WEST, False, True),      # B's EAST <-> A's WEST
        ]

        # Apply canonical sorting (same as _pad_halo_mpi_tiled):
        # Send sorted by nbr_edge (entry[2])
        # Recv sorted by edge (entry[0])
        send_order_A = sorted(entries_A, key=lambda e: e[2])
        recv_order_A = sorted(entries_A, key=lambda e: e[0])
        send_order_B = sorted(entries_B, key=lambda e: e[2])
        recv_order_B = sorted(entries_B, key=lambda e: e[0])

        # Rank A sends strips in order of nbr_edge: EAST(1) then NORTH(3)
        # Rank B receives and unpacks in order of edge: EAST(1) then NORTH(3)
        # B's recv order must match A's send order (by nbr_edge perspective):
        a_send_nbr_edges = [e[2] for e in send_order_A]  # A sends sorted by nbr_edge
        b_recv_edges = [e[0] for e in recv_order_B]       # B unpacks sorted by own edge

        # A sends for B's edges [EAST, NORTH] sorted = [1, 3]
        # B unpacks by own edges [EAST, NORTH] sorted = [1, 3]
        assert a_send_nbr_edges == b_recv_edges, (
            f"Sender A nbr_edges {a_send_nbr_edges} != receiver B edges {b_recv_edges}"
        )

        # Symmetrically: B sends sorted by nbr_edge, A unpacks sorted by edge
        b_send_nbr_edges = [e[2] for e in send_order_B]
        a_recv_edges = [e[0] for e in recv_order_A]
        assert b_send_nbr_edges == a_recv_edges, (
            f"Sender B nbr_edges {b_send_nbr_edges} != receiver A edges {a_recv_edges}"
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
    """State partitioning and gathering.

    Uses the conftest ``cube_face_layout`` fixture: the autouse per-test reset
    clears the active layout, so it must be re-established at SETUP —
    a bare ``get_active_layout()`` here always returned None.
    """

    def test_partition_zeros_non_local(self, topology, cube_face_layout):
        """scatter_to_local extracts only local faces."""
        n = cube_face_layout.global_n
        data = jnp.ones((6, n, n), dtype=jnp.float32)
        result = scatter_to_local(data)

        n_local = len(topology.local_face_ids)
        assert result.shape == (n_local, n, n), (
            f"Expected ({n_local}, {n}, {n}), got {result.shape}"
        )
        # All local faces should have value 1.0
        assert jnp.allclose(result, 1.0)

    def test_gather_recovers_full(self, topology, cube_face_layout):
        """gather_to_global(scatter_to_local(x)) == x on rank 0."""
        n = cube_face_layout.global_n
        data = jnp.ones((6, n, n), dtype=jnp.float32)
        for f in range(6):
            data = data.at[f].set(float(f + 1))

        partitioned = scatter_to_local(data)
        gathered = gather_to_global(partitioned)

        if topology.rank == 0:
            assert gathered.shape == (6, n, n), f"Expected (6,{n},{n}), got {gathered.shape}"
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

        # Compare only owned faces: the MPI version zeroes non-owned faces,
        # so non-owned face data will not match the reference.
        owned = jnp.array(list(topology.local_face_ids))
        # MPI allreduce uses different FP summation order than serial,
        # producing O(1e-5) differences in float32.  Use atol=1e-4
        # to accommodate the worst-case rounding discrepancy.
        assert jnp.allclose(
            fixed_part.eta.data[owned], ref_state.eta.data[owned], atol=1e-4,
        ), "eta mismatch on owned faces"
        assert jnp.allclose(
            fixed_part.T.data[owned], ref_state.T.data[owned], atol=1e-4,
        ), "T mismatch on owned faces"
        assert jnp.allclose(
            fixed_part.S.data[owned], ref_state.S.data[owned], atol=1e-4,
        ), "S mismatch on owned faces"
