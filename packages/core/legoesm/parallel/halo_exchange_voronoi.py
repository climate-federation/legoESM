"""Halo exchange for domain-decomposed Voronoi meshes.

Provides MPI-based halo exchange that communicates field values for
halo (ghost) entities from their owning ranks.  Also provides a
simulated local exchange for testing without MPI.

Usage
-----
::

    # MPI mode (under mpirun):
    halo = VoronoiHaloExchange(partition, backend="mpi")
    h_local = halo.exchange_cell_field(h_local)
    u_local = halo.exchange_edge_field(u_local)

    # Testing mode (single process, multiple partitions):
    local_fields = exchange_local_simulated(
        partitions, local_fields, entity="cell")
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.parallel.voronoi_partition import (
    HaloCommSchedule,
    VoronoiPartition,
)


class VoronoiHaloExchange:
    """Halo exchange manager for one rank of a partitioned Voronoi mesh.

    Parameters
    ----------
    partition : VoronoiPartition
        Partition descriptor for this rank.
    backend : str
        ``"mpi"`` for distributed execution, ``"local"`` for
        single-process mode (exchange is a no-op; call
        :func:`exchange_local_simulated` instead for testing).
    """

    def __init__(self, partition: VoronoiPartition, backend: str = "local"):
        self.partition = partition
        self.backend = backend

    def exchange_cell_field(self, field: jnp.ndarray) -> jnp.ndarray:
        """Exchange halo values for a cell-centered field.

        Parameters
        ----------
        field : jax.Array, shape ``(n_local_cells, ...)``

        Returns
        -------
        jax.Array
            Same shape, with halo positions updated.
        """
        return self._exchange(field, self.partition.cell_comm, entity_type=0)

    def exchange_edge_field(self, field: jnp.ndarray) -> jnp.ndarray:
        """Exchange halo values for an edge-centered field."""
        return self._exchange(field, self.partition.edge_comm, entity_type=1)

    def exchange_vertex_field(self, field: jnp.ndarray) -> jnp.ndarray:
        """Exchange halo values for a vertex-centered field."""
        return self._exchange(field, self.partition.vertex_comm, entity_type=2)

    def _exchange(
        self,
        field: jnp.ndarray,
        comm: HaloCommSchedule,
        entity_type: int,
    ) -> jnp.ndarray:
        if self.backend == "mpi":
            return _exchange_mpi(
                field, comm, self.partition.rank, entity_type,
            )
        # local backend: no-op (halos are already filled or irrelevant)
        return field


# ============================================================================
# MPI exchange
# ============================================================================

def _exchange_mpi(
    field: jnp.ndarray,
    comm: HaloCommSchedule,
    rank: int,
    entity_type: int,
) -> jnp.ndarray:
    """Perform MPI halo exchange using mpi4jax sendrecv.

    Iterates over neighbor ranks, performing a blocking sendrecv for
    each.  The send buffer is packed from ``comm.send_idx`` and the
    received data is scattered into ``comm.recv_idx``.

    Works for fields of any shape ``(n_local, ...)`` — multi-level
    fields are handled automatically.
    """
    from legoesm.parallel.reductions import (
        _mpi4jax_array_result,
        _require_mpi_stack,
    )

    mpi4jax, MPI = _require_mpi_stack()

    # Use a type-based tag offset to avoid collisions between
    # cell / edge / vertex exchanges in the same JIT trace.
    TAG_BASE = entity_type * 1_000_000

    s_offset = 0
    r_offset = 0

    for i, nbr_rank in enumerate(comm.neighbor_ranks):
        s_count = comm.send_counts[i]
        r_count = comm.recv_counts[i]

        send_idx = comm.send_idx[s_offset:s_offset + s_count]
        recv_idx = comm.recv_idx[r_offset:r_offset + r_count]

        send_buf = field[send_idx]

        if field.ndim == 1:
            recv_buf = jnp.zeros(r_count, dtype=field.dtype)
        else:
            recv_buf = jnp.zeros(
                (r_count,) + field.shape[1:], dtype=field.dtype,
            )

        recv_data = _mpi4jax_array_result(
            mpi4jax.sendrecv(
                send_buf,
                recv_buf,
                source=nbr_rank,
                dest=nbr_rank,
                sendtag=TAG_BASE + rank * 1000 + nbr_rank,
                recvtag=TAG_BASE + nbr_rank * 1000 + rank,
                comm=MPI.COMM_WORLD,
            ),
        )

        field = field.at[recv_idx].set(recv_data)

        s_offset += s_count
        r_offset += r_count

    return field


# ============================================================================
# Local (non-MPI) utilities for testing
# ============================================================================

def exchange_local_simulated(
    partitions: list[VoronoiPartition],
    local_fields: list[jnp.ndarray],
    entity: str = "cell",
) -> list[jnp.ndarray]:
    """Simulate MPI halo exchange in a single process.

    Reconstructs a global field from all ranks' owned values, then
    copies the correct halo values into each rank's local field.

    Parameters
    ----------
    partitions : list of VoronoiPartition
        One partition per simulated rank.
    local_fields : list of jnp.ndarray
        One local field per rank, shape ``(n_local_*, ...)``.
    entity : str
        ``"cell"``, ``"edge"``, or ``"vertex"``.

    Returns
    -------
    list of jnp.ndarray
        Updated local fields with halo values filled.
    """
    if entity == "cell":
        g_size = partitions[0].nCells_global
        n_own = lambda p: p.n_owned_cells
        local_ents = lambda p: p.local_cells
    elif entity == "edge":
        g_size = partitions[0].nEdges_global
        n_own = lambda p: p.n_owned_edges
        local_ents = lambda p: p.local_edges
    elif entity == "vertex":
        g_size = partitions[0].nVertices_global
        n_own = lambda p: p.n_owned_vertices
        local_ents = lambda p: p.local_vertices
    else:
        raise ValueError(f"Unknown entity type: {entity!r}")

    dtype = local_fields[0].dtype
    trailing = local_fields[0].shape[1:]
    global_field = jnp.zeros((g_size,) + trailing, dtype=dtype)

    # Scatter owned values into the global field.
    for p, f in zip(partitions, local_fields):
        n = n_own(p)
        idx = local_ents(p)[:n]
        global_field = global_field.at[idx].set(f[:n])

    # Fill each rank's halo from the global field.
    result = []
    for p, f in zip(partitions, local_fields):
        n = n_own(p)
        halo_idx = local_ents(p)[n:]
        updated = f.at[n:].set(global_field[halo_idx])
        result.append(updated)

    return result
