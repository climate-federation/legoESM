"""Distributed halo exchange using mpi4jax.

Provides MPI-based halo exchange for multi-node cubed-sphere runs.
The single-node version (using JAX's automatic SPMD) is in
:mod:`legoesm.grids.halo` and requires no changes.

This module is only imported when MPI is active.  ``mpi4jax`` is
an optional dependency.

Usage
-----
The halo exchange is set up once at initialization via
:func:`legoesm.grids.halo.set_halo_backend`::

    from legoesm.parallel.comm import build_comm_topology
    from legoesm.grids.halo import set_halo_backend

    topology = build_comm_topology(rank, n_processes)
    set_halo_backend("mpi", topology)

After that, all operators automatically use MPI halo exchange.

Optimization
------------
Edges to the same neighbor rank are packed into a single send buffer,
reducing the number of MPI messages from ~4 per face (one per edge)
to ~2-3 per rank (one per unique neighbor rank).  This cuts MPI
latency overhead significantly on high-latency interconnects.
"""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp

from legoesm.grids.halo import (
    CONNECTIVITY,
    WEST,
    EAST,
    SOUTH,
    NORTH,
    _extract_edge_strip,
)
from legoesm.parallel.comm import CommTopology
from legoesm.parallel.reductions import _mpi4jax_array_result


_EDGES = (WEST, EAST, SOUTH, NORTH)


def _place_strip(padded: jax.Array, face: int, edge: int, strip: jax.Array) -> jax.Array:
    """Place a received strip into the correct halo position."""
    if edge == WEST:
        padded = padded.at[face, 0, 1:-1].set(strip)
    elif edge == EAST:
        padded = padded.at[face, -1, 1:-1].set(strip)
    elif edge == SOUTH:
        padded = padded.at[face, 1:-1, 0].set(strip)
    elif edge == NORTH:
        padded = padded.at[face, 1:-1, -1].set(strip)
    return padded


def pad_halo_mpi(
    data: jax.Array,
    topology: CommTopology,
) -> jax.Array:
    """Pad a scalar field with inter-face halo data using MPI.

    This mirrors :func:`legoesm.grids.halo.pad_halo` but uses
    ``mpi4jax.sendrecv`` for edges that cross MPI rank boundaries.

    Edges to the same neighbor rank are packed into a single buffer
    to reduce the number of MPI messages (latency optimization).

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
        Scalar field on the cubed-sphere.  Non-local faces should
        contain valid data for local exchanges or zeros.
    topology : CommTopology
        Pre-computed communication topology.

    Returns
    -------
    padded : jax.Array, shape (6, n+2, n+2)
    """
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "MPI halo exchange requires mpi4jax and mpi4py. "
            "Install with: pip install mpi4jax mpi4py"
        ) from exc

    n = data.shape[1]
    padded = jnp.zeros((6, n + 2, n + 2), dtype=data.dtype)

    # Place interior data.
    padded = padded.at[:, 1:-1, 1:-1].set(data)

    # --- Phase 1: Classify edges as local vs remote, group remote by rank ---
    local_edges = []   # (face, edge, nbr_face, nbr_edge, is_reversed)
    remote_by_rank = defaultdict(list)  # nbr_rank -> [(face, edge, nbr_face, nbr_edge, is_reversed), ...]

    for face in topology.local_face_ids:
        for edge in _EDGES:
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[
                (face, edge)
            ]
            nbr_rank = topology.neighbor_ranks[(face, edge)]

            entry = (face, edge, nbr_face, nbr_edge, is_reversed)
            if nbr_rank == topology.rank:
                local_edges.append(entry)
            else:
                remote_by_rank[nbr_rank].append(entry)

    # --- Phase 2: Handle local edges (no MPI) ---
    for face, edge, nbr_face, nbr_edge, is_reversed in local_edges:
        strip = _extract_edge_strip(data, nbr_face, nbr_edge)
        if is_reversed:
            strip = strip[::-1]
        padded = _place_strip(padded, face, edge, strip)

    # --- Phase 3: Handle remote edges, packed per neighbor rank ---
    for nbr_rank in sorted(remote_by_rank):
        edges = remote_by_rank[nbr_rank]
        n_edges = len(edges)
        # Canonical edge ordering for packed buffer correctness.
        # Both sides must iterate shared edges in the SAME order so
        # that strip i on the sender matches strip i on the receiver.
        # Canonical key = sender's (face, edge):
        #   - Sender sorts by (face, edge)           [its local key]
        #   - Receiver sorts by (nbr_face, nbr_edge) [= sender's key]
        # Without this, 2-rank and 3-rank decompositions corrupt halos.
        send_order = sorted(edges, key=lambda e: (e[0], e[1]))
        recv_order = sorted(edges, key=lambda e: (e[2], e[3]))

        # Pack all outgoing strips into one buffer.
        send_strips = []
        for face, edge, nbr_face, nbr_edge, is_reversed in send_order:
            send_strips.append(_extract_edge_strip(data, face, edge))
        send_buf = jnp.concatenate(send_strips, axis=0)  # (n_edges * n,)

        recv_buf = jnp.zeros(n_edges * n, dtype=data.dtype)

        # Single sendrecv per neighbor rank.
        # Tag encodes (sender_rank, receiver_rank) for uniqueness.
        send_tag = topology.rank * 1000 + nbr_rank
        recv_tag = nbr_rank * 1000 + topology.rank

        recv_buf = _mpi4jax_array_result(
            mpi4jax.sendrecv(
                send_buf,
                recv_buf,
                source=nbr_rank,
                dest=nbr_rank,
                sendtag=send_tag,
                recvtag=recv_tag,
                comm=MPI.COMM_WORLD,
            ),
        )

        # Unpack received buffer and place strips.
        for i, (face, edge, nbr_face, nbr_edge, is_reversed) in enumerate(recv_order):
            strip = recv_buf[i * n : (i + 1) * n]
            if is_reversed:
                strip = strip[::-1]
            padded = _place_strip(padded, face, edge, strip)

    return padded


def pad_halo_vector_mpi(
    u_data: jax.Array,
    v_data: jax.Array,
    cos_angle: jax.Array,
    sin_angle: jax.Array,
    cos_angle_padded: jax.Array,
    sin_angle_padded: jax.Array,
    topology: CommTopology,
) -> tuple[jax.Array, jax.Array]:
    """Pad vector components with MPI halo exchange.

    Same algorithm as :func:`legoesm.grids.halo.pad_halo_vector`
    but uses :func:`pad_halo_mpi` for the scalar exchanges.

    Parameters
    ----------
    u_data, v_data : jax.Array, shape (6, n, n)
        Grid-aligned velocity components.
    cos_angle, sin_angle : jax.Array, shape (6, n, n)
        Precomputed trig of grid rotation angle.
    cos_angle_padded, sin_angle_padded : jax.Array, shape (6, n+2, n+2)
        Precomputed trig of padded grid rotation angle.
    topology : CommTopology
        Pre-computed communication topology.

    Returns
    -------
    u_padded, v_padded : jax.Array, shape (6, n+2, n+2)
    """
    # Rotate to geographic (east, north).
    u_east = cos_angle * u_data - sin_angle * v_data
    v_north = sin_angle * u_data + cos_angle * v_data

    # Pad geographic components as scalars via MPI.
    u_east_padded = pad_halo_mpi(u_east, topology)
    v_north_padded = pad_halo_mpi(v_north, topology)

    # Rotate back to grid-aligned.
    u_padded = (
        cos_angle_padded * u_east_padded
        + sin_angle_padded * v_north_padded
    )
    v_padded = (
        -sin_angle_padded * u_east_padded
        + cos_angle_padded * v_north_padded
    )

    return u_padded, v_padded
