"""Distributed halo exchange using mpi4jax.

Provides MPI-based halo exchange for multi-node cubed-sphere runs.
The single-node version (using JAX's automatic SPMD) is in
:mod:`legoesm.grids.halo` and requires no changes.

This module is only imported when MPI is active.  ``mpi4jax`` is
an optional dependency.

Supports two modes:

1. **Face-only** (≤6 MPI ranks): each rank owns one or more full faces.
   Halo exchange happens between faces on different ranks.

2. **Sub-face tiling** (6 × k² ranks): each rank owns one tile of one
   face.  Halo exchange happens both between tiles on the same face
   (intra-face) and between tiles on adjacent faces (inter-face).

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
Face-only mode: edges to the same neighbor rank are packed into a
single send buffer, reducing MPI messages from ~4 per face to ~2-3
per rank.

Sub-face tiling mode: one sendrecv per edge direction (4 total).
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
    _extract_edge_strip_at_depth,
)
from legoesm.parallel.comm import CommTopology
from legoesm.parallel.reductions import _mpi4jax_array_result


_EDGES = (WEST, EAST, SOUTH, NORTH)
_DIR_TO_EDGE = {"west": WEST, "east": EAST, "south": SOUTH, "north": NORTH}
_OPPOSITE_EDGE = {WEST: EAST, EAST: WEST, SOUTH: NORTH, NORTH: SOUTH}


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


def _place_strip_h2(
    padded: jax.Array, face: int, edge: int,
    strip_d0: jax.Array, strip_d1: jax.Array,
) -> jax.Array:
    """Place two received strips into halo=2 positions.

    depth 0 = adjacent to interior, depth 1 = outer.
    Interior is at [2:-2, 2:-2].
    """
    n = padded.shape[1] - 4
    if edge == WEST:
        padded = padded.at[face, 1, 2:-2].set(strip_d0)
        padded = padded.at[face, 0, 2:-2].set(strip_d1)
    elif edge == EAST:
        padded = padded.at[face, n + 2, 2:-2].set(strip_d0)
        padded = padded.at[face, n + 3, 2:-2].set(strip_d1)
    elif edge == SOUTH:
        padded = padded.at[face, 2:-2, 1].set(strip_d0)
        padded = padded.at[face, 2:-2, 0].set(strip_d1)
    elif edge == NORTH:
        padded = padded.at[face, 2:-2, n + 2].set(strip_d0)
        padded = padded.at[face, 2:-2, n + 3].set(strip_d1)
    return padded


# ======================================================================
# Face-only MPI halo exchange (≤6 ranks)
# ======================================================================

def _pad_halo_mpi_face_only(
    data: jax.Array,
    topology: CommTopology,
    halo: int,
    mpi4jax,
    MPI,
) -> jax.Array:
    """Face-only halo exchange (original behavior)."""
    n = data.shape[1]
    h2 = 2 * halo
    padded = jnp.zeros((6, n + h2, n + h2), dtype=data.dtype)
    padded = padded.at[:, halo:-halo, halo:-halo].set(data)

    # --- Classify edges as local vs remote, group remote by rank ---
    local_edges = []
    remote_by_rank = defaultdict(list)

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

    # --- Handle local edges (no MPI) ---
    for face, edge, nbr_face, nbr_edge, is_reversed in local_edges:
        if halo == 1:
            strip = _extract_edge_strip(data, nbr_face, nbr_edge)
            if is_reversed:
                strip = strip[::-1]
            padded = _place_strip(padded, face, edge, strip)
        else:
            strip_d0 = _extract_edge_strip_at_depth(data, nbr_face, nbr_edge, 0)
            strip_d1 = _extract_edge_strip_at_depth(data, nbr_face, nbr_edge, 1)
            if is_reversed:
                strip_d0 = strip_d0[::-1]
                strip_d1 = strip_d1[::-1]
            padded = _place_strip_h2(padded, face, edge, strip_d0, strip_d1)

    # --- Handle remote edges, packed per neighbor rank ---
    strips_per_edge = halo  # 1 strip for halo=1, 2 strips for halo=2
    for nbr_rank in sorted(remote_by_rank):
        edges = remote_by_rank[nbr_rank]
        n_edges = len(edges)
        send_order = sorted(edges, key=lambda e: (e[0], e[1]))
        recv_order = sorted(edges, key=lambda e: (e[2], e[3]))

        # Pack all outgoing strips into one buffer.
        send_strips = []
        for face, edge, nbr_face, nbr_edge, is_reversed in send_order:
            if halo == 1:
                send_strips.append(_extract_edge_strip(data, face, edge))
            else:
                for depth in range(halo):
                    send_strips.append(
                        _extract_edge_strip_at_depth(data, face, edge, depth)
                    )
        send_buf = jnp.concatenate(send_strips, axis=0)

        recv_buf = jnp.zeros(n_edges * strips_per_edge * n, dtype=data.dtype)

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
            if halo == 1:
                strip = recv_buf[i * n : (i + 1) * n]
                if is_reversed:
                    strip = strip[::-1]
                padded = _place_strip(padded, face, edge, strip)
            else:
                base = i * halo * n
                strip_d0 = recv_buf[base : base + n]
                strip_d1 = recv_buf[base + n : base + 2 * n]
                if is_reversed:
                    strip_d0 = strip_d0[::-1]
                    strip_d1 = strip_d1[::-1]
                padded = _place_strip_h2(padded, face, edge, strip_d0, strip_d1)

    return padded


# ======================================================================
# Sub-face tiling MPI halo exchange (>6 ranks)
# ======================================================================

def _pad_halo_mpi_tiled(
    data: jax.Array,
    topology: CommTopology,
    halo: int,
    mpi4jax,
    MPI,
) -> jax.Array:
    """Sub-face tiling halo exchange.

    Each rank owns one tile of one face.  For each edge direction:
    - If the neighbor is on the same face (intra-face): simple exchange,
      no index reversal needed.
    - If the neighbor is on a different face (inter-face boundary):
      exchange with possible reversal from CONNECTIVITY.

    Tags use ``rank * 4 + edge`` to uniquely identify each message.
    """
    face = topology.local_face_ids[0]
    n = data.shape[1]  # tile dimension (same for i and j since tx == ty)
    h2 = 2 * halo
    padded = jnp.zeros((6, n + h2, n + h2), dtype=data.dtype)
    padded = padded.at[:, halo:-halo, halo:-halo].set(data)

    for direction, edge in _DIR_TO_EDGE.items():
        tile_nbr_rank = topology.tile_neighbors.get(direction)

        if tile_nbr_rank is not None:
            # ----- Intra-face tile exchange (no reversal) -----
            opp = _OPPOSITE_EDGE[edge]
            # Tags: I send with tag = my_rank*4+edge, receive with tag = nbr*4+opp
            sendtag = topology.rank * 4 + edge
            recvtag = tile_nbr_rank * 4 + opp

            if halo == 1:
                my_strip = _extract_edge_strip(data, face, edge)
                recv_buf = jnp.zeros(n, dtype=data.dtype)
                recv_strip = _mpi4jax_array_result(
                    mpi4jax.sendrecv(
                        my_strip, recv_buf,
                        source=tile_nbr_rank, dest=tile_nbr_rank,
                        sendtag=sendtag, recvtag=recvtag,
                        comm=MPI.COMM_WORLD,
                    ),
                )
                padded = _place_strip(padded, face, edge, recv_strip)
            else:
                strips = []
                for depth in range(halo):
                    strips.append(
                        _extract_edge_strip_at_depth(data, face, edge, depth)
                    )
                send_buf = jnp.concatenate(strips, axis=0)
                recv_buf = jnp.zeros(halo * n, dtype=data.dtype)
                recv_data = _mpi4jax_array_result(
                    mpi4jax.sendrecv(
                        send_buf, recv_buf,
                        source=tile_nbr_rank, dest=tile_nbr_rank,
                        sendtag=sendtag, recvtag=recvtag,
                        comm=MPI.COMM_WORLD,
                    ),
                )
                strip_d0 = recv_data[:n]
                strip_d1 = recv_data[n : 2 * n]
                padded = _place_strip_h2(padded, face, edge, strip_d0, strip_d1)

        elif (face, edge) in topology.neighbor_info:
            # ----- Inter-face boundary exchange (with possible reversal) -----
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[(face, edge)]
            nbr_rank = topology.neighbor_ranks[(face, edge)]

            sendtag = topology.rank * 4 + edge
            recvtag = nbr_rank * 4 + nbr_edge

            if halo == 1:
                my_strip = _extract_edge_strip(data, face, edge)
                recv_buf = jnp.zeros(n, dtype=data.dtype)
                recv_strip = _mpi4jax_array_result(
                    mpi4jax.sendrecv(
                        my_strip, recv_buf,
                        source=nbr_rank, dest=nbr_rank,
                        sendtag=sendtag, recvtag=recvtag,
                        comm=MPI.COMM_WORLD,
                    ),
                )
                if is_reversed:
                    recv_strip = recv_strip[::-1]
                padded = _place_strip(padded, face, edge, recv_strip)
            else:
                strips = []
                for depth in range(halo):
                    strips.append(
                        _extract_edge_strip_at_depth(data, face, edge, depth)
                    )
                send_buf = jnp.concatenate(strips, axis=0)
                recv_buf = jnp.zeros(halo * n, dtype=data.dtype)
                recv_data = _mpi4jax_array_result(
                    mpi4jax.sendrecv(
                        send_buf, recv_buf,
                        source=nbr_rank, dest=nbr_rank,
                        sendtag=sendtag, recvtag=recvtag,
                        comm=MPI.COMM_WORLD,
                    ),
                )
                strip_d0 = recv_data[:n]
                strip_d1 = recv_data[n : 2 * n]
                if is_reversed:
                    strip_d0 = strip_d0[::-1]
                    strip_d1 = strip_d1[::-1]
                padded = _place_strip_h2(padded, face, edge, strip_d0, strip_d1)

        # else: no neighbor in this direction (shouldn't happen on a cube)

    return padded


# ======================================================================
# Main entry point
# ======================================================================

def pad_halo_mpi(
    data: jax.Array,
    topology: CommTopology,
    halo: int = 1,
) -> jax.Array:
    """Pad a scalar field with halo data using MPI.

    Dispatches to face-only or sub-face tiling mode based on the
    topology's tiling configuration.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
        Scalar field on the cubed-sphere.  Non-local faces should
        contain valid data for local exchanges or zeros.
    topology : CommTopology
        Pre-computed communication topology.
    halo : int
        Halo width (1 or 2).

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo)
    """
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "MPI halo exchange requires mpi4jax and mpi4py. "
            "Install with: pip install mpi4jax mpi4py"
        ) from exc

    if topology.tiling != (1, 1):
        return _pad_halo_mpi_tiled(data, topology, halo, mpi4jax, MPI)
    return _pad_halo_mpi_face_only(data, topology, halo, mpi4jax, MPI)


def pad_halo_vector_mpi(
    u_data: jax.Array,
    v_data: jax.Array,
    cos_angle: jax.Array,
    sin_angle: jax.Array,
    cos_angle_padded: jax.Array,
    sin_angle_padded: jax.Array,
    topology: CommTopology,
    halo: int = 1,
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
    cos_angle_padded, sin_angle_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
        Precomputed trig of padded grid rotation angle.
    topology : CommTopology
        Pre-computed communication topology.
    halo : int
        Halo width (1 or 2).

    Returns
    -------
    u_padded, v_padded : jax.Array, shape (6, n+2*halo, n+2*halo)
    """
    # Rotate to geographic (east, north).
    u_east = cos_angle * u_data - sin_angle * v_data
    v_north = sin_angle * u_data + cos_angle * v_data

    # Pad geographic components as scalars via MPI.
    u_east_padded = pad_halo_mpi(u_east, topology, halo=halo)
    v_north_padded = pad_halo_mpi(v_north, topology, halo=halo)

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
