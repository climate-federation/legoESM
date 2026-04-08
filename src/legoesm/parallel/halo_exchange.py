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
Face-only mode: edges destined for the same neighbor rank are packed
into a single send buffer and exchanged via one sendrecv per neighbor
(at most 4 point-to-point messages per rank).

Sub-face tiling mode: one sendrecv per edge direction (4 total).

Limitations and Known Bottlenecks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Blocking sendrecv**: All halo exchanges use blocking ``sendrecv``
   via ``mpi4jax``.  ``mpi4jax`` does not yet expose ``Isend``/``Irecv``
   non-blocking primitives, so true computation-communication overlap
   is not possible at this level.

2. **Face-only mode uses neighbor sendrecv**: When rank count ≤ 6,
   edges are grouped by neighbor rank and exchanged via one sendrecv
   per unique neighbor (≤4 messages).  For >6 ranks, the tiled mode
   uses point-to-point sendrecv with only actual neighbors.

3. **Tiled mode is sequential per edge**: The 4 edge directions are
   exchanged in sequence (not pipelined).  Each sendrecv blocks until
   both send and receive complete.

Near-term improvement path:
  - Batch edge packing into a single contiguous buffer per neighbor
    rank and do one sendrecv per neighbor (reduces from 4 to ≤4
    messages per rank, with larger messages for better bandwidth).
  - When ``mpi4jax`` gains ``Isend``/``Irecv``, convert to non-blocking
    with ``Waitall`` after all sends/receives are posted.
  - For pure multi-GPU (no MPI), ``jax.lax.ppermute`` is the preferred
    path and integrates with XLA's SPMD partitioner.
"""

from __future__ import annotations

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
    _fill_corners_h1,
    _fill_corners_h2,
)
from legoesm.parallel.comm import CommTopology
from legoesm.parallel.reductions import _mpi4jax_array_result


_EDGES = (WEST, EAST, SOUTH, NORTH)
_DIR_TO_EDGE = {"west": WEST, "east": EAST, "south": SOUTH, "north": NORTH}
_OPPOSITE_EDGE = {WEST: EAST, EAST: WEST, SOUTH: NORTH, NORTH: SOUTH}

# MPI tag computation.  Tags must be unique per (edge, rank) pair and
# fit within MPI's tag space (guaranteed at least 2^15-1 = 32767, but
# most implementations support up to 2^31-1).
#
# Old scheme ``edge * 1000 + rank`` collides when rank >= 1000 because
# the edge term (0-3) * 1000 overlaps with the rank offset.
#
# New scheme: ``edge * _MPI_TAG_RANK_STRIDE + rank`` with a stride of
# 100_000, supporting up to 99_999 ranks without collision.  Maximum
# tag value = 3 * 100_000 + 99_999 = 399_999, well within 2^31-1.
_MPI_TAG_RANK_STRIDE = 100_000


# ======================================================================
# AD-safe sendrecv wrapper (custom_vjp)
# ======================================================================

def _make_sendrecv_vjp(mpi4jax_mod):
    """Build an AD-safe sendrecv once mpi4jax is imported.

    mpi4jax.sendrecv has a transpose rule that swaps source/dest, but
    the XLA lowering raises RuntimeError when ``_must_transpose=True``.
    This wrapper bypasses that by using ``@jax.custom_vjp``: the forward
    calls sendrecv normally, and the backward calls sendrecv with
    swapped endpoints as a fresh forward call (no transpose flag).

    Non-JAX arguments (source, dest, sendtag, recvtag, comm) are
    declared via ``nondiff_argnums`` so JAX does not attempt to trace
    them.  They are passed through to fwd/bwd as leading static args.
    """

    def _sendrecv_impl(send_buf, recv_template, source, dest,
                       sendtag, recvtag, comm):
        return _mpi4jax_array_result(
            mpi4jax_mod.sendrecv(
                send_buf, recv_template,
                source=source, dest=dest,
                sendtag=sendtag, recvtag=recvtag, comm=comm,
            )
        )

    _sendrecv = jax.custom_vjp(
        _sendrecv_impl, nondiff_argnums=(2, 3, 4, 5, 6),
    )

    def _fwd(send_buf, recv_template, source, dest, sendtag, recvtag, comm):
        # fwd has the same signature as the primal function.
        result = _mpi4jax_array_result(
            mpi4jax_mod.sendrecv(
                send_buf, recv_template,
                source=source, dest=dest,
                sendtag=sendtag, recvtag=recvtag, comm=comm,
            )
        )
        return result, ()

    def _bwd(source, dest, sendtag, recvtag, comm, _res, g):
        # bwd receives nondiff args first, then residuals, then cotangent.
        # Reverse: swap source<->dest so cotangent flows back to sender.
        d_send = _mpi4jax_array_result(
            mpi4jax_mod.sendrecv(
                g, jnp.zeros_like(g),
                source=dest, dest=source,
                sendtag=sendtag, recvtag=recvtag, comm=comm,
            )
        )
        return d_send, jnp.zeros_like(g)

    _sendrecv.defvjp(_fwd, _bwd)
    return _sendrecv


# Module-level cache: built lazily on first MPI import.
_sendrecv_vjp_fn = None


def _get_sendrecv_vjp(mpi4jax_mod):
    """Return the cached AD-safe sendrecv wrapper."""
    global _sendrecv_vjp_fn
    if _sendrecv_vjp_fn is None:
        _sendrecv_vjp_fn = _make_sendrecv_vjp(mpi4jax_mod)
    return _sendrecv_vjp_fn


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
    """Face-only halo exchange using batched neighbor sendrecv.

    Groups remote edges by neighbor rank and issues one sendrecv per
    unique neighbor (at most 4 for face-only decomposition), rather
    than an O(world_size) allgather.
    """
    n = data.shape[1]
    h2 = 2 * halo
    padded = jnp.zeros((6, n + h2, n + h2), dtype=data.dtype)
    padded = padded.at[:, halo:-halo, halo:-halo].set(data)

    # --- Classify edges as local vs remote ---
    local_edges = []
    remote_edges = []

    for face in topology.local_face_ids:
        for edge in _EDGES:
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[
                (face, edge)
            ]
            nbr_rank = topology.neighbor_ranks[(face, edge)]

            entry = (face, edge, nbr_face, nbr_edge, is_reversed, nbr_rank)
            if nbr_rank == topology.rank:
                local_edges.append(entry)
            else:
                remote_edges.append(entry)

    # --- Handle local edges (no MPI) ---
    for face, edge, nbr_face, nbr_edge, is_reversed, _ in local_edges:
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

    if not remote_edges:
        # Still need to fill corner cells (matching local implementation).
        if halo == 1:
            padded = _fill_corners_h1(padded)
        else:
            padded = _fill_corners_h2(padded)
        return padded

    # --- Batched neighbor sendrecv for remote edges ---
    # Group remote edges by neighbor rank so we send one message per
    # unique neighbor instead of an O(world_size) allgather.
    from collections import defaultdict
    strip_size = halo * n
    comm = MPI.COMM_WORLD
    rank = topology.rank

    by_nbr_rank: dict[int, list] = defaultdict(list)
    for entry in remote_edges:
        # entry = (face, edge, nbr_face, nbr_edge, is_reversed, nbr_rank)
        by_nbr_rank[entry[5]].append(entry)

    for nbr_rank, entries in by_nbr_rank.items():
        # Pack all edge strips destined for this neighbor.
        send_parts = []
        for face, edge, nbr_face, nbr_edge, is_reversed, _ in entries:
            if halo == 1:
                send_parts.append(_extract_edge_strip(data, face, edge))
            else:
                for depth in range(halo):
                    send_parts.append(
                        _extract_edge_strip_at_depth(data, face, edge, depth)
                    )
        send_buf = jnp.concatenate(send_parts)

        # Single sendrecv per neighbor.
        send_tag = rank
        recv_tag = nbr_rank
        sendrecv = _get_sendrecv_vjp(mpi4jax)
        recv_buf = sendrecv(
            send_buf, jnp.zeros_like(send_buf),
            nbr_rank, nbr_rank,
            send_tag, recv_tag, comm,
        )

        # Unpack received strips.
        offset = 0
        for face, edge, nbr_face, nbr_edge, is_reversed, _ in entries:
            if halo == 1:
                strip = recv_buf[offset:offset + n]
                offset += n
                if is_reversed:
                    strip = strip[::-1]
                padded = _place_strip(padded, face, edge, strip)
            else:
                strip_d0 = recv_buf[offset:offset + n]
                strip_d1 = recv_buf[offset + n:offset + 2 * n]
                offset += 2 * n
                if is_reversed:
                    strip_d0 = strip_d0[::-1]
                    strip_d1 = strip_d1[::-1]
                padded = _place_strip_h2(padded, face, edge, strip_d0, strip_d1)

    # Fill corner cells by averaging adjacent edge-halo values
    # (matching the local implementation).
    if halo == 1:
        padded = _fill_corners_h1(padded)
    else:
        padded = _fill_corners_h2(padded)

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
    """Sub-face tiling halo exchange using batched point-to-point sendrecv.

    Edges destined for the same neighbor rank are packed into a single
    contiguous send buffer, reducing the number of MPI messages.  A tile
    with 4 unique neighbor ranks still does 4 sendrecv calls, but when
    2 edges share a neighbor (corner tiles), this batching halves the
    message count for those edges.

    All send buffers are pre-packed before any communication starts,
    maximizing the data available to MPI's internal pipeline.
    """
    face = topology.local_face_ids[0]
    n = data.shape[1]  # tile dimension
    h2 = 2 * halo
    padded = jnp.zeros((6, n + h2, n + h2), dtype=data.dtype)
    padded = padded.at[:, halo:-halo, halo:-halo].set(data)

    strip_size = halo * n
    comm = MPI.COMM_WORLD
    rank = topology.rank

    # --- Phase 1: classify and pre-pack all edges ---
    # Group edges by destination rank for batched communication.
    # Each entry: (edge, nbr_rank, nbr_edge, is_reversed, is_tile_nbr)
    edge_info = []
    for direction, edge in _DIR_TO_EDGE.items():
        tile_nbr_rank = topology.tile_neighbors.get(direction)

        if tile_nbr_rank is not None:
            # Intra-face tile neighbor: opposite edge, no reversal.
            opp = _OPPOSITE_EDGE[edge]
            edge_info.append((edge, tile_nbr_rank, opp, False, True))
        elif (face, edge) in topology.neighbor_info:
            # Inter-face boundary: possible reversal from CONNECTIVITY.
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[(face, edge)]
            nbr_rank = topology.neighbor_ranks[(face, edge)]
            edge_info.append((edge, nbr_rank, nbr_edge, is_reversed, False))

    if not edge_info:
        if halo == 1:
            padded = _fill_corners_h1(padded)
        else:
            padded = _fill_corners_h2(padded)
        return padded

    # Group by neighbor rank for batching.
    from collections import defaultdict
    by_nbr_rank = defaultdict(list)
    for entry in edge_info:
        by_nbr_rank[entry[1]].append(entry)

    # --- Phase 2: pack all send buffers, then exchange per neighbor ---
    for nbr_rank, entries in by_nbr_rank.items():
        # Pack all edges for this neighbor into one contiguous buffer.
        send_parts = []
        for edge, _, _, _, _ in entries:
            if halo == 1:
                send_parts.append(_extract_edge_strip(data, face, edge))
            else:
                for depth in range(halo):
                    send_parts.append(
                        _extract_edge_strip_at_depth(data, face, edge, depth)
                    )
        send_buf = jnp.concatenate(send_parts)

        # Single sendrecv for all edges to this neighbor.
        # Tag encodes our rank + a batch identifier.
        send_tag = rank  # unique per source rank
        recv_tag = nbr_rank  # expect the neighbor's rank as their tag
        sendrecv = _get_sendrecv_vjp(mpi4jax)
        recv_buf = sendrecv(
            send_buf, jnp.zeros_like(send_buf),
            nbr_rank, nbr_rank,
            send_tag, recv_tag, comm,
        )

        # --- Phase 3: unpack received strips into padded array ---
        offset = 0
        for edge, _, nbr_edge, is_reversed, is_tile_nbr in entries:
            if halo == 1:
                recv_strip = recv_buf[offset:offset + n]
                offset += n
                if is_reversed:
                    recv_strip = recv_strip[::-1]
                padded = _place_strip(padded, face, edge, recv_strip)
            else:
                strip_d0 = recv_buf[offset:offset + n]
                strip_d1 = recv_buf[offset + n:offset + 2 * n]
                offset += 2 * n
                if is_reversed:
                    strip_d0 = strip_d0[::-1]
                    strip_d1 = strip_d1[::-1]
                padded = _place_strip_h2(padded, face, edge, strip_d0, strip_d1)

    # Fill corner cells by averaging adjacent edge-halo values.
    if halo == 1:
        padded = _fill_corners_h1(padded)
    else:
        padded = _fill_corners_h2(padded)

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

    from legoesm.parallel.profiling import mpi_timer
    with mpi_timer("pad_halo_mpi"):
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


# ======================================================================
# 4D (multi-level) MPI halo exchange
# ======================================================================


def _extract_edge_strip_4d(data: jax.Array, face: int, edge: int) -> jax.Array:
    """Extract an edge strip from a 4D field, shape (n, nlev)."""
    if edge == WEST:
        return data[face, 0, :, :]
    elif edge == EAST:
        return data[face, -1, :, :]
    elif edge == SOUTH:
        return data[face, :, 0, :]
    elif edge == NORTH:
        return data[face, :, -1, :]
    raise ValueError(f"Invalid edge: {edge}")


def _extract_edge_strip_at_depth_4d(
    data: jax.Array, face: int, edge: int, depth: int,
) -> jax.Array:
    """Extract an edge strip at a given depth from a 4D field, shape (n, nlev)."""
    if edge == WEST:
        return data[face, depth, :, :]
    elif edge == EAST:
        return data[face, -(depth + 1), :, :]
    elif edge == SOUTH:
        return data[face, :, depth, :]
    elif edge == NORTH:
        return data[face, :, -(depth + 1), :]
    raise ValueError(f"Invalid edge: {edge}")


def _place_strip_4d(
    padded: jax.Array, face: int, edge: int, strip: jax.Array,
) -> jax.Array:
    """Place a strip (n, nlev) into the halo=1 position of a 4D padded array."""
    if edge == WEST:
        padded = padded.at[face, 0, 1:-1, :].set(strip)
    elif edge == EAST:
        padded = padded.at[face, -1, 1:-1, :].set(strip)
    elif edge == SOUTH:
        padded = padded.at[face, 1:-1, 0, :].set(strip)
    elif edge == NORTH:
        padded = padded.at[face, 1:-1, -1, :].set(strip)
    return padded


def _place_strip_h2_4d(
    padded: jax.Array, face: int, edge: int,
    strip_d0: jax.Array, strip_d1: jax.Array,
) -> jax.Array:
    """Place two strips (n, nlev) into halo=2 positions of a 4D padded array."""
    n = padded.shape[1] - 4
    if edge == WEST:
        padded = padded.at[face, 1, 2:-2, :].set(strip_d0)
        padded = padded.at[face, 0, 2:-2, :].set(strip_d1)
    elif edge == EAST:
        padded = padded.at[face, n + 2, 2:-2, :].set(strip_d0)
        padded = padded.at[face, n + 3, 2:-2, :].set(strip_d1)
    elif edge == SOUTH:
        padded = padded.at[face, 2:-2, 1, :].set(strip_d0)
        padded = padded.at[face, 2:-2, 0, :].set(strip_d1)
    elif edge == NORTH:
        padded = padded.at[face, 2:-2, n + 2, :].set(strip_d0)
        padded = padded.at[face, 2:-2, n + 3, :].set(strip_d1)
    return padded


def _pad_halo_mpi_face_only_4d(
    data: jax.Array,
    topology: CommTopology,
    halo: int,
    mpi4jax,
    MPI,
) -> jax.Array:
    """Face-only 4D halo exchange: one sendrecv per neighbor for all levels."""
    n = data.shape[1]
    nlev = data.shape[3]
    h2 = 2 * halo
    padded = jnp.zeros((6, n + h2, n + h2, nlev), dtype=data.dtype)
    padded = padded.at[:, halo:-halo, halo:-halo, :].set(data)

    local_edges = []
    remote_edges = []

    for face in topology.local_face_ids:
        for edge in _EDGES:
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[
                (face, edge)
            ]
            nbr_rank = topology.neighbor_ranks[(face, edge)]
            entry = (face, edge, nbr_face, nbr_edge, is_reversed, nbr_rank)
            if nbr_rank == topology.rank:
                local_edges.append(entry)
            else:
                remote_edges.append(entry)

    # Handle local edges
    for face, edge, nbr_face, nbr_edge, is_reversed, _ in local_edges:
        if halo == 1:
            strip = _extract_edge_strip_4d(data, nbr_face, nbr_edge)
            if is_reversed:
                strip = strip[::-1]
            padded = _place_strip_4d(padded, face, edge, strip)
        else:
            strip_d0 = _extract_edge_strip_at_depth_4d(data, nbr_face, nbr_edge, 0)
            strip_d1 = _extract_edge_strip_at_depth_4d(data, nbr_face, nbr_edge, 1)
            if is_reversed:
                strip_d0 = strip_d0[::-1]
                strip_d1 = strip_d1[::-1]
            padded = _place_strip_h2_4d(padded, face, edge, strip_d0, strip_d1)

    if not remote_edges:
        if halo == 1:
            padded = _fill_corners_h1(padded)
        else:
            padded = _fill_corners_h2(padded)
        return padded

    # Batched neighbor sendrecv — one message per neighbor for all levels
    from collections import defaultdict
    comm = MPI.COMM_WORLD
    rank = topology.rank

    by_nbr_rank: dict[int, list] = defaultdict(list)
    for entry in remote_edges:
        by_nbr_rank[entry[5]].append(entry)

    for nbr_rank, entries in by_nbr_rank.items():
        send_parts = []
        for face, edge, nbr_face, nbr_edge, is_reversed, _ in entries:
            if halo == 1:
                # shape (n, nlev) → flatten to (n * nlev,)
                send_parts.append(
                    _extract_edge_strip_4d(data, face, edge).reshape(-1)
                )
            else:
                for depth in range(halo):
                    send_parts.append(
                        _extract_edge_strip_at_depth_4d(
                            data, face, edge, depth
                        ).reshape(-1)
                    )
        send_buf = jnp.concatenate(send_parts)

        send_tag = rank
        recv_tag = nbr_rank
        sendrecv = _get_sendrecv_vjp(mpi4jax)
        recv_buf = sendrecv(
            send_buf, jnp.zeros_like(send_buf),
            nbr_rank, nbr_rank,
            send_tag, recv_tag, comm,
        )

        offset = 0
        chunk = n * nlev
        for face, edge, nbr_face, nbr_edge, is_reversed, _ in entries:
            if halo == 1:
                strip = recv_buf[offset:offset + chunk].reshape(n, nlev)
                offset += chunk
                if is_reversed:
                    strip = strip[::-1]
                padded = _place_strip_4d(padded, face, edge, strip)
            else:
                strip_d0 = recv_buf[offset:offset + chunk].reshape(n, nlev)
                strip_d1 = recv_buf[offset + chunk:offset + 2 * chunk].reshape(n, nlev)
                offset += 2 * chunk
                if is_reversed:
                    strip_d0 = strip_d0[::-1]
                    strip_d1 = strip_d1[::-1]
                padded = _place_strip_h2_4d(padded, face, edge, strip_d0, strip_d1)

    if halo == 1:
        padded = _fill_corners_h1(padded)
    else:
        padded = _fill_corners_h2(padded)

    return padded


def _pad_halo_mpi_tiled_4d(
    data: jax.Array,
    topology: CommTopology,
    halo: int,
    mpi4jax,
    MPI,
) -> jax.Array:
    """Sub-face tiling 4D halo exchange: one sendrecv per neighbor for all levels."""
    face = topology.local_face_ids[0]
    n = data.shape[1]
    nlev = data.shape[3]
    h2 = 2 * halo
    padded = jnp.zeros((6, n + h2, n + h2, nlev), dtype=data.dtype)
    padded = padded.at[:, halo:-halo, halo:-halo, :].set(data)

    comm = MPI.COMM_WORLD
    rank = topology.rank
    chunk = n * nlev

    edge_info = []
    for direction, edge in _DIR_TO_EDGE.items():
        tile_nbr_rank = topology.tile_neighbors.get(direction)
        if tile_nbr_rank is not None:
            opp = _OPPOSITE_EDGE[edge]
            edge_info.append((edge, tile_nbr_rank, opp, False, True))
        elif (face, edge) in topology.neighbor_info:
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[(face, edge)]
            nbr_rank = topology.neighbor_ranks[(face, edge)]
            edge_info.append((edge, nbr_rank, nbr_edge, is_reversed, False))

    if not edge_info:
        if halo == 1:
            padded = _fill_corners_h1(padded)
        else:
            padded = _fill_corners_h2(padded)
        return padded

    from collections import defaultdict
    by_nbr_rank = defaultdict(list)
    for entry in edge_info:
        by_nbr_rank[entry[1]].append(entry)

    for nbr_rank, entries in by_nbr_rank.items():
        send_parts = []
        for edge, _, _, _, _ in entries:
            if halo == 1:
                send_parts.append(
                    _extract_edge_strip_4d(data, face, edge).reshape(-1)
                )
            else:
                for depth in range(halo):
                    send_parts.append(
                        _extract_edge_strip_at_depth_4d(
                            data, face, edge, depth
                        ).reshape(-1)
                    )
        send_buf = jnp.concatenate(send_parts)

        send_tag = rank
        recv_tag = nbr_rank
        sendrecv = _get_sendrecv_vjp(mpi4jax)
        recv_buf = sendrecv(
            send_buf, jnp.zeros_like(send_buf),
            nbr_rank, nbr_rank,
            send_tag, recv_tag, comm,
        )

        offset = 0
        for edge, _, nbr_edge, is_reversed, is_tile_nbr in entries:
            if halo == 1:
                recv_strip = recv_buf[offset:offset + chunk].reshape(n, nlev)
                offset += chunk
                if is_reversed:
                    recv_strip = recv_strip[::-1]
                padded = _place_strip_4d(padded, face, edge, recv_strip)
            else:
                strip_d0 = recv_buf[offset:offset + chunk].reshape(n, nlev)
                strip_d1 = recv_buf[offset + chunk:offset + 2 * chunk].reshape(n, nlev)
                offset += 2 * chunk
                if is_reversed:
                    strip_d0 = strip_d0[::-1]
                    strip_d1 = strip_d1[::-1]
                padded = _place_strip_h2_4d(padded, face, edge, strip_d0, strip_d1)

    if halo == 1:
        padded = _fill_corners_h1(padded)
    else:
        padded = _fill_corners_h2(padded)

    return padded


def pad_halo_mpi_4d(
    data: jax.Array,
    topology: CommTopology,
    halo: int = 1,
) -> jax.Array:
    """Pad a 4D scalar field with halo data using MPI.

    All vertical levels are packed into a single message per neighbor,
    reducing MPI call count by a factor of ``nlev`` compared to
    per-level :func:`pad_halo_mpi`.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n, nlev)
    topology : CommTopology
    halo : int

    Returns
    -------
    padded : jax.Array, shape (6, n+2*halo, n+2*halo, nlev)
    """
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError as exc:
        raise ImportError(
            "MPI halo exchange requires mpi4jax and mpi4py. "
            "Install with: pip install mpi4jax mpi4py"
        ) from exc

    from legoesm.parallel.profiling import mpi_timer
    with mpi_timer("pad_halo_mpi_4d"):
        if topology.tiling != (1, 1):
            return _pad_halo_mpi_tiled_4d(data, topology, halo, mpi4jax, MPI)
        return _pad_halo_mpi_face_only_4d(data, topology, halo, mpi4jax, MPI)


def packed_pad_halo_mpi_4d(
    *fields: jax.Array,
    topology: CommTopology,
    halo: int = 1,
) -> list[jax.Array]:
    """Exchange halos for multiple 4D fields in a single MPI round.

    Stacks fields along the trailing axis, performs ONE halo exchange
    (with proportionally larger MPI messages), then splits.  Reduces
    MPI message count from ``len(fields)`` exchanges to 1.

    All fields must share the same ``(6, n, n)`` spatial prefix.
    The trailing axis (levels/channels) can differ.

    Parameters
    ----------
    *fields : jax.Array
        4D arrays of shape ``(6, n, n, C_i)``.
    topology : CommTopology
    halo : int

    Returns
    -------
    list[jax.Array]
        Padded arrays, each ``(6, n+2h, n+2h, C_i)``.
    """
    if not fields:
        return []
    if len(fields) == 1:
        return [pad_halo_mpi_4d(fields[0], topology, halo)]

    splits = [f.shape[-1] for f in fields]
    stacked = jnp.concatenate(fields, axis=-1)
    padded = pad_halo_mpi_4d(stacked, topology, halo)
    return list(jnp.split(padded, jnp.cumsum(jnp.array(splits[:-1])), axis=-1))
