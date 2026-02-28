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
)
from legoesm.parallel.comm import CommTopology


_EDGES = (WEST, EAST, SOUTH, NORTH)


def pad_halo_mpi(
    data: jax.Array,
    topology: CommTopology,
) -> jax.Array:
    """Pad a scalar field with inter-face halo data using MPI.

    This mirrors :func:`legoesm.grids.halo.pad_halo` but uses
    ``mpi4jax.sendrecv`` for edges that cross MPI rank boundaries.

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

    # Token for MPI ordering inside JIT.
    token = jax.lax.create_token()

    for face in topology.local_face_ids:
        for edge in _EDGES:
            nbr_face, nbr_edge, is_reversed = topology.neighbor_info[
                (face, edge)
            ]
            nbr_rank = topology.neighbor_ranks[(face, edge)]

            if nbr_rank == topology.rank:
                # Local exchange — no MPI needed.
                strip = _extract_edge_strip(data, nbr_face, nbr_edge)
            else:
                # Send our edge strip to the neighbor, receive theirs.
                send_strip = _extract_edge_strip(data, face, edge)
                recv_strip = jnp.zeros(n, dtype=data.dtype)

                # Unique tag from (face, edge, nbr_face, nbr_edge).
                tag = face * 100 + edge * 10 + nbr_face
                recv_strip, token = mpi4jax.sendrecv(
                    send_strip,
                    recv_strip,
                    source=nbr_rank,
                    dest=nbr_rank,
                    sendtag=tag,
                    recvtag=nbr_face * 100 + nbr_edge * 10 + face,
                    comm=MPI.COMM_WORLD,
                    token=token,
                )
                strip = recv_strip

            # Reverse if needed.
            if is_reversed:
                strip = strip[::-1]

            # Place in halo position.
            if edge == WEST:
                padded = padded.at[face, 0, 1:-1].set(strip)
            elif edge == EAST:
                padded = padded.at[face, -1, 1:-1].set(strip)
            elif edge == SOUTH:
                padded = padded.at[face, 1:-1, 0].set(strip)
            elif edge == NORTH:
                padded = padded.at[face, 1:-1, -1].set(strip)

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
