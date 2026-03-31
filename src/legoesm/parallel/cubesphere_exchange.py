"""Explicit cubed-sphere halo exchange for multi-GPU SPMD.

Replaces the implicit cross-shard reads that ``pad_halo`` generates
under face-axis sharding with **explicit collective operations**
(``lax.all_gather``) inside ``shard_map``.  This produces
``all-gather`` HLO collectives in the compiled XLA program, which
NCCL (GPU) and ICI (TPU) handle far more efficiently than the
dynamic-slice pattern that XLA synthesises for implicit reads.

Key design decisions
--------------------
1. Uses ``shard_map`` so each device runs per-face logic with an
   explicit ``all_gather`` for neighbor data.  This makes the
   communication visible to XLA's latency-hiding scheduler.

2. Supports **packed multi-field exchange**: callers can stack
   several ``(6, n, n, nlev)`` fields along the level axis, do
   ONE all_gather, then unstack — reducing the number of
   collective calls from O(fields) to O(1).

3. Falls back gracefully to the standard ``pad_halo`` when not
   running on a multi-device mesh (single GPU, CPU).

References
----------
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- JAX shard_map: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
"""

from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp

from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

logger = logging.getLogger("legoesm.parallel.cubesphere_exchange")

# ---------------------------------------------------------------------------
# Static connectivity tables as JAX arrays (built once, reused forever).
# NBR_FACES[face, edge]  = which face provides the halo data
# NBR_EDGES[face, edge]  = which edge of that neighbor to extract
# IS_REVERSED[face, edge] = whether the strip needs flipping
# ---------------------------------------------------------------------------

_NBR_FACES = jnp.array([
    [3, 1, 5, 4],   # face 0: W←3, E←1, S←5, N←4
    [0, 2, 5, 4],   # face 1
    [1, 3, 5, 4],   # face 2
    [2, 0, 5, 4],   # face 3
    [3, 1, 0, 2],   # face 4
    [3, 1, 2, 0],   # face 5
], dtype=jnp.int32)

_NBR_EDGES = jnp.array([
    [EAST, WEST, NORTH, SOUTH],    # face 0
    [EAST, WEST, EAST,  EAST],     # face 1
    [EAST, WEST, SOUTH, NORTH],    # face 2
    [EAST, WEST, WEST,  WEST],     # face 3
    [NORTH, NORTH, NORTH, NORTH],  # face 4
    [SOUTH, SOUTH, SOUTH, SOUTH],  # face 5
], dtype=jnp.int32)

_IS_REVERSED = jnp.array([
    [0, 0, 0, 0],  # face 0
    [0, 0, 1, 0],  # face 1
    [0, 0, 1, 1],  # face 2
    [0, 0, 0, 1],  # face 3
    [1, 0, 0, 1],  # face 4
    [0, 1, 1, 0],  # face 5
], dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Core exchange: 3D scalar  (6, n, n) → (6, n+2, n+2)
# ---------------------------------------------------------------------------

def _make_exchange_3d(mesh):
    """Return a shard_map-based 3D halo exchange function."""
    try:
        from jax.shard_map import shard_map  # JAX >= 0.8
    except (ImportError, ModuleNotFoundError):
        from jax.experimental.shard_map import shard_map

    P = jax.sharding.PartitionSpec

    @partial(shard_map, mesh=mesh,
             in_specs=P("face", None, None),
             out_specs=P("face", None, None),
             check_rep=False)
    def _exchange(local_shard):
        # local_shard: (1, n, n) — this device's face shard.
        n = local_shard.shape[1]
        my_face = local_shard[0]  # (n, n)

        # Explicit collective: gather all 6 face shards → (6, n, n).
        all_faces = jax.lax.all_gather(local_shard, "face",
                                       tiled=True)  # (6, n, n)

        my_idx = jax.lax.axis_index("face")  # scalar int (traced)

        # Look up my 4 neighbors.
        my_nbr_f = _NBR_FACES[my_idx]      # (4,)
        my_nbr_e = _NBR_EDGES[my_idx]      # (4,)
        my_rev   = _IS_REVERSED[my_idx]     # (4,)

        # Pre-extract ALL edge strips from ALL faces:
        #   edge 0 (WEST):  data[:, 0, :]   → (6, n)
        #   edge 1 (EAST):  data[:, -1, :]  → (6, n)
        #   edge 2 (SOUTH): data[:, :, 0]   → (6, n)
        #   edge 3 (NORTH): data[:, :, -1]  → (6, n)
        all_strips = jnp.stack([
            all_faces[:, 0, :],     # WEST  strips
            all_faces[:, -1, :],    # EAST  strips
            all_faces[:, :, 0],     # SOUTH strips
            all_faces[:, :, -1],    # NORTH strips
        ], axis=1)  # (6, 4, n)

        # Build padded output.
        padded = jnp.zeros((n + 2, n + 2), dtype=my_face.dtype)
        padded = padded.at[1:-1, 1:-1].set(my_face)

        # Fill the 4 halo edges (loop is unrolled at trace time).
        for e in range(4):
            strip = all_strips[my_nbr_f[e], my_nbr_e[e]]   # (n,)
            strip = jnp.where(my_rev[e], strip[::-1], strip)
            if e == WEST:
                padded = padded.at[0, 1:-1].set(strip)
            elif e == EAST:
                padded = padded.at[-1, 1:-1].set(strip)
            elif e == SOUTH:
                padded = padded.at[1:-1, 0].set(strip)
            else:  # NORTH
                padded = padded.at[1:-1, -1].set(strip)

        # Corner cells: average the two adjacent edge-halo values.
        padded = padded.at[0, 0].set(
            0.5 * (padded[0, 1] + padded[1, 0]))
        padded = padded.at[0, -1].set(
            0.5 * (padded[0, -2] + padded[1, -1]))
        padded = padded.at[-1, 0].set(
            0.5 * (padded[-1, 1] + padded[-2, 0]))
        padded = padded.at[-1, -1].set(
            0.5 * (padded[-1, -2] + padded[-2, -1]))

        return padded[None]  # (1, n+2, n+2) to match output sharding

    return _exchange


# ---------------------------------------------------------------------------
# Core exchange: 4D scalar  (6, n, n, nlev) → (6, n+2, n+2, nlev)
# ---------------------------------------------------------------------------

def _make_exchange_4d(mesh):
    """Return a shard_map-based 4D halo exchange function."""
    try:
        from jax.shard_map import shard_map  # JAX >= 0.8
    except (ImportError, ModuleNotFoundError):
        from jax.experimental.shard_map import shard_map

    P = jax.sharding.PartitionSpec

    @partial(shard_map, mesh=mesh,
             in_specs=P("face", None, None, None),
             out_specs=P("face", None, None, None),
             check_rep=False)
    def _exchange(local_shard):
        # local_shard: (1, n, n, C) — one face shard, C = nlev or packed.
        n = local_shard.shape[1]
        C = local_shard.shape[3]
        my_face = local_shard[0]  # (n, n, C)

        all_faces = jax.lax.all_gather(local_shard, "face",
                                       tiled=True)  # (6, n, n, C)

        my_idx = jax.lax.axis_index("face")
        my_nbr_f = _NBR_FACES[my_idx]
        my_nbr_e = _NBR_EDGES[my_idx]
        my_rev   = _IS_REVERSED[my_idx]

        # Edge strips: (6, 4, n, C)
        all_strips = jnp.stack([
            all_faces[:, 0, :, :],     # WEST
            all_faces[:, -1, :, :],    # EAST
            all_faces[:, :, 0, :],     # SOUTH
            all_faces[:, :, -1, :],    # NORTH
        ], axis=1)

        padded = jnp.zeros((n + 2, n + 2, C), dtype=my_face.dtype)
        padded = padded.at[1:-1, 1:-1, :].set(my_face)

        for e in range(4):
            strip = all_strips[my_nbr_f[e], my_nbr_e[e]]   # (n, C)
            strip = jnp.where(my_rev[e], strip[::-1], strip)
            if e == WEST:
                padded = padded.at[0, 1:-1, :].set(strip)
            elif e == EAST:
                padded = padded.at[-1, 1:-1, :].set(strip)
            elif e == SOUTH:
                padded = padded.at[1:-1, 0, :].set(strip)
            else:
                padded = padded.at[1:-1, -1, :].set(strip)

        # Corners
        padded = padded.at[0, 0, :].set(
            0.5 * (padded[0, 1, :] + padded[1, 0, :]))
        padded = padded.at[0, -1, :].set(
            0.5 * (padded[0, -2, :] + padded[1, -1, :]))
        padded = padded.at[-1, 0, :].set(
            0.5 * (padded[-1, 1, :] + padded[-2, 0, :]))
        padded = padded.at[-1, -1, :].set(
            0.5 * (padded[-1, -2, :] + padded[-2, -1, :]))

        return padded[None]  # (1, n+2, n+2, C)

    return _exchange


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_exchange_3d_cache: dict[int, object] = {}
_exchange_4d_cache: dict[int, object] = {}


def explicit_pad_halo(data, mesh, halo=1):
    """Explicit 3D halo exchange via shard_map + all_gather.

    Drop-in replacement for ``pad_halo`` that produces explicit
    collective operations in the XLA program.

    Parameters
    ----------
    data : jax.Array, shape (6, n, n)
    mesh : jax.sharding.Mesh
    halo : int
        Only ``halo=1`` is supported on the explicit path.

    Returns
    -------
    jax.Array, shape (6, n+2, n+2)
    """
    if halo != 1:
        from legoesm.grids.halo import _pad_halo_local, _pad_halo_local_h2
        return _pad_halo_local_h2(data) if halo == 2 else _pad_halo_local(data)

    mesh_id = id(mesh)
    if mesh_id not in _exchange_3d_cache:
        _exchange_3d_cache[mesh_id] = _make_exchange_3d(mesh)
    return _exchange_3d_cache[mesh_id](data)


def explicit_pad_halo_4d(data, mesh, halo=1):
    """Explicit 4D halo exchange via shard_map + all_gather.

    Drop-in replacement for ``pad_halo_4d``.  The trailing axis can
    hold vertical levels **or** packed multi-field channels (see
    :func:`packed_pad_halo_4d`).

    Parameters
    ----------
    data : jax.Array, shape (6, n, n, C)
    mesh : jax.sharding.Mesh
    halo : int

    Returns
    -------
    jax.Array, shape (6, n+2, n+2, C)
    """
    if halo != 1:
        from legoesm.grids.halo import _pad_halo_local_4d, _pad_halo_local_h2_4d
        fn = _pad_halo_local_h2_4d if halo == 2 else _pad_halo_local_4d
        return fn(data)

    mesh_id = id(mesh)
    if mesh_id not in _exchange_4d_cache:
        _exchange_4d_cache[mesh_id] = _make_exchange_4d(mesh)
    return _exchange_4d_cache[mesh_id](data)


# ---------------------------------------------------------------------------
# Packed multi-field exchange
# ---------------------------------------------------------------------------

def packed_pad_halo_4d(*fields, mesh):
    """Exchange halos for multiple fields in a single all_gather.

    Stacks the trailing (nlev) axes of all fields, performs ONE
    explicit halo exchange, then splits the result back.  This reduces
    the number of collective operations from ``len(fields)`` to 1.

    All fields must have shape ``(6, n, n, nlev_i)`` with the same
    ``(6, n, n)`` prefix.

    Parameters
    ----------
    *fields : jax.Array
        4D fields to exchange.
    mesh : jax.sharding.Mesh
        Device mesh with a ``"face"`` axis.

    Returns
    -------
    list[jax.Array]
        Padded fields, each ``(6, n+2, n+2, nlev_i)``.
    """
    if not fields:
        return []

    splits = [f.shape[-1] for f in fields]
    stacked = jnp.concatenate(fields, axis=-1)  # (6, n, n, sum(nlev_i))
    padded = explicit_pad_halo_4d(stacked, mesh, halo=1)
    return list(jnp.split(padded, jnp.cumsum(jnp.array(splits[:-1])), axis=-1))


# ---------------------------------------------------------------------------
# SPMD backend activation
# ---------------------------------------------------------------------------

_spmd_mesh = None


def activate_spmd_halo_backend(mesh) -> None:
    """Switch the global halo backend to explicit SPMD exchange.

    After this call, every ``pad_halo`` / ``pad_halo_4d`` invocation
    in the codebase will use ``shard_map`` + ``all_gather`` instead of
    the implicit cross-shard read path.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Device mesh with a ``"face"`` axis of size 6.
    """
    global _spmd_mesh
    _spmd_mesh = mesh
    from legoesm.grids import halo
    halo._halo_backend = "spmd"
    halo._spmd_mesh = mesh
    logger.info(
        "SPMD halo backend activated (mesh=%s, %d devices)",
        mesh.axis_names, len(mesh.devices.flat),
    )


def deactivate_spmd_halo_backend() -> None:
    """Revert to the default local halo backend."""
    global _spmd_mesh
    _spmd_mesh = None
    from legoesm.grids import halo
    halo._halo_backend = "local"
    halo._spmd_mesh = None
