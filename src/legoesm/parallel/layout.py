"""Distributed layout objects for rank-local ownership.

Describes how global cubed-sphere state ``(6, n, n, ...)`` is decomposed
across MPI ranks.  Two decomposition modes are supported:

1. **Face-only** (1–6 ranks): each rank owns one or more complete faces.
   Local shape: ``(n_local_faces, n, n, ...)``.

2. **Sub-face tiling** (6 × k² ranks, k ≥ 2): each face is split into
   a ``(k, k)`` tile grid; each rank owns one tile of one face.
   Local shape: ``(1, n_tile, n_tile, ...)``.

Key principle: ranks store **only local data**.  There are no zero-masked
global arrays.  ``scatter`` extracts local data from a global array;
``gather`` reconstructs the global array from local contributions via
MPI allgather.

Design
------
``DistributedLayout`` is an immutable NamedTuple that is cheap to
construct and safe to pass through JAX pytree boundaries.  It carries
enough information for scatter, gather, reduction, halo exchange, and
I/O to work without consulting any global singleton.

For single-rank execution, ``SingleRankLayout`` provides the same API
but is a no-op: scatter and gather are identity functions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


# =========================================================================
# Layout descriptors
# =========================================================================

class FaceOwnership(NamedTuple):
    """Describes which faces (or tiles) this rank owns.

    For face-only mode: ``face_ids`` has 1–6 entries, ``tile`` is
    ``(0, 0)``, ``tiling`` is ``(1, 1)``, ``tile_size`` equals ``n``.

    For tiled mode: ``face_ids`` has exactly one entry,
    ``tile`` is ``(ti, tj)``, ``tiling`` is ``(tx, ty)``,
    ``tile_size`` equals ``n // tx``.
    """
    face_ids: tuple[int, ...]
    tile: tuple[int, int]       # (ti, tj) within the face
    tiling: tuple[int, int]     # (tx, ty) tile grid per face
    tile_size: int              # n_tile (= n for face-only)


class DistributedLayout(NamedTuple):
    """Immutable descriptor of rank-local data ownership.

    Attributes
    ----------
    rank : int
        MPI rank of this process.
    n_ranks : int
        Total number of MPI ranks.
    global_n : int
        Global per-face grid size.
    ownership : FaceOwnership
        Which faces/tiles this rank owns.
    local_shape_2d : tuple[int, ...]
        Shape of a rank-local 2-D field: ``(n_local_faces, n_tile, n_tile)``
        for face-only, or ``(1, n_tile, n_tile)`` for tiled.
    is_tiled : bool
        True for sub-face tiling mode.
    """
    rank: int
    n_ranks: int
    global_n: int
    ownership: FaceOwnership
    local_shape_2d: tuple[int, ...]
    is_tiled: bool


class SingleRankLayout(NamedTuple):
    """Layout for single-rank (non-distributed) execution.

    Scatter and gather are identity.  This lets all code use the same
    API regardless of whether MPI is active.
    """
    rank: int           # always 0
    n_ranks: int        # always 1
    global_n: int
    local_shape_2d: tuple[int, ...]
    is_tiled: bool      # always False


# =========================================================================
# Constructors
# =========================================================================

def make_layout(rank: int, n_ranks: int, global_n: int) -> DistributedLayout | SingleRankLayout:
    """Create the appropriate layout for this rank.

    Parameters
    ----------
    rank : int
        MPI rank (0-based).
    n_ranks : int
        Total number of MPI ranks.
    global_n : int
        Per-face grid resolution.

    Returns
    -------
    DistributedLayout or SingleRankLayout
    """
    if n_ranks == 1:
        return SingleRankLayout(
            rank=0,
            n_ranks=1,
            global_n=global_n,
            local_shape_2d=(6, global_n, global_n),
            is_tiled=False,
        )

    if n_ranks <= 6:
        return _make_face_layout(rank, n_ranks, global_n)

    return _make_tiled_layout(rank, n_ranks, global_n)


def _make_face_layout(rank: int, n_ranks: int, global_n: int) -> DistributedLayout:
    """Face-only layout: each rank owns complete faces."""
    if 6 % n_ranks != 0:
        raise ValueError(
            f"n_ranks={n_ranks} does not evenly divide 6 faces. "
            f"Use 1, 2, 3, or 6."
        )
    faces_per_rank = 6 // n_ranks
    start = rank * faces_per_rank
    face_ids = tuple(range(start, start + faces_per_rank))

    ownership = FaceOwnership(
        face_ids=face_ids,
        tile=(0, 0),
        tiling=(1, 1),
        tile_size=global_n,
    )
    n_local = len(face_ids)
    return DistributedLayout(
        rank=rank,
        n_ranks=n_ranks,
        global_n=global_n,
        ownership=ownership,
        local_shape_2d=(n_local, global_n, global_n),
        is_tiled=False,
    )


def _make_tiled_layout(rank: int, n_ranks: int, global_n: int) -> DistributedLayout:
    """Sub-face tiled layout: each rank owns one tile of one face."""
    if n_ranks % 6 != 0:
        raise ValueError(
            f"n_ranks={n_ranks} is not a multiple of 6."
        )
    tiles_per_face = n_ranks // 6
    k = int(math.isqrt(tiles_per_face))
    if k * k != tiles_per_face:
        raise ValueError(
            f"n_ranks={n_ranks} gives tiles_per_face={tiles_per_face} "
            f"which is not a perfect square."
        )
    if global_n % k != 0:
        raise ValueError(
            f"global_n={global_n} is not divisible by tile factor k={k}."
        )

    face = rank // tiles_per_face
    tile_idx = rank % tiles_per_face
    ti = tile_idx // k
    tj = tile_idx % k
    n_tile = global_n // k

    ownership = FaceOwnership(
        face_ids=(face,),
        tile=(ti, tj),
        tiling=(k, k),
        tile_size=n_tile,
    )
    return DistributedLayout(
        rank=rank,
        n_ranks=n_ranks,
        global_n=global_n,
        ownership=ownership,
        local_shape_2d=(1, n_tile, n_tile),
        is_tiled=True,
    )


# =========================================================================
# Scatter: global → rank-local
# =========================================================================

def scatter(global_array: jax.Array, layout) -> jax.Array:
    """Extract the rank-local portion from a global ``(6, n, n, ...)`` array.

    Parameters
    ----------
    global_array : jax.Array
        Full global array with leading face dimension of size 6.
    layout : DistributedLayout or SingleRankLayout
        Layout descriptor for this rank.

    Returns
    -------
    jax.Array
        Rank-local array with shape ``(n_local_faces, n_tile, n_tile, ...)``.
    """
    if isinstance(layout, SingleRankLayout):
        return global_array

    own = layout.ownership
    trailing = global_array.shape[3:]  # everything after (6, n, n)

    if not layout.is_tiled:
        # Face-only: extract owned faces.
        indices = jnp.array(own.face_ids)
        return global_array[indices]
    else:
        # Tiled: extract the tile from the single owned face.
        face = own.face_ids[0]
        ti, tj = own.tile
        tx, ty = own.tiling
        nt = own.tile_size
        i0, j0 = ti * nt, tj * nt
        tile = global_array[face, i0:i0 + nt, j0:j0 + nt]
        return tile[jnp.newaxis]  # (1, n_tile, n_tile, ...)


def scatter_pytree(pytree, layout):
    """Apply :func:`scatter` to every array leaf in a pytree."""
    if isinstance(layout, SingleRankLayout):
        return pytree

    def _scatter_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        if leaf.ndim < 3 or leaf.shape[0] != 6:
            return leaf  # not a face-indexed array
        return scatter(leaf, layout)

    return jax.tree.map(_scatter_leaf, pytree)


# =========================================================================
# Gather: rank-local → global
# =========================================================================

def gather(local_array: jax.Array, layout) -> jax.Array:
    """Reconstruct a global ``(6, n, n, ...)`` array from rank-local data.

    For single-rank, this is identity.
    For multi-rank, uses MPI allgather + reassembly.

    Parameters
    ----------
    local_array : jax.Array
        Rank-local array.
    layout : DistributedLayout or SingleRankLayout
        Layout descriptor.

    Returns
    -------
    jax.Array
        Global array with shape ``(6, n, n, ...)``.
    """
    if isinstance(layout, SingleRankLayout):
        return local_array

    from legoesm.parallel.reductions import _require_mpi_stack, _mpi4jax_array_result

    mpi4jax, MPI = _require_mpi_stack()
    comm = MPI.COMM_WORLD
    own = layout.ownership
    n = layout.global_n
    trailing = local_array.shape[3:] if local_array.ndim > 3 else ()

    if not layout.is_tiled:
        # Face-only: each rank contributes its faces into the global array.
        # Use allgather to collect from all ranks, then reassemble.
        all_data = _mpi4jax_array_result(
            mpi4jax.allgather(local_array, comm=comm),
        )
        # all_data shape: (n_ranks, n_local_faces, n, n, ...)
        # Flatten to (6, n, n, ...) since total faces = 6.
        n_local = local_array.shape[0]
        return all_data.reshape((6,) + (n,) * 2 + trailing)
    else:
        # Tiled: each rank has (1, n_tile, n_tile, ...).
        # Allgather gives (n_ranks, 1, n_tile, n_tile, ...).
        all_tiles = _mpi4jax_array_result(
            mpi4jax.allgather(local_array, comm=comm),
        )
        # Reassemble into (6, n, n, ...).
        k = own.tiling[0]
        nt = own.tile_size
        tiles_per_face = k * k
        global_shape = (6, n, n) + trailing
        result = jnp.zeros(global_shape, dtype=local_array.dtype)
        for r in range(layout.n_ranks):
            face = r // tiles_per_face
            tile_idx = r % tiles_per_face
            ti = tile_idx // k
            tj = tile_idx % k
            i0, j0 = ti * nt, tj * nt
            result = result.at[face, i0:i0 + nt, j0:j0 + nt].set(
                all_tiles[r, 0]
            )
        return result


def gather_pytree(pytree, layout):
    """Apply :func:`gather` to every array leaf in a pytree."""
    if isinstance(layout, SingleRankLayout):
        return pytree

    def _gather_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        # Heuristic: only gather face-indexed arrays.
        # Face-only: leading dim = n_local_faces; tiled: leading dim = 1.
        own = layout.ownership
        expected_leading = len(own.face_ids) if not layout.is_tiled else 1
        if leaf.ndim < 3 or leaf.shape[0] != expected_leading:
            return leaf
        nt = own.tile_size
        if leaf.shape[1] != nt or leaf.shape[2] != nt:
            return leaf
        return gather(leaf, layout)

    return jax.tree.map(_gather_leaf, pytree)


# =========================================================================
# Local reductions
# =========================================================================

def local_sum(field: jax.Array, weights: jax.Array | None = None) -> jax.Array:
    """Compute a weighted sum over local data only.

    Parameters
    ----------
    field : jax.Array
        Rank-local field, shape ``(n_faces, ni, nj, ...)``.
    weights : jax.Array, optional
        Area weights, same spatial shape as *field*.

    Returns
    -------
    jax.Array
        Scalar local contribution.
    """
    if weights is not None:
        field = field * weights
    return jnp.sum(field)


def local_max(field: jax.Array) -> jax.Array:
    """Return the maximum over local data."""
    return jnp.max(field)


def local_min(field: jax.Array) -> jax.Array:
    """Return the minimum over local data."""
    return jnp.min(field)


def global_reduce(local_value: jax.Array, layout, op: str = "sum") -> jax.Array:
    """MPI-reduce a local scalar to a global scalar.

    For single-rank, returns ``local_value`` unchanged.

    Parameters
    ----------
    local_value : jax.Array
        Scalar contribution from this rank.
    layout : DistributedLayout or SingleRankLayout
    op : ``"sum"``, ``"max"``, or ``"min"``
    """
    if isinstance(layout, SingleRankLayout):
        return local_value

    from legoesm.parallel.reductions import (
        global_sum_mpi, global_max_mpi, global_min_mpi,
    )
    _ops = {"sum": global_sum_mpi, "max": global_max_mpi, "min": global_min_mpi}
    if op not in _ops:
        raise ValueError(f"Unknown op={op!r}. Choose from {list(_ops)}.")
    return _ops[op](local_value)
