"""Communication topology for distributed cubed-sphere.

Supports two decomposition modes:

1. **Face-only** (1, 2, 3, or 6 processes):
   Each process owns one or more complete faces.

2. **Sub-face tiling** (6 × tx × ty processes):
   Each face is split into a (tx × ty) tile grid.
   Each process owns one tile of one face.
   Halo exchange happens both between faces AND between tiles
   on the same face.

Face-to-rank assignment (deterministic, contiguous):

    Face-only:
        6 processes → face *i* on rank *i*
        3 processes → rank 0: [0,1], rank 1: [2,3], rank 2: [4,5]
        2 processes → rank 0: [0,1,2], rank 1: [3,4,5]
        1 process  → all local

    Sub-face:
        N processes where N = 6 × tx × ty
        rank = face * (tx*ty) + tile_i * ty + tile_j
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

# Number of cubed-sphere faces.
_N_FACES = 6
_EDGES = (WEST, EAST, SOUTH, NORTH)


class CommTopology(NamedTuple):
    """Pre-computed communication pattern for MPI halo exchange.

    Attributes
    ----------
    rank : int
        This process's MPI rank.
    n_processes : int
        Total number of MPI processes.
    local_face_ids : tuple[int, ...]
        Global face indices owned by this process.
    neighbor_ranks : dict[tuple[int, int], int]
        Maps ``(face, edge)`` → rank of process owning the neighbor.
    neighbor_info : dict[tuple[int, int], tuple[int, int, bool]]
        Maps ``(face, edge)`` → ``(neighbor_face, neighbor_edge, reversed)``.
    tiling : tuple[int, int]
        Sub-face tile grid ``(tx, ty)``.  ``(1, 1)`` for face-only.
    tile_index : tuple[int, int]
        This rank's tile position ``(ti, tj)`` within its face.
        ``(0, 0)`` for face-only mode.
    tile_neighbors : dict[str, int | None]
        Maps direction ``"west"/"east"/"south"/"north"`` to the MPI rank
        of the neighboring tile on the SAME face, or ``None`` if
        the neighbor is on a different face (handled by inter-face halo).
        Empty dict for face-only mode.
    """
    rank: int
    n_processes: int
    local_face_ids: tuple[int, ...]
    neighbor_ranks: dict  # (face, edge) -> rank
    neighbor_info: dict   # (face, edge) -> (nbr_face, nbr_edge, reversed)
    tiling: tuple[int, int]
    tile_index: tuple[int, int]
    tile_neighbors: dict  # direction -> rank or None


# ==============================================================================
# Face-only decomposition
# ==============================================================================

def _face_to_rank(face: int, n_processes: int) -> int:
    """Deterministic mapping: face index → MPI rank (face-only mode)."""
    faces_per_rank = _N_FACES // n_processes
    return face // faces_per_rank


def _rank_to_faces(rank: int, n_processes: int) -> tuple[int, ...]:
    """Return the global face indices owned by *rank* (face-only mode)."""
    faces_per_rank = _N_FACES // n_processes
    start = rank * faces_per_rank
    return tuple(range(start, start + faces_per_rank))


# ==============================================================================
# Sub-face tiling decomposition
# ==============================================================================

def _tile_rank(face: int, ti: int, tj: int, tx: int, ty: int) -> int:
    """Map (face, tile_i, tile_j) to MPI rank for sub-face tiling."""
    return face * (tx * ty) + ti * ty + tj


def _rank_to_tile(rank: int, tx: int, ty: int) -> tuple[int, int, int]:
    """Map MPI rank to (face, tile_i, tile_j) for sub-face tiling."""
    tiles_per_face = tx * ty
    face = rank // tiles_per_face
    tile_idx = rank % tiles_per_face
    ti = tile_idx // ty
    tj = tile_idx % ty
    return face, ti, tj


def _build_tile_neighbors(
    face: int, ti: int, tj: int, tx: int, ty: int, n_processes: int
) -> dict[str, int | None]:
    """Find neighboring tile ranks on the SAME face.

    Returns None for edges that border another face (handled by
    inter-face halo exchange).
    """
    neighbors = {}

    # West neighbor (ti-1)
    if ti > 0:
        neighbors["west"] = _tile_rank(face, ti - 1, tj, tx, ty)
    else:
        neighbors["west"] = None  # inter-face boundary

    # East neighbor (ti+1)
    if ti < tx - 1:
        neighbors["east"] = _tile_rank(face, ti + 1, tj, tx, ty)
    else:
        neighbors["east"] = None  # inter-face boundary

    # South neighbor (tj-1)
    if tj > 0:
        neighbors["south"] = _tile_rank(face, ti, tj - 1, tx, ty)
    else:
        neighbors["south"] = None  # inter-face boundary

    # North neighbor (tj+1)
    if tj < ty - 1:
        neighbors["north"] = _tile_rank(face, ti, tj + 1, tx, ty)
    else:
        neighbors["north"] = None  # inter-face boundary

    return neighbors


# ==============================================================================
# Main topology builder
# ==============================================================================

def build_comm_topology(rank: int, n_processes: int) -> CommTopology:
    """Build the communication topology for a given rank.

    Automatically selects face-only or sub-face tiling based on
    ``n_processes``.

    Parameters
    ----------
    rank : int
        This process's MPI rank.
    n_processes : int
        Total number of MPI processes.

        - 1, 2, 3, or 6: face-only decomposition.
        - Multiple of 6 and >6: sub-face tiling with
          ``tiles_per_face = n_processes // 6``.
        - Other values: error.

    Returns
    -------
    CommTopology
    """
    # --- Face-only mode ---
    if n_processes <= 6:
        if _N_FACES % n_processes != 0:
            raise ValueError(
                f"n_processes={n_processes} does not evenly divide "
                f"{_N_FACES} faces.  Choose 1, 2, 3, or 6."
            )
        return _build_face_only(rank, n_processes)

    # --- Sub-face tiling mode ---
    if n_processes % 6 != 0:
        raise ValueError(
            f"n_processes={n_processes} is not a multiple of 6. "
            f"For >6 processes, use a multiple of 6 (e.g., 12, 24, 54, 96, 150, 384, 600)."
        )

    tiles_per_face = n_processes // 6
    # Require square tiles (tx == ty) because inter-face boundaries with
    # axis swaps need matching strip lengths on both sides.
    import math
    t = int(math.isqrt(tiles_per_face))
    if t * t != tiles_per_face:
        raise ValueError(
            f"n_processes={n_processes} gives tiles_per_face={tiles_per_face} "
            f"which is not a perfect square.  Sub-face tiling requires "
            f"n_processes = 6 × k² (e.g., 24, 54, 96, 150, 384, 600)."
        )
    tx, ty = t, t

    return _build_tiled(rank, n_processes, tx, ty)


def _build_face_only(rank: int, n_processes: int) -> CommTopology:
    """Build face-only topology (original behavior)."""
    local_faces = _rank_to_faces(rank, n_processes)

    neighbor_ranks: dict[tuple[int, int], int] = {}
    neighbor_info: dict[tuple[int, int], tuple[int, int, bool]] = {}

    for face in local_faces:
        for edge in _EDGES:
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
            nbr_rank = _face_to_rank(nbr_face, n_processes)
            neighbor_ranks[(face, edge)] = nbr_rank
            neighbor_info[(face, edge)] = (nbr_face, nbr_edge, is_reversed)

    return CommTopology(
        rank=rank,
        n_processes=n_processes,
        local_face_ids=local_faces,
        neighbor_ranks=neighbor_ranks,
        neighbor_info=neighbor_info,
        tiling=(1, 1),
        tile_index=(0, 0),
        tile_neighbors={},
    )


def _build_tiled(rank: int, n_processes: int, tx: int, ty: int) -> CommTopology:
    """Build sub-face tiling topology."""
    face, ti, tj = _rank_to_tile(rank, tx, ty)

    # This rank owns exactly one tile of one face.
    local_faces = (face,)

    # Inter-face neighbors (only relevant for edge tiles).
    neighbor_ranks: dict[tuple[int, int], int] = {}
    neighbor_info: dict[tuple[int, int], tuple[int, int, bool]] = {}

    for edge in _EDGES:
        nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
        # Determine which tile on the neighbor face borders this tile.
        # This depends on which edge we're at and the tile position.
        if edge == WEST and ti == 0:
            # Border tile on west edge → neighbor face's east border
            nbr_ti = tx - 1
            nbr_tj = tj  # May need reversal
            if is_reversed:
                nbr_tj = ty - 1 - tj
            nbr_rank = _tile_rank(nbr_face, nbr_ti, nbr_tj, tx, ty)
            neighbor_ranks[(face, edge)] = nbr_rank
            neighbor_info[(face, edge)] = (nbr_face, nbr_edge, is_reversed)
        elif edge == EAST and ti == tx - 1:
            nbr_ti = 0
            nbr_tj = tj
            if is_reversed:
                nbr_tj = ty - 1 - tj
            nbr_rank = _tile_rank(nbr_face, nbr_ti, nbr_tj, tx, ty)
            neighbor_ranks[(face, edge)] = nbr_rank
            neighbor_info[(face, edge)] = (nbr_face, nbr_edge, is_reversed)
        elif edge == SOUTH and tj == 0:
            nbr_ti = ti
            nbr_tj = ty - 1
            if is_reversed:
                nbr_ti = tx - 1 - ti
            nbr_rank = _tile_rank(nbr_face, nbr_ti, nbr_tj, tx, ty)
            neighbor_ranks[(face, edge)] = nbr_rank
            neighbor_info[(face, edge)] = (nbr_face, nbr_edge, is_reversed)
        elif edge == NORTH and tj == ty - 1:
            nbr_ti = ti
            nbr_tj = 0
            if is_reversed:
                nbr_ti = tx - 1 - ti
            nbr_rank = _tile_rank(nbr_face, nbr_ti, nbr_tj, tx, ty)
            neighbor_ranks[(face, edge)] = nbr_rank
            neighbor_info[(face, edge)] = (nbr_face, nbr_edge, is_reversed)

    # Intra-face tile neighbors
    tile_neighbors = _build_tile_neighbors(face, ti, tj, tx, ty, n_processes)

    return CommTopology(
        rank=rank,
        n_processes=n_processes,
        local_face_ids=local_faces,
        neighbor_ranks=neighbor_ranks,
        neighbor_info=neighbor_info,
        tiling=(tx, ty),
        tile_index=(ti, tj),
        tile_neighbors=tile_neighbors,
    )
