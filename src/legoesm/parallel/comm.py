"""Communication topology for distributed cubed-sphere.

Pre-computes the mapping between face edges and MPI ranks,
determining which process owns each neighbor face for halo exchange.

Face-to-rank assignment (deterministic, contiguous):
    6 processes → face *i* on rank *i*
    3 processes → rank 0: [0,1], rank 1: [2,3], rank 2: [4,5]
    2 processes → rank 0: [0,1,2], rank 1: [3,4,5]
    1 process  → all local (no MPI needed)
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
        Same information as :data:`CONNECTIVITY` but filtered for
        local faces.
    """
    rank: int
    n_processes: int
    local_face_ids: tuple[int, ...]
    neighbor_ranks: dict  # (face, edge) -> rank
    neighbor_info: dict   # (face, edge) -> (nbr_face, nbr_edge, reversed)


def _face_to_rank(face: int, n_processes: int) -> int:
    """Deterministic mapping: face index → MPI rank.

    Faces are assigned contiguously:
        faces_per_rank = 6 // n_processes
        rank = face // faces_per_rank
    """
    faces_per_rank = _N_FACES // n_processes
    return face // faces_per_rank


def _rank_to_faces(rank: int, n_processes: int) -> tuple[int, ...]:
    """Return the global face indices owned by *rank*."""
    faces_per_rank = _N_FACES // n_processes
    start = rank * faces_per_rank
    return tuple(range(start, start + faces_per_rank))


def build_comm_topology(rank: int, n_processes: int) -> CommTopology:
    """Build the communication topology for a given rank.

    Parameters
    ----------
    rank : int
        This process's MPI rank.
    n_processes : int
        Total number of MPI processes.  Must divide 6.

    Returns
    -------
    CommTopology
    """
    if _N_FACES % n_processes != 0:
        raise ValueError(
            f"n_processes={n_processes} does not evenly divide "
            f"{_N_FACES} faces.  Choose 1, 2, 3, or 6."
        )

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
    )
