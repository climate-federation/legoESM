"""Domain decomposition for Voronoi (MPAS-style) meshes.

Partitions an unstructured Voronoi mesh across MPI ranks or JAX devices
and constructs local sub-meshes with halo (ghost) entities for parallel
stencil computation.

Two partitioning methods:

1. **Geometric (RCB)**: Recursive Coordinate Bisection on cell-center
   Cartesian coordinates.  No external dependencies.
2. **METIS** (optional): k-way graph partitioning via ``pymetis``.

After partitioning, each rank holds owned + halo entities.  The halo
exchange (:mod:`legoesm.parallel.halo_exchange_voronoi`) updates halo
values from their owning ranks between timesteps.

Usage
-----
::

    partition = partition_voronoi_mesh(mesh, n_ranks=4, rank=0)
    local_mesh = build_local_mesh(mesh, partition)

    # In the time loop, exchange halo data:
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
    halo = VoronoiHaloExchange(partition, backend="mpi")
    h_local = halo.exchange_cell_field(h_local)
"""

from __future__ import annotations

from typing import NamedTuple

import importlib.util
import logging

import numpy as np
import jax.numpy as jnp

from legoesm.grids.voronoi import VoronoiMesh

logger = logging.getLogger("legoesm.parallel.voronoi_partition")

# One-time log guard so a per-rank/per-call "auto" resolution does not spam.
_AUTO_METHOD_LOGGED = False

# Hilbert space-filling-curve resolution: a 2^order x 2^order (lat, lon) grid.
# order=10 -> 1024^2 ~ 1.05e6 buckets, finer than any production Voronoi mesh
# (level-9 SCVT ~2.6e6 cells is the practical ceiling; ties break by stable
# sort), so distinct cells almost never collide. Module constant, not config:
# it is a numerics resolution knob, not a tunable.
_DEFAULT_HILBERT_ORDER = 10


def _metis_available() -> bool:
    """True if the optional ``pymetis`` graph-partitioning package is importable."""
    return importlib.util.find_spec("pymetis") is not None


def resolve_partition_method(method: str) -> str:
    """Resolve a partition method, expanding ``"auto"`` by available capability.

    ``"auto"`` (the default) selects ``"metis"`` when ``pymetis`` is importable —
    graph partitioning minimizes the edge cut, giving better load balance and
    smaller halos on irregular/variable-resolution meshes (the MPAS lesson:
    geometric RCB leaves lopsided cell counts and fat halos at scale) — and
    otherwise falls back to ``"geometric"`` (RCB, no dependency).

    ``"geometric"``, ``"metis"``, and any unknown value pass through UNCHANGED so
    the caller's own dispatch guard still raises on an unknown method. Returns the
    concrete method name.
    """
    global _AUTO_METHOD_LOGGED
    if method != "auto":
        return method
    chosen = "metis" if _metis_available() else "geometric"
    if not _AUTO_METHOD_LOGGED:
        _AUTO_METHOD_LOGGED = True
        if chosen == "metis":
            logger.info(
                "Voronoi partition method='auto' -> 'metis' (pymetis available; "
                "graph partitioning for load balance + smaller halos)."
            )
        else:
            logger.info(
                "Voronoi partition method='auto' -> 'geometric' RCB (pymetis not "
                "installed; `pip install pymetis` for better load balance at scale)."
            )
    return chosen


# ============================================================================
# Data structures
# ============================================================================

class HaloCommSchedule(NamedTuple):
    """Communication schedule for halo exchange of one entity type.

    For neighbor rank ``neighbor_ranks[i]``:

    - Send ``send_counts[i]`` values starting at cumulative offset in
      ``send_idx``.
    - Recv ``recv_counts[i]`` values starting at cumulative offset in
      ``recv_idx``.
    """
    neighbor_ranks: tuple[int, ...]
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    send_idx: jnp.ndarray   # (total_send,) local indices to pack
    recv_idx: jnp.ndarray   # (total_recv,) local indices to fill


class BatchedHaloSchedule(NamedTuple):
    """Union-neighbor comm schedule joining the cell + edge index spaces.

    Built once at layout-build time by :func:`build_batched_halo_schedule`
    from a partition's ``cell_comm`` and ``edge_comm``.  Lets the MPAS
    state exchange send ONE message per neighbor per dtype group (u edges
    + T/p_s cells + tracer cells packed into a single flat buffer) instead
    of one message per neighbor per entity exchange.

    ``neighbor_ranks`` is the sorted union of the cell and edge neighbor
    lists.  A rank present in only one of the two entity schedules gets
    zero counts for the other entity (zero-length pack segments).  The
    union relation is symmetric across ranks whenever the underlying
    entity schedules are (rank A lists B iff B lists A) — see
    ``tests/distributed/test_voronoi_batched_halo.py`` for the mechanical
    cross-rank check.

    For union neighbor ``i``:

    - cell send rows: ``cell_send_idx[sum(cell_send_counts[:i]) : ... +
      cell_send_counts[i]]`` (local OWNED cell indices to pack);
    - cell recv rows: same slicing of ``cell_recv_idx`` (local HALO cell
      indices to fill);
    - edge send/recv rows: identical layout in ``edge_send_idx`` /
      ``edge_recv_idx``.

    All counts are Python ints (layout constants) so every pack/unpack
    slice has a static shape under JIT.  Concatenating the per-neighbor
    recv rows in union order yields exactly ``cell_recv_idx`` /
    ``edge_recv_idx``, so the unpack can do a single functional scatter
    per field.
    """
    neighbor_ranks: tuple[int, ...]
    cell_send_counts: tuple[int, ...]
    cell_recv_counts: tuple[int, ...]
    edge_send_counts: tuple[int, ...]
    edge_recv_counts: tuple[int, ...]
    cell_send_idx: jnp.ndarray   # (total_cell_send,) local indices to pack
    cell_recv_idx: jnp.ndarray   # (total_cell_recv,) local indices to fill
    edge_send_idx: jnp.ndarray   # (total_edge_send,)
    edge_recv_idx: jnp.ndarray   # (total_edge_recv,)

    def messages_per_exchange(self, n_dtype_groups: int = 1) -> int:
        """Messages one batched state exchange posts per rank.

        Pure schedule math for the homogeneous case where every dtype
        group touches both index spaces (the expected production case:
        all prognostic fields share one dtype, so ``n_dtype_groups=1``).
        For heterogeneous groups (e.g. a cell-only dtype group facing an
        edge-only neighbor) the exact count is
        :func:`legoesm.parallel.halo_exchange_voronoi.count_batched_messages`,
        which never exceeds this bound.
        """
        return len(self.neighbor_ranks) * n_dtype_groups


class VoronoiPartition(NamedTuple):
    """Domain decomposition descriptor for one rank of a Voronoi mesh.

    Entities are ordered: owned first (sorted by global index), then
    halo (sorted by global index).
    """
    rank: int
    n_ranks: int

    # Global counts
    nCells_global: int
    nEdges_global: int
    nVertices_global: int

    # Owned counts
    n_owned_cells: int
    n_owned_edges: int
    n_owned_vertices: int

    # Local counts (owned + halo)
    n_local_cells: int
    n_local_edges: int
    n_local_vertices: int

    # Global indices of local entities (owned first, then halo)
    local_cells: np.ndarray       # (n_local_cells,)
    local_edges: np.ndarray       # (n_local_edges,)
    local_vertices: np.ndarray    # (n_local_vertices,)

    # Global-to-local mapping (-1 for non-local entities)
    cell_g2l: np.ndarray          # (nCells_global,)
    edge_g2l: np.ndarray          # (nEdges_global,)
    vertex_g2l: np.ndarray        # (nVertices_global,)

    # Communication schedules
    cell_comm: HaloCommSchedule
    edge_comm: HaloCommSchedule
    vertex_comm: HaloCommSchedule


# ============================================================================
# Partitioners
# ============================================================================

def partition_cells_geometric(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
    """Partition cells via Recursive Coordinate Bisection (RCB).

    Uses cell-center Cartesian coordinates on the unit sphere.

    Parameters
    ----------
    mesh : VoronoiMesh
    n_ranks : int

    Returns
    -------
    cell_owner : np.ndarray, shape (nCells,), dtype int32
        ``cell_owner[c]`` is the rank that owns cell ``c``.
    """
    coords = np.stack([
        np.asarray(mesh.xCell) / mesh.radius,
        np.asarray(mesh.yCell) / mesh.radius,
        np.asarray(mesh.zCell) / mesh.radius,
    ], axis=1)
    return _rcb(coords, n_ranks)


def _rcb(coords: np.ndarray, n_ranks: int) -> np.ndarray:
    """Recursive Coordinate Bisection on a point cloud."""
    n = len(coords)
    if n_ranks <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int32)

    axis = int(np.argmax(np.ptp(coords, axis=0)))
    order = np.argsort(coords[:, axis])

    n_left_ranks = n_ranks // 2
    n_right_ranks = n_ranks - n_left_ranks
    split = max(1, min(n - 1, n * n_left_ranks // n_ranks))

    left, right = order[:split], order[split:]
    result = np.empty(n, dtype=np.int32)
    result[left] = _rcb(coords[left], n_left_ranks)
    result[right] = _rcb(coords[right], n_right_ranks) + n_left_ranks
    return result


def partition_cells_metis(mesh: VoronoiMesh, n_ranks: int) -> np.ndarray:
    """Partition cells via METIS k-way graph partitioning.

    Requires the ``pymetis`` package.

    Parameters
    ----------
    mesh : VoronoiMesh
    n_ranks : int

    Returns
    -------
    cell_owner : np.ndarray, shape (nCells,), dtype int32
    """
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError(
            "METIS partitioning requires pymetis.  "
            "Install with: pip install pymetis"
        ) from exc

    coc = np.asarray(mesh.cellsOnCell)
    nec = np.asarray(mesh.nEdgesOnCell)
    adjacency = []
    for c in range(mesh.nCells):
        nbrs = [int(coc[k, c]) for k in range(int(nec[c])) if coc[k, c] >= 0]
        adjacency.append(np.array(nbrs, dtype=np.int32))

    _, membership = pymetis.part_graph(n_ranks, adjacency=adjacency)
    return np.array(membership, dtype=np.int32)


def _hilbert_xy2d(order: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Hilbert-curve distance ``d`` for integer grid coords ``(x, y)``.

    Vectorized form of the canonical Wikipedia ``xy2d`` integer algorithm on a
    ``2^order x 2^order`` grid (rotation uses the full side length ``n``, not the
    current level ``s``).  Returns a bijection ``[0, n)^2 -> [0, n^2)`` whose
    1-D ordering preserves 2-D locality: cells adjacent on the curve are spatially
    close, which keeps each contiguous partition compact (small halo surface).
    """
    n = 1 << order
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    d = np.zeros(x.shape, dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rot(n, x, y, rx, ry): reflect when ry==0 (and x,y when rx==1), then swap.
        ry0 = ry == 0
        flip = ry0 & (rx == 1)
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        tx = np.where(ry0, y, x)
        ty = np.where(ry0, x, y)
        x, y = tx, ty
        s >>= 1
    return d


def hilbert_cell_keys(mesh: VoronoiMesh, order: int = _DEFAULT_HILBERT_ORDER) -> np.ndarray:
    """Per-cell Hilbert space-filling-curve key from cell (lat, lon).

    Maps each cell center to a ``2^order x 2^order`` (lon, lat) grid and returns
    its Hilbert distance.  Sorting cells by this key yields a 1-D ordering with
    strong 2-D spatial locality — used to build compact, contiguous partitions
    and locality-friendly local indexings.

    Parameters
    ----------
    mesh : VoronoiMesh
    order : int
        SFC grid resolution (side = ``2^order``).

    Returns
    -------
    np.ndarray, shape (nCells,), dtype int64
    """
    two_pi = 2.0 * np.pi
    lon = np.mod(np.asarray(mesh.lonCell, dtype=np.float64), two_pi)
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    n = 1 << order
    u = lon / two_pi                       # [0, 1)
    v = (lat + 0.5 * np.pi) / np.pi        # [0, 1]
    gx = np.clip((u * n).astype(np.int64), 0, n - 1)
    gy = np.clip((v * n).astype(np.int64), 0, n - 1)
    return _hilbert_xy2d(order, gx, gy)


def partition_cells_sfc(
    mesh: VoronoiMesh, n_ranks: int, order: int = _DEFAULT_HILBERT_ORDER,
) -> np.ndarray:
    """Partition cells into contiguous Hilbert space-filling-curve chunks.

    Orders cells along a Hilbert curve, then assigns ``n_ranks`` balanced
    contiguous runs.  Dependency-free (unlike METIS) and gives compact,
    spatially-local partitions (smaller halos than RCB on irregular meshes).

    Returns
    -------
    cell_owner : np.ndarray, shape (nCells,), dtype int32
    """
    n_cells = mesh.nCells
    if n_ranks <= 1 or n_cells <= 1:
        return np.zeros(n_cells, dtype=np.int32)
    keys = hilbert_cell_keys(mesh, order)
    order_idx = np.argsort(keys, kind="stable")
    pos = np.empty(n_cells, dtype=np.int64)
    pos[order_idx] = np.arange(n_cells, dtype=np.int64)
    return (pos * n_ranks // n_cells).astype(np.int32)


# ============================================================================
# Halo computation
# ============================================================================

def compute_halo_cells(
    cell_owner: np.ndarray,
    cellsOnCell: np.ndarray,
    maxEdges: int,
    rank: int,
    halo_depth: int,
) -> set[int]:
    """Compute the set of halo cells for *rank* up to *halo_depth* rings."""
    owned = set(np.where(cell_owner == rank)[0].tolist())
    halo: set[int] = set()
    frontier = set(owned)
    for _ in range(halo_depth):
        new_frontier: set[int] = set()
        for c in frontier:
            for k in range(maxEdges):
                nbr = int(cellsOnCell[k, c])
                if nbr >= 0 and nbr not in owned and nbr not in halo:
                    halo.add(nbr)
                    new_frontier.add(nbr)
        frontier = new_frontier
    return halo


# ============================================================================
# Internal helpers
# ============================================================================

def _group_by_owner(halo_entities, owner_array):
    """Group halo entities by their owner rank, sorted by global index."""
    by_rank: dict[int, list[int]] = {}
    for g in halo_entities:
        r = int(owner_array[int(g)])
        by_rank.setdefault(r, []).append(int(g))
    for r in by_rank:
        by_rank[r].sort()
    return by_rank


def _assemble_schedule(recv_by_rank, send_by_rank, g2l, neighbor_ranks):
    """Build HaloCommSchedule from per-rank send/recv lists."""
    send_idx_all: list[int] = []
    recv_idx_all: list[int] = []
    send_counts: list[int] = []
    recv_counts: list[int] = []
    active_ranks: list[int] = []

    for r in neighbor_ranks:
        s = send_by_rank.get(r, [])
        rv = recv_by_rank.get(r, [])
        if not s and not rv:
            continue
        active_ranks.append(r)
        send_idx_all.extend(int(g2l[g]) for g in s)
        send_counts.append(len(s))
        recv_idx_all.extend(int(g2l[g]) for g in rv)
        recv_counts.append(len(rv))

    return HaloCommSchedule(
        neighbor_ranks=tuple(active_ranks),
        send_counts=tuple(send_counts),
        recv_counts=tuple(recv_counts),
        send_idx=(jnp.array(send_idx_all, dtype=jnp.int32)
                  if send_idx_all
                  else jnp.empty(0, dtype=jnp.int32)),
        recv_idx=(jnp.array(recv_idx_all, dtype=jnp.int32)
                  if recv_idx_all
                  else jnp.empty(0, dtype=jnp.int32)),
    )


def build_batched_halo_schedule(
    cell_comm: HaloCommSchedule,
    edge_comm: HaloCommSchedule,
) -> BatchedHaloSchedule:
    """Join the cell and edge comm schedules into one union-neighbor schedule.

    Per union neighbor (sorted union of the two neighbor lists), the
    per-entity send/recv index slices are re-laid-out in union-neighbor
    order; ranks absent from one entity schedule get a zero count for
    that entity.  Everything here is host-side layout math (numpy) run
    once at layout-build time — the resulting index arrays are JIT
    constants.

    The per-neighbor message sequence stays in sorted-rank order on every
    rank, exactly like the entity schedules it replaces, so the blocking
    ``sendrecv`` pairing properties of the existing exchange carry over
    unchanged.
    """

    def _per_neighbor(comm: HaloCommSchedule):
        """rank -> (send_rows, recv_rows) numpy slices for one entity."""
        send_idx = np.asarray(comm.send_idx)
        recv_idx = np.asarray(comm.recv_idx)
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        s_off = r_off = 0
        for i, r in enumerate(comm.neighbor_ranks):
            s_cnt = comm.send_counts[i]
            r_cnt = comm.recv_counts[i]
            out[r] = (
                send_idx[s_off:s_off + s_cnt],
                recv_idx[r_off:r_off + r_cnt],
            )
            s_off += s_cnt
            r_off += r_cnt
        return out

    cell_by_rank = _per_neighbor(cell_comm)
    edge_by_rank = _per_neighbor(edge_comm)
    union = sorted(set(cell_by_rank) | set(edge_by_rank))

    _empty = np.empty(0, dtype=np.int32)
    cell_send_chunks, cell_recv_chunks = [], []
    edge_send_chunks, edge_recv_chunks = [], []
    cell_send_counts, cell_recv_counts = [], []
    edge_send_counts, edge_recv_counts = [], []
    for r in union:
        c_s, c_r = cell_by_rank.get(r, (_empty, _empty))
        e_s, e_r = edge_by_rank.get(r, (_empty, _empty))
        cell_send_chunks.append(c_s)
        cell_recv_chunks.append(c_r)
        edge_send_chunks.append(e_s)
        edge_recv_chunks.append(e_r)
        cell_send_counts.append(int(len(c_s)))
        cell_recv_counts.append(int(len(c_r)))
        edge_send_counts.append(int(len(e_s)))
        edge_recv_counts.append(int(len(e_r)))

    def _cat(chunks):
        if chunks:
            flat = np.concatenate(chunks).astype(np.int32)
        else:
            flat = np.empty(0, dtype=np.int32)
        return jnp.asarray(flat)

    return BatchedHaloSchedule(
        neighbor_ranks=tuple(int(r) for r in union),
        cell_send_counts=tuple(cell_send_counts),
        cell_recv_counts=tuple(cell_recv_counts),
        edge_send_counts=tuple(edge_send_counts),
        edge_recv_counts=tuple(edge_recv_counts),
        cell_send_idx=_cat(cell_send_chunks),
        cell_recv_idx=_cat(cell_recv_chunks),
        edge_send_idx=_cat(edge_send_chunks),
        edge_recv_idx=_cat(edge_recv_chunks),
    )


# ============================================================================
# Main entry point
# ============================================================================

def partition_voronoi_mesh(
    mesh: VoronoiMesh,
    n_ranks: int,
    rank: int,
    *,
    method: str = "auto",
    halo_depth: int = 2,
    cell_owner: np.ndarray | None = None,
) -> VoronoiPartition:
    """Partition a Voronoi mesh and build decomposition for *rank*.

    Parameters
    ----------
    mesh : VoronoiMesh
        Global mesh.
    n_ranks : int
        Total number of MPI ranks / devices.
    rank : int
        This rank (0-based).
    method : str
        ``"auto"`` (default: METIS if ``pymetis`` available, else RCB),
        ``"geometric"`` (RCB), ``"metis"``, or ``"sfc"`` (Hilbert
        space-filling-curve contiguous chunks).
    halo_depth : int
        Number of halo cell layers (default 2 for del4 support).
    cell_owner : np.ndarray or None
        Pre-computed cell ownership.  If ``None``, computed via *method*.

    Returns
    -------
    VoronoiPartition
    """
    # Validate at entry on the static method value (CLAUDE.md: fail early) so an
    # unknown method raises even when ``cell_owner`` is supplied or the method is
    # otherwise unused.
    method = resolve_partition_method(method)
    if method not in ("geometric", "metis", "sfc"):
        raise ValueError(f"Unknown partitioning method: {method!r}")
    if cell_owner is None:
        if method == "geometric":
            cell_owner = partition_cells_geometric(mesh, n_ranks)
        elif method == "metis":
            cell_owner = partition_cells_metis(mesh, n_ranks)
        else:  # "sfc" (validated above)
            cell_owner = partition_cells_sfc(mesh, n_ranks)

    # Convert mesh connectivity to numpy for the setup phase.
    cellsOnCell = np.asarray(mesh.cellsOnCell)       # (maxEdges, nCells)
    cellsOnEdge = np.asarray(mesh.cellsOnEdge)       # (2, nEdges)
    cellsOnVertex = np.asarray(mesh.cellsOnVertex)   # (vDeg, nVertices)

    # ------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------
    owned_cells = np.sort(np.where(cell_owner == rank)[0]).astype(np.int64)
    halo_cells_set = compute_halo_cells(
        cell_owner, cellsOnCell, mesh.maxEdges, rank, halo_depth,
    )
    halo_cells = np.array(sorted(halo_cells_set), dtype=np.int64)
    local_cells = np.concatenate([owned_cells, halo_cells])
    local_cells_set = set(owned_cells.tolist()) | halo_cells_set

    cell_g2l = np.full(mesh.nCells, -1, dtype=np.int64)
    for i, g in enumerate(local_cells):
        cell_g2l[g] = i

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    # An edge is local if at least one of its cells is local.
    c1_all = np.asarray(cellsOnEdge[0])
    c2_all = np.asarray(cellsOnEdge[1])
    edge_local_mask = np.zeros(mesh.nEdges, dtype=bool)
    for e in range(mesh.nEdges):
        if int(c1_all[e]) in local_cells_set or int(c2_all[e]) in local_cells_set:
            edge_local_mask[e] = True
    all_local_edges = np.where(edge_local_mask)[0]

    # Edge owner = owner of the cell with the smaller global index.
    edge_owner_all = cell_owner[np.minimum(c1_all, c2_all)]

    owned_edges = np.array(
        sorted(int(e) for e in all_local_edges if edge_owner_all[e] == rank),
        dtype=np.int64,
    )
    halo_edges = np.array(
        sorted(int(e) for e in all_local_edges if edge_owner_all[e] != rank),
        dtype=np.int64,
    )
    local_edges = np.concatenate([owned_edges, halo_edges])

    edge_g2l = np.full(mesh.nEdges, -1, dtype=np.int64)
    for i, g in enumerate(local_edges):
        edge_g2l[g] = i

    # ------------------------------------------------------------------
    # Vertices
    # ------------------------------------------------------------------
    # A vertex is local if at least one of its cells is local.
    cov_np = np.asarray(cellsOnVertex)
    vertex_local_mask = np.zeros(mesh.nVertices, dtype=bool)
    for v in range(mesh.nVertices):
        for k in range(mesh.vertexDegree):
            c = int(cov_np[k, v])
            if c >= 0 and c in local_cells_set:
                vertex_local_mask[v] = True
                break
    all_local_verts = np.where(vertex_local_mask)[0]

    # Vertex owner = owner of the cell with the smallest global index
    # among the vertex's cells.
    cov_safe = np.where(cov_np >= 0, cov_np, mesh.nCells)
    min_cell_v = np.min(cov_safe, axis=0)
    vertex_owner_all = np.where(
        min_cell_v < mesh.nCells,
        cell_owner[np.minimum(min_cell_v, mesh.nCells - 1)],
        0,
    ).astype(np.int32)

    owned_vertices = np.array(
        sorted(int(v) for v in all_local_verts if vertex_owner_all[v] == rank),
        dtype=np.int64,
    )
    halo_vertices = np.array(
        sorted(int(v) for v in all_local_verts if vertex_owner_all[v] != rank),
        dtype=np.int64,
    )
    local_vertices = np.concatenate([owned_vertices, halo_vertices])

    vertex_g2l = np.full(mesh.nVertices, -1, dtype=np.int64)
    for i, g in enumerate(local_vertices):
        vertex_g2l[g] = i

    # ------------------------------------------------------------------
    # Communication schedules
    # ------------------------------------------------------------------
    # Recv side: group halo entities by their owner rank.
    cell_recv = _group_by_owner(halo_cells, cell_owner)
    edge_recv = _group_by_owner(halo_edges, edge_owner_all)
    vertex_recv = _group_by_owner(halo_vertices, vertex_owner_all)

    neighbor_ranks = sorted(cell_recv.keys())

    # For each neighbor rank R, precompute which cells are in R's local
    # domain (owned + halo).  This lets us determine which of our owned
    # edges/vertices R needs as halo.
    # Candidate ranks for the SEND schedule must be a SUPERSET of the cell-recv
    # neighbours.  A rank can share only an EDGE or VERTEX boundary with me — it
    # holds an edge/vertex I own in its halo — without its cell-halo reaching my
    # cells, so it is absent from `neighbor_ranks` (= cell-recv owners) and the
    # cell-neighbour-only `cell_to_nbr` would never mark it as needing that edge:
    # I would not send, its blocking sendrecv to me would hang.  This is the
    # np>=64 multi-node deadlock (asymmetric edge schedule; cells were fine).
    # An edge/vertex spans exactly one cell-ring beyond the cell halo, so owners
    # of cells within (halo_depth + 1) rings of my owned cells are a provably
    # sufficient superset (an edge I own has one cell of mine and one neighbour
    # cell; any rank needing it is within halo_depth of that neighbour cell,
    # i.e. within halo_depth + 1 of my cell).
    send_candidate_cells = compute_halo_cells(
        cell_owner, cellsOnCell, mesh.maxEdges, rank, halo_depth + 1,
    )
    cell_to_nbr_ranks = sorted(
        (set(neighbor_ranks)
         | {int(cell_owner[c]) for c in send_candidate_cells})
        - {rank}
    )
    cell_to_nbr: dict[int, set[int]] = {}
    for R in cell_to_nbr_ranks:
        R_owned = set(np.where(cell_owner == R)[0].tolist())
        R_halo = compute_halo_cells(
            cell_owner, cellsOnCell, mesh.maxEdges, R, halo_depth,
        )
        for c in R_owned | R_halo:
            cell_to_nbr.setdefault(c, set()).add(R)

    # --- Cell send ---
    cell_send: dict[int, list[int]] = {r: [] for r in neighbor_ranks}
    for c in owned_cells:
        c_int = int(c)
        for r in cell_to_nbr.get(c_int, ()):
            if r != rank:
                cell_send.setdefault(r, []).append(c_int)
    cell_send = {r: sorted(set(v)) for r, v in cell_send.items()}

    # --- Edge send ---
    edge_send: dict[int, list[int]] = {r: [] for r in neighbor_ranks}
    for e in owned_edges:
        c1, c2 = int(cellsOnEdge[0, e]), int(cellsOnEdge[1, e])
        nbrs: set[int] = set()
        nbrs |= cell_to_nbr.get(c1, set())
        nbrs |= cell_to_nbr.get(c2, set())
        for r in nbrs:
            if r != rank:
                edge_send.setdefault(r, []).append(int(e))
    edge_send = {r: sorted(set(v)) for r, v in edge_send.items()}

    # --- Vertex send ---
    vert_send: dict[int, list[int]] = {r: [] for r in neighbor_ranks}
    for v in owned_vertices:
        nbrs_v: set[int] = set()
        for k in range(mesh.vertexDegree):
            c = int(cov_np[k, v])
            if c >= 0:
                nbrs_v |= cell_to_nbr.get(c, set())
        for r in nbrs_v:
            if r != rank:
                vert_send.setdefault(r, []).append(int(v))
    vert_send = {r: sorted(set(v)) for r, v in vert_send.items()}

    # Assemble schedules.
    cell_comm = _assemble_schedule(
        cell_recv, cell_send, cell_g2l, neighbor_ranks)

    edge_nbrs = sorted(
        set(edge_recv.keys())
        | {k for k, v in edge_send.items() if v}
    )
    edge_comm = _assemble_schedule(
        edge_recv, edge_send, edge_g2l, edge_nbrs)

    vert_nbrs = sorted(
        set(vertex_recv.keys())
        | {k for k, v in vert_send.items() if v}
    )
    vertex_comm = _assemble_schedule(
        vertex_recv, vert_send, vertex_g2l, vert_nbrs)

    return VoronoiPartition(
        rank=rank,
        n_ranks=n_ranks,
        nCells_global=mesh.nCells,
        nEdges_global=mesh.nEdges,
        nVertices_global=mesh.nVertices,
        n_owned_cells=len(owned_cells),
        n_owned_edges=len(owned_edges),
        n_owned_vertices=len(owned_vertices),
        n_local_cells=len(local_cells),
        n_local_edges=len(local_edges),
        n_local_vertices=len(local_vertices),
        local_cells=local_cells,
        local_edges=local_edges,
        local_vertices=local_vertices,
        cell_g2l=cell_g2l,
        edge_g2l=edge_g2l,
        vertex_g2l=vertex_g2l,
        cell_comm=cell_comm,
        edge_comm=edge_comm,
        vertex_comm=vertex_comm,
    )


# ============================================================================
# Local mesh construction
# ============================================================================

def build_local_mesh(
    mesh: VoronoiMesh,
    partition: VoronoiPartition,
) -> VoronoiMesh:
    """Build a VoronoiMesh with local indices from a partition.

    The returned mesh uses local (0-based) indices in its connectivity
    arrays.  Entries that reference entities outside the local domain
    are set to -1 (TRiSK operators already mask these).

    Parameters
    ----------
    mesh : VoronoiMesh
        Global mesh.
    partition : VoronoiPartition

    Returns
    -------
    VoronoiMesh
        Local mesh with ``nCells = n_local_cells``, etc.
    """
    lc = partition.local_cells
    le = partition.local_edges
    lv = partition.local_vertices

    # Global-to-local maps as JAX arrays for indexing.
    c_g2l = jnp.array(partition.cell_g2l, dtype=jnp.int32)
    e_g2l = jnp.array(partition.edge_g2l, dtype=jnp.int32)
    v_g2l = jnp.array(partition.vertex_g2l, dtype=jnp.int32)

    def remap(conn, local_ents, g2l_map):
        """Select columns for local entities and remap values."""
        sel = conn[:, local_ents]        # (K, n_local)
        valid = sel >= 0
        safe = jnp.maximum(sel, 0)
        return jnp.where(valid, g2l_map[safe], -1).astype(jnp.int32)

    return VoronoiMesh(
        nCells=partition.n_local_cells,
        nEdges=partition.n_local_edges,
        nVertices=partition.n_local_vertices,
        maxEdges=mesh.maxEdges,
        vertexDegree=mesh.vertexDegree,
        radius=mesh.radius,
        # Cell coordinates
        latCell=mesh.latCell[lc],
        lonCell=mesh.lonCell[lc],
        xCell=mesh.xCell[lc],
        yCell=mesh.yCell[lc],
        zCell=mesh.zCell[lc],
        # Edge coordinates
        latEdge=mesh.latEdge[le],
        lonEdge=mesh.lonEdge[le],
        xEdge=mesh.xEdge[le],
        yEdge=mesh.yEdge[le],
        zEdge=mesh.zEdge[le],
        # Vertex coordinates
        latVertex=mesh.latVertex[lv],
        lonVertex=mesh.lonVertex[lv],
        xVertex=mesh.xVertex[lv],
        yVertex=mesh.yVertex[lv],
        zVertex=mesh.zVertex[lv],
        # Connectivity (column-select + value-remap)
        cellsOnEdge=remap(mesh.cellsOnEdge, le, c_g2l),
        edgesOnCell=remap(mesh.edgesOnCell, lc, e_g2l),
        verticesOnCell=remap(mesh.verticesOnCell, lc, v_g2l),
        verticesOnEdge=remap(mesh.verticesOnEdge, le, v_g2l),
        edgesOnVertex=remap(mesh.edgesOnVertex, lv, e_g2l),
        cellsOnVertex=remap(mesh.cellsOnVertex, lv, c_g2l),
        cellsOnCell=remap(mesh.cellsOnCell, lc, c_g2l),
        edgesOnEdge=remap(mesh.edgesOnEdge, le, e_g2l),
        nEdgesOnCell=mesh.nEdgesOnCell[lc],
        nEdgesOnEdge=mesh.nEdgesOnEdge[le],
        # Geometry (column-select only)
        areaCell=mesh.areaCell[lc],
        areaTriangle=mesh.areaTriangle[lv],
        dcEdge=mesh.dcEdge[le],
        dvEdge=mesh.dvEdge[le],
        angleEdge=mesh.angleEdge[le],
        # Weights (column-select only, values are floats not indices)
        weightsOnEdge=mesh.weightsOnEdge[:, le],
        kiteAreasOnVertex=mesh.kiteAreasOnVertex[:, lv],
        fEdge=mesh.fEdge[le],
        fVertex=mesh.fVertex[lv],
        edgeSignOnCell=mesh.edgeSignOnCell[:, lc],
        edgeSignOnVertex=mesh.edgeSignOnVertex[:, lv],
        meshDensity=mesh.meshDensity[lc],
        # Optional per-cell surface fields: slice like any cell field so
        # the LOCAL mesh keeps the SSO/land-fraction the global mesh
        # carries (None stays None — legacy meshes unchanged).
        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
                             else mesh.subgrid_topo_stddev[lc]),
        land_frac=(None if mesh.land_frac is None
                   else mesh.land_frac[lc]),
    )


# ============================================================================
# Convenience utilities
# ============================================================================

def scatter_to_local(
    global_field: jnp.ndarray,
    partition: VoronoiPartition,
    entity: str = "cell",
) -> jnp.ndarray:
    """Extract local portion (owned + halo) of a global field.

    Parameters
    ----------
    global_field : jax.Array
        Full-mesh field, shape ``(nEntities, ...)`` or ``(nEntities,)``.
    partition : VoronoiPartition
    entity : str
        ``"cell"``, ``"edge"``, or ``"vertex"``.

    Returns
    -------
    jax.Array, shape ``(n_local_entities, ...)``
    """
    idx = {
        "cell": partition.local_cells,
        "edge": partition.local_edges,
        "vertex": partition.local_vertices,
    }[entity]
    return global_field[idx]


# ============================================================================
# Mesh padding for even sharding
# ============================================================================

def _pad_voronoi_for_sharding(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
    """Pad cell/edge arrays so their sizes are divisible by *n_devices*.

    Adds ghost cells/edges that are inert in physics:
    - Ghost cells: ``areaCell=1`` (avoids 0/0 NaN in divergence), all
      connectivity = -1 (masked by operators), signs/weights = 0.
    - Ghost edges: ``dvEdge=0`` (zero flux contribution), ``dcEdge=1``
      (avoids 0/0 in gradient), ``cellsOnEdge=[0,0]`` (valid references
      for unmasked operators like ``gradient_edge`` and ``cell_to_edge_avg``).

    Returns *mesh* unchanged when no padding is required.
    """
    pad_cells = (-mesh.nCells) % n_devices
    pad_edges = (-mesh.nEdges) % n_devices

    if pad_cells == 0 and pad_edges == 0:
        return mesh

    # --- helpers ---
    def pad_1d(arr, n_pad, fill=0.0):
        if n_pad == 0:
            return arr
        return jnp.concatenate([arr, jnp.full((n_pad,), fill, dtype=arr.dtype)])

    def pad_2d_col(arr, n_pad, fill=0):
        """Pad along axis 1 (entity axis for (K, nEntities) layout)."""
        if n_pad == 0:
            return arr
        K = arr.shape[0]
        return jnp.concatenate(
            [arr, jnp.full((K, n_pad), fill, dtype=arr.dtype)], axis=1,
        )

    return VoronoiMesh(
        # --- dimensions ---
        nCells=mesh.nCells + pad_cells,
        nEdges=mesh.nEdges + pad_edges,
        nVertices=mesh.nVertices,
        maxEdges=mesh.maxEdges,
        vertexDegree=mesh.vertexDegree,
        radius=mesh.radius,
        # --- cell coordinates (ghost at origin) ---
        latCell=pad_1d(mesh.latCell, pad_cells, 0.0),
        lonCell=pad_1d(mesh.lonCell, pad_cells, 0.0),
        xCell=pad_1d(mesh.xCell, pad_cells, 0.0),
        yCell=pad_1d(mesh.yCell, pad_cells, 0.0),
        zCell=pad_1d(mesh.zCell, pad_cells, 0.0),
        # --- edge coordinates (ghost at origin) ---
        latEdge=pad_1d(mesh.latEdge, pad_edges, 0.0),
        lonEdge=pad_1d(mesh.lonEdge, pad_edges, 0.0),
        xEdge=pad_1d(mesh.xEdge, pad_edges, 0.0),
        yEdge=pad_1d(mesh.yEdge, pad_edges, 0.0),
        zEdge=pad_1d(mesh.zEdge, pad_edges, 0.0),
        # --- vertex coordinates (unchanged) ---
        latVertex=mesh.latVertex,
        lonVertex=mesh.lonVertex,
        xVertex=mesh.xVertex,
        yVertex=mesh.yVertex,
        zVertex=mesh.zVertex,
        # --- connectivity ---
        # cellsOnEdge: operators (gradient_edge, cell_to_edge_avg) index
        # directly without masking, so ghost edges need valid cell refs.
        cellsOnEdge=pad_2d_col(mesh.cellsOnEdge, pad_edges, fill=0),
        edgesOnCell=pad_2d_col(mesh.edgesOnCell, pad_cells, fill=-1),
        verticesOnCell=pad_2d_col(mesh.verticesOnCell, pad_cells, fill=-1),
        verticesOnEdge=pad_2d_col(mesh.verticesOnEdge, pad_edges, fill=0),
        edgesOnVertex=mesh.edgesOnVertex,  # vertex-indexed, unchanged
        cellsOnVertex=mesh.cellsOnVertex,  # vertex-indexed, unchanged
        cellsOnCell=pad_2d_col(mesh.cellsOnCell, pad_cells, fill=-1),
        edgesOnEdge=pad_2d_col(mesh.edgesOnEdge, pad_edges, fill=-1),
        nEdgesOnCell=pad_1d(mesh.nEdgesOnCell, pad_cells, fill=0),
        nEdgesOnEdge=pad_1d(mesh.nEdgesOnEdge, pad_edges, fill=0),
        # --- geometry ---
        # areaCell=1 for ghosts avoids 0/0 NaN in divergence (numerator is
        # exactly zero because all connectivity = -1 and signs = 0).
        areaCell=pad_1d(mesh.areaCell, pad_cells, fill=1.0),
        areaTriangle=mesh.areaTriangle,  # vertex-indexed, unchanged
        dcEdge=pad_1d(mesh.dcEdge, pad_edges, fill=1.0),  # avoid /0
        dvEdge=pad_1d(mesh.dvEdge, pad_edges, fill=0.0),  # zero flux
        angleEdge=pad_1d(mesh.angleEdge, pad_edges, fill=0.0),
        # --- weights and signs ---
        weightsOnEdge=pad_2d_col(mesh.weightsOnEdge, pad_edges, fill=0.0),
        kiteAreasOnVertex=mesh.kiteAreasOnVertex,  # vertex-indexed
        fEdge=pad_1d(mesh.fEdge, pad_edges, fill=0.0),
        fVertex=mesh.fVertex,  # vertex-indexed, unchanged
        edgeSignOnCell=pad_2d_col(mesh.edgeSignOnCell, pad_cells, fill=0.0),
        edgeSignOnVertex=mesh.edgeSignOnVertex,  # vertex-indexed
        meshDensity=pad_1d(mesh.meshDensity, pad_cells, fill=0.0),
        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
                             else pad_1d(mesh.subgrid_topo_stddev,
                                         pad_cells, fill=0.0)),
        land_frac=(None if mesh.land_frac is None
                   else pad_1d(mesh.land_frac, pad_cells, fill=0.0)),
    )


# ============================================================================
# Mesh reordering for JAX SPMD sharding
# ============================================================================

def resolve_sharding_partition_method(method: str) -> str:
    """Concrete ownership for the SPMD/ppermute path (``auto`` -> ``sfc``).

    Separate from :func:`resolve_partition_method`, whose ``auto`` prefers
    METIS: METIS minimizes edge CUT, while this path is bound by the number of
    sequential halo exchanges, and measured they move oppositely (subdiv-8 at
    128 devices: sfc 14 rounds, metis 19).  ONE definition, so a scorer that
    reports which ownership ran cannot drift from what the reorder does.
    """
    if method == "auto":
        return "sfc"
    return resolve_partition_method(method)


def reorder_voronoi_for_sharding(
    mesh: VoronoiMesh,
    n_devices: int,
    *,
    method: str = "auto",
    edge_order: str = "hilbert",
) -> VoronoiMesh:
    """Reorder a Voronoi mesh so that JAX NamedSharding gives spatial locality.

    Partitions cells via RCB (or METIS), then reorders cells, edges, and
    vertices so that entities owned by device 0 come first, then device 1,
    etc.  When JAX splits the reordered arrays into ``n_devices`` contiguous
    chunks along axis 0, each chunk corresponds to a spatially contiguous
    domain — minimizing cross-device communication in TRiSK stencils.

    Parameters
    ----------
    mesh : VoronoiMesh
        Original global mesh.
    n_devices : int
        Number of devices (partitions).
    method : str
        ``"auto"`` -> ``"sfc"`` on THIS path (see below), ``"geometric"``
        (RCB), ``"metis"``, or ``"sfc"`` (Hilbert space-filling-curve
        contiguous chunks).
    edge_order : str
        Order of edges and vertices INSIDE each owner block: ``"hilbert"``
        (along the cells' Hilbert curve; faster MPAS atmosphere step) or
        ``"owner"`` (generator order, the layout before the Hilbert relabel),
        or ``"block"``: generator order, but each edge belongs to the device
        BLOCK of its smaller (reordered) cell index and every block is padded
        to the same size, so a device's edge shard holds exactly the edges of
        its own cells (see :func:`_block_align_edges`).  The ocean MPAS lanes
        pass ``"block"``: with even chunks the edge shards drift off the cell
        shards and every device's halo grows by the foreign cells (s7 @ 128
        devices: 2745 -> ~2322 local cells per device).  Values are
        identical; only the layout (and the padded edge count) differs.

    Returns
    -------
    VoronoiMesh
        Mesh with reordered entities and remapped connectivity.
    """
    # ``auto`` resolves to SFC HERE, not to the global METIS-if-available
    # policy.  This is the SPMD/ppermute path, where the cost that binds at
    # high device counts is the number of collective-permute ROUNDS -- equal
    # to the max degree of the post-reorder depth-3-plus-closure comm graph,
    # since the edge coloring already reaches that lower bound.  METIS
    # minimizes its ``cellsOnCell`` EDGE CUT, which is a different objective,
    # and measured on the real halo-aware layout the two move OPPOSITELY:
    #
    #   rounds (= max_degree)      64 dev   128 dev
    #     subdiv-8  geometric        16       21
    #     subdiv-8  sfc              12       14
    #     subdiv-8  metis            13       19
    #     subdiv-9  geometric        14       18
    #     subdiv-9  sfc              11       13
    #     subdiv-9  metis            14       18
    #
    # SFC wins at every mesh and device count; at SSP-RK3's 3 halo fills per
    # step, auto->metis would cost +15 collective-permutes/step at 128 on
    # both meshes.  This became live rather than theoretical when pymetis
    # became importable in the venvs, which silently flipped auto to the
    # worst choice for this path.  The high-count launchers pin ``sfc``
    # explicitly, so their receipts are unaffected either way.
    #
    # SCOPE: only this function.  ``initialize_voronoi_mpi`` (route-A MPI)
    # and ``partition_voronoi_mesh`` keep the global policy -- their halo
    # exchange is not this ppermute schedule and no census was run for them.
    # Validate at entry (CLAUDE.md: fail early) BEFORE the single-device shortcut,
    # so an unknown method raises even when no partitioning happens.
    method = resolve_sharding_partition_method(method)
    method = resolve_partition_method(method)
    if method not in ("geometric", "metis", "sfc"):
        raise ValueError(f"Unknown partitioning method: {method!r}")
    if edge_order not in ("hilbert", "owner", "block"):
        raise ValueError(
            f"reorder_voronoi_for_sharding: edge_order must be 'hilbert', "
            f"'owner' or 'block', got {edge_order!r}")
    if n_devices <= 1:
        return mesh

    # --- Partition cells ---
    if method == "geometric":
        cell_owner = partition_cells_geometric(mesh, n_devices)
    elif method == "metis":
        cell_owner = partition_cells_metis(mesh, n_devices)
    else:  # "sfc" (validated above)
        cell_owner = partition_cells_sfc(mesh, n_devices)

    # --- Cell permutation: group by owner (primary), then order WITHIN each
    # owner by the Hilbert space-filling curve (secondary) so each contiguous
    # NamedSharding shard is spatially compact -> better cache/GPU locality and
    # smaller cross-shard stencil reach.  Ownership is unchanged; this sets only
    # the intra-shard order (the prior stable sort left it as arbitrary mesh
    # order).
    hkeys = hilbert_cell_keys(mesh)
    cell_perm = np.lexsort((hkeys, cell_owner))
    cell_inv = np.empty_like(cell_perm)
    cell_inv[cell_perm] = np.arange(len(cell_perm))

    # --- Edge owner: owner of the cell with the smaller global index ---
    cellsOnEdge_np = np.asarray(mesh.cellsOnEdge)  # (2, nEdges)
    c0 = cellsOnEdge_np[0]
    c1 = cellsOnEdge_np[1]
    edge_owner = cell_owner[np.minimum(c0, c1)]
    # Within each owner group, order edges (and vertices below) along the same
    # Hilbert curve as the cells, keyed on the min original cell -- the cell
    # that defines the owner, except an all-sentinel vertex (owner 0, keyed on
    # the last cell).  A stable sort by owner alone leaves the generator's
    # order inside the group, and the 10-neighbour edgesOnEdge gather then
    # jumps ~190k rows between consecutive edges on the s9 mesh.
    if edge_order == "hilbert":
        edge_perm = np.lexsort((hkeys[np.minimum(c0, c1)], edge_owner))
    else:
        edge_perm = np.argsort(edge_owner, kind="stable")
    edge_inv = np.empty_like(edge_perm)
    edge_inv[edge_perm] = np.arange(len(edge_perm))

    # --- Vertex owner: owner of the cell with the smallest global index ---
    cellsOnVertex_np = np.asarray(mesh.cellsOnVertex)  # (vertexDegree, nVertices)
    cov_safe = np.where(cellsOnVertex_np >= 0, cellsOnVertex_np, mesh.nCells)
    min_cell_v = np.min(cov_safe, axis=0)
    vertex_owner = np.where(
        min_cell_v < mesh.nCells,
        cell_owner[np.minimum(min_cell_v, mesh.nCells - 1)],
        0,
    ).astype(np.int32)
    if edge_order == "hilbert":
        vert_perm = np.lexsort(
            (hkeys[np.minimum(min_cell_v, mesh.nCells - 1)], vertex_owner))
    else:
        vert_perm = np.argsort(vertex_owner, kind="stable")
    vert_inv = np.empty_like(vert_perm)
    vert_inv[vert_perm] = np.arange(len(vert_perm))

    # --- Helper: remap connectivity values through an inverse permutation ---
    def remap_conn(conn, inv_perm):
        """Remap integer connectivity array: old_global → new_global."""
        arr = np.asarray(conn)
        valid = arr >= 0
        safe = np.where(valid, arr, 0)
        remapped = np.where(valid, inv_perm[safe], -1)
        return jnp.array(remapped, dtype=conn.dtype)

    # --- Helper: reorder along entity axis (last axis for (K, nEntities)) ---
    def reorder_col(arr, perm):
        """Reorder columns: arr[:, perm] for 2D, arr[perm] for 1D."""
        a = np.asarray(arr)
        if a.ndim == 1:
            return jnp.array(a[perm], dtype=arr.dtype)
        return jnp.array(a[:, perm], dtype=arr.dtype)

    def reorder_1d(arr, perm):
        a = np.asarray(arr)
        return jnp.array(a[perm], dtype=arr.dtype)

    reordered = VoronoiMesh(
        nCells=mesh.nCells,
        nEdges=mesh.nEdges,
        nVertices=mesh.nVertices,
        maxEdges=mesh.maxEdges,
        vertexDegree=mesh.vertexDegree,
        radius=mesh.radius,
        # Cell coordinates (reorder by cell_perm)
        latCell=reorder_1d(mesh.latCell, cell_perm),
        lonCell=reorder_1d(mesh.lonCell, cell_perm),
        xCell=reorder_1d(mesh.xCell, cell_perm),
        yCell=reorder_1d(mesh.yCell, cell_perm),
        zCell=reorder_1d(mesh.zCell, cell_perm),
        # Edge coordinates (reorder by edge_perm)
        latEdge=reorder_1d(mesh.latEdge, edge_perm),
        lonEdge=reorder_1d(mesh.lonEdge, edge_perm),
        xEdge=reorder_1d(mesh.xEdge, edge_perm),
        yEdge=reorder_1d(mesh.yEdge, edge_perm),
        zEdge=reorder_1d(mesh.zEdge, edge_perm),
        # Vertex coordinates (reorder by vert_perm)
        latVertex=reorder_1d(mesh.latVertex, vert_perm),
        lonVertex=reorder_1d(mesh.lonVertex, vert_perm),
        xVertex=reorder_1d(mesh.xVertex, vert_perm),
        yVertex=reorder_1d(mesh.yVertex, vert_perm),
        zVertex=reorder_1d(mesh.zVertex, vert_perm),
        # Connectivity: reorder columns AND remap values
        cellsOnEdge=remap_conn(reorder_col(mesh.cellsOnEdge, edge_perm), cell_inv),
        edgesOnCell=remap_conn(reorder_col(mesh.edgesOnCell, cell_perm), edge_inv),
        verticesOnCell=remap_conn(reorder_col(mesh.verticesOnCell, cell_perm), vert_inv),
        verticesOnEdge=remap_conn(reorder_col(mesh.verticesOnEdge, edge_perm), vert_inv),
        edgesOnVertex=remap_conn(reorder_col(mesh.edgesOnVertex, vert_perm), edge_inv),
        cellsOnVertex=remap_conn(reorder_col(mesh.cellsOnVertex, vert_perm), cell_inv),
        cellsOnCell=remap_conn(reorder_col(mesh.cellsOnCell, cell_perm), cell_inv),
        edgesOnEdge=remap_conn(reorder_col(mesh.edgesOnEdge, edge_perm), edge_inv),
        nEdgesOnCell=reorder_1d(mesh.nEdgesOnCell, cell_perm),
        nEdgesOnEdge=reorder_1d(mesh.nEdgesOnEdge, edge_perm),
        # Geometry (reorder by entity)
        areaCell=reorder_1d(mesh.areaCell, cell_perm),
        areaTriangle=reorder_1d(mesh.areaTriangle, vert_perm),
        dcEdge=reorder_1d(mesh.dcEdge, edge_perm),
        dvEdge=reorder_1d(mesh.dvEdge, edge_perm),
        angleEdge=reorder_1d(mesh.angleEdge, edge_perm),
        # Weights and signs (reorder columns by entity)
        weightsOnEdge=reorder_col(mesh.weightsOnEdge, edge_perm),
        kiteAreasOnVertex=reorder_col(mesh.kiteAreasOnVertex, vert_perm),
        fEdge=reorder_1d(mesh.fEdge, edge_perm),
        fVertex=reorder_1d(mesh.fVertex, vert_perm),
        edgeSignOnCell=reorder_col(mesh.edgeSignOnCell, cell_perm),
        edgeSignOnVertex=reorder_col(mesh.edgeSignOnVertex, vert_perm),
        meshDensity=reorder_1d(mesh.meshDensity, cell_perm),
        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
                             else reorder_1d(mesh.subgrid_topo_stddev,
                                             cell_perm)),
        land_frac=(None if mesh.land_frac is None
                   else reorder_1d(mesh.land_frac, cell_perm)),
    )

    if edge_order == "block":
        reordered = _block_align_edges(reordered, n_devices)
    # --- Pad so that nCells and nEdges are divisible by n_devices ---
    return _pad_voronoi_for_sharding(reordered, n_devices)


def _block_align_edges(mesh: VoronoiMesh, n_devices: int) -> VoronoiMesh:
    """Group edges by the device block of their smaller cell index and pad
    EACH block to the largest block's size, so device ``d``'s contiguous edge
    shard holds exactly the edges whose owning cell lies in its contiguous
    cell shard (cells are padded at the tail to ``ceil(nCells/n)`` per block).

    With even edge chunks over owner-grouped edges (``"owner"``/``"hilbert"``)
    the chunk boundaries drift from the cell blocks, so a device owns edges of
    other devices' cells and its halo must carry those cells plus their
    closure rings (measured s7 @ 128: 2745 local cells per device vs 2322
    within four rings).  Ghost edges use the inert tail recipe of
    :func:`_pad_voronoi_for_sharding`, except that they reference a cell (and
    a vertex of that cell) of their OWN block, so no device's halo is pulled
    toward another block.  Expects an unpadded, already reordered mesh.
    """
    n_cells = int(mesh.nCells)
    cp = -(-n_cells // n_devices)
    coe = np.asarray(mesh.cellsOnEdge)
    if np.any(coe < 0):
        raise ValueError("block edge order needs every edge to have two cells")
    block = coe.min(axis=0) // cp
    order = np.argsort(block, kind="stable")
    counts = np.bincount(block, minlength=n_devices)
    e_max = int(counts.max())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank_in_block = np.arange(order.size) - np.repeat(starts, counts)
    pos = np.empty(order.size, dtype=np.int64)
    pos[order] = block[order] * e_max + rank_in_block   # old edge -> new slot
    n_new = n_devices * e_max
    ghost = np.ones(n_new, dtype=bool)
    ghost[pos] = False
    ghost_block = np.flatnonzero(ghost) // e_max
    # Ghosts reference the block cell FARTHEST from the block boundary, so no
    # other device's halo holds it (a ghost whose cell sits in a neighbour's
    # halo becomes a halo edge there and rides every edge exchange).
    coc = np.asarray(mesh.cellsOnCell)
    nb_blk = np.where(coc >= 0, np.maximum(coc, 0) // cp, -1)
    cell_blk = np.arange(n_cells) // cp
    depth = np.where(((nb_blk != cell_blk) & (nb_blk >= 0)).any(axis=0), 0, -1)
    k = 0
    while (depth < 0).any():
        front = depth == k
        nxt = np.zeros(n_cells, dtype=bool)
        nb = coc[:, front].ravel()
        nxt[nb[nb >= 0]] = True
        new = nxt & (depth < 0)
        if not new.any():
            break
        depth[new] = k + 1
        k += 1
    deepest = np.full(n_devices, -1, dtype=np.int64)
    order_d = np.lexsort((-depth, cell_blk))          # per block, deepest first
    first = np.unique(cell_blk[order_d], return_index=True)
    deepest[first[0]] = order_d[first[1]]
    ghost_cell = deepest[ghost_block]
    ghost_cell = np.where(ghost_cell >= 0, ghost_cell, n_cells - 1)
    ghost_vert = np.asarray(mesh.verticesOnCell)[0, ghost_cell]

    def col(arr, fill, ghost_vals=None):
        a = np.asarray(arr)
        shape = (n_new,) if a.ndim == 1 else (a.shape[0], n_new)
        out = np.full(shape, fill, dtype=a.dtype)
        out[..., pos] = a
        if ghost_vals is not None:
            out[..., ghost] = ghost_vals
        return jnp.asarray(out)

    def remap(conn):
        a = np.asarray(conn)
        return jnp.asarray(np.where(a >= 0, pos[np.maximum(a, 0)], -1).astype(a.dtype))

    eoe = np.asarray(mesh.edgesOnEdge)
    eoe_new = np.where(eoe >= 0, pos[np.maximum(eoe, 0)], -1).astype(eoe.dtype)
    return mesh._replace(
        nEdges=n_new,
        latEdge=col(mesh.latEdge, 0.0), lonEdge=col(mesh.lonEdge, 0.0),
        xEdge=col(mesh.xEdge, 0.0), yEdge=col(mesh.yEdge, 0.0),
        zEdge=col(mesh.zEdge, 0.0),
        cellsOnEdge=col(mesh.cellsOnEdge, 0, ghost_cell),
        verticesOnEdge=col(mesh.verticesOnEdge, 0, ghost_vert),
        edgesOnEdge=col(eoe_new, -1),
        nEdgesOnEdge=col(mesh.nEdgesOnEdge, 0),
        dcEdge=col(mesh.dcEdge, 1.0), dvEdge=col(mesh.dvEdge, 0.0),
        angleEdge=col(mesh.angleEdge, 0.0),
        weightsOnEdge=col(mesh.weightsOnEdge, 0.0),
        fEdge=col(mesh.fEdge, 0.0),
        edgesOnCell=remap(mesh.edgesOnCell),
        edgesOnVertex=remap(mesh.edgesOnVertex),
    )


def complete_cell_rings(mesh: VoronoiMesh, partitions, max_rings: int = 8) -> int:
    """Largest k such that, on EVERY partition, all cells within k cellsOnCell
    hops of the owned block (owned cells included) are local cells AND have
    every incident edge and every neighbour cell local — so a cell -> edge ->
    cell stencil evaluated on the local mesh equals the owner's value on rings
    0..k.  (Neighbours are checked explicitly: the MPI partition keeps an edge
    when only ONE of its cells is local.)  Capped at ``max_rings``.
    """
    coc = np.asarray(mesh.cellsOnCell)
    eoc = np.asarray(mesh.edgesOnCell)
    neoc = np.asarray(mesh.nEdgesOnCell)
    slot = np.arange(eoc.shape[0])[:, None]
    best = max_rings
    for part in partitions:
        is_lc = np.zeros(coc.shape[1], bool)
        is_lc[np.asarray(part.local_cells)] = True
        is_le = np.zeros(int(mesh.nEdges), bool)
        is_le[np.asarray(part.local_edges)] = True
        seen = np.zeros(coc.shape[1], bool)
        ring = np.asarray(part.local_cells[:part.n_owned_cells])
        seen[ring] = True
        k = -1
        while k < best:
            e = np.where(slot < neoc[ring], eoc[:, ring], -1)
            nb = np.where(slot < neoc[ring], coc[:, ring], -1).ravel()
            nb = np.unique(nb[nb >= 0])
            if not (is_lc[ring].all() and is_le[e[e >= 0]].all()
                    and is_lc[nb].all()):
                break
            k += 1
            ring = nb[~seen[nb]]
            seen[ring] = True
        best = min(best, k)
    return max(best, 0)
