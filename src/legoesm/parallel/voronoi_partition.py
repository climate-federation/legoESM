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

import numpy as np
import jax.numpy as jnp

from legoesm.grids.voronoi import VoronoiMesh


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


# ============================================================================
# Halo computation
# ============================================================================

def _compute_halo_cells(
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


# ============================================================================
# Main entry point
# ============================================================================

def partition_voronoi_mesh(
    mesh: VoronoiMesh,
    n_ranks: int,
    rank: int,
    *,
    method: str = "geometric",
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
        ``"geometric"`` (RCB) or ``"metis"``.
    halo_depth : int
        Number of halo cell layers (default 2 for del4 support).
    cell_owner : np.ndarray or None
        Pre-computed cell ownership.  If ``None``, computed via *method*.

    Returns
    -------
    VoronoiPartition
    """
    if cell_owner is None:
        if method == "geometric":
            cell_owner = partition_cells_geometric(mesh, n_ranks)
        elif method == "metis":
            cell_owner = partition_cells_metis(mesh, n_ranks)
        else:
            raise ValueError(f"Unknown partitioning method: {method!r}")

    # Convert mesh connectivity to numpy for the setup phase.
    cellsOnCell = np.asarray(mesh.cellsOnCell)       # (maxEdges, nCells)
    cellsOnEdge = np.asarray(mesh.cellsOnEdge)       # (2, nEdges)
    cellsOnVertex = np.asarray(mesh.cellsOnVertex)   # (vDeg, nVertices)

    # ------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------
    owned_cells = np.sort(np.where(cell_owner == rank)[0]).astype(np.int64)
    halo_cells_set = _compute_halo_cells(
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
    cell_to_nbr: dict[int, set[int]] = {}
    for R in neighbor_ranks:
        R_owned = set(np.where(cell_owner == R)[0].tolist())
        R_halo = _compute_halo_cells(
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
                cell_send[r].append(c_int)
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
# Mesh reordering for JAX SPMD sharding
# ============================================================================

def reorder_voronoi_for_sharding(
    mesh: VoronoiMesh,
    n_devices: int,
    *,
    method: str = "geometric",
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
        ``"geometric"`` (RCB) or ``"metis"``.

    Returns
    -------
    VoronoiMesh
        Mesh with reordered entities and remapped connectivity.
    """
    if n_devices <= 1:
        return mesh

    # --- Partition cells ---
    if method == "geometric":
        cell_owner = partition_cells_geometric(mesh, n_devices)
    else:
        cell_owner = partition_cells_metis(mesh, n_devices)

    # --- Cell permutation: group by owner, stable sort within each group ---
    cell_perm = np.argsort(cell_owner, kind="stable")
    cell_inv = np.empty_like(cell_perm)
    cell_inv[cell_perm] = np.arange(len(cell_perm))

    # --- Edge owner: owner of the cell with the smaller global index ---
    cellsOnEdge_np = np.asarray(mesh.cellsOnEdge)  # (2, nEdges)
    c0 = cellsOnEdge_np[0]
    c1 = cellsOnEdge_np[1]
    edge_owner = cell_owner[np.minimum(c0, c1)]
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

    return VoronoiMesh(
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
    )
