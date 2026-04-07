"""MPI-based domain decomposition for MPAS Voronoi dynamics.

Each MPI rank owns a partition of the Voronoi mesh obtained via
Recursive Coordinate Bisection (RCB) or METIS graph partitioning.
Halo exchange uses ``mpi4jax.sendrecv`` to update ghost cell/edge
values from their owning ranks between each RK stage.

Design
------
1. Global mesh is partitioned via :func:`partition_voronoi_mesh`.
2. Each rank builds a local sub-mesh via :func:`build_local_mesh`.
3. The tendency function is wrapped to exchange halos before computing
   tendencies, so every RK stage sees fresh ghost values.
4. Mass fixer uses :func:`global_sum_mpi` over owned cells only.

This module does NOT touch the global ``_active_topology`` or
``_active_layout`` singletons in ``distributed.py``.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState
from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    mpas_hydrostatic_tendencies,
)
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.parallel.voronoi_partition import (
    VoronoiPartition,
    partition_voronoi_mesh,
    build_local_mesh,
    scatter_to_local,
)
from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange
from legoesm.parallel.reductions import (
    _require_mpi_stack,
    global_sum_mpi,
    batch_allreduce_mpi,
)
from legoesm.timestepping.dispatch import dispatch_integrator

logger = logging.getLogger(__name__)


# ============================================================================
# Layout
# ============================================================================

class VoronoiPartitionLayout(NamedTuple):
    """MPI layout for domain-decomposed Voronoi mesh.

    Wraps a :class:`VoronoiPartition` with the halo exchange manager
    and ownership masks needed for MPI-safe stepping and conservation.
    """
    rank: int
    n_ranks: int
    partition: VoronoiPartition
    halo_exchange: VoronoiHaloExchange
    local_mesh: VoronoiMesh
    owned_mask_cells: jnp.ndarray   # (n_local_cells,) bool
    owned_mask_edges: jnp.ndarray   # (n_local_edges,) bool


def make_voronoi_partition_layout(
    global_mesh: VoronoiMesh,
    rank: int,
    n_ranks: int,
    *,
    method: str = "geometric",
    halo_depth: int = 2,
    cell_owner: np.ndarray | None = None,
) -> VoronoiPartitionLayout:
    """Build a Voronoi MPI layout for rank ``rank``.

    Parameters
    ----------
    global_mesh : VoronoiMesh
        Full global mesh.
    rank, n_ranks : int
        MPI rank and total ranks.
    method : str
        Partitioning method (``"geometric"`` or ``"metis"``).
    halo_depth : int
        Number of halo cell rings.
    cell_owner : np.ndarray, optional
        Pre-computed cell ownership array.  If None, computed via *method*.

    Returns
    -------
    VoronoiPartitionLayout
    """
    partition = partition_voronoi_mesh(
        global_mesh, n_ranks, rank,
        method=method, halo_depth=halo_depth,
        cell_owner=cell_owner,
    )

    local_mesh = build_local_mesh(global_mesh, partition)
    halo_ex = VoronoiHaloExchange(partition, backend="mpi")

    # Ownership masks: True for owned, False for halo
    owned_mask_cells = jnp.arange(partition.n_local_cells) < partition.n_owned_cells
    owned_mask_edges = jnp.arange(partition.n_local_edges) < partition.n_owned_edges

    return VoronoiPartitionLayout(
        rank=rank,
        n_ranks=n_ranks,
        partition=partition,
        halo_exchange=halo_ex,
        local_mesh=local_mesh,
        owned_mask_cells=owned_mask_cells,
        owned_mask_edges=owned_mask_edges,
    )


# ============================================================================
# Scatter / gather
# ============================================================================

def scatter_voronoi_field(
    global_field: jnp.ndarray,
    partition: VoronoiPartition,
    entity: str = "cell",
) -> jnp.ndarray:
    """Extract rank-local (owned + halo) portion of a global field."""
    return scatter_to_local(global_field, partition, entity)


def scatter_state_voronoi(
    global_state: MPASHydrostaticState,
    partition: VoronoiPartition,
) -> MPASHydrostaticState:
    """Extract rank-local state from a global MPASHydrostaticState.

    u is edge-centered; T, p_s, phis are cell-centered.
    """
    return MPASHydrostaticState(
        u=global_state.u.replace(
            data=scatter_to_local(global_state.u.data, partition, "edge")),
        T=global_state.T.replace(
            data=scatter_to_local(global_state.T.data, partition, "cell")),
        p_s=global_state.p_s.replace(
            data=scatter_to_local(global_state.p_s.data, partition, "cell")),
        phis=global_state.phis.replace(
            data=scatter_to_local(global_state.phis.data, partition, "cell")),
    )


def gather_voronoi_field(
    local_field: jnp.ndarray,
    partition: VoronoiPartition,
    entity: str = "cell",
) -> jnp.ndarray:
    """Reconstruct global field from rank-local data via MPI Allgatherv.

    Only owned entities contribute; halo values are discarded.
    """
    _require_mpi_stack()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    if entity == "cell":
        n_owned = partition.n_owned_cells
        n_global = partition.nCells_global
        local_ents = partition.local_cells
    elif entity == "edge":
        n_owned = partition.n_owned_edges
        n_global = partition.nEdges_global
        local_ents = partition.local_edges
    elif entity == "vertex":
        n_owned = partition.n_owned_vertices
        n_global = partition.nVertices_global
        local_ents = partition.local_vertices
    else:
        raise ValueError(f"Unknown entity: {entity!r}")

    # Send only owned values
    owned_data = np.asarray(local_field[:n_owned])
    owned_indices = np.asarray(local_ents[:n_owned])

    # Gather owned data and indices from all ranks
    all_data = comm.allgather(owned_data)
    all_indices = comm.allgather(owned_indices)

    # Reconstruct global field
    trailing = local_field.shape[1:] if local_field.ndim > 1 else ()
    global_field = np.zeros((n_global,) + trailing, dtype=np.asarray(local_field).dtype)
    for data_chunk, idx_chunk in zip(all_data, all_indices):
        global_field[idx_chunk] = data_chunk

    return jnp.array(global_field)


def gather_state_voronoi(
    local_state: MPASHydrostaticState,
    partition: VoronoiPartition,
) -> MPASHydrostaticState:
    """Reconstruct global MPASHydrostaticState from rank-local data."""
    return MPASHydrostaticState(
        u=local_state.u.replace(
            data=gather_voronoi_field(local_state.u.data, partition, "edge")),
        T=local_state.T.replace(
            data=gather_voronoi_field(local_state.T.data, partition, "cell")),
        p_s=local_state.p_s.replace(
            data=gather_voronoi_field(local_state.p_s.data, partition, "cell")),
        phis=local_state.phis.replace(
            data=gather_voronoi_field(local_state.phis.data, partition, "cell")),
    )


# ============================================================================
# MPI-safe mass fixer
# ============================================================================

def _fix_mass_mpi(
    state_new: MPASHydrostaticState,
    state_old: MPASHydrostaticState,
    mesh: VoronoiMesh,
    owned_mask: jnp.ndarray,
) -> MPASHydrostaticState:
    """Global mass fixer: sum only owned cells, allreduce across ranks.

    Parameters
    ----------
    state_new, state_old : MPASHydrostaticState
    mesh : VoronoiMesh (local)
    owned_mask : jax.Array, shape (n_local_cells,), bool
        True for owned cells, False for halos.
    """
    area = mesh.areaCell
    owned_area = jnp.where(owned_mask, area, 0.0)

    local_mass_old = jnp.sum(state_old.p_s.data * owned_area)
    local_mass_new = jnp.sum(state_new.p_s.data * owned_area)
    local_total_area = jnp.sum(owned_area)
    mass_old, mass_new, total_area = batch_allreduce_mpi(
        [local_mass_old, local_mass_new, local_total_area], op="sum",
    )

    correction = (mass_old - mass_new) / total_area
    p_s_fixed = state_new.p_s.replace(
        data=state_new.p_s.data + correction)
    return state_new._replace(p_s=p_s_fixed)


# ============================================================================
# Initialization
# ============================================================================

def initialize_voronoi_mpi(
    global_mesh: VoronoiMesh,
    *,
    method: str = "geometric",
    halo_depth: int = 2,
) -> tuple[int, int, VoronoiPartitionLayout]:
    """Initialize MPI for Voronoi domain decomposition.

    Returns (rank, n_ranks, layout).

    Does NOT call ``jax.distributed.initialize()`` or modify any global
    state in ``distributed.py``.
    """
    _require_mpi_stack()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Compute cell ownership on all ranks (deterministic, no communication)
    from legoesm.parallel.voronoi_partition import partition_cells_geometric
    if method == "geometric":
        cell_owner = partition_cells_geometric(global_mesh, n_ranks)
    else:
        from legoesm.parallel.voronoi_partition import partition_cells_metis
        cell_owner = partition_cells_metis(global_mesh, n_ranks)

    layout = make_voronoi_partition_layout(
        global_mesh, rank, n_ranks,
        method=method, halo_depth=halo_depth,
        cell_owner=cell_owner,
    )

    logger.info(
        "Voronoi MPI init: rank=%d/%d, owned_cells=%d, halo_cells=%d, "
        "owned_edges=%d, halo_edges=%d",
        rank, n_ranks,
        layout.partition.n_owned_cells,
        layout.partition.n_local_cells - layout.partition.n_owned_cells,
        layout.partition.n_owned_edges,
        layout.partition.n_local_edges - layout.partition.n_owned_edges,
    )
    return rank, n_ranks, layout


# ============================================================================
# MPI-aware step function
# ============================================================================

def make_voronoi_mpi_step(
    model,
    layout: VoronoiPartitionLayout,
    sigma_coord,
    config: MPASPrimitiveEquationConfig | None = None,
):
    """Build an MPI-parallel step function for MPAS dynamics.

    The returned function ``step(state, dt) -> state`` operates on
    rank-local state.  The tendency function includes halo exchange
    so that every RK stage sees fresh ghost values.  Mass conservation
    uses a global allreduce over owned cells.

    Parameters
    ----------
    model : MPASPrimitiveEquationModel
    layout : VoronoiPartitionLayout
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig, optional

    Returns
    -------
    callable : ``(state, dt) -> state``
    """
    if config is None:
        config = model.config

    local_mesh = layout.local_mesh
    halo_ex = layout.halo_exchange
    owned_mask = layout.owned_mask_cells

    def _exchange_mpas_state(state: MPASHydrostaticState) -> MPASHydrostaticState:
        """Exchange halos for all prognostic MPAS fields."""
        u_ex = halo_ex.exchange_edge_field(state.u.data)
        T_ex = halo_ex.exchange_cell_field(state.T.data)
        ps_ex = halo_ex.exchange_cell_field(state.p_s.data)
        # phis is static — exchange once is enough, but for simplicity
        # we exchange every time (cost is negligible)
        phis_ex = halo_ex.exchange_cell_field(state.phis.data)

        return MPASHydrostaticState(
            u=state.u.replace(data=u_ex),
            T=state.T.replace(data=T_ex),
            p_s=state.p_s.replace(data=ps_ex),
            phis=state.phis.replace(data=phis_ex),
        )

    def _mpi_tendency_fn(state: MPASHydrostaticState) -> MPASHydrostaticState:
        """Exchange halos then compute tendencies on local mesh."""
        state_ex = _exchange_mpas_state(state)
        tend = mpas_hydrostatic_tendencies(
            state_ex, local_mesh, sigma_coord, config,
        )
        return MPASHydrostaticState(
            u=state.u.replace(data=tend.du_dt.data),
            T=state.T.replace(data=tend.dT_dt.data),
            p_s=state.p_s.replace(data=tend.dp_s_dt.data),
            phis=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        )

    @jax.jit
    def _step(state: MPASHydrostaticState, dt: float) -> MPASHydrostaticState:
        state_new = dispatch_integrator(
            state, _mpi_tendency_fn, dt, config.time_integrator,
        )

        # Temperature floor
        if config.T_min > 0:
            T_clipped = jnp.maximum(state_new.T.data, config.T_min)
            state_new = state_new._replace(
                T=state_new.T.replace(data=T_clipped))

        # Global mass fixer
        if config.fix_mass:
            state_new = _fix_mass_mpi(state_new, state, local_mesh, owned_mask)

        return state_new

    logger.info(
        "Voronoi MPI step ready: rank=%d/%d, %d owned cells, %d local cells",
        layout.rank, layout.n_ranks,
        layout.partition.n_owned_cells,
        layout.partition.n_local_cells,
    )
    return _step
