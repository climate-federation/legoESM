"""MPI-based domain decomposition for MPAS Voronoi dynamics.

Each MPI rank owns a partition of the Voronoi mesh obtained via
Recursive Coordinate Bisection (RCB) or METIS graph partitioning.
Ghost cell/edge values are refreshed from their owning ranks between
each RK stage.  The DEFAULT state exchange is the batched union-neighbor
exchange
(:func:`legoesm.parallel.halo_exchange_voronoi.batched_halo_exchange`,
one message per neighbor per dtype group); the legacy per-entity path
(u edges; T+p_s packed cells; tracers packed cells) is opt-OUT via
``LEGOESM_VORONOI_BATCHED_HALO=0``.  Batched is now the default because it
is MEASURED faster at every size benchmarked — I4/I5 f32 np8 -12%/-5%
(job 8488057) and I6 f64 np8/np16 -4.6%/-3.5% (job 8488023); the historical
18x regression (435.9 ms I5/f32, 2026-06-10) was fixed by the one-scatter
pack/unpack rework.  On both paths every message routes through the AD-safe
``@jax.custom_vjp`` sendrecv wrapper (reverse-mode differentiable, never raw
``mpi4jax.sendrecv``), and pack/unpack is pure gather/scatter so the two
paths are bit-identical.

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
import os
from typing import Callable, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.core.precision import cast_pytree
from legoesm.core.state import MPASHydrostaticState
# NOTE: the MPAS dynamics live in the atmosphere component (a layer ABOVE this
# shared-substrate ``parallel`` package).  Importing them here would make
# legoesm-core depend on legoesm-atmosphere (a cycle), so — exactly as
# ``latlon_mpi`` does — the rank-local model is re-instantiated from the
# passed-in instance via ``type(model)(...)`` and its ``.tendencies`` method is
# used, never the atmosphere module.  The ``MPASPrimitiveEquationConfig``
# annotation is a lazy string (``from __future__ import annotations``), so it
# needs no import either.
from legoesm.grids.voronoi import VoronoiMesh
from legoesm.parallel.voronoi_partition import (
    BatchedHaloSchedule,
    VoronoiPartition,
    build_batched_halo_schedule,
    build_local_mesh,
    partition_cells_geometric,
    partition_cells_metis,
    partition_cells_sfc,
    partition_voronoi_mesh,
    resolve_partition_method,
    scatter_to_local,
)
from legoesm.parallel.halo_exchange_voronoi import (
    VoronoiHaloExchange,
    batched_halo_exchange,
)
from legoesm.parallel.reductions import (
    require_mpi_stack,
    batch_allreduce_mpi,
    global_min_mpi,
    global_max_mpi,
    global_sum_mpi,
)
from legoesm.timestepping.dispatch import dispatch_integrator

logger = logging.getLogger(__name__)


# ============================================================================
# Halo-exchange path switch
# ============================================================================

# Selects the MPAS state-exchange implementation in ``make_voronoi_mpi_step``
# (both the per-RK-stage exchange and the post-physics exchange).  Read ONCE
# at import time:
#
#   * unset / ``"1"`` (DEFAULT) -> batched union-neighbor exchange
#     (``batched_halo_exchange``, one message per union neighbor per dtype
#     group).  Default because MEASURED faster at every benchmarked size:
#     I4/I5 f32 np8 -12%/-5% (job 8488057), I6 f64 np8/np16 -4.6%/-3.5%
#     (job 8488023).  The historical 18x regression (435.9 vs 24.1 ms/step,
#     I5/f32, job 8457273, 2026-06-10) was fixed by the one-scatter pack/unpack
#     rework — re-measured GONE 2026-06-15.
#   * ``"0"`` -> legacy per-entity exchange (opt-OUT): per neighbor, one u edge
#     message + ONE packed T/p_s cell message + ONE packed tracer cell message,
#     via ``VoronoiHaloExchange`` -> ``_exchange_mpi`` -> the AD-safe
#     ``get_sendrecv_vjp`` wrapper.  Pack/unpack is pure gather/scatter, so the
#     two paths are bit-identical (kept as a fallback + parity reference).
#
# Trace-time semantics: the flag is consulted as a static Python bool while
# ``make_voronoi_mpi_step`` builds the exchange closure, which is then traced
# into the jitted ``_step`` — the choice is baked into the compiled step.
# Flipping the environment variable after this module is imported has NO
# effect; tests/profiling opt out either by setting the env var before first
# import or by monkeypatching ``voronoi_mpi._USE_BATCHED_HALO`` BEFORE
# calling ``make_voronoi_mpi_step``.
_USE_BATCHED_HALO = os.environ.get("LEGOESM_VORONOI_BATCHED_HALO", "1") == "1"


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
    # Union-neighbor batched schedule for the fused u/T/p_s/tracer state
    # exchange (one message per neighbor per dtype group).  Built at
    # layout time; ``None`` only for layouts constructed by hand —
    # ``make_voronoi_mpi_step`` rebuilds it on demand then.
    batched_comm: BatchedHaloSchedule | None = None


def voronoi_partition_metrics(
    layout: VoronoiPartitionLayout, *, n_dtype_groups: int = 1
) -> dict:
    """Per-rank partition-quality metrics for benchmark telemetry (roadmap #6).

    PURE (reads ownership masks + the batched-halo schedule constants); issues
    NO collectives, so it is unit-testable without an MPI stack.  Cross-rank
    aggregates (cells-per-rank min/max, total edge-cut) are added by
    :func:`reduce_partition_metrics`.  Call OUTSIDE the timed loop — the mask
    ``sum`` forces one device->host transfer.

    Keys
    ----
    n_owned_cells, n_halo_cells
        Exact owned / ghost cell counts from ``owned_mask_cells``.
    owned_halo_ratio
        ``n_halo / n_owned`` — the halo (communication) overhead of this
        rank's tile (``inf`` if a rank owns nothing).
    n_neighbor_ranks
        Number of MPI neighbours (0 without a batched schedule).
    messages_per_exchange
        Messages one batched state exchange posts (``BatchedHaloSchedule``).
    halo_recv_cells
        Ghost cells pulled per exchange = this rank's edge-cut contribution
        (a proxy for the METIS graph edge-cut).
    owned_send_cells
        Owned cells shipped to neighbours per exchange.
    """
    owned = layout.owned_mask_cells
    n_owned = int(jnp.sum(owned))
    n_halo = int(owned.shape[0]) - n_owned
    bc = layout.batched_comm
    if bc is not None:
        n_neighbors = len(bc.neighbor_ranks)
        halo_recv = int(sum(bc.cell_recv_counts))
        owned_send = int(sum(bc.cell_send_counts))
        msgs = int(bc.messages_per_exchange(n_dtype_groups))
    else:
        # No schedule (hand-built layout): fall back to the ghost-cell count.
        n_neighbors = 0
        halo_recv = n_halo
        owned_send = 0
        msgs = 0
    return {
        "n_owned_cells": n_owned,
        "n_halo_cells": n_halo,
        "owned_halo_ratio": (n_halo / n_owned) if n_owned else float("inf"),
        "n_neighbor_ranks": n_neighbors,
        "messages_per_exchange": msgs,
        "halo_recv_cells": halo_recv,
        "owned_send_cells": owned_send,
    }


def reduce_partition_metrics(local: dict) -> dict:
    """Add cross-rank aggregates to a per-rank :func:`voronoi_partition_metrics`
    dict via MPI reductions (``global_min/max/sum_mpi`` — diagnostics-only, NOT
    differentiable).  Degrades to the per-rank values on a single rank.  Call
    OUTSIDE the timed loop (each reduction is a collective).

    Adds: ``cells_per_rank_min`` / ``cells_per_rank_max`` (load balance),
    ``edge_cut_total`` (sum of ghost cells over ranks), ``max_neighbor_ranks``.
    """
    n_owned = jnp.asarray(float(local["n_owned_cells"]))
    halo_recv = jnp.asarray(float(local["halo_recv_cells"]))
    neighbors = jnp.asarray(float(local["n_neighbor_ranks"]))
    out = dict(local)
    out["cells_per_rank_min"] = int(global_min_mpi(n_owned))
    out["cells_per_rank_max"] = int(global_max_mpi(n_owned))
    out["edge_cut_total"] = int(global_sum_mpi(halo_recv))
    out["max_neighbor_ranks"] = int(global_max_mpi(neighbors))
    return out


def make_voronoi_partition_layout(
    global_mesh: VoronoiMesh,
    rank: int,
    n_ranks: int,
    *,
    method: str = "auto",
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
        batched_comm=build_batched_halo_schedule(
            partition.cell_comm, partition.edge_comm),
    )


# ============================================================================
# Scatter / gather
# ============================================================================

def scatter_state_voronoi(
    global_state: MPASHydrostaticState,
    partition: VoronoiPartition,
) -> MPASHydrostaticState:
    """Extract rank-local state from a global MPASHydrostaticState.

    u is edge-centered; T, p_s, phis and every tracer field are
    cell-centered.  ``v`` is None for MPAS (edge-normal u only) and passes
    through unchanged.

    Tracers (``q_v``/``q_c``/``q_r`` for a moist run) MUST be scattered here
    too: the per-step halo exchange only FILLS halos, it does not distribute
    the initial global field.  Without this, a moist run's local state would
    have ``tracers=None`` and the Kessler ``physics_fn`` would raise.
    """
    def _cell(f):
        return f.replace(data=scatter_to_local(f.data, partition, "cell"))

    new = global_state._replace(
        u=global_state.u.replace(
            data=scatter_to_local(global_state.u.data, partition, "edge")),
        T=_cell(global_state.T),
        p_s=_cell(global_state.p_s),
        phis=_cell(global_state.phis),
    )
    if global_state.tracers is not None:
        new = new._replace(
            tracers={k: _cell(f) for k, f in global_state.tracers.items()}
        )
    return new


def scatter_state_mpas_ocean(global_state, partition: VoronoiPartition):
    """Rank-local ``MPASOceanState`` from a global one.

    ``u`` is edge-centered; ``T, S, eta, w, H_bathy, land_mask`` are
    cell-centered.  ``rho_ref_z`` (a static reference profile, no
    horizontal axis) is passed through unsliced.  Mirrors
    :func:`scatter_state_voronoi` (the atmosphere twin) — each rank
    slices its (owned + halo) entities; no communication.
    """
    # Schema-drift tripwire (codex MINOR): a NEW MPASOceanState field
    # would pass through _replace UNSLICED silently — fail loudly so
    # the scatter/gather pair is extended deliberately.
    _expected = {"u", "T", "S", "eta", "w", "H_bathy", "land_mask",
                 "rho_ref_z"}
    if set(global_state._fields) != _expected:
        raise ValueError(
            "scatter_state_mpas_ocean: MPASOceanState schema changed "
            f"({sorted(set(global_state._fields) ^ _expected)}); extend "
            "the scatter/gather pair (and this set) deliberately."
        )

    def _cell(f):
        return f.replace(data=scatter_to_local(f.data, partition, "cell"))

    new = global_state._replace(
        u=global_state.u.replace(
            data=scatter_to_local(global_state.u.data, partition, "edge")),
        T=_cell(global_state.T),
        S=_cell(global_state.S),
        eta=_cell(global_state.eta),
        w=_cell(global_state.w),
        H_bathy=_cell(global_state.H_bathy),
        land_mask=_cell(global_state.land_mask),
    )
    return new


def gather_state_mpas_ocean(local_state, partition: VoronoiPartition):
    """Global ``MPASOceanState`` from rank-local states (owned entities
    only; Allgatherv per field).  Inverse of
    :func:`scatter_state_mpas_ocean`; ``rho_ref_z`` passes through."""
    def _g(f, entity):
        return f.replace(
            data=gather_voronoi_field(f.data, partition, entity))

    return local_state._replace(
        u=_g(local_state.u, "edge"),
        T=_g(local_state.T, "cell"),
        S=_g(local_state.S, "cell"),
        eta=_g(local_state.eta, "cell"),
        w=_g(local_state.w, "cell"),
        H_bathy=_g(local_state.H_bathy, "cell"),
        land_mask=_g(local_state.land_mask, "cell"),
    )


def exchange_state_mpas_ocean(local_state, layout: VoronoiPartitionLayout):
    """Packed full-state halo refresh for a rank-local ``MPASOceanState``.

    The OCEAN twin of the atmosphere step's ``_exchange_mpas_state``
    (see :func:`make_voronoi_mpi_step`): ONE batched union-neighbor
    message per neighbor per dtype group for the whole prognostic state
    — ``u`` (edge) plus ``T``, ``S``, ``eta``, ``w`` (cell) — via
    :func:`legoesm.parallel.halo_exchange_voronoi.batched_halo_exchange`
    (AD-safe ``custom_vjp`` sendrecv; identity when the schedule has no
    neighbors, i.e. np=1).  ``H_bathy``, ``land_mask`` and ``rho_ref_z``
    are static after :func:`scatter_state_mpas_ocean` (halo filled at
    scatter) and pass through unexchanged.

    ``w`` is included even though the step re-diagnoses it: KPP/TKE
    profile functions and diagnostics may read ``state.w`` at halo
    cells, and one extra channel in the packed message is cheaper than
    a stale-field audit on every consumer.

    NOTE (stage correctness): this refreshes the step INPUT.  The ocean
    step itself still consumes more stencil hops than ``halo_depth``
    between refreshes — see
    ``docs/performance/scaling/mpas_ocean_distributed_stage_audit.md``.
    """
    # Schema-drift tripwire (mirrors scatter/gather): a NEW MPASOceanState
    # field would silently pass through UNEXCHANGED — fail loudly so the
    # scatter/gather/exchange triple is extended deliberately.
    _expected = {"u", "T", "S", "eta", "w", "H_bathy", "land_mask",
                 "rho_ref_z"}
    if set(local_state._fields) != _expected:
        raise ValueError(
            "exchange_state_mpas_ocean: MPASOceanState schema changed "
            f"({sorted(set(local_state._fields) ^ _expected)}); extend "
            "the scatter/gather/exchange helpers (and this set) "
            "deliberately."
        )

    sched = layout.batched_comm
    if sched is None:
        # Hand-built layout (mirrors make_voronoi_mpi_step's rebuild).
        sched = build_batched_halo_schedule(
            layout.partition.cell_comm, layout.partition.edge_comm,
        )

    (u_ex,), (T_ex, S_ex, eta_ex, w_ex) = batched_halo_exchange(
        (local_state.u.data,),
        (local_state.T.data, local_state.S.data,
         local_state.eta.data, local_state.w.data),
        sched, layout.rank,
    )
    return local_state._replace(
        u=local_state.u.replace(data=u_ex),
        T=local_state.T.replace(data=T_ex),
        S=local_state.S.replace(data=S_ex),
        eta=local_state.eta.replace(data=eta_ex),
        w=local_state.w.replace(data=w_ex),
    )


class MPASOceanHaloRefresh(NamedTuple):
    """Packed IN-STEP halo refresh callables for the distributed MPAS ocean
    step (the stage-correctness lever — see
    ``docs/performance/scaling/mpas_ocean_distributed_stage_audit.md``).

    The per-step entry refresh (:func:`exchange_state_mpas_ocean`) restores
    ``halo_depth`` rings at step ENTRY only; the step's internal stencil
    chains (del4 = 4 hops, forward-backward Coriolis = 2 hops on UPDATED u,
    2-6 hops PER barotropic substep, TVD tracer advection = 2 hops on
    UPDATED T/S) consume more hops than ``halo_depth = 2`` between
    refreshes, silently corrupting owned cells at partition boundaries.
    Threading this object through ``MPASOceanModel.step(halo_refresh=...)``
    re-arms the halo at each dependency frontier.

    Each callable issues ONE batched union-neighbor message per neighbor
    per dtype group (:func:`batched_halo_exchange` — AD-safe ``custom_vjp``
    sendrecv, safe inside ``lax.scan``; identity when the schedule has no
    neighbors).  ``None`` (the serial default everywhere) keeps every
    consumer byte-identical — the refresh sites are static Python
    ``if halo_refresh is not None`` branches.

    Fields
    ------
    edges : Callable(*fields) -> tuple
        Refresh edge-indexed fields ``(n_local_edges, ...)``.
    cells : Callable(*fields) -> tuple
        Refresh cell-indexed fields ``(n_local_cells, ...)``.
    both : Callable(edge_fields, cell_fields) -> (tuple, tuple)
        Refresh a mixed group in the SAME packed message (e.g. the
        barotropic substep's ``u_bar`` edge + ``eta`` cell pair).
    vertices : Callable(*fields) -> tuple
        Refresh vertex-indexed fields ``(n_local_vertices, ...)`` via the
        per-field :class:`~legoesm.parallel.halo_exchange_voronoi.
        VoronoiHaloExchange` (``partition.vertex_comm``) — NOT the packed
        cell/edge message (the batched schedule carries no vertex lane;
        the ONLY vertex consumer is the K_zeta_bih vorticity-biharmonic
        intermediate, one field per tendency call, so a per-field
        exchange is proportionate).  Same AD-safe custom_vjp sendrecv.
    """

    edges: Callable
    cells: Callable
    both: Callable
    vertices: Callable


def make_mpas_ocean_halo_refresh(layout) -> MPASOceanHaloRefresh:
    """Build the in-step refresh callables for an armed
    :class:`VoronoiPartitionLayout` (see :class:`MPASOceanHaloRefresh`)."""
    from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange

    sched = layout.batched_comm
    if sched is None:
        # Hand-built layout (mirrors make_voronoi_mpi_step's rebuild).
        sched = build_batched_halo_schedule(
            layout.partition.cell_comm, layout.partition.edge_comm,
        )
    rank = layout.rank
    _vx = VoronoiHaloExchange(layout.partition, backend="mpi")

    def _edges(*fields):
        out_e, _ = batched_halo_exchange(fields, (), sched, rank)
        return out_e

    def _cells(*fields):
        _, out_c = batched_halo_exchange((), fields, sched, rank)
        return out_c

    def _both(edge_fields, cell_fields):
        return batched_halo_exchange(edge_fields, cell_fields, sched, rank)

    def _vertices(*fields):
        # np=1 / no-neighbor degeneracy: identity WITHOUT touching the
        # MPI stack (mirrors batched_halo_exchange's early return —
        # _exchange_mpi calls require_mpi_stack unconditionally).
        if not layout.partition.vertex_comm.neighbor_ranks:
            return tuple(fields)
        return tuple(_vx.exchange_vertex_field(f) for f in fields)

    return MPASOceanHaloRefresh(edges=_edges, cells=_cells, both=_both,
                                vertices=_vertices)


def gather_voronoi_field(
    local_field: jnp.ndarray,
    partition: VoronoiPartition,
    entity: str = "cell",
) -> jnp.ndarray:
    """Reconstruct global field from rank-local data via MPI Allgatherv.

    Only owned entities contribute; halo values are discarded.
    """
    require_mpi_stack()
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
    owned_area: jnp.ndarray,
    total_area: float,
) -> MPASHydrostaticState:
    """Global mass fixer: sum only owned cells, allreduce across ranks.

    Parameters
    ----------
    state_new, state_old : MPASHydrostaticState
    owned_area : jax.Array, shape (n_local_cells,)
        Per-cell area masked to zero on halo cells.  Pre-computed in
        :func:`make_voronoi_mpi_step` so the per-step path does not pay
        for the (state-independent) ``where`` and the wasted scalar in
        the batch allreduce.
    total_area : float
        Globally-allreduced total area.  Pre-computed once at setup; do
        not include it in the per-step batch allreduce.
    """
    # Promote the area-weighted mass sums to the fp64 budget accumulator
    # before reducing — mirrors the serial ``_fix_mass_mpas_hydro``.  Plain
    # fp32 reductions over ~10^4-10^5 cells leak ~N·eps noise into
    # ``mass_old``/``mass_new`` and leave an avoidable dry-mass drift, which
    # under MPI would also break serial parity on fp32 runs.  (With
    # JAX_ENABLE_X64 off, ``float64`` falls back to float32 — same as serial.)
    acc = jnp.float64
    area_acc = owned_area.astype(acc)
    local_mass_old = jnp.sum(state_old.p_s.data.astype(acc) * area_acc)
    local_mass_new = jnp.sum(state_new.p_s.data.astype(acc) * area_acc)
    mass_old, mass_new = batch_allreduce_mpi(
        [local_mass_old, local_mass_new], op="sum",
    )

    # fp64 correction; the storage-precision cast at the end of ``_step``
    # returns p_s to its carry dtype (mirrors serial cast_pytree(..., "storage")).
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
    method: str = "auto",
    halo_depth: int = 2,
) -> tuple[int, int, VoronoiPartitionLayout]:
    """Initialize MPI for Voronoi domain decomposition.

    Returns (rank, n_ranks, layout).

    Does NOT call ``jax.distributed.initialize()`` or modify any global
    state in ``distributed.py``.
    """
    require_mpi_stack()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Compute cell ownership on all ranks (deterministic, no communication)
    method = resolve_partition_method(method)
    if method == "geometric":
        cell_owner = partition_cells_geometric(global_mesh, n_ranks)
    elif method == "metis":
        cell_owner = partition_cells_metis(global_mesh, n_ranks)
    elif method == "sfc":
        cell_owner = partition_cells_sfc(global_mesh, n_ranks)
    else:
        raise ValueError(f"Unknown partitioning method: {method!r}")

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
    global _active_voronoi_layout
    _active_voronoi_layout = layout
    return rank, n_ranks, layout


# Module-level active layout (mirrors ``distributed._active_topology``):
# the MPAS barotropic solver needs the partition (halo exchanger +
# owned-cell mask) without threading a layout argument through the
# grid-agnostic model/config plumbing.  Accessor-only — never import the
# global directly (private-cross-import rule).
_active_voronoi_layout: "VoronoiPartitionLayout | None" = None


def get_active_voronoi_layout() -> "VoronoiPartitionLayout | None":
    """Active :class:`VoronoiPartitionLayout`, or ``None`` pre-init."""
    return _active_voronoi_layout


def reset_voronoi_layout() -> None:
    """Forget the active Voronoi layout (test isolation / multi-run)."""
    global _active_voronoi_layout
    _active_voronoi_layout = None


def get_matching_voronoi_layout(mesh) -> "VoronoiPartitionLayout | None":
    """Active layout IFF its local mesh matches ``mesh``, else ``None``.

    The size check (cell count) guards against a STALE layout from a
    previous run/test routing a global-mesh or different-mesh solve
    into the distributed branch with the wrong partition/owned mask
    (codex 2026-06-11 MAJOR).  Every distributed-MPAS consumer (the
    implicit-PCG dispatch, the explicit-substep eta-floor clamps, the
    conservation fixer) keys on THIS accessor, not the raw active
    layout.
    """
    lay = _active_voronoi_layout
    if lay is None:
        return None
    if int(lay.local_mesh.areaCell.shape[0]) != int(mesh.areaCell.shape[0]):
        return None
    return lay


# ============================================================================
# MPI-aware step function
# ============================================================================

def make_voronoi_mpi_step(
    model,
    layout: VoronoiPartitionLayout,
    sigma_coord,
    config=None,
    physics_fn=None,
    return_phys_state: bool = False,
):
    """Build an MPI-parallel step function for MPAS dynamics + AMIP physics.

    The returned function operates on rank-local state.  The tendency
    function includes halo exchange so that every RK stage sees fresh ghost
    values; mass conservation uses a global allreduce over owned cells.

    With ``return_phys_state=False`` (default) the contract is the legacy
    ``step(state, dt) -> state`` (dynamics + stateless physics).  With
    ``return_phys_state=True`` it is ``step(state, dt, forcing=None,
    phys_state=None) -> (state, phys_state_out)`` — full operator-split AMIP
    parity with the serial ``_step_jit`` (tracer advection + halo exchange,
    traced per-step ``forcing`` such as prescribed ``T_sfc``, and the
    prognostic ``phys_state`` carry threaded through ``physics_fn``).

    .. note::
       **Differentiable halo path.**  The Voronoi halo exchange
       (:mod:`legoesm.parallel.halo_exchange_voronoi`) routes every
       message through the AD-safe ``@jax.custom_vjp`` sendrecv wrapper
       (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`) — the
       same path the lat-lon / cubed-sphere halos use — so ``jax.grad``
       through this step propagates halo cotangents back to the owning
       rank (see ``tests/distributed/test_mpi_differentiability.py::
       TestVoronoiHaloMPIGrad``).  This holds on BOTH state-exchange
       paths: the DEFAULT batched union-neighbor exchange (one message
       per union (cell ∪ edge) neighbor per dtype group —
       ``n_union_nbrs`` messages vs the legacy ``n_edge_nbrs +
       2*n_cell_nbrs``) and the opt-OUT legacy per-entity exchange (u
       edges; T+p_s packed cells; tracers packed cells), selected at
       import via ``LEGOESM_VORONOI_BATCHED_HALO`` (see
       ``_USE_BATCHED_HALO``; batched now default — measured faster at
       all sizes, the 18x regression is fixed/gone).  Note the
       mass fixer's ``allreduce`` remains the only other collective; as
       everywhere, keep ``global_max/min`` out of losses.

    Parameters
    ----------
    model : MPASPrimitiveEquationModel
    layout : VoronoiPartitionLayout
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : MPASPrimitiveEquationConfig, optional
    physics_fn : callable, optional
        Operator-split physics, called as
        ``physics_fn(state, mesh, sigma_coord, phys_state=, forcing=)``
        and returning ``MPASHydrostaticTendencies`` (or a
        ``(tendencies, phys_state_out)`` tuple — the carry is ignored
        here).  Applied ONCE on the post-dynamics state and integrated
        forward over ``dt`` (``state += dt * tendency``), mirroring the
        serial ``MPASPrimitiveEquationModel._step_jit`` operator-split
        convention exactly (dynamics RK → physics → floor → fix_mass).
        The physics is evaluated on the rank-LOCAL mesh after a fresh
        halo exchange, so boundary-owned edges/cells see valid neighbour
        pressure.  ``physics_fn`` must be a column-/cell-local closure (it
        adds NO horizontal halo coupling beyond the pre-physics exchange).
        When ``return_phys_state=True`` the prognostic ``phys_state`` carry
        (TKE / convection state) AND the traced ``forcing`` are threaded
        through, so the full AMIP physics package (radiation, convection,
        turbulence) runs — not just stateless/additive physics.
        Passed in as a callable (not imported) so this core ``parallel``
        module keeps no dependency on the atmosphere component.

    Returns
    -------
    callable : ``(state, dt) -> state``
    """
    # Issue #405/#413: never silently run stateful physics without a carry.
    # The shared predicates also see a partial / __wrapped__ wrapper that
    # hides the _requires_phys_state tag.
    from legoesm.timestepping.integration import (
        physics_requires_phys_state,
        refuse_unthreaded_stateful_physics,
    )
    # ``return_phys_state=False`` is a state-only contract that DROPS the
    # carry — refuse a stateful physics_fn at build time.
    if not return_phys_state and physics_requires_phys_state(physics_fn):
        raise NotImplementedError(
            "make_voronoi_mpi_step(return_phys_state=False) does not "
            "thread the PhysicsState carry, so the configured stateful "
            "physics would silently reseed every step (issue #405/#413).  "
            "Pass return_phys_state=True and thread the returned carry, or "
            "use a diagnostic scheme."
        )
    # Column-local physics (Newtonian relaxation, warm-rain microphysics) reads
    # ONLY its own column and never touches halo cells, so the pre-physics halo
    # exchange below is wasted work.  Skipping it removes one halo exchange per
    # step (~1/4 of a moist step's exchanges: 3 RK-stage + 1 pre-physics) with
    # NO effect on results — a latency-bound strong-scaling win.  Schemes opt in
    # via a ``_column_local = True`` attribute; unknown physics defaults to
    # exchanging (safe).  Static at trace time (set on the closure at build).
    _phys_col_local = (
        bool(getattr(physics_fn, "_column_local", False))
        and os.environ.get("LEGOESM_NO_COLUMN_LOCAL_SKIP") != "1"
    )
    if config is None:
        config = model.config

    local_mesh = layout.local_mesh
    owned_mask = layout.owned_mask_cells

    # Re-instantiate the dynamics model on the rank-LOCAL mesh using the same
    # class as the passed-in instance (``type(model)``) — so this substrate
    # ``parallel`` module never imports the atmosphere component yet still drives
    # the exact MPAS tendency the global model would.  ``.tendencies(state)``
    # forwards to ``mpas_hydrostatic_tendencies(state, mesh, sigma_coord, config)``
    # with the matching defaults (physics_tendency=None, dt=0.0).
    local_model = type(model)(local_mesh, sigma_coord, config)
    # The floors' conserving tracer clamp weights columns by THIS
    # sigma_coord's dsigma while the serial path uses model.sigma_coord — a
    # direct caller passing a mismatched coordinate would silently conserve
    # against the wrong thickness (codex 2026-07-26 round 2, finding 6 nit).
    # Setup-time host check, zero hot-path cost.
    import numpy as _np
    if not _np.array_equal(_np.asarray(sigma_coord.dsigma),
                           _np.asarray(model.sigma_coord.dsigma)):
        raise ValueError(
            "make_voronoi_mpi_step: sigma_coord.dsigma differs from "
            "model.sigma_coord.dsigma — the MPI floors would conserve "
            "tracers against the wrong layer thicknesses. Pass the model's "
            "own vertical coordinate.")

    # Pre-compute the owned-area mask and the global total area once at
    # setup time.  Both are state-independent constants:
    #   * ``owned_area`` keeps the per-step ``jnp.where`` out of the JIT
    #     hot path.
    #   * ``total_area`` is allreduced once (via Python MPI, not on the
    #     XLA stream) and closed over, so the per-step batch allreduce
    #     in ``_fix_mass_mpi`` shrinks from 3 scalars (mass_old, mass_new,
    #     total_area) to 2.
    _owned_area = jnp.where(owned_mask, local_mesh.areaCell, 0.0)
    # fp64 area sum to match the fp64 mass-budget numerator in _fix_mass_mpi
    # (the correction is mass_diff/total_area — a fp32 denominator would
    # silently downcast the ratio and break serial parity on fp32 runs).
    _local_total_area = float(jnp.sum(_owned_area.astype(jnp.float64)))
    try:
        from mpi4py import MPI
        _total_area_global = MPI.COMM_WORLD.allreduce(
            _local_total_area, op=MPI.SUM,
        )
    except ImportError:
        # Single-rank fallback (no MPI): the local sum *is* the global sum.
        _total_area_global = _local_total_area

    # phis (surface geopotential) is set at initialisation and never
    # changes during the integration.  The caller is responsible for
    # ensuring ``state.phis.data`` already has its halo filled (the
    # initial-condition path does this); we simply pass it through here
    # rather than paying for a per-step ``exchange_cell_field``.

    # Static Python switch (module-level ``_USE_BATCHED_HALO``, env
    # ``LEGOESM_VORONOI_BATCHED_HALO`` read at import): both branches bind
    # the same ``_exchange_mpas_state`` name, and BOTH route every message
    # through the AD-safe ``get_sendrecv_vjp`` sendrecv wrapper.  The choice
    # is baked into the traced/jitted ``_step`` — the dynamics RK stages AND
    # the post-physics exchange below use this same closure.
    if _USE_BATCHED_HALO:
        # Union-neighbor batched schedule: ONE message per neighbor per dtype
        # group for the whole prognostic state.  A layout constant — built in
        # ``make_voronoi_partition_layout``; rebuilt here only for layouts
        # constructed by hand (e.g. direct ``VoronoiPartitionLayout(...)``).
        _batched_sched = layout.batched_comm
        if _batched_sched is None:
            _batched_sched = build_batched_halo_schedule(
                layout.partition.cell_comm, layout.partition.edge_comm,
            )
        _rank = layout.rank

        def _exchange_mpas_state(
                state: MPASHydrostaticState) -> MPASHydrostaticState:
            """Batched halo exchange for u (edge) + T/p_s/tracers (cell).

            DEFAULT path (opt-OUT via ``LEGOESM_VORONOI_BATCHED_HALO=0``):
            measured faster than the legacy per-entity path at every
            benchmarked size (I4/I5 f32 -12%/-5% job 8488057; I6 f64
            np8/np16 -4.6%/-3.5% job 8488023).  The historical 18x
            regression (job 8457273, 2026-06-10) was fixed by the
            one-scatter pack/unpack rework; see ``_USE_BATCHED_HALO`` and
            the ``halo_exchange_voronoi`` module docstring.

            All prognostic fields go out in **one** flat message per union
            (cell ∪ edge) neighbor per dtype group via
            :func:`legoesm.parallel.halo_exchange_voronoi.batched_halo_exchange`
            — ``n_union_nbrs`` messages instead of the legacy
            ``n_edge_nbrs + 2*n_cell_nbrs`` (u edges; T+p_s cells; tracer
            cells).  Pack/unpack is linear gather/scatter, so exchanged
            values are bit-identical to the per-entity path.  Tracer
            advection in the dynamics tendency reads neighbour-cell values,
            so boundary-owned cells need fresh halo tracer columns each RK
            stage exactly as they need fresh T/p_s.  ``phis`` is static
            (halo filled at setup) and ``v`` is None on MPAS (edge-normal
            ``u`` is the wind), so neither is exchanged.

            The tracer WIRE order is canonicalized to ``sorted(keys)`` —
            every rank packs/unpacks tracers in the same order even if dict
            insertion order ever diverged across ranks (same-dtype tracers
            would otherwise swap silently: identical message sizes/tags,
            wrong assignment).  Values map back by key, so sorting changes
            nothing semantically; the tracer set itself is fixed at trace
            time.
            """
            _tnames = (sorted(state.tracers)
                       if state.tracers is not None else [])
            cell_fields = (state.T.data, state.p_s.data) + tuple(
                state.tracers[k].data for k in _tnames)
            (u_ex,), cell_ex = batched_halo_exchange(
                (state.u.data,), cell_fields, _batched_sched, _rank,
            )

            new = MPASHydrostaticState(
                u=state.u.replace(data=u_ex),
                T=state.T.replace(data=cell_ex[0]),
                p_s=state.p_s.replace(data=cell_ex[1]),
                phis=state.phis,  # static after scatter; halos correct
                v=state.v,
                tracers=state.tracers,
            )
            if _tnames:
                new = new._replace(tracers={
                    k: state.tracers[k].replace(data=cell_ex[2 + i])
                    for i, k in enumerate(_tnames)
                })
            return new
    else:
        # opt-OUT (``LEGOESM_VORONOI_BATCHED_HALO=0``): legacy per-entity
        # exchange.  ``halo_ex`` dispatches through ``_exchange_mpi`` — every
        # message via the AD-safe ``get_sendrecv_vjp`` custom-VJP wrapper, same
        # as the (now default) batched path; kept as the parity reference.
        halo_ex = layout.halo_exchange

        def _exchange_mpas_state(
                state: MPASHydrostaticState) -> MPASHydrostaticState:
            """Exchange halos for u (edge), T+p_s (cell), and tracers (cell).

            ``T`` and ``p_s`` are stacked into a single
            ``(n_local_cells, nlev + 1)`` tensor so we issue **one**
            ``exchange_cell_field`` call instead of two.  Any prognostic
            ``tracers`` (q_v, q_c, ...) are stacked along a trailing axis
            and exchanged in **one** further cell collective — tracer
            advection in the dynamics tendency reads neighbour-cell values,
            so a boundary-owned cell needs fresh halo tracer columns each
            RK stage exactly as it needs fresh T/p_s.  ``phis`` is static
            (halo filled at setup) and ``v`` is None on MPAS (edge-normal
            ``u`` is the wind), so neither is exchanged.
            """
            u_ex = halo_ex.exchange_edge_field(state.u.data)
            # Pack T (nlev) + p_s (1) along the trailing axis: one exchange.
            Tp_packed = jnp.concatenate(
                [state.T.data, state.p_s.data[..., None]], axis=-1,
            )
            Tp_ex = halo_ex.exchange_cell_field(Tp_packed)
            nlev = state.T.data.shape[-1]
            T_ex = Tp_ex[..., :nlev]
            ps_ex = Tp_ex[..., nlev]

            new = MPASHydrostaticState(
                u=state.u.replace(data=u_ex),
                T=state.T.replace(data=T_ex),
                p_s=state.p_s.replace(data=ps_ex),
                phis=state.phis,  # static after scatter; halos correct
                v=state.v,
                tracers=state.tracers,
            )
            if state.tracers is not None and len(state.tracers) > 0:
                # Canonical sorted(keys) WIRE order — same cross-rank
                # consistency guard as the batched path (values map back
                # by key, so sorting changes nothing semantically).
                _tnames = sorted(state.tracers)
                # (n_local_cells, nlev, n_tracers) — one collective for all.
                q_packed = jnp.stack(
                    [state.tracers[k].data for k in _tnames], axis=-1,
                )
                q_ex = halo_ex.exchange_cell_field(q_packed)
                new = new._replace(tracers={
                    k: state.tracers[k].replace(data=q_ex[..., i])
                    for i, k in enumerate(_tnames)
                })
            return new

    def _mpi_tendency_fn(state: MPASHydrostaticState) -> MPASHydrostaticState:
        """Exchange halos then compute tendencies on local mesh.

        Returns a tendency-shaped ``MPASHydrostaticState`` (carrying tracer
        ADVECTION tendencies under the same tracer keys) so the pytree RK
        integrator advances moisture mass-consistently with u/T/p_s —
        mirroring the serial ``_step_jit.dyn_tendency_fn``.
        """
        state_ex = _exchange_mpas_state(state)
        tend = local_model.tendencies(state_ex)
        new = MPASHydrostaticState(
            u=state.u.replace(data=tend.du_dt.data),
            T=state.T.replace(data=tend.dT_dt.data),
            p_s=state.p_s.replace(data=tend.dp_s_dt.data),
            phis=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
            v=state.v,
            tracers=state.tracers,
        )
        if state.tracers is not None and tend.tracer_tendencies is not None:
            new = new._replace(tracers={
                k: state.tracers[k].replace(
                    data=tend.tracer_tendencies[k].data)
                for k in state.tracers
            })
        return new

    @jax.jit
    def _step(state: MPASHydrostaticState, dt: float, forcing=None,
              phys_state=None):
        """Advance one MPI step; returns ``(state_new, phys_state_out)``.

        Full operator-split parity with the serial
        ``MPASPrimitiveEquationModel._step_jit``: dynamics RK (incl. tracer
        advection, with a fresh halo exchange each stage) → operator-split
        physics evaluated ONCE on the post-dynamics state with the traced
        per-step ``forcing`` (e.g. prescribed ``T_sfc``) and the prognostic
        ``phys_state`` carry threaded through → floors → owned-cell global
        mass fixer.  ``forcing`` / ``phys_state`` are jit arguments (NOT
        static) so new values each step do not retrace.
        """
        # Precision parity with the serial ``_step_jit``: integrate in compute
        # precision, store the result in storage precision.  Without this a
        # mixed-precision run's owned cells could diverge from serial beyond
        # the reduction-order residual, and the fp64 mass-fixer correction
        # would otherwise leave p_s in fp64 (carry-dtype instability).
        state = cast_pytree(state, None, "compute")
        state_new = dispatch_integrator(
            state, _mpi_tendency_fn, dt, config.time_integrator,
        )

        # Operator-split physics: evaluate ONCE on the post-dynamics state and
        # apply forward over dt (state += dt * tendency).  Mirrors the serial
        # ``_step_jit`` convention.  A fresh halo exchange precedes the call so
        # boundary-owned cells/edges see valid neighbour values (e.g. edge
        # sigma from ``cellsOnEdge``); column-local AMIP physics is unaffected
        # by it but the exchange keeps the boundary consistent.
        phys_state_out = phys_state
        # Surface-flux diagnostic (8-slot contract: sw_net_sfc, lw_net_sfc,
        # precip, then the CMOR TOA/turbulent-flux extras) the coupler reads
        # from ``_carry_aux`` for the daily ocean/land forcing.  Mirrors
        # the serial ``primitive_eq_mpas._step_jit``: extract it from the physics
        # tendency and publish it (rank-local, matching the rank-local state the
        # MPI-voronoi coupler already sees).  Without this the coupled MPI-voronoi
        # ocean/land was forced with zero surface flux (the serial path exported
        # it, the MPI bypass did not).
        sfc_diag = (None, None, None)
        if physics_fn is not None:
            # Column-local physics needs no neighbor cells -> skip the exchange.
            state_phys_in = (
                state_new if _phys_col_local
                else _exchange_mpas_state(state_new))
            _pr = physics_fn(
                state_phys_in, local_mesh, sigma_coord,
                phys_state=phys_state, forcing=forcing,
            )
            # NB ``type(... ) is tuple`` (not isinstance): the tendencies
            # object is itself a NamedTuple, so isinstance(_pr, tuple) is
            # always True — mirror the serial ``_step_jit`` guard exactly.
            if type(_pr) is tuple:
                _pt, phys_state_out = _pr[0], _pr[1]
            else:
                _pt = _pr
            _sw_sfc = getattr(_pt, "sw_net_sfc", None)
            _lw_sfc = getattr(_pt, "lw_net_sfc", None)
            _pr_sfc = getattr(_pt, "precip", None)
            # CMOR TOA + surface turbulent-flux extras — mirror the serial
            # producer's 8-slot contract (primitive_eq_mpas.step) EXACTLY so the
            # one-rank MPI-voronoi coupled lane exports rlut/rsut/rsdt/hfss/hfls
            # too. Slot order: (sw_net, lw_net, precip, lw_up_toa, sw_up_toa,
            # sw_down_toa, shflx, lhflx) — the consumer (model_driver
            # _feed_mpas_cmip_accumulators) reads slots 3-7 by this order.
            _extras = tuple(getattr(_pt, _k, None) for _k in (
                "lw_up_toa", "sw_up_toa", "sw_down_toa",
                "shflx_sfc", "lhflx_sfc"))
            if (_sw_sfc is not None or _lw_sfc is not None
                    or _pr_sfc is not None
                    or any(_e is not None for _e in _extras)):
                sfc_diag = (_sw_sfc, _lw_sfc, _pr_sfc) + _extras
            state_new = MPASHydrostaticState(
                u=state_phys_in.u.replace(
                    data=state_phys_in.u.data + dt * _pt.du_dt.data),
                T=state_phys_in.T.replace(
                    data=state_phys_in.T.data + dt * _pt.dT_dt.data),
                p_s=state_phys_in.p_s.replace(
                    data=state_phys_in.p_s.data + dt * _pt.dp_s_dt.data),
                phis=state_new.phis,
                v=state_new.v,
                tracers=state_phys_in.tracers,
            )
            if (state_new.tracers is not None
                    and _pt.tracer_tendencies is not None):
                state_new = state_new._replace(tracers={
                    k: (state_new.tracers[k].replace(
                            data=state_new.tracers[k].data
                            + dt * _pt.tracer_tendencies[k].data)
                        if k in _pt.tracer_tendencies
                        else state_new.tracers[k])
                    for k in state_new.tracers
                })

        # Global mass fixer BEFORE the floors, mirroring the serial ordering
        # (codex 2026-07-26 round 2, finding 5): the fixer touches ONLY p_s
        # and the floors touch ONLY T/tracers, so the stages commute — but
        # diagnosed column water is sum(q*p_s*dsigma)/g, so correcting p_s
        # AFTER the conserving tracer clamp shifted water by c*B/g.  With p_s
        # finalised first, the clamp conserves against the final p_s exactly.
        # (Owned cells + allreduce — correct under the cell partition, unlike
        # the model's internal fixer which would double-count halo cells.)
        if config.fix_mass:
            state_new = _fix_mass_mpi(
                state_new, state, _owned_area, _total_area_global,
            )

        # Floors: temperature and tracer non-negativity (advection is not
        # positive-definite; clamp before tracers feed saturation).
        if config.T_min > 0:
            T_clipped = jnp.maximum(state_new.T.data, config.T_min)
            state_new = state_new._replace(
                T=state_new.T.replace(data=T_clipped))
        if state_new.tracers is not None:
            # Mirror the serial floors EXACTLY (codex 2026-07-26 review of
            # fc7e7dce8: this block previously hard-coded the plain clamp, so
            # ``conservative_tracer_clamp`` was SILENTLY INERT under MPI — the
            # repo's recurring dropped-flag defect class).  The borrow is
            # column-local, so it needs no halo/allreduce and is identical on
            # owned and halo cells.
            if getattr(config, "conservative_tracer_clamp", False):
                from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
                    _borrow_eligible,
                )
                from legoesm.core.conservation import (
                    conservative_positive_clip,
                )
                # PER-MASS tracers borrowed (mixing ratios + N_i/N_s/N_g) —
                # mirrors the serial floors exactly (see primitive_eq_mpas:
                # the naive clip INVENTED per-mass number every step, x2.2/day
                # measured -> N_i overflow NaN; per-volume N_c/N_r keep the
                # plain clip pending density-aware repair).
                # TRUE layer-mass dp weight (post-mass-fix p_s): identical
                # rescale on pure sigma (per-column p_s cancels), correct on
                # hybrid where dsigma is not the layer mass (codex
                # 2026-07-28 round 2).  Non-positive dp zero-weighted.
                _ph = sigma_coord.pressure_at_half(state_new.p_s.data)
                _dp = jnp.maximum(_ph[..., 1:] - _ph[..., :-1], 0.0)
                state_new = state_new._replace(tracers={
                    k: f.replace(data=(
                        conservative_positive_clip(f.data, _dp)[0]
                        if _borrow_eligible(k)
                        else jnp.maximum(f.data, 0.0)))
                    for k, f in state_new.tracers.items()
                })
            else:
                state_new = state_new._replace(tracers={
                    k: f.replace(data=jnp.maximum(f.data, 0.0))
                    for k, f in state_new.tracers.items()
                })

        # (mass fixer moved above the floors — codex round-2 finding 5.)

        return cast_pytree(state_new, None, "storage"), phys_state_out, sfc_diag

    logger.info(
        "Voronoi MPI step ready: rank=%d/%d, %d owned cells, %d local cells",
        layout.rank, layout.n_ranks,
        layout.partition.n_owned_cells,
        layout.partition.n_local_cells,
    )
    if return_phys_state:
        # Full AMIP contract: ``(state, dt, forcing=None, phys_state=None)``
        # -> ``(state, phys_state_out)`` so the driver can thread the
        # operator-split physics carry (TKE / convection state) across steps.
        # This contract CAN thread the carry, but a caller that forgets to
        # pass ``phys_state`` for a stateful physics_fn would still silently
        # reseed every step.  Guard at CALL time (mirroring the dycore step
        # APIs) via a thin Python wrapper around the jitted ``_step`` — the
        # check is Python-level (static physics_fn, ``phys_state is None``),
        # so it adds no retrace and runs even under an outer trace.
        def _step_carry(state, dt, forcing=None, phys_state=None):
            refuse_unthreaded_stateful_physics(
                physics_fn, phys_state,
                where="make_voronoi_mpi_step(return_phys_state=True)")
            _state_out, _phys_out, _sfc = _step(state, dt, forcing, phys_state)
            # Publish the surface-flux diagnostic on the model so ``_run_mpas``
            # stashes it into ``_carry_aux`` for the coupler — the coupled
            # MPI-voronoi path otherwise forced the ocean/land with zero surface
            # flux.  Guard against stashing a Tracer (an outer jit/scan/grad would
            # leak it into the next trace, gh-417) and merge ELEMENT-WISE
            # keep-last-non-None — both mirror the serial
            # ``primitive_eq_mpas.step()`` (a held-radiation step refreshes precip
            # while retaining the last radiation sw/lw).  Python-level side stash
            # (post-jit), same as the serial ``self._sfc_diag``.
            if not any(isinstance(leaf, jax.core.Tracer)
                       for leaf in jax.tree_util.tree_leaves(_sfc)):
                # Pad the shorter of (prev, new) so a length mismatch (a
                # held-radiation step that returns the 3-slot default vs an
                # 8-slot published prev, or a mid-session contract growth) merges
                # slot-wise instead of truncating via zip — mirrors the serial
                # primitive_eq_mpas merge.
                _prev = getattr(model, "_sfc_diag", None) or ()
                _n = max(len(_sfc), len(_prev))
                _prev = _prev + (None,) * (_n - len(_prev))
                _sfc = _sfc + (None,) * (_n - len(_sfc))
                model._sfc_diag = tuple(
                    new if new is not None else old
                    for new, old in zip(_sfc, _prev))
            return _state_out, _phys_out
        return _step_carry

    # Backward-compatible contract for the dynamics / stateless-physics
    # callers (test_voronoi_mpi, the scaling benches): ``step(state, dt)``
    # -> ``state``.  ``forcing`` / ``phys_state`` default to None and the
    # carry is dropped.  ``_step`` is already jit; this thin wrapper only
    # selects the first element.
    def _step_state_only(state, dt, forcing=None, phys_state=None):
        return _step(state, dt, forcing, phys_state)[0]

    return _step_state_only
