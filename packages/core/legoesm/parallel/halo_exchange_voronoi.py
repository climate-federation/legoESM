"""Halo exchange for domain-decomposed Voronoi meshes.

Provides MPI-based halo exchange that communicates field values for
halo (ghost) entities from their owning ranks.  Also provides a
simulated local exchange for testing without MPI.

All MPI messages route through the AD-safe ``@jax.custom_vjp`` sendrecv
wrapper (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`) so the
Voronoi halo exchange is reverse-mode differentiable, like the lat-lon
and cubed-sphere halos.

Usage
-----
::

    # MPI mode (under mpirun):
    halo = VoronoiHaloExchange(partition, backend="mpi")
    h_local = halo.exchange_cell_field(h_local)
    u_local = halo.exchange_edge_field(u_local)

    # Batched state exchange (one message per union neighbor per dtype
    # group, instead of one per neighbor per entity exchange):
    sched = build_batched_halo_schedule(partition.cell_comm,
                                        partition.edge_comm)
    (u,), (T, p_s) = batched_halo_exchange((u,), (T, p_s), sched, rank)

    # Testing mode (single process, multiple partitions):
    local_fields = exchange_local_simulated(
        partitions, local_fields, entity="cell")

.. note::
   **The batched path is now the DEFAULT** (opt-OUT via
   ``LEGOESM_VORONOI_BATCHED_HALO=0``; ``voronoi_mpi._USE_BATCHED_HALO``).
   The historical ~18x CPU regression (435.9 vs 24.1 ms/step, I5 np8 f32,
   job 8457273, 2026-06-10 — per-field/per-neighbour scatter copies) was
   FIXED by the one-scatter pack/unpack (one whole-field scatter; see
   :func:`unpack_batched_recvs`).  Re-measured 2026-06-15: batched is
   FASTER than legacy at every benchmarked size — I4/I5 f32 np8 -12%/-5%
   (job 8488057), I6 f64 np8/np16 -4.6%/-3.5% (job 8488023).  Pack/unpack
   is pure gather/reshape/concat/split/scatter, so both paths are
   bit-identical (the legacy path is kept as the parity reference + fallback).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.parallel.voronoi_partition import (
    BatchedHaloSchedule,
    HaloCommSchedule,
    VoronoiPartition,
    build_batched_halo_schedule,  # noqa: F401  (re-export: schedule + exchange live together)
)


# ============================================================================
# Message-count instrumentation
# ============================================================================

# Incremented once per MPI sendrecv POST.  The increment happens when the
# exchange function body runs — i.e. at TRACE time under ``jax.jit`` (once
# per compilation, not per execution) and per call when running eagerly.
# That is exactly the "messages per exchange" schedule quantity the
# batching work optimizes; use it eagerly (or per fresh trace) when
# asserting message counts in tests/benchmarks.
_halo_message_count = 0


def reset_halo_message_count() -> None:
    """Reset the traced-sendrecv message counter to zero."""
    global _halo_message_count
    _halo_message_count = 0


def get_halo_message_count() -> int:
    """Number of MPI sendrecv posts traced since the last reset."""
    return _halo_message_count


def _count_message() -> None:
    global _halo_message_count
    _halo_message_count += 1


class VoronoiHaloExchange:
    """Halo exchange manager for one rank of a partitioned Voronoi mesh.

    Parameters
    ----------
    partition : VoronoiPartition
        Partition descriptor for this rank.
    backend : str
        ``"mpi"`` for distributed execution, ``"local"`` for
        single-process mode (exchange is a no-op; call
        :func:`exchange_local_simulated` instead for testing).
    """

    def __init__(self, partition: VoronoiPartition, backend: str = "local"):
        self.partition = partition
        self.backend = backend

    def exchange_cell_field(self, field: jnp.ndarray) -> jnp.ndarray:
        """Exchange halo values for a cell-centered field.

        Parameters
        ----------
        field : jax.Array, shape ``(n_local_cells, ...)``

        Returns
        -------
        jax.Array
            Same shape, with halo positions updated.
        """
        return self._exchange(field, self.partition.cell_comm, entity_type=0)

    def exchange_edge_field(self, field: jnp.ndarray) -> jnp.ndarray:
        """Exchange halo values for an edge-centered field."""
        return self._exchange(field, self.partition.edge_comm, entity_type=1)

    def exchange_vertex_field(self, field: jnp.ndarray) -> jnp.ndarray:
        """Exchange halo values for a vertex-centered field."""
        return self._exchange(field, self.partition.vertex_comm, entity_type=2)

    def _exchange(
        self,
        field: jnp.ndarray,
        comm: HaloCommSchedule,
        entity_type: int,
    ) -> jnp.ndarray:
        if self.backend == "mpi":
            return _exchange_mpi(
                field, comm, self.partition.rank, entity_type,
            )
        # local backend: no-op (halos are already filled or irrelevant)
        return field


# ============================================================================
# MPI exchange
# ============================================================================

def _exchange_mpi(
    field: jnp.ndarray,
    comm: HaloCommSchedule,
    rank: int,
    entity_type: int,
) -> jnp.ndarray:
    """Perform MPI halo exchange using mpi4jax sendrecv.

    Gathers every neighbor's send buffer from the ORIGINAL field, issues
    one blocking sendrecv per neighbor, then scatters all received data
    back in a SINGLE functional update.

    Works for fields of any shape ``(n_local, ...)`` — multi-level
    fields are handled automatically.

    Why one scatter at the end (iter 3, strong-scaling fix): the previous
    version did ``field = field.at[recv_idx].set(recv_data)`` *inside* the
    neighbor loop.  That (a) copied the whole field once per neighbor
    (an O(n_local) functional update × n_neighbors — e.g. a 7 MB edge
    field copied 6× per exchange) and (b) created a false
    read-after-write dependency: ``send_buf = field[send_idx]`` for the
    next neighbor textually read the just-mutated ``field``, forcing XLA
    to serialize the blocking sendrecvs even though ``send_idx`` are
    OWNED entities that never overlap the halo ``recv_idx`` written by any
    neighbor.  Collecting all sends from the original field and scattering
    once removes both costs.  Measured comm overhead was 22 ms/step at
    np=4 (18 serialized sendrecv) and 30 ms at np=8 (36) — see
    docs/performance/scaling/amip_mpi_scaling.md.

    Correctness: ``send_idx`` ⊂ owned, ``recv_idx`` ⊂ halo (disjoint), and
    each halo entity is owned by exactly one neighbor, so the per-neighbor
    recv chunks are disjoint — concatenating them in neighbor order and
    writing at ``comm.recv_idx`` (laid out in the same order) reproduces
    the previous result exactly, order-independent.

    Differentiability: every message goes through the ``@jax.custom_vjp``
    sendrecv wrapper (:func:`legoesm.parallel.halo_exchange.get_sendrecv_vjp`)
    — the same AD-safe path the lat-lon / cubed-sphere halos use — so
    ``jax.grad`` routes halo cotangents back to the owning rank instead of
    hitting mpi4jax's broken transpose rule.  Never call raw
    ``mpi4jax.sendrecv`` here (enforced by the AST guard in
    ``tests/distributed/test_voronoi_batched_halo.py``).
    """
    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    from legoesm.parallel.reductions import require_mpi_stack

    mpi4jax, MPI = require_mpi_stack()
    sendrecv = get_sendrecv_vjp(mpi4jax)

    # Tag = the entity type ALONE (cell=0 / edge=1 / vertex=2).  mpi4jax
    # ``sendrecv`` already matches on (source, dest), so the rank PAIR is never
    # encoded in the tag -- the only thing source/dest does NOT separate is two
    # messages of DIFFERENT entity type between the same pair (a cell vs an edge
    # buffer, different sizes).  Same-entity multi-field exchanges between a pair
    # are paired by mpi4jax's token + MPI's non-overtaking guarantee.  Dropping
    # the old ``rank * 1000 + nbr`` pair offset removes the < 1000-rank ceiling
    # (and the MPI_TAG_UB pressure): the tag is now rank-INDEPENDENT, so the halo
    # scales to ANY rank count.  (``rank`` stays in the signature for API
    # stability with the message-count siblings; the rank-free tag no longer
    # uses it.)
    tag = _entity_tag(entity_type)
    if comm.neighbor_ranks:
        _check_tag_bound(tag, MPI, "_exchange_mpi")

    s_offset = 0
    recv_chunks = []

    for i, nbr_rank in enumerate(comm.neighbor_ranks):
        s_count = comm.send_counts[i]
        r_count = comm.recv_counts[i]

        send_idx = comm.send_idx[s_offset:s_offset + s_count]

        # Gather from the ORIGINAL field — independent of every recv.
        send_buf = field[send_idx]

        if field.ndim == 1:
            recv_buf = jnp.zeros(r_count, dtype=field.dtype)
        else:
            recv_buf = jnp.zeros(
                (r_count,) + field.shape[1:], dtype=field.dtype,
            )

        _count_message()
        recv_data = sendrecv(
            send_buf, recv_buf,
            nbr_rank, nbr_rank,
            tag,
            tag,
            MPI.COMM_WORLD,
        )

        recv_chunks.append(recv_data)
        s_offset += s_count

    if not recv_chunks:
        return field

    # Single scatter: one whole-field copy instead of n_neighbors.
    # comm.recv_idx is the concatenation of the per-neighbor recv indices
    # in the same order recv_chunks was built, so a flat set lines up.
    all_recv = jnp.concatenate(recv_chunks, axis=0)
    return field.at[comm.recv_idx].set(all_recv)


# ============================================================================
# Batched union-neighbor exchange (one message per neighbor per dtype group)
# ============================================================================
#
# Replaces the per-entity message pattern of ``_exchange_mpas_state``
# (u edges + T/p_s cells + tracer cells = n_edge_nbrs + 2*n_cell_nbrs
# messages) with ONE flat message per union neighbor per dtype group
# (n_union_nbrs messages when all prognostic fields share a dtype).
# Pack/unpack is pure gather/reshape/concatenate/split/scatter — no
# arithmetic — so exchanged values are bit-identical to the per-entity
# path, and the VJP (split/scatter-add duals of the pack, gather duals of
# the unpack) is exact.

# MPI tags here are RANK-INDEPENDENT: mpi4jax ``sendrecv`` matches on
# (source, dest), so the rank pair is never encoded in the tag and the scheme
# scales to ANY rank count (no < 1000-rank ceiling).  A tag only separates
# concurrent messages between the SAME (source, dest) pair: per-entity
# exchanges by entity type (cell=0 / edge=1 / vertex=2), the batched path by
# dtype-group index (``_BATCH_TAG_BASE + g``, offset above the entity tags so
# the two schemes stay disjoint).  Group tags grow with the NUMBER OF DTYPE
# GROUPS (tiny in practice -- one per distinct prognostic dtype, not per field);
# a pathological group count is caught by ``_check_tag_bound`` (raises, never
# silently exceeds MPI_TAG_UB) -- it is NOT bounded for unlimited groups.
#
# ORDERING INVARIANT (required -- the SAME one the cube / lat-lon / plane halos
# rely on, which likewise reuse a fixed tag across every exchange call): two
# messages that share (comm, source, dest, tag) -- e.g. the production MPAS
# step's two cell exchanges (T+p_s, then tracers) between one pair -- are
# disambiguated NOT by the tag but by ORDER.  Every rank runs the same SPMD
# program order; mpi4jax's ordered effect keeps the sendrecvs from being
# reordered/parallelised; MPI's non-overtaking guarantee then pairs them 1:1.
# A NEW halo path that issues same-(source,dest,tag) messages MUST preserve that
# program-order property (regression: test_voronoi_mpi
# ::TestHaloExchange::test_repeated_same_entity_exchange_no_cross_match).
_BATCH_TAG_BASE = 8  # > the entity tags {0, 1, 2}; leaves headroom


def _entity_tag(entity_type: int) -> int:
    """MPI tag for a per-entity halo message: the entity type ALONE.

    Rank-independent: mpi4jax ``sendrecv`` matches on (source, dest), so the
    pair is never encoded in the tag.  The entity type is the only thing
    source/dest does not separate (cell vs edge vs vertex between one pair).
    """
    return entity_type


def _batch_group_tag(group: int) -> int:
    """MPI tag for a batched dtype-group message: ``_BATCH_TAG_BASE + group``.

    Rank-independent like :func:`_entity_tag`; the group index separates the
    concurrent dtype-group messages between a pair, and the base sits above the
    per-entity tags so the two schemes never collide.
    """
    return _BATCH_TAG_BASE + group


def _mpi_tag_ub(MPI) -> int | None:
    """The MPI implementation's largest valid tag (None if unqueryable)."""
    try:
        return MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)
    except Exception:  # pragma: no cover - exotic MPI without TAG_UB attr
        return None


def _check_tag_bound(max_tag: int, MPI, context: str) -> None:
    """Fail EARLY (host-side, at trace time) if a tag exceeds MPI_TAG_UB.

    The MPI standard only guarantees tags up to ``MPI_TAG_UB`` (>= 32767);
    common implementations allow ~2**31-1, which the multi-million tag
    bases here rely on.  A silent overflow would mis-pair messages, so
    raise loudly instead.
    """
    tag_ub = _mpi_tag_ub(MPI)
    if tag_ub is not None and max_tag > tag_ub:
        raise ValueError(
            f"{context}: computed MPI tag {max_tag} exceeds this MPI "
            f"implementation's MPI_TAG_UB={tag_ub}; the halo tag scheme "
            f"cannot address this rank count / group count here."
        )


def _field_width(arr: jnp.ndarray) -> int:
    """Static per-row element count (product of trailing dims)."""
    w = 1
    for d in arr.shape[1:]:
        w *= int(d)
    return w


def _prefix_offsets(counts: tuple[int, ...]) -> list[int]:
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + int(c))
    return offs


def _dtype_field_groups(edge_fields, cell_fields):
    """Group fields by dtype — one MPI buffer per dtype group per neighbor.

    Never mixes dtypes in a buffer (a flat ``concatenate`` across dtypes
    would silently promote/narrow and break serial parity).  Group order
    is first-appearance order over (edge fields..., cell fields...) in
    the caller's field order — a trace-time constant, identical on every
    rank for identically-structured states.

    Returns
    -------
    list of ``(dtype, members)`` with ``members = [(kind, pos), ...]``,
    ``kind`` in {"edge", "cell"} and ``pos`` the index into the
    corresponding field tuple.
    """
    groups: list[tuple] = []
    index: dict = {}
    for kind, fields in (("edge", edge_fields), ("cell", cell_fields)):
        for j, f in enumerate(fields):
            dt = jnp.dtype(f.dtype)
            if dt not in index:
                index[dt] = len(groups)
                groups.append((dt, []))
            groups[index[dt]][1].append((kind, j))
    return groups


def _message_schedule(edge_fields, cell_fields, sched: BatchedHaloSchedule):
    """Static per-(group, neighbor) message plan.

    Single source of truth shared by :func:`batched_halo_exchange` and
    :func:`count_batched_messages` so the pure counter can never diverge
    from what the exchange actually posts.

    Returns ``(groups, plan)`` where ``plan[g][i] = (post, send_size,
    recv_size)`` in elements.  A message is skipped (``post=False``) only
    when BOTH sizes are zero — a symmetric predicate across the pair
    (A's send size to B equals B's recv size from A by schedule-count
    symmetry), so both ranks skip together and the collective sequence
    stays uniform.
    """
    groups = _dtype_field_groups(edge_fields, cell_fields)
    plan = []
    for _dtype, members in groups:
        row = []
        for i in range(len(sched.neighbor_ranks)):
            s_sz = 0
            r_sz = 0
            for kind, j in members:
                if kind == "edge":
                    w = _field_width(edge_fields[j])
                    s_sz += sched.edge_send_counts[i] * w
                    r_sz += sched.edge_recv_counts[i] * w
                else:
                    w = _field_width(cell_fields[j])
                    s_sz += sched.cell_send_counts[i] * w
                    r_sz += sched.cell_recv_counts[i] * w
            row.append((s_sz > 0 or r_sz > 0, s_sz, r_sz))
        plan.append(row)
    return groups, plan


def count_batched_messages(edge_fields, cell_fields,
                           sched: BatchedHaloSchedule) -> int:
    """Messages :func:`batched_halo_exchange` posts for these fields.

    Pure schedule math (no MPI): callable on any host with synthetic
    fields/schedules.  Equals ``sched.messages_per_exchange(n_groups)``
    when every dtype group has traffic with every union neighbor (the
    homogeneous production case).
    """
    _, plan = _message_schedule(edge_fields, cell_fields, sched)
    return sum(1 for row in plan for (post, _, _) in row if post)


def pack_batched_sends(edge_fields, cell_fields,
                       sched: BatchedHaloSchedule):
    """Build flat per-(dtype group, union neighbor) send buffers.

    For each group (see :func:`_dtype_field_groups`) and each union
    neighbor, gathers every member field's send rows (edge fields from
    ``edge_send_idx``, cell fields from ``cell_send_idx``), flattens, and
    concatenates in member order.  All slice bounds come from the
    schedule's Python-int counts — static shapes under JIT.

    Returns ``buffers[g][i]`` — flat 1-D arrays of the group dtype
    (zero-length where the neighbor has no send rows for the group).
    """
    groups = _dtype_field_groups(edge_fields, cell_fields)
    c_offs = _prefix_offsets(sched.cell_send_counts)
    e_offs = _prefix_offsets(sched.edge_send_counts)

    buffers = []
    for dtype, members in groups:
        bufs = []
        for i in range(len(sched.neighbor_ranks)):
            chunks = []
            for kind, j in members:
                if kind == "edge":
                    rows = sched.edge_send_idx[e_offs[i]:e_offs[i + 1]]
                    arr = edge_fields[j]
                else:
                    rows = sched.cell_send_idx[c_offs[i]:c_offs[i + 1]]
                    arr = cell_fields[j]
                chunks.append(arr[rows].reshape(-1))
            bufs.append(jnp.concatenate(chunks) if chunks
                        else jnp.zeros(0, dtype=dtype))
        buffers.append(bufs)
    return buffers


def unpack_batched_recvs(edge_fields, cell_fields,
                         sched: BatchedHaloSchedule, recv_buffers):
    """Scatter received flat buffers back into halo rows of each field.

    Exact inverse layout of :func:`pack_batched_sends` on the recv side:
    ``recv_buffers[g][i]`` splits into per-member-field chunks (sizes =
    recv rows x field width, static).  Per field, the per-neighbor chunks
    are disjoint halo rows; concatenating them in union-neighbor order
    lines up with ``cell_recv_idx`` / ``edge_recv_idx``, so each field is
    updated with a SINGLE functional scatter (one whole-field copy, same
    optimization as :func:`_exchange_mpi`).

    Returns ``(edge_fields_out, cell_fields_out)`` tuples in input order.
    """
    groups = _dtype_field_groups(edge_fields, cell_fields)
    n_nbrs = len(sched.neighbor_ranks)

    # Per-field flat recv segments, collected in union-neighbor order.
    segments: dict = {}
    for g, (_dtype, members) in enumerate(groups):
        for i in range(n_nbrs):
            buf = recv_buffers[g][i]
            off = 0
            for kind, j in members:
                if kind == "edge":
                    n_rows = sched.edge_recv_counts[i]
                    w = _field_width(edge_fields[j])
                else:
                    n_rows = sched.cell_recv_counts[i]
                    w = _field_width(cell_fields[j])
                sz = n_rows * w
                segments.setdefault((kind, j), []).append(buf[off:off + sz])
                off += sz

    edge_out = list(edge_fields)
    cell_out = list(cell_fields)
    for (kind, j), chunks in segments.items():
        if kind == "edge":
            arr = edge_fields[j]
            recv_idx = sched.edge_recv_idx
        else:
            arr = cell_fields[j]
            recv_idx = sched.cell_recv_idx
        total_rows = int(recv_idx.shape[0])
        if total_rows == 0:
            continue
        flat = jnp.concatenate(chunks)
        vals = flat.reshape((total_rows,) + arr.shape[1:])
        updated = arr.at[recv_idx].set(vals)
        if kind == "edge":
            edge_out[j] = updated
        else:
            cell_out[j] = updated
    return tuple(edge_out), tuple(cell_out)


def batched_halo_exchange(edge_fields, cell_fields,
                          sched: BatchedHaloSchedule, rank: int):
    """One MPI halo exchange for edge AND cell fields together.

    Issues ONE blocking sendrecv per union neighbor per dtype group
    (vs one per neighbor per entity exchange), via the AD-safe
    ``custom_vjp`` sendrecv wrapper — fully reverse-mode differentiable.
    Pack/unpack are linear gather/scatter, so results are bit-identical
    to the per-entity exchanges for the production same-compute-dtype
    state (T, p_s, and tracers share the compute dtype; the legacy path's
    ``concatenate``/``stack`` of a hypothetical MIXED-dtype T/p_s could
    type-promote where the batched dtype-grouping would not — not a
    production config).  The collective schedule is uniform: every rank
    walks its sorted union-neighbor list with matching (tag, size) pairs.

    DEFAULT in production (opt-OUT via ``LEGOESM_VORONOI_BATCHED_HALO=0``);
    the historical 18x CPU regression was fixed by the one-scatter
    pack/unpack and re-measured faster at all sizes — see the module
    docstring.

    Parameters
    ----------
    edge_fields : sequence of jax.Array, shape ``(n_local_edges, ...)``
    cell_fields : sequence of jax.Array, shape ``(n_local_cells, ...)``
        Field set, order, dtypes, and trailing shapes must be identical
        on every rank (fixed at trace time).
    sched : BatchedHaloSchedule
        From :func:`build_batched_halo_schedule` (layout-build constant).
    rank : int
        This MPI rank (for tag construction).

    Returns
    -------
    (edge_fields_out, cell_fields_out) : tuples of jax.Array
        Same order/shapes/dtypes, halo rows updated.
    """
    edge_fields = tuple(edge_fields)
    cell_fields = tuple(cell_fields)
    if not sched.neighbor_ranks:
        # np=1 / no-neighbor degeneracy: identity, no MPI dependency.
        return edge_fields, cell_fields

    from legoesm.parallel.halo_exchange import get_sendrecv_vjp
    from legoesm.parallel.reductions import require_mpi_stack

    mpi4jax, MPI = require_mpi_stack()
    sendrecv = get_sendrecv_vjp(mpi4jax)

    groups, plan = _message_schedule(edge_fields, cell_fields, sched)
    send_buffers = pack_batched_sends(edge_fields, cell_fields, sched)

    # Rank-INDEPENDENT tags: one per dtype group (``_BATCH_TAG_BASE + g``).
    # mpi4jax matches on (source, dest), so a tag must only separate the
    # concurrent dtype-group messages between the SAME pair -- the group index
    # does that.  No rank in the tag => no rank ceiling.  Validate the largest
    # tag up front (defensive; it is tiny).  (``rank`` stays in the signature
    # for API stability; the rank-free tag no longer uses it.)
    max_tag = _batch_group_tag(len(groups) - 1)
    _check_tag_bound(max_tag, MPI, "batched_halo_exchange")

    recv_buffers = []
    for g, (dtype, _members) in enumerate(groups):
        tag = _batch_group_tag(g)
        g_recv = []
        for i, nbr_rank in enumerate(sched.neighbor_ranks):
            post, _s_sz, r_sz = plan[g][i]
            if not post:
                g_recv.append(jnp.zeros(0, dtype=dtype))
                continue
            _count_message()
            g_recv.append(sendrecv(
                send_buffers[g][i],
                jnp.zeros(r_sz, dtype=dtype),
                nbr_rank, nbr_rank,
                tag,
                tag,
                MPI.COMM_WORLD,
            ))
        recv_buffers.append(g_recv)

    return unpack_batched_recvs(edge_fields, cell_fields, sched,
                                recv_buffers)


# ============================================================================
# Local (non-MPI) utilities for testing
# ============================================================================

def exchange_local_simulated(
    partitions: list[VoronoiPartition],
    local_fields: list[jnp.ndarray],
    entity: str = "cell",
) -> list[jnp.ndarray]:
    """Simulate MPI halo exchange in a single process.

    Reconstructs a global field from all ranks' owned values, then
    copies the correct halo values into each rank's local field.

    Parameters
    ----------
    partitions : list of VoronoiPartition
        One partition per simulated rank.
    local_fields : list of jnp.ndarray
        One local field per rank, shape ``(n_local_*, ...)``.
    entity : str
        ``"cell"``, ``"edge"``, or ``"vertex"``.

    Returns
    -------
    list of jnp.ndarray
        Updated local fields with halo values filled.
    """
    if entity == "cell":
        g_size = partitions[0].nCells_global
        n_own = lambda p: p.n_owned_cells
        local_ents = lambda p: p.local_cells
    elif entity == "edge":
        g_size = partitions[0].nEdges_global
        n_own = lambda p: p.n_owned_edges
        local_ents = lambda p: p.local_edges
    elif entity == "vertex":
        g_size = partitions[0].nVertices_global
        n_own = lambda p: p.n_owned_vertices
        local_ents = lambda p: p.local_vertices
    else:
        raise ValueError(f"Unknown entity type: {entity!r}")

    dtype = local_fields[0].dtype
    trailing = local_fields[0].shape[1:]
    global_field = jnp.zeros((g_size,) + trailing, dtype=dtype)

    # Scatter owned values into the global field.
    for p, f in zip(partitions, local_fields):
        n = n_own(p)
        idx = local_ents(p)[:n]
        global_field = global_field.at[idx].set(f[:n])

    # Fill each rank's halo from the global field.
    result = []
    for p, f in zip(partitions, local_fields):
        n = n_own(p)
        halo_idx = local_ents(p)[n:]
        updated = f.at[n:].set(global_field[halo_idx])
        result.append(updated)

    return result
