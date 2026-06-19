"""Compose the distributed-MPAS correction-campaign hooks into one entry point.

The capstone of the distributed-MPAS work (iters 86–88): each piece is a separate
hook on the (single-process-default) campaign, and a real distributed run must
supply ALL of them together, consistently keyed off the SAME rank-local Voronoi
partition.  This module builds that triple from the active partition layout so the
caller cannot mismatch them:

* **owned-cell ``valid_mask``** (iter 86) — each rank ranks only its OWNED cells.
* **global top-k ``manifest_reducer``** (iter 87) — reduces the per-rank owned
  rankings to the GLOBAL ``n_worst`` worst columns (each diagnosed on its owner).
* **collective ``global_reduce``** (iter 88) — ``global_sum_mpi`` so the loop's
  no-op gate, line search, and accept gate use the GLOBAL bias / count (no
  rank-divergent control flow → no deadlock).

Plus :func:`slice_reference_to_local` to cut the GLOBAL ERA5 reference down to the
rank's local cells (owned + halo) so the rank-local model state and the reference
are aligned on the same cell axis before the compare.

These are pure composition (the ``manifest_reducer`` / ``global_reduce`` it returns
are MPI-collective when CALLED, but building them is not), so they unit-test
without MPI; the end-to-end collective behaviour is covered by the distributed
test.  Grid-agnostic over anything exposing ``owned_mask_cells`` +
``partition.local_cells`` (the Voronoi/MPAS layout today).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
from legoesm.training.compare_reanalysis import owned_cell_valid_mask
from legoesm.training.distributed_manifest import gather_global_worst_columns


def distributed_campaign_hooks(
    layout: Any, n_worst: int, *, base_valid_mask: Any = None
) -> tuple[Any, Any, Any]:
    """Build the ``(valid_mask, manifest_reducer, global_reduce)`` triple for a
    DISTRIBUTED-MPAS campaign from the active partition ``layout``.

    ``layout`` is the rank's :class:`VoronoiPartitionLayout` (needs
    ``owned_mask_cells`` ``(n_local,)`` bool + ``partition.local_cells``
    ``(n_local,)`` local→global cell ids).  ``n_worst`` is the GLOBAL worst-column
    budget (the same value passed to the campaign).  ``base_valid_mask`` (optional)
    is a further per-cell restriction (e.g. an ocean/land mask) AND-ed into the
    owned mask.  All three returned hooks are keyed off the SAME ``layout``, so a
    caller cannot accidentally pair an owned mask from one partition with a reducer
    from another.  Pass them straight to ``build_correction_campaign(valid_mask=…,
    manifest_reducer=…, global_reduce=…, n_worst=n_worst)``.
    """
    from legoesm.parallel.reductions import global_sum_mpi

    valid_mask = owned_cell_valid_mask(layout, base_mask=base_valid_mask)
    local_to_global = np.asarray(layout.partition.local_cells)
    manifest_reducer = partial(
        gather_global_worst_columns,
        local_to_global=local_to_global, n_worst=int(n_worst),
    )
    return valid_mask, manifest_reducer, global_sum_mpi


def assert_partition_covers_global(layout: Any, *, global_reduce: Any = None) -> None:
    """Pre-flight COLLECTIVE check that the rank partition covers EVERY global cell
    EXACTLY ONCE — raises ``ValueError`` on a gap or an overlap.

    The distributed campaign ranks each rank's OWNED cells, reduces to the GLOBAL
    ``n_worst`` worst, and corrects them on their owners.  That is only correct if the
    owned sets across ranks PARTITION the global mesh: a cell owned by NO rank is
    silently NEVER ranked/corrected (a permanent bias the loop cannot see); a cell
    owned by >1 rank is double-counted in the global top-k and double-corrected.  A
    partitioner regression, a wrong ``n_ranks``, or a layout built against a different
    mesh all break this invariant — and a JAX scatter would otherwise hide it.  This
    catches it BEFORE a multi-day run instead of silently corrupting the science.

    Each rank scatters a one-hot owned-count over the ``nCells_global`` global cells;
    ``global_reduce`` (an allreduce SUM — ``global_sum_mpi`` by default; inject a
    plain sum/identity for a single-rank unit test) sums them; every global cell must
    end at exactly 1.  MUST be called on EVERY rank (it is collective).  A ONE-TIME
    ``O(nCells_global)`` diagnostic (like ``gather_to_global`` for I/O) — NOT a
    per-step op.  ``layout`` needs ``owned_mask_cells`` ``(n_local,)`` bool +
    ``partition.local_cells`` ``(n_local,)`` + ``partition.nCells_global``.

    PRECONDITION: ``partition.nCells_global`` sets the reduced vector's shape
    (``n_global + 2``), so it MUST be present and the SAME on every rank — a rank that
    cannot read it cannot form the matching buffer and so cannot participate in the
    collective at all.  ``partition.local_cells`` must likewise be present (it is read
    to build the owned ids before the collective).  Both hold by construction for a
    layout from ``make_voronoi_partition_layout`` (the same NamedTuple on every rank).
    Detected DATA malformations (mask shape, out-of-range id, gaps, overlaps) ARE
    synchronized so every rank raises together (no pre-collective rank-divergent raise).
    """
    if global_reduce is None:
        from legoesm.parallel.reductions import global_sum_mpi
        global_reduce = global_sum_mpi

    n_global = int(layout.partition.nCells_global)
    local_cells = np.asarray(layout.partition.local_cells)
    owned = np.asarray(layout.owned_mask_cells, dtype=bool)

    # COLLECTIVE-SAFE structural checks: a per-rank malformation (mask-length mismatch
    # or out-of-range owned id) is detected LOCALLY into a flag, then synchronized via
    # the SAME allreduce as the owned counts — so EVERY rank raises together. A rank-
    # local `raise` BEFORE the collective would hang the others inside the allreduce
    # (Codex iter 98, HIGH). The flags ride two EXTRA slots of the reduced vector
    # (slot n_global = #ranks with a length mismatch, n_global+1 = #ranks with an
    # out-of-range id), so one SUM reduction carries everything (no second primitive).
    # a non-1-D mask (or a length mismatch) is a malformed layout — fold the ndim
    # check in so ``local_cells[owned]`` below cannot raise an IndexError BEFORE the
    # collective on one rank only (which would hang the others).
    mask_bad = owned.ndim != 1 or owned.shape[0] != local_cells.shape[0]
    oob_bad = False
    owned_global = np.empty(0, dtype=local_cells.dtype)
    if not mask_bad:
        owned_global = local_cells[owned]            # global ids this rank OWNS
        if owned_global.size and (
                int(owned_global.min()) < 0 or int(owned_global.max()) >= n_global):
            oob_bad = True                           # skip the scatter (would clamp)
    counts_local = jnp.zeros(n_global + 2, dtype=jnp.int32)
    if not mask_bad and not oob_bad and owned_global.size:
        counts_local = counts_local.at[jnp.asarray(owned_global)].add(1)
    counts_local = counts_local.at[n_global].add(int(mask_bad))
    counts_local = counts_local.at[n_global + 1].add(int(oob_bad))

    counts_global = np.asarray(global_reduce(counts_local))
    n_mask_bad = int(counts_global[n_global])
    n_oob_bad = int(counts_global[n_global + 1])
    if n_mask_bad:                                   # raised on ALL ranks (synced)
        raise ValueError(
            "assert_partition_covers_global: owned_mask_cells must be a 1-D mask the "
            "same length as local_cells (one owned flag per local cell) — malformed "
            f"on {n_mask_bad} rank(s).")
    if n_oob_bad:
        raise ValueError(
            "assert_partition_covers_global: an owned global cell id is out of range "
            f"[0, {n_global}) — the partition is malformed on {n_oob_bad} rank(s).")
    owned_counts = counts_global[:n_global]          # how many ranks own each cell
    gaps = np.nonzero(owned_counts == 0)[0]
    overlaps = np.nonzero(owned_counts > 1)[0]
    if gaps.size or overlaps.size:
        raise ValueError(
            "assert_partition_covers_global: the rank partition does NOT cover the "
            f"global mesh exactly once — {gaps.size} cell(s) owned by NO rank (e.g. "
            f"{gaps[:5].tolist()}), {overlaps.size} owned by >1 rank (e.g. "
            f"{overlaps[:5].tolist()}). A gap is never corrected; an overlap is "
            "double-counted in the global top-k.")


def slice_reference_to_local(
    reference: Any, local_cells: Any, *, expected_n_cells: int | None = None
) -> Any:
    """Cut a GLOBAL reference :class:`ColumnState` to a rank's LOCAL cells.

    The distributed model runs rank-local (each rank holds its ``local_cells`` =
    owned + halo), so the GLOBAL ERA5 reference must be sliced to the SAME cell axis
    before the compare.  ``local_cells`` are the global cell ids of this rank's
    local cells (``layout.partition.local_cells``); every per-cell field (the
    leading axis is the cell axis) is gathered by them, and ``None`` fields (e.g.
    ``precip_mm_day`` absent on a cell-wind reference) pass through.  Order is
    PRESERVED so the sliced reference aligns row-for-row with the rank-local model
    state.

    ``expected_n_cells`` (optional) is the partitioned GLOBAL cell count
    (``layout.partition.nCells_global``); when given, the reference's cell-axis
    length MUST equal it (an EXACT mesh-identity check that also rejects a reference
    LONGER than the mesh, which the bounds check alone would miss).

    The reference MUST be a CELL-space comparison state: every field's leading axis
    is the cell axis, so ``u_edge`` (the model's NATIVE edge velocity, ``nEdges`` —
    an EDGE field that a cell-id slice would corrupt) MUST be ``None``.  A non-``None``
    ``u_edge`` means a MODEL state was passed as the reference; that is REJECTED
    LOUDLY rather than silently mis-sliced (Codex iter 89).
    """
    if getattr(reference, "u_edge", None) is not None:
        raise ValueError(
            "slice_reference_to_local: the reference must be a CELL-space state "
            "(u_edge is the model's native EDGE velocity on nEdges — it cannot be "
            "sliced by cell ids); got a non-None u_edge (a model state was passed "
            "as the reference?)."
        )
    idx_np = np.asarray(local_cells)
    # FAIL FAST on a global/reference cell-count MISMATCH: a JAX gather (``x[idx]``)
    # SILENTLY CLAMPS out-of-bounds indices to the edge, so a wrong-mesh ERA5
    # reference (or one for fewer cells than the partition spans) would corrupt the
    # compare on a multi-day run with NO error. ``local_cells`` + the reference shapes
    # are concrete (host) here (campaign-build time, never traced), so a Python check
    # is safe. EVERY per-cell field is gathered, so ALL must share the cell-axis
    # length (checking only the first would let an inconsistent later field clamp).
    cell_lens = {
        f: int(np.asarray(getattr(reference, f)).shape[0])
        for f in reference._fields if getattr(reference, f) is not None
    }
    if not cell_lens:
        raise ValueError(
            "slice_reference_to_local: reference has no per-cell fields to slice.")
    n_cells = next(iter(cell_lens.values()))
    bad = {f: length for f, length in cell_lens.items() if length != n_cells}
    if bad:
        raise ValueError(
            "slice_reference_to_local: inconsistent reference cell-axis lengths "
            f"(expected {n_cells}, got {bad}) — every per-cell field is sliced by the "
            "same cell ids, so they must agree.")
    if expected_n_cells is not None and n_cells != expected_n_cells:
        raise ValueError(
            f"slice_reference_to_local: reference has {n_cells} cells but the "
            f"partitioned global mesh has {expected_n_cells} — the GLOBAL ERA5 "
            "reference must be defined on the SAME mesh as the model run.")
    if idx_np.size:
        lo, hi = int(idx_np.min()), int(idx_np.max())
        if lo < 0 or hi >= n_cells:
            raise ValueError(
                "slice_reference_to_local: local_cells index out of range "
                f"[{lo}, {hi}] for a reference with {n_cells} cells (a JAX gather "
                "would otherwise silently clamp out-of-range ids and corrupt the "
                "compare).")
    idx = jnp.asarray(idx_np)

    def _slice(x: Any) -> Any:
        return None if x is None else jnp.asarray(x)[idx]

    return reference._replace(
        **{f: _slice(getattr(reference, f)) for f in reference._fields}
    )


def assemble_global_field(
    layout: Any, local_field: Any, *, global_reduce: Any = None,
    verify_coverage: bool = True,
) -> np.ndarray:
    """Re-assemble a rank-local corrected coefficient field into the GLOBAL field.

    The exact FORWARD inverse of :func:`slice_reference_to_local`: that cuts a global
    per-cell field DOWN to a rank's local cells before the run; this gathers each
    rank's OWNED-cell corrected values back UP into one ``(nCells_global,)`` global
    field after the run, so the multi-day distributed campaign's result can be
    persisted (or compared) as a single global array.  Without it a user composing an
    MPI driver from :func:`build_distributed_mpas_campaign` has only a rank-LOCAL
    ``result.final_field`` (each rank corrected only its owned cells); writing that
    per rank races on the path AND each file holds a PARTIAL field.

    Mechanics mirror :func:`assert_partition_covers_global`: each rank scatters its
    OWNED cells' values over a ``nCells_global``-length vector by GLOBAL id (non-owned
    halo cells are ignored), and ``global_reduce`` (an allreduce SUM —
    ``global_sum_mpi`` by default; inject a plain identity/sum for a single-rank unit
    test) sums them.  Because the owned sets PARTITION the global mesh (every cell
    owned by exactly one rank), each global cell receives exactly one contribution, so
    the SUM reproduces the value with no double-count.  That partition invariant is the
    precondition; with ``verify_coverage`` (default) it is re-checked up front via
    :func:`assert_partition_covers_global` (the canonical gate — NOT re-derived here),
    so an unverified caller still fails LOUDLY on a gap/overlap instead of silently
    summing a doubled value.

    ``local_field`` is the rank-local corrected field (``result.final_field``); it is
    flattened to ``(n_local,)`` and MUST align with ``partition.local_cells`` /
    ``owned_mask_cells`` (a length mismatch, like an out-of-range owned id, is folded
    into the SAME reduction's two sentinel slots so EVERY rank raises together — never
    a rank-local raise that would hang the others inside the collective, Codex iter 98).

    A ONE-TIME ``O(nCells_global)`` collective (like ``gather_to_global`` for I/O), NOT
    a per-step op.  Runs in float64 (the summed coefficients) — enable ``x64`` (the
    campaign does); MUST be called on EVERY rank.  Returns the global field identically
    on every rank (allreduce); write it on rank 0 ONLY (e.g. ``if rank == 0:
    _atomic_write_json(...)``) so the persist does not race.
    """
    if global_reduce is None:
        from legoesm.parallel.reductions import global_sum_mpi
        global_reduce = global_sum_mpi
    if verify_coverage:
        assert_partition_covers_global(layout, global_reduce=global_reduce)

    n_global = int(layout.partition.nCells_global)
    local_cells = np.asarray(layout.partition.local_cells)
    owned = np.asarray(layout.owned_mask_cells, dtype=bool)
    field_flat = np.asarray(local_field).reshape(-1)

    # Per-rank structural defects crash the scatter LOCALLY on one rank only (which
    # would hang the others in the allreduce), so detect them into flags and ride two
    # EXTRA reduced slots — one SUM carries the values AND the flags (mirrors the
    # owned-count gate). A non-1-D / mis-length owned mask OR a field that does not
    # align with local_cells is malformed; an owned global id out of range would clamp.
    mask_bad = (
        owned.ndim != 1
        or owned.shape[0] != local_cells.shape[0]
        or field_flat.shape[0] != local_cells.shape[0]
    )
    oob_bad = False
    owned_global = np.empty(0, dtype=local_cells.dtype)
    owned_vals = np.empty(0, dtype=np.float64)
    if not mask_bad:
        owned_global = local_cells[owned]
        owned_vals = field_flat[owned].astype(np.float64)
        if owned_global.size and (
                int(owned_global.min()) < 0 or int(owned_global.max()) >= n_global):
            oob_bad = True                           # skip the scatter (would clamp)

    buf = jnp.zeros(n_global + 2, dtype=jnp.float64)
    if not mask_bad and not oob_bad and owned_global.size:
        buf = buf.at[jnp.asarray(owned_global)].add(jnp.asarray(owned_vals))
    buf = buf.at[n_global].add(float(mask_bad))
    buf = buf.at[n_global + 1].add(float(oob_bad))

    red = np.asarray(global_reduce(buf))
    n_mask_bad = int(round(float(red[n_global])))
    n_oob_bad = int(round(float(red[n_global + 1])))
    if n_mask_bad:                                   # raised on ALL ranks (synced)
        raise ValueError(
            "assemble_global_field: owned_mask_cells / local_field must be 1-D and the "
            "same length as local_cells (one value + owned flag per local cell) — "
            f"malformed on {n_mask_bad} rank(s).")
    if n_oob_bad:
        raise ValueError(
            "assemble_global_field: an owned global cell id is out of range "
            f"[0, {n_global}) — the partition is malformed on {n_oob_bad} rank(s).")
    return red[:n_global]


def assemble_global_campaign_result(
    result: Any, layout: Any, *, corrected_field: str | None = None,
    global_reduce: Any = None, verify_coverage: bool = True,
) -> Any:
    """Replace a DISTRIBUTED campaign result's RANK-LOCAL corrected field(s) with the
    GLOBAL assembled field(s) — the one rank-local→global step an MPI driver needs
    between the campaign and the rank-0 persist.

    A distributed campaign returns a result whose biases / accepted flags / summary
    inputs are ALREADY global (the loop reduces them with ``global_reduce``); the ONLY
    rank-local pieces are the corrected coefficient FIELDS.  This calls
    :func:`assemble_global_field` on each and returns a new result (NamedTuple
    ``_replace``) whose ``final_field`` / ``final_fields`` (and the single
    ``final_config.<coef>``) are GLOBAL — so the EXISTING ``summarize_campaign`` +
    ``build_campaign_output_dict`` + deploy path produce GLOBAL output with NO change.
    Pure (no I/O, no rank-0 logic — the driver does the rank-0 write); call it on EVERY
    rank (``assemble_global_field`` is collective), then ``if rank == 0: write(...)``.

    SINGLE (``CampaignResult``): pass ``corrected_field`` (the coefficient name, e.g.
    ``"C_K"``) — BOTH ``result.final_field`` AND ``result.final_config.<corrected_field>``
    are replaced (the two places the summary + output consumers read it).  MULTI
    (``MultiCampaignResult``, detected by a ``final_fields`` attribute): ``corrected_field``
    is ignored and EVERY ``final_fields[key]`` is replaced (the consumers read
    ``final_fields``, not the config, for multi).

    Coverage is verified ONCE (the first field) then reused for the rest via
    ``verify_coverage=False`` — the partition is identical across the fields, so
    re-running the (collective) gate per field would only add redundant reductions.
    """
    verified = [not verify_coverage]      # list cell so the closure can flip it

    def _to_global(field: Any) -> Any:
        vc = not verified[0]
        verified[0] = True
        return jnp.asarray(assemble_global_field(
            layout, field, global_reduce=global_reduce, verify_coverage=vc))

    if hasattr(result, "final_fields"):              # MultiCampaignResult
        global_fields = {k: _to_global(v) for k, v in result.final_fields.items()}
        return result._replace(final_fields=global_fields)

    if corrected_field is None:                      # CampaignResult (single)
        raise ValueError(
            "assemble_global_campaign_result: a single-coefficient CampaignResult needs "
            "corrected_field (the coefficient name, e.g. 'C_K') so final_config.<coef> "
            "can be replaced too; only a MultiCampaignResult (final_fields) infers it.")
    global_field = _to_global(result.final_field)
    new_config = result.final_config._replace(**{corrected_field: global_field})
    return result._replace(final_field=global_field, final_config=new_config)
