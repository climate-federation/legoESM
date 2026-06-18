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
