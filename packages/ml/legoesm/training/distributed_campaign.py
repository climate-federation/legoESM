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


def slice_reference_to_local(reference: Any, local_cells: Any) -> Any:
    """Cut a GLOBAL reference :class:`ColumnState` to a rank's LOCAL cells.

    The distributed model runs rank-local (each rank holds its ``local_cells`` =
    owned + halo), so the GLOBAL ERA5 reference must be sliced to the SAME cell axis
    before the compare.  ``local_cells`` are the global cell ids of this rank's
    local cells (``layout.partition.local_cells``); every per-cell field (the
    leading axis is the cell axis) is gathered by them, and ``None`` fields (e.g.
    ``precip_mm_day`` absent on a cell-wind reference) pass through.  Order is
    PRESERVED so the sliced reference aligns row-for-row with the rank-local model
    state.

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
    idx = jnp.asarray(np.asarray(local_cells))

    def _slice(x: Any) -> Any:
        return None if x is None else jnp.asarray(x)[idx]

    return reference._replace(
        **{f: _slice(getattr(reference, f)) for f in reference._fields}
    )
