"""Cross-rank global top-k of the per-rank worst-column manifests (distributed MPAS).

The companion to the iter-86 owned-cell mask
(:func:`legoesm.training.compare_reanalysis.owned_cell_valid_mask`).  That mask
makes each MPI rank rank ONLY the cells it owns; this module reduces those
per-rank owned rankings to the GLOBAL ``n_worst`` worst columns, so a distributed
campaign spins off exactly ``n_worst`` LES (the globally worst cells) instead of
``R × n_worst`` (each rank's local worst).  Without it, R ranks each pick their
own ``n_worst`` and the campaign over-spawns LES on cells that are not globally
worst.

Architecture (most logic is PURE + unit-testable; MPI is a thin shell):

* :func:`select_global_top_k` — pure host-side selection of the ``n_worst``
  highest-score, DISTINCT-id candidates from the gathered per-rank arrays.
* :func:`owned_subset_of_global_top_k` — pure: given the gathered candidates +
  this rank's local manifest + its local→global id map, return the rank's OWNED
  records that made the global cut (in global score order).
* :func:`gather_global_worst_columns` — the MPI reducer: build this rank's
  fixed-size padded candidate arrays, ``allgather_mpi`` them (fixed shape per
  rank, the allgather contract), and call the pure kernels.  Used ONLY under MPI
  (the single-process runner needs no reduction — every cell is owned and the
  local top-k IS the global top-k); it is the ``manifest_reducer`` a distributed
  driver passes to :func:`legoesm.training.correction_loop.make_compare_fn`.

This is host-side ORCHESTRATION (which columns to diagnose), not a traced /
differentiated quantity, so the non-AD ``allgather`` (CLAUDE.md: allgather is
diagnostics/IO only) is appropriate here.  Selection is plain NumPy on the small
``R × n_worst`` gathered set — no JAX top-k, no tracing.

.. note::
   This reducer makes the per-rank RANKING globally correct; the downstream
   correction LOOP was made COLLECTIVE in iter 88, so the reducer is safely wired
   into a fully distributed campaign.  The two iter-87 deadlock/divergence hazards
   are RESOLVED via ``global_reduce=global_sum_mpi``: (1) the ``if not records:``
   no-op gate could skip the second ``compare_fn`` (model re-run) on a rank owning
   NONE of the globally-selected columns while others re-run (an MPI
   collective-count mismatch); (2) the line-search improvement test used the
   RANK-LOCAL bias, so ranks could pick different step fractions.  Now
   :func:`legoesm.training.correction_loop.run_correction_iteration` reduces the bias
   globally (``global_sum_mpi`` of the weighted numerator/denominator), chooses the
   step fraction collectively, and runs ``compare_fn`` in lockstep on every rank.
   The single-process runner leaves ``manifest_reducer=None`` (gate + line search
   behave exactly as before).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def select_global_top_k(
    scores: Any, global_ids: Any, valid: Any, n_worst: int
) -> list[int]:
    """The ``n_worst`` highest-score, DISTINCT-global-id candidates (descending).

    ``scores`` / ``global_ids`` / ``valid`` are flat ``(M,)`` arrays of the gathered
    per-rank candidates (``M = n_ranks × n_worst``, padded slots flagged
    ``valid=False``).  Returns the selected GLOBAL cell ids, ordered by descending
    score.  Invalid (padding) slots are never selected.  De-duplicates by global id
    keeping the highest score — owned-masking already makes ids distinct across
    ranks, but a duplicate (e.g. a caller that forgot the owned mask) would
    otherwise consume two of the ``n_worst`` slots; de-dup keeps the result the
    globally-worst DISTINCT cells regardless.
    """
    s = np.asarray(scores, dtype=float).reshape(-1)
    ids = np.asarray(global_ids).reshape(-1)
    ok = np.asarray(valid, dtype=bool).reshape(-1)
    if not (s.shape == ids.shape == ok.shape):
        raise ValueError(
            f"select_global_top_k: scores/global_ids/valid must share shape; got "
            f"{s.shape}, {ids.shape}, {ok.shape}."
        )
    # Route NOT-valid AND non-finite scores to -inf so they are never selected: a
    # NaN reaching argsort would otherwise sort to the END (ascending) → the FRONT
    # of the reversed (descending) order → be wrongly picked as "worst" (Codex).
    finite_ok = ok & np.isfinite(s)
    masked = np.where(finite_ok, s, -np.inf)
    # Descending by score; ties broken by ASCENDING global id — a canonical,
    # PARTITION-INDEPENDENT key (iter 213).  Without a gid tie-break, the prior
    # ``argsort(...)[::-1]`` resolved equal scores by the gathered candidate
    # POSITION, so the SAME global field decomposed across a different rank count
    # could select a DIFFERENT set of equally-worst columns (verified: a 3-way
    # tie for 2 slots gives {C,B} vs {B,A} depending on which rank owns which
    # cell) — breaking the campaign's resume/decomposition reproducibility.  The
    # gid tie-break also matches the single-rank path (``rank_worst_columns`` /
    # ``jax.lax.top_k``, which breaks ties by ascending flat index == ascending
    # global id here), so serial and distributed runs diagnose the same columns.
    # lexsort's LAST key is primary: ``-masked`` ascending == score descending;
    # ``ids`` ascending == lowest global id first on a tie.  Padding/NaN already
    # mapped to ``-inf`` ⇒ ``+inf`` primary key ⇒ sorts last (loop breaks there).
    ids_int = np.asarray(ids).reshape(-1).astype(np.int64)
    order = np.lexsort((ids_int, -masked))
    selected: list[int] = []
    seen: set[int] = set()
    for i in order:
        if not np.isfinite(masked[i]):
            break                                     # only -inf (invalid) remain
        gid = int(ids[i])
        if gid in seen:
            continue
        seen.add(gid)
        selected.append(gid)
        if len(selected) >= int(n_worst):
            break
    return selected


def owned_subset_of_global_top_k(
    gathered_scores: Any,
    gathered_ids: Any,
    gathered_valid: Any,
    local_manifest: list,
    local_to_global: Any,
    n_worst: int,
) -> list:
    """This rank's OWNED records that made the global top-k, in global score order.

    ``local_manifest`` is this rank's owned-only worst-column manifest (each
    :class:`~legoesm.training.column_manifest.ColumnRecord`, sorted worst-first);
    ``local_to_global`` maps a record's rank-local ``flat_index`` to its GLOBAL cell
    id (for MPAS: ``VoronoiPartition.local_cells``).  Given the gathered candidate
    arrays from all ranks, returns the records this rank owns that are among the
    global ``n_worst`` (a record's global id is the discriminator), so each globally
    worst cell is diagnosed on exactly its owning rank — no double-count.
    """
    l2g = np.asarray(local_to_global)
    selected = select_global_top_k(
        gathered_scores, gathered_ids, gathered_valid, n_worst)
    recs = list(local_manifest)[:int(n_worst)]
    by_gid = {int(l2g[r.flat_index]): r for r in recs}
    return [by_gid[g] for g in selected if g in by_gid]


def _local_candidate_arrays(local_manifest: list, local_to_global: Any, n_worst: int):
    """Fixed-size ``(n_worst,)`` (score, global_id, valid) for this rank, padded.

    The ``allgather`` contract requires every rank to contribute the SAME shape, so
    fewer-than-``n_worst`` owned candidates are padded with ``score=-inf``,
    ``global_id=-1``, ``valid=0``.  ``valid`` is gathered as ``int32`` (not bool)
    for cross-backend allgather robustness.
    """
    l2g = np.asarray(local_to_global)
    recs = list(local_manifest)[:int(n_worst)]
    scores = np.full((int(n_worst),), -np.inf, dtype=np.float64)
    gids = np.full((int(n_worst),), -1, dtype=np.int64)
    valid = np.zeros((int(n_worst),), dtype=np.int32)
    for k, r in enumerate(recs):
        scores[k] = float(r.combined_score)
        gids[k] = int(l2g[r.flat_index])
        valid[k] = 1
    return scores, gids, valid


def gather_global_worst_columns(
    local_manifest: list, local_to_global: Any, n_worst: int
) -> list:
    """MPI reducer: this rank's owned subset of the GLOBAL ``n_worst`` worst columns.

    A ``manifest_reducer`` for :func:`legoesm.training.correction_loop.make_compare_fn`
    under MPI: each rank passes its owned-only ``local_manifest`` (produced with the
    iter-86 owned mask) + its ``local_to_global`` id map; the per-rank top-``n_worst``
    candidates are ``allgather``-ed (fixed shape) and the global ``n_worst`` selected,
    returning the records THIS rank owns (so the campaign diagnoses each globally
    worst cell on exactly one rank).  Single-rank / non-distributed callers should
    NOT use this (no MPI) — the local manifest already IS the global top-k.
    """
    import jax.numpy as jnp
    from legoesm.parallel.reductions import allgather_mpi

    scores, gids, valid = _local_candidate_arrays(
        local_manifest, local_to_global, n_worst)
    g_scores = np.asarray(allgather_mpi(jnp.asarray(scores))).reshape(-1)
    g_ids = np.asarray(allgather_mpi(jnp.asarray(gids))).reshape(-1)
    g_valid = np.asarray(allgather_mpi(jnp.asarray(valid))).reshape(-1)
    return owned_subset_of_global_top_k(
        g_scores, g_ids, g_valid, local_manifest, local_to_global, n_worst)
