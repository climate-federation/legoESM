"""Time-mean accumulation of :class:`ColumnState` for climatological comparison.

AMIP/CMIP-vs-reanalysis comparison (``docs/COMPARE_REANALYSIS.md`` §1) is a
**climatology** comparison: a monthly/seasonal time mean of the model state
against the ERA5 climatology — not an instantaneous snapshot, which is dominated
by synoptic noise and would mis-rank the worst columns.  This accumulates a
:class:`~legoesm.training.compare_reanalysis.ColumnState` over a run into its
time mean, which then feeds :func:`compare_state_to_reference` as ``model``.

The accumulator is a clean pytree carry (running field sums + a sample count), so
it works either as a ``lax.scan`` carry over the integration *or* by folding a
list of restart snapshots (:func:`time_mean_column_states`).  ``jax.tree``
handles the optional ``precip_mm_day`` / ``sst_K`` / ``u_edge`` fields
automatically — a ``None`` field is an empty pytree subtree, so it is preserved
(never summed), and accumulating a state whose optional fields differ in presence
raises a structure mismatch (loud), not a silent drop.  The optional ``u_edge``
(native MPAS edge velocity, ``nEdges`` cardinality — not a per-column field) is
summed/meaned like any present leaf; this is CONSISTENT with the meaned cell
``u``/``v`` because the Perot reconstruction is linear
(``mean(reconstruct(u_edge)) == reconstruct(mean(u_edge))``).  Pure-JAX and
differentiable w.r.t. the accumulated states.

Precision: a plain running SUM is exact in float64 (the model's ``x64`` mode) for
the sample counts of a climatology, but in float32 a long series risks
accumulation error; run the accumulation in float64 (or switch to a Welford mean)
if time-meaning a float32 state over very many samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.training.compare_reanalysis import ColumnState


class ColumnStateAccumulator(NamedTuple):
    """Running field sums + sample count for a :class:`ColumnState` time mean."""

    sum_state: ColumnState   # per-field running sums (None where the field is absent)
    count: jax.Array         # scalar sample count


def init_column_accumulator(template: ColumnState) -> ColumnStateAccumulator:
    """Zero accumulator shaped like ``template`` (same present/absent fields).

    ``count`` starts at 0; the optional ``precip_mm_day`` / ``sst_K`` fields stay
    ``None`` if absent in ``template`` (a ``None`` is an empty pytree subtree, so
    ``zeros_like`` is never applied to it).
    """
    sum_state = jax.tree.map(jnp.zeros_like, template)
    return ColumnStateAccumulator(sum_state=sum_state, count=jnp.asarray(0, dtype=jnp.int32))


def accumulate_column_state(
    acc: ColumnStateAccumulator, state: ColumnState
) -> ColumnStateAccumulator:
    """Add one sample to the accumulator (``lax.scan``-friendly).

    ``state`` MUST have the same present/absent optional fields as the
    accumulator's ``sum_state``.  This is checked EXPLICITLY (a treedef compare)
    so it fails the same deterministic way in both directions — a sample that
    suddenly carries ``sst_K`` when the accumulator does not (``ValueError:
    Expected None, got Array``) *and* the reverse (which would otherwise fail
    obscurely as ``Array + None`` inside the map) — because a field that
    appears/disappears mid-run is a bug, not something to average over.  The
    compare is on static pytree structure, so it is ``jax.jit`` / ``lax.scan``
    safe (evaluated once at trace time).
    """
    acc_struct = jax.tree.structure(acc.sum_state)
    state_struct = jax.tree.structure(state)
    if acc_struct != state_struct:
        raise ValueError(
            "accumulate_column_state: ColumnState structure mismatch — the "
            "sample's present/absent optional fields differ from the "
            f"accumulator ({state_struct} vs {acc_struct})."
        )
    new_sum = jax.tree.map(lambda s, x: s + x, acc.sum_state, state)
    return ColumnStateAccumulator(sum_state=new_sum, count=acc.count + 1)


def mean_column_state(acc: ColumnStateAccumulator) -> ColumnState:
    """Finalize the time mean ``Σ state / count``.

    ``count`` must be ``>= 1``; the division is guarded with ``maximum(count, 1)``
    so an *empty* accumulator returns the zero template rather than ``0/0=NaN``
    (the host-side :func:`time_mean_column_states` rejects the empty case loudly —
    prefer it when the sample count is known; this guard only protects a
    ``lax.scan`` carry that was never advanced).
    """
    denom = jnp.maximum(acc.count, 1)
    return jax.tree.map(lambda s: s / denom.astype(s.dtype), acc.sum_state)


def time_mean_column_states(states: Sequence[ColumnState]) -> ColumnState:
    """Time mean of a non-empty sequence of column states (host-side fold).

    Convenience for the restart-snapshot case (e.g. monthly means from saved
    AMIP restarts); for an in-line integration use the accumulator as a
    ``lax.scan`` carry instead.  Raises on an empty sequence.
    """
    if len(states) == 0:
        raise ValueError("time_mean_column_states: need at least one ColumnState.")
    acc = init_column_accumulator(states[0])
    for state in states:
        acc = accumulate_column_state(acc, state)
    return mean_column_state(acc)
