"""The two-leaf canopy Newton solve runs in column groups, not one batch.

Under a single vmap every column iterates until the slowest column of the rank
stops, so one column at the iteration cap made every land column pay 60
iterations (measured: the slowest rank's land step dominated the post-physics
wait of every rank).  Grouping must leave the per-column answer unchanged.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import legoesm.land.surface_scheme.two_leaf_canopy as tlc
from tests.land.unit.test_canopy_warm_start import _canopy_call

jax.config.update("jax_enable_x64", True)

_NCOL = 2 * tlc._CANOPY_SOLVE_GROUP + 3      # two full groups plus a remainder


def _while_batch_sizes(ncol):
    jaxpr = jax.make_jaxpr(lambda: _canopy_call(ncol=ncol))()
    sizes = set()

    def walk(jp):
        for eqn in jp.eqns:
            if eqn.primitive.name == "while":
                sizes.update(v.aval.shape[0] for v in eqn.invars
                             if getattr(v.aval, "shape", ()) and v.aval.ndim >= 1)
            for p in eqn.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(sub, "jaxpr"):
                        walk(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub)
                    elif hasattr(sub, "eqns"):
                        walk(sub)
    walk(jaxpr.jaxpr)
    return sizes


def test_newton_iterations_run_per_group_not_per_rank(monkeypatch):
    # An odd group size no other array in the canopy call has, so finding it on
    # a while loop can only mean the grouped Newton loop.
    monkeypatch.setattr(tlc, "_CANOPY_SOLVE_GROUP", 11)
    sizes = _while_batch_sizes(25)
    assert 11 in sizes and 3 in sizes          # two groups of 11 plus a remainder of 3
    assert 25 not in sizes


def test_grouping_does_not_change_the_answer(monkeypatch):
    grouped = _canopy_call(ncol=_NCOL)
    monkeypatch.setattr(tlc, "_CANOPY_SOLVE_GROUP", _NCOL)   # one group holding every column
    single = _canopy_call(ncol=_NCOL)
    np.testing.assert_array_equal(np.asarray(grouped.converged), np.asarray(single.converged))
    for name in ("shflx", "lhflx", "T_surface", "G_soil", "gpp"):
        np.testing.assert_allclose(np.asarray(getattr(grouped, name)),
                                   np.asarray(getattr(single, name)), rtol=0, atol=1e-9)


def test_later_passes_group_hard_columns_together(monkeypatch):
    """Picard passes after the first order their columns by the previous
    pass's Newton iterations, so the hard columns share groups instead of
    making every group run to the slowest column's count."""
    calls = []
    orig = jax.lax.map

    def rec(f, xs, **kw):
        out = orig(f, xs, **kw)
        calls.append((np.asarray(xs[0]), np.asarray(out[0]), np.asarray(out[1]),
                      np.asarray(out[2])))
        return out

    monkeypatch.setattr(jax.lax, "map", rec)
    _canopy_call(ncol=_NCOL)
    assert len(calls) >= 2
    (_, x_prev, n_prev, conv_prev), (seed_next, _, _, _) = calls[0], calls[1]
    assert len(set(n_prev.tolist())) > 1, "uniform iteration counts: nothing to order"
    # Map each second-pass row back to the first pass by its seed (a converged
    # column is reseeded with its own solution).
    key = {tuple(x): i for i, x in enumerate(x_prev) if conv_prev[i]}
    assert len(key) == int(conv_prev.sum()), "two columns share a root: seeds cannot be mapped back"
    order = [key[tuple(s)] for s in seed_next if tuple(s) in key]
    assert len(order) > _NCOL // 2
    assert np.any(np.diff(n_prev[sorted(order)]) > 0), "already ordered: test cannot fail"
    assert np.all(np.diff(n_prev[order]) <= 0), (
        "second pass is not ordered by the first pass's iteration counts")


def test_sorting_does_not_change_the_answer_or_its_gradient(monkeypatch):
    """Ordering the later Picard passes changes only which columns share a
    group.  Eagerly every output leaf matches the unsorted solve bit for bit.
    The gradient is compiled, and a permuted batch compiles to a different
    rounding: measured max relative difference 2.5e-6 (a dropped un-permutation
    gives O(1)), so it is compared at rtol 1e-5."""
    orig = jax.lax.map
    w = jnp.arange(1, _NCOL + 1, dtype=jnp.float64)   # unequal weights: a misrouted column shows

    def run(sort):
        monkeypatch.setattr(tlc, "_SORT_BY_PREV_ITERS", sort)
        seeds = []

        def rec(f, xs, **kw):
            seeds.append(np.asarray(xs[0]))
            return orig(f, xs, **kw)

        monkeypatch.setattr(jax.lax, "map", rec)
        out = _canopy_call(ncol=_NCOL)
        monkeypatch.setattr(jax.lax, "map", orig)
        g = jax.jit(jax.grad(lambda s: jnp.sum(w * _canopy_call(ncol=_NCOL, soil_offset=s).shflx)))(
            jnp.zeros(_NCOL, jnp.float64))
        return out, np.asarray(g), seeds

    on, g_on, s_on = run(True)
    off, g_off, s_off = run(False)
    # The two arms really differ: same first pass, same second-pass columns,
    # different second-pass order.
    np.testing.assert_array_equal(s_on[0], s_off[0])
    assert sorted(map(tuple, s_on[1].tolist())) == sorted(map(tuple, s_off[1].tolist()))
    assert not np.array_equal(s_on[1], s_off[1]), "sorting was a no-op: test cannot fail"
    assert np.asarray(on.shflx).dtype == np.float64          # fluxes solve in float64 under x64
    leaves_on, leaves_off = jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(off)
    assert len(leaves_on) == len(leaves_off) > 5
    for a, b in zip(leaves_on, leaves_off):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert np.all(np.isfinite(g_on)) and np.count_nonzero(g_on) > _NCOL // 2
    np.testing.assert_allclose(g_on, g_off, rtol=1e-5, atol=0)
