"""Batching of the two-leaf canopy Newton solve (CanopyConfig.solve_group).

A batch iterates until its slowest column stops.  0 (production, user
2026-11-07) solves every column of the call in one batch; a positive value
splits the call into batches of that many columns (bounds memory).  The
per-column answer must not depend on the batching.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.config import CanopyConfig
from tests.land.unit.test_canopy_warm_start import _canopy_call

jax.config.update("jax_enable_x64", True)

_NCOL = 25


def _while_batch_sizes(ncol, solve_group):
    jaxpr = jax.make_jaxpr(lambda: _canopy_call(ncol=ncol, solve_group=solve_group))()
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


def test_default_is_one_batch():
    assert CanopyConfig().solve_group == 0


@pytest.mark.parametrize("bad", [-1, 2.0])
def test_solve_group_must_be_nonnegative_int(bad):
    with pytest.raises(ValueError, match="solve_group"):
        CanopyConfig(solve_group=bad).validate()


def test_newton_loop_batch_follows_solve_group():
    # 11 is a size no other array in the canopy call has, so finding it on a
    # while loop can only mean the batched Newton loop.
    grouped = _while_batch_sizes(_NCOL, 11)
    assert 11 in grouped and 3 in grouped        # two batches of 11 plus a remainder of 3
    assert _NCOL not in grouped
    single = _while_batch_sizes(_NCOL, 0)
    assert _NCOL in single and 11 not in single
    larger = _while_batch_sizes(_NCOL, 40)       # a batch larger than the call: one batch
    assert _NCOL in larger and 40 not in larger


def test_batching_does_not_change_the_answer_or_its_gradient():
    """One vmap and lax.map batches compile to different roundings of the same
    per-column solve: measured max relative difference 5.7e-10 in shflx (the
    canopy-call bench, 7403 columns: 2e-10).  A misrouted column gives O(1)."""
    grouped = _canopy_call(ncol=_NCOL, solve_group=11)
    single = _canopy_call(ncol=_NCOL, solve_group=0)
    np.testing.assert_array_equal(np.asarray(grouped.converged), np.asarray(single.converged))
    for name in ("shflx", "lhflx", "T_surface", "G_soil", "gpp"):
        np.testing.assert_allclose(np.asarray(getattr(grouped, name)),
                                   np.asarray(getattr(single, name)), rtol=1e-8, atol=0)
    w = jnp.arange(1, _NCOL + 1, dtype=jnp.float64)   # unequal weights: a misrouted column shows

    def grad(group):
        return np.asarray(jax.jit(jax.grad(lambda s: jnp.sum(w * _canopy_call(
            ncol=_NCOL, soil_offset=s, solve_group=group).shflx)))(
                jnp.zeros(_NCOL, jnp.float64)))
    g_single, g_grouped = grad(0), grad(11)
    assert np.all(np.isfinite(g_single)) and np.count_nonzero(g_single) > _NCOL // 2
    np.testing.assert_allclose(g_grouped, g_single, rtol=1e-5, atol=0)
