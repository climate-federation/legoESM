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
