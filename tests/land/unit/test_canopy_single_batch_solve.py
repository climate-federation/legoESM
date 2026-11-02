"""The two-leaf canopy Newton solve runs every column in one batch.

On GPU every iteration of the batched while_loop was profiled to read its
predicate back to the host, so one wide batch beats column groups (measured
1.77x on the AMIP production deck).  Gradients through the solve are covered
by test_canopy_solver_grad.py.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax

from tests.land.unit.test_canopy_warm_start import _canopy_call

jax.config.update("jax_enable_x64", True)

_NCOL = 131      # odd and > 64, so a grouped solve could not carry it


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


def test_newton_iterations_run_over_all_columns_at_once():
    sizes = _while_batch_sizes(_NCOL)
    assert _NCOL in sizes, sizes
