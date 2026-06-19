"""Shared ``jax.lax.scan`` time-integration loop for the SSP-RK integrators.

``ssp_rk3`` / ``ssp_rk34`` / ``ssp_rk54`` each carried a byte-identical
``integrate_scan`` + ``_scan_step`` + ``_scan_step_no_output`` triplet,
differing only in which single-step function they invoked.  This factors
the scan / checkpoint / trajectory plumbing into one helper parameterized
by the already-bound single step, so the primary differentiable
integration loop has a single source of truth.
"""
from __future__ import annotations

from typing import Callable, TypeVar

import jax

State = TypeVar("State")


def integrate_scan_generic(
    state: State,
    step: Callable[[State], State],
    n_steps: int,
    checkpoint_interval: int = 0,
    return_trajectory: bool = True,
) -> tuple[State, State | None]:
    """Run ``n_steps`` of ``step`` forward via ``jax.lax.scan``.

    Parameters
    ----------
    state : pytree
        Initial state.
    step : callable
        Full single time step with its tendency function and ``dt`` already
        bound: ``step(state) -> new_state``.
    n_steps : int
        Number of time steps.
    checkpoint_interval : int
        If > 0, wrap each scan step with :func:`jax.checkpoint` to trade
        recomputation for reduced memory during backpropagation.
    return_trajectory : bool
        If True (default), stack every intermediate state (leading
        ``n_steps`` axis); else return ``(final_state, None)``.

    Returns
    -------
    final_state : pytree
        State after ``n_steps``.
    trajectory : pytree or None
        All intermediate states if ``return_trajectory`` else ``None``.
    """
    if return_trajectory:
        def scan_fn(s, _):
            new_state = step(s)
            return new_state, new_state
    else:
        def scan_fn(s, _):
            new_state = step(s)
            return new_state, None

    if checkpoint_interval > 0:
        scan_fn = jax.checkpoint(scan_fn)

    return jax.lax.scan(scan_fn, state, xs=None, length=n_steps)
