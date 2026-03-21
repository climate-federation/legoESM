"""Strong Stability Preserving Runge-Kutta 3rd order (SSP-RK3) time integrator.

The SSP-RK3 scheme (Shu & Osher 1988) is a 3-stage explicit method that
preserves strong stability properties (e.g., TVD, positivity) of the
forward Euler method. It is the workhorse for hyperbolic conservation laws.

The scheme:
    k1 = state + dt * F(state)
    k2 = 3/4 * state + 1/4 * (k1 + dt * F(k1))
    k3 = 1/3 * state + 2/3 * (k2 + dt * F(k2))

where F(state) computes tendencies (time derivatives).

References
----------
- Shu, C.-W., & Osher, S. (1988). Efficient Implementation of Essentially
  Non-oscillatory Shock-Capturing Schemes. J. Comp. Phys.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import jax

from legoesm.timestepping.pytree_ops import pytree_axpy as _pytree_axpy
from legoesm.timestepping.pytree_ops import pytree_linear_combination as _pytree_linear_combination

State = TypeVar("State")


def ssp_rk3_step(
    state: State,
    tendency_fn: Callable[[State], State],
    dt: float,
) -> State:
    """Perform one SSP-RK3 time step.

    Parameters
    ----------
    state : pytree
        The current model state (any JAX pytree: NamedTuple, dataclass, etc.).
    tendency_fn : callable
        Function that computes tendencies: tendency_fn(state) -> tendencies.
        The tendencies must be the same pytree structure as state.
    dt : float
        Time step size [seconds].

    Returns
    -------
    state : pytree
        The state advanced by one time step.
    """
    # Stage 1: k1 = state + dt * F(state)
    tend_0 = tendency_fn(state)
    k1 = _pytree_axpy(state, tend_0, dt)

    # Stage 2: k2 = 3/4 * state + 1/4 * (k1 + dt * F(k1))
    tend_1 = tendency_fn(k1)
    k1_plus_dt_tend1 = _pytree_axpy(k1, tend_1, dt)
    k2 = _pytree_linear_combination(state, k1_plus_dt_tend1, 0.75, 0.25)

    # Stage 3: k3 = 1/3 * state + 2/3 * (k2 + dt * F(k2))
    tend_2 = tendency_fn(k2)
    k2_plus_dt_tend2 = _pytree_axpy(k2, tend_2, dt)
    k3 = _pytree_linear_combination(state, k2_plus_dt_tend2, 1.0 / 3.0, 2.0 / 3.0)

    return k3


def integrate_scan(
    state: State,
    tendency_fn: Callable[[State], State],
    n_steps: int,
    dt: float,
    checkpoint_interval: int = 0,
    return_trajectory: bool = True,
) -> tuple[State, State | None]:
    """Integrate forward in time using jax.lax.scan with SSP-RK3.

    This is the primary integration method for legoESM. Using lax.scan
    ensures efficient compilation and enables automatic differentiation
    through the full time integration via jax.grad.

    Parameters
    ----------
    state : pytree
        Initial state.
    tendency_fn : callable
        Tendency function: tendency_fn(state) -> tendencies.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size [seconds].
    checkpoint_interval : int
        If > 0, wrap each scan step with jax.checkpoint to trade
        recomputation for reduced memory during backpropagation.
        0 = no checkpointing (full trajectory stored in memory).
    return_trajectory : bool
        If True (default), return all intermediate states with each leaf
        having shape (n_steps, ...). If False, return only the final state
        and None for the trajectory — essential for long production runs
        where storing O(n_steps) states is infeasible.

    Returns
    -------
    final_state : pytree
        State after n_steps.
    trajectory : pytree or None
        All intermediate states if return_trajectory=True, else None.
    """
    if return_trajectory:
        step_fn = lambda state, _: _scan_step(state, tendency_fn, dt)
    else:
        step_fn = lambda state, _: _scan_step_no_output(state, tendency_fn, dt)

    if checkpoint_interval > 0:
        step_fn = jax.checkpoint(step_fn)

    final_state, trajectory = jax.lax.scan(
        step_fn, state, xs=None, length=n_steps
    )
    return final_state, trajectory


def _scan_step(state, tendency_fn, dt):
    """Single step for use inside jax.lax.scan (stores output)."""
    new_state = ssp_rk3_step(state, tendency_fn, dt)
    return new_state, new_state


def _scan_step_no_output(state, tendency_fn, dt):
    """Single step for use inside jax.lax.scan (no output stacking)."""
    new_state = ssp_rk3_step(state, tendency_fn, dt)
    return new_state, None
