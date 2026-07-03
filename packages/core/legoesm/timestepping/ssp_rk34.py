"""Strong Stability Preserving Runge-Kutta (4,3) time integrator.

Implements the 4-stage, 3rd-order SSPRK(4,3) scheme in Shu-Osher form.
This method has a larger SSP CFL coefficient than SSP-RK3 while keeping
third-order accuracy.

Scheme:
    u1 = u0 + 1/2 dt F(u0)
    u2 = u1 + 1/2 dt F(u1)
    u3 = 2/3 u0 + 1/3 u2 + 1/6 dt F(u2)
    u4 = u3 + 1/2 dt F(u3)

References
----------
- Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). Strong Stability-Preserving
  High-Order Time Discretization Methods. SIAM Review.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import jax

from legoesm.timestepping.pytree_ops import pytree_axpy as _pytree_axpy
from legoesm.timestepping.pytree_ops import pytree_linear_combination as _pytree_linear_combination
from legoesm.timestepping.scan_loop import integrate_scan_generic

State = TypeVar("State")


def ssp_rk34_step(
    state: State,
    tendency_fn: Callable[[State], State],
    dt: float,
) -> State:
    """Perform one SSP-RK(4,3) time step."""
    # Stage 1
    f0 = tendency_fn(state)
    u1 = _pytree_axpy(state, f0, 0.5 * dt)

    # Stage 2
    f1 = tendency_fn(u1)
    u2 = _pytree_axpy(u1, f1, 0.5 * dt)

    # Stage 3
    f2 = tendency_fn(u2)
    u3 = _pytree_linear_combination(state, u2, 2.0 / 3.0, 1.0 / 3.0)
    u3 = _pytree_axpy(u3, f2, (1.0 / 6.0) * dt)

    # Stage 4
    f3 = tendency_fn(u3)
    u4 = _pytree_axpy(u3, f3, 0.5 * dt)

    return u4


def integrate_scan(
    state: State,
    tendency_fn: Callable[[State], State],
    n_steps: int,
    dt: float,
    checkpoint_interval: int = 0,
    return_trajectory: bool = True,
) -> tuple[State, State | None]:
    """Integrate forward in time using jax.lax.scan with SSP-RK(4,3)."""
    return integrate_scan_generic(
        state,
        lambda s: ssp_rk34_step(s, tendency_fn, dt),
        n_steps,
        checkpoint_interval,
        return_trajectory,
    )

