"""Strong Stability Preserving Runge-Kutta (5,4) time integrator.

The optimal 5-stage, 4th-order SSP Runge-Kutta method of Spiteri & Ruuth
(2002).  It has SSP coefficient c = 1.508, meaning the time step can be
up to 1.508 times the forward-Euler stability limit while still
preserving strong stability (TVD / positivity) properties.

The scheme is expressed in Shu-Osher form (alpha, beta coefficients):

  Stage 1:
    u1 = u0 + 0.391752226571890 * dt * F(u0)

  Stage 2:
    u2 = 0.444370493651235 * u0
       + 0.555629506348765 * u1
       + 0.368410593050371 * dt * F(u1)

  Stage 3:
    u3 = 0.620101851488403 * u0
       + 0.379898148511597 * u2
       + 0.251891774271694 * dt * F(u2)

  Stage 4:
    u4 = 0.178079954393132 * u0
       + 0.821920045606868 * u3
       + 0.544974750228521 * dt * F(u3)

  Stage 5:
    u5 = 0.517231671970585 * u2
       + 0.096059710526147 * u3 + 0.063692468666290 * dt * F(u3)
       + 0.386708617503269 * u4 + 0.226007483236906 * dt * F(u4)

References
----------
- Spiteri, R. J., & Ruuth, S. J. (2002). A New Class of Optimal
  High-Order Strong-Stability-Preserving Time Discretization Methods.
  SIAM J. Numer. Anal., 40(2), 469-491.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import jax
import jax.numpy as jnp

State = TypeVar("State")


# -- Shu-Osher coefficients --------------------------------------------------

# alpha_ij (convex combination weights, rows sum to 1)
_a10 = 0.391752226571890

_a20 = 0.444370493651235
_a21 = 0.555629506348765

_a30 = 0.620101851488403
_a32 = 0.379898148511597

_a40 = 0.178079954393132
_a43 = 0.821920045606868

_a52 = 0.517231671970585
_a53 = 0.096059710526147
_a54 = 0.386708617503269

# beta_ij (tendency weights, beta_ij = alpha_ij * dt_eff / dt)
_b10 = 0.391752226571890

_b21 = 0.368410593050371

_b32 = 0.251891774271694

_b43 = 0.544974750228521

_b53 = 0.063692468666290
_b54 = 0.226007483236906


def ssp_rk54_step(
    state: State,
    tendency_fn: Callable[[State], State],
    dt: float,
) -> State:
    """Perform one SSP-RK(5,4) time step.

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
    # Stage 1: u1 = u0 + b10 * dt * F(u0)
    F0 = tendency_fn(state)
    u1 = _pytree_axpy(state, F0, _b10 * dt)

    # Stage 2: u2 = a20*u0 + a21*u1 + b21*dt*F(u1)
    F1 = tendency_fn(u1)
    u2 = _pytree_linear_combination(state, u1, _a20, _a21)
    u2 = _pytree_axpy(u2, F1, _b21 * dt)

    # Stage 3: u3 = a30*u0 + a32*u2 + b32*dt*F(u2)
    F2 = tendency_fn(u2)
    u3 = _pytree_linear_combination(state, u2, _a30, _a32)
    u3 = _pytree_axpy(u3, F2, _b32 * dt)

    # Stage 4: u4 = a40*u0 + a43*u3 + b43*dt*F(u3)
    F3 = tendency_fn(u3)
    u4 = _pytree_linear_combination(state, u3, _a40, _a43)
    u4 = _pytree_axpy(u4, F3, _b43 * dt)

    # Stage 5: u5 = a52*u2 + a53*u3 + b53*dt*F(u3) + a54*u4 + b54*dt*F(u4)
    F4 = tendency_fn(u4)
    u5 = jax.tree.map(
        lambda s2, s3, s4, f3, f4: (
            _a52 * s2
            + _a53 * s3 + _b53 * dt * f3
            + _a54 * s4 + _b54 * dt * f4
        ),
        u2, u3, u4, F3, F4,
    )

    return u5


def _pytree_axpy(x, y, alpha):
    """Compute x + alpha * y for two pytrees with the same structure."""
    return jax.tree.map(lambda xi, yi: xi + alpha * yi, x, y)


def _pytree_linear_combination(x, y, a, b):
    """Compute a * x + b * y for two pytrees with the same structure."""
    return jax.tree.map(lambda xi, yi: a * xi + b * yi, x, y)


def integrate_scan(
    state: State,
    tendency_fn: Callable[[State], State],
    n_steps: int,
    dt: float,
    checkpoint_interval: int = 0,
    return_trajectory: bool = True,
) -> tuple[State, State | None]:
    """Integrate forward in time using jax.lax.scan with SSP-RK(5,4).

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
    return_trajectory : bool
        If True (default), return all intermediate states. If False,
        return only the final state and None for trajectory.

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
    new_state = ssp_rk54_step(state, tendency_fn, dt)
    return new_state, new_state


def _scan_step_no_output(state, tendency_fn, dt):
    """Single step for use inside jax.lax.scan (no output stacking)."""
    new_state = ssp_rk54_step(state, tendency_fn, dt)
    return new_state, None
