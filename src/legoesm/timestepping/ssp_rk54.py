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

from legoesm.timestepping.pytree_ops import pytree_axpy as _pytree_axpy
from legoesm.timestepping.pytree_ops import pytree_linear_combination as _pytree_linear_combination

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


def _state_inexact_dtype(state: State):
    """Return first inexact (float/complex) dtype found in a pytree state."""
    for leaf in jax.tree.leaves(state):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            return dtype
    return None


def _cast_scalar(value, dtype):
    """Cast scalar constants to the model state's inexact dtype when available."""
    if dtype is None:
        return value
    return jnp.asarray(value, dtype=dtype)


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
    # Keep scalar coefficients in the same floating precision as the state.
    # This makes x32 behavior explicit and avoids backend-dependent weak-scalar
    # promotion subtleties in long integrations.
    dtype = _state_inexact_dtype(state)
    dt_t = _cast_scalar(dt, dtype)
    a20 = _cast_scalar(_a20, dtype)
    a21 = _cast_scalar(_a21, dtype)
    a30 = _cast_scalar(_a30, dtype)
    a32 = _cast_scalar(_a32, dtype)
    a40 = _cast_scalar(_a40, dtype)
    a43 = _cast_scalar(_a43, dtype)
    a52 = _cast_scalar(_a52, dtype)
    a53 = _cast_scalar(_a53, dtype)
    a54 = _cast_scalar(_a54, dtype)
    b10 = _cast_scalar(_b10, dtype)
    b21 = _cast_scalar(_b21, dtype)
    b32 = _cast_scalar(_b32, dtype)
    b43 = _cast_scalar(_b43, dtype)
    b53 = _cast_scalar(_b53, dtype)
    b54 = _cast_scalar(_b54, dtype)

    # Stage 1: u1 = u0 + b10 * dt * F(u0)
    F0 = tendency_fn(state)
    u1 = _pytree_axpy(state, F0, b10 * dt_t)

    # Stage 2: u2 = a20*u0 + a21*u1 + b21*dt*F(u1)
    F1 = tendency_fn(u1)
    u2 = _pytree_linear_combination(state, u1, a20, a21)
    u2 = _pytree_axpy(u2, F1, b21 * dt_t)

    # Stage 3: u3 = a30*u0 + a32*u2 + b32*dt*F(u2)
    F2 = tendency_fn(u2)
    u3 = _pytree_linear_combination(state, u2, a30, a32)
    u3 = _pytree_axpy(u3, F2, b32 * dt_t)

    # Stage 4: u4 = a40*u0 + a43*u3 + b43*dt*F(u3)
    F3 = tendency_fn(u3)
    u4 = _pytree_linear_combination(state, u3, a40, a43)
    u4 = _pytree_axpy(u4, F3, b43 * dt_t)

    # Stage 5: u5 = a52*u2 + a53*u3 + b53*dt*F(u3) + a54*u4 + b54*dt*F(u4)
    F4 = tendency_fn(u4)
    u5 = jax.tree.map(
        lambda s2, s3, s4, f3, f4: (
            a52 * s2
            + a53 * s3 + b53 * dt_t * f3
            + a54 * s4 + b54 * dt_t * f4
        ),
        u2, u3, u4, F3, F4,
    )

    return u5


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
