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
from legoesm.timestepping.scan_loop import integrate_scan_generic

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


def ssp_rk3_step_scan(
    state: State,
    tendency_fn: Callable[[State], State],
    dt: float,
) -> State:
    """SSP-RK3 step with the 3 stages folded into a single ``lax.scan`` body.

    Task #25 (JIT-compile bloat at multi-rank): the inline variant
    above calls ``tendency_fn`` 3× sequentially, so XLA inlines THREE
    copies of the full tendency pipeline (advection + polar filter +
    diffusion + hydrostatic + vertical advection) into one XLA module.
    With ``jax.lax.scan`` over a 3-iteration body XLA optimizes the
    tendency code ONCE and loops 3×.  The math is identical to
    ``ssp_rk3_step`` — numerically equivalent to ~1e-9 relative, not
    bit-exact: XLA may fuse/associate the single compiled scan body's
    FMAs differently from the three unrolled stages, and that choice is
    JAX/XLA-version dependent.  (Pinned by
    ``tests/timestepping/test_ssp_rk3_scan_bit_equivalence.py``.)

    Stage update arithmetic:

        k_{n+1} = α_n * state_init + β_n * (k_n + dt * F(k_n))

    with α = (0, 0.75, 1/3), β = (1, 0.25, 2/3).  Pass coefficients
    inside the scan-carry as JAX arrays so the body is fully
    closed-form (no Python-side branching).

    Parameters
    ----------
    state, tendency_fn, dt
        Same as :func:`ssp_rk3_step`.

    Returns
    -------
    Final state pytree, same structure as ``state``.
    """
    import jax.numpy as jnp

    # α_n weights ``state_init`` (kept as scan carry); β_n weights the
    # axpy ``(k_n + dt * F(k_n))``.  Build as Float arrays so they're
    # part of the scan's xs (the only thing that differs per stage).
    alpha = jnp.asarray([0.0, 0.75, 1.0 / 3.0])
    beta = jnp.asarray([1.0, 0.25, 2.0 / 3.0])

    def scan_body(carry, stage_idx):
        k_curr, state_init = carry
        tend = tendency_fn(k_curr)
        k_axpy = _pytree_axpy(k_curr, tend, dt)
        a = alpha[stage_idx]
        b = beta[stage_idx]
        # Smoke 8070583 surfaced: ``jnp.asarray([…])`` produces a
        # STRONGLY-typed float64 array.  Indexing it gives a float64
        # scalar; multiplying with a float32 state leaf upcasts to
        # float64.  ``jax.lax.scan`` then refuses to close because the
        # scan body's carry-in (float32) and carry-out (float64) types
        # do not match.  Fix: cast ``a`` and ``b`` to each leaf's own
        # dtype inside ``jax.tree.map`` so the linear combination
        # preserves the leaf dtype — same arithmetic as
        # ``_pytree_linear_combination`` but dtype-stable for mixed-
        # precision pytrees.
        def _comb(si, ki):
            # Cast BOTH the RK coefficients AND the tendency leaf ``ki`` to the
            # STATE leaf dtype.  The hydrostatic lat-lon C-grid dycore emits a
            # float64 ``p_s`` tendency against a float32 state; without casting
            # ``ki``, ``b_typed(f32) * ki(f64)`` upcasts the stage to float64 and
            # ``jax.lax.scan`` refuses to close (carry-in float32 != carry-out
            # float64).  Casting to ``si.dtype`` keeps the combination dtype-
            # stable at the storage precision, matching the non-scan ssp_rk3
            # (which has no carry-type check) bit-for-bit on equal-dtype leaves.
            # See #835.
            a_typed = a.astype(si.dtype)
            b_typed = b.astype(si.dtype)
            return a_typed * si + b_typed * ki.astype(si.dtype)

        k_new = jax.tree.map(_comb, state_init, k_axpy)
        return (k_new, state_init), None

    (k_final, _), _ = jax.lax.scan(
        scan_body, (state, state), jnp.arange(3),
    )
    return k_final


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
    return integrate_scan_generic(
        state,
        lambda s: ssp_rk3_step(s, tendency_fn, dt),
        n_steps,
        checkpoint_interval,
        return_trajectory,
    )
