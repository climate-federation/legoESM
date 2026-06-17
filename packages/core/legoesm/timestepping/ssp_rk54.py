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
from legoesm.timestepping.scan_loop import integrate_scan_generic

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
    """Return the float dtype matching the state's precision policy.

    Walks pytree leaves until it finds an inexact (float / complex)
    leaf, then returns the *real* dtype with matching precision —
    ``complex128 → float64``, ``complex64 → float32``.  This keeps
    scalar coefficients (``dt``, RK weights) real-valued so multiplying
    them against a real-tracer leaf does not promote the result to
    complex.  ``complex_state + real_dt * complex_tendency`` still
    yields ``complex_state`` via JAX's natural promotion, so the
    spectral-field math is unchanged.
    """
    for leaf in jax.tree.leaves(state):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or not jnp.issubdtype(dtype, jnp.inexact):
            continue
        if jnp.issubdtype(dtype, jnp.complexfloating):
            # Map complex → real of matching precision.
            return jnp.float32 if dtype == jnp.complex64 else jnp.float64
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


# -- Generalized Shu-Osher coefficient matrices (rows i = stage 1..5,
#    cols k = source stage 0..4).  Row 0 unused (u^0 = state).  These encode
#    EXACTLY the same scheme as ``ssp_rk54_step`` above:
#        u^i = sum_k A[i,k] u^k + dt * sum_k B[i,k] F(u^k)
# Used by the scan-folded variant so XLA compiles the (gather-heavy) tendency
# ONCE instead of inlining it five times.  See ``ssp_rk54_step_scan``.
_SHU_OSHER_A = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (_a20, _a21, 0.0, 0.0, 0.0),
    (_a30, 0.0, _a32, 0.0, 0.0),
    (_a40, 0.0, 0.0, _a43, 0.0),
    (0.0, 0.0, _a52, _a53, _a54),
)
_SHU_OSHER_B = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (_b10, 0.0, 0.0, 0.0, 0.0),
    (0.0, _b21, 0.0, 0.0, 0.0),
    (0.0, 0.0, _b32, 0.0, 0.0),
    (0.0, 0.0, 0.0, _b43, 0.0),
    (0.0, 0.0, 0.0, _b53, _b54),
)


def ssp_rk54_step_scan(
    state: State,
    tendency_fn: Callable[[State], State],
    dt: float,
) -> State:
    """SSP-RK(5,4) step, scan-folded so the tendency compiles ONCE.

    Numerically identical scheme to :func:`ssp_rk54_step` (same Spiteri-Ruuth
    Shu-Osher coefficients), but the five stages are evaluated inside a
    ``jax.lax.scan`` rather than inlined.  The inlined form forces XLA to emit
    five separate copies of ``tendency_fn``; for a gather-heavy unstructured
    (MPAS/TRiSK) tendency the resulting graph crosses an XLA-CPU op-count
    threshold that de-vectorizes the gathers, making the inlined step ~8x
    slower than its nominal 5-evaluation cost.  Compiling the stage body once
    avoids that blowup while preserving the scheme's stability region.

    The stage states ``u^0..u^5`` and tendencies ``F(u^0)..F(u^4)`` are carried
    as leading-axis stacks; stage ``i`` reads only ``k < i`` slots (the upper
    Shu-Osher coefficients are zero), so the constant-initialised unused slots
    never contaminate the result.

    Because the per-stage update is summed over the stage axis (vs the explicit
    two-term combinations of the inlined form), floating-point rounding differs
    at the ~1e-9 relative level — far below the scheme's truncation error.
    """
    dtype = _state_inexact_dtype(state)
    dt_t = _cast_scalar(dt, dtype)
    A = jnp.asarray(_SHU_OSHER_A)
    B = jnp.asarray(_SHU_OSHER_B)
    if dtype is not None:
        A = A.astype(dtype)
        B = B.astype(dtype)

    leaves0, treedef = jax.tree.flatten(state)
    # Promote every dynamic leaf to a concrete array and require inexact
    # (float/complex) dtype.  The Shu-Osher carry slots are allocated at a
    # FIXED dtype per leaf, so — unlike the inlined form's per-op promotion —
    # an integer leaf would be silently advanced in integer arithmetic.  SSP
    # RK on a non-inexact prognostic field is meaningless, so reject it loudly
    # rather than degrade (also handles Python-scalar leaves via asarray).
    arr0 = [jnp.asarray(l) for l in leaves0]
    for a in arr0:
        if not jnp.issubdtype(a.dtype, jnp.inexact):
            raise TypeError(
                "ssp_rk54_step_scan requires inexact (float/complex) state "
                f"leaves; got dtype {a.dtype}.  Cast the state to floating "
                "point before integrating.")
    # Each carry slot is allocated at the dtype the inlined form's per-leaf
    # update PRODUCES — ``result_type(leaf, scalar)`` — NOT the bare leaf dtype.
    # The inline step computes ``a*u + dt*b*F`` with the RK scalars (``A``/``B``
    # /``dt``) cast to the first inexact leaf's precision ``dtype``; under
    # standard promotion that upcasts a leaf narrower than ``dtype`` (e.g. an
    # f32 tracer in an otherwise-f64 MPAS state -> f64).  Allocating the carry
    # at the bare leaf dtype instead would force the f64 stage combination back
    # into an f32 slot on every ``.set`` — silently diverging from the inline
    # scheme AND riding a deprecated narrowing cast that a future JAX turns into
    # a hard error.  Promoting the slot to ``result_type(leaf, dtype)`` matches
    # the inline output exactly (verified leaf-for-leaf) and keeps a real+complex
    # mix at one precision complex (real scalars never narrow a complex leaf).
    acc_dtypes = [jnp.result_type(a.dtype, dtype) if dtype is not None
                  else a.dtype for a in arr0]
    # U[leaf]: stack (6, *leaf) of stage states u^0..u^5; F[leaf]: (5, *leaf).
    # u^0 := state; stages 1..5 overwritten before they are ever read.
    U = [jnp.broadcast_to(a.astype(acc), (6,) + a.shape)
         for a, acc in zip(arr0, acc_dtypes)]
    F = [jnp.zeros((5,) + a.shape, acc) for a, acc in zip(arr0, acc_dtypes)]

    def body(carry, i):
        U, F = carry
        prev = jax.tree.unflatten(treedef, [u[i - 1] for u in U])
        Fi, Fi_def = jax.tree.flatten(tendency_fn(prev))
        # Tendency must share the state's pytree structure, else the leaf
        # zip below would mis-align fields (e.g. a None/non-None mismatch).
        # The inlined form's tree.map raises on this; preserve that contract.
        if Fi_def != treedef:
            raise ValueError(
                "ssp_rk54_step_scan: tendency_fn output structure does not "
                f"match the state structure.\n  state:    {treedef}\n  "
                f"tendency: {Fi_def}")
        # The F carry slot is promoted to ``result_type(state_leaf, dt)`` (see
        # ``acc_dtypes`` above), which already absorbs a tendency at or below
        # the state's precision.  A tendency leaf WIDER than that promoted slot
        # (e.g. an f64 tendency for an all-f32 state) would still be narrowed by
        # the scatter below — silently diverging from the inline form (which
        # widens the whole update to f64) and riding the deprecated narrowing
        # cast.  That case cannot be sized away without a pre-trace of the
        # tendency, so reject it loudly, like the integer-leaf guard above
        # (dtypes are static, so this fires at trace time, never per-step).
        # Pass ``fi`` itself (not ``fi.dtype``) to ``result_type``: a tendency
        # leaf may be a weakly-typed Python scalar with no ``.dtype`` — the
        # inline form and the ``.set`` below both accept it — and a weak scalar
        # never forces a narrowing, so it correctly passes the guard.
        for f, fi in zip(F, Fi):
            if jnp.result_type(fi, f.dtype) != f.dtype:
                raise TypeError(
                    "ssp_rk54_step_scan: tendency leaf dtype "
                    f"{jnp.result_type(fi)} exceeds the state's compute "
                    f"precision ({f.dtype}); the fixed-dtype scan carry would "
                    "narrow it.  Return the tendency at the state's precision.")
        F = [f.at[i - 1].set(fi) for f, fi in zip(F, Fi)]
        a_row = A[i]               # (5,) weights over stages k = 0..4
        b_row = B[i]
        U_new = []
        for u, f in zip(U, F):
            comb = (jnp.tensordot(a_row, u[:5], axes=(0, 0))
                    + dt_t * jnp.tensordot(b_row, f, axes=(0, 0)))
            U_new.append(u.at[i].set(comb))
        return (U_new, F), None

    (U, _), _ = jax.lax.scan(body, (U, F), jnp.arange(1, 6))
    return jax.tree.unflatten(treedef, [u[5] for u in U])


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
    return integrate_scan_generic(
        state,
        lambda s: ssp_rk54_step(s, tendency_fn, dt),
        n_steps,
        checkpoint_interval,
        return_trajectory,
    )
