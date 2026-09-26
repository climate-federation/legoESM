"""On-device minimization for 4D-Var inner loop.

Both L-BFGS and CG run entirely inside jax.lax.while_loop for
full JIT compilation — no Python-side optimizer loops.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)


class MinimizationResult(NamedTuple):
    """Result of a minimization run."""
    x: jax.Array              # Solution
    fun: jax.Array            # Final cost value
    grad_norm: jax.Array      # Final gradient norm
    n_iter: jax.Array         # Number of iterations performed
    converged: jax.Array      # Boolean: converged?
    history: jax.Array        # Cost at each iteration, shape (max_iter,)
    # L-BFGS only: True when it stopped because no step met the Armijo
    # condition (x is the last accepted point). CG accepts such steps and
    # always reports False.
    line_search_failed: jax.Array = jnp.array(False)


# ---------------------------------------------------------------------------
# Backtracking line search (Armijo condition)
# ---------------------------------------------------------------------------

def _backtracking_line_search(
    cost_and_grad_fn: Callable,
    x: jax.Array,
    f: jax.Array,
    g: jax.Array,
    d: jax.Array,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    max_backtracks: int = 20,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Backtracking line search with Armijo sufficient decrease.

    Returns (alpha, f_new, g_new, x_new, armijo_ok). ``armijo_ok`` is False
    when ``max_backtracks`` ran out before sufficient decrease was reached.
    """
    slope = jnp.sum(g * d)

    class LSState(NamedTuple):
        alpha: jax.Array
        f_new: jax.Array
        g_new: jax.Array
        x_new: jax.Array
        k: jax.Array

    def ls_cond(state):
        return (state.f_new > f + c1 * state.alpha * slope) & (state.k < max_backtracks)

    def ls_body(state):
        alpha = state.alpha * 0.5
        x_new = x + alpha * d
        f_new, g_new = cost_and_grad_fn(x_new)
        return LSState(alpha, f_new, g_new, x_new, state.k + 1)

    x_init = x + alpha_init * d
    f_init, g_init = cost_and_grad_fn(x_init)
    init_state = LSState(
        jnp.array(alpha_init),
        f_init, g_init, x_init,
        jnp.array(0),
    )

    final = jax.lax.while_loop(ls_cond, ls_body, init_state)
    armijo_ok = final.f_new <= f + c1 * final.alpha * slope
    return final.alpha, final.f_new, final.g_new, final.x_new, armijo_ok


# ---------------------------------------------------------------------------
# L-BFGS
# ---------------------------------------------------------------------------

def minimize_lbfgs(
    cost_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.Array,
    max_iter: int = 50,
    gtol: float = 1e-5,
    ftol: float = 1e-8,
    m: int = 10,
    line_search: str = "backtracking",
) -> MinimizationResult:
    """L-BFGS minimization via jax.lax.while_loop.

    The entire optimization runs on-device (XLA-compiled).

    Parameters
    ----------
    cost_and_grad_fn : callable
        Returns (cost, gradient) given control vector.
    x0 : jax.Array
        Initial guess.
    max_iter : int
        Maximum iterations.
    gtol : float
        Gradient norm tolerance.
    ftol : float
        Relative function decrease tolerance.
    m : int
        L-BFGS memory (number of correction pairs).
    line_search : str
        "backtracking" (only option currently).
    """
    if line_search != "backtracking":
        # Dispatch hardening: the loop calls _backtracking_line_search
        # unconditionally, so any other value would silently run backtracking.
        raise ValueError(
            f"unsupported line_search {line_search!r}; only 'backtracking' "
            "is implemented."
        )
    n = x0.shape[0]
    f0, g0 = cost_and_grad_fn(x0)

    # Pre-allocate L-BFGS buffers
    S = jnp.zeros((m, n))  # s_k = x_{k+1} - x_k
    Y = jnp.zeros((m, n))  # y_k = g_{k+1} - g_k
    rho = jnp.zeros(m)     # 1 / (y_k . s_k)
    history = jnp.full(max_iter, jnp.inf)
    history = history.at[0].set(f0)

    class LBFGSState(NamedTuple):
        x: jax.Array
        f: jax.Array
        g: jax.Array
        S: jax.Array
        Y: jax.Array
        rho: jax.Array
        k: jax.Array
        n_pairs: jax.Array  # correction pairs accepted so far (ring-buffer head)
        history: jax.Array
        converged: jax.Array
        f_prev: jax.Array
        stalled: jax.Array  # line search found no sufficient decrease

    init_state = LBFGSState(
        x=x0, f=f0, g=g0,
        S=S, Y=Y, rho=rho,
        k=jnp.array(0),
        n_pairs=jnp.array(0),
        history=history,
        converged=jnp.array(False),
        f_prev=jnp.array(jnp.inf),
        stalled=jnp.array(False),
    )

    def cond_fn(state):
        grad_small = jnp.linalg.norm(state.g) < gtol
        f_decrease_small = (
            jnp.abs(state.f_prev - state.f)
            / jnp.maximum(jnp.abs(state.f), 1.0)
            < ftol
        ) & (state.k > 0)
        return ((~grad_small) & (~f_decrease_small) & (state.k < max_iter)
                & (~state.converged) & (~state.stalled))

    def body_fn(state):
        # L-BFGS two-loop recursion to compute search direction
        q = state.g
        n_use = jnp.minimum(state.n_pairs, m)

        # Allocate alpha buffer for two-loop recursion
        alpha_buf = jnp.zeros(m)

        # First loop (backward)
        def first_loop_body(i, carry):
            q, alpha_buf = carry
            idx = (state.n_pairs - 1 - i) % m
            a = state.rho[idx] * jnp.sum(state.S[idx] * q)
            alpha_buf = alpha_buf.at[idx].set(a)
            q = q - a * state.Y[idx]
            # Only apply if we have enough stored pairs
            mask = i < n_use
            return (jnp.where(mask, q, carry[0]),
                    jnp.where(mask, alpha_buf, carry[1]))

        q, alpha_buf = jax.lax.fori_loop(0, m, first_loop_body, (q, alpha_buf))

        # Initial Hessian approximation: gamma * I
        # gamma = (s_{k-1} . y_{k-1}) / (y_{k-1} . y_{k-1}); with no pairs yet,
        # a unit-length first step (gamma = 1/||g||, as in L-BFGS-B) so a
        # badly scaled cost does not exhaust the backtracking on step one.
        last_idx = (state.n_pairs - 1) % m
        gamma = jnp.where(
            state.n_pairs > 0,
            jnp.sum(state.S[last_idx] * state.Y[last_idx])
            / jnp.maximum(jnp.sum(state.Y[last_idx] * state.Y[last_idx]), _TINY),
            1.0 / jnp.maximum(jnp.linalg.norm(state.g), _TINY),
        )
        r = gamma * q

        # Second loop (forward)
        def second_loop_body(i, r):
            idx = (state.n_pairs - n_use + i) % m
            beta = state.rho[idx] * jnp.sum(state.Y[idx] * r)
            update = state.S[idx] * (alpha_buf[idx] - beta)
            mask = i < n_use
            return jnp.where(mask, r + update, r)

        r = jax.lax.fori_loop(0, m, second_loop_body, r)
        d = -r

        # Line search. A step without sufficient decrease is rejected: keep
        # the current point and stop (the loop exits on ``stalled``).
        _, f_ls, g_ls, x_ls, armijo_ok = _backtracking_line_search(
            cost_and_grad_fn, state.x, state.f, state.g, d,
        )
        x_new = jnp.where(armijo_ok, x_ls, state.x)
        f_new = jnp.where(armijo_ok, f_ls, state.f)
        g_new = jnp.where(armijo_ok, g_ls, state.g)

        # Update L-BFGS buffers only with positive-curvature pairs, so the
        # inverse-Hessian approximation stays positive definite and d is a
        # descent direction.
        s_k = x_new - state.x
        y_k = g_new - state.g
        sy = jnp.sum(s_k * y_k)
        keep = armijo_ok & (sy > _TINY)
        rho_k = 1.0 / jnp.where(keep, sy, 1.0)

        store_idx = state.n_pairs % m
        S_new = jnp.where(keep, state.S.at[store_idx].set(s_k), state.S)
        Y_new = jnp.where(keep, state.Y.at[store_idx].set(y_k), state.Y)
        rho_new = jnp.where(keep, state.rho.at[store_idx].set(rho_k), state.rho)

        new_k = state.k + 1
        history_new = state.history.at[jnp.minimum(new_k, max_iter - 1)].set(f_new)

        converged = jnp.linalg.norm(g_new) < gtol

        return LBFGSState(
            x=x_new, f=f_new, g=g_new,
            S=S_new, Y=Y_new, rho=rho_new,
            k=new_k,
            n_pairs=state.n_pairs + keep.astype(state.n_pairs.dtype),
            history=history_new,
            converged=converged,
            f_prev=state.f,
            stalled=~armijo_ok,
        )

    final = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return MinimizationResult(
        x=final.x,
        fun=final.f,
        grad_norm=jnp.linalg.norm(final.g),
        n_iter=final.k,
        converged=final.converged,
        history=final.history,
        line_search_failed=final.stalled,
    )


# ---------------------------------------------------------------------------
# Conjugate Gradient (Polak-Ribière)
# ---------------------------------------------------------------------------

def minimize_cg(
    cost_and_grad_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    x0: jax.Array,
    max_iter: int = 50,
    gtol: float = 1e-5,
    preconditioner: Callable | None = None,
) -> MinimizationResult:
    """Preconditioned conjugate gradient (Polak-Ribière).

    Parameters
    ----------
    cost_and_grad_fn : callable
        Returns (cost, gradient).
    x0 : jax.Array
        Initial guess.
    max_iter : int
        Maximum iterations.
    gtol : float
        Gradient norm tolerance.
    preconditioner : callable or None
        P^{-1} @ g. If None, identity preconditioning.
    """
    f0, g0 = cost_and_grad_fn(x0)

    if preconditioner is not None:
        z0 = preconditioner(g0)
    else:
        z0 = g0

    d0 = -z0
    history = jnp.full(max_iter, jnp.inf)
    history = history.at[0].set(f0)

    class CGState(NamedTuple):
        x: jax.Array
        f: jax.Array
        g: jax.Array
        z: jax.Array
        d: jax.Array
        k: jax.Array
        history: jax.Array
        converged: jax.Array

    init_state = CGState(
        x=x0, f=f0, g=g0, z=z0, d=d0,
        k=jnp.array(0),
        history=history,
        converged=jnp.array(False),
    )

    def cond_fn(state):
        return (jnp.linalg.norm(state.g) >= gtol) & (state.k < max_iter) & (~state.converged)

    def body_fn(state):
        # Line search
        _, f_new, g_new, x_new, _ = _backtracking_line_search(
            cost_and_grad_fn, state.x, state.f, state.g, state.d,
        )

        if preconditioner is not None:
            z_new = preconditioner(g_new)
        else:
            z_new = g_new

        # Polak-Ribière beta
        beta = jnp.sum(z_new * (g_new - state.g)) / jnp.maximum(
            jnp.sum(state.z * state.g), _TINY
        )
        beta = jnp.maximum(beta, 0.0)  # Restart if beta < 0

        d_new = -z_new + beta * state.d

        new_k = state.k + 1
        history_new = state.history.at[jnp.minimum(new_k, max_iter - 1)].set(f_new)
        converged = jnp.linalg.norm(g_new) < gtol

        return CGState(
            x=x_new, f=f_new, g=g_new, z=z_new, d=d_new,
            k=new_k,
            history=history_new,
            converged=converged,
        )

    final = jax.lax.while_loop(cond_fn, body_fn, init_state)

    return MinimizationResult(
        x=final.x,
        fun=final.f,
        grad_norm=jnp.linalg.norm(final.g),
        n_iter=final.k,
        converged=final.converged,
        history=final.history,
    )
