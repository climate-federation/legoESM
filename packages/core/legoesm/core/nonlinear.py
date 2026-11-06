"""Differentiable nonlinear root solvers shared by legoESM components."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def attach_implicit_gradient(
    x_root: jax.Array,
    residual_fn: Callable,
    *,
    lo: float | None = None,
    hi: float | None = None,
    slope_floor: float = 1.0e-6,
) -> jax.Array:
    """Exact implicit-function gradient for an elementwise scalar root.

    ``x_root`` comes from any forward iteration (Newton, bisection, ...); it is
    detached and replaced by one Newton correction at the root with the slope
    held constant, ``x = x* - f(x*) / sg(f'(x*))``.  The value moves only by
    the residual left at ``x*``; the gradient w.r.t. everything ``residual_fn``
    closes over is ``-(df/dtheta)/(df/dx)``, and reverse mode never replays
    the forward iteration.  ``f'`` comes from a jvp, so at a kink it is the
    one-sided derivative of the active branch (``residual_fn`` decides which
    branch owns an exact tie).  ``slope_floor`` keeps the division finite
    (|f'| below it is replaced by +floor, as the Newton loops it replaces did);
    a NaN slope stays NaN.  ``lo``/``hi`` re-apply a clip of the forward
    iteration: a forward value at or beyond a bound is not a root; it is
    returned unchanged with a zero gradient (``jnp.clip`` alone would pass
    half of it at the bound).
    """
    x0 = jax.lax.stop_gradient(x_root)
    f, slope = jax.jvp(residual_fn, (x0,), (jnp.ones_like(x0),))
    slope = jax.lax.stop_gradient(jnp.where(
        jnp.isnan(slope) | (jnp.abs(slope) > slope_floor), slope, slope_floor))
    x = x0 - f / slope
    if lo is not None or hi is not None:
        x = jnp.clip(x, lo, hi)
        pinned = jnp.zeros(x0.shape, dtype=bool)
        if lo is not None:
            pinned = pinned | (x0 <= lo)
        if hi is not None:
            pinned = pinned | (x0 >= hi)
        x = jnp.where(pinned, x0, x)        # the forward value, no gradient
    return x.astype(x_root.dtype)


def make_implicit_newton_solver(
    residual_fn: Callable,
    *,
    x_scale: jax.Array,
    f_scale: jax.Array,
    max_iters: int,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-12,
    lambda_initial: float = 1.0e-2,
    lambda_max: float = 1.0e10,
    n_sq_max: float | None = None,
    admissible: Callable | None = None,
):
    """Build a scaled Levenberg--Marquardt root solver with an IFT VJP.

    ``residual_fn(x, parameters)`` must return a one-dimensional square
    residual.  ``x_scale`` and ``f_scale`` are fixed characteristic magnitudes;
    the forward solve works in ``z = x / x_scale`` and ``f / f_scale``.

    The result is ``(x, n_iters, converged, n_sq, n_sq_rel, damping, hit_cap)``.
    Convergence is based on the scaled squared residual: a reduction by
    ``rtol`` relative to the SEED's residual.  That alone cannot certify a root
    when the seed is far off (a huge seed residual makes the relative gate
    vacuous), so a caller can also require an absolute ceiling ``n_sq_max`` and
    a physical ``admissible(x) -> bool``; both default to off, which leaves the
    contract unchanged.  The combined flag drives the stop test, the returned
    ``converged`` and the adjoint mask alike.  The forward
    linear least-squares step uses augmented QR and Nielsen gain-ratio damping
    with rejected steps.  The backward pass uses an exact, column-equilibrated
    solve and returns zero cotangents for every non-converged root.
    """
    x_scale = jnp.asarray(x_scale)
    f_scale = jnp.asarray(f_scale)

    def accepted(x, n_sq):
        ok = jnp.asarray(True)
        if n_sq_max is not None:
            ok = ok & (n_sq <= n_sq_max)
        if admissible is not None:
            ok = ok & admissible(x)
        return ok

    def scaled_residual(x, parameters):
        return residual_fn(x, parameters) / f_scale

    def forward(x0, parameters):
        def fun(x):
            return scaled_residual(x, parameters)

        def fun_twice(x):
            f = fun(x)
            return f, f

        # The forward sweep's primal IS the residual: one pass gives both.
        jac_and_fun = jax.jacfwd(fun_twice, has_aux=True)

        def residual_jacobian(x):
            jacobian, residual = jac_and_fun(x)
            return residual, jacobian * x_scale

        def step(jacobian_z, residual, damping):
            n = residual.shape[0]
            augmented = jnp.vstack(
                [jacobian_z, jnp.sqrt(damping) * jnp.eye(n, dtype=residual.dtype)]
            )
            rhs = jnp.concatenate([-residual, jnp.zeros(n, residual.dtype)])
            q, r = jnp.linalg.qr(augmented)
            delta_z = jax.scipy.linalg.solve_triangular(
                r, q.T @ rhs, lower=False
            )
            predicted = residual + jacobian_z @ delta_z
            return delta_z, jnp.sum(predicted * predicted)

        residual_0, jacobian_z_0 = residual_jacobian(x0)
        n_sq_0 = jnp.sum(residual_0 * residual_0)

        def cond(state):
            return ~state[-1]

        def body(state):
            x, residual, jacobian_z, n_sq, damping, iteration, _ = state
            delta_z, n_sq_predicted = step(jacobian_z, residual, damping)
            x_trial = x + x_scale * delta_z
            residual_trial, jacobian_trial = residual_jacobian(x_trial)
            n_sq_trial = jnp.sum(residual_trial * residual_trial)
            denominator = jnp.maximum(
                n_sq - n_sq_predicted,
                jnp.asarray(1.0e-30, dtype=n_sq.dtype),
            )
            rho = (n_sq - n_sq_trial) / denominator
            accept = (rho > 0.0) & jnp.all(jnp.isfinite(residual_trial))

            x_new = jnp.where(accept, x_trial, x)
            residual_new = jnp.where(accept, residual_trial, residual)
            n_sq_new = jnp.where(accept, n_sq_trial, n_sq)
            jacobian_new = jnp.where(accept, jacobian_trial, jacobian_z)

            # Nielsen gain-ratio adaptation.  The negated comparison is
            # intentional: a NaN gain ratio must increase damping.
            bad = ~(rho >= 0.25)
            good = rho > 0.75
            damping_new = jnp.clip(
                jnp.where(
                    bad,
                    damping * 4.0,
                    jnp.where(good, damping / 3.0, damping),
                ),
                0.0,
                lambda_max,
            )
            iteration_new = iteration + 1
            converged = (n_sq_new <= atol + rtol * n_sq_0) & accepted(x_new, n_sq_new)
            done = (
                converged
                | (damping_new >= lambda_max)
                | (iteration_new >= max_iters)
            )
            return (
                x_new,
                residual_new,
                jacobian_new,
                n_sq_new,
                damping_new,
                iteration_new,
                done,
            )

        initial_converged = (n_sq_0 <= atol) & accepted(x0, n_sq_0)
        x_final, _, _, n_sq_final, damping, n_iters, _ = jax.lax.while_loop(
            cond,
            body,
            (
                x0,
                residual_0,
                jacobian_z_0,
                n_sq_0,
                jnp.asarray(lambda_initial, dtype=x0.dtype),
                jnp.array(0),
                initial_converged,
            ),
        )
        # The verdict reuses the loop's own residual.  Re-evaluating it here
        # compiles differently from the residual inside the Jacobian sweep, and
        # at the tolerance edge that rounding flipped a column the loop had
        # accepted onto its fallback.
        converged = ((n_sq_final <= atol + rtol * n_sq_0)
                     & accepted(x_final, n_sq_final))
        tiny = jnp.finfo(n_sq_final.dtype).tiny
        return (
            x_final,
            n_iters,
            converged,
            n_sq_final,
            n_sq_final / jnp.maximum(n_sq_0, tiny),
            damping,
            (n_iters >= max_iters).astype(x_final.dtype),
        )

    @jax.custom_vjp
    def solve(x0, parameters):
        return forward(x0, parameters)

    def solve_fwd(x0, parameters):
        result = forward(x0, parameters)
        return result, (result[0], parameters, result[2])

    def solve_bwd(saved, cotangents):
        x_star, parameters, converged = saved
        g_x = cotangents[0]
        x_safe = jnp.where(jnp.isfinite(x_star), x_star, jnp.zeros_like(x_star))
        input_invalid = jnp.any(~jnp.isfinite(x_star)) | jnp.any(~jnp.isfinite(g_x))

        jacobian = jax.jacfwd(
            lambda x: scaled_residual(x, parameters)
        )(x_safe)
        transpose = jacobian.T
        column_norm = jnp.linalg.norm(transpose, axis=0)
        column_scale = jnp.where(column_norm > 0.0, 1.0 / column_norm, 1.0)
        equilibrated = transpose * column_scale
        adjoint_scaled = jnp.linalg.solve(equilibrated, g_x)
        adjoint = column_scale * adjoint_scaled

        _, vjp = jax.vjp(lambda p: scaled_residual(x_safe, p), parameters)
        grad_parameters = vjp(-adjoint)[0]
        solve_failed = (
            input_invalid | ~converged | jnp.any(~jnp.isfinite(adjoint))
        )

        def mask(value):
            return jnp.where(
                solve_failed | ~jnp.isfinite(value), jnp.zeros_like(value), value
            )

        return jnp.zeros_like(x_star), jax.tree.map(mask, grad_parameters)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
