"""Differentiable nonlinear root solvers shared by legoESM components."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


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
    loop: str = "while",
):
    """Build a scaled Levenberg--Marquardt root solver with an IFT VJP.

    ``residual_fn(x, parameters)`` must return a one-dimensional square
    residual.  ``x_scale`` and ``f_scale`` are fixed characteristic magnitudes;
    the forward solve works in ``z = x / x_scale`` and ``f / f_scale``.

    The result is ``(x, n_iters, converged, n_sq, n_sq_rel, damping, hit_cap)``.
    Convergence is based only on the scaled squared residual.  The forward
    linear least-squares step uses augmented QR and Nielsen gain-ratio damping
    with rejected steps.  The backward pass uses an exact, column-equilibrated
    solve and returns zero cotangents for every non-converged root.

    ``loop`` selects how the iteration is expressed, and ONLY that -- the two
    modes return bit-identical results (pinned by
    ``tests/unit/test_nonlinear_loop_modes.py``), because the fixed-length body
    is masked by the same ``done`` flag the ``while`` condition tests, so an
    iteration after convergence is an exact no-op:

    * ``"while"`` (default) -- ``lax.while_loop``, which STOPS as soon as every
      root has converged.  Cheapest, and what production runs.
    * ``"fixed"`` -- ``lax.scan`` over exactly ``max_iters`` iterations.  Slower
      whenever the solve converges early, and the reason it exists is
      differentiability, not speed: forward-over-reverse (an ``hvp``, a
      Gauss-Newton inner solve, a Lanczos spectrum) through a ``while_loop``
      produced a program the XLA CPU backend ABORTED on -- a core dump, not an
      exception (#1736).  ``"fixed"`` has no ``while`` primitive, so those
      operators can run on objectives containing this solve.
    """
    if loop not in ("while", "fixed"):
        raise ValueError(
            "make_implicit_newton_solver loop must be 'while' or 'fixed'; "
            f"got {loop!r}.")
    x_scale = jnp.asarray(x_scale)
    f_scale = jnp.asarray(f_scale)

    def scaled_residual(x, parameters):
        return residual_fn(x, parameters) / f_scale

    def forward(x0, parameters):
        def fun(x):
            return scaled_residual(x, parameters)

        jac = jax.jacfwd(fun)

        def residual_jacobian(x):
            return fun(x), jac(x) * x_scale

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
            residual_trial = fun(x_trial)
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
            jacobian_new = jnp.where(
                accept, jac(x_trial) * x_scale, jacobian_z
            )

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
            converged = n_sq_new <= atol + rtol * n_sq_0
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

        initial_converged = n_sq_0 <= atol
        state_0 = (
            x0,
            residual_0,
            jacobian_z_0,
            n_sq_0,
            jnp.asarray(lambda_initial, dtype=x0.dtype),
            jnp.array(0),
            initial_converged,
        )
        if loop == "while":
            x_final, _, _, _, damping, n_iters, _ = jax.lax.while_loop(
                cond, body, state_0,
            )
        else:
            # Fixed-length form (#1736): the same body, run exactly max_iters
            # times, with every carry element FROZEN once ``done`` is set.  The
            # while_loop stops at that same flag, so the two give bit-identical
            # finals; this one merely keeps iterating over a state that no
            # longer changes.  No ``while`` primitive, so forward-over-reverse
            # can be compiled (an hvp through the while form aborted XLA).
            def masked_body(state, _):
                done = state[-1]
                stepped = body(state)
                frozen = tuple(
                    jnp.where(done, old, new)
                    for old, new in zip(state, stepped)
                )
                return frozen, None

            state_f, _ = jax.lax.scan(
                masked_body, state_0, None, length=int(max_iters),
            )
            x_final, _, _, _, damping, n_iters, _ = state_f
        residual_final = fun(x_final)
        n_sq_final = jnp.sum(residual_final * residual_final)
        converged = n_sq_final <= atol + rtol * n_sq_0
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
