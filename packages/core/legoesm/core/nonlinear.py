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
    adjoint_unconverged: bool = False,
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
    solve and returns zero cotangents for every non-converged root, unless
    ``adjoint_unconverged``: then a non-converged ``x`` (a caller keeping the
    last iterate) gets the implicit-function adjoint at that iterate,
    ``-(df/dx)^-1 df/dp``, which approaches the root's as the residual left
    there shrinks.

    The working dtype is ``jnp.result_type(x0.dtype, residual.dtype)``:
    ``x_scale``, ``f_scale``, and the ``while_loop`` carry are aligned to that
    working dtype so import-time ``float64`` scale constants never trigger a
    carry mismatch or silent promotion on ``float32`` solves under
    ``JAX_ENABLE_X64=1``.  When ``work_dtype != x0.dtype`` (mixed-precision
    solve), floating outputs are cast back to ``x0.dtype`` (matching
    :func:`attach_implicit_gradient` and the ``custom_vjp`` cotangent dtype)
    and ``converged``, ``n_sq``, ``n_sq_rel``, and ``admissible`` are certified
    on the returned ``x0.dtype`` state so callers never receive a ``True``
    certificate invalidated by downcast rounding.
    """

    def _freeze_scale(scale):
        if isinstance(scale, jax.Array):
            return scale
        if hasattr(scale, "tolist"):
            scale = scale.tolist()
        if isinstance(scale, list):
            return tuple(scale)
        return scale

    x_scale_ref = _freeze_scale(x_scale)
    f_scale_ref = _freeze_scale(f_scale)

    def accepted(x, n_sq):
        ok = jnp.asarray(True)
        if n_sq_max is not None:
            ok = ok & (n_sq <= n_sq_max)
        if admissible is not None:
            ok = ok & admissible(x)
        return ok

    def _forward_with_work(x0, parameters):
        x0 = jnp.asarray(x0)
        x_dtype = x0.dtype
        res_dtype = jax.eval_shape(residual_fn, x0, parameters).dtype
        work_dtype = jnp.result_type(x_dtype, res_dtype)
        x_scale_w = jnp.asarray(x_scale_ref, dtype=work_dtype)
        f_scale_w = jnp.asarray(f_scale_ref, dtype=work_dtype)
        x_init = x0.astype(work_dtype)
        lam_zero = jnp.asarray(0.0, dtype=work_dtype)
        lam_max = jnp.asarray(lambda_max, dtype=work_dtype)

        def fun(x):
            return residual_fn(x, parameters) / f_scale_w

        def fun_twice(x):
            f = fun(x)
            return f, f

        # The forward sweep's primal IS the residual: one pass gives both.
        jac_and_fun = jax.jacfwd(fun_twice, has_aux=True)

        def residual_jacobian(x):
            jacobian, residual = jac_and_fun(x)
            return residual, jacobian * x_scale_w

        def step(jacobian_z, residual, damping):
            n = residual.shape[0]
            augmented = jnp.vstack(
                [jacobian_z, jnp.sqrt(damping) * jnp.eye(n, dtype=residual.dtype)]
            )
            rhs = jnp.concatenate([-residual, jnp.zeros(n, residual.dtype)])
            # Q^T rhs from the packed reflectors, never forming Q: on GPU the
            # explicit-Q (orgqr) step runs one small kernel chain PER MATRIX,
            # which was ~90% of the land canopy solve's wall time.
            packed, taus = jnp.linalg.qr(augmented, mode="raw")
            h = packed.T
            qt_rhs = rhs
            for k in range(n):
                v = jnp.concatenate([jnp.zeros(k, h.dtype),
                                     jnp.ones(1, h.dtype), h[k + 1:, k]])
                qt_rhs = qt_rhs - taus[k] * jnp.dot(v, qt_rhs) * v
            delta_z = jax.scipy.linalg.solve_triangular(
                jnp.triu(h[:n, :]), qt_rhs[:n], lower=False
            )
            predicted = residual + jacobian_z @ delta_z
            return delta_z, jnp.sum(predicted * predicted)

        residual_0, jacobian_z_0 = residual_jacobian(x_init)
        n_sq_0 = jnp.sum(residual_0 * residual_0)

        def cond(state):
            return ~state[-1]

        def body(state):
            x, residual, jacobian_z, n_sq, damping, iteration, _ = state
            delta_z, n_sq_predicted = step(jacobian_z, residual, damping)
            x_trial = x + x_scale_w * delta_z
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
                lam_zero,
                lam_max,
            )
            iteration_new = iteration + 1
            converged = (n_sq_new <= atol + rtol * n_sq_0) & accepted(x_new, n_sq_new)
            done = (
                converged
                | (damping_new >= lam_max)
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

        initial_converged = (n_sq_0 <= atol) & accepted(x_init, n_sq_0)
        x_final, _, _, n_sq_final, damping, n_iters, _ = jax.lax.while_loop(
            cond,
            body,
            (
                x_init,
                residual_0,
                jacobian_z_0,
                n_sq_0,
                jnp.asarray(lambda_initial, dtype=work_dtype),
                jnp.array(0),
                initial_converged,
            ),
        )
        # The verdict reuses the loop's own residual when no downcast occurs.
        # Re-evaluating it on a same-dtype solve compiles differently from the
        # residual inside the Jacobian sweep, and at the tolerance edge that
        # rounding flipped a column the loop had accepted onto its fallback.
        # When work_dtype != x_dtype, casting x_final down to x_dtype moves the
        # returned state, so re-evaluate the residual and admissibility on the
        # actual returned state so the convergence certificate matches it.
        x_out = x_final.astype(x_dtype)
        if work_dtype != x_dtype:
            x_cert = x_out.astype(work_dtype)
            r_out = fun(x_cert)
            n_sq_cert = jnp.sum(r_out * r_out)
            admissible_ok = accepted(x_cert, n_sq_cert)
        else:
            x_cert = x_final
            n_sq_cert = n_sq_final
            admissible_ok = accepted(x_final, n_sq_final)
        converged = (n_sq_cert <= atol + rtol * n_sq_0) & admissible_ok
        tiny = jnp.finfo(n_sq_cert.dtype).tiny
        result = (
            x_out,
            n_iters,
            converged,
            n_sq_cert.astype(x_dtype),
            (n_sq_cert / jnp.maximum(n_sq_0, tiny)).astype(x_dtype),
            damping.astype(x_dtype),
            (n_iters >= max_iters).astype(x_dtype),
        )
        return result, x_cert

    @jax.custom_vjp
    def solve(x0, parameters):
        return _forward_with_work(x0, parameters)[0]

    def solve_fwd(x0, parameters):
        result, x_final_work = _forward_with_work(x0, parameters)
        return result, (x_final_work, parameters, result[2])

    def solve_bwd(saved, cotangents):
        x_star, parameters, converged = saved
        g_x = cotangents[0]
        work_dtype = x_star.dtype
        f_scale_w = jnp.asarray(f_scale_ref, dtype=work_dtype)
        x_safe = jnp.where(jnp.isfinite(x_star), x_star, jnp.zeros_like(x_star))
        input_invalid = jnp.any(~jnp.isfinite(x_star)) | jnp.any(~jnp.isfinite(g_x))

        jacobian = jax.jacfwd(
            lambda x: residual_fn(x, parameters) / f_scale_w
        )(x_safe)
        transpose = jacobian.T
        column_norm = jnp.linalg.norm(transpose, axis=0)
        column_scale = jnp.where(column_norm > 0.0, 1.0 / column_norm, 1.0)
        equilibrated = transpose * column_scale
        adjoint_scaled = jnp.linalg.solve(equilibrated, g_x.astype(work_dtype))
        adjoint = column_scale * adjoint_scaled

        _, vjp = jax.vjp(lambda p: residual_fn(x_safe, p) / f_scale_w, parameters)
        grad_parameters = vjp(-adjoint)[0]
        solve_failed = input_invalid | jnp.any(~jnp.isfinite(adjoint))
        if not adjoint_unconverged:
            solve_failed = solve_failed | ~converged

        def mask(value):
            return jnp.where(
                solve_failed | ~jnp.isfinite(value), jnp.zeros_like(value), value
            )

        return jnp.zeros_like(g_x), jax.tree.map(mask, grad_parameters)

    solve.defvjp(solve_fwd, solve_bwd)
    return solve
