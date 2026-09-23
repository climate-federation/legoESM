"""Contract tests for the shared implicit nonlinear root solver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.core.nonlinear import make_implicit_newton_solver


def _solver(fun, *, x_scale, f_scale, max_iters=80, **kwargs):
    return make_implicit_newton_solver(
        fun,
        x_scale=jnp.asarray(x_scale),
        f_scale=jnp.asarray(f_scale),
        max_iters=max_iters,
        **kwargs,
    )


def test_fixed_state_and_residual_scaling_recovers_mixed_unit_root():
    """Both supplied scales are needed for this fp32 mixed-unit system."""
    def residual(x, target):
        return jnp.array([
            x[0] + 1.0e6 * x[1] - target[0],
            1.0e-6 * x[0] - x[1] - target[1],
        ])

    target_x = jnp.array([3.0, 2.0e-6])
    target = jnp.array([target_x[0] + 1.0e6 * target_x[1],
                        1.0e-6 * target_x[0] - target_x[1]])
    solve = _solver(
        residual,
        x_scale=[1.0, 1.0e-6],
        f_scale=[1.0, 1.0e-6],
    )
    x, _, converged, *_ = solve(jnp.zeros(2), target)
    assert bool(converged)
    assert jnp.allclose(x, target_x, rtol=2.0e-5, atol=2.0e-8)


def test_convergence_flag_uses_residual_not_damped_step():
    def residual(x, target):
        return x - target

    solve = _solver(
        residual,
        x_scale=[1.0],
        f_scale=[1.0],
        max_iters=1,
        lambda_initial=1.0e8,
        lambda_max=1.0e12,
    )
    x, _, converged, n_sq, *_ = solve(jnp.zeros(1), jnp.ones(1))
    assert float(jnp.linalg.norm(x)) < 1.0e-6  # tiny damping-induced step
    assert float(n_sq) > 0.9
    assert not bool(converged)


def test_rejected_nonfinite_trial_raises_damping_and_keeps_iterate():
    def residual(x, _):
        return jnp.sqrt(x)

    solve = _solver(
        residual,
        x_scale=[1.0],
        f_scale=[1.0],
        max_iters=1,
        lambda_initial=1.0e-2,
    )
    x0 = jnp.array([0.1])
    x, _, converged, _, _, damping, _ = solve(x0, ())
    assert not bool(converged)
    assert jnp.array_equal(x, x0)  # invalid proposal was rejected
    assert jnp.isclose(damping, 4.0e-2)  # Nielsen bad-step update


def test_forward_step_uses_augmented_qr(monkeypatch):
    qr_called = False
    original_qr = jnp.linalg.qr

    def recording_qr(value):
        nonlocal qr_called
        qr_called = True
        return original_qr(value)

    monkeypatch.setattr(jnp.linalg, "qr", recording_qr)

    def residual(x, target):
        return x - target

    solve = _solver(
        residual,
        x_scale=[1.0],
        f_scale=[1.0],
        max_iters=20,
    )
    x, _, converged, *_ = solve(jnp.zeros(1), jnp.array([2.0]))
    assert bool(converged)
    assert jnp.allclose(x, 2.0, atol=1.0e-4)
    assert qr_called


def test_column_equilibrated_exact_adjoint_matches_analytic_gradient(monkeypatch):
    norm_called = False
    solve_called = False
    original_norm = jnp.linalg.norm
    original_solve = jnp.linalg.solve

    def recording_norm(value, *args, **kwargs):
        nonlocal norm_called
        norm_called = True
        return original_norm(value, *args, **kwargs)

    def recording_solve(a, b):
        nonlocal solve_called
        solve_called = True
        return original_solve(a, b)

    monkeypatch.setattr(jnp.linalg, "norm", recording_norm)
    monkeypatch.setattr(jnp.linalg, "solve", recording_solve)
    matrix = jnp.array([[1.0e-8, 1.0], [2.0e-8, -1.0]])

    def residual(x, parameters):
        return matrix @ x - parameters

    solve = _solver(
        residual,
        x_scale=[1.0e8, 1.0],
        f_scale=[1.0, 1.0],
        max_iters=40,
    )

    def objective(parameters):
        return jnp.array([1.0, -0.25]) @ solve(jnp.zeros(2), parameters)[0]

    parameters = jnp.array([1.25, -0.75])
    gradient = jax.grad(objective)(parameters)
    expected = jnp.linalg.solve(matrix.T, jnp.array([1.0, -0.25]))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, expected, rtol=2.0e-5, atol=2.0e-5)
    assert norm_called  # column equilibration
    assert solve_called  # exact solve, not least-squares truncation


def test_nonconverged_root_has_zero_parameter_gradient():
    def residual(x, parameter):
        return x * x - parameter

    solve = _solver(
        residual,
        x_scale=[1.0],
        f_scale=[1.0],
        max_iters=1,
    )

    def objective(parameter):
        return solve(jnp.array([0.1]), parameter)[0][0]

    result = solve(jnp.array([0.1]), jnp.array([4.0]))
    assert not bool(result[2])
    assert float(jax.grad(objective)(jnp.array([4.0]))[0]) == 0.0
