"""Contract tests for the shared implicit nonlinear root solver."""

from __future__ import annotations

import jax
import pytest
import jax.numpy as jnp
from legoesm.core.nonlinear import attach_implicit_gradient, make_implicit_newton_solver


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


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cube_root_newton(a, iters=40):
    return jax.lax.fori_loop(
        0, iters, lambda _, x: x - (x ** 3 - a) / (3.0 * x ** 2), jnp.ones_like(a))


def test_attach_implicit_gradient_value_gradient_and_no_replay(x64):
    a = jnp.array([2.0, 27.0, 0.5])

    def root(a):
        return attach_implicit_gradient(_cube_root_newton(a), lambda x: x ** 3 - a)

    x = root(a)
    assert jnp.allclose(x, jnp.cbrt(a), rtol=0, atol=1e-14)
    g = jax.vmap(jax.grad(lambda ai: root(ai)))(a)
    assert jnp.allclose(g, 1.0 / (3.0 * jnp.cbrt(a) ** 2), rtol=1e-12)
    # Reverse mode must not replay the forward iteration.
    _, lin = jax.linearize(root, a)
    assert "while" not in str(jax.make_jaxpr(lin)(a))


def test_attach_implicit_gradient_clip_gives_zero_gradient(x64):
    a = jnp.array([8.0, 1000.0])

    def root(a):
        return attach_implicit_gradient(
            jnp.clip(_cube_root_newton(a), 0.0, 5.0), lambda x: x ** 3 - a,
            lo=0.0, hi=5.0)

    assert jnp.allclose(root(a), jnp.array([2.0, 5.0]))
    g = jax.vmap(jax.grad(root))(a)
    assert g[1] == 0.0 and g[0] > 0.0


def test_attach_implicit_gradient_bound_and_nan(x64):
    a = jnp.array([125.0, 8.0])

    def root(a):
        # x0 exactly AT the bound (5.0): zero gradient, not clip's half.
        return attach_implicit_gradient(
            jnp.clip(_cube_root_newton(a), 0.0, 5.0), lambda x: x ** 3 - a,
            lo=0.0, hi=5.0)

    g = jax.vmap(jax.grad(root))(a)
    assert g[0] == 0.0 and g[1] > 0.0
    # A pinned value keeps its forward value even if the step points inward.
    x = attach_implicit_gradient(jnp.array([5.0]), lambda x: x ** 3 - 100.0,
                                 lo=0.0, hi=5.0)
    assert x[0] == 5.0
    # A NaN slope with a finite residual must give NaN, not a value moved by
    # residual / floor.
    @jax.custom_jvp
    def resid(x):
        return x - 0.5

    @resid.defjvp
    def _resid_jvp(primals, tangents):
        return resid(primals[0]), tangents[0] * jnp.nan

    x = attach_implicit_gradient(jnp.array([1.0]), resid)
    assert jnp.isnan(x).all()


def _split_residual(offset):
    """|f| as the Jacobian sweep sees it (x - 1) and as a plain call sees it
    (x - 1 + offset).  The deliberate gap stands in for the rounding gap between
    two compilations of one residual; keep it, or the tests below go vacuous."""
    @jax.custom_jvp
    def f(x):
        return x - 1.0 + offset

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        return x - 1.0, dx

    return lambda x, _: f(x)


def test_a_seed_within_tolerance_is_converged_not_sent_to_the_fallback():
    """The verdict must be the loop's own: the sweep sees |f| = 0.9e-6 at the
    seed, a plain call 1.1e-6, either side of atol = 1e-12 on |f|^2.  A seed the
    loop accepts must come back converged, or its column falls back to the
    cold-state fluxes."""
    solve = _solver(_split_residual(2.0e-7), x_scale=[1.0], f_scale=[1.0], atol=1.0e-12)
    x0 = jnp.array([1.0 + 0.9e-6])
    x, n_iters, converged, n_sq, *_ = solve(x0, ())
    assert int(n_iters) == 0
    assert bool(converged)
    assert float(n_sq) < 1.0e-12
    assert float(x[0]) == float(x0[0])


def test_a_root_the_loop_reaches_is_converged_not_sent_to_the_fallback():
    """Same contract after iterating: the loop stops on its own residual; a
    plain call at that root reads |f| ~ 2e-6, |f|^2 ~ 4e-12, above the gate."""
    solve = _solver(_split_residual(2.0e-6), x_scale=[1.0], f_scale=[1.0], atol=1.0e-12)
    x, n_iters, converged, n_sq, *_ = solve(jnp.array([1.001]), ())
    assert int(n_iters) >= 1
    assert bool(converged)
    assert float(n_sq) <= 1.0e-12 + 1.0e-8 * 1.0e-6
