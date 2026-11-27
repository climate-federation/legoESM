"""Contract tests for the shared implicit nonlinear root solver."""

from __future__ import annotations

import re
import jax
import numpy as np
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

    def recording_qr(value, **kwargs):
        nonlocal qr_called
        qr_called = True
        return original_qr(value, **kwargs)

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


def test_forward_step_never_forms_q():
    """The step applies Q^T from the packed reflectors: on GPU the explicit-Q
    (householder_product / orgqr) lowering of ``qr`` runs per matrix and
    dominated the land canopy solve."""
    def residual(x, target):
        return jnp.stack([x[0] - target[0], x[1] ** 3 + x[0] - target[1]])

    solve = _solver(residual, x_scale=[1.0, 1.0], f_scale=[1.0, 1.0],
                    max_iters=20)
    # Seed off x1 = 0, where the Jacobian is singular and the solve stalls.
    args = (jnp.array([1.5, 0.5]), jnp.array([2.0, 3.0]))
    text = str(jax.make_jaxpr(solve)(*args))
    assert "geqrf" in text
    # The explicit-Q path is the ``qr`` primitive, which lowers to
    # geqrf + householder_product; it must not appear in the step.
    assert re.search(r"\bqr\[", text) is None
    x, _, converged, *_ = solve(*args)
    assert bool(converged)
    assert jnp.allclose(x, jnp.array([2.0, 1.0]), atol=1.0e-4)


def test_one_step_equals_damped_least_squares(x64):
    """One accepted step on a linear residual is the damped normal-equations
    solution; a wrong reflector layout or order gives a different step."""
    rng = np.random.default_rng(0)
    n = 6
    jac = rng.normal(size=(n, n))
    b = rng.normal(size=n)
    damping = 0.3

    def residual(x, b_):
        return jnp.asarray(jac) @ x - b_

    solve = _solver(residual, x_scale=np.ones(n), f_scale=np.ones(n),
                    max_iters=1, lambda_initial=damping)
    x, n_iters, *_ = solve(jnp.zeros(n), jnp.asarray(b))
    expected = np.linalg.solve(jac.T @ jac + damping * np.eye(n), jac.T @ b)
    assert int(n_iters) == 1
    np.testing.assert_allclose(np.asarray(x), expected, rtol=1e-10, atol=1e-12)


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



def test_adjoint_unconverged_gives_implicit_gradient_at_the_last_iterate(x64):
    """``adjoint_unconverged``: a non-converged x carries -(df/dx)^-1 df/dp at x."""
    from jax.test_util import check_grads

    def residual(x, parameter):
        return x * x - parameter

    one_step = _solver(residual, x_scale=[1.0], f_scale=[1.0], max_iters=1,
                       adjoint_unconverged=True)
    x1, _, converged, *_ = one_step(jnp.array([0.1]), jnp.array([4.0]))
    assert not bool(converged)
    g = jax.grad(lambda p: one_step(jnp.array([0.1]), p)[0][0])(jnp.array([4.0]))
    np.testing.assert_allclose(float(g[0]), 1.0 / (2.0 * float(x1[0])), rtol=1e-12)

    # an unreachable ceiling: the loop stops at the cap on the root, never
    # "converged"
    near_root = _solver(residual, x_scale=[1.0], f_scale=[1.0], max_iters=60,
                        n_sq_max=-1.0, adjoint_unconverged=True)

    def root(p):
        return near_root(jnp.array([1.0]), p)[0]

    assert not bool(near_root(jnp.array([1.0]), jnp.array([4.0]))[2])
    np.testing.assert_allclose(
        float(jax.grad(lambda p: root(p)[0])(jnp.array([4.0]))[0]), 0.25, rtol=1e-10)
    # order 1 only: the forward solve's raw QR (geqrf) has no differentiation
    # rule, so no column of this solver, converged or not, has a second
    # reverse derivative.
    check_grads(root, (jnp.array([4.0]),), order=1, modes=["rev"])

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


def test_fp32_solve_stays_fp32_under_x64_with_fp64_scales(x64):
    """Import-time fp64 scales must not crash or promote an fp32 solve under x64."""
    def residual(x, target):
        return jnp.stack([
            x[0] + 1.0e6 * x[1] - target[0],
            1.0e-6 * x[0] - x[1] - target[1],
        ])

    solve = make_implicit_newton_solver(
        residual,
        x_scale=jnp.array([1.0, 1.0e-6], dtype=jnp.float64),
        f_scale=jnp.array([1.0, 1.0e-6], dtype=jnp.float64),
        max_iters=80,
    )
    target_x = jnp.array([3.0, 2.0e-6], dtype=jnp.float32)
    target = residual(target_x, jnp.zeros(2, dtype=jnp.float32))
    x0 = jnp.zeros(2, dtype=jnp.float32)

    jaxpr_text = str(jax.make_jaxpr(solve)(x0, target))
    assert "while[" in jaxpr_text
    assert "f64" not in jaxpr_text[jaxpr_text.index("while["):]

    for runner in (solve, jax.jit(solve)):
        x, _, converged, n_sq, n_sq_rel, damping, hit_cap = runner(x0, target)
        assert bool(converged)
        assert jnp.allclose(x, target_x, rtol=2.0e-5, atol=2.0e-8)
        for arr in (x, n_sq, n_sq_rel, damping, hit_cap):
            assert arr.dtype == jnp.float32

    batch_targets = jnp.stack([target, target * jnp.float32(1.5)])
    batch_x0 = jnp.zeros((2, 2), dtype=jnp.float32)
    bx, _, bconv, *_ = jax.vmap(solve)(batch_x0, batch_targets)
    assert bool(jnp.all(bconv))
    assert bx.dtype == jnp.float32
    assert jnp.allclose(bx[1], target_x * jnp.float32(1.5), rtol=2.0e-5, atol=2.0e-8)

    def objective(x_seed, t):
        return solve(x_seed, t)[0][0]

    g_x0, g_t = jax.grad(objective, argnums=(0, 1))(x0, target)
    assert g_x0.dtype == jnp.float32
    assert jnp.array_equal(g_x0, jnp.zeros_like(x0))
    assert g_t.dtype == jnp.float32
    assert jnp.allclose(g_t, jnp.array([0.5, 5.0e5], dtype=jnp.float32), rtol=2.0e-5, atol=2.0e-5)


def test_fp32_seed_with_mixed_fp64_parameters_preserves_seed_dtype_and_adjoint(x64):
    """An fp32 seed with mixed fp64/fp32 parameters solves cleanly and preserves dtypes."""
    def residual(x, params):
        a, b = params
        return jnp.stack([x[0] ** 2 - a, x[1] ** 3 + x[0] - b])

    solve = make_implicit_newton_solver(
        residual,
        x_scale=jnp.array([1.0, 1.0], dtype=jnp.float64),
        f_scale=jnp.array([1.0, 1.0], dtype=jnp.float64),
        max_iters=40,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    x0 = jnp.array([1.5, 0.5], dtype=jnp.float32)
    params = (jnp.array(4.0, dtype=jnp.float64), jnp.array(3.0, dtype=jnp.float32))
    expected_x = jnp.array([2.0, 1.0], dtype=jnp.float32)

    for runner in (solve, jax.jit(solve)):
        x, _, converged, n_sq, n_sq_rel, damping, hit_cap = runner(x0, params)
        assert bool(converged)
        assert jnp.allclose(x, expected_x, rtol=1.0e-6, atol=1.0e-6)
        for arr in (x, n_sq, n_sq_rel, damping, hit_cap):
            assert arr.dtype == jnp.float32

    bx, _, bconv, *_ = jax.vmap(solve, in_axes=(0, (0, 0)))(
        jnp.stack([x0, x0]),
        (jnp.array([4.0, 9.0], dtype=jnp.float64), jnp.array([3.0, 11.0], dtype=jnp.float32)),
    )
    assert bool(jnp.all(bconv))
    assert bx.dtype == jnp.float32
    assert jnp.allclose(bx, jnp.array([[2.0, 1.0], [3.0, 2.0]], dtype=jnp.float32), rtol=1.0e-6)

    def objective(x_seed, p):
        x_sol = solve(x_seed, p)[0]
        return x_sol[0] + jnp.float32(2.0) * x_sol[1]

    g_x0, (g_a, g_b) = jax.grad(objective, argnums=(0, 1))(x0, params)
    assert g_x0.dtype == jnp.float32
    assert jnp.array_equal(g_x0, jnp.zeros_like(x0))
    assert g_a.dtype == jnp.float64
    assert g_b.dtype == jnp.float32
    np.testing.assert_allclose(float(g_a), 1.0 / 12.0, rtol=1.0e-6)
    np.testing.assert_allclose(float(g_b), 2.0 / 3.0, rtol=1.0e-6)

    # Same mixed-dtype contract when adjoint_unconverged differentiates at the last iterate.
    one_step = make_implicit_newton_solver(
        residual,
        x_scale=jnp.array([1.0, 1.0], dtype=jnp.float64),
        f_scale=jnp.array([1.0, 1.0], dtype=jnp.float64),
        max_iters=1,
        adjoint_unconverged=True,
    )
    x1, _, conv1, *_ = one_step(x0, params)
    assert not bool(conv1)
    assert x1.dtype == jnp.float32

    def one_step_objective(p):
        x_sol = one_step(x0, p)[0]
        return x_sol[0] + jnp.float32(2.0) * x_sol[1]

    g1_a, g1_b = jax.grad(one_step_objective)(params)
    assert g1_a.dtype == jnp.float64
    assert g1_b.dtype == jnp.float32
    x1_0, x1_1 = float(x1[0]), float(x1[1])
    expected_g1_a = (1.0 - 2.0 / (3.0 * x1_1 ** 2)) / (2.0 * x1_0)
    expected_g1_b = 2.0 / (3.0 * x1_1 ** 2)
    np.testing.assert_allclose(float(g1_a), expected_g1_a, rtol=1.0e-5)
    np.testing.assert_allclose(float(g1_b), expected_g1_b, rtol=1.0e-5)


def test_python_scales_keep_fp64_precision_when_x64_enabled_later():
    """Scales supplied as Python sequences are cast at solve time, not truncated at build."""
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        s = 1.0 + 1.0e-9
        lam = 1.0

        def residual(x, target):
            return x - target

        solve = make_implicit_newton_solver(
            residual,
            x_scale=[s],
            f_scale=[1.0],
            max_iters=1,
            lambda_initial=lam,
        )
        jax.config.update("jax_enable_x64", True)
        x, n_iters, *_ = solve(
            jnp.zeros(1, dtype=jnp.float64),
            jnp.ones(1, dtype=jnp.float64),
        )
        assert int(n_iters) == 1
        assert x.dtype == jnp.float64
        expected = (s * s) / (s * s + lam)
        np.testing.assert_allclose(float(x[0]), expected, rtol=1.0e-12, atol=1.0e-12)
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_mutating_python_scale_list_does_not_alter_solver(x64):
    """Python-list scales are snapshotted at build time so later mutation cannot drift."""
    def residual(x, target):
        return x - target

    scale = [1.0]
    solve = make_implicit_newton_solver(
        residual,
        x_scale=scale,
        f_scale=[1.0],
        max_iters=1,
        lambda_initial=1.0,
    )
    x0 = jnp.zeros(1, dtype=jnp.float64)
    target = jnp.ones(1, dtype=jnp.float64)
    compiled = jax.jit(solve)
    before = float(compiled(x0, target)[0][0])
    scale[0] = 2.0
    after_eager = float(solve(x0, target)[0][0])
    after_jit = float(compiled(x0, target)[0][0])
    np.testing.assert_allclose(before, 0.5, rtol=1.0e-12)
    np.testing.assert_allclose(after_eager, 0.5, rtol=1.0e-12)
    np.testing.assert_allclose(after_jit, 0.5, rtol=1.0e-12)


def test_mixed_precision_certificate_reflects_returned_fp32_state(x64):
    """Downcast rounding to fp32 must be reflected in converged and reported n_sq."""
    def residual(x, target):
        return x - target

    solve = make_implicit_newton_solver(
        residual,
        x_scale=jnp.ones(1, dtype=jnp.float64),
        f_scale=jnp.ones(1, dtype=jnp.float64),
        max_iters=20,
    )
    # Cold start from far away: fp32-rounded state 298.0 still beats the relative gate.
    cold_seed = jnp.array([295.0], dtype=jnp.float32)
    target = jnp.array([298.00001], dtype=jnp.float64)
    x_cold, _, conv_cold, n_sq_cold, *_ = solve(cold_seed, target)
    actual_cold = float(jnp.sum(residual(x_cold.astype(jnp.float64), target) ** 2))
    assert x_cold.dtype == jnp.float32
    assert bool(conv_cold)
    np.testing.assert_allclose(float(n_sq_cold), actual_cold, rtol=1.0e-5)

    # Warm start from the returned fp32 state: rounding leaves residual 1e-10 > atol=1e-12,
    # so the certificate must report unconverged and match the returned state's residual.
    x_warm, _, conv_warm, n_sq_warm, *_ = solve(x_cold, target)
    actual_warm = float(jnp.sum(residual(x_warm.astype(jnp.float64), target) ** 2))
    assert x_warm.dtype == jnp.float32
    assert not bool(conv_warm)
    assert actual_warm > 1.0e-11
    np.testing.assert_allclose(float(n_sq_warm), actual_warm, rtol=1.0e-5)
    g_warm = jax.grad(lambda t: solve(x_cold, t)[0][0])(target)
    assert g_warm.dtype == jnp.float64
    assert float(g_warm[0]) == 0.0

    # Admissibility is also certified on the returned fp32 state (298.0 < 298.000005),
    # even though the internal fp64 root (298.00001) satisfied it.
    solve_adm = make_implicit_newton_solver(
        residual,
        x_scale=jnp.ones(1, dtype=jnp.float64),
        f_scale=jnp.ones(1, dtype=jnp.float64),
        max_iters=20,
        admissible=lambda x: x[0] >= 298.000005,
    )
    x_adm, _, conv_adm, *_ = solve_adm(cold_seed, target)
    assert x_adm.dtype == jnp.float32
    assert not bool(conv_adm)
