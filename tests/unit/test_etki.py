"""ETKI optimizer unit tests — linear-model analytics + robustness.

For a LINEAR forward model G(θ) = Hθ with n_e > n_p (ensemble spans
parameter space) the ETKI mean iterates follow the preconditioned
gradient flow of ||R^{-1/2}(y - Hθ)||² and converge to the least-squares
solution as Σδt → ∞ — the textbook EKI property (Iglesias et al. 2013;
Schillings & Stuart 2017). These tests pin that behavior plus the
implementation contracts (shrink monotonicity, blow-up handling,
subspace property).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.training.etki import etki_update, run_etki


def _linear_problem(key, n_p=4, n_o=12, n_e=40, noise=0.0):
    k1, k2, k3 = jax.random.split(key, 3)
    H = jax.random.normal(k1, (n_o, n_p), dtype=jnp.float64)
    theta_star = jax.random.normal(k2, (n_p,), dtype=jnp.float64)
    y = H @ theta_star
    if noise > 0.0:
        y = y + noise * jax.random.normal(k3, (n_o,), dtype=jnp.float64)
    theta0 = 2.0 * jax.random.normal(
        jax.random.fold_in(key, 7), (n_e, n_p), dtype=jnp.float64)
    return H, theta_star, y, theta0


def test_linear_convergence_to_truth():
    """Noise-free linear model, full-rank H: mean -> theta_star."""
    H, theta_star, y, theta0 = _linear_problem(jax.random.PRNGKey(0))

    def fwd(theta):
        return theta @ H.T

    theta, misfits = run_etki(fwd, theta0, y, n_iterations=40, dt=1.0)
    theta_mean = jnp.mean(theta, axis=0)
    rel = float(jnp.linalg.norm(theta_mean - theta_star)
                / jnp.linalg.norm(theta_star))
    # Ensemble collapse bounds the practical accuracy (Schillings & Stuart
    # 2017): expect percent-level recovery from a 2-sigma-wide prior, not
    # machine precision.
    assert rel < 2e-2, f"ETKI mean did not converge: rel={rel:.2e}"
    # misfit decreases monotonically on a noise-free linear problem
    assert all(b <= a * (1 + 1e-12) for a, b in zip(misfits, misfits[1:])), (
        f"misfit not monotone: {misfits[:6]}...")
    assert misfits[-1] < 1e-4 * misfits[0]


def test_ensemble_shrinks_to_consensus():
    H, _, y, theta0 = _linear_problem(jax.random.PRNGKey(1))

    def fwd(theta):
        return theta @ H.T

    spread0 = float(jnp.mean(jnp.var(theta0, axis=0)))
    theta = theta0
    spreads = [spread0]
    for _ in range(5):
        step = etki_update(theta, fwd(theta), y, dt=1.0)
        theta = step.theta
        spreads.append(float(jnp.mean(jnp.var(theta, axis=0))))
    assert all(b < a for a, b in zip(spreads, spreads[1:])), (
        f"ensemble spread must shrink every iteration: {spreads}")


def test_update_stays_in_ensemble_span():
    """EKI subspace property: with n_e <= n_p the update never leaves the
    affine span of the initial ensemble."""
    key = jax.random.PRNGKey(2)
    n_p, n_e, n_o = 6, 3, 8
    H = jax.random.normal(key, (n_o, n_p), dtype=jnp.float64)
    theta0 = jax.random.normal(
        jax.random.fold_in(key, 1), (n_e, n_p), dtype=jnp.float64)
    y = jax.random.normal(jax.random.fold_in(key, 2), (n_o,),
                          dtype=jnp.float64)
    step = etki_update(theta0, theta0 @ H.T, y, dt=0.7)
    # span basis: centered initial members
    mean0 = jnp.mean(theta0, axis=0)
    basis = (theta0 - mean0).T                       # (n_p, n_e)
    resid = step.theta - mean0[None, :]
    # project residuals onto the basis; reconstruction must be exact
    coef, *_ = jnp.linalg.lstsq(basis, resid.T, rcond=None)
    recon = (basis @ coef).T
    assert float(jnp.max(jnp.abs(recon - resid))) < 1e-9


def test_blowup_members_dropped():
    """Non-finite forward evaluations are excluded; update still works."""
    H, theta_star, y, theta0 = _linear_problem(jax.random.PRNGKey(3))
    g = theta0 @ H.T
    g = g.at[0, :].set(jnp.nan)
    g = g.at[3, 1].set(jnp.inf)
    step = etki_update(theta0, g, y, dt=1.0)
    assert step.n_valid == theta0.shape[0] - 2
    assert bool(jnp.all(jnp.isfinite(step.theta)))


def test_all_blowup_raises():
    H, _, y, theta0 = _linear_problem(jax.random.PRNGKey(4), n_e=4)
    g = jnp.full((4, y.shape[0]), jnp.nan)
    with pytest.raises(RuntimeError, match="finite forward"):
        etki_update(theta0, g, y)


def test_r_diag_weighting():
    """Down-weighted observations influence the update less."""
    key = jax.random.PRNGKey(5)
    n_p, n_e = 2, 20
    H = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    theta_star = jnp.asarray([1.0, -1.0])
    y = H @ theta_star
    # poison observation 0, then de-weight it: recovery of theta[1] must
    # be unaffected and theta[0] must follow the (poisoned) obs less when
    # its variance is huge
    y_bad = y.at[0].add(10.0)
    theta0 = jax.random.normal(key, (n_e, n_p), dtype=jnp.float64)

    def fwd(theta):
        return theta @ H.T

    th_unif, _ = run_etki(fwd, theta0, y_bad, n_iterations=20, dt=1.0)
    r = jnp.asarray([1e6, 1.0, 1.0])
    th_wt, _ = run_etki(fwd, theta0, y_bad, n_iterations=20, dt=1.0, r_diag=r)
    err_unif = abs(float(jnp.mean(th_unif, axis=0)[0]) - 1.0)
    err_wt = abs(float(jnp.mean(th_wt, axis=0)[0]) - 1.0)
    assert err_wt < err_unif, (
        f"de-weighting the poisoned obs must reduce its pull: "
        f"{err_wt:.3f} vs {err_unif:.3f}")


def test_noise_robust_recovery():
    """Noisy observations: mean lands near truth (the chaotic-statistics
    robustness this method was chosen for, in its simplest avatar)."""
    H, theta_star, y, theta0 = _linear_problem(
        jax.random.PRNGKey(6), n_o=40, n_e=60, noise=0.05)

    def fwd(theta):
        return theta @ H.T

    theta, _ = run_etki(fwd, theta0, y, n_iterations=15, dt=1.0)
    rel = float(jnp.linalg.norm(jnp.mean(theta, axis=0) - theta_star)
                / jnp.linalg.norm(theta_star))
    assert rel < 0.05, f"noisy recovery too far off: rel={rel:.3f}"


def test_package_reexports():
    from legoesm.training import etki_update as e, run_etki as r
    assert e is etki_update and r is run_etki
