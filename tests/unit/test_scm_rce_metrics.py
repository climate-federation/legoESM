"""Direct unit tests for :mod:`legoesm.training.scm_rce_metrics` leaf helpers.

The SCM-RCE metric module is the shared, differentiable profile/precip score reused
by ``column_era5_metrics`` (the clause-3 worst-column ranking) AND by the SCM-RCE
parameter-tuning loss.  Its gradient-safety is therefore load-bearing for parameter
training, but until now the module had NO direct test file — ``safe_sqrt`` /
``weighted_rmse`` were exercised only TRANSITIVELY (through ``per_column_weighted_rmse``
and the train/realism integration tests).  These tests lock the gradient-safe-sqrt
contract head-on, so a regression to a naive ``jnp.sqrt`` is caught at the leaf.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from legoesm.training.scm_rce_metrics import safe_sqrt, weighted_rmse


def test_safe_sqrt_primal_exact_and_gradient_safe_at_zero():
    """``safe_sqrt`` is the EXACT ``sqrt`` for ``x > 0`` (it IS the RMSE — the primal
    must not be perturbed) and routes a FINITE (zero) gradient through ``x <= 0``,
    where the true derivative ``1/(2√x) → ∞``.  A naive ``jnp.sqrt`` would give an
    infinite gradient at ``x = 0``; the double-where idiom regularises ONLY the
    gradient."""
    assert float(safe_sqrt(jnp.asarray(0.0))) == 0.0
    assert float(safe_sqrt(jnp.asarray(4.0))) == pytest.approx(2.0)
    # Normal branch: the gradient is the TRUE derivative, unchanged — d√x/dx|_4 = 0.25.
    assert float(jax.grad(safe_sqrt)(4.0)) == pytest.approx(0.25)
    # x = 0: the true derivative is +∞; safe_sqrt routes a finite ZERO gradient.
    g0 = jax.grad(safe_sqrt)(0.0)
    assert jnp.isfinite(g0) and float(g0) == 0.0
    # x < 0 (defensive): primal 0, finite gradient (no NaN from sqrt of a negative).
    assert jnp.isfinite(jax.grad(safe_sqrt)(-1.0))


def test_weighted_rmse_value_matches_mass_weighted_definition():
    """``weighted_rmse(diff, w) = √(Σ wₖ diffₖ²)`` — a mass-weighted vertical RMSE."""
    diff = jnp.array([2.0, 0.0, 4.0])
    weights = jnp.array([0.25, 0.25, 0.5])   # Σ w·diff² = .25·4 + 0 + .5·16 = 9 ⇒ √9 = 3
    assert float(weighted_rmse(diff, weights)) == pytest.approx(3.0)


def test_weighted_rmse_zero_error_gradient_is_finite():
    """The load-bearing training-convergence property: as the optimizer drives the
    model toward the reference, the weighted RMSE → 0 (``safe_sqrt(0)``).  A naive
    ``jnp.sqrt`` gives an INFINITE gradient EXACTLY at convergence (``d√s/ds = ∞`` at
    ``s = 0``, propagated as ``∞·0 = NaN`` through the chain rule), so a parameter
    training run would blow up at the optimum.  ``safe_sqrt`` routes a finite gradient.
    Locked w.r.t. a model scale at an EXACT match (RMSE == 0)."""
    ref = jnp.array([285.0, 280.0, 275.0])
    weights = jnp.array([0.3, 0.5, 0.2])

    def rmse_of(scale):                  # scale = 1 ⇒ model == ref ⇒ RMSE == 0
        return weighted_rmse(scale * ref - ref, weights)

    assert float(rmse_of(1.0)) == 0.0
    g = jax.grad(rmse_of)(1.0)
    assert jnp.isfinite(g)
