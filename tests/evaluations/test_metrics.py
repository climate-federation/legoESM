"""Tests for the optional mask on evaluations.metrics.bias / acc (Stage 0, Task 7)."""
import numpy as np
import jax.numpy as jnp

from evaluations.metrics import rmse, acc, bias


def test_bias_mask_excludes_cells():
    pred = jnp.array([[1.0, 100.0], [1.0, 100.0]])
    target = jnp.zeros((2, 2))
    w = jnp.ones(2)
    mask = jnp.array([[1.0, 0.0], [1.0, 0.0]])   # exclude column 1
    b = bias(pred, target, w, mask=mask)
    assert np.isclose(float(b), 1.0)             # only column 0 (=1) counts, not 100


def test_acc_mask_excludes_anticorrelated_cells():
    clim = jnp.zeros((2, 3))
    pred = jnp.array([[1.0, 2.0, 999.0], [3.0, 4.0, -999.0]])
    target = jnp.array([[1.0, 2.0, -999.0], [3.0, 4.0, 999.0]])   # col 2 anti-correlated
    w = jnp.ones(2)
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    a_masked = acc(pred, target, clim, w, mask=mask)
    a_unmasked = acc(pred, target, clim, w)
    assert float(a_masked) > 0.99                # masking the bad column -> ~perfect corr
    assert float(a_unmasked) < float(a_masked)   # unmasked is dragged down


def test_rmse_mask_leading_dims():
    # (..., n_lat, n_lon) contract: masked RMSE must not double-count leading dims.
    pred = jnp.ones((3, 2, 2))
    target = jnp.zeros((3, 2, 2))
    w = jnp.ones(2)
    mask = jnp.ones((2, 2))
    assert np.isclose(float(rmse(pred, target, w, mask=mask)), 1.0)   # all errors 1 -> RMSE 1


def test_bias_mask_leading_dims():
    # (..., n_lat, n_lon) contract: a leading dim must not double-count the denominator.
    pred = jnp.ones((2, 2, 2))
    target = jnp.zeros((2, 2, 2))
    w = jnp.ones(2)
    mask = jnp.ones((2, 2))
    assert np.isclose(float(bias(pred, target, w, mask=mask)), 1.0)


def test_mask_none_matches_unmasked():
    pred = jnp.array([[2.0, 3.0]])
    target = jnp.array([[1.0, 1.0]])
    w = jnp.ones(1)
    assert np.isclose(float(bias(pred, target, w)), 1.5)     # mean(1, 2)
    clim = jnp.zeros((1, 2))
    assert float(acc(pred, pred, clim, w)) > 0.999           # ACC of a field with itself -> 1
    # rmse mask=None path (already supported) still works
    assert np.isclose(float(rmse(pred, target, w)),
                      float(jnp.sqrt(jnp.mean((pred - target) ** 2))))
