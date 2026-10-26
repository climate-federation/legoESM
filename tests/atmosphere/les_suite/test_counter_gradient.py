"""Unit tests for the Q1 counter-gradient diagnostic."""
from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.atmosphere.les_suite.bridge import LESTruth
from legoesm.atmosphere.les_suite.counter_gradient import (
    CounterGradientResult,
    centered_dtheta_dz,
    counter_gradient_diagnostic,
    diagnose_truth,
)


def test_centered_gradient_linear_profile():
    # θ = 300 + 0.004 z → dθ/dz = 0.004 everywhere (uniform grid)
    z = jnp.linspace(0.0, 1000.0, 11)
    theta = 300.0 + 0.004 * z
    grad = centered_dtheta_dz(theta, z)
    assert jnp.allclose(grad, 0.004, atol=1e-5)  # precision-robust (x32/x64)


def test_centered_gradient_nonuniform_grid():
    z = jnp.array([0.0, 10.0, 30.0, 70.0, 150.0])
    theta = 300.0 + 0.01 * z  # exact linear → gradient 0.01 even on stretched grid
    grad = centered_dtheta_dz(theta, z)
    assert jnp.allclose(grad, 0.01, atol=1e-5)  # precision-robust (x32/x64)


def test_pure_downgradient_no_counter_layer():
    # Stable BL: θ increases with z (dθ/dz>0) and flux is downward (negative).
    # flux*grad < 0 everywhere → down-gradient, no counter-gradient layer.
    z = jnp.linspace(0.0, 400.0, 20)
    theta = 300.0 + 0.02 * z
    flux = jnp.full_like(z, -0.01)
    res = counter_gradient_diagnostic(flux, theta, z)
    assert not res.has_counter_gradient_layer
    assert res.layer_base_m is None
    assert bool(jnp.all(~res.is_counter_gradient))
    assert res.counter_gradient_fraction == pytest.approx(0.0)


def test_convective_mixed_layer_has_counter_gradient():
    # Classic CBL signature: a mixed layer that is very slightly STABLE
    # (dθ/dz > 0) yet carries an UPWARD heat flux (flux > 0) → up-gradient.
    z = jnp.linspace(0.0, 1000.0, 40)
    # weakly stable mid-layer gradient
    theta = 300.0 + 1.0e-3 * z
    flux = jnp.full_like(z, 0.05)  # upward heat flux throughout
    res = counter_gradient_diagnostic(flux, theta, z)
    assert res.has_counter_gradient_layer
    assert res.layer_base_m is not None and res.layer_top_m is not None
    assert res.layer_top_m > res.layer_base_m
    assert res.counter_gradient_fraction > 0.5


def test_single_cell_blip_rejected():
    # One isolated counter-gradient cell must NOT count as a layer (min 2 levels).
    z = jnp.linspace(0.0, 500.0, 25)
    theta = 300.0 + 0.02 * z          # stable
    flux = jnp.full_like(z, -0.01)     # down-gradient everywhere...
    flux = flux.at[10].set(+0.02)      # ...except one cell (up-gradient)
    res = counter_gradient_diagnostic(flux, theta, z, min_layer_levels=2)
    assert bool(res.is_counter_gradient[10])
    assert not res.has_counter_gradient_layer


def test_two_cell_layer_accepted():
    z = jnp.linspace(0.0, 500.0, 25)
    theta = 300.0 + 0.02 * z
    flux = jnp.full_like(z, -0.01)
    flux = flux.at[10].set(+0.02)
    flux = flux.at[11].set(+0.02)      # two contiguous up-gradient cells
    res = counter_gradient_diagnostic(flux, theta, z, min_layer_levels=2)
    assert res.has_counter_gradient_layer
    # first qualifying layer spans levels 10-11
    assert res.layer_base_m == pytest.approx(float(z[10]))
    assert res.layer_top_m == pytest.approx(float(z[11]))


def test_product_gate_masks_near_zero():
    # Near-zero gradient AND flux → product below gate → not counter-gradient.
    z = jnp.linspace(0.0, 500.0, 20)
    theta = jnp.full_like(z, 300.0) + 1.0e-8 * z  # essentially neutral
    flux = jnp.full_like(z, 1.0e-6)
    res = counter_gradient_diagnostic(flux, theta, z, product_gate=1.0e-7)
    assert not res.has_counter_gradient_layer


def test_diagnose_truth_snapshot():
    z = jnp.linspace(0.0, 1000.0, 30)
    theta = 300.0 + 1.0e-3 * z
    flux = jnp.full_like(z, 0.05)
    truth = LESTruth(
        case_name="cbl_test",
        heights_m=z,
        times_s=jnp.array([3600.0]),
        theta=theta,
        u=jnp.zeros_like(z),
        v=jnp.zeros_like(z),
        wtheta=flux,
    )
    res = diagnose_truth(truth)
    assert isinstance(res, CounterGradientResult)
    assert res.has_counter_gradient_layer


def test_diagnose_truth_rejects_prognostic_series():
    z = jnp.linspace(0.0, 1000.0, 5)
    truth = LESTruth(
        case_name="cbl_test",
        heights_m=z,
        times_s=jnp.array([0.0, 3600.0]),
        theta=jnp.zeros((2, 5)),   # multi-time → 2-D theta
        u=jnp.zeros((2, 5)),
        v=jnp.zeros((2, 5)),
        wtheta=jnp.zeros((2, 5)),
    )
    with pytest.raises(ValueError):
        diagnose_truth(truth)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        counter_gradient_diagnostic(
            jnp.zeros(5), jnp.zeros(6), jnp.zeros(5)
        )
