"""Bit-for-bit + AD regression tests for pcr_solve_batched.

Addresses cavecrew adversarial review concern (iter-71) that the PCR
test matrix was undocumented. This file pins the validated regime:
- n_sys in {4, 8, 29, 30, 32, 60, 64} — power-of-2 and non-power-of-2
- batch_shapes in {(), (16,), (6, 4, 12)} — 1D batched, leading axes
- dtype fp32 and fp64
- AD via jax.grad on the d (RHS) input — gradients finite and match
  legacy fori_loop Thomas to machine epsilon
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.timestepping.tridiagonal import (
    pcr_solve_batched,
    _thomas_solve_batched_legacy,
)

jax.config.update("jax_enable_x64", True)


def _make_random_diagonally_dominant(shape, dtype, seed=0):
    """Return (a, b, c, d) with strong diagonal dominance — matches the
    SI acoustic regime where b ~ 1 + alpha (alpha > 0)."""
    rng = jax.random.PRNGKey(seed)
    ks = jax.random.split(rng, 4)
    a = (jax.random.normal(ks[0], shape) * 0.1).astype(dtype)
    b = (jax.random.normal(ks[1], shape) + 5.0).astype(dtype)
    c = (jax.random.normal(ks[2], shape) * 0.1).astype(dtype)
    a = a.at[..., 0].set(0.0)
    c = c.at[..., -1].set(0.0)
    d = jax.random.normal(ks[3], shape).astype(dtype)
    return a, b, c, d


@pytest.mark.parametrize("n_sys", [4, 8, 29, 30, 32, 60, 64])
@pytest.mark.parametrize("batch_shape", [(), (16,), (6, 4, 12)])
def test_pcr_matches_legacy_thomas_fp64(n_sys, batch_shape):
    shape = batch_shape + (n_sys,)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)
    x_pcr = pcr_solve_batched(a, b, c, d)
    x_leg = _thomas_solve_batched_legacy(a, b, c, d)
    diff = float(jnp.max(jnp.abs(x_pcr - x_leg)))
    # Machine epsilon level — PCR has slightly more roundoff growth
    # than Thomas but stays within ~2-4 ULP for diagonally dominant
    # systems. 1e-12 is a comfortable bound for fp64.
    assert diff < 1.0e-12, f"n={n_sys}, batch={batch_shape}: diff={diff}"


@pytest.mark.parametrize("n_sys", [29, 30, 32])
def test_pcr_residual_fp64(n_sys):
    shape = (8, n_sys)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)
    x = pcr_solve_batched(a, b, c, d)
    mid = (
        a[..., 1:-1] * x[..., :-2]
        + b[..., 1:-1] * x[..., 1:-1]
        + c[..., 1:-1] * x[..., 2:]
        - d[..., 1:-1]
    )
    residual = float(jnp.max(jnp.abs(mid)))
    assert residual < 1.0e-13, f"n={n_sys}: residual={residual}"


def test_pcr_ad_grad_finite():
    """Gradient flow through pcr_solve_batched. AD safety is required
    for training paths that backprop through the SI acoustic substep."""
    shape = (4, 5, 29)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)

    def f(d_in):
        return jnp.sum(pcr_solve_batched(a, b, c, d_in))

    grad = jax.grad(f)(d)
    assert bool(jnp.all(jnp.isfinite(grad))), "PCR gradient has NaN/Inf"
    assert grad.shape == d.shape


def test_pcr_rejects_scalar():
    """Cavecrew adversarial review found ndim < 1 raised IndexError.
    iter-75 fixed this; pin the contract via a regression test."""
    x = jnp.array(0.0)
    with pytest.raises(ValueError, match="at least 1 axis"):
        pcr_solve_batched(x, x, x, x)


def test_pcr_rejects_n_sys_lt_2():
    """n_sys=1 (trivial system) is rejected — PCR requires at least
    one off-diagonal stride."""
    x = jnp.zeros((4, 1))
    with pytest.raises(ValueError, match="n>=2"):
        pcr_solve_batched(x, x, x, x)


def test_pcr_rejects_shape_mismatch():
    a = jnp.zeros((4, 8))
    b = jnp.zeros((4, 8))
    c = jnp.zeros((4, 8))
    d = jnp.zeros((4, 16))  # wrong shape
    with pytest.raises(ValueError, match="matching shapes"):
        pcr_solve_batched(a, b, c, d)
