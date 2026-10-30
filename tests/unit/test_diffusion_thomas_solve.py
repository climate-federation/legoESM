"""Direct tests for legoesm.timestepping.tridiagonal.diffusion_thomas_solve:
the levels-first backward-Euler diffusion solve that builds its bands inside
the sweep. Checked against thomas_solve on explicitly built bands (value and
gradient), against finite differences (first and second order), and through
the ocean wrapper that now uses it."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm.timestepping.tridiagonal import diffusion_thomas_solve, thomas_solve


def _inputs(n=7, cols=(5,), per_level_extra=False, seed=0):
    rng = np.random.default_rng(seed)
    dtf = jnp.asarray(rng.uniform(0.1, 30.0, (n - 1, *cols)))
    inv = jnp.asarray(1.0 / rng.uniform(5.0, 50.0, (n, *cols)))
    extra = (jnp.asarray(rng.uniform(0.0, 0.5, (n, *cols))) if per_level_extra
             else jnp.asarray(0.2))
    rhs = jnp.asarray(rng.standard_normal((n, *cols)))
    return dtf, inv, extra, rhs


def _reference(dtf, inv, extra, rhs):
    """thomas_solve on the explicitly built bands, levels LAST."""
    zero = jnp.zeros_like(dtf[:1])
    alpha = jnp.concatenate([zero, dtf]) * inv
    beta = jnp.concatenate([dtf, zero]) * inv
    mv = lambda v: jnp.moveaxis(v, 0, -1)
    x = thomas_solve(mv(-alpha), mv(1.0 + alpha + beta + extra), mv(-beta), mv(rhs))
    return jnp.moveaxis(x, -1, 0)


@pytest.mark.parametrize("per_level_extra", [False, True])
@pytest.mark.parametrize("cols", [(5,), (2, 3), ()])
def test_matches_thomas_on_built_bands(per_level_extra, cols):
    args = _inputs(cols=cols, per_level_extra=per_level_extra)
    np.testing.assert_allclose(diffusion_thomas_solve(*args), _reference(*args),
                               rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("per_level_extra", [False, True])
def test_gradient_matches_thomas_path(per_level_extra):
    args = _inputs(per_level_extra=per_level_extra, seed=1)
    w = jnp.asarray(np.random.default_rng(2).standard_normal(args[3].shape))
    g_new = jax.grad(lambda *a: jnp.sum(w * diffusion_thomas_solve(*a)),
                     argnums=(0, 1, 2, 3))(*args)
    g_ref = jax.grad(lambda *a: jnp.sum(w * _reference(*a)),
                     argnums=(0, 1, 2, 3))(*args)
    for gn, gr in zip(g_new, g_ref):
        np.testing.assert_allclose(gn, gr, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("per_level_extra", [False, True])
def test_check_grads_second_order(per_level_extra):
    check_grads(diffusion_thomas_solve,
                _inputs(n=5, cols=(3,), per_level_extra=per_level_extra, seed=3),
                order=2, modes=["rev"], atol=1e-6, rtol=1e-6, eps=1e-6)


def test_forward_over_reverse():
    args = _inputs(n=5, cols=(3,), seed=4)
    f = lambda rhs: jnp.sum(diffusion_thomas_solve(args[0], args[1], args[2], rhs) ** 2)
    v = jnp.ones_like(args[3])
    _, hv = jax.jvp(jax.grad(f), (args[3],), (v,))
    eps = 1e-6
    fd = (jax.grad(f)(args[3] + eps * v) - jax.grad(f)(args[3] - eps * v)) / (2 * eps)
    np.testing.assert_allclose(hv, fd, rtol=1e-6, atol=1e-8)


def test_refuses_single_level():
    with pytest.raises(ValueError, match="at least 2 levels"):
        diffusion_thomas_solve(jnp.zeros((0, 3)), jnp.ones((1, 3)),
                               jnp.asarray(0.0), jnp.ones((1, 3)))


def test_ocean_wrapper_matches_built_tridiag_solve():
    """implicit_vertical_diffusion_ocean (fused path) == thomas_solve on the
    bands _build_implicit_tridiag builds, values and gradients."""
    from legoesm.ocean.physics.vertical_mixing import implicit_solver as vm
    rng = np.random.default_rng(5)
    field = jnp.asarray(rng.standard_normal((11, 9)))
    K = jnp.asarray(rng.uniform(0.0, 1e-2, (11, 8)))
    dz = jnp.asarray(rng.uniform(5.0, 80.0, (11, 9)))
    dzh = 0.5 * (dz[:, 1:] + dz[:, :-1])
    extra = jnp.asarray(rng.uniform(0.0, 0.1, (11, 9)))

    def ref(field, K):
        return thomas_solve(*vm._build_implicit_tridiag(field, K, dz, dzh, 600.0,
                                                         extra_diag=extra))

    def new(field, K):
        return vm.implicit_vertical_diffusion_ocean(field, K, dz, dzh, 600.0,
                                                    extra_diag=extra)

    np.testing.assert_allclose(new(field, K), ref(field, K), rtol=1e-13, atol=1e-14)
    w = jnp.asarray(rng.standard_normal(field.shape))
    gn = jax.grad(lambda f, k: jnp.sum(w * new(f, k)), argnums=(0, 1))(field, K)
    gr = jax.grad(lambda f, k: jnp.sum(w * ref(f, k)), argnums=(0, 1))(field, K)
    for a, b in zip(gn, gr):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-13)


def test_python_float_extra_and_bad_extra_shape():
    dtf, inv, _, rhs = _inputs(seed=6)
    np.testing.assert_allclose(diffusion_thomas_solve(dtf, inv, 0.2, rhs),
                               _reference(dtf, inv, jnp.asarray(0.2), rhs), rtol=1e-13)
    g = jax.grad(lambda e: jnp.sum(diffusion_thomas_solve(dtf, inv, e, rhs)))(0.2)
    assert np.isfinite(g)
    with pytest.raises(ValueError, match="scalar or shaped like rhs"):
        diffusion_thomas_solve(dtf, inv, jnp.ones((rhs.shape[0], 1)), rhs)


def test_f32_coefficients_with_f64_rhs():
    """f32 coefficients with an f64 RHS: bands rounded in f32, solve in f64.
    Agreement with thomas_solve on prebuilt f32 bands is to f32 rounding
    (XLA may contract the band arithmetic differently in the two paths)."""
    dtf, inv, extra, rhs = _inputs(seed=7)
    f32 = jnp.float32
    args = (dtf.astype(f32), inv.astype(f32), extra.astype(f32), rhs)
    x = diffusion_thomas_solve(*args)
    assert x.dtype == jnp.float64
    np.testing.assert_allclose(x, _reference(*args), rtol=1e-6, atol=1e-6)
