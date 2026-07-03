"""LAPACK/FFI tridiagonal solve (scaling review 2026-06-13 lever #8).

``LEGOESM_TRIDIAG=lapack`` routes ``thomas_solve_batched`` to
``jax.lax.linalg.tridiagonal_solve`` on ANY backend — LAPACK ``gtsv`` on
CPU (the ~2000x win over the fori_loop legacy on the Ginsburg CPU nodes),
cuSPARSE on CUDA.  The primitive carries a JVP + transpose rule so it stays
reverse-mode differentiable (the end-to-end ``jax.grad`` requirement).

Pins: parity with the legacy Thomas to machine precision, finite + correct
gradients, the round-trip identity, and the env-var dispatch.  Skips
cleanly on a jaxlib too old to expose the CPU ``gtsv_ffi`` lowering.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.timestepping.tridiagonal import (  # noqa: E402
    _ffi_tridiagonal_solve,
    _thomas_solve_batched_legacy,
    thomas_solve_batched,
)


def _make_random_diagonally_dominant(shape, dtype, seed=0):
    """Random strongly diagonally-dominant tridiagonal system.  Supports
    real and complex dtypes (the spectral/nonhydrostatic solves use complex
    tridiagonals)."""
    is_complex = jnp.issubdtype(dtype, jnp.complexfloating)
    rng = jax.random.PRNGKey(seed)
    ks = jax.random.split(rng, 8)

    def mk(i_re, i_im, scale, shift=0.0):
        v = jax.random.normal(ks[i_re], shape) * scale + shift
        if is_complex:
            v = v + 1j * (jax.random.normal(ks[i_im], shape) * scale)
        return v.astype(dtype)

    a = mk(0, 4, 0.1)
    b = mk(1, 5, 1.0, 5.0)          # strong (real) diagonal dominance
    c = mk(2, 6, 0.1)
    d = mk(3, 7, 1.0)
    a = a.at[..., 0].set(0.0)
    c = c.at[..., -1].set(0.0)
    return a, b, c, d


def _ffi_available() -> bool:
    """True if the RAW ``jax.lax.linalg.tridiagonal_solve`` lowers on this
    backend (recent jaxlib exposes the CPU LAPACK ``gtsv_ffi``; older ones
    do not).  Probe the PRIMITIVE directly — not the wrapper under test — so
    a wrapper regression fails the tests loudly instead of silently skipping
    the whole file (codex 2026-06-13)."""
    try:
        from jax.lax.linalg import tridiagonal_solve
    except (ImportError, AttributeError):
        return False
    try:
        dl = jnp.array([[0.0, 1.0]])
        d = jnp.array([[2.0, 2.0]])
        du = jnp.array([[1.0, 0.0]])
        b = jnp.array([[1.0, 1.0]])
        out = tridiagonal_solve(dl, d, du, b[..., None])
        out.block_until_ready()
        return True
    except Exception:
        # Missing lowering / unavailable backend -> skip (the wrapper is not
        # in this probe, so a wrapper bug cannot mask itself here).
        return False


pytestmark = pytest.mark.skipif(
    not _ffi_available(),
    reason="jax.lax.linalg.tridiagonal_solve FFI lowering unavailable "
           "on this backend/jaxlib",
)


@pytest.mark.parametrize("n_sys", [4, 8, 29, 30, 32, 60, 64])
@pytest.mark.parametrize("batch_shape", [(1,), (16,), (6, 4, 12)])
def test_lapack_matches_legacy_thomas_fp64(n_sys, batch_shape):
    shape = batch_shape + (n_sys,)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)
    x_ffi = _ffi_tridiagonal_solve(a, b, c, d)
    x_leg = _thomas_solve_batched_legacy(a, b, c, d)
    diff = float(jnp.max(jnp.abs(x_ffi - x_leg)))
    assert diff < 1.0e-12, f"n={n_sys}, batch={batch_shape}: diff={diff}"


@pytest.mark.parametrize("n_sys", [29, 30, 32])
def test_lapack_residual_fp64(n_sys):
    shape = (8, n_sys)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)
    x = _ffi_tridiagonal_solve(a, b, c, d)
    mid = (
        a[..., 1:-1] * x[..., :-2]
        + b[..., 1:-1] * x[..., 1:-1]
        + c[..., 1:-1] * x[..., 2:]
        - d[..., 1:-1]
    )
    residual = float(jnp.max(jnp.abs(mid)))
    assert residual < 1.0e-13, f"n={n_sys}: residual={residual}"


def test_lapack_ad_grad_matches_legacy():
    """Reverse-mode grad through the FFI solve is finite AND matches the
    legacy Thomas grad for ALL FOUR inputs (a, b, c, d) — the registered
    JVP+transpose rule is correct for the coefficients too, not just the
    RHS, so training paths that backprop through K-build → implicit vertical
    solve are unaffected by opting into the LAPACK backend (codex
    2026-06-13: prior test only differentiated d)."""
    shape = (4, 5, 29)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)

    def _loss(fn):
        def _l(abcd):
            aa, bb, cc, dd = abcd
            return jnp.sum(fn(aa, bb, cc, dd) ** 2)
        return _l

    g_ffi = jax.grad(_loss(_ffi_tridiagonal_solve))((a, b, c, d))
    g_leg = jax.grad(_loss(_thomas_solve_batched_legacy))((a, b, c, d))
    for gf, gl, name in zip(g_ffi, g_leg, "abcd"):
        assert bool(jnp.all(jnp.isfinite(gf))), f"FFI grad {name} non-finite"
        diff = float(jnp.max(jnp.abs(gf - gl)))
        assert diff < 1.0e-8, f"FFI grad {name} disagrees with legacy: {diff}"


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_lapack_parity_and_residual_by_dtype(dtype):
    """Parity vs legacy + scaled residual for real AND complex systems —
    LAPACK gtsv success on real data does not prove the complex lowering
    works (spectral_nh uses complex tridiagonals) (codex 2026-06-13)."""
    shape = (8, 30)
    a, b, c, d = _make_random_diagonally_dominant(shape, dtype)
    x_ffi = _ffi_tridiagonal_solve(a, b, c, d)
    x_leg = _thomas_solve_batched_legacy(a, b, c, d)
    assert float(jnp.max(jnp.abs(x_ffi - x_leg))) < 1.0e-11, "dtype parity"
    # scaled (relative) residual — robust to gtsv's partial pivoting
    mid = (
        a[..., 1:-1] * x_ffi[..., :-2]
        + b[..., 1:-1] * x_ffi[..., 1:-1]
        + c[..., 1:-1] * x_ffi[..., 2:]
        - d[..., 1:-1]
    )
    rel = float(jnp.max(jnp.abs(mid)) / jnp.max(jnp.abs(d)))
    assert rel < 1.0e-13, f"{dtype}: scaled residual {rel}"


def test_lapack_zero_size_batch():
    """A zero-size leading batch must not crash — the dispatch flattens via
    ``math.prod(spatial_shape)`` which is 0 here, not clamped to 1 (codex
    2026-06-13 MED: the old ``max(1, …)`` reshaped a size-0 array into
    (1, n_sys) and raised)."""
    shape = (0, 8)
    z = jnp.zeros(shape)
    x = _ffi_tridiagonal_solve(z, jnp.ones(shape), z, z)
    assert x.shape == shape


def test_lapack_round_trip_identity_system():
    rng = jax.random.PRNGKey(42)
    shape = (8, 29)
    d = jax.random.normal(rng, shape, dtype=jnp.float64)
    a = jnp.zeros(shape)
    b = jnp.ones(shape)
    c = jnp.zeros(shape)
    x = _ffi_tridiagonal_solve(a, b, c, d)
    assert float(jnp.max(jnp.abs(x - d))) < 1.0e-14, "Identity solve failed"


def test_lapack_dispatch_via_env(monkeypatch):
    """LEGOESM_TRIDIAG=lapack routes thomas_solve_batched to the FFI path
    and returns the correct solution (env read at trace time)."""
    shape = (8, 30)
    a, b, c, d = _make_random_diagonally_dominant(shape, jnp.float64)
    monkeypatch.setenv("LEGOESM_TRIDIAG", "lapack")
    x = thomas_solve_batched(a, b, c, d)
    x_leg = _thomas_solve_batched_legacy(a, b, c, d)
    assert float(jnp.max(jnp.abs(x - x_leg))) < 1.0e-12
