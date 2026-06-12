"""Unit tests for the CLUBB tridiagonal-solve adapter (now in ``clubb.py``).

Part of the fuller CLUBB port — see ``PORT_CLUBB.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    penta_solve,
    tridiag_solve,
)

_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


def _band_system(ng=3, n=10, seed=0):
    """Random diagonally-dominant CLUBB-band LHS [super, main, sub] + rhs."""
    rng = np.random.default_rng(seed)
    sup = rng.standard_normal((ng, n))
    sub = rng.standard_normal((ng, n))
    sup[:, -1] = 0.0   # no super on the top level
    sub[:, 0] = 0.0    # no sub on the bottom level
    # Diagonally dominant main diagonal -> well-conditioned, unique solution.
    mid = 2.0 + np.abs(sup) + np.abs(sub) + rng.random((ng, n))
    lhs = jnp.asarray(np.stack([sup, mid, sub]))      # (3, ng, n)
    rhs = jnp.asarray(rng.standard_normal((ng, n)))
    return lhs, rhs


def _apply_band(lhs, x):
    """Reconstruct lhs @ x for the [super, main, sub] band layout."""
    sup, mid, sub = lhs[0], lhs[1], lhs[2]
    out = mid * x
    out = out.at[:, 1:].add(sub[:, 1:] * x[:, :-1])
    out = out.at[:, :-1].add(sup[:, :-1] * x[:, 1:])
    return out


def test_solves_the_system_exactly():
    lhs, rhs = _band_system()
    x = tridiag_solve(lhs, rhs)
    assert x.shape == rhs.shape
    residual = jnp.max(jnp.abs(_apply_band(lhs, x) - rhs))
    assert float(residual) < 1e-12


def test_identity_lhs_returns_rhs():
    ng, n = 2, 6
    lhs = jnp.zeros((3, ng, n)).at[1].set(1.0)   # main diag = 1, off-diags 0
    rhs = jnp.asarray(np.random.default_rng(1).standard_normal((ng, n)))
    np.testing.assert_allclose(np.asarray(tridiag_solve(lhs, rhs)), np.asarray(rhs), rtol=1e-12)


def test_jit_and_grad_clean():
    lhs, rhs = _band_system()

    def loss(r):
        return jnp.sum(tridiag_solve(lhs, r) ** 2)

    assert jnp.isfinite(jax.jit(loss)(rhs))
    g = jax.grad(loss)(rhs)
    assert g.shape == rhs.shape and jnp.all(jnp.isfinite(g))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_matches_clubb_reference_solver():
    """Thomas (reuse) vs CLUBB tridiag_lu_solve: same solve, agree to round-off."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    from clubb_jax.src.CLUBB_core.tridiag_lu_solver import tridiag_lu_solve_jax

    lhs, rhs = _band_system(ng=4, n=20, seed=3)
    mine = tridiag_solve(lhs, rhs)
    ref = tridiag_lu_solve_jax(lhs, rhs)
    np.testing.assert_allclose(np.asarray(mine), np.asarray(ref), rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------------------
# Pentadiagonal solver
# ---------------------------------------------------------------------------

def _penta_system(ng=3, n=15, seed=0):
    """Random diagonally-dominant CLUBB-band [super2,super1,diag,sub1,sub2] + rhs."""
    rng = np.random.default_rng(seed)
    s2 = rng.standard_normal((ng, n))
    s1 = rng.standard_normal((ng, n))
    b1 = rng.standard_normal((ng, n))
    b2 = rng.standard_normal((ng, n))
    s2[:, -2:] = 0.0   # no 2nd super on the top two levels
    s1[:, -1] = 0.0    # no 1st super on the top level
    b1[:, 0] = 0.0     # no 1st sub on the bottom level
    b2[:, :2] = 0.0    # no 2nd sub on the bottom two levels
    diag = 4.0 + np.abs(s2) + np.abs(s1) + np.abs(b1) + np.abs(b2) + rng.random((ng, n))
    lhs = jnp.asarray(np.stack([s2, s1, diag, b1, b2]))   # (5, ng, n)
    rhs = jnp.asarray(rng.standard_normal((ng, n)))
    return lhs, rhs


def _apply_penta(lhs, x):
    s2, s1, diag, b1, b2 = lhs[0], lhs[1], lhs[2], lhs[3], lhs[4]
    out = diag * x
    out = out.at[:, :-1].add(s1[:, :-1] * x[:, 1:])
    out = out.at[:, :-2].add(s2[:, :-2] * x[:, 2:])
    out = out.at[:, 1:].add(b1[:, 1:] * x[:, :-1])
    out = out.at[:, 2:].add(b2[:, 2:] * x[:, :-2])
    return out


def test_penta_solves_exactly():
    lhs, rhs = _penta_system()
    x = penta_solve(lhs, rhs)
    assert x.shape == rhs.shape
    residual = jnp.max(jnp.abs(_apply_penta(lhs, x) - rhs))
    assert float(residual) < 1e-10


def test_penta_identity_returns_rhs():
    ng, n = 2, 9
    lhs = jnp.zeros((5, ng, n)).at[2].set(1.0)
    rhs = jnp.asarray(np.random.default_rng(2).standard_normal((ng, n)))
    np.testing.assert_allclose(np.asarray(penta_solve(lhs, rhs)), np.asarray(rhs), rtol=1e-12)


def test_penta_reduces_to_tridiag():
    """Zero 2nd bands -> penta solve equals the tridiagonal solve."""
    rng = np.random.default_rng(4)
    ng, n = 2, 12
    s1 = rng.standard_normal((ng, n))
    s1[:, -1] = 0.0
    b1 = rng.standard_normal((ng, n))
    b1[:, 0] = 0.0
    diag = 3.0 + np.abs(s1) + np.abs(b1) + rng.random((ng, n))
    rhs = jnp.asarray(rng.standard_normal((ng, n)))
    penta_lhs = jnp.asarray(np.stack([np.zeros((ng, n)), s1, diag, b1, np.zeros((ng, n))]))
    tri_lhs = jnp.asarray(np.stack([s1, diag, b1]))
    np.testing.assert_allclose(
        np.asarray(penta_solve(penta_lhs, rhs)), np.asarray(tridiag_solve(tri_lhs, rhs)),
        rtol=1e-10, atol=1e-12)


def test_penta_jit_and_grad():
    lhs, rhs = _penta_system()

    def loss(r):
        return jnp.sum(penta_solve(lhs, r) ** 2)

    assert jnp.isfinite(jax.jit(loss)(rhs))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(rhs)))


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_penta_matches_clubb_reference():
    """Bit-exact vs CLUBB penta_lu_solve_jax (verbatim algorithm port).

    The reference is ``jit``-compiled; JIT mine too so XLA fuses both identically
    (eager vs jitted float reassociation otherwise differs at round-off).
    """
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    from clubb_jax.src.CLUBB_core.penta_lu_solver import penta_lu_solve_jax

    lhs, rhs = _penta_system(ng=4, n=25, seed=7)
    np.testing.assert_array_equal(
        np.asarray(jax.jit(penta_solve)(lhs, rhs)), np.asarray(penta_lu_solve_jax(lhs, rhs)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
