"""The CLUBB banded solves batched along the column axis equal the per-system
solves bit for bit (both solvers are elementwise across the batch), and the
batched helpers refuse mismatched shapes."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812


def _systems(rng, n_sys, ncol, n, nb):
    out = []
    for _ in range(n_sys):
        lhs = jnp.asarray(rng.standard_normal((nb, ncol, n)))
        # diagonally dominant main band so every system is well posed
        lhs = lhs.at[nb // 2].add(10.0)
        rhs = jnp.asarray(rng.standard_normal((ncol, n)))
        out.append((lhs, rhs))
    return out


@pytest.mark.parametrize("n_sys", [1, 2, 3])
def test_batched_tridiag_equals_sequential(n_sys):
    rng = np.random.default_rng(1)
    systems = _systems(rng, n_sys, ncol=13, n=33, nb=3)
    batched = C._batched_tridiag_solve(systems)
    for (lhs, rhs), x in zip(systems, batched):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(C.tridiag_solve(lhs, rhs)))
    jitted = jax.jit(C._batched_tridiag_solve)(systems)
    for (lhs, rhs), x in zip(systems, jitted):
        np.testing.assert_array_equal(
            np.asarray(x), np.asarray(jax.jit(C.tridiag_solve)(lhs, rhs)))


@pytest.mark.parametrize("n_sys", [1, 2])
def test_batched_penta_equals_sequential(n_sys):
    rng = np.random.default_rng(2)
    systems = _systems(rng, n_sys, ncol=11, n=65, nb=5)
    batched = C._batched_penta_solve(systems)
    for (lhs, rhs), x in zip(systems, batched):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(C.penta_solve(lhs, rhs)))
    jitted = jax.jit(C._batched_penta_solve)(systems)
    for (lhs, rhs), x in zip(systems, jitted):
        np.testing.assert_array_equal(
            np.asarray(x), np.asarray(jax.jit(C.penta_solve)(lhs, rhs)))


def test_batched_solve_refuses_mismatched_columns():
    rng = np.random.default_rng(3)
    (lhs, rhs), = _systems(rng, 1, ncol=5, n=9, nb=3)
    with pytest.raises(ValueError):
        C._batched_tridiag_solve([(lhs, rhs), (lhs[:, :3], rhs[:3])])
