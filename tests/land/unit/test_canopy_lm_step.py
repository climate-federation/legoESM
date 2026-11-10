"""The canopy Newton solver's damped Levenberg-Marquardt step
(legoesm.land.canopy.solver.damped_lm_step), ported from legoESM main
0e6e65a4f9 (#2398): Q^T rhs from the packed Householder reflectors, never
forming Q."""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.solver import damped_lm_step


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.mark.parametrize("lam", [0.0, 1e-3, 1.0, 1e4])
def test_step_equals_the_damped_normal_equations(lam):
    """dz solves (J^T J + lam I) dz = -J^T F (the augmented least-squares
    problem), and the returned norm is ||F + J dz||^2."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        J = jnp.asarray(rng.normal(size=(6, 6)) * np.logspace(0, 3, 6))   # cond ~1e3+
        F = jnp.asarray(rng.normal(size=6))
        dz, nsq = damped_lm_step(J, F, jnp.asarray(lam))
        ref = jnp.linalg.solve(J.T @ J + lam * jnp.eye(6), -J.T @ F)
        np.testing.assert_allclose(np.asarray(dz), np.asarray(ref), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(float(nsq), float(jnp.sum((F + J @ dz) ** 2)), rtol=1e-12)


def test_step_never_forms_q():
    """The explicit-Q path is the ``qr`` primitive (geqrf + householder
    product); the step must carry only the packed factorisation (geqrf)."""
    J = jnp.eye(6) + 0.1
    text = str(jax.make_jaxpr(damped_lm_step)(J, jnp.ones(6), jnp.asarray(0.5)))
    assert "geqrf" in text
    assert re.search(r"\bqr\[", text) is None
    assert "householder_product" not in text
