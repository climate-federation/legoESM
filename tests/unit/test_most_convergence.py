"""MOST fixed-point convergence residual (``return_convergence``).

The Obukhov-length iteration is a fixed ``n_iter`` ``fori_loop`` (AD-safe).
``return_convergence=True`` exposes the final-iteration relative ``u*`` change
so callers/tests can verify the fixed count actually converged and flag the
strong-stability columns where it did not — the gap vs a tolerance-based root
solve.  The diagnostic must NOT change the fluxes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import compute_most_fluxes


def _args(T_sfc):
    """A single air-sea column; T_sfc sets the stability (cold sfc = stable)."""
    f = lambda v: jnp.asarray(v)
    return dict(
        u_rel=f(6.0), v_rel=f(1.0), T_atm=f(290.0), q_atm=f(0.008),
        T_sfc=f(T_sfc), q_sfc=f(0.010), rho=f(1.2),
        z_ref=10.0, z0_init=1e-4, scheme="coare3",
    )


def test_default_signature_byte_identical():
    """return_convergence=False keeps the 5-tuple and the exact fluxes; the
    True call's first five outputs are bit-identical (residual is diagnostic)."""
    a = _args(292.0)
    base = compute_most_fluxes(**a, n_iter=5)
    assert len(base) == 5
    conv = compute_most_fluxes(**a, n_iter=5, return_convergence=True)
    assert len(conv) == 6
    for x, y in zip(base, conv[:5]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_residual_finite_nonnegative():
    _, _, _, _, _, resid = compute_most_fluxes(
        **_args(285.0), n_iter=5, return_convergence=True)
    assert jnp.isfinite(resid)
    assert float(resid) >= 0.0


def test_more_iterations_reduce_residual():
    """The residual monotonically shrinks as the fixed iteration count grows —
    i.e. it genuinely measures fixed-point convergence."""
    a = _args(283.0)   # stable (warm air over cold water) — slower to converge
    r2 = float(compute_most_fluxes(**a, n_iter=2, return_convergence=True)[-1])
    r5 = float(compute_most_fluxes(**a, n_iter=5, return_convergence=True)[-1])
    r20 = float(compute_most_fluxes(**a, n_iter=20, return_convergence=True)[-1])
    assert r2 >= r5 >= r20
    assert r20 < 1e-3          # well converged after 20 iterations
    assert r20 < r2            # strictly improved


def test_converged_flux_matches_more_iterations():
    """Once converged (large n_iter), the residual is tiny AND the flux stops
    moving — the residual is a faithful proxy for 'fluxes have settled'."""
    a = _args(283.0)
    *f20, r20 = compute_most_fluxes(**a, n_iter=20, return_convergence=True)
    *f40, r40 = compute_most_fluxes(**a, n_iter=40, return_convergence=True)
    assert float(r40) <= float(r20)
    # u* (last flux element) has settled to < the residual tolerance
    np.testing.assert_allclose(np.asarray(f20[4]), np.asarray(f40[4]), rtol=1e-3)


def test_return_convergence_with_return_2m():
    """Combined flags append T_2m then the residual (documented order)."""
    out = compute_most_fluxes(
        **_args(288.0), n_iter=5, return_2m=True, return_convergence=True)
    assert len(out) == 7
    tau_x, tau_y, shflx, lhflx, ustar, T_2m, resid = out
    assert jnp.isfinite(T_2m) and jnp.isfinite(resid)


def test_differentiable_through_convergence_path():
    """Adding the residual carry leaf must not break reverse-mode AD — both
    for the fluxes and for the residual itself (the abs/divide path)."""
    def flux_loss(T_sfc):
        tau_x, _, shflx, lhflx, _, resid = compute_most_fluxes(
            **{**_args(285.0), "T_sfc": T_sfc}, n_iter=6,
            return_convergence=True)
        return jnp.sum(shflx ** 2 + lhflx ** 2 + tau_x ** 2)
    assert jnp.isfinite(jax.grad(flux_loss)(jnp.asarray(285.0)))

    # Differentiate THROUGH the residual (the abs/max/divide chain) — it must
    # stay AD-finite too, not just be carried dead.
    def resid_loss(T_sfc):
        return compute_most_fluxes(
            **{**_args(283.0), "T_sfc": T_sfc}, n_iter=4,
            return_convergence=True)[-1]
    assert jnp.isfinite(jax.grad(resid_loss)(jnp.asarray(283.0)))


@pytest.mark.parametrize("scheme", ("most", "coare3", "large_yeager"))
def test_residual_available_for_every_most_scheme(scheme):
    out = compute_most_fluxes(
        **{**_args(286.0), "scheme": scheme}, n_iter=8, return_convergence=True)
    assert len(out) == 6 and float(out[-1]) >= 0.0
