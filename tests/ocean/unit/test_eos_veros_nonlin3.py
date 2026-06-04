"""Tests for ``eos="veros_nonlin3"`` — Veros's ``eq_of_state_type=3``.

Bit-for-bit verification against the formula in
``veros/core/density/nonlinear_eq3.py``. This EOS is what
``veros.setups.acc.ACCSetup`` uses, so tier-2 PGF / density comparison
on the ACC recipe depends on matching it exactly.

Veros source (relevant lines)::

    rho0 = 1024.0
    theta0 = 283.0 - 273.15
    S0 = 35.0
    betaT = 1.67e-4
    betaTs = 1e-5 / 2.0
    betaS = 0
    grav = 9.81
    z0 = 0.0

    def nonlin3_eq_of_state_rho(sa, ct):
        thetas = ct - theta0
        return -(betaT * thetas + betaTs * thetas**2
                 - betaS * (sa - S0)) * rho0

The Veros function returns the density **anomaly** (rho - rho0); legoESM's
``veros_nonlin3_eos`` returns the in-situ density (rho_0 added back) to
match the legoESM EOS API. Tests account for this offset.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.eos import (
    VerosNonlin3Config,
    make_eos_fn,
    veros_nonlin3_eos,
)

jax.config.update("jax_enable_x64", True)


# Veros source-of-truth coefficients.
_VEROS_RHO0 = 1024.0
_VEROS_THETA0 = 283.0 - 273.15
_VEROS_S0 = 35.0
_VEROS_BETAT = 1.67e-4
_VEROS_BETATS = 1.0e-5 / 2.0
_VEROS_BETAS = 0.0


def _veros_nonlin3_anomaly(sa, ct):
    """Veros's exact nonlin3_eq_of_state_rho formula (returns anomaly)."""
    thetas = ct - _VEROS_THETA0
    return -(
        _VEROS_BETAT * thetas
        + _VEROS_BETATS * thetas ** 2
        - _VEROS_BETAS * (sa - _VEROS_S0)
    ) * _VEROS_RHO0


def test_default_config_matches_veros_source():
    """Default ``VerosNonlin3Config`` carries Veros's module-level
    constants verbatim."""
    cfg = VerosNonlin3Config()
    assert cfg.rho_0 == _VEROS_RHO0
    assert cfg.theta0_C == _VEROS_THETA0
    assert cfg.S0 == _VEROS_S0
    assert cfg.betaT == _VEROS_BETAT
    assert cfg.betaTs == _VEROS_BETATS
    assert cfg.betaS == _VEROS_BETAS


def test_bit_exact_against_veros_formula():
    """legoESM ``veros_nonlin3_eos`` minus rho_0 must equal Veros's
    ``nonlin3_eq_of_state_rho`` (density anomaly) to machine
    precision on a representative T, S grid."""
    T = jnp.linspace(-2.0, 30.0, 17)
    S = jnp.linspace(30.0, 38.0, 9)
    Tg, Sg = jnp.meshgrid(T, S, indexing="ij")
    p = jnp.zeros_like(Tg)   # pressure irrelevant for this EOS
    rho = veros_nonlin3_eos(Tg, Sg, p)
    anom_lego = rho - _VEROS_RHO0
    anom_veros = _veros_nonlin3_anomaly(np.asarray(Sg), np.asarray(Tg))
    # Tolerance accounts for ~5 float64 ops of round-off (~5·eps relative).
    np.testing.assert_allclose(
        np.asarray(anom_lego), anom_veros,
        rtol=1e-12, atol=1e-12,
    )


def test_zero_dependency_on_salt():
    """With Veros's default ``betaS = 0``, salinity must have ZERO
    effect on density — Veros ACC relies on this (initialises S to
    35 PSU uniform and treats it as a passive tracer)."""
    T = jnp.asarray(10.0)
    rho_a = float(veros_nonlin3_eos(T, jnp.asarray(30.0), jnp.asarray(0.0)))
    rho_b = float(veros_nonlin3_eos(T, jnp.asarray(40.0), jnp.asarray(0.0)))
    assert rho_a == rho_b


def test_zero_dependency_on_pressure():
    """``betaP = 0`` and no pressure terms in the polynomial — ``p`` is
    accepted but ignored."""
    rho_a = float(veros_nonlin3_eos(jnp.asarray(10.0), jnp.asarray(35.0),
                                     jnp.asarray(0.0)))
    rho_b = float(veros_nonlin3_eos(jnp.asarray(10.0), jnp.asarray(35.0),
                                     jnp.asarray(5e7)))  # 500 dbar
    assert rho_a == rho_b


def test_at_theta0_equals_rho0():
    """At T = theta0, S = S0, the anomaly is zero so rho = rho_0."""
    rho = float(veros_nonlin3_eos(
        jnp.asarray(_VEROS_THETA0),
        jnp.asarray(_VEROS_S0),
        jnp.asarray(0.0),
    ))
    np.testing.assert_allclose(rho, _VEROS_RHO0, atol=1e-12)


def test_density_increases_with_cooling():
    """Cooling water densifies (d_rho/dT < 0 at T > theta0/2 by Veros
    nonlin3's quadratic form)."""
    rho_warm = float(veros_nonlin3_eos(jnp.asarray(15.0), jnp.asarray(35.0),
                                        jnp.asarray(0.0)))
    rho_cold = float(veros_nonlin3_eos(jnp.asarray(5.0), jnp.asarray(35.0),
                                        jnp.asarray(0.0)))
    assert rho_cold > rho_warm


def test_dispatch_via_make_eos_fn():
    fn = make_eos_fn("veros_nonlin3")
    rho = float(fn(jnp.asarray(_VEROS_THETA0), jnp.asarray(_VEROS_S0),
                   jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, _VEROS_RHO0, atol=1e-12)


def test_dispatch_accepts_custom_config():
    """Passing a custom ``VerosNonlin3Config`` (e.g., non-zero betaS)
    flows through make_eos_fn."""
    cfg = VerosNonlin3Config(betaS=1e-3)
    fn = make_eos_fn("veros_nonlin3", eos_veros_nonlin3=cfg)
    rho_a = float(fn(jnp.asarray(10.0), jnp.asarray(30.0), jnp.asarray(0.0)))
    rho_b = float(fn(jnp.asarray(10.0), jnp.asarray(40.0), jnp.asarray(0.0)))
    # With non-zero betaS, salinity must now influence density.
    assert rho_a != rho_b


def test_differentiable_wrt_T():
    """d_rho/d_T = -(betaT + 2·betaTs·(T - theta0)) · rho_0."""
    T0 = jnp.asarray(15.0)
    fn = lambda T: veros_nonlin3_eos(T, jnp.asarray(35.0), jnp.asarray(0.0))
    grad = jax.grad(fn)(T0)
    thetas = float(T0) - _VEROS_THETA0
    expected = -(_VEROS_BETAT + 2 * _VEROS_BETATS * thetas) * _VEROS_RHO0
    np.testing.assert_allclose(float(grad), expected, rtol=1e-12)
