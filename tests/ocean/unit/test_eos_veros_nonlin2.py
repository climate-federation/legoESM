"""Tests for ``eos="veros_nonlin2"`` — Veros's ``eq_of_state_type=3``.

Verified bit-for-bit against
``veros/core/density/nonlinear_eq2.py``. This is what Veros's
``get_rho`` dispatcher actually calls for ``eq_of_state_type=3``
(despite the ``nonlinear_eq3.py`` file name on the type=4 EOS).
Vallis 2008 nonlinear seawater EOS with quadratic-in-T, linear-S,
and pressure-coupled-T-and-z terms.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.eos import (
    VerosNonlin2Config,
    make_eos_fn,
    veros_nonlin2_eos,
)

jax.config.update("jax_enable_x64", True)


# Source-of-truth coefficients from veros/core/density/nonlinear_eq2.py.
_RHO0 = 1024.0
_THETA0 = 283.0 - 273.15
_S0 = 35.0
_GRAV = 9.81
_CS0 = 1490.0
_BETAT = 1.67e-4
_BETATS = 1.0e-5
_BETAS = 0.78e-3
_GAMMAS = 1.1e-8
_Z0 = 0.0


def _veros_nonlin2_anomaly_depth_m(sa, ct, depth_m):
    """Verbatim port of Veros's ``nonlin2_eq_of_state_rho`` —
    Veros passes ``press = abs(zt)`` (depth in meters)."""
    zz = -depth_m - _Z0
    thetas = ct - _THETA0
    return -(
        _GRAV * zz / _CS0**2
        + _BETAT * (1 - _GAMMAS * _GRAV * zz * _RHO0) * thetas
        + _BETATS / 2 * thetas**2
        - _BETAS * (sa - _S0)
    ) * _RHO0


def test_default_config_matches_veros_source():
    cfg = VerosNonlin2Config()
    assert cfg.rho_0 == _RHO0
    assert cfg.theta0_C == _THETA0
    assert cfg.S0 == _S0
    assert cfg.grav == _GRAV
    assert cfg.cs0 == _CS0
    assert cfg.betaT == _BETAT
    assert cfg.betaTs == _BETATS
    assert cfg.betaS == _BETAS
    assert cfg.gammas == _GAMMAS


def test_bit_exact_against_veros_formula():
    """Cover a representative T × S × depth grid.

    legoESM ``veros_nonlin2_eos`` takes pressure in Pa and converts
    internally to depth_m = p / (rho_0 g). The reference Veros formula
    uses depth_m directly. We construct the comparison so both
    receive the same depth_m: legoESM gets p_pa = depth_m · rho_0 g.
    """
    T = jnp.linspace(-2.0, 30.0, 9)
    S = jnp.linspace(30.0, 38.0, 5)
    DEPTH_M = jnp.linspace(0.0, 4000.0, 6)
    Tg, Sg, Dg = jnp.meshgrid(T, S, DEPTH_M, indexing="ij")
    Pg_pa = Dg * _RHO0 * _GRAV
    rho = veros_nonlin2_eos(Tg, Sg, Pg_pa)
    anom_lego = rho - _RHO0
    anom_veros = _veros_nonlin2_anomaly_depth_m(
        np.asarray(Sg), np.asarray(Tg), np.asarray(Dg),
    )
    np.testing.assert_allclose(
        np.asarray(anom_lego), anom_veros,
        rtol=1e-12, atol=1e-12,
    )


def test_salinity_dependence_active():
    """nonlin2 has βS = 7.8e-4 (≠0), so salinity must affect density."""
    rho_a = float(veros_nonlin2_eos(jnp.asarray(10.0), jnp.asarray(30.0),
                                      jnp.asarray(0.0)))
    rho_b = float(veros_nonlin2_eos(jnp.asarray(10.0), jnp.asarray(40.0),
                                      jnp.asarray(0.0)))
    assert rho_a != rho_b
    # ΔS = +10 → Δρ ≈ +βS·ρ₀·10 = +7.99 kg/m³
    expected = _BETAS * _RHO0 * 10.0
    np.testing.assert_allclose(rho_b - rho_a, expected, rtol=1e-4)


def test_pressure_dependence_active():
    """Unlike nonlin3, nonlin2 has pressure dependence — increasing
    pressure (deeper water) increases density. legoESM API uses Pa."""
    p_1000m_pa = 1000.0 * _RHO0 * _GRAV   # ~1e7 Pa = 1000 dbar
    rho_0p = float(veros_nonlin2_eos(jnp.asarray(10.0), jnp.asarray(35.0),
                                       jnp.asarray(0.0)))
    rho_1000 = float(veros_nonlin2_eos(jnp.asarray(10.0), jnp.asarray(35.0),
                                         jnp.asarray(p_1000m_pa)))
    assert rho_1000 > rho_0p


def test_at_reference_state_equals_rho0_at_surface():
    """T = theta0, S = S0, p = 0 → anomaly = 0, in-situ = rho_0."""
    rho = float(veros_nonlin2_eos(
        jnp.asarray(_THETA0), jnp.asarray(_S0), jnp.asarray(0.0),
    ))
    np.testing.assert_allclose(rho, _RHO0, atol=1e-12)


def test_dispatch_via_make_eos_fn():
    fn = make_eos_fn("veros_nonlin2")
    rho = float(fn(jnp.asarray(_THETA0), jnp.asarray(_S0),
                   jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, _RHO0, atol=1e-12)


def test_differentiable_wrt_T_S_p():
    fn_T = lambda T: veros_nonlin2_eos(T, jnp.asarray(35.0), jnp.asarray(0.0))
    fn_S = lambda S: veros_nonlin2_eos(jnp.asarray(10.0), S, jnp.asarray(0.0))
    fn_p = lambda p: veros_nonlin2_eos(jnp.asarray(10.0), jnp.asarray(35.0), p)
    gT = float(jax.grad(fn_T)(jnp.asarray(10.0)))
    gS = float(jax.grad(fn_S)(jnp.asarray(35.0)))
    gp = float(jax.grad(fn_p)(jnp.asarray(0.0)))
    assert gT < 0   # warming reduces density
    assert gS > 0   # higher salt increases density
    # ∂ρ/∂p > 0 — increasing pressure/depth densifies water. The
    # nonlin2 formula uses zz = -p so the chain rule flips the sign
    # of the inner expression, but with the outer negative on the
    # anomaly the net derivative is +grav/cs0² · ρ₀ ≈ +4.5e-3 kg/m³/dbar.
    assert gp > 0
    # ∂ρ/∂p_Pa = (1/(rho_0·g)) · ∂ρ/∂depth ≈ 1/cs0² ≈ 4.5e-7 kg/m³/Pa
    np.testing.assert_allclose(gp, 1.0 / _CS0**2, rtol=1e-3)
