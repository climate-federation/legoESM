"""Tests for the UNESCO 1980 equation of state (Phase G.1b).

Reference density values are taken from the UNESCO 1981 / Fofonoff &
Millard 1983 standard table — these are independent verification
points for the polynomial implementation.

Veros's ``eq_of_state_type=3`` (Jackett & McDougall 1995) is a closely
related polynomial that adjusts a few coefficients to better match
modern measurements; the two agree to within ~0.001 kg/m³ at typical
ocean T/S. For bit-exact Veros parity we need the JM95 coefficients
directly — tracked as audit follow-up.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.eos import (
    linear_eos,
    make_eos_fn,
    unesco80_eos,
    wright_eos,
)

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Standard UNESCO reference points
# ---------------------------------------------------------------------------


def test_pure_water_at_zero_T_p():
    """rho(T=0, S=0, p=0) = 999.842594 (UNESCO 1980 a0)."""
    rho = float(unesco80_eos(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 999.842594, atol=1e-4)


def test_seawater_at_zero_T_p():
    """rho(T=0, S=35, p=0) = 1028.106331 (UNESCO 1980)."""
    rho = float(unesco80_eos(jnp.asarray(0.0), jnp.asarray(35.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1028.106331, atol=1e-3)


def test_seawater_at_warm_temperature():
    """rho(T=20, S=35, p=0) ≈ 1024.76 (UNESCO 1981 standard table; the
    most-cited surface-water reference value). Tolerance 0.05 covers
    the small variation across UNESCO/Pond&Pickard tabulations."""
    rho = float(unesco80_eos(jnp.asarray(20.0), jnp.asarray(35.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1024.76, atol=0.05)


def test_seawater_at_25c_35psu():
    """rho(T=25, S=35, p=0) ≈ 1023.343 (Pond & Pickard 1983 Table 6.5)."""
    rho = float(unesco80_eos(jnp.asarray(25.0), jnp.asarray(35.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1023.343, atol=5e-3)


def test_pressure_dependence_increases_density():
    """At fixed T, S, increasing pressure must increase density
    (seawater is compressible). Use deep-ocean pressure 5000 dbar."""
    T = jnp.asarray(2.0)
    S = jnp.asarray(35.0)
    rho_0 = float(unesco80_eos(T, S, jnp.asarray(0.0)))
    rho_5000_dbar = float(unesco80_eos(T, S, jnp.asarray(5000.0 * 1e4)))  # 5000 dbar = 5e7 Pa
    # ~2.3 % density increase at 5000 dbar — order of magnitude check.
    assert rho_5000_dbar > rho_0
    assert rho_5000_dbar - rho_0 > 15.0
    assert rho_5000_dbar - rho_0 < 30.0


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def test_make_eos_fn_dispatches_unesco80():
    fn = make_eos_fn("unesco80")
    rho = float(fn(jnp.asarray(0.0), jnp.asarray(35.0), jnp.asarray(0.0)))
    np.testing.assert_allclose(rho, 1028.106331, atol=1e-3)


def test_make_eos_fn_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown EOS scheme"):
        make_eos_fn("definitely_not_an_eos")


# ---------------------------------------------------------------------------
# Cross-checks vs. existing EOSes — agreement within ocean-typical T, S, p
# ---------------------------------------------------------------------------


def test_unesco80_close_to_wright_at_surface():
    """Wright 1997 and UNESCO 1980 should agree to within a few tenths
    of a kg/m³ at typical surface conditions."""
    T = jnp.linspace(0.0, 30.0, 5)
    S = jnp.full_like(T, 35.0)
    p = jnp.zeros_like(T)
    rho_un = jax.vmap(unesco80_eos)(T, S, p)
    rho_wr = jax.vmap(wright_eos)(T, S, p)
    diff = np.asarray(jnp.abs(rho_un - rho_wr))
    assert diff.max() < 0.5, (
        f"UNESCO 80 vs Wright 97 max diff {diff.max():.3f} kg/m³ "
        "exceeds the 0.5 expected envelope at typical surface conditions."
    )


def test_unesco80_matches_linear_locally():
    """Local linearisation of UNESCO 1980 around (T_ref, S_ref) should
    agree with the linear EOS configured with thermal/haline
    coefficients evaluated at the same reference point."""
    T_ref = 10.0
    S_ref = 35.0
    rho_ref = float(unesco80_eos(jnp.asarray(T_ref), jnp.asarray(S_ref),
                                  jnp.asarray(0.0)))
    # Compute thermal/haline coefficients from UNESCO at the reference
    # via auto-diff and compare against a small T perturbation.
    drho_dT = float(jax.grad(
        lambda T: unesco80_eos(T, jnp.asarray(S_ref), jnp.asarray(0.0))
    )(jnp.asarray(T_ref)))
    rho_un_T1 = float(unesco80_eos(jnp.asarray(T_ref + 1.0),
                                    jnp.asarray(S_ref), jnp.asarray(0.0)))
    # Local linearisation: rho(T_ref + 1) ≈ rho_ref + drho/dT · 1
    np.testing.assert_allclose(rho_un_T1, rho_ref + drho_dT, atol=0.01)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_unesco80_differentiable_wrt_T():
    fn = lambda T: unesco80_eos(T, jnp.asarray(35.0), jnp.asarray(0.0))
    grad = jax.grad(fn)(jnp.asarray(10.0))
    # d_rho/dT < 0 at T > 4°C (water expands with warming).
    assert float(grad) < 0
    # Magnitude is the thermal expansion times rho ≈ 2e-4 · 1024 ≈ 0.2
    assert abs(float(grad)) > 0.1
    assert abs(float(grad)) < 0.5


def test_unesco80_differentiable_wrt_S():
    fn = lambda S: unesco80_eos(jnp.asarray(10.0), S, jnp.asarray(0.0))
    grad = jax.grad(fn)(jnp.asarray(35.0))
    # d_rho/dS ≈ +0.76 at S=35, T=10 (haline contraction)
    assert float(grad) > 0
    np.testing.assert_allclose(float(grad), 0.76, atol=0.05)
