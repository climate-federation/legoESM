"""Wright (1997) EOS ORACLE-FAITHFULNESS tests.

``ocean/eos.py::wright_eos`` is the DEFAULT ocean equation of state, yet — unlike
every ALTERNATE EOS (TEOS-10 ``veros_gsw``, UNESCO-80, NEMO-SEOS, veros-nonlin2/3,
all value-pinned) — it had only qualitative checks (warm < cold, salty > fresh,
1020 < ρ < 1035).  These pin it to round-off against an INDEPENDENT scalar
reimplementation of the Wright (1997) rational-polynomial form and cross-validate
against the already-pinned UNESCO-80 EOS.

Wright (1997) / MOM6 ``MOM_EOS_Wright.F90`` (reduced-range fit), T [degC], S [PSU],
p [Pa], ρ [kg/m^3]:

    ρ    = (p + p0) / (λ + al0·(p + p0))
    al0  = a0 + a1·T + a2·S
    p0   = (b0 + b4·S) + T·(b1 + T·(b2 + b3·T) + b5·S)
    λ    = (c0 + c4·S) + T·(c1 + T·(c2 + c3·T) + c5·S)

What is pinned, in order of authority:
1. Density closed form to round-off (rel 1e-12) across a T×S×p grid vs an
   independent EXPANDED-monomial reimplementation — a different association than
   the module's nested Horner, so a transcription / parenthesization bug in
   ``wright_eos`` (not just a coefficient typo) is caught.
2. INDEPENDENT physical cross-check: ρ_Wright ≈ ρ_UNESCO-80 to < 0.01 kg/m^3 at
   surface mid-range — a separate published EOS catches any material coefficient
   error affecting these points (four surface values cannot pin down all 15
   coefficients, but a gross error would diverge by >> 0.01).
3. α = -(1/ρ)∂ρ/∂T and β = (1/ρ)∂ρ/∂S (which the module gets via ``jax.grad``)
   pinned to independent ANALYTIC derivatives of the closed form (rel 1e-9).
4. Coefficient canary vs the published MOM6/Wright values.
5. Physical monotonicity (ρ ↓ with T in the warm range, ↑ with S, ↑ with p) and
   x64/float32 AD-finiteness.

Precision: the EOS polynomial runs in the active precision policy's COMPUTE dtype
(float32 by default even under ``JAX_ENABLE_X64=1``), so the autouse fixture sets
the fp64 policy for the round-off pins and restores the entry policy after.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(autouse=True)
def _force_fp64():
    """float64 + fp64 precision policy for the round-off pins; restore both in
    finally so no policy/x64 state leaks into other modules.  Entry state is
    captured per-test (not at import) so restoration is correct regardless of the
    order tests run in."""
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    entry_policy = get_policy()
    entry_x64 = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(entry_policy)
        jax.config.update("jax_enable_x64", entry_x64)


from legoesm.ocean import eos as _eos                                    # noqa: E402
from legoesm.ocean.eos import (                                          # noqa: E402
    wright_eos, thermal_expansion_coeff, haline_contraction_coeff, unesco80_eos,
)

# Independent transcription of the MOM6 (MOM_EOS_Wright.F90) reduced-range Wright
# (1997) coefficients — canaried against the module below.
_O_A0, _O_A1, _O_A2 = 7.057924e-4, 3.480336e-7, -1.112733e-7
_O_B0, _O_B1, _O_B2 = 5.790749e8, 3.516535e6, -4.002714e4
_O_B3, _O_B4, _O_B5 = 2.084372e2, 5.944068e5, -9.643486e3
_O_C0, _O_C1, _O_C2 = 1.704853e5, 7.904722e2, -7.984422
_O_C3, _O_C4, _O_C5 = 5.140652e-2, -2.302158e2, -3.079464


def _rho_oracle(T, S, p):
    """Independent Wright (1997) density in FULLY EXPANDED monomial form.

    The polynomials p0 and λ are summed monomial-by-monomial (a0+a1·T+…), a
    genuinely different association than the module's nested-Horner evaluation, so
    a parenthesization / evaluation-order transcription bug in ``wright_eos`` (not
    just a coefficient typo) shows up as a mismatch."""
    T2 = T * T
    T3 = T2 * T
    al0 = _O_A0 + _O_A1 * T + _O_A2 * S
    p0 = _O_B0 + _O_B4 * S + _O_B1 * T + _O_B2 * T2 + _O_B3 * T3 + _O_B5 * S * T
    lam = _O_C0 + _O_C4 * S + _O_C1 * T + _O_C2 * T2 + _O_C3 * T3 + _O_C5 * S * T
    return (p + p0) / (lam + al0 * (p + p0))


def _alpha_beta_oracle(T, S, p):
    """Independent ANALYTIC α, β from the closed form (not jax.grad); polynomials
    in the same fully expanded monomial form as ``_rho_oracle``."""
    T2 = T * T
    T3 = T2 * T
    al0 = _O_A0 + _O_A1 * T + _O_A2 * S
    p0 = _O_B0 + _O_B4 * S + _O_B1 * T + _O_B2 * T2 + _O_B3 * T3 + _O_B5 * S * T
    lam = _O_C0 + _O_C4 * S + _O_C1 * T + _O_C2 * T2 + _O_C3 * T3 + _O_C5 * S * T
    N = p + p0
    D = lam + al0 * N
    rho = N / D
    dN_dT = _O_B1 + 2 * _O_B2 * T + 3 * _O_B3 * T2 + _O_B5 * S
    dD_dT = (_O_C1 + 2 * _O_C2 * T + 3 * _O_C3 * T2 + _O_C5 * S) + _O_A1 * N + al0 * dN_dT
    dN_dS = _O_B4 + _O_B5 * T
    dD_dS = (_O_C4 + _O_C5 * T) + _O_A2 * N + al0 * dN_dS
    drho_dT = (dN_dT * D - N * dD_dT) / D ** 2
    drho_dS = (dN_dS * D - N * dD_dS) / D ** 2
    return -drho_dT / rho, drho_dS / rho


def _a(x):
    return jnp.array(float(x))


# T endpoints -2 and 40 degC are the documented reduced-range limits (the cubic
# terms are most stressed at 40 degC); include both.
_TS_P_GRID = [(T, S, p)
              for T in (-2.0, -1.0, 5.0, 15.0, 25.0, 35.0, 40.0)
              for S in (0.0, 20.0, 35.0, 42.0)
              for p in (0.0, 1.0e6, 4.0e7)]


# --- density closed form -------------------------------------------------------

@pytest.mark.parametrize("T,S,p", _TS_P_GRID)
def test_wright_density_matches_oracle(T, S, p):
    got = float(wright_eos(_a(T), _a(S), _a(p)))
    assert got == pytest.approx(_rho_oracle(T, S, p), rel=1e-12, abs=0.0)


def test_wright_check_value():
    """Regression check value (float64): ρ(T=10, S=35, p=0) = 1026.9514889."""
    assert float(wright_eos(_a(10.0), _a(35.0), _a(0.0))) == pytest.approx(
        1026.951488884, rel=1e-10)


# --- independent physical cross-check vs UNESCO-80 -----------------------------

@pytest.mark.parametrize("T,S", [(10.0, 35.0), (20.0, 35.0), (2.0, 34.0), (25.0, 36.0)])
def test_wright_agrees_with_unesco80_surface(T, S):
    """Independent oracle: the Wright fit must agree with the (separately pinned)
    UNESCO-80 EOS to < 0.01 kg/m^3 at the surface over the open ocean — a
    different published EOS catches material coefficient errors affecting these
    points (four surface values cannot pin down all 15 coefficients, but a gross
    error would diverge by >> 0.01)."""
    w = float(wright_eos(_a(T), _a(S), _a(0.0)))
    u = float(unesco80_eos(_a(T), _a(S), _a(0.0)))
    assert abs(w - u) < 0.01, f"Wright {w} vs UNESCO-80 {u}"


# --- alpha / beta vs independent analytic derivatives --------------------------

@pytest.mark.parametrize("T,S,p", [(5.0, 35.0, 0.0), (25.0, 35.0, 0.0),
                                   (2.0, 34.0, 4.0e7), (15.0, 20.0, 1.0e6)])
def test_thermal_expansion_matches_analytic(T, S, p):
    """α = -(1/ρ)∂ρ/∂T (module uses jax.grad) vs the closed-form analytic derivative."""
    ao, _ = _alpha_beta_oracle(T, S, p)
    got = float(thermal_expansion_coeff(_a(T), _a(S), _a(p)))
    assert got == pytest.approx(ao, rel=1e-9)


@pytest.mark.parametrize("T,S,p", [(5.0, 35.0, 0.0), (25.0, 35.0, 0.0),
                                   (2.0, 34.0, 4.0e7), (15.0, 20.0, 1.0e6)])
def test_haline_contraction_matches_analytic(T, S, p):
    """β = (1/ρ)∂ρ/∂S (module uses jax.grad) vs the closed-form analytic derivative."""
    _, bo = _alpha_beta_oracle(T, S, p)
    got = float(haline_contraction_coeff(_a(T), _a(S), _a(p)))
    assert got == pytest.approx(bo, rel=1e-9)


def test_alpha_positive_warm_beta_positive():
    """Physical signs: α > 0 for warm water (density falls as it warms), β > 0
    (density rises with salinity)."""
    assert float(thermal_expansion_coeff(_a(25.0), _a(35.0), _a(0.0))) > 0.0
    assert float(haline_contraction_coeff(_a(25.0), _a(35.0), _a(0.0))) > 0.0


# --- coefficient canary --------------------------------------------------------

def test_wright_coefficients_match_module():
    assert (_eos.WRIGHT_A0, _eos.WRIGHT_A1, _eos.WRIGHT_A2) == (_O_A0, _O_A1, _O_A2)
    assert (_eos.WRIGHT_B0, _eos.WRIGHT_B1, _eos.WRIGHT_B2) == (_O_B0, _O_B1, _O_B2)
    assert (_eos.WRIGHT_B3, _eos.WRIGHT_B4, _eos.WRIGHT_B5) == (_O_B3, _O_B4, _O_B5)
    assert (_eos.WRIGHT_C0, _eos.WRIGHT_C1, _eos.WRIGHT_C2) == (_O_C0, _O_C1, _O_C2)
    assert (_eos.WRIGHT_C3, _eos.WRIGHT_C4, _eos.WRIGHT_C5) == (_O_C3, _O_C4, _O_C5)


# --- physical monotonicity -----------------------------------------------------

def test_wright_monotonic_physical():
    """ρ decreases with T (warm range), increases with S, increases with p."""
    assert float(wright_eos(_a(25.0), _a(35.0), _a(0.0))) < float(wright_eos(_a(5.0), _a(35.0), _a(0.0)))
    assert float(wright_eos(_a(15.0), _a(38.0), _a(0.0))) > float(wright_eos(_a(15.0), _a(30.0), _a(0.0)))
    assert float(wright_eos(_a(15.0), _a(35.0), _a(4.0e7))) > float(wright_eos(_a(15.0), _a(35.0), _a(0.0)))


# --- AD-safety -----------------------------------------------------------------

def test_wright_grad_finite_x64_and_float32():
    """grad of ρ wrt (T, S, p) is finite in x64 (fp64 policy) and float32 (fp32
    policy — the EOS polynomial then genuinely runs in float32)."""
    from legoesm.core.precision import set_policy, PrecisionPolicy

    def _rho(T, S, p):
        return wright_eos(T, S, p)

    def _check(dtype):
        args = (jnp.asarray(15.0, dtype), jnp.asarray(35.0, dtype), jnp.asarray(1.0e6, dtype))
        for k in range(3):
            g = jax.grad(_rho, argnums=k)(*args)
            assert bool(jnp.isfinite(g)) and float(jnp.abs(g)) > 0.0

    _check(jnp.float64)   # fp64 policy from the fixture
    jax.config.update("jax_enable_x64", False)
    set_policy(PrecisionPolicy.fp32())
    _check(jnp.float32)   # autouse fixture restores the entry policy + x64 afterwards
